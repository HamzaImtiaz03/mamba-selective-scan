r"""Stage 3 — Triton selective-scan: chunked associative scan, fwd + bwd.

Real Triton implementation as a ``torch.autograd.Function`` (``SelectiveScanTriton``)
that ports the math proven on CPU in ``backward_math.py`` (float64 gradcheck) to GPU.

Design
------
Parallelism: one Triton *program* per (batch b, channel d) lane; the state dim N is
vectorized inside the program as a length-BLOCK_N register vector. The B*D lanes run
concurrently — that is the parallelism.

* Forward: each lane walks L in chunks of CHUNK and runs a work-efficient
  ``tl.associative_scan`` over the chunk in SRAM, with the combine operator
  ``(a_l,b_l) ∘ (a_r,b_r) = (a_l·a_r, a_r·b_l + b_r)``. The boundary state h is carried
  across chunks by folding ``a_first · h_carry`` into the first row's b before scanning.
  Per-timestep states h are written to HBM once for the backward (nothing is O(L²)).

* Backward: the adjoint of a linear scan is a linear scan,
  ``gh_t = dh_y_t + a_{t+1}·gh_{t+1}``. Rather than risk an off-by-one in a reverse
  *tree* scan, each lane runs this recurrence SEQUENTIALLY in reverse (B*D lanes still
  parallel). It is a direct, unambiguous transcription of the verified formula. Input
  grads (dx, ddelta, dA, dB, dC, dD) follow the closed forms in ``backward_math``;
  grads that reduce across lanes (dA, dB, dC, dD) use ``tl.atomic_add``.

Shared-memory budget (T4, sm_75, ~48KB/block): the forward working tile is
CHUNK×BLOCK_N fp32 = 64·16·4 B = 4 KB for each of (a, b); a few live tiles, well under
48 KB. fp16 inputs are accumulated in fp32 (h buffer is fp32).

Validation status
-----------------
The math is float64-gradcheck-verified on CPU (``tests/sanity_backward.py``). The
Triton kernels are validated against the reference on CUDA — run the test suite on a
Colab T4 (see ``notebooks/colab_runner.ipynb``). With no CUDA present,
``SelectiveScanTriton.apply`` routes to the gradcheck-verified reference and prints a
one-time notice; it never silently pretends a kernel ran.
"""

from __future__ import annotations

import warnings
import torch
from torch import Tensor

from .backward_math import forward_capture, selective_scan_backward

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton absent on CPU-only installs
    _HAS_TRITON = False

# Per-step sequence tile for the forward scan. Small enough for the T4 SRAM budget.
_CHUNK = 64


if _HAS_TRITON:

    @triton.jit
    def _combine(a_l, b_l, a_r, b_r):
        """Associative combine: fuse earlier segment (l) into later segment (r)."""
        return a_l * a_r, a_r * b_l + b_r

    @triton.jit
    def _fwd_kernel(
        x_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr, y_ptr, h_ptr,
        B, L, D, N,
        sx_b, sx_l, sx_d,            # strides of x/delta/y (B,L,D)
        sb_b, sb_l, sb_n,            # strides of B/C (B,L,N)
        sA_d, sA_n,                  # strides of A (D,N)
        sh_b, sh_l, sh_d, sh_n,      # strides of h (B,L,D,N)
        HAS_D: tl.constexpr,
        CHUNK: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)              # one lane per (b, d)
        b = pid // D
        d = pid % D

        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        A_row = tl.load(A_ptr + d * sA_d + offs_n * sA_n, mask=mask_n, other=0.0)  # (N,)
        d_skip = tl.load(D_ptr + d) if HAS_D else 0.0

        h_carry = tl.zeros((BLOCK_N,), dtype=tl.float32)   # boundary state across chunks

        for c in range(0, L, CHUNK):
            offs_t = c + tl.arange(0, CHUNK)
            mask_t = offs_t < L
            delta_c = tl.load(delta_ptr + b * sx_b + offs_t * sx_l + d * sx_d,
                              mask=mask_t, other=0.0).to(tl.float32)            # (CHUNK,)
            x_c = tl.load(x_ptr + b * sx_b + offs_t * sx_l + d * sx_d,
                          mask=mask_t, other=0.0).to(tl.float32)               # (CHUNK,)
            B_c = tl.load(B_ptr + b * sb_b + offs_t[:, None] * sb_l + offs_n[None, :] * sb_n,
                          mask=mask_t[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
            C_c = tl.load(C_ptr + b * sb_b + offs_t[:, None] * sb_l + offs_n[None, :] * sb_n,
                          mask=mask_t[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

            a = tl.exp(delta_c[:, None] * A_row[None, :])                 # (CHUNK,N)
            bb = delta_c[:, None] * B_c * x_c[:, None]                    # (CHUNK,N)

            # Fold carry into the first row: bb_row0 += a_row0 * h_carry.
            is_first = (offs_t == c)
            bb = bb + tl.where(is_first[:, None], a * h_carry[None, :], 0.0)

            # Inclusive associative scan over the time axis (axis=0): b-component == h.
            _, h_c = tl.associative_scan((a, bb), 0, _combine)           # (CHUNK,N)

            y_c = tl.sum(C_c * h_c, axis=1)                              # (CHUNK,)
            if HAS_D:
                y_c = y_c + d_skip * x_c
            tl.store(y_ptr + b * sx_b + offs_t * sx_l + d * sx_d, y_c, mask=mask_t)
            tl.store(h_ptr + b * sh_b + offs_t[:, None] * sh_l + d * sh_d + offs_n[None, :] * sh_n,
                     h_c, mask=mask_t[:, None] & mask_n[None, :])

            # Carry the last valid state in the chunk forward.
            last = tl.minimum(CHUNK, L - c) - 1
            h_carry = tl.sum(tl.where((tl.arange(0, CHUNK))[:, None] == last, h_c, 0.0), axis=0)

    @triton.jit
    def _bwd_kernel(
        dy_ptr, x_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr, h_ptr,
        dx_ptr, ddelta_ptr, dA_ptr, dB_ptr, dC_ptr, dD_ptr,
        B, L, D, N,
        sx_b, sx_l, sx_d,
        sb_b, sb_l, sb_n,
        sA_d, sA_n,
        sh_b, sh_l, sh_d, sh_n,
        HAS_D: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // D
        d = pid % D
        offs_n = tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        A_row = tl.load(A_ptr + d * sA_d + offs_n * sA_n, mask=mask_n, other=0.0)
        d_skip = tl.load(D_ptr + d) if HAS_D else 0.0

        # Reverse adjoint recurrence: gh_t = dh_y_t + a_{t+1}*gh_{t+1}  (sequential).
        G = tl.zeros((BLOCK_N,), dtype=tl.float32)        # gh_{t+1}
        mult = tl.zeros((BLOCK_N,), dtype=tl.float32)     # a_{t+1}
        dA_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        dD_acc = tl.zeros((1,), dtype=tl.float32)

        for i in range(0, L):
            t = L - 1 - i
            delta_t = tl.load(delta_ptr + b * sx_b + t * sx_l + d * sx_d).to(tl.float32)
            x_t = tl.load(x_ptr + b * sx_b + t * sx_l + d * sx_d).to(tl.float32)
            dy_t = tl.load(dy_ptr + b * sx_b + t * sx_l + d * sx_d).to(tl.float32)
            B_t = tl.load(B_ptr + b * sb_b + t * sb_l + offs_n * sb_n, mask=mask_n, other=0.0).to(tl.float32)
            C_t = tl.load(C_ptr + b * sb_b + t * sb_l + offs_n * sb_n, mask=mask_n, other=0.0).to(tl.float32)
            h_t = tl.load(h_ptr + b * sh_b + t * sh_l + d * sh_d + offs_n * sh_n, mask=mask_n, other=0.0)
            # h_{t-1}; zero at t==0.
            h_prev = tl.load(h_ptr + b * sh_b + (t - 1) * sh_l + d * sh_d + offs_n * sh_n,
                             mask=mask_n & (t >= 1), other=0.0)

            a_t = tl.exp(delta_t * A_row)                 # (N,)
            dh_y = dy_t * C_t                             # (N,)
            gh = dh_y + mult * G                          # (N,)  gh_t

            g_bb = gh
            g_a = gh * h_prev

            # Per-(b,t,d) grads (no atomics: this lane owns (b,d), writes all t).
            ddelta = tl.sum(g_bb * B_t * x_t) + tl.sum(g_a * a_t * A_row)
            dx = tl.sum(g_bb * delta_t * B_t)
            if HAS_D:
                dx += dy_t * d_skip
                dD_acc += dy_t * x_t
            tl.store(ddelta_ptr + b * sx_b + t * sx_l + d * sx_d, ddelta)
            tl.store(dx_ptr + b * sx_b + t * sx_l + d * sx_d, dx)

            # Reductions across lanes -> atomics.
            dA_acc += g_a * a_t * delta_t
            tl.atomic_add(dB_ptr + b * sb_b + t * sb_l + offs_n * sb_n, g_bb * delta_t * x_t, mask=mask_n)
            tl.atomic_add(dC_ptr + b * sb_b + t * sb_l + offs_n * sb_n, dy_t * h_t, mask=mask_n)

            # Advance the reverse recurrence.
            mult = a_t
            G = gh

        tl.atomic_add(dA_ptr + d * sA_d + offs_n * sA_n, dA_acc, mask=mask_n)
        if HAS_D:
            tl.atomic_add(dD_ptr + d, tl.sum(dD_acc))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _triton_forward(x, delta, A, B_mat, C_mat, D_skip):
    B, L, D = x.shape
    N = A.shape[1]
    x, delta = x.contiguous(), delta.contiguous()
    A, B_mat, C_mat = A.contiguous(), B_mat.contiguous(), C_mat.contiguous()
    has_d = D_skip is not None
    D_skip_c = D_skip.contiguous() if has_d else x.new_zeros(D)

    y = torch.empty(B, L, D, device=x.device, dtype=x.dtype)
    h = torch.empty(B, L, D, N, device=x.device, dtype=torch.float32)
    BLOCK_N = _next_pow2(N)
    grid = (B * D,)
    _fwd_kernel[grid](
        x, delta, A, B_mat, C_mat, D_skip_c, y, h,
        B, L, D, N,
        x.stride(0), x.stride(1), x.stride(2),
        B_mat.stride(0), B_mat.stride(1), B_mat.stride(2),
        A.stride(0), A.stride(1),
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        HAS_D=has_d, CHUNK=_CHUNK, BLOCK_N=BLOCK_N,
    )
    return y, h


def _triton_backward(dy, x, delta, A, B_mat, C_mat, D_skip, h):
    B, L, D = x.shape
    N = A.shape[1]
    has_d = D_skip is not None
    D_skip_c = D_skip.contiguous() if has_d else x.new_zeros(D)
    dy = dy.contiguous()

    dx = torch.empty_like(x)
    ddelta = torch.empty_like(delta)
    # Parameter grads accumulate via atomics -> must start at zero, fp32 for accuracy.
    dA = torch.zeros_like(A, dtype=torch.float32)
    dB = torch.zeros(B, L, N, device=x.device, dtype=torch.float32)
    dC = torch.zeros(B, L, N, device=x.device, dtype=torch.float32)
    dD = torch.zeros(D, device=x.device, dtype=torch.float32)
    BLOCK_N = _next_pow2(N)
    grid = (B * D,)
    _bwd_kernel[grid](
        dy, x, delta, A, B_mat, C_mat, D_skip_c, h,
        dx, ddelta, dA, dB, dC, dD,
        B, L, D, N,
        x.stride(0), x.stride(1), x.stride(2),
        B_mat.stride(0), B_mat.stride(1), B_mat.stride(2),
        A.stride(0), A.stride(1),
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        HAS_D=has_d, BLOCK_N=BLOCK_N,
    )
    cast = lambda g: g.to(x.dtype)
    return (cast(dx), cast(ddelta), cast(dA), cast(dB), cast(dC),
            (cast(dD) if has_d else None))


_WARNED_CPU = False


def _use_triton(x: Tensor) -> bool:
    return _HAS_TRITON and x.is_cuda


class SelectiveScanTriton(torch.autograd.Function):
    """Autograd Function dispatching to the Triton kernels on CUDA.

    On a non-CUDA device (or if Triton is unavailable) it routes to the
    float64-gradcheck-verified reference math and emits a one-time warning so the
    fallback is explicit, never silent.
    """

    @staticmethod
    def forward(ctx, x, delta, A, B_mat, C_mat, D_skip):
        if _use_triton(x):
            y, h = _triton_forward(x, delta, A, B_mat, C_mat, D_skip)
            ctx.used_triton = True
        else:
            global _WARNED_CPU
            if not _WARNED_CPU:
                warnings.warn(
                    "SelectiveScanTriton: no CUDA/Triton available -> using the "
                    "gradcheck-verified reference math (NOT the Triton kernel).",
                    RuntimeWarning, stacklevel=2)
                _WARNED_CPU = True
            y, _a, h = forward_capture(x, delta, A, B_mat, C_mat, D_skip)
            ctx.used_triton = False
        ctx.save_for_backward(
            x, delta, A, B_mat, C_mat,
            D_skip if D_skip is not None else x.new_zeros(0), h)
        ctx.has_dskip = D_skip is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        x, delta, A, B_mat, C_mat, D_skip_or_empty, h = ctx.saved_tensors
        D_skip = D_skip_or_empty if ctx.has_dskip else None
        if ctx.used_triton:
            grads = _triton_backward(dy.contiguous(), x, delta, A, B_mat, C_mat, D_skip, h)
        else:
            # Reference path needs the transition coefficients a; recompute (cheap).
            a = torch.exp(delta.unsqueeze(-1) * A)
            grads = selective_scan_backward(
                dy.contiguous(), x, delta, A, B_mat, C_mat, D_skip, a, h)
        return (*grads,)


def selective_scan_triton(x, delta, A, B_mat, C_mat, D_skip=None):
    """Public entry point for the Triton selective scan."""
    return SelectiveScanTriton.apply(x, delta, A, B_mat, C_mat, D_skip)
