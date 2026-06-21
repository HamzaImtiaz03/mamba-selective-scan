r"""The analytical backward of the selective scan — derived once, verified on CPU.

This module is the **single source of truth** for the gradient math. The Triton
kernel (Stage 3) and the CUDA kernel (Stage 5) port these exact formulas; here we
keep a transparent pure-torch version that ``torch.autograd.gradcheck`` can validate
in float64. If gradcheck passes here, the math is correct and the kernels only have
to reproduce it.

Notation (all per channel, indices b∈B, t∈L, d∈D, n∈N)
------------------------------------------------------
Forward:
    a[b,t,d,n]  = exp(delta[b,t,d] * A[d,n])                       # transition
    bb[b,t,d,n] = delta[b,t,d] * B[b,t,n] * x[b,t,d]               # Bbar * x_t
    h[b,t]      = a[b,t] * h[b,t-1] + bb[b,t]    (h[b,-1] = 0)     # recurrence
    y[b,t,d]    = Σ_n C[b,t,n] * h[b,t,d,n]  +  Dskip[d] * x[b,t,d]

Backward (given upstream dy = ∂L/∂y)
------------------------------------
Readout:
    dC[b,t,n]   = Σ_d dy[b,t,d] * h[b,t,d,n]
    dDskip[d]   = Σ_{b,t} dy[b,t,d] * x[b,t,d]
    dh_y[b,t,d,n] = dy[b,t,d] * C[b,t,n]            # grad of y wrt h

Adjoint (REVERSE linear scan — the gradient of a scan is a scan):
    gh[b,t] = dh_y[b,t] + a[b,t+1] * gh[b,t+1]      (gh[b,L] = 0)

Local grads of the recurrence:
    g_bb[b,t] = gh[b,t]
    g_a[b,t]  = gh[b,t] * h[b,t-1]                  (h[b,-1] = 0)

Input grads (note delta appears in BOTH bb and a → the delta·A coupling):
    ddelta[b,t,d] = Σ_n [ g_bb*B*x ]_n  +  Σ_n [ g_a * a * A ]_n
    dx[b,t,d]     = dy[b,t,d]*Dskip[d]  +  Σ_n g_bb[b,t,d,n] * delta[b,t,d] * B[b,t,n]
    dB[b,t,n]     = Σ_d g_bb[b,t,d,n] * delta[b,t,d] * x[b,t,d]
    dA[d,n]       = Σ_{b,t} g_a[b,t,d,n] * a[b,t,d,n] * delta[b,t,d]
"""

from __future__ import annotations

import torch
from torch import Tensor


def forward_capture(
    x: Tensor, delta: Tensor, A: Tensor, B_mat: Tensor, C_mat: Tensor, D_skip: Tensor | None
):
    """Sequential forward that also returns the tensors backward needs.

    Returns:
        y: (B, L, D)
        a: (B, L, D, N) transition coefficients
        h: (B, L, D, N) hidden states h_1..h_L
    """
    B, L, D = x.shape
    N = A.shape[1]
    a = torch.exp(delta.unsqueeze(-1) * A)                              # (B, L, D, N)
    bb = delta.unsqueeze(-1) * B_mat.unsqueeze(2) * x.unsqueeze(-1)     # (B, L, D, N)

    h = torch.empty(B, L, D, N, dtype=x.dtype, device=x.device)
    prev = torch.zeros(B, D, N, dtype=x.dtype, device=x.device)
    for t in range(L):
        prev = a[:, t] * prev + bb[:, t]
        h[:, t] = prev

    y = (C_mat.unsqueeze(2) * h).sum(dim=-1)                            # (B, L, D)
    if D_skip is not None:
        y = y + D_skip * x
    return y, a, h


def selective_scan_backward(
    dy: Tensor,
    x: Tensor,
    delta: Tensor,
    A: Tensor,
    B_mat: Tensor,
    C_mat: Tensor,
    D_skip: Tensor | None,
    a: Tensor,
    h: Tensor,
):
    """Analytical backward. Returns grads matching the input signature.

    Args:
        dy: (B, L, D) upstream gradient.
        x, delta, B_mat, C_mat, D_skip: forward inputs.
        A: (D, N).
        a, h: (B, L, D, N) captured by :func:`forward_capture`.

    Returns:
        (dx, ddelta, dA, dB_mat, dC_mat, dD_skip). dD_skip is None if D_skip is None.
    """
    B, L, D = x.shape
    N = A.shape[1]

    # --- Readout gradients ---
    # dh_y[b,t,d,n] = dy[b,t,d] * C[b,t,n]
    dh_y = dy.unsqueeze(-1) * C_mat.unsqueeze(2)                        # (B, L, D, N)
    # dC[b,t,n] = sum_d dy[b,t,d] * h[b,t,d,n]
    dC_mat = (dy.unsqueeze(-1) * h).sum(dim=2)                          # (B, L, N)

    # --- Adjoint reverse scan: gh[b,t] = dh_y[b,t] + a[b,t+1]*gh[b,t+1] ---
    gh = torch.empty_like(h)
    carry = torch.zeros(B, D, N, dtype=h.dtype, device=h.device)       # a_{t+1}*gh_{t+1}
    for t in range(L - 1, -1, -1):
        gh[:, t] = dh_y[:, t] + carry
        carry = a[:, t] * gh[:, t]

    # --- Local grads of the recurrence ---
    g_bb = gh
    # h_prev[:, t] = h[:, t-1], with h[:, -1] = 0
    h_prev = torch.cat([torch.zeros(B, 1, D, N, dtype=h.dtype, device=h.device), h[:, :-1]], dim=1)
    g_a = gh * h_prev

    # --- Input gradients ---
    # ddelta = sum_n( g_bb * B * x )  +  sum_n( g_a * a * A )
    ddelta = (g_bb * B_mat.unsqueeze(2) * x.unsqueeze(-1)).sum(-1) \
        + (g_a * a * A).sum(-1)                                        # (B, L, D)
    # dx (recurrence part) = sum_n g_bb * delta * B
    dx = (g_bb * delta.unsqueeze(-1) * B_mat.unsqueeze(2)).sum(-1)     # (B, L, D)
    # dB[b,t,n] = sum_d g_bb * delta * x
    dB_mat = (g_bb * delta.unsqueeze(-1) * x.unsqueeze(-1)).sum(dim=2) # (B, L, N)
    # dA[d,n] = sum_{b,t} g_a * a * delta
    dA = (g_a * a * delta.unsqueeze(-1)).sum(dim=(0, 1))               # (D, N)

    # --- Skip connection grads ---
    if D_skip is not None:
        dx = dx + dy * D_skip                                         # (B, L, D)
        dD_skip = (dy * x).sum(dim=(0, 1))                            # (D,)
    else:
        dD_skip = None

    return dx, ddelta, dA, dB_mat, dC_mat, dD_skip


class SelectiveScanRef(torch.autograd.Function):
    """Reference autograd.Function: explicit forward + ANALYTICAL backward.

    Used by ``gradcheck`` to validate the backward math on CPU/float64. The Triton
    and CUDA autograd Functions implement the identical contract on GPU.
    """

    @staticmethod
    def forward(ctx, x, delta, A, B_mat, C_mat, D_skip):
        y, a, h = forward_capture(x, delta, A, B_mat, C_mat, D_skip)
        ctx.save_for_backward(x, delta, A, B_mat, C_mat, D_skip if D_skip is not None else x.new_zeros(0), a, h)
        ctx.has_dskip = D_skip is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        x, delta, A, B_mat, C_mat, D_skip_or_empty, a, h = ctx.saved_tensors
        D_skip = D_skip_or_empty if ctx.has_dskip else None
        dx, ddelta, dA, dB_mat, dC_mat, dD_skip = selective_scan_backward(
            dy.contiguous(), x, delta, A, B_mat, C_mat, D_skip, a, h)
        return dx, ddelta, dA, dB_mat, dC_mat, dD_skip


def selective_scan_ref_autograd(x, delta, A, B_mat, C_mat, D_skip=None):
    """Convenience wrapper around :class:`SelectiveScanRef`."""
    return SelectiveScanRef.apply(x, delta, A, B_mat, C_mat, D_skip)
