"""Stage 2 — the selective scan as an ASSOCIATIVE SCAN, in pure PyTorch.

This is the conceptual bridge between the sequential reference and the custom
kernels. It computes *exactly* the same thing as :func:`selective_scan_ref`, but
instead of a Python loop over ``L`` it uses a parallel prefix scan over the
associative combine operator. No custom CUDA/Triton — just torch tensor ops — so
it is easy to read, fully autograd-differentiable, and `allclose` to the oracle.

Why this works
--------------
The hidden recurrence is a *first-order linear recurrence*:

    h_t = a_t · h_{t-1} + b_t                       (elementwise per (B, D, N) channel)

with
    a_t = Abar_t = exp(delta_t · A)                 # (B, L, D, N)
    b_t = Bbar_t · x_t = (delta_t · B_t) · x_t      # (B, L, D, N)

A first-order linear recurrence is an associative scan. Represent each timestep by
the pair (a_t, b_t). The combine operator that fuses an earlier segment L with a
later segment R is:

    (a_L, b_L) ∘ (a_R, b_R) = (a_L · a_R,  a_R · b_L + b_R)

It is associative (proof: both groupings give the affine map x ↦ a_L a_R x + a_R b_L
+ b_R). Because h_0 = 0, the inclusive prefix scan's b-component *is* the state:
h_t = b_{1..t}. We then read out y_t = Σ_N C_t · h_t (+ D_skip · x_t).

Here we use a Hillis-Steele inclusive scan: O(L·log L) work, O(log L) depth. It is
not work-efficient (the kernels use the work-efficient Blelloch scan), but it is the
simplest correct vectorized formulation — exactly what a "stepping stone" should be.
"""

from __future__ import annotations

import torch
from torch import Tensor


def combine(a_l: Tensor, b_l: Tensor, a_r: Tensor, b_r: Tensor) -> tuple[Tensor, Tensor]:
    """The associative combine operator: fuse earlier segment L into later segment R.

        (a_l, b_l) ∘ (a_r, b_r) = (a_l·a_r,  a_r·b_l + b_r)

    Returns the (a, b) of the combined segment.
    """
    return a_l * a_r, a_r * b_l + b_r


def hillis_steele_scan(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Inclusive associative scan along dim=1 (the sequence dim L).

    Args:
        a, b: (B, L, D, N) per-timestep coefficients of the linear recurrence.

    Returns:
        (a_scan, b_scan): inclusive prefix combine over L. b_scan[:, t] == h_t.
    """
    L = a.shape[1]
    offset = 1
    while offset < L:
        # Earlier (left) operands are positions [0 .. L-offset-1]; later (right) are
        # [offset .. L-1]. Combine pairs (t-offset, t) for all t >= offset.
        a_left, b_left = a[:, : L - offset], b[:, : L - offset]
        a_right, b_right = a[:, offset:], b[:, offset:]
        a_comb, b_comb = combine(a_left, b_left, a_right, b_right)

        # Out-of-place write so the step reads a consistent previous level (not in-place
        # aliased), which also keeps autograd happy.
        a = torch.cat([a[:, :offset], a_comb], dim=1)
        b = torch.cat([b[:, :offset], b_comb], dim=1)
        offset *= 2
    return a, b


def selective_scan_parallel(
    x: Tensor,
    delta: Tensor,
    A: Tensor,
    B_mat: Tensor,
    C_mat: Tensor,
    D_skip: Tensor | None = None,
    *,
    return_states: bool = False,
):
    """Pure-torch associative-scan implementation of the S6 selective scan.

    Identical contract to :func:`mamba_scan.reference.selective_scan_ref`.

    Args:
        x:      (B, L, D)
        delta:  (B, L, D)   positive
        A:      (D, N)      negative
        B_mat:  (B, L, N)
        C_mat:  (B, L, N)
        D_skip: (D,) or None
        return_states: if True, also return the (B, L, D, N) hidden states.

    Returns:
        y: (B, L, D)   (and hs if return_states).
    """
    B, L, D = x.shape
    N = A.shape[1]

    # --- Discretization: build the per-timestep recurrence coefficients (a_t, b_t). ---
    # a_t = exp(delta_t · A): delta (B,L,D)->(B,L,D,1), A (D,N)->(1,1,D,N) => (B,L,D,N)
    a = torch.exp(delta.unsqueeze(-1) * A)                          # (B, L, D, N)
    # b_t = (delta_t · B_t) · x_t.  delta (B,L,D,1), B_mat (B,L,1,N), x (B,L,D,1)
    b = delta.unsqueeze(-1) * B_mat.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, D, N)

    # --- Inclusive associative scan over L. b-component == hidden state h_t. ---
    _, h = hillis_steele_scan(a, b)                                # (B, L, D, N)

    # --- Readout: y_t = sum_N (C_t · h_t)  (+ skip). ---
    y = (C_mat.unsqueeze(2) * h).sum(dim=-1)                       # (B, L, D)
    if D_skip is not None:
        y = y + D_skip * x

    if return_states:
        return y, h
    return y
