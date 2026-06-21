"""Ground-truth sequential reference for the Mamba selective-scan (S6).

This module implements the selective state-space recurrence as an explicit
Python ``for`` loop over the sequence length ``L``. It is intentionally slow but
*obviously correct* — every other implementation in this project (pure-torch
associative scan, Triton kernel, CUDA kernel) is validated against this oracle.

Shapes (the single most important thing to get right)
-----------------------------------------------------
    x       : (B, L, D)      input sequence, D = d_inner
    delta   : (B, L, D)      input-dependent step size, already > 0 (post-softplus)
    A       : (D, N)         state-transition parameter; negative (A = -exp(A_log))
    B_mat   : (B, L, N)      input-dependent input matrix (selective)
    C_mat   : (B, L, N)      input-dependent output matrix (selective)
    D_skip  : (D,)           skip-connection parameter (optional)
    -> y    : (B, L, D)      output sequence

Discretization (Mamba's simplified Zero-Order-Hold)
---------------------------------------------------
    Abar_t = exp(delta_t[..., None] * A)          # (B, D, N)
    Bbar_t = delta_t[..., None] * B_mat_t         # (B, D, N)  (B broadcast over D)

Recurrence (h_0 = 0, hidden state h_t has shape (B, D, N))
---------------------------------------------------------
    h_t = Abar_t ⊙ h_{t-1} + Bbar_t ⊙ x_t[..., None]
    y_t = sum_N (C_mat_t ⊙ h_t)                   # contract the state dim N
    y_t = y_t + D_skip ⊙ x_t                      # skip connection
"""

from __future__ import annotations

import torch
from torch import Tensor


def selective_scan_ref(
    x: Tensor,
    delta: Tensor,
    A: Tensor,
    B_mat: Tensor,
    C_mat: Tensor,
    D_skip: Tensor | None = None,
) -> Tensor:
    """Sequential reference implementation of the S6 selective scan.

    Args:
        x:      (B, L, D) input.
        delta:  (B, L, D) positive step sizes (already passed through softplus).
        A:      (D, N) negative state-transition parameter.
        B_mat:  (B, L, N) input-dependent input projection.
        C_mat:  (B, L, N) input-dependent output projection.
        D_skip: (D,) optional skip parameter. If ``None``, the skip term is omitted.

    Returns:
        y: (B, L, D) output sequence.
    """
    B, L, D = x.shape
    N = A.shape[1]
    assert A.shape == (D, N), f"A must be (D, N)=({D},{N}), got {tuple(A.shape)}"
    assert delta.shape == (B, L, D), f"delta must be {(B, L, D)}, got {tuple(delta.shape)}"
    assert B_mat.shape == (B, L, N), f"B_mat must be {(B, L, N)}, got {tuple(B_mat.shape)}"
    assert C_mat.shape == (B, L, N), f"C_mat must be {(B, L, N)}, got {tuple(C_mat.shape)}"

    # Hidden state h: (B, D, N), initialized to zero.
    h = torch.zeros(B, D, N, dtype=x.dtype, device=x.device)
    ys = []

    for t in range(L):
        delta_t = delta[:, t, :]                 # (B, D)
        x_t = x[:, t, :]                          # (B, D)
        B_t = B_mat[:, t, :]                      # (B, N)
        C_t = C_mat[:, t, :]                      # (B, N)

        # Discretization for this timestep.
        # delta_t[..., None]: (B, D, 1); A: (D, N) -> (B, D, N)
        Abar_t = torch.exp(delta_t[..., None] * A)              # (B, D, N)
        # B_t[:, None, :]: (B, 1, N) broadcast over D -> (B, D, N)
        Bbar_t = delta_t[..., None] * B_t[:, None, :]           # (B, D, N)

        # Linear recurrence step.
        h = Abar_t * h + Bbar_t * x_t[..., None]               # (B, D, N)

        # Output contraction over the state dimension N.
        y_t = (C_t[:, None, :] * h).sum(dim=-1)                # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)                                 # (B, L, D)

    if D_skip is not None:
        y = y + D_skip * x                                     # (D,) broadcast over (B, L, D)

    return y


def selective_scan_ref_with_states(
    x: Tensor,
    delta: Tensor,
    A: Tensor,
    B_mat: Tensor,
    C_mat: Tensor,
    D_skip: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Same as :func:`selective_scan_ref` but also returns all hidden states.

    Useful for debugging the parallel scans, which can be checked not just on the
    final output ``y`` but on the per-timestep states ``h_t`` as well.

    Returns:
        y: (B, L, D) output.
        hs: (B, L, D, N) stacked hidden states h_1..h_L.
    """
    B, L, D = x.shape
    N = A.shape[1]
    h = torch.zeros(B, D, N, dtype=x.dtype, device=x.device)
    ys, hs = [], []

    for t in range(L):
        delta_t = delta[:, t, :]
        x_t = x[:, t, :]
        B_t = B_mat[:, t, :]
        C_t = C_mat[:, t, :]

        Abar_t = torch.exp(delta_t[..., None] * A)
        Bbar_t = delta_t[..., None] * B_t[:, None, :]
        h = Abar_t * h + Bbar_t * x_t[..., None]

        ys.append((C_t[:, None, :] * h).sum(dim=-1))
        hs.append(h)

    y = torch.stack(ys, dim=1)
    hs = torch.stack(hs, dim=1)                                # (B, L, D, N)
    if D_skip is not None:
        y = y + D_skip * x
    return y, hs
