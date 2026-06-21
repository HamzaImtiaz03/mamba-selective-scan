"""Shared test helpers: realistic random input generators for the selective scan.

These build inputs with the *correct Mamba parameterization* so tests exercise the
real numerical regime:
    * A is negative           (A = -exp(A_log), so Abar = exp(delta*A) in (0, 1])
    * delta is positive       (delta = softplus(raw))
This keeps the recurrence stable (decaying), matching how Mamba actually runs.
"""

from __future__ import annotations

import torch


def make_inputs(
    B: int = 2,
    L: int = 16,
    D: int = 8,
    N: int = 4,
    *,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
    with_dskip: bool = True,
    requires_grad: bool = False,
    seed: int = 0,
):
    """Build a realistic set of selective-scan inputs.

    Returns a dict with keys: x, delta, A, B_mat, C_mat, D_skip (D_skip is None if
    ``with_dskip`` is False). ``delta`` is produced via softplus (positive) and ``A``
    via -exp(A_log) (negative), matching Mamba's parameterization.
    """
    g = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=dtype, device=device)

    x = rand(B, L, D)
    # delta = softplus(raw) keeps it strictly positive; offset keeps it away from 0.
    delta_raw = rand(B, L, D) - 1.0
    delta = torch.nn.functional.softplus(delta_raw)
    # A = -exp(A_log) is negative, magnitudes ~ O(1).
    A_log = rand(D, N)
    A = -torch.exp(A_log)
    B_mat = rand(B, L, N)
    C_mat = rand(B, L, N)
    D_skip = rand(D) if with_dskip else None

    tensors = {"x": x, "delta": delta, "A": A, "B_mat": B_mat, "C_mat": C_mat, "D_skip": D_skip}

    if requires_grad:
        for k, v in tensors.items():
            if v is not None:
                v.requires_grad_(True)

    return tensors
