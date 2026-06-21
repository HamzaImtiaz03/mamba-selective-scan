"""Stage 1 sanity check: validate the sequential reference against analytic cases.

Run directly:  python tests/sanity_stage1.py
These are closed-form checks that do not depend on any other implementation, so they
establish that the *oracle itself* is trustworthy before we build anything on it.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mamba_scan.reference import selective_scan_ref, selective_scan_ref_with_states


def test_cumsum_identity():
    """With A=0, delta=1, B=1, C=1, N=1, the recurrence reduces to a cumulative sum.

    Abar = exp(delta*A) = exp(0) = 1, Bbar = delta*B = 1, so h_t = h_{t-1} + x_t,
    and y_t = C * h_t = cumsum(x). A clean analytic ground truth.
    """
    B, L, D, N = 2, 6, 3, 1
    x = torch.randn(B, L, D, dtype=torch.float64)
    delta = torch.ones(B, L, D, dtype=torch.float64)
    A = torch.zeros(D, N, dtype=torch.float64)
    B_mat = torch.ones(B, L, N, dtype=torch.float64)
    C_mat = torch.ones(B, L, N, dtype=torch.float64)

    y = selective_scan_ref(x, delta, A, B_mat, C_mat, D_skip=None)
    expected = torch.cumsum(x, dim=1)
    err = (y - expected).abs().max().item()
    assert torch.allclose(y, expected, atol=1e-10), f"cumsum identity failed, err={err}"
    print(f"[ok] cumsum identity (A=0,B=C=1)            max_err={err:.2e}")


def test_single_step():
    """At L=1, h_1 = Bbar_1 * x_1, y_1 = sum_N(C_1 * h_1) + D_skip*x_1. Check by hand."""
    B, L, D, N = 1, 1, 2, 3
    torch.manual_seed(1)
    x = torch.randn(B, L, D, dtype=torch.float64)
    delta = torch.nn.functional.softplus(torch.randn(B, L, D, dtype=torch.float64))
    A = -torch.exp(torch.randn(D, N, dtype=torch.float64))
    B_mat = torch.randn(B, L, N, dtype=torch.float64)
    C_mat = torch.randn(B, L, N, dtype=torch.float64)
    D_skip = torch.randn(D, dtype=torch.float64)

    y = selective_scan_ref(x, delta, A, B_mat, C_mat, D_skip)

    # Manual single-step computation.
    d0 = delta[0, 0]                       # (D,)
    Bbar = d0[:, None] * B_mat[0, 0][None, :]      # (D, N)
    h1 = Bbar * x[0, 0][:, None]                   # (D, N)
    y_manual = (C_mat[0, 0][None, :] * h1).sum(-1) + D_skip * x[0, 0]
    err = (y[0, 0] - y_manual).abs().max().item()
    assert torch.allclose(y[0, 0], y_manual, atol=1e-12), f"single-step failed, err={err}"
    print(f"[ok] single-step (L=1) hand computation     max_err={err:.2e}")


def test_decay_property():
    """Inject input only at t=0, then watch the state decay (A<0 => Abar in (0,1)).

    For t>=1 there is no injection, so h_t = Abar_t * h_{t-1} and |h_t| must shrink
    monotonically toward 0. This actually exercises the decay (unlike B=0 everywhere,
    which leaves h identically 0).
    """
    B, L, D, N = 1, 30, 4, 4
    torch.manual_seed(2)
    x = torch.ones(B, L, D, dtype=torch.float64)
    delta = torch.nn.functional.softplus(torch.randn(B, L, D, dtype=torch.float64)) + 0.5
    A = -torch.exp(torch.randn(D, N, dtype=torch.float64))
    B_mat = torch.zeros(B, L, N, dtype=torch.float64)
    B_mat[:, 0, :] = 1.0                                  # inject only at t=0
    C_mat = torch.ones(B, L, N, dtype=torch.float64)

    _, hs = selective_scan_ref_with_states(x, delta, A, B_mat, C_mat, D_skip=None)
    norms = hs[0].abs().amax(dim=(-2, -1))               # (L,) per-step max magnitude
    injected = norms[0].item()
    final = norms[-1].item()
    # Strictly non-increasing after the injection step.
    diffs = norms[1:] - norms[:-1]
    assert (diffs <= 1e-12).all(), f"state did not decay monotonically: {norms.tolist()}"
    assert injected > 0 and final < injected, f"no decay: {injected} -> {final}"
    print(f"[ok] decay property (inject@t=0): |h_0|={injected:.2e} -> |h_L|={final:.2e}")


def test_shapes():
    """Output shape must be (B, L, D) for a range of sizes."""
    for (B, L, D, N) in [(1, 1, 1, 1), (3, 7, 5, 8), (2, 64, 16, 16)]:
        x = torch.randn(B, L, D, dtype=torch.float64)
        delta = torch.nn.functional.softplus(torch.randn(B, L, D, dtype=torch.float64))
        A = -torch.exp(torch.randn(D, N, dtype=torch.float64))
        B_mat = torch.randn(B, L, N, dtype=torch.float64)
        C_mat = torch.randn(B, L, N, dtype=torch.float64)
        D_skip = torch.randn(D, dtype=torch.float64)
        y = selective_scan_ref(x, delta, A, B_mat, C_mat, D_skip)
        assert y.shape == (B, L, D), f"got {tuple(y.shape)} expected {(B, L, D)}"
    print("[ok] shapes (B,L,D) correct across sizes")


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 1 SANITY: sequential reference (the oracle)")
    print("=" * 60)
    test_cumsum_identity()
    test_single_step()
    test_decay_property()
    test_shapes()
    print("-" * 60)
    print("STAGE 1: all sanity checks passed.")
