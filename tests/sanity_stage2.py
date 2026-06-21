"""Stage 2 sanity: the pure-torch associative scan must match the oracle exactly.

Run directly:  python tests/sanity_stage2.py
Checks both the output y AND the per-timestep hidden states h_t against the
sequential reference, across a range of (B, L, D, N) including non-power-of-2 L.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from conftest import make_inputs
from mamba_scan.reference import selective_scan_ref, selective_scan_ref_with_states
from mamba_scan.parallel_scan_torch import selective_scan_parallel, combine


def test_combine_associativity():
    """The combine operator must be associative: (X∘Y)∘Z == X∘(Y∘Z)."""
    torch.manual_seed(0)
    shp = (3, 5)
    X = (torch.rand(shp), torch.randn(shp))
    Y = (torch.rand(shp), torch.randn(shp))
    Z = (torch.rand(shp), torch.randn(shp))

    left = combine(*combine(*X, *Y), *Z)        # (X∘Y)∘Z
    right = combine(*X, *combine(*Y, *Z))       # X∘(Y∘Z)
    err = max((left[0] - right[0]).abs().max().item(),
              (left[1] - right[1]).abs().max().item())
    assert torch.allclose(left[0], right[0]) and torch.allclose(left[1], right[1])
    print(f"[ok] combine operator is associative        max_err={err:.2e}")


def test_matches_reference():
    """Parallel scan == sequential reference on y and on hidden states."""
    configs = [
        (1, 1, 1, 1), (2, 7, 8, 4), (2, 8, 8, 4), (1, 64, 16, 16),
        (3, 13, 5, 8), (2, 100, 4, 8),
    ]
    worst_y = 0.0
    worst_h = 0.0
    for (B, L, D, N) in configs:
        for with_dskip in (True, False):
            t = make_inputs(B, L, D, N, dtype=torch.float64, with_dskip=with_dskip, seed=B + L + D + N)
            y_ref, h_ref = selective_scan_ref_with_states(
                t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
            y_par, h_par = selective_scan_parallel(
                t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"], return_states=True)
            ey = (y_ref - y_par).abs().max().item()
            eh = (h_ref - h_par).abs().max().item()
            worst_y, worst_h = max(worst_y, ey), max(worst_h, eh)
            assert torch.allclose(y_ref, y_par, atol=1e-10), f"y mismatch {(B,L,D,N)} dskip={with_dskip} err={ey}"
            assert torch.allclose(h_ref, h_par, atol=1e-10), f"h mismatch {(B,L,D,N)} dskip={with_dskip} err={eh}"
    print(f"[ok] parallel == reference (12 configs)      max_y_err={worst_y:.2e}  max_h_err={worst_h:.2e}")


def test_fp32_tolerance():
    """In fp32 the two implementations should still agree to ~1e-4."""
    t = make_inputs(2, 256, 16, 16, dtype=torch.float32, seed=42)
    y_ref = selective_scan_ref(t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    y_par = selective_scan_parallel(t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    err = (y_ref - y_par).abs().max().item()
    assert torch.allclose(y_ref, y_par, atol=1e-3), f"fp32 mismatch err={err}"
    print(f"[ok] fp32 agreement (L=256)                  max_err={err:.2e}")


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 2 SANITY: pure-torch associative scan vs oracle")
    print("=" * 60)
    test_combine_associativity()
    test_matches_reference()
    test_fp32_tolerance()
    print("-" * 60)
    print("STAGE 2: parallel scan matches the reference.")
