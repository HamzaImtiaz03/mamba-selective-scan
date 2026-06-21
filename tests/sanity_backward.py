"""Backward-correctness gate: float64 gradcheck of the ANALYTICAL backward + grad
allclose vs autograd of the pure-torch parallel scan.

Run directly:  python tests/sanity_backward.py

This is grading criterion #1. The analytical backward lives in backward_math.py and
is exercised here through SelectiveScanRef.apply, so gradcheck numerically perturbs
each input and compares against MY formulas (not autograd). The Triton/CUDA kernels
port these same formulas.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from conftest import make_inputs
from mamba_scan.backward_math import selective_scan_ref_autograd
from mamba_scan.parallel_scan_torch import selective_scan_parallel


def test_gradcheck_with_dskip():
    """Primary proof: gradcheck in float64 on tiny sizes, WITH skip connection."""
    t = make_inputs(B=1, L=8, D=4, N=4, dtype=torch.float64, with_dskip=True,
                    requires_grad=True, seed=7)
    inputs = (t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    ok = torch.autograd.gradcheck(selective_scan_ref_autograd, inputs,
                                  atol=1e-6, rtol=1e-4, raise_exception=True)
    assert ok
    print("[ok] gradcheck float64 (B1 L8 D4 N4, +D_skip)   PASSED")


def test_gradcheck_no_dskip():
    """gradcheck WITHOUT the skip connection (D_skip=None path)."""
    t = make_inputs(B=1, L=8, D=4, N=4, dtype=torch.float64, with_dskip=False,
                    requires_grad=True, seed=11)
    inputs = (t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], None)
    ok = torch.autograd.gradcheck(selective_scan_ref_autograd, inputs,
                                  atol=1e-6, rtol=1e-4, raise_exception=True)
    assert ok
    print("[ok] gradcheck float64 (B1 L8 D4 N4, no D_skip) PASSED")


def test_gradcheck_nonpow2():
    """gradcheck at a non-power-of-2 length to catch off-by-one in the reverse scan."""
    t = make_inputs(B=2, L=7, D=3, N=5, dtype=torch.float64, with_dskip=True,
                    requires_grad=True, seed=3)
    inputs = (t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    ok = torch.autograd.gradcheck(selective_scan_ref_autograd, inputs,
                                  atol=1e-6, rtol=1e-4, raise_exception=True)
    assert ok
    print("[ok] gradcheck float64 (B2 L7 D3 N5, +D_skip)   PASSED")


def test_grad_allclose_vs_parallel_autograd():
    """Every input's grad must match autograd of the independent parallel scan.

    The parallel scan (Hillis-Steele) builds its graph entirely from torch ops, so
    autograd differentiates it without any of our custom backward code. Matching it
    cross-checks the analytical backward against a fully independent gradient path.
    """
    worst = {}
    for (B, L, D, N) in [(2, 16, 8, 4), (1, 13, 5, 8), (3, 32, 4, 16)]:
        t = make_inputs(B, L, D, N, dtype=torch.float64, with_dskip=True, seed=B * 100 + L)
        names = ["x", "delta", "A", "B_mat", "C_mat", "D_skip"]

        # Path 1: analytical backward via SelectiveScanRef.
        ins1 = {k: t[k].clone().requires_grad_(True) for k in names}
        y1 = selective_scan_ref_autograd(ins1["x"], ins1["delta"], ins1["A"],
                                         ins1["B_mat"], ins1["C_mat"], ins1["D_skip"])
        g = torch.randn_like(y1)
        y1.backward(g)

        # Path 2: autograd through the pure-torch parallel scan.
        ins2 = {k: t[k].clone().requires_grad_(True) for k in names}
        y2 = selective_scan_parallel(ins2["x"], ins2["delta"], ins2["A"],
                                     ins2["B_mat"], ins2["C_mat"], ins2["D_skip"])
        y2.backward(g)

        for k in names:
            e = (ins1[k].grad - ins2[k].grad).abs().max().item()
            worst[k] = max(worst.get(k, 0.0), e)
            assert torch.allclose(ins1[k].grad, ins2[k].grad, atol=1e-9), \
                f"grad mismatch {k} cfg={(B,L,D,N)} err={e}"
    line = "  ".join(f"{k}:{v:.1e}" for k, v in worst.items())
    print(f"[ok] analytical grads == parallel autograd grads")
    print(f"     max abs err per input -> {line}")


if __name__ == "__main__":
    print("=" * 64)
    print("BACKWARD GATE (grading criterion #1): analytical backward")
    print("=" * 64)
    test_gradcheck_with_dskip()
    test_gradcheck_no_dskip()
    test_gradcheck_nonpow2()
    test_grad_allclose_vs_parallel_autograd()
    print("-" * 64)
    print("BACKWARD GATE: float64 gradcheck + cross-check PASSED.")
