"""Stage 3 CPU check: validate the SelectiveScanTriton autograd plumbing.

On a CUDA box this same Function runs the real Triton kernels (validate via
tests/test_forward.py + test_backward.py on Colab). On CPU it must (a) emit a loud
one-time warning, (b) produce the correct forward, and (c) produce correct grads
through the autograd.Function machinery (save_for_backward / dispatch / fallback).
Run directly:  python tests/sanity_triton_dispatch.py
"""

import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from conftest import make_inputs
from mamba_scan import selective_scan_triton, selective_scan_ref
from mamba_scan.triton_scan import _HAS_TRITON


def test_cpu_warns_once():
    """The CPU fallback must warn that it is NOT running the Triton kernel."""
    import mamba_scan.triton_scan as ts
    ts._WARNED_CPU = False
    t = make_inputs(1, 8, 4, 4, dtype=torch.float32, seed=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        selective_scan_triton(t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    assert any("NOT the Triton kernel" in str(x.message) for x in w), "expected explicit fallback warning"
    print(f"[ok] CPU path warns explicitly (triton present in env: {_HAS_TRITON})")


def test_cpu_forward_matches_reference():
    t = make_inputs(2, 32, 8, 8, dtype=torch.float64, seed=2)
    y_ref = selective_scan_ref(t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    y_tri = selective_scan_triton(t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    err = (y_ref - y_tri).abs().max().item()
    assert torch.allclose(y_ref, y_tri, atol=1e-10), f"forward mismatch err={err}"
    print(f"[ok] CPU-dispatch forward == reference            max_err={err:.2e}")


def test_cpu_gradcheck():
    """gradcheck through SelectiveScanTriton.apply on the CPU path (validates plumbing)."""
    t = make_inputs(1, 8, 4, 4, dtype=torch.float64, with_dskip=True, requires_grad=True, seed=5)
    inputs = (t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    ok = torch.autograd.gradcheck(selective_scan_triton, inputs, atol=1e-6, rtol=1e-4)
    assert ok
    print("[ok] CPU-dispatch gradcheck float64               PASSED")


if __name__ == "__main__":
    print("=" * 64)
    print("STAGE 3 (CPU plumbing): SelectiveScanTriton autograd.Function")
    print("=" * 64)
    test_cpu_warns_once()
    test_cpu_forward_matches_reference()
    test_cpu_gradcheck()
    print("-" * 64)
    print("STAGE 3: autograd plumbing OK. Triton kernels validate on Colab T4.")
