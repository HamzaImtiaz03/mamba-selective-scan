"""Edge cases: varying L, N, D, with/without D_skip.

Spec: L in {7, 64, 1000}, N in {8, 16}, D in {16, 64}, +/- D_skip.
On CPU we test the pure-torch parallel scan and the Triton CPU-dispatch path against
the reference. On CUDA, the same matrix exercises the real Triton kernel.
"""

import itertools
import pytest
import torch

from conftest import make_inputs
from mamba_scan.reference import selective_scan_ref
from mamba_scan.parallel_scan_torch import selective_scan_parallel
from mamba_scan.triton_scan import selective_scan_triton

HAS_CUDA = torch.cuda.is_available()

LENGTHS = [7, 64, 1000]
NSTATES = [8, 16]
DINNER = [16, 64]
DSKIP = [True, False]
GRID = list(itertools.product(LENGTHS, NSTATES, DINNER, DSKIP))


def _run(impl, L, N, D, with_dskip, device, dtype, atol):
    t = make_inputs(1, L, D, N, dtype=dtype, with_dskip=with_dskip, device=device,
                    seed=L * 7 + N * 3 + D + int(with_dskip))
    y_ref = selective_scan_ref(t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    y = impl(t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    assert y.shape == (1, L, D)
    err = (y_ref - y).abs().max().item()
    assert torch.allclose(y_ref, y, atol=atol), f"L={L} N={N} D={D} dskip={with_dskip} err={err}"


@pytest.mark.parametrize("L,N,D,with_dskip", GRID)
def test_edge_cpu_parallel(L, N, D, with_dskip):
    # float64 on CPU keeps the long-L (1000) accumulation clean.
    _run(selective_scan_parallel, L, N, D, with_dskip, "cpu", torch.float64, atol=1e-8)


@pytest.mark.parametrize("L,N,D,with_dskip", GRID)
def test_edge_cpu_triton_dispatch(L, N, D, with_dskip):
    _run(selective_scan_triton, L, N, D, with_dskip, "cpu", torch.float64, atol=1e-8)


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA (Colab T4)")
@pytest.mark.parametrize("L,N,D,with_dskip", GRID)
def test_edge_cuda_triton(L, N, D, with_dskip):
    _run(selective_scan_triton, L, N, D, with_dskip, "cuda", torch.float32, atol=2e-3)


def test_minimal_L1():
    """Degenerate single-timestep sequence."""
    _run(selective_scan_parallel, 1, 8, 16, True, "cpu", torch.float64, atol=1e-10)
    _run(selective_scan_triton, 1, 8, 16, True, "cpu", torch.float64, atol=1e-10)

