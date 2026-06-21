"""Backward correctness (grading criterion #1).

Primary proof: torch.autograd.gradcheck in FLOAT64 on tiny sizes through the
analytical backward (SelectiveScanRef) AND through SelectiveScanTriton's dispatch.
Secondary: grad allclose in fp32 vs autograd of the reference, for every input
(x, delta, A, B_mat, C_mat, D_skip).
"""

import pytest
import torch

from conftest import make_inputs
from mamba_scan.backward_math import selective_scan_ref_autograd
from mamba_scan.parallel_scan_torch import selective_scan_parallel
from mamba_scan.triton_scan import selective_scan_triton

HAS_CUDA = torch.cuda.is_available()

GRADCHECK_FNS = [
    ("ref_autograd", selective_scan_ref_autograd),
    ("triton_dispatch", selective_scan_triton),
]


@pytest.mark.parametrize("name,fn", GRADCHECK_FNS)
@pytest.mark.parametrize("with_dskip", [True, False])
def test_gradcheck_float64(name, fn, with_dskip):
    """Primary proof: float64 gradcheck, B=1 L=8 D=4 N=4."""
    t = make_inputs(1, 8, 4, 4, dtype=torch.float64, with_dskip=with_dskip,
                    requires_grad=True, seed=7)
    inputs = (t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    assert torch.autograd.gradcheck(fn, inputs, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("name,fn", GRADCHECK_FNS)
def test_gradcheck_nonpow2_len(name, fn):
    """Catch reverse-scan off-by-one at non-power-of-2 length."""
    t = make_inputs(2, 7, 3, 5, dtype=torch.float64, with_dskip=True,
                    requires_grad=True, seed=3)
    inputs = (t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    assert torch.autograd.gradcheck(fn, inputs, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("with_dskip", [True, False])
def test_grad_allclose_vs_reference(with_dskip):
    """Every input's grad from the analytical backward matches autograd of the
    independent pure-torch parallel scan, in fp32."""
    names = ["x", "delta", "A", "B_mat", "C_mat", "D_skip"]
    t = make_inputs(2, 48, 8, 8, dtype=torch.float64, with_dskip=with_dskip, seed=21)

    a_ins = {k: (t[k].clone().requires_grad_(True) if t[k] is not None else None) for k in names}
    y_a = selective_scan_ref_autograd(a_ins["x"], a_ins["delta"], a_ins["A"],
                                      a_ins["B_mat"], a_ins["C_mat"], a_ins["D_skip"])
    g = torch.randn_like(y_a)
    y_a.backward(g)

    b_ins = {k: (t[k].clone().requires_grad_(True) if t[k] is not None else None) for k in names}
    y_b = selective_scan_parallel(b_ins["x"], b_ins["delta"], b_ins["A"],
                                  b_ins["B_mat"], b_ins["C_mat"], b_ins["D_skip"])
    y_b.backward(g)

    for k in names:
        if a_ins[k] is None:
            continue
        e = (a_ins[k].grad - b_ins[k].grad).abs().max().item()
        assert torch.allclose(a_ins[k].grad, b_ins[k].grad, atol=1e-9), f"{k}: err={e}"


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA (Colab T4)")
@pytest.mark.parametrize("with_dskip", [True, False])
def test_triton_grads_match_reference_cuda(with_dskip):
    """Real Triton kernel grads vs analytical reference grads, fp32, on CUDA."""
    names = ["x", "delta", "A", "B_mat", "C_mat", "D_skip"]
    t = make_inputs(2, 96, 16, 16, dtype=torch.float32, with_dskip=with_dskip,
                    device="cuda", seed=33)

    tr = {k: (t[k].clone().requires_grad_(True) if t[k] is not None else None) for k in names}
    y_tr = selective_scan_triton(tr["x"], tr["delta"], tr["A"], tr["B_mat"], tr["C_mat"], tr["D_skip"])
    g = torch.randn_like(y_tr)
    y_tr.backward(g)

    rf = {k: (t[k].clone().requires_grad_(True) if t[k] is not None else None) for k in names}
    y_rf = selective_scan_ref_autograd(rf["x"], rf["delta"], rf["A"], rf["B_mat"], rf["C_mat"], rf["D_skip"])
    y_rf.backward(g)

    for k in names:
        if tr[k] is None:
            continue
        e = (tr[k].grad - rf[k].grad).abs().max().item()
        assert torch.allclose(tr[k].grad, rf[k].grad, atol=1e-2, rtol=1e-2), f"{k}: err={e}"
