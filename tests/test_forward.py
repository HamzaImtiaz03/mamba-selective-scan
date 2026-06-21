"""Forward correctness: every implementation vs the sequential reference oracle.

Tolerances (from the project spec): fp32 atol=1e-3, fp16 atol=2e-2.

What runs where:
  * CPU: reference, pure-torch parallel scan, and SelectiveScanTriton's CPU path.
  * CUDA (Colab T4): additionally the real Triton kernel and the JIT CUDA kernel.
"""

import pytest
import torch

from conftest import make_inputs
from mamba_scan.reference import selective_scan_ref
from mamba_scan.parallel_scan_torch import selective_scan_parallel
from mamba_scan.triton_scan import selective_scan_triton

HAS_CUDA = torch.cuda.is_available()


def _cuda_impls():
    """Implementations that require a CUDA device (real Triton + CUDA kernels)."""
    impls = [("triton", selective_scan_triton)]
    try:
        from mamba_scan.cuda_scan import selective_scan_cuda  # noqa: F401
        impls.append(("cuda", selective_scan_cuda))
    except Exception:
        pass
    return impls


# ---- CPU-runnable implementations (always available) ----
CPU_IMPLS = [
    ("parallel", selective_scan_parallel),
    ("triton_cpu", selective_scan_triton),   # CPU fallback path
]


@pytest.mark.parametrize("name,impl", CPU_IMPLS)
@pytest.mark.parametrize("with_dskip", [True, False])
def test_forward_fp32_cpu(name, impl, with_dskip):
    t = make_inputs(2, 64, 16, 8, dtype=torch.float32, with_dskip=with_dskip, seed=10)
    y_ref = selective_scan_ref(t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    y = impl(t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    assert torch.allclose(y_ref, y, atol=1e-3), f"{name}: max_err={(y_ref-y).abs().max():.2e}"


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA (Colab T4)")
@pytest.mark.parametrize("name,impl", _cuda_impls())
@pytest.mark.parametrize("with_dskip", [True, False])
def test_forward_fp32_cuda(name, impl, with_dskip):
    t = make_inputs(2, 128, 32, 16, dtype=torch.float32, with_dskip=with_dskip,
                    device="cuda", seed=11)
    y_ref = selective_scan_ref(t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    y = impl(t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"])
    assert torch.allclose(y_ref, y, atol=1e-3), f"{name}: max_err={(y_ref-y).abs().max():.2e}"


@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA (Colab T4)")
@pytest.mark.parametrize("name,impl", _cuda_impls())
def test_forward_fp16_cuda(name, impl):
    t = make_inputs(2, 128, 32, 16, dtype=torch.float16, device="cuda", seed=12)
    # Reference in fp32 for a clean ground truth, compare against fp16 kernel output.
    t32 = {k: (v.float() if v is not None else None) for k, v in t.items()}
    y_ref = selective_scan_ref(t32["x"], t32["delta"], t32["A"], t32["B_mat"], t32["C_mat"], t32["D_skip"])
    y = impl(t["x"], t["delta"], t["A"], t["B_mat"], t["C_mat"], t["D_skip"]).float()
    assert torch.allclose(y_ref, y, atol=2e-2), f"{name}: max_err={(y_ref-y).abs().max():.2e}"
