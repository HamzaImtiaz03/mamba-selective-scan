r"""Stage 5 — JIT-compiled CUDA selective scan, wrapped as an autograd.Function.

This loads ``csrc/`` with ``torch.utils.cpp_extension.load`` (no setup.py needed) and
exposes ``selective_scan_cuda`` / ``SelectiveScanCUDA.apply``.

Forward  : Blelloch work-efficient prefix scan in shared memory (scan_fwd_kernel.cu).
Backward : reverse-scan adjoint (scan_bwd_kernel.cu), porting the float64-gradcheck
           verified formulas in ``backward_math.py``.

Targets Turing sm_75 (Colab T4). Compilation requires nvcc + a CUDA device, so the
extension is built lazily on first use. On a machine without CUDA this module imports
fine but calling the kernel raises a clear error (no silent fallback).
"""

from __future__ import annotations

import os
import warnings
import torch
from torch import Tensor

_EXT = None


def _csrc_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "csrc"))


def _load_extension():
    """JIT-compile and cache the CUDA extension. Raises on failure (never fakes).

    A successful build is cached. A *failed* build is NOT cached: the next call
    retries, so a fixable environment issue (e.g. ninja missing, then installed)
    can be resolved without restarting the Python kernel.
    """
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        raise RuntimeError(
            "selective_scan_cuda requires a CUDA device (Colab T4). None is available.")
    from torch.utils.cpp_extension import load
    csrc = _csrc_dir()
    _EXT = load(   # raises on failure; _EXT stays None so the next call retries
        name="selective_scan_cuda_ext",
        sources=[
            os.path.join(csrc, "selective_scan.cpp"),
            os.path.join(csrc, "scan_fwd_kernel.cu"),
            os.path.join(csrc, "scan_bwd_kernel.cu"),
        ],
        extra_include_paths=[os.path.join(csrc, "include")],
        extra_cuda_cflags=[
            "-O3",
            "-gencode", "arch=compute_75,code=sm_75",  # Turing / T4
            "--use_fast_math",
        ],
        verbose=True,
    )
    return _EXT


class SelectiveScanCUDA(torch.autograd.Function):
    """Autograd Function backed by the JIT CUDA kernels."""

    @staticmethod
    def forward(ctx, x, delta, A, B_mat, C_mat, D_skip):
        ext = _load_extension()
        x = x.contiguous(); delta = delta.contiguous()
        A = A.contiguous(); B_mat = B_mat.contiguous(); C_mat = C_mat.contiguous()
        D_in = D_skip.contiguous() if D_skip is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        y, h = ext.fwd(x, delta, A, B_mat, C_mat, D_in)
        ctx.save_for_backward(x, delta, A, B_mat, C_mat, D_in, h)
        ctx.has_dskip = D_skip is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        ext = _load_extension()
        x, delta, A, B_mat, C_mat, D_in, h = ctx.saved_tensors
        dx, ddelta, dA, dB, dC, dD = ext.bwd(
            dy.contiguous(), x, delta, A, B_mat, C_mat, D_in, h)
        if not ctx.has_dskip:
            dD = None
        return dx, ddelta, dA, dB, dC, dD


def selective_scan_cuda(x, delta, A, B_mat, C_mat, D_skip=None):
    """Public entry point for the CUDA selective scan."""
    return SelectiveScanCUDA.apply(x, delta, A, B_mat, C_mat, D_skip)
