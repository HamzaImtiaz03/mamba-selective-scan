"""OPTIONAL ahead-of-time build of the CUDA selective-scan extension.

The project normally JIT-compiles via ``mamba_scan/cuda_scan.py`` (no build step). This
setup.py is only for users who prefer a prebuilt extension:

    pip install -e .            # builds selective_scan_cuda_ext for sm_75 (T4)

Requires a CUDA toolkit (nvcc). On CPU-only machines the CUDA extension is skipped.
"""

import os
from setuptools import setup, find_packages

ext_modules = []
cmdclass = {}
try:
    import torch  # noqa: F401
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1":
        csrc = os.path.join(os.path.dirname(__file__), "csrc")
        ext_modules = [
            CUDAExtension(
                name="selective_scan_cuda_ext",
                sources=[
                    os.path.join(csrc, "selective_scan.cpp"),
                    os.path.join(csrc, "scan_fwd_kernel.cu"),
                    os.path.join(csrc, "scan_bwd_kernel.cu"),
                ],
                include_dirs=[os.path.join(csrc, "include")],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "-gencode", "arch=compute_75,code=sm_75", "--use_fast_math"],
                },
            )
        ]
        cmdclass = {"build_ext": BuildExtension}
except Exception as e:  # torch not importable at setup time
    print(f"[setup] skipping CUDA extension: {e}")

setup(
    name="mamba_scan",
    version="0.1.0",
    description="From-scratch Mamba selective-scan (S6) kernel: reference, Triton, CUDA.",
    packages=find_packages(include=["mamba_scan", "mamba_scan.*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
)
