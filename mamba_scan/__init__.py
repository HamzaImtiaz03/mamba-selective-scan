"""mamba_scan — a from-scratch Mamba selective-scan (S6) kernel.

Public API:
    selective_scan_ref          -- sequential ground-truth (reference.py)
    selective_scan_parallel     -- pure-torch associative scan (parallel_scan_torch.py)

Optional GPU implementations (imported lazily; require CUDA):
    SelectiveScanTriton.apply   -- Triton forward+backward (triton_scan.py)
    selective_scan_cuda         -- JIT CUDA kernel wrapper (cuda_scan.py)
"""

from .reference import selective_scan_ref, selective_scan_ref_with_states
from .parallel_scan_torch import selective_scan_parallel
from .backward_math import selective_scan_ref_autograd, SelectiveScanRef
from .triton_scan import selective_scan_triton, SelectiveScanTriton

__all__ = [
    "selective_scan_ref",
    "selective_scan_ref_with_states",
    "selective_scan_parallel",
    "selective_scan_ref_autograd",
    "SelectiveScanRef",
    "selective_scan_triton",
    "SelectiveScanTriton",
]
