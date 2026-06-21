r"""Stage 6 — a full Mamba block built around our selective-scan kernel.

Pipeline (per the Mamba paper / mamba-ssm reference block):

    u (B,L,d_model)
      -> in_proj           -> x (B,L,d_inner),  z (B,L,d_inner)   [gate]
      -> causal depthwise conv1d over x  -> SiLU
      -> x_proj            -> dt_raw (dt_rank), B_mat (d_state), C_mat (d_state)
      -> dt_proj + softplus -> delta (B,L,d_inner)  (positive)
      -> SELECTIVE SCAN(x, delta, A, B_mat, C_mat, D)  -> y (B,L,d_inner)
      -> y * SiLU(z)       [gating]
      -> out_proj          -> (B,L,d_model)

The selective scan is pluggable via ``backend`` so the same block can run the pure
reference, the pure-torch parallel scan, the Triton kernel, or the CUDA kernel. The
SSM parameterization matches the kernel contract: ``A = -exp(A_log)`` (negative),
``delta`` positive via softplus.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .reference import selective_scan_ref
from .parallel_scan_torch import selective_scan_parallel
from .triton_scan import selective_scan_triton


def _get_backend(name: str):
    if name == "reference":
        return selective_scan_ref
    if name == "parallel":
        return selective_scan_parallel
    if name == "triton":
        return selective_scan_triton
    if name == "cuda":
        from .cuda_scan import selective_scan_cuda
        return selective_scan_cuda
    raise ValueError(f"unknown backend {name!r}; use reference|parallel|triton|cuda")


class MambaBlock(nn.Module):
    """A single Mamba (S6) block.

    Args:
        d_model:  model width.
        d_state:  SSM state dim N (4..16 typical).
        d_conv:   causal conv kernel size.
        expand:   inner expansion factor (d_inner = expand * d_model).
        dt_rank:  rank of the low-rank dt projection ("auto" -> ceil(d_model/16)).
        backend:  selective-scan implementation: reference|parallel|triton|cuda.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_rank: int | str = "auto", bias: bool = False, conv_bias: bool = True,
                 backend: str = "triton"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.backend = backend
        self._scan = _get_backend(backend)

        # in_proj produces x and the gate z.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # Causal depthwise conv1d over the inner channels.
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=conv_bias)

        # x_proj -> (dt_raw, B, C); dt_proj lifts dt_rank -> d_inner.
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A parameterized as A = -exp(A_log) (negative, stable). Shape (d_inner, d_state).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))   # skip connection

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, d_model) -> (B, L, d_model)."""
        B, L, _ = u.shape

        xz = self.in_proj(u)                                  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)                            # each (B, L, d_inner)

        # Causal depthwise conv: (B, d_inner, L), trim the right padding to keep length L.
        x = x.transpose(1, 2)                                 # (B, d_inner, L)
        x = self.conv1d(x)[..., :L]                          # causal: drop look-ahead
        x = x.transpose(1, 2)                                 # (B, L, d_inner)
        x = F.silu(x)

        # Input-dependent dt, B, C.
        x_dbl = self.x_proj(x)                                # (B, L, dt_rank + 2N)
        dt_raw, B_mat, C_mat = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_raw))             # (B, L, d_inner), positive

        A = -torch.exp(self.A_log.float())                   # (d_inner, d_state), negative

        # The selective scan: (B,L,d_inner).
        y = self._scan(x, delta, A, B_mat.contiguous(), C_mat.contiguous(), self.D)

        y = y * F.silu(z)                                    # gated output
        return self.out_proj(y)                              # (B, L, d_model)
