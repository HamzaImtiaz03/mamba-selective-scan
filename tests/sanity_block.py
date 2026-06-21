"""Stage 6 sanity: the full Mamba block runs end-to-end with correct shapes, and
gradients flow through the whole pipeline into every parameter.

Run directly:  python tests/sanity_block.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mamba_scan.mamba_block import MambaBlock


def test_forward_shapes():
    torch.manual_seed(0)
    for backend in ("reference", "parallel", "triton"):  # triton -> CPU fallback here
        for (B, L, d_model) in [(2, 16, 32), (1, 7, 8), (3, 64, 48)]:
            block = MambaBlock(d_model=d_model, d_state=8, backend=backend).double()
            u = torch.randn(B, L, d_model, dtype=torch.float64)
            y = block(u)
            assert y.shape == (B, L, d_model), f"{backend} {(B,L,d_model)} -> {tuple(y.shape)}"
    print("[ok] forward shapes (B,L,d_model) across backends & sizes")


def test_backends_agree():
    """reference, parallel, and triton(CPU-fallback) must give identical block output."""
    from mamba_scan.mamba_block import _get_backend
    torch.manual_seed(1)
    d_model = 32
    u = torch.randn(2, 24, d_model, dtype=torch.float64)
    # One block instance, swap only the scan backend so all weights are identical.
    block = MambaBlock(d_model=d_model, d_state=8, backend="reference").double()

    ys = {}
    for backend in ("reference", "parallel", "triton"):
        block._scan = _get_backend(backend)
        ys[backend] = block(u)

    e_par = (ys["reference"] - ys["parallel"]).abs().max().item()
    e_tri = (ys["reference"] - ys["triton"]).abs().max().item()
    assert torch.allclose(ys["reference"], ys["parallel"], atol=1e-9), f"parallel err={e_par}"
    assert torch.allclose(ys["reference"], ys["triton"], atol=1e-9), f"triton err={e_tri}"
    print(f"[ok] backends agree on block output   parallel_err={e_par:.2e}  triton_err={e_tri:.2e}")


def test_grads_flow():
    """Loss.backward() populates a gradient for every parameter (no dead params)."""
    torch.manual_seed(2)
    block = MambaBlock(d_model=24, d_state=8, backend="triton").double()
    u = torch.randn(2, 20, 24, dtype=torch.float64, requires_grad=True)
    y = block(u)
    loss = y.square().mean()
    loss.backward()

    missing = [name for name, p in block.named_parameters() if p.grad is None]
    assert not missing, f"params with no grad: {missing}"
    assert u.grad is not None and torch.isfinite(u.grad).all()
    n_params = sum(p.numel() for p in block.parameters())
    print(f"[ok] grads flow to all {len(list(block.parameters()))} params "
          f"({n_params} scalars); input grad finite")


if __name__ == "__main__":
    print("=" * 64)
    print("STAGE 6 SANITY: full Mamba block end-to-end")
    print("=" * 64)
    test_forward_shapes()
    test_backends_agree()
    test_grads_flow()
    print("-" * 64)
    print("STAGE 6: Mamba block runs end-to-end; grads flow to all params.")
