"""bench_memory.py — peak VRAM vs sequence length L: linear scan vs quadratic attention.

The headline, honest result: our O(N) selective scan keeps peak memory ~linear in L,
while an equal-width softmax-attention baseline materializes an (L,L) matrix and OOMs
on a T4 as L grows. OOMs are caught and recorded, not hidden.

Usage:
    python benchmarks/bench_memory.py                 # auto: CUDA if available
    python benchmarks/bench_memory.py --smoke         # tiny L, for CPU smoke-testing

Writes benchmarks/results/memory.json (consumed by plot_results.py).
"""

from __future__ import annotations

import argparse
import gc
import torch

from _common import (device_info, peak_mem_mb, save_json, AttentionBaseline)
from mamba_scan.mamba_block import MambaBlock


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def measure(make_model, B, L, d_model, device, dtype):
    """Return peak forward memory (MB) or the string 'OOM'."""
    try:
        model = make_model().to(device=device, dtype=dtype).eval()
        u = torch.randn(B, L, d_model, device=device, dtype=dtype)
        with torch.no_grad():
            mem = peak_mem_mb(lambda: model(u))
        del model, u
        _free()
        return mem if mem is not None else -1.0
    except torch.cuda.OutOfMemoryError:
        _free()
        return "OOM"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            _free()
            return "OOM"
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny lengths for CPU smoke test")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    info = device_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    backend = "triton" if device == "cuda" else "parallel"

    lengths = [128, 256, 512] if args.smoke else [1024, 4096, 16384, 65536, 100000]
    print(f"device={info}  backend={backend}  dtype={dtype}  d_model={args.d_model}")
    print(f"{'L':>8} | {'scan (MB)':>12} | {'attention (MB)':>16}")
    print("-" * 44)

    results = {"info": info, "backend": backend, "d_model": args.d_model,
               "batch": args.batch, "dtype": str(dtype), "lengths": lengths,
               "scan_mb": [], "attn_mb": []}

    for L in lengths:
        scan_mem = measure(
            lambda: MambaBlock(d_model=args.d_model, d_state=16, backend=backend),
            args.batch, L, args.d_model, device, dtype)
        attn_mem = measure(
            lambda: AttentionBaseline(d_model=args.d_model),
            args.batch, L, args.d_model, device, dtype)
        results["scan_mb"].append(scan_mem)
        results["attn_mb"].append(attn_mem)
        s = f"{scan_mem:12.1f}" if isinstance(scan_mem, float) else f"{scan_mem:>12}"
        a = f"{attn_mem:16.1f}" if isinstance(attn_mem, float) else f"{attn_mem:>16}"
        print(f"{L:>8} | {s} | {a}")

    path = save_json("memory.json", results)
    print(f"\nsaved -> {path}")
    if device == "cpu":
        print("NOTE: CPU run reports no VRAM (peak_mem=-1). Run on Colab T4 for real numbers.")


if __name__ == "__main__":
    main()
