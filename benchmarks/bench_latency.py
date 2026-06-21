"""bench_latency.py — tokens/s vs sequence length for each selective-scan backend.

Compares (where available):
    * sequential reference (the slow oracle)
    * pure-torch parallel scan
    * our Triton kernel
    * our CUDA kernel
    * official mamba-ssm, IF it imports on this runtime (it frequently needs a matching
      CUDA build; if it won't install we say so and skip it — never fabricated).

CUDA events, warmup, median of 30. Writes benchmarks/results/latency.json.

Usage:
    python benchmarks/bench_latency.py            # auto
    python benchmarks/bench_latency.py --smoke    # tiny, for CPU smoke test
"""

from __future__ import annotations

import argparse
import torch

from _common import device_info, cuda_time_ms, save_json
from mamba_scan.reference import selective_scan_ref
from mamba_scan.parallel_scan_torch import selective_scan_parallel
from mamba_scan.triton_scan import selective_scan_triton


def _try_official():
    """Return a callable matching our signature backed by official mamba-ssm, or None."""
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except Exception as e:
        print(f"[latency] official mamba-ssm not available: {type(e).__name__}: {e}")
        return None

    def official(x, delta, A, B_mat, C_mat, D_skip):
        # mamba-ssm expects (B, D, L) for u/delta and (B, N, L) for B/C.
        u = x.transpose(1, 2).contiguous()
        d = delta.transpose(1, 2).contiguous()
        Bt = B_mat.transpose(1, 2).contiguous()
        Ct = C_mat.transpose(1, 2).contiguous()
        y = selective_scan_fn(u, d, A, Bt, Ct, D=D_skip, delta_softplus=False)
        return y.transpose(1, 2)

    return official


def make_inputs(B, L, D, N, device, dtype):
    x = torch.randn(B, L, D, device=device, dtype=dtype)
    delta = torch.nn.functional.softplus(torch.randn(B, L, D, device=device, dtype=dtype))
    A = -torch.exp(torch.randn(D, N, device=device, dtype=dtype))
    B_mat = torch.randn(B, L, N, device=device, dtype=dtype)
    C_mat = torch.randn(B, L, N, device=device, dtype=dtype)
    D_skip = torch.randn(D, device=device, dtype=dtype)
    return x, delta, A, B_mat, C_mat, D_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--d_inner", type=int, default=256)
    ap.add_argument("--d_state", type=int, default=16)
    args = ap.parse_args()

    info = device_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    backends = {"reference": selective_scan_ref, "parallel": selective_scan_parallel}
    if device == "cuda":
        backends["triton"] = selective_scan_triton
        try:
            from mamba_scan.cuda_scan import selective_scan_cuda
            backends["cuda"] = selective_scan_cuda
        except Exception as e:
            print(f"[latency] cuda kernel unavailable: {e}")
        official = _try_official()
        if official is not None:
            backends["mamba_ssm_official"] = official

    lengths = [128, 256, 512] if args.smoke else [512, 1024, 2048, 4096, 8192]
    # The sequential reference is O(L) python — cap it so it doesn't dominate runtime.
    ref_cap = 2048

    print(f"device={info}  dtype={dtype}  d_inner={args.d_inner}  d_state={args.d_state}")
    header = "    L | " + " | ".join(f"{k[:12]:>12}" for k in backends)
    print(header)
    print("-" * len(header))

    results = {"info": info, "dtype": str(dtype), "lengths": lengths,
               "d_inner": args.d_inner, "d_state": args.d_state,
               "ms": {k: [] for k in backends}, "tokens_per_s": {k: [] for k in backends}}

    for L in lengths:
        ins = make_inputs(args.batch, L, args.d_inner, args.d_state, device, dtype)
        row = []
        for name, fn in backends.items():
            if name == "reference" and L > ref_cap:
                results["ms"][name].append(None)
                results["tokens_per_s"][name].append(None)
                row.append(f"{'skip':>12}")
                continue
            try:
                ms = cuda_time_ms(lambda: fn(*ins), warmup=3, iters=30 if device == "cuda" else 7)
                toks = args.batch * L / (ms / 1e3)
                results["ms"][name].append(ms)
                results["tokens_per_s"][name].append(toks)
                row.append(f"{ms:12.3f}")
            except Exception as e:
                results["ms"][name].append(None)
                results["tokens_per_s"][name].append(None)
                row.append(f"{'ERR':>12}")
                print(f"  [{name} @ L={L}] {type(e).__name__}: {str(e)[:60]}")
        print(f"{L:>5} | " + " | ".join(row))

    path = save_json("latency.json", results)
    print(f"\nsaved -> {path}  (times are median ms; tokens/s in JSON)")


if __name__ == "__main__":
    main()
