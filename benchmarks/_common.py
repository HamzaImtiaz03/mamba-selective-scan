"""Shared benchmark utilities: device/timing helpers and a softmax-attention baseline."""

from __future__ import annotations

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def save_json(name: str, obj: dict):
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def load_json(name: str) -> dict:
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def device_info():
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return {"device": "cuda", "name": p.name,
                "total_mem_gb": round(p.total_memory / 1e9, 2),
                "capability": f"sm_{p.major}{p.minor}"}
    return {"device": "cpu", "name": "cpu", "total_mem_gb": None, "capability": None}


def cuda_time_ms(fn, warmup: int = 5, iters: int = 30) -> float:
    """Median wall time of fn() in ms using CUDA events (or perf_counter on CPU)."""
    if torch.cuda.is_available():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        return times[len(times) // 2]
    else:
        for _ in range(max(1, warmup // 2)):
            fn()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        return times[len(times) // 2]


def peak_mem_mb(fn) -> float | None:
    """Run fn() and return peak CUDA memory in MB (None on CPU)."""
    if not torch.cuda.is_available():
        fn()
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


class AttentionBaseline(nn.Module):
    """Minimal softmax self-attention of equal width — the O(L^2)-memory comparator.

    Materializes the (L, L) attention matrix, so peak memory grows quadratically with L
    and OOMs on a T4 well before the linear-memory scan does.
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, u):
        B, L, Dm = u.shape
        qkv = self.qkv(u).reshape(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # (B, h, L, dh)
        # Explicit scores to make the O(L^2) memory cost real (no flash-attention fusion).
        scores = (q @ k.transpose(-1, -2)) / (self.dh ** 0.5)  # (B, h, L, L)
        attn = F.softmax(scores, dim=-1)
        causal = torch.tril(torch.ones(L, L, device=u.device, dtype=torch.bool))
        attn = attn.masked_fill(~causal, 0.0)
        y = attn @ v                                          # (B, h, L, dh)
        y = y.transpose(1, 2).reshape(B, L, Dm)
        return self.out(y)
