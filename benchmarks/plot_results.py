"""plot_results.py — render the two headline figures from the benchmark JSON.

    benchmarks/figures/memory_vs_length.png   (log-y VRAM vs L; scan flat, attn OOMs)
    benchmarks/figures/latency_vs_length.png  (median latency vs L per backend)

Usage:  python benchmarks/plot_results.py
Run bench_memory.py and bench_latency.py first to produce the JSON.
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import load_json, FIGURES_DIR, ensure_dirs


def plot_memory():
    try:
        data = load_json("memory.json")
    except FileNotFoundError:
        print("[plot] memory.json not found; run bench_memory.py first")
        return
    L = data["lengths"]
    scan = data["scan_mb"]
    attn = data["attn_mb"]

    def split(series):
        xs_ok, ys_ok, xs_oom = [], [], []
        for x, y in zip(L, series):
            if isinstance(y, (int, float)) and y >= 0:
                xs_ok.append(x); ys_ok.append(y)
            elif y == "OOM":
                xs_oom.append(x)
        return xs_ok, ys_ok, xs_oom

    sx, sy, _ = split(scan)
    ax_, ay, a_oom = split(attn)

    fig, ax = plt.subplots(figsize=(7, 5))
    if sy:
        ax.plot(sx, sy, "o-", label="selective scan (ours, O(L) mem)", color="#1f77b4")
    if ay:
        ax.plot(ax_, ay, "s-", label="softmax attention (O(L²) mem)", color="#d62728")
    for x in a_oom:
        ax.axvline(x, ls=":", color="#d62728", alpha=0.4)
        ax.text(x, ax.get_ylim()[1] if ay else 1, "attn OOM", rotation=90,
                va="top", ha="right", color="#d62728", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sequence length L"); ax.set_ylabel("peak forward VRAM (MB)")
    title = f"Peak memory vs L — {data['info'].get('name','?')}"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if sy or ay:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no VRAM data (CPU run)\nrun on Colab T4", ha="center",
                va="center", transform=ax.transAxes, color="gray")
    ensure_dirs()
    out = os.path.join(FIGURES_DIR, "memory_vs_length.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_latency():
    try:
        data = load_json("latency.json")
    except FileNotFoundError:
        print("[plot] latency.json not found; run bench_latency.py first")
        return
    L = data["lengths"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, ms in data["ms"].items():
        xs = [x for x, m in zip(L, ms) if m is not None]
        ys = [m for m in ms if m is not None]
        if xs:
            ax.plot(xs, ys, "o-", label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sequence length L"); ax.set_ylabel("median forward latency (ms)")
    ax.set_title(f"Latency vs L — {data['info'].get('name','?')}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ensure_dirs()
    out = os.path.join(FIGURES_DIR, "latency_vs_length.png")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    plot_memory()
    plot_latency()
