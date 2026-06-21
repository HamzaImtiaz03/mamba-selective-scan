"""CPU simulation of the CUDA forward Blelloch scan algorithm.

This mirrors, step for step, the algorithm in csrc/scan_fwd_kernel.cu (up-sweep,
set-root-identity, exclusive down-sweep, inclusive conversion with cross-chunk carry).
It lets us verify the *algorithm* on CPU — independently of nvcc and a GPU — so a logic
bug (such as a swapped operand in the non-commutative down-sweep combine) is caught here
rather than only surfacing on Colab.

The combine operator is non-commutative:  (aL,bL) o (aR,bR) = (aL*aR, aR*bL + bR).
"""

import numpy as np
import pytest


def _combine(aL, bL, aR, bR):
    return aL * aR, aR * bL + bR


def blelloch_lane(a, b, TILE):
    """Chunked Blelloch inclusive scan for one (b,d) lane, matching the CUDA kernel.

    Args:
        a, b: (L, N) per-timestep recurrence coefficients.
        TILE: power-of-two chunk size (the CUDA block width).
    Returns:
        h: (L, N) hidden states.
    """
    L, N = a.shape
    h = np.zeros((L, N))
    cA, cB = np.ones(N), np.zeros(N)                      # carry = identity (1, 0)

    for ts in range(0, L, TILE):
        sa, sb = np.ones((TILE, N)), np.zeros((TILE, N))  # scan buffers
        ea, eb = np.ones((TILE, N)), np.zeros((TILE, N))  # original elements (identity pad)
        for p in range(TILE):
            t = ts + p
            if t < L:
                sa[p], sb[p], ea[p], eb[p] = a[t], b[t], a[t], b[t]

        # up-sweep (reduce)
        stride = 1
        while stride < TILE:
            k = 0
            while True:
                idx = (k + 1) * 2 * stride - 1
                if idx >= TILE:
                    break
                left = idx - stride
                sa[idx], sb[idx] = _combine(sa[left], sb[left], sa[idx], sb[idx])
                k += 1
            stride *= 2

        # set root to identity, then exclusive down-sweep
        sa[TILE - 1], sb[TILE - 1] = 1.0, 0.0
        stride = TILE // 2
        while stride >= 1:
            k = 0
            while True:
                idx = (k + 1) * 2 * stride - 1
                if idx >= TILE:
                    break
                left = idx - stride
                tA, tB = sa[left].copy(), sb[left].copy()      # left subtree total
                xA, xB = sa[idx].copy(), sb[idx].copy()        # parent prefix
                sa[left], sb[left] = xA, xB
                # combine(parent_prefix, left_total): parent is the EARLIER segment
                sa[idx], sb[idx] = xA * tA, tA * xB + tB
                k += 1
            stride //= 2

        # inclusive value with carry, then update carry by the full-tile total
        for p in range(TILE):
            t = ts + p
            if t < L:
                pA, pB = _combine(cA, cB, sa[p], sb[p])
                _, iB = _combine(pA, pB, ea[p], eb[p])
                h[t] = iB
        totA, totB = _combine(sa[TILE - 1], sb[TILE - 1], ea[TILE - 1], eb[TILE - 1])
        cA, cB = _combine(cA, cB, totA, totB)
    return h


def _reference(a, b):
    L, N = a.shape
    h = np.zeros((L, N))
    prev = np.zeros(N)
    for t in range(L):
        prev = a[t] * prev + b[t]
        h[t] = prev
    return h


@pytest.mark.parametrize("L,N,TILE", [
    (7, 4, 8), (128, 16, 256), (256, 16, 256), (300, 8, 128), (1000, 16, 256),
])
def test_blelloch_matches_reference(L, N, TILE):
    rng = np.random.default_rng(L * 100 + N + TILE)
    delta = np.log1p(np.exp(rng.standard_normal((L, 1))))     # softplus > 0
    A = -np.exp(rng.standard_normal((1, N)))                  # < 0
    Bm = rng.standard_normal((L, N))
    x = rng.standard_normal((L, 1))
    a = np.exp(delta * A)
    b = delta * Bm * x

    h_ref = _reference(a, b)
    h_blel = blelloch_lane(a, b, TILE)
    err = np.abs(h_ref - h_blel).max()
    assert err < 1e-10, f"Blelloch scan algorithm wrong: L={L} N={N} TILE={TILE} err={err:.2e}"
