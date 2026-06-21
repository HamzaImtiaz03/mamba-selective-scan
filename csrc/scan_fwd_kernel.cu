// scan_fwd_kernel.cu — forward selective scan via Blelloch work-efficient prefix scan.
//
// Layout: one CUDA block per (batch b, channel d) lane. The state dim N is a batched
// dimension carried by every thread. Each block walks the sequence L in tiles of
// blockDim.x timesteps; within a tile it runs a Blelloch (work-efficient, O(T)) prefix
// scan in shared memory over the associative combine operator
//
//     (a_l, b_l) o (a_r, b_r) = (a_l*a_r,  a_r*b_l + b_r),   identity = (1, 0)
//
// and carries the boundary state (the scan total) across tiles. Compute is fp32;
// inputs may be fp16 or fp32. h (B,L,D,N) is written to HBM for the backward.
//
// Shared memory: 2 * TILE * N floats (the a- and b-components of the in-tile scan) plus
// a tiny per-N carry. The host picks TILE so 2*TILE*N*4 bytes <= ~45KB (T4 has 48KB/block).

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "include/selective_scan.h"

// combine: fuse earlier segment (l) into later segment (r).
__device__ __forceinline__ void combine(
    float aL, float bL, float aR, float bR, float& ao, float& bo) {
  ao = aL * aR;
  bo = aR * bL + bR;
}

template <typename scalar_t>
__global__ void scan_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ delta,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B_mat,
    const scalar_t* __restrict__ C_mat,
    const scalar_t* __restrict__ D_skip,   // may be nullptr
    scalar_t* __restrict__ y,
    float* __restrict__ h,
    int Bsz, int L, int D, int N, bool has_D) {

  const int bd = blockIdx.x;            // lane (b, d)
  const int b = bd / D;
  const int d = bd % D;
  const int tid = threadIdx.x;
  const int TILE = blockDim.x;          // power of two

  extern __shared__ float smem[];
  float* sa = smem;                     // [TILE*N]
  float* sb = smem + TILE * N;          // [TILE*N]
  float* cA = smem + 2 * TILE * N;      // [N] carry a
  float* cB = cA + N;                   // [N] carry b

  if (tid < N) { cA[tid] = 1.0f; cB[tid] = 0.0f; }   // carry = identity
  __syncthreads();

  const float d_skip = has_D ? static_cast<float>(D_skip[d]) : 0.0f;

  // Thread tid owns timestep `t = tile_start + tid`. Keep this thread's element (a,b)
  // for all n in registers for the inclusive conversion after the exclusive scan.
  for (int tile_start = 0; tile_start < L; tile_start += TILE) {
    const int t = tile_start + tid;
    const bool valid = (t < L);

    // --- Load this thread's element (a_n, b_n) for all n; pad with identity. ---
    // We need them again after the exclusive scan, so stash in shared (sa,sb) AND we
    // recompute cheaply below; here we just fill sa/sb with the element values.
    const float dt = valid ? static_cast<float>(delta[(b * L + t) * D + d]) : 0.0f;
    const float xt = valid ? static_cast<float>(x[(b * L + t) * D + d]) : 0.0f;
    for (int n = 0; n < N; ++n) {
      float a_e, b_e;
      if (valid) {
        const float Adn = static_cast<float>(A[d * N + n]);
        a_e = __expf(dt * Adn);
        b_e = dt * static_cast<float>(B_mat[(b * L + t) * N + n]) * xt;
      } else {
        a_e = 1.0f; b_e = 0.0f;   // identity for padding
      }
      sa[tid * N + n] = a_e;
      sb[tid * N + n] = b_e;
    }
    __syncthreads();

    // ----- Blelloch up-sweep (reduce) over the time axis, per n -----
    for (int stride = 1; stride < TILE; stride <<= 1) {
      const int idx = (tid + 1) * stride * 2 - 1;
      if (idx < TILE) {
        const int left = idx - stride;
        for (int n = 0; n < N; ++n) {
          float ao, bo;
          combine(sa[left * N + n], sb[left * N + n], sa[idx * N + n], sb[idx * N + n], ao, bo);
          sa[idx * N + n] = ao; sb[idx * N + n] = bo;
        }
      }
      __syncthreads();
    }

    // The full-tile reduction now sits at position TILE-1. For the exclusive down-sweep
    // we set the root to identity; the tile total needed for the carry is recomputed at
    // the end from exclusive[TILE-1] o elem[TILE-1] (no need to stash it here).
    if (tid == 0) {
      for (int n = 0; n < N; ++n) { sa[(TILE - 1) * N + n] = 1.0f; sb[(TILE - 1) * N + n] = 0.0f; }
    }
    __syncthreads();

    // ----- Blelloch down-sweep -> exclusive prefix scan, per n -----
    for (int stride = TILE >> 1; stride >= 1; stride >>= 1) {
      const int idx = (tid + 1) * stride * 2 - 1;
      if (idx < TILE) {
        const int left = idx - stride;
        for (int n = 0; n < N; ++n) {
          const float tA = sa[left * N + n], tB = sb[left * N + n];
          const float xA = sa[idx * N + n], xB = sb[idx * N + n];
          sa[left * N + n] = xA; sb[left * N + n] = xB;     // x[left] = x[idx]
          // x[idx] = combine(t, x_idx_old): a = tA*xA, b = xA*tB + xB
          sa[idx * N + n] = tA * xA;
          sb[idx * N + n] = xA * tB + xB;
        }
      }
      __syncthreads();
    }
    // Now sa[p],sb[p] = EXCLUSIVE prefix (within tile, from identity) for position p.

    // ----- Inclusive value with carry, output, and h -----
    if (valid) {
      float y_acc = 0.0f;
      for (int n = 0; n < N; ++n) {
        // recompute this position's element (cheap) for the inclusive conversion
        const float Adn = static_cast<float>(A[d * N + n]);
        const float a_e = __expf(dt * Adn);
        const float b_e = dt * static_cast<float>(B_mat[(b * L + t) * N + n]) * xt;
        // pref = combine(carry, exclusive[p]); incl = combine(pref, elem)
        float pA, pB, iA, iB;
        combine(cA[n], cB[n], sa[tid * N + n], sb[tid * N + n], pA, pB);
        combine(pA, pB, a_e, b_e, iA, iB);
        const float h_tn = iB;                 // hidden state h_{t,n}
        h[((b * L + t) * D + d) * N + n] = h_tn;
        y_acc += static_cast<float>(C_mat[(b * L + t) * N + n]) * h_tn;
      }
      if (has_D) y_acc += d_skip * xt;
      y[(b * L + t) * D + d] = static_cast<scalar_t>(y_acc);
    }
    __syncthreads();

    // ----- Update carry = combine(carry, tile_total) -----
    // tile_total (within tile, from identity) = combine(exclusive[TILE-1], elem[TILE-1]).
    if (tid < N) {
      const int n = tid;
      const int t_last = tile_start + (TILE - 1);
      float a_last, b_last;
      if (t_last < L) {
        const float dtl = static_cast<float>(delta[(b * L + t_last) * D + d]);
        const float xtl = static_cast<float>(x[(b * L + t_last) * D + d]);
        const float Adn = static_cast<float>(A[d * N + n]);
        a_last = __expf(dtl * Adn);
        b_last = dtl * static_cast<float>(B_mat[(b * L + t_last) * N + n]) * xtl;
      } else { a_last = 1.0f; b_last = 0.0f; }
      float totA, totB, newA, newB;
      combine(sa[(TILE - 1) * N + n], sb[(TILE - 1) * N + n], a_last, b_last, totA, totB);
      combine(cA[n], cB[n], totA, totB, newA, newB);
      cA[n] = newA; cB[n] = newB;
    }
    __syncthreads();
  }
}

// ---- host launcher ----
static int pick_tile(int N) {
  // Largest power of two TILE with 2*TILE*N*4 bytes <= ~45000, capped at 256.
  int tile = 256;
  while (tile > 1 && (size_t)2 * tile * N * sizeof(float) > 45000) tile >>= 1;
  return tile;
}

std::vector<at::Tensor> selective_scan_fwd_cuda(
    const at::Tensor& x, const at::Tensor& delta, const at::Tensor& A,
    const at::Tensor& B_mat, const at::Tensor& C_mat, const at::Tensor& D_skip) {
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const int Bsz = x.size(0), L = x.size(1), D = x.size(2), N = A.size(1);
  const bool has_D = D_skip.defined() && D_skip.numel() > 0;

  auto y = at::empty({Bsz, L, D}, x.options());
  auto h = at::empty({Bsz, L, D, N}, x.options().dtype(at::kFloat));

  const int TILE = pick_tile(N);
  const dim3 grid(Bsz * D);
  const dim3 block(TILE);
  const size_t shmem = (size_t)(2 * TILE * N + 2 * N) * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "selective_scan_fwd", [&] {
    scan_fwd_kernel<scalar_t><<<grid, block, shmem, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<scalar_t>(), delta.data_ptr<scalar_t>(), A.data_ptr<scalar_t>(),
        B_mat.data_ptr<scalar_t>(), C_mat.data_ptr<scalar_t>(),
        has_D ? D_skip.data_ptr<scalar_t>() : nullptr,
        y.data_ptr<scalar_t>(), h.data_ptr<float>(),
        Bsz, L, D, N, has_D);
  });
  return {y, h};
}
