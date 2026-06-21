// scan_bwd_kernel.cu — backward of the selective scan (the adjoint reverse scan).
//
// The adjoint of a linear scan is a linear scan: gh_t = dh_y_t + a_{t+1}*gh_{t+1}.
// Each thread owns one (b, d) lane and walks the recurrence SEQUENTIALLY in reverse —
// a direct, unambiguous transcription of the float64-gradcheck-verified formulas in
// mamba_scan/backward_math.py. The B*D lanes run in parallel. Parameter grads that
// reduce across lanes (dA over b,t; dB,dC over d; dD over b,t) use atomicAdd into fp32
// accumulators. Compute is fp32; I/O may be fp16 or fp32.
//
//   ddelta[b,t,d] = sum_n( g_bb*B*x )  +  sum_n( g_a * a * A )
//   dx[b,t,d]     = sum_n( g_bb*delta*B )  [+ dy*Dskip]
//   dB[b,t,n]    += sum_d( g_bb*delta*x )
//   dC[b,t,n]    += sum_d( dy*h )
//   dA[d,n]      += sum_{b,t}( g_a * a * delta )
//   dD[d]        += sum_{b,t}( dy*x )
// with g_bb = gh, g_a = gh*h_{t-1}.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "include/selective_scan.h"

template <typename scalar_t>
__global__ void scan_bwd_kernel(
    const scalar_t* __restrict__ dy,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ delta,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B_mat,
    const scalar_t* __restrict__ C_mat,
    const scalar_t* __restrict__ D_skip,    // may be nullptr
    const float* __restrict__ h,
    scalar_t* __restrict__ dx,
    scalar_t* __restrict__ ddelta,
    float* __restrict__ dA,                 // (D,N) fp32 accumulator
    float* __restrict__ dB,                 // (B,L,N) fp32 accumulator
    float* __restrict__ dC,                 // (B,L,N) fp32 accumulator
    float* __restrict__ dD,                 // (D,) fp32 accumulator
    int Bsz, int L, int D, int N, bool has_D, int MAXN) {

  const int bd = blockIdx.x * blockDim.x + threadIdx.x;   // lane (b, d)
  if (bd >= Bsz * D) return;
  const int b = bd / D;
  const int d = bd % D;

  const float d_skip = has_D ? static_cast<float>(D_skip[d]) : 0.0f;

  // Reverse-recurrence carry per n. MAXN bounds the local arrays (>= N).
  float G[32];      // gh_{t+1}
  float mult[32];   // a_{t+1}
  float dA_acc[32]; // local accumulation of dA[d,:]
  for (int n = 0; n < N; ++n) { G[n] = 0.0f; mult[n] = 0.0f; dA_acc[n] = 0.0f; }
  float dD_acc = 0.0f;

  for (int i = 0; i < L; ++i) {
    const int t = L - 1 - i;
    const float dt = static_cast<float>(delta[(b * L + t) * D + d]);
    const float xt = static_cast<float>(x[(b * L + t) * D + d]);
    const float dyt = static_cast<float>(dy[(b * L + t) * D + d]);

    float ddelta_acc = 0.0f;
    float dx_acc = 0.0f;
    for (int n = 0; n < N; ++n) {
      const float Adn = static_cast<float>(A[d * N + n]);
      const float Btn = static_cast<float>(B_mat[(b * L + t) * N + n]);
      const float Ctn = static_cast<float>(C_mat[(b * L + t) * N + n]);
      const float a_t = __expf(dt * Adn);
      const float h_t = h[((b * L + t) * D + d) * N + n];
      const float h_prev = (t >= 1) ? h[((b * L + (t - 1)) * D + d) * N + n] : 0.0f;

      const float dh_y = dyt * Ctn;
      const float gh = dh_y + mult[n] * G[n];     // gh_t
      const float g_bb = gh;
      const float g_a = gh * h_prev;

      ddelta_acc += g_bb * Btn * xt + g_a * a_t * Adn;
      dx_acc     += g_bb * dt * Btn;

      dA_acc[n]  += g_a * a_t * dt;
      atomicAdd(&dB[(b * L + t) * N + n], g_bb * dt * xt);
      atomicAdd(&dC[(b * L + t) * N + n], dyt * h_t);

      mult[n] = a_t;     // advance reverse recurrence
      G[n]    = gh;
    }
    if (has_D) { dx_acc += dyt * d_skip; dD_acc += dyt * xt; }

    ddelta[(b * L + t) * D + d] = static_cast<scalar_t>(ddelta_acc);
    dx[(b * L + t) * D + d]     = static_cast<scalar_t>(dx_acc);
  }

  for (int n = 0; n < N; ++n) atomicAdd(&dA[d * N + n], dA_acc[n]);
  if (has_D) atomicAdd(&dD[d], dD_acc);
}

std::vector<at::Tensor> selective_scan_bwd_cuda(
    const at::Tensor& dy, const at::Tensor& x, const at::Tensor& delta,
    const at::Tensor& A, const at::Tensor& B_mat, const at::Tensor& C_mat,
    const at::Tensor& D_skip, const at::Tensor& h) {
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const int Bsz = x.size(0), L = x.size(1), D = x.size(2), N = A.size(1);
  const bool has_D = D_skip.defined() && D_skip.numel() > 0;
  TORCH_CHECK(N <= 32, "CUDA backward supports N<=32 (got ", N, ")");

  auto dx = at::empty_like(x);
  auto ddelta = at::empty_like(delta);
  // fp32 accumulators for the atomic reductions; cast back to input dtype at the end.
  auto dA = at::zeros({D, N}, x.options().dtype(at::kFloat));
  auto dB = at::zeros({Bsz, L, N}, x.options().dtype(at::kFloat));
  auto dC = at::zeros({Bsz, L, N}, x.options().dtype(at::kFloat));
  auto dD = at::zeros({D}, x.options().dtype(at::kFloat));

  const int threads = 128;
  const int blocks = (Bsz * D + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "selective_scan_bwd", [&] {
    scan_bwd_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        dy.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), delta.data_ptr<scalar_t>(),
        A.data_ptr<scalar_t>(), B_mat.data_ptr<scalar_t>(), C_mat.data_ptr<scalar_t>(),
        has_D ? D_skip.data_ptr<scalar_t>() : nullptr, h.data_ptr<float>(),
        dx.data_ptr<scalar_t>(), ddelta.data_ptr<scalar_t>(),
        dA.data_ptr<float>(), dB.data_ptr<float>(), dC.data_ptr<float>(), dD.data_ptr<float>(),
        Bsz, L, D, N, has_D, N);
  });

  auto dA_o = dA.to(x.dtype());
  auto dB_o = dB.to(x.dtype());
  auto dC_o = dC.to(x.dtype());
  auto dD_o = has_D ? dD.to(x.dtype()) : at::Tensor();
  return {dx, ddelta, dA_o, dB_o, dC_o, dD_o};
}
