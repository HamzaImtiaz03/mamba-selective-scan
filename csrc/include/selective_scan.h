// selective_scan.h — declarations for the CUDA selective-scan kernels.
//
// Forward: Blelloch work-efficient parallel prefix scan over each L-tile in shared
// memory, carrying the boundary state between tiles. Backward: reverse linear-scan
// adjoint. fp16/fp32 in, fp32 accumulate. Targets Turing sm_75 (Colab T4).
#pragma once

#include <torch/extension.h>
#include <vector>

// Forward. Returns {y (B,L,D), h (B,L,D,N) fp32}. h is saved for the backward.
//   x, delta : (B, L, D)
//   A        : (D, N)
//   B_mat    : (B, L, N)
//   C_mat    : (B, L, N)
//   D_skip   : (D,) or undefined tensor (no skip)
std::vector<at::Tensor> selective_scan_fwd_cuda(
    const at::Tensor& x,
    const at::Tensor& delta,
    const at::Tensor& A,
    const at::Tensor& B_mat,
    const at::Tensor& C_mat,
    const at::Tensor& D_skip);

// Backward. Returns {dx, ddelta, dA, dB_mat, dC_mat, dD_skip}.
//   dy : (B, L, D) upstream gradient
//   h  : (B, L, D, N) fp32 hidden states from the forward
std::vector<at::Tensor> selective_scan_bwd_cuda(
    const at::Tensor& dy,
    const at::Tensor& x,
    const at::Tensor& delta,
    const at::Tensor& A,
    const at::Tensor& B_mat,
    const at::Tensor& C_mat,
    const at::Tensor& D_skip,
    const at::Tensor& h);
