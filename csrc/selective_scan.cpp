// selective_scan.cpp — torch bindings + input validation/dispatch for the CUDA kernels.
#include <torch/extension.h>
#include "include/selective_scan.h"

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

static void check_inputs(const at::Tensor& x, const at::Tensor& delta, const at::Tensor& A,
                         const at::Tensor& B_mat, const at::Tensor& C_mat) {
  CHECK_CUDA(x); CHECK_CUDA(delta); CHECK_CUDA(A); CHECK_CUDA(B_mat); CHECK_CUDA(C_mat);
  CHECK_CONTIG(x); CHECK_CONTIG(delta); CHECK_CONTIG(A); CHECK_CONTIG(B_mat); CHECK_CONTIG(C_mat);
  TORCH_CHECK(x.dim() == 3, "x must be (B,L,D)");
  TORCH_CHECK(A.dim() == 2, "A must be (D,N)");
  const int D = x.size(2), N = A.size(1);
  TORCH_CHECK(A.size(0) == D, "A.size(0) must equal D");
  TORCH_CHECK(B_mat.size(2) == N && C_mat.size(2) == N, "B_mat/C_mat last dim must equal N");
}

std::vector<at::Tensor> fwd(const at::Tensor& x, const at::Tensor& delta, const at::Tensor& A,
                            const at::Tensor& B_mat, const at::Tensor& C_mat,
                            const at::Tensor& D_skip) {
  check_inputs(x, delta, A, B_mat, C_mat);
  return selective_scan_fwd_cuda(x, delta, A, B_mat, C_mat, D_skip);
}

std::vector<at::Tensor> bwd(const at::Tensor& dy, const at::Tensor& x, const at::Tensor& delta,
                            const at::Tensor& A, const at::Tensor& B_mat, const at::Tensor& C_mat,
                            const at::Tensor& D_skip, const at::Tensor& h) {
  check_inputs(x, delta, A, B_mat, C_mat);
  CHECK_CUDA(dy); CHECK_CUDA(h);
  return selective_scan_bwd_cuda(dy.contiguous(), x, delta, A, B_mat, C_mat, D_skip, h);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &fwd, "Selective scan forward (Blelloch, CUDA)");
  m.def("bwd", &bwd, "Selective scan backward (reverse scan, CUDA)");
}
