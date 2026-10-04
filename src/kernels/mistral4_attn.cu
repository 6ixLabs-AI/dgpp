// Mistral-Small-4 attention kernel — UNVERIFIED DRAFT (2026-10-04): see the
// header. Never compiled by nvcc, never run.
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/mistral4_attn.hpp"

namespace dgpp {
namespace {

constexpr int kThreads = 256;

// One thread per element; the row's multiplier is read once per thread.
__global__ void llama4_scale_kernel(uint16_t* __restrict__ q, int64_t q_stride, int width,
                                    const int64_t* __restrict__ pos, int rows,
                                    const float* __restrict__ table, int steps, int original) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<int64_t>(rows) * width) return;
  const int r = static_cast<int>(idx / width);
  const int i = static_cast<int>(idx - static_cast<int64_t>(r) * width);
  const int64_t p = pos[r];
  if (p < 0) return;
  int64_t step = p / original;
  if (step == 0) return;  // the multiplier is exactly 1
  if (step >= steps) step = steps - 1;
  uint16_t* e = q + static_cast<int64_t>(r) * q_stride + i;
  *e = float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(*e), table[step]));
}

}  // namespace

void mistral4_llama4_scale(uint16_t* q, int64_t q_stride, int width, const int64_t* pos, int rows,
                           const float* table, int steps, int original, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!q || !pos || !table) throw std::invalid_argument("mistral4_llama4_scale: null pointer");
  if (width <= 0 || q_stride < width || steps <= 0 || original <= 0)
    throw std::invalid_argument("mistral4_llama4_scale: width / stride / steps / original");
  const int64_t total = static_cast<int64_t>(rows) * width;
  const unsigned blocks = static_cast<unsigned>((total + kThreads - 1) / kThreads);
  llama4_scale_kernel<<<blocks, kThreads, 0, stream>>>(q, q_stride, width, pos, rows, table, steps,
                                                       original);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
