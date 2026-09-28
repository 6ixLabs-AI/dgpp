#include "kernels/qwen_norm.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/qwen_norm.cuh"

namespace dgpp {
namespace {

// One block per row; the (1 + w) norm with a single rounding.
__global__ void rmsnorm_kernel(const uint16_t* __restrict__ x,
                               const uint16_t* __restrict__ weight,
                               uint16_t* __restrict__ y, int dim, float eps) {
  extern __shared__ float staged[];
  __shared__ float warp_sums[32];
  const int64_t row = blockIdx.x;
  const uint16_t* xr = x + row * dim;
  uint16_t* yr = y + row * dim;
  const float total = qwen_norm::block_sum_squares(xr, dim, staged, warp_sums);
  const float rstd = rsqrtf(total / static_cast<float>(dim) + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float w1 = 1.0f + bf16_bits_to_float(weight[i]);
    yr[i] = float_to_bf16_bits(staged[i] * rstd * w1);
  }
}

// One block per (row, group): the group's own statistics, the weight
// indexed by the flat position in the row.
__global__ void group_rmsnorm_kernel(const uint16_t* __restrict__ x,
                                     const uint16_t* __restrict__ weight,
                                     uint16_t* __restrict__ y, int groups,
                                     int group_dim, float eps) {
  extern __shared__ float staged[];
  __shared__ float warp_sums[32];
  const int64_t row = blockIdx.x;
  const int g = blockIdx.y;
  const int64_t base = row * static_cast<int64_t>(groups) * group_dim +
                       static_cast<int64_t>(g) * group_dim;
  const uint16_t* xr = x + base;
  uint16_t* yr = y + base;
  const uint16_t* wr = weight + static_cast<int64_t>(g) * group_dim;
  const float total = qwen_norm::block_sum_squares(xr, group_dim, staged, warp_sums);
  const float rstd = rsqrtf(total / static_cast<float>(group_dim) + eps);
  for (int i = threadIdx.x; i < group_dim; i += blockDim.x) {
    const float w1 = 1.0f + bf16_bits_to_float(wr[i]);
    yr[i] = float_to_bf16_bits(staged[i] * rstd * w1);
  }
}

__global__ void gated_rmsnorm_kernel(const uint16_t* __restrict__ x,
                                     const uint16_t* __restrict__ gate,
                                     const uint16_t* __restrict__ weight,
                                     uint16_t* __restrict__ y, int dim, float eps) {
  extern __shared__ float staged[];
  __shared__ float warp_sums[32];
  const int64_t row = blockIdx.x;
  const uint16_t* xr = x + row * dim;
  const uint16_t* gr = gate + row * dim;
  uint16_t* yr = y + row * dim;
  const float total = qwen_norm::block_sum_squares(xr, dim, staged, warp_sums);
  // The reference: 1 / sqrt(mean + eps) — rsqrt's rounding differs from
  // 1/sqrtf by an ulp on some inputs; keep the division form.
  const float rstd = 1.0f / sqrtf(total / static_cast<float>(dim) + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float u = bf16_bits_to_float(float_to_bf16_bits(staged[i] * rstd));
    const float p = bf16_bits_to_float(float_to_bf16_bits(u * bf16_bits_to_float(weight[i])));
    const float g = bf16_bits_to_float(gr[i]);
    const float sig = 1.0f / (1.0f + expf(-g));
    yr[i] = float_to_bf16_bits(p * sig);
  }
}

void check(int64_t rows, int dim, const char* who) {
  if (rows <= 0 || dim <= 0) throw std::invalid_argument(std::string(who) + ": empty problem");
  if (dim > 12288) throw std::invalid_argument(std::string(who) + ": dim exceeds the smem stage");
}

}  // namespace

void qwen_rmsnorm_bf16(const void* x, const void* weight, void* y, int64_t rows,
                       int dim, float eps, cudaStream_t stream) {
  check(rows, dim, "qwen rmsnorm");
  constexpr int kBlock = 256;
  const size_t shmem = sizeof(float) * static_cast<size_t>(dim);
  cudaGetLastError();
  rmsnorm_kernel<<<static_cast<unsigned>(rows), kBlock, shmem, stream>>>(
      static_cast<const uint16_t*>(x), static_cast<const uint16_t*>(weight),
      static_cast<uint16_t*>(y), dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void qwen_group_rmsnorm_bf16(const void* x, const void* weight, void* y,
                             int64_t rows, int groups, int group_dim, float eps,
                             cudaStream_t stream) {
  check(rows, group_dim, "qwen group rmsnorm");
  if (groups <= 0) throw std::invalid_argument("qwen group rmsnorm: no groups");
  constexpr int kBlock = 256;
  const size_t shmem = sizeof(float) * static_cast<size_t>(group_dim);
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(groups), 1);
  cudaGetLastError();
  group_rmsnorm_kernel<<<grid, kBlock, shmem, stream>>>(
      static_cast<const uint16_t*>(x), static_cast<const uint16_t*>(weight),
      static_cast<uint16_t*>(y), groups, group_dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

void gdn_gated_rmsnorm_bf16(const void* x, const void* gate, const void* weight,
                             void* y, int64_t rows, int dim, float eps,
                             cudaStream_t stream) {
  check(rows, dim, "gdn gated rmsnorm");
  constexpr int kBlock = 128;
  const size_t shmem = sizeof(float) * static_cast<size_t>(dim);
  cudaGetLastError();
  gated_rmsnorm_kernel<<<static_cast<unsigned>(rows), kBlock, shmem, stream>>>(
      static_cast<const uint16_t*>(x), static_cast<const uint16_t*>(gate),
      static_cast<const uint16_t*>(weight), static_cast<uint16_t*>(y), dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

__global__ void gated_rmsnorm_swish_kernel(const uint16_t* __restrict__ x,
                                           const uint16_t* __restrict__ gate,
                                           const uint16_t* __restrict__ weight,
                                           uint16_t* __restrict__ y, int dim, float eps) {
  extern __shared__ float staged[];
  __shared__ float warp_sums[32];
  const int64_t row = blockIdx.x;
  const uint16_t* xr = x + row * dim;
  const uint16_t* gr = gate + row * dim;
  uint16_t* yr = y + row * dim;
  const float total = qwen_norm::block_sum_squares(xr, dim, staged, warp_sums);
  // Same division form as the sigmoid kernel (1/sqrtf, not rsqrtf).
  const float rstd = 1.0f / sqrtf(total / static_cast<float>(dim) + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float u = bf16_bits_to_float(float_to_bf16_bits(staged[i] * rstd));
    const float p = bf16_bits_to_float(float_to_bf16_bits(u * bf16_bits_to_float(weight[i])));
    const float g = bf16_bits_to_float(gr[i]);
    const float sig = 1.0f / (1.0f + expf(-g));
    yr[i] = float_to_bf16_bits(p * (g * sig));
  }
}

void gdn_gated_rmsnorm_swish_bf16(const void* x, const void* gate, const void* weight,
                                  void* y, int64_t rows, int dim, float eps,
                                  cudaStream_t stream) {
  check(rows, dim, "gdn gated rmsnorm swish");
  constexpr int kBlock = 128;
  const size_t shmem = sizeof(float) * static_cast<size_t>(dim);
  cudaGetLastError();
  gated_rmsnorm_swish_kernel<<<static_cast<unsigned>(rows), kBlock, shmem, stream>>>(
      static_cast<const uint16_t*>(x), static_cast<const uint16_t*>(gate),
      static_cast<const uint16_t*>(weight), static_cast<uint16_t*>(y), dim, eps);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
