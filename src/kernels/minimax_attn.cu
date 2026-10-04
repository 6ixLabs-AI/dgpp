// MiniMax-M2 attention kernels — UNVERIFIED DRAFT (2026-10-04): see the
// header. Never compiled by nvcc, never run.
#include <cmath>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/glm4_attn.hpp"
#include "kernels/minimax_attn.hpp"

namespace dgpp {
namespace {

constexpr int D = kGlm4HeadDim;
constexpr int kRstdThreads = 256;

__device__ __forceinline__ float round_bf16(float v) {
  return bf16_bits_to_float(float_to_bf16_bits(v));
}

// ---- the row's rstd -----------------------------------------------------------
// One block per row. Thread t sums the squares of the bf16-rounded elements
// t, t + 256, ... in ascending order; the partials fold through shared
// memory in a fixed tree (256 -> 128 -> ... -> 1), so the sum is the same
// bits whatever the launch shape.
__global__ void qk_rstd_kernel(const float* __restrict__ dot, int64_t stride, int width,
                               const int64_t* __restrict__ pos, float eps,
                               float* __restrict__ rstd) {
  __shared__ float part[kRstdThreads];
  const int r = static_cast<int>(blockIdx.x);
  if (pos[r] < 0) return;
  const float* row = dot + static_cast<int64_t>(r) * stride;
  float ss = 0.f;
  for (int i = static_cast<int>(threadIdx.x); i < width; i += kRstdThreads) {
    const float x = round_bf16(row[i]);
    ss = __fmaf_rn(x, x, ss);
  }
  part[threadIdx.x] = ss;
  __syncthreads();
  for (int half = kRstdThreads / 2; half > 0; half >>= 1) {
    if (static_cast<int>(threadIdx.x) < half) part[threadIdx.x] += part[threadIdx.x + half];
    __syncthreads();
  }
  if (threadIdx.x == 0) rstd[r] = rsqrtf(part[0] / static_cast<float>(width) + eps);
}

// ---- qkv finish ---------------------------------------------------------------
// glm4_attn.cu's qkv_finish_kernel with the norm changed: one warp per
// (row, head), lane l owns dims [4l, 4l + 4); the rstd is the row's (read,
// not summed here) and the weight is the element's own.
__global__ void qkv_finish_kernel(
    const float* __restrict__ q_dot, int64_t q_stride, const float* __restrict__ k_dot,
    int64_t k_stride, const float* __restrict__ v_dot, int64_t v_stride,
    const uint16_t* __restrict__ q_norm, const uint16_t* __restrict__ k_norm,
    const float* __restrict__ q_rstd, const float* __restrict__ k_rstd,
    const float* __restrict__ inv_freq, int rotary_dim, const int32_t* __restrict__ req_ids,
    const int64_t* __restrict__ pos, int rows, int local_heads, int kv_heads,
    const int32_t* __restrict__ block_tables, int blocks_per_request, int block_tokens,
    uint16_t* __restrict__ q_out, int64_t q_out_stride, uint16_t* __restrict__ k_cache,
    uint16_t* __restrict__ v_cache) {
  const int heads_total = local_heads + 2 * kv_heads;
  const int item = static_cast<int>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  if (item >= rows * heads_total) return;
  const int r = item / heads_total;
  const int hh = item - r * heads_total;
  const int64_t p = pos[r];
  if (p < 0) return;
  const int lane = threadIdx.x % 32;
  const int d0 = lane * 4;
  int kind, h;  // 0 = q, 1 = k, 2 = v
  if (hh < local_heads) {
    kind = 0;
    h = hh;
  } else if (hh < local_heads + kv_heads) {
    kind = 1;
    h = hh - local_heads;
  } else {
    kind = 2;
    h = hh - local_heads - kv_heads;
  }
  const float* src = kind == 0   ? q_dot + static_cast<int64_t>(r) * q_stride + h * D
                     : kind == 1 ? k_dot + static_cast<int64_t>(r) * k_stride + h * D
                                 : v_dot + static_cast<int64_t>(r) * v_stride + h * D;
  const uint16_t* norm = kind == 0 ? q_norm : kind == 1 ? k_norm : nullptr;
  // The Linear's rounding (no bias).
  float x[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = round_bf16(src[d0 + i]);
  if (norm != nullptr) {
    // The per-layer norm: the row's rstd, the element's weight.
    const float rstd = kind == 0 ? q_rstd[r] : k_rstd[r];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float u = round_bf16(x[i] * rstd);
      x[i] = round_bf16(bf16_bits_to_float(norm[h * D + d0 + i]) * u);
    }
    // RoPE on dims [0, rotary_dim): pair i = (i, i + half), the partner
    // lane half/4 lanes away (glm4_attn.cu's shuffle).
    if (rotary_dim > 0) {
      const int half = rotary_dim / 2;
      const int lane_off = half / 4;
      float mate[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) mate[j] = __shfl_xor_sync(~0u, x[j], lane_off);
      if (d0 < rotary_dim) {
        const bool first = d0 < half;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i = (first ? d0 : d0 - half) + j;  // the pair index
          const float ang = __fmul_rn(static_cast<float>(p), inv_freq[i]);
          const float c = round_bf16(cosf(ang));
          const float s = round_bf16(sinf(ang));
          x[j] = first ? round_bf16(round_bf16(x[j] * c) + round_bf16(-mate[j] * s))
                       : round_bf16(round_bf16(x[j] * c) + round_bf16(mate[j] * s));
        }
      }
    }
  }
  uint16_t packed[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) packed[i] = float_to_bf16_bits(x[i]);
  uint2 word;
  word.x = static_cast<uint32_t>(packed[0]) | (static_cast<uint32_t>(packed[1]) << 16);
  word.y = static_cast<uint32_t>(packed[2]) | (static_cast<uint32_t>(packed[3]) << 16);
  if (kind == 0) {
    *reinterpret_cast<uint2*>(q_out + static_cast<int64_t>(r) * q_out_stride + h * D + d0) = word;
    return;
  }
  const int32_t blk =
      block_tables[static_cast<int64_t>(req_ids[r]) * blocks_per_request + p / block_tokens];
  const int64_t phys = static_cast<int64_t>(blk) * block_tokens + p % block_tokens;
  uint16_t* cache = kind == 1 ? k_cache : v_cache;
  *reinterpret_cast<uint2*>(cache + phys * (static_cast<int64_t>(kv_heads) * D) + h * D + d0) =
      word;
}

}  // namespace

void minimax_qk_rstd(const float* dot, int64_t stride, int width, const int64_t* pos, int rows,
                     float eps, float* rstd, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!dot || !pos || !rstd) throw std::invalid_argument("minimax_qk_rstd: null pointer");
  if (width <= 0 || stride < width) throw std::invalid_argument("minimax_qk_rstd: width / stride");
  qk_rstd_kernel<<<rows, kRstdThreads, 0, stream>>>(dot, stride, width, pos, eps, rstd);
  DGPP_CUDA_OK(cudaGetLastError());
}

void minimax_qkv_finish(const float* q_dot, int64_t q_stride, const float* k_dot, int64_t k_stride,
                        const float* v_dot, int64_t v_stride, const uint16_t* q_norm,
                        const uint16_t* k_norm, const float* q_rstd, const float* k_rstd,
                        const float* inv_freq, int rotary_dim, const int32_t* req_ids,
                        const int64_t* pos, int rows, int local_heads, int kv_heads,
                        const int32_t* block_tables, int blocks_per_request, int block_tokens,
                        uint16_t* q_out, int64_t q_out_stride, uint16_t* k_cache, uint16_t* v_cache,
                        cudaStream_t stream) {
  if (rows <= 0) return;
  if (!q_dot || !k_dot || !v_dot || !q_norm || !k_norm || !q_rstd || !k_rstd || !req_ids || !pos ||
      !block_tables || !q_out || !k_cache || !v_cache)
    throw std::invalid_argument("minimax_qkv_finish: null pointer");
  if (local_heads <= 0 || kv_heads <= 0 || local_heads % kv_heads != 0)
    throw std::invalid_argument("minimax_qkv_finish: heads");
  if (rotary_dim < 0 || rotary_dim > D || rotary_dim % 8 != 0 || (rotary_dim > 0 && !inv_freq))
    throw std::invalid_argument(
        "minimax_qkv_finish: rotary_dim (a multiple of 8, at most head_dim)");
  const int items = rows * (local_heads + 2 * kv_heads);
  const int blocks = (items + 7) / 8;
  qkv_finish_kernel<<<blocks, 256, 0, stream>>>(
      q_dot, q_stride, k_dot, k_stride, v_dot, v_stride, q_norm, k_norm, q_rstd, k_rstd, inv_freq,
      rotary_dim, req_ids, pos, rows, local_heads, kv_heads, block_tables, blocks_per_request,
      block_tokens, q_out, q_out_stride, k_cache, v_cache);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
