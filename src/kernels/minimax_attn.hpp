#pragma once
// MiniMax-M2 attention kernels — UNVERIFIED DRAFT (2026-10-04,
// docs/minimax_m27_plan.md §3). Written without a GPU: never compiled by
// nvcc, never run. Built only under -DDGPP_BUILD_MINIMAX_DRAFT=ON; the
// first thing to do with it is tests/cuda/minimax_attn_test.cu against the
// host reference (models/minimax/attn_reference.hpp), which IS tested.
//
// The model's attention is GLM-4.7's (kernels/glm4_attn.hpp) with one step
// changed: the q / k norm is one RMS over the WHOLE projection row — every
// head — with a weight per element, where GLM-4.7 norms each head alone.
// Two kernels replace glm4_qkv_finish; the paged split-KV attention and its
// combine (glm4_attn_partial, glm4_attn_combine) are used unchanged.
//
//   qk rstd      per row: rstd = rsqrt(sum_i bf16(dot_i)^2 / width + eps)
//                over all `width` elements of the row's projection (every
//                head — under tensor parallelism the rank holds the whole
//                q_proj / k_proj for exactly this). fp32, a fixed order:
//                thread t of the row's block sums its strided elements
//                ascending, then a fixed shared-memory tree. The host
//                reference sums in double; the two agree to fp32 rounding;
//   qkv finish   per (row, head) of the rank's slice: bf16(dot), the
//                two-rounding norm with the row's rstd and the element's
//                own weight (u = bf16(x * rstd), y = bf16(w * u)), the
//                half-split partial RoPE with bf16 ops, q heads to a bf16
//                buffer and k / v heads appended to the paged caches —
//                glm4_qkv_finish's lanes, sinks and padding rule.
//
// Deterministic and capturable (fixed grids, no host reads). dim is 128.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// rstd[r] for rows [0, rows): dot fp32 [rows, width] (row stride `stride`
// elements). A row with pos < 0 writes nothing.
void minimax_qk_rstd(const float* dot, int64_t stride, int width, const int64_t* pos, int rows,
                     float eps, float* rstd, cudaStream_t stream);

// The rank's heads of rows [0, rows). q_dot / k_dot / v_dot point at the
// rank's first head of row 0 (row strides in elements — the whole
// projection's width when the buffer holds every head); q_norm / k_norm at
// the rank's first head's weights (bf16 [local_heads * 128] /
// [kv_heads * 128]); q_rstd / k_rstd fp32 [rows] from minimax_qk_rstd.
// inv_freq fp32 [rotary_dim / 2] (rotary_dim 0: no RoPE). kv_heads is the
// rank's K/V head count. A row with pos < 0 writes nothing.
void minimax_qkv_finish(const float* q_dot, int64_t q_stride, const float* k_dot, int64_t k_stride,
                        const float* v_dot, int64_t v_stride, const uint16_t* q_norm,
                        const uint16_t* k_norm, const float* q_rstd, const float* k_rstd,
                        const float* inv_freq, int rotary_dim, const int32_t* req_ids,
                        const int64_t* pos, int rows, int local_heads, int kv_heads,
                        const int32_t* block_tables, int blocks_per_request, int block_tokens,
                        uint16_t* q_out, int64_t q_out_stride, uint16_t* k_cache, uint16_t* v_cache,
                        cudaStream_t stream);

}  // namespace dgpp
