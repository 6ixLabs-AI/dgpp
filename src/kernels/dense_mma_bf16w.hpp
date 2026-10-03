#pragma once
// The dense tensor-core GEMM over BF16 weights (2026-10-01, the
// DeepSeek-V4-Flash prefill's compressor projections):
//
//   out[m, n] = act[m, k] x W[n, k]^T      act, W bf16; fp32 accumulation
//
// in 64-row m-tiles against 128-row n-tiles, the weights and activations
// staged through cp.async rings and read by ldmatrix / packed loads into
// mma.sync m16n8k16 — the fp8 ldmatrix kernel's shape (glm_moe.cu,
// moe_grouped_mma_fp8_ldm_kernel) without its dequant. An output element's
// chain is the mma's over the k16 slices in ascending k, whatever rows share
// the launch: a prefill chunk's rows are bitwise the rows of any other
// chunking. It is not the streaming form's chain (mma_gemv.hpp), which a
// decode walk keeps.
//
// WHY: the streaming form re-reads the weights per 128 rows and spends a
// warp per eight weight rows on them — a bandwidth shape, 12 TFLOPS on a
// prefill chunk, where this one runs the tile at the tensor cores' rate.
//
// CONTRACT: k a positive multiple of 32; W rows 16-byte aligned (w aligned,
// k % 8 == 0); act rows 16-byte aligned with a stride % 8 == 0 take the
// async copies, any other layout a scalar staging; out row stride >= n
// (0: n).
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

bool dense_mma_bf16w_shape_ok(const void* w, int k);
void launch_dense_mma_bf16w_f32(const uint16_t* act, size_t act_stride, const uint16_t* w, float* out, int m,
                                int n, int k, cudaStream_t stream, size_t out_stride = 0);
void launch_dense_mma_bf16w_bf16(const uint16_t* act, size_t act_stride, const uint16_t* w, uint16_t* out, int m,
                                 int n, int k, cudaStream_t stream, size_t out_stride = 0);

// The gated ring form (2026-10-02, the decode walk's lazy ratio-128
// compressors): for every batch row r whose position completes a group
// ((pos[r] + 1) % group == 0, pos[r] >= 0)
//
//   out[r, i, :] = ring[req_ids[r], (pos[r] - group + 1 + i) % slots, :] x W^T,   i in [0, group)
//
// and nothing for the other rows (their blocks return before a load: the
// launch costs a captured step its fixed latency alone). ring: bf16
// [requests][slots][k], 16-byte aligned, slots >= group; out: fp32
// [rows][group][n]. Each output row is bitwise launch_dense_mma_bf16w_f32's
// on the same activation row (the same tile chain).
void launch_dense_mma_bf16w_groups_f32(const uint16_t* ring, int slots, const int32_t* req_ids, const int64_t* pos,
                                       int rows, int group, const uint16_t* w, float* out, int n, int k,
                                       cudaStream_t stream);

}  // namespace dgpp
