#pragma once
// The streaming tensor-core decode GEMM (2026-09-14, the batched decode's
// dense projections and the lm head): out[m, n] = act[m, k] x W[n, k]^T for
// m <= 32 activation rows, the weights read ONCE whatever m.
//
// WHY: the decode GEMV cores (fp8_gemv.cuh, bf16_gemv.cu) chunk the rows by
// four and pay kRows FMAs per weight element on the CUDA cores, so past a
// few rows a projection is issue-bound (the six-slot DeepSeek-V4.1 batch:
// 30 rows = eight chunks, 56 ms of a 249 ms step in dense fp8 projections
// and 18 ms in the 331 MB head). This kernel keeps the GEMV's memory shape
// — one warp per eight weight rows, a quad of lanes per row, 16-byte chunks,
// eight in flight per lane, no barrier in the k loop — and applies every
// chunk to all m rows with mma.sync m16n8k16 (bf16 in, fp32 accumulate):
// the per-weight cost is a dequant, a fixed in-quad shuffle and a quarter
// of an mma, whatever m.
//
// NUMERICS: the weight VALUES are the dequant bridge's (bf16(e4m3 x scale),
// exact for the e8m0 scales); the activations are the bf16 rows as given;
// the fp32 accumulation is the mma's over the k16 slices in ascending k
// order — deterministic, and without split-K the SAME chain whatever m
// (a padded row never touches another row's accumulators), so a batched row is
// bitwise the row alone at m = 1 through this kernel. It is not bitwise
// the GEMV cores' chain (a different fp32 order, inside the oracle budgets).
// With workspace, groups of at most 32 rows may use split-K. Their chain
// is independent of the group's row count, but differs from wider groups'
// unsplit chain. A caller changing batch size must preserve this boundary.
//
// CONTRACT: m >= 1 (rows 1..128 in one launch — 1/2/4/8 sixteen-row tiles
// by the count — and wider m in 128-row groups, each group its own launch
// re-reading the weights); k % 64 == 0 (a quad of lanes loads 4 x 16 k; a chunk lies inside one scale block:
// cs >= 4); 16-byte-aligned weight rows; act rows 16-byte aligned with an
// even-16-byte stride; out row stride >= n. fp8: w [n, k] e4m3 with block
// scales f32 [ceil(n / 2^rs), ceil(k / 2^cs)]. bf16: w [n, k] bf16.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kMmaGemvMaxRows = 32;
// Rows per launch: the widest single form (8 tiles); m above it runs in
// groups of this many rows, the weights read once per group.
constexpr int kMmaGemvMaxRowsPerLaunch = 128;

// fp8 weights with block scales; out bf16 or f32 (the epilogue store is the
// only difference: bf16(out_f32) == out_bf16 bit for bit).
// ws / ws_bytes (2026-09-21): a device workspace lets the decode forms (m <=
// 32) split the k range across blocks when a small n leaves the grid
// under-filled (fp32 partials in ws, one reduce launch; the split count a
// function of the shape only, so a row's chain is still the same whatever
// m rides in the launch). nullptr: the unsplit form, as before.
void launch_mma_gemv_fp8_bf16(const uint16_t* act, size_t act_stride, const uint8_t* w,
                              const float* scales, uint16_t* out, int m, int n, int k,
                              size_t out_stride, int rs, int cs, cudaStream_t stream,
                              void* ws = nullptr, size_t ws_bytes = 0);
void launch_mma_gemv_fp8_f32(const uint16_t* act, size_t act_stride, const uint8_t* w,
                             const float* scales, float* out, int m, int n, int k,
                             size_t out_stride, int rs, int cs, cudaStream_t stream,
                             void* ws = nullptr, size_t ws_bytes = 0);
// The swiglu form (2026-10-02, the DeepSeek-V4-Flash shared expert's down
// projection): out = swiglu(gate, up) x W^T, the activation
//   g = min(gate, limit), u = clamp(up, -limit, limit), bf16(bf16(g * sigmoid(g)) * u)
// computed as the rows are staged — bitwise launch_moe_swiglu_clamp followed
// by launch_mma_gemv_fp8_f32 on its output, without the launch between (a
// kernel that becomes ready behind a queued one waits for it). gate / up:
// bf16 [m, act_stride], both 16-byte aligned. The unsplit chain.
void launch_mma_gemv_fp8_swiglu_f32(const uint16_t* gate, const uint16_t* up, float limit, size_t act_stride,
                                    const uint8_t* w, const float* scales, float* out, int m, int n, int k,
                                    size_t out_stride, int rs, int cs, cudaStream_t stream);
// bf16 weights (the lm head); out bf16 or f32.
void launch_mma_gemv_bf16_bf16(const uint16_t* act, size_t act_stride, const uint16_t* w,
                               uint16_t* out, int m, int n, int k, size_t out_stride,
                               cudaStream_t stream, void* ws = nullptr, size_t ws_bytes = 0);
void launch_mma_gemv_bf16_f32(const uint16_t* act, size_t act_stride, const uint16_t* w,
                              float* out, int m, int n, int k, size_t out_stride,
                              cudaStream_t stream, void* ws = nullptr, size_t ws_bytes = 0);
// The shape the kernel takes (k a multiple of 16, aligned pointers, m in range).
bool mma_gemv_shape_ok(const void* w, const void* act, size_t act_stride, int m, int k);

// The decode forms' block width override (warps of 8 weight rows: 1, 2, 4,
// 8; 0 restores the rule by n) — the timing sweep's knob, not a serving
// setting: the rows' chains do not depend on it.
void mma_gemv_set_decode_width(int warps);

}  // namespace dgpp
