#pragma once
// Packed int4/int8 prefill: exact integer codes on BF16 tensor cores,
// then FP32 scaling of each 64-element group's partial dot. Weights are
// never rounded to BF16. The summation order differs from packq_gemv.
#include "kernels/glm_moe_launch.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

// Short-prefill reassociation changes MTP acceptance and can reduce C1
// throughput even when accuracy gates pass. Keep the established GEMV
// path there and amortize packed tiles over bulk-prefill batches.
constexpr int kPackqMmaFromRows = 128;

// D[m,n] = Act[m,k] W[n,k]^T. K is a positive multiple of 64; packed
// weight rows are 16-byte aligned. Activations may have a padded stride.
// These explicit GEMM entry points do not change the GEMV decode launchers.
// `variant` (2026-09-29): -1 = the default kernel (the 64 x 128 wide tile
// for int4 rows unless DGPP_PACKQ_GEMM=narrow), 0 = the narrow 32 x 64
// kernel, 1 = the wide one, 2 (2026-09-30, DGPP_PACKQ_GEMM=wide3) = the
// wide tile with its B fragments decoded in registers and three pipeline
// stages. All give bitwise the same D.
void launch_packq_gemm_bf16(const uint16_t* act, size_t act_stride, const GlmPackedMatrix& w,
                            uint16_t* out, int m, int n, int k, cudaStream_t stream);
void launch_packq_gemm_f32(const uint16_t* act, size_t act_stride, const GlmPackedMatrix& w,
                           float* out, int m, int n, int k, cudaStream_t stream);
// The same with the kernel pinned (the bench and the tests; the entries
// above keep their signatures for the callers that take them by pointer).
void launch_packq_gemm_bf16_variant(const uint16_t* act, size_t act_stride, const GlmPackedMatrix& w,
                                    uint16_t* out, int m, int n, int k, cudaStream_t stream, int variant);
void launch_packq_gemm_f32_variant(const uint16_t* act, size_t act_stride, const GlmPackedMatrix& w,
                                   float* out, int m, int n, int k, cudaStream_t stream, int variant);

// Device-resident expert segments, including empty/ragged segments.
// max_rows bounds the largest segment; act_rows optionally maps gathered
// row indices to original activation rows. Output uses segment row indices.
// Each selected view must have the supplied bit width, scale format
// (kPackedScale*, quant_matrix.hpp) and N/K geometry.
// `tiles` (2026-09-29): the compact tile list of the segments at
// kPackqGemmWideRows rows a tile (launch_moe_tile_list) with its device
// count and host capacity; the wide kernel then runs a 1-D grid over the
// real tiles instead of n_segs x the longest segment's m-tiles (the same
// blocks do the same tiles: bitwise). Any max_rows > 0 is accepted with a
// list; the narrow kernel ignores the list and keeps the max_rows grid.
constexpr int kPackqGemmWideRows = 64;
// `out2` / `which2` (2026-09-30, the bf16 form): a second projection of the
// same segments in the same launch — the wide kernel's n-tiles past `n`
// read view `which2` and write `out2` (the same stride), each token row
// gathered once for both; every element's chain is the separate launch's
// (bitwise). The gate and up projections of the packed chain run so.
void launch_moe_grouped_mma_packq_bf16(const uint16_t* act, size_t act_stride,
                                       const MoeSegment* segs, int n_segs, int max_rows,
                                       const MoeExpertView* views, int which, uint16_t* out,
                                       size_t out_stride, int n, int k, int bits,
                                       cudaStream_t stream, const int32_t* act_rows = nullptr,
                                       int scale_fmt = 0, int variant = -1,
                                       const MoeTile* tiles = nullptr,
                                       const int32_t* tile_count = nullptr, int tile_cap = 0,
                                       uint16_t* out2 = nullptr, int which2 = -1);
void launch_moe_grouped_mma_packq_f32(const uint16_t* act, size_t act_stride,
                                      const MoeSegment* segs, int n_segs, int max_rows,
                                      const MoeExpertView* views, int which, float* out,
                                      size_t out_stride, int n, int k, int bits,
                                      cudaStream_t stream, const int32_t* act_rows = nullptr,
                                      int scale_fmt = 0, int variant = -1,
                                      const MoeTile* tiles = nullptr,
                                      const int32_t* tile_count = nullptr, int tile_cap = 0);
// The kernel the default `variant` selects: 1 = the wide tile, 0 = narrow.
int packq_gemm_variant_default();

}  // namespace dgpp
