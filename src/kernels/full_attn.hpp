#pragma once
// Dense causal paged attention for the Qwen3.8-27B full-attention layers:
// each row r attends positions [0, pos[r]] over the paged K/V cache — no
// scoring, no selection. Two forms (full_attn.cu): the row form (one warp
// per (row, KV group), the group's heads as the mma rows, the keys split
// across kMaxSplits ranges that a combine pass merges) for decode rows of
// arbitrary requests, and the tile form (one CTA per 16 consecutive rows of
// ONE request and KV group, one warp per head, 32-token K/V tiles shared by
// the group) for prefill chunks.
//
// Numerics follow the QSA rule (probabilities rounded to bf16 for P V, the
// denominator summing the unrounded values); tolerance-equal to a dense
// reference, not bitwise, and the forms differ from each other in their
// rescaling points.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// dim 256 and at most 16 query heads per KV head.
bool full_attn_supported(int dim, int local_heads, int kv_heads);

// The decode form. out fp32 [rows, local_heads, 256]. q bf16 rows of
// local_heads x dim pairs ([q|gate] interleave, row stride q_row_stride,
// heads contiguous); caches as qsa_kv_append writes them (kv_heads local);
// every query head h reads kv head h / (local_heads / kv_heads). pos int64
// [rows]: each row's position (a negative position is a padding row: zero
// output); req_ids [rows] select the block table. `partials` (null, or
// full_attn_partials_bytes(rows, kv_heads) bytes) enables the split walk:
// each (row, group) walks full_attn_max_splits() key ranges in parallel and
// a combine pass merges them — the long-context form.
void full_attn_decode(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                      const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos, int rows,
                      int local_heads, int kv_heads, int block_tokens, const int32_t* block_tables,
                      int blocks_per_request, float scale, float* out, float* partials,
                      cudaStream_t stream);
size_t full_attn_partials_bytes(int rows, int kv_heads);
int full_attn_max_splits();

// The prefill form: `rows` consecutive rows of one request (block_table =
// that request's row of the tables), positions per row. Groups wider than
// eight heads take the row form (req_ids / block_tables as above).
void full_attn_prefill(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                       const uint16_t* v_cache, const int32_t* block_table, const int32_t* req_ids,
                       const int64_t* pos, int rows, int local_heads, int kv_heads, int block_tokens,
                       const int32_t* block_tables, int blocks_per_request, float scale, float* out,
                       cudaStream_t stream);

// The unsplit row form (every row's whole range in one warp): the tests'
// reference path and the original kernel's contract.
void full_attn_prefill_warp(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                            const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos,
                            int rows, int local_heads, int kv_heads, int block_tokens,
                            const int32_t* block_tables, int blocks_per_request, float scale,
                            float* out, cudaStream_t stream);

}  // namespace dgpp
