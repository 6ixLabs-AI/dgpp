#pragma once
// Mistral-Small-4 attention kernel — UNVERIFIED DRAFT (2026-10-04,
// docs/mistral_small4_plan.md §4). Written without a GPU: never compiled
// by nvcc, never run. Built only under -DDGPP_BUILD_MISTRAL_DRAFT=ON; the
// first thing to do with it is tests/cuda/mistral4_attn_test.cu against the
// host reference (models/mistral4/mla_reference.hpp), which IS tested.
//
// The model's attention is the full GLM-5.3's latent attention
// (kernels/dsa.hpp) without an indexer, and every step but one already has
// a kernel there: the fused q_a / kv_a norm (dsa_fused_qkv_rmsnorm), the
// interleaved rope from a table (dsa_rope_interleave — the table here is
// built from the YaRN inverse frequencies, mistral4_ref::rope_table), the
// latent append with its rope tail (dsa_latent_append), the absorbed query
// (dsa_absorb_q), the attention (dsa_attn_dense / dsa_attn_partial), its
// combine and the value projection (dsa_attn_combine, dsa_vout_gemm).
//
// The one step that is new is the Llama-4 query scale: after the rotation,
// every element of the query of the token at position p — nope and rope
// lanes, every head — is bf16(x * l4), l4 = 1 + beta * log(1 + floor(p /
// original)). The multiplier is read from a host-built fp32 table indexed
// by the step floor(p / original) (mistral4_llama4_table_host: the host
// reference's own expression), so the device has no log to disagree about
// and the result is the reference's bits. A row whose step is 0 (the first
// `original` positions: l4 exactly 1) and a padding row (pos < 0) are left
// untouched.
#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace dgpp {

// table[s] = 1 + beta * log(1 + s) in fp32, for the steps of positions
// [0, max_positions): ceil(max_positions / original) entries.
inline std::vector<float> mistral4_llama4_table_host(float beta, int original,
                                                     int64_t max_positions) {
  const int64_t steps = (max_positions + original - 1) / original;
  std::vector<float> table(static_cast<size_t>(steps > 0 ? steps : 1));
  for (size_t s = 0; s < table.size(); ++s)
    table[s] = 1.0f + beta * std::log(1.0f + static_cast<float>(s));
  return table;
}

// In place over rows [0, rows): q bf16 [rows, width] (row stride q_stride
// elements; width = local_heads * (nope + rope)). pos: device int64 [rows];
// table: device fp32 [steps]; a position past the table reads its last
// entry (the layer bounds positions at admission). Deterministic and
// capturable (a fixed grid for a given rows x width, no host reads).
void mistral4_llama4_scale(uint16_t* q, int64_t q_stride, int width, const int64_t* pos, int rows,
                           const float* table, int steps, int original, cudaStream_t stream);

}  // namespace dgpp
