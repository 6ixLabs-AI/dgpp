#pragma once
// DeepSeek-V4-Flash attention kernels (2026-10-01): the pieces the CSA2 /
// DSA kernels (kernels/csa2.hpp, kernels/dsa.hpp) do not already provide.
// The layer object (models/dsv4/attn_layer.hpp) composes them with those:
//   * the window rows live in per-request rings and the compressed entries
//     in the paged pool, both as bf16 rows of 512 (K == V): the 448
//     non-rotary elements quantize-dequantized with the release's
//     act_quant (e4m3, one power-of-two scale per 64) and the rotated tail
//     of 64 kept as it is — exactly the values the reference caches;
//   * every compressing layer owns a gated-pooling compressor: ratio 4
//     pools OVERLAPPING groups (an entry sees the previous group through
//     the first half of the projection and its own group through the
//     second), ratio 128 plain groups. The per-token projections go to a
//     positional ring (slot = pos % slots, fp32 kv and score + ape), so a
//     rejected speculative row leaves nothing to roll back: its slots are
//     rewritten before any accepted group reads them;
//   * a ratio-4 layer's indexer pools its own 128-wide keys the same way,
//     rotates them by the 128-point Hadamard transform and stores them as
//     the reference's fp4 e8m0/32 values in the planar index cache
//     (kernels/csa2.hpp's exact e4m3 + row-scale form).
// Numerics mirror inference/model.py + kernel.py at their rounding points:
// fp32 pooling with a bf16 rounding before the one-rounding RMSNorm, the
// complex rotation over adjacent pairs in fp32 with one rounding, the
// release's quantizers.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kDsv4Latent = 512;      // head_dim: the latent width, K == V
constexpr int kDsv4Rope = 64;         // the rotated tail (qk_rope_head_dim)
constexpr int kDsv4Nope = kDsv4Latent - kDsv4Rope;
constexpr int kDsv4NopeGroup = 64;    // act_quant's block on the cached rows
constexpr int kDsv4IndexDim = 128;    // index_head_dim

// ---- the query's per-head normalization -------------------------------------
// q[r, h, :] *= rsqrt(mean(q[r, h, :]^2) + eps) in place: fp32 interior, one
// bf16 rounding (the reference's `q *= torch.rsqrt(q.square().mean(-1) +
// eps)`). q: bf16 [rows, heads, dim].
void dsv4_q_head_rmsnorm(void* q, int64_t rows, int heads, int dim, float eps, cudaStream_t stream);

// ---- act_quant on the cached rows ------------------------------------------
// The release's act_quant(x[..., :448], 64, ue8m0, inplace) on the first
// kDsv4Nope elements of every 512-wide row: per 64 elements s =
// 2^ceil(log2(max(absmax, 1e-4) / 448)), x <- bf16(e4m3(x / s) * s). x:
// bf16 rows at x + r * row_stride (elements). pos (optional): rows with
// pos[r] < 0 are left untouched.
void dsv4_act_quant_nope(void* x, int64_t row_stride, int64_t rows, const int64_t* pos, cudaStream_t stream);

// ---- the decode walk's fused tails --------------------------------------------
// dsv4_q_head_rmsnorm_rope: dsv4_q_head_rmsnorm, then the rotation of each
// head's last 64 elements at pos[row] (csa2_rope_apply forward; rows with
// pos -1 are normed and not rotated) — one launch, bitwise the two.
// q: bf16 [rows, heads, 512].
void dsv4_q_head_rmsnorm_rope(void* q, int64_t rows, int heads, float eps, const int64_t* pos,
                              const float* inv_freq, cudaStream_t stream);
// dsv4_kv_tail: the K/V row's chain behind its projection in one launch —
// csa2_rmsnorm_bf16 (norm_weight [512]), the rotation of the last 64,
// dsv4_act_quant_nope, the row into its ring slot (the bf16 row format;
// slot pos % ring_slots of block ring_table[req]: dsa_latent_append) and,
// with wlist / wcounts, the row's window list (csa2_window_slots_decode).
// kv: bf16 [rows, 512] in place. Bitwise the six launches; a pos -1 row
// keeps its normed latent and lists nothing.
void dsv4_kv_tail(void* kv, const void* norm_weight, float eps, const int32_t* req_ids, const int64_t* pos,
                  const float* inv_freq, const int32_t* ring_table, int ring_slots, void* ring, int rows, int window,
                  int32_t* wlist, int32_t* wcounts, cudaStream_t stream);

// ---- the 128-point Hadamard rotation ---------------------------------------------
// x[r, :] <- bf16(H x[r, :] * 128^-0.5) in place over `rows` rows of 128
// (the Sylvester transform in fp32, one rounding): the indexer's
// rotate_activation on its q heads and its pooled keys.
void dsv4_hadamard128(void* x, int64_t rows, cudaStream_t stream);

// w_folded[i] = fp32(bf16(fp32(w[i]) * scale)) * q_scale[i]: the reference's
// `weights_proj(x) * (softmax_scale * n_heads^-0.5)` (a bf16 tensor times a
// scalar: one bf16 rounding) with the fp8 q's row scale folded in.
void dsv4_fold_weights(const void* w_bf16, const float* q_scale, float* out, int64_t n, float scale,
                       cudaStream_t stream);

// ---- the compressors -------------------------------------------------------------
// One compressor's geometry: `ratio` tokens pool into one `dim`-wide entry;
// `overlap` (ratio 4): the projections are 2 * dim wide and an entry pools
// the previous group's first halves with its own group's second halves.
// `slots`: the positional ring's slots per request (a power of two is not
// required; slots >= (overlap ? 2 : 1) * ratio + the speculative rows).
struct Dsv4CompGeom {
  int ratio = 0;
  int dim = 0;
  int wide = 0;
  bool overlap = false;
  int slots = 0;
  // A ring's fp32 elements per request: [slots][2][wide] (kv then score + ape).
  size_t ring_elems() const { return static_cast<size_t>(slots) * 2 * static_cast<size_t>(wide); }
};
// The ring slots a compressor of `ratio` needs beside `spec_rows`
// speculative rows.
constexpr int dsv4_comp_ring_slots(int ratio, bool overlap, int spec_rows) {
  return (overlap ? 2 : 1) * ratio + spec_rows + ((overlap ? 2 : 1) * ratio + spec_rows) % 2;
}

// The rows' projections into their requests' rings: slot pos % slots gets
// kv[r, :] and score[r, :] + ape[pos % ratio, :]. Rows with pos < 0 write
// nothing; two rows of a call must not share a (request, slot) — a prefill
// passes its chunk's last `slots` rows. kv / score: fp32 [rows, wide];
// ape: fp32 [ratio, wide]; ring: fp32 [max_requests][slots][2][wide].
void dsv4_comp_ring_write(const float* kv, const float* score, const float* ape, const int32_t* req_ids,
                          const int64_t* pos, int rows, const Dsv4CompGeom& g, float* ring, cudaStream_t stream);

// ---- the lazy form (a compressor without overlap: ratio 128) ----------------
// The decode walk keeps the layer INPUTS of the open group instead of their
// projections and projects the whole group when it completes
// (launch_dense_mma_bf16w_groups_f32): a pass that completes no group reads
// no compressor weights. dsv4_u_ring_write: hidden[r, :] (bf16, `hidden`
// elements at a row stride of hidden_stride) into ring[req_ids[r], pos[r] %
// slots, :]; rows with pos < 0 write nothing. ring: bf16
// [max_requests][slots][hidden], 16-byte aligned; hidden % 8 == 0.
void dsv4_u_ring_write(const void* hidden_rows, size_t hidden_stride, const int32_t* req_ids, const int64_t* pos,
                       int rows, int hidden, int slots, void* ring, cudaStream_t stream);
// dsv4_comp_publish_main_lazy: every row whose position completes a group
// publishes entry pos / ratio from the group's projections kv / score
// (fp32 [rows][ratio][wide], row r's group at r — the groups form's
// output; ape added here), exactly dsv4_comp_publish_main_prefill's entry
// on the same projections. g.overlap must be false.
void dsv4_comp_publish_main_lazy(const float* kv, const float* score, const float* ape, const Dsv4CompGeom& g,
                                 const int32_t* req_ids, const int64_t* pos, int rows, const void* norm_w, float eps,
                                 const float* inv_freq, const int32_t* block_tables, int blocks_per_request,
                                 int entries_per_block, void* cache, cudaStream_t stream);

// Decode, the main entries: every row whose position completes a group
// ((pos + 1) % ratio == 0) pools the group off its request's ring (the
// rows of this call are already in it), rounds to bf16, applies the
// one-rounding RMSNorm (norm_w bf16 [512]), rotates the tail at the
// group's first position, quantizes the 448 with act_quant and writes the
// row of entry pos / ratio through the block table. cache: bf16 rows of
// 512 per physical entry slot.
void dsv4_comp_publish_main_decode(const float* ring, const Dsv4CompGeom& g, const int32_t* req_ids,
                                   const int64_t* pos, int rows, const void* norm_w, float eps, const float* inv_freq,
                                   const int32_t* block_tables, int blocks_per_request, int entries_per_block,
                                   void* cache, cudaStream_t stream);
// Decode, the index keys (dim 128): pooled, rounded, normed (norm_w bf16
// [128]), the tail of 64 rotated at the group's first position, the
// Hadamard rotation, then the fp4 e8m0/32 values as exact e4m3 codes with
// a power-of-two row scale (csa2_index_k_append's form; *violations counts
// rows that could not be exact, may be null).
void dsv4_comp_publish_index_decode(const float* ring, const Dsv4CompGeom& g, const int32_t* req_ids,
                                    const int64_t* pos, int rows, const void* norm_w, float eps,
                                    const float* inv_freq, const int32_t* block_tables, int blocks_per_request,
                                    int entries_per_block, void* index_k, float* index_scale, unsigned* violations,
                                    cudaStream_t stream);
// Prefill: one request's chunk of T rows at positions pos0 + i (pos0 a
// multiple of ratio): the T / ratio complete groups, entry pos0 / ratio +
// j, from the chunk's own projections (kv / score fp32 [T, wide], ape added
// here); an overlapping compressor's first group reads the previous group
// off the request's ring (`ring_req`: this request's [slots][2][wide]) when
// pos0 > 0. The ring must still hold the rows before the chunk: publish
// before dsv4_comp_ring_write. block_table: the request's row.
void dsv4_comp_publish_main_prefill(const float* kv, const float* score, const float* ape, const float* ring_req,
                                    const Dsv4CompGeom& g, int64_t pos0, int T, const void* norm_w, float eps,
                                    const float* inv_freq, const int32_t* block_table, int entries_per_block,
                                    void* cache, cudaStream_t stream);
void dsv4_comp_publish_index_prefill(const float* kv, const float* score, const float* ape, const float* ring_req,
                                     const Dsv4CompGeom& g, int64_t pos0, int T, const void* norm_w, float eps,
                                     const float* inv_freq, const int32_t* block_table, int entries_per_block,
                                     void* index_k, float* index_scale, unsigned* violations, cudaStream_t stream);

// ---- the dense list of a ratio-128 layer ----------------------------------------------
// list[r, k] = k for k < pos_sel[r] + 1 (every compressed entry a row sees),
// -1 past it; counts[r] = max(pos_sel[r] + 1, 0) clamped to stride.
void dsv4_dense_list(const int64_t* pos_sel, int rows, int stride, int32_t* list, int32_t* counts,
                     cudaStream_t stream);

// ---- hash routing ---------------------------------------------------------------------
// The router's per-row selection bias of a hash layer: out[r, e] = 1e30 for
// the top_k experts tid2eid[tokens[r]] names, 0 elsewhere — the biased
// scores then select exactly those experts while the weights stay the raw
// scores'. tid2eid: int32 [vocab, top_k]; out: fp32 [rows, n_experts].
void dsv4_hash_bias_rows(const int64_t* tokens, int rows, const int32_t* tid2eid, int vocab, int top_k,
                         int n_experts, float* out, cudaStream_t stream);

}  // namespace dgpp
