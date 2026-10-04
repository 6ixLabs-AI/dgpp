#pragma once
// Host reference of the Mistral-Small-4 attention layer (2026-10-04,
// docs/mistral_small4_plan.md §4): dense multi-head latent attention with
// the YaRN-blended interleaved rope and the Llama-4 query scale. Two roles,
// as models/dsa_reference.hpp has for the GLM-5.3 layer it is modelled on:
// the test oracle for the device kernels (Acc = float mirrors the device's
// fp32 accumulation, Acc = double measures its drift), and the executable
// statement of the numerics — tools/mistral4_reference.py computes the same
// equations in full precision, and tests/unit/mistral4_mla_test.cpp holds
// the two against each other on a synthetic checkpoint.
//
// All tensors are host memory; bf16 is carried as raw uint16 bits. Rounding
// points, in order (the GLM-5.3 MLA path's, so the kernels of
// kernels/dsa.hpp serve this model with a zero-width indexer):
//   * the fused projection [wq_a | wkv_a_with_mqa] of the layer's input:
//     bf16 GEMM output;
//   * q latent and kv latent: RMSNorm, fp32 compute, one bf16 rounding
//     (x * rstd * w), eps = norm_eps;
//   * q = wq_b @ q latent: bf16 GEMM output, per head [nope | rope];
//   * rope, INTERLEAVED pairs kept in place — (x[2i], x[2i+1]) ->
//     (bf16(bf16(x0 c) - bf16(x1 s)), bf16(bf16(x1 c) + bf16(x0 s))) — on
//     each head's rope slice and on the token's rope key (the last
//     qk_rope_head_dim outputs of wkv_a_with_mqa, NOT normed). (c, s) are
//     bf16: the angle is fp32(pos) * inv_freq[i] in fp32, the trig taken in
//     double and rounded to fp32 then bf16 (kernels/dsa.cu's table recipe),
//     with inv_freq the YaRN blend (Mistral4TextConfig::rope_inv_freq).
//     transformers' apply_rotary_pos_emb_interleave writes the rotated
//     pairs de-interleaved; q and k are permuted alike, so the scores are
//     the same and the in-place layout is the cache's;
//   * the Llama-4 scale: after the rotation every element of the token's
//     query (nope and rope lanes, every head) is bf16(x * l4(pos)), l4 in
//     fp32 = 1 + beta * log(1 + floor(pos / original)); positions below
//     `original` multiply by exactly 1 and are left untouched;
//   * the cache row of a token: [kv latent | rotated rope key], bf16;
//   * attention of the token at position p over cache rows [0, p]: the
//     absorbed form — q~ = [W_uk^T q_nope | q_rope] with the absorbed part
//     rounded to bf16, score = (q~ . row) * scale in Acc, softmax in Acc,
//     probabilities rounded to bf16 for the value accumulation c = sum p *
//     latent, out_h = W_uv c rounded to bf16 — with W_uk / W_uv head h's
//     first nope / last v rows of wkv_b. scale = 1 / sqrt(nope + rope):
//     no YaRN magnitude (params.json yarn.apply_scale false);
//   * the output projection wo: bf16 GEMM output.
#include <cstdint>
#include <vector>

#include "models/mistral4/config.hpp"

namespace dgpp::mistral4_ref {

// The layer's shape. `heads` is the number of heads the weight views carry
// (all of them at world 1; a rank's slice of wq_b / wkv_b rows and wo
// columns under tensor parallelism — the latents are replicated).
struct Geometry {
  int hidden = 0;
  int heads = 0;
  int q_lora = 0;
  int kv_lora = 0;
  int nope = 0;
  int rope = 0;
  int v = 0;
  float eps = 1e-6f;
  float scale = 0.0f;  // 1 / sqrt(nope + rope)
  float llama4_beta = 0.0f;
  int llama4_original = 1;

  int q_head() const { return nope + rope; }
  int kv_head() const { return nope + v; }
  int row_width() const { return kv_lora + rope; }  // a cache row
  static Geometry from_config(const Mistral4TextConfig& cfg, int tp_size = 1);
};

// bf16 weight views, row-major [out, in] as the checkpoint stores them.
struct HostWeights {
  const uint16_t* wq_a = nullptr;       // [q_lora, hidden]
  const uint16_t* q_a_norm = nullptr;   // [q_lora]
  const uint16_t* wq_b = nullptr;       // [heads * (nope + rope), q_lora]
  const uint16_t* wkv_a = nullptr;      // [kv_lora + rope, hidden] (wkv_a_with_mqa)
  const uint16_t* kv_a_norm = nullptr;  // [kv_lora]
  const uint16_t* wkv_b = nullptr;      // [heads * (nope + v), kv_lora]
  const uint16_t* wo = nullptr;         // [hidden, heads * v]
};

// One request's cache in logical order: row t is the token at position
// first_pos + t.
struct HostState {
  std::vector<uint16_t> rows;  // [num_tokens, kv_lora + rope] bf16
  int64_t num_tokens = 0;
  int64_t first_pos = 0;
  void reset(int64_t first_position) {
    rows.clear();
    num_tokens = 0;
    first_pos = first_position;
  }
};

// What the python reference taps, for stage-by-stage comparison (optional).
struct Taps {
  std::vector<uint16_t> q;       // [tokens, heads, nope + rope] after rope and the Llama-4 scale
  std::vector<uint16_t> latent;  // [tokens, kv_lora]
  std::vector<uint16_t> k_rope;  // [tokens, rope] rotated
  std::vector<uint16_t> heads;   // [tokens, heads, v] the attention output before wo
};

// The Llama-4 query multiplier at `pos`, fp32 (1 for pos < original).
float llama4_scale(float beta, int original, int64_t pos);

// The (cos | sin) row of one position: bf16 [2][rope / 2], the table
// recipe of kernels/dsa.cu with explicit inverse frequencies.
void rope_row(const float* inv_freq, int rope_dim, int64_t pos, uint16_t* cos_sin);
// The device table: bf16 [positions][2][rope / 2] for positions [0, positions).
void rope_table(const float* inv_freq, int rope_dim, int64_t positions, uint16_t* out);
// Rotates the first rope_dim elements of a bf16 row in place.
void rope_interleave_row(uint16_t* x, int rope_dim, const uint16_t* cos_sin);

// The layer over `tokens` rows of its (already attention_norm'ed) input,
// hidden_in [tokens, hidden] bf16, the first at position
// state.first_pos + state.num_tokens. Appends the rows to the cache and
// writes layer_out [tokens, hidden] bf16. inv_freq: [rope / 2].
template <typename Acc>
void layer_forward(const HostWeights& w, const Geometry& g, const float* inv_freq,
                   const uint16_t* hidden_in, HostState& state, int tokens, uint16_t* layer_out,
                   Taps* taps = nullptr);

}  // namespace dgpp::mistral4_ref
