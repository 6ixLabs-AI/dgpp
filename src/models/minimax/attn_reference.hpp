#pragma once
// Host reference of the MiniMax-M2 attention layer (2026-10-04,
// docs/minimax_m27_plan.md §3): GQA with the PER-LAYER q/k norm. Two roles,
// as models/glm4/attn_reference.hpp has for the GLM-4.7 layer it differs
// from in one step: the test oracle for the device kernels, and the
// executable statement of the numerics — tools/minimax_m2_reference.py
// computes the same equations in full precision and
// tests/unit/minimax_attn_test.cpp holds the two against each other on a
// synthetic checkpoint.
//
// All tensors are host memory; bf16 is carried as raw uint16 bits. Rounding
// points, in order (GLM-4.7's — kernels/glm4_attn.hpp — except the norm):
//   * q / k / v projections: fp32 dots, bf16(dot) — the Linear's one
//     rounding (no biases);
//   * THE NORM: one RMS over the WHOLE row of the projection — every query
//     head's 128 elements for q (6144), every K/V head's for k (1024) —
//     rstd = 1 / sqrt(fp32(sum x^2 / n) + eps), the sum in double over the
//     bf16 values; then, per element, u = bf16(x * rstd), y = bf16(w * u)
//     with w the element's own weight (q_norm [6144], k_norm [1024]).
//     GLM-4.7 takes the RMS per head with one [128] weight;
//   * the partial RoPE on the first rotary_dim dims of each q and k head,
//     half-split pairs (i, i + rotary_dim / 2) — transformers' rotate_half —
//     with bf16 ops: cos / sin of the fp32 angle rounded to bf16, x*cos and
//     rotate(x)*sin rounded, the sum rounded;
//   * the k and v heads appended to the K/V rows, bf16;
//   * attention: glm4_ref::attention (fp32 scores and softmax,
//     probabilities rounded to bf16 for the value sum, the denominator
//     unrounded), a query head reading K/V head h / (heads / kv_heads),
//     scale 1 / sqrt(head_dim); the heads' output rounded to bf16;
//   * o_proj: the fp32 dot, unrounded here — at world 1 the caller rounds
//     it to bf16; under tensor parallelism it is a rank's PARTIAL sum.
//
// Tensor parallelism (the plan's option A): the RMS spans every head, so a
// rank cannot norm its own slice from its own slice. Every rank holds the
// whole q_proj and k_proj (and norm weights), computes and norms the whole
// row — bitwise the same on every rank — and keeps its slice: query heads
// [head_begin, head_begin + local_heads), K/V heads [kv_begin, kv_begin +
// local_kv). v_proj rows and o_proj columns are sliced. `Geometry` carries
// the slice; world 1 is the slice of everything.
#include <cstdint>
#include <vector>

#include "models/minimax/config.hpp"

namespace dgpp::minimax_ref {

struct Geometry {
  int hidden = 0;
  int heads = 0;     // every query head (the norm's span)
  int kv_heads = 0;  // every K/V head
  int dim = 0;
  int rotary_dim = 0;
  float eps = 1e-6f;
  float scale = 0.0f;  // 1 / sqrt(dim)
  // This rank's slice.
  int head_begin = 0;
  int local_heads = 0;
  int kv_begin = 0;
  int local_kv = 0;

  int q_width() const { return heads * dim; }
  int kv_width() const { return kv_heads * dim; }
  static Geometry from_config(const MinimaxTextConfig& cfg, int rank = 0, int world = 1);
};

// bf16 weight views, row-major [out, in]. q_proj, k_proj and the two norm
// weights are WHOLE (replicated across ranks); v_proj holds the rank's K/V
// heads' rows, o_proj the rank's query heads' columns.
struct HostWeights {
  const uint16_t* q_proj = nullptr;  // [heads * dim, hidden]
  const uint16_t* k_proj = nullptr;  // [kv_heads * dim, hidden]
  const uint16_t* v_proj = nullptr;  // [local_kv * dim, hidden]
  const uint16_t* o_proj = nullptr;  // [hidden, local_heads * dim]
  const uint16_t* q_norm = nullptr;  // [heads * dim]
  const uint16_t* k_norm = nullptr;  // [kv_heads * dim]
};

// One request's K/V rows on this rank, in position order.
struct HostState {
  std::vector<uint16_t> k;  // [num_tokens, local_kv * dim] bf16
  std::vector<uint16_t> v;  // [num_tokens, local_kv * dim] bf16
  int64_t num_tokens = 0;
  int64_t first_pos = 0;
  void reset(int64_t first_position) {
    k.clear();
    v.clear();
    num_tokens = 0;
    first_pos = first_position;
  }
};

struct Taps {
  std::vector<uint16_t> q;      // [tokens, local_heads, dim] normed and rotated
  std::vector<uint16_t> k;      // [tokens, local_kv, dim] normed and rotated
  std::vector<uint16_t> v;      // [tokens, local_kv, dim]
  std::vector<uint16_t> heads;  // [tokens, local_heads, dim] the attention output before o_proj
};

// The whole-row finish of a q or k projection: dot fp32 [heads * dim] (every
// head), norm bf16 [heads * dim]; writes out bf16 [heads * dim], normed and
// rotated at `pos`. inv_freq: [rotary_dim / 2].
void qk_finish(const float* dot, const uint16_t* norm, float eps, const float* inv_freq,
               int rotary_dim, int64_t pos, int heads, int dim, uint16_t* out);

// The layer over `tokens` rows of its (already input_layernorm'ed) input,
// hidden_in [tokens, hidden] bf16, the first at position state.first_pos +
// state.num_tokens. Appends this rank's K/V rows and writes the o_proj
// dots, UNROUNDED fp32, to out [tokens, hidden]: the layer's output at
// world 1 (round it to bf16), a rank's partial sum otherwise.
void layer_forward(const HostWeights& w, const Geometry& g, const float* inv_freq,
                   const uint16_t* hidden_in, HostState& state, int tokens, float* out,
                   Taps* taps = nullptr);

}  // namespace dgpp::minimax_ref
