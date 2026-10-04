#pragma once
// Double-precision oracle for the plain Qwen3 MoE (Qwen3MoeSparseMoeBlock,
// 2026-10-04): the softmax top-k router (glm_moe_ref_router in its
// SoftmaxTopk mode), NVFP4 routed experts, and NO shared expert — the chain
// GlmMoeLayer runs with n_shared_experts = 0, rounded once by
// launch_moe_round_bf16. The existing oracles each cover a neighbour and not
// this: glm_moe_ref_forward always adds a shared expert as the chain's last
// term, qwen_moe_ref_forward takes FP8 experts and Flash-Next's BF16 shared
// expert.
//
// The engine's rounding points, in double (models/glm/moe_reference.cpp's
// NVFP4 rules):
//   weights   e2m1(code) x e4m3(scale), exact; the matrix's global scale
//             divides the FINISHED dot once (the kernels' epilogue)
//   gate, up  bf16(dot / global)
//   act       bf16(bf16(silu(gate)) * up)         two roundings, no clamps
//   down      dot / global, unrounded
//   out       bf16(sum_e w_e * down_e)            one fma per expert in
//             ASCENDING expert id, one rounding
// `global` is the DIVISOR the loader stores: compressed-tensors'
// weight_global_scale as shipped, modelopt's weight_scale_2 as 1 / it.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "models/glm/moe.hpp"
#include "models/glm/moe_reference.hpp"

namespace dgpp {

struct Qwen3MoeHostWeights {
  int hidden = 0;
  int inter = 0;      // the routed slice width on this rank (I / world)
  int n_experts = 0;
  std::vector<uint16_t> router;   // bf16 [E, H]
  // E triples (gate [I, H], up [I, H], down [H, I]), matrix after matrix:
  // the code bytes [rows, cols / 2] (low nibble = even column), the e4m3
  // block scales [rows, cols / 16], and one divisor per matrix.
  std::vector<uint8_t> payloads;
  std::vector<uint8_t> scales;
  std::vector<float> globals;     // [E * 3]

  int64_t rows(int m) const { return m == 2 ? hidden : inter; }
  int64_t cols(int m) const { return m == 2 ? inter : hidden; }
  size_t payload_bytes(int m) const { return static_cast<size_t>(rows(m) * (cols(m) / 2)); }
  size_t scale_bytes(int m) const { return static_cast<size_t>(rows(m) * (cols(m) / 16)); }
  // Offsets of expert e's matrix m (index e * 3 + m).
  size_t payload_offset(int index) const;
  size_t scale_offset(int index) const;
  // Sizes the vectors to the geometry (zeros, globals 1).
  void allocate();
  GlmFp4MatrixHost view(int index) const;
};

// hidden [tokens, H] bf16 in; out [tokens, H] bf16 bits. `route` (optional)
// receives the router's decision (ids ascending, bf16 weights, the logit
// row in `biased`). cfg: Qwen3TextConfig::moe_config().
void qwen3_moe_ref_forward(const uint16_t* hidden, const Qwen3MoeHostWeights& w,
                           const GlmMoeConfig& cfg, int tokens, std::vector<uint16_t>& out,
                           GlmMoeRouterRef* route = nullptr);

}  // namespace dgpp
