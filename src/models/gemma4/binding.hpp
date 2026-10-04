#pragma once
// Expected-tensor table for Gemma 4 (Gemma4ForConditionalGeneration,
// 2026-10-04): every text-backbone tensor the checkpoint must contain —
// names, dtypes, exact shapes — derived from the parsed config, not observed
// from one file. The table drives the offline validator (apps/gemma4_bind_check)
// and the resident loader, as the other families' tables do.
//
// Naming is checkpoint truth (nvidia/Gemma-4-31B-IT-NVFP4 @ 4135a98a): the
// text model sits under `model.language_model.` — layers under
// `model.language_model.layers.L.`, the globals
// `model.language_model.embed_tokens.weight` and
// `model.language_model.norm.weight`. There is NO lm_head tensor: the head
// is the embedding (tie_word_embeddings). The vision tower
// (`model.vision_tower.*`) and its projector (`model.embed_vision.*`) are in
// the checkpoint and NOT served: the validator counts them as ignored, never
// as unexpected.
//
// Per layer (L sliding or full, config.layer_types):
//   input_layernorm / post_attention_layernorm / pre_feedforward_layernorm /
//   post_feedforward_layernorm .weight        BF16 [H]
//   layer_scalar                              BF16 [1]
//   self_attn.q_norm / k_norm .weight         BF16 [head_dim of the layer]
//             (the value norm has no weight, so no tensor)
//   self_attn.q_proj                          [heads x head_dim, H]
//   self_attn.k_proj                          [kv_heads x head_dim, H]
//   self_attn.v_proj                          [kv_heads x head_dim, H]   sliding layers only
//   self_attn.o_proj                          [H, heads x head_dim]
//   mlp.gate_proj / up_proj                   [I, H]
//   mlp.down_proj                             [H, I]
// and, with the MoE block (Gemma-4-26B-A4B; names as
// bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16 @ main ships them):
//   post_feedforward_layernorm_1 / _2, pre_feedforward_layernorm_2 .weight
//                                             BF16 [H]
//   router.proj.weight                        BF16 [E, H]
//   router.scale                              BF16 [H]
//   router.per_expert_scale                   BF16 [E]
//   moe.experts.E.gate_proj / up_proj         [Im, H]     one Linear per expert
//   moe.experts.E.down_proj                   [H, Im]     (the quantizer's unfused form)
//
// Format contract, by the config's recipe (models/gemma4/config.hpp):
//   NVFP4 (modelopt)  X.weight U8 [N, K/2] (e2m1 pairs, low nibble = even
//                     column) + X.weight_scale F8_E4M3 [N, K/16] +
//                     X.weight_scale_2 F32 [] (+ X.input_scale F32 [] under
//                     Nvfp4Mlp: the recipe's activation scale, unused);
//   BF16              X.weight [N, K].
//   Nvfp4Mlp:        the MLP is NVFP4, the attention projections BF16.
//   Nvfp4WeightOnly: the MLP and the attention projections are NVFP4.
//   The experts are NVFP4 under both; the router is always BF16.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/gemma4/config.hpp"

namespace dgpp {

enum class Gemma4WeightClass : int {
  Embed,       // embed_tokens (also the head)
  FinalNorm,
  LayerNorm,   // the four sandwich norms of a layer
  LayerScalar,
  Attention,   // q/k/v/o projections, q_norm, k_norm
  DenseMlp,    // gate/up/down
  Router,      // the MoE block's proj, scale, per_expert_scale
  RoutedExpert,  // one expert's gate/up/down
};

enum class Gemma4TensorRole : uint8_t {
  Plain,
  Fp4Payload,  // U8 [N, K/2] e2m1 pairs
  Fp4Scale,    // F8_E4M3 [N, K/16]
  Fp4Global,   // F32 [], the matrix's weight_scale_2
  InputScale,  // F32 [], the recipe's activation scale (unused: BF16 activations)
};

struct Gemma4ExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  Gemma4WeightClass cls = Gemma4WeightClass::Attention;
  int layer = -1;   // layer index; -1 for globals
  Gemma4TensorRole role = Gemma4TensorRole::Plain;
  int expert = -1;  // routed-expert id, -1 otherwise

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
  // Present and validated, never read by the loader.
  bool unused() const { return role == Gemma4TensorRole::InputScale; }
};

// "model.language_model.layers.L."
std::string gemma4_layer_prefix(int layer);
// Whether a checkpoint tensor name belongs to a part of the release the
// engine does not serve (the vision / audio encoders and their projectors).
bool gemma4_ignored_tensor(const std::string& name);

std::vector<Gemma4ExpectedTensor> gemma4_expected_text_tensors(const Gemma4TextConfig& cfg);
std::vector<Gemma4ExpectedTensor> gemma4_expected_layer_tensors(const Gemma4TextConfig& cfg, int layer);
std::vector<Gemma4ExpectedTensor> gemma4_expected_global_tensors(const Gemma4TextConfig& cfg);

struct Gemma4TensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct Gemma4BindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t ignored = 0;        // encoder tensors present and skipped
  size_t fp4_matrices = 0;
  size_t bf16_matrices = 0;  // 2-D BF16 tensors (projections, routers, the embedding)
  size_t expert_matrices = 0;  // of the NVFP4 ones, the routed experts'
  size_t bytes = 0;          // matched tensors' bytes
  std::vector<std::string> errors;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

Gemma4BindReport gemma4_validate_text_binding(
    const Gemma4TextConfig& cfg,
    const std::unordered_map<std::string, Gemma4TensorDesc>& present,
    size_t max_errors = 32);

}  // namespace dgpp
