#pragma once
// Mistral-Small-4 configuration, parsed from the checkpoint's params.json
// root (2026-10-04, docs/mistral_small4_plan.md §1). The release this engine
// reads — mistralai/Mistral-Small-4-119B-2603-NVFP4 — is in Mistral's own
// format: params.json and `consolidated-*.safetensors`, no config.json. The
// same policy as the other parsers: every field the assembly consumes is
// parsed into a known-supported value or rejected with a message naming the
// field, at load time.
//
// The references are transformers' Mistral4ForCausalLM
// (models/mistral4/modeling_mistral4.py, with convert_mistral4_weight_to_hf.py
// for what each params.json key means) and vLLM's reading of params.json
// (transformers_utils/configs/mistral.py -> MistralLarge3ForCausalLM, a
// DeepSeek-V3 model). The model is DeepSeek-V3's shape, not a plain
// GQA transformer: multi-head latent attention (a 1024-wide query latent,
// a 256-wide key/value latent cached per token, a 64-wide decoupled rope
// key) over a softmax-routed MoE with one shared expert. Three things are
// its own:
//   * the rope is YaRN's frequency blend (factor 128 over an 8192-token
//     original context) with the INTERLEAVED pair layout;
//   * `yarn.apply_scale` is false: no YaRN magnitude correction anywhere —
//     the softmax scale is 1 / sqrt(qk_nope + qk_rope) and the cos / sin are
//     unscaled. vLLM reads the flag this way (rope `attention_factor` 1,
//     the `deepseek_llama_scaling` branch); transformers main multiplies the
//     scale by (0.1 ln factor + 1)^2 since its PR #47435, 5.8.1 does not.
//     The engine follows the checkpoint's flag; a `true` is refused;
//   * `llama_4_scaling`: the query of the token at position p is multiplied
//     by 1 + beta ln(1 + floor(p / original_max_position_embeddings)).
//
// The quantization contract (compressed-tensors 0.13 `nvfp4-pack-quantized`):
// every routed-expert and shared-expert matrix is the quadruple
// X.weight_packed U8 [N, K/2] (e2m1 pairs, the low nibble the even column),
// X.weight_scale F8_E4M3 [N, K/16], X.weight_global_scale F32 [1] (a
// DIVISOR: dequant = e2m1 x e4m3 / global) and X.input_global_scale [1] —
// the recipe's W4A4 activation scale, F32 on routed experts and BF16 on
// shared ones, unused here (the engine feeds the experts full-precision
// activations, as it does for every other NVFP4 release). Attention, the
// router, the norms, the embedding and the head are BF16. Which module is
// quantized is derived from the file's `ignore` rules as compressed-tensors
// applies them — against the module names of the runtime that wrote them
// (vLLM's: `model.layers.L.self_attn.q_a_proj`, `...mlp.experts.E.gate_proj`,
// `lm_head`) — and checked to be the one shape the loader implements.
//
// Text only: the Pixtral tower and its projector are in the checkpoint
// (222 BF16 tensors) and in the binding table, and are never loaded.
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm/moe.hpp"

namespace dgpp {

// The vision tower's shape, read only so the binding table can account for
// its tensors (params.json `vision_encoder`).
struct Mistral4VisionConfig {
  bool present = false;
  int hidden_size = 1024;
  int num_hidden_layers = 24;
  int intermediate_size = 4096;
  int num_channels = 3;
  int patch_size = 14;
  int spatial_merge_size = 2;
  bool pre_mm_projector_norm = true;  // add_pre_mm_projector_layer_norm
};

struct Mistral4TextConfig {
  // --- model shape (the params.json key in brackets) -------------------------
  int hidden_size = 4096;            // [dim]
  int vocab_size = 131072;           // [vocab_size]
  int num_hidden_layers = 36;        // [n_layers]
  float rms_norm_eps = 1e-6f;        // [norm_eps]
  bool tie_word_embeddings = false;  // [tied_embeddings]
  int max_position_embeddings = 1048576;
  int intermediate_size =
      12288;  // [hidden_dim] — the dense MLP's width; no layer has one (first_k_dense_replace 0)
  // params.json carries no token ids; the tokenizer's are <s> 1, </s> 2,
  // <pad> 11 (the transformers conversion pins the same three).
  int64_t bos_token_id = 1;
  int64_t eos_token_id = 2;
  int64_t pad_token_id = 11;

  // --- MLA attention -----------------------------------------------------------
  int num_attention_heads = 32;  // [n_heads]
  int q_lora_rank = 1024;
  int kv_lora_rank = 256;
  int qk_nope_head_dim = 64;
  int qk_rope_head_dim = 64;
  int v_head_dim = 128;

  // --- rope: YaRN, interleaved pairs --------------------------------------------
  double rope_theta = 10000.0;
  double yarn_factor = 128.0;                        // [yarn.factor]
  int yarn_original_max_position_embeddings = 8192;  // [yarn.original_max_position_embeddings]
  double yarn_beta_fast = 32.0;                      // [yarn.beta]
  double yarn_beta_slow = 1.0;                       // [yarn.alpha]
  // [llama_4_scaling]
  double llama4_scaling_beta = 0.1;
  int llama4_original_max_position_embeddings = 8192;

  // --- MoE (every layer; [moe.*]) -------------------------------------------------
  int moe_intermediate_size = 2048;    // [moe.expert_hidden_dim]
  int n_routed_experts = 128;          // [moe.num_experts]
  int n_shared_experts = 1;            // [moe.num_shared_experts]
  int num_experts_per_tok = 4;         // [moe.num_experts_per_tok]
  float routed_scaling_factor = 1.0f;  // [moe.routed_scale]

  // --- weight formats ----------------------------------------------------------
  int fp4_group_size = 16;  // the NVFP4 block (the kernels implement 16)
  // The recipe quantized the experts' input activations too (W4A4): recorded,
  // not applied — the binding table expects the input_global_scale tensors.
  bool activation_scales_present = true;

  Mistral4VisionConfig vision;

  // Parses params.json's root object. Throws std::runtime_error naming the
  // offending field on anything unsupported.
  static Mistral4TextConfig parse(const minijson::Value& root);
  // `path`: params.json, or the snapshot directory holding it.
  static Mistral4TextConfig from_json_file(const std::string& path);
  // parse() accepts every shape the model class defines (the host
  // references and the synthetic test checkpoints are not bound to one
  // geometry). The device assembly is: this throws std::invalid_argument
  // naming the dimension the draft kernels do not implement
  // (docs/mistral_small4_plan.md §4) — a 64-wide rope key, latent and
  // intermediate widths on the NVFP4 / MLA cores' multiples.
  void require_kernel_geometry() const;

  int qk_head_dim() const { return qk_nope_head_dim + qk_rope_head_dim; }
  // The fused [kv latent | rope key] projection's rows.
  int kv_a_rows() const { return kv_lora_rank + qk_rope_head_dim; }
  int shared_expert_inter() const { return n_shared_experts * moe_intermediate_size; }
  // 1 / sqrt(qk_head_dim): the whole softmax scale (apply_scale false).
  float attention_scale() const;
  // The token-at-`pos` query multiplier, in the fp32 the references compute it.
  float llama4_query_scale(int64_t pos) const;
  // The rope's [qk_rope_head_dim / 2] inverse frequencies (the YaRN blend;
  // kernels/rope_scaling.hpp's builder with this config's band).
  std::vector<float> rope_inv_freq() const;

  // The routed chain's configuration (models/glm/moe.hpp): the softmax
  // router, the NVFP4 shared expert in the chain, no swiglu clamps.
  // `local_inter` is this rank's slice of moe_intermediate_size.
  GlmMoeConfig moe_config(int local_inter) const;
};

}  // namespace dgpp
