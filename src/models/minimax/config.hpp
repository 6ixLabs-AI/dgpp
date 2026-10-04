#pragma once
// MiniMax-M2.7 (MiniMaxM2ForCausalLM, model_type minimax_m2) configuration,
// parsed from the checkpoint's config.json root (2026-10-04,
// docs/minimax_m27_plan.md §1). The same policy as the other parsers: every
// field the assembly consumes is parsed into a known-supported value or
// rejected with a message naming the field, at load time. The references
// are the modeling_minimax_m2.py the checkpoint ships and transformers'
// minimax_m2 (the same model: FlexOlmo's attention over Mixtral's block).
//
// The model is GLM-4.7's shape with four differences that matter here:
//   * the q / k norm is PER LAYER ("qk_norm_type": "per_layer"): one RMSNorm
//     over the whole q projection (48 heads x 128) and one over the whole k
//     projection (8 x 128), with a weight per element, before the split into
//     heads — GLM-4.7 norms each head on its own with one shared [128]
//     weight. Under tensor parallelism the RMS therefore couples every
//     rank's heads (docs/minimax_m27_plan.md §4);
//   * no attention biases, no shared expert ("shared_intermediate_size": 0),
//     no dense layers: every layer is attention + a routed MoE;
//   * the router's weights are the picked sigmoid scores renormalized, with
//     no scaling factor (GLM's routed_scaling_factor is 1 here);
//   * config.json declares an MTP head (`use_mtp`, `num_mtp_modules` 3,
//     `mtp_transformer_layers` 1) that no published checkpoint carries and
//     no published code defines: the fields are recorded, the binding table
//     has no draft tensors, and the engine has no draft head for this model.
//
// The quantization contract (lukealonso/MiniMax-M2.7-NVFP4, modelopt 0.39):
// only the routed experts' w1 / w2 / w3 are NVFP4 — e2m1 codes two per byte
// (`weight`, U8 [N, K/2]), e4m3 block scales per 16 (`weight_scale`,
// [N, K/16]) and one fp32 per-tensor scale (`weight_scale_2`, a multiplier);
// `input_scale` (the W4A4 activation scale, in a file of its own, absent
// for experts the calibration never routed to) rides along unused — the
// model card itself runs the experts on 16-bit activations. Attention, the
// router and its bias, the norms, the embedding and the head are BF16.
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm/moe.hpp"

namespace dgpp {

struct MinimaxTextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 3072;
  int vocab_size = 200064;
  int num_hidden_layers = 62;
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = false;
  int max_position_embeddings = 196608;
  // config.json's own token ids. In the release they are 1 and 2, which
  // are byte tokens, not the tokenizer's control tokens: serving resolves
  // its stop ids from generation_config.json (eos 200020) and the tokenizer.
  int64_t config_bos_token_id = -1;
  int64_t config_eos_token_id = -1;

  // --- attention (GQA, no biases, per-layer q/k norm, partial RoPE) -------
  int num_attention_heads = 48;
  int num_key_value_heads = 8;
  int head_dim = 128;
  int rotary_dim = 64;  // head_dim * partial_rotary_factor; half-split pairs (i, i + 32)
  double rope_theta = 5e6;

  // --- MoE (every layer) ---------------------------------------------------
  int moe_intermediate_size = 1536;  // config.json `intermediate_size`: an expert's width
  int n_routed_experts = 256;        // `num_local_experts`
  int num_experts_per_tok = 8;

  // --- declared, not present ------------------------------------------------
  bool mtp_declared = false;     // `use_mtp`
  int mtp_modules_declared = 0;  // `num_mtp_modules`

  // --- weight formats -----------------------------------------------------
  int fp4_group_size = 16;  // the NVFP4 block (the kernels implement 16)

  // Parses config.json's root object (flat: no text_config). Throws
  // std::runtime_error naming the offending field on anything unsupported.
  static MinimaxTextConfig parse(const minijson::Value& root);
  // `path`: config.json, or the snapshot directory holding it.
  static MinimaxTextConfig from_json_file(const std::string& path);
  // parse() accepts every shape the model class defines; this throws
  // std::invalid_argument naming the dimension the attention and NVFP4
  // kernels do not implement (128-wide heads, K multiples of 32).
  void require_kernel_geometry() const;

  int q_heads_per_kv() const { return num_attention_heads / num_key_value_heads; }
  int q_width() const { return num_attention_heads * head_dim; }
  int kv_width() const { return num_key_value_heads * head_dim; }
  float attention_scale() const;
  // The rope's [rotary_dim / 2] inverse frequencies, fp32 as torch builds
  // them: 1 / theta^(2i / rotary_dim).
  std::vector<float> rope_inv_freq() const;
  // The routed chain's configuration (models/glm/moe.hpp): the sigmoid
  // router with its bias, no shared expert, no swiglu clamps, no scaling.
  // `local_inter` is this rank's slice of moe_intermediate_size.
  GlmMoeConfig moe_config(int local_inter) const;
};

}  // namespace dgpp
