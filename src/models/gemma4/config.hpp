#pragma once
// Gemma 4 (Gemma4ForConditionalGeneration, model_type gemma4) configuration,
// parsed from the checkpoint's config.json (2026-10-04). The same policy as
// the other families' parsers: every field the assembly consumes is parsed
// into a known-supported value or rejected with a message naming the field,
// at load time. The reference is the model's class in Hugging Face
// transformers (models/gemma4/modeling_gemma4.py, 5.8.1 and main agree on
// the text model).
//
// STATE (2026-10-04): written with no access to a Spark. The parser, the
// tensor table and the host reference are unit-tested on a Mac; nothing in
// this family has been compiled with nvcc, has loaded a weight or has
// produced a token.
//
// The text backbone (the vision tower and its projector are in the
// checkpoints and are not served):
//
//   x = embed_tokens[ids] * bf16(sqrt(hidden_size))
//   per layer:  x = x + rms(attn(rms(x, input_layernorm)), post_attention_layernorm)
//               x = x + rms(ffn(x), post_feedforward_layernorm)
//               x = x * layer_scalar
//   logits = embed_tokens @ rms(x, norm)          the head is the embedding (tied)
//   logits = tanh(logits / cap) * cap             final_logit_softcapping (30)
//
//   rms(x, w) = x * (mean(x^2) + eps)^-0.5 * w    PLAIN weight (Gemma 2 / 3 wrote 1 + w)
//
//   ffn (dense):    down(gelu_tanh(gate(h)) * up(h)),  h = rms(x, pre_feedforward_layernorm)
//   ffn (MoE block, enable_moe_block — Gemma-4-26B-A4B): the dense MLP and a
//               routed MoE side by side, each under its own norm:
//                 rms(dense(h), post_feedforward_layernorm_1)
//               + rms(experts(rms(x, pre_feedforward_layernorm_2), route(x)), post_feedforward_layernorm_2)
//               route(x): softmax over proj @ (rms_noscale(x) * scale * hidden^-0.5), top_k_experts,
//               the picked probabilities renormalized to sum 1, then times per_expert_scale[e];
//               an expert is down(gelu_tanh(gate(h)) * up(h)) on its own h.
//
// Attention, two kinds on `layer_types` (every sixth layer is full):
//   sliding_attention  head_dim wide heads, num_key_value_heads KV heads, a
//                      separate v_proj, a row reads keys (pos - window, pos]
//                      (`kv_idx > q_idx - sliding_window`), rotary over the
//                      whole head: pairs (i, i + head_dim/2), angle
//                      pos * theta^(-2i/head_dim), theta 1e4;
//   full_attention     global_head_dim wide heads, num_global_key_value_heads
//                      KV heads, NO v_proj: the value is the k_proj output
//                      (attention_k_eq_v), causal over the whole sequence,
//                      "proportional" rotary: the first
//                      int(partial_rotary_factor * global_head_dim / 2) pairs
//                      (i, i + global_head_dim/2) rotate with angle
//                      pos * theta^(-2i/global_head_dim) (the exponent's
//                      divisor is the WHOLE head), theta 1e6; the rest of the
//                      head is not rotated.
//   both: q = rope(rms(q_proj(h), q_norm)), k = rope(rms(k_proj(h), k_norm)),
//         v = rms_noscale(v_proj(h) | k_proj(h)) — the norms run per head over
//         the head's dims, the value's has no weight; logits = q . k with NO
//         1/sqrt(d) (the reference's scaling is 1.0) and no attention
//         soft-cap; query head h reads KV head h / (heads / kv_heads).
//
// The quantization contract — NVIDIA modelopt NVFP4 (e2m1 codes two per byte,
// low nibble = even column; an e4m3 scale per 16 along K; an F32 per-tensor
// `weight_scale_2` that multiplies). Two recipes, told apart by the ignore
// list of config.json's quantization_config:
//   Nvfp4Mlp         nvidia/Gemma-4-31B-IT-NVFP4 (modelopt 0.37.0): every
//                    layer's self_attn is ignored (BF16); the dense MLP's
//                    gate / up / down are NVFP4 and each carries a calibrated
//                    activation scale (`input_scale`, F32) the engine does not
//                    use — activations stay BF16 here, as for every NVFP4
//                    family in this engine;
//   Nvfp4WeightOnly  bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16 (modelopt
//                    0.43, weight-only): attention, dense MLP and experts are
//                    NVFP4 with no activation scale; every layer's router is
//                    ignored (BF16). The experts are stored UNFUSED, one
//                    Linear each (`layers.L.moe.experts.E.gate_proj` /
//                    `up_proj` / `down_proj` — the release's quantizer split
//                    the reference's stacked `experts.gate_up_proj` [E, 2I, H]
//                    into its gate half (rows 0..I) and its up half).
// Whether the activation scales are in the file is the release's, not the
// config's, to say (both configs declare `input_activations`), so it is
// pinned to the recipe; the binding check is what catches a release that
// differs.
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"

namespace dgpp {

enum class Gemma4Recipe : int {
  Nvfp4Mlp,         // BF16 attention, NVFP4 MLP with input_scale
  Nvfp4WeightOnly,  // NVFP4 attention / MLP / experts, no input_scale, BF16 router
};

constexpr const char* gemma4_recipe_name(Gemma4Recipe r) {
  switch (r) {
    case Gemma4Recipe::Nvfp4Mlp: return "nvfp4-mlp";
    case Gemma4Recipe::Nvfp4WeightOnly: return "nvfp4-weight-only";
  }
  return "?";
}

struct Gemma4TextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 5376;
  int vocab_size = 262144;
  int num_hidden_layers = 60;
  float rms_norm_eps = 1e-6f;
  int max_position_embeddings = 262144;
  std::string hidden_activation = "gelu_pytorch_tanh";
  bool tie_word_embeddings = true;      // the checkpoint has no lm_head: the head is the embedding
  float final_logit_softcapping = 30.0f;  // 0: none (config null)
  std::vector<uint8_t> sliding_layer;   // [num_hidden_layers]: 1 = sliding_attention, 0 = full_attention

  // --- tokens ---------------------------------------------------------------
  std::vector<int64_t> eos_token_ids;   // config.json's root list (<eos>, <turn|>); serving adds <|tool_response>
  int64_t bos_token_id = 2;
  int64_t pad_token_id = 0;
  // The multimodal placeholders (root config). Text-only serving refuses a
  // prompt that carries one: its embedding row is not what the model reads
  // there (the reference substitutes the encoder's soft tokens).
  int64_t image_token_id = -1;
  int64_t audio_token_id = -1;
  int64_t video_token_id = -1;

  // --- attention -----------------------------------------------------------
  int num_attention_heads = 32;         // both kinds share the query head count
  int num_key_value_heads = 16;         // sliding layers
  int num_global_key_value_heads = 4;   // full layers (attention_k_eq_v)
  int head_dim = 256;                   // sliding layers
  int global_head_dim = 512;            // full layers
  bool attention_k_eq_v = true;         // full layers: value = the k_proj output, no v_proj
  int sliding_window = 1024;            // a sliding row reads (pos - W, pos]
  double sliding_rope_theta = 1e4;
  double global_rope_theta = 1e6;
  int sliding_rotary_pairs = 128;       // head_dim / 2: the whole head rotates
  int global_rotary_pairs = 64;         // int(partial_rotary_factor * global_head_dim / 2)

  // --- MLP -------------------------------------------------------------------
  int intermediate_size = 21504;        // the dense GeGLU MLP of every layer

  // --- the MoE block (enable_moe_block: Gemma-4-26B-A4B) ------------------
  // Every layer then runs the dense MLP AND a routed MoE side by side:
  //   rms(dense(rms(x, pre_feedforward_layernorm)), post_feedforward_layernorm_1)
  // + rms(experts(rms(x, pre_feedforward_layernorm_2), route(x)), post_feedforward_layernorm_2)
  // route(x): logits = proj @ (rms_noscale(x) * scale * hidden^-0.5), an fp32
  // softmax over all experts, the top_k_experts probabilities renormalized to
  // sum 1, each then times per_expert_scale[e]. An expert is the dense MLP's
  // form on its own input: down(gelu_tanh(gate(h)) * up(h)).
  bool enable_moe_block = false;
  int num_experts = 0;
  int top_k_experts = 0;
  int moe_intermediate_size = 0;        // per expert

  // --- weight formats ---------------------------------------------------------
  Gemma4Recipe recipe = Gemma4Recipe::Nvfp4Mlp;
  int nvfp4_group = 16;                 // the e4m3 scale's group along K

  static Gemma4TextConfig parse(const minijson::Value& root);
  static Gemma4TextConfig from_json_file(const std::string& path);

  bool is_sliding_layer(int l) const { return sliding_layer[static_cast<size_t>(l)] != 0; }
  int num_sliding_layers() const {
    int n = 0;
    for (uint8_t s : sliding_layer) n += s != 0;
    return n;
  }
  int num_full_layers() const { return num_hidden_layers - num_sliding_layers(); }
  // Per-layer attention geometry.
  int head_dim_of(int l) const { return is_sliding_layer(l) ? head_dim : global_head_dim; }
  int kv_heads_of(int l) const { return is_sliding_layer(l) ? num_key_value_heads : num_global_key_value_heads; }
  // Whether layer `l` has its own v_proj (the full layers reuse k_proj's output).
  bool has_v_proj(int l) const { return is_sliding_layer(l) || !attention_k_eq_v; }
  double rope_theta_of(int l) const { return is_sliding_layer(l) ? sliding_rope_theta : global_rope_theta; }
  // Rotated pairs of layer `l`: pair i couples dims (i, i + head_dim_of(l)/2)
  // with inverse frequency theta^(-2i / head_dim_of(l)).
  int rotary_pairs_of(int l) const { return is_sliding_layer(l) ? sliding_rotary_pairs : global_rotary_pairs; }
  int64_t q_rows(int l) const { return static_cast<int64_t>(num_attention_heads) * head_dim_of(l); }
  int64_t kv_rows(int l) const { return static_cast<int64_t>(kv_heads_of(l)) * head_dim_of(l); }
  // The embedding scale as the reference applies it: sqrt(hidden_size)
  // rounded to bf16 (the embedding table's dtype) — 73.5 for 5376.
  float embed_scale() const;
  // The router's input scale as the reference applies it on a bf16
  // checkpoint: hidden_size^-0.5 rounded to bf16 (a Python float against a
  // bf16 tensor in transformers; `root_size.to(x.dtype)` in the vLLM model
  // file the 26B release ships) — 0.018798828125 for 2816, not 0.0188445.
  float router_input_scale() const;
  // The recipe's per-class formats.
  bool attention_nvfp4() const { return recipe == Gemma4Recipe::Nvfp4WeightOnly; }
  bool activation_scales() const { return recipe == Gemma4Recipe::Nvfp4Mlp; }
};

}  // namespace dgpp
