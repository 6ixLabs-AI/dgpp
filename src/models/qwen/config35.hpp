#pragma once
// Qwen3.8-27B (Qwen3_5ForConditionalGeneration) text-model configuration,
// parsed from the checkpoint's config.json text_config — and, with the
// routed MoE in the dense MLP's place, Qwen3.6-35B-A3B's
// (Qwen3_5MoeForConditionalGeneration, text_config type qwen3_5_moe_text). Same policy as the
// other families: every field the assembly consumes is parsed into a
// known-supported value or rejected with a message naming the field, at
// load time. The reference is transformers' modular_qwen3_5.py.
//
// This family differs from Qwen4Exp (Flash-Next): no hyper-connections, no
// MoE, no indexer, no n-gram table. Its layers are plain Gated DeltaNet
// (linear_attention, swish output gate) and plain full attention (GQA with
// partial rotary + MROPE + a per-head output gate), sandwiching a dense
// SwiGLU MLP, between two RMSNorms. Text-only scope: vision_config is
// ignored (image requests stay refused until the tower lands).
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "kernels/rope_scaling.hpp"

#include "loaders/minijson.hpp"

namespace dgpp {

enum class Qwen35LayerKind : int { Gdn, Full };

enum class Qwen35QuantKind : int {
  Fp8Block,   // e4m3 + BF16 128x128 block scales, dynamic activations
  // The NVIDIA Qwen3.6-35B-A3B release (modelopt MIXED_PRECISION, 2026-10-04):
  // an 8-bit float group — the GDN in_proj_qkv / in_proj_z / out_proj and the
  // attention q/k/v/o as e4m3 codes x ONE F32 scale per tensor — and a 4-bit
  // float group per 16 — the routed experts, the shared expert and the lm
  // head as the modelopt NVFP4 set; the draft layer BF16 (its experts as two
  // stacked tensors). The binding table exists for the MoE shape only.
  Nvfp4Mixed,
  // The NVIDIA Qwen3-Next release (modelopt): the routed experts, the shared
  // expert, the attention o_proj and the GDN out_proj as e2m1 codes x e4m3
  // scales per 16 x an F32 per-tensor scale; every other matrix BF16.
  Nvfp4Modelopt,
  // RedHatAI/Qwen3-Coder-Next-NVFP4 (compressed-tensors `nvfp4-pack-quantized`,
  // 2026-10-04): the routed experts, the shared expert and the attention
  // q/k/v/o as `weight_packed` e2m1 codes x `weight_scale` e4m3 per 16 over
  // an F32 `weight_global_scale` (a divisor, where modelopt's weight_scale_2
  // multiplies); the GDN projections, the router and the head BF16. No
  // K/V-cache scales and no draft layer in that release.
  //   The same container on the Qwen3.5 dialect's routed-MoE models
  // (Sehyo/Qwen3.5-122B-A10B-NVFP4, Sehyo/Qwen3.5-35B-A3B-NVFP4): the same
  // matrices quantized, the split GDN projections BF16, and a BF16 draft
  // layer with per-expert matrices (the recipe ignores `mtp.*`).
  Nvfp4Packed,
  // nvidia/Qwen3.5-122B-A10B-NVFP4 (modelopt `quant_algo` NVFP4 with one
  // group and an ignore list, 2026-10-04): the routed experts alone as the
  // modelopt NVFP4 set; the GDN, the attention, the shared expert, the router
  // and the head BF16; a BF16 draft layer with per-expert matrices. The
  // config's ignore list is held to exactly that (config35.cpp).
  Nvfp4Experts,
  // An unquantized release (no quantization_config: Qwen/Qwen3.5-0.8B,
  // 2026-10-04): every matrix BF16 as the model was trained, bound as it
  // ships — or, under engine.dense_weights = fp8, encoded to block FP8 at
  // load like the Qwen3Next dialect's BF16 projections. The GDN's A_log and
  // its output norm weight are F32 in that release and BF16 in others of
  // this kind (binding.hpp's Bf16OrF32).
  Bf16,
};

// Which checkpoint layout the config describes. Both run the same walk
// (pre-norm residual, swish-gated GDN, gated GQA, one draft layer); the
// dialect picks the tensor names, the fused or split GDN projections and
// the MLP (dense SwiGLU or the routed MoE).
enum class Qwen35Dialect : int {
  Qwen35,     // Qwen3_5ForConditionalGeneration: text_config, model.language_model.*
  // Qwen3NextForCausalLM (Qwen3-Next-80B-A3B, 2026-10-03): a flat config,
  // `model.layers.L.*`, the GDN projections fused and interleaved per key
  // head (`in_proj_qkvz`, `in_proj_ba`), a routed MoE in every layer.
  Qwen3Next,
};

struct Qwen35TextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 5120;
  int vocab_size = 248320;
  int num_hidden_layers = 64;
  float rms_norm_eps = 1e-6f;
  // Tied: the checkpoint stores no `lm_head.weight` and the head reads the
  // embedding matrix (the Qwen3.5 dialect's small models; the Qwen3Next
  // dialect refuses it).
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  int max_position_embeddings = 262144;
  std::vector<Qwen35LayerKind> layers;  // size == num_hidden_layers
  std::vector<int64_t> eos_token_ids;
  int64_t bos_token_id = -1;

  // --- Gated DeltaNet (linear_attention) ----------------------------------
  int gdn_key_heads = 16;
  int gdn_value_heads = 48;
  int gdn_key_head_dim = 128;
  int gdn_value_head_dim = 128;
  int gdn_conv_width = 4;
  // This family trains the swish output gate; the gated kernels land with
  // the layer slice (the config records it now so checkpoints bind today).
  std::string output_gate_type = "swish";

  // --- full attention (GQA + partial rotary + MROPE + output gate) --------
  int num_attention_heads = 24;
  int num_key_value_heads = 4;
  int head_dim = 256;
  int rotary_dim = 64;  // head_dim * partial_rotary_factor
  double rope_theta = 1e7;
  std::vector<int> mrope_section;  // [11, 11, 10]; text positions are 1-D
  bool mrope_interleaved = true;
  // The q projection stacks [q | gate] per head (attn_output_gate), so its
  // row count is 2 * heads * head_dim.
  bool attn_output_gate = true;

  // --- dense SwiGLU MLP ----------------------------------------------------
  int intermediate_size = 17408;

  // --- routed MoE (Qwen3Next, Qwen3_5Moe; 0 experts = the dense MLP) ---------
  // Softmax top-k routed experts plus a shared expert weighted by
  // sigmoid(x . g): Flash-Next's block (models/qwen/moe_layer.hpp) on the
  // plain residual. Every layer is sparse (decoder_sparse_step 1, no
  // mlp_only_layers), the draft layer included.
  int num_experts = 0;
  int num_experts_per_tok = 0;
  int moe_intermediate_size = 0;
  int shared_expert_intermediate_size = 0;
  bool norm_topk_prob = true;

  // --- MTP ------------------------------------------------------------------
  int mtp_num_layers = 1;  // 0 or 1; the draft layer is a Full layer

  // --- weight formats -------------------------------------------------------
  Qwen35QuantKind quant_kind = Qwen35QuantKind::Fp8Block;
  Qwen35Dialect dialect = Qwen35Dialect::Qwen35;

  static Qwen35TextConfig parse(const minijson::Value& text_config,
                                const minijson::Value* quantization_config);
  // The Qwen3Next dialect: `root` is the whole (flat) config.json object,
  // its quantization_config included. The config names no draft layer, so
  // mtp_num_layers follows the release the recipe identifies: 1 under the
  // modelopt recipe (Qwen3-Next-80B-A3B carries `mtp.*`; the binding refuses
  // a checkpoint without it by name), 0 under the compressed-tensors recipe
  // (Qwen3-Coder-Next carries none; a `mtp.*` tensor is then unexpected). A
  // config that does name mtp_num_hidden_layers is taken at its word.
  static Qwen35TextConfig parse_qwen3_next(const minijson::Value& root);
  // Reads config.json from disk (the root object) and dispatches to parse()
  // (a text_config object) or parse_qwen3_next() (model_type qwen3_next).
  static Qwen35TextConfig from_json_file(const std::string& path);

  bool next() const { return dialect == Qwen35Dialect::Qwen3Next; }
  bool moe() const { return num_experts > 0; }

  int num_gdn_layers() const;
  int num_full_layers() const;
  // Layer index of the MTP draft layer (num_hidden_layers), -1 when absent.
  int mtp_layer() const { return mtp_num_layers == 1 ? num_hidden_layers : -1; }
  // The opt-in YaRN ramp (engine.rope_scaling), set by the serving layer on
  // the parsed config for the Qwen3Next dialect — never by the checkpoint,
  // whose rope must stay unscaled. Qwen validated this model to 1M tokens
  // with YaRN (factor 4 over 262,144; factor 2 for 524,288). The full
  // attention layers build the ramp's table and its attention scale from it
  // (QwenFullAttnLayer); the GDN layers have no positions to scale. Empty =
  // the plain table, bit for bit what this stack built before.
  std::optional<RopeScaling> rope_scaling;
  // One request's positional ceiling: the checkpoint's, or the ramp's.
  int64_t context_limit() const {
    return rope_scaling.has_value() ? rope_scaling->context_limit()
                                    : static_cast<int64_t>(max_position_embeddings);
  }
};

}  // namespace dgpp
