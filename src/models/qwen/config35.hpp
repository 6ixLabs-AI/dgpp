#pragma once
// Qwen3.8-27B (Qwen3_5ForConditionalGeneration) text-model configuration,
// parsed from the checkpoint's config.json text_config. Same policy as the
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
#include <string>
#include <vector>

#include "loaders/minijson.hpp"

namespace dgpp {

enum class Qwen35LayerKind : int { Gdn, Full };

enum class Qwen35QuantKind : int {
  Fp8Block,   // e4m3 + BF16 128x128 block scales, dynamic activations
  Nvfp4Mixed, // compressed-tensors mixed: MLP nvfp4 group16, attn FP8, kv 8b hint
  // The NVIDIA Qwen3-Next release (modelopt): the routed experts, the shared
  // expert, the attention o_proj and the GDN out_proj as e2m1 codes x e4m3
  // scales per 16 x an F32 per-tensor scale; every other matrix BF16.
  Nvfp4Modelopt,
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

  // --- routed MoE (the Qwen3Next dialect; 0 experts = the dense MLP) ---------
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
  // its quantization_config included. The config names no draft layer; the
  // released checkpoints carry one (`mtp.*`), so mtp_num_layers defaults to
  // 1 and the binding refuses a checkpoint without it by name.
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
  int64_t context_limit() const { return static_cast<int64_t>(max_position_embeddings); }
};

}  // namespace dgpp
