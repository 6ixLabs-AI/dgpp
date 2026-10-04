#pragma once
// The "plain" Qwen3 family (2026-10-04): the Qwen3 architecture without the
// Gated DeltaNet layers the qwen3_5 stack is built around — every layer is
// ordinary full attention followed by a dense SwiGLU MLP or a routed MoE.
// Parsed from the checkpoint's config.json with the families' policy: every
// field the assembly consumes becomes a known-supported value or is rejected
// with a message naming the field, at load time.
//
// The reference is transformers 4.57.1: modeling_qwen3.py (Qwen3ForCausalLM),
// modeling_qwen3_moe.py (Qwen3MoeForCausalLM) and modeling_qwen3_vl_moe.py
// (Qwen3VLMoeForConditionalGeneration, whose text model is the same walk).
//
//   h = embed(x)
//   per layer: h += attn(rmsnorm(h, input_layernorm))
//              h += mlp(rmsnorm(h, post_attention_layernorm))
//   logits = lm_head(rmsnorm(h, norm))                (lm_head tied to embed in the dense models)
//
//   rmsnorm(x, w) = w * (x * rsqrt(mean(x^2) + eps))  (a PLAIN weight: not the
//                   zero-centred 1 + w of Qwen3-Next)
//   attention: unbiased q/k/v/o; q and k RMS-normed per head over head_dim
//              (q_norm / k_norm, one weight vector shared by the heads), then
//              FULL rotary over all head_dim dims in transformers' rotate_half
//              pairing (i, i + head_dim/2), angle pos * theta^(-2i/head_dim);
//              GQA (query head h reads kv head h / (heads / kv_heads)), scale
//              head_dim^-0.5, causal. No output gate, no partial rotary.
//   dense MLP: down(silu(gate(x)) * up(x)).
//   routed MoE (Qwen3MoeSparseMoeBlock): p = softmax(gate(x)) over all
//              experts in fp32, top-k, weights p[top] / sum(p[top])
//              (norm_topk_prob), y = sum_e w_e * down_e(silu(gate_e x) * up_e x).
//              NO shared expert and no router bias.
//
// Three dialects, one walk:
//   Dense   Qwen3ForCausalLM / qwen3 — flat config, `model.layers.L.*` (the
//           retrieval models Qwen3-Reranker-0.6B / Qwen3-Embedding-0.6B: BF16,
//           tied embeddings; the Embedding release stores the BASE model's
//           state dict, without the `model.` prefix and without a head).
//   Moe     Qwen3MoeForCausalLM / qwen3_moe — flat config, `model.layers.L.*`
//           (nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4, modelopt).
//   VlMoe   Qwen3VLMoeForConditionalGeneration / qwen3_vl_moe — text_config +
//           vision_config, `model.language_model.layers.L.*`, the tower under
//           `model.visual.*` (ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4,
//           compressed-tensors). TEXT ONLY here: the tower's tensors are bound
//           (the checkpoint validates whole) and never loaded; image inputs
//           are refused by name. A text-only prompt gives all three MROPE
//           axes the same position, so the interleaved MROPE of the text
//           model is exactly the plain 1-D rope above.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm/moe.hpp"
#include "models/glm4/config.hpp"

namespace dgpp {

enum class Qwen3Dialect : int {
  Dense,  // Qwen3ForCausalLM: a dense SwiGLU MLP in every layer
  Moe,    // Qwen3MoeForCausalLM: a routed MoE in every layer
  VlMoe,  // Qwen3VLMoeForConditionalGeneration: Moe's text walk + a vision tower (bound, not served)
};

enum class Qwen3QuantKind : int {
  // No quantization_config: every matrix BF16 (the dense retrieval models).
  Bf16,
  // NVIDIA modelopt (nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4): the routed
  // experts and the attention o_proj as `weight` U8 [N, K/2] e2m1 pairs,
  // `weight_scale` e4m3 [N, K/16], F32 [] `weight_scale_2` (a multiplier)
  // and F32 [] `input_scale`; q/k/v, the router and the head BF16; the
  // recipe's FP8 K/V-cache scales `k_proj.k_scale` / `v_proj.v_scale` F32 [].
  Nvfp4Modelopt,
  // compressed-tensors `nvfp4-pack-quantized` (ig1/Qwen3-VL-30B-A3B-Instruct-
  // NVFP4, llm-compressor): the routed experts and the attention q/k/v/o as
  // `weight_packed` U8 [N, K/2], `weight_scale` e4m3 [N, K/16], F32 [1]
  // `weight_global_scale` (a DIVISOR) and F32 [1] `input_global_scale`; the
  // router, the head and the whole vision tower BF16; no K/V-cache scales.
  Nvfp4Packed,
};

// The Qwen3-VL vision tower's shape (vision_config): parsed so its tensors
// bind by name, dtype and shape. Nothing here runs it.
struct Qwen3VisionConfig {
  int depth = 0;
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_heads = 0;
  int in_channels = 3;
  int patch_size = 16;
  int temporal_patch_size = 2;
  int spatial_merge_size = 2;
  int num_position_embeddings = 0;
  int out_hidden_size = 0;
  std::vector<int> deepstack_visual_indexes;  // one deepstack merger per entry
  // The merger's input width: spatial_merge_size^2 patches of hidden_size.
  int64_t merged_width() const {
    return static_cast<int64_t>(hidden_size) * spatial_merge_size * spatial_merge_size;
  }
};

struct Qwen3TextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 2048;
  int vocab_size = 151936;
  int num_hidden_layers = 48;
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  int max_position_embeddings = 262144;
  std::vector<int64_t> eos_token_ids;
  int64_t bos_token_id = -1;
  int64_t pad_token_id = -1;

  // --- attention (GQA, q/k head norm, full rotary) --------------------------
  int num_attention_heads = 32;
  int num_key_value_heads = 4;
  int head_dim = 128;
  double rope_theta = 5e6;
  // The VL text model's MROPE sections (three, summing to head_dim / 2);
  // empty in the text-only dialects. Text positions are 1-D on all three
  // axes, so the sections never change a served request (the header).
  std::vector<int> mrope_section;

  // --- dense SwiGLU MLP (the Dense dialect; unused by the MoE dialects) -----
  int intermediate_size = 6144;

  // --- routed MoE (0 experts = the dense MLP) ---------------------------------
  int num_experts = 0;
  int num_experts_per_tok = 0;
  int moe_intermediate_size = 0;
  bool norm_topk_prob = true;

  // --- the vision tower (VlMoe; bound, never served) --------------------------
  std::optional<Qwen3VisionConfig> vision;
  // The checkpoint's image / video placeholder and delimiter ids (VlMoe):
  // a prompt that carries one of the two pads asks for the tower.
  int64_t image_token_id = -1;
  int64_t video_token_id = -1;
  int64_t vision_start_token_id = -1;
  int64_t vision_end_token_id = -1;

  // --- weight formats and names -------------------------------------------------
  Qwen3QuantKind quant_kind = Qwen3QuantKind::Bf16;
  Qwen3Dialect dialect = Qwen3Dialect::Moe;
  // modelopt's FP8 K/V-cache scheme: `k_proj.k_scale` / `v_proj.v_scale`
  // F32 [] beside every layer's k and v (bound; nothing reads them yet).
  bool kv_cache_scales = false;
  // Qwen3-Embedding stores the base model's state dict: `layers.L.*`,
  // `embed_tokens.weight`, `norm.weight` — no `model.` prefix, no head. The
  // config cannot say so (its config.json is the Reranker's but for two
  // fields); the checkpoint's header does (qwen3_apply_header_naming).
  bool bare_names = false;

  // Parses config.json's root object. Throws std::runtime_error naming the
  // offending field on anything unsupported.
  static Qwen3TextConfig parse(const minijson::Value& root);
  static Qwen3TextConfig from_json_file(const std::string& path);

  bool moe() const { return num_experts > 0; }
  bool vl() const { return dialect == Qwen3Dialect::VlMoe; }
  bool fp4() const { return quant_kind != Qwen3QuantKind::Bf16; }
  bool fp4_packed() const { return quant_kind == Qwen3QuantKind::Nvfp4Packed; }
  int q_heads_per_kv() const { return num_attention_heads / num_key_value_heads; }
  // The family name the serving layer reports and gates options on.
  const char* family_name() const {
    return dialect == Qwen3Dialect::Dense ? "qwen3" : dialect == Qwen3Dialect::Moe ? "qwen3_moe" : "qwen3_vl_moe";
  }
  // Human name of the object a refusal is about ("Qwen3-MoE config").
  const char* scope_name() const;

  // The routed chain's configuration (models/glm/moe.hpp): the softmax
  // top-k router, NO shared expert in or beside the chain, no scaling
  // factor, no swiglu clamps — QwenMoeLayer::routed_config's values, taken
  // here so the host library needs no CUDA header. `local_inter` is this
  // rank's slice of moe_intermediate_size.
  GlmMoeConfig moe_config(int local_inter) const;
  // The GLM-4.7 attention layer's view of this model (models/glm4/layers.hpp
  // reads hidden_size, head_dim, rotary_dim, rope_theta, rms_norm_eps and
  // use_qk_norm): the same 128-wide GQA kernels with no biases and the
  // rotary over the whole head (rotary_dim = head_dim).
  Glm4TextConfig attention_view() const;
};

}  // namespace dgpp
