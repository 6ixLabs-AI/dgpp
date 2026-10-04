#pragma once
// Which model family a checkpoint's config.json describes (Q1/Q2,
// 2026-09-09): the apps dispatch on this before parsing a family's config.
// Read from `architectures[0]` (the transformers class name), with
// `model_type` as the cross-check; anything else is refused by name — the
// engine serves the families it implements, it does not guess.
//
// A Mistral-native checkpoint (2026-10-04) has no config.json: its
// description is `params.json`, which names no class. Such a file is
// recognized by its shape (detect_architecture_params), the way vLLM's
// adapter dispatches it.
#include <string>

#include "loaders/minijson.hpp"

namespace dgpp {

enum class ModelArchitecture : int {
  Glm5,     // Glm5ForConditionalGeneration / glm5_next (GLM-5.3-Flash)
  Qwen4Exp, // Qwen4ExpForConditionalGeneration / qwen4_exp (Qwen3.8-Flash-Next)
  Glm4Moe,  // Glm4MoeForCausalLM / glm4_moe (GLM-4.7, 2026-09-09)
  GlmMoeDsa, // GlmMoeDsaForCausalLM / glm_moe_dsa (full GLM-5.3, 2026-09-12)
  DeepseekV41, // DeepseekV41ForCausalLM / deepseek_v41 (DeepSeek-V4.1-Flash, 2026-09-13)
  MimoV2,      // MiMoV2ForCausalLM / mimo_v2 (MiMo-V2.6-Flash, 2026-09-22)
  Qwen3_5,     // Qwen3_5ForConditionalGeneration / qwen3_5 (Qwen3.8-27B, text-only support lands first)
  DeepseekV4,  // DeepseekV4ForCausalLM / deepseek_v4 (DeepSeek-V4-Flash-0731, 2026-10-01)
  Qwen3Next,   // Qwen3NextForCausalLM / qwen3_next (Qwen3-Next-80B-A3B: the qwen3_5 stack with a routed MoE)
  NemotronH,   // NemotronHForCausalLM / nemotron_h (Nemotron-3 Nano / Super, 2026-10-04: config + binding only)
  Gemma4,      // Gemma4ForConditionalGeneration / gemma4 (Gemma-4-31B, 2026-10-04: host side; kernels a draft)
  // Mistral-Small-4 (2026-10-04, docs/mistral_small4_plan.md): a Mistral-native params.json — latent
  // attention over a softmax-routed MoE. Config, binding table and host references; the device
  // assembly is a draft behind DGPP_BUILD_MISTRAL_DRAFT.
  Mistral4,
  // MiniMax-M2.7 (2026-10-04, docs/minimax_m27_plan.md): MiniMaxM2ForCausalLM / minimax_m2 — GQA
  // with a per-layer q/k norm over a sigmoid-routed MoE. Config, binding table and host references;
  // the device assembly is a draft behind DGPP_BUILD_MINIMAX_DRAFT.
  MiniMaxM2,
  // The plain Qwen3 family (2026-10-04, models/qwen3): full attention in
  // every layer, no Gated DeltaNet.
  Qwen3,       // Qwen3ForCausalLM / qwen3 (dense MLP: Qwen3-Reranker / Qwen3-Embedding; config and binding only)
  Qwen3Moe,    // Qwen3MoeForCausalLM / qwen3_moe (Qwen3-235B-A22B: a routed MoE, no shared expert)
  Qwen3VlMoe,  // Qwen3VLMoeForConditionalGeneration / qwen3_vl_moe (Qwen3-VL-30B-A3B: the text path only)
};

constexpr const char* model_architecture_name(ModelArchitecture a) {
  switch (a) {
    case ModelArchitecture::Qwen4Exp: return "qwen4_exp";
    case ModelArchitecture::Glm4Moe: return "glm4_moe";
    case ModelArchitecture::GlmMoeDsa: return "glm_moe_dsa";
    case ModelArchitecture::Glm5: return "glm5";
    case ModelArchitecture::DeepseekV41: return "deepseek_v41";
    case ModelArchitecture::MimoV2: return "mimo_v2";
    case ModelArchitecture::Qwen3_5: return "qwen3_5";
    case ModelArchitecture::DeepseekV4: return "deepseek_v4";
    case ModelArchitecture::Qwen3Next: return "qwen3_next";
    case ModelArchitecture::NemotronH: return "nemotron_h";
    case ModelArchitecture::Gemma4: return "gemma4";
    case ModelArchitecture::Mistral4: return "mistral4";
    case ModelArchitecture::MiniMaxM2: return "minimax_m2";
    case ModelArchitecture::Qwen3: return "qwen3";
    case ModelArchitecture::Qwen3Moe: return "qwen3_moe";
    case ModelArchitecture::Qwen3VlMoe: return "qwen3_vl_moe";
  }
  return "glm5";
}

// From a parsed config.json root; throws std::runtime_error naming the
// unsupported value.
ModelArchitecture detect_architecture(const minijson::Value& root);
// From a parsed Mistral-native params.json root: Mistral4 when the file has
// the family's shape (an MLA block, a `moe` object with a shared expert and
// `llama_4_scaling` — what vLLM's adapter sends to its DeepSeek-V3 class);
// throws std::runtime_error naming what is missing otherwise.
ModelArchitecture detect_architecture_params(const minijson::Value& root);
// Reads DIR/config.json (or the file itself when `path` names a file). A
// directory without a config.json but with a params.json — or a `path`
// that names a params.json, or a DIR/config.json whose directory holds only
// a params.json (the apps pass that spelling) — is read as Mistral-native.
ModelArchitecture detect_architecture_file(const std::string& path);

}  // namespace dgpp
