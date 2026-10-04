#include "loaders/architecture.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace dgpp {

ModelArchitecture detect_architecture(const minijson::Value& root) {
  if (!root.is_object())
    throw std::runtime_error("config.json: root is not an object");
  std::string arch;
  if (const minijson::Value* a = root.find("architectures")) {
    if (!a->is_array() || a->items().empty() || !a->items()[0].is_string())
      throw std::runtime_error("config.json: architectures must be a non-empty string array");
    arch = std::string(a->items()[0].as_string());
  }
  std::string type;
  if (const minijson::Value* t = root.find("model_type"))
    if (t->is_string()) type = std::string(t->as_string());
  // Full GLM-5.3 (2026-09-12) is `GlmMoeDsaForCausalLM` / `glm_moe_dsa`;
  // the Flash checkpoint's class starts with `Glm5` (its model_type is
  // `glm5_next`, an older release wrote `glm_moe_dsa` there too, which is
  // why the class name is read first).
  if (arch.rfind("GlmMoeDsa", 0) == 0 || (arch.empty() && type == "glm_moe_dsa"))
    return ModelArchitecture::GlmMoeDsa;
  if (arch.rfind("Glm5", 0) == 0 || (arch.empty() && type == "glm5_next"))
    return ModelArchitecture::Glm5;
  if (arch.rfind("Qwen4Exp", 0) == 0 || (arch.empty() && type == "qwen4_exp"))
    return ModelArchitecture::Qwen4Exp;
  if (arch.rfind("Glm4Moe", 0) == 0 || (arch.empty() && type == "glm4_moe"))
    return ModelArchitecture::Glm4Moe;
  // DeepSeek-V4.1-Flash (2026-09-13, docs/deepseek_v41_flash_plan.md):
  // `DeepseekV41ForCausalLM` / `deepseek_v41` (its text_config's type is
  // `deepseek_v41_text`).
  if (arch.rfind("DeepseekV41", 0) == 0 || (arch.empty() && type == "deepseek_v41"))
    return ModelArchitecture::DeepseekV41;
  // DeepSeek-V4-Flash (the 0731 release, 2026-10-01): `DeepseekV4ForCausalLM`
  // / `deepseek_v4` — a flat config, not V4.1's nested text_config.
  if (arch.rfind("DeepseekV4For", 0) == 0 || (arch.empty() && type == "deepseek_v4"))
    return ModelArchitecture::DeepseekV4;
  // MiMo-V2.6-Flash (2026-09-22, docs/mimo_v26_flash_plan.md):
  // `MiMoV2ForCausalLM` / `mimo_v2`.
  if (arch.rfind("MiMoV2", 0) == 0 || (arch.empty() && type == "mimo_v2"))
    return ModelArchitecture::MimoV2;
  // Qwen3.8-27B (2026-09-27): `Qwen3_5ForConditionalGeneration` / `qwen3_5`
  // (its text_config's type is `qwen3_5_text`).
  // Qwen3.6-35B-A3B (2026-10-04) is the same stack with a routed MoE:
  // `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe`.
  if (arch.rfind("Qwen3_5", 0) == 0 || (arch.empty() && (type == "qwen3_5" || type == "qwen3_5_moe")))
    return ModelArchitecture::Qwen3_5;
  // Qwen3-Next-80B-A3B (2026-10-03): `Qwen3NextForCausalLM` / `qwen3_next`
  // — a flat config (no text_config), served by the qwen3_5 stack with the
  // Flash-Next routed MoE in place of the dense MLP.
  if (arch.rfind("Qwen3Next", 0) == 0 || (arch.empty() && type == "qwen3_next"))
    return ModelArchitecture::Qwen3Next;
  // MiniMax-M2.7 (2026-10-04, docs/minimax_m27_plan.md):
  // `MiniMaxM2ForCausalLM` / `minimax_m2` — a flat config.
  if (arch.rfind("MiniMaxM2", 0) == 0 || (arch.empty() && type == "minimax_m2"))
    return ModelArchitecture::MiniMaxM2;
  // Mistral-Small-4 in transformers' format (`Mistral3ForConditionalGeneration`
  // over a `mistral4` text_config, or `Mistral4ForCausalLM`): the same model
  // with fused expert tensors and other names. Only the Mistral-native
  // release (params.json) has a binding table here.
  if (arch.rfind("Mistral4", 0) == 0 || arch.rfind("Mistral3", 0) == 0 || type == "mistral4" || type == "mistral3")
    throw std::runtime_error(
        "config.json: architecture '" + arch + "' (model_type '" + type +
        "') is Mistral's transformers-format release, which is not implemented; the engine reads the "
        "Mistral-native release (params.json, e.g. mistralai/Mistral-Small-4-119B-2603-NVFP4)");
  throw std::runtime_error(
      "config.json: unsupported architecture '" + arch + "' (model_type '" +
      type + "'); the engine implements Glm5*, Qwen4Exp*, Glm4Moe*, GlmMoeDsa*, DeepseekV41*, DeepseekV4*, MiMoV2*, Qwen3_5*, Qwen3Next* and MiniMaxM2*, and Mistral-native params.json (Mistral-Small-4)");
}

ModelArchitecture detect_architecture_params(const minijson::Value& root) {
  if (!root.is_object()) throw std::runtime_error("params.json: root is not an object");
  const auto has = [&](std::string_view key) {
    const minijson::Value* v = root.find(key);
    return v && !v->is_null();
  };
  const minijson::Value* moe = root.find("moe");
  const bool shared_moe = moe && moe->is_object() && moe->find("num_shared_experts") &&
                          moe->find("num_shared_experts")->as_int() > 0;
  if (has("qk_nope_head_dim") && has("kv_lora_rank") && shared_moe && has("llama_4_scaling"))
    return ModelArchitecture::Mistral4;
  throw std::runtime_error(
      "params.json: unsupported Mistral-native model; the engine implements the latent-attention MoE family "
      "(Mistral-Small-4: qk_nope_head_dim / kv_lora_rank, moe.num_shared_experts > 0 and llama_4_scaling)");
}

namespace {

std::string read_text(const std::filesystem::path& p, const char* what) {
  FILE* f = std::fopen(p.c_str(), "rb");
  if (!f)
    throw std::runtime_error(std::string("cannot open ") + what + " " + p.string() + ": " +
                             std::strerror(errno));
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  return text;
}

}  // namespace

ModelArchitecture detect_architecture_file(const std::string& path) {
  namespace fs = std::filesystem;
  const fs::path p = fs::is_directory(path) ? fs::path(path) / "config.json"
                                            : fs::path(path);
  // The Mistral-native layout: params.json and no config.json. `p` is the
  // config.json this function would read; when it does not exist and its
  // directory holds a params.json, that file describes the checkpoint.
  fs::path params;
  if (p.filename() == "params.json")
    params = p;
  else if (!fs::exists(p) && fs::exists(p.parent_path() / "params.json"))
    params = p.parent_path() / "params.json";
  if (!params.empty()) {
    const std::string text = read_text(params, "params");
    const auto parsed = minijson::parse(text);
    return detect_architecture_params(parsed.root);
  }
  const std::string text = read_text(p, "config");
  const auto parsed = minijson::parse(text);
  return detect_architecture(parsed.root);
}

}  // namespace dgpp
