// qwen35_bind_check: offline validation of a qwen3_5-stack checkpoint
// against the family's expected-tensor table — Qwen3.8-27B (the Qwen35
// dialect) or Qwen3-Next-80B-A3B (the Qwen3Next dialect: flat config, fused
// GDN projections, the routed MoE, the modelopt NVFP4 set). Reads
// config.json and every safetensors header; no payload bytes are touched.
// Exit 0 only when the binding is exact: every expected tensor present with
// its dtype and shape, nothing unexpected.
//
//   qwen35_bind_check --model ORG/NAME | (--config <dir>/config.json
//                                        --checkpoint-dir <dir>) [--max-errors N]
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "loaders/architecture.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/safetensors.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"

namespace {

int run(int argc, char** argv) {
  std::string config_path, checkpoint_dir, model_id;
  size_t max_errors = 32;
  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    auto next = [&]() -> std::string_view {
      if (i + 1 >= argc) throw std::runtime_error(std::format("missing value for {}", a));
      return argv[++i];
    };
    if (a == "--config") config_path = next();
    else if (a == "--checkpoint-dir") checkpoint_dir = next();
    else if (a == "--model") model_id = next();
    else if (a == "--max-errors") max_errors = std::stoul(std::string(next()));
    else throw std::runtime_error(std::format("unknown argument {}", a));
  }
  if (!model_id.empty()) {
    if (!checkpoint_dir.empty()) throw std::runtime_error("--model and --checkpoint-dir are mutually exclusive");
    std::string err;
    const std::string snapshot = dgpp::hf::model_dir(model_id, &err);
    if (snapshot.empty()) throw std::runtime_error(std::format("--model {}: {}", model_id, err));
    checkpoint_dir = snapshot;
    if (config_path.empty()) config_path = snapshot + "/config.json";
    std::printf("model: %s -> %s\n", model_id.c_str(), snapshot.c_str());
  }
  if (checkpoint_dir.empty())
    throw std::runtime_error(
        "usage: qwen35_bind_check --model ORG/NAME | (--config <config.json> --checkpoint-dir <dir>) [--max-errors N]");
  if (config_path.empty()) config_path = checkpoint_dir + "/config.json";

  const dgpp::ModelArchitecture arch = dgpp::detect_architecture_file(config_path);
  if (arch != dgpp::ModelArchitecture::Qwen3_5 && arch != dgpp::ModelArchitecture::Qwen3Next)
    throw std::runtime_error(std::format("{} is a {} checkpoint, not qwen3_5 or qwen3_next", config_path,
                                         dgpp::model_architecture_name(arch)));
  const dgpp::Qwen35TextConfig cfg = dgpp::Qwen35TextConfig::from_json_file(config_path);
  std::printf("config: %s, %d layers (%d GDN + %d full attention) + %d draft, hidden %d, vocab %d\n",
              dgpp::model_architecture_name(arch), cfg.num_hidden_layers, cfg.num_gdn_layers(),
              cfg.num_full_layers(), cfg.mtp_num_layers, cfg.hidden_size, cfg.vocab_size);
  if (cfg.moe())
    std::printf("mlp: routed MoE, %d experts top-%d, intermediate %d, shared expert %d\n", cfg.num_experts,
                cfg.num_experts_per_tok, cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size);
  else
    std::printf("mlp: dense SwiGLU, intermediate %d\n", cfg.intermediate_size);

  namespace fs = std::filesystem;
  std::vector<fs::path> shards;
  for (const auto& entry : fs::directory_iterator(checkpoint_dir))
    if (entry.path().extension() == ".safetensors") shards.push_back(entry.path());
  if (shards.empty()) throw std::runtime_error("no .safetensors shards in " + checkpoint_dir);
  std::sort(shards.begin(), shards.end());

  std::unordered_map<std::string, dgpp::QwenTensorDesc> present;
  size_t total_tensors = 0;
  for (const auto& shard : shards) {
    auto f = dgpp::SafetensorsFile::open(shard.string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      ++total_tensors;
      auto [it, inserted] = present.emplace(t.name, dgpp::QwenTensorDesc{t.dtype, t.shape});
      if (!inserted) throw std::runtime_error(std::format("duplicate tensor '{}' in {}", t.name, shard.string()));
    });
  }
  std::printf("checkpoint: %zu shards, %zu tensors in headers\n", shards.size(), total_tensors);

  const dgpp::QwenBindReport rep = dgpp::qwen35_validate_text_binding(cfg, present, max_errors);
  std::printf("binding: expected %zu | matched %zu (missing %zu, dtype %zu, shape %zu) | unexpected %zu | "
              "out of scope %zu | vision skipped %zu\n",
              rep.expected, rep.matched, rep.missing, rep.dtype_mismatch, rep.shape_mismatch, rep.unexpected,
              rep.out_of_scope, rep.vision);
  std::printf("quantized matrices: %zu\n", rep.quantized_matrices);
  for (const auto& err : rep.errors) std::printf("  error: %s\n", err.c_str());
  if (rep.errors.size() >= max_errors) std::printf("  ... error list capped at %zu\n", max_errors);
  if (!rep.ok()) {
    std::printf("BINDING FAILED\n");
    return 1;
  }
  std::printf("binding OK: every tensor of the table is present with its dtype and shape\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "qwen35_bind_check: %s\n", e.what());
    return 2;
  }
}
