// gemma4_bind_check: offline validation of a Gemma 4 checkpoint against the
// family's expected-tensor table (models/gemma4/binding.hpp) —
// nvidia/Gemma-4-31B-IT-NVFP4 or bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16
// as shipped. Reads config.json and every
// safetensors header; no payload bytes are touched (host-only: no CUDA).
// Exit 0 only when the binding is exact: every expected tensor present with
// its dtype and shape, nothing unexpected, and the table plus the ignored
// encoder tensors equal to the header count.
//
//   gemma4_bind_check --model ORG/NAME | (--config <dir>/config.json
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
#include "models/gemma4/binding.hpp"
#include "models/gemma4/config.hpp"

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
        "usage: gemma4_bind_check --model ORG/NAME | (--config <config.json> --checkpoint-dir <dir>) [--max-errors N]");
  if (config_path.empty()) config_path = checkpoint_dir + "/config.json";

  const dgpp::ModelArchitecture arch = dgpp::detect_architecture_file(config_path);
  if (arch != dgpp::ModelArchitecture::Gemma4)
    throw std::runtime_error(std::format("{} is a {} checkpoint, not gemma4", config_path,
                                         dgpp::model_architecture_name(arch)));
  const dgpp::Gemma4TextConfig cfg = dgpp::Gemma4TextConfig::from_json_file(config_path);
  std::printf("config: gemma4, %d layers (%d sliding + %d full), hidden %d, vocab %d, recipe %s\n",
              cfg.num_hidden_layers, cfg.num_sliding_layers(), cfg.num_full_layers(), cfg.hidden_size, cfg.vocab_size,
              dgpp::gemma4_recipe_name(cfg.recipe));
  std::printf("attention: %d heads; sliding %d KV heads x %d, window %d, %d rotary pairs (theta %g); "
              "full %d KV heads x %d, %d rotary pairs (theta %g), value from k_proj\n",
              cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.sliding_window,
              cfg.sliding_rotary_pairs, cfg.sliding_rope_theta, cfg.num_global_key_value_heads, cfg.global_head_dim,
              cfg.global_rotary_pairs, cfg.global_rope_theta);
  std::printf("mlp: dense GeGLU, intermediate %d; embedding scale %g; logit soft-cap %g\n", cfg.intermediate_size,
              static_cast<double>(cfg.embed_scale()), static_cast<double>(cfg.final_logit_softcapping));
  if (cfg.enable_moe_block)
    std::printf("moe: %d experts top-%d beside the dense MLP, intermediate %d, router input scale %.12g\n",
                cfg.num_experts, cfg.top_k_experts, cfg.moe_intermediate_size,
                static_cast<double>(cfg.router_input_scale()));

  namespace fs = std::filesystem;
  std::vector<fs::path> shards;
  for (const auto& entry : fs::directory_iterator(checkpoint_dir))
    if (entry.path().extension() == ".safetensors") shards.push_back(entry.path());
  if (shards.empty()) throw std::runtime_error("no .safetensors shards in " + checkpoint_dir);
  std::sort(shards.begin(), shards.end());

  std::unordered_map<std::string, dgpp::Gemma4TensorDesc> present;
  size_t total_tensors = 0;
  for (const auto& shard : shards) {
    auto f = dgpp::SafetensorsFile::open(shard.string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      ++total_tensors;
      auto [it, inserted] = present.emplace(t.name, dgpp::Gemma4TensorDesc{t.dtype, t.shape});
      if (!inserted) throw std::runtime_error(std::format("duplicate tensor '{}' in {}", t.name, shard.string()));
    });
  }
  std::printf("checkpoint: %zu shards, %zu tensors in headers\n", shards.size(), total_tensors);

  const dgpp::Gemma4BindReport rep = dgpp::gemma4_validate_text_binding(cfg, present, max_errors);
  std::printf("binding: expected %zu | matched %zu (missing %zu, dtype %zu, shape %zu) | unexpected %zu | "
              "ignored (encoders) %zu\n",
              rep.expected, rep.matched, rep.missing, rep.dtype_mismatch, rep.shape_mismatch, rep.unexpected,
              rep.ignored);
  std::printf("matrices: %zu NVFP4 (%zu of them experts), %zu BF16; matched bytes %zu\n", rep.fp4_matrices,
              rep.expert_matrices, rep.bf16_matrices, rep.bytes);
  std::printf("table %zu + ignored %zu = %zu; headers %zu\n", rep.expected, rep.ignored, rep.expected + rep.ignored,
              total_tensors);
  for (const auto& err : rep.errors) std::printf("  error: %s\n", err.c_str());
  if (rep.errors.size() >= max_errors) std::printf("  ... error list capped at %zu\n", max_errors);
  if (!rep.ok() || rep.expected + rep.ignored != total_tensors) {
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
    std::fprintf(stderr, "gemma4_bind_check: %s\n", e.what());
    return 2;
  }
}
