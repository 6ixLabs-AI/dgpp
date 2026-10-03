// dsv4_bind_check: offline validation of a DeepSeek-V4-Flash checkpoint
// (deepseek-ai/DeepSeek-V4-Flash-0731) against the family's expected-tensor
// table. Reads config.json and every
// safetensors header; no payload bytes are touched. Exit 0 only when the
// binding is exact: every expected tensor present with its dtype and shape,
// nothing unexpected.
//
//   dsv4_bind_check --model ORG/NAME | (--config <dir>/config.json
//                                       --checkpoint-dir <dir>) [--max-errors N]
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "loaders/hf_cache.hpp"
#include "loaders/safetensors.hpp"
#include "models/dsv4/binding.hpp"
#include "models/dsv4/config.hpp"

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
        "usage: dsv4_bind_check --model ORG/NAME | (--config <config.json> --checkpoint-dir <dir>) [--max-errors N]");
  if (config_path.empty()) config_path = checkpoint_dir + "/config.json";

  const dgpp::Dsv4Config cfg = dgpp::Dsv4Config::from_json_file(config_path);
  std::printf("config: %d layers + %d draft stages, %d compressed caches (%d indexed), %d hash layers, experts %d "
              "top-%d, vocab %d\n",
              cfg.num_hidden_layers, cfg.draft_stages, cfg.num_caches(), cfg.num_index_caches(), cfg.num_hash_layers,
              cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.vocab_size);

  namespace fs = std::filesystem;
  std::vector<fs::path> shards;
  for (const auto& entry : fs::directory_iterator(checkpoint_dir))
    if (entry.path().extension() == ".safetensors") shards.push_back(entry.path());
  if (shards.empty()) throw std::runtime_error("no .safetensors shards in " + checkpoint_dir);
  std::sort(shards.begin(), shards.end());

  std::unordered_map<std::string, dgpp::Dsv4TensorDesc> present;
  size_t total_tensors = 0;
  for (const auto& shard : shards) {
    auto f = dgpp::SafetensorsFile::open(shard.string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      ++total_tensors;
      auto [it, inserted] = present.emplace(t.name, dgpp::Dsv4TensorDesc{t.dtype, t.shape});
      if (!inserted) throw std::runtime_error(std::format("duplicate tensor '{}' in {}", t.name, shard.string()));
    });
  }
  std::printf("checkpoint: %zu shards, %zu tensors in headers\n", shards.size(), total_tensors);

  const dgpp::Dsv4BindReport rep = dgpp::dsv4_validate_binding(cfg, present, max_errors);
  std::printf("binding: expected %zu | matched %zu (missing %zu, dtype %zu, shape %zu) | unexpected %zu | beyond stack %d\n",
              rep.expected, rep.matched, rep.missing, rep.dtype_mismatch, rep.shape_mismatch, rep.unexpected,
              rep.beyond_stack);
  std::printf("matrices: MXFP4 %zu | fp8 %zu\n", rep.fp4_matrices, rep.fp8_matrices);
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
    std::fprintf(stderr, "dsv4_bind_check: %s\n", e.what());
    return 2;
  }
}
