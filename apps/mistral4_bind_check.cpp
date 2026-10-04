// mistral4_bind_check: offline validation of a Mistral-Small-4 checkpoint
// (the Mistral-native NVFP4 release: params.json + consolidated-*.safetensors)
// against the family's expected-tensor table, and the weight bytes a rank
// would hold. Reads params.json, the shard index and every safetensors
// header; no payload bytes are touched. Exit 0 only when the binding is
// exact: every expected tensor present with its dtype and shape, nothing
// unexpected.
//
//   mistral4_bind_check --model ORG/NAME | --checkpoint-dir <dir>
//                       [--world N] [--pool-tokens T] [--max-errors N]
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
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "models/mistral4/binding.hpp"
#include "models/mistral4/config.hpp"

namespace {

std::string read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  return text;
}

// The model's shards: the index's file list when the release has one (the
// directory may hold other safetensors files), every *.safetensors otherwise.
std::vector<std::filesystem::path> shard_list(const std::string& dir) {
  namespace fs = std::filesystem;
  std::vector<fs::path> shards;
  const fs::path index = fs::path(dir) / "consolidated.safetensors.index.json";
  if (fs::exists(index)) {
    const std::string text = read_file(index.string());
    const auto parsed = dgpp::minijson::parse(text);
    const dgpp::minijson::Value* map = parsed.root.find("weight_map");
    if (!map || !map->is_object()) throw std::runtime_error(index.string() + ": no weight_map");
    for (const auto& m : map->members()) {
      const fs::path p = fs::path(dir) / std::string(m.value.as_string());
      if (std::find(shards.begin(), shards.end(), p) == shards.end()) shards.push_back(p);
    }
  } else {
    for (const auto& entry : fs::directory_iterator(dir))
      if (entry.path().extension() == ".safetensors") shards.push_back(entry.path());
  }
  if (shards.empty()) throw std::runtime_error("no .safetensors shards in " + dir);
  std::sort(shards.begin(), shards.end());
  return shards;
}

int run(int argc, char** argv) {
  std::string checkpoint_dir, model_id;
  size_t max_errors = 32;
  int world = 1;
  long long pool_tokens = 262144;
  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    auto next = [&]() -> std::string_view {
      if (i + 1 >= argc) throw std::runtime_error(std::format("missing value for {}", a));
      return argv[++i];
    };
    if (a == "--checkpoint-dir")
      checkpoint_dir = next();
    else if (a == "--model")
      model_id = next();
    else if (a == "--world")
      world = std::stoi(std::string(next()));
    else if (a == "--pool-tokens")
      pool_tokens = std::stoll(std::string(next()));
    else if (a == "--max-errors")
      max_errors = std::stoul(std::string(next()));
    else
      throw std::runtime_error(std::format("unknown argument {}", a));
  }
  if (!model_id.empty()) {
    if (!checkpoint_dir.empty())
      throw std::runtime_error("--model and --checkpoint-dir are mutually exclusive");
    std::string err;
    const std::string snapshot = dgpp::hf::model_dir(model_id, &err);
    if (snapshot.empty()) throw std::runtime_error(std::format("--model {}: {}", model_id, err));
    checkpoint_dir = snapshot;
    std::printf("model: %s -> %s\n", model_id.c_str(), snapshot.c_str());
  }
  if (checkpoint_dir.empty())
    throw std::runtime_error(
        "usage: mistral4_bind_check --model ORG/NAME | --checkpoint-dir <dir> [--world N] "
        "[--pool-tokens T] "
        "[--max-errors N]");

  const dgpp::ModelArchitecture arch = dgpp::detect_architecture_file(checkpoint_dir);
  if (arch != dgpp::ModelArchitecture::Mistral4)
    throw std::runtime_error(std::format("{} is a {} checkpoint, not mistral4", checkpoint_dir,
                                         dgpp::model_architecture_name(arch)));
  const dgpp::Mistral4TextConfig cfg = dgpp::Mistral4TextConfig::from_json_file(checkpoint_dir);
  std::printf("config: mistral4, %d layers, hidden %d, vocab %d\n", cfg.num_hidden_layers,
              cfg.hidden_size, cfg.vocab_size);
  std::printf(
      "attention: latent (MLA), %d heads, q latent %d, kv latent %d, nope %d + rope %d, v %d; yarn "
      "factor "
      "%g over %d, llama-4 beta %g over %d\n",
      cfg.num_attention_heads, cfg.q_lora_rank, cfg.kv_lora_rank, cfg.qk_nope_head_dim,
      cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.yarn_factor,
      cfg.yarn_original_max_position_embeddings, cfg.llama4_scaling_beta,
      cfg.llama4_original_max_position_embeddings);
  std::printf(
      "mlp: routed MoE (softmax), %d experts top-%d, intermediate %d, %d shared expert; vision "
      "tower %s\n",
      cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size,
      cfg.n_shared_experts, cfg.vision.present ? "present (never loaded)" : "absent");

  const std::vector<std::filesystem::path> shards = shard_list(checkpoint_dir);
  std::unordered_map<std::string, dgpp::Mistral4TensorDesc> present;
  size_t total_tensors = 0;
  for (const auto& shard : shards) {
    auto f = dgpp::SafetensorsFile::open(shard.string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      ++total_tensors;
      auto [it, inserted] = present.emplace(t.name, dgpp::Mistral4TensorDesc{t.dtype, t.shape});
      if (!inserted)
        throw std::runtime_error(
            std::format("duplicate tensor '{}' in {}", t.name, shard.string()));
    });
  }
  std::printf("checkpoint: %zu shards, %zu tensors in headers\n", shards.size(), total_tensors);

  const dgpp::Mistral4BindReport rep = dgpp::mistral4_validate_binding(cfg, present, max_errors);
  std::printf(
      "binding: expected %zu | matched %zu (missing %zu, dtype %zu, shape %zu) | unexpected %zu\n",
      rep.expected, rep.matched, rep.missing, rep.dtype_mismatch, rep.shape_mismatch,
      rep.unexpected);
  std::printf(
      "nvfp4 matrices: %zu; never loaded (activation scales, vision): %zu tensors; bytes in "
      "headers %zu, "
      "of which the text model %zu\n",
      rep.fp4_matrices, rep.unused, rep.bytes, rep.loaded_bytes);
  for (const auto& err : rep.errors) std::printf("  error: %s\n", err.c_str());
  if (rep.errors.size() >= max_errors) std::printf("  ... error list capped at %zu\n", max_errors);
  if (!rep.ok()) {
    std::printf("BINDING FAILED\n");
    return 1;
  }
  std::printf("binding OK: every tensor of the table is present with its dtype and shape\n");

  // What a rank holds, as stored (no re-encoding at load), and the latent
  // cache: a bf16 latent and a bf16 rope key per token per layer.
  const double gib = 1073741824.0;
  const dgpp::Mistral4WeightBytes b = dgpp::mistral4_weight_bytes(cfg, world);
  std::printf(
      "weights per rank at world %d, as stored: %zu bytes = %.3f GiB (routed experts %.3f, shared "
      "%.3f, "
      "attention %.3f, embedding %.3f, head %.3f, routers and norms %.3f)\n",
      world, b.total(), b.total() / gib, b.routed_experts / gib, b.shared_experts / gib,
      b.attention / gib, b.embedding / gib, b.head / gib, b.routers_and_norms / gib);
  const size_t per_token =
      static_cast<size_t>(cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 2 * cfg.num_hidden_layers;
  std::printf(
      "latent cache (replicated on every rank): %zu bytes per token; a %lld-token pool is %.3f "
      "GiB\n",
      per_token, pool_tokens,
      static_cast<double>(per_token) * static_cast<double>(pool_tokens) / gib);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mistral4_bind_check: %s\n", e.what());
    return 2;
  }
}
