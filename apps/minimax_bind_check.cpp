// minimax_bind_check: offline validation of a MiniMax-M2 checkpoint (the
// modelopt NVFP4 release) against the family's expected-tensor table, and
// the weight and K/V bytes a rank would hold. Reads config.json,
// model.safetensors.index.json and the headers of the shards the index
// names — the repository holds other safetensors files that repeat tensor
// names and are not the model; no payload bytes are touched. Exit 0 only
// when the binding is exact: every required tensor present with its dtype
// and shape, every activation scale well-formed, nothing unexpected.
//
//   minimax_bind_check --model ORG/NAME | --checkpoint-dir <dir>
//                      [--world N] [--pool-tokens T] [--max-errors N]
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
#include "models/minimax/binding.hpp"
#include "models/minimax/config.hpp"

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

int run(int argc, char** argv) {
  std::string checkpoint_dir, model_id;
  size_t max_errors = 32;
  int world = 2;
  long long pool_tokens = 196608;
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
        "usage: minimax_bind_check --model ORG/NAME | --checkpoint-dir <dir> [--world N] "
        "[--pool-tokens T] "
        "[--max-errors N]");

  namespace fs = std::filesystem;
  const dgpp::ModelArchitecture arch = dgpp::detect_architecture_file(checkpoint_dir);
  if (arch != dgpp::ModelArchitecture::MiniMaxM2)
    throw std::runtime_error(std::format("{} is a {} checkpoint, not minimax_m2", checkpoint_dir,
                                         dgpp::model_architecture_name(arch)));
  const dgpp::MinimaxTextConfig cfg = dgpp::MinimaxTextConfig::from_json_file(checkpoint_dir);
  std::printf("config: minimax_m2, %d layers, hidden %d, vocab %d\n", cfg.num_hidden_layers,
              cfg.hidden_size, cfg.vocab_size);
  std::printf(
      "attention: GQA, %d query heads over %d kv heads x %d, per-layer q/k norm, rotary %d (theta "
      "%g)\n",
      cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim,
      cfg.rope_theta);
  std::printf(
      "mlp: routed MoE (sigmoid + bias), %d experts top-%d, intermediate %d, no shared expert\n",
      cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size);
  if (cfg.mtp_declared)
    std::printf(
        "note: config.json declares an MTP head (%d modules); no published checkpoint carries one "
        "and the "
        "table expects none\n",
        cfg.mtp_modules_declared);

  const std::string index_path =
      (fs::path(checkpoint_dir) / "model.safetensors.index.json").string();
  const std::string index_text = read_file(index_path);
  const auto index = dgpp::minijson::parse(index_text);
  const std::vector<std::string> shards = dgpp::minimax_shard_files(index.root);
  size_t strays = 0;
  for (const auto& entry : fs::directory_iterator(checkpoint_dir)) {
    if (entry.path().extension() != ".safetensors") continue;
    bool listed = false;
    for (const std::string& s : shards) listed = listed || s == entry.path().filename().string();
    strays += !listed;
  }
  std::unordered_map<std::string, dgpp::MinimaxTensorDesc> present;
  size_t total_tensors = 0;
  for (const std::string& shard : shards) {
    auto f = dgpp::SafetensorsFile::open((fs::path(checkpoint_dir) / shard).string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      ++total_tensors;
      auto [it, inserted] = present.emplace(t.name, dgpp::MinimaxTensorDesc{t.dtype, t.shape});
      if (!inserted)
        throw std::runtime_error(std::format("duplicate tensor '{}' in {}", t.name, shard));
    });
  }
  std::printf(
      "checkpoint: %zu shards named by the index, %zu tensors in their headers; %zu other "
      "safetensors "
      "files in the directory ignored\n",
      shards.size(), total_tensors, strays);

  const dgpp::MinimaxBindReport rep = dgpp::minimax_validate_binding(cfg, present, max_errors);
  std::printf(
      "binding: required %zu | matched %zu (missing %zu, dtype %zu, shape %zu) | unexpected %zu\n",
      rep.expected, rep.matched, rep.missing, rep.dtype_mismatch, rep.shape_mismatch,
      rep.unexpected);
  std::printf(
      "nvfp4 matrices: %zu; activation scales (never loaded): %zu present, %zu absent; bound %zu "
      "of %zu "
      "tensors; bytes in headers %zu, of which the model %zu\n",
      rep.fp4_matrices, rep.optional_present, rep.optional_absent, rep.bound(), total_tensors,
      rep.bytes, rep.loaded_bytes);
  for (const auto& err : rep.errors) std::printf("  error: %s\n", err.c_str());
  if (rep.errors.size() >= max_errors) std::printf("  ... error list capped at %zu\n", max_errors);
  if (!rep.ok()) {
    std::printf("BINDING FAILED\n");
    return 1;
  }
  std::printf("binding OK: every required tensor is present with its dtype and shape\n");

  const double gib = 1073741824.0;
  for (const bool replicate_qk : {true, false}) {
    if (world == 1 && !replicate_qk) continue;
    const dgpp::MinimaxWeightBytes b = dgpp::minimax_weight_bytes(cfg, world, replicate_qk);
    std::printf(
        "weights per rank at world %d, as stored (q_proj / k_proj %s): %zu bytes = %.3f GiB "
        "(routed "
        "experts %.3f, attention %.3f, embedding %.3f, head %.3f, routers and norms %.3f)\n",
        world, replicate_qk ? "replicated" : "head-sharded", b.total(), b.total() / gib,
        b.routed_experts / gib, b.attention / gib, b.embedding / gib, b.head / gib,
        b.routers_and_norms / gib);
  }
  const size_t per_token = dgpp::minimax_kv_bytes_per_token(cfg, world);
  std::printf("K/V cache per rank: %zu bytes per token; a %lld-token pool is %.3f GiB\n", per_token,
              pool_tokens, static_cast<double>(per_token) * static_cast<double>(pool_tokens) / gib);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "minimax_bind_check: %s\n", e.what());
    return 2;
  }
}
