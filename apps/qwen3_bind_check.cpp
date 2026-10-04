// qwen3_bind_check: offline validation of a plain-Qwen3 checkpoint against
// the family's expected-tensor table — Qwen3-VL-30B-A3B's compressed-tensors
// NVFP4 set (the vision tower's tensors included), Qwen3-235B-A22B's modelopt
// NVFP4 set, or the BF16 dense retrieval models. Reads config.json and every
// safetensors header; host only, no GPU. Exit 0 only when the binding is
// exact: every expected tensor present with its dtype and shape, nothing
// unexpected, and the table's count and bytes equal the headers'.
//
//   qwen3_bind_check --model ORG/NAME | --checkpoint-dir <dir>
//                    [--layers N] [--max-errors N] [--world W]
//                    [--host-forward ids.json [--engine-rounding]]
//
// --world W prints what each rank of a W-node world holds resident (the
// sharding plan's arithmetic; nothing is loaded).
// --host-forward runs the family's host reference (models/qwen3/reference.hpp,
// double precision, one thread) over the token ids and prints the
// teacher-forced log-probabilities in the forward check's format — the same
// lines sixlabs/bench/compare_forward_check.py reads, so the C++ reading of
// the checkpoint can be held to tools/qwen3_reference.py without a GPU. Use
// --layers to keep it short on a real checkpoint (each layer's attention is
// a plain triple loop).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "loaders/architecture.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen3/binding.hpp"
#include "models/qwen3/config.hpp"
#include "models/qwen3/reference.hpp"

namespace {

std::vector<int64_t> read_ids(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();
  const auto parsed = dgpp::minijson::parse(text);
  const dgpp::minijson::Value* list = &parsed.root;
  if (list->is_object()) list = list->find("token_ids") ? list->find("token_ids") : list->find("ids");
  if (list == nullptr || !list->is_array() || list->items().empty())
    throw std::runtime_error(path + ": expected a non-empty JSON list of token ids");
  std::vector<int64_t> ids;
  for (const auto& v : list->items()) ids.push_back(v.as_int(-1));
  return ids;
}

int run(int argc, char** argv) {
  std::string checkpoint_dir, model_id, ids_path;
  size_t max_errors = 32;
  int layers = -1, world = 0;
  bool engine_rounding = false;
  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    auto next = [&]() -> std::string_view {
      if (i + 1 >= argc) throw std::runtime_error(std::format("missing value for {}", a));
      return argv[++i];
    };
    if (a == "--checkpoint-dir") checkpoint_dir = next();
    else if (a == "--model") model_id = next();
    else if (a == "--max-errors") max_errors = std::stoul(std::string(next()));
    else if (a == "--layers") layers = std::stoi(std::string(next()));
    else if (a == "--world") world = std::stoi(std::string(next()));
    else if (a == "--host-forward") ids_path = next();
    else if (a == "--engine-rounding") engine_rounding = true;
    else throw std::runtime_error(std::format("unknown argument {}", a));
  }
  if (!model_id.empty()) {
    if (!checkpoint_dir.empty()) throw std::runtime_error("--model and --checkpoint-dir are mutually exclusive");
    std::string err;
    const std::string snapshot = dgpp::hf::model_dir(model_id, &err);
    if (snapshot.empty()) throw std::runtime_error(std::format("--model {}: {}", model_id, err));
    checkpoint_dir = snapshot;
    std::printf("model: %s -> %s\n", model_id.c_str(), snapshot.c_str());
  }
  if (checkpoint_dir.empty())
    throw std::runtime_error(
        "usage: qwen3_bind_check --model ORG/NAME | --checkpoint-dir <dir> [--layers N] [--max-errors N] "
        "[--world W] [--host-forward ids.json [--engine-rounding]]");
  const std::string config_path = checkpoint_dir + "/config.json";

  const dgpp::ModelArchitecture arch = dgpp::detect_architecture_file(config_path);
  if (arch != dgpp::ModelArchitecture::Qwen3 && arch != dgpp::ModelArchitecture::Qwen3Moe &&
      arch != dgpp::ModelArchitecture::Qwen3VlMoe)
    throw std::runtime_error(std::format("{} is a {} checkpoint, not qwen3, qwen3_moe or qwen3_vl_moe", config_path,
                                         dgpp::model_architecture_name(arch)));

  // The host checkpoint parses the config, maps the shards, takes the naming
  // from the header and validates the whole header against the table; a
  // mismatch is reported here in full rather than thrown.
  dgpp::Qwen3TextConfig cfg = dgpp::Qwen3TextConfig::from_json_file(config_path);
  dgpp::Qwen3BindReport rep;
  size_t header_tensors = 0, header_bytes = 0;
  {
    namespace fs = std::filesystem;
    std::vector<std::string> shards;
    for (const auto& entry : fs::directory_iterator(checkpoint_dir))
      if (entry.is_regular_file() && entry.path().extension() == ".safetensors") shards.push_back(entry.path().string());
    if (shards.empty()) throw std::runtime_error("no .safetensors shards in " + checkpoint_dir);
    std::sort(shards.begin(), shards.end());
    dgpp::Qwen3PresentMap present;
    for (const std::string& shard : shards) {
      const auto f = dgpp::SafetensorsFile::open(shard);
      f->for_each([&](const dgpp::TensorInfo& t) {
        ++header_tensors;
        header_bytes += t.nbytes();
        if (!present.emplace(t.name, dgpp::Qwen3TensorDesc{t.dtype, t.shape}).second)
          throw std::runtime_error(std::format("duplicate tensor '{}' in {}", t.name, shard));
      });
    }
    dgpp::qwen3_apply_header_naming(cfg, present);
    if (layers > 0 && layers < cfg.num_hidden_layers) cfg.num_hidden_layers = layers;
    std::printf("config: %s, %d layers%s, hidden %d, %d query / %d kv heads x %d, rope theta %.0f, vocab %d, "
                "max positions %d\n",
                cfg.family_name(), cfg.num_hidden_layers, layers > 0 ? " (truncated by --layers)" : "", cfg.hidden_size,
                cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rope_theta, cfg.vocab_size,
                cfg.max_position_embeddings);
    if (cfg.moe())
      std::printf("mlp: routed MoE, %d experts top-%d, intermediate %d, no shared expert\n", cfg.num_experts,
                  cfg.num_experts_per_tok, cfg.moe_intermediate_size);
    else
      std::printf("mlp: dense SwiGLU, intermediate %d%s\n", cfg.intermediate_size,
                  cfg.tie_word_embeddings ? ", tied embeddings" : "");
    std::printf("weights: %s%s%s\n",
                cfg.quant_kind == dgpp::Qwen3QuantKind::Bf16 ? "BF16"
                : cfg.fp4_packed() ? "NVFP4, compressed-tensors nvfp4-pack-quantized (q/k/v/o and the experts)"
                                   : "NVFP4, modelopt (o_proj and the experts; q/k/v BF16)",
                cfg.kv_cache_scales ? ", FP8 K/V-cache scales bound (unread)" : "",
                cfg.bare_names ? ", base-model names (no model. prefix)" : "");
    if (cfg.vision.has_value())
      std::printf("vision: a %d-block tower (hidden %d) is BOUND and NOT SERVED — text path only, image inputs are refused\n",
                  cfg.vision->depth, cfg.vision->hidden_size);
    std::printf("checkpoint: %zu shards, %zu tensors in headers, %zu tensor bytes\n", shards.size(), header_tensors,
                header_bytes);
    rep = dgpp::qwen3_validate_binding(cfg, present, max_errors);
  }
  const size_t table_text = dgpp::qwen3_expected_text_tensors(cfg).size();
  std::printf("table: %zu tensors (%zu text + %zu vision)\n", rep.expected, table_text, rep.expected - table_text);
  std::printf("binding: expected %zu | matched %zu (missing %zu, dtype %zu, shape %zu) | unexpected %zu | "
              "out of scope %zu | vision bound %zu | tied head copies %zu | matched bytes %zu\n",
              rep.expected, rep.matched, rep.missing, rep.dtype_mismatch, rep.shape_mismatch, rep.unexpected,
              rep.out_of_scope, rep.vision, rep.tied_head_copies, rep.bytes);
  std::printf("nvfp4 matrices: %zu\n", rep.fp4_matrices);
  for (const auto& err : rep.errors) std::printf("  error: %s\n", err.c_str());
  if (rep.errors.size() >= max_errors) std::printf("  ... error list capped at %zu\n", max_errors);
  if (!rep.ok()) {
    std::printf("BINDING FAILED\n");
    return 1;
  }
  // The whole-checkpoint identity (not under --layers: the truncated table
  // is smaller by construction).
  if (layers <= 0) {
    const bool counts = rep.expected + rep.tied_head_copies == header_tensors;
    const bool bytes = rep.bytes == header_bytes || rep.tied_head_copies != 0;
    if (rep.tied_head_copies == 0) {
      std::printf("table count %zu %s header count %zu; table bytes %zu %s header bytes %zu\n", rep.expected,
                  counts ? "==" : "!=", header_tensors, rep.bytes, bytes ? "==" : "!=", header_bytes);
    } else {
      // A tied head stored a second time is in the headers and not in the table: say so rather
      // than print "==" between two numbers that differ by it.
      std::printf("table count %zu + %zu tied head cop%s %s header count %zu; table bytes %zu, header bytes %zu "
                  "(the difference is the tied cop%s, not compared)\n",
                  rep.expected, rep.tied_head_copies, rep.tied_head_copies == 1 ? "y" : "ies", counts ? "==" : "!=",
                  header_tensors, rep.bytes, header_bytes, rep.tied_head_copies == 1 ? "y" : "ies");
    }
    if (!counts || !bytes) {
      std::printf("BINDING FAILED\n");
      return 1;
    }
  }
  std::printf("binding OK: every tensor of the table is present with its dtype and shape\n");

  if (world > 0) {
    try {
      for (int rank = 0; rank < world; ++rank) {
        const dgpp::Qwen3RankBytes b = dgpp::qwen3_rank_resident_bytes(cfg, rank, world);
        std::printf("resident, world %d rank %d: experts %zu, attention %zu, dense mlp %zu, router %zu, norms %zu, "
                    "embedding %zu, head %zu | total %zu bytes (%.2f GiB)\n",
                    world, rank, b.experts, b.attention, b.dense_mlp, b.router, b.norms, b.embed, b.head, b.total(),
                    static_cast<double>(b.total()) / (1024.0 * 1024.0 * 1024.0));
      }
    } catch (const std::invalid_argument& e) {
      // The geometry does not shard at this world: said, not fatal to the bind check.
      std::printf("resident, world %d: refused — %s\n", world, e.what());
    }
  }

  if (!ids_path.empty()) {
    const std::vector<int64_t> ids = read_ids(ids_path);
    const dgpp::qwen3_ref::HostCheckpoint host(checkpoint_dir, layers);
    const auto arithmetic = engine_rounding ? dgpp::qwen3_ref::Arithmetic::Engine : dgpp::qwen3_ref::Arithmetic::Exact;
    std::printf("host forward: %zu tokens, %d layers, %s arithmetic\n", ids.size(), host.config().num_hidden_layers,
                engine_rounding ? "engine-rounding" : "exact (double)");
    const std::vector<double> logits = dgpp::qwen3_ref::forward(host, ids, arithmetic);
    const int V = host.config().vocab_size;
    const std::vector<double> lp = dgpp::qwen3_ref::teacher_forced_logprobs(logits, ids, V);
    std::string lp_line, am_line;
    double mean = 0;
    for (size_t i = 0; i < lp.size(); ++i) {
      lp_line += std::format("{}{:.6f}", i ? "," : "", lp[i]);
      mean += lp[i] / static_cast<double>(lp.size());
    }
    for (size_t t = 0; t < ids.size(); ++t) {
      const double* row = logits.data() + t * static_cast<size_t>(V);
      am_line += std::format("{}{}", t ? "," : "", static_cast<long>(std::max_element(row, row + V) - row));
    }
    std::printf("token logprobs:%s\n", lp_line.c_str());
    std::printf("argmax ids:%s\n", am_line.c_str());
    std::printf("mean logprob %.5f over %zu tokens\n", mean, lp.size());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "qwen3_bind_check: %s\n", e.what());
    return 2;
  }
}
