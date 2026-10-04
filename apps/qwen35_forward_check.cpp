// qwen35_forward_check: the qwen3_5 stack's world-1 diagnostic forward over
// a real checkpoint (Qwen3.8-27B or, the Qwen3Next dialect, Qwen3-Next-80B and
// Qwen3-Coder-Next) —
// one prompt of token ids through the loader, the per-layer residual
// magnitudes, every position's top-k next-token logits, and the
// teacher-forced log-probability of each given token (the number an
// `echo` + `logprobs` completion reports, and what the host reference
// tools/qwen3next_reference.py `score` prints). No tokenizer: ids come in on
// the command line or from a JSON list.
//
//   qwen35_forward_check --model ORG/NAME | --checkpoint-dir DIR
//                        --ids 1,2,3,... | --ids-json FILE
//                        [--topk K] [--layers N] [--resident] [--image-dir DIR|off]
//                        [--dump-states FILE] [--plan SLOTS,CONTEXT]
//                        [--dense-weights checkpoint|fp8] [--rope-scaling FACTOR]
//
// --layers N runs the first N layers (the head then reads a residual the
//   checkpoint never trained it on: for layer-by-layer comparison only).
// --dump-states writes the captured residual after each layer, then the
//   final normed hidden, as raw bf16 [T, hidden] blocks in that order.
// --plan prints the serving memory plan for SLOTS requests over a CONTEXT
//   token pool (resident, MTP on) and exits without loading anything.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "loaders/architecture.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/model35.hpp"

namespace {

std::vector<int64_t> read_ids_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();  // the parsed values view the text
  const auto parsed = dgpp::minijson::parse(text);
  if (!parsed.root.is_array()) throw std::runtime_error(path + ": not a JSON array of token ids");
  std::vector<int64_t> ids;
  for (const auto& item : parsed.root.items()) {
    if (!item.is_number()) throw std::runtime_error(path + ": non-numeric token id");
    ids.push_back(item.as_int());
  }
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_id, ckpt, ids_text, ids_json, dump_states, image_dir, plan_text, dense_weights;
  double rope_factor = 0.0;  // --rope-scaling: the YaRN factor (0: the plain rope)
  int topk = 5, layers = -1;
  bool resident = false;
  auto next = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error("missing value after " + std::string(argv[i]));
    return argv[++i];
  };
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--model") model_id = next(i);
      else if (a == "--checkpoint-dir") ckpt = next(i);
      else if (a == "--ids") ids_text = next(i);
      else if (a == "--ids-json") ids_json = next(i);
      else if (a == "--topk") topk = std::stoi(next(i));
      else if (a == "--layers") layers = std::stoi(next(i));
      else if (a == "--dump-states") dump_states = next(i);
      else if (a == "--resident") resident = true;
      else if (a == "--image-dir") image_dir = next(i);
      else if (a == "--plan") plan_text = next(i);
      else if (a == "--dense-weights") dense_weights = next(i);
      else if (a == "--rope-scaling") rope_factor = std::stod(next(i));
      else throw std::runtime_error("unknown argument " + a);
    }
    if (ckpt.empty()) {
      if (model_id.empty()) throw std::runtime_error("--model or --checkpoint-dir is required");
      std::string err;
      ckpt = dgpp::hf::model_dir(model_id, &err);
      if (ckpt.empty()) throw std::runtime_error("cannot resolve " + model_id + ": " + err);
    }
    const std::string cfg_path = (std::filesystem::path(ckpt) / "config.json").string();
    const dgpp::ModelArchitecture arch = dgpp::detect_architecture_file(cfg_path);
    if (arch != dgpp::ModelArchitecture::Qwen3_5 && arch != dgpp::ModelArchitecture::Qwen3Next)
      throw std::runtime_error("not a qwen3_5 or qwen3_next checkpoint: " + ckpt);
    dgpp::Qwen35TextConfig cfg = dgpp::Qwen35TextConfig::from_json_file(cfg_path);
    // --rope-scaling FACTOR: the YaRN ramp as the server's engine.rope_scaling
    // sets it on this dialect (plain 1-D rope: the band from the original
    // positions), to check the ramp against tools/qwen3next_reference.py --yarn-factor.
    if (rope_factor > 0.0) {
      if (!cfg.next()) throw std::runtime_error("--rope-scaling is for the Qwen3-Next dialect");
      dgpp::RopeScaling rs;
      rs.factor = rope_factor;
      rs.original_max_position_embeddings = cfg.max_position_embeddings;
      rs.mrope_cache_factor = 1.0;
      rs.validate("--rope-scaling");
      cfg.rope_scaling = rs;
      std::printf("rope scaling: yarn x%g over %d positions, mscale %.10g, context limit %lld\n", rs.factor,
                  cfg.max_position_embeddings, static_cast<double>(rs.mscale()),
                  static_cast<long long>(cfg.context_limit()));
    }
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    // engine.dense_weights: the block-FP8 lm head, and for a checkpoint that
    // ships dense projections outside block FP8, those encoded at load.
    if (!dense_weights.empty() && dense_weights != "fp8" && dense_weights != "checkpoint")
      throw std::runtime_error("--dense-weights takes checkpoint or fp8");
    if (dense_weights == "fp8") {
      dgpp::Qwen35Model::set_dense_weights_fp8(true);
      if (cfg.next() || cfg.quant_kind != dgpp::Qwen35QuantKind::Fp8Block)
        dgpp::Qwen35LayerStream::set_dense_weights_fp8(true);
    }

    if (!plan_text.empty()) {
      const size_t comma = plan_text.find(',');
      if (comma == std::string::npos) throw std::runtime_error("--plan takes SLOTS,CONTEXT");
      const int slots = std::stoi(plan_text.substr(0, comma));
      const int64_t context = std::stoll(plan_text.substr(comma + 1));
      const bool mtp = cfg.mtp_layer() >= 0;
      const dgpp::MemoryPlan plan = dgpp::Qwen35Model::plan_memory(
          cfg, dgpp::Qwen35Model::prefill_chunk_tokens(), context, 0, 1, dgpp::LoaderResidency::Resident,
          slots, mtp, slots);
      std::printf("memory plan: %s, %d slots, %lld-token pool, resident, mtp %s\n",
                  dgpp::model_architecture_name(arch), slots, static_cast<long long>(context),
                  mtp ? "on" : "off");
      for (const auto& item : plan.items)
        std::printf("  %9.3f GiB device %8.3f GiB pinned  %s\n", item.device / kGiB, item.pinned / kGiB,
                    item.name.c_str());
      std::printf("  %9.3f GiB device %8.3f GiB pinned  TOTAL (+ %.3f GiB per prefix snapshot)\n",
                  plan.device_bytes() / kGiB, plan.pinned_bytes() / kGiB,
                  dgpp::Qwen35Model::session_snapshot_bytes(cfg, 1, mtp) / kGiB);
      return 0;
    }

    std::vector<int64_t> ids;
    if (!ids_json.empty()) {
      ids = read_ids_json(ids_json);
    } else {
      std::stringstream ss(ids_text);
      std::string item;
      while (std::getline(ss, item, ',')) if (!item.empty()) ids.push_back(std::stoll(item));
    }
    if (ids.empty()) throw std::runtime_error("--ids or --ids-json is required");
    for (int64_t id : ids)
      if (id < 0 || id >= cfg.vocab_size) throw std::runtime_error("token id outside the vocabulary");
    const bool truncated = layers > 0 && layers < cfg.num_hidden_layers;
    if (truncated) {
      cfg.num_hidden_layers = layers;
      cfg.layers.resize(static_cast<size_t>(layers));
    }
    const int T = static_cast<int>(ids.size());
    const int H = cfg.hidden_size;
    if (!image_dir.empty()) dgpp::Qwen35LayerStream::set_resident_image_dir(image_dir == "off" ? "" : image_dir);
    const dgpp::LoaderResidency residency =
        resident ? dgpp::LoaderResidency::Resident : dgpp::LoaderResidency::Streaming;
    DGPP_LOG_INFO("qwen35_forward_check: {} ({}) — {} tokens, {} layers{}, {}", ckpt,
                  dgpp::model_architecture_name(arch), T, cfg.num_hidden_layers,
                  truncated ? " (truncated)" : "", resident ? "resident" : "streaming");

    const auto t0 = std::chrono::steady_clock::now();
    dgpp::Qwen35Model model(cfg, ckpt, T, T + 64, residency, /*boundary=*/nullptr, /*rank=*/0, /*world=*/1,
                            /*max_requests=*/1, /*decode_rows=*/0);
    const auto t1 = std::chrono::steady_clock::now();
    const dgpp::Qwen35Model::Outputs out = model.forward(ids, true);
    const auto t2 = std::chrono::steady_clock::now();

    auto fnv = [](const std::vector<uint16_t>& v) {
      uint64_t h = 1469598103934665603ull;
      for (uint16_t x : v) { h ^= x; h *= 1099511628211ull; }
      return h;
    };
    for (size_t l = 0; l < out.layer_states.size(); ++l) {
      double rms = 0, mx = 0;
      for (uint16_t v : out.layer_states[l]) {
        const float f = dgpp::bf16_bits_to_float(v);
        rms += static_cast<double>(f) * f;
        mx = std::max(mx, static_cast<double>(std::fabs(f)));
      }
      rms = std::sqrt(rms / static_cast<double>(out.layer_states[l].size()));
      std::printf("layer %2zu: residual rms %.4g max %.4g digest %016llx\n", l, rms, mx,
                  static_cast<unsigned long long>(fnv(out.layer_states[l])));
    }
    std::printf("final hidden digest %016llx\n", static_cast<unsigned long long>(fnv(out.final_hidden_bits)));

    const int64_t V = out.lm_vocab_count;
    const auto top = dgpp::Qwen35Model::topk(out.logits, T, V, topk);
    for (int t = 0; t < T; ++t) {
      std::printf("pos %3d id %7lld ->", t, static_cast<long long>(ids[static_cast<size_t>(t)]));
      for (const auto& [id, v] : top[static_cast<size_t>(t)]) std::printf(" %d(%.3f)", id + out.lm_vocab_begin, v);
      std::printf("\n");
    }
    std::printf("argmax ids:");
    for (int t = 0; t < T; ++t) std::printf("%s%d", t ? "," : " ", top[static_cast<size_t>(t)][0].first + out.lm_vocab_begin);
    std::printf("\n");
    // Teacher forcing: log P(ids[t + 1] | ids[0..t]) from row t's logits
    // (the log-softmax in double over the full vocabulary).
    std::printf("token logprobs:");
    double sum = 0;
    for (int t = 0; t + 1 < T; ++t) {
      const float* row = out.logits.data() + static_cast<size_t>(t) * static_cast<size_t>(V);
      double mx = row[0];
      for (int64_t v = 1; v < V; ++v) mx = std::max(mx, static_cast<double>(row[v]));
      double z = 0;
      for (int64_t v = 0; v < V; ++v) z += std::exp(static_cast<double>(row[v]) - mx);
      const double lp = static_cast<double>(row[ids[static_cast<size_t>(t) + 1]]) - mx - std::log(z);
      sum += lp;
      std::printf("%s%.5f", t ? "," : " ", lp);
    }
    std::printf("\n");
    if (T > 1) std::printf("mean token logprob %.5f over %d tokens\n", sum / (T - 1), T - 1);

    if (!dump_states.empty()) {
      std::ofstream f(dump_states, std::ios::binary);
      for (const auto& st : out.layer_states)
        f.write(reinterpret_cast<const char*>(st.data()), static_cast<std::streamsize>(st.size() * 2));
      f.write(reinterpret_cast<const char*>(out.final_hidden_bits.data()),
              static_cast<std::streamsize>(out.final_hidden_bits.size() * 2));
      std::printf("wrote %s: %zu layers x [%d, %d] + final [%d, %d] bf16\n", dump_states.c_str(),
                  out.layer_states.size(), T, H, T, H);
    }
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::printf("boot %.0f ms, forward %.0f ms\n", ms(t0, t1), ms(t1, t2));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "qwen35_forward_check: %s\n", e.what());
    return 1;
  }
}
