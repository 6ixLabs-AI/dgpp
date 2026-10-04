// UNVERIFIED DRAFT (2026-10-04): never compiled with nvcc / GCC, never run.
// Built only under -DDGPP_BUILD_QWEN3_PLAIN_DRAFT=ON.
//
// qwen3_forward_check: the plain Qwen3-MoE stack's world-1 diagnostic forward
// over a real checkpoint (Qwen3-VL-30B-A3B's text path, or Qwen3-235B-A22B
// truncated to what one box holds) — one prompt of token ids through the
// loader, the per-layer residual magnitudes, every position's top-k
// next-token logits, and the teacher-forced log-probability of each given
// token (what tools/qwen3_reference.py `score` prints, and the two lines
// sixlabs/bench/compare_forward_check.py reads). No tokenizer: ids come in
// on the command line or from a JSON list.
//
//   qwen3_forward_check --model ORG/NAME | --checkpoint-dir DIR
//                       --ids 1,2,3,... | --ids-json FILE
//                       [--topk K] [--layers N] [--resident] [--image-dir DIR|off]
//                       [--dump-states FILE] [--plan SLOTS,CONTEXT[,WORLD]]
//                       [--host-compare]
//
// --layers N runs the first N layers (the head then reads a residual the
//   checkpoint never trained it on: for layer-by-layer comparison only, and
//   the only way a 235B forward fits one box).
// --dump-states writes the captured residual after each layer, then the
//   final normed hidden, as raw bf16 [T, hidden] blocks in that order.
// --plan prints the serving memory plan for SLOTS requests over a CONTEXT
//   token pool (resident; rank 0 of WORLD, default 1) and exits without
//   loading anything.
// --host-compare also runs the family's host reference (models/qwen3/
//   reference.hpp) with the engine's rounding points over the same ids and
//   prints, per layer, the relative l2 distance between the GPU residual and
//   the host one — the first layer where they part is where a defect is.
//   Slow on a real checkpoint (plain loops, one thread): use with --layers.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "loaders/architecture.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen3/config.hpp"
#include "models/qwen3/model.hpp"
#include "models/qwen3/reference.hpp"

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
  std::string model_id, ckpt, ids_text, ids_json, dump_states, image_dir, plan_text;
  int topk = 5, layers = -1;
  bool resident = false, host_compare = false;
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
      else if (a == "--host-compare") host_compare = true;
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
    if (arch != dgpp::ModelArchitecture::Qwen3Moe && arch != dgpp::ModelArchitecture::Qwen3VlMoe)
      throw std::runtime_error("not a qwen3_moe or qwen3_vl_moe checkpoint: " + ckpt +
                               (arch == dgpp::ModelArchitecture::Qwen3
                                    ? " (the dense dialect has no GPU path: qwen3_bind_check --host-forward)"
                                    : ""));
    dgpp::Qwen3TextConfig cfg = dgpp::Qwen3TextConfig::from_json_file(cfg_path);
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    if (cfg.vl())
      std::printf("vision: this checkpoint's tower is bound and NOT served — text path only\n");

    if (!plan_text.empty()) {
      std::vector<long long> parts;
      std::stringstream ps(plan_text);
      for (std::string item; std::getline(ps, item, ',');) parts.push_back(std::stoll(item));
      if (parts.size() < 2 || parts.size() > 3) throw std::runtime_error("--plan takes SLOTS,CONTEXT[,WORLD]");
      const int slots = static_cast<int>(parts[0]);
      const int64_t context = parts[1];
      const int world = parts.size() == 3 ? static_cast<int>(parts[2]) : 1;
      const dgpp::MemoryPlan plan = dgpp::Qwen3Model::plan_memory(
          cfg, dgpp::Qwen3Model::prefill_chunk_tokens(), context, 0, world, dgpp::LoaderResidency::Resident, slots,
          slots);
      std::printf("memory plan: %s, rank 0 of %d, %d slots, %lld-token pool, resident, no draft layer\n",
                  dgpp::model_architecture_name(arch), world, slots, static_cast<long long>(context));
      for (const auto& item : plan.items)
        std::printf("  %9.3f GiB device %8.3f GiB pinned  %s\n", item.device / kGiB, item.pinned / kGiB,
                    item.name.c_str());
      std::printf("  %9.3f GiB device %8.3f GiB pinned  TOTAL\n", plan.device_bytes() / kGiB,
                  plan.pinned_bytes() / kGiB);
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
    if (truncated) cfg.num_hidden_layers = layers;
    const int T = static_cast<int>(ids.size());
    const int H = cfg.hidden_size;
    if (!image_dir.empty()) dgpp::Qwen3LayerStream::set_resident_image_dir(image_dir == "off" ? "" : image_dir);
    const dgpp::LoaderResidency residency =
        resident ? dgpp::LoaderResidency::Resident : dgpp::LoaderResidency::Streaming;
    DGPP_LOG_INFO("qwen3_forward_check: {} ({}) — {} tokens, {} layers{}, {}", ckpt,
                  dgpp::model_architecture_name(arch), T, cfg.num_hidden_layers,
                  truncated ? " (truncated)" : "", resident ? "resident" : "streaming");

    const auto t0 = std::chrono::steady_clock::now();
    dgpp::Qwen3Model model(cfg, ckpt, T, T + 64, residency, /*boundary=*/nullptr, /*rank=*/0, /*world=*/1,
                           /*max_requests=*/1, /*decode_rows=*/0);
    const auto t1 = std::chrono::steady_clock::now();
    const dgpp::Qwen3Model::Outputs out = model.forward(ids, true);
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
    std::vector<int> argmax(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t) {
      const float* row = out.logits.data() + static_cast<size_t>(t) * static_cast<size_t>(V);
      std::vector<int> order(static_cast<size_t>(V));
      for (int64_t v = 0; v < V; ++v) order[static_cast<size_t>(v)] = static_cast<int>(v);
      const int k = static_cast<int>(std::min<int64_t>(std::max(topk, 1), V));
      // Largest first; a tie goes to the lower id.
      std::partial_sort(order.begin(), order.begin() + k, order.end(),
                        [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
      argmax[static_cast<size_t>(t)] = order[0] + out.lm_vocab_begin;
      std::printf("pos %3d id %7lld ->", t, static_cast<long long>(ids[static_cast<size_t>(t)]));
      for (int i = 0; i < k; ++i) std::printf(" %d(%.3f)", order[static_cast<size_t>(i)] + out.lm_vocab_begin, row[order[static_cast<size_t>(i)]]);
      std::printf("\n");
    }
    std::printf("argmax ids:");
    for (int t = 0; t < T; ++t) std::printf("%s%d", t ? "," : " ", argmax[static_cast<size_t>(t)]);
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

    if (host_compare) {
      // The host walk with the engine's rounding points over the same ids:
      // states[0] is the embedding, states[l + 1] the residual after layer l.
      const dgpp::qwen3_ref::HostCheckpoint host(ckpt, truncated ? layers : -1);
      dgpp::qwen3_ref::Trace trace;
      const auto h0 = std::chrono::steady_clock::now();
      const std::vector<double> host_logits =
          dgpp::qwen3_ref::forward(host, ids, dgpp::qwen3_ref::Arithmetic::Engine, &trace);
      const auto h1 = std::chrono::steady_clock::now();
      double worst = 0;
      for (size_t l = 0; l < out.layer_states.size() && l + 1 < trace.states.size(); ++l) {
        double num = 0, den = 0;
        for (size_t i = 0; i < out.layer_states[l].size(); ++i) {
          const double g = dgpp::bf16_bits_to_float(out.layer_states[l][i]);
          const double d = g - trace.states[l + 1][i];
          num += d * d;
          den += trace.states[l + 1][i] * trace.states[l + 1][i];
        }
        const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
        worst = std::max(worst, rel);
        std::printf("host compare layer %2zu: rel l2 %.5f\n", l, rel);
      }
      const std::vector<double> host_lp = dgpp::qwen3_ref::teacher_forced_logprobs(host_logits, ids, cfg.vocab_size);
      double host_mean = 0;
      for (double v : host_lp) host_mean += v / static_cast<double>(std::max<size_t>(host_lp.size(), 1));
      std::printf("host compare: worst layer rel l2 %.5f; mean token logprob host %.5f, gpu %.5f (host walk %.0f ms)\n",
                  worst, host_mean, T > 1 ? sum / (T - 1) : 0.0, ms(h0, h1));
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "qwen3_forward_check: %s\n", e.what());
    return 1;
  }
}
