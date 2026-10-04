// gemma4_host_check: the Gemma 4 HOST reference (models/gemma4/reference.hpp)
// on a checkpoint — no GPU, no CUDA. It reads the weights through the
// family's tensor table, runs the fp32 forward on the CPU and prints the
// teacher-forced log-probabilities in the forward checks' format, so
// sixlabs/bench/compare_forward_check.py can set it against the numpy
// reference's `score` of the same ids (tools/gemma4_reference.py). What it
// gates: the table, the loader's dequantization and the C++ operators on
// REAL weights, before any kernel exists to blame.
//
//   gemma4_host_check --model ORG/NAME | --checkpoint-dir DIR
//                     --ids 1,2,3,... | --ids-json FILE
//                     [--layers N] [--topk K]
//
// --layers N loads and runs the first N layers only (the head then reads a
// residual the model was not trained to read: only meaningful against the
// reference run with the same --layers). The reference keeps a layer's
// weights in fp32 (the 31B: about 2 GiB a layer, 5.6 GiB for the
// embedding) and is a triple loop: budget about half a second per token per
// layer on one core, and N x 2 GiB of memory.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "loaders/architecture.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/minijson.hpp"
#include "models/gemma4/config.hpp"
#include "models/gemma4/reference.hpp"

namespace {

std::vector<int64_t> parse_ids(const std::string& text) {
  std::vector<int64_t> ids;
  std::stringstream ss(text);
  for (std::string part; std::getline(ss, part, ',');)
    if (!part.empty()) ids.push_back(std::stoll(part));
  return ids;
}

std::vector<int64_t> read_ids_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();
  const auto parsed = dgpp::minijson::parse(text);
  const dgpp::minijson::Value* list = &parsed.root;
  if (list->is_object()) {
    const dgpp::minijson::Value* inner = list->find("token_ids");
    if (inner == nullptr) inner = list->find("ids");
    if (inner == nullptr) throw std::runtime_error(path + ": expected a list of token ids");
    list = inner;
  }
  std::vector<int64_t> ids;
  for (const auto& v : list->items()) ids.push_back(v.as_int());
  return ids;
}

int run(int argc, char** argv) {
  std::string checkpoint_dir, model_id, ids_text, ids_json;
  int layers = -1, topk = 5;
  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::format("missing value for {}", a));
      return argv[++i];
    };
    if (a == "--checkpoint-dir") checkpoint_dir = next();
    else if (a == "--model") model_id = next();
    else if (a == "--ids") ids_text = next();
    else if (a == "--ids-json") ids_json = next();
    else if (a == "--layers") layers = std::stoi(next());
    else if (a == "--topk") topk = std::stoi(next());
    else throw std::runtime_error(std::format("unknown argument {}", a));
  }
  if (!model_id.empty()) {
    std::string err;
    checkpoint_dir = dgpp::hf::model_dir(model_id, &err);
    if (checkpoint_dir.empty()) throw std::runtime_error(std::format("--model {}: {}", model_id, err));
  }
  if (checkpoint_dir.empty())
    throw std::runtime_error("usage: gemma4_host_check --model ORG/NAME | --checkpoint-dir DIR --ids ...|--ids-json FILE "
                             "[--layers N] [--topk K]");
  const std::vector<int64_t> ids = !ids_json.empty() ? read_ids_json(ids_json) : parse_ids(ids_text);
  if (ids.empty()) throw std::runtime_error("--ids or --ids-json is required");

  const std::string config_path = checkpoint_dir + "/config.json";
  if (dgpp::detect_architecture_file(config_path) != dgpp::ModelArchitecture::Gemma4)
    throw std::runtime_error(config_path + " is not a gemma4 checkpoint");
  const dgpp::Gemma4TextConfig cfg = dgpp::Gemma4TextConfig::from_json_file(config_path);
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  const dgpp::gemma4_ref::Model model(cfg, dgpp::gemma4_ref::load_weights(cfg, checkpoint_dir, layers));
  const auto t1 = clock::now();
  std::printf("gemma4 host reference: %d of %d layers, %zu tokens, hidden %d, vocab %d\n", model.layers(),
              cfg.num_hidden_layers, ids.size(), cfg.hidden_size, cfg.vocab_size);

  const int H = cfg.hidden_size, V = cfg.vocab_size, T = static_cast<int>(ids.size());
  dgpp::gemma4_ref::State st;
  const std::vector<float> x = model.forward(ids, st, [&](std::string_view name, const std::vector<float>& v) {
    if (name.size() != 4 || name.substr(0, 2) != "h_") return;  // the residual stream after each layer
    double ss = 0.0, mx = 0.0;
    for (const float f : v) {
      ss += static_cast<double>(f) * f;
      mx = std::max(mx, static_cast<double>(std::fabs(f)));
    }
    std::printf("%.*s: residual rms %.6g max %.6g\n", static_cast<int>(name.size()), name.data(),
                std::sqrt(ss / static_cast<double>(v.size())), mx);
  });
  // The head row by row: the tied embedding is [vocab, hidden].
  std::vector<int> argmax(static_cast<size_t>(T));
  std::vector<double> token_lp;
  for (int t = 0; t < T; ++t) {
    const std::vector<float> row(x.begin() + static_cast<long>(t) * H, x.begin() + static_cast<long>(t + 1) * H);
    const std::vector<float> lp = model.logprobs(row);
    std::vector<std::pair<float, int>> best;
    for (int v = 0; v < V; ++v) best.emplace_back(lp[static_cast<size_t>(v)], v);
    const int k = std::min(topk, V);
    std::partial_sort(best.begin(), best.begin() + k, best.end(), [](const auto& a, const auto& b) {
      return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    argmax[static_cast<size_t>(t)] = best[0].second;
    std::printf("pos %3d id %7lld ->", t, static_cast<long long>(ids[static_cast<size_t>(t)]));
    for (int j = 0; j < k; ++j) std::printf(" %d(%.3f)", best[static_cast<size_t>(j)].second, best[static_cast<size_t>(j)].first);
    std::printf("\n");
    if (t + 1 < T) token_lp.push_back(lp[static_cast<size_t>(ids[static_cast<size_t>(t + 1)])]);
  }
  const auto t2 = clock::now();
  std::printf("argmax ids:");
  for (int t = 0; t < T; ++t) std::printf("%s%d", t ? "," : " ", argmax[static_cast<size_t>(t)]);
  std::printf("\n");
  std::printf("token logprobs:");
  double sum = 0.0;
  for (size_t i = 0; i < token_lp.size(); ++i) {
    std::printf("%s%.5f", i ? "," : " ", token_lp[i]);
    sum += token_lp[i];
  }
  std::printf("\n");
  if (!token_lp.empty()) std::printf("mean token logprob %.5f over %zu tokens\n", sum / static_cast<double>(token_lp.size()), token_lp.size());
  const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  std::printf("load %.0f ms, forward %.0f ms (host, fp32, one core)\n", ms(t0, t1), ms(t1, t2));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gemma4_host_check: %s\n", e.what());
    return 2;
  }
}
