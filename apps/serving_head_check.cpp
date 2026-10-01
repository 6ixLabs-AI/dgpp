// Teacher-forced scoring of real serving prefill tails. Run twice with
// --full and --compact, then join TP scores with fabric_logprob.py --prefill.
// Unlike a full-forward logprob test, every scored row uses serving's head.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "engine/graph_engine.hpp"
#include "engine/tp_bus.hpp"
#include "loaders/architecture.hpp"
#include "loaders/hf_cache.hpp"
#include "models/dsv41/model.hpp"
#include "models/glm/forward.hpp"
#include "models/glm4/forward.hpp"
#include "models/glm_dsa/model.hpp"
#include "models/mimo/forward.hpp"
#include "models/qwen/forward.hpp"
#include "net/collective_bus.hpp"
#include "text/tokenizer.hpp"

using namespace dgpp;
namespace {
template <class Model>
void score(Model& model, const std::vector<int64_t>& ids, int trials, int rank, int world) {
  const int lengths[] = {1, 4, 8, 17, 64, 129, 257, 1024};
  double elapsed = 0;
  for (int trial = 0; trial < trials; ++trial) {
    const int length = lengths[trial % 8];
    const size_t start = (size_t(trial) * 104729) % (ids.size() - length);
    const std::vector<int64_t> prompt(ids.begin() + start, ids.begin() + start + length);
    const auto t0 = std::chrono::steady_clock::now();
    const auto out = model.session_prefill(0, prompt);
    elapsed += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const int64_t target = ids[start + length];
    const auto& logits = out.logits;
    const int best = std::max_element(logits.begin(), logits.end()) - logits.begin();
    double z = 0;
    for (float x : logits) {
      if (!std::isfinite(x)) throw std::runtime_error("nonfinite head output");
      z += std::exp(double(x) - logits[best]);
    }
    std::string target_value = "nan";
    if (target >= out.lm_vocab_begin && target < out.lm_vocab_begin + out.lm_vocab_count) {
      char value[64];
      std::snprintf(value, sizeof(value), "%.9g", logits[target - out.lm_vocab_begin]);
      target_value = value;
    }
    uint64_t hash = 1469598103934665603ull;
    for (uint16_t x : out.final_hidden_bits) {
      hash ^= x;
      hash *= 1099511628211ull;
    }
    std::printf(
        "[prefill_tf] rank %d step %d: target %lld argmax %d lmax %.9g lse %.12g target_logit %s\n",
        rank, trial, static_cast<long long>(target), best + out.lm_vocab_begin, logits[best],
        logits[best] + std::log(z), target_value.c_str());
    std::printf("[head_hidden] rank %d step %d: %016llx length %d\n", rank, trial,
                static_cast<unsigned long long>(hash), length);
    model.session_close(0);
  }
  std::printf("[prefill_tf_end] rank %d world %d positions %d\n", rank, world, trials);
  std::printf("[head_time] rank %d seconds %.6f capacity %d\n", rank, elapsed,
              model.logits_capacity_rows());
}
}  // namespace
int main(int argc, char** argv) {
  try {
    std::string id, checkpoint, teacher, peer, image_dir;
    int world = 1, rank = 0, port = 29950, trials = 256;
    bool compact = true;
    auto next = [&](int& i) -> std::string {
      if (++i >= argc) throw std::runtime_error("missing argument");
      return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--model")
        id = next(i);
      else if (arg == "--checkpoint-dir")
        checkpoint = next(i);
      else if (arg == "--teacher-file")
        teacher = next(i);
      else if (arg == "--world")
        world = std::stoi(next(i));
      else if (arg == "--rank")
        rank = std::stoi(next(i));
      else if (arg == "--peer")
        peer = next(i);
      else if (arg == "--port")
        port = std::stoi(next(i));
      else if (arg == "--trials")
        trials = std::stoi(next(i));
      else if (arg == "--image-dir")
        image_dir = next(i);
      else if (arg == "--full")
        compact = false;
      else if (arg == "--compact")
        compact = true;
      else
        throw std::runtime_error("unknown argument: " + arg);
    }
    if (trials < 8 || world < 1 || rank < 0 || rank >= world)
      throw std::runtime_error("invalid trial or rank count");
    if (checkpoint.empty()) {
      std::string error;
      checkpoint = hf::model_dir(id, &error);
      if (checkpoint.empty()) throw std::runtime_error("cannot resolve model: " + error);
    }
    std::ifstream input(teacher, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read teacher file");
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    const auto tokenizer =
        text::Tokenizer::load((std::filesystem::path(checkpoint) / "tokenizer.json").string());
    const auto ids = tokenizer.encode(text);
    if (ids.size() <= 1024) throw std::runtime_error("teacher text needs more than 1024 tokens");
    const std::string config = (std::filesystem::path(checkpoint) / "config.json").string();
    std::unique_ptr<net::CollectiveBus> bus;
    std::unique_ptr<BusBoundaryReducer> reducer;
    if (world > 1) {
      bus = std::make_unique<net::CollectiveBus>(fabric_bus_options(
          rank, world, uint16_t(port), peer, 120000, size_t(1024) * 32768 * 2 + 4096));
      std::string error;
      if (!bus->start(&error)) throw std::runtime_error(error);
      reducer = std::make_unique<BusBoundaryReducer>(*bus, 600000);
    }
    const auto run = [&](auto& model) { score(model, ids, trials, rank, world); };
    switch (detect_architecture_file(config)) {
      case ModelArchitecture::Glm4Moe: {
        auto cfg = Glm4TextConfig::from_json_file(config);
        Glm4LayerStream::set_resident_image_dir(image_dir);
        Glm4Model model(cfg, checkpoint, 1024, 2048, Glm4Residency::Resident, reducer.get(), rank,
                        world, 1, false, 8, compact);
        run(model);
        break;
      }
      case ModelArchitecture::GlmMoeDsa: {
        auto cfg = GlmDsaTextConfig::from_json_file(config);
        GlmDsaLayerStream::set_resident_image_dir(image_dir);
        GlmDsaModel model(cfg, checkpoint, 1024, 2048, GlmDsaResidency::Resident, reducer.get(),
                          rank, world, 1, false, 8, LatentFormat::kBf16, compact);
        run(model);
        break;
      }
      case ModelArchitecture::MimoV2: {
        auto cfg = MimoTextConfig::from_json_file(config);
        MimoLayerStream::set_resident_image_dir(image_dir);
        MimoModel model(cfg, checkpoint, 1024, 2048, MimoResidency::Resident, reducer.get(), rank,
                        world, 1, false, 8, LatentFormat::kBf16, compact);
        run(model);
        break;
      }
      case ModelArchitecture::DeepseekV41: {
        auto cfg = Dsv41TextConfig::from_json_file(config);
        Dsv41LayerStream::set_resident_image_dir(image_dir);
        Dsv41Model model(cfg, checkpoint, 1024, 2048, Dsv41Residency::Resident, reducer.get(), rank,
                         world, 1, false, 8, compact);
        run(model);
        break;
      }
      case ModelArchitecture::Qwen4Exp: {
        auto cfg = QwenTextConfig::from_json_file(config);
        QwenLayerStream::set_resident_image_dir(image_dir);
        QwenLayerStream::set_dense_weights_fp8(true);
        QwenLayerStream::set_ngram_table_mmap(true);
        QwenModel model(cfg, checkpoint, 1024, 2048, QwenResidency::Resident, reducer.get(), rank,
                        world, 1, false, 8, false, compact);
        run(model);
        break;
      }
      case ModelArchitecture::Glm5: {
        auto cfg = GlmTextConfig::from_json_file(config);
        GlmLayerStream::set_resident_image_dir(image_dir);
        GlmDiagnosticModel model(cfg, checkpoint, 1024, 2048, reducer.get(), rank, world,
                                 GlmResidency::Resident,
                                 world > 1 ? GlmHeadSharding::VocabSharded : GlmHeadSharding::Full,
                                 1, false, LatentFormat::kBf16, compact);
        run(model);
        break;
      }
      default:
        throw std::runtime_error("unsupported architecture");
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "serving_head_check: %s\n", e.what());
    return 1;
  }
}
