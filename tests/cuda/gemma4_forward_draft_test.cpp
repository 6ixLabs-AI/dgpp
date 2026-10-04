// The Gemma 4 DRAFT GPU forward (models/gemma4/forward_draft.hpp) against the
// numpy reference's vectors on the synthetic checkpoint "Gemma4Synth v1" —
// the test the draft must pass on a Spark before the real checkpoint is
// worth loading. Built only with -DDGPP_BUILD_GEMMA_DRAFT=ON.
//
// The same test was run on a Mac with forward_draft.cu compiled as plain
// C++ (its kernels executed as loops): that is all the evidence the draft
// has as of 2026-10-04. It needs no checkpoint: the fixture writes its own.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "gemma4_reference_vectors.hpp"
#include "gemma4_synth.hpp"
#include "loaders/minijson.hpp"
#include "models/gemma4/config.hpp"
#include "models/gemma4/forward_draft.hpp"

namespace {

namespace vec = gemma4_vectors;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

void close(const float* got, const float* want, size_t n, float tol, const std::string& what) {
  float worst = 0.0f;
  size_t at = 0;
  for (size_t i = 0; i < n; ++i) {
    const float d = std::fabs(got[i] - want[i]);
    if (!(d <= worst)) {
      worst = d;
      at = i;
    }
  }
  if (!(worst <= tol))
    throw std::runtime_error(what + ": max |diff| " + std::to_string(worst) + " at " + std::to_string(at) + " (got " +
                             std::to_string(got[at]) + ", want " + std::to_string(want[at]) + ")");
}

struct SynthCheckpoint {
  std::filesystem::path dir;
  dgpp::Gemma4TextConfig cfg;

  // The 31B-shaped recipe by default; `moe` the 26B-A4B-shaped one (NVFP4 attention, the MoE block).
  explicit SynthCheckpoint(bool moe = false) {
    const char* config_json = moe ? vec::kConfigJsonMoe : vec::kConfigJsonMlp;
    const std::string text = config_json;
    const auto parsed = dgpp::minijson::parse(text);
    cfg = dgpp::Gemma4TextConfig::parse(parsed.root);
    const auto tensors = gemma4_synth::tensors(cfg, vec::kSeed);
    require(gemma4_synth::content_hash(tensors) == (moe ? vec::kContentHashMoe : vec::kContentHashMlp),
            "the fixture is not the Python tool's");
    static int serial = 0;
    dir = std::filesystem::temp_directory_path() /
          ("gemma4_draft_" + std::to_string(::getpid()) + "_" + std::to_string(serial++));
    gemma4_synth::write_checkpoint(dir, config_json, tensors);
  }
  ~SynthCheckpoint() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

std::vector<int64_t> ids() { return std::vector<int64_t>(vec::kIds, vec::kIds + vec::kTokens); }

constexpr float kTol = 2e-4f;

}  // namespace

DGPP_TEST(gemma4_draft_forward_matches_the_numpy_reference) {
  const SynthCheckpoint ck;
  dgpp::Gemma4DraftForward model(ck.cfg, ck.dir.string(), -1, 64);
  require(model.layers() == 6 && model.pos() == 0 && model.device_bytes() > 0, "construction");
  std::vector<std::vector<float>> trace;
  const std::vector<float> x = model.forward(ids(), &trace);
  require(model.pos() == vec::kTokens && trace.size() == 7 && x.size() == 12 * 32, "state / shape");
  const float* hidden[7] = {vec::kHidden0Mlp, vec::kHidden1Mlp, vec::kHidden2Mlp, vec::kHidden3Mlp,
                            vec::kHidden4Mlp, vec::kHidden5Mlp, vec::kHidden6Mlp};
  for (int l = 0; l <= 6; ++l)
    close(trace[static_cast<size_t>(l)].data(), hidden[l], 384, kTol, "residual stream after layer " + std::to_string(l));
  close(x.data(), vec::kHidden6Mlp, 384, kTol, "returned stream");
  const std::vector<float> logits = model.logits(x);
  require(logits.size() == 12 * 96, "logits shape");
  close(logits.data(), vec::kLogitsMlp, 1152, kTol, "soft-capped logits");
}

DGPP_TEST(gemma4_draft_incremental_forward_equals_whole_sequence) {
  // The K/V cache path: one token at a time, and in chunks of 5 (a chunk boundary inside a window).
  const SynthCheckpoint ck;
  for (int chunk : {1, 5}) {
    dgpp::Gemma4DraftForward model(ck.cfg, ck.dir.string(), -1, 64);
    std::vector<float> x;
    const std::vector<int64_t> all = ids();
    for (size_t s = 0; s < all.size(); s += static_cast<size_t>(chunk)) {
      const std::vector<int64_t> part(all.begin() + static_cast<long>(s),
                                      all.begin() + static_cast<long>(std::min(all.size(), s + static_cast<size_t>(chunk))));
      const std::vector<float> y = model.forward(part);
      x.insert(x.end(), y.begin(), y.end());
    }
    require(model.pos() == 12, "position");
    close(x.data(), vec::kHidden6Mlp, 384, kTol, "chunk " + std::to_string(chunk));
  }
}

DGPP_TEST(gemma4_draft_limits) {
  const SynthCheckpoint ck;
  // --layers 3 stops after layer 2.
  {
    dgpp::Gemma4DraftForward model(ck.cfg, ck.dir.string(), 3, 64);
    const std::vector<float> x = model.forward(ids());
    close(x.data(), vec::kHidden3Mlp, 384, kTol, "three layers");
  }
  // A sequence past the cache, and an id outside the vocabulary, are refused.
  dgpp::Gemma4DraftForward small(ck.cfg, ck.dir.string(), 1, 8);
  bool threw = false;
  try {
    (void)small.forward(ids());
  } catch (const std::runtime_error&) {
    threw = true;
  }
  require(threw, "a sequence longer than max_tokens");
  threw = false;
  try {
    (void)small.forward({96});
  } catch (const std::out_of_range&) {
    threw = true;
  }
  require(threw, "an out-of-range id");
}

DGPP_TEST(gemma4_draft_moe_forward_matches_the_numpy_reference) {
  // The 26B-A4B's shape: NVFP4 attention projections, the router, the picked experts beside the
  // dense MLP — whole sequence, then one token at a time.
  const SynthCheckpoint ck(/*moe=*/true);
  const float* hidden[7] = {vec::kHidden0Moe, vec::kHidden1Moe, vec::kHidden2Moe, vec::kHidden3Moe,
                            vec::kHidden4Moe, vec::kHidden5Moe, vec::kHidden6Moe};
  {
    dgpp::Gemma4DraftForward model(ck.cfg, ck.dir.string(), -1, 64);
    std::vector<std::vector<float>> trace;
    const std::vector<float> x = model.forward(ids(), &trace);
    require(trace.size() == 7, "trace");
    for (int l = 0; l <= 6; ++l)
      close(trace[static_cast<size_t>(l)].data(), hidden[l], 384, kTol, "residual stream after layer " + std::to_string(l));
    close(model.logits(x).data(), vec::kLogitsMoe, 1152, kTol, "soft-capped logits");
  }
  dgpp::Gemma4DraftForward model(ck.cfg, ck.dir.string(), -1, 64);
  std::vector<float> x;
  for (const int64_t id : ids()) {
    const std::vector<float> y = model.forward({id});
    x.insert(x.end(), y.begin(), y.end());
  }
  close(x.data(), vec::kHidden6Moe, 384, kTol, "one token at a time");
}

int main() { return ::dgpp::test::run_all(); }
