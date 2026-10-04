// UNVERIFIED DRAFT (2026-10-04): never compiled with nvcc / GCC, never run.
// Built only under -DDGPP_BUILD_QWEN3_PLAIN_DRAFT=ON.
//
// The plain Qwen3-MoE draft on a GPU, against the family's host side on
// synthetic checkpoints of both NVFP4 containers (tests/unit/qwen3_fixture.hpp
// — the generator the unit gate holds to tools/qwen3_synth.py — at GLM-4.7's
// fixture geometry: hidden 256, 4 query / 2 kv heads of 128, 8 experts of
// 128, a 512-token vocabulary, three layers):
//
//   (the loader's resident values and rank slices: tests/cuda/qwen3_loader_test.cpp)
//   - the cold forward: every layer's residual, the final hidden state and
//     the teacher-forced log-probabilities against the host reference with
//     the engine's rounding points (models/qwen3/reference.hpp), and the
//     router's decisions against it;
//   - the session surface: a prefill's last row and T=1 decode steps against
//     the cold re-forward;
//   - the VL dialect's refusal of a vision placeholder.
//
// The bounds are the draft's first guesses (bf16 noise over three layers);
// the first run on a Spark is what sets them.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "models/qwen3/config.hpp"
#include "models/qwen3/model.hpp"
#include "models/qwen3/reference.hpp"
#include "qwen3_gpu_fixture.hpp"

namespace {

using namespace qwen3_gpu_fixture;
using dgpp::Qwen3Model;
using dgpp::qwen3_ref::Arithmetic;

std::vector<double> logprobs_of(const std::vector<float>& logits, const std::vector<int64_t>& tok, int V) {
  const std::vector<double> wide(logits.begin(), logits.end());
  return dgpp::qwen3_ref::teacher_forced_logprobs(wide, tok, V);
}

void check_forward(const char* tag, bool vl) {
  const Fixture fx = write_fixture(tag, vl);
  const HostCheckpoint ck(fx.dir);
  const Qwen3TextConfig& c = ck.config();
  const std::vector<int64_t> tok = ids();
  const int T = static_cast<int>(tok.size()), H = c.hidden_size, V = c.vocab_size;
  dgpp::qwen3_ref::Trace host;
  const std::vector<double> host_logits = dgpp::qwen3_ref::forward(ck, tok, Arithmetic::Engine, &host);

  Qwen3Model model(c, fx.dir, T, T + 64);  // route traces are on by default (the session core's)
  const Qwen3Model::Outputs out = model.forward(tok, /*capture_layers=*/true);
  require(out.lm_vocab_count == V && out.logits.size() == static_cast<size_t>(T) * V, std::string(tag) + ": logits shape");
  require(out.layer_states.size() == static_cast<size_t>(c.num_hidden_layers), std::string(tag) + ": one state per layer");
  for (size_t l = 0; l < out.layer_states.size(); ++l) {
    double num = 0, den = 0;
    for (size_t i = 0; i < out.layer_states[l].size(); ++i) {
      const double g = dgpp::bf16_bits_to_float(out.layer_states[l][i]);
      num += (g - host.states[l + 1][i]) * (g - host.states[l + 1][i]);
      den += host.states[l + 1][i] * host.states[l + 1][i];
    }
    const double rel = std::sqrt(num / den);
    std::printf("%s layer %zu: residual rel l2 vs the host walk %.6f\n", tag, l, rel);
    require(rel < 0.02, std::string(tag) + ": layer " + std::to_string(l) + " residual rel l2 " + std::to_string(rel));
  }
  {
    double num = 0, den = 0;
    for (size_t i = 0; i < out.final_hidden_bits.size(); ++i) {
      const double g = dgpp::bf16_bits_to_float(out.final_hidden_bits[i]);
      num += (g - host.final_hidden[i]) * (g - host.final_hidden[i]);
      den += host.final_hidden[i] * host.final_hidden[i];
    }
    require(std::sqrt(num / den) < 0.02, std::string(tag) + ": final hidden rel l2 " + std::to_string(std::sqrt(num / den)));
    require(static_cast<int>(out.final_hidden_bits.size()) == T * H, std::string(tag) + ": final hidden shape");
  }
  const std::vector<double> gpu_lp = logprobs_of(out.logits, tok, V);
  const std::vector<double> host_lp = dgpp::qwen3_ref::teacher_forced_logprobs(host_logits, tok, V);
  double worst = 0, mean = 0;
  for (size_t i = 0; i < gpu_lp.size(); ++i) {
    worst = std::max(worst, std::fabs(gpu_lp[i] - host_lp[i]));
    mean += std::fabs(gpu_lp[i] - host_lp[i]) / static_cast<double>(gpu_lp.size());
  }
  std::printf("%s: teacher-forced log-probabilities vs the host walk: max %.5f mean %.5f nat\n", tag, worst, mean);
  require(worst < 0.1 && mean < 0.03, std::string(tag) + ": log-probabilities drift");
  // The routes: the same experts wherever the host's selection is not a near tie.
  require(out.route_ids.size() == static_cast<size_t>(c.num_hidden_layers), std::string(tag) + ": one route per layer");
  size_t same = 0, total = 0;
  for (size_t l = 0; l < out.route_ids.size(); ++l)
    for (size_t i = 0; i < out.route_ids[l].size(); ++i, ++total) same += out.route_ids[l][i] == host.router_ids[l][i];
  std::printf("%s: routes equal on %zu of %zu (token, slot) pairs\n", tag, same, total);
  require(same * 10 >= total * 9, std::string(tag) + ": routes diverge from the host walk");
}

int argmax(const float* row, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (row[i] > row[best]) best = i;
  return best;
}

}  // namespace

DGPP_TEST(qwen3_forward_matches_the_host_walk_packed) { check_forward("vl_fwd", /*vl=*/true); }
DGPP_TEST(qwen3_forward_matches_the_host_walk_modelopt) { check_forward("moe_fwd", /*vl=*/false); }

DGPP_TEST(qwen3_session_decode_agrees_with_the_cold_forward) {
  const Fixture fx = write_fixture("session", /*vl=*/false);
  const Qwen3TextConfig cfg = Qwen3TextConfig::from_json_file(fx.dir + "/config.json");
  const std::vector<int64_t> tok = ids();
  const int V = cfg.vocab_size;
  const size_t P = 6;
  const std::vector<int64_t> prompt(tok.begin(), tok.begin() + static_cast<std::ptrdiff_t>(P));
  Qwen3Model model(cfg, fx.dir, static_cast<int>(tok.size()), 256);
  const Qwen3Model::Outputs cold = model.forward(tok);
  // A prefill's last row is the cold forward's row P - 1.
  Qwen3Model::Outputs o = model.session_prefill(0, prompt);
  require(model.session_position(0) == static_cast<int64_t>(P), "prefill: position");
  for (size_t step = P; step <= tok.size(); ++step) {
    const float* want = cold.logits.data() + (step - 1) * static_cast<size_t>(V);
    double worst = 0;
    for (int i = 0; i < V; ++i) worst = std::max(worst, static_cast<double>(std::fabs(o.logits[static_cast<size_t>(i)] - want[i])));
    std::printf("session row %zu: max |d logit| vs the cold forward %.5f\n", step - 1, worst);
    require(worst < 0.05, "session row " + std::to_string(step - 1) + " drifts from the cold forward: " + std::to_string(worst));
    require(argmax(o.logits.data(), V) == argmax(want, V) || worst < 0.05, "session argmax");
    if (step == tok.size()) break;
    o = model.session_step(0, tok[step]);  // teacher-forced: the next given token
  }
  model.session_close(0);
  require(model.session_position(0) == 0, "close: position");
}

DGPP_TEST(qwen3_vl_forward_refuses_a_vision_placeholder) {
  const Fixture fx = write_fixture("refusal", /*vl=*/true);
  Qwen3Model model(fx.cfg, fx.dir, 8, 64);
  bool refused = false;
  try {
    (void)model.forward({1, fx.cfg.image_token_id, 3});
  } catch (const std::invalid_argument& e) {
    refused = std::string(e.what()).find("vision tower is not served") != std::string::npos;
  }
  require(refused, "an image placeholder is refused by name");
  (void)model.forward({1, 2, 3});  // and the slot is still usable
}

int main() { return ::dgpp::test::run_all(); }
