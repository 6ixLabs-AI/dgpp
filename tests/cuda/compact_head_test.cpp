// Serving allocation and row selection across the non-Qwen families.
// Both head paths are checked against FP64 on identical BF16 operands.
// Hidden state, fixed-token continuation, draft and cache restore stay exact.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "dsv41_fixture.hpp"
#include "glm4_fixture.hpp"
#include "glm_dsa_fixture.hpp"
#include "loaders/safetensors.hpp"
#include "mimo_fixture.hpp"
#include "models/dsv41/model.hpp"
#include "models/glm/forward.hpp"
#include "models/glm4/forward.hpp"
#include "models/glm_dsa/model.hpp"
#include "models/mimo/forward.hpp"

using namespace dgpp;
GlmTextConfig glm_tp_test_config();
void glm_tp_write_fixture(const std::string& dir);

namespace {
void require(bool ok, const char* msg) {
  if (!ok) throw std::runtime_error(msg);
}
bool exact(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
std::vector<int64_t> tokens(int n, int vocab, int salt) {
  std::vector<int64_t> ids(n);
  for (int i = 0; i < n; ++i) ids[i] = (i * 17 + salt) % vocab;
  return ids;
}
struct Accuracy {
  double full_l2 = 0, compact_l2 = 0, max_logprob_delta = 0;
  int rows = 0;
  template <class Output>
  void compare(const Output& a, const Output& b, const uint16_t* weight, int hidden) {
    require(a.final_hidden_bits == b.final_hidden_bits, "head changed hidden state");
    require(a.logits.size() == b.logits.size() && !a.logits.empty(), "head output shape");
    const size_t vocab = a.logits.size();
    double norm = 0, ea = 0, eb = 0;
    double za = 0, zb = 0;
    const double ma = *std::max_element(a.logits.begin(), a.logits.end());
    const double mb = *std::max_element(b.logits.begin(), b.logits.end());
    for (size_t v = 0; v < vocab; ++v) {
      double oracle = 0;
      for (int k = 0; k < hidden; ++k)
        oracle += static_cast<double>(bf16_bits_to_float(a.final_hidden_bits[k])) *
                  static_cast<double>(bf16_bits_to_float(weight[v * hidden + k]));
      require(std::isfinite(a.logits[v]) && std::isfinite(b.logits[v]), "nonfinite head output");
      norm += oracle * oracle;
      ea += std::pow(a.logits[v] - oracle, 2);
      eb += std::pow(b.logits[v] - oracle, 2);
      za += std::exp(a.logits[v] - ma);
      zb += std::exp(b.logits[v] - mb);
    }
    full_l2 = std::max(full_l2, std::sqrt(ea / (norm + 1e-30)));
    compact_l2 = std::max(compact_l2, std::sqrt(eb / (norm + 1e-30)));
    const double shift = ma + std::log(za) - mb - std::log(zb);
    for (size_t v = 0; v < vocab; ++v)
      max_logprob_delta = std::max(max_logprob_delta, std::abs(a.logits[v] - b.logits[v] - shift));
    require(full_l2 < 1e-5 && compact_l2 < 1e-5, "head exceeds FP64 oracle budget");
    require(max_logprob_delta < 2e-4, "head changes token log probability beyond rounding budget");
    ++rows;
  }
};

template <class Model, class Config, class Factory, class Planner>
void check(const Config& cfg, const std::string& dir, Factory make, Planner plan, int capacity) {
  auto full = make(false);
  auto compact = make(true);
  const bool enabled = compact_logits_rows(true, capacity, 2) > 0;
  require(full->logits_capacity_rows() == 256, "diagnostic logits capacity");
  require(compact->logits_capacity_rows() == (enabled ? capacity : 256), "serving logits capacity");
  const size_t saving =
      enabled ? size_t(256 - capacity) * full->lm_vocab_count() * sizeof(float) : 0;
  require(plan(false).total_bytes() - plan(true).total_bytes() == saving,
          "logits memory accounting");
  auto file = SafetensorsFile::open(dir + "/model.safetensors");
  const auto& tensor =
      file->at(std::is_same_v<Model, Dsv41Model> ? "head.weight" : "lm_head.weight");
  require(tensor.dtype == DType::BF16, "oracle needs BF16 head weights");
  const auto* weight = static_cast<const uint16_t*>(tensor.data);
  Accuracy accuracy;
  for (int length : {1, 4, 5, 8, 9, 17, 127, 128, 129, 257}) {
    if constexpr (std::is_same_v<Model, GlmDiagnosticModel>) {
      if (length > 256) continue;  // Flash chunks require a 2048-row allocation.
    }
    const auto ids = tokens(length, cfg.vocab_size, length + 7);
    const auto a = full->session_prefill(0, ids);
    const auto b = compact->session_prefill(0, ids);
    accuracy.compare(a, b, weight, cfg.hidden_size);
    // Draft starts from the saved hidden, never from the rounded head output.
    const auto da = full->session_draft(0, {3});
    const auto db = compact->session_draft(0, {3});
    require(exact(da.logits, db.logits), "compact prefill changed draft logits");
    for (int step = 0; step < 3; ++step) {
      const auto ca = full->session_step(0, 11 + step);
      const auto cb = compact->session_step(0, 11 + step);
      require(exact(ca.logits, cb.logits), "compact prefill changed fixed-token decode");
      require(ca.final_hidden_bits == cb.final_hidden_bits,
              "compact prefill changed decode hidden");
    }
    full->session_close(0);
    compact->session_close(0);
  }
  // Nonuniform groups, including one-row spans and the capacity boundary.
  for (int length : {1, 4, 8, 15, 17, 65, 127}) {
    const auto a = tokens(length, cfg.vocab_size, 3);
    const auto b = tokens(length + 1, cfg.vocab_size, 27);
    const auto f = full->session_prefill_group({1, 0}, {&a, &b});
    const auto c = compact->session_prefill_group({1, 0}, {&a, &b});
    for (int i = 0; i < 2; ++i) {
      accuracy.compare(f[i], c[i], weight, cfg.hidden_size);
      require(exact(full->session_step(1 - i, 3).logits, compact->session_step(1 - i, 3).logits),
              "group tail or request mapping changed decode");
    }
    for (auto* m : {full.get(), compact.get()}) {
      m->session_close(0);
      m->session_close(1);
    }
  }
  // A prefix snapshot on a block boundary, restored to the other slot.
  const auto prefix = tokens(128, cfg.vocab_size, 19);
  const auto suffix = tokens(17, cfg.vocab_size, 37);
  for (auto* m : {full.get(), compact.get()}) {
    (void)m->session_prefill(0, prefix);
    void* snapshot = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&snapshot, m->session_snapshot_bytes()));
    const auto meta = m->session_snapshot(0, snapshot);
    m->session_close(0);
    m->session_attach(1, snapshot, meta);
    const auto hot = m->session_prefill_resume(1, suffix, {});
    m->session_close(1);
    m->session_attach(0, snapshot, meta);
    const auto repeat = m->session_prefill_resume(0, suffix, {});
    require(exact(hot.logits, repeat.logits), "prefix restore changed logits across slots");
    m->session_close(0);
    m->session_release_snapshot(meta);
    DGPP_CUDA_OK(cudaFree(snapshot));
  }
  if (enabled) {
    bool refused = false;
    try {
      (void)compact->forward(tokens(capacity + 1, cfg.vocab_size, 17));
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    require(refused, "compact storage must reject diagnostic all-row overflow");
    compact->session_close(0);
  }
  std::printf(
      "[ .. ] %s: %d rows, FP64 relative L2 full=%.3g compact=%.3g, max token logprob delta=%.3g, "
      "saved=%zu bytes\n",
      dir.c_str(), accuracy.rows, accuracy.full_l2, accuracy.compact_l2, accuracy.max_logprob_delta,
      saving);
}
}  // namespace

DGPP_TEST(compact_head_glm4) {
  const auto cfg = glm4fx::tiny_config();
  const std::string dir = "compact_glm4_fixture";
  glm4fx::write_fixture(cfg, dir);
  check<Glm4Model>(
      cfg, dir,
      [&](bool compact) {
        return std::make_unique<Glm4Model>(cfg, dir, 256, 1024, Glm4Residency::Resident, nullptr, 0,
                                           1, 2, true, 16, compact);
      },
      [&](bool compact) {
        return Glm4Model::plan_memory(cfg, 256, 1024, 0, 1, Glm4Residency::Resident, 2, true, 16,
                                      compact);
      },
      16);
}
DGPP_TEST(compact_head_glm_dsa) {
  const auto cfg = glmdsafx::tiny_config();
  const std::string dir = "compact_glm_dsa_fixture";
  glmdsafx::write_fixture(cfg, dir);
  check<GlmDsaModel>(
      cfg, dir,
      [&](bool compact) {
        return std::make_unique<GlmDsaModel>(cfg, dir, 256, 1024, GlmDsaResidency::Resident,
                                             nullptr, 0, 1, 2, true, 16, LatentFormat::kBf16,
                                             compact);
      },
      [&](bool compact) {
        return GlmDsaModel::plan_memory(cfg, 256, 1024, 0, 1, GlmDsaResidency::Resident, 2, true,
                                        16, LatentFormat::kBf16, compact);
      },
      16);
}
DGPP_TEST(compact_head_mimo) {
  const auto cfg = mimofx::tiny_config();
  const std::string dir = "compact_mimo_fixture";
  mimofx::write_fixture(cfg, dir);
  check<MimoModel>(
      cfg, dir,
      [&](bool compact) {
        return std::make_unique<MimoModel>(cfg, dir, 256, 1024, MimoResidency::Resident, nullptr, 0,
                                           1, 2, true, 16, LatentFormat::kBf16, compact);
      },
      [&](bool compact) {
        return MimoModel::plan_memory(cfg, 256, 1024, 0, 1, MimoResidency::Resident, 2, true, 16,
                                      LatentFormat::kBf16, compact);
      },
      16);
}
DGPP_TEST(compact_head_dsv41) {
  const auto cfg = dsv41fx::tiny_config();
  const std::string dir = "compact_dsv41_fixture";
  dsv41fx::write_fixture(cfg, dir);
  (void)dsv41fx::fixture_sidecar(cfg, dir);
  for (bool bounded : {false, true})
    check<Dsv41Model>(
        cfg, dir,
        [&](bool compact) {
          auto m = std::make_unique<Dsv41Model>(cfg, dir, 256, 1024, Dsv41Residency::Resident,
                                                nullptr, 0, 1, 2, true, 16, compact);
          m->set_prefill_bounded(bounded);
          return m;
        },
        [&](bool compact) {
          return Dsv41Model::plan_memory(cfg, 256, 1024, 0, 1, Dsv41Residency::Resident, 2, true,
                                         16, compact);
        },
        16);
}
DGPP_TEST(compact_head_glm_flash) {
  const auto cfg = glm_tp_test_config();
  const std::string dir = "compact_glm_flash_fixture";
  glm_tp_write_fixture(dir);
  check<GlmDiagnosticModel>(
      cfg, dir,
      [&](bool compact) {
        return std::make_unique<GlmDiagnosticModel>(cfg, dir, 256, 1024, nullptr, 0, 1,
                                                    GlmResidency::Resident, GlmHeadSharding::Full,
                                                    2, true, LatentFormat::kBf16, compact);
      },
      [&](bool compact) {
        return GlmDiagnosticModel::plan_memory(cfg, 256, 1024, 0, 1, GlmResidency::Resident,
                                               GlmHeadSharding::Full, 2, true, LatentFormat::kBf16,
                                               compact);
      },
      8);
}
int main() {
  return dgpp::test::run_all();
}
