// The Qwen3.8-Flash-Next TP forwards: loopback worlds of
// 2 and 4 ranks in one process (real verbs QPs over 127.0.0.1, the bus the
// fabric uses) over the tiny fixture, against the world-1 model as the
// oracle. Asserted:
//   * cross-rank: every layer's hyper state, the final read, the routing
//     decisions and the lm-head slices' top-1 are bitwise identical on
//     every rank (the canonical rank-order fold; a divergence is the
//     silent-corruption class);
//   * vs world 1: per-layer hyper states within the numerics budget (the
//     partial sums fold in bf16 on the wire, so the block outputs differ
//     from the single fp32 accumulation at rounding level), the final
//     read likewise, the top-1 per token equal or a certified near tie;
//   * the sub-128 expert scale grid: the fixture's 64-wide experts slice
//     to 32 (world 2) and 16 (world 4) — the GEMV core's re-blocked grid.
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "common/test.hpp"
#include "engine/tp_bus.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/forward.hpp"
#include "net/collective_bus.hpp"
#include "qwen_fixture.hpp"

namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::BusBoundaryReducer;
using dgpp::QwenModel;
using dgpp::QwenResidency;
using dgpp::QwenTextConfig;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

int wait_timeout_ms() {
  const char* v = std::getenv("DGPP_TEST_WAIT_TIMEOUT_MS");
  return v ? std::atoi(v) : 60000;
}

BusOptions loop_options(int rank, int world, uint16_t port) {
  BusOptions o;
  o.world_size = world;
  o.my_rank = rank;
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : "127.0.0.1";
  o.rendezvous_timeout_ms = 20000;
  o.lat_slots = 8;
  // A 256 KB latency slot: every fold of this test (72 rows x 256 or
  // 1024 bf16) rides the chunked latency path. The bulk RS+AG machine
  // stalls at world 4 when a buffer spans fewer stripes than ranks (the
  // 36 KB attention fold: one stripe, three ranks with no shard, the AG
  // phase waiting on an arrival that never comes — 2026-09-09, recorded
  // in the plan's risks; the fabric's prefill folds are megabytes and its
  // decode folds ride the recorded collectives, so production is not on
  // that path).
  o.lat_slot_bytes = 262144;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = [] {
    const char* ms = std::getenv("DGPP_TEST_BUS_TIMEOUT_MS");
    return ms ? std::atoi(ms) : 5000;
  }();
  o.consumer_deadline_s = [] {
    const char* s = std::getenv("DGPP_TEST_CONSUMER_DEADLINE_S");
    return s ? std::atof(s) : 20.0;
  }();
  o.launch_consumers = false;
  return o;
}

std::vector<std::unique_ptr<CollectiveBus>> start_world(int world, uint16_t port) {
  std::vector<std::unique_ptr<CollectiveBus>> out;
  for (int r = 0; r < world; ++r) out.push_back(std::make_unique<CollectiveBus>(loop_options(r, world, port)));
  std::vector<std::string> errors(static_cast<size_t>(world));
  std::thread listener([&] {
    if (!out[0]->start(&errors[0])) DGPP_LOG_ERROR("qwen tp world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("qwen tp world rank {}: {}", r, errors[static_cast<size_t>(r)]);
    });
  listener.join();
  for (auto& t : connectors) t.join();
  for (int r = 0; r < world; ++r)
    if (!errors[static_cast<size_t>(r)].empty()) return {};
  return out;
}

struct ConstructBarrier {
  std::mutex mu;
  std::condition_variable cv;
  int left;
  explicit ConstructBarrier(int world) : left(world) {}
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mu);
    if (--left == 0) cv.notify_all();
    else cv.wait(lock, [this] { return left == 0; });
  }
};

struct RankOutcome {
  std::string error;
  QwenModel::Outputs out;
};

void rank_work(int rank, int world, const std::string& dir, const QwenTextConfig& cfg,
               const std::vector<int64_t>& tokens, CollectiveBus* bus, ConstructBarrier* barrier,
               RankOutcome* out) {
  bool arrived = false;
  auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    QwenModel model(cfg, dir, static_cast<int>(tokens.size()), 256, QwenResidency::Streaming, &reducer,
                    rank, world);
    arrive_once();
    out->out = model.forward(tokens, true);
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("qwen tp rank {} failed: {}", rank, e.what());
    out->error = "rank " + std::to_string(rank) + ": " + e.what();
    arrive_once();
  }
}

std::vector<int64_t> make_tokens(const QwenTextConfig& cfg, int n) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = 0xA5A5F00DBEEF1234ull;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  if (n > 9) t[9] = cfg.eos_token_ids.empty() ? 0 : cfg.eos_token_ids[0];
  return t;
}

bool bits_equal(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * 2) == 0);
}

struct Stats {
  double l2 = 0, max_ulps = 0;
  long hard = 0, total = 0;
};

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

Stats compare(const std::vector<uint16_t>& got, const std::vector<uint16_t>& want) {
  Stats s;
  s.total = static_cast<long>(got.size());
  double rms = 0;
  for (uint16_t v : want) rms += std::pow(bf16_bits_to_float(v), 2);
  rms = std::sqrt(rms / std::max<size_t>(want.size(), 1));
  double d2 = 0, o2 = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = bf16_bits_to_float(got[i]), w = bf16_bits_to_float(want[i]);
    int u = bf16_ulps(got[i], want[i]);
    if (std::fabs(g - w) <= 0.02 * rms) u = 0;
    s.max_ulps = std::max(s.max_ulps, static_cast<double>(u));
    if (u > 128) ++s.hard;
    d2 += (g - w) * (g - w);
    o2 += w * w;
  }
  s.l2 = std::sqrt(d2) / std::sqrt(o2 + 1e-30);
  return s;
}

// The merged top-1 over the ranks' vocab slices: the highest logit, ties
// to the lower id.
std::vector<std::pair<int32_t, float>> merged_top1(const std::vector<RankOutcome>& ranks, int T) {
  std::vector<std::pair<int32_t, float>> best(static_cast<size_t>(T), {-1, -INFINITY});
  for (const RankOutcome& r : ranks)
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < r.out.lm_vocab_count; ++c) {
        const float v = r.out.logits[static_cast<size_t>(t) * r.out.lm_vocab_count + c];
        const int32_t id = r.out.lm_vocab_begin + c;
        auto& b = best[static_cast<size_t>(t)];
        if (v > b.second || (v == b.second && id < b.first)) b = {id, v};
      }
  return best;
}

void check_world(int world, uint16_t port, const std::string& dir, const QwenTextConfig& cfg,
                 const std::vector<int64_t>& tokens, const QwenModel::Outputs& ref) {
  DGPP_LOG_INFO("qwen TP loopback world={} tokens={}", world, tokens.size());
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(world, port);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<RankOutcome> ranks(static_cast<size_t>(world));
  ConstructBarrier barrier(world);
  std::vector<std::thread> workers;
  for (int r = 0; r < world; ++r)
    workers.emplace_back(rank_work, r, world, dir, std::cref(cfg), std::cref(tokens),
                         buses[static_cast<size_t>(r)].get(), &barrier, &ranks[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < world; ++r) require(ranks[static_cast<size_t>(r)].error.empty(), ranks[static_cast<size_t>(r)].error);
  const int T = static_cast<int>(tokens.size());
  const int L = cfg.num_hidden_layers;
  // ---- cross-rank bitwise ------------------------------------------------------
  for (int r = 1; r < world; ++r) {
    const QwenModel::Outputs& a = ranks[0].out;
    const QwenModel::Outputs& b = ranks[static_cast<size_t>(r)].out;
    require(bits_equal(a.final_hidden_bits, b.final_hidden_bits), "final hidden differs across ranks");
    require(a.layer_states.size() == static_cast<size_t>(L) && b.layer_states.size() == static_cast<size_t>(L),
            "layer captures missing");
    for (int l = 0; l < L; ++l)
      require(bits_equal(a.layer_states[static_cast<size_t>(l)], b.layer_states[static_cast<size_t>(l)]),
              "layer " + std::to_string(l) + " hyper state differs across ranks");
    require(a.route_ids == b.route_ids, "routing differs across ranks");
  }
  // ---- vs world 1 ---------------------------------------------------------------
  const QwenModel::Outputs& o = ranks[0].out;
  for (int l = 0; l < L; ++l) {
    const Stats s = compare(o.layer_states[static_cast<size_t>(l)], ref.layer_states[static_cast<size_t>(l)]);
    std::printf("[ .. ] world %d layer %d: l2 %.3g max %g ulps hard %ld of %ld\n", world, l, s.l2, s.max_ulps,
                s.hard, s.total);
    require(s.l2 < 0.02 && s.hard == 0, "world " + std::to_string(world) + " layer " + std::to_string(l) +
                                            " outside the numerics budget");
  }
  const Stats f = compare(o.final_hidden_bits, ref.final_hidden_bits);
  std::printf("[ .. ] world %d final hidden: l2 %.3g max %g ulps hard %ld of %ld\n", world, f.l2, f.max_ulps, f.hard,
              f.total);
  require(f.l2 < 0.02 && f.hard == 0, "world " + std::to_string(world) + " final hidden outside the budget");
  // Routing: flips are near ties at most (counted, bounded).
  long flips = 0, slots = 0;
  for (int l = 0; l < L; ++l)
    for (size_t i = 0; i < ref.route_ids[static_cast<size_t>(l)].size(); ++i, ++slots)
      if (o.route_ids[static_cast<size_t>(l)][i] != ref.route_ids[static_cast<size_t>(l)][i]) ++flips;
  std::printf("[ .. ] world %d routing: %ld of %ld slots differ from world 1\n", world, flips, slots);
  require(flips <= slots / 20, "too many routing flips against world 1");
  // The merged top-1 vs world 1's: equal, or a near tie in world 1's logits.
  const auto merged = merged_top1(ranks, T);
  const std::vector<std::pair<int32_t, float>> single = merged_top1({RankOutcome{"", ref}}, T);
  int mism = 0;
  for (int t = 0; t < T; ++t) {
    if (merged[static_cast<size_t>(t)].first == single[static_cast<size_t>(t)].first) continue;
    // world 1's logit for the TP top-1 must be within 2 % of its own top-1.
    const float v_tp = ref.logits[static_cast<size_t>(t) * ref.lm_vocab_count + merged[static_cast<size_t>(t)].first];
    const float v_1 = single[static_cast<size_t>(t)].second;
    if (std::fabs(v_1 - v_tp) > 0.02 * std::fabs(v_1)) ++mism;
  }
  std::printf("[ .. ] world %d top-1: %d of %d rows beyond a near tie\n", world, mism, T);
  require(mism == 0, "world " + std::to_string(world) + " top-1 differs beyond a near tie");
}

}  // namespace

DGPP_TEST(qwen_tp_loopback_worlds_2_and_4_match_world_1) {
  const std::string dir = (fs::current_path() / "qwen_tp_fixture").string();
  const QwenTextConfig cfg = qwenfx::tiny_config();
  qwenfx::write_fixture_for(cfg, dir);
  const std::vector<int64_t> tokens = make_tokens(cfg, 72);
  QwenModel::Outputs ref;
  {
    QwenModel single(cfg, dir, static_cast<int>(tokens.size()), 256);
    ref = single.forward(tokens, true);
  }
  check_world(2, 29942, dir, cfg, tokens, ref);
  check_world(4, 29943, dir, cfg, tokens, ref);
}

int main() { return dgpp::test::run_all(); }
