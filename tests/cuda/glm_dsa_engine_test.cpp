// The full GLM-5.3 decode engines over the engine core (docs/glm53_plan.md
// G5): EagerEngineAdapter<GlmDsaModel> and GraphEngineAdapter<GlmDsaModel>
// on the fixture, a loopback world of 2 (real verbs QPs over 127.0.0.1)
// against the world-1 eager engine. The decode batch is capped at eight
// rows (plan D9), so the depth-2 world runs two slots (2 x 3 rows).
//
// Gates: the graph engine's SCALAR replays (the device-driven T=1 graph:
// device positions, the recorded commit, the pinned token upload) and its
// ROW-BATCHED replays (the fixed slot-major batch off the persistent
// feeds, closed slots padding at -1, the 2-slot and full families) produce
// the eager engine's transcripts at the same world exactly (the recorded
// kernels are the eager kernels); the world-2 eager transcripts follow the
// world-1 ones (reported; the folds reassociate, so a near tie may flip a
// late token — the first tokens must agree).
// Graph models use the compact serving allocation; eager references retain
// full heads. Cache, batching and speculative contracts keep their strict gates.
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
#include "models/glm_dsa/config.hpp"
#include "models/glm_dsa/model.hpp"
#include "net/collective_bus.hpp"
#include "glm_dsa_fixture.hpp"
#include "engine/eager_engine.hpp"
#include "engine/graph_engine.hpp"
#include "engine_test_ties.hpp"
#include "engine/tp_bus.hpp"

namespace fs = std::filesystem;
using dgpp::BusBoundaryReducer;
using dgpp::EagerEngineAdapter;
using dgpp::GraphEngineAdapter;
using dgpp::GlmDsaModel;
using dgpp::GlmDsaResidency;
using dgpp::GlmDsaTextConfig;
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
  o.lat_slot_bytes = 262144;  // every fold on the latency path (glm4_tp_test's note)
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
    if (!out[0]->start(&errors[0])) DGPP_LOG_ERROR("glm_dsa engine world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("glm_dsa engine world rank {}: {}", r, errors[static_cast<size_t>(r)]);
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

std::vector<int64_t> smoke_tokens(const GlmDsaTextConfig& cfg, int n, uint64_t seed) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = seed;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  return t;
}

std::string ids_text(const std::vector<int32_t>& ids) {
  std::string s;
  for (size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
  return s;
}

// The engine's greedy transcript for one prompt in slot `req`: the
// prefill's pick then `steps` scalar steps.
template <class Engine>
std::vector<int32_t> solo(Engine& eng, int req, const std::vector<int64_t>& prompt, int steps) {
  std::vector<int32_t> out;
  out.push_back(eng.prefill(req, prompt));
  eng.reserve(req, static_cast<int64_t>(prompt.size()) + steps + 1);
  for (int s = 0; s < steps; ++s) {
    const std::vector<int32_t> t = eng.step(req);
    require(t.size() == 1, "a scalar step decides one token");
    out.push_back(t[0]);
  }
  eng.close(req);
  return out;
}

size_t agreeing_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  size_t n = 0;
  while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
  return n;
}

constexpr int kSlots = 3;
constexpr int kDepth2Slots = 2;  // 2 x 3 verify rows within the 8-row cap
constexpr int kSteps = 60;
constexpr int kMaxTokens = 64;
constexpr int64_t kCache = 512;
constexpr int kWorld = 2;
constexpr uint16_t kPort = 29954;  // 29954 (graph vs eager), 29955 (mtp), 29956 (depth 2), 29957 (sharded), 29960 (sixteen rows)


// The batched engines against the eager one at the same world (2026-09-14):
// the dense sites' lowering follows the rows of a launch (kernels/gemm.hpp
// dense_gemv_rows — the GEMV chunks to four rows, cuBLASLt's algorithm or
// the streaming tensor-core form above), so a batched step's rows and the
// eager scalar's are tolerance-equal, not bitwise; on this fixture a near
// tie flips and every later token follows. The rule (engine_test_ties.hpp):
// the first kAgreePositions decisions agree, or the first difference among
// them sits on a near tie of the world-1 reference's own decision (its
// top-2 margin under kTieMargin); later differences are reported.
constexpr float kTieMargin = 0.1f;
constexpr size_t kAgreePositions = 3;

struct Ref {
  std::vector<int32_t> a, b, c;
  std::vector<float> am, bm, cm;  // the world-1 reference's top-2 margin at every decision
};

Ref world1_reference(const GlmDsaTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
                     const std::vector<int64_t>& B, const std::vector<int64_t>& C) {
  GlmDsaModel m(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, nullptr, 0, 1, kSlots);
  std::vector<float> margins;
  EagerEngineAdapter<GlmDsaModel> eng(&m, kSlots, engine_ties::margin_pick(cfg.vocab_size, &margins));
  Ref r;
  r.a = solo(eng, 0, A, kSteps);
  r.am = margins; margins.clear();
  r.b = solo(eng, 1, B, kSteps);
  r.bm = margins; margins.clear();
  r.c = solo(eng, 2, C, kSteps);
  r.cm = margins;
  return r;
}

struct RankOutcome {
  std::string error;
  std::vector<int32_t> ea, eb, ec;   // the eager engine, world 2
  std::vector<int32_t> ga;           // the graph engine, scalar
  std::vector<int32_t> ba, bb, bc;   // the graph engine, batched
  std::vector<int32_t> ma, mb, mc;   // the MTP graph engine: scalar A, batched B and C
  int mtp_steps_a = 0;               // scalar MTP steps A took (< kSteps: drafts stood)
};

void rank_work(int r, const GlmDsaTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
               const std::vector<int64_t>& B, const std::vector<int64_t>& C, CollectiveBus* bus,
               ConstructBarrier* barrier, RankOutcome* out) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  uint16_t* scratch = nullptr;
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    GlmDsaModel eager(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots);
    GlmDsaModel graph(
        cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots,
        /*mtp=*/false, /*decode_rows=*/0, dgpp::LatentFormat::kBf16, /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<GlmDsaModel> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    GraphEngineAdapter<GlmDsaModel> graph_engine(&graph, bus, r, kWorld, scratch, cfg.vocab_size,
                                               wait_timeout_ms(), /*batch_min_live=*/2);

    // 1. The eager engine at world 2.
    out->ea = solo(eager_engine, 0, A, kSteps);
    out->eb = solo(eager_engine, 1, B, kSteps);
    out->ec = solo(eager_engine, 2, C, kSteps);

    // 2. The graph engine, scalar replays (one live slot).
    out->ga = solo(graph_engine, 1, A, kSteps);
    graph_engine.drain();

    // 3. The row-batched replays: A and B live (the 2-slot family), then
    //    C joins (the full family), then B closes (slot 1 pads).
    out->ba.push_back(graph_engine.prefill(0, A));
    graph_engine.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 1);
    out->bb.push_back(graph_engine.prefill(1, B));
    graph_engine.reserve(1, static_cast<int64_t>(B.size()) + kSteps + 1);
    for (int s = 0; s < 25; ++s) {
      const auto t = graph_engine.step_batch({0, 1});
      require(t.size() == 2 && t[0].size() == 1 && t[1].size() == 1, "batch step shape");
      out->ba.push_back(t[0][0]);
      out->bb.push_back(t[1][0]);
    }
    graph_engine.drain();
    out->bc.push_back(graph_engine.prefill(2, C));
    graph_engine.reserve(2, static_cast<int64_t>(C.size()) + kSteps + 1);
    for (int s = 0; s < 20; ++s) {
      const auto t = graph_engine.step_batch({0, 1, 2});
      require(t.size() == 3, "batch step shape");
      out->ba.push_back(t[0][0]);
      out->bb.push_back(t[1][0]);
      out->bc.push_back(t[2][0]);
    }
    graph_engine.close(1);
    for (int s = 0; s < 15; ++s) {
      const auto t = graph_engine.step_batch({0, 2});
      require(t.size() == 2, "batch step shape");
      out->ba.push_back(t[0][0]);
      out->bc.push_back(t[1][0]);
    }
    graph_engine.close(0);
    graph_engine.close(2);
    graph_engine.drain();
    cudaFreeHost(scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

// The MTP graph engine (the one-graph draft) in its own bus world (one
// graph era per bus): the T=2 verify with the recorded commit, the in-graph
// draft and its pick, the [next, draft] feed — scalar for A, then the
// batched families for B and C — against the plain eager engine.
void rank_work_mtp(int r, const GlmDsaTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
                   const std::vector<int64_t>& B, const std::vector<int64_t>& C, CollectiveBus* bus,
                   ConstructBarrier* barrier, RankOutcome* out) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  uint16_t* scratch = nullptr;
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    GlmDsaModel eager(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots);
    GlmDsaModel mtp(
        cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots,
        /*mtp=*/true, /*decode_rows=*/0, dgpp::LatentFormat::kBf16, /*serving_logits=*/true);
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<GlmDsaModel> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    out->ea = solo(eager_engine, 0, A, kSteps);
    out->eb = solo(eager_engine, 1, B, kSteps);
    out->ec = solo(eager_engine, 2, C, kSteps);
    {
      GraphEngineAdapter<GlmDsaModel> mtp_engine(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                               /*batch_min_live=*/2);
      out->ma.push_back(mtp_engine.prefill(0, A));
      mtp_engine.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 2);
      while (out->ma.size() < static_cast<size_t>(kSteps) + 1) {
        const std::vector<int32_t> t = mtp_engine.step(0);
        require(!t.empty() && t.size() <= 2, "mtp step shape");
        out->ma.insert(out->ma.end(), t.begin(), t.end());
        ++out->mtp_steps_a;
      }
      out->ma.resize(static_cast<size_t>(kSteps) + 1);
      mtp_engine.close(0);
      out->mb.push_back(mtp_engine.prefill(1, B));
      mtp_engine.reserve(1, static_cast<int64_t>(B.size()) + kSteps + 2);
      out->mc.push_back(mtp_engine.prefill(2, C));
      mtp_engine.reserve(2, static_cast<int64_t>(C.size()) + kSteps + 2);
      while (out->mb.size() < static_cast<size_t>(kSteps) + 1 || out->mc.size() < static_cast<size_t>(kSteps) + 1) {
        const auto t = mtp_engine.step_batch({1, 2});
        require(t.size() == 2, "mtp batch step shape");
        out->mb.insert(out->mb.end(), t[0].begin(), t[0].end());
        out->mc.insert(out->mc.end(), t[1].begin(), t[1].end());
      }
      out->mb.resize(static_cast<size_t>(kSteps) + 1);
      out->mc.resize(static_cast<size_t>(kSteps) + 1);
      mtp_engine.close(1);
      mtp_engine.close(2);
      mtp_engine.drain();
    }
    cudaFreeHost(scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

// Sixteen decode rows (2026-09-13, the batch's cap lifted from eight):
// eight slots at depth 1, all live, on the every-slot family — the select's
// two row groups, the pick's sixteen request slots and the sixteen-row
// folds in one world. Each slot's batched transcript is the eager engine's,
// bitwise, with BOTH models configured for the same decode rows: the GEMM
// interface lowers every bf16 call up to the model's decode rows to the
// row-independent GEMV chain (kernels/gemm.hpp), so a prompt of 9–16 tokens
// prefills through it on a sixteen-row model and through cuBLASLt on an
// eight-row one — last-bit differences that flip a near tie thirty tokens
// later (found 2026-09-13 with the eager model at eight rows; the
// DGPP_WIDE_PREFILL_PROBE=1 aid prints which prompts' prefill logits agree).
// DGPP_WIDE_SLOTS=n and DGPP_WIDE_DEPTH=d reshape the gate (n slots at
// depth d; the bus allows 2n + 2 x families <= 32 variants), and
// DGPP_WIDE_ROTATE=k puts prompt k in slot 0.
constexpr int kWideSlotsMax = 16;
constexpr int64_t kWideCache = 4096;  // sixteen sessions x up to three 64-token blocks
inline int wide_slots() {
  const char* v = std::getenv("DGPP_WIDE_SLOTS");
  const int n = v ? std::atoi(v) : 8;
  return n < 2 ? 2 : n > kWideSlotsMax ? kWideSlotsMax : n;
}
inline int wide_depth() {
  const char* v = std::getenv("DGPP_WIDE_DEPTH");
  const int d = v ? std::atoi(v) : 1;
  return d < 1 ? 1 : d > 3 ? 3 : d;
}
struct WideOutcome {
  std::vector<std::vector<int32_t>> eager, batched;
  uint64_t full_steps = 0;
  std::string error;
};
void rank_work_wide(int r, const GlmDsaTextConfig& cfg, const std::string& dir,
                    const std::vector<std::vector<int64_t>>& prompts, CollectiveBus* bus, ConstructBarrier* barrier,
                    WideOutcome* out) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  const int slots = wide_slots();
  const int rows_per_slot = 1 + wide_depth();
  const int decode_rows = slots * rows_per_slot;
  uint16_t* scratch = nullptr;
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    GlmDsaModel eager(cfg, dir, kMaxTokens, kWideCache, GlmDsaResidency::Resident, &reducer, r, kWorld, slots,
                      /*mtp=*/false, decode_rows);
    GlmDsaModel mtp(cfg, dir, kMaxTokens, kWideCache, GlmDsaResidency::Resident, &reducer, r,
                    kWorld, slots,
                    /*mtp=*/true, decode_rows, dgpp::LatentFormat::kBf16, /*serving_logits=*/true);
    require(mtp.max_decode_rows() == std::max(8, decode_rows), "the slots make the batch");
    if (std::getenv("DGPP_WIDE_PREFILL_PROBE")) {
      for (int i = 0; i < slots; ++i) {
        const GlmDsaModel::Outputs a = eager.session_prefill(0, prompts[static_cast<size_t>(i)]);
        eager.session_close(0);
        const GlmDsaModel::Outputs b = mtp.session_prefill(0, prompts[static_cast<size_t>(i)]);
        mtp.session_close(0);
        size_t first = a.logits.size();
        for (size_t k = 0; k < a.logits.size() && k < b.logits.size(); ++k)
          if (a.logits[k] != b.logits[k]) { first = k; break; }
        DGPP_LOG_INFO("rank {} prefill probe: prompt {} ({} tokens): logits {} ({} elems, first difference at {})", r, i,
                      prompts[static_cast<size_t>(i)].size(), first == a.logits.size() ? "bitwise" : "DIFFER",
                      a.logits.size(), first);
      }
    }
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<GlmDsaModel> eager_engine(
        &eager, slots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    for (int i = 0; i < slots; ++i) out->eager.push_back(solo(eager_engine, i, prompts[static_cast<size_t>(i)], kSteps));
    {
      GraphEngineAdapter<GlmDsaModel> mtp_engine(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                               /*batch_min_live=*/2, nullptr, nullptr, dgpp::kSamplingCandidates, nullptr, 0,
                                               /*mtp_depth=*/wide_depth());
      out->batched.assign(static_cast<size_t>(slots), {});
      std::vector<int> reqs;
      for (int i = 0; i < slots; ++i) {
        const std::vector<int64_t>& p = prompts[static_cast<size_t>(i)];
        out->batched[static_cast<size_t>(i)].push_back(mtp_engine.prefill(i, p));
        // Room for a run that accepts every draft (rows_per_slot tokens a step).
        mtp_engine.reserve(i, static_cast<int64_t>(p.size()) + rows_per_slot * kSteps + 2);
        reqs.push_back(i);
      }
      const auto all_done = [&] {
        for (const auto& t : out->batched)
          if (t.size() < static_cast<size_t>(kSteps) + 1) return false;
        return true;
      };
      while (!all_done()) {
        const auto t = mtp_engine.step_batch(reqs);
        require(t.size() == static_cast<size_t>(slots), "wide batch step shape");
        for (int i = 0; i < slots; ++i)
          out->batched[static_cast<size_t>(i)].insert(out->batched[static_cast<size_t>(i)].end(),
                                                      t[static_cast<size_t>(i)].begin(), t[static_cast<size_t>(i)].end());
      }
      for (auto& t : out->batched) t.resize(static_cast<size_t>(kSteps) + 1);
      // The every-slot family is the last (graph_engine.hpp).
      out->full_steps = mtp_engine.batch_family_steps(static_cast<int>(mtp_engine.batch_families().size()) - 1);
      for (int i = 0; i < slots; ++i) mtp_engine.close(i);
      mtp_engine.drain();
    }
    cudaFreeHost(scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

}  // namespace

DGPP_TEST(glm_dsa_engines_loopback_world_2_sixteen_row_batch_matches_eager) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config();
  const std::string dir = "glm_dsa_engine_fixture";
  glmdsafx::write_fixture(cfg, dir);
  const int slots = wide_slots();
  std::vector<std::vector<int64_t>> prompts;
  const int lengths[kWideSlotsMax] = {23, 17, 11, 19, 13, 21, 15, 9, 25, 14, 10, 18, 12, 20, 16, 22};
  for (int i = 0; i < kWideSlotsMax; ++i)
    prompts.push_back(smoke_tokens(cfg, lengths[i], 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull * static_cast<uint64_t>(i + 1)));
  if (const char* rot = std::getenv("DGPP_WIDE_ROTATE"))
    std::rotate(prompts.begin(), prompts.begin() + (std::atoi(rot) % kWideSlotsMax), prompts.end());
  prompts.resize(static_cast<size_t>(slots));
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, 29960);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<WideOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_wide, r, std::cref(cfg), std::cref(dir), std::cref(prompts),
                         buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].eager == outs[0].eager && outs[static_cast<size_t>(r)].batched == outs[0].batched,
            "the ranks' transcripts differ");
  const WideOutcome& o = outs[0];
  DGPP_LOG_INFO("world 2 {}-row batch ({} slots x {} rows): {} every-slot steps; slot 0 eager {} | batched {}",
                slots * (1 + wide_depth()), slots, 1 + wide_depth(), o.full_steps, ids_text(o.eager[0]), ids_text(o.batched[0]));
  require(o.full_steps > 0, "the every-slot family replayed");
  // The world-1 reference's margins per prompt (the near-tie rule's evidence).
  std::vector<std::vector<float>> margins(static_cast<size_t>(slots));
  {
    GlmDsaModel w1(cfg, dir, kMaxTokens, kWideCache, GlmDsaResidency::Resident, nullptr, 0, 1, slots);
    std::vector<float> m;
    EagerEngineAdapter<GlmDsaModel> eng(&w1, slots, engine_ties::margin_pick(cfg.vocab_size, &m));
    for (int i = 0; i < slots; ++i) {
      (void)solo(eng, i, prompts[static_cast<size_t>(i)], kSteps);
      margins[static_cast<size_t>(i)] = m;
      m.clear();
    }
  }
  int differ = 0;
  for (int i = 0; i < slots; ++i) {
    const auto& e = o.eager[static_cast<size_t>(i)];
    const auto& b = o.batched[static_cast<size_t>(i)];
    if (b != e) {
      ++differ;
      DGPP_LOG_INFO("slot {} (prompt {} tokens): batched differs from eager at index {}/{}: eager {} | batched {}", i,
                    prompts[static_cast<size_t>(i)].size(), agreeing_prefix(b, e), e.size(), ids_text(e), ids_text(b));
    }
    engine_ties::require_agrees_or_tie(b, e, margins[static_cast<size_t>(i)], kTieMargin, kAgreePositions,
                                       ("slot " + std::to_string(i) + " of the wide batch").c_str());
  }
  DGPP_LOG_INFO("wide batch: {} of {} slots differ from the eager engine past the near-tie rule's positions", differ, slots);
}

// Depth 2: the T=3 verify with two drafts, the second off the
// block's chain row (its own output row as the hidden, the first draft's
// pick as the token), in its own bus world — A on the scalar graph, then B
// and C together on the ROW-BATCHED graph (the batched chain: every slot's
// chain row in one draft-block run). The decode batch is capped at eight
// rows (the fused DSA select, plan D9), so this world runs two slots at
// 3 rows each (6 rows: slots 1 and 2 share the 2-slot family). The
// transcripts must be the plain eager engine's exactly; the pass count
// reports the second draft's worth.
void rank_work_mtp_depth2(int r, const GlmDsaTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
                          const std::vector<int64_t>& B, const std::vector<int64_t>& C, CollectiveBus* bus,
                          ConstructBarrier* barrier, RankOutcome* out) {
  bool arrived = false;
  const auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  uint16_t* scratch = nullptr;
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    GlmDsaModel eager(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld, kSlots);
    // The decode-row ceiling of the depth-2 shape: kDepth2Slots x 3 rows = 6
    // (the model's cap is 8, plan D9).
    GlmDsaModel mtp(cfg, dir, kMaxTokens, kCache, GlmDsaResidency::Resident, &reducer, r, kWorld,
                    kDepth2Slots,
                    /*mtp=*/true, /*decode_rows=*/kDepth2Slots * 3, dgpp::LatentFormat::kBf16,
                    /*serving_logits=*/true);
    require(mtp.max_decode_rows() == 8, "the model floors the decode rows at eight (the batch's cap)");
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                               sizeof(uint16_t) * dgpp::kPickScratchElems(kWorld), cudaHostAllocDefault));
    arrive_once();
    EagerEngineAdapter<GlmDsaModel> eager_engine(
        &eager, kSlots, dgpp::make_fabric_pick(bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms()));
    out->ea = solo(eager_engine, 0, A, kSteps);
    out->eb = solo(eager_engine, 1, B, kSteps);
    out->ec = solo(eager_engine, 2, C, kSteps);
    {
      GraphEngineAdapter<GlmDsaModel> d2(&mtp, bus, r, kWorld, scratch, cfg.vocab_size, wait_timeout_ms(),
                                       /*batch_min_live=*/2, nullptr, nullptr, dgpp::kSamplingCandidates, nullptr, 0,
                                       /*mtp_depth=*/2);
      require(!d2.batch_families().empty() && d2.batch_families().front() == 2,
              "the depth-2 world builds the 2-slot batch family over the 8-row ceiling");
      // A alone: the scalar depth-2 graph.
      out->ma.push_back(d2.prefill(0, A));
      d2.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 3);
      while (out->ma.size() < static_cast<size_t>(kSteps) + 1) {
        const std::vector<int32_t> t = d2.step(0);
        require(!t.empty() && t.size() <= 3, "mtp depth-2 step shape");
        out->ma.insert(out->ma.end(), t.begin(), t.end());
        ++out->mtp_steps_a;
      }
      out->ma.resize(static_cast<size_t>(kSteps) + 1);
      d2.close(0);
      // B and C together: the row-batched depth-2 graph (the two slots of
      // this world form the 2-slot family).
      out->mb.push_back(d2.prefill(0, B));
      d2.reserve(0, static_cast<int64_t>(B.size()) + kSteps + 3);
      out->mc.push_back(d2.prefill(1, C));
      d2.reserve(1, static_cast<int64_t>(C.size()) + kSteps + 3);
      while (out->mb.size() < static_cast<size_t>(kSteps) + 1 || out->mc.size() < static_cast<size_t>(kSteps) + 1) {
        const auto t = d2.step_batch({0, 1});
        require(t.size() == 2 && !t[0].empty() && t[0].size() <= 3 && !t[1].empty() && t[1].size() <= 3,
                "mtp depth-2 batch step shape");
        out->mb.insert(out->mb.end(), t[0].begin(), t[0].end());
        out->mc.insert(out->mc.end(), t[1].begin(), t[1].end());
      }
      out->mb.resize(static_cast<size_t>(kSteps) + 1);
      out->mc.resize(static_cast<size_t>(kSteps) + 1);
      d2.close(0);
      d2.close(1);
      d2.drain();
      require(d2.batch_family_steps(0) > 0, "the batched depth-2 steps ran on the 2-slot family");
    }
    cudaFreeHost(scratch);
  } catch (const std::exception& e) {
    if (scratch) cudaFreeHost(scratch);
    out->error = "rank " + std::to_string(r) + ": " + e.what();
    arrive_once();
  }
}

DGPP_TEST(glm_dsa_engines_loopback_world_2_mtp_depth2_graph_matches_plain_decode) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config();
  const std::string dir = "glm_dsa_engine_fixture";
  glmdsafx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, kPort + 2);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<RankOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_mtp_depth2, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B),
                         std::cref(C), buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].ma == outs[0].ma && outs[static_cast<size_t>(r)].mb == outs[0].mb &&
                outs[static_cast<size_t>(r)].mc == outs[0].mc,
            "the ranks' depth-2 transcripts differ");
  const Ref ref = world1_reference(cfg, dir, A, B, C);
  const RankOutcome& o = outs[0];
  DGPP_LOG_INFO("world 2 MTP depth 2: scalar A {} ({} steps for {} tokens) | batched B {} | C {}", ids_text(o.ma),
                o.mtp_steps_a, kSteps, ids_text(o.mb), ids_text(o.mc));
  require(o.ma == o.ea, "the scalar depth-2 transcript of A differs from the plain eager engine's");
  engine_ties::require_agrees_or_tie(o.mb, o.eb, ref.bm, kTieMargin, kAgreePositions, "the batched depth-2 transcript of B");
  engine_ties::require_agrees_or_tie(o.mc, o.ec, ref.cm, kTieMargin, kAgreePositions, "the batched depth-2 transcript of C");
}

DGPP_TEST(glm_dsa_engines_loopback_world_2_mtp_graph_matches_plain_decode) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config();
  const std::string dir = "glm_dsa_engine_fixture";
  glmdsafx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, kPort + 1);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<RankOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work_mtp, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B),
                         std::cref(C), buses[static_cast<size_t>(r)].get(), &barrier,
                         &outs[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r)
    require(outs[static_cast<size_t>(r)].ma == outs[0].ma && outs[static_cast<size_t>(r)].mb == outs[0].mb,
            "the ranks' MTP transcripts differ");
  const Ref ref = world1_reference(cfg, dir, A, B, C);
  const RankOutcome& o = outs[0];
  DGPP_LOG_INFO("world 2 MTP graph: A {} ({} steps) | B {} | C {}", ids_text(o.ma), o.mtp_steps_a, ids_text(o.mb),
                ids_text(o.mc));
  require(o.ma == o.ea, "the MTP scalar transcript differs from the plain eager engine's");
  engine_ties::require_agrees_or_tie(o.mb, o.eb, ref.bm, kTieMargin, kAgreePositions, "the MTP batched transcript of B");
  engine_ties::require_agrees_or_tie(o.mc, o.ec, ref.cm, kTieMargin, kAgreePositions, "the MTP batched transcript of C");
  // A random-weight fixture drafts by chance only (the acceptance rate is
  // the real checkpoint's measurement, scripts/fabric_mtp_classes.sh).
  DGPP_LOG_INFO("world 2 MTP graph: A took {} steps for {} tokens", o.mtp_steps_a, kSteps);
}

RankOutcome graph_world(const GlmDsaTextConfig& cfg, const std::string& dir, const std::vector<int64_t>& A,
                        const std::vector<int64_t>& B, const std::vector<int64_t>& C, const Ref& ref, uint16_t port) {
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(kWorld, port);
  require(!buses.empty(), "the loopback bus world failed to start");
  std::vector<RankOutcome> outs(kWorld);
  ConstructBarrier barrier(kWorld);
  std::vector<std::thread> workers;
  for (int r = 0; r < kWorld; ++r)
    workers.emplace_back(rank_work, r, std::cref(cfg), std::cref(dir), std::cref(A), std::cref(B), std::cref(C),
                         buses[static_cast<size_t>(r)].get(), &barrier, &outs[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < kWorld; ++r) require(outs[static_cast<size_t>(r)].error.empty(), outs[static_cast<size_t>(r)].error);
  for (int r = 1; r < kWorld; ++r) {
    require(outs[static_cast<size_t>(r)].ea == outs[0].ea && outs[static_cast<size_t>(r)].ga == outs[0].ga &&
                outs[static_cast<size_t>(r)].ba == outs[0].ba,
            "the ranks' transcripts differ");
  }
  const RankOutcome& o = outs[0];
  DGPP_LOG_INFO("world 2 eager: A {} | B {} | C {}", ids_text(o.ea), ids_text(o.eb), ids_text(o.ec));
  DGPP_LOG_INFO("world 2 graph scalar A {}", ids_text(o.ga));
  DGPP_LOG_INFO("world 2 graph batched: A {} | B {} | C {}", ids_text(o.ba), ids_text(o.bb), ids_text(o.bc));
  // The world-2 eager engine against world 1: reported, the first tokens
  // must agree (the folds reassociate; a near tie may flip a late token).
  const size_t pa = agreeing_prefix(o.ea, ref.a), pb = agreeing_prefix(o.eb, ref.b), pc = agreeing_prefix(o.ec, ref.c);
  DGPP_LOG_INFO("world 2 vs world 1 agreeing prefixes: A {}/{} B {}/{} C {}/{}", pa, ref.a.size(), pb, ref.b.size(),
                pc, ref.c.size());
  require(pa >= 3 && pb >= 3 && pc >= 3, "the world-2 eager engine diverges from world 1 at the start");
  // The graph engine against the eager engine at the same world: exact.
  require(o.ga == o.ea, "the scalar graph transcript differs from the eager engine's");
  // 61 tokens for A (23 + 60: the batched rows cross the pool's 64-token
  // block), 46 for B, 36 for C — each a prefix of the eager transcript.
  require(o.ba == o.ea, "the batched graph transcript of A differs from the eager engine's");
  require(std::equal(o.bb.begin(), o.bb.end(), o.eb.begin()), "the batched graph transcript of B differs");
  require(std::equal(o.bc.begin(), o.bc.end(), o.ec.begin()), "the batched graph transcript of C differs");
  require(o.bb.size() == 46 && o.bc.size() == 36, "the batched transcripts' lengths");
  return o;
}

DGPP_TEST(glm_dsa_engines_loopback_world_2_graph_matches_eager) {
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config();
  const std::string dir = "glm_dsa_engine_fixture";
  glmdsafx::write_fixture(cfg, dir);
  const std::vector<int64_t> A = smoke_tokens(cfg, 23, 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> B = smoke_tokens(cfg, 17, 0xD1B54A32D192ED03ull);
  const std::vector<int64_t> C = smoke_tokens(cfg, 11, 0x2545F4914F6CDD1Dull);
  const Ref ref = world1_reference(cfg, dir, A, B, C);
  DGPP_LOG_INFO("world 1 eager: A {} | B {} | C {}", ids_text(ref.a), ids_text(ref.b), ids_text(ref.c));
  const RankOutcome replicated = graph_world(cfg, dir, A, B, C, ref, kPort);
  // engine.embed_sharding = vocab: the same world with each rank's rows of
  // the embedding summed by a recorded fold (the main path's and the
  // draft's) — eager and graph transcripts bitwise the replicated ones.
  dgpp::GlmDsaLayerStream::set_embed_vocab_sharded(true);
  const RankOutcome sharded = graph_world(cfg, dir, A, B, C, ref, static_cast<uint16_t>(kPort + 3));
  dgpp::GlmDsaLayerStream::set_embed_vocab_sharded(false);
  require(sharded.ea == replicated.ea && sharded.eb == replicated.eb && sharded.ec == replicated.ec,
          "the vocab-sharded embedding changes the eager transcripts");
  require(sharded.ga == replicated.ga && sharded.ba == replicated.ba && sharded.bb == replicated.bb &&
              sharded.bc == replicated.bc,
          "the vocab-sharded embedding changes the graph transcripts");
}

int main() { return dgpp::test::run_all(); }
