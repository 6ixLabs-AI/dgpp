// M6 Stage 1d: glm_gen_check — greedy generation end-to-end at real dims.
// The "it speaks" instrument: prompt token-ids in, generated token-ids out,
// through the exact serving path (resident TP forward + vocab-sharded head
// + the distributed greedy pick over the bus).
//
// MODES
//   world=1 (no --world): the local reference — full head, streaming
//     residency, plain argmax. No bus, no peers; the loop logic and the
//     w1 sequence land in the record before the fabric run.
//   world>1: the fabric TP shape — every rank a process, VocabSharded
//     head, Resident residency by default (the M6 serving contract;
//     --streaming keeps the M4 diagnostic loader), the per-step winner
//     through bus_greedy_pick (the same exact pick the CI gate pins:
//     glm_tp_greedy_gen_loopback). Every rank writes its tokens file;
//     the run record cross-checks them (md5) — the pick's rank
//     consistency is by construction (canonical order + broadcast),
//     the files are the audit trail.
//
// PROMPT: either raw text (--text "...", tokenized by the Stage 3 exact
// tokenizer, whose ids match HF tokenizers 0.23.1 byte-for-byte) or
// comma-separated token ids (--prompt "1,2,3"). Generated tokens decode
// through the same tokenizer; EOS stops the loop (config's eos_token_ids,
// --no-eos disables).
//
// SCHEDULER MODE (M6 Stage 2b): --requests FILE runs a manifest of
// CONCURRENT conversations through the deterministic scheduler (strict
// alternation, full-reserve admission, round-robin decode, scripted
// cancellation) — DESIGN §11's identical-rank-order contract, so all
// ranks issue the same ops in the same order and the fabric's
// collectives stay aligned. The manifest is JSONL, one object per line:
//   {"id": "a", "text": "...", "steps": 8}
//   {"id": "b", "chat": "...", "system": "...", "steps": 16,
//    "cancel_after": 8}
// Knobs (NInfer-shaped): --max-concurrency N (engine session slots;
// default min(8, requests) — 8 is the decode-row bound) and
// --kv-capacity TOKENS (the SHARED DSA pool; default: the sum of every
// request's reservation, so all fit immediately — shrink it to force
// budget deferral). --sched-plan prints the memory receipt and the
// admission forecast WITHOUT loading the model, then exits. The file
// must exist identically on every rank (scripts/fabric_run.sh
// --stage-file stages it); its FNV-1a-64 hash is logged per rank as the
// cross-rank identity check.
//
// T² IS DIAGNOSTIC HERE: every step re-forwards the whole sequence
// (fresh KDA/DSA state per call — GlmDiagnosticModel's contract). The
// incremental decode engine is Stage 2; do not read this app's per-step
// time as serving latency. EOS-stop is likewise Stage 2 policy (the
// sampler/service interface); this loop runs exactly --steps steps.
//
// ALLOCATION DISCIPLINE (the burst-wedge lesson, now a rule): every
// device allocation — the pick scratch included — happens BEFORE the
// world forms. Nothing allocates between collectives; the per-step
// host buffers are hoisted out of the loop.
#include <fcntl.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <functional>
#include <cmath>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/minijson.hpp"
#include "text/chat_template.hpp"
#include "models/glm/fabric_engine.hpp"
#include "models/glm/forward.hpp"
#include "models/glm/gen_engine.hpp"
#include "models/glm/graph_check.hpp"
#include "sample/sampler.hpp"
#include "sched/scheduler.hpp"
#include "models/glm/speculative.hpp"
#include "models/glm/step_timing.hpp"
#include "text/tokenizer.hpp"
#include "models/glm/tp_bus.hpp"
#include "net/collective_bus.hpp"

namespace fs = std::filesystem;
using dgpp::DsaConfig;
using dgpp::DsaGeometry;
using dgpp::DsaStatePool;
using dgpp::GenEngineAdapter;
using dgpp::GlmDiagnosticModel;
using dgpp::GlmTextConfig;
using dgpp::KdaConfig;
using dgpp::KdaGeometry;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// "Where did the calling thread go?" — a snapshot of the scheduler's and
// the VM's view of THIS thread: on-CPU time and runnable-but-waiting time
// (/proc/thread-self/schedstat), page faults and context switches
// (getrusage), and the core it runs on. Two snapshots bracket a code region
// that has no business taking milliseconds; their delta says whether the
// thread was preempted (wait grows), stuck in the kernel on its own behalf
// (run grows — direct reclaim, a slow syscall), paging (majflt), or moved.
// The 2026-09-02 hunt needed exactly this for the ~10 ms the peers lost
// between two log lines while every bus and GPU stamp read "fine".
class ThreadProbe {
 public:
  struct Sample {
    double run_ms = 0, wait_ms = 0;
    long majflt = 0, minflt = 0, nvcsw = 0, nivcsw = 0;
    int cpu = -1;
  };

  ThreadProbe() : fd_(::open("/proc/thread-self/schedstat", O_RDONLY)) {}
  ~ThreadProbe() {
    if (fd_ >= 0) ::close(fd_);
  }
  ThreadProbe(const ThreadProbe&) = delete;
  ThreadProbe& operator=(const ThreadProbe&) = delete;

  Sample sample() const {
    Sample s;
    // procfs regenerates the file on every read from offset 0: one pread on
    // the fd opened once, not open/read/close per step.
    char buf[96] = {};
    if (fd_ >= 0 && ::pread(fd_, buf, sizeof(buf) - 1, 0) > 0) {
      unsigned long long run_ns = 0, wait_ns = 0, slices = 0;
      if (std::sscanf(buf, "%llu %llu %llu", &run_ns, &wait_ns, &slices) >= 2) {
        s.run_ms = static_cast<double>(run_ns) / 1e6;
        s.wait_ms = static_cast<double>(wait_ns) / 1e6;
      }
    }
    rusage ru{};
    if (::getrusage(RUSAGE_THREAD, &ru) == 0) {
      s.majflt = ru.ru_majflt;
      s.minflt = ru.ru_minflt;
      s.nvcsw = ru.ru_nvcsw;
      s.nivcsw = ru.ru_nivcsw;
    }
    s.cpu = ::sched_getcpu();
    return s;
  }

  static std::string delta_text(const Sample& a, const Sample& b) {
    return std::format(
        "sched run {:.2f} wait {:.2f} majflt {} minflt {} nvcsw {} nivcsw {} "
        "cpu {}->{}",
        b.run_ms - a.run_ms, b.wait_ms - a.wait_ms, b.majflt - a.majflt,
        b.minflt - a.minflt, b.nvcsw - a.nvcsw, b.nivcsw - a.nivcsw, a.cpu,
        b.cpu);
  }

 private:
  int fd_;
};

// Real-mesh bus budgets + the fabric pick now live in the shared interface
// (models/glm_fabric_engine.hpp) — dgpp-serve and this app ride the
// same closure, so the pick path cannot drift between the smoke
// instrument and the serving deployment.

std::vector<int64_t> parse_prompt_ids(const std::string& text,
                                      int64_t vocab_size) {
  std::vector<int64_t> ids;
  std::string num;
  const auto flush = [&] {
    require(!num.empty(), "empty token id in --prompt");
    char* end = nullptr;
    const long long v = std::strtoll(num.c_str(), &end, 10);
    require(end && *end == '\0', "bad token id in --prompt: " + num);
    require(v >= 0 && v < vocab_size,
            "token id out of range [0, vocab): " + num);
    ids.push_back(static_cast<int64_t>(v));
    num.clear();
  };
  for (const char c : text) {
    if (c == ',' || c == ' ') flush();
    else num.push_back(c);
  }
  flush();
  require(!ids.empty(), "--prompt produced no token ids");
  return ids;
}

void write_tokens_file(const std::string& path, int rank, int world,
                       const std::vector<int64_t>& prompt,
                       const std::vector<int64_t>& generated) {
  std::string txt = "rank " + std::to_string(rank) + " world " +
                    std::to_string(world) + "\nprompt";
  for (int64_t t : prompt) txt += " " + std::to_string(t);
  txt += "\ngenerated";
  for (int64_t t : generated) txt += " " + std::to_string(t);
  txt += "\n";
  std::FILE* f = std::fopen(path.c_str(), "wb");
  require(f != nullptr, "cannot write " + path);
  const size_t n = std::fwrite(txt.data(), 1, txt.size(), f);
  std::fclose(f);
  require(n == txt.size(), "short write to " + path);
}

std::string ids_line(const std::vector<int64_t>& ids) {
  std::string s;
  for (int64_t t : ids) {
    if (!s.empty()) s += ",";
    s += std::to_string(t);
  }
  return s;
}

// ---------------------------------------------------------------------------
// Scheduler mode (M6 Stage 2b)
// ---------------------------------------------------------------------------

uint64_t fnv1a64(const std::string& bytes) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : bytes) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

// Parses the manifest from its bytes (read once — the FNV hash logged
// across ranks must cover exactly the bytes that were parsed). One JSON
// object per line ('#' comments and blanks skipped); throws with the
// line number on any malformed entry. The same bytes must reach every
// rank (fabric_run.sh --stage-file stages it).
std::vector<dgpp::sched::SchedulerRequest> parse_manifest(
    const std::string& manifest, const std::string& display_path,
    int default_steps, const dgpp::text::Tokenizer& tok, const fs::path& ckpt) {
  std::vector<dgpp::sched::SchedulerRequest> requests;
  std::unique_ptr<dgpp::text::ChatTemplate> chat_tpl;
  std::istringstream in(manifest);
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    const auto fail = [&](const std::string& what) {
      throw std::runtime_error("--requests " + display_path + " line " +
                               std::to_string(line_no) + ": " + what);
    };
    const size_t first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    dgpp::minijson::ParseResult parsed;
    try {
      parsed = dgpp::minijson::parse(line);
    } catch (const std::exception& e) {
      fail(std::string("JSON: ") + e.what());
    }
    const dgpp::minijson::Value& v = parsed.root;
    if (!v.is_object()) fail("expected a JSON object");
    const dgpp::minijson::Value& id = v.at("id");
    if (!id.is_string() || id.as_string().empty())
      fail("id must be a non-empty string");
    const std::string rid(id.as_string());

    const dgpp::minijson::Value* text = v.find("text");
    const dgpp::minijson::Value* chat = v.find("chat");
    const dgpp::minijson::Value* system = v.find("system");
    if ((text != nullptr) == (chat != nullptr))
      fail("exactly one of text|chat is required");
    if (system != nullptr && chat == nullptr)
      fail("system requires chat");

    int steps = default_steps;
    if (const dgpp::minijson::Value* s = v.find("steps")) {
      steps = static_cast<int>(s->as_int(0));
      if (steps < 1) fail("steps must be >= 1");
    }
    int cancel_after = 0;
    if (const dgpp::minijson::Value* c = v.find("cancel_after")) {
      cancel_after = static_cast<int>(c->as_int(0));
      if (cancel_after < 0) fail("cancel_after must be >= 0");
      if (cancel_after > steps)
        fail("cancel_after " + std::to_string(cancel_after) +
             " can never fire (steps " + std::to_string(steps) + ")");
    }

    // The same prompt builders as the single-request path.
    std::vector<int64_t> prompt;
    if (chat != nullptr) {
      if (!chat_tpl)
        chat_tpl = std::make_unique<dgpp::text::ChatTemplate>(
            dgpp::text::ChatTemplate::load(
                (ckpt / "chat_template.jinja").string()));
      std::vector<dgpp::text::Value> messages;
      if (system != nullptr && !system->as_string().empty()) {
        dgpp::text::Value::Members sys_msg;
        sys_msg.emplace_back("role",
                             dgpp::text::Value::string_value("system"));
        sys_msg.emplace_back(
            "content", dgpp::text::Value::string_value(
                           std::string(system->as_string())));
        messages.push_back(dgpp::text::Value::map_value(std::move(sys_msg)));
      }
      dgpp::text::Value::Members user_msg;
      user_msg.emplace_back("role", dgpp::text::Value::string_value("user"));
      user_msg.emplace_back(
          "content",
          dgpp::text::Value::string_value(std::string(chat->as_string())));
      messages.push_back(dgpp::text::Value::map_value(std::move(user_msg)));
      dgpp::text::Value::Members globals;
      globals.emplace_back(
          "messages", dgpp::text::Value::list_value(std::move(messages)));
      globals.emplace_back("add_generation_prompt",
                           dgpp::text::Value::boolean(true));
      const std::string rendered = chat_tpl->render(
          dgpp::text::Value::map_value(std::move(globals)));
      prompt = tok.encode(rendered);
      DGPP_LOG_INFO("request '{}': chat template {} rendered to {} ids",
                    rid, chat_tpl->source_hash(), prompt.size());
    } else {
      prompt = tok.encode(std::string(text->as_string()));
      DGPP_LOG_INFO("request '{}': text encoded to {} ids by the exact "
                    "tokenizer",
                    rid, prompt.size());
    }
    if (prompt.empty()) fail("request '" + rid + "' produced no tokens");

    dgpp::sched::SchedulerRequest r;
    r.id = rid;
    r.prompt = std::move(prompt);
    r.max_steps = steps;
    r.cancel_after = cancel_after;
    requests.push_back(std::move(r));
  }
  require(!requests.empty(), "--requests produced no requests");
  return requests;
}

std::string read_whole_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  require(static_cast<bool>(f), "cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Blocks covering `tokens` tokens — the same arithmetic as the model's
// pool (DsaStatePool::block_count_for_tokens); standalone so the plan
// paths can run without a device.
int64_t model_blocks_for(int64_t tokens, int64_t block_tokens) {
  return (tokens + block_tokens - 1) / block_tokens;
}

// The shared-pool sizing every mode agrees on (mirrors GlmDiagnosticModel's
// own arithmetic: tp-sliced sub-configs, block-rounded capacity).
struct SchedSizing {
  int64_t pool_tokens = 0;  // block-rounded token capacity
  int64_t blocks_total = 0;
  int64_t block_tokens = 0;
};

SchedSizing sched_sizing(const GlmTextConfig& cfg, int world,
                         int64_t kv_capacity) {
  DsaConfig dsa = cfg.dsa_config();
  dsa.tp_size = world;
  const int64_t bt = dsa.block_tokens;
  SchedSizing z;
  z.block_tokens = bt;
  z.pool_tokens = ((kv_capacity + bt - 1) / bt) * bt;
  z.blocks_total = z.pool_tokens / bt;
  const int64_t pools =
      z.blocks_total * DsaGeometry::from_config(dsa).pools_per_block;
  // The selection kernels pack pool ids into 21 composite-key bits —
  // DsaStatePool::init refuses this shape; the plan refuses it EARLIER.
  if (pools >= (int64_t(1) << 21))
    throw std::runtime_error(
        "--kv-capacity " + std::to_string(kv_capacity) +
        " exceeds the DSA pool-id space (2^21 pools) — shrink it");
  return z;
}

// The admission policy for the manifest runs and the forecast (M6 6d):
// --admission full|grow, --admission-window N.
dgpp::sched::AdmissionPolicy g_admission;

// The bulk all-reduce bench (2026-09-04, the prefill fold's cost): the
// world forms, no model — bf16 all-reduces of 2, 4, .. g_bulk_bench_mb MiB,
// g_bulk_bench_iters each, timed submit-to-completion; the pool overrides
// size the experiment. --bulk-bench MIB [--bulk-bench-iters N]
// [--bulk-slots N] [--bulk-slot-bytes B].
int g_bulk_bench_mb = 0, g_bulk_bench_iters = 10, g_bulk_slots_override = 0;
int g_bulk_inflight_override = -1;
double g_bulk_pace_override = -1.0;
int64_t g_bulk_slot_bytes_override = 0;
// --prefill-repeat N: prefill the prompt N times (the slot reopens each
// time) and log each — the first pays the process's one-time setup (the
// lazily built layer objects, their scratch, the GEMM plans), the later
// ones are the steady state the service sees after its warm-up.
int g_prefill_repeat = 1;
// --gr-probe N (Q0, 2026-09-09): one extra collective after the attention
// fold of the first N layers of every decode row — the sliced-GR cost probe
// (GlmBoundaryReducer::probe). Bounded by the graph's node budget:
// kBusMaxGraphGens = 128 less the step's own ~92 nodes.
int g_gr_probe_layers = 0;
// --group-check-b: the second prompt of the grouped-prefill site check (below).
static std::string g_group_check_b;
static std::vector<int64_t> g_group_check_b_tokens;

// The memory receipt: exactly what the model pre-allocates for this knob
// combination, by region, plus the per-request reserve math. Runs with or
// without a GPU (--sched-plan uses it before any device work).
void print_memory_receipt(const GlmTextConfig& cfg, int world,
                          int max_requests, const SchedSizing& z,
                          const std::vector<dgpp::sched::SchedulerRequest>& reqs,
                          bool with_forecast) {
  DsaConfig dsa = cfg.dsa_config();
  dsa.tp_size = world;
  KdaConfig kda = cfg.kda_config();
  kda.tp_size = world;
  const DsaGeometry geo = DsaGeometry::from_config(dsa);
  const KdaGeometry kgeo = KdaGeometry::from_config(kda);
  const int64_t pools = z.blocks_total * geo.pools_per_block;
  const int L = dsa.num_dsa_layers;
  const size_t latent =
      static_cast<size_t>(L) * static_cast<size_t>(z.pool_tokens) *
      geo.latent_bytes_per_token;
  const size_t index_k = static_cast<size_t>(L) *
                         static_cast<size_t>(pools) *
                         geo.index_k_bytes_per_pool;
  const size_t index_scale =
      static_cast<size_t>(L) * static_cast<size_t>(pools) * sizeof(float);
  const size_t tails = static_cast<size_t>(L) *
                       static_cast<size_t>(max_requests) *
                       geo.tail_bytes_per_request;
  const size_t tables = static_cast<size_t>(max_requests) *
                        static_cast<size_t>(z.blocks_total) * sizeof(int32_t);
  const size_t pool_total =
      DsaStatePool::cache_bytes(dsa, max_requests, z.pool_tokens);
  const size_t kda_state =
      static_cast<size_t>(kda.num_kda_layers) *
      static_cast<size_t>(max_requests) *
      (kgeo.recurrent_bytes + kgeo.conv_committed_bytes);

  DGPP_LOG_INFO(
      "sched receipt: pool {} tokens ({} blocks of {}), max_concurrency {}",
      z.pool_tokens, z.blocks_total, z.block_tokens, max_requests);
  DGPP_LOG_INFO(
      "sched receipt: DSA pool {:.1f} MiB = latent {:.1f} + index_k {:.1f} "
      "+ index_scale {:.1f} + tails {:.1f} + tables {:.1f}",
      pool_total / 1048576.0, latent / 1048576.0, index_k / 1048576.0,
      index_scale / 1048576.0, tails / 1048576.0, tables / 1048576.0);
  DGPP_LOG_INFO("sched receipt: KDA state {:.1f} MiB ({} slots x {} layers)",
                kda_state / 1048576.0, max_requests, kda.num_kda_layers);
  DGPP_LOG_INFO("sched receipt: session reserve = blocks_for(prompt + "
                "steps), block {} tokens — held for the request's lifetime",
                z.block_tokens);
  if (with_forecast) {
    int64_t held = 0;
    int used_slots = 0, admitted = 0, deferred = 0;
    for (const auto& r : reqs) {
      const int64_t reserve = model_blocks_for(
          static_cast<int64_t>(r.prompt.size()) + r.max_steps, z.block_tokens);
      if (reserve > z.blocks_total)
        throw std::runtime_error(
            "request '" + r.id + "' reserves " + std::to_string(reserve) +
            " blocks against a " + std::to_string(z.blocks_total) +
            "-block pool — it can NEVER fit (raise --kv-capacity or lower "
            "its steps)");
      if (used_slots < max_requests && held + reserve <= z.blocks_total) {
        held += reserve;
        ++used_slots;
        ++admitted;
      } else {
        ++deferred;
      }
    }
    DGPP_LOG_INFO(
        "sched plan: {} of {} requests admitted at start ({} blocks "
        "reserved of {}), {} deferred until peers retire",
        admitted, reqs.size(), held, z.blocks_total, deferred);
    // The same forecast under grow-on-demand: the initial reservation is
    // prompt + min(steps, window) (a plain decode's width plus one token
    // at least); growth and the shed rule take over at tick top.
    const int64_t window = std::max<int64_t>(g_admission.window_tokens, 2);
    int64_t held_grow = 0;
    int used_grow = 0, admitted_grow = 0, deferred_grow = 0;
    for (const auto& r : reqs) {
      const int64_t reserve = model_blocks_for(
          static_cast<int64_t>(r.prompt.size()) +
              std::min<int64_t>(r.max_steps, window),
          z.block_tokens);
      if (used_grow < max_requests && held_grow + reserve <= z.blocks_total) {
        held_grow += reserve;
        ++used_grow;
        ++admitted_grow;
      } else {
        ++deferred_grow;
      }
    }
    DGPP_LOG_INFO(
        "sched plan (grow-on-demand, window {}): {} of {} requests admitted "
        "at start ({} blocks reserved of {}), {} deferred; reservations grow "
        "at tick top and the youngest sheds (finish length) at exhaustion{}",
        window, admitted_grow, reqs.size(), held_grow, z.blocks_total,
        deferred_grow,
        g_admission.mode == dgpp::sched::AdmissionPolicy::Mode::kGrowOnDemand
            ? " — THIS RUN's policy"
            : " — not this run's policy (--admission grow selects it)");
  }
}

// The engine binding lives in src/models/glm_gen_engine.hpp (shared
// with the Stage 4 serving app): GenEngineAdapter + the w1/fabric
// picks. Both worlds keep their local pick closures below.

// The process's sampling mode (M6 6b). Absent params: the exact greedy
// loop every gate pins — this diagnostic does not apply the checkpoint's
// stochastic defaults on its own (dgpp-serve does); --sample or any
// override opts in, at the model's defaults with the overrides applied.
// `seed` is the fixed seed (request i of a manifest draws from seed + i).
struct SamplingRun {
  std::optional<dgpp::sample::Params> params;
  uint64_t seed = 0;
  bool on() const { return params.has_value(); }
};

// The host pick — one driver for every loop that picks on the host (PLAN
// M8's FabricPicker interface, 2026-09-05): world 1's full-head argmax or exact
// sampler and the fabric's bus merge or bus sampler behind the same
// greedy / sampling branch, the same RNG, the same log lines (the ones
// scripts/fabric_xrank.py and fabric_xcript.py read). The plain loops use
// it at every step, eager or under the T=1 graph (the pick rides between
// windows); the speculative loop uses it for its prefill pick and its
// eager row picks. The in-graph pick is DevicePicker (a recorded node)
// — the other driver, by nature.
class HostPick {
 public:
  HostPick(int world, int rank, CollectiveBus* bus, uint16_t* pick_scratch,
           GenEngineAdapter::Sample sample, const SamplingRun& sampling,
           int64_t vocab)
      : world_(world), rank_(rank), bus_(bus), scratch_(pick_scratch),
        sample_(std::move(sample)), sampling_(sampling), vocab_(vocab),
        rng_{sampling.seed, 0} {}

  // One pick over `o`'s (last) row: sampled at the run's settings with
  // `context` (the penalties' count table) or greedy; `what` and `s` name
  // the step in the log. The Candidate's logit is the winner's when this
  // rank holds it (world 1 always), else -inf.
  dgpp::sample::Candidate pick(const GlmDiagnosticModel::Outputs& o,
                                   const char* what, int s,
                                   const std::vector<int32_t>& context) {
    if (sampling_.on()) {
      const dgpp::sample::Result r =
          sample_(o, *sampling_.params, rng_, context, nullptr, nullptr);
      if (world_ == 1) {
        DGPP_LOG_INFO("[gen] sampled token {} logprob {:.4f} (counter {})",
                      r.token, r.logprob, rng_.counter);
        return {r.token, o.logits[static_cast<size_t>(r.token)]};
      }
      DGPP_LOG_INFO(
          "[gen] rank {} {} {}: sampled token {} logprob {:.4f} (counter "
          "{}; local slice [{},{}))",
          rank_, what, s, r.token, r.logprob, rng_.counter,
          o.lm_vocab_begin, o.lm_vocab_begin + o.lm_vocab_count);
      return {r.token, -INFINITY};
    }
    const std::vector<float>& fslice = o.logits;
    const dgpp::sample::Candidate local = dgpp::sample::local_max(
        fslice.data(), o.lm_vocab_count, o.lm_vocab_begin);
    if (world_ == 1) return local;  // the full head: the argmax is the pick
    // The slice's runner-up: with the four ranks' lines side by side
    // (fabric_xrank), the global top-2 margin of every pick follows, and
    // a transcript that diverges between two builds can be judged — a
    // near-tie flip is rounding, a wide-margin flip is a bug.
    float second = -INFINITY;
    for (int i = 0; i < o.lm_vocab_count; ++i)
      if (o.lm_vocab_begin + i != local.id && fslice[i] > second)
        second = fslice[i];
    const int32_t token =
        dgpp::bus_greedy_pick(*bus_, rank_, world_, local, scratch_, 60000);
    require(token >= 0 && token < vocab_,
            "generated id out of range: " + std::to_string(token));
    DGPP_LOG_INFO(
        "[gen] rank {} {} {}: token {} (local slice [{},{}) best "
        "{} logit {:.4f} second {:.4f})",
        rank_, what, s, token, o.lm_vocab_begin,
        o.lm_vocab_begin + o.lm_vocab_count, local.id, local.logit, second);
    return {token, token == local.id ? local.logit : -INFINITY};
  }
  // Greedy picks over several rows' local maxima (the speculative loop's
  // prefill and eager row picks): the same bus merge, one collective.
  std::vector<int32_t> pick_rows(
      const std::vector<dgpp::sample::Candidate>& locals) {
    if (world_ == 1) {
      std::vector<int32_t> out;
      for (const dgpp::sample::Candidate& c : locals) out.push_back(c.id);
      return out;
    }
    return dgpp::bus_greedy_pick_rows(*bus_, rank_, world_, locals, scratch_,
                                      60000);
  }

 private:
  int world_;
  int rank_;
  CollectiveBus* bus_;
  uint16_t* scratch_;
  GenEngineAdapter::Sample sample_;
  SamplingRun sampling_;
  int64_t vocab_;
  dgpp::sample::Rng rng_;
};

// Pinned words for the sampler's two collectives (the candidate/LSE fold
// and the fallback gather), allocated before the world forms.
struct PinnedWords {
  uint16_t* data = nullptr;
  explicit PinnedWords(size_t elems) {
    DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&data),
                               sizeof(uint16_t) * std::max<size_t>(elems, 2),
                               cudaHostAllocDefault));
  }
  ~PinnedWords() {
    if (data) cudaFreeHost(data);
  }
  PinnedWords(const PinnedWords&) = delete;
  PinnedWords& operator=(const PinnedWords&) = delete;
};

// --group-check-b: the site-by-site view of a grouped prefill against the
// prefills alone on this rank (GlmDiagnosticModel::set_walk_capture): the
// --text/--chat prompt A and prompt B alone, then as the spans of one walk.
// Every span's rows after the embedding and after every attention and FFN
// stream update, and its last-row logits, must be bitwise. Returns 0 when
// they are; logs the first differing sites otherwise.
static int run_group_check(GlmDiagnosticModel& model, int rank, const std::vector<int64_t>& A,
                           const std::vector<int64_t>& B) {
  // Greedy eager steps off each prefill: the step-by-step view of the
  // request's state (KV, recurrent, index caches) after a solo and a group
  // prefill. The argmax over this rank's vocab rows is enough to drive
  // identical steps on every rank as long as they stay bitwise.
  constexpr int kSteps = 12;
  const auto local_argmax = [](const std::vector<float>& row) {
    return static_cast<int64_t>(std::max_element(row.begin(), row.end()) - row.begin());
  };
  const auto steps_from = [&](int req, const GlmDiagnosticModel::Outputs& first) {
    std::vector<std::vector<float>> rows;
    int64_t tok = local_argmax(first.logits);
    for (int s = 0; s < kSteps; ++s) {
      const GlmDiagnosticModel::Outputs o = model.session_step(req, tok);
      rows.push_back(o.logits);
      tok = local_argmax(o.logits);
    }
    return rows;
  };
  std::vector<std::vector<uint16_t>> ca, cb, cg;
  model.set_walk_capture(&ca);
  const GlmDiagnosticModel::Outputs pa = model.session_prefill(0, A);
  model.set_walk_capture(nullptr);
  const auto steps_a = steps_from(0, pa);
  model.session_close(0);
  model.set_walk_capture(&cb);
  const GlmDiagnosticModel::Outputs pb = model.session_prefill(1, B);
  model.set_walk_capture(nullptr);
  const auto steps_b = steps_from(1, pb);
  model.session_close(1);
  model.set_walk_capture(&cg);
  const std::vector<GlmDiagnosticModel::Outputs> g = model.session_prefill_group({0, 1}, {&A, &B});
  model.set_walk_capture(nullptr);
  const auto gsteps_a = steps_from(0, g[0]);
  const auto gsteps_b = steps_from(1, g[1]);
  model.session_close(0);
  model.session_close(1);
  int first_step_bad_a = -1, first_step_bad_b = -1;
  for (int s = 0; s < kSteps; ++s) {
    if (first_step_bad_a < 0 && gsteps_a[static_cast<size_t>(s)] != steps_a[static_cast<size_t>(s)]) first_step_bad_a = s;
    if (first_step_bad_b < 0 && gsteps_b[static_cast<size_t>(s)] != steps_b[static_cast<size_t>(s)]) first_step_bad_b = s;
  }
  DGPP_LOG_INFO("rank {} group check: {} eager greedy steps off the group's cache vs the solo cache: span A {}, span B {}",
                rank, kSteps, first_step_bad_a < 0 ? std::string("bitwise") : "first differing step " + std::to_string(first_step_bad_a),
                first_step_bad_b < 0 ? std::string("bitwise") : "first differing step " + std::to_string(first_step_bad_b));
  const size_t la = A.size(), lb = B.size();
  if (ca.size() != cg.size() || cb.size() != cg.size() || cg.empty()) {
    DGPP_LOG_ERROR("rank {} group check: capture counts {} / {} / {}", rank, ca.size(), cb.size(), cg.size());
    return 2;
  }
  const size_t elems = cg[0].size() / (la + lb);
  const auto bf = [](uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  int bad_sites = 0;
  for (size_t s = 0; s < cg.size(); ++s) {
    for (int span = 0; span < 2; ++span) {
      const std::vector<uint16_t>& solo = span == 0 ? ca[s] : cb[s];
      const size_t rows = span == 0 ? la : lb, off = span == 0 ? 0 : la * elems;
      size_t n = 0, first_row = rows, last_row = 0;
      float worst = 0.f;
      for (size_t r = 0; r < rows; ++r)
        for (size_t i = 0; i < elems; ++i) {
          const uint16_t x = cg[s][off + r * elems + i], y = solo[r * elems + i];
          if (x != y) {
            ++n;
            first_row = std::min(first_row, r);
            last_row = std::max(last_row, r);
            worst = std::max(worst, std::abs(bf(x) - bf(y)));
          }
        }
      if (n == 0) continue;
      if (bad_sites < 8)
        DGPP_LOG_INFO("rank {} group check: site {} ({}, layer {}) span {}: {} of {} elements differ, rows {}..{} of {}, max abs {:.3g}",
                      rank, s, s == 0 ? "embedding" : ((s - 1) % 2 ? "FFN" : "attention"),
                      s == 0 ? 0 : (s - 1) / 2, span == 0 ? "A" : "B", n, rows * elems, first_row, last_row,
                      rows, worst);
      ++bad_sites;
    }
  }
  const auto logit_diff = [](const std::vector<float>& x, const std::vector<float>& y) {
    size_t n = 0;
    float worst = 0.f;
    for (size_t i = 0; i < x.size() && i < y.size(); ++i)
      if (x[i] != y[i]) {
        ++n;
        worst = std::max(worst, std::abs(x[i] - y[i]));
      }
    return std::make_pair(n, worst);
  };
  const auto da = logit_diff(g[0].logits, pa.logits), db = logit_diff(g[1].logits, pb.logits);
  DGPP_LOG_INFO("rank {} group check: {} + {} rows, {} sites, {} span-sites with differences; last-row logits "
                "(this rank's {} vocab rows): A {} differ (max abs {:.3g}), B {} differ (max abs {:.3g})",
                rank, la, lb, cg.size(), bad_sites, g[0].logits.size(), da.first, da.second, db.first, db.second);
  return (bad_sites == 0 && da.first == 0 && db.first == 0 && first_step_bad_a < 0 && first_step_bad_b < 0) ? 0 : 2;
}

const char* sched_status_name(const dgpp::sched::Scheduler::Result& r) {
  using S = dgpp::sched::Scheduler::Result::Status;
  switch (r.status) {
    case S::kDone: return "done";
    case S::kCancelled: return "cancelled";
    case S::kActive: return "active";
    default: return "queued";
  }
}

// Runs the whole manifest through the scheduler. Both worlds share the
// body; only the pick differs (full-head argmax vs the bus merge).
int run_scheduler(const GlmTextConfig& cfg, const std::string& ckpt,
                  int world, int rank, uint16_t port, const std::string& peer,
                  std::vector<dgpp::sched::SchedulerRequest> requests,
                  int max_requests, int64_t kv_capacity, bool resident,
                  bool no_eos, const dgpp::text::Tokenizer& tok,
                  const std::string& out_prefix, int rendezvous_timeout_ms,
                  int64_t vocab, uint64_t manifest_hash,
                  const SamplingRun& sampling) {
  const SchedSizing z = sched_sizing(cfg, world, kv_capacity);
  // Every manifest request takes the process's sampling mode; request i
  // draws from seed + i so the transcripts are reproducible per request.
  for (size_t i = 0; i < requests.size(); ++i) {
    requests[i].sampling = sampling.on() ? *sampling.params
                                         : dgpp::sample::greedy_params();
    requests[i].seed = sampling.seed + i;
  }
  for (const auto& r : requests) {
    const int64_t reserve =
        model_blocks_for(static_cast<int64_t>(r.prompt.size()) + r.max_steps,
                         z.block_tokens);
    if (reserve > z.blocks_total)
      throw std::runtime_error(
          "request '" + r.id + "' reserves " + std::to_string(reserve) +
          " blocks against a " + std::to_string(z.blocks_total) +
          "-block pool — raise --kv-capacity or lower its steps");
  }
  int max_tokens = 0;
  for (const auto& r : requests)
    max_tokens = std::max(max_tokens,
                          static_cast<int>(r.prompt.size()) + r.max_steps + 1);
  std::vector<int64_t> eos =
      no_eos ? std::vector<int64_t>{} : cfg.eos_token_ids;

  const auto log_results = [&](dgpp::sched::Scheduler& sched) {
    for (size_t i = 0; i < requests.size(); ++i) {
      const auto& spec = requests[i];
      const auto& res = sched.results()[i];
      write_tokens_file(out_prefix + "." + spec.id + ".tokens.txt", rank,
                        world, spec.prompt, res.generated);
      std::string text;
      for (int64_t t : res.generated)
        text += tok.decode(t, /*skip_special_tokens=*/false);
      DGPP_LOG_INFO("rank {} request {} generated ids: {}", rank, spec.id,
                    ids_line(res.generated));
      DGPP_LOG_INFO("rank {} request {} generated text: {}", rank, spec.id,
                    text);
      DGPP_LOG_INFO("rank {} request {} finished: {} ({} tokens)", rank,
                    spec.id, sched_status_name(res), res.steps_done);
    }
  };

  // ---- world 1: no bus, full head, plain argmax pick ------------------
  if (world == 1) {
    dgpp::prepare_serving_process(rank);
    const auto t_construct = std::chrono::steady_clock::now();
    GlmDiagnosticModel model(cfg, ckpt, max_tokens, z.pool_tokens,
                             /*boundary=*/nullptr, /*tp_rank=*/0,
                             /*tp_world=*/1, dgpp::GlmResidency::Streaming,
                             dgpp::GlmHeadSharding::Full, max_requests);
    DGPP_LOG_INFO(
        "w1 model constructed in {:.1f}s (streaming, {} request slots)",
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      t_construct)
                .count(),
        max_requests);
    // The shared w1 pick (full-vocab argmax; the float row is hoisted
    // inside the closure — no per-token allocation).
    GenEngineAdapter engine(&model, max_requests, dgpp::make_w1_pick(vocab),
                            dgpp::make_w1_sample(vocab));
    dgpp::sched::Scheduler sched(&engine, eos, 0, g_admission);
    // Submit COPIES: the manifest entries stay intact for the audit
    // trail below (log_results reads spec.id/spec.prompt AFTER the run —
    // moving into the scheduler would leave husks there).
    for (const auto& r : requests) {
      dgpp::sched::SchedulerRequest copy = r;
      sched.submit(std::move(copy));
    }
    const auto t0 = std::chrono::steady_clock::now();
    sched.run_to_completion();
    DGPP_LOG_INFO(
        "w1 scheduler: {} requests in {:.1f}ms", requests.size(),
                  std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                      .count());
    log_results(sched);
    return 0;
  }

  // ---- fabric TP: bus, sharded head, resident by default ---------------
  uint16_t* pick_scratch = nullptr;
  // Pinned (see the fabric-path note below): no UVM residency dependence
  // on the decode path, no migration ping-pong per pick.
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&pick_scratch),
                              sizeof(uint16_t) * dgpp::kPickScratchElems(world),
                              cudaHostAllocDefault));
  PinnedWords sample_prefix(dgpp::fabric_sampling_prefix_scratch_elems(world));
  PinnedWords sample_gather(dgpp::sampling_gather_scratch_elems(vocab));
  std::unique_ptr<CollectiveBus> bus;
  try {
    bus = std::make_unique<CollectiveBus>(dgpp::fabric_bus_options(
        rank, world, port, peer, rendezvous_timeout_ms));
    std::string err;
    if (!bus->start(&err))
      throw std::runtime_error("rank " + std::to_string(rank) +
                              " bus start: " + err);
    dgpp::GlmBusBoundaryReducer reducer(*bus);
    dgpp::prepare_serving_process(rank);
    const auto t_construct = std::chrono::steady_clock::now();
    GlmDiagnosticModel model(
        cfg, ckpt, max_tokens, z.pool_tokens, &reducer, rank, world,
        resident ? dgpp::GlmResidency::Resident : dgpp::GlmResidency::Streaming,
        dgpp::GlmHeadSharding::VocabSharded, max_requests);
    DGPP_LOG_INFO(
        "rank {} model constructed in {:.1f}s ({}, {} request slots)", rank,
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       t_construct)
                .count(),
        resident ? "resident" : "streaming", max_requests);

    // The shared fabric pick (glm_fabric_engine.hpp) — the same
    // closure dgpp-serve runs in production; this app's runs are its
    // regression gate.
    GenEngineAdapter engine(&model, max_requests,
                            dgpp::make_fabric_pick(bus.get(), rank, world,
                                                   pick_scratch, vocab),
                            dgpp::make_fabric_sample(
                                bus.get(), rank, world, sample_prefix.data,
                                sample_gather.data, vocab));
    dgpp::sched::Scheduler sched(&engine, eos, 0, g_admission);
    // Submit COPIES (the manifest entries stay intact for the audit
    // trail below — log_results reads spec.id/spec.prompt after the run).
    for (const auto& r : requests) {
      dgpp::sched::SchedulerRequest copy = r;
      sched.submit(std::move(copy));
    }
    const auto t0 = std::chrono::steady_clock::now();
    sched.run_to_completion();
    DGPP_LOG_INFO("rank {} scheduler: {} requests (manifest {:016x}) in "
                  "{:.1f}ms",
                  rank, requests.size(), manifest_hash,
                  std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                      .count());
    log_results(sched);
    dgpp::step_timing::report(("rank " + std::to_string(rank) +
                               " scheduler mode (prefill+decode mixed)")
                                  .c_str());
    const auto stats = bus->stats();
    DGPP_LOG_INFO("rank {}: lat {} bulk {} collectives served",
                  rank, stats.latency.latency_us.size(),
                  stats.bulk.latency_us.size());
    bus->stop();
  } catch (...) {
    if (pick_scratch) cudaFreeHost(pick_scratch);
    throw;
  }
  cudaFreeHost(pick_scratch);
  return 0;
}

// Teacher forcing (--teacher-file): the step's logits are scored against
// the KNOWN next token instead of driving the pick. Each rank logs what it
// can compute from its slice — its max, its log-sum-exp (double, over the
// bf16 logits as the sampler would see them), and the target's logit when
// the target lives in the slice; scripts/fabric_logprob.py joins the ranks
// (logaddexp over the slices' lse) into log p(target) per position and a
// perplexity. That is the numerics gate for reassociated kernels: a
// transcript md5 flips on a 1-ulp tie and says nothing about magnitude;
// the mean NLL over a fixed text says exactly how far the distribution
// moved. World 1 logs the same line over the full head.
double log_teacher_stats(int rank, int step,
                         const GlmDiagnosticModel::Outputs& out,
                         int64_t target, int32_t argmax) {
  const std::vector<float>& slice = out.logits;
  const double lse =
      dgpp::sample::slice_logsumexp(slice.data(), out.lm_vocab_count);
  double lmax = -INFINITY;
  for (int i = 0; i < out.lm_vocab_count; ++i)
    lmax = std::max(lmax,
                    static_cast<double>(slice[static_cast<size_t>(i)]));
  const int64_t local = target - out.lm_vocab_begin;
  const bool in_slice = local >= 0 && local < out.lm_vocab_count;
  DGPP_LOG_INFO("[tf] rank {} step {}: target {} argmax {} lmax {:.4f} lse "
                "{:.6f} target_logit {}",
                rank, step, target, argmax, lmax, lse,
                in_slice ? std::format("{:.4f}",
                                       slice[static_cast<size_t>(local)])
                         : std::string("nan"));
  return lse;
}

// The exact sizing evidence for M6 6b. A point is global (every rank gets
// the same candidate table), but every rank logs it so the analysis script
// can make rank consistency part of the measurement rather than an
// assumption. At top_p=0.95, mass below 0.95 means the candidate prefix
// cannot contain the nucleus cut and that position must fall back.
class SamplingMassSummary {
 public:
  static constexpr double kTopP = 0.95;

  void add(int rank, int step,
           const std::array<double, dgpp::kSamplingProfileTopKs.size()>&
               masses) {
    DGPP_LOG_INFO(
        "[sample_mass] rank {} step {}: k32 {:.17f} k64 {:.17f} k128 "
        "{:.17f} k256 {:.17f}",
        rank, step, masses[0], masses[1], masses[2], masses[3]);
    for (size_t i = 0; i < masses.size(); ++i) {
      require(std::isfinite(masses[i]) && masses[i] >= 0.0 &&
                  masses[i] <= 1.0 + 1e-9,
              "sampling profile mass outside [0,1]");
      if (i > 0)
        require(masses[i] + 1e-15 >= masses[i - 1],
                "sampling profile mass is not monotonic");
      values_[i].push_back(masses[i]);
    }
  }

  void report(int rank) const {
    require(!values_[0].empty(), "sampling profile has no positions");
    for (size_t i = 0; i < values_.size(); ++i) {
      std::vector<double> sorted = values_[i];
      std::sort(sorted.begin(), sorted.end());
      double sum = 0.0;
      int fallbacks = 0;
      for (double mass : sorted) {
        sum += mass;
        fallbacks += mass < kTopP;
      }
      const auto percentile = [&](double p) {
        const size_t at = std::min(
            sorted.size() - 1,
            static_cast<size_t>(p * static_cast<double>(sorted.size())));
        return sorted[at];
      };
      DGPP_LOG_INFO(
          "[sample_mass_summary] rank {}: T=1 top_p={:.2f} k={} "
          "positions={} mass min/mean/p50/p99={:.6f}/{:.6f}/{:.6f}/"
          "{:.6f} fallback={}/{} ({:.3f}%)",
          rank, kTopP, dgpp::kSamplingProfileTopKs[i], sorted.size(),
          sorted.front(), sum / sorted.size(), percentile(0.50),
          percentile(0.99), fallbacks, sorted.size(),
          100.0 * static_cast<double>(fallbacks) / sorted.size());
    }
  }

 private:
  std::array<std::vector<double>, dgpp::kSamplingProfileTopKs.size()> values_;
};

std::array<double, dgpp::kSamplingProfileTopKs.size()>
local_sampling_topk_masses(const GlmDiagnosticModel::Outputs& out,
                           double logsumexp) {
  const std::vector<dgpp::sample::Candidate> top =
      dgpp::sample::local_topk(
          out.logits.data(), out.lm_vocab_count, out.lm_vocab_begin,
          dgpp::kSamplingProfileMaxK);
  const std::vector<double> masses =
      dgpp::sample::topk_probability_masses(
          top, logsumexp,
          std::vector<int>(dgpp::kSamplingProfileTopKs.begin(),
                           dgpp::kSamplingProfileTopKs.end()));
  std::array<double, dgpp::kSamplingProfileTopKs.size()> out_masses{};
  std::copy(masses.begin(), masses.end(), out_masses.begin());
  return out_masses;
}

// ---------------------------------------------------------------------------
// --mtp: greedy speculative decode on the fabric (DESIGN §9), the on-device
// step. With --decode-graph EVERY step is one graph replay: the T=2 verify
// of [next, draft], the recorded pick behind the head (DevicePicker: the
// local argmax, the candidate gather as a collective node, the verdict),
// the commit (the rejected row's rollback and the position advance, on the
// device), the draft block's fixed two rows off the verdict (the second a
// padding row after a miss), its head on both rows, the recorded draft pick
// on the last accepted row, and the next replay's fed tokens written on the
// device. The host launches, syncs, finishes the bus window, reads two
// pinned verdicts, settles its position mirrors and logs. Without the graph
// the same step runs eagerly (host rollback, eager draft, eager picks). The
// transcript is the plain loop's, token for token — the [gen] lines below
// carry the same fields per generated token, so fabric_xcript judges an
// --mtp run against a plain run directly (and must say IDENTICAL).
// ---------------------------------------------------------------------------
struct SpecRunStats {
  int steps = 0;
  int accepted = 0;
  double step_ms = 0;   // graph: the replay (launch..bus finish); eager: verify
  double draft_ms = 0;  // eager only
  double pick_ms = 0;   // eager only
};

// The per-generated-token line fabric_xcript/fabric_logprob read: the
// committed token and this rank's slice argmax with its runner-up.
void log_gen_line(int rank, int gen_index, int vocab_begin, int vocab_count,
                  int32_t best_id, float best_logit, float second,
                  int32_t token) {
  DGPP_LOG_INFO(
      "[gen] rank {} step {}: token {} (local slice [{},{}) best {} logit "
      "{:.4f} second {:.4f})",
      rank, gen_index, token, vocab_begin, vocab_begin + vocab_count, best_id,
      best_logit, second);
}

void log_row_pick(int rank, int gen_index, const GlmDiagnosticModel::Outputs& o,
                  int row, const dgpp::sample::Candidate& local,
                  int32_t token) {
  const float* slice =
      o.logits.data() + static_cast<size_t>(row) * o.lm_vocab_count;
  float second = -INFINITY;
  for (int i = 0; i < o.lm_vocab_count; ++i)
    if (o.lm_vocab_begin + i != local.id && slice[i] > second) second = slice[i];
  log_gen_line(rank, gen_index, o.lm_vocab_begin, o.lm_vocab_count, local.id,
               local.logit, second, token);
}

void run_speculative(GlmDiagnosticModel& model, CollectiveBus& bus, int rank,
                     int world, const GlmTextConfig& cfg,
                     const dgpp::text::Tokenizer& tok,
                     const std::vector<int64_t>& prompt, int steps,
                     bool no_eos, bool decode_graph, uint16_t* pick_scratch,
                     std::vector<int64_t>* generated, std::string* text,
                     double* forward_ms_total, const SamplingRun& sampling,
                     uint16_t* sample_prefix_scratch,
                     uint16_t* sample_gather_scratch) {
  using Clock = std::chrono::steady_clock;
  const auto ms_since = [](Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  };
  const auto is_eos = [&](int64_t id) {
    return !no_eos && std::find(cfg.eos_token_ids.begin(),
                                cfg.eos_token_ids.end(), id) !=
                          cfg.eos_token_ids.end();
  };
  // The picker outlives the graph it records into (its buffers are the
  // recorded nodes' baked addresses). Slot 0: the verify; slot 1: the draft.
  dgpp::DevicePicker picker(bus, rank, world, 60000);
  // `graph_feed`: the recorded verify compares row 0's winner with the
  // token the graph FED to row 1 — the slot's persistent feed rows
  // (device_feed, rows kDecodeRows onward since 2026-09-06), which the
  // recorded token feed rewrites every replay. The eager verify uploads
  // its rows at device_tokens() instead. Handing the graph the eager rows
  // compared against a token nothing rewrites: every draft rejected, the
  // transcript still exact.
  const auto pick_inputs = [&](int rows, int slot,
                               const dgpp::PickVerdict* row_select,
                               bool graph_feed = false) {
    dgpp::DevicePicker::Inputs in;
    in.logits = model.device_logits();
    in.rows = rows;
    in.vocab_count = model.lm_vocab_count();
    in.vocab_begin = model.lm_vocab_begin();
    in.fed = graph_feed ? model.device_feed(0, rows) : model.device_tokens();
    in.slot = slot;
    in.row_select = row_select;
    return in;
  };

  // ---- prefill (main stack + the draft block over the prompt) -----------
  // The prefill's pick stays on the host path: its logits row is the last
  // row of a prompt-sized chunk, already mirrored to the host.
  const auto t_pre = Clock::now();
  const GlmDiagnosticModel::Outputs pre = model.session_prefill(prompt);
  const double prefill_ms = ms_since(t_pre);
  DGPP_LOG_INFO("rank {} prefill: {} tokens in {:.0f}ms (main stack + draft "
                "block)", rank, prompt.size(), prefill_ms);
  const std::vector<dgpp::sample::Candidate> pre_local =
      dgpp::local_row_maxes(pre, 1);
  *forward_ms_total = prefill_ms;
  dgpp::step_timing::reset();
  // The host picks of this loop (the prefill's, the eager rows'): HostPick,
  // greedy — the sampled speculator below draws through its own interface.
  HostPick host(world, rank, &bus, pick_scratch, GenEngineAdapter::Sample{},
                SamplingRun{}, cfg.vocab_size);

  // ---- exact speculative SAMPLING, the eager driver (M6 6b) --------------
  // The sampled prefill pick, then SampledSpeculator: row 0's accept test
  // and residual over the bus, row 1's sample when the draft stands, the
  // block's argmax draft over the accepted rows; two draws per step.
  if (sampling.on()) {
    const auto pick_rows =
        [&](const std::vector<dgpp::sample::Candidate>& locals) {
          return host.pick_rows(locals);
        };
    const dgpp::GenEngineAdapter::Sample row1 = dgpp::make_fabric_sample(
        &bus, rank, world, sample_prefix_scratch, sample_gather_scratch,
        cfg.vocab_size);
    dgpp::sample::Rng rng{sampling.seed, 0};
    const std::vector<int32_t> prompt_context(prompt.begin(), prompt.end());
    const dgpp::sample::Result first =
        row1(pre, *sampling.params, rng, prompt_context, nullptr, nullptr);
    DGPP_LOG_INFO("[gen] rank {} prefill+sample: token {} logprob {:.4f}",
                  rank, first.token, first.logprob);
    dgpp::SampledSpeculator spec(
        model, 0, pick_rows,
        dgpp::make_fabric_spec_row0(&bus, rank, world, sample_prefix_scratch,
                                    sample_gather_scratch, cfg.vocab_size),
        row1, *sampling.params, rng, prompt);
    spec.start(first.token);
    bool stop = false;
    while (!stop && static_cast<int>(generated->size()) < steps) {
      const auto t_step = Clock::now();
      const std::vector<int32_t> committed = spec.step();
      for (const int32_t token : committed) {
        generated->push_back(token);
        *text += tok.decode(token, /*skip_special_tokens=*/false);
        if (is_eos(token)) {
          DGPP_LOG_INFO("rank {} eos stop at token {}", rank,
                        generated->size() - 1);
          stop = true;
          break;
        }
        if (static_cast<int>(generated->size()) >= steps) {
          stop = true;
          break;
        }
      }
      const double ms = ms_since(t_step);
      *forward_ms_total += ms;
      DGPP_LOG_INFO(
          "rank {} sampled spec step {}: {:.1f}ms, accepted {} ({} tokens), "
          "counter {}",
          rank, spec.steps(), ms, static_cast<int>(committed.size()) - 1,
          committed.size(), spec.rng().counter);
    }
    DGPP_LOG_INFO(
        "rank {} sampled speculative summary: {} tokens in {} steps ({:.1f}% "
        "drafts accepted, {:.3f} tokens/step); the transcript is an exact "
        "sample at the request's settings",
        rank, generated->size(), spec.steps(),
        spec.steps() ? 100.0 * spec.accepted_drafts() / spec.steps() : 0.0,
        spec.steps() ? static_cast<double>(generated->size()) / spec.steps()
                     : 0.0);
    return;
  }

  int32_t next = host.pick_rows(pre_local)[0];
  log_row_pick(rank, 0, pre, 0, pre_local[0], next);

  // ---- the eager draft (the first proposal; every step's, without graph) --
  SpecRunStats st;
  const auto draft_after = [&](const std::vector<int64_t>& rows) -> int32_t {
    const auto t0 = Clock::now();
    (void)model.session_draft(0, rows);
    st.draft_ms += ms_since(t0);
    const auto t1 = Clock::now();
    const int32_t id =
        picker.run(model.stream(), pick_inputs(1, 1, nullptr)).next;
    st.pick_ms += ms_since(t1);
    return id;
  };
  int32_t draft = draft_after({next});

  // ---- the step graph, once -----------------------------------------------
  cudaGraphExec_t graph_exec = nullptr;
  std::unique_ptr<dgpp::GlmGraphRecordReducer> recorder;
  if (decode_graph) {
    const auto t_capture = Clock::now();
    model.set_decode_route_traces(false);
    model.set_decode_tail_mirrors(false);
    model.session_graph_prepare();
    model.session_reserve_blocks(0, model.max_tokens());
    recorder = std::make_unique<dgpp::GlmGraphRecordReducer>(bus, model.stream());
    dgpp::GlmBoundaryReducer* eager_reducer = model.set_boundary(recorder.get());
    std::string gerr;
    require(bus.graph_record_begin(&gerr), "graph_record_begin: " + gerr);
    cudaGraph_t graph = nullptr;
    DGPP_CUDA_OK(cudaStreamBeginCapture(model.stream(),
                                        cudaStreamCaptureModeThreadLocal));
    model.session_graph_capture_step(0, std::vector<int64_t>{next, draft},
                                     /*device_positions=*/true,
                                     /*device_tokens=*/true);
    picker.record(model.stream(),
                  pick_inputs(2, 0, nullptr, /*graph_feed=*/true));
    model.session_graph_capture_commit(0, picker.device_verdict(0));
    model.session_graph_capture_draft(0, picker.device_verdict(0));
    picker.record(model.stream(), pick_inputs(1, 1, picker.device_verdict(0)));
    model.session_graph_capture_next_tokens(0, picker.device_verdict(1));
    DGPP_CUDA_OK(cudaStreamEndCapture(model.stream(), &graph));
    require(graph != nullptr, "step-graph capture produced no graph");
    require(bus.graph_record_end(&gerr), "graph_record_end: " + gerr);
    model.set_boundary(eager_reducer);
    dgpp::glm_check_decode_graph(graph, rank, "step graph");
    DGPP_CUDA_OK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    cudaGraphDestroy(graph);
    model.session_graph_seed_tokens(0, {next, draft});
    DGPP_LOG_INFO("rank {} step graph (verify T=2 + pick + commit + draft + "
                  "pick + token feed) recorded+instantiated in {:.0f}ms",
                  rank, ms_since(t_capture));
  }

  // One step. Returns the verify's verdict with the state, positions and
  // (graph) the next draft settled; `draft` is updated for the next step.
  const auto step = [&](const std::vector<int64_t>& fed) -> dgpp::PickVerdict {
    const auto t0 = Clock::now();
    if (graph_exec == nullptr) {
      (void)model.session_verify(0, fed);
      const dgpp::PickVerdict v =
          picker.run(model.stream(), pick_inputs(2, 0, nullptr));
      if (v.accepted < 2) model.session_rollback(0, v.accepted);
      st.step_ms += ms_since(t0);
      return v;
    }
    std::string gerr;
    require(bus.graph_replay_arm(&gerr), "graph_replay_arm: " + gerr);
    DGPP_CUDA_OK(cudaGraphLaunch(graph_exec, model.stream()));
    DGPP_CUDA_OK(cudaStreamSynchronize(model.stream()));
    require(bus.graph_replay_finish(60000, &gerr),
            "graph_replay_finish: " + gerr);
    const dgpp::PickVerdict v = picker.verdict(0);
    draft = picker.verdict(1).next;
    model.session_graph_settle(0, v.accepted);
    st.step_ms += ms_since(t0);
    return v;
  };

  // ---- the loop -----------------------------------------------------------
  const auto commit = [&](int32_t token) {
    generated->push_back(token);
    *text += tok.decode(token, /*skip_special_tokens=*/false);
  };
  bool stop = false;
  while (!stop && static_cast<int>(generated->size()) < steps) {
    const auto t_step = Clock::now();
    const std::vector<int64_t> fed{next, draft};
    const dgpp::PickVerdict v = step(fed);
    ++st.steps;
    st.accepted += v.accepted - 1;
    // Commit: fed[0] was decided last step; a standing row 1 makes the
    // draft final too. Row r's winner is the pick line of the token that
    // follows committed[r] — the plain loop's step index is that token's
    // position in the transcript.
    for (int r = 0; r < v.accepted; ++r) {
      const int gen_index = static_cast<int>(generated->size());
      const int32_t committed = static_cast<int32_t>(fed[static_cast<size_t>(r)]);
      commit(committed);
      const dgpp::PickLocal& local = picker.local(r, 0);
      log_gen_line(rank, gen_index + 1, model.lm_vocab_begin(),
                   model.lm_vocab_count(), local.best_id, local.best_logit,
                   local.second_logit, v.winners[r]);
      if (is_eos(committed)) {
        DGPP_LOG_INFO("rank {} eos stop at token {}", rank, gen_index);
        stop = true;
        break;
      }
      if (static_cast<int>(generated->size()) >= steps) {
        stop = true;
        break;
      }
    }
    next = v.next;
    const double device_ms = ms_since(t_step);
    if (!stop && graph_exec == nullptr)
      draft = draft_after(std::vector<int64_t>(v.winners, v.winners + v.accepted));
    const double total_ms = ms_since(t_step);
    *forward_ms_total += total_ms;
    DGPP_LOG_INFO("rank {} spec step {}: {} {:.1f}ms, draft {:.1f}ms, "
                  "accepted {} ({} tokens)",
                  rank, st.steps, graph_exec ? "graph" : "verify+pick",
                  device_ms, total_ms - device_ms, v.accepted - 1, v.accepted);
  }
  if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
  recorder.reset();
  const double decode_ms = *forward_ms_total - prefill_ms;
  const double per_step = st.steps ? decode_ms / st.steps : 0.0;
  const double device_per_step = st.steps ? st.step_ms / st.steps : 0.0;
  DGPP_LOG_INFO(
      "rank {} speculative summary: {} tokens in {} steps ({:.1f}% drafts "
      "accepted, {:.3f} tokens/step); decode {:.1f}ms = {:.2f} ms/token "
      "effective, {:.2f} ms/step ({} {:.2f} + draft {:.2f} + picks {:.2f} + "
      "host {:.2f} per step)",
      rank, generated->size(), st.steps,
      st.steps ? 100.0 * st.accepted / st.steps : 0.0,
      st.steps ? static_cast<double>(generated->size()) / st.steps : 0.0,
      decode_ms, generated->empty() ? 0.0 : decode_ms / generated->size(),
      per_step, graph_exec || decode_graph ? "graph" : "verify",
      device_per_step, st.steps ? st.draft_ms / st.steps : 0.0,
      st.steps ? st.pick_ms / st.steps : 0.0,
      per_step - device_per_step -
          (st.steps ? (st.draft_ms + st.pick_ms) / st.steps : 0.0));
}

int run(const GlmTextConfig& cfg, const std::string& ckpt, int world,
        int rank, uint16_t port, const std::string& peer,
        const std::vector<int64_t>& prompt, int steps, bool resident,
        bool incremental, bool no_eos, bool decode_graph, bool mtp,
        bool sampling_profile,
        const dgpp::text::Tokenizer& tok,
        const std::string& out_prefix, int rendezvous_timeout_ms,
        int64_t kv_capacity, const std::vector<int64_t>& teacher,
        const SamplingRun& sampling) {
  // Teacher forcing runs the text's length and never stops at EOS: the
  // scored positions are the text's, not the model's choices.
  const bool teaching = !teacher.empty();
  if (teaching) {
    steps = static_cast<int>(teacher.size());
    no_eos = true;
  }
  // The group check walks both prompts as one chunk.
  const int max_tokens = static_cast<int>(prompt.size() + g_group_check_b_tokens.size()) + steps + 1;
  // --kv-capacity overrides the pool bound here too (0 = the historical
  // default); the model rounds it up to a block multiple.
  const int64_t cache =
      kv_capacity > 0 ? kv_capacity : std::max<int64_t>(128, max_tokens);

  // ---- world 1: no bus, full head, plain argmax ----------------------
  if (world == 1) {
    dgpp::prepare_serving_process(rank);
    const auto t_construct = std::chrono::steady_clock::now();
    GlmDiagnosticModel model(cfg, ckpt, max_tokens, cache);
    if (g_gr_probe_layers > 0) model.set_gr_probe_layers(g_gr_probe_layers);
    const double construct_s = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() -
                                   t_construct)
                                   .count();
    DGPP_LOG_INFO("w1 model constructed in {:.1f}s (streaming)", construct_s);
    std::vector<int64_t> generated;
    SamplingMassSummary mass_summary;
    std::string generated_text;  // decoded via the exact tokenizer
    const auto is_eos = [&](int64_t id) {
      return !no_eos && std::find(cfg.eos_token_ids.begin(),
                                 cfg.eos_token_ids.end(), id) !=
                           cfg.eos_token_ids.end();
    };
    if (incremental) {
      // The serving path (Stage 2): prefill once, then one stateful
      // step per token — constant work per step, no T^2 re-forward.
      const auto t0 = std::chrono::steady_clock::now();
      const GlmDiagnosticModel::Outputs out = model.session_prefill(prompt);
      const double prefill_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0)
                                    .count();
      // The pick (HostPick, the one host driver): greedy argmax, or the
      // exact w1 sampler over the full head (context = prompt + the picks,
      // the penalties' count table).
      HostPick picker(/*world=*/1, /*rank=*/0, /*bus=*/nullptr,
                      /*pick_scratch=*/nullptr,
                      dgpp::make_w1_sample(cfg.vocab_size), sampling,
                      cfg.vocab_size);
      std::vector<int32_t> context(prompt.begin(), prompt.end());
      dgpp::sample::Candidate best = picker.pick(out, "prefill+pick", 0, context);
      context.push_back(best.id);
      DGPP_LOG_INFO("[gen] prefill: {} tokens in {:.0f}ms", prompt.size(),
                    prefill_ms);
      // Under teacher forcing the fed token is the text's; the pick is
      // still computed (and logged) as the top-1 hit signal.
      const auto force = [&](int s, const GlmDiagnosticModel::Outputs& o) {
        if (!teaching) return;
        const double lse =
            log_teacher_stats(0, s, o, teacher[static_cast<size_t>(s)],
                              best.id);
        if (sampling_profile)
          mass_summary.add(0, s, local_sampling_topk_masses(o, lse));
        best.id = static_cast<int32_t>(teacher[static_cast<size_t>(s)]);
      };
      force(0, out);
      for (int s = 0; s < steps; ++s) {
        generated.push_back(best.id);
        generated_text += tok.decode(best.id, /*skip_special_tokens=*/false);
        if (is_eos(best.id)) {
          DGPP_LOG_INFO("[gen] eos stop at step {} (token {})", s, best.id);
          break;
        }
        if (s + 1 == steps) break;
        const auto t0s = std::chrono::steady_clock::now();
        const GlmDiagnosticModel::Outputs step =
            model.session_step(generated.back());
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0s)
                              .count();
        best = picker.pick(step, "step", s + 1, context);
        context.push_back(best.id);
        DGPP_LOG_INFO("[gen] step {}: token {} (logit {:.4f}) [{:.0f}ms]",
                      s + 1, best.id, best.logit, ms);
        force(s + 1, step);
      }
    } else {
      // The re-forward reference (the T^2 diagnostic loop).
      std::vector<int64_t> toks = prompt;
      for (int s = 0; s < steps; ++s) {
        const auto t0 = std::chrono::steady_clock::now();
        const GlmDiagnosticModel::Outputs out = model.forward(toks);
        const int T = static_cast<int>(toks.size());
        require(out.lm_vocab_count == cfg.vocab_size,
                "w1 head must be full-vocab");
        const float* row = out.logits.data() +
                           static_cast<size_t>(T - 1) * cfg.vocab_size;
        const dgpp::sample::Candidate best =
            dgpp::sample::local_max(row, cfg.vocab_size, 0);
        toks.push_back(best.id);
        generated.push_back(best.id);
        generated_text += tok.decode(best.id, /*skip_special_tokens=*/false);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        DGPP_LOG_INFO("[gen] step {}: token {} (logit {:.4f}) [{:.0f}ms]",
                      s, best.id, best.logit, ms);
      }
    }
    write_tokens_file(out_prefix + ".tokens.txt", rank, world, prompt,
                      generated);
    if (sampling_profile) mass_summary.report(rank);
    DGPP_LOG_INFO("w1 generated ids: {}", ids_line(generated));
    DGPP_LOG_INFO("w1 generated text: {}", generated_text);
    return 0;
  }

  // ---- fabric TP: bus, sharded head, resident by default --------------
  // The pick scratch (bus_greedy_pick's caller-owned gather table) is
  // allocated BEFORE the world forms — the discipline at the top of this
  // file. Model construction happens after bus.start like glm_tp_check:
  // separate processes put a spinning peer kernel on the PEER's device;
  // it cannot block this rank's construction-phase device syncs.
  uint16_t* pick_scratch = nullptr;
  // Pinned, not managed: the pick scratch is host-written between
  // collectives and device-read by the kernel — pinning removes the UVM
  // migration ping-pong (two faults per pick) and the coherence
  // dependence entirely; the bus's own cells are pinned for the same
  // reason (the 2026-09-01 hunt's lesson: the decode path's shared
  // buffers do not ride managed memory).
  const size_t pick_scratch_elems =
      sampling_profile
          ? std::max(dgpp::kPickScratchElems(world),
                     dgpp::sampling_profile_scratch_elems(world))
          : dgpp::kPickScratchElems(world);
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&pick_scratch),
                             sizeof(uint16_t) * pick_scratch_elems,
                             cudaHostAllocDefault));
  PinnedWords sample_prefix(dgpp::fabric_sampling_prefix_scratch_elems(world));
  PinnedWords sample_gather(
      dgpp::sampling_gather_scratch_elems(cfg.vocab_size));
  std::unique_ptr<CollectiveBus> bus;
  try {
    auto bus_options = dgpp::fabric_bus_options(rank, world, port, peer,
                                                rendezvous_timeout_ms);
    if (g_bulk_slots_override > 0) bus_options.bulk_slots = g_bulk_slots_override;
    if (g_bulk_inflight_override >= 0)
      bus_options.bulk_inflight_per_lane = g_bulk_inflight_override;
    if (g_bulk_pace_override >= 0) bus_options.bulk_pace_gbps = g_bulk_pace_override;
    if (g_bulk_slot_bytes_override > 0)
      bus_options.bulk_slot_bytes = static_cast<size_t>(g_bulk_slot_bytes_override);
    bus = std::make_unique<CollectiveBus>(bus_options);
    std::string err;
    if (!bus->start(&err))
      throw std::runtime_error("rank " + std::to_string(rank) +
                               " bus start: " + err);

    if (g_bulk_bench_mb > 0) {
      // The bulk all-reduce bench: the prefill fold's collective, timed by
      // payload size on the formed world, nothing else running.
      const size_t max_bytes = static_cast<size_t>(g_bulk_bench_mb) << 20;
      void* d_buf = nullptr;
      DGPP_CUDA_OK(cudaMalloc(&d_buf, max_bytes));
      DGPP_CUDA_OK(cudaMemset(d_buf, 0x3c, max_bytes));  // bf16 1.0-ish
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_LOG_INFO("rank {} bulk bench: world {}, pool {} x {} B, inflight cap {} per lane, pace {:.1f} Gb/s per QP ({}), {} iters per size",
                    rank, world, bus_options.bulk_slots, bus_options.bulk_slot_bytes,
                    bus_options.bulk_inflight_per_lane, bus->bulk_pace_gbps(),
                    g_bulk_pace_override >= 0 ? "explicit" : "derived from the port rate",
                    g_bulk_bench_iters);
      for (size_t bytes = size_t{2} << 20; bytes <= max_bytes; bytes <<= 1) {
        const size_t elems = bytes / 2;
        std::vector<double> ms;
        std::vector<double> ms_in_order;
        const uint64_t redos_before = bus->stats().bulk_gate_redos;
        for (int it = 0; it < g_bulk_bench_iters + 2; ++it) {
          const auto t0 = std::chrono::steady_clock::now();
          std::string berr;
          const uint64_t id = bus->allreduce_bulk(d_buf, d_buf, elems, &berr);
          if (id == 0) throw std::runtime_error("bulk bench: rejected: " + berr);
          const dgpp::net::BusAllReduceResult res = bus->wait_allreduce(id, 60000);
          if (!res.ok) throw std::runtime_error("bulk bench: " + res.error);
          const double t = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
          if (it >= 2) ms.push_back(t);  // two warm-ups
          ms_in_order.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        const double p50 = ms[ms.size() / 2];
        {
          // Every iteration in run order (the two warm-ups first): the
          // distribution's shape — a bimodal one is a wire-side event, not
          // kernel noise.
          std::string series;
          for (double t : ms_in_order) {
            char buf[32];
            std::snprintf(buf, sizeof buf, " %.2f", t);
            series += buf;
          }
          DGPP_LOG_INFO("rank {} bulk bench: {:>5} MiB iterations (ms):{}  placement redos {}",
                        rank, bytes >> 20, series,
                        bus->stats().bulk_gate_redos - redos_before);
        }
        // Wire bytes per rank: 2(W-1)/W of the buffer (RS + AG).
        const double wire_mb = 2.0 * (world - 1) / world * bytes / 1048576.0;
        DGPP_LOG_INFO(
            "rank {} bulk bench: {:>5} MiB  min {:7.2f} ms  p50 {:7.2f} ms  max "
            "{:7.2f} ms  ({:.2f} GB/s of wire traffic at p50)",
            rank, bytes >> 20, ms.front(), p50, ms.back(),
            wire_mb / 1024.0 / (p50 / 1000.0));
      }
      cudaFree(d_buf);
      bus->stop();
      return 0;
    }

    dgpp::GlmBusBoundaryReducer reducer(*bus);
    dgpp::prepare_serving_process(rank);
    const auto t_construct = std::chrono::steady_clock::now();
    GlmDiagnosticModel model(
        cfg, ckpt, max_tokens, cache, &reducer, rank, world,
        resident ? dgpp::GlmResidency::Resident
                 : dgpp::GlmResidency::Streaming,
        dgpp::GlmHeadSharding::VocabSharded,
        /*max_requests=*/g_group_check_b_tokens.empty() ? 1 : 2, mtp);
    if (g_gr_probe_layers > 0) model.set_gr_probe_layers(g_gr_probe_layers);
    const double construct_s = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() -
                                   t_construct)
                                   .count();
    DGPP_LOG_INFO("rank {} model constructed in {:.1f}s ({}{})", rank,
                  construct_s, resident ? "resident" : "streaming",
                  mtp ? ", + MTP draft layer" : "");
    if (!g_group_check_b_tokens.empty()) {
      const int rc = run_group_check(model, rank, prompt, g_group_check_b_tokens);
      bus->stop();
      return rc;
    }

    std::vector<int64_t> toks = prompt;
    std::vector<int64_t> generated;
    SamplingMassSummary mass_summary;
    double forward_ms_total = 0.0;
    const auto is_eos = [&](int64_t id) {
      return !no_eos && std::find(cfg.eos_token_ids.begin(),
                                  cfg.eos_token_ids.end(), id) !=
                            cfg.eos_token_ids.end();
    };
    // The pick (HostPick, the one host driver): the fabric sampler is the
    // same closure dgpp-serve's eager engine runs (M6 6b), the greedy pick
    // the bus merge; context = prompt + generated so far.
    HostPick picker(world, rank, bus.get(), pick_scratch,
                    dgpp::make_fabric_sample(bus.get(), rank, world,
                                             sample_prefix.data,
                                             sample_gather.data,
                                             cfg.vocab_size),
                    sampling, cfg.vocab_size);
    // The step body: run one row (forward or stateful step), pick the
    // winner, record it. Returns the picked token.
    const auto run_step = [&](const GlmDiagnosticModel::Outputs& out,
                              const char* what, int s) -> int32_t {
      std::vector<int32_t> context(prompt.begin(), prompt.end());
      context.insert(context.end(), generated.begin(), generated.end());
      return picker.pick(out, what, s, context).id;
    };
    std::string generated_text;
    if (mtp) {
      run_speculative(model, *bus, rank, world, cfg, tok, prompt, steps,
                      no_eos, decode_graph, pick_scratch, &generated,
                      &generated_text, &forward_ms_total, sampling,
                      sample_prefix.data, sample_gather.data);
    } else if (incremental) {
      // The serving path (Stage 2): prefill once, one stateful step per
      // token, distributed pick over the sharded head's slices.
      // Repeats before the timed one: the first prefill of a process pays
      // the one-time setup; --prefill-repeat exposes the steady state.
      for (int rep = 1; rep < g_prefill_repeat; ++rep) {
        const auto tr = std::chrono::steady_clock::now();
        (void)model.session_prefill(prompt);
        DGPP_LOG_INFO("rank {} prefill (repeat {} of {}): {} tokens in {:.0f}ms",
                      rank, rep, g_prefill_repeat, prompt.size(),
                      std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - tr)
                          .count());
      }
      const auto t0 = std::chrono::steady_clock::now();
      const GlmDiagnosticModel::Outputs pre = model.session_prefill(prompt);
      const double prefill_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0)
                                    .count();
      DGPP_LOG_INFO("rank {} prefill: {} tokens in {:.0f}ms{}", rank,
                    prompt.size(), prefill_ms,
                    g_prefill_repeat > 1 ? " (steady state: after the repeats)" : "");
      // Teacher forcing: score this step's logits against the text's next
      // token, then feed THAT token (every rank holds the same text, so
      // the ranks agree without the pick; the pick still runs for its
      // argmax and to keep the step's shape identical to serving).
      const auto force = [&](int s, const GlmDiagnosticModel::Outputs& o,
                             int32_t picked) -> int32_t {
        if (!teaching) return picked;
        const double lse = log_teacher_stats(
            rank, s, o, teacher[static_cast<size_t>(s)], picked);
        if (sampling_profile) {
          mass_summary.add(
              rank, s,
              dgpp::bus_sampling_topk_masses(
                  *bus, rank, world, o.logits.data(), o.lm_vocab_count,
                  o.lm_vocab_begin, lse, pick_scratch, 60000));
        }
        return static_cast<int32_t>(teacher[static_cast<size_t>(s)]);
      };
      const auto tp0 = std::chrono::steady_clock::now();
      int32_t token = force(0, pre, run_step(pre, "prefill+pick", 0));
      forward_ms_total =
          prefill_ms + std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - tp0)
                           .count();
      // The budget below is decode-steps only: prefill's folds and
      // host-path MoE syncs are a different (amortized) story.
      dgpp::step_timing::reset();

      // ---- the graph era (--decode-graph, DESIGN §6.2) ---------------
      // Record the decode step ONCE — the whole launch sequence
      // including the bus's collective nodes — then replay per token:
      // stage -> arm -> launch -> sync -> finish -> collect, with the
      // EAGER pick riding BETWEEN windows (the mixed era). The capture
      // executes NOTHING: state, positions, and staging are untouched,
      // so the first replay performs step 1 exactly as the eager path
      // would.
      cudaGraphExec_t graph_exec = nullptr;
      // The recorder OWNS the stable boundary buffer every recorded node
      // bakes in, so it must outlive the graph exec, not just the capture
      // (a block-scoped recorder here was a use-after-free that pinned
      // memory's sticky mapping happened to forgive).
      std::unique_ptr<dgpp::GlmGraphRecordReducer> recorder;
      if (decode_graph) {
        const auto t_capture = std::chrono::steady_clock::now();
        // The serving loop reads logits only; the per-layer route traces
        // are 126 D2H nodes per token it would otherwise replay for nobody.
        model.set_decode_route_traces(false);
        // No D2H mirror nodes either: the decode graph is kernels-only
        // (glm_check_decode_graph; docs/batched_mtp_graph_stall.md) —
        // session_graph_collect copies the tail eagerly after each replay.
        model.set_decode_tail_mirrors(false);
        // The MoE expert-view tables land in their graph slots here, once;
        // the capture below records kernels that read them in place.
        model.session_graph_prepare();
        recorder =
            std::make_unique<dgpp::GlmGraphRecordReducer>(*bus, model.stream());
        dgpp::GlmBoundaryReducer* eager_reducer =
            model.set_boundary(recorder.get());
        std::string gerr;
        require(bus->graph_record_begin(&gerr),
                "graph_record_begin: " + gerr);
        cudaGraph_t graph = nullptr;
        DGPP_CUDA_OK(cudaStreamBeginCapture(
            model.stream(), cudaStreamCaptureModeThreadLocal));
        model.session_graph_capture_step(0, token);
        DGPP_CUDA_OK(cudaStreamEndCapture(model.stream(), &graph));
        require(graph != nullptr, "decode-graph capture produced no graph");
        require(bus->graph_record_end(&gerr),
                "graph_record_end: " + gerr);
        model.set_boundary(eager_reducer);
        dgpp::glm_check_decode_graph(graph, rank, "decode graph");
        DGPP_CUDA_OK(cudaGraphInstantiate(&graph_exec, graph, nullptr,
                                          nullptr, 0));
        cudaGraphDestroy(graph);
        DGPP_LOG_INFO(
            "rank {} decode graph recorded+instantiated in {:.0f}ms "
            "(the bus logged the collective node count)",
            rank,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_capture)
                .count());
      }

      // The inter-step interface (t_pick of step k -> t0s of step k+1) is the
      // only host region no phase below times, and it is where the peers
      // lost ~10 ms in lockstep (2026-09-02, seen only as rank 0's gen-0
      // handshake). It holds two log lines and a token decode — so the
      // probe brackets it and the next step's phases line reports it.
      using Clock = std::chrono::steady_clock;
      ThreadProbe probe;
      Clock::time_point t_pick_prev{}, t_logged_prev{};
      ThreadProbe::Sample probe_prev{};
      bool have_prev = false;

      for (int s = 0; s < steps; ++s) {
        generated.push_back(token);
        toks.push_back(token);
        generated_text += tok.decode(token, /*skip_special_tokens=*/false);
        if (is_eos(token)) {
          DGPP_LOG_INFO("rank {} eos stop at step {} (token {})", rank, s,
                        token);
          break;
        }
        if (s + 1 == steps) break;
        // Boundary phases, timed: where a step's time goes OUTSIDE the
        // replay (stage+arm+launch, the sync, the window finish, the
        // collect, the eager pick) is exactly what the bus's in-window
        // timeline cannot see — and a stall there on one rank shows up on
        // every other rank as "everyone else was late".
        const auto t0s = Clock::now();
        const ThreadProbe::Sample probe_start = probe.sample();
        std::string gap_text;
        if (have_prev) {
          const auto ms_of = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
          };
          gap_text = std::format(
              "; gap {:.2f} = log {:.2f} + top {:.2f}; {}",
              ms_of(t_pick_prev, t0s), ms_of(t_pick_prev, t_logged_prev),
              ms_of(t_logged_prev, t0s),
              ThreadProbe::delta_text(probe_prev, probe_start));
        }
        auto t_launch = t0s, t_sync = t0s, t_finish = t0s, t_collect = t0s;
        const GlmDiagnosticModel::Outputs step = [&] {
          if (graph_exec == nullptr)
            return model.session_step(generated.back());
          // The replay path: the graph re-uploads the staged pinned
          // members, runs every recorded node, and re-folds through
          // the bus's window; the pick stays EAGER between windows.
          std::string gerr;
          model.session_graph_stage(0, generated.back());
          require(bus->graph_replay_arm(&gerr),
                  "graph_replay_arm: " + gerr);
          DGPP_CUDA_OK(cudaGraphLaunch(graph_exec, model.stream()));
          t_launch = Clock::now();
          DGPP_CUDA_OK(cudaStreamSynchronize(model.stream()));
          t_sync = Clock::now();
          require(bus->graph_replay_finish(60000, &gerr),
                  "graph_replay_finish: " + gerr);
          t_finish = Clock::now();
          GlmDiagnosticModel::Outputs out = model.session_graph_collect(0);
          t_collect = Clock::now();
          return out;
        }();
        token = force(s + 1, step, run_step(step, "step", s + 1));
        const auto t_pick = Clock::now();
        probe_prev = probe.sample();
        const auto ms_between = [](Clock::time_point a, Clock::time_point b) {
          return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const double ms = ms_between(t0s, t_pick);
        forward_ms_total += ms;
        DGPP_LOG_INFO("rank {} step {:.0f}ms ({} + pick)", rank, ms,
                      graph_exec ? "graph replay" : "stateful step");
        if (graph_exec != nullptr) {
          // launch -> the graph's first node, through the bus's calibrated
          // globaltimer offset (CLOCK_MONOTONIC == steady_clock on Linux).
          const int64_t launch_gt =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  t_launch.time_since_epoch())
                  .count() +
              bus->globaltimer_offset_ns();
          const int64_t start_gt =
              static_cast<int64_t>(model.graph_start_globaltimer());
          DGPP_LOG_INFO(
              "rank {} step phases ms: launch {:.2f} sync {:.2f} finish {:.2f} "
              "collect {:.2f} pick {:.2f}; launch->gpu_start {:.2f}{}",
              rank, ms_between(t0s, t_launch), ms_between(t_launch, t_sync),
              ms_between(t_sync, t_finish), ms_between(t_finish, t_collect),
              ms_between(t_collect, t_pick),
              static_cast<double>(start_gt - launch_gt) / 1e6, gap_text);
        }
        t_pick_prev = t_pick;
        t_logged_prev = Clock::now();
        have_prev = true;
      }
      if (graph_exec != nullptr) cudaGraphExecDestroy(graph_exec);
      recorder.reset();  // after the exec: its buffer is baked into the nodes
    } else {
      // The re-forward reference (the T^2 loop).
      for (int s = 0; s < steps; ++s) {
        const auto t0 = std::chrono::steady_clock::now();
        const GlmDiagnosticModel::Outputs out = model.forward(toks);
        const int32_t token = run_step(out, "step", s);
        generated_text += tok.decode(token, /*skip_special_tokens=*/false);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();
        forward_ms_total += ms;
        toks.push_back(token);
        generated.push_back(token);
      }
    }

    write_tokens_file(out_prefix + ".tokens.txt", rank, world, prompt,
                      generated);
    if (sampling_profile) mass_summary.report(rank);
    DGPP_LOG_INFO("rank {} generated ids: {}", rank, ids_line(generated));
    DGPP_LOG_INFO("rank {} generated text: {}", rank, generated_text);
    dgpp::step_timing::report(
        ("rank " + std::to_string(rank) + " gen_check").c_str());

    const auto stats = bus->stats();
    const auto summarize = [](const std::vector<double>& us) {
      std::vector<double> sorted = us;
      std::sort(sorted.begin(), sorted.end());
      const auto pct = [&](double f) {
        return sorted.empty()
                   ? 0.0
                   : sorted[std::min(sorted.size() - 1,
                                     static_cast<size_t>(f * sorted.size()))];
      };
      return std::make_pair(sorted.size(),
                            std::make_pair(pct(0.5), pct(0.99)));
    };
    const auto [lat_n, lat_tails] = summarize(stats.latency.latency_us);
    const auto [bulk_n, bulk_tails] = summarize(stats.bulk.latency_us);
    DGPP_LOG_INFO(
        "rank {}: {} of {} step cap in {:.1f}ms total ({:.0f}ms/step avg "
        "incl. prefill; {}), lat {} "
        "(p50={:.1f}us p99={:.1f}us), bulk {} (p50={:.1f}us p99={:.1f}us)",
        rank, generated.size(), steps, forward_ms_total,
        forward_ms_total / generated.size(),
        incremental ? "incremental engine — constant per-step, the serving "
                       "path (steady-state from the per-step logs)"
                     : "re-forward T^2 diagnostic",
        lat_n, lat_tails.first, lat_tails.second, bulk_n, bulk_tails.first,
        bulk_tails.second);
    bus->stop();
  } catch (...) {
    if (pick_scratch) cudaFreeHost(pick_scratch);
    throw;
  }
  cudaFreeHost(pick_scratch);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  static constexpr const char* kUsage =
      "usage: glm_gen_check --model ORG/NAME | --checkpoint-dir DIR\n"
      "  (--prompt ID,ID,... | --text TEXT | --chat TEXT [--system TEXT]\n"
      "   | --requests FILE)\n"
      "  [--steps N] [--world N --rank R --peer HOST --port N]\n"
      "  [--prompt-file PATH (the ids as a comma- or newline-separated file)]\n"
      "  [--mtp  speculative decode through the checkpoint's MTP draft layer\n"
      "          (fabric only; the transcript is the plain loop's, faster)]\n"
      "  [--streaming] [--engine incremental|reforward] [--no-eos]\n"
      "  [--teacher-file F  score the text's tokens instead of generating:\n"
      "   per-step [tf] lines for scripts/fabric_logprob.py; steps = its\n"
      "   token count; the file must exist on every rank (--stage-file)]\n"
      "  [--sampling-profile  with --teacher-file, measure exact global\n"
      "   top-{32,64,128,256} mass at T=1 for the top_p=0.95 fallback]\n"
      "  [--decode-graph] (fabric decode step as a CUDA graph: record once,\n"
      "   replay per token, eager pick between windows; resident only)\n"
      "  [--step-timing] [--rendezvous-timeout-ms N] [--out PREFIX]\n"
      "  sampling (M6 6b; the default stays the exact greedy loop):\n"
      "  [--sample  sample at the checkpoint's generation_config.json\n"
      "   defaults] [--temperature X] [--top-p X] [--top-k N] [--min-p X]\n"
      "  [--repetition-penalty X] (each implies --sample and overrides the\n"
      "   file) [--seed N (default 0; manifest request i draws from N+i)]\n"
      "   (--mtp: the eager sampled speculator; not with --mtp\n"
      "    --decode-graph, --teacher-file or --engine reforward)\n"
      "scheduler mode (--requests): [--max-concurrency N] [--kv-capacity N]\n"
      "  [--bulk-bench MIB [--bulk-bench-iters N] [--bulk-slots N]\n"
      "   [--bulk-slot-bytes B]]  (the fabric's bulk all-reduce timed by size;\n"
      "   the world forms, no model loads)\n"
      "  [--prefill-repeat N]  (prefill N times; the last is the steady state)\n"
      "  [--sched-plan]\n"
      "  [--admission full|grow] [--admission-window N (256)]  (M6 6d)\n";

  std::string config_path, ckpt, peer, prompt_text, text_prompt, out_prefix =
                                                    "glm_gen",
              model_id, requests_path;
  int world = 1, rank = 0, steps = 8, rendezvous_timeout_ms = 120000;
  int max_concurrency = 0, kv_capacity = 0;
  bool sched_plan = false, step_timing = false;

  uint16_t port = 29970;
  bool resident = true, incremental = true, no_eos = false;
  bool decode_graph = false;
  bool mtp = false;
  bool sampling_profile = false;
  bool sample = false;
  std::optional<float> temperature, top_p, min_p, repetition_penalty;
  std::optional<int> top_k;
  uint64_t seed = 0;
  std::string system_prompt, chat_text, teacher_file;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string {
      require(i + 1 < argc, "missing value for " + a);
      return argv[++i];
    };
    if (a == "--model") model_id = next();
    else if (a == "--checkpoint-dir") ckpt = next();
    else if (a == "--world") world = std::stoi(next());
    else if (a == "--rank") rank = std::stoi(next());
    else if (a == "--peer") peer = next();
    else if (a == "--port") port = static_cast<uint16_t>(std::stoi(next()));
    else if (a == "--prompt") prompt_text = next();
    else if (a == "--prompt-file") {
      // The ids as a file (a 32K-token prompt outruns the argument limit).
      std::ifstream in(next());
      require(in.good(), "--prompt-file: cannot open the ids file");
      prompt_text.assign(std::istreambuf_iterator<char>(in),
                         std::istreambuf_iterator<char>());
      for (char& c : prompt_text)
        if (c == '\n' || c == '\r') c = ',';
      while (!prompt_text.empty() &&
             (prompt_text.back() == ',' || prompt_text.back() == ' '))
        prompt_text.pop_back();
    }
    else if (a == "--text") text_prompt = next();
    else if (a == "--chat") chat_text = next();
    else if (a == "--system") system_prompt = next();
    else if (a == "--steps") steps = std::stoi(next());
    else if (a == "--streaming") resident = false;
    else if (a == "--step-timing") step_timing = true;
    else if (a == "--gr-probe") g_gr_probe_layers = std::stoi(next());
    else if (a == "--group-check-b") g_group_check_b = next();
    else if (a == "--decode-graph") decode_graph = true;
    else if (a == "--mtp") mtp = true;
    else if (a == "--engine") {
      const std::string v = next();
      if (v == "incremental") incremental = true;
      else if (v == "reforward") incremental = false;
      else {
        DGPP_LOG_ERROR("--engine must be incremental|reforward");
        return 1;
      }
    }
    else if (a == "--no-eos") no_eos = true;
    else if (a == "--teacher-file") teacher_file = next();
    else if (a == "--sampling-profile") sampling_profile = true;
    else if (a == "--requests") requests_path = next();
    else if (a == "--max-concurrency") max_concurrency = std::stoi(next());
    else if (a == "--kv-capacity") kv_capacity = std::stoll(next());
    else if (a == "--sched-plan") sched_plan = true;
    else if (a == "--admission") {
      const std::string m = next();
      if (m != "full" && m != "grow") {
        DGPP_LOG_ERROR("--admission must be full or grow");
        return 1;
      }
      g_admission.mode = m == "grow"
                             ? dgpp::sched::AdmissionPolicy::Mode::kGrowOnDemand
                             : dgpp::sched::AdmissionPolicy::Mode::kFullReserve;
    } else if (a == "--admission-window") {
      g_admission.window_tokens = std::stoi(next());
      if (g_admission.window_tokens < 1) {
        DGPP_LOG_ERROR("--admission-window must be at least 1");
        return 1;
      }
    }
    else if (a == "--rendezvous-timeout-ms") rendezvous_timeout_ms = std::stoi(next());
    else if (a == "--bulk-bench") g_bulk_bench_mb = std::stoi(next());
    else if (a == "--prefill-repeat") g_prefill_repeat = std::max(1, std::stoi(next()));
    else if (a == "--bulk-bench-iters") g_bulk_bench_iters = std::stoi(next());
    else if (a == "--bulk-slots") g_bulk_slots_override = std::stoi(next());
    else if (a == "--bulk-inflight") g_bulk_inflight_override = std::stoi(next());
    else if (a == "--bulk-pace-gbps") g_bulk_pace_override = std::stod(next());
    else if (a == "--bulk-slot-bytes") g_bulk_slot_bytes_override = std::stoll(next());
    else if (a == "--out") out_prefix = next();
    else if (a == "--sample") sample = true;
    else if (a == "--temperature") { temperature = std::stof(next()); sample = true; }
    else if (a == "--top-p") { top_p = std::stof(next()); sample = true; }
    else if (a == "--top-k") { top_k = std::stoi(next()); sample = true; }
    else if (a == "--min-p") { min_p = std::stof(next()); sample = true; }
    else if (a == "--repetition-penalty") {
      repetition_penalty = std::stof(next());
      sample = true;
    }
    else if (a == "--seed") seed = std::stoull(next());
    else {
      std::fputs(kUsage, stderr);
      return a == "--help" ? 0 : 1;
    }
  }
  if (step_timing) dgpp::step_timing::set_enabled(true);
  if (world > 1) {
    require(rank >= 0 && rank < world, "--rank outside --world");
    require(!peer.empty() || rank == 0,
            "ranks > 0 need --peer (rank 0's fabric IP)");
  }
  if (steps < 1) {
    DGPP_LOG_ERROR("--steps must be >= 1");
    return 1;
  }
  if (mtp && (world < 2 || !incremental || !requests_path.empty() ||
              !teacher_file.empty())) {
    DGPP_LOG_ERROR(
        "--mtp is the fabric's speculative single-session decode (world > 1, "
        "incremental engine; no --requests, no --teacher-file)");
    return 1;
  }
  if (sampling_profile && teacher_file.empty()) {
    DGPP_LOG_ERROR("--sampling-profile requires --teacher-file");
    return 1;
  }
  if (sample && ((mtp && decode_graph) || !incremental ||
                 !teacher_file.empty())) {
    DGPP_LOG_ERROR(
        "--sample/--temperature/... drive the exact sampler on the eager "
        "engines, the eager --mtp speculator, and this app's plain "
        "--decode-graph loop (its pick stays on the host between windows): "
        "not with --mtp --decode-graph, --teacher-file or --engine reforward "
        "(the one-graph MTP step's sampler is M6 6b's next slice)");
    return 1;
  }
  if (!teacher_file.empty() && (!requests_path.empty() || !incremental)) {
    DGPP_LOG_ERROR(
        "--teacher-file needs the incremental engine and no --requests");
    return 1;
  }
  if (decode_graph) {
    require(world > 1 && incremental && resident,
            "--decode-graph is the fabric decode step's graph era: it "
            "needs --world > 1, --engine incremental, and RESIDENT "
            "weights (the streaming loader refills one resident per "
            "layer — a recorded graph would bake whichever layer was "
            "last resident; resident bindings are lifetime-stable)");
    require(requests_path.empty(),
             "--decode-graph is the single-session decode gate (no "
             "--requests scheduler mode)");
  }
  if (sched_plan && requests_path.empty()) {
    DGPP_LOG_ERROR("--sched-plan requires --requests");
    return 1;
  }
  if (!requests_path.empty() && !incremental) {
    DGPP_LOG_ERROR(
        "--requests drives the incremental session engine (leave --engine "
        "at its incremental default)");
    return 1;
  }
  if (max_concurrency < 0 || kv_capacity < 0) {
    DGPP_LOG_ERROR("--max-concurrency and --kv-capacity must be >= 1");
    return 1;
  }
  if (!model_id.empty()) {
    if (!ckpt.empty()) {
      DGPP_LOG_ERROR("--model and --checkpoint-dir are mutually exclusive");
      return 1;
    }
    std::string err;
    const std::string snapshot = dgpp::hf::model_dir(model_id, &err);
    if (snapshot.empty()) {
      DGPP_LOG_ERROR("--model {}: {}", model_id, err);
      return 1;
    }
    ckpt = snapshot;
    DGPP_LOG_INFO("model {} -> {}", model_id, snapshot);
  }
  if (ckpt.empty()) {
    std::fputs(kUsage, stderr);
    return 1;
  }

  // --sched-plan is a sizing calculator: the receipt prints from
  // config.json + the manifest alone — no device required.
  if (!sched_plan) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      DGPP_LOG_ERROR("no CUDA device visible");
      return 1;
    }
  }
  try {
    GlmTextConfig cfg =
        GlmTextConfig::from_json_file((fs::path(ckpt) / "config.json").string());
    const dgpp::GlmGenerationDefaults generation_defaults =
        dgpp::GlmGenerationDefaults::from_checkpoint_dir(ckpt,
                                                         cfg.vocab_size);
    // The generation file owns EOS for generation programs. config.json's
    // ids remain the compatibility fallback for synthetic fixtures and old
    // checkpoints without generation_config.json.
    if (generation_defaults.eos_token_ids.has_value())
      cfg.eos_token_ids = *generation_defaults.eos_token_ids;
    SamplingRun sampling;
    sampling.seed = seed;
    if (sample) {
      dgpp::sample::Params p;
      p.temperature = generation_defaults.effective_temperature();
      p.top_p = generation_defaults.effective_top_p();
      p.top_k = generation_defaults.effective_top_k();
      p.min_p = generation_defaults.effective_min_p();
      p.repetition_penalty =
          generation_defaults.effective_repetition_penalty();
      if (temperature) p.temperature = *temperature;
      if (top_p) p.top_p = *top_p;
      if (top_k) p.top_k = *top_k;
      if (min_p) p.min_p = *min_p;
      if (repetition_penalty) p.repetition_penalty = *repetition_penalty;
      dgpp::sample::validate_params(p);
      if (p.temperature > 0.0f) sampling.params = p;
      DGPP_LOG_INFO(
          "sampling: temperature {} top_p {} top_k {} min_p {} "
          "repetition_penalty {} seed {} ({})",
          p.temperature, p.top_p, p.top_k, p.min_p, p.repetition_penalty,
          seed,
          p.temperature > 0.0f ? "the exact eager sampler"
                               : "temperature 0: the greedy loop");
    }
    if (cfg.vocab_size > (1 << 18)) {
      DGPP_LOG_ERROR(
          "vocab {} exceeds the 6-bit-triplet pick encoding (2^18) — this "
          "app needs a wider id path before it can serve this checkpoint",
          cfg.vocab_size);
      return 1;
    }
    // --text goes through the exact tokenizer (Stage 3); --prompt stays
    // for raw ids; --chat renders the chat template (Stage 3b) with the
    // value as the user message body. --requests (Stage 2b) is the
    // concurrent manifest. All four are mutually exclusive.
    const int prompt_kinds = static_cast<int>(!text_prompt.empty()) +
                             static_cast<int>(!prompt_text.empty()) +
                             static_cast<int>(!chat_text.empty()) +
                             static_cast<int>(!requests_path.empty());
    if (prompt_kinds > 1) {
      DGPP_LOG_ERROR(
          "--text, --prompt, --chat and --requests are mutually exclusive");
      return 1;
    }
    std::vector<int64_t> prompt;
    const dgpp::text::Tokenizer tok = dgpp::text::Tokenizer::load(
        (fs::path(ckpt) / "tokenizer.json").string());

    // ---- scheduler mode (M6 Stage 2b) --------------------------------
    if (!requests_path.empty()) {
      // Read ONCE: the hash logged across ranks must cover exactly the
      // bytes that were parsed.
      const std::string manifest = read_whole_file(requests_path);
      const uint64_t manifest_hash = fnv1a64(manifest);
      std::vector<dgpp::sched::SchedulerRequest> requests = parse_manifest(
          manifest, requests_path, steps, tok, fs::path(ckpt));
      const int n = static_cast<int>(requests.size());
      const int slots =
          max_concurrency > 0 ? max_concurrency : std::min(8, n);
      int64_t default_cap = 0;
      for (const auto& r : requests)
        default_cap +=
            static_cast<int64_t>(r.prompt.size()) + r.max_steps;
      const int64_t cap = kv_capacity > 0 ? kv_capacity : default_cap;
      const SchedSizing z = sched_sizing(cfg, world, cap);
      // The receipt + forecast validate everything the scheduler will
      // assume (pool-id space, per-request fit) BEFORE any allocation.
      print_memory_receipt(cfg, world, slots, z, requests,
                           /*with_forecast=*/true);
      DGPP_LOG_INFO(
          "manifest {} hash {:016x} — {} requests, {} slot(s), "
          "{}-token shared pool ({} tokens reserved by default)",
          requests_path, manifest_hash, n, slots, z.pool_tokens,
          default_cap);
      if (sched_plan) {
        DGPP_LOG_INFO(
            "--sched-plan: receipt printed; no model loaded, nothing run");
        return 0;
      }
      return run_scheduler(cfg, ckpt, world, rank, port, peer,
                           std::move(requests), slots, cap, resident, no_eos,
                           tok, out_prefix, rendezvous_timeout_ms,
                           cfg.vocab_size, manifest_hash, sampling);
    }

    if (!chat_text.empty()) {
      const dgpp::text::ChatTemplate chat_tpl = dgpp::text::ChatTemplate::load(
          (fs::path(ckpt) / "chat_template.jinja").string());
      std::vector<dgpp::text::Value> messages;
      if (!system_prompt.empty()) {
        dgpp::text::Value::Members sys_msg;
        sys_msg.emplace_back("role", dgpp::text::Value::string_value("system"));
        sys_msg.emplace_back("content", dgpp::text::Value::string_value(system_prompt));
        messages.push_back(dgpp::text::Value::map_value(std::move(sys_msg)));
      }
      dgpp::text::Value::Members user_msg;
      user_msg.emplace_back("role", dgpp::text::Value::string_value("user"));
      user_msg.emplace_back("content", dgpp::text::Value::string_value(chat_text));
      messages.push_back(dgpp::text::Value::map_value(std::move(user_msg)));
      dgpp::text::Value::Members globals;
      globals.emplace_back("messages",
                           dgpp::text::Value::list_value(std::move(messages)));
      globals.emplace_back("add_generation_prompt", dgpp::text::Value::boolean(true));
      const std::string rendered = chat_tpl.render(
          dgpp::text::Value::map_value(std::move(globals)));
      prompt = tok.encode(rendered);
      DGPP_LOG_INFO("chat template {} rendered to {} bytes, {} ids",
                    chat_tpl.source_hash(), rendered.size(), prompt.size());
      require(!prompt.empty(), "--chat produced no tokens");
    } else if (!text_prompt.empty()) {
      prompt = tok.encode(text_prompt);
      DGPP_LOG_INFO("prompt encoded to {} ids by the exact tokenizer",
                    prompt.size());
      require(!prompt.empty(), "--text produced no tokens");
    } else {
      prompt = parse_prompt_ids(prompt_text, cfg.vocab_size);
    }
    if (!g_group_check_b.empty()) {
      g_group_check_b_tokens = tok.encode(g_group_check_b);
      require(!g_group_check_b_tokens.empty(), "--group-check-b produced no tokens");
      DGPP_LOG_INFO("group check: prompt B encoded to {} ids", g_group_check_b_tokens.size());
    }
    std::vector<int64_t> teacher;
    if (!teacher_file.empty()) {
      require(requests_path.empty() && incremental,
              "--teacher-file needs the incremental engine and no --requests");
      std::ifstream in(teacher_file, std::ios::binary);
      require(in.good(), "--teacher-file not readable: " + teacher_file);
      std::stringstream buf;
      buf << in.rdbuf();
      teacher = tok.encode(buf.str());
      require(!teacher.empty(), "--teacher-file produced no tokens");
      DGPP_LOG_INFO("teacher text {} bytes -> {} ids (fnv1a {:016x}); "
                    "--steps ignored",
                    buf.str().size(), teacher.size(),
                    fnv1a64(buf.str()));
    }
    return run(cfg, ckpt, world, rank, port, peer, prompt, steps, resident,
               incremental, no_eos, decode_graph, mtp, sampling_profile, tok,
               out_prefix, rendezvous_timeout_ms, kv_capacity, teacher,
               sampling);
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("rank {}: {}", rank, e.what());
    return 1;
  }
}
