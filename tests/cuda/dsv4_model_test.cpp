// The DeepSeek-V4-Flash model on the fixture at world 1 — the session
// surface's gates:
//   * the cold forward runs every layer (the residual streams and the
//     ratio-4 layers' selections captured), and a session prefill's last
//     row is bitwise the forward's;
//   * a decode row against the prefill path's same row, layer by layer
//     (the GEMV chain and the decode publication against the tile kernels
//     and the prefill publication), and decode steps against a re-forward
//     — past the first ratio-128 entry published by a decode row;
//   * a verify batch with a rollback: the rolled-back slot continues
//     bitwise like a slot that never saw the rejected rows (the
//     compressors' rings are positional: a rejected row leaves nothing);
//   * two interleaved slots are bitwise the slots alone; a closed slot
//     reopens bitwise and the pool empties;
//   * the chunked prefill (cuts on and off the 128-token block grid) is
//     bitwise the one-shot walk, and the prefix snapshots: hot == cold
//     bitwise at a cut and mid-decode (the window rings and the ratio-4
//     compressor rings travel with the entry);
//   * the group prefill: three cold prompts as one walk, the last rows and
//     the first steps bitwise the prefills alone;
//   * the DSpark draft through the eager session API: the block's rows,
//     the chain, a verify, a rollback and the next draft.
// The transcript-level speculative gates are dsv4_engine_test's, the TP
// worlds dsv4_tp_test's, the reference parity tools/dsv4_torch_reference.py.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "dsv4_fixture.hpp"
#include "models/dsv4/model.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::Dsv4Model;
using dgpp::Dsv4Residency;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  dgpp::Dsv4Config cfg;
  std::string dir;
};
Fixture make_fixture() {
  Fixture fx;
  fx.cfg = dsv4fx::tiny_config();
  fx.dir = (fs::current_path() / "dsv4_model_fixture").string();
  dsv4fx::write_fixture(fx.cfg, fx.dir);
  return fx;
}
std::vector<int64_t> tokens_of(uint64_t seed, int n, int vocab) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = seed | 1;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(vocab));
  }
  return t;
}
// Relative l2 between two logits rows and the argmax.
double rel_l2(const float* a, const float* b, size_t n) {
  double d = 0, m = 0;
  for (size_t i = 0; i < n; ++i) { d += double(a[i] - b[i]) * (a[i] - b[i]); m += double(b[i]) * b[i]; }
  return std::sqrt(d) / std::max(std::sqrt(m), 1e-30);
}
int argmax(const float* a, size_t n) {
  int best = 0;
  for (size_t i = 1; i < n; ++i) if (a[i] > a[size_t(best)]) best = int(i);
  return best;
}
bool bitwise(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && !a.empty() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
// The top-1 rule of two rows that differ at rounding level: equal, or the
// reference's own top-2 margin within 2 % of its top logit (a near tie).
bool top1_or_near_tie(const float* got, const float* want, size_t n) {
  const int g = argmax(got, n), w = argmax(want, n);
  if (g == w) return true;
  return std::fabs(want[size_t(w)] - want[size_t(g)]) <= 0.02f * std::fabs(want[size_t(w)]);
}

constexpr int kRows = 256;    // the walk's rows (max_tokens)
constexpr int64_t kCache = 1024;
}  // namespace

DGPP_TEST(dsv4_model_forward_prefill_decode_and_rollback_at_world_1) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  const int L = fx.cfg.num_hidden_layers;
  const int n_index = fx.cfg.num_index_caches();
  Dsv4Model model(fx.cfg, fx.dir, kRows, kCache, Dsv4Residency::Streaming, nullptr, 0, 1, /*max_requests=*/2,
                  /*mtp=*/false, /*decode_rows=*/8);
  // 1) the cold forward over 150 tokens: 37 ratio-4 entries (past the
  // fixture's 16-entry selection) and one ratio-128 entry.
  const int P = 150;
  const auto prompt = tokens_of(0xD5, P, V);
  auto fwd = model.forward(prompt, /*capture_layers=*/true);
  require(fwd.logits.size() == size_t(P) * V, "forward logits rows");
  for (const float v : fwd.logits) require(std::isfinite(v), "forward logits finite");
  require(fwd.layer_states.size() == size_t(L), "every layer's streams captured");
  require(fwd.dsa_selections.size() == size_t(n_index), "the ratio-4 layers' selections captured");
  for (const auto& st : fwd.layer_states)
    for (const uint16_t b : st) require(std::isfinite(bf16_bits_to_float(b)), "stream finite");
  require(rel_l2(fwd.logits.data(), fwd.logits.data() + size_t(P - 1) * V, size_t(V)) > 1e-3, "rows differ");
  // The last row selects among 37 visible entries: its 16 picks are a
  // proper subset (every pick a visible entry, none repeated).
  for (int s = 0; s < n_index; ++s) {
    const int ms = fx.cfg.index_topk;
    const int32_t* row = fwd.dsa_selections[size_t(s)].data() + size_t(P - 1) * ms;
    std::vector<int32_t> seen;
    for (int i = 0; i < ms; ++i)
      if (row[i] >= 0) {
        require(row[i] < P / 4, "a selected entry is visible");
        require(std::find(seen.begin(), seen.end(), row[i]) == seen.end(), "selections are distinct");
        seen.push_back(row[i]);
      }
    require(static_cast<int>(seen.size()) == ms, "the last row fills its selection");
  }

  // 2) prefill == forward bitwise on the last row.
  auto pre = model.session_prefill(0, prompt);
  require(pre.logits.size() == size_t(V), "prefill returns the last row");
  for (int i = 0; i < V; ++i)
    require(pre.logits[size_t(i)] == fwd.logits[size_t(P - 1) * V + i], "prefill's last row bitwise the forward's");

  // 3) the decode path against the prefill path row by row: a 149-token
  // prefill, then the prompt's own last token as a decode step, its layer
  // rows against the forward's row 149. The decode row runs the GEMV chain
  // and the split window attention against the tile kernels: rounding
  // level, unless the row's selection flips on a near tie (reported; the
  // layers from the flip on then hold the flip budget and the argmax).
  {
    Dsv4Model m2(fx.cfg, fx.dir, kRows, kCache, Dsv4Residency::Streaming, nullptr, 0, 1, 1, false, 8);
    const std::vector<int64_t> head(prompt.begin(), prompt.begin() + (P - 1));
    (void)m2.session_prefill(0, head);
    setenv("DGPP_DSV4_CAPTURE_DECODE", "1", 1);
    const auto st = m2.session_step(0, prompt[size_t(P - 1)]);
    unsetenv("DGPP_DSV4_CAPTURE_DECODE");
    const int H4 = 4 * fx.cfg.hidden_size;
    require(st.layer_states.size() == size_t(L), "the decode walk captured every layer");
    require(st.dsa_selections.size() == size_t(n_index), "the decode walk captured its selections");
    int first_flip_layer = 99;
    for (int l = 0; l < L; ++l) {
      if (!fx.cfg.indexed(l)) continue;
      const int s = fx.cfg.index_ordinal(l);
      const int ms = fx.cfg.index_topk;
      std::vector<int32_t> a(st.dsa_selections[size_t(s)].begin(), st.dsa_selections[size_t(s)].begin() + ms);
      std::vector<int32_t> b(fwd.dsa_selections[size_t(s)].begin() + size_t(P - 1) * ms,
                             fwd.dsa_selections[size_t(s)].begin() + size_t(P) * ms);
      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());
      const bool flipped = a != b;
      std::printf("[INFO] decode vs prefill, layer %d selection: %s\n", l, flipped ? "FLIPPED" : "equal");
      if (flipped) first_flip_layer = std::min(first_flip_layer, l);
    }
    for (int l = 0; l < L; ++l) {
      double d = 0, m = 0;
      for (int i = 0; i < H4; ++i) {
        const double a = bf16_bits_to_float(st.layer_states[size_t(l)][size_t(i)]);
        const double b = bf16_bits_to_float(fwd.layer_states[size_t(l)][size_t(P - 1) * H4 + i]);
        d += (a - b) * (a - b); m += b * b;
      }
      const double l2 = std::sqrt(d) / std::max(std::sqrt(m), 1e-30);
      std::printf("[INFO] decode vs prefill, layer %d (ratio %d): rel l2 %.3e\n", l, fx.cfg.compress_ratio(l), l2);
      if (l < first_flip_layer) require(l2 < 0.01, "layer " + std::to_string(l) + " within the decode budget");
      // A flipped row attends to a different set: its move is the attention's,
      // an order above rounding and far under a wrong row (order one).
      else require(l2 < 0.3, "layer " + std::to_string(l) + " past a selection flip within a flip's move");
    }
    if (first_flip_layer > L)
      require(rel_l2(st.logits.data(), fwd.logits.data() + size_t(P - 1) * V, size_t(V)) < 0.01, "decode logits within the budget");
    else
      require(top1_or_near_tie(st.logits.data(), fwd.logits.data() + size_t(P - 1) * V, size_t(V)),
              "decode top-1 matches the prefill's past a selection flip");
    m2.session_close(0);
  }

  // 4) decode steps against a re-forward over the prompt plus the fed
  // tokens. Twelve steps from 150: the ratio-4 groups close at 152, 156,
  // 160 (entries published by decode rows, read by later rows). The decode
  // rows sit at rounding level from the prefill's (the GEMV chains, split
  // across the part, against the tile kernels), and on this random-weight
  // fixture a top-16 boundary is often that close: a step whose selection
  // differs from the re-forward's row reads a different attention set, and
  // its cached entries move every later row. Rows before the first such
  // flip hold the decode budget; from it on the top-1 rule is the gate.
  std::vector<int64_t> history = prompt;
  const auto feed = tokens_of(0xD6, 12, V);
  std::vector<std::vector<std::vector<int32_t>>> step_sels;  // per step, per indexed layer, the row's picks
  bool flipped = false;
  setenv("DGPP_DSV4_CAPTURE_DECODE", "1", 1);
  for (int k = 0; k < 12; ++k) {
    const auto step = model.session_step(0, feed[size_t(k)]);
    history.push_back(feed[size_t(k)]);
    step_sels.push_back(step.dsa_selections);
    if (k % 4 != 3) continue;  // a re-forward every fourth step
    Dsv4Model ref(fx.cfg, fx.dir, kRows, kCache, Dsv4Residency::Streaming, nullptr, 0, 1, 1, false, 8);
    const auto again = ref.forward(history, true);
    const int ms = fx.cfg.index_topk;
    for (int j = 0; j <= k && !flipped; ++j)
      for (int sidx = 0; sidx < n_index && !flipped; ++sidx) {
        std::vector<int32_t> a(step_sels[size_t(j)][size_t(sidx)].begin(), step_sels[size_t(j)][size_t(sidx)].begin() + ms);
        const size_t row = size_t(P + j);
        std::vector<int32_t> b(again.dsa_selections[size_t(sidx)].begin() + row * ms,
                               again.dsa_selections[size_t(sidx)].begin() + (row + 1) * ms);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) {
          flipped = true;
          std::printf("[INFO] decode step %d: the selection of indexed layer %d differs from the re-forward's (a near tie)\n", j, sidx);
        }
      }
    const float* want = again.logits.data() + (history.size() - 1) * V;
    const double l2 = rel_l2(step.logits.data(), want, size_t(V));
    std::printf("[INFO] decode step %d: rel l2 %.2e vs the re-forward, argmax %d / %d%s\n", k, l2,
                argmax(step.logits.data(), size_t(V)), argmax(want, size_t(V)), flipped ? " (past a selection flip)" : "");
    require(top1_or_near_tie(step.logits.data(), want, size_t(V)),
            "decode step " + std::to_string(k) + " top-1 matches the re-forward");
    if (!flipped) require(l2 < 0.12, "decode step " + std::to_string(k) + " within the decode budget");
  }
  unsetenv("DGPP_DSV4_CAPTURE_DECODE");

  // 5) a verify batch of six rows (the DSpark shape), one accepted, then
  // two more rows — against a second slot that fed the same accepted
  // tokens with no rejected row in between: bitwise (the rings are
  // positional, a rejected row's writes are overwritten before any read),
  // and against the re-forward within the budget.
  const auto more = tokens_of(0xD9, 6, V);
  const std::vector<int64_t> drafts(more.begin(), more.end());
  const auto ver = model.session_verify(0, drafts);
  require(ver.logits.size() == size_t(6) * V, "verify returns every row");
  model.session_rollback(0, 1);
  const auto tok2 = tokens_of(0xD7, 2, V);
  const auto ver2 = model.session_verify(0, tok2);
  {
    (void)model.session_prefill(1, prompt);
    for (int k = 0; k < 12; ++k) (void)model.session_step(1, feed[size_t(k)]);
    const auto one = model.session_step(1, drafts[0]);
    const auto two = model.session_verify(1, tok2);
    const bool row0 = std::memcmp(one.logits.data(), ver.logits.data(), size_t(V) * 4) == 0;
    std::printf("[INFO] verify row 0 of a six-row batch vs the same row alone: %s (rel l2 %.2e)\n",
                row0 ? "bitwise" : "differs", rel_l2(ver.logits.data(), one.logits.data(), size_t(V)));
    require(row0, "a verify batch's first row is bitwise the row stepped alone");
    require(bitwise(ver2.logits, two.logits), "the rows after a rollback are bitwise a slot's that never saw the rejected rows");
    model.session_close(1);
  }
  history.push_back(drafts[0]);
  {
    std::vector<int64_t> h = history;
    h.insert(h.end(), tok2.begin(), tok2.end());
    Dsv4Model ref(fx.cfg, fx.dir, kRows, kCache, Dsv4Residency::Streaming, nullptr, 0, 1, 1, false, 8);
    const auto again = ref.forward(h, false);
    for (int r = 0; r < 2; ++r) {
      const float* want = again.logits.data() + (history.size() + size_t(r)) * V;
      const double l2 = rel_l2(ver2.logits.data() + size_t(r) * V, want, size_t(V));
      std::printf("[INFO] post-rollback row %d: rel l2 %.2e vs the re-forward\n", r, l2);
      require(top1_or_near_tie(ver2.logits.data() + size_t(r) * V, want, size_t(V)), "post-rollback row top-1 matches the re-forward");
      if (!flipped) require(l2 < 0.12, "post-rollback row within the decode budget");
    }
  }
  model.session_close(0);
  require(model.kv_blocks_in_use() == 0, "every block released");
}

// Two slots stepped alternately are bitwise the slots alone; a closed slot
// reopens bitwise.
DGPP_TEST(dsv4_model_interleaved_slots_and_reopen_are_bitwise) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  const auto A = tokens_of(0xB1, 141, V), B = tokens_of(0xB2, 37, V);
  const int kSteps = 20;  // A crosses its 128-group... B its ratio-4 groups
  Dsv4Model m(fx.cfg, fx.dir, kRows, kCache, Dsv4Residency::Streaming, nullptr, 0, 1, /*max_requests=*/2, false, 8);
  auto run_alone = [&](int slot, const std::vector<int64_t>& p) {
    std::vector<std::vector<float>> rows;
    auto o = m.session_prefill(slot, p);
    rows.push_back(o.logits);
    for (int s = 0; s < kSteps; ++s) {
      o = m.session_step(slot, argmax(rows.back().data(), size_t(V)));
      rows.push_back(o.logits);
    }
    m.session_close(slot);
    return rows;
  };
  const auto ra = run_alone(0, A);
  const auto rb = run_alone(1, B);
  require(m.kv_blocks_in_use() == 0, "the pool is empty after the closes");
  auto oa = m.session_prefill(0, A);
  auto ob = m.session_prefill(1, B);
  require(bitwise(oa.logits, ra[0]) && bitwise(ob.logits, rb[0]), "the prefills beside another slot are bitwise the prefills alone");
  for (int s = 0; s < kSteps; ++s) {
    oa = m.session_step(0, argmax(oa.logits.data(), size_t(V)));
    ob = m.session_step(1, argmax(ob.logits.data(), size_t(V)));
    require(bitwise(oa.logits, ra[size_t(s) + 1]), "slot 0 step " + std::to_string(s) + " differs beside slot 1");
    require(bitwise(ob.logits, rb[size_t(s) + 1]), "slot 1 step " + std::to_string(s) + " differs beside slot 0");
  }
  m.session_close(0);
  // The reopened slot 0 with B's prompt (a shorter request over the
  // rings a longer one left behind).
  const auto rb0 = run_alone(0, B);
  for (size_t i = 0; i < rb.size(); ++i) require(bitwise(rb0[i], rb[i]), "a reopened slot differs from a fresh one");
  m.session_close(1);
  require(m.kv_blocks_in_use() == 0, "every block released");
  std::printf("[ OK ] interleaved slots and a reopened slot: bitwise the slots alone over %d steps\n", kSteps);
}

// The chunked prefill and the prefix snapshots.
DGPP_TEST(dsv4_model_chunked_prefill_and_prefix_snapshots) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  const int block = Dsv4Model::kv_block_tokens_static();
  const auto C = tokens_of(0xC3, 300, V);
  Dsv4Model m(fx.cfg, fx.dir, /*max_tokens=*/512, /*max_cache_tokens=*/2048, Dsv4Residency::Streaming, nullptr, 0, 1,
              /*max_requests=*/2, false, 8);
  require(m.session_snapshot_align() == block, "snapshots sit on the pool's block grid");
  const auto one = m.session_prefill(0, C);
  const auto one_step = m.session_step(0, 7);
  m.session_close(0);
  // 1) the chunked walk against the one-shot: bitwise. The prefill is
  // split-invariant — every dense site's kernel is row-invariant, a row
  // reads its window, its compressor tails and its entries back from the
  // rings and the pool (the same stored values whichever chunk wrote
  // them), and the selection scores the stored keys — so a prefix-cache
  // cut, a chat-template boundary or the chunk budget cannot move a
  // transcript. Cuts on the block grid, then off it (inside a ratio-4
  // group and a ratio-128 group; a last chunk of 10 rows takes the dense
  // sites' decode form).
  for (const std::vector<int64_t>& bounds : {std::vector<int64_t>{block, 2 * block}, std::vector<int64_t>{101, 133, 290}}) {
    const auto p = m.session_prefill(0, C, bounds);
    const auto st = m.session_step(0, 7);
    std::printf("[INFO] chunked prefill (first cut %ld, %zu cuts): rel l2 %.3e vs the one-shot, the next step %.3e\n",
                static_cast<long>(bounds[0]), bounds.size(), rel_l2(p.logits.data(), one.logits.data(), size_t(V)),
                rel_l2(st.logits.data(), one_step.logits.data(), size_t(V)));
    require(bitwise(p.logits, one.logits), "chunked prefill: the last row differs from the one-shot walk's");
    require(bitwise(st.logits, one_step.logits), "chunked prefill: the step after it differs from the one-shot walk's");
    m.session_close(0);
  }
  // 2) hot == cold bitwise: a snapshot at the cut 256 of the 300-token
  // prompt, attached in the other slot, the suffix resumed.
  std::vector<uint8_t*> arena(2, nullptr);
  const size_t bytes = m.session_snapshot_bytes();
  require(bytes > 0, "prefix: the snapshot has bytes");
  for (uint8_t*& p : arena) require(cudaMalloc(reinterpret_cast<void**>(&p), bytes) == cudaSuccess, "prefix: arena");
  Dsv4Model::SessionSnapshotMeta meta;
  {
    const std::vector<int64_t> bounds{2 * block};
    Dsv4Model::SnapshotRequest snap;
    snap.position = 2 * block;
    snap.dst = arena[0];
    snap.meta = &meta;
    const auto cold = m.session_prefill(0, C, bounds, &snap);
    require(snap.taken && meta.position == 2 * block, "prefix: the snapshot was taken at the cut");
    const auto cold_step = m.session_step(0, 7);
    const auto cold_step2 = m.session_step(0, 11);
    m.session_close(0);
    m.session_attach(1, arena[0], meta);
    require(m.session_position(1) == 2 * block, "prefix: attached at the snapshot position");
    const auto hot = m.session_prefill_resume(1, std::vector<int64_t>(C.begin() + 2 * block, C.end()), bounds);
    require(bitwise(hot.logits, cold.logits), "prefix: the hot prefill's last row differs from the cold one's");
    require(bitwise(m.session_step(1, 7).logits, cold_step.logits), "prefix: the first step after the attach differs");
    require(bitwise(m.session_step(1, 11).logits, cold_step2.logits), "prefix: the second step after the attach differs");
    m.session_close(1);
    // The entry outlives its writer and serves a second attach with a
    // different suffix against that suffix's cold walk.
    std::vector<int64_t> D(C.begin(), C.begin() + 2 * block);
    const auto tail = tokens_of(0xC4, 21, V);
    D.insert(D.end(), tail.begin(), tail.end());
    const auto cold_d = m.session_prefill(0, D, bounds);
    m.session_close(0);
    m.session_attach(1, arena[0], meta);
    const auto hot_d = m.session_prefill_resume(1, tail, bounds);
    require(bitwise(hot_d.logits, cold_d.logits), "prefix: a second suffix on the same entry differs from its cold walk");
    m.session_close(1);
  }
  // 3) mid-decode at an aligned position: a 107-token prompt + 21 steps =
  // 128 (the ratio-128 entry of the first block published by a decode row).
  {
    const auto A = tokens_of(0xC5, 107, V);
    auto o = m.session_prefill(1, A);
    int64_t pending = argmax(o.logits.data(), size_t(V));
    for (int s = 0; s < block - static_cast<int>(A.size()); ++s) {
      o = m.session_step(1, pending);
      pending = argmax(o.logits.data(), size_t(V));
    }
    require(m.session_position(1) == block, "prefix: position 128");
    Dsv4Model::SessionSnapshotMeta meta2 = m.session_snapshot(1, arena[1]);
    const auto cont = m.session_step(1, pending);
    const auto cont2 = m.session_step(1, 5);
    const auto cont3 = m.session_verify(1, tokens_of(0xC6, 6, V));
    m.session_close(1);
    m.session_attach(0, arena[1], meta2);
    require(bitwise(m.session_step(0, pending).logits, cont.logits), "prefix: the step after a mid-decode attach differs");
    require(bitwise(m.session_step(0, 5).logits, cont2.logits), "prefix: the second step after a mid-decode attach differs");
    require(bitwise(m.session_verify(0, tokens_of(0xC6, 6, V)).logits, cont3.logits),
            "prefix: the verify batch after a mid-decode attach differs");
    m.session_close(0);
    m.session_release_snapshot(meta2);
  }
  m.session_release_snapshot(meta);
  require(m.kv_blocks_in_use() == 0, "prefix: every block released with the entries");
  for (uint8_t* p : arena) cudaFree(p);
  std::printf("[ OK ] prefix snapshots: hot == cold bitwise at a cut, on a second suffix and mid-decode\n");
}

// The group prefill (session_prefill_group): three cold prompts as the
// spans of one walk against each prefilled alone — the last rows' logits
// bitwise (the dense sites and the MoE tile kernel are row-invariant, the
// attention, the compressors and the publication run per span), and the
// first decode step of each request bitwise the alone model's (the draft
// state, positions and rings per request). One prompt holds compressed
// entries of both ratios.
DGPP_TEST(dsv4_model_group_prefill_is_bitwise_the_prefills_alone) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  const std::vector<std::vector<int64_t>> prompts = {tokens_of(0xA1, 141, V), tokens_of(0xA2, 37, V), tokens_of(0xA3, 11, V)};
  std::vector<std::vector<float>> alone_logits, alone_step;
  {
    Dsv4Model m(fx.cfg, fx.dir, kRows, kCache, Dsv4Residency::Streaming, nullptr, 0, 1, 3, /*mtp=*/true, 8);
    for (int r = 0; r < 3; ++r) {
      const auto o = m.session_prefill(r, prompts[static_cast<size_t>(r)]);
      alone_logits.push_back(o.logits);
      const int32_t first = argmax(o.logits.data(), size_t(V));
      const auto st = m.session_step(r, first);
      alone_step.push_back(st.logits);
    }
  }
  {
    Dsv4Model m(fx.cfg, fx.dir, kRows, kCache, Dsv4Residency::Streaming, nullptr, 0, 1, 3, /*mtp=*/true, 8);
    require(m.prefill_group_span_limit() >= 141, "the span limit is the walk's rows");
    const std::vector<const std::vector<int64_t>*> pp = {&prompts[0], &prompts[1], &prompts[2]};
    const auto outs = m.session_prefill_group({2, 0, 1}, {pp[2], pp[0], pp[1]});  // a permuted slot order
    require(outs.size() == 3, "one output per request");
    const int order[3] = {2, 0, 1};
    for (int i = 0; i < 3; ++i) {
      const int r = order[i];
      require(outs[static_cast<size_t>(i)].logits.size() == size_t(V), "a last-row logits vector per request");
      require(outs[static_cast<size_t>(i)].logits == alone_logits[static_cast<size_t>(r)],
              "group prefill logits bitwise the prefill alone (request " + std::to_string(r) + ")");
    }
    for (int i = 0; i < 3; ++i) {
      const int r = order[i];
      const int32_t first = argmax(alone_logits[static_cast<size_t>(r)].data(), size_t(V));
      const auto st = m.session_step(r, first);
      require(st.logits == alone_step[static_cast<size_t>(r)],
              "the first step after a group prefill bitwise the alone model's (request " + std::to_string(r) + ")");
    }
    std::printf("[ OK ] group prefill of 141 + 37 + 11 rows: last rows and first steps bitwise the prefills alone\n");
  }
}

// The resumable prefill and its group advance: three prompts read in
// through their cursors, their next chunks the spans of ONE walk per tick
// (a 128-token share each; a prompt joins a tick late), a peer slot
// decoding between the ticks — every prompt's logits, its first draft and
// its first step bitwise the one-shot prefill's, the peer's steps bitwise
// an undisturbed peer's. One cursor alone (session_prefill_advance) the
// same, and a cancelled cursor's slot reopens clean.
DGPP_TEST(dsv4_model_resumable_prefill_and_group_advance_are_bitwise_the_one_shot) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  const int block = Dsv4Model::kv_block_tokens_static();
  const std::vector<std::vector<int64_t>> prompts = {tokens_of(0xB1, 700, V), tokens_of(0xB2, 300, V),
                                                     tokens_of(0xB3, 141, V)};
  const auto peer = tokens_of(0xB4, 37, V);
  Dsv4Model ref(fx.cfg, fx.dir, /*max_tokens=*/1024, /*max_cache_tokens=*/4096, Dsv4Residency::Streaming, nullptr, 0, 1,
                /*max_requests=*/4, /*mtp=*/true, 8);
  Dsv4Model got(fx.cfg, fx.dir, 1024, 4096, Dsv4Residency::Streaming, nullptr, 0, 1, 4, true, 8);
  require(Dsv4Model::kResumablePrefill && Dsv4Model::kPrefillGroupAdvance, "the family yields and groups its prefill");
  require(got.session_snapshot_align() == block, "chunks sit on the pool's block grid");
  std::vector<std::vector<float>> one_logits, one_draft, one_step;
  for (int r = 0; r < 3; ++r) {
    const auto o = ref.session_prefill(r, prompts[static_cast<size_t>(r)]);
    const int32_t first = argmax(o.logits.data(), size_t(V));
    one_logits.push_back(o.logits);
    one_draft.push_back(ref.session_draft(r, {first}).logits);
    one_step.push_back(ref.session_step(r, first).logits);
  }
  auto peer_ref = ref.session_prefill(3, peer);
  auto peer_got = got.session_prefill(3, peer);
  require(peer_ref.logits == peer_got.logits, "the peer starts identically");
  using Cursor = Dsv4Model::PrefillCursor;
  std::vector<Cursor> cursors;
  cursors.reserve(3);
  cursors.push_back(got.session_prefill_begin(0, prompts[0], 700 + 16, block));
  cursors.push_back(got.session_prefill_begin(1, prompts[1], 300 + 16, block));
  int ticks = 0, widest = 0;
  for (;;) {
    // The third prompt arrives two ticks late and joins the walk.
    if (ticks == 2) cursors.push_back(got.session_prefill_begin(2, prompts[2], 141 + 16, block));
    std::vector<Cursor*> open;
    for (Cursor& c : cursors)
      if (c.next < c.end) open.push_back(&c);
    if (open.empty() && ticks > 2) break;
    if (!open.empty()) {
      std::vector<int64_t> before;
      for (Cursor* c : open) before.push_back(c->next);
      const std::vector<bool> done = got.session_prefill_advance_group(open, std::vector<int64_t>(open.size(), block));
      widest = std::max(widest, static_cast<int>(open.size()));
      for (size_t i = 0; i < open.size(); ++i) {
        require(open[i]->next > before[i] && open[i]->next - before[i] <= block, "a share of at most one block per tick");
        require(done[i] == (open[i]->next == open[i]->end), "done at the prompt's end");
      }
    }
    // The peer decodes between the ticks, undisturbed by the unfinished slots.
    const int32_t token = argmax(peer_ref.logits.data(), size_t(V));
    require(ref.session_draft(3, {token}).logits == got.session_draft(3, {token}).logits,
            "a yield preserves the peer's draft state");
    peer_ref = ref.session_step(3, token);
    peer_got = got.session_step(3, token);
    require(peer_ref.logits == peer_got.logits, "the peer's decode survives every yield bitwise");
    ++ticks;
  }
  require(widest == 3, "three prompts shared a walk");
  for (int r = 0; r < 3; ++r) {
    const size_t i = static_cast<size_t>(r);
    require(cursors[i].output.logits == one_logits[i],
            "group advance: the prompt's logits are the one-shot's (request " + std::to_string(r) + ")");
    const int32_t first = argmax(one_logits[i].data(), size_t(V));
    require(got.session_draft(r, {first}).logits == one_draft[i],
            "group advance: the first draft is the one-shot's (request " + std::to_string(r) + ")");
    require(got.session_step(r, first).logits == one_step[i],
            "group advance: the first step is the one-shot's (request " + std::to_string(r) + ")");
  }
  std::printf("[ OK ] 700 + 300 + 141 rows read in over %d ticks (up to 3 prompts a walk, a peer decoding between): "
              "logits, first drafts and first steps bitwise the one-shot prefills\n", ticks);
  // One cursor alone, cancelled once: the slot reopens clean.
  got.session_close(1);
  auto cancelled = got.session_prefill_begin(1, prompts[1], 300 + 16, block);
  require(!got.session_prefill_advance(cancelled), "a 300-token prompt yields after one block");
  got.session_close(1);
  auto again = got.session_prefill_begin(1, prompts[1], 300 + 16, 2 * block);
  int chunks = 0;
  while (!got.session_prefill_advance(again)) ++chunks;
  require(chunks == 1, "256-token chunks: one yield inside a 300-token prompt");
  require(again.output.logits == one_logits[1], "one cursor alone: the logits are the one-shot's");
  const int32_t first1 = argmax(one_logits[1].data(), size_t(V));
  require(got.session_draft(1, {first1}).logits == one_draft[1], "one cursor alone: the first draft is the one-shot's");
  require(got.session_step(1, first1).logits == one_step[1], "one cursor alone: the first step is the one-shot's");
  std::printf("[ OK ] a cancelled cursor's slot reopens clean; one cursor at 256-token chunks bitwise the one-shot\n");
}

// The DSpark draft through the eager session API: after a prefill, the
// first draft call (the accepted row's target hiddens into the draft
// rings, the block over [next, noise x 4], the Markov-biased row 0) and
// four chain rows; then a verify over [next, drafts], a rollback and the
// next draft — every row finite, the block's rows distinct, the confidence
// logits finite.
DGPP_TEST(dsv4_model_dspark_draft_rows_and_chain_at_world_1) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  const int B = fx.cfg.dspark_block_size;
  require(B == 5, "the fixture's block");
  Dsv4Model model(fx.cfg, fx.dir, kRows, kCache, Dsv4Residency::Resident, nullptr, 0, 1, /*max_requests=*/1,
                  /*mtp=*/true, /*decode_rows=*/6);
  const auto prompt = tokens_of(0xD8, 40, V);
  const auto pre = model.session_prefill(0, prompt);
  const int next = argmax(pre.logits.data(), size_t(V));
  std::vector<int32_t> ids{static_cast<int32_t>(next)};
  std::vector<std::vector<float>> rows;
  const auto d0 = model.session_draft(0, {next});
  require(d0.logits.size() == size_t(V), "the draft returns one row");
  rows.push_back(d0.logits);
  for (int i = 0; i < B - 1; ++i) {
    const int tok = argmax(rows.back().data(), size_t(V));
    ids.push_back(tok);
    require(model.session_draft_chain_fits(0, i), "the chain fits the context");
    const auto dc = model.session_draft_chain(0, tok, i, i == 0, i == B - 2);
    require(dc.logits.size() == size_t(V), "a chain row returns one row");
    rows.push_back(dc.logits);
  }
  ids.push_back(argmax(rows.back().data(), size_t(V)));
  for (const auto& r : rows)
    for (const float v : r) require(std::isfinite(v), "draft logits finite");
  for (int i = 1; i < B; ++i) require(rel_l2(rows[size_t(i)].data(), rows[0].data(), size_t(V)) > 1e-4, "block rows differ");
  std::vector<float> conf(static_cast<size_t>(B));
  DGPP_CUDA_OK(cudaMemcpy(conf.data(), model.debug_confidence(), size_t(B) * 4, cudaMemcpyDeviceToHost));
  for (const float c : conf) require(std::isfinite(c), "confidence finite");
  std::printf("[INFO] dspark ids: next %d drafts", next);
  for (size_t i = 1; i < ids.size(); ++i) std::printf(" %d", ids[i]);
  std::printf("; confidence %.3f %.3f %.3f %.3f %.3f\n", conf[0], conf[1], conf[2], conf[3], conf[4]);
  // The verify over [next, d1 .. d5], greedy acceptance, the rollback and
  // the next draft (the accepted rows' target hiddens appended to the rings).
  std::vector<int64_t> fed;
  for (const int32_t t : ids) fed.push_back(t);
  const auto ver = model.session_verify(0, fed);
  require(ver.logits.size() == size_t(B + 1) * V, "the verify's rows");
  int accepted = 1;
  while (accepted < B + 1 && argmax(ver.logits.data() + size_t(accepted - 1) * V, size_t(V)) == ids[size_t(accepted)]) ++accepted;
  if (accepted < B + 1) model.session_rollback(0, accepted);
  std::printf("[INFO] verify accepted %d of %d rows\n", accepted, B + 1);
  // The draft consumes the accepted rows' tokens: row i's token is the
  // token AFTER row i (the winner); the last accepted row's is the new next.
  std::vector<int64_t> draft_tokens;
  for (int i = 1; i < accepted; ++i) draft_tokens.push_back(fed[size_t(i)]);
  draft_tokens.push_back(argmax(ver.logits.data() + size_t(accepted - 1) * V, size_t(V)));
  const auto d1 = model.session_draft(0, draft_tokens);
  for (const float v : d1.logits) require(std::isfinite(v), "second draft finite");
  const auto dc1 = model.session_draft_chain(0, argmax(d1.logits.data(), size_t(V)), 0, true, false);
  for (const float v : dc1.logits) require(std::isfinite(v), "second chain finite");
  model.session_close(0);
}

// The screened draft head (the block's base logits off the block-FP8 copy
// of the lm head, every logit within reach of a pick's maximum recomputed
// from the bf16 rows) against the bf16 head itself: the same picks down the
// chain, the picked logits at the bf16 head's values, and the rest of a row
// within the copy's noise.
DGPP_TEST(dsv4_model_screened_draft_head_picks_are_the_bf16_heads) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  const int B = fx.cfg.dspark_block_size;
  const auto chain = [&](bool screened, uint64_t seed, std::vector<int32_t>* ids) {
    Dsv4Model model(fx.cfg, fx.dir, kRows, kCache, Dsv4Residency::Resident, nullptr, 0, 1, /*max_requests=*/1,
                    /*mtp=*/true, /*decode_rows=*/6);
    require(model.debug_draft_screened(), "the fixture's head takes the screen");
    if (!screened) model.debug_drop_draft_screen();
    const auto prompt = tokens_of(seed, 40, V);
    const auto pre = model.session_prefill(0, prompt);
    const int next = argmax(pre.logits.data(), size_t(V));
    std::vector<std::vector<float>> rows;
    rows.push_back(model.session_draft(0, {next}).logits);
    ids->push_back(next);
    for (int i = 0; i < B - 1; ++i) {
      const int tok = argmax(rows.back().data(), size_t(V));
      ids->push_back(tok);
      rows.push_back(model.session_draft_chain(0, tok, i, i == 0, i == B - 2).logits);
    }
    ids->push_back(argmax(rows.back().data(), size_t(V)));
    model.session_close(0);
    return rows;
  };
  double worst_pick = 0, worst_row = 0;
  for (const uint64_t seed : {0xD8ull, 0x3Bull, 0x91ull}) {
    std::vector<int32_t> ids_s, ids_b;
    const auto rows_s = chain(true, seed, &ids_s);
    const auto rows_b = chain(false, seed, &ids_b);
    require(ids_s == ids_b, "screened draft: a pick differs from the bf16 head's");
    for (int i = 0; i < B; ++i) {
      const int pick = ids_s[size_t(i) + 1];
      const double want = rows_b[size_t(i)][size_t(pick)], got = rows_s[size_t(i)][size_t(pick)];
      worst_pick = std::max(worst_pick, std::fabs(got - want) / (std::fabs(want) + 1e-3));
      worst_row = std::max(worst_row, rel_l2(rows_s[size_t(i)].data(), rows_b[size_t(i)].data(), size_t(V)));
    }
  }
  std::printf("[INFO] screened draft head: picks equal over 3 chains; picked logits within %.2e of the bf16 head's, rows within %.2e\n",
              worst_pick, worst_row);
  require(worst_pick < 1e-4, "screened draft: a picked logit is not the bf16 head's");
  require(worst_row < 0.05, "screened draft: a row sits outside the copy's noise");
}

int main() { return dgpp::test::run_all(); }
