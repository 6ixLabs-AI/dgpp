// Forced K/V exhaustion at snapshot time, using the real Qwen session core
// and GLM's separate implementation. No checkpoint downloads or RDMA needed.
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine/prefix_arena.hpp"
#include "models/glm/forward.hpp"
#include "models/qwen/forward.hpp"

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

bool bitwise(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

template <class Model>
void check_snapshots(Model& model, int vocab) {
  require(model.kv_blocks_total() == 4, "fixture needs four cache blocks");
  const int64_t block = model.kv_block_tokens();
  const int64_t hop = block + 12;
  dgpp::PrefixArena<Model> arena(&model, 3);
  std::vector<int64_t> prompt(static_cast<size_t>(block + 19));
  for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = (i * 7 + 3) % vocab;
  const std::vector<int64_t> cuts{block, hop, block + 16};
  const std::vector<int64_t> peer(prompt.begin(), prompt.begin() + 8);

  // At the first block boundary, the snapshot pins an existing block. The
  // two later partial-block copies must be skipped when the peer holds the
  // other half of the pool. Both failures must unwind the full-block pin.
  for (const bool chunked : {false, true}) {
    const auto run = [&](bool snapshots, bool full) {
      if (full) {
        (void)model.session_prefill(1, peer);
        model.session_reserve_blocks(1, 2 * block);
      }
      auto first = arena.request(0, block);
      auto second = arena.request(1, hop);
      auto third = arena.request(2, block + 16);
      first.next = &second;
      second.next = &third;
      typename Model::Outputs out;
      if (chunked) {
        auto cursor = model.session_prefill_begin(0, prompt, 2 * block, block, cuts, snapshots ? &first : nullptr);
        while (!model.session_prefill_advance(cursor)) {}
        require(cursor.next == block + 19, "skipped snapshot stalled the cursor");
        out = std::move(cursor.output);
      } else {
        out = model.session_prefill(0, prompt, cuts, snapshots ? &first : nullptr);
      }
      arena.commit(0, first);
      arena.commit(1, second);
      arena.commit(2, third);
      require(model.session_position(0) == block + 19, "prefill did not finish");
      if (snapshots) {
        require(first.taken && arena.filled(0), "full-block snapshot should still succeed");
        require(second.taken == !full && third.taken == !full,
                "partial snapshots did not follow pool availability");
        require(arena.filled(1) == !full && arena.filled(2) == !full, "incorrect arena occupancy");
        require(model.kv_blocks_in_use() == 4, "snapshot failure changed pool accounting");
      }
      const auto draft = model.session_draft(0, {7});
      const auto step = model.session_step(0, 7);
      model.session_close(0);
      if (full) model.session_close(1);
      for (int slot = 0; slot < 3; ++slot) arena.release(slot);
      require(model.kv_blocks_in_use() == 0, "snapshot leaked a full or partial block pin");
      return std::vector<std::vector<float>>{out.logits, draft.logits, step.logits};
    };
    const auto baseline = run(false, true);
    for (const bool full : {true, false}) {
      const auto got = run(true, full);
      for (size_t i = 0; i < got.size(); ++i)
        require(bitwise(got[i], baseline[i]), "snapshot changed prefill, draft or decode logits");
    }
  }

  // A full-block snapshot already occupies the slot. Replacing it with a
  // partial hop fails after release, so neither the new nor old position
  // survives. Repeat past the timer-ring length, then retry with free space.
  std::vector<int64_t> prefix(prompt.begin(), prompt.begin() + hop - 1);
  auto old = arena.request(0, block);
  (void)model.session_prefill(0, prefix, {block}, &old);
  arena.commit(0, old);
  require(arena.filled(0), "old snapshot missing");
  model.session_reserve_blocks(0, 2 * block);
  (void)model.session_prefill(1, peer);
  model.session_reserve_blocks(1, 2 * block);
  (void)model.session_draft(0, {7});
  (void)model.session_verify(0, {7, 8});
  require(model.session_position(0) == hop + 1, "verify did not hop over the snapshot position");
  const int64_t before = arena.snapshots();
  for (int attempt = 0; attempt < 8; ++attempt) {
    arena.snapshot_post_row0(0, 0, hop, 0);
    require(!arena.filled(0), "skipped replacement retained stale state");
    require(arena.snapshots() == before, "skip counted as a successful snapshot");
    require(model.kv_blocks_in_use() == 4, "hop failure changed pool accounting");
  }
  model.session_close(1);
  arena.snapshot_post_row0(0, 0, hop, 0);
  require(arena.filled(0) && arena.position(0) == hop && arena.snapshots() == before + 1,
          "slot or timer could not be reused after skipped hops");
  model.session_close(0);
  arena.release(0);
  require(model.kv_blocks_in_use() == 0, "hop failure leaked block pins");
  (void)arena.snapshot_ms();
}

struct FailingModel {
  struct SessionSnapshotMeta { int64_t position = 0; };
  struct SnapshotRequest {};
  size_t session_snapshot_bytes() const { return 4; }
  int64_t session_position(int) const { return 77; }
  cudaStream_t stream() const { return nullptr; }
  void session_release_snapshot(const SessionSnapshotMeta&) {}
  SessionSnapshotMeta session_snapshot_post_row0(int, void*, int, int) {
    throw std::runtime_error("non-pool snapshot failure");
  }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::invalid_argument("usage: prefix_snapshot_test QWEN_FIXTURE GLM_FIXTURE");
    {
      const auto cfg = dgpp::QwenTextConfig::from_json_file(std::string(argv[1]) + "/config.json");
      dgpp::QwenModel model(cfg, argv[1], 128, 256, dgpp::QwenResidency::Streaming, nullptr,
                            0, 1, 2, true);
      check_snapshots(model, cfg.vocab_size);
      std::puts("[ OK ] Qwen: full-pool prefill/hop skips, parity, pin cleanup and slot reuse");
    }
    {
      const auto cfg = dgpp::GlmTextConfig::from_json_file(std::string(argv[2]) + "/config.json");
      dgpp::GlmDiagnosticModel model(cfg, argv[2], 256, 512, nullptr, 0, 1,
                                     dgpp::GlmResidency::Streaming, dgpp::GlmHeadSharding::Full, 2, true);
      check_snapshots(model, cfg.vocab_size);
      std::puts("[ OK ] GLM: full-pool prefill/hop skips, parity, pin cleanup and slot reuse");
    }
    FailingModel model;
    dgpp::PrefixArena<FailingModel> arena(&model, 1);
    bool propagated = false;
    try {
      arena.snapshot_post_row0(0, 0, 76, 0);
    } catch (const std::runtime_error& e) {
      propagated = std::string(e.what()) == "non-pool snapshot failure";
    }
    require(propagated, "non-pool errors must still propagate");
    std::puts("[ OK ] non-pool snapshot errors remain fatal");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
