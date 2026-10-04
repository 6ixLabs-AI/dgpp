#pragma once
// UNVERIFIED DRAFT (2026-10-04): written without access to a GPU or to nvcc.
// It has been syntax-checked against a CUDA header shim on a Mac and has
// never been compiled for a device, loaded a weight or produced a token.
// Built only under -DDGPP_BUILD_QWEN3_PLAIN_DRAFT=ON (CMakeLists.txt);
// sixlabs/ports/qwen3-vl-30b-a3b/gpu-steps.md is the sequence that would
// verify it, against tools/qwen3_reference.py and the host reference
// (models/qwen3/reference.hpp), both of which HAVE run.
//
// The plain Qwen3-MoE model (Qwen3MoeForCausalLM, and the text model of
// Qwen3VLMoeForConditionalGeneration) with resident or streaming weights on
// the shared session core. No new kernel: the walk is assembled from layers
// that serve other families.
//
//   h = embed(x)
//   per layer: h += attn(norm(h, input_layernorm))      Glm4AttentionLayer — GLM-4.7's GQA over
//                                                       128-wide heads, here with no biases, the
//                                                       q/k head norms, and the rotary over the
//                                                       WHOLE head (rotary_dim = head_dim: pairs
//                                                       (i, i + 64), the layer's own half-split)
//              h += moe(norm(h, post_attention_layernorm))   GlmMoeLayer's routed chain alone —
//                                                       the softmax top-k router, NVFP4 experts,
//                                                       NO shared expert — the fp32 sum rounded
//                                                       once (MiMo's arrangement of the same
//                                                       layer; QwenMoeLayer adds a shared expert
//                                                       this model does not have)
//   logits = lm_head(norm(h, norm))
//
// The norms are the two-rounding RMSNorm with a plain weight
// (glm_rmsnorm_bf16) — Qwen3's, not Qwen3-Next's zero-centred 1 + w.
//
// One row walk (run_rows) serves every entry point — the cold diagnostic
// forward, a session's prefill chunks, the eager decode rows and the
// captured decode graphs — and engine/session_model.hpp owns everything
// around it. State per request slot: its row of the paged K/V pool
// (Glm4KvPool, 64-token blocks). Nothing else: no recurrent state, so a
// prefix snapshot is the block list, and a rejected verify row leaves stale
// K/V rows past the position that no later row reads.
//
// TP: `tp_world` > 1 loads this rank's slices (heads / W query heads, the
// rank's kv heads, every expert's intermediate slice, the lm-head vocab
// slice) and folds the attention and MoE block outputs per layer through
// `boundary` — the GLM-4.7 walk's two folds. The code path is the one
// world 1 takes with an identity reducer; it is written, and it is the
// part of this draft furthest from any evidence (the 235B's plan,
// sixlabs/ports/qwen3-235b-a22b/sharding-plan.md, is what it must match).
//
// No draft layer: these checkpoints carry no MTP head, engine.mtp is
// refused by name.
//
// The VL dialect is served as TEXT ONLY. The checkpoint's vision tower is
// validated with the header and never loaded; supports_images() stays false
// (the session core's default), so the service refuses an image part, and
// the family says why (ServeFamily::image_refusal).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "engine/boundary_reducer.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/memory_plan.hpp"
#include "engine/session_model.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_spec.hpp"
#include "models/glm/moe_layer.hpp"
#include "models/glm4/kv_pool.hpp"
#include "models/glm4/layers.hpp"
#include "models/qwen3/config.hpp"
#include "models/qwen3/loader.hpp"

namespace dgpp {

class Qwen3Model : public SessionModel<Qwen3Model> {
 public:
  // Several cold prompts as the spans of one walk (session_prefill_group):
  // the attention rows carry their own request ids and positions, so a span
  // attends to its own request's cache only; every span within max_tokens.
  int64_t prefill_group_span_limit() const { return max_tokens_; }
  using Base = SessionModel<Qwen3Model>;
  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  using RowRun = Base::RowRun;

  // max_tokens bounds a walk's rows (a prefill chunk, the diagnostic
  // forward); max_cache_tokens the paged pool's capacity in tokens (rounded
  // up to a block) — shared by every slot. max_requests: the session slots.
  // decode_rows: the fixed decode batch's row ceiling (0 = kDecodeRows).
  Qwen3Model(const Qwen3TextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
             int64_t max_cache_tokens, LoaderResidency residency = LoaderResidency::Streaming,
             BoundaryReducer* boundary = nullptr, int tp_rank = 0, int tp_world = 1,
             int max_requests = 1, int decode_rows = 0, bool serving_logits = false);
  ~Qwen3Model();
  Qwen3Model(const Qwen3Model&) = delete;
  Qwen3Model& operator=(const Qwen3Model&) = delete;

  // Every byte the constructor (and its layer objects) will allocate for a
  // shape, from the same formulas BEFORE anything is allocated.
  using MemoryPlan = dgpp::MemoryPlan;
  static MemoryPlan plan_memory(const Qwen3TextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                int tp_rank = 0, int tp_world = 1,
                                LoaderResidency residency = LoaderResidency::Streaming,
                                int max_requests = 1, int decode_rows = 0, bool serving_logits = false);

  // The cold diagnostic forward: one request on slot 0 (which must be
  // closed), fresh state, every row's logits; the slot is closed after.
  // capture_layers: every layer's residual output [T, H] in layer_states.
  Outputs forward(const std::vector<int64_t>& token_ids, bool capture_layers = false);

  // The attention kernel takes at most 32 query heads per kv head and the
  // session core's fixed batch; 32 is GLM-4.7's measured ceiling for the
  // same attention layer.
  static constexpr int decode_rows_cap() { return 32; }
  static constexpr int prefill_chunk_tokens() { return kPrefillChunkTokens; }
  static constexpr int kv_block_tokens_static() { return kBlockTokens; }
  static size_t session_snapshot_bytes(const Qwen3TextConfig&, int, bool) { return kSnapshotStamp; }
  using Base::session_snapshot_bytes;

  const Qwen3TextConfig& config() const { return cfg_; }
  const Glm4KvPool& kv_pool() const { return pool_; }

  // ---- the session core's hooks (engine/session_model.hpp) --------------------
  Outputs run_rows(const RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int req, int snapshot_row0 = 0) const;
  size_t snapshot_state_bytes() const { return kSnapshotStamp; }
  size_t draft_state_bytes() const { return 0; }
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_state_snapshot(int, const uint8_t*) {}
  void read_draft_snapshot(int, const uint8_t*) {}
  bool has_pool() const { return true; }
  Glm4KvPool& pool() { return pool_; }
  const Glm4KvPool& pool() const { return pool_; }
  void graph_prepare();
  // No draft layer in this model class: the core never reaches these with
  // mtp off, and the serving family refuses engine.mtp before the build.
  void mtp_run_rows(int, const int64_t*, int64_t, int, bool, bool, int, int);
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) {}
  static constexpr bool kDraftChain = false;
  static constexpr bool kBatchedDraftChain = false;
  const uint16_t* draft_hidden_rows() const { return nullptr; }
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}

 private:
  static constexpr int kBlockTokens = 64;
  static constexpr int kPrefillChunkTokens = 2048;
  // No state families beside the pool's blocks: a 16-byte zero stamp keeps
  // the prefix arena's slots addressable (it refuses a zero-byte model).
  static constexpr size_t kSnapshotStamp = 16;

  void build_layer_objects(const Qwen3LayerResident& r);
  static GlmMoeWeights moe_view(const Qwen3MoeResident& m);
  int table_slots() const;
  // The layer's two blocks over `resid` (the residual, updated in place):
  // the attention block and the MoE block with their folds.
  struct WalkRows {
    const int32_t* req_ids = nullptr;
    const int64_t* pos = nullptr;
    bool decode = false;
    bool capture = false;
    int moe_table_slot = -1;   // >= 0: the capture's graph table
    MoeTraceStaging* trace = nullptr;
  };
  void enqueue_layer(const Qwen3LayerResident& r, int pool_layer, uint16_t* resid, int T, const WalkRows& rows);
  uint16_t* stage(uint16_t* fallback, int T, int width, bool capture);
  void fold(uint16_t* buf, int T, int width, bool capture);

  Qwen3TextConfig cfg_;
  Glm4TextConfig attn_cfg_;  // cfg_.attention_view(): what Glm4AttentionLayer reads
  Qwen3LayerStream loader_;
  CublasLtGemm gemm_;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  Glm4GemmWorkspace gw_;
  Qwen3GlobalsResident globals_;
  GlmMoeConfig moe_cfg_;
  int n_split_ = 32;

  // Layer objects (built at first use, rebound per layer).
  std::unique_ptr<Glm4AttentionLayer> attn_;
  std::unique_ptr<GlmMoeLayer> moe_;

  Glm4KvPool pool_;  // every layer's paged K/V

  // Activations [M rows] (the token rows, the head's outputs and the tail
  // mirrors are the core's).
  uint16_t* resid_ = nullptr;  // [M, H] the residual
  uint16_t* x_ = nullptr;      // [M, H] the normed block input
  uint16_t* y_ = nullptr;      // [M, H] the block output (the fold's fallback)
  float* moe_acc_ = nullptr;   // [M, H] the routed chain's fp32 sum
  // The prefill walk's route traces: one pinned staging slot per layer
  // ([layers][M][top_k]), materialized after the walk's final sync.
  int32_t* h_route_ids_ = nullptr;
  float* h_route_weights_ = nullptr;
};

}  // namespace dgpp
