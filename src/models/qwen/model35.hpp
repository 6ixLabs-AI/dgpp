// Qwen3.5-27B dense model (FP8 text-only, standard pre-norm residual).
//
// Text-only FP8 first boot: 48 GDN layers (swish gate) + 16 Full GQA layers,
// dense SwiGLU MLPs, no MoE / hyperconnections / PLE / vision. The
// attention and GDN layers are the shared qwen4_exp components; the K/V pool
// is a slim QwenKvPool clone without the indexer structures; GDN recurrent
// and conv state is model-owned per request slot.
//
// MTP: one Full draft layer (mtp.layers.0) + fused fc [H, 2H] over
// cat(pre_fc_norm_embedding(embed), pre_fc_norm_hidden(hidden)) + mtp.norm,
// sharing the lm head. Draft width is H (no hyperconnections); the session
// core's window holds the post-norm final hidden rows (h_).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "engine/paged_blocks.hpp"
#include "engine/session_model.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_spec.hpp"
#include "loaders/resident_stream.hpp"
#include "models/quant_matrix.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/layers.hpp"
#include "models/qwen/loader.hpp"
#include "models/qwen/loader35.hpp"
#include "engine/memory_plan.hpp"

namespace dgpp {

struct Qwen35KvPoolShape {
  int layers = 0;  // Full layers served
  int kv_heads = 0;
  int dim = 0;  // head_dim
  int block_tokens = 0;
  int max_requests = 0;
  int64_t token_slots = 0;  // pool capacity in tokens, a multiple of block_tokens
};

// The Full layers' paged K/V caches: per-layer K and V bf16 rows for the
// rank's kv heads plus one block table int32 [max_requests,
// blocks_per_request] shared by every layer. Same protocol as QwenKvPool
// (engine/paged_blocks.hpp owns the table mechanics); no index cache, no
// ring — dense attention reads only the K/V planes.
class Qwen35KvPool {
 public:
  Qwen35KvPool() = default;
  ~Qwen35KvPool();
  Qwen35KvPool(const Qwen35KvPool&) = delete;
  Qwen35KvPool& operator=(const Qwen35KvPool&) = delete;

  void init(const Qwen35KvPoolShape& shape);
  bool initialized() const { return initialized_; }
  // The device bytes init allocates for a shape (the memory plan).
  static size_t cache_bytes(const Qwen35KvPoolShape& shape);

  const Qwen35KvPoolShape& shape() const { return shape_; }
  int64_t total_blocks() const { return table_.total_blocks(); }
  int64_t token_slots() const { return shape_.token_slots; }
  int64_t blocks_in_use() const { return table_.blocks_in_use(); }
  int64_t free_blocks() const { return table_.free_blocks(); }
  int64_t block_count_for_tokens(int64_t tokens) const { return table_.block_count_for_tokens(tokens); }
  const PagedBlockTable& blocks() const { return table_; }

  // The kernels' view of one layer's caches (the shared table inside).
  QwenFullAttnCache view(int layer) const;

  // ---- block management (the shared table's protocol) --------------------
  bool ensure_request_blocks(int req, int64_t tokens, cudaStream_t stream) {
    return table_.ensure_request_blocks(req, tokens, stream);
  }
  void release_request_blocks(int req, cudaStream_t stream) {
    table_.release_request_blocks(req, stream);
  }
  int64_t request_blocks(int req) const { return table_.request_blocks(req); }
  const int32_t* request_table_row(int req) const { return table_.request_table_row(req); }
  // Per-request open: release its blocks (K/V rows are always written
  // before they are read, so no zeroing is needed).
  void reset_request(int req, cudaStream_t stream);
  // Cold start: zero every cache plane and table row; all blocks free.
  void reset_all(cudaStream_t stream);

  // ---- sharing (the prefix cache) --------------------------------------
  bool share_blocks_into(int req, const int32_t* blocks, int64_t n, cudaStream_t stream) {
    return table_.share_blocks_into(req, blocks, n, stream);
  }
  void pin_blocks(const int32_t* blocks, int64_t n) { table_.pin_blocks(blocks, n); }
  void unpin_blocks(const int32_t* blocks, int64_t n) { table_.unpin_blocks(blocks, n); }
  int32_t acquire_pinned_block() { return table_.acquire_pinned_block(); }
  // Every layer's rows of physical block `src` into `dst`, stream-ordered.
  void copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream);
  int32_t block_refcount(int32_t block) const { return table_.block_refcount(block); }

 private:
  Qwen35KvPoolShape shape_;
  bool initialized_ = false;
  PagedBlockTable table_;
  uint16_t* k_base_ = nullptr;  // [layers][token_slots][kv_heads * dim]
  uint16_t* v_base_ = nullptr;
  size_t kv_row_elems() const { return static_cast<size_t>(shape_.kv_heads) * shape_.dim; }
  size_t layer_kv_elems() const { return static_cast<size_t>(shape_.token_slots) * kv_row_elems(); }
  void check_req(int req, const char* what) const;
};

// Fused SwiGLU (model35_kernels.cu): out[i] = silu(gate[i]) * up[i], bf16.
void qwen35_swiglu_bf16(const uint16_t* gate, const uint16_t* up, uint16_t* out, int64_t n,
                        cudaStream_t stream);

// The MTP fusion concat: out[t, :] = [e[t, :], h[t, :]] (model35_kernels.cu).
void qwen35_mtp_concat_bf16(const uint16_t* e, const uint16_t* h, uint16_t* out, int64_t rows,
                            int64_t hidden, cudaStream_t stream);

class Qwen35Model : public SessionModel<Qwen35Model> {
 public:
  using Base = SessionModel<Qwen35Model>;
  Qwen35Model(const Qwen35TextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
              int64_t max_cache_tokens, LoaderResidency residency, BoundaryReducer* boundary, int rank,
              int world, int max_requests, int decode_rows, bool mtp = false);
  ~Qwen35Model();

  static constexpr int decode_rows_cap() { return 32; }
  static constexpr int prefill_chunk_tokens() { return 4096; }
  static constexpr int kv_block_tokens_static() { return 64; }
  // Prefix-snapshot arena size: all GDN recurrent + conv state of one slot.
  static size_t session_snapshot_bytes(const Qwen35TextConfig& cfg, int world, bool mtp);
  size_t session_snapshot_bytes() const { return session_snapshot_bytes(cfg_, 1, mtp_); }
  static MemoryPlan plan_memory(const Qwen35TextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                int rank, int world, LoaderResidency residency, int max_requests,
                                bool mtp = false, int decode_rows = 0);

  const Qwen35TextConfig& config() const { return cfg_; }

  // ---- the session core's hooks (engine/session_model.hpp) --------------------
  typename Base::Outputs run_rows(const typename Base::RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int req, int snapshot_row0 = 0) const;
  size_t snapshot_state_bytes() const;
  // Paged K/V: the draft block reads its own pool plane, so there is no
  // draft ring to snapshot; the core's window row rides the core plan.
  size_t draft_state_bytes() const { return 0; }
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void read_state_snapshot(int req, const uint8_t* src);
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_draft_snapshot(int, const uint8_t*) {}
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) {}
  // The MTP draft chain (depth >= 2): the chain rows append draft-plane
  // K/V at uncommitted positions only, and every draft position is written
  // before it is ever read as committed (rejected guesses are overwritten
  // in place by the next draft), so there is no recurrent draft state to
  // snapshot — unlike QwenModel's ring, the paged draft plane needs no
  // chain copy. The hooks stay as documented no-ops.
  static constexpr bool kDraftChain = true;
  static constexpr bool kBatchedDraftChain = true;
  const uint16_t* draft_hidden_rows() const { return mtp_h_; }
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}
  bool has_pool() const { return true; }
  Qwen35KvPool& pool() { return pool_; }
  const Qwen35KvPool& pool() const { return pool_; }
  void graph_prepare();
  void mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row,
                    bool capture, int head_rows, int batch_requests);
  Outputs mtp_forward(const std::vector<int64_t>& token_ids);

 private:
  void build_layer_objects(const Qwen35LayerResident& r);
  void dense_mlp(const uint16_t* x, uint16_t* out, int tokens, const Qwen35DenseMlpResident& m,
                 cudaStream_t stream);
  float* gdn_rec(int slot, int ord) const {
    return gdn_rec_base_ + (static_cast<size_t>(slot) * num_gdn_ + ord) * rec_elems_;
  }
  uint16_t* gdn_conv(int slot, int ord) const {
    return gdn_conv_base_ + (static_cast<size_t>(slot) * num_gdn_ + ord) * conv_elems_;
  }

  Qwen35TextConfig cfg_;
  Qwen35LayerStream loader_;
  CublasLtGemm gemm_;
  QwenGemmWorkspace gw_;
  Qwen35GlobalsResident globals_;
  QwenTextConfig qcfg_;  // adapter: the fields the qwen4_exp layer ctors read
  std::unique_ptr<QwenFullAttnLayer> full_;
  std::unique_ptr<QwenGdnLayer> gdn_;
  Qwen35KvPool pool_;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  // Activation scratch at max_tokens rows.
  uint16_t *resid_ = nullptr, *x_ = nullptr, *attn_out_ = nullptr, *mlp_out_ = nullptr;
  uint16_t *gate_tmp_ = nullptr, *up_tmp_ = nullptr;  // dense MLP [M, I]
  // MTP draft scratch at max_tokens rows (null when MTP is off): embed rows
  // and their norm, the gathered/gated main hidden and its norm, the
  // [M, 2H] concat, the draft residual, its norm (the chain rows), and the
  // gathered window rows decode drafts read.
  uint16_t *mtp_e_ = nullptr, *mtp_en_ = nullptr, *mtp_hn_ = nullptr, *mtp_hin_ = nullptr;
  uint16_t *mtp_cat_ = nullptr, *mtp_r_ = nullptr, *mtp_h_ = nullptr;
  bool mtp_ = false;
  // Model-owned GDN state: [max_requests][num_gdn][elems], plus the
  // verify's per-row snapshots ([max_decode_rows][num_gdn][elems]) the
  // speculative rollback reads (engine/session_model.hpp's commit).
  float* gdn_rec_base_ = nullptr;
  uint16_t* gdn_conv_base_ = nullptr;
  float* spec_rec_ = nullptr;
  uint16_t* spec_conv_ = nullptr;
  int num_gdn_ = 0, num_full_ = 0;
  int64_t rec_elems_ = 0, conv_elems_ = 0;
};

}  // namespace dgpp
