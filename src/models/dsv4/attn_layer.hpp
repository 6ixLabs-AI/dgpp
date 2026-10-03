#pragma once
// One DeepSeek-V4-Flash attention layer's forward (2026-10-01; the
// reference `Attention`, `Compressor`, `Indexer`, `DSparkAttention`): the
// low-rank q (wq_a -> RMSNorm -> wq_b per local head -> the per-head
// normalization -> the tail rotated), the window latent (wkv -> RMSNorm ->
// rotated -> act_quant on the 448 -> the ring row), and on a compressing
// layer its own compressor (the per-token projections into the positional
// ring, an entry published when its group completes), on a ratio-4 layer
// its indexer (the pooled, Hadamard-rotated fp4 keys; the q heads rotated
// and quantized the same way; the top-512 entries), on a ratio-128 layer
// every compressed entry; then the two-source attention (the window ring
// and the layer's own compressed rows, the sink in the denominator), the
// inverse rotation, the grouped wo_a and wo_b.
//
// One object serves every layer of a model: rebind() points it at a
// layer's weights. Nothing flows between layers (every compressing layer
// selects for itself), so a call is self-contained.
//
// The decode path is graph-capturable (call prepare() outside capture;
// the batch's req_ids / pos are device buffers re-uploaded between
// replays); prefill is a control-path operation (one request per call,
// block-aligned chunks, host staging of positions).
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/dsv4_attn.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/l2_prefetch.hpp"
#include "models/dsv4/compressor.hpp"
#include "models/dsv4/state.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

struct Dsv4AttnConfig {
  // The dense fp8 projections' decode form: the streaming tensor-core GEMM
  // (rows 1..32 in one launch, the weights read once) or the 4-row GEMV
  // chunks. Tolerance-equal forms; the model sets every site alike.
  bool dense_mma = true;
  int hidden = 4096;
  int q_lora = 1024;
  int o_lora = 1024;
  int num_heads = 64;
  int o_groups = 8;
  int index_heads = 64;
  int index_topk = 512;
  int window = 128;
  int ring_slots = 160;
  int block_tokens = 128;
  float eps = 1e-6f;
  int tp = 1;
  int local_heads() const { return num_heads / tp; }
  int local_groups() const { return o_groups / tp; }
  int heads_per_group() const { return num_heads / o_groups; }
  static void validate(const Dsv4AttnConfig& c);
};

struct Dsv4AttnWeights {
  GlmQuantMatrix wq_a, wkv, wq_b, wo_a, wo_b;  // fp8, 128 x 128 grids
  const uint16_t* q_norm = nullptr;
  const uint16_t* kv_norm = nullptr;
  const float* attn_sink = nullptr;    // f32 [local_heads]
  Dsv4CompressorResident comp;         // ratio > 0
  GlmQuantMatrix idx_wq_b;             // ratio 4
  const uint16_t* idx_wp = nullptr;
  Dsv4CompressorResident idx_comp;
  const float* inv_freq = nullptr;     // device [32]: the layer's rotary table
  int ratio = 0;                       // 0: window only
  int cache_ord = -1;                  // the layer's compressed cache (ratio > 0)
};

class Dsv4AttnLayer {
 public:
  Dsv4AttnLayer(IGemm& gemm, const Dsv4AttnConfig& cfg, int max_tokens, int64_t max_cache_tokens, void* scratch,
                size_t scratch_capacity, void* gemm_workspace, size_t gemm_ws_bytes, int max_decode_rows = 16,
                int decode_n_split = 32, size_t dot_budget = 256ull << 20);
  ~Dsv4AttnLayer();
  Dsv4AttnLayer(const Dsv4AttnLayer&) = delete;
  Dsv4AttnLayer& operator=(const Dsv4AttnLayer&) = delete;
  static size_t scratch_bytes(const Dsv4AttnConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                              int max_decode_rows = 16, int decode_n_split = 32, size_t dot_budget = 256ull << 20);

  void rebind(const Dsv4AttnWeights& w, int layer);
  // The decode walk's in-layer L2 prefetch window (kernels/l2_prefetch.hpp;
  // bit-identical on or off): the output projections under the selection
  // and the attention. Null (the default) or a disabled prefetcher: none.
  void set_prefetcher(WeightPrefetcher* p) { prefetch_ = p; }
  // GEMM plans for `tokens` rows and the shared-memory opt-ins; outside capture.
  bool prepare(int tokens);

  // One request's chunk [pos0, pos0 + tokens): pos0 a multiple of the block
  // size (a compressing layer publishes its entries); the request's blocks
  // must already cover pos0 + tokens.
  void enqueue_prefill(const void* hidden_in, Dsv4StatePool& pool, int req, int64_t pos0, int tokens, void* out,
                       cudaStream_t stream);
  // A decode batch of `tokens` rows (contiguous rows per request in
  // position order, pos -1 padding rows).
  void enqueue_decode(const void* hidden_in, Dsv4StatePool& pool, const int32_t* req_ids, const int64_t* pos,
                      int tokens, void* out, cudaStream_t stream);

  // ---- the DSpark draft on a window-only draft stage ----------------------------
  // append_window_rows: the ring rows of `n` real tokens — x [n, hidden] the
  // projected main hidden (main_x), the latent rotated at pos and quantized,
  // the row at pos % ring_slots; pos -1 rows write nothing; n <=
  // min(max_tokens, ring_slots) (one slot per row: the caller passes a
  // chunk's tail).
  void append_window_rows(const void* x, Dsv4StatePool& pool, const int32_t* req_ids, const int64_t* pos, int n,
                          cudaStream_t stream);
  // enqueue_draft_block: the block's rows (`rows` = groups x block; req_ids /
  // pos device per row, pos = the block's first position + the row's index
  // in its block, -1 for a padding block): q and the rows' own window
  // latents from `hidden_in` (into the ring at the block's slots — past the
  // window of every later real query, so never read again), the attention
  // over the ring's real rows up to the block and the block itself
  // (bidirectional inside it), wo_a / wo_b into `out`.
  void enqueue_draft_block(const void* hidden_in, Dsv4StatePool& pool, const int32_t* req_ids, const int64_t* pos,
                           int rows, int block, void* out, cudaStream_t stream);
  static constexpr int kMaxDraftBlock = 8;

  const Dsv4AttnConfig& config() const { return cfg_; }
  int layer() const { return layer_; }
  // Index-key / q exactness violations so far (a host read; synchronizes).
  unsigned index_violations() const;
  // Probes (valid until the next enqueue on the shared scratch).
  const int32_t* debug_topk() const { return topk_; }
  const int32_t* debug_counts() const { return counts_; }
  const uint16_t* debug_attn_out() const { return o_; }
  const uint16_t* debug_q() const { return q_; }
  const uint16_t* debug_kv() const { return kv_; }
  const uint8_t* debug_index_q_codes() const { return q_fp8_; }   // [rows * heads, 128] e4m3
  const float* debug_index_q_scales() const { return q_scale_; }  // [rows * heads]
  // The last prefill select tile's logits [rows, stride] (fp32, -inf past
  // the row's visible entries; the parity tests' certification source).
  const float* debug_logits() const { return logits_; }
  int debug_logits_rows() const { return dbg_logits_rows_; }
  int64_t debug_logits_stride() const { return dbg_logits_stride_; }
  int64_t debug_logits_entries() const { return dbg_logits_entries_; }

  static constexpr int kPrefillAttnRows = 128;
  static constexpr int kPrefillSplit = 4;
  // The decode window attention splits its 128 keys this many ways (a FIXED
  // count, independent of the row count, so a token's window is computed
  // identically in a one-row step and as row 0 of a verify batch).
  static constexpr int kWinDecodeSplit = 8;
  static constexpr int kEntryPad = 256;
  // An in-layer prefetch window's byte budget (the L2 holds 24 MB).
  static constexpr size_t kWindowBudget = size_t{18} << 20;

 private:
  struct Layout {
    size_t total = 0;
    size_t qr, kv, q, o, oa, idx_q, q_fp8, q_scale, w, w_folded, comp_kv, comp_score;
    size_t pos, req_ids, req_zero, slots, pos_sel, scratch_pos, iota, one_block;
    size_t wlist, wcounts, dlist, dcounts, wscratch, topk, counts, mlist;
    size_t m_main, l_main, c_main, m_win, l_win, c_win;
    size_t gather_k, gather_scale, dot, logits, select_ws, counter, violations;
    int tile_cap = 0;
    int64_t max_entries = 0;   // the indexed (ratio 4) caches' entries, padded
    int dense_stride = 0;      // the ratio-128 caches' entries
    int ws_slots = 0;
    int ws_win_rows = 0;
    int list_rows = 0;         // rows the main list holds (an attention tile)
  };
  static Layout layout(const Dsv4AttnConfig& cfg, int max_tokens, int64_t max_cache_tokens, int max_decode_rows,
                       int decode_n_split, size_t dot_budget);

  // The projections and the window row of `tokens` rows at `pos` (device).
  // `ring` (the decode walks): the K/V rows go straight into their ring
  // slots, with their window lists, in the K/V tail's one launch
  // (dsv4_kv_tail); null (the prefill, the draft block): the separate chain.
  struct RingAppend {
    const int32_t* req_ids;
    const int32_t* table;
    uint8_t* ring;
    bool window_lists;
  };
  void project_q_kv(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream,
                    const RingAppend* ring = nullptr);
  void project_kv(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream,
                  const RingAppend* ring = nullptr);
  void indexer_query(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream);
  void compress_project(const void* hidden_in, const Dsv4CompressorResident& c, int tokens, cudaStream_t stream);
  void stage_prefill_rows(int req, int64_t pos0, int tokens, cudaStream_t stream);
  // The attention of rows [row0, row0 + rows) of the current call: the
  // window partials over `win_cache` (a ring or the prefill scratch; the
  // per-row slot lists `list` of stride `list_stride` with `counts`), the
  // main partials over the layer's own cache, the finish into o_.
  void attend_rows(Dsv4StatePool& pool, const int32_t* req_ids_win, const uint8_t* win_cache, int win_block_tokens,
                   const int32_t* win_table, int win_blocks, const int32_t* req_ids_main, const int64_t* pos, int row0,
                   int rows,
                   int n_split_main, cudaStream_t stream, const int32_t* list, int list_stride,
                   const int32_t* counts, bool split_window);
  // Its two halves (attend_rows = both, in order): the window partials,
  // which need the queries and the ring alone, and the main partials with
  // the finish, which need the compressed cache and the selection.
  void attend_window(const int32_t* req_ids_win, const uint8_t* win_cache, int win_block_tokens,
                     const int32_t* win_table, int win_blocks, int row0, int rows, int n_split_main,
                     cudaStream_t stream, const int32_t* list, int list_stride, const int32_t* counts,
                     bool split_window);
  void attend_main(Dsv4StatePool& pool, const int32_t* req_ids_main, const int64_t* pos, int row0, int rows,
                   int n_split_main, cudaStream_t stream, bool split_window);
  void project_out(int tokens, void* out, cudaStream_t stream);
  void validate_pool(const Dsv4StatePool& pool) const;
  // A dense fp8 projection: the decode walks' streaming form, or — inside a
  // prefill chunk (prefill_tiles_) — the grouped tensor-core GEMM on the
  // chunk's view table (upload_dense_views), `view` the matrix's entry.
  static constexpr int kViewWqA = 0, kViewWkv = 1, kViewWqB = 2, kViewWoB = 3, kViewIdxWqB = 4, kViewWoA = 5;
  static constexpr int kDenseViewsMax = 15;        // kViewWoA + at most eight local groups, a multiple of three
  static constexpr int kDensePrefillSplit = 128;   // rows per block along z (the kernel's m-tile)
  void dense(int view, const uint16_t* act, size_t act_stride, const uint8_t* payload, const float* scales,
             uint16_t* out, int m, int n, int k, size_t out_stride, cudaStream_t stream);
  void upload_dense_views(int tokens, cudaStream_t stream);
  bool prefill_tiles_ = false;  // set for the span of a prefill enqueue
  int dense_views_live_ = 0;
  MoeExpertView* d_dense_views_ = nullptr;
  MoeSegment* d_dense_segs_ = nullptr;
  std::vector<MoeExpertView> h_dense_views_;
  std::vector<MoeSegment> h_dense_segs_;

  // The decode windows (see set_prefetcher): opened at the points of the
  // chain where the bytes before them have been consumed.
  bool windows() const { return decode_windows_ && prefetch_ != nullptr && prefetch_->enabled(); }
  void window_fp8(const GlmQuantMatrix& m);
  void open_compress_window(cudaStream_t stream);   // a window-only layer's output window, before wq_b
  void open_output_window(cudaStream_t stream);     // wo_a / wo_b, before the latency-bound attention
  WeightPrefetcher* prefetch_ = nullptr;
  bool decode_windows_ = false;  // set for the span of a decode / draft enqueue
  // A compressing layer's decode window attention runs on this side stream,
  // forked behind the projections and joined before the main partials: it
  // needs the queries and the ring alone, and its two latency-bound blocks
  // hide under the compressor's and the selection's launches.
  cudaStream_t win_side_ = nullptr;
  cudaEvent_t win_fork_ = nullptr;
  cudaEvent_t win_join_ = nullptr;
  // The dense fp8 sites' split-K workspace: the decode walks' alone (a short
  // prefill chunk keeps the unsplit chain, bitwise the rows of a long one).
  void* split_ws() const { return decode_windows_ ? gemm_ws_ : nullptr; }
  size_t split_ws_bytes() const { return decode_windows_ ? gemm_ws_bytes_ : 0; }

  IGemm& gemm_;
  Dsv4AttnConfig cfg_;
  Dsv4AttnWeights w_;
  int layer_ = -1;
  int max_tokens_ = 0;
  int64_t max_cache_tokens_ = 0;
  int max_decode_rows_ = 16;
  int decode_n_split_ = 32;
  int tile_cap_ = 0;
  int64_t max_entries_ = 0;
  int dense_stride_ = 0;
  int64_t gather_zeroed_ = 0;
  int ws_slots_ = 0;
  int ws_win_rows_ = 0;
  int list_rows_ = 0;
  float attn_scale_ = 0.f;
  float index_weight_scale_ = 0.f;
  int dbg_logits_rows_ = 0;
  int64_t dbg_logits_stride_ = 0, dbg_logits_entries_ = 0;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  std::vector<int64_t> host_i64_;
  std::vector<int32_t> host_i32_;

  uint8_t* scratch_ = nullptr;
  uint16_t* qr_ = nullptr;        // [T, q_lora]
  uint16_t* kv_ = nullptr;        // [T, 512]
  uint16_t* q_ = nullptr;         // [T, lh * 512]
  uint16_t* o_ = nullptr;         // [T, lh * 512]
  uint16_t* oa_ = nullptr;        // [T, lg * o_lora]
  uint16_t* idx_q_ = nullptr;     // [T, index_heads * 128]
  uint8_t* q_fp8_ = nullptr;      // [T * index_heads, 128]
  float* q_scale_ = nullptr;      // [T * index_heads]
  uint16_t* iw_ = nullptr;        // [T, index_heads] the indexer weights
  float* w_folded_ = nullptr;     // [T * index_heads]
  float* comp_kv_ = nullptr;      // [T, 1024]
  float* comp_score_ = nullptr;   // [T, 1024]
  int64_t* pos_ = nullptr;        // [T] (prefill staging)
  int32_t* req_ids_ = nullptr;    // [T] (prefill staging)
  int32_t* req_zero_ = nullptr;   // [T] zeros
  int64_t* slots_ = nullptr;      // [T] ring slots
  int64_t* pos_sel_ = nullptr;    // [T]
  int64_t* scratch_pos_ = nullptr;  // [T] window-scratch rows
  int64_t* iota_ = nullptr;       // [T]
  int32_t* one_block_ = nullptr;  // [1] = 0
  int32_t* wlist_ = nullptr;      // [T, window]
  int32_t* wcounts_ = nullptr;    // [T]
  int32_t* dlist_ = nullptr;      // [decode rows, window + kMaxDraftBlock] the draft block's lists
  int32_t* dcounts_ = nullptr;    // [decode rows]
  uint8_t* wscratch_ = nullptr;   // [(window - 1 + T) rows]
  int32_t* topk_ = nullptr;       // [T, index_topk]
  int32_t* counts_ = nullptr;     // [T]
  int32_t* mlist_ = nullptr;      // [list_rows, dense_stride] a ratio-128 tile's entry lists
  float* m_main_ = nullptr;       // [ws_slots, lh]
  float* l_main_ = nullptr;
  float* c_main_ = nullptr;       // [ws_slots, lh, 512]
  float* m_win_ = nullptr;
  float* l_win_ = nullptr;
  float* c_win_ = nullptr;
  uint8_t* gather_k_ = nullptr;   // [max_entries, 128]
  float* gather_scale_ = nullptr; // [max_entries]
  float* dot_ = nullptr;          // [tile_cap * index_heads, max_entries]
  float* logits_ = nullptr;       // [tile_cap, max_entries]
  void* select_ws_ = nullptr;
  int32_t* counter_ws_ = nullptr;
  unsigned* violations_ = nullptr;
};

}  // namespace dgpp
