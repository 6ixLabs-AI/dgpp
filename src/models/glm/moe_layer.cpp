#include "models/glm/moe_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/moe_w4a4.hpp"
#include "kernels/packq_gemm.hpp"
#include "kernels/scale_gemm.hpp"
#include "models/glm/step_timing.hpp"

namespace dgpp {

size_t GlmMoeLayer::scratch_bytes(const GlmMoeConfig& cfg, int max_tokens,
                                  int decode_slots, int graph_table_slots,
                                  size_t* pinned_bytes) {
  const size_t M = static_cast<size_t>(std::max(max_tokens, 0));
  const size_t H = static_cast<size_t>(cfg.hidden);
  const size_t I = static_cast<size_t>(cfg.inter);
  const size_t K = static_cast<size_t>(cfg.top_k);
  const size_t E = static_cast<size_t>(cfg.n_experts);
  const size_t rows_total = M * (K + 1);
  const size_t tk_max = M * K;
  const size_t segs_max = E + 1;
  size_t dev = 0, pin = 0;
  dev += M * K * 4 * 2;              // d_ids_, d_weights_
  dev += M * E * 4 * 2;              // d_biased_, d_scores_
  dev += rows_total * 4;             // d_rows_
  dev += tk_max * 4 * 2;             // d_row_w_, d_slot_row_
  dev += segs_max * sizeof(MoeSegment);
  dev += segs_max * 3 * sizeof(MoeExpertView);
  dev += static_cast<size_t>(moe_tile_list_capacity(static_cast<int>(segs_max), static_cast<int>(tk_max),
                                                    kPackqGemmWideRows)) * sizeof(MoeTile) + 4;
  pin += tk_max * 4 * 2;             // h_ids_pinned_, h_weights_pinned_
  pin += M * E * 4;                  // h_biased_pinned_
  pin += rows_total * 4;             // h_seg_rows_
  pin += tk_max * 4;                 // h_slot_row_
  pin += segs_max * sizeof(MoeSegment);
  pin += static_cast<size_t>(kViewRing) * segs_max * 3 * sizeof(MoeExpertView);
  dev += rows_total * H * 2;         // d_gather_
  dev += rows_total * I * 2 * 3;     // d_gate_, d_up_, d_act_
  dev += rows_total * H * 4;         // d_down_
  dev += M * H * 4;                  // d_acc_
  if (decode_slots > 0) {
    const size_t rows = static_cast<size_t>(decode_slots) * (K + 1);
    dev += rows * I * 2 + rows * H * 4 + rows * 4;
    dev += static_cast<size_t>(decode_slots) * sizeof(int);
    dev += sizeof(MoeExpertView) * (E + 1) * 3;
    if (graph_table_slots > 0) {
      const size_t table = sizeof(MoeExpertView) * (E + 1) * 3 * static_cast<size_t>(graph_table_slots);
      pin += table;
      dev += table;
    }
  }
  if (pinned_bytes) *pinned_bytes = pin;
  return dev;
}

GlmMoeLayer::GlmMoeLayer(const GlmMoeWeights& weights, const GlmMoeConfig& cfg,
                          int max_tokens, int decode_slots,
                          int graph_table_slots)
    : w_(weights), cfg_(cfg), max_tokens_(max_tokens),
      decode_slots_(decode_slots), graph_table_slots_(graph_table_slots) {
  GlmMoeConfig::validate_config(cfg_);
  if (max_tokens_ <= 0)
    throw std::invalid_argument("GlmMoeLayer: max_tokens must be positive");
  if (decode_slots_ < 0)
    throw std::invalid_argument("GlmMoeLayer: decode_slots must be >= 0");
  if (graph_table_slots_ < 0)
    throw std::invalid_argument(
        "GlmMoeLayer: graph_table_slots must be >= 0");
  const bool bias_required = cfg_.router_mode == MoeRouterMode::SigmoidBias;
  if (!w_.router_gate || (bias_required && !w_.router_bias) ||
      (!w_.experts && !w_.experts_fp4 && !w_.experts_packed) ||
      (has_shared() && !w_.shared[0].payload && !w_.shared_fp4[0].payload &&
       !w_.shared_packed[0].packed))
    throw std::invalid_argument("GlmMoeLayer: null weight pointer");
  if (w_.shared_nvfp4() && !w_.nvfp4())
    throw std::invalid_argument(
        "GlmMoeLayer: an NVFP4 shared expert needs NVFP4 routed experts (one table)");
  if (w_.shared_packq() && !w_.packq())
    throw std::invalid_argument(
        "GlmMoeLayer: a packed shared expert needs packed routed experts (one table)");
  if (w_.packq() && has_shared() && !w_.shared_packq())
    throw std::invalid_argument(
        "GlmMoeLayer: packed routed experts need a packed shared expert (the slot kernels' table)");
  if ((w_.experts != nullptr) + (w_.experts_fp4 != nullptr) + (w_.experts_packed != nullptr) != 1)
    throw std::invalid_argument(
        "GlmMoeLayer: routed experts must be bound in exactly one format");

  const int M = max_tokens_;
  const size_t H = static_cast<size_t>(cfg_.hidden);
  const size_t I = static_cast<size_t>(cfg_.inter);
  // Plain device memory, not managed: nothing on the host
  // ever dereferences these (the traces and the diagnostic path leave via
  // cudaMemcpyAsync), and on the GB10 managed pages are the slow
  // translation path for every kernel that touches them — the decode
  // slot chain touches them ~700 times per token.
  DGPP_CUDA_OK(cudaMalloc(&d_ids_, static_cast<size_t>(M) * cfg_.top_k * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_weights_, static_cast<size_t>(M) * cfg_.top_k * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_biased_,
                                  static_cast<size_t>(M) * cfg_.n_experts * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_scores_,
                                  static_cast<size_t>(M) * cfg_.n_experts * 4));
  // The grouped prefill path's rows: every routed (token, slot) plus the
  // tokens once more for the shared expert.
  const size_t rows_total = static_cast<size_t>(M) * (cfg_.top_k + 1);
  const size_t tk_max = static_cast<size_t>(M) * cfg_.top_k;
  const size_t segs_max = static_cast<size_t>(cfg_.n_experts) + 1;
  DGPP_CUDA_OK(cudaMalloc(&d_rows_, rows_total * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_row_w_, tk_max * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_slot_row_, tk_max * 4));
  DGPP_CUDA_OK(cudaMalloc(&d_segs_, segs_max * sizeof(MoeSegment)));
  DGPP_CUDA_OK(cudaMalloc(&d_views_prefill_, segs_max * 3 * sizeof(MoeExpertView)));
  tiles_cap_ = moe_tile_list_capacity(static_cast<int>(segs_max), static_cast<int>(tk_max), kPackqGemmWideRows);
  DGPP_CUDA_OK(cudaMalloc(&d_tiles_, static_cast<size_t>(tiles_cap_) * sizeof(MoeTile)));
  DGPP_CUDA_OK(cudaMalloc(&d_tile_count_, sizeof(int32_t)));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_ids_pinned_), tk_max * 4,
                             cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_weights_pinned_),
                             tk_max * 4, cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_biased_pinned_),
                             static_cast<size_t>(M) * cfg_.n_experts * 4,
                             cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_seg_rows_),
                             rows_total * 4, cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_slot_row_), tk_max * 4,
                             cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_segs_),
                             segs_max * sizeof(MoeSegment), cudaHostAllocDefault));
  // The expert-view upload ring (see the member's comment).
  view_table_entries_ = segs_max * 3;
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_view_ring_),
                             static_cast<size_t>(kViewRing) * view_table_entries_ *
                                 sizeof(MoeExpertView),
                             cudaHostAllocDefault));
  for (int i = 0; i < kViewRing; ++i)
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&view_ring_event_[i],
                                          cudaEventDisableTiming));
  DGPP_CUDA_OK(cudaMalloc(&d_gather_, rows_total * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_gate_, rows_total * I * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_up_, rows_total * I * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_act_, rows_total * I * 2));
  DGPP_CUDA_OK(cudaMalloc(&d_down_, rows_total * H * sizeof(float)));
  DGPP_CUDA_OK(cudaMalloc(&d_acc_, M * H * sizeof(float)));
  h_counts_.assign(cfg_.n_experts, 0);

  // Decode-slot scratch: tokens*(top_k+1) rows — routed slots plus the
  // shared expert's, per token. Sized by the decode-row bound, not
  // max_tokens: a short-prompt model still decodes full slots.
  if (decode_slots_ > 0) {
    const size_t rows =
        static_cast<size_t>(decode_slots_) * (cfg_.top_k + 1);
    DGPP_CUDA_OK(cudaMalloc(&d_slot_act_, rows * I * 2));
    DGPP_CUDA_OK(cudaMalloc(&d_slot_down_, rows * H * sizeof(float)));
    DGPP_CUDA_OK(cudaMalloc(&d_slot_order_, rows * sizeof(int32_t)));
    // The fused router selection's tickets: one per decode row, zero at
    // rest (the last block of each launch resets its own).
    DGPP_CUDA_OK(cudaMalloc(&d_router_counters_,
                            static_cast<size_t>(decode_slots_) * sizeof(int)));
    DGPP_CUDA_OK(cudaMemset(d_router_counters_, 0,
                            static_cast<size_t>(decode_slots_) * sizeof(int)));
    // One table of every expert's three views (plus the NVFP4 shared
    // expert's three, entry n_experts — the (E+1)-entry table, sized so
    // whatever the shared format), re-uploaded per eager enqueue_decode
    // from the upload ring (see h_view_ring_'s comment).
    DGPP_CUDA_OK(cudaMalloc(
        &d_expert_views_,
        sizeof(MoeExpertView) * static_cast<size_t>(cfg_.n_experts + 1) * 3));
    // Per-slot capture sources: each recorded upload node bakes its
    // slot's address, whose contents freeze at capture time (resident
    // bindings). The eager path never touches these.
    if (graph_table_slots_ > 0) {
      const size_t table_bytes =
          sizeof(MoeExpertView) * static_cast<size_t>(cfg_.n_experts + 1) * 3 *
          static_cast<size_t>(graph_table_slots_);
      DGPP_CUDA_OK(cudaHostAlloc(
          reinterpret_cast<void**>(&h_expert_views_graph_), table_bytes,
          cudaHostAllocDefault));
      DGPP_CUDA_OK(cudaMalloc(&d_expert_views_graph_, table_bytes));
      graph_table_ready_.assign(static_cast<size_t>(graph_table_slots_),
                                false);
    }
  }
}

GlmMoeLayer::~GlmMoeLayer() {
  cudaFree(d_ids_);
  cudaFree(d_slot_row_);
  cudaFree(d_segs_);
  cudaFree(d_views_prefill_);
  cudaFree(d_tiles_);
  cudaFree(d_tile_count_);
  if (h_ids_pinned_) cudaFreeHost(h_ids_pinned_);
  if (h_weights_pinned_) cudaFreeHost(h_weights_pinned_);
  if (h_biased_pinned_) cudaFreeHost(h_biased_pinned_);
  if (h_seg_rows_) cudaFreeHost(h_seg_rows_);
  if (h_slot_row_) cudaFreeHost(h_slot_row_);
  if (h_segs_) cudaFreeHost(h_segs_);
  if (h_view_ring_) cudaFreeHost(h_view_ring_);
  for (int i = 0; i < kViewRing; ++i)
    if (view_ring_event_[i]) cudaEventDestroy(view_ring_event_[i]);
  cudaFree(d_weights_);
  cudaFree(d_biased_);
  cudaFree(d_scores_);
  cudaFree(d_rows_);
  cudaFree(d_row_w_);
  cudaFree(d_gather_);
  cudaFree(d_gate_);
  cudaFree(d_up_);
  cudaFree(d_act_);
  cudaFree(d_q_codes_);
  cudaFree(d_q_scales_);
  cudaFree(d_q_gs_);
  cudaFree(d_down_);
  cudaFree(d_acc_);
  cudaFree(d_slot_act_);
  cudaFree(d_slot_down_);
  cudaFree(d_slot_order_);
  cudaFree(d_router_counters_);
  cudaFree(d_expert_views_);
  cudaFree(d_expert_views_graph_);
  cudaFreeHost(h_expert_views_graph_);
}

void GlmMoeLayer::prepare_graph_table(int table_slot, cudaStream_t stream) {
  if (table_slot < 0 || table_slot >= graph_table_slots_ ||
      h_expert_views_graph_ == nullptr)
    throw std::invalid_argument(
        "GlmMoeLayer: graph table slot out of range (construct with "
        "graph_table_slots)");
  const size_t E3 = static_cast<size_t>(cfg_.n_experts) * 3;
  const size_t stride = E3 + 3;  // the (E+1)-entry table
  const size_t off = static_cast<size_t>(table_slot) * stride;
  MoeExpertView* src = h_expert_views_graph_ + off;
  for (size_t i = 0; i < E3; ++i)
    src[i] = w_.nvfp4() ? MoeExpertView::of(w_.experts_fp4[i])
             : w_.packq() ? MoeExpertView::of(w_.experts_packed[i])
                          : MoeExpertView::of(w_.experts[i]);
  size_t n = E3;
  if (w_.shared_nvfp4()) {
    for (int m = 0; m < 3; ++m) src[E3 + static_cast<size_t>(m)] = MoeExpertView::of(w_.shared_fp4[m]);
    n += 3;
  } else if (w_.shared_packq()) {
    for (int m = 0; m < 3; ++m) src[E3 + static_cast<size_t>(m)] = MoeExpertView::of(w_.shared_packed[m]);
    n += 3;
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(d_expert_views_graph_ + off, src,
                               sizeof(MoeExpertView) * n,
                               cudaMemcpyHostToDevice, stream));
  graph_table_ready_[static_cast<size_t>(table_slot)] = true;
}


// Every expert triple must share the routed geometry (the loader's
// contract: one slice width per rank); the shared triple has its own inter
// but the same hidden. Checked once per enqueue — the views can rebind.
void GlmMoeLayer::check_expert_geometry() const {
  const int H = cfg_.hidden, E = cfg_.n_experts;
  if (w_.packq()) {
    const GlmPackedMatrix& g0 = w_.experts_packed[0];
    if (g0.bits != 4 && g0.bits != 8)
      throw std::runtime_error("GlmMoeLayer: the packed routed width must be 4 or 8");
    for (int e = 0; e < E; ++e) {
      const GlmPackedMatrix* m = w_.experts_packed + static_cast<size_t>(e) * 3;
      if (m[0].cols != H || m[1].cols != H || m[0].rows != g0.rows ||
          m[1].rows != g0.rows || m[2].rows != H || m[2].cols != g0.rows ||
          m[0].bits != g0.bits || m[1].bits != g0.bits || m[2].bits != g0.bits ||
          m[0].scale_fmt != g0.scale_fmt || m[1].scale_fmt != g0.scale_fmt ||
          m[2].scale_fmt != g0.scale_fmt ||
          !m[0].packed || !m[1].packed || !m[2].packed || !m[0].scales || !m[1].scales ||
          !m[2].scales)
        throw std::runtime_error(
            "GlmMoeLayer: inconsistent packed routed expert matrices (expert " +
            std::to_string(e) + ")");
    }
    if (g0.scale_fmt != kPackedScaleBf16G64 && g0.scale_fmt != kPackedScaleF16G128)
      throw std::runtime_error("GlmMoeLayer: unknown packed routed scale format");
    if (has_shared()) {
      const GlmPackedMatrix* sh = w_.shared_packed;
      if (sh[0].rows != sh[1].rows || sh[2].cols != sh[0].rows || sh[2].rows != H ||
          sh[0].cols != H || sh[1].cols != H || !sh[0].packed || !sh[1].packed ||
          !sh[2].packed || !sh[0].scales || !sh[1].scales || !sh[2].scales)
        throw std::runtime_error("GlmMoeLayer: inconsistent packed shared matrices");
      if (sh[0].bits != 8 || sh[1].bits != 8 || sh[2].bits != 8)
        throw std::runtime_error("GlmMoeLayer: the packed shared expert is int8 (the slot kernels' width)");
      if (sh[0].scale_fmt != kPackedScaleBf16G64 || sh[1].scale_fmt != kPackedScaleBf16G64 ||
          sh[2].scale_fmt != kPackedScaleBf16G64)
        throw std::runtime_error("GlmMoeLayer: the packed shared expert carries bf16 scales per 64");
      // The slot kernels run the shared slot at the routed K.
      if (sh[2].cols != g0.rows)
        throw std::runtime_error(
            "GlmMoeLayer: the packed shared expert's inter must equal the routed experts' (the "
            "slot kernels' K)");
    }
    return;
  }
  if (w_.nvfp4()) {
    const GlmFp4Matrix& g0 = w_.experts_fp4[0];
    if (g0.scale_group != kFp4Group && g0.scale_group != kMxfp4Group)
      throw std::runtime_error("GlmMoeLayer: the fp4 scale group must be 16 (NVFP4) or 32 (MXFP4)");
    // NVFP4 matrices carry a global; MXFP4 ones (2026-09-13, DeepSeek-V4.1)
    // carry none — one group across the table.
    const bool needs_global = !g0.mxfp4();
    for (int e = 0; e < E; ++e) {
      const GlmFp4Matrix* m = w_.experts_fp4 + static_cast<size_t>(e) * 3;
      if (m[0].cols != H || m[1].cols != H || m[0].rows != g0.rows ||
          m[1].rows != g0.rows || m[2].rows != H || m[2].cols != g0.rows ||
          m[0].scale_group != g0.scale_group || m[1].scale_group != g0.scale_group ||
          m[2].scale_group != g0.scale_group ||
          (needs_global && (!m[0].global_scale || !m[1].global_scale || !m[2].global_scale)))
        throw std::runtime_error(
            "GlmMoeLayer: inconsistent fp4 routed expert matrices (expert " +
            std::to_string(e) + ")");
    }
    if (has_shared() && w_.shared_nvfp4()) {
      const GlmFp4Matrix* sh = w_.shared_fp4;
      if (sh[0].rows != sh[1].rows || sh[2].cols != sh[0].rows || sh[2].rows != H ||
          sh[0].cols != H || sh[1].cols != H ||
          sh[0].scale_group != g0.scale_group || sh[1].scale_group != g0.scale_group ||
          sh[2].scale_group != g0.scale_group ||
          (needs_global && (!sh[0].global_scale || !sh[1].global_scale || !sh[2].global_scale)))
        throw std::runtime_error("GlmMoeLayer: inconsistent NVFP4 shared matrices");
      // The slot kernels run the shared slot at the routed K (D3): the
      // shared down's K is the shared inter, which must be the routed's.
      if (sh[2].cols != g0.rows)
        throw std::runtime_error(
            "GlmMoeLayer: the NVFP4 shared expert's inter must equal the routed experts' (the "
            "slot kernels' K)");
    } else if (has_shared() &&
               (w_.shared[0].rows != w_.shared[1].rows ||
                w_.shared[2].cols != w_.shared[0].rows || w_.shared[2].rows != H ||
                w_.shared[0].cols != H || w_.shared[1].cols != H)) {
      throw std::runtime_error("GlmMoeLayer: inconsistent shared matrices");
    }
    return;
  }
  const GlmQuantMatrix& g0 = w_.experts[0];
  auto grid_ok = [](int b) { return b >= 16 && (b & (b - 1)) == 0 && b <= 128; };
  for (int e = 0; e < E; ++e) {
    const GlmQuantMatrix* m = w_.experts + static_cast<size_t>(e) * 3;
    if (m[0].cols != H || m[1].cols != H || m[0].rows != g0.rows ||
        m[1].rows != g0.rows || m[2].rows != H || m[2].cols != g0.rows)
      throw std::runtime_error(
          "GlmMoeLayer: inconsistent routed expert matrices (expert " +
          std::to_string(e) + ")");
    // One scale grid per matrix index across the experts (the sliced axis
    // re-blocked at a power of two >= 16: gate/up rows, down columns; the
    // other axis stays the checkpoint's 128).
    for (int i = 0; i < 3; ++i) {
      const GlmQuantMatrix& gi = w_.experts[i];
      if (m[i].scale_block_rows != gi.scale_block_rows || m[i].scale_block_cols != gi.scale_block_cols ||
          !grid_ok(m[i].scale_block_rows) || !grid_ok(m[i].scale_block_cols))
        throw std::runtime_error(
            "GlmMoeLayer: inconsistent or unsupported expert scale grid (expert " +
            std::to_string(e) + ")");
    }
  }
  if (has_shared() &&
      (w_.shared[0].rows != w_.shared[1].rows ||
       w_.shared[2].cols != w_.shared[0].rows || w_.shared[2].rows != H ||
       w_.shared[0].cols != H || w_.shared[1].cols != H))
    throw std::runtime_error("GlmMoeLayer: inconsistent shared matrices");
}

void GlmMoeLayer::enqueue(const uint16_t* hidden, uint16_t* out, int tokens,
                          cudaStream_t stream, MoeExpertKernel kernel) {
  enqueue_host(hidden, out, nullptr, tokens, stream, kernel);
}

void GlmMoeLayer::enqueue_f32(const uint16_t* hidden, float* out, int tokens,
                              cudaStream_t stream, MoeExpertKernel kernel) {
  enqueue_host(hidden, nullptr, out, tokens, stream, kernel);
}

void GlmMoeLayer::enqueue_host(const uint16_t* hidden, uint16_t* out_bf16,
                               float* out_f32, int tokens, cudaStream_t stream,
                               MoeExpertKernel kernel) {
  step_timing::Scope tick(step_timing::kMoe);
  if (tokens <= 0) return;
  if (tokens > max_tokens_)
    throw std::invalid_argument("GlmMoeLayer: tokens exceed max_tokens");
  if (!hidden || (!out_bf16 && !out_f32))
    throw std::invalid_argument("GlmMoeLayer: null pointer");
  const int E = cfg_.n_experts, K = cfg_.top_k;
  const bool shared = has_shared();
  check_expert_geometry();

  // 1. Router + one sync: the ids/weights round-trip is the diagnostic
  //    mode's cost; the production path keeps segmentation device-side.
  launch_moe_router(hidden, w_.router_gate, w_.router_bias, d_ids_,
                    d_weights_, d_scores_, d_biased_, cfg_, tokens, stream);
  const size_t tk = static_cast<size_t>(tokens) * K;
  DGPP_CUDA_OK(cudaMemcpyAsync(h_ids_pinned_, d_ids_, tk * 4,
                               cudaMemcpyDeviceToHost, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(h_weights_pinned_, d_weights_, tk * 4,
                               cudaMemcpyDeviceToHost, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(h_biased_pinned_, d_biased_,
                               static_cast<size_t>(tokens) * E * 4,
                               cudaMemcpyDeviceToHost, stream));
  {
    step_timing::Scope sync_tick(step_timing::kMoeSync);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  }
  h_ids_.assign(h_ids_pinned_, h_ids_pinned_ + tk);
  h_weights_.assign(h_weights_pinned_, h_weights_pinned_ + tk);
  h_biased_.assign(h_biased_pinned_,
                   h_biased_pinned_ + static_cast<size_t>(tokens) * E);

  // 2. Segment by expert (ascending expert id — the accumulation order)
  //    straight into the pinned staging: the routed rows in segment order,
  //    then every token once more for the shared expert; per (token, slot)
  //    its gathered row; the segment table; the expert views (the shared
  //    triple last). one upload of each per layer.
  std::fill(h_counts_.begin(), h_counts_.end(), 0);
  for (size_t i = 0; i < tk; ++i) ++h_counts_[h_ids_[i]];
  std::vector<int> seg_begin(E, 0);
  for (int e = 1; e < E; ++e) seg_begin[e] = seg_begin[e - 1] + h_counts_[e - 1];
  std::vector<int> fill(seg_begin.begin(), seg_begin.end());
  for (int t = 0; t < tokens; ++t)
    for (int i = 0; i < K; ++i) {
      const int e = h_ids_[static_cast<size_t>(t) * K + i];
      h_seg_rows_[fill[e]] = t;
      h_slot_row_[static_cast<size_t>(t) * K + i] = fill[e];
      ++fill[e];
    }
  // The shared segment (when the chain has one): every token once more,
  // after the routed rows; shared_row0 = -1 tells the accumulation to end
  // the chain after the routed slots.
  const int shared_row0 = shared ? static_cast<int>(tk) : -1;
  if (shared)
    for (int t = 0; t < tokens; ++t) h_seg_rows_[tk + t] = t;
  int n_segs = 0, max_rows = 1;
  for (int e = 0; e < E; ++e) {
    if (h_counts_[e] == 0) continue;
    h_segs_[n_segs++] = MoeSegment{seg_begin[e], h_counts_[e], e};
    max_rows = std::max(max_rows, h_counts_[e]);
  }
  if (shared) h_segs_[n_segs] = MoeSegment{shared_row0, tokens, E};
  const size_t rows_total = tk + (shared ? static_cast<size_t>(tokens) : 0);
  DGPP_CUDA_OK(cudaMemcpyAsync(d_rows_, h_seg_rows_, rows_total * 4,
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_slot_row_, h_slot_row_, tk * 4,
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_segs_, h_segs_,
                               static_cast<size_t>(n_segs + (shared ? 1 : 0)) *
                                   sizeof(MoeSegment),
                               cudaMemcpyHostToDevice, stream));
  upload_expert_views(d_views_prefill_, /*with_shared=*/shared, stream);
  const int tile_cap = tile_list_for(kernel, n_segs, static_cast<int>(tk), stream);

  // 3. The grouped chain: gather every row once; gate and up over the
  //    routed segments in one launch each and the shared segment in one
  //    more (its inter may differ); swiglu over every row; the down
  //    projection in the selected chain's FP32 or BF16 format.
  //    The inter dims come from the matrix views (the rank's slices).
  grouped_expert_chain(kernel, hidden, d_segs_, n_segs, max_rows,
                       shared ? d_segs_ + n_segs : nullptr, tokens, rows_total,
                       stream, tile_cap ? d_tiles_ : nullptr, d_tile_count_, tile_cap);

  // 4. The ordered accumulation: per token its K slots in ascending expert
  //    id, then the shared row (weight 1), the fmaf chain from zero, one
  //    rounding onto the wire buffer — or the chain unrounded, for a
  //    caller that continues it.
  accumulate_grouped(out_bf16, out_f32, shared_row0, tokens, stream);
}

void GlmMoeLayer::accumulate_grouped(uint16_t* out_bf16, float* out_f32, int shared_row0,
                                     int tokens, cudaStream_t stream) {
  const int H = cfg_.hidden, K = cfg_.top_k;
  if (down_bf16_) {
    const auto* down = reinterpret_cast<const uint16_t*>(d_down_);
    if (out_bf16)
      launch_moe_accum_ordered_bf16down(out_bf16, down, H, d_slot_row_, d_ids_, d_weights_, tokens,
                                        K, H, stream);
    else
      launch_moe_accum_ordered_f32_bf16down(out_f32, down, H, d_slot_row_, d_ids_, d_weights_,
                                            tokens, K, H, stream);
  } else if (out_bf16)
    launch_moe_accum_ordered(out_bf16, d_down_, H, d_slot_row_, d_ids_,
                             d_weights_, shared_row0, tokens, K,
                             static_cast<int>(H), stream);
  else
    launch_moe_accum_ordered_f32(out_f32, d_down_, H, d_slot_row_, d_ids_,
                                 d_weights_, shared_row0, tokens, K,
                                 static_cast<int>(H), stream);
}

void GlmMoeLayer::upload_expert_views(MoeExpertView* d_dst, bool with_shared,
                                      cudaStream_t stream) {
  const int E = cfg_.n_experts;
  const int slot = view_ring_next_;
  view_ring_next_ = (view_ring_next_ + 1) % kViewRing;
  // The entry's previous upload must have executed before the fill
  // overwrites its source; the host is only ever made to wait here when
  // it is kViewRing uploads ahead of the stream.
  if (view_ring_armed_[slot])
    DGPP_CUDA_OK(cudaEventSynchronize(view_ring_event_[slot]));
  MoeExpertView* h = h_view_ring_ + static_cast<size_t>(slot) * view_table_entries_;
  for (size_t i = 0; i < static_cast<size_t>(E) * 3; ++i)
    h[i] = w_.nvfp4() ? MoeExpertView::of(w_.experts_fp4[i])
           : w_.packq() ? MoeExpertView::of(w_.experts_packed[i])
                        : MoeExpertView::of(w_.experts[i]);
  size_t n = static_cast<size_t>(E) * 3;
  if (with_shared) {
    for (int w = 0; w < 3; ++w)
      h[n + static_cast<size_t>(w)] = w_.shared_nvfp4() ? MoeExpertView::of(w_.shared_fp4[w])
                                      : w_.shared_packq() ? MoeExpertView::of(w_.shared_packed[w])
                                                          : MoeExpertView::of(w_.shared[w]);
    n += 3;
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(d_dst, h, n * sizeof(MoeExpertView),
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaEventRecord(view_ring_event_[slot], stream));
  view_ring_armed_[slot] = true;
}

bool GlmMoeLayer::packq_tile_list() const {
  static const bool enabled = [] {
    const char* e = std::getenv("DGPP_MOE_TILE_LIST");
    return e == nullptr || e[0] != '0';  // default on; =0 keeps the max_rows grid
  }();
  return enabled && w_.packq() && w_.experts_packed[0].bits == 4 && packq_gemm_variant_default() == 1;
}

// Builds the routed segments' tile list on the stream when the chain's
// tensor-core launches take it and returns its capacity (0: not built).
int GlmMoeLayer::tile_list_for(MoeExpertKernel kernel, int n_segs, int routed_rows,
                               cudaStream_t stream) {
  if (kernel != MoeExpertKernel::kMma || !packq_tile_list() || n_segs <= 0) return 0;
  const int cap = moe_tile_list_capacity(n_segs, routed_rows, kPackqGemmWideRows);
  if (cap > tiles_cap_) throw std::logic_error("GlmMoeLayer: the tile list outgrew its buffer");
  launch_moe_tile_list(d_segs_, n_segs, kPackqGemmWideRows, d_tiles_, d_tile_count_, stream);
  return cap;
}

int GlmMoeLayer::routed_seg_max_rows(const MoeSegment* d_segs, int n_segs,
                                     int fallback, cudaStream_t stream) {
  static const bool enabled = [] {
    const char* e = std::getenv("DGPP_MOE_SEG_MAX");
    return e == nullptr || e[0] != '0';  // default on; =0 keeps tokens-wide
  }();
  if (!enabled || d_segs == nullptr || n_segs <= 0) return fallback;
  // h_segs_ is pinned staging sized to n_experts + 1 (the host path uploads
  // from it; here it is download scratch — the uses never interleave).
  DGPP_CUDA_OK(cudaMemcpyAsync(h_segs_, d_segs,
                               static_cast<size_t>(n_segs) * sizeof(MoeSegment),
                               cudaMemcpyDeviceToHost, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  int longest = 1;
  for (int e = 0; e < n_segs; ++e) longest = std::max(longest, h_segs_[e].rows);
  return longest;
}

void GlmMoeLayer::enqueue_prefill(const uint16_t* hidden, uint16_t* out,
                                  int tokens, MoeTraceStaging* trace,
                                  cudaStream_t stream) {
  step_timing::Scope tick(step_timing::kMoe);
  if (tokens <= 0) return;
  if (!has_shared())
    throw std::logic_error(
        "GlmMoeLayer: the prefill path without the shared expert is not wired "
        "(the Qwen engine milestone)");
  if (tokens > max_tokens_)
    throw std::invalid_argument("GlmMoeLayer: tokens exceed max_tokens");
  if (!hidden || !out)
    throw std::invalid_argument("GlmMoeLayer: null pointer");
  const int E = cfg_.n_experts, K = cfg_.top_k;
  check_expert_geometry();
  const size_t tk = static_cast<size_t>(tokens) * K;

  // 1. Router; the traces ride async copies into the caller's pinned
  //    staging (no round trip).
  launch_moe_router(hidden, w_.router_gate, w_.router_bias, d_ids_,
                    d_weights_, d_scores_, d_biased_, cfg_, tokens, stream);
  if (trace) {
    // ids and weights ride async copies into the pinned staging; the biased
    // scores only when the caller stages them (GLM-5.3 does, GLM-4.7 not).
    if (!trace->ids || !trace->weights)
      throw std::invalid_argument("GlmMoeLayer: incomplete trace staging");
    DGPP_CUDA_OK(cudaMemcpyAsync(trace->ids, d_ids_, tk * 4,
                                 cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(trace->weights, d_weights_, tk * 4,
                                 cudaMemcpyDeviceToHost, stream));
    if (trace->biased)
      DGPP_CUDA_OK(cudaMemcpyAsync(trace->biased, d_biased_,
                                   static_cast<size_t>(tokens) * E * 4,
                                   cudaMemcpyDeviceToHost, stream));
  }

  // 2. Segmentation on the device: rows, slot map, segment table (every
  //    expert, empty ones included; the shared segment last).
  launch_moe_segment(d_ids_, tokens, K, E, d_rows_, d_slot_row_, d_segs_,
                     stream);
  // The expert views: the same table the host path uploads, from the
  // upload ring (the host runs ahead of the stream here — no per-layer
  // sync — so the source must not be a single table; see h_view_ring_).
  upload_expert_views(d_views_prefill_, /*with_shared=*/true, stream);

  // 3. The grouped chain over every expert's segment (an empty one's
  //    blocks exit at once) and the shared segment.
  const int shared_row0 = static_cast<int>(tk);
  const size_t rows_total = tk + static_cast<size_t>(tokens);
  // Packed tensor-core partials are scaled per group without rounding
  // the weights; the smallest batches retain the GEMV launchers.
  const bool mma = mma_takes_grid() && (!w_.packq() || tokens >= kPackqMmaFromRows);
  const MoeExpertKernel kernel = mma ? MoeExpertKernel::kMma : MoeExpertKernel::kGemv;
  // With the tile list the grid no longer needs the longest segment (and
  // the host no longer syncs for it): any bound serves as max_rows.
  const int tile_cap = tile_list_for(kernel, E, static_cast<int>(tk), stream);
  grouped_expert_chain(kernel, hidden, d_segs_, E,
                       tile_cap ? std::max(tokens, 1)
                                : routed_seg_max_rows(d_segs_, E, std::max(tokens, 1), stream),
                       d_segs_ + E, tokens, rows_total, stream, tile_cap ? d_tiles_ : nullptr,
                       d_tile_count_, tile_cap);
  accumulate_grouped(out, nullptr, shared_row0, tokens, stream);
}

bool GlmMoeLayer::mma_takes_grid() const {
  // The fp4 tile kernel reads NVFP4 and (2026-09-14) MXFP4 tables.
  if (w_.experts_fp4) return true;
  if (w_.experts_packed)
    return cfg_.hidden % kPackedGroup == 0 && w_.experts_packed[0].rows % kPackedGroup == 0;
  const int H = cfg_.hidden;
  const int I_r = static_cast<int>(w_.experts[0].rows);
  const int br = w_.experts[0].scale_block_rows, bc = w_.experts[0].scale_block_cols;
  if (br == 128 && bc == 128) return true;
  return (H % 32) == 0 && (I_r % 32) == 0 && br >= 32 && bc >= 32;
}

void GlmMoeLayer::enqueue_prefill_f32(const uint16_t* hidden, float* out, int tokens,
                                      MoeTraceStaging* trace, cudaStream_t stream,
                                      MoeExpertKernel kernel) {
  step_timing::Scope tick(step_timing::kMoe);
  if (tokens <= 0) return;
  if (tokens > max_tokens_)
    throw std::invalid_argument("GlmMoeLayer: tokens exceed max_tokens");
  if (!hidden || !out)
    throw std::invalid_argument("GlmMoeLayer: null pointer");
  const int E = cfg_.n_experts, K = cfg_.top_k;
  check_expert_geometry();
  const size_t tk = static_cast<size_t>(tokens) * K;
  launch_moe_router(hidden, w_.router_gate, w_.router_bias, d_ids_,
                    d_weights_, d_scores_, d_biased_, cfg_, tokens, stream);
  if (trace) {
    // ids and weights ride async copies into the pinned staging; the biased
    // scores only when the caller stages them (the Qwen model does not).
    if (!trace->ids || !trace->weights)
      throw std::invalid_argument("GlmMoeLayer: incomplete trace staging");
    DGPP_CUDA_OK(cudaMemcpyAsync(trace->ids, d_ids_, tk * 4,
                                 cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(trace->weights, d_weights_, tk * 4,
                                 cudaMemcpyDeviceToHost, stream));
    if (trace->biased)
      DGPP_CUDA_OK(cudaMemcpyAsync(trace->biased, d_biased_,
                                   static_cast<size_t>(tokens) * E * 4,
                                   cudaMemcpyDeviceToHost, stream));
  }
  launch_moe_segment(d_ids_, tokens, K, E, d_rows_, d_slot_row_, d_segs_,
                     stream);
  upload_expert_views(d_views_prefill_, /*with_shared=*/false, stream);
  // The routed chain alone (no shared segment) on the tensor-core kernel,
  // the fp32 chain handed back unrounded (shared_row0 < 0).
  const int tile_cap = tile_list_for(kernel, E, static_cast<int>(tk), stream);
  grouped_expert_chain(kernel, hidden, d_segs_, E,
                       tile_cap ? std::max(tokens, 1)
                                : routed_seg_max_rows(d_segs_, E, std::max(tokens, 1), stream),
                       /*shared_seg=*/nullptr, tokens, tk, stream, tile_cap ? d_tiles_ : nullptr,
                       d_tile_count_, tile_cap);
  accumulate_grouped(nullptr, out, /*shared_row0=*/-1, tokens, stream);
}

void GlmMoeLayer::ensure_w4a4(size_t rows, int k) {
  if (rows <= q_rows_cap_ && static_cast<size_t>(k) <= q_k_cap_) return;
  const size_t r = std::max(rows, q_rows_cap_), kk = std::max(static_cast<size_t>(k), q_k_cap_);
  cudaFree(d_q_codes_);
  cudaFree(d_q_scales_);
  cudaFree(d_q_gs_);
  DGPP_CUDA_OK(cudaMalloc(&d_q_codes_, r * kk / 2));
  DGPP_CUDA_OK(cudaMalloc(&d_q_scales_, r * nvfp4_act_scale_stride(static_cast<int>(kk))));
  DGPP_CUDA_OK(cudaMalloc(&d_q_gs_, r * sizeof(float)));
  q_rows_cap_ = r;
  q_k_cap_ = kk;
}

// The routed NVFP4 experts' prefill GEMMs on the native block-scaled FP4
// tensor cores (src/kernels/moe_w4a4.cu), activations quantized to NVFP4 per
// row with the checkpoint's static input_scale -- what SGLang's
// flashinfer_cutlass MoE runs for these checkpoints. On by default where the
// loader provides that calibrated scale (Qwen3.8 NVFP4); DGPP_MOE_W4A4=0 keeps
// W4A16 everywhere, DGPP_MOE_W4A4=1 also takes it without a static scale
// (dynamic per-row activation scale). Chains of at least
// DGPP_MOE_W4A4_MIN_ROWS routed rows (default 256), eager only.
static int moe_w4a4_mode() {  // 0 off, 1 forced on, -1 default (on with a static scale)
  static const int v = [] {
    const char* e = std::getenv("DGPP_MOE_W4A4");
    return e == nullptr || *e == '\0' ? -1 : e[0] == '0' ? 0 : 1;
  }();
  return v;
}
// DGPP_MOE_W4A4_DYNAMIC=1: a per-row dynamic activation global instead of
// the checkpoint's static input_scale (the default, SGLang's form).
static bool moe_w4a4_static() {
  static const bool v = std::getenv("DGPP_MOE_W4A4_DYNAMIC") == nullptr;
  return v;
}
static size_t moe_w4a4_min_rows() {
  static const size_t v = [] {
    const char* e = std::getenv("DGPP_MOE_W4A4_MIN_ROWS");
    return e != nullptr ? static_cast<size_t>(std::atoll(e)) : size_t{256};
  }();
  return v;
}

static bool moe_w4a4_eligible(bool calibrated, int hidden, int inter, size_t rows) {
  const int mode = moe_w4a4_mode();
  return (mode == 1 || (mode == -1 && calibrated)) && rows >= moe_w4a4_min_rows() && hidden > 0 &&
         inter > 0 && hidden % 64 == 0 && inter % 64 == 0 && hidden <= 16384 && inter <= 16384;
}

size_t GlmMoeLayer::w4a4_scratch_bytes(const GlmMoeConfig& cfg, int max_tokens, bool calibrated) {
  const size_t rows =
      static_cast<size_t>(std::max(max_tokens, 0)) * (cfg.top_k + cfg.n_shared_experts);
  if (!moe_w4a4_eligible(calibrated, cfg.hidden, cfg.inter, rows)) return 0;
  const int k = std::max(cfg.hidden, cfg.inter);
  return rows * (static_cast<size_t>(k) / 2 + nvfp4_act_scale_stride(k) + sizeof(float));
}

void GlmMoeLayer::grouped_expert_chain(MoeExpertKernel kernel,
                                       const uint16_t* hidden,
                                       const MoeSegment* segs, int n_segs,
                                       int max_rows, const MoeSegment* shared_seg,
                                       int tokens, size_t rows_total,
                                       cudaStream_t stream, const MoeTile* tiles,
                                       const int32_t* tile_count, int tile_cap) {
  const int H = static_cast<int>(cfg_.hidden);
  const bool fp4 = w_.nvfp4();
  const bool packq = w_.packq();
  const bool shared_fp4 = w_.shared_nvfp4();
  const int I_r = static_cast<int>(fp4 ? w_.experts_fp4[0].rows
                                   : packq ? w_.experts_packed[0].rows
                                           : w_.experts[0].rows);
  // shared_seg == nullptr: the chain has no shared expert (n_shared_experts 0).
  const int I_s = shared_seg ? static_cast<int>(shared_fp4 ? w_.shared_fp4[0].rows
                                                : packq ? w_.shared_packed[0].rows
                                                        : w_.shared[0].rows)
                             : 0;
  const int routed_bits = packq ? w_.experts_packed[0].bits : 0;
  const int shared_bits = (packq && shared_seg) ? w_.shared_packed[0].bits : 0;
  const int routed_sf = packq ? w_.experts_packed[0].scale_fmt : 0;  // the shared triple: format 0
  const int fp4_group = fp4 ? w_.experts_fp4[0].scale_group : kFp4Group;
  const size_t I_max = static_cast<size_t>(std::max(I_r, I_s));
  const bool w4a4 = fp4 && !packq && fp4_group == kFp4Group && kernel == MoeExpertKernel::kMma &&
                    moe_w4a4_eligible(w_.act_scales_dev != nullptr, H, I_r, rows_total);
  // The W4A4 chain's down rows in bf16 (half the write and the ordered
  // accumulation's read; SGLang's CUTLASS MoE keeps a bf16 intermediate too)
  // unless DGPP_MOE_W4A4_F32_DOWN=1; only without a shared segment, whose
  // rows would share the buffer in fp32.
  static const bool f32_down = std::getenv("DGPP_MOE_W4A4_F32_DOWN") != nullptr;
  down_bf16_ = w4a4 && shared_seg == nullptr && !f32_down;
  if (w4a4) {
    static const bool logged = [&] {
      DGPP_LOG_INFO("moe: W4A4 NVFP4 prefill experts on ({} activation scale; DGPP_MOE_W4A4=0 turns it off)",
                    moe_w4a4_static() && w_.act_scales_dev != nullptr ? "the checkpoint's static" : "a dynamic per-row");
      return true;
    }();
    (void)logged;
    ensure_w4a4(std::max(static_cast<size_t>(tokens), rows_total), std::max(H, I_r));
    launch_quantize_rows_nvfp4(hidden, static_cast<size_t>(H), tokens, H, d_q_codes_, d_q_scales_, d_q_gs_,
                               stream, 0.f, moe_w4a4_static() ? w_.act_scales_dev : nullptr);
  }
  // The shared segment is every token: split across blocks along z (the
  // GEMV core in 16-row pieces, the tensor-core kernel in whole m-tiles).
  const bool mma = kernel == MoeExpertKernel::kMma;
  // The tensor-core kernels read 128 x 128 block scales; a re-blocked TP
  // slice (a sub-128 grid on the sliced axis) takes the GEMV core.
  // A re-blocked grid (a TP slice at gcd(128, I/W), plan D2) runs on the
  // ldmatrix fp8 tile kernel, which reads per-row scales and one scale
  // column per 32-deep stage — every k a multiple of 32; the
  // older tile kernel (other widths) knows the 128 grid only.
  if (mma && !mma_takes_grid())
    throw std::invalid_argument(
        "GlmMoeLayer: the tensor-core expert kernel needs the 128x128 scale grid, or a "
        "32/64 grid with hidden and the slice both multiples of 32; use "
        "MoeExpertKernel::kGemv");
  const int shared_split = mma ? 128 : 16;
  // The GEMV core reads a gathered copy of the rows; the tensor-core kernel
  // reads the hidden rows through the row map directly (2026-09-05: the
  // gather was 1.26 ms per layer at 2048 tokens).
  if (!mma)
    launch_moe_gather_rows(hidden, d_rows_, d_gather_, static_cast<int>(rows_total),
                           H, stream);
  // `routed` selects the routed experts' kernel family: the fp4 kernels
  // (tensor-core or GEMV) for NVFP4 tables, the fp8 ones otherwise; the
  // shared segment (FP8 under both formats) always takes the fp8 kernels.
  // The fp4 tile kernel is the ldmatrix kernel on both shapes (2026-09-08
  // evening, moe_tile_bench: gate 2.37 vs the reference tile's 3.70 ms per
  // launch at 2,048 tokens, 5.0 vs 9.9 at 8,192; down 3.21 vs the two-stage
  // kernel's 3.84, 7.89 vs 9.58) — launch_moe_grouped_mma_fp4_{bf16,f32}.
  // `routed` selects the segment class; an NVFP4 shared expert (GLM-4.7,
  // view-table entry E) takes the fp4 kernels like the routed segments.
  auto gemm_bf16 = [&](const MoeSegment* sg, int ns, int mr, int split, int which,
                       uint16_t* out, int n, bool routed_arg) {
    const bool routed = routed_arg || shared_fp4;
    if (packq && mma)
      launch_moe_grouped_mma_packq_bf16(hidden, H, sg, ns, mr, d_views_prefill_, which, out, I_max,
                                        n, H, routed_arg ? routed_bits : shared_bits, stream,
                                        d_rows_, routed_arg ? routed_sf : 0, -1,
                                        routed_arg ? tiles : nullptr, tile_count, tile_cap);
    else if (packq)
      launch_moe_grouped_gemv_packq_bf16(d_gather_, H, sg, ns, mr, split, d_views_prefill_,
                                         which, out, I_max, n, H,
                                         routed_arg ? routed_bits : shared_bits, stream,
                                         routed_arg ? routed_sf : 0);
    else if (w4a4 && routed_arg)
      launch_moe_grouped_w4a4_bf16(d_q_codes_, d_q_scales_, d_q_gs_, d_rows_, sg, ns, mr, d_views_prefill_, which, out,
                                   I_max, n, H, stream);
    else if (mma && routed && fp4)
      launch_moe_grouped_mma_fp4_bf16(hidden, H, sg, ns, mr, split, d_views_prefill_,
                                      which, out, I_max, n, H, stream, d_rows_, fp4_group);
    else if (mma)
      launch_moe_grouped_mma_bf16(hidden, H, sg, ns, mr, split, d_views_prefill_,
                                  which, out, I_max, n, H, stream, d_rows_);
    else if (routed && fp4)
      launch_moe_grouped_gemv_fp4_bf16(d_gather_, H, sg, ns, mr, split, d_views_prefill_,
                                       which, out, I_max, n, H, stream, fp4_group);
    else
      launch_moe_grouped_gemv_bf16(d_gather_, H, sg, ns, mr, split, d_views_prefill_,
                                   which, out, I_max, n, H, stream);
  };
  auto gemm_f32 = [&](const MoeSegment* sg, int ns, int mr, int split, int k,
                      bool routed_arg) {
    const bool routed = routed_arg || shared_fp4;
    if (packq && mma)
      launch_moe_grouped_mma_packq_f32(d_act_, I_max, sg, ns, mr, d_views_prefill_, 2, d_down_, H,
                                       H, k, routed_arg ? routed_bits : shared_bits, stream,
                                       nullptr, routed_arg ? routed_sf : 0, -1,
                                       routed_arg ? tiles : nullptr, tile_count, tile_cap);
    else if (packq)
      launch_moe_grouped_gemv_packq_f32(d_act_, I_max, sg, ns, mr, split, d_views_prefill_,
                                        2, d_down_, H, H, k,
                                        routed_arg ? routed_bits : shared_bits, stream,
                                        routed_arg ? routed_sf : 0);
    else if (w4a4 && routed_arg && down_bf16_)
      launch_moe_grouped_w4a4_bf16(d_q_codes_, d_q_scales_, d_q_gs_, nullptr, sg, ns, mr, d_views_prefill_, 2,
                                   reinterpret_cast<uint16_t*>(d_down_), H, H, k, stream);
    else if (w4a4 && routed_arg)
      launch_moe_grouped_w4a4_f32(d_q_codes_, d_q_scales_, d_q_gs_, nullptr, sg, ns, mr, d_views_prefill_, 2, d_down_, H, H,
                                  k, stream);
    else if (mma && routed && fp4)
      launch_moe_grouped_mma_fp4_f32(d_act_, I_max, sg, ns, mr, split, d_views_prefill_,
                                     2, d_down_, H, H, k, stream, nullptr, fp4_group);
    else if (mma)
      launch_moe_grouped_mma_f32(d_act_, I_max, sg, ns, mr, split, d_views_prefill_,
                                 2, d_down_, H, H, k, stream);
    else if (routed && fp4)
      launch_moe_grouped_gemv_fp4_f32(d_act_, I_max, sg, ns, mr, split, d_views_prefill_,
                                      2, d_down_, H, H, k, stream, fp4_group);
    else
      launch_moe_grouped_gemv_f32(d_act_, I_max, sg, ns, mr, split, d_views_prefill_,
                                  2, d_down_, H, H, k, stream);
  };
  gemm_bf16(segs, n_segs, max_rows, 0, 0, d_gate_, I_r, true);
  if (shared_seg) gemm_bf16(shared_seg, 1, tokens, shared_split, 0, d_gate_, I_s, false);
  gemm_bf16(segs, n_segs, max_rows, 0, 1, d_up_, I_r, true);
  if (shared_seg) gemm_bf16(shared_seg, 1, tokens, shared_split, 1, d_up_, I_s, false);
  // The down projection's activations to NVFP4 (routed rows; the shared
  // expert's rows, when present, sit past them and keep the bf16 path).
  // Without a shared segment the bf16 act buffer feeds nothing but that
  // quantizer, so the activation runs inside it (bitwise the two-kernel chain;
  // DGPP_MOE_SWIGLU_QUANT=0 keeps the two launches).
  static const bool fused_act = [] {
    const char* e = std::getenv("DGPP_MOE_SWIGLU_QUANT");
    return e == nullptr || e[0] != '0';
  }();
  const bool fused_swiglu =
      w4a4 && shared_seg == nullptr && fused_act && I_max == static_cast<size_t>(I_r);
  if (fused_swiglu) {
    launch_swiglu_quantize_rows_nvfp4(d_gate_, d_up_, I_max, static_cast<int>(rows_total), I_r, cfg_.swiglu_limit,
                                      d_q_codes_, d_q_scales_, d_q_gs_, stream,
                                      0.f, moe_w4a4_static() && w_.act_scales_dev ? w_.act_scales_dev + 1 : nullptr);
  } else {
    launch_moe_swiglu_clamp(d_gate_, d_up_, d_act_,
                            static_cast<int64_t>(rows_total) * I_max,
                            cfg_.swiglu_limit, stream);
    if (w4a4)
      launch_quantize_rows_nvfp4(d_act_, I_max, static_cast<int>(rows_total), I_r, d_q_codes_, d_q_scales_, d_q_gs_,
                                 stream, 0.f, moe_w4a4_static() && w_.act_scales_dev ? w_.act_scales_dev + 1 : nullptr);
  }
  gemm_f32(segs, n_segs, max_rows, 0, I_r, true);
  if (shared_seg) gemm_f32(shared_seg, 1, tokens, shared_split, I_s, false);
  if (std::getenv("DGPP_MOE_CHAIN_DUMP") != nullptr) {
    // Hunt instrument: per-stage checksums of the chain's buffers.
    // Materialize stages bypassed by the optimized chain before reading them.
    if (mma)
      launch_moe_gather_rows(hidden, d_rows_, d_gather_, static_cast<int>(rows_total), H, stream);
    if (fused_swiglu)
      launch_moe_swiglu_clamp(d_gate_, d_up_, d_act_, static_cast<int64_t>(rows_total) * I_max,
                              cfg_.swiglu_limit, stream);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    auto sum_bf16 = [&](const uint16_t* d, size_t n) {
      std::vector<uint16_t> h(n);
      DGPP_CUDA_OK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
      double acc = 0;
      for (uint16_t v : h) acc += std::fabs(bf16_bits_to_float(v));
      return acc;
    };
    auto sum_f32 = [&](const float* d, size_t n) {
      std::vector<float> h(n);
      DGPP_CUDA_OK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
      double acc = 0;
      for (float v : h) acc += std::fabs(v);
      return acc;
    };
    std::vector<MoeSegment> hs(static_cast<size_t>(n_segs) + (shared_seg ? 1 : 0));
    DGPP_CUDA_OK(cudaMemcpy(hs.data(), segs, n_segs * sizeof(MoeSegment),
                            cudaMemcpyDeviceToHost));
    if (shared_seg)
      DGPP_CUDA_OK(cudaMemcpy(&hs[n_segs], shared_seg, sizeof(MoeSegment),
                              cudaMemcpyDeviceToHost));
    std::string segtxt;
    for (const MoeSegment& sg : hs)
      if (sg.rows > 0)
        segtxt += " (" + std::to_string(sg.row0) + "," + std::to_string(sg.rows) + ",e" +
                  std::to_string(sg.expert) + ")";
    DGPP_LOG_INFO(
        "[chain {}] rows_total={} I_r={} I_s={} segs:{} | gather {:.6g} gate {:.6g} "
        "up {:.6g} act {:.6g} down {:.6g}",
        mma ? "mma" : "gemv", rows_total, I_r, I_s, segtxt, sum_bf16(d_gather_, rows_total * H),
        sum_bf16(d_gate_, rows_total * I_max), sum_bf16(d_up_, rows_total * I_max),
        sum_bf16(d_act_, rows_total * I_max),
        down_bf16_ ? sum_bf16(reinterpret_cast<const uint16_t*>(d_down_), rows_total * H)
                   : sum_f32(d_down_, rows_total * H));
  }
}

void GlmMoeLayer::enqueue_decode(const uint16_t* hidden, uint16_t* out,
                                 int tokens, MoeTraceStaging* trace,
                                 cudaStream_t stream, int table_slot) {
  if (!has_shared())
    throw std::logic_error(
        "GlmMoeLayer: enqueue_decode needs the shared expert in the chain "
        "(enqueue_decode_f32 runs the routed chain alone)");
  enqueue_decode_impl(hidden, out, nullptr, tokens, trace, stream, table_slot);
}

void GlmMoeLayer::enqueue_decode_f32(const uint16_t* hidden, float* out,
                                     int tokens, MoeTraceStaging* trace,
                                     cudaStream_t stream, int table_slot, bool accumulate) {
  enqueue_decode_impl(hidden, nullptr, out, tokens, trace, stream, table_slot, accumulate);
}

void GlmMoeLayer::enqueue_decode_impl(const uint16_t* hidden, uint16_t* out_bf16,
                                      float* out_f32, int tokens,
                                      MoeTraceStaging* trace, cudaStream_t stream,
                                      int table_slot, bool accumulate) {
  step_timing::Scope tick(step_timing::kMoe);
  if (tokens <= 0) return;
  const bool with_shared = out_bf16 != nullptr;
  if (with_shared == (out_f32 != nullptr))
    throw std::invalid_argument("GlmMoeLayer: exactly one decode output");
  if (with_shared && !has_shared())
    throw std::logic_error("GlmMoeLayer: no shared expert for the full chain");
  if (decode_slots_ <= 0)
    throw std::runtime_error(
        "GlmMoeLayer: decode path not provisioned (construct with "
        "decode_slots > 0)");
  if (tokens > decode_slots_)
    throw std::invalid_argument(
        "GlmMoeLayer: decode rows exceed decode_slots");
  if (!hidden)
    throw std::invalid_argument("GlmMoeLayer: null pointer");
  const int H = cfg_.hidden, E = cfg_.n_experts, K = cfg_.top_k;
  check_expert_geometry();

  // 1. Router, dots and selection in one launch — ids ASCENDING per row,
  //    on device.
  launch_moe_router(hidden, w_.router_gate, w_.router_bias, d_ids_,
                    d_weights_, d_scores_, d_biased_, cfg_, tokens, stream,
                    d_router_counters_);
  // 2. Route traces ride ASYNC copies into the caller's pinned staging;
  //    the caller materializes them after its next stream sync (the
  //    decode step's final sync). No round-trip on the hot path.
  if (trace) {
    if (!trace->ids || !trace->weights || !trace->biased)
      throw std::invalid_argument("GlmMoeLayer: incomplete trace staging");
    DGPP_CUDA_OK(cudaMemcpyAsync(trace->ids, d_ids_,
                                  static_cast<size_t>(tokens) * K * 4,
                                  cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(trace->weights, d_weights_,
                                  static_cast<size_t>(tokens) * K * 4,
                                  cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(trace->biased, d_biased_,
                                  static_cast<size_t>(tokens) * E * 4,
                                  cudaMemcpyDeviceToHost, stream));
  }

  // 3. The slot chain. Slot layout: tokens*(K+1); slot t*(K+1)+j is row
  //    t's routed expert j (ascending id — the router's contract) and
  //    slot ..+K is the shared expert. EAGER: the expert-view table is
  //    RE-UPLOADED EVERY CALL (see the member's comment: the streaming
  //    loader makes binding-keyed caching a wrong-weights factory; one
  //    small async upload per layer per step is the honest price).
  //    CAPTURE (table_slot >= 0): the kernels read the slot's OWN device
  //    table, prepared before the capture — no upload node at all. A
  //    slot never prepared is refused (the wrong-weights class the 4c
  //    cache bug taught, caught at capture time instead of in the
  //    transcript).
  const MoeExpertView* table = d_expert_views_;
  if (table_slot >= 0) {
    if (table_slot >= graph_table_slots_ || d_expert_views_graph_ == nullptr)
      throw std::invalid_argument(
          "GlmMoeLayer: graph table slot out of range (construct with "
          "graph_table_slots)");
    if (!graph_table_ready_[static_cast<size_t>(table_slot)])
      throw std::logic_error(
          "GlmMoeLayer: graph table slot " + std::to_string(table_slot) +
          " was not prepared (call prepare_graph_table before capturing)");
    table = d_expert_views_graph_ +
            static_cast<size_t>(table_slot) * static_cast<size_t>(E + 1) * 3;
  } else {
    upload_expert_views(d_expert_views_, /*with_shared=*/w_.shared_nvfp4() || w_.shared_packq(),
                        stream);
  }

  const int slots = tokens * (K + 1);
  const bool fp4 = w_.nvfp4();
  const bool packq = w_.packq();
  const int I_r = static_cast<int>(fp4 ? w_.experts_fp4[0].rows
                                   : packq ? w_.experts_packed[0].rows
                                           : w_.experts[0].rows);  // routed inter slice
  // The shared inter slice; 0 = no shared slot (the routed chain alone).
  // An NVFP4 shared expert (GLM-4.7) is view-table entry E, read through
  // the fp4 core; the FP8 one rides the launch arguments.
  const bool shared_fp4 = with_shared && w_.shared_nvfp4();
  const int shared_view_base = shared_fp4 ? E * 3 : -1;
  const int I_s = with_shared ? static_cast<int>(shared_fp4 ? w_.shared_fp4[0].rows
                                                 : packq ? w_.shared_packed[0].rows
                                                         : w_.shared[0].rows)
                              : 0;
  const bool sh_args = with_shared && !shared_fp4 && !packq;
  // The fp8 shared expert's own scale grid (the checkpoint's 128 x 128,
  // or the DeepSeek-V4.1 release's 32 x 32).
  const auto log2_of = [](int b) {
    int s = 0;
    while ((1 << s) < b) ++s;
    return s;
  };
  const int sh_rs = sh_args ? log2_of(w_.shared[0].scale_block_rows) : 7;
  const int sh_cs = sh_args ? log2_of(w_.shared[0].scale_block_cols) : 7;
  const uint8_t* sh_gate_p = sh_args ? w_.shared[0].payload : nullptr;
  const float* sh_gate_s = sh_args ? w_.shared[0].scales : nullptr;
  const uint8_t* sh_up_p = sh_args ? w_.shared[1].payload : nullptr;
  const float* sh_up_s = sh_args ? w_.shared[1].scales : nullptr;
  const uint8_t* sh_down_p = sh_args ? w_.shared[2].payload : nullptr;
  const float* sh_down_s = sh_args ? w_.shared[2].scales : nullptr;
  // Multi-token batches (a speculative verify) run their slots in
  // expert order so an expert two rows share is read from DRAM once (see
  // launch_moe_slot_order); one token has nothing to share.
  const int32_t* order = nullptr;
  if (tokens > 1) {
    launch_moe_slot_order(d_ids_, d_slot_order_, slots, K, E, stream);
    order = d_slot_order_;
  }
  // Gate + up + swiglu in one launch (bit-identical to the three-launch
  // chain — see the launcher). Per-slot bounds are consumed downstream
  // (the down GEMV reads only k=I_s of the shared slot).
  // The shared slot's gate/up k and down n (0: no shared slot — its blocks
  // find n == 0 and return).
  const int K_s = with_shared ? H : 0;
  const int N_s = with_shared ? H : 0;
  if (packq) {
    // The packed table: the shared expert (int8) is view-table entry E,
    // read at the routed K; no launch-argument matrices.
    const int rb = w_.experts_packed[0].bits;
    const int rsf = w_.experts_packed[0].scale_fmt;
    const int sb = with_shared ? w_.shared_packed[0].bits : 0;
    const int sbase = with_shared ? E * 3 : -1;
    launch_moe_slot_gate_up_swiglu_packq(hidden, H, d_ids_, order, table, I_r, H, rb, I_s, sb,
                                         d_slot_act_, I_r, slots, K, cfg_.swiglu_limit, stream,
                                         sbase, rsf);
    launch_moe_slot_down_packq(d_slot_act_, I_r, d_ids_, order, table, H, I_r, rb, N_s, sb,
                               d_slot_down_, H, slots, K, stream, sbase, rsf);
  } else if (fp4) {
    const int fp4_group = w_.experts_fp4[0].scale_group;
    launch_moe_slot_gate_up_swiglu_fp4(
        hidden, H, d_ids_, order, table, I_r, H, I_s, K_s, sh_gate_p, sh_gate_s,
        sh_up_p, sh_up_s, d_slot_act_, I_r, slots, K, cfg_.swiglu_limit, stream,
        shared_view_base, fp4_group, sh_rs, sh_cs);
    launch_moe_slot_down_fp4(d_slot_act_, I_r, d_ids_, order, table, H, I_r, N_s,
                             I_s, sh_down_p, sh_down_s, d_slot_down_, H, slots, K,
                             stream, shared_view_base, fp4_group, sh_rs, sh_cs);
  } else {
    launch_moe_slot_gate_up_swiglu(
        hidden, H, d_ids_, order, table, I_r, H, I_s, K_s, sh_gate_p, sh_gate_s,
        sh_up_p, sh_up_s, d_slot_act_, I_r, slots, K, cfg_.swiglu_limit, stream, sh_rs,
        sh_cs);
    launch_moe_slot_down(d_slot_act_, I_r, d_ids_, order, table, H, I_r, N_s,
                         I_s, sh_down_p, sh_down_s, d_slot_down_, H, slots, K,
                         stream, sh_rs, sh_cs);
  }
  if (with_shared)
    launch_moe_slot_accum(out_bf16, d_slot_down_, d_weights_, tokens, H, K, stream);
  else if (accumulate)
    launch_moe_slot_accum_routed_f32(out_f32, d_slot_down_, d_weights_, tokens, H,
                                     K, stream);
  // else: the consumer folds the accumulation from decode_slot_down() /
  // decode_weights() (the Qwen shared tail, 2026-09-29).
}

}  // namespace dgpp
