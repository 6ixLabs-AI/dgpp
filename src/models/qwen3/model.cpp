// UNVERIFIED DRAFT (2026-10-04): see models/qwen3/model.hpp. Never compiled
// with nvcc / GCC, never run.
#include "models/qwen3/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/cuda_check.hpp"
#include "kernels/glm4_attn.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/kernels.hpp"

namespace dgpp {
namespace {

template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  return p;
}

template <class T>
T* pinned_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&p), std::max<size_t>(n, 1) * sizeof(T),
                             cudaHostAllocMapped));
  return p;
}

int64_t round_to_blocks(int64_t tokens, int block) {
  return ((tokens + block - 1) / block) * block;
}

}  // namespace

Qwen3Model::Qwen3Model(const Qwen3TextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
                       int64_t max_cache_tokens, LoaderResidency residency, BoundaryReducer* boundary,
                       int tp_rank, int tp_world, int max_requests, int decode_rows, bool serving_logits)
    : cfg_(cfg),
      attn_cfg_(cfg.attention_view()),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency,
              tp_world > 1 ? LoaderHeadSharding::VocabSharded : LoaderHeadSharding::Full) {
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("Qwen3Model: decode_rows exceeds the limit of 32");
  if (max_tokens <= 0) throw std::invalid_argument("Qwen3Model: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("Qwen3Model: max_requests must be in [1, kPickMaxRequests]");
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument(
        "Qwen3Model: a boundary reducer is required exactly when tp_world > 1 — without one the "
        "block-boundary partials would be returned silently as results");
  if (cfg_.eos_token_ids.empty()) throw std::invalid_argument("Qwen3Model: the config names no EOS token");
  if (cfg_.head_dim != kGlm4HeadDim) throw std::invalid_argument("Qwen3Model: 128-wide heads (the kernels' shape)");
  if (!cfg_.moe()) throw std::invalid_argument("Qwen3Model: the dense dialect has no GPU path");
  init_stream();
  loader_.set_reader_stream(stream_);
  const int H = cfg_.hidden_size;
  globals_ = loader_.load_globals();
  {
    SessionParams sp;
    sp.max_tokens = max_tokens;
    sp.max_cache_tokens = round_to_blocks(std::max<int64_t>(max_cache_tokens, max_tokens), kBlockTokens);
    sp.rank = tp_rank;
    sp.world = tp_world;
    sp.boundary = boundary;
    sp.max_requests = max_requests;
    sp.decode_rows = decode_rows;
    sp.logits_rows = compact_logits_rows(serving_logits, decode_rows, max_requests);
    sp.mtp = false;
    sp.vocab_size = cfg_.vocab_size;
    sp.hidden = H;
    sp.lm_vocab_begin = globals_.lm_vocab_begin;
    sp.lm_vocab_count = globals_.lm_vocab_count > 0 ? globals_.lm_vocab_count : cfg_.vocab_size;
    sp.max_position_embeddings = cfg_.max_position_embeddings;
    sp.block_tokens = kBlockTokens;
    sp.snapshot_align = 1;  // no recurrent state: any position snapshots
    sp.draft_width = H;
    sp.eos = static_cast<int32_t>(cfg_.eos_token_ids[0]);
    init_session(sp);
  }
  gemm_ws_bytes_ = std::max<size_t>(64u << 20, gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, H, DType::BF16));
  gemm_ws_ = dev_alloc<char>(gemm_ws_bytes_);
  gw_ = Glm4GemmWorkspace{&gemm_, gemm_ws_, gemm_ws_bytes_};
  // The attention projections' lowering (kernels/gemm.hpp dense_gemv_rows):
  // the GEMV chunks to the bound, cuBLASLt's algorithm above it.
  gemm_.set_decode_rows(std::min(max_decode_rows_, dense_gemv_rows()));
  moe_cfg_ = cfg_.moe_config(static_cast<int>(loader_.geometry().local_inter));
  n_split_ = Glm4AttentionLayer::default_decode_splits();

  const Qwen3LocalGeometry& geo = loader_.geometry();
  {
    Glm4KvPoolShape shape;
    shape.layers = cfg_.num_hidden_layers;
    shape.kv_heads = geo.local_kv_heads;
    shape.dim = cfg_.head_dim;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = max_requests_;
    shape.token_slots = max_cache_tokens_;
    pool_.init(shape);
  }
  const size_t M = static_cast<size_t>(max_tokens_);
  resid_ = dev_alloc<uint16_t>(M * H);
  x_ = dev_alloc<uint16_t>(M * H);
  y_ = dev_alloc<uint16_t>(M * H);
  moe_acc_ = dev_alloc<float>(M * H);
  {
    const size_t layers = static_cast<size_t>(cfg_.num_hidden_layers);
    const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
    h_route_ids_ = pinned_alloc<int32_t>(layers * M * K);
    h_route_weights_ = pinned_alloc<float>(layers * M * K);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
}

Qwen3Model::~Qwen3Model() {
  cudaFree(resid_);
  cudaFree(x_);
  cudaFree(y_);
  cudaFree(moe_acc_);
  cudaFreeHost(h_route_ids_);
  cudaFreeHost(h_route_weights_);
  cudaFree(gemm_ws_);
}

int Qwen3Model::table_slots() const {
  return loader_.residency() == LoaderResidency::Resident ? cfg_.num_hidden_layers : 0;
}

Qwen3Model::MemoryPlan Qwen3Model::plan_memory(const Qwen3TextConfig& cfg, int max_tokens,
                                               int64_t max_cache_tokens, int tp_rank, int tp_world,
                                               LoaderResidency residency, int max_requests,
                                               int decode_rows, bool serving_logits) {
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests must be in [1, kPickMaxRequests]");
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("plan_memory: decode_rows exceeds the limit of 32");
  // The fixed batch's row ceiling, floored as the session core floors it.
  const int rows = std::max({kDecodeRows, decode_rows, max_requests});
  const LoaderHeadSharding head = tp_world > 1 ? LoaderHeadSharding::VocabSharded : LoaderHeadSharding::Full;
  const Qwen3LocalGeometry geo = Qwen3LocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const int64_t cache_tokens = round_to_blocks(std::max<int64_t>(max_cache_tokens, max_tokens), kBlockTokens);
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.max_position_embeddings);
  const size_t M = static_cast<size_t>(max_tokens);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t V = static_cast<size_t>(Qwen3LayerStream::lm_vocab_count(cfg, tp_rank, tp_world, head));
  const int n_split = Glm4AttentionLayer::default_decode_splits();

  if (residency == LoaderResidency::Resident) {
    plan.add("model weights (resident; the vision tower stays on disk)",
             Qwen3LayerStream::resident_bytes(cfg, tp_rank, tp_world, head, false));
    plan.add("loader staging (pinned host, freed when the last layer is resident)", 0,
             Qwen3LayerStream::staging_plan_bytes(cfg, tp_rank, tp_world, head, false));
  } else {
    size_t largest = 0;
    for (int l = 0; l < cfg.num_hidden_layers; ++l)
      largest = std::max(largest, Qwen3LayerStream::layer_bytes(cfg, l, tp_rank, tp_world));
    plan.add("model weights (one streamed layer + globals)",
             largest + Qwen3LayerStream::globals_bytes(cfg, tp_rank, tp_world, head));
  }
  plan.add("gemm workspace (at least)", size_t{64} << 20);
  {
    Glm4KvPoolShape shape;
    shape.layers = cfg.num_hidden_layers;
    shape.kv_heads = geo.local_kv_heads;
    shape.dim = cfg.head_dim;
    shape.block_tokens = kBlockTokens;
    shape.max_requests = max_requests;
    shape.token_slots = cache_tokens;
    plan.add("kv cache pool (K/V bf16)", Glm4KvPool::cache_bytes(shape));
  }
  {
    size_t core_dev = 0, core_pin = 0;
    session_core_plan_bytes(max_tokens, max_requests, cfg.hidden_size, static_cast<int>(V), /*mtp=*/false,
                            cfg.hidden_size, &core_dev, &core_pin, rows, serving_logits);
    const size_t layers = static_cast<size_t>(cfg.num_hidden_layers);
    const size_t K = static_cast<size_t>(cfg.num_experts_per_tok);
    plan.add("activations (session core, residual, block io, the moe chain's fp32 sum, route staging)",
             core_dev + 3 * M * H * 2 + M * H * 4, core_pin + layers * M * K * 8);
  }
  {
    const Glm4TextConfig attn_cfg = cfg.attention_view();
    plan.add("layer objects (attention)",
             Glm4AttentionLayer::scratch_bytes(attn_cfg, geo.local_heads, geo.local_kv_heads, max_tokens, rows, n_split));
    const GlmMoeConfig moe_cfg = cfg.moe_config(static_cast<int>(geo.local_inter));
    const int slots = residency == LoaderResidency::Resident ? cfg.num_hidden_layers : 0;
    size_t moe_pinned = 0;
    const size_t moe_dev = GlmMoeLayer::scratch_bytes(moe_cfg, max_tokens, rows, slots, &moe_pinned);
    plan.add("moe scratch (routed slots, graph tables)", moe_dev, moe_pinned);
    plan.add("moe W4A4 activation workspace", GlmMoeLayer::w4a4_scratch_bytes(moe_cfg, max_tokens));
  }
  return plan;
}

GlmMoeWeights Qwen3Model::moe_view(const Qwen3MoeResident& m) {
  GlmMoeWeights w;
  w.router_gate = m.router;
  w.router_bias = nullptr;  // the softmax router carries no bias
  w.experts = nullptr;
  w.experts_fp4 = m.experts.data();
  return w;
}

void Qwen3Model::build_layer_objects(const Qwen3LayerResident& r) {
  if (!attn_) {
    attn_ = std::make_unique<Glm4AttentionLayer>(r.attn, gw_, attn_cfg_, max_tokens_, max_decode_rows_, n_split_);
  } else {
    attn_->rebind(r.attn);
  }
  if (!moe_) {
    moe_ = std::make_unique<GlmMoeLayer>(moe_view(r.moe), moe_cfg_, max_tokens_, max_decode_rows_, table_slots());
  } else {
    moe_->rebind(moe_view(r.moe));
  }
}

// The state every row walk starts from on a fresh request: no blocks (the
// K/V rows are written before any row reads them).
void Qwen3Model::reset_slot_state(int req) { pool_.release_request_blocks(req, stream_); }

GlmSpecSegments Qwen3Model::spec_segments(int, int) const { return GlmSpecSegments{}; }

void Qwen3Model::write_state_snapshot(int, uint8_t* d, int) {
  DGPP_CUDA_OK(cudaMemsetAsync(d, 0, kSnapshotStamp, stream_));
}

void Qwen3Model::mtp_run_rows(int, const int64_t*, int64_t, int, bool, bool, int, int) {
  throw std::logic_error("Qwen3Model: this model class carries no draft layer (engine.mtp must be false)");
}

// ---------------------------------------------------------------------------
// The row walk.
// ---------------------------------------------------------------------------
// The boundary folds: the producer writes its partial into the reducer's
// staged buffer when the shape fits (the collective sends straight from
// there; under capture the recorder's one stable buffer), else into
// `fallback`. The eager producer quiesces before the collective; under
// capture the fold is a recorded node and the stream order IS the drain.
uint16_t* Qwen3Model::stage(uint16_t* fallback, int T, int width, bool capture) {
  if (!boundary_) return fallback;
  uint16_t* s = boundary_->stage(T, width);
  if (s == nullptr && capture) throw std::runtime_error("run_rows: a capture fold does not fit the recorder's staged buffer");
  return s ? s : fallback;
}

void Qwen3Model::fold(uint16_t* buf, int T, int width, bool capture) {
  if (!boundary_) return;
  if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  boundary_->reduce(buf, T, width);
}

void Qwen3Model::enqueue_layer(const Qwen3LayerResident& r, int pool_layer, uint16_t* resid, int T,
                               const WalkRows& rows) {
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  const int64_t n = static_cast<int64_t>(T) * H;
  build_layer_objects(r);
  // ---- the attention block --------------------------------------------------
  glm_rmsnorm_bf16(resid, r.input_norm, x_, T, H, eps, stream_);
  uint16_t* attn_out = stage(y_, T, H, rows.capture);
  {
    Glm4AttnRows arows;
    arows.req_ids = rows.req_ids;
    arows.pos = rows.pos;
    arows.decode = rows.decode;
    attn_->enqueue(x_, T, arows, pool_.view(pool_layer), attn_out, stream_);
  }
  fold(attn_out, T, H, rows.capture);  // block boundary 1: o_proj's partial
  glm_residual_add_bf16(resid, attn_out, n, stream_);
  // ---- the MoE block ----------------------------------------------------------
  glm_rmsnorm_bf16(resid, r.post_norm, x_, T, H, eps, stream_);
  uint16_t* ffn_out = stage(y_, T, H, rows.capture);
  // The routed chain alone (no shared expert): the fp32 sum in ascending
  // expert order, rounded once here (MiMo's arrangement; GlmMoeLayer's
  // enqueue_decode / enqueue_prefill fold a shared expert into the chain
  // and refuse without one).
  if (rows.decode)
    moe_->enqueue_decode_f32(x_, moe_acc_, T, rows.trace, stream_, rows.moe_table_slot);
  else
    moe_->enqueue_prefill_f32(x_, moe_acc_, T, rows.trace, stream_,
                              moe_->mma_takes_grid() ? MoeExpertKernel::kMma : MoeExpertKernel::kGemv);
  launch_moe_round_bf16(ffn_out, moe_acc_, n, stream_);
  fold(ffn_out, T, H, rows.capture);  // block boundary 2: the sliced down projections
  glm_residual_add_bf16(resid, ffn_out, n, stream_);
}

Qwen3Model::Outputs Qwen3Model::run_rows(const RowRun& run) {
  const int T = run.T;
  if (run.capture && loader_.residency() != LoaderResidency::Resident)
    throw std::logic_error("run_rows: a capture needs a resident stack");
  const int H = cfg_.hidden_size;
  const RowInputs in = begin_run(run);
  embed_gather_bf16(globals_.embed, in.tokens, resid_, T, H, stream_);
  Outputs out;
  const bool traces = !run.decode && route_traces_;
  const size_t K = static_cast<size_t>(cfg_.num_experts_per_tok);
  WalkRows rows;
  rows.req_ids = in.req_ids;
  rows.pos = in.pos;
  rows.decode = run.decode;
  rows.capture = run.capture;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const Qwen3LayerResident& r = loader_.load_layer(layer);
    MoeTraceStaging trace;
    rows.trace = nullptr;
    rows.moe_table_slot = run.capture ? layer : -1;
    if (traces) {
      const size_t slot = static_cast<size_t>(layer) * static_cast<size_t>(max_tokens_) * K;
      trace.ids = h_route_ids_ + slot;
      trace.weights = h_route_weights_ + slot;
      trace.biased = nullptr;
      rows.trace = &trace;
      out.route_ids.emplace_back();  // filled after the sync
      out.route_weights.emplace_back();
    }
    enqueue_layer(r, layer, resid_, T, rows);
    if (run.capture_layers) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      std::vector<uint16_t> snap(static_cast<size_t>(T) * H);
      DGPP_CUDA_OK(cudaMemcpy(snap.data(), resid_, snap.size() * 2, cudaMemcpyDeviceToHost));
      out.layer_states.push_back(std::move(snap));
    }
  }
  // Serving projects only each request's last prefill row; all-row
  // diagnostics and decode keep their full head.
  glm_rmsnorm_bf16(resid_, globals_.final_norm, h_, T, H, cfg_.rms_norm_eps, stream_);
  const bool packed_logits = project_head_rows(run, [&](int input_row, int output_row, int head_rows) {
    gemm_.matmul(h_ + static_cast<size_t>(input_row) * H, globals_.lm_head,
                 logits_ + static_cast<size_t>(output_row) * lm_vocab_count_, head_rows, lm_vocab_count_,
                 H, DType::BF16, GemmOut::F32, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_,
                 stream_);
  });
  out = finish_run(run, std::move(out), packed_logits);
  if (!run.capture && traces) {
    for (size_t l = 0; l < out.route_ids.size(); ++l) {
      const size_t slot = l * static_cast<size_t>(max_tokens_) * K;
      out.route_ids[l].assign(h_route_ids_ + slot, h_route_ids_ + slot + static_cast<size_t>(T) * K);
      out.route_weights[l].assign(h_route_weights_ + slot, h_route_weights_ + slot + static_cast<size_t>(T) * K);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The cold diagnostic forward: slot 0, fresh state, every row.
// ---------------------------------------------------------------------------
Qwen3Model::Outputs Qwen3Model::forward(const std::vector<int64_t>& token_ids, bool capture_layers) {
  const int T = static_cast<int>(token_ids.size());
  if (T <= 0) throw std::invalid_argument("forward: empty token batch");
  if (T > max_tokens_) throw std::invalid_argument("forward: tokens exceed max_tokens");
  if (T > max_context_) throw std::invalid_argument("forward: tokens exceed the context bound");
  for (int64_t id : token_ids) {
    if (id < 0 || id >= cfg_.vocab_size) throw std::invalid_argument("forward: token id out of range");
    // The text path has no features to put behind a visual placeholder.
    if (cfg_.vl() && (id == cfg_.image_token_id || id == cfg_.video_token_id))
      throw std::invalid_argument(
          "forward: the prompt carries an image / video placeholder — the vision tower is not served "
          "(text path only)");
  }
  if (session_pos_[0] != 0) throw std::logic_error("forward: slot 0 holds an open session (close it first)");
  open_slot(0);
  if (!pool_.ensure_request_blocks(0, T, stream_)) throw std::runtime_error("forward: the cache pool cannot cover the batch");
  RowRun run;
  run.req = 0;
  run.ids = token_ids.data();
  run.T = T;
  run.pos0 = 0;
  run.decode = false;
  run.all_rows = true;
  run.capture_layers = capture_layers;
  Outputs out = run_rows(run);
  session_close(0);
  return out;
}

// ---------------------------------------------------------------------------
// The graph era.
// ---------------------------------------------------------------------------
void Qwen3Model::graph_prepare() {
  if (loader_.residency() != LoaderResidency::Resident)
    throw std::logic_error("session_graph_prepare: the decode graph needs a resident stack");
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const Qwen3LayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r);
    moe_->prepare_graph_table(layer, stream_);
  }
}

}  // namespace dgpp
