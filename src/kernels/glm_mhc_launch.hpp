#pragma once
// Launchers for the mHC module kernels (DESIGN §7.3; semantics documented in
// models/glm/mhc.hpp). All kernels are deterministic (fixed reduction order)
// and CUDA-graph capturable: caller-owned scratch only, no host reads, no
// dynamic smem.
#include <cuda_runtime.h>

#include "models/glm/mhc.hpp"

namespace dgpp {

// Computes the mHC mapping for `tokens` rows:
//   streams   bf16 [tokens, n, D]      (the residual streams, pre-collapse)
//   collapsed bf16 [tokens, D] out     (weighted sum over streams, the
//                                       sublayer input)
//   post      bf16 [tokens, n] out     (block-output placement weights)
//   comb      bf16 [tokens, n, n] out  (stream mixer, ~doubly stochastic)
// pre is internal (consumed by the collapse) and not exported.
// logits_scratch: f32 [tokens, cfg.coeff_rows()] device scratch the caller
// owns — the hand-off between the per-coefficient dots kernel and the
// finish kernel (two launches; see glm_mhc.cu for why).
// Tests: switch the prefill-sized (>= 16 tokens, vector path) dots between
// the token-tiled form (default; bitwise the per-coefficient form) and the
// per-coefficient form.
void mhc_set_tiled_form(bool on);
// The prefill row count at or above which the mHC dots take the tiled
// form (16 by default; a family with a group prefill sets 1 so a row's
// coefficients never depend on the rows sharing the launch).
void mhc_set_tile_min_tokens(int tokens);
int mhc_tile_min_tokens();
// Tests: switch the prefill-sized dots between the tensor-core GEMM form
// (default; fp32 mma.sync accumulation, inv_rms applied to the finished
// logits — within the oracle budgets, not bitwise the per-coefficient form)
// and the token-tiled kernel (bitwise the per-coefficient form).
void mhc_set_prefill_gemm(bool on);

void launch_mhc_compute(const uint16_t* streams, const GlmMhcWeights& w,
                        const GlmMhcConfig& cfg, uint16_t* collapsed,
                        uint16_t* post, uint16_t* comb, float* logits_scratch,
                        int tokens, cudaStream_t stream);

// The same, plus the sublayer's two-rounding RMSNorm of the collapsed row
// (glm_norm.hpp's semantics) in the finish kernel's tail:
//   normed bf16 [tokens, D] out = rmsnorm(collapsed, ln, ln_eps)
// One launch fewer per site than launch_mhc_compute + glm_rmsnorm_bf16;
// ln and normed are both null (plain compute) or both set. With normed
// requested, collapsed may be null: the row is then not stored (the sites
// consume normed alone).
// finish_counters (int32 [tokens] device scratch, zero at rest; the caller
// zeroes it once at allocation) fuses the finish phase into the dots
// launch: the last dots block of a token runs it. Null keeps the two-
// launch form. Returns true when comb was DEFERRED (defer_comb with the
// fused per-coefficient form): the caller must then launch_mhc_comb
// before anything reads comb; the tiled prefill form never defers.
// The single-pass form (2026-09-13, DeepSeek-V4.1-Flash,
// docs/deepseek_v41_flash_plan.md D4): every sublayer collapses its input
// with the coefficients the PREVIOUS sublayer predicted. pre_in (fp32
// [tokens, n]) replaces this site's own pre in the collapse (null: the GLM
// form); pre_out receives this site's own pre for the next sublayer;
// post_f32 / comb_f32 receive the coefficients in fp32 beside the bf16
// exports (the one-rounding update below reads them). A deferred comb
// takes its fp32 export from launch_mhc_comb instead.
struct MhcSinglePass {
  const float* pre_in = nullptr;
  float* pre_out = nullptr;
  float* post_f32 = nullptr;
  float* comb_f32 = nullptr;
  // The fused norm's rounding (2026-10-01, DeepSeek-V4-Flash's decode
  // sites): true = the DeepSeek reference's RMSNorm on the bf16 collapsed
  // row — fp32 interior, the weight applied in fp32, ONE bf16 rounding
  // (csa2_rmsnorm_bf16's arithmetic; its sum of squares in the finish's
  // tree order, a rounding-level difference in the scale) — in place of
  // GLM's two-rounding norm.
  bool one_round_norm = false;
};

bool launch_mhc_compute_normed(const uint16_t* streams, const GlmMhcWeights& w,
                               const GlmMhcConfig& cfg, uint16_t* collapsed,
                               uint16_t* post, uint16_t* comb,
                               float* logits_scratch, const uint16_t* ln,
                               uint16_t* normed, float ln_eps, int tokens,
                               cudaStream_t stream,
                               int* finish_counters = nullptr,
                               bool defer_comb = false,
                               const MhcSinglePass* single_pass = nullptr,
                               bool decode_rows = false);
// decode_rows (2026-09-14): the rows are a decode batch — keep the
// per-coefficient fused form whatever their count. The prefill-sized forms
// (>= 16 tokens: the token-tiled kernel, or the tensor-core GEMM) are for
// prefill; a batched decode row must be bitwise the row alone, and the
// GEMM form is not (DeepSeek-V4.1's six-slot batch is 30 rows).

// The deferred comb (2026-09-08; fused finish only): with defer_comb the
// finish above writes collapsed/post/normed and leaves comb to this
// launch, which reads the dots' logits_scratch (one warp per token, the
// in-block arithmetic exactly — bitwise). Only the stream update reads
// comb, so decode runs it on a side stream forked after the finish and
// joined before the update, off the sublayer's critical path.
void launch_mhc_comb(const float* logits_scratch, const GlmMhcWeights& w,
                     const GlmMhcConfig& cfg, uint16_t* comb, int tokens,
                     cudaStream_t stream, float* comb_f32 = nullptr);

// The one-rounding stream update of the single-pass form (the reference's
// hc_post: fp32 products and sum, one cast): for every token,
//   streams_out[i] = bf16(post[i] * sublayer_out + sum_j comb[j,i] * streams_in[j])
// with post / comb the fp32 exports above.
void launch_mhc_stream_update_f32(const float* post, const float* comb,
                                  const uint16_t* sublayer_out, const uint16_t* streams_in,
                                  uint16_t* streams_out, const GlmMhcConfig& cfg, int tokens,
                                  cudaStream_t stream);

// The weighted collapse with the sublayer's RMSNorm (the single-pass
// form's head input: hc_pre(streams, pre) then the norm): collapsed
// [tokens, D] = bf16(sum_j pre[j] * streams[j]) (may be null), normed =
// rmsnorm(collapsed, ln, ln_eps) with the two-rounding norm (ln and normed
// both null or both set). Bitwise launch_mhc_compute_normed's normed row
// at the same pre_in.
void launch_mhc_collapse_normed(const uint16_t* streams, const float* pre,
                                const uint16_t* ln, float ln_eps, uint16_t* collapsed,
                                uint16_t* normed, const GlmMhcConfig& cfg, int tokens,
                                cudaStream_t stream);

// Stream update after the sublayer: for every token,
//   streams_out[i] = bf16(bf16(post[i] * sublayer_out)
//                         + bf16(sum_j comb[j,i] * streams_in[j]))
// with post/comb already bf16 (as produced by launch_mhc_compute) — the
// reference's dtype choreography, including both intermediate roundings.
// streams_in and streams_out must not alias.
void launch_mhc_stream_update(const uint16_t* post, const uint16_t* comb,
                              const uint16_t* sublayer_out,
                              const uint16_t* streams_in,
                              uint16_t* streams_out, const GlmMhcConfig& cfg,
                              int tokens, cudaStream_t stream);

// Final head: out[d] = bf16(mean over streams), fp32 accumulate.
void launch_mhc_final_mean(const uint16_t* streams, uint16_t* out,
                           const GlmMhcConfig& cfg, int tokens,
                           cudaStream_t stream);

}  // namespace dgpp
