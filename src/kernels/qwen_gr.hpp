#pragma once
// The gated residual's elementwise kernels (Q3, 2026-09-09;
// docs/qwen38_flash_next_plan.md §1.5, transformers Qwen4ExpTextGatedResidual).
// The two gate matmuls (down [r, hc*H] and up [hc*H, r]) and the inject
// dots go through the GEMM/GEMV interface; these kernels are what sits between
// them, each pinned to the reference's bf16 rounding points:
//   mix:     Rn = group_rmsnorm(R)                    (qwen_norm.hpp)
//            t  = bf16(silu(bf16(bf16(W_down Rn) / hc)))     gate_act
//            G  = bf16(sigmoid(bf16(W_up t)))                 |
//            x  = bf16(mean_i bf16(G_i * Rn_i))               | mix_finish
//   combine: s_i = 2 * bf16(sigmoid(bf16(W_inj Rn)_i / hc))   |
//            R_i = bf16(R_i + bf16(y * s_i))                   | combine
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/bf12_gemv.hpp"

namespace dgpp {

// In place on t [rows, r] bf16: t = silu(t / hc).
void qwen_gr_gate_act_bf16(void* t, int64_t rows, int r, int hc, cudaStream_t stream);

// x[rows, H] from the gate logits [rows, hc*H] and Rn [rows, hc*H].
void qwen_gr_mix_finish_bf16(const void* logits, const void* rn, void* x, int64_t rows,
                             int hc, int hidden, cudaStream_t stream);

// R[rows, hc*H] += y[rows, H] * s_i, with s from W_inj [hc, hc*H] and
// Rn [rows, hc*H] (the inject dots computed here: hc dots of hc*H per row,
// staged through `gates` — F32 [rows, hc] device scratch — between the
// dots kernel and the row-wide apply kernel, 2026-09-09).
void qwen_gr_combine_bf16(void* r_state, const void* rn, const void* w_inject,
                          const void* y, float* gates, int64_t rows, int hc, int hidden,
                          cudaStream_t stream);

// The combine's two halves as their own launches, so the dots — which read
// only Rn, and are therefore ready as soon as the mix has normalized the
// row — can be issued on a side stream at mix time and joined before the
// apply (2026-09-10, the decode profile: one block per row walking the
// hc*H-wide chain cost 21 us per site at T=1 and 23 at two rows, 96 sites
// per step, all of it on the chain between the boundary collective and the
// next mix). Same kernels, same order, same values: bitwise
// qwen_gr_combine_bf16 either way.
void qwen_gr_combine_dots_bf16(const void* rn, const void* w_inject, float* gates,
                               int64_t rows, int hc, int hidden, cudaStream_t stream);
void qwen_gr_combine_apply_bf16(void* r_state, const float* gates, const void* y,
                                int64_t rows, int hc, int hidden, cudaStream_t stream);
// The previous site's combine applied while the next site's group norm
// reads R (2026-09-29): one launch in place of combine_apply + the norm.
// R[row, i*H + j] = bf16(R + bf16(y[row, j] * gates[row, i])) in place, then
// rn = group_rmsnorm(R) over the updated values — the apply is
// combine_apply_kernel's ops and the norm is group_rmsnorm_kernel's
// reduction (the same per-thread element order, the same fmaf chain), so
// both R and rn are bitwise the two-launch chain's.
void qwen_gr_combine_norm_bf16(void* r_state, const float* gates, const void* y, const void* norm_w,
                               void* rn, int64_t rows, int hc, int hidden, float eps, cudaStream_t stream);

// The mix's two GEMVs with their producers folded into the staging
//, each bitwise the two-launch chain it replaces, for the
// decode rows (rows <= 8, chunked by four): norm_down = the group norm
// (block_sum_squares' order per group; `rn` gets the normalized rows when
// given) staged into the down GEMV, t = bf16(Rn . down_w^T); act_up = the
// gate activation silu(t / hc) staged into the up GEMV, logits = bf16(act
// . up_w^T). R bf16 [rows, hc*hidden] (row stride r_stride), norm_w
// [hc*hidden], down_w [lowrank, hc*hidden], up_w [hc*hidden, lowrank].
bool qwen_gr_fused_mix_accepts(int hc, int hidden, int lowrank);
// `w_inject` and `gates` (both or neither) add the combine's inject dots to
// the same launch — one extra block reads the normalized row out of the
// staged copy every block already holds, in combine_dots_kernel's order, so
// the gates are bitwise that kernel's and the chain loses a 21 us launch
//.
void qwen_gr_norm_down_bf16(const void* r, size_t r_stride, const void* norm_w, int hc, int hidden,
                            float eps, void* rn, const void* down_w, void* t, int lowrank,
                            int64_t rows, cudaStream_t stream, const void* w_inject = nullptr,
                            float* gates = nullptr);
void qwen_gr_act_up_bf16(const void* t, int lowrank, int hc, const void* up_w, void* logits, int hidden,
                         int64_t rows, cudaStream_t stream);

// The batched rows' (1..8) down GEMV over an already-normalized Rn with the
// inject rows appended: t = bf16(Rn . down_w^T) bitwise the
// bf16 GEMV's, gates as qwen_gr_combine_dots_bf16's — one launch per four
// rows in place of the chain's down GEMV plus a side-stream dots kernel.
void qwen_gr_down_inject_bf16(const void* rn, const void* down_w, void* t, int lowrank,
                              const void* w_inject, float* gates, int hc, int hidden,
                              int64_t rows, cudaStream_t stream);
// The batched rows' group norm staged into the down + inject GEMV
// (2026-09-29): one block reduction for every (row, group), so Rn (block
// 0 writes it), t and the gates are bitwise the group norm + down_inject
// chain — one launch fewer per site. A pending combine of the previous
// site must have landed on R first (every block reads R).
void qwen_gr_norm_down_inject_bf16(const void* r, size_t r_stride, const void* norm_w, int hc, int hidden, float eps,
                                   void* rn, const void* down_w, void* t, int lowrank, const void* w_inject,
                                   float* gates, int64_t rows, cudaStream_t stream);
void qwen_gr_norm_down_inject_fp8(const void* r, size_t r_stride, const void* norm_w, int hc, int hidden, float eps,
                                  void* rn, const uint8_t* down_p, const float* down_s, void* t, int lowrank,
                                  const void* w_inject, float* gates, int64_t rows, cudaStream_t stream);
// The same with the down matrix's lossless 12-bit companion
// (engine.bf16_weights; kernels/bf12_gemv.hpp): the rows through
// bf12_gemv::row_dots, bitwise the bf16 chain, 0.75 of the down bytes
// (2026-09-29, world 1 where the site streams from DRAM).
void qwen_gr_down_inject_bf12(const void* rn, const Bf12Matrix& down, void* t, int lowrank,
                              const void* w_inject, float* gates, int hc, int hidden,
                              int64_t rows, cudaStream_t stream);
// The same three with the down / up matrices in block FP8 (2026-09-10,
// engine.dense_weights = "fp8"): E4M3 payload [rows, k] + fp32 128 x 128
// scales, the rows through fp8_gemv::row_dots — the scale GEMM's GEMV
// chain — so t / logits are bitwise the unfused fp8 chain's (the norm
// kernel + launch_scale_gemm_bf16, gate_act + launch_scale_gemm_bf16);
// the inject rows stay BF16 and their gates combine_dots_kernel's. k
// (hc * hidden for down, lowrank for up) a multiple of 16.
void qwen_gr_norm_down_fp8(const void* r, size_t r_stride, const void* norm_w, int hc, int hidden,
                           float eps, void* rn, const uint8_t* down_p, const float* down_s, void* t,
                           int lowrank, int64_t rows, cudaStream_t stream,
                           const void* w_inject = nullptr, float* gates = nullptr);
void qwen_gr_act_up_fp8(const void* t, int lowrank, int hc, const uint8_t* up_p, const float* up_s,
                        void* logits, int hidden, int64_t rows, cudaStream_t stream);
// act_up with the mix folded into its epilogue (2026-09-29): x = the
// act_up + mix_finish chain's output, bitwise, in one launch and without the
// logits round trip. rn is the site's normalized rows [rows, hc*hidden].
void qwen_gr_act_up_mix_bf16(const void* t, int lowrank, int hc, const void* up_w, const void* rn, void* x,
                             int hidden, int64_t rows, cudaStream_t stream);
void qwen_gr_act_up_mix_fp8(const void* t, int lowrank, int hc, const uint8_t* up_p, const float* up_s,
                            const void* rn, void* x, int hidden, int64_t rows, cudaStream_t stream);
void qwen_gr_down_inject_fp8(const void* rn, const uint8_t* down_p, const float* down_s, void* t,
                             int lowrank, const void* w_inject, float* gates, int hc, int hidden,
                             int64_t rows, cudaStream_t stream);

}  // namespace dgpp
