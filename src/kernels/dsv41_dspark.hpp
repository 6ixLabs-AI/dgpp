#pragma once
// The DSpark draft's glue kernels (docs/deepseek_v41_flash_plan.md §1.7,
// D8; the reference `DSparkBlock`): the target layers' stream mean, the
// block rows off the accepted verify rows, the Markov-biased head row and
// the confidence logit. Every launch is graph-capturable (kernels only).
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// out[t, :] (row stride out_stride elements) = bf16(mean over the hc_mult
// streams of streams[t, :, :]) — the fp32 sum in stream order, one
// rounding (torch's bf16 mean: `h.mean(dim=2)` of the attention input at
// a DSpark target layer).
void dsv41_stream_mean_bf16(const uint16_t* streams, int hc_mult, int hidden, int rows, uint16_t* out,
                            int64_t out_stride, cudaStream_t stream);

// The block rows off the draft rows the core staged (glm_spec_draft_rows'
// contract: step_pos[r] the accepted rows' positions, -1 padding, tokens[r]
// their winners; the eager path stages every row real). For each group g
// of `rows_per_group` rows: P = 1 + max(step_pos) over the group (-1 when
// no row is real: the group's block is padding) and `next` = the token of
// that row; the block's rows g * block + k get pos P + k (or -1), the
// tokens [next, noise, noise, ...], the group's request id (row
// g * rows_per_group's) and one span per group (spans[g] = g * block,
// spans[groups] = groups * block).
void dsv41_dspark_block_rows(const int64_t* step_pos, const int64_t* tokens, const int32_t* req_ids, int groups,
                             int rows_per_group, int block, int64_t noise_id, int64_t* pos_out, int64_t* tok_out,
                             int32_t* req_out, int32_t* spans_out, cudaStream_t stream);

// The Markov-biased head row: for each group g and each j in [0, rows_out),
//   out[g * out_group_stride + j * count + v] = base[g * base_group_stride + block_row * count + v]
//                                             + sum_r embed[tok_g][r] * head[vocab_begin + v][r]
// over this rank's vocab slice [vocab_begin, vocab_begin + count) — the
// reference's `logits[:, i].add_(markov_head(output_ids[:, i]))` with the
// bf16 embedding row and the bf16 head rows, fp32 accumulation (the head's
// linear over one row: one fp32 dot per vocab entry, in rank order).
// tok_g = tok[g * tok_stride]. block_max (optional, fp32
// [groups][ceil(count / 256)]): the maximum of each 256 consecutive
// biased entries, for dsv41_dspark_rescore.
void dsv41_dspark_markov_bias(const float* base, int64_t base_group_stride, int block_row, const uint16_t* markov_embed,
                              const uint16_t* markov_head, int rank, int vocab_begin, int count, const int64_t* tok,
                              int tok_stride, int groups, float* out, int64_t out_group_stride, int rows_out,
                              cudaStream_t stream, float* block_max = nullptr);

// The screened draft head's exact pass (2026-10-02, DeepSeek-V4-Flash). The
// draft rows' base logits come from a block-FP8 copy of the lm head (half
// the bytes of the bf16 head the verify rows already read once a pass);
// after the Markov bias of a pick's row, this pass recomputes, from the
// bf16 head rows themselves, the base logit of every entry within `delta`
// of the biased row's maximum — the only ones the pick can land on — so
// the draft is the bf16 head's, not the copy's. For each group g and v with
//   biased[g, 0, v] >= max_v biased[g, 0, :] - delta:
//   biased[g, j, v] += sum_d h[g, block_row, d] * head[v, d] - base[g, block_row, v]     (j < rows_out)
// At most kDsparkRescoreCap entries per 256 consecutive ones are
// recomputed (the highest of them when a flat row offers more; the rest
// keep the copy's values), so a row's maximum is always the bf16 head's. block_max: the biased row's maxima per 256 entries, which
// dsv41_dspark_markov_bias writes when handed the buffer
// ([groups][ceil(count / 256)]). h: bf16, row (g, block_row) at
// g * h_group_stride + block_row * hidden; head: bf16 [count, hidden];
// hidden % 8 == 0, both 16-byte aligned (h_group_stride % 8 == 0).
constexpr int kDsparkRescoreCap = 8;
void dsv41_dspark_rescore(float* biased, int64_t out_group_stride, int rows_out, const float* base,
                          int64_t base_group_stride, int block_row, int count, const uint16_t* h,
                          int64_t h_group_stride, int hidden, const uint16_t* head, const float* block_max,
                          float delta, int groups, cudaStream_t stream);

// The confidence logit of block row `block_row` per group (the reference's
// `confidence_head(x, markov_embed)`: the fp32 projection of
// [x_k | embed(tok_{k-1})], x the hc_pre'd hidden BEFORE the norm):
//   conf[g * conf_stride] = sum_d w[d] * x[g * x_group_stride + block_row * hidden + d]
//                         + sum_r w[hidden + r] * embed[tok_g][r]
void dsv41_dspark_confidence(const uint16_t* x, int64_t x_group_stride, int block_row, int hidden,
                             const uint16_t* markov_embed, int rank, const int64_t* tok, int tok_stride,
                             const float* w, int groups, float* conf_out, int conf_stride, cudaStream_t stream);

}  // namespace dgpp
