#pragma once
// Qwen3.8-Flash-Next PLE — the hashed n-gram embedding layer (Q3,
// 2026-09-09; docs/qwen38_flash_next_plan.md §1.7, D4; the reference is
// transformers Qwen4ExpTextPLELayer / Qwen4ExpTextNGramEmbedding). Four
// kernels, each deterministic and capturable; the projections run through
// the GEMM interface and the norms through kernels/qwen_norm.hpp.
//
//   hash   token ids -> the 16 table rows per token (int64 math, the
//          reference's multipliers / primes / offsets, the EOS rule);
//   gather this rank's hash heads' rows out of its e4m3 table slice as
//          bf16(e4m3 x scale) — the K-slice of the embedding;
//   gate   sigmoid(signed sqrt(<key_n, query_n> / sqrt(H))) x value per
//          hyper branch, with the reference's bf16 rounding points;
//   conv   the dilated depthwise conv over the normalized gated value with
//          its (width-1) x dilation-step state, silu, the residual adds.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// ids[t, h] for the sequence tokens[0..n) whose two preceding ids are
// ctx_prev1 (t-1) and ctx_prev2 (t-2) — the reference's stored context,
// EOS at a request's start. The EOS rule: y1 = x[t-1]; y2 = EOS when
// x[t-1] is EOS, else x[t-2]. Head h < heads_per_ngram hashes (y0, y1),
// the rest (y0, y1, y2): id = (mix mod head_vocab[h]) + head_offset[h].
// multipliers [ngram_size >= 3], head_vocab / head_offset [heads]: int64
// device arrays. ids: int32 [n, heads] (the padded table has < 2^31 rows).
void qwen_ple_hash_ids(const int32_t* tokens, int n, int32_t ctx_prev1,
                       int32_t ctx_prev2, int32_t eos, const int64_t* multipliers,
                       const int64_t* head_vocab, const int64_t* head_offset,
                       int heads, int heads_per_ngram, int32_t* ids,
                       cudaStream_t stream);

// The row form (Q6, the decode graphs): rows of one or more requests in
// span order (req_spans int32 [num_requests, 2] = start, len into the
// batch; a request's rows contiguous, in position order; pos < 0 marks a
// padding row). Each request's stored context is ctx[req_ids[row] * 4 + 0]
// (t-1) and [.. + 1] (t-2) — device int32 [max_requests, 4], the two
// ids and two pad words (a 16-byte state family the spec commit copies).
// tokens: int64 [rows]. Padding rows hash their (valid) token like any
// other; nothing reads the result.
void qwen_ple_hash_ids_rows(const int64_t* tokens, int rows, const int32_t* req_ids,
                            const int64_t* pos, const int32_t* req_spans,
                            int num_requests, const int32_t* ctx, int32_t eos,
                            const int64_t* multipliers, const int64_t* head_vocab,
                            const int64_t* head_offset, int heads,
                            int heads_per_ngram, int32_t* ids, cudaStream_t stream);
// ctx_rows[row * 4 + {0, 1}] = the request's context AFTER that row (its
// token and the one before it) — the per-row snapshots a rollback copies
// back (row accepted-1); the span's last real row's context lands in the
// request's ctx IN PLACE (the state after every row stood). A padding row
// repeats the context as it stands. Enqueue after the hash (which reads
// the incoming context).
void qwen_ple_context_rows(const int64_t* tokens, int rows, const int32_t* req_ids,
                           const int64_t* pos, const int32_t* req_spans,
                           int num_requests, int32_t* ctx, int32_t* ctx_rows,
                           cudaStream_t stream);
// out[t, hl * head_dim + d] = bf16(e4m3(table[ids[t, head_begin + hl] -
// row_begin][d]) x scale) for the rank's heads_local heads. table: this
// rank's rows [row_begin, row_begin + rows) of the padded table, e4m3
// [rows, head_dim]; head_dim % 8 == 0. An id outside the slice is a
// contract violation (traps).
void qwen_ple_gather_bf16(const uint8_t* table, int64_t row_begin, int64_t rows,
                          float scale, const int32_t* ids, int n, int heads,
                          int head_begin, int heads_local, int head_dim,
                          uint16_t* out, cudaStream_t stream);

// gated[t, i, :] = bf16(bf16(sigmoid(g)) x value[t, :]) with
//   g = bf16(sqrt(max(|g0|, 1e-6))) x sign(g0),
//   g0 = bf16(bf16(sum_d bf16(key_n[t, i, d] x query_n[t, i, d])) / sqrt(H)),
// the reference's bf16 ops one by one (fp32 sum of the rounded products).
// key_n / query_n / gated: bf16 [n, hc * H]; value: bf16 [n, H].
// The staged form (2026-09-10, the mmap'ed table): the rows already
// gathered by the host into `staged` — e4m3 [n, heads_local, head_dim],
// row (t, hl) the table row of ids[t, head_begin + hl] — converted with the
// same per-element arithmetic as qwen_ple_gather_bf16 (bf16(e4m3 x scale)),
// so out is bitwise the device gather's. `staged` may be pinned host memory.
void qwen_ple_gather_staged_bf16(const uint8_t* staged, float scale, int n, int heads_local,
                                 int head_dim, uint16_t* out, cudaStream_t stream);
// The staging handshake's device half (2026-09-29): after the hash kernel,
// publishes the walk's row count and a sequence number to pinned words the
// gather thread polls (the ids are already in pinned memory). One thread:
// *pinned_rows = rows, then seq = ++*device_seq released to *pinned_seq.
// The gather kernel's turn waits on the thread's answer through
// glm_stage_wait (kernels/glm_spec.hpp). Capturable.
void qwen_ple_publish_stage(uint64_t* device_seq, uint64_t* pinned_seq, int32_t* pinned_rows, int rows,
                            cudaStream_t stream);

void qwen_ple_gate_bf16(const uint16_t* key_n, const uint16_t* query_n,
                        const uint16_t* value, uint16_t* gated, int n, int hc,
                        int hidden, cudaStream_t stream);

// The conv (width 4, dilation 3 — the only instantiation) over un [n, C]
// with state [C, 9] (the previous nine inputs, oldest first; in/out —
// after the call the last nine inputs of state ++ un):
//   c[t, ch] = bf16(silu(bf16(sum_k w[ch, k] x u[t - 9 + 3k])))   (k < 4; u[t] the 4th tap)
//   ple      = bf16(gv[t, ch] + c)
//   out      = residual ? bf16(residual[t, ch] + ple) : ple   (out may alias residual)
// weight: bf16 [C, 4] (the checkpoint's [C, 1, 4]).
void qwen_ple_conv_bf16(const uint16_t* un, const uint16_t* gv, uint16_t* state,
                        const uint16_t* weight, const uint16_t* residual,
                        uint16_t* out, int n, int channels, int width,
                        int dilation, cudaStream_t stream);
// The row form: request req's state at states + req * state_stride (bf16
// elements); each span rolls its own request's state through its rows in
// order, a padding row (pos < 0) writes its residual (or zero) through
// and leaves the state alone. snapshots (optional, bf16 [rows, C * 9]):
// the state as it stands AFTER every row — the spec rollback's source.
void qwen_ple_conv_rows_bf16(const uint16_t* un, const uint16_t* gv, uint16_t* states,
                             int64_t state_stride, const uint16_t* weight,
                             const uint16_t* residual, uint16_t* out, const int32_t* req_ids,
                             const int64_t* pos, const int32_t* req_spans, int num_requests,
                             int channels, int width, int dilation, cudaStream_t stream,
                             uint16_t* snapshots = nullptr);

}  // namespace dgpp
