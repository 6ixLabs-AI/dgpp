#pragma once
// KDA (Kimi Delta Attention) forward kernels, DESIGN §7.1.
//
// The recurrence implemented here is pinned bit-for-bit (module fp32
// reduction order) to the vLLM GLM-5 reference kernels
// (flash_linear_attention/ops/fused_recurrent.py with IS_KDA/SIGMOID_BETA/
// COMPUTE_GATE/SAFE_GATE, and causal_conv1d with activation="silu"):
//
//   per head h, state S in R^{V x K} (fp32), per token t:
//     q, k   <- l2norm(q, k) with eps=1e-6 inside the sqrt
//     q      <- q * K^-1/2
//     g[k]   <- lower_bound / (1 + exp(-exp(A_log[h]) * (g_raw[h,k] + dt_bias[h,k])))
//     S[v,k] <- exp(g[k]) * S[v,k]                    (decay, K-major)
//     u[v]   <- v[v] - sum_k S[v,k] * k[k]             (delta error)
//     S[v,k] <- S[v,k] + sigmoid(beta_raw[h]) * u[v] * k[k]
//     o[v]   <- sum_k S[v,k] * q[k]                    (post-update read)
//
// One shared implementation serves chunked prefill and single-token decode:
// the only difference is the token count. State round-trips through memory
// in fp32, so carrying state across chunk boundaries is exact.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// Post-row state snapshots for speculative decode (DESIGN §9). When
// `states` is set and a call carries T > 1 rows, the state as it stands
// after row t (t < T-1) is stored to states + t * stride_elems, in the
// state's own layout. Row T-1's state lands in place as always, so a
// caller that accepts every row does nothing; a caller that accepts only
// rows [0, a) copies snapshot a-1 over the committed state. Null = off.
struct KdaStateSnapshots {
  float* states = nullptr;
  int64_t stride_elems = 0;  // >= heads * v_dim * k_dim
};
struct KdaConvSnapshots {
  void* states = nullptr;
  int64_t stride_elems = 0;  // >= channels * state_width (bf16 elems)
};

// The checkpoint-and-replay form of the speculative recurrent state
// (2026-09-29, the Qwen depth-3 profile: three snapshot stores per layer per
// pass plus the commit's copy of one back were 500 MB of a 12 GB pass). The
// live state buffer holds the CHECKPOINT — the state before the last pass's
// rows — and the pass's own rows are kept as their inputs (the post-conv
// q|k|v row, a_raw, beta_raw: ~21 KB a row against 3.15 MB a state). A
// pass first replays the rows the verdict accepted from the previous pass
// (count[req] of them, from `in`), stores that state as the new checkpoint,
// then runs its own rows from registers and saves their inputs to `save`
// for the next pass; nothing else is written. The replayed chain is the
// same per-token code on the same inputs, so the state after the replay is
// bitwise the snapshot it replaces. `in` and `save` are distinct buffers
// (one block's replay may still be reading a slice another block has
// already saved); the caller alternates them by a parity the commit flips.
// A consumer that needs the state as it stands (a prefill continuation,
// the prefix cache's snapshot) runs the materialize form: replay
// `materialize_rows` rows (or count[req] when negative), store, no own rows.
// Scalar-gate (GDN) recurrences only: a KDA gate row is heads * K wide.
struct KdaReplay {
  const uint16_t* in = nullptr;     // saved rows [rows_cap][in_stride] (request stride request_stride)
  uint16_t* save = nullptr;         // this pass's rows, same layout
  const int32_t* count = nullptr;   // per request: rows to replay (device)
  int64_t in_stride = 0;            // elements per saved row: the qkv row + heads (a) + heads (beta)
  int64_t request_stride = 0;       // elements between requests' saved blocks
  int rows_cap = 0;                 // saved rows per request (>= the pass's rows)
  bool checkpoint = false;          // replay + checkpoint store + save; no final store
  bool materialize = false;         // the materialize form: replay, store, no own rows
  int materialize_rows = -1;        // materialize: replay this many rows (-1: count[req])
  float* dst = nullptr;             // materialize destination (nullptr: the live state, in place)
};

// A decode batch's device-side row map. Span i is (start, length) in the row
// batch, every non-padding row in that span carries the same request id, and
// a negative position marks a fixed-shape padding row. Eager callers may pass
// only active spans; a fixed-shape graph may pass one span per configured slot,
// including all-padding spans. The kernels obtain the state slot from
// request_ids[start..start+length), so active slots may be any subset (for
// example {1, 6}) without placeholder spans for the slots between them. Rows
// for one request must be contiguous because its recurrence is ordered.
struct KdaRequestRows {
  const int32_t* request_ids = nullptr;  // device [rows]
  const int64_t* positions = nullptr;    // device [rows], -1 = padding
  const int32_t* spans = nullptr;        // device [num_requests, 2]
  int num_requests = 0;
};

// Causal depthwise short conv over the merged q|k|v channels with silu
// activation, matching the reference's runtime-merged causal_conv1d call.
//   src:            bf16 [tokens, channels], row stride src_row_stride elems
//   weight:         bf16 [channels, conv_width] (merged q|k|v conv weights)
//   conv_state:     bf16 [channels, state_width], in/out. Columns
//                   [0, conv_width-1) hold the committed history; the
//                   speculative reserve columns are left untouched (M8).
//   dst:            bf16 [tokens, channels] contiguous, silu applied.
// The state stores raw pre-activation inputs; after the call it holds the
// last conv_width-1 inputs (blended with the previous state when
// tokens < conv_width-1).
void kda_causal_conv_silu_bf16(const void* src, int64_t src_row_stride,
                               const void* weight, void* conv_state,
                               int state_width, void* dst, int tokens,
                               int channels, int conv_width,
                               cudaStream_t stream,
                               const KdaConvSnapshots& snap = {});

// Request-indexed decode form. `conv_states` is the layer's state for slot 0
// and request_state_stride is the distance between slots in bf16 elements.
// Each active span rolls only its selected slot; padding rows write zero to
// dst and leave all state untouched. Snapshot row indices are global batch
// row indices, which lets a later per-request commit select its own row.
void kda_causal_conv_silu_bf16_batched(
    const void* src, int64_t src_row_stride, const void* weight,
    void* conv_states, int64_t request_state_stride, int state_width,
    void* dst, int rows, int channels, int conv_width,
    const KdaRequestRows& requests, cudaStream_t stream,
    const KdaConvSnapshots& snap = {});

// Gated RMSNorm with sigmoid gate (reference o_norm: FusedRMSNormGated,
// activation="sigmoid"): y = rmsnorm(x) * w * sigmoid(gate), fp32 internal.
//   x/gate/y: bf16 [rows, dim] contiguous; weight: bf16 [dim].
void kda_gated_rmsnorm_sigmoid_bf16(const void* x, const void* gate,
                                    const void* weight, void* y, int64_t rows,
                                    int dim, float eps, cudaStream_t stream);

// The KDA recurrence above, sequential over tokens, FP32 state held in
// registers for the whole call. Handles any token count >= 1; decode is the
// tokens==1 case of the same code path.
//   qkv:      bf16 [tokens, H*K + H*K + H*V] contiguous (post-conv q|k|v)
//   g_raw:    bf16 [tokens, heads*k_dim] contiguous (f_b projection output)
//   beta_raw: bf16 [tokens, heads], row stride beta_row_stride elems
//   a_log:    fp32 [heads]; dt_bias: fp32 [heads, k_dim]
//   state:    fp32 [heads, v_dim, k_dim] in/out, k contiguous
//   out:      bf16 [tokens, heads, v_dim]
// Requires k_dim % 4 == 0 and k_dim in {32, 64, 128}.
void kda_recurrent_fwd(const void* qkv, const void* g_raw, const void* beta_raw,
                       int64_t beta_row_stride, const float* a_log,
                       const float* dt_bias, float* state, void* out,
                       int tokens, int heads, int k_dim, int v_dim,
                       float lower_bound, float scale, cudaStream_t stream,
                       const KdaStateSnapshots& snap = {});

// Request-indexed decode form. `states` is this layer's recurrent state for
// slot 0 and request_state_stride is the distance between slots in fp32
// elements. Spans run independently in one launch; rows within each span run
// sequentially against that request's register-resident state. Padding rows
// produce zero output and do not advance state.
void kda_recurrent_fwd_batched(
    const void* qkv, const void* g_raw, const void* beta_raw,
    int64_t beta_row_stride, const float* a_log, const float* dt_bias,
    float* states, int64_t request_state_stride, void* out, int rows,
    int heads, int k_dim, int v_dim, float lower_bound, float scale,
    const KdaRequestRows& requests, cudaStream_t stream,
    const KdaStateSnapshots& snap = {});

// The Gated DeltaNet recurrence (Q3, 2026-09-09; docs/qwen38_flash_next_plan.md
// §1.3): the KDA kernel's scalar-gate mode. Per head h and token t the state
// decays by exp(-exp(A_log[h]) * softplus(a_raw[t,h] + dt_bias[h])); value
// head h reads key head h / kv_ratio's q and k (the reference's
// repeat_interleave); everything else — l2norm, K^-1/2, sigmoid(beta), the
// delta update, the post-update read, snapshots, batching — is the KDA
// recurrence above.
//   qkv:      bf16 [tokens, Hk*K + Hk*K + H*V] (post-conv q|k|v), Hk = H/kv_ratio
//   a_raw:    bf16 [tokens, H], row stride a_row_stride elems (in_proj_a)
//   beta_raw: bf16 [tokens, H], row stride beta_row_stride elems (in_proj_b)
//   a_log:    fp32 [H]; dt_bias: fp32 [H]
//   state:    fp32 [H, v_dim, k_dim] in/out, k contiguous
//   out:      bf16 [tokens, H, v_dim]
void gdn_recurrent_fwd(const void* qkv, const void* a_raw, int64_t a_row_stride,
                       const void* beta_raw, int64_t beta_row_stride,
                       const float* a_log, const float* dt_bias, float* state,
                       void* out, int tokens, int heads, int kv_ratio,
                       int k_dim, int v_dim, float scale, cudaStream_t stream,
                       const KdaStateSnapshots& snap = {}, const KdaReplay& replay = {});

void gdn_recurrent_fwd_batched(
    const void* qkv, const void* a_raw, int64_t a_row_stride,
    const void* beta_raw, int64_t beta_row_stride, const float* a_log,
    const float* dt_bias, float* states, int64_t request_state_stride,
    void* out, int rows, int heads, int kv_ratio, int k_dim, int v_dim,
    float scale, const KdaRequestRows& requests, cudaStream_t stream,
    const KdaStateSnapshots& snap = {}, const KdaReplay& replay = {});

}  // namespace dgpp
