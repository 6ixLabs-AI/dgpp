#pragma once
// Host reference of the Nemotron-H Mamba2 mixer between its two projections
// (2026-10-04, docs/nemotron3_plan.md §3): the executable specification of
// the reference numerics (modeling_nemotron_h.py NemotronHMamba2Mixer; the
// served path is mamba_ssm's causal_conv1d_fn + mamba_chunk_scan_combined on
// a prefill and causal_conv1d_update + selective_state_update on a decode
// step, MambaRMSNormGated with norm_before_gate = False) and the oracle the
// device kernels are to be tested against. Acc = float mirrors the device
// fp32 math; Acc = double measures drift and is what the unit test holds to
// tools/nemotron_reference.py.
//
// in_proj's output row is [gate z (inner) | x B C (conv_dim) | dt (heads)]
// with inner = heads * head_dim and conv_dim = inner + 2 * groups * state.
//   conv   c[t, ch] = silu(b[ch] + sum_j w[ch, j] * xBC[t - (K-1) + j, ch])
//          — depthwise, causal: w[ch, K-1] multiplies the current token; rows
//          before the sequence start are zero
//   split  c -> x [heads, head_dim] | B [groups, state] | C [groups, state];
//          head h reads group h / (heads / groups)
//   step   dt = softplus(dt_raw[h] + dt_bias[h])        (no clamp)
//          S[h] = S[h] * exp(-exp(A_log[h]) * dt) + outer(dt * x[h], B[g])
//          y[h] = S[h] C[g] + D[h] * x[h]
//   norm   u = y * silu(z); per group of inner / groups channels
//          u * rsqrt(mean(u^2) + eps) * w
// The chunked SSD algorithm the reference runs on a whole sequence is this
// recurrence regrouped (tools/nemotron_synth.py selftest holds the two
// equal), so one formulation serves prefill and decode.
#include <cstdint>
#include <vector>

namespace dgpp::mamba2_ref {

struct Geometry {
  int heads = 0;
  int head_dim = 0;
  int state = 0;   // ssm_state_size
  int groups = 0;  // n_groups: B/C groups and the gated norm's groups
  int conv_kernel = 0;
  int inner() const { return heads * head_dim; }
  int conv_dim() const { return inner() + 2 * groups * state; }
  int proj_rows() const { return inner() + conv_dim() + heads; }
};

// The layer's vectors as fp32 (the checkpoint's BF16 values, exact).
struct Weights {
  const float* conv_w = nullptr;   // conv1d.weight [conv_dim, K] (the [conv_dim, 1, K] tensor)
  const float* conv_b = nullptr;   // conv1d.bias [conv_dim]
  const float* a_log = nullptr;    // [heads]
  const float* d = nullptr;        // [heads]
  const float* dt_bias = nullptr;  // [heads]
  const float* norm_w = nullptr;   // mixer.norm.weight [inner]
};

// Where the reference's BF16 module rounds between its fp32 kernels: the
// convolution's output, the scan's output and the gated norm's output are
// written in the activations' dtype; the state stays fp32. None keeps every
// value in Acc (the comparison against the numpy reference); Bf16 applies
// those three roundings (read off the reference's dtype flow — no device
// kernel exists yet to hold it to).
enum class Rounding { None, Bf16 };

// One sequence's recurrent state, zero at the start.
template <typename Acc>
struct State {
  std::vector<Acc> conv;  // [K - 1, conv_dim]: the last K - 1 pre-conv rows, oldest first
  std::vector<Acc> ssm;   // [heads, head_dim, state]
  explicit State(const Geometry& g)
      : conv(static_cast<size_t>(g.conv_kernel - 1) * static_cast<size_t>(g.conv_dim()), Acc(0)),
        ssm(static_cast<size_t>(g.heads) * static_cast<size_t>(g.head_dim) *
                static_cast<size_t>(g.state),
            Acc(0)) {}
};

// The intermediates of a mixer() call, for tests and dumps.
template <typename Acc>
struct Trace {
  std::vector<Acc> conv;  // [tokens, conv_dim] after the conv and SiLU
  std::vector<Acc> dt;    // [tokens, heads] after softplus
  std::vector<Acc> scan;  // [tokens, inner]: S C + D x, before the gated norm
};

// The convolution and SiLU over `tokens` rows of xBC (row stride `stride`),
// reading and advancing the tail. out: [tokens, conv_dim].
template <typename Acc>
void conv_silu(const Geometry& g, const Weights& w, const float* xbc, int64_t stride, int tokens,
               Acc* tail, Acc* out, Rounding rounding);

// The recurrence over `tokens` rows: conv [tokens, conv_dim] is conv_silu's
// output, dt_raw rows have stride `dt_stride`. ssm is updated in place.
// y: [tokens, inner]; dt_out: [tokens, heads] or null.
template <typename Acc>
void scan(const Geometry& g, const Weights& w, const Acc* conv, const float* dt_raw,
          int64_t dt_stride, int tokens, Acc* ssm, Acc* y, Acc* dt_out, Rounding rounding);

// The gated norm: y [tokens, inner], gate rows of stride `gate_stride`.
template <typename Acc>
void gated_group_rmsnorm(const Geometry& g, const Acc* y, const float* gate, int64_t gate_stride,
                         const float* norm_w, Acc* out, int tokens, double eps, Rounding rounding);

// The mixer between in_proj and out_proj: proj [tokens, proj_rows] ->
// out [tokens, inner], advancing the state. Feeding a sequence whole or in
// pieces performs the same operations in the same order.
template <typename Acc>
void mixer(const Geometry& g, const Weights& w, const float* proj, int tokens, State<Acc>& state,
           double eps, Acc* out, Rounding rounding = Rounding::None, Trace<Acc>* trace = nullptr);

}  // namespace dgpp::mamba2_ref
