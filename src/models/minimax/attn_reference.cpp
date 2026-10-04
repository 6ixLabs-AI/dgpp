// Host reference of the MiniMax-M2 attention layer (see the header for the
// pinned numerics).
#include "models/minimax/attn_reference.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/dtypes.hpp"
#include "models/glm4/attn_reference.hpp"

namespace dgpp::minimax_ref {
namespace {

float rb(float v) {
  return bf16_bits_to_float(float_to_bf16_bits(v));
}

// dot[n] = sum_k x[k] * w[n, k] in fp32 (the GEMM interface's fp32 output).
void project(const uint16_t* x, const uint16_t* w, int n, int k, float* dot) {
  for (int nn = 0; nn < n; ++nn) {
    const uint16_t* wr = w + static_cast<int64_t>(nn) * k;
    float acc = 0.f;
    for (int kk = 0; kk < k; ++kk)
      acc = std::fma(bf16_bits_to_float(x[kk]), bf16_bits_to_float(wr[kk]), acc);
    dot[nn] = acc;
  }
}

}  // namespace

Geometry Geometry::from_config(const MinimaxTextConfig& cfg, int rank, int world) {
  if (world < 1 || rank < 0 || rank >= world || cfg.num_attention_heads % world != 0 ||
      cfg.num_key_value_heads % world != 0)
    throw std::invalid_argument(
        "minimax_ref::Geometry: world must divide the query and the kv heads");
  Geometry g;
  g.hidden = cfg.hidden_size;
  g.heads = cfg.num_attention_heads;
  g.kv_heads = cfg.num_key_value_heads;
  g.dim = cfg.head_dim;
  g.rotary_dim = cfg.rotary_dim;
  g.eps = cfg.rms_norm_eps;
  g.scale = cfg.attention_scale();
  g.local_heads = g.heads / world;
  g.head_begin = g.local_heads * rank;
  g.local_kv = g.kv_heads / world;
  g.kv_begin = g.local_kv * rank;
  return g;
}

void qk_finish(const float* dot, const uint16_t* norm, float eps, const float* inv_freq,
               int rotary_dim, int64_t pos, int heads, int dim, uint16_t* out) {
  const int n = heads * dim;
  std::vector<float> x(static_cast<size_t>(n));
  // The Linear's rounding, then ONE mean of squares over every head.
  double ss = 0.0;
  for (int i = 0; i < n; ++i) {
    x[static_cast<size_t>(i)] = rb(dot[i]);
    ss += static_cast<double>(x[static_cast<size_t>(i)]) * x[static_cast<size_t>(i)];
  }
  const float rstd = 1.0f / std::sqrt(static_cast<float>(ss / n) + eps);
  for (int i = 0; i < n; ++i) {
    const float u = rb(x[static_cast<size_t>(i)] * rstd);
    x[static_cast<size_t>(i)] = rb(bf16_bits_to_float(norm[i]) * u);
  }
  // Half-split pairs (i, i + half) on each head: transformers' rotate_half.
  const int half = rotary_dim / 2;
  for (int h = 0; h < heads; ++h) {
    float* xh = x.data() + static_cast<size_t>(h) * dim;
    for (int i = 0; i < half; ++i) {
      const float ang = static_cast<float>(pos) * inv_freq[i];
      const float c = rb(std::cos(ang));
      const float s = rb(std::sin(ang));
      const float x1 = xh[i], x2 = xh[i + half];
      xh[i] = rb(rb(x1 * c) + rb(-x2 * s));
      xh[i + half] = rb(rb(x2 * c) + rb(x1 * s));
    }
  }
  for (int i = 0; i < n; ++i) out[i] = float_to_bf16_bits(x[static_cast<size_t>(i)]);
}

void layer_forward(const HostWeights& w, const Geometry& g, const float* inv_freq,
                   const uint16_t* hidden_in, HostState& state, int tokens, float* out,
                   Taps* taps) {
  if (tokens <= 0) return;
  if (g.local_heads <= 0 || g.local_kv <= 0 || g.local_heads % g.local_kv != 0 ||
      g.head_begin / (g.heads / g.kv_heads) != g.kv_begin)
    throw std::invalid_argument(
        "minimax_ref::layer_forward: a rank's query heads must be its kv heads' groups");
  const int H = g.hidden, d = g.dim, qw = g.q_width(), kw = g.kv_width();
  const int lq = g.local_heads * d, lk = g.local_kv * d;
  std::vector<float> dot(static_cast<size_t>(std::max(qw, kw)));
  std::vector<uint16_t> q_all(static_cast<size_t>(qw)), k_all(static_cast<size_t>(kw));
  std::vector<uint16_t> attn(static_cast<size_t>(lq));
  std::vector<float> c;
  if (taps) {
    taps->q.clear();
    taps->k.clear();
    taps->v.clear();
    taps->heads.clear();
  }
  for (int t = 0; t < tokens; ++t) {
    const uint16_t* x = hidden_in + static_cast<int64_t>(t) * H;
    const int64_t pos = state.first_pos + state.num_tokens;
    // q and k: the WHOLE row projected and finished, then this rank's slice.
    project(x, w.q_proj, qw, H, dot.data());
    qk_finish(dot.data(), w.q_norm, g.eps, inv_freq, g.rotary_dim, pos, g.heads, d, q_all.data());
    project(x, w.k_proj, kw, H, dot.data());
    qk_finish(dot.data(), w.k_norm, g.eps, inv_freq, g.rotary_dim, pos, g.kv_heads, d,
              k_all.data());
    const uint16_t* q = q_all.data() + static_cast<size_t>(g.head_begin) * d;
    // The K/V rows of this token.
    state.k.insert(state.k.end(), k_all.begin() + static_cast<int64_t>(g.kv_begin) * d,
                   k_all.begin() + static_cast<int64_t>(g.kv_begin) * d + lk);
    project(x, w.v_proj, lk, H, dot.data());
    for (int i = 0; i < lk; ++i) state.v.push_back(float_to_bf16_bits(dot[static_cast<size_t>(i)]));
    ++state.num_tokens;
    // Attention over rows [0, pos] of the request, the heads' output rounded.
    glm4_ref::attention(q, state.k.data(), state.v.data(), static_cast<int>(state.num_tokens),
                        g.local_heads, g.local_kv, d, g.scale, c);
    for (int i = 0; i < lq; ++i)
      attn[static_cast<size_t>(i)] = float_to_bf16_bits(c[static_cast<size_t>(i)]);
    project(attn.data(), w.o_proj, H, lq, out + static_cast<int64_t>(t) * H);
    if (taps) {
      taps->q.insert(taps->q.end(), q, q + lq);
      taps->k.insert(taps->k.end(), state.k.end() - lk, state.k.end());
      taps->v.insert(taps->v.end(), state.v.end() - lk, state.v.end());
      taps->heads.insert(taps->heads.end(), attn.begin(), attn.end());
    }
  }
}

}  // namespace dgpp::minimax_ref
