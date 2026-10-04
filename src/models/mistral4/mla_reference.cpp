// Host reference of the Mistral-Small-4 attention layer (see the header for
// the pinned numerics). Templated on the accumulation type: float mirrors
// the device fp32 path, double is the high-precision oracle.
#include "models/mistral4/mla_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "common/dtypes.hpp"

namespace dgpp::mistral4_ref {
namespace {

float rb(float v) {
  return bf16_bits_to_float(float_to_bf16_bits(v));
}

// out[m, n] = bf16(sum_k act[m, k] * w[n, k]), accumulation in Acc.
template <typename Acc>
void gemm_bf16(const uint16_t* act, int64_t act_row_stride, const uint16_t* w, uint16_t* out, int m,
               int n, int k) {
  for (int mm = 0; mm < m; ++mm) {
    const uint16_t* a = act + static_cast<int64_t>(mm) * act_row_stride;
    for (int nn = 0; nn < n; ++nn) {
      const uint16_t* wr = w + static_cast<int64_t>(nn) * k;
      Acc acc = 0;
      for (int kk = 0; kk < k; ++kk)
        acc += static_cast<Acc>(bf16_bits_to_float(a[kk])) *
               static_cast<Acc>(bf16_bits_to_float(wr[kk]));
      out[static_cast<int64_t>(mm) * n + nn] = float_to_bf16_bits(static_cast<float>(acc));
    }
  }
}

// y = bf16(x * rsqrt(mean(x^2) + eps) * w): one rounding.
template <typename Acc>
void rmsnorm_bf16(const uint16_t* x, const uint16_t* w, uint16_t* y, int dim, float eps) {
  Acc ss = 0;
  for (int d = 0; d < dim; ++d)
    ss += static_cast<Acc>(bf16_bits_to_float(x[d])) * static_cast<Acc>(bf16_bits_to_float(x[d]));
  const Acc inv = 1 / std::sqrt(ss / dim + static_cast<Acc>(eps));
  for (int d = 0; d < dim; ++d)
    y[d] = float_to_bf16_bits(static_cast<float>(static_cast<Acc>(bf16_bits_to_float(x[d])) * inv *
                                                 static_cast<Acc>(bf16_bits_to_float(w[d]))));
}

// One query's absorbed attention over cache rows [0, n): q [heads, nope +
// rope] bf16 (rotated and scaled), rows [n, kv_lora + rope] bf16; writes
// out [heads, v] bf16.
template <typename Acc>
void absorbed_attn(const uint16_t* q, const uint16_t* rows, int64_t n, const uint16_t* wkv_b,
                   const Geometry& g, uint16_t* out) {
  const int width = g.row_width();
  std::vector<uint16_t> q_tilde(static_cast<size_t>(width));
  std::vector<Acc> s(static_cast<size_t>(n)), p(static_cast<size_t>(n)),
      c(static_cast<size_t>(g.kv_lora));
  for (int h = 0; h < g.heads; ++h) {
    const uint16_t* w_uk = wkv_b + static_cast<int64_t>(h) * g.kv_head() * g.kv_lora;
    const uint16_t* w_uv = w_uk + static_cast<int64_t>(g.nope) * g.kv_lora;
    const uint16_t* qh = q + static_cast<int64_t>(h) * g.q_head();
    // q~ = [W_uk^T q_nope | q_rope], the absorbed part rounded to bf16.
    for (int cc = 0; cc < g.kv_lora; ++cc) {
      Acc acc = 0;
      for (int d = 0; d < g.nope; ++d)
        acc += static_cast<Acc>(bf16_bits_to_float(qh[d])) *
               static_cast<Acc>(bf16_bits_to_float(w_uk[static_cast<int64_t>(d) * g.kv_lora + cc]));
      q_tilde[static_cast<size_t>(cc)] = float_to_bf16_bits(static_cast<float>(acc));
    }
    for (int i = 0; i < g.rope; ++i) q_tilde[static_cast<size_t>(g.kv_lora + i)] = qh[g.nope + i];
    // Scores over the latent and the rope key; softmax in Acc.
    Acc m = -std::numeric_limits<Acc>::infinity();
    for (int64_t t = 0; t < n; ++t) {
      const uint16_t* row = rows + t * width;
      Acc acc = 0;
      for (int cc = 0; cc < width; ++cc)
        acc += static_cast<Acc>(bf16_bits_to_float(q_tilde[static_cast<size_t>(cc)])) *
               static_cast<Acc>(bf16_bits_to_float(row[cc]));
      s[static_cast<size_t>(t)] = acc * static_cast<Acc>(g.scale);
      m = std::max(m, s[static_cast<size_t>(t)]);
    }
    Acc denom = 0;
    for (int64_t t = 0; t < n; ++t) {
      p[static_cast<size_t>(t)] = std::exp(s[static_cast<size_t>(t)] - m);
      denom += p[static_cast<size_t>(t)];
    }
    // c = sum_t bf16(p_t) * latent_t: the probabilities round to bf16 for
    // the value accumulation, the denominator does not.
    std::fill(c.begin(), c.end(), Acc(0));
    for (int64_t t = 0; t < n; ++t) {
      const uint16_t* row = rows + t * width;
      const Acc pt = static_cast<Acc>(rb(static_cast<float>(p[static_cast<size_t>(t)] / denom)));
      for (int cc = 0; cc < g.kv_lora; ++cc)
        c[static_cast<size_t>(cc)] += pt * static_cast<Acc>(bf16_bits_to_float(row[cc]));
    }
    // out_h = W_uv c, bf16 GEMM rounding.
    for (int d = 0; d < g.v; ++d) {
      Acc acc = 0;
      for (int cc = 0; cc < g.kv_lora; ++cc)
        acc +=
            static_cast<Acc>(bf16_bits_to_float(w_uv[static_cast<int64_t>(d) * g.kv_lora + cc])) *
            c[static_cast<size_t>(cc)];
      out[static_cast<int64_t>(h) * g.v + d] = float_to_bf16_bits(static_cast<float>(acc));
    }
  }
}

}  // namespace

Geometry Geometry::from_config(const Mistral4TextConfig& cfg, int tp_size) {
  if (tp_size < 1 || cfg.num_attention_heads % tp_size != 0)
    throw std::invalid_argument("mistral4_ref::Geometry: tp_size must divide the heads");
  Geometry g;
  g.hidden = cfg.hidden_size;
  g.heads = cfg.num_attention_heads / tp_size;
  g.q_lora = cfg.q_lora_rank;
  g.kv_lora = cfg.kv_lora_rank;
  g.nope = cfg.qk_nope_head_dim;
  g.rope = cfg.qk_rope_head_dim;
  g.v = cfg.v_head_dim;
  g.eps = cfg.rms_norm_eps;
  g.scale = cfg.attention_scale();
  g.llama4_beta = static_cast<float>(cfg.llama4_scaling_beta);
  g.llama4_original = cfg.llama4_original_max_position_embeddings;
  return g;
}

float llama4_scale(float beta, int original, int64_t pos) {
  if (pos < 0 || original <= 0) return 1.0f;
  const float steps = std::floor(static_cast<float>(pos) / static_cast<float>(original));
  return 1.0f + beta * std::log(1.0f + steps);
}

void rope_row(const float* inv_freq, int rope_dim, int64_t pos, uint16_t* cos_sin) {
  if (rope_dim <= 0 || rope_dim % 2 != 0 || inv_freq == nullptr || cos_sin == nullptr)
    throw std::invalid_argument("mistral4_ref::rope_row: bad arguments");
  const int half = rope_dim / 2;
  for (int i = 0; i < half; ++i) {
    // The fp32 position product (the references'), then the trig in double
    // rounded to fp32 and to bf16.
    const float ang = static_cast<float>(pos) * inv_freq[i];
    cos_sin[i] = float_to_bf16_bits(static_cast<float>(std::cos(static_cast<double>(ang))));
    cos_sin[half + i] = float_to_bf16_bits(static_cast<float>(std::sin(static_cast<double>(ang))));
  }
}

void rope_table(const float* inv_freq, int rope_dim, int64_t positions, uint16_t* out) {
  if (positions <= 0 || out == nullptr)
    throw std::invalid_argument("mistral4_ref::rope_table: bad arguments");
  for (int64_t p = 0; p < positions; ++p) rope_row(inv_freq, rope_dim, p, out + p * rope_dim);
}

void rope_interleave_row(uint16_t* x, int rope_dim, const uint16_t* cos_sin) {
  const int half = rope_dim / 2;
  for (int i = 0; i < half; ++i) {
    const float c = bf16_bits_to_float(cos_sin[i]);
    const float s = bf16_bits_to_float(cos_sin[half + i]);
    const float x0 = bf16_bits_to_float(x[2 * i]);
    const float x1 = bf16_bits_to_float(x[2 * i + 1]);
    const float t1 = rb(x0 * c), t2 = rb(x1 * s);
    const float u1 = rb(x1 * c), u2 = rb(x0 * s);
    x[2 * i] = float_to_bf16_bits(t1 - t2);
    x[2 * i + 1] = float_to_bf16_bits(u1 + u2);
  }
}

template <typename Acc>
void layer_forward(const HostWeights& w, const Geometry& g, const float* inv_freq,
                   const uint16_t* hidden_in, HostState& state, int tokens, uint16_t* layer_out,
                   Taps* taps) {
  if (tokens <= 0) return;
  if (g.rope <= 0 || g.rope % 2 != 0)
    throw std::invalid_argument("mistral4_ref::layer_forward: rope must be even");
  const int H = g.hidden, width = g.row_width(), q_rows = g.heads * g.q_head(),
            v_rows = g.heads * g.v;
  std::vector<uint16_t> q_a(static_cast<size_t>(g.q_lora)), q_c(static_cast<size_t>(g.q_lora));
  std::vector<uint16_t> kv_a(static_cast<size_t>(width)), q(static_cast<size_t>(q_rows));
  std::vector<uint16_t> cos_sin(static_cast<size_t>(g.rope)),
      attn(static_cast<size_t>(tokens) * v_rows);
  if (taps) {
    taps->q.assign(static_cast<size_t>(tokens) * q_rows, 0);
    taps->latent.assign(static_cast<size_t>(tokens) * g.kv_lora, 0);
    taps->k_rope.assign(static_cast<size_t>(tokens) * g.rope, 0);
    taps->heads.assign(static_cast<size_t>(tokens) * v_rows, 0);
  }
  for (int t = 0; t < tokens; ++t) {
    const uint16_t* x = hidden_in + static_cast<int64_t>(t) * H;
    const int64_t pos = state.first_pos + state.num_tokens;
    // The two latent projections (the device fuses them into one GEMM).
    gemm_bf16<Acc>(x, H, w.wq_a, q_a.data(), 1, g.q_lora, H);
    gemm_bf16<Acc>(x, H, w.wkv_a, kv_a.data(), 1, width, H);
    rmsnorm_bf16<Acc>(q_a.data(), w.q_a_norm, q_c.data(), g.q_lora, g.eps);
    // The cache row: [normed latent | rotated rope key].
    state.rows.resize(static_cast<size_t>(state.num_tokens + 1) * width);
    uint16_t* row = state.rows.data() + static_cast<size_t>(state.num_tokens) * width;
    rmsnorm_bf16<Acc>(kv_a.data(), w.kv_a_norm, row, g.kv_lora, g.eps);
    rope_row(inv_freq, g.rope, pos, cos_sin.data());
    for (int i = 0; i < g.rope; ++i) row[g.kv_lora + i] = kv_a[static_cast<size_t>(g.kv_lora + i)];
    rope_interleave_row(row + g.kv_lora, g.rope, cos_sin.data());
    ++state.num_tokens;
    // The query: expand, rotate each head's rope slice, then the Llama-4 scale.
    gemm_bf16<Acc>(q_c.data(), g.q_lora, w.wq_b, q.data(), 1, q_rows, g.q_lora);
    for (int h = 0; h < g.heads; ++h)
      rope_interleave_row(q.data() + h * g.q_head() + g.nope, g.rope, cos_sin.data());
    const float l4 = llama4_scale(g.llama4_beta, g.llama4_original, pos);
    if (l4 != 1.0f)
      for (uint16_t& e : q) e = float_to_bf16_bits(bf16_bits_to_float(e) * l4);
    absorbed_attn<Acc>(q.data(), state.rows.data(), state.num_tokens, w.wkv_b, g,
                       attn.data() + static_cast<size_t>(t) * v_rows);
    if (taps) {
      std::copy(q.begin(), q.end(), taps->q.begin() + static_cast<int64_t>(t) * q_rows);
      std::copy(row, row + g.kv_lora, taps->latent.begin() + static_cast<int64_t>(t) * g.kv_lora);
      std::copy(row + g.kv_lora, row + width,
                taps->k_rope.begin() + static_cast<int64_t>(t) * g.rope);
    }
  }
  if (taps) taps->heads = attn;
  gemm_bf16<Acc>(attn.data(), v_rows, w.wo, layer_out, tokens, H, v_rows);
}

// Explicit instantiations for the two oracle precisions.
template void layer_forward<float>(const HostWeights&, const Geometry&, const float*,
                                   const uint16_t*, HostState&, int, uint16_t*, Taps*);
template void layer_forward<double>(const HostWeights&, const Geometry&, const float*,
                                    const uint16_t*, HostState&, int, uint16_t*, Taps*);

}  // namespace dgpp::mistral4_ref
