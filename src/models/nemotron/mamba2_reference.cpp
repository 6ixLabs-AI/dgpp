#include "models/nemotron/mamba2_reference.hpp"

#include <cmath>
#include <stdexcept>

#include "common/dtypes.hpp"

namespace dgpp::mamba2_ref {
namespace {

template <typename Acc>
Acc softplus(Acc x) {
  // torch.nn.functional.softplus, beta 1, threshold 20 (mamba_ssm's kernels
  // use the same cut).
  return x > Acc(20) ? x : std::log1p(std::exp(x));
}

template <typename Acc>
Acc silu(Acc x) {
  return x / (Acc(1) + std::exp(-x));
}

template <typename Acc>
Acc rounded(Acc v, Rounding r) {
  if (r == Rounding::None) return v;
  return static_cast<Acc>(bf16_bits_to_float(float_to_bf16_bits(static_cast<float>(v))));
}

void check(const Geometry& g) {
  if (g.heads <= 0 || g.head_dim <= 0 || g.state <= 0 || g.groups <= 0 || g.conv_kernel < 2)
    throw std::invalid_argument("mamba2_ref: geometry must be positive (conv_kernel >= 2)");
  if (g.heads % g.groups != 0) throw std::invalid_argument("mamba2_ref: groups must divide heads");
}

}  // namespace

template <typename Acc>
void conv_silu(const Geometry& g, const Weights& w, const float* xbc, int64_t stride, int tokens,
               Acc* tail, Acc* out, Rounding rounding) {
  check(g);
  const int K = g.conv_kernel, cd = g.conv_dim();
  // Row r of the window: tail rows 0 .. K-2 (oldest first), then the chunk.
  auto in = [&](int64_t r, int ch) -> Acc {
    return r < K - 1 ? tail[r * cd + ch] : static_cast<Acc>(xbc[(r - (K - 1)) * stride + ch]);
  };
  for (int t = 0; t < tokens; ++t)
    for (int ch = 0; ch < cd; ++ch) {
      Acc acc = w.conv_b != nullptr ? static_cast<Acc>(w.conv_b[ch]) : Acc(0);
      for (int j = 0; j < K; ++j)
        acc += static_cast<Acc>(w.conv_w[static_cast<int64_t>(ch) * K + j]) *
               in(static_cast<int64_t>(t) + j, ch);
      out[static_cast<int64_t>(t) * cd + ch] = rounded(silu(acc), rounding);
    }
  // The new tail: the last K - 1 rows of [tail | chunk].
  std::vector<Acc> next(static_cast<size_t>(K - 1) * static_cast<size_t>(cd));
  for (int r = 0; r < K - 1; ++r)
    for (int ch = 0; ch < cd; ++ch)
      next[static_cast<size_t>(r) * static_cast<size_t>(cd) + static_cast<size_t>(ch)] =
          in(static_cast<int64_t>(tokens) + r, ch);
  for (size_t i = 0; i < next.size(); ++i) tail[i] = next[i];
}

template <typename Acc>
void scan(const Geometry& g, const Weights& w, const Acc* conv, const float* dt_raw,
          int64_t dt_stride, int tokens, Acc* ssm, Acc* y, Acc* dt_out, Rounding rounding) {
  check(g);
  const int hd = g.head_dim, N = g.state, inner = g.inner(), cd = g.conv_dim();
  const int per_group = g.heads / g.groups;
  for (int t = 0; t < tokens; ++t) {
    const Acc* row = conv + static_cast<int64_t>(t) * cd;
    for (int h = 0; h < g.heads; ++h) {
      const int grp = h / per_group;
      const Acc* x = row + static_cast<int64_t>(h) * hd;
      const Acc* B = row + inner + static_cast<int64_t>(grp) * N;
      const Acc* C =
          row + inner + static_cast<int64_t>(g.groups) * N + static_cast<int64_t>(grp) * N;
      const Acc dt =
          softplus<Acc>(static_cast<Acc>(dt_raw[static_cast<int64_t>(t) * dt_stride + h]) +
                        static_cast<Acc>(w.dt_bias[h]));
      if (dt_out != nullptr) dt_out[static_cast<int64_t>(t) * g.heads + h] = dt;
      const Acc decay = std::exp(-std::exp(static_cast<Acc>(w.a_log[h])) * dt);
      Acc* s = ssm + static_cast<int64_t>(h) * hd * N;
      for (int p = 0; p < hd; ++p) {
        Acc* sp = s + static_cast<int64_t>(p) * N;
        const Acc dtx = dt * x[p];
        Acc o = 0;
        for (int n = 0; n < N; ++n) {
          sp[n] = sp[n] * decay + dtx * B[n];
          o += sp[n] * C[n];
        }
        y[static_cast<int64_t>(t) * inner + static_cast<int64_t>(h) * hd + p] =
            rounded(o + static_cast<Acc>(w.d[h]) * x[p], rounding);
      }
    }
  }
}

template <typename Acc>
void gated_group_rmsnorm(const Geometry& g, const Acc* y, const float* gate, int64_t gate_stride,
                         const float* norm_w, Acc* out, int tokens, double eps, Rounding rounding) {
  check(g);
  const int inner = g.inner(), gs = inner / g.groups;
  std::vector<Acc> u(static_cast<size_t>(gs));
  for (int t = 0; t < tokens; ++t)
    for (int grp = 0; grp < g.groups; ++grp) {
      const int64_t base = static_cast<int64_t>(grp) * gs;
      Acc ss = 0;
      for (int i = 0; i < gs; ++i) {
        const Acc z = static_cast<Acc>(gate[static_cast<int64_t>(t) * gate_stride + base + i]);
        u[static_cast<size_t>(i)] = y[static_cast<int64_t>(t) * inner + base + i] * silu(z);
        ss += u[static_cast<size_t>(i)] * u[static_cast<size_t>(i)];
      }
      const Acc rstd = Acc(1) / std::sqrt(ss / static_cast<Acc>(gs) + static_cast<Acc>(eps));
      for (int i = 0; i < gs; ++i)
        out[static_cast<int64_t>(t) * inner + base + i] = rounded(
            u[static_cast<size_t>(i)] * rstd * static_cast<Acc>(norm_w[base + i]), rounding);
    }
}

template <typename Acc>
void mixer(const Geometry& g, const Weights& w, const float* proj, int tokens, State<Acc>& state,
           double eps, Acc* out, Rounding rounding, Trace<Acc>* trace) {
  check(g);
  const int inner = g.inner(), cd = g.conv_dim(), rows = g.proj_rows();
  std::vector<Acc> conv(static_cast<size_t>(tokens) * static_cast<size_t>(cd));
  std::vector<Acc> y(static_cast<size_t>(tokens) * static_cast<size_t>(inner));
  std::vector<Acc> dt(static_cast<size_t>(tokens) * static_cast<size_t>(g.heads));
  conv_silu<Acc>(g, w, proj + inner, rows, tokens, state.conv.data(), conv.data(), rounding);
  scan<Acc>(g, w, conv.data(), proj + inner + cd, rows, tokens, state.ssm.data(), y.data(),
            dt.data(), rounding);
  gated_group_rmsnorm<Acc>(g, y.data(), proj, rows, w.norm_w, out, tokens, eps, rounding);
  if (trace != nullptr) {
    trace->conv.insert(trace->conv.end(), conv.begin(), conv.end());
    trace->dt.insert(trace->dt.end(), dt.begin(), dt.end());
    trace->scan.insert(trace->scan.end(), y.begin(), y.end());
  }
}

template void conv_silu<float>(const Geometry&, const Weights&, const float*, int64_t, int, float*,
                               float*, Rounding);
template void conv_silu<double>(const Geometry&, const Weights&, const float*, int64_t, int,
                                double*, double*, Rounding);
template void scan<float>(const Geometry&, const Weights&, const float*, const float*, int64_t, int,
                          float*, float*, float*, Rounding);
template void scan<double>(const Geometry&, const Weights&, const double*, const float*, int64_t,
                           int, double*, double*, double*, Rounding);
template void gated_group_rmsnorm<float>(const Geometry&, const float*, const float*, int64_t,
                                         const float*, float*, int, double, Rounding);
template void gated_group_rmsnorm<double>(const Geometry&, const double*, const float*, int64_t,
                                          const float*, double*, int, double, Rounding);
template void mixer<float>(const Geometry&, const Weights&, const float*, int, State<float>&,
                           double, float*, Rounding, Trace<float>*);
template void mixer<double>(const Geometry&, const Weights&, const float*, int, State<double>&,
                            double, double*, Rounding, Trace<double>*);

}  // namespace dgpp::mamba2_ref
