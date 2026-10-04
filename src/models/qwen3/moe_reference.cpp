#include "models/qwen3/moe_reference.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"

namespace dgpp {
namespace {

double bf16_to_d(uint16_t v) { return static_cast<double>(bf16_bits_to_float(v)); }
uint16_t d_to_bf16(double v) { return float_to_bf16_bits(static_cast<float>(v)); }
double sigmoid_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// e2m1(code) x e4m3(scale): exact in double (and in fp32 — at most six
// significant bits); the global is not folded in.
std::vector<double> dequant_fp4(const GlmFp4MatrixHost& m) {
  const int64_t pc = m.cols / 2, sc = m.cols / kFp4Group;
  std::vector<double> w(static_cast<size_t>(m.rows * m.cols));
  for (int64_t r = 0; r < m.rows; ++r)
    for (int64_t c = 0; c < m.cols; ++c) {
      const uint8_t byte = m.payload[static_cast<size_t>(r * pc + c / 2)];
      const uint8_t code = (c & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0xFu);
      const double s = static_cast<double>(fp8_e4m3_bits_to_float(m.scales[static_cast<size_t>(r * sc + c / kFp4Group)]));
      w[static_cast<size_t>(r * m.cols + c)] = static_cast<double>(fp4_e2m1_bits_to_float(code)) * s;
    }
  return w;
}

// out[n] = sum_k act[k] * w[n, k] / divisor for one row, unrounded.
void dots(const uint16_t* act, const std::vector<double>& w, int n, int k, double divisor,
          std::vector<double>& out) {
  out.assign(static_cast<size_t>(n), 0.0);
  for (int nn = 0; nn < n; ++nn) {
    const double* wrow = w.data() + static_cast<size_t>(nn) * static_cast<size_t>(k);
    double acc = 0.0;
    for (int kk = 0; kk < k; ++kk) acc += bf16_to_d(act[kk]) * wrow[kk];
    out[static_cast<size_t>(nn)] = acc / divisor;
  }
}

}  // namespace

size_t Qwen3MoeHostWeights::payload_offset(int index) const {
  const size_t per_expert = payload_bytes(0) + payload_bytes(1) + payload_bytes(2);
  size_t off = static_cast<size_t>(index / 3) * per_expert;
  for (int m = 0; m < index % 3; ++m) off += payload_bytes(m);
  return off;
}

size_t Qwen3MoeHostWeights::scale_offset(int index) const {
  const size_t per_expert = scale_bytes(0) + scale_bytes(1) + scale_bytes(2);
  size_t off = static_cast<size_t>(index / 3) * per_expert;
  for (int m = 0; m < index % 3; ++m) off += scale_bytes(m);
  return off;
}

void Qwen3MoeHostWeights::allocate() {
  payloads.assign(payload_offset(n_experts * 3), 0);
  scales.assign(scale_offset(n_experts * 3), 0);
  globals.assign(static_cast<size_t>(n_experts) * 3, 1.0f);
  router.assign(static_cast<size_t>(n_experts) * static_cast<size_t>(hidden), 0);
}

GlmFp4MatrixHost Qwen3MoeHostWeights::view(int index) const {
  if (index < 0 || index >= n_experts * 3) throw std::invalid_argument("Qwen3MoeHostWeights::view: index out of range");
  const int m = index % 3;
  GlmFp4MatrixHost v;
  v.payload = payloads.data() + payload_offset(index);
  v.scales = scales.data() + scale_offset(index);
  v.global_scale = globals[static_cast<size_t>(index)];
  v.rows = rows(m);
  v.cols = cols(m);
  v.scale_group = kFp4Group;
  return v;
}

void qwen3_moe_ref_forward(const uint16_t* hidden, const Qwen3MoeHostWeights& w,
                           const GlmMoeConfig& cfg, int tokens, std::vector<uint16_t>& out,
                           GlmMoeRouterRef* route) {
  GlmMoeConfig::validate_config(cfg);
  if (cfg.router_mode != MoeRouterMode::SoftmaxTopk || cfg.n_shared_experts != 0)
    throw std::invalid_argument("qwen3_moe_ref_forward: the config must be the routed-only softmax config");
  if (cfg.hidden != w.hidden || cfg.inter != w.inter || cfg.n_experts != w.n_experts)
    throw std::invalid_argument("qwen3_moe_ref_forward: config/weights geometry mismatch");
  if (w.hidden % kFp4Group != 0 || w.inter % kFp4Group != 0)
    throw std::invalid_argument("qwen3_moe_ref_forward: hidden and inter must be multiples of 16");
  const int E = cfg.n_experts, H = cfg.hidden, I = cfg.inter, K = cfg.top_k;

  GlmMoeRouterRef local_route;
  GlmMoeRouterRef& r = route ? *route : local_route;
  glm_moe_ref_router(hidden, w.router.data(), nullptr, cfg, tokens, r);

  // The selected experts' matrices, dequantized on first use.
  std::vector<std::vector<double>> mats(static_cast<size_t>(E) * 3);
  auto mat = [&](int index) -> const std::vector<double>& {
    std::vector<double>& m = mats[static_cast<size_t>(index)];
    if (m.empty()) m = dequant_fp4(w.view(index));
    return m;
  };

  out.assign(static_cast<size_t>(tokens) * static_cast<size_t>(H), 0);
  std::vector<double> gate_d, up_d, down_d, acc(static_cast<size_t>(H));
  std::vector<uint16_t> act(static_cast<size_t>(I));
  for (int t = 0; t < tokens; ++t) {
    const uint16_t* x = hidden + static_cast<size_t>(t) * static_cast<size_t>(H);
    std::fill(acc.begin(), acc.end(), 0.0);
    for (int i = 0; i < K; ++i) {
      const int e = r.ids[static_cast<size_t>(t) * K + i];
      const double we = static_cast<double>(r.weights[static_cast<size_t>(t) * K + i]);
      dots(x, mat(e * 3 + 0), I, H, static_cast<double>(w.globals[static_cast<size_t>(e) * 3 + 0]), gate_d);
      dots(x, mat(e * 3 + 1), I, H, static_cast<double>(w.globals[static_cast<size_t>(e) * 3 + 1]), up_d);
      for (int j = 0; j < I; ++j) {
        // gate and up leave their GEMVs in bf16; silu(gate) * up with the
        // engine's two roundings, no clamps.
        const double g = bf16_to_d(d_to_bf16(gate_d[static_cast<size_t>(j)]));
        const double u = bf16_to_d(d_to_bf16(up_d[static_cast<size_t>(j)]));
        const uint16_t s = d_to_bf16(g * sigmoid_d(g));
        act[static_cast<size_t>(j)] = d_to_bf16(bf16_to_d(s) * u);
      }
      dots(act.data(), mat(e * 3 + 2), H, I, static_cast<double>(w.globals[static_cast<size_t>(e) * 3 + 2]), down_d);
      for (int d = 0; d < H; ++d) acc[static_cast<size_t>(d)] += we * down_d[static_cast<size_t>(d)];
    }
    for (int d = 0; d < H; ++d)
      out[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(d)] = d_to_bf16(acc[static_cast<size_t>(d)]);
  }
}

}  // namespace dgpp
