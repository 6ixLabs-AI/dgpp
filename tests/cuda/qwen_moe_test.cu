// The Qwen3.8-Flash-Next MoE: the SoftmaxTopk router at
// the real geometry (E=512, H=2560, K=10) against the double oracle — its
// three device forms (tiled dots, warp dots, fused select) bitwise one
// another, every id equal to the oracle's or certified as a near-tie on
// the bf16 logits, every weight a bf16 value within an ulp of the
// oracle's — and the whole layer (routed FP8 experts + the BF16 shared
// expert under its sigmoid gate) on small geometry against the oracle on
// both expert kernels, at a GEMV-shaped and a GEMM-shaped token count,
// deterministic run to run.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/qwen_moe.hpp"
#include "kernels/scale_gemm.hpp"
#include "loaders/fp8_quant.hpp"
#include "models/glm/moe.hpp"
#include "models/glm/moe_reference.hpp"
#include "models/qwen/moe_layer.hpp"
#include "models/qwen/moe_reference.hpp"
#include "models/quant_matrix.hpp"

using namespace dgpp::kda_test;
using dgpp::GlmMoeConfig;
using dgpp::GlmMoeRouterRef;
using dgpp::GlmQuantMatrix;
using dgpp::QwenMoeHostWeights;
using dgpp::QwenMoeLayer;
using dgpp::QwenMoeWeights;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

double bf16_ulp_at(double x) {
  if (x == 0.0) return std::ldexp(1.0, -133);
  int e = 0;
  std::frexp(std::fabs(x), &e);
  return std::ldexp(1.0, e - 8);
}

// ---- the router at the real geometry --------------------------------------

struct RouterRun {
  std::vector<int32_t> ids;
  std::vector<float> w;
  std::vector<float> biased;
};

RouterRun run_router(const GlmMoeConfig& cfg, int tokens, const uint16_t* d_x,
                     const uint16_t* d_gate, int mode) {
  const int E = cfg.n_experts, K = cfg.top_k;
  DevBuf ids(static_cast<size_t>(tokens) * K * 4), w(static_cast<size_t>(tokens) * K * 4),
      scores(static_cast<size_t>(tokens) * E * 4), biased(static_cast<size_t>(tokens) * E * 4),
      counters(static_cast<size_t>(tokens) * 4);
  DGPP_CUDA_OK(cudaMemset(counters.p, 0, static_cast<size_t>(tokens) * 4));
  cudaStream_t st = test_stream();
  // mode 0: the tiled dots + select; 1: the warp dots + select; 2: fused.
  dgpp::launch_moe_router(d_x, d_gate, nullptr, static_cast<int32_t*>(ids.p),
                          static_cast<float*>(w.p), static_cast<float*>(scores.p),
                          static_cast<float*>(biased.p), cfg, tokens, st,
                          mode == 2 ? static_cast<int*>(counters.p) : nullptr,
                          /*allow_tiled=*/mode == 0);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  RouterRun r;
  r.ids.resize(static_cast<size_t>(tokens) * K);
  r.w.resize(static_cast<size_t>(tokens) * K);
  r.biased.resize(static_cast<size_t>(tokens) * E);
  ids.download(r.ids.data(), r.ids.size() * 4);
  w.download(r.w.data(), r.w.size() * 4);
  biased.download(r.biased.data(), r.biased.size() * 4);
  if (mode == 2) {
    std::vector<int> c(static_cast<size_t>(tokens));
    counters.download(c.data(), c.size() * 4);
    for (int v : c) require(v == 0, "fused router left a ticket set");
  }
  return r;
}

}  // namespace

DGPP_TEST(qwen_moe_softmax_router_matches_the_oracle_at_real_geometry) {
  GlmMoeConfig cfg = QwenMoeLayer::routed_config(2560, 640, 512, 10, true);
  const int tokens = 48, E = cfg.n_experts, H = cfg.hidden, K = cfg.top_k;
  const std::vector<uint16_t> x = random_bf16_normal(101, static_cast<int64_t>(tokens) * H, 1.0f);
  const std::vector<uint16_t> gate = random_bf16_normal(102, static_cast<int64_t>(E) * H, 0.04f);
  DevBuf dx(x.size() * 2), dg(gate.size() * 2);
  dx.upload(x.data(), x.size() * 2);
  dg.upload(gate.data(), gate.size() * 2);
  const uint16_t* px = static_cast<const uint16_t*>(dx.p);
  const uint16_t* pg = static_cast<const uint16_t*>(dg.p);

  const RouterRun tiled = run_router(cfg, tokens, px, pg, 0);
  const RouterRun warp = run_router(cfg, tokens, px, pg, 1);
  const RouterRun fused = run_router(cfg, tokens, px, pg, 2);
  require(tiled.ids == warp.ids && tiled.ids == fused.ids, "router forms: ids differ");
  require(std::memcmp(tiled.w.data(), warp.w.data(), tiled.w.size() * 4) == 0 &&
              std::memcmp(tiled.w.data(), fused.w.data(), tiled.w.size() * 4) == 0,
          "router forms: weights differ");
  require(std::memcmp(tiled.biased.data(), warp.biased.data(), tiled.biased.size() * 4) == 0 &&
              std::memcmp(tiled.biased.data(), fused.biased.data(), tiled.biased.size() * 4) == 0,
          "router forms: logit rows differ");

  GlmMoeRouterRef ref;
  dgpp::glm_moe_ref_router(x.data(), gate.data(), nullptr, cfg, tokens, ref);

  long certified = 0;
  double max_w_ulps = 0, max_logit_ulps = 0;
  for (int t = 0; t < tokens; ++t) {
    // Ids: equal, or a near-tie swap certified on the oracle's logits — a
    // swapped pair's logits are within one bf16 ulp (both sides round the
    // fp32 dot to bf16; a swap needs the two roundings to land apart).
    std::vector<int> got(tiled.ids.begin() + t * K, tiled.ids.begin() + (t + 1) * K);
    std::vector<int> exp(ref.ids.begin() + t * K, ref.ids.begin() + (t + 1) * K);
    require(std::is_sorted(got.begin(), got.end()), "ids not ascending");
    std::vector<int> only_got, only_exp;
    for (int e : got)
      if (std::find(exp.begin(), exp.end(), e) == exp.end()) only_got.push_back(e);
    for (int e : exp)
      if (std::find(got.begin(), got.end(), e) == got.end()) only_exp.push_back(e);
    require(only_got.size() == only_exp.size(), "id set sizes differ");
    for (size_t i = 0; i < only_got.size(); ++i) {
      const double la = ref.biased[static_cast<size_t>(t) * E + only_got[i]];
      const double lb = ref.biased[static_cast<size_t>(t) * E + only_exp[i]];
      require(std::fabs(la - lb) <= 1.01 * bf16_ulp_at(std::max(std::fabs(la), std::fabs(lb))),
              "router id swap beyond the near-tie certification");
      ++certified;
    }
    // The exported row is the bf16 logit (a bf16 value) within an ulp of
    // the oracle's rounded logit.
    for (int e = 0; e < E; ++e) {
      const float l = tiled.biased[static_cast<size_t>(t) * E + e];
      require(dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(l)) == l, "logit not a bf16 value");
      const double d = std::fabs(l - ref.biased[static_cast<size_t>(t) * E + e]);
      max_logit_ulps = std::max(max_logit_ulps, d / bf16_ulp_at(ref.biased[static_cast<size_t>(t) * E + e]));
    }
    // Weights: bf16 values; where the ids agree, within an ulp of the
    // oracle's; the row sums to one within bf16 rounding.
    double sum = 0;
    for (int i = 0; i < K; ++i) {
      const float w = tiled.w[static_cast<size_t>(t) * K + i];
      require(dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(w)) == w, "weight not a bf16 value");
      require(w > 0.f && w <= 1.f, "weight outside (0, 1]");
      sum += w;
      if (only_got.empty()) {
        const int u = bf16_ulps(dgpp::float_to_bf16_bits(w),
                                dgpp::float_to_bf16_bits(ref.weights[static_cast<size_t>(t) * K + i]));
        max_w_ulps = std::max(max_w_ulps, static_cast<double>(u));
      }
    }
    require(std::fabs(sum - 1.0) < 0.02, "weights do not sum to one");
  }
  std::printf("[ .. ] softmax router: %d tokens x %d experts, %ld certified swaps, "
              "logits within %.2f ulp, weights within %g ulp\n",
              tokens, E, certified, max_logit_ulps, max_w_ulps);
  require(max_logit_ulps <= 1.0, "logits beyond one bf16 ulp of the oracle");
  require(max_w_ulps <= 1.0, "weights beyond one bf16 ulp of the oracle");
  require(certified <= tokens, "too many near-tie swaps to trust");
}

// ---- the whole layer on small geometry --------------------------------------

namespace {

struct SmallCase {
  GlmMoeConfig cfg;
  QwenMoeHostWeights hw;
  int tokens = 0;
  std::vector<uint16_t> x;
  // device
  DevBuf dx, drouter, dsg, dsw0, dsw1, dsw2;
  std::vector<DevBuf> dpayload, dscale;
  std::vector<GlmQuantMatrix> experts;
  QwenMoeWeights dev;

  // hidden x inter (the routed slice width), the slice axis's scale grid
  // `scale_block` (128 = the checkpoint's; 64 / 32 = the TP=2 / TP=4
  // re-blocked slices of plan D2).
  static SmallCase make(int tokens, uint64_t seed, int hidden = 256, int inter = 128,
                        int scale_block = 128) {
    SmallCase c;
    c.cfg = QwenMoeLayer::routed_config(hidden, inter, 16, 4, true);
    c.tokens = tokens;
    QwenMoeHostWeights& w = c.hw;
    w.hidden = c.cfg.hidden;
    w.inter = c.cfg.inter;
    w.n_experts = c.cfg.n_experts;
    w.shared_inter = 192;
    w.scale_block = scale_block;
    w.allocate();
    const int H = w.hidden, E = w.n_experts, S = w.shared_inter;
    w.router = random_bf16_normal(seed + 1, static_cast<int64_t>(E) * H, 0.05f);
    w.shared_gate = random_bf16_normal(seed + 2, H, 0.05f);
    w.shared_w[0] = random_bf16_normal(seed + 3, static_cast<int64_t>(S) * H, 0.06f);
    w.shared_w[1] = random_bf16_normal(seed + 4, static_cast<int64_t>(S) * H, 0.06f);
    w.shared_w[2] = random_bf16_normal(seed + 5, static_cast<int64_t>(H) * S, 0.06f);
    // fp8 payloads and scales: e4m3 of N(0, 1.5) codes, scales in [0.5, 2)
    // times a small factor so the expert outputs stay O(1).
    for (size_t i = 0; i < w.payloads.size(); ++i)
      w.payloads[i] = dgpp::float_to_fp8_e4m3_bits(1.5f * normal_f(seed + 6, static_cast<int64_t>(i)));
    for (size_t i = 0; i < w.scales.size(); ++i)
      w.scales[i] = 0.02f * std::exp2(uniform_pm1(seed + 7, static_cast<int64_t>(i), 1.0f));
    c.x = random_bf16_normal(seed + 8, static_cast<int64_t>(tokens) * H, 1.0f);

    c.dx = DevBuf(c.x.size() * 2);
    c.dx.upload(c.x.data(), c.x.size() * 2);
    c.drouter = DevBuf(w.router.size() * 2);
    c.drouter.upload(w.router.data(), w.router.size() * 2);
    c.dsg = DevBuf(w.shared_gate.size() * 2);
    c.dsg.upload(w.shared_gate.data(), w.shared_gate.size() * 2);
    c.dsw0 = DevBuf(w.shared_w[0].size() * 2);
    c.dsw0.upload(w.shared_w[0].data(), w.shared_w[0].size() * 2);
    c.dsw1 = DevBuf(w.shared_w[1].size() * 2);
    c.dsw1.upload(w.shared_w[1].data(), w.shared_w[1].size() * 2);
    c.dsw2 = DevBuf(w.shared_w[2].size() * 2);
    c.dsw2.upload(w.shared_w[2].data(), w.shared_w[2].size() * 2);
    c.experts.resize(static_cast<size_t>(E) * 3);
    for (int i = 0; i < E * 3; ++i) {
      const int m = i % 3;
      DevBuf p(w.payload_bytes(m)), s(w.scale_count(m) * 4);
      p.upload(w.payloads.data() + w.payload_offset(i), w.payload_bytes(m));
      s.upload(w.scales.data() + w.scale_offset(i), w.scale_count(m) * 4);
      GlmQuantMatrix q;
      q.payload = static_cast<const uint8_t*>(p.p);
      q.scales = static_cast<const float*>(s.p);
      q.rows = w.rows(m);
      q.cols = w.cols(m);
      q.scale_block_rows = w.scale_block_rows(m);
      q.scale_block_cols = w.scale_block_cols(m);
      c.experts[static_cast<size_t>(i)] = q;
      c.dpayload.push_back(std::move(p));
      c.dscale.push_back(std::move(s));
    }
    c.dev.router = static_cast<const uint16_t*>(c.drouter.p);
    c.dev.shared_gate = static_cast<const uint16_t*>(c.dsg.p);
    c.dev.shared_gate_proj = static_cast<const uint16_t*>(c.dsw0.p);
    c.dev.shared_up_proj = static_cast<const uint16_t*>(c.dsw1.p);
    c.dev.shared_down_proj = static_cast<const uint16_t*>(c.dsw2.p);
    c.dev.shared_inter = S;
    c.dev.experts = c.experts.data();
    return c;
  }
};

// The GLM expert-path budget (glm_moe_test): per-element bf16 ulps against
// the double oracle — hard 12, soft 4 on < 2 % — with an absolute floor at
// 2 % of the output's RMS (cancellation), and a relative l2.
void require_within_expert_budget(const std::vector<uint16_t>& got,
                                  const std::vector<uint16_t>& oracle, const char* label) {
  require(got.size() == oracle.size(), "size mismatch");
  double rms = 0;
  for (uint16_t v : oracle) rms += std::pow(dgpp::bf16_bits_to_float(v), 2);
  rms = std::sqrt(rms / static_cast<double>(oracle.size()));
  const double floor_abs = 2e-2 * rms;
  long hard = 0, soft = 0;
  double max_ulps = 0, sum_d2 = 0, sum_o2 = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = dgpp::bf16_bits_to_float(got[i]), o = dgpp::bf16_bits_to_float(oracle[i]);
    int u = bf16_ulps(got[i], oracle[i]);
    if (std::fabs(g - o) <= floor_abs) u = 0;
    max_ulps = std::max(max_ulps, static_cast<double>(u));
    sum_d2 += (g - o) * (g - o);
    sum_o2 += o * o;
    if (u > 4) ++soft;
    if (u > 12) ++hard;
  }
  const double l2 = sum_o2 > 0 ? std::sqrt(sum_d2 / sum_o2) : 0;
  std::printf("[ .. ] %s: max %g ulps, %ld/%zu over soft, %ld hard, l2=%.2g (rms %.3g)\n",
              label, max_ulps, soft, got.size(), hard, l2, rms);
  require(hard == 0, std::string(label) + ": hard ulp violations");
  require(static_cast<double>(soft) / static_cast<double>(got.size()) < 0.02,
          std::string(label) + ": soft ulp budget");
  require(l2 < 4e-3, std::string(label) + ": l2 budget");
}

std::vector<uint16_t> run_layer(SmallCase& c, dgpp::IGemm& gemm, dgpp::MoeExpertKernel kernel,
                                GlmMoeRouterRef* route_out = nullptr) {
  QwenMoeLayer layer(c.dev, c.cfg, gemm, c.tokens);
  DevBuf out(static_cast<size_t>(c.tokens) * c.cfg.hidden * 2);
  cudaStream_t st = test_stream();
  layer.enqueue(static_cast<const uint16_t*>(c.dx.p), static_cast<uint16_t*>(out.p), c.tokens,
                st, kernel);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  std::vector<uint16_t> got(static_cast<size_t>(c.tokens) * c.cfg.hidden);
  out.download(got.data(), got.size() * 2);
  if (route_out) {
    route_out->ids = layer.routed().last_ids();
    route_out->weights = layer.routed().last_weights();
  }
  return got;
}

void small_case(int tokens, uint64_t seed) {
  SmallCase c = SmallCase::make(tokens, seed);
  dgpp::CublasLtGemm gemm;
  std::vector<uint16_t> oracle;
  GlmMoeRouterRef ref_route;
  dgpp::qwen_moe_ref_forward(c.x.data(), c.hw, c.cfg, c.tokens, oracle, &ref_route);

  GlmMoeRouterRef dev_route;
  const std::vector<uint16_t> gemv = run_layer(c, gemm, dgpp::MoeExpertKernel::kGemv, &dev_route);
  // Small geometry: no near-ties expected — the routing must be the
  // oracle's exactly (otherwise the output comparison would be moot).
  require(dev_route.ids == ref_route.ids,
          "small case: device routing differs from the oracle's");
  for (size_t i = 0; i < dev_route.weights.size(); ++i)
    require(bf16_ulps(dgpp::float_to_bf16_bits(dev_route.weights[i]),
                      dgpp::float_to_bf16_bits(ref_route.weights[i])) <= 1,
            "small case: routing weight beyond one ulp");
  const std::string label = "tokens=" + std::to_string(tokens);
  require_within_expert_budget(gemv, oracle, (label + " gemv").c_str());
  const std::vector<uint16_t> mma = run_layer(c, gemm, dgpp::MoeExpertKernel::kMma);
  require_within_expert_budget(mma, oracle, (label + " mma").c_str());
  const std::vector<uint16_t> again = run_layer(c, gemm, dgpp::MoeExpertKernel::kGemv);
  require(again == gemv, "small case: not deterministic run to run");
}

}  // namespace

DGPP_TEST(qwen_moe_layer_matches_the_oracle_at_decode_shape) { small_case(3, 200); }

DGPP_TEST(qwen_moe_layer_matches_the_oracle_at_prefill_shape) { small_case(40, 300); }

// ---- the TP slice grids on the tensor-core kernels -------------

namespace {

int log2_of(int b) {
  int sh = 0;
  while ((1 << sh) < b) ++sh;
  require((1 << sh) == b, "scale block not a power of two");
  return sh;
}

// GIVEN a slice geometry (H, I_r) with the sliced axis re-blocked at
// `scale_block` (gate/up rows, down columns — the other axis 128),
// THEN the grouped tensor-core kernel (the ldmatrix fp8 tile kernel: per-row
// scales, one scale column per 32-deep stage) is bitwise the reference tile
// kernel per segment on the same (rs, cs) grid — segments of 1..130 rows,
// ragged n-tiles (I_r % 128 != 0), the fp32 down variant with k = I_r.
void grouped_mma_on_slice_grid(int H, int I, int scale_block, uint64_t seed) {
  constexpr int kRows = 300;
  SmallCase c = SmallCase::make(kRows, seed, H, I, scale_block);
  const int E = c.cfg.n_experts;
  const int lens[] = {1, 7, 64, 3, 130, 95};
  std::vector<dgpp::MoeSegment> segs;
  int row0 = 0;
  for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); ++i) {
    segs.push_back(dgpp::MoeSegment{row0, lens[i], static_cast<int>((i * 5) % E)});
    row0 += lens[i];
  }
  require(row0 == kRows, "segments cover the rows");
  dgpp::MoeSegment* d_segs = nullptr;
  dgpp::MoeExpertView* d_views = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_segs, segs.size() * sizeof(dgpp::MoeSegment)));
  std::memcpy(d_segs, segs.data(), segs.size() * sizeof(dgpp::MoeSegment));
  DGPP_CUDA_OK(cudaMallocManaged(&d_views, static_cast<size_t>(E) * 3 * sizeof(dgpp::MoeExpertView)));
  for (int m = 0; m < E * 3; ++m) {
    dgpp::MoeExpertView v;
    v.payload = c.experts[static_cast<size_t>(m)].payload;
    v.scales = c.experts[static_cast<size_t>(m)].scales;
    v.scale_shift_rows = log2_of(c.experts[static_cast<size_t>(m)].scale_block_rows);
    v.scale_shift_cols = log2_of(c.experts[static_cast<size_t>(m)].scale_block_cols);
    d_views[m] = v;
  }
  const uint16_t* act = static_cast<const uint16_t*>(c.dx.p);  // [kRows, H]
  uint16_t *d_grouped = nullptr, *d_ref = nullptr;
  const size_t gate_bytes = static_cast<size_t>(kRows) * I * 2;
  DGPP_CUDA_OK(cudaMallocManaged(&d_grouped, gate_bytes));
  DGPP_CUDA_OK(cudaMallocManaged(&d_ref, gate_bytes));
  DGPP_CUDA_OK(cudaMemset(d_grouped, 0xA5, gate_bytes));
  DGPP_CUDA_OK(cudaMemset(d_ref, 0x5A, gate_bytes));
  const int rs_gate = log2_of(c.hw.scale_block_rows(0)), cs_gate = log2_of(c.hw.scale_block_cols(0));
  const int rs_down = log2_of(c.hw.scale_block_rows(2)), cs_down = log2_of(c.hw.scale_block_cols(2));
  require(rs_gate == log2_of(scale_block) && cs_gate == 7 && rs_down == 7 &&
              cs_down == log2_of(scale_block),
          "the fixture's grid is the slice grid");
  for (int which = 0; which < 2; ++which) {
    dgpp::launch_moe_grouped_mma_bf16(act, H, d_segs, static_cast<int>(segs.size()), 130,
                                      /*rows_per_block=*/0, d_views, which, d_grouped, I, I,
                                      H, nullptr);
    for (const dgpp::MoeSegment& sg : segs) {
      const GlmQuantMatrix& g = c.experts[static_cast<size_t>(sg.expert) * 3 + which];
      dgpp::launch_scale_gemm_tile_bf16(act + static_cast<size_t>(sg.row0) * H, H, g.payload,
                                        g.scales, d_ref + static_cast<size_t>(sg.row0) * I,
                                        sg.rows, I, H, nullptr, rs_gate, cs_gate);
    }
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(std::memcmp(d_grouped, d_ref, gate_bytes) == 0,
            ("grouped mma " + std::string(which == 0 ? "gate" : "up") +
             " output bitwise the tile GEMM per segment on the " + std::to_string(scale_block) +
             " grid (I=" + std::to_string(I) + ")")
                .c_str());
  }
  {
    const size_t down_bytes = static_cast<size_t>(kRows) * H * 4;
    float *d_gd = nullptr, *d_rd = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_gd, down_bytes));
    DGPP_CUDA_OK(cudaMallocManaged(&d_rd, down_bytes));
    DGPP_CUDA_OK(cudaMemset(d_gd, 0xA5, down_bytes));
    DGPP_CUDA_OK(cudaMemset(d_rd, 0x5A, down_bytes));
    dgpp::launch_moe_grouped_mma_f32(d_grouped, I, d_segs, static_cast<int>(segs.size()), 130,
                                     /*rows_per_block=*/0, d_views, 2, d_gd, H, H, I, nullptr);
    for (const dgpp::MoeSegment& sg : segs) {
      const GlmQuantMatrix& d = c.experts[static_cast<size_t>(sg.expert) * 3 + 2];
      dgpp::launch_scale_gemm_tile_f32(d_grouped + static_cast<size_t>(sg.row0) * I, I,
                                       d.payload, d.scales, d_rd + static_cast<size_t>(sg.row0) * H,
                                       sg.rows, H, I, nullptr, rs_down, cs_down);
    }
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(std::memcmp(d_gd, d_rd, down_bytes) == 0,
            ("grouped mma down output bitwise the tile GEMM per segment (fp32) on the " +
             std::to_string(scale_block) + " grid (I=" + std::to_string(I) + ")")
                .c_str());
    cudaFree(d_gd);
    cudaFree(d_rd);
  }
  std::printf("[ OK ] grouped mma on the %d grid (H=%d, I=%d): %zu segments (1..130 rows) "
              "bitwise the tile GEMM, gate/up/down\n",
              scale_block, H, I, segs.size());
  cudaFree(d_grouped);
  cudaFree(d_ref);
  cudaFree(d_segs);
  cudaFree(d_views);
}

// The device-segmented prefill chain (the model's prefill walk, 2026-09-09)
// on a slice grid: the routing lands in the pinned trace and is the
// oracle's; the output is bitwise the host chain's on the same kernel and
// within the expert budget of the oracle; deterministic run to run.
void prefill_chain_on_slice_grid(int H, int I, int scale_block, int tokens, uint64_t seed) {
  SmallCase c = SmallCase::make(tokens, seed, H, I, scale_block);
  dgpp::CublasLtGemm gemm;
  std::vector<uint16_t> oracle;
  GlmMoeRouterRef ref_route;
  dgpp::qwen_moe_ref_forward(c.x.data(), c.hw, c.cfg, c.tokens, oracle, &ref_route);
  const std::string label = "H=" + std::to_string(H) + " I=" + std::to_string(I) + " grid " +
                            std::to_string(scale_block) + " tokens=" + std::to_string(tokens);
  GlmMoeRouterRef dev_route;
  const std::vector<uint16_t> gemv = run_layer(c, gemm, dgpp::MoeExpertKernel::kGemv, &dev_route);
  require(dev_route.ids == ref_route.ids, label + ": host routing differs from the oracle's");
  require_within_expert_budget(gemv, oracle, (label + " host gemv").c_str());
  const std::vector<uint16_t> mma = run_layer(c, gemm, dgpp::MoeExpertKernel::kMma);
  require_within_expert_budget(mma, oracle, (label + " host mma").c_str());

  QwenMoeLayer layer(c.dev, c.cfg, gemm, c.tokens);
  require(layer.routed().mma_takes_grid(), label + ": the tensor-core chain takes the grid");
  const size_t tk = static_cast<size_t>(tokens) * c.cfg.top_k;
  int32_t* h_ids = nullptr;
  float* h_w = nullptr;
  DGPP_CUDA_OK(cudaMallocHost(&h_ids, tk * 4));
  DGPP_CUDA_OK(cudaMallocHost(&h_w, tk * 4));
  dgpp::MoeTraceStaging trace;
  trace.ids = h_ids;
  trace.weights = h_w;
  trace.biased = nullptr;
  DevBuf out(static_cast<size_t>(tokens) * H * 2);
  cudaStream_t st = test_stream();
  std::vector<uint16_t> prefill(static_cast<size_t>(tokens) * H), again(prefill.size());
  layer.enqueue_prefill(static_cast<const uint16_t*>(c.dx.p), static_cast<uint16_t*>(out.p),
                        tokens, st, &trace);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  out.download(prefill.data(), prefill.size() * 2);
  require(std::vector<int32_t>(h_ids, h_ids + tk) == ref_route.ids,
          label + ": the staged routing differs from the oracle's");
  for (size_t i = 0; i < tk; ++i)
    require(bf16_ulps(dgpp::float_to_bf16_bits(h_w[i]),
                      dgpp::float_to_bf16_bits(ref_route.weights[i])) <= 1,
            label + ": staged routing weight beyond one ulp");
  require(prefill == mma, label + ": the prefill chain is not bitwise the host tensor-core chain");
  require_within_expert_budget(prefill, oracle, (label + " prefill chain").c_str());
  layer.enqueue_prefill(static_cast<const uint16_t*>(c.dx.p), static_cast<uint16_t*>(out.p),
                        tokens, st, nullptr);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  out.download(again.data(), again.size() * 2);
  require(again == prefill, label + ": the prefill chain is not deterministic run to run");
  cudaFreeHost(h_ids);
  cudaFreeHost(h_w);
  std::printf("[ OK ] prefill chain %s: routing == oracle, bitwise the host mma chain\n",
              label.c_str());
}

}  // namespace

// The decode path — the routed slot chain plus the fused two-launch shared
// tail — bitwise the host path at every decode row count, on
// the checkpoint grid and the TP=4 slice grid.
void decode_path_case(int tokens, int H, int I, int scale_block, uint64_t seed) {
  SmallCase c = SmallCase::make(tokens, seed, H, I, scale_block);
  dgpp::CublasLtGemm gemm;
  const std::vector<uint16_t> host = run_layer(c, gemm, dgpp::MoeExpertKernel::kGemv);
  QwenMoeLayer layer(c.dev, c.cfg, gemm, c.tokens, /*decode_slots=*/std::max(8, tokens), /*graph_table_slots=*/0);
  DevBuf out(static_cast<size_t>(tokens) * H * 2);
  cudaStream_t st = test_stream();
  layer.enqueue_decode(static_cast<const uint16_t*>(c.dx.p), static_cast<uint16_t*>(out.p), tokens,
                       st, /*table_slot=*/-1);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  std::vector<uint16_t> got(static_cast<size_t>(tokens) * H);
  out.download(got.data(), got.size() * 2);
  const std::string label = "decode tokens=" + std::to_string(tokens) + " I=" + std::to_string(I) +
                            " grid " + std::to_string(scale_block);
  size_t mism = 0;
  for (size_t i = 0; i < got.size(); ++i) mism += got[i] != host[i];
  require(mism == 0, label + ": the decode path differs from the host path in " +
                         std::to_string(mism) + " of " + std::to_string(got.size()) + " elements");
  std::printf("[ OK ] %s: the slot chain + fused shared tail bitwise the host path\n", label.c_str());
}

// The FP8 fused decode tail (2026-09-10, engine.dense_weights = "fp8"):
// the three shared matrices encoded to block FP8, the two-launch tail
// against the unfused fp8 chain (launch_scale_gemm_bf16 x2, the swiglu,
// launch_scale_gemm_f32, the gate, the accumulate, the round): bitwise at
// one, three and six rows.
DGPP_TEST(qwen_moe_wide_fp8_decode_matches_the_host_chain) {
  for (const int tokens : {9, 12, 16}) {
    SmallCase c = SmallCase::make(tokens, 1000 + tokens, 256, 160, 32);
    std::vector<DevBuf> payloads, scales;
    std::vector<GlmQuantMatrix> shared(3);
    for (int m = 0; m < 3; ++m) {
      const int rows = m == 2 ? c.cfg.hidden : c.hw.shared_inter;
      const int cols = m == 2 ? c.hw.shared_inter : c.cfg.hidden;
      std::vector<uint8_t> encoded(static_cast<size_t>(rows) * cols);
      std::vector<float> factors(static_cast<size_t>(dgpp::fp8_quant::scale_rows(rows)) *
                                  dgpp::fp8_quant::scale_cols(cols));
      dgpp::fp8_quant::encode_block128(c.hw.shared_w[m].data(), cols, rows, cols,
                                       encoded.data(), factors.data(), 1);
      payloads.emplace_back(encoded.size());
      scales.emplace_back(factors.size() * sizeof(float));
      payloads.back().upload(encoded.data(), encoded.size());
      scales.back().upload(factors.data(), factors.size() * sizeof(float));
      shared[m].payload = payloads.back().as<uint8_t>();
      shared[m].scales = scales.back().as<float>();
      shared[m].rows = rows;
      shared[m].cols = cols;
    }
    c.dev.shared_fp8 = shared.data();
    c.dev.shared_gate_proj = c.dev.shared_up_proj = c.dev.shared_down_proj = nullptr;
    dgpp::CublasLtGemm gemm;
    QwenMoeLayer layer(c.dev, c.cfg, gemm, tokens, tokens);
    DevBuf out(static_cast<size_t>(tokens) * c.cfg.hidden * 2);
    cudaStream_t stream = test_stream();
    for (const int threshold : {0, 9}) {
      layer.set_mma_from_rows(threshold);
      layer.enqueue(c.dx.as<uint16_t>(), out.as<uint16_t>(), tokens, stream);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      std::vector<uint16_t> expected(static_cast<size_t>(tokens) * c.cfg.hidden), got(expected.size());
      out.download(expected.data(), expected.size() * 2);
      layer.enqueue_decode(c.dx.as<uint16_t>(), out.as<uint16_t>(), tokens, stream);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      out.download(got.data(), got.size() * 2);
      require(got == expected, "wide FP8 decode differs from the matching host lowering");
    }
  }
}

DGPP_TEST(qwen_moe_fp8_fused_tail_is_bitwise_the_fp8_chain) {
  cudaStream_t st = test_stream();
  const int H = 256, S = 64;
  for (const int tokens : {1, 3, 6}) {
    const std::vector<uint16_t> x = random_bf16_normal(401 + tokens, static_cast<int64_t>(tokens) * H, 1.0f);
    const std::vector<uint16_t> gate = random_bf16_normal(402, static_cast<int64_t>(S) * H, 0.05f);
    const std::vector<uint16_t> up = random_bf16_normal(403, static_cast<int64_t>(S) * H, 0.05f);
    const std::vector<uint16_t> down = random_bf16_normal(404, static_cast<int64_t>(H) * S, 0.05f);
    const std::vector<uint16_t> g = random_bf16_normal(405, H, 0.1f);
    std::vector<float> acc(static_cast<size_t>(tokens) * H);
    for (size_t i = 0; i < acc.size(); ++i) acc[i] = 0.01f * static_cast<float>((i * 7919) % 1000) - 5.0f;
    std::vector<uint8_t> gp(gate.size()), upp(up.size()), dp(down.size());
    std::vector<float> gs(static_cast<size_t>(dgpp::fp8_quant::scale_rows(S)) * dgpp::fp8_quant::scale_cols(H)), us(gs.size());
    std::vector<float> ds(static_cast<size_t>(dgpp::fp8_quant::scale_rows(H)) * dgpp::fp8_quant::scale_cols(S));
    dgpp::fp8_quant::encode_block128(gate.data(), H, S, H, gp.data(), gs.data(), 1);
    dgpp::fp8_quant::encode_block128(up.data(), H, S, H, upp.data(), us.data(), 1);
    dgpp::fp8_quant::encode_block128(down.data(), S, H, S, dp.data(), ds.data(), 1);
    DevBuf dx(x.size() * 2), dg(g.size() * 2), dgp(gp.size()), dgs(gs.size() * 4), dupp(upp.size()), dus(us.size() * 4),
        ddp(dp.size()), dds(ds.size() * 4), dacc(acc.size() * 4), dacc2(acc.size() * 4);
    dx.upload(x.data(), x.size() * 2);
    dg.upload(g.data(), g.size() * 2);
    dgp.upload(gp.data(), gp.size());
    dgs.upload(gs.data(), gs.size() * 4);
    dupp.upload(upp.data(), upp.size());
    dus.upload(us.data(), us.size() * 4);
    ddp.upload(dp.data(), dp.size());
    dds.upload(ds.data(), ds.size() * 4);
    dacc.upload(acc.data(), acc.size() * 4);
    dacc2.upload(acc.data(), acc.size() * 4);
    const size_t an = static_cast<size_t>(tokens) * S, on = static_cast<size_t>(tokens) * H;
    // The chain.
    DevBuf dsg(an * 2), dsu(an * 2), dsact(an * 2), dsdown(on * 4), dsw(static_cast<size_t>(tokens) * 4), dout(on * 2);
    std::vector<int32_t> ident(static_cast<size_t>(tokens));
    for (int i = 0; i < tokens; ++i) ident[static_cast<size_t>(i)] = i;
    DevBuf drows(ident.size() * 4);
    drows.upload(ident.data(), ident.size() * 4);
    dgpp::launch_scale_gemm_bf16(dx.as<uint16_t>(), static_cast<size_t>(H), dgp.as<uint8_t>(), static_cast<const float*>(dgs.p),
                                 dsg.as<uint16_t>(), tokens, S, H, st, static_cast<size_t>(S));
    dgpp::launch_scale_gemm_bf16(dx.as<uint16_t>(), static_cast<size_t>(H), dupp.as<uint8_t>(), static_cast<const float*>(dus.p),
                                 dsu.as<uint16_t>(), tokens, S, H, st, static_cast<size_t>(S));
    dgpp::launch_moe_swiglu_clamp(dsg.as<uint16_t>(), dsu.as<uint16_t>(), dsact.as<uint16_t>(), static_cast<int64_t>(an),
                                  std::numeric_limits<float>::infinity(), st);
    dgpp::launch_scale_gemm_f32(dsact.as<uint16_t>(), static_cast<size_t>(S), ddp.as<uint8_t>(), static_cast<const float*>(dds.p),
                                static_cast<float*>(dsdown.p), tokens, H, S, st, static_cast<size_t>(H));
    dgpp::qwen_moe_shared_gate_bf16(dx.as<uint16_t>(), dg.as<uint16_t>(), static_cast<float*>(dsw.p), tokens, H, st);
    dgpp::launch_moe_accum(static_cast<float*>(dacc.p), static_cast<const float*>(dsdown.p), static_cast<const int32_t*>(drows.p),
                           static_cast<const float*>(dsw.p), tokens, H, st);
    dgpp::launch_moe_round_bf16(dout.as<uint16_t>(), static_cast<const float*>(dacc.p), static_cast<int64_t>(on), st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    std::vector<uint16_t> out_chain(on), act_chain(an);
    dout.download(out_chain.data(), on * 2);
    dsact.download(act_chain.data(), an * 2);
    // The fused fp8 tail.
    DevBuf dact2(an * 2), dsw2(static_cast<size_t>(tokens) * 4), dout2(on * 2);
    dgpp::qwen_moe_shared_tail_decode_fp8(dx.as<uint16_t>(), static_cast<size_t>(H), dgp.as<uint8_t>(), static_cast<const float*>(dgs.p),
                                          dupp.as<uint8_t>(), static_cast<const float*>(dus.p), ddp.as<uint8_t>(),
                                          static_cast<const float*>(dds.p), dg.as<uint16_t>(), dact2.as<uint16_t>(),
                                          static_cast<float*>(dsw2.p), static_cast<const float*>(dacc2.p), dout2.as<uint16_t>(),
                                          tokens, H, S, st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    std::vector<uint16_t> out2(on), act2(an);
    dout2.download(out2.data(), on * 2);
    dact2.download(act2.data(), an * 2);
    require(std::memcmp(act2.data(), act_chain.data(), an * 2) == 0, "fp8 fused tail: act differs from the chain");
    require(std::memcmp(out2.data(), out_chain.data(), on * 2) == 0, "fp8 fused tail: out differs from the chain");
    std::printf("[ OK ] the fp8 fused shared tail is bitwise the fp8 chain at %d rows\n", tokens);
    // The routed accumulation folded into the down epilogue: per-slot
    // contributions and weights whose accumulate-kernel sum (fma in slot
    // order) is `acc` up to fp32 — here computed on the host in that order
    // and handed to the plain tail, against the routed tail on the slots.
    {
      const int K = 5;
      std::vector<float> contrib(static_cast<size_t>(tokens) * (K + 1) * H), weights(static_cast<size_t>(tokens) * K);
      for (size_t i = 0; i < contrib.size(); ++i) contrib[i] = 0.001f * static_cast<float>((i * 104729) % 2000) - 1.0f;
      for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.1f + 0.05f * static_cast<float>(i % 7);
      std::vector<float> acc3(on);
      for (int t = 0; t < tokens; ++t)
        for (int c = 0; c < H; ++c) {
          float a = 0.f;
          for (int j = 0; j < K; ++j)
            a = std::fmaf(weights[static_cast<size_t>(t) * K + j], contrib[(static_cast<size_t>(t) * (K + 1) + j) * H + c], a);
          acc3[static_cast<size_t>(t) * H + c] = a;
        }
      DevBuf dcontrib(contrib.size() * 4), dweights(weights.size() * 4), dacc3(on * 4), dact3(an * 2), dsw3(static_cast<size_t>(tokens) * 4),
          dout3(on * 2), dact4(an * 2), dsw4(static_cast<size_t>(tokens) * 4), dout4(on * 2);
      dcontrib.upload(contrib.data(), contrib.size() * 4);
      dweights.upload(weights.data(), weights.size() * 4);
      dacc3.upload(acc3.data(), on * 4);
      dgpp::qwen_moe_shared_tail_decode_fp8(dx.as<uint16_t>(), static_cast<size_t>(H), dgp.as<uint8_t>(), static_cast<const float*>(dgs.p),
                                            dupp.as<uint8_t>(), static_cast<const float*>(dus.p), ddp.as<uint8_t>(),
                                            static_cast<const float*>(dds.p), dg.as<uint16_t>(), dact3.as<uint16_t>(),
                                            static_cast<float*>(dsw3.p), static_cast<const float*>(dacc3.p), dout3.as<uint16_t>(),
                                            tokens, H, S, st);
      dgpp::qwen_moe_shared_tail_decode_routed_fp8(dx.as<uint16_t>(), static_cast<size_t>(H), dgp.as<uint8_t>(), static_cast<const float*>(dgs.p),
                                                   dupp.as<uint8_t>(), static_cast<const float*>(dus.p), ddp.as<uint8_t>(),
                                                   static_cast<const float*>(dds.p), dg.as<uint16_t>(), dact4.as<uint16_t>(),
                                                   static_cast<float*>(dsw4.p), static_cast<const float*>(dcontrib.p),
                                                   static_cast<const float*>(dweights.p), K, dout4.as<uint16_t>(), tokens, H, S, st);
      DGPP_CUDA_OK(cudaStreamSynchronize(st));
      std::vector<uint16_t> out3(on), out4(on);
      dout3.download(out3.data(), on * 2);
      dout4.download(out4.data(), on * 2);
      require(std::memcmp(out3.data(), out4.data(), on * 2) == 0, "fp8 routed tail: out differs from the accumulate + tail chain");
      std::printf("[ OK ] the fp8 tail with the routed accumulation folded is bitwise the chain at %d rows\n", tokens);
    }
  }
}

DGPP_TEST(qwen_moe_decode_path_is_bitwise_the_host_path) {
  decode_path_case(1, 256, 128, 128, 700);
  decode_path_case(3, 256, 128, 128, 701);
  decode_path_case(8, 256, 160, 32, 702);
  decode_path_case(5, 256, 320, 64, 703);
}

DGPP_TEST(qwen_moe_wide_decode_matches_the_host_chain) {
  for (int rows : {9, 12, 16}) {
    decode_path_case(rows, 256, 160, 32, 800 + rows);
    decode_path_case(rows, 256, 320, 64, 900 + rows);
  }
}

DGPP_TEST(qwen_moe_grouped_mma_on_the_tp2_grid_is_bitwise_the_tile_gemm) {
  grouped_mma_on_slice_grid(/*H=*/256, /*I=*/320, /*scale_block=*/64, 0x64);
}

DGPP_TEST(qwen_moe_grouped_mma_on_the_tp4_grid_is_bitwise_the_tile_gemm) {
  grouped_mma_on_slice_grid(/*H=*/256, /*I=*/160, /*scale_block=*/32, 0x32);
}

DGPP_TEST(qwen_moe_prefill_chain_matches_the_oracle_on_the_checkpoint_grid) {
  prefill_chain_on_slice_grid(256, 128, 128, /*tokens=*/40, 400);
}

DGPP_TEST(qwen_moe_prefill_chain_matches_the_oracle_on_the_tp2_grid) {
  prefill_chain_on_slice_grid(256, 320, 64, /*tokens=*/40, 500);
}

DGPP_TEST(qwen_moe_prefill_chain_matches_the_oracle_on_the_tp4_grid) {
  prefill_chain_on_slice_grid(256, 160, 32, /*tokens=*/40, 600);
}

int main() { return dgpp::test::run_all(); }
