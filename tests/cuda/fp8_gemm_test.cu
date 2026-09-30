// The opt-in fp8 prefill GEMM (kernels/fp8_gemm, engine.prefill_fp8_gemm,
// 2026-09-30): the per-token quantizer's contract (amax / 448 scales, e4m3
// codes bitwise the host encoder), the GEMM against an exact reference on
// its own quantized inputs (fp32-accumulation agreement), the bf16 output
// as the f32 output rounded once, and the tolerance claim against the
// dequantized bf16 chain the lever replaces.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/fp8_gemm.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    std::exit(1);
  }
}

template <typename T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  explicit Dev(size_t count) : n(count) { DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T))); }
  ~Dev() { cudaFree(p); }
  void upload(const std::vector<T>& v) { DGPP_CUDA_OK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> download() const {
    std::vector<T> v(n);
    DGPP_CUDA_OK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};

struct Shape {
  int m, n, k, sbr;
};

void run(const Shape& sh, std::mt19937& rng) {
  const int m = sh.m, n = sh.n, k = sh.k, sbr = sh.sbr;
  const int groups = k / 128;
  const int scale_rows = (n + sbr - 1) / sbr;
  std::normal_distribution<float> normal(0.f, 1.f);
  std::uniform_real_distribution<float> uni(0.5f, 2.f);
  // Activations: bf16 rows at a padded stride, a few rows scaled up so the
  // per-group amax varies; one all-zero group.
  const size_t stride = static_cast<size_t>(k) + 8;
  std::vector<uint16_t> act(static_cast<size_t>(m) * stride, 0);
  for (int r = 0; r < m; ++r) {
    const float rs = (r % 7 == 0) ? 8.f : 1.f;
    for (int c = 0; c < k; ++c) act[r * stride + c] = dgpp::float_to_bf16_bits(normal(rng) * rs);
  }
  for (int c = 0; c < 128; ++c) act[static_cast<size_t>(m - 1) * stride + c] = 0;  // an all-zero group
  // Weights: e4m3 codes (no NaN codes) and a 128 x sbr scale grid.
  std::vector<uint8_t> w(static_cast<size_t>(n) * k);
  std::uniform_int_distribution<int> code(0, 255);
  for (auto& b : w) {
    int c = code(rng);
    if ((c & 0x7F) == 0x7F) c &= 0x7E;
    b = static_cast<uint8_t>(c);
  }
  std::vector<float> ws(static_cast<size_t>(scale_rows) * groups);
  for (auto& s : ws) s = uni(rng) * 0.01f;

  Dev<uint16_t> d_act(act.size());
  d_act.upload(act);
  Dev<uint8_t> d_q(static_cast<size_t>(m) * k);
  Dev<float> d_as(static_cast<size_t>(m) * groups);
  Dev<uint8_t> d_w(w.size());
  d_w.upload(w);
  Dev<float> d_ws(ws.size());
  d_ws.upload(ws);
  Dev<float> d_out(static_cast<size_t>(m) * n);
  Dev<uint16_t> d_out16(static_cast<size_t>(m) * n);
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  dgpp::launch_fp8_quantize_rows(d_act.p, stride, m, k, d_q.p, d_as.p, stream);
  dgpp::launch_fp8_gemm_f32(d_q.p, d_as.p, d_w.p, d_ws.p, sbr, d_out.p, m, n, k, stream);
  dgpp::launch_fp8_gemm_bf16(d_q.p, d_as.p, d_w.p, d_ws.p, sbr, d_out16.p, m, n, k, stream);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  const auto q = d_q.download();
  const auto as = d_as.download();
  const auto out = d_out.download();
  const auto out16 = d_out16.download();

  // 1. The quantizer: scale = amax / 448 (1 for an all-zero group), each
  //    code the host encoder's of x / scale.
  for (int r = 0; r < m; ++r)
    for (int g = 0; g < groups; ++g) {
      float amax = 0.f;
      for (int c = 0; c < 128; ++c) amax = std::max(amax, std::fabs(dgpp::bf16_bits_to_float(act[r * stride + g * 128 + c])));
      const float want = amax > 0.f ? amax / 448.f : 1.f;
      require(as[static_cast<size_t>(r) * groups + g] == want, "quantizer scale = amax / 448");
      for (int c = 0; c < 128; ++c) {
        const float x = dgpp::bf16_bits_to_float(act[r * stride + g * 128 + c]);
        const uint8_t got = q[static_cast<size_t>(r) * k + g * 128 + c];
        const uint8_t host = dgpp::float_to_fp8_e4m3_bits(x / want);
        require(got == host || (dgpp::fp8_e4m3_bits_to_float(got) == dgpp::fp8_e4m3_bits_to_float(host)),
                "quantizer code = e4m3(x / scale)");
      }
    }
  // 2. The GEMM on its own inputs against an exact reference (double), and
  //    the tolerance claim against the dequantized bf16 chain it replaces.
  std::vector<float> aq(static_cast<size_t>(m) * k), wq(static_cast<size_t>(n) * k), wdq(static_cast<size_t>(n) * k);
  for (size_t i = 0; i < aq.size(); ++i) aq[i] = dgpp::fp8_e4m3_bits_to_float(q[i]);
  for (int c = 0; c < n; ++c)
    for (int j = 0; j < k; ++j) {
      const float code_v = dgpp::fp8_e4m3_bits_to_float(w[static_cast<size_t>(c) * k + j]);
      wq[static_cast<size_t>(c) * k + j] = code_v;
      const float sc = ws[static_cast<size_t>(c / sbr) * groups + j / 128];
      wdq[static_cast<size_t>(c) * k + j] = dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(code_v * sc));
    }
  double err_exact = 0, err_chain = 0, den = 0, ref_sq = 0, max_abs = 0;
  size_t bf16_mismatch = 0;
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      double ref = 0, chain = 0;
      for (int g = 0; g < groups; ++g) {
        double dot = 0;
        for (int j = g * 128; j < (g + 1) * 128; ++j)
          dot += static_cast<double>(aq[static_cast<size_t>(r) * k + j]) * wq[static_cast<size_t>(c) * k + j];
        ref += static_cast<double>(as[static_cast<size_t>(r) * groups + g]) * ws[static_cast<size_t>(c / sbr) * groups + g] * dot;
      }
      for (int j = 0; j < k; ++j)
        chain += static_cast<double>(dgpp::bf16_bits_to_float(act[r * stride + j])) * wdq[static_cast<size_t>(c) * k + j];
      const double got = out[static_cast<size_t>(r) * n + c];
      require(std::isfinite(got), "finite output");
      err_exact += (got - ref) * (got - ref);
      err_chain += (got - chain) * (got - chain);
      den += chain * chain;
      ref_sq += ref * ref;
      max_abs = std::max(max_abs, std::fabs(got - ref));
      if (out16[static_cast<size_t>(r) * n + c] != dgpp::float_to_bf16_bits(out[static_cast<size_t>(r) * n + c])) ++bf16_mismatch;
    }
  }
  const double rel_exact = std::sqrt(err_exact / den), rel_chain = std::sqrt(err_chain / den);
  const double ref_rms = std::sqrt(ref_sq / (static_cast<double>(m) * n));
  std::printf("m=%d n=%d k=%d sbr=%d: vs exact %.2e (max %.2e of the output RMS), vs the bf16 chain %.3f, "
              "bf16 mismatches %zu\n",
              m, n, k, sbr, rel_exact, max_abs / ref_rms, rel_chain, bf16_mismatch);
  require(rel_exact < 1e-5, "the GEMM agrees with the exact sum of its own quantized inputs (fp32 accumulation)");
  require(max_abs < 1e-4 * ref_rms, "no element far from the exact sum (against the output's RMS)");
  require(bf16_mismatch == 0, "the bf16 output is the f32 output rounded once");
  // The e4m3 activation carries ~2^-4 relative error an element; the dot
  // product's relative RMS error stays near that (independent errors).
  require(rel_chain < 0.06, "within the quantized model's tolerance of the dequantized bf16 chain");
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("no CUDA device: skipped\n");
    return 2;
  }
  std::mt19937 rng(20260930);
  for (const Shape& sh : {Shape{129, 320, 128, 128}, Shape{300, 1000, 2560, 64}, Shape{512, 2560, 1280, 128},
                          Shape{5, 40, 256, 32}})
    run(sh, rng);
  std::printf("fp8_gemm_test: OK\n");
  return 0;
}
