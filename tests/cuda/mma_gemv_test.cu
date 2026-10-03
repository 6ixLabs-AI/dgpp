// The streaming tensor-core decode GEMM (kernels/mma_gemv.hpp): against a
// double oracle over the dequant bridge's weight values (the tile kernel's
// budget: fp32-order differences only), bitwise m-invariance (a row's
// result identical at m = 1 and inside an m = 30 batch, fp8 and bf16), the
// fp32 and bf16 epilogues agreeing bit for bit, and a cold timing line
// beside the GEMV cores it replaces (informational).
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/dense_mma_bf16w.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/scale_gemm.hpp"

namespace {
void require(bool c, const std::string& w) { if (!c) throw std::runtime_error(w); }

struct Problem {
  int m, n, k, rs, cs;
  std::vector<uint8_t> w8;      // [n, k] e4m3
  std::vector<uint16_t> w16;    // [n, k] bf16
  std::vector<float> scales;    // [n>>rs][k>>cs]
  std::vector<uint16_t> act;    // [m, k] bf16
};

Problem make(int m, int n, int k, int rs, int cs, uint64_t seed) {
  Problem p{m, n, k, rs, cs, {}, {}, {}, {}};
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> code(0, 255);
  std::uniform_real_distribution<float> uf(-1.f, 1.f);
  p.w8.resize(size_t(n) * k);
  for (auto& b : p.w8) { uint8_t c = code(rng); if ((c & 0x7F) == 0x7F) c &= 0x7E; b = c; }  // no NaN codes
  p.w16.resize(size_t(n) * k);
  for (auto& x : p.w16) x = dgpp::float_to_bf16_bits(uf(rng));
  const int sr = (n + (1 << rs) - 1) >> rs, sc = (k + (1 << cs) - 1) >> cs;
  p.scales.resize(size_t(sr) * sc);
  for (auto& s : p.scales) s = std::ldexp(1.f, -(rng() % 8));  // e8m0-like powers of two
  p.act.resize(size_t(m) * k);
  for (auto& x : p.act) x = dgpp::float_to_bf16_bits(uf(rng));
  return p;
}

// The dequant bridge's weight value: bf16(e4m3 x scale).
float wval8(const Problem& p, int r, int c) {
  const float s = p.scales[size_t(r >> p.rs) * ((p.k + (1 << p.cs) - 1) >> p.cs) + (c >> p.cs)];
  return dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(p.w8[size_t(r) * p.k + c]) * s));
}

template <typename F>
double max_rel_err(const Problem& p, const std::vector<float>& got, F wv) {
  double worst = 0;
  for (int i = 0; i < p.m; ++i)
    for (int j = 0; j < p.n; ++j) {
      double ref = 0, mag = 0;
      for (int c = 0; c < p.k; ++c) {
        const double a = dgpp::bf16_bits_to_float(p.act[size_t(i) * p.k + c]), w = wv(j, c);
        ref += a * w; mag += std::fabs(a * w);
      }
      const double err = std::fabs(got[size_t(i) * p.n + j] - ref) / (mag + 1e-6);
      if (err > worst) worst = err;
    }
  return worst;
}

struct Dev {
  uint8_t* w8 = nullptr; uint16_t* w16 = nullptr; float* scales = nullptr; uint16_t* act = nullptr;
  float* outf = nullptr; uint16_t* outb = nullptr;
  explicit Dev(const Problem& p) {
    DGPP_CUDA_OK(cudaMalloc(&w8, p.w8.size())); DGPP_CUDA_OK(cudaMemcpy(w8, p.w8.data(), p.w8.size(), cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&w16, p.w16.size() * 2)); DGPP_CUDA_OK(cudaMemcpy(w16, p.w16.data(), p.w16.size() * 2, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&scales, p.scales.size() * 4)); DGPP_CUDA_OK(cudaMemcpy(scales, p.scales.data(), p.scales.size() * 4, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&act, p.act.size() * 2)); DGPP_CUDA_OK(cudaMemcpy(act, p.act.data(), p.act.size() * 2, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&outf, size_t(std::max(32, p.m)) * p.n * 4)); DGPP_CUDA_OK(cudaMalloc(&outb, size_t(std::max(32, p.m)) * p.n * 2));
  }
  ~Dev() { cudaFree(w8); cudaFree(w16); cudaFree(scales); cudaFree(act); cudaFree(outf); cudaFree(outb); }
};
std::vector<float> fetch_f(const float* d, size_t n) { std::vector<float> h(n); DGPP_CUDA_OK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return h; }
std::vector<uint16_t> fetch_b(const uint16_t* d, size_t n) { std::vector<uint16_t> h(n); DGPP_CUDA_OK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost)); return h; }
}  // namespace

DGPP_TEST(mma_gemv_fp8_matches_oracle_and_is_m_invariant) {
  const Problem p = make(30, 1024, 512, 5, 5, 0x9E3779B97F4A7C15ull);
  Dev d(p);
  // fp32 epilogue at m = 30 vs the oracle.
  dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, p.m, p.n, p.k, 0, p.rs, p.cs, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> full = fetch_f(d.outf, size_t(p.m) * p.n);
  const double err = max_rel_err(p, full, [&](int r, int c) { return wval8(p, r, c); });
  std::printf("[ .. ] mma_gemv fp8 m=30 n=%d k=%d: max rel err vs the double oracle %.3e\n", p.n, p.k, err);
  require(err < 2e-3, "fp8: outside the tile kernel's oracle budget");
  // The bf16 epilogue is the fp32 one rounded.
  dgpp::launch_mma_gemv_fp8_bf16(d.act, p.k, d.w8, d.scales, d.outb, p.m, p.n, p.k, 0, p.rs, p.cs, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> fullb = fetch_b(d.outb, size_t(p.m) * p.n);
  for (size_t i = 0; i < fullb.size(); ++i)
    require(fullb[i] == dgpp::float_to_bf16_bits(full[i]), "fp8: bf16 epilogue != bf16(f32 epilogue)");
  // m-invariance: every row alone (m = 1), and rows in m = 7 and m = 16
  // batches, bitwise the m = 30 result.
  for (int m2 : {1, 7, 16}) {
    for (int r0 = 0; r0 + m2 <= p.m; r0 += (m2 == 1 ? 7 : m2)) {
      dgpp::launch_mma_gemv_fp8_f32(d.act + size_t(r0) * p.k, p.k, d.w8, d.scales, d.outf, m2, p.n, p.k, 0, p.rs, p.cs, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> part = fetch_f(d.outf, size_t(m2) * p.n);
      for (int i = 0; i < m2; ++i)
        require(std::memcmp(part.data() + size_t(i) * p.n, full.data() + size_t(r0 + i) * p.n, size_t(p.n) * 4) == 0,
                "fp8: row " + std::to_string(r0 + i) + " differs at m=" + std::to_string(m2) + " from m=30");
    }
  }
  std::printf("[ OK ] mma_gemv fp8: bitwise m-invariant (m = 1, 7, 16 vs 30), bf16 == bf16(f32)\n");
}

// The wide forms (4 and 8 tiles) and the 128-row grouping above them: the
// oracle at 200 rows, and every row bitwise its m = 1 result (the forms
// share one chain).
DGPP_TEST(mma_gemv_fp8_wide_forms_match_oracle_and_the_scalar_form) {
  const Problem p = make(200, 512, 1024, 5, 5, 0x2545F4914F6CDD1Dull);
  Dev d(p);
  dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, p.m, p.n, p.k, 0, p.rs, p.cs, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> full = fetch_f(d.outf, size_t(p.m) * p.n);
  const double err = max_rel_err(p, full, [&](int r, int c) { return wval8(p, r, c); });
  std::printf("[ .. ] mma_gemv fp8 m=200 (128 + 72 rows) n=%d k=%d: max rel err %.3e\n", p.n, p.k, err);
  require(err < 2e-3, "fp8 wide: outside the oracle budget");
  for (int m2 : {1, 40, 64, 100}) {
    for (int r0 = 0; r0 + m2 <= p.m; r0 += (m2 == 1 ? 37 : m2)) {
      dgpp::launch_mma_gemv_fp8_f32(d.act + size_t(r0) * p.k, p.k, d.w8, d.scales, d.outf, m2, p.n, p.k, 0, p.rs, p.cs, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> part = fetch_f(d.outf, size_t(m2) * p.n);
      for (int i = 0; i < m2; ++i)
        require(std::memcmp(part.data() + size_t(i) * p.n, full.data() + size_t(r0 + i) * p.n, size_t(p.n) * 4) == 0,
                "fp8 wide: row " + std::to_string(r0 + i) + " differs at m=" + std::to_string(m2) + " from m=200");
    }
  }
  std::printf("[ OK ] mma_gemv fp8: the 1/2/4/8-tile forms and the grouped launch are bitwise one chain\n");
  // The fixture-sized shapes (k under one 512-k window, small n) at every form.
  for (const auto [n2, k2] : {std::pair{64, 128}, std::pair{128, 256}, std::pair{256, 64}, std::pair{320, 448}}) {
    for (int m2 : {1, 24, 40, 70, 100, 130}) {
      const Problem q = make(m2, n2, k2, 5, 5, 0xA0761D6478BD642Full + m2 + n2);
      Dev e(q);
      dgpp::launch_mma_gemv_fp8_f32(e.act, q.k, e.w8, e.scales, e.outf, q.m, q.n, q.k, 0, q.rs, q.cs, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> got = fetch_f(e.outf, size_t(q.m) * q.n);
      const double e2 = max_rel_err(q, got, [&](int r, int c) { return wval8(q, r, c); });
      require(e2 < 2e-3, "fp8 shape n=" + std::to_string(n2) + " k=" + std::to_string(k2) + " m=" + std::to_string(m2) +
                             ": max rel err " + std::to_string(e2));
    }
  }
  std::printf("[ OK ] mma_gemv fp8: fixture-sized shapes at every form match the oracle\n");
}

DGPP_TEST(mma_gemv_bf16_matches_oracle_and_is_m_invariant) {
  const Problem p = make(30, 1024, 512, 7, 7, 0xD1B54A32D192ED03ull);
  Dev d(p);
  dgpp::launch_mma_gemv_bf16_f32(d.act, p.k, d.w16, d.outf, p.m, p.n, p.k, 0, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> full = fetch_f(d.outf, size_t(p.m) * p.n);
  const double err = max_rel_err(p, full, [&](int r, int c) { return dgpp::bf16_bits_to_float(p.w16[size_t(r) * p.k + c]); });
  std::printf("[ .. ] mma_gemv bf16 m=30: max rel err vs the double oracle %.3e\n", err);
  require(err < 2e-3, "bf16: outside the oracle budget");
  for (int m2 : {1, 16}) {
    for (int r0 = 0; r0 + m2 <= p.m; r0 += (m2 == 1 ? 11 : m2)) {
      dgpp::launch_mma_gemv_bf16_f32(d.act + size_t(r0) * p.k, p.k, d.w16, d.outf, m2, p.n, p.k, 0, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> part = fetch_f(d.outf, size_t(m2) * p.n);
      for (int i = 0; i < m2; ++i)
        require(std::memcmp(part.data() + size_t(i) * p.n, full.data() + size_t(r0 + i) * p.n, size_t(p.n) * 4) == 0,
                "bf16: row differs across m");
    }
  }
  std::printf("[ OK ] mma_gemv bf16: bitwise m-invariant\n");
}

// The dense tensor-core GEMM over bf16 weights (kernels/dense_mma_bf16w.hpp,
// the DeepSeek-V4-Flash prefill's compressor projections): inside the
// double oracle's budget on a ragged shape (m and n off the tile sizes),
// bitwise the same row whatever rows share the launch — a chunked prefill's
// contract — and the bf16 epilogue the rounded fp32 one.
DGPP_TEST(dense_mma_bf16w_matches_oracle_and_is_row_invariant) {
  const Problem p = make(150, 344, 512, 7, 7, 0x9E3779B97F4A7C15ull);
  Dev d(p);
  require(dgpp::dense_mma_bf16w_shape_ok(d.w16, p.k), "the shape the kernel takes");
  dgpp::launch_dense_mma_bf16w_f32(d.act, p.k, d.w16, d.outf, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> full = fetch_f(d.outf, size_t(p.m) * p.n);
  const double err = max_rel_err(p, full, [&](int r, int c) { return dgpp::bf16_bits_to_float(p.w16[size_t(r) * p.k + c]); });
  std::printf("[ .. ] dense_mma_bf16w [150 x 344 x 512]: max rel err vs the double oracle %.3e\n", err);
  require(err < 2e-3, "dense bf16w: outside the oracle budget");
  // Any chunking of the rows gives the same rows.
  for (int chunk : {1, 37, 64, 101}) {
    for (int r0 = 0; r0 < p.m; r0 += (chunk == 1 ? 29 : chunk)) {
      const int rows = std::min(chunk, p.m - r0);
      dgpp::launch_dense_mma_bf16w_f32(d.act + size_t(r0) * p.k, p.k, d.w16, d.outf, rows, p.n, p.k, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> part = fetch_f(d.outf, size_t(rows) * p.n);
      for (int i = 0; i < rows; ++i)
        require(std::memcmp(part.data() + size_t(i) * p.n, full.data() + size_t(r0 + i) * p.n, size_t(p.n) * 4) == 0,
                "dense bf16w: a row differs across chunkings");
    }
  }
  dgpp::launch_dense_mma_bf16w_bf16(d.act, p.k, d.w16, d.outb, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> fb = fetch_b(d.outb, size_t(p.m) * p.n);
  for (size_t i = 0; i < fb.size(); ++i)
    require(fb[i] == dgpp::float_to_bf16_bits(full[i]), "dense bf16w: the bf16 epilogue is the rounded fp32 one");
  std::printf("[ OK ] dense_mma_bf16w: row-invariant across chunkings, epilogues agree\n");
}

// The swiglu form: bitwise the clamp launch followed by the plain fp8 GEMV.
DGPP_TEST(mma_gemv_fp8_swiglu_form_is_bitwise_the_clamp_then_the_gemv) {
  const Problem p = make(6, 1024, 512, 7, 7, 0x7F4A7C159E3779B9ull);
  Dev d(p);
  // gate = the problem's activations, up = a second set (the weights' first rows reused as bf16 values).
  uint16_t* up; uint16_t* act;
  DGPP_CUDA_OK(cudaMalloc(&up, size_t(p.m) * p.k * 2)); DGPP_CUDA_OK(cudaMalloc(&act, size_t(p.m) * p.k * 2));
  DGPP_CUDA_OK(cudaMemcpy(up, p.w16.data(), size_t(p.m) * p.k * 2, cudaMemcpyHostToDevice));
  for (const float limit : {10.0f, 0.25f}) {
    for (int m : {1, 4, 6}) {
      dgpp::launch_moe_swiglu_clamp(d.act, up, act, int64_t(m) * p.k, limit, nullptr);
      dgpp::launch_mma_gemv_fp8_f32(act, p.k, d.w8, d.scales, d.outf, m, p.n, p.k, 0, p.rs, p.cs, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> want = fetch_f(d.outf, size_t(m) * p.n);
      DGPP_CUDA_OK(cudaMemset(d.outf, 0xA5, size_t(m) * p.n * 4));
      dgpp::launch_mma_gemv_fp8_swiglu_f32(d.act, up, limit, p.k, d.w8, d.scales, d.outf, m, p.n, p.k, 0, p.rs, p.cs, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> got = fetch_f(d.outf, size_t(m) * p.n);
      require(std::memcmp(want.data(), got.data(), want.size() * 4) == 0, "swiglu form: differs from the clamp launch + the GEMV");
    }
  }
  cudaFree(up); cudaFree(act);
  std::printf("[ OK ] mma_gemv swiglu form: bitwise the clamp launch then the GEMV (m = 1, 4, 6; two limits)\n");
}

// The gated ring form (the lazy ratio-128 compressors): a row that
// completes a group gets the group's rows off its request's ring, bitwise
// the plain launch on the same rows; every other row's output is untouched.
DGPP_TEST(dense_mma_bf16w_groups_are_the_plain_rows_and_gated) {
  const int requests = 3, slots = 104, group = 96, n = 344, k = 512;
  const Problem p = make(requests * slots, n, k, 7, 7, 0xC2B2AE3D27D4EB4Full);  // act: the ring [requests][slots][k]
  Dev d(p);
  struct Row { int req; int64_t pos; };
  // (req 0, 95): the first group, slots 0..95.  (req 1, 191): the second,
  // positions 96..191 = slots 96..103, 0..87 (the wrap).  The rest: no group.
  const std::vector<Row> rows = {{0, 95}, {0, 96}, {1, 191}, {2, -1}, {2, 94}};
  const int R = int(rows.size());
  std::vector<int32_t> req(R); std::vector<int64_t> pos(R);
  for (int i = 0; i < R; ++i) { req[size_t(i)] = rows[size_t(i)].req; pos[size_t(i)] = rows[size_t(i)].pos; }
  int32_t* dreq; int64_t* dpos; float* dout; uint16_t* dgather; float* dref;
  DGPP_CUDA_OK(cudaMalloc(&dreq, size_t(R) * 4)); DGPP_CUDA_OK(cudaMalloc(&dpos, size_t(R) * 8));
  DGPP_CUDA_OK(cudaMalloc(&dout, size_t(R) * group * n * 4)); DGPP_CUDA_OK(cudaMalloc(&dgather, size_t(group) * k * 2));
  DGPP_CUDA_OK(cudaMalloc(&dref, size_t(group) * n * 4));
  DGPP_CUDA_OK(cudaMemcpy(dreq, req.data(), size_t(R) * 4, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(dpos, pos.data(), size_t(R) * 8, cudaMemcpyHostToDevice));
  std::vector<float> sentinel(size_t(R) * group * n, -777.25f);
  DGPP_CUDA_OK(cudaMemcpy(dout, sentinel.data(), sentinel.size() * 4, cudaMemcpyHostToDevice));
  dgpp::launch_dense_mma_bf16w_groups_f32(d.act, slots, dreq, dpos, R, group, d.w16, dout, n, k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> got = fetch_f(dout, size_t(R) * group * n);
  int completed = 0;
  for (int i = 0; i < R; ++i) {
    const float* mine = got.data() + size_t(i) * group * n;
    if (pos[size_t(i)] < 0 || (pos[size_t(i)] + 1) % group != 0) {
      for (size_t e = 0; e < size_t(group) * n; ++e) require(mine[e] == -777.25f, "groups: a row without a group wrote");
      continue;
    }
    ++completed;
    std::vector<uint16_t> gathered(size_t(group) * k);
    for (int t = 0; t < group; ++t) {
      const int64_t slot = (pos[size_t(i)] - group + 1 + t) % slots;
      std::memcpy(gathered.data() + size_t(t) * k, p.act.data() + (size_t(req[size_t(i)]) * slots + size_t(slot)) * k, size_t(k) * 2);
    }
    DGPP_CUDA_OK(cudaMemcpy(dgather, gathered.data(), gathered.size() * 2, cudaMemcpyHostToDevice));
    dgpp::launch_dense_mma_bf16w_f32(dgather, k, d.w16, dref, group, n, k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> ref = fetch_f(dref, size_t(group) * n);
    require(std::memcmp(mine, ref.data(), ref.size() * 4) == 0, "groups: a group differs from the plain launch on its rows");
  }
  require(completed == 2, "groups: the test's two completing rows");
  cudaFree(dreq); cudaFree(dpos); cudaFree(dout); cudaFree(dgather); cudaFree(dref);
  std::printf("[ OK ] dense_mma_bf16w groups: bitwise the plain rows (a wrapped ring included), gated rows untouched\n");
}

DGPP_TEST(mma_gemv_cold_timing_beside_the_gemv_cores) {
  // Informational: the dense projection shape [5120 x 5120] fp8 (32 x 32
  // scales) and the head shape [32320 x 5120] bf16, cold (weights cycled
  // past the L2), at m = 1, 6, 16, 30.
  const int n = 5120, k = 5120, copies = 24;
  std::vector<uint8_t*> w(copies); std::vector<float*> s(copies);
  const size_t sb = size_t(n / 32) * (k / 32) * 4;
  std::vector<float> hs(sb / 4, 0.5f);
  for (int c = 0; c < copies; ++c) { DGPP_CUDA_OK(cudaMalloc(&w[c], size_t(n) * k)); DGPP_CUDA_OK(cudaMemset(w[c], 0x38, size_t(n) * k)); DGPP_CUDA_OK(cudaMalloc(&s[c], sb)); DGPP_CUDA_OK(cudaMemcpy(s[c], hs.data(), sb, cudaMemcpyHostToDevice)); }
  uint16_t* act; DGPP_CUDA_OK(cudaMalloc(&act, size_t(2048) * k * 2)); DGPP_CUDA_OK(cudaMemset(act, 0x3f, size_t(2048) * k * 2));
  uint16_t* out; DGPP_CUDA_OK(cudaMalloc(&out, size_t(2048) * n * 2));
  cudaEvent_t e0, e1; DGPP_CUDA_OK(cudaEventCreate(&e0)); DGPP_CUDA_OK(cudaEventCreate(&e1));
  std::printf("[ .. ] fp8 [%d x %d] cold, us per product: m  gemv-chunks  mma-stream\n", n, k);
  for (int m : {1, 2, 3, 4, 6, 16, 30}) {
    float t[2];
    for (int path = 0; path < 2; ++path) {
      for (int i = 0; i < 4; ++i) { if (path == 0) dgpp::launch_scale_gemm_grid_bf16(act, k, w[i], s[i], out, m, n, k, nullptr, 0, 5, 5); else dgpp::launch_mma_gemv_fp8_bf16(act, k, w[i], s[i], out, m, n, k, 0, 5, 5, nullptr); }
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaEventRecord(e0));
      for (int i = 0; i < 48; ++i) { const int c = i % copies; if (path == 0) dgpp::launch_scale_gemm_grid_bf16(act, k, w[c], s[c], out, m, n, k, nullptr, 0, 5, 5); else dgpp::launch_mma_gemv_fp8_bf16(act, k, w[c], s[c], out, m, n, k, 0, 5, 5, nullptr); }
      DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1)); t[path] = ms * 1000.f / 48;
    }
    std::printf("[ .. ]   m=%2d  %8.1f us  %8.1f us  (%.0f GB/s of weights on the mma stream)\n", m, t[0], t[1], double(n) * k / t[1] / 1e3);
  }
  // The prefill's rows: the chunk kernels to 128 rows, the tile kernel above.
  std::printf("[ .. ] fp8 [%d x %d] cold, us per product: m  chunks/tile  mma-forms\n", n, k);
  // The full GLM-5.3 DSA fp8 shapes at one rank of four (kv_a, q_a, q_b, o_proj):
  // the streaming form's block count is n / 64 (a small-n site fills few SMs).
  for (auto [sn, sk] : std::vector<std::pair<int, int>>{{576, 6144}, {2048, 6144}, {4096, 2048}, {5120, 5120}, {6144, 4096}, {12288, 5120}, {32320, 5120}}) {
    uint8_t* sw[4]; float* ss[4];
    const int sc = ((sn + 127) / 128) * ((sk + 127) / 128);
    for (int i = 0; i < 4; ++i) {
      DGPP_CUDA_OK(cudaMalloc(&sw[i], static_cast<size_t>(sn) * sk)); DGPP_CUDA_OK(cudaMemset(sw[i], 0x38, static_cast<size_t>(sn) * sk));
      DGPP_CUDA_OK(cudaMalloc(&ss[i], static_cast<size_t>(sc) * 4)); DGPP_CUDA_OK(cudaMemset(ss[i], 0, static_cast<size_t>(sc) * 4));
    }
    for (int m : {1, 16}) {
      float t[6];
      for (int path = 0; path < 6; ++path) {
        // path 0: the chunks; 1..5: the mma at widths 8, 4, 2, 1 warps and the rule.
        dgpp::mma_gemv_set_decode_width(path == 1 ? 8 : path == 2 ? 4 : path == 3 ? 2 : path == 4 ? 1 : 0);
        auto run = [&](int i) { if (path == 0) dgpp::launch_scale_gemm_bf16(act, sk, sw[i], ss[i], out, m, sn, sk, nullptr, 0); else dgpp::launch_mma_gemv_fp8_bf16(act, sk, sw[i], ss[i], out, m, sn, sk, 0, 7, 7, nullptr); };
        for (int i = 0; i < 4; ++i) run(i);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        DGPP_CUDA_OK(cudaEventRecord(e0));
        for (int i = 0; i < 16; ++i) run(i % 4);
        DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
        float ms = 0; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1)); t[path] = ms * 1000.f / 16;
      }
      dgpp::mma_gemv_set_decode_width(0);
      std::printf("[ .. ]   fp8 [%d x %d] m=%3d  chunks %8.1f us  mma w8 %8.1f  w4 %8.1f  w2 %8.1f  w1 %8.1f  rule %8.1f\n", sn, sk, m, t[0], t[1], t[2], t[3], t[4], t[5]);
    }
    for (int i = 0; i < 4; ++i) { cudaFree(sw[i]); cudaFree(ss[i]); }
  }
  // The non-grid scale GEMM (the session-core families' dense sites): the
  // GEMV chunk to 128 rows, the 128-row dense tensor-core kernel above.
  for (int m : {30, 64, 128, 256, 512, 2048}) {
    float t[2];
    for (int path = 0; path < 2; ++path) {
      for (int i = 0; i < 4; ++i) { if (path == 0) dgpp::launch_scale_gemm_bf16(act, k, w[i], s[i], out, m, n, k, nullptr, 0); else dgpp::launch_mma_gemv_fp8_bf16(act, k, w[i], s[i], out, m, n, k, 0, 7, 7, nullptr); }
      DGPP_CUDA_OK(cudaEventRecord(e0)); 
      for (int i = 0; i < 4; ++i) { if (path == 0) dgpp::launch_scale_gemm_bf16(act, k, w[i], s[i], out, m, n, k, nullptr, 0); else dgpp::launch_mma_gemv_fp8_bf16(act, k, w[i], s[i], out, m, n, k, 0, 7, 7, nullptr); }
      DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
      DGPP_CUDA_OK(cudaEventElapsedTime(&t[path], e0, e1)); t[path] = t[path] * 1000.f / 4;
    }
    std::printf("[ .. ]   non-grid scale GEMM vs mma  m=%4d  %9.1f us  %9.1f us\n", m, t[0], t[1]);
  }
  for (int m : {64, 128, 256, 2048}) {
    float t[2];
    const int reps = m > 256 ? 6 : 24;
    for (int path = 0; path < 2; ++path) {
      for (int i = 0; i < 2; ++i) { if (path == 0) dgpp::launch_scale_gemm_grid_bf16(act, k, w[i], s[i], out, m, n, k, nullptr, 0, 5, 5); else dgpp::launch_mma_gemv_fp8_bf16(act, k, w[i], s[i], out, m, n, k, 0, 5, 5, nullptr); }
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaEventRecord(e0));
      for (int i = 0; i < reps; ++i) { const int c = i % copies; if (path == 0) dgpp::launch_scale_gemm_grid_bf16(act, k, w[c], s[c], out, m, n, k, nullptr, 0, 5, 5); else dgpp::launch_mma_gemv_fp8_bf16(act, k, w[c], s[c], out, m, n, k, 0, 5, 5, nullptr); }
      DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1)); t[path] = ms * 1000.f / reps;
    }
    std::printf("[ .. ]   m=%4d  %8.1f us  %8.1f us  (%.1f TFLOP/s on the mma forms)\n", m, t[0], t[1], 2.0 * m * n * k / t[1] / 1e6);
  }
  for (int c = 0; c < copies; ++c) { cudaFree(w[c]); cudaFree(s[c]); }
  // The head.
  const int hn = 32320; const int hc = 3;
  std::vector<uint16_t*> hw(hc); for (int c = 0; c < hc; ++c) { DGPP_CUDA_OK(cudaMalloc(&hw[c], size_t(hn) * k * 2)); DGPP_CUDA_OK(cudaMemset(hw[c], 0x3c, size_t(hn) * k * 2)); }
  float* outf; DGPP_CUDA_OK(cudaMalloc(&outf, size_t(32) * hn * 4));
  std::printf("[ .. ] bf16 head [%d x %d] cold, us per full product: m  gemv-chunks  mma-stream\n", hn, k);
  for (int m : {1, 6, 16, 30}) {
    float t[2];
    for (int path = 0; path < 2; ++path) {
      DGPP_CUDA_OK(cudaEventRecord(e0));
      for (int i = 0; i < 9; ++i) { const int c = i % hc;
        if (path == 0) { for (int r0 = 0; r0 < m; r0 += 4) { const int rows = std::min(4, m - r0); dgpp::launch_bf16_gemv(act + size_t(r0) * k, k, hw[c], outf + size_t(r0) * hn, true, rows, hn, k, nullptr); } }
        else dgpp::launch_mma_gemv_bf16_f32(act, k, hw[c], outf, m, hn, k, 0, nullptr); }
      DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
      float ms = 0; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1)); t[path] = ms * 1000.f / 9;
    }
    std::printf("[ .. ]   m=%2d  %8.1f us  %8.1f us\n", m, t[0], t[1]);
  }
  for (int c = 0; c < hc; ++c) cudaFree(hw[c]);
  cudaFree(outf); cudaFree(act); cudaFree(out);
}

// Split-K at the small-n sites (2026-09-21): with a workspace the decode
// forms split the k range across blocks. Against the oracle, deterministic
// across launches, the bf16 epilogue bf16(f32), a row's result the same at
// m = 16 and inside m = 30 (the split is a function of the shape), and a
// cold timing line beside the unsplit form.
DGPP_TEST(mma_gemv_split_k_small_n_sites) {
  void* ws = nullptr; const size_t ws_bytes = 4u << 20;
  DGPP_CUDA_OK(cudaMalloc(&ws, ws_bytes));
  for (auto [n, k] : std::vector<std::pair<int, int>>{{320, 10240}, {320, 2560}, {256, 2560}, {640, 2560}}) {
    const Problem p = make(30, n, k, 7, 7, 0xA5A5A5A5ull + n * 131 + k);
    Dev d(p);
    dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, 30, p.n, p.k, 0, p.rs, p.cs, nullptr, ws, ws_bytes);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> a = fetch_f(d.outf, size_t(30) * p.n);
    const double err = max_rel_err(p, a, [&](int r, int c) { return wval8(p, r, c); });
    require(err < 2e-3, "split-K fp8 [" + std::to_string(n) + " x " + std::to_string(k) + "] vs oracle: " + std::to_string(err));
    dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, 30, p.n, p.k, 0, p.rs, p.cs, nullptr, ws, ws_bytes);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> b = fetch_f(d.outf, size_t(30) * p.n);
    require(std::memcmp(a.data(), b.data(), a.size() * 4) == 0, "split-K: two launches differ");
    dgpp::launch_mma_gemv_fp8_f32(d.act, p.k, d.w8, d.scales, d.outf, 16, p.n, p.k, 0, p.rs, p.cs, nullptr, ws, ws_bytes);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> m16 = fetch_f(d.outf, size_t(16) * p.n);
    require(std::memcmp(a.data(), m16.data(), m16.size() * 4) == 0, "split-K: rows differ between m = 16 and m = 30");
    dgpp::launch_mma_gemv_fp8_bf16(d.act, p.k, d.w8, d.scales, d.outb, 30, p.n, p.k, 0, p.rs, p.cs, nullptr, ws, ws_bytes);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<uint16_t> ob = fetch_b(d.outb, size_t(30) * p.n);
    for (size_t i = 0; i < ob.size(); ++i)
      require(ob[i] == dgpp::float_to_bf16_bits(a[i]), "split-K: bf16 epilogue != bf16(f32)");
    std::printf("[ OK ] mma_gemv split-K fp8 [%d x %d]: oracle %.3e, deterministic, m-invariant, bf16 == bf16(f32)\n", n, k, err);
  }
  // Cold timing (weights cycled past the 24 MB L2): unsplit vs split, m = 16 and 30.
  cudaEvent_t e0, e1; DGPP_CUDA_OK(cudaEventCreate(&e0)); DGPP_CUDA_OK(cudaEventCreate(&e1));
  uint16_t* act; DGPP_CUDA_OK(cudaMalloc(&act, size_t(32) * 10240 * 2)); DGPP_CUDA_OK(cudaMemset(act, 0x3f, size_t(32) * 10240 * 2));
  uint16_t* out; DGPP_CUDA_OK(cudaMalloc(&out, size_t(32) * 640 * 2));
  for (auto [n, k] : std::vector<std::pair<int, int>>{{320, 10240}, {320, 2560}, {256, 2560}, {640, 2560}}) {
    const size_t bytes = size_t(n) * k;
    const int copies = std::min<size_t>(64, (48u << 20) / bytes + 1);
    std::vector<uint8_t*> w(copies); std::vector<float*> sc(copies);
    const size_t sb = size_t((n + 127) / 128) * ((k + 127) / 128) * 4;
    for (int c = 0; c < copies; ++c) {
      DGPP_CUDA_OK(cudaMalloc(&w[c], bytes)); DGPP_CUDA_OK(cudaMemset(w[c], 0x38, bytes));
      DGPP_CUDA_OK(cudaMalloc(&sc[c], sb)); DGPP_CUDA_OK(cudaMemset(sc[c], 0, sb));
    }
    for (int m : {16, 30}) {
      float t[2];
      for (int path = 0; path < 2; ++path) {
        auto run = [&](int i) {
          dgpp::launch_mma_gemv_fp8_bf16(act, k, w[i], sc[i], out, m, n, k, 0, 7, 7, nullptr,
                                         path ? ws : nullptr, path ? ws_bytes : 0);
        };
        for (int i = 0; i < copies; ++i) run(i);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        const int reps = copies * 3;
        DGPP_CUDA_OK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i) run(i % copies);
        DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
        float ms = 0; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1)); t[path] = ms * 1000.f / reps;
      }
      std::printf("[ .. ]   fp8 [%d x %d] m=%2d cold: unsplit %7.1f us (%.0f GB/s)  split-K %7.1f us (%.0f GB/s)\n",
                  n, k, m, t[0], bytes / t[0] / 1e3, t[1], bytes / t[1] / 1e3);
    }
    for (int c = 0; c < copies; ++c) { cudaFree(w[c]); cudaFree(sc[c]); }
  }
  cudaFree(act); cudaFree(out); cudaFree(ws);
}

int main() { return dgpp::test::run_all(); }
