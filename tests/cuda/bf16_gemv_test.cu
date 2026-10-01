// Parity tests for the bf16 decode GEMV (M6 Stage 2 round 3): the GEMM
// interface's m<=8 decode path (chunked above four rows), against an fp64 oracle
// over the same bf16
// operands. Exercises both outputs (bf16, f32), strided activation views
// (the KDA f_a/g_a K-column slices), ragged n, short and long k, and the
// row-independence property (row r's bits at m=1 == its bits in m=3/m=8).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/bf12_companions.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/gemm.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;

struct Problem {
  int m, n, k;
  size_t act_stride;             // elements between activation rows
  std::vector<uint16_t> act;     // [m, act_stride] (k columns read)
  std::vector<uint16_t> weight;  // [n, k]
};

Problem make_problem(int m, int n, int k, size_t act_stride, uint64_t seed) {
  Problem p{m, n, k, act_stride, {}, {}};
  Rng rng(seed);
  p.act.resize(static_cast<size_t>(m) * act_stride);
  p.weight.resize(static_cast<size_t>(n) * k);
  fill_act(rng, p.act);
  fill_act(rng, p.weight);
  return p;
}

// fp64 dot over the bf16 operands: the strict oracle (fp64 accumulation).
std::vector<double> oracle(const Problem& p) {
  std::vector<double> out(static_cast<size_t>(p.m) * p.n);
  for (int r = 0; r < p.m; ++r)
    for (int c = 0; c < p.n; ++c) {
      double acc = 0.0;
      for (int kk = 0; kk < p.k; ++kk)
        acc += static_cast<double>(bf16_to_float(p.act[r * p.act_stride + kk])) *
               static_cast<double>(bf16_to_float(p.weight[static_cast<size_t>(c) * p.k + kk]));
      out[static_cast<size_t>(r) * p.n + c] = acc;
    }
  return out;
}

struct Device {
  uint16_t* act = nullptr;
  uint16_t* w = nullptr;
  void* out = nullptr;
  explicit Device(const Problem& p) {
    DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&w, p.weight.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 4));
    std::memcpy(act, p.act.data(), p.act.size() * 2);
    std::memcpy(w, p.weight.data(), p.weight.size() * 2);
  }
  ~Device() {
    cudaFree(act);
    cudaFree(w);
    cudaFree(out);
  }
};

// Runs through the SEAM (the production entry) and returns bf16 bits.
std::vector<uint16_t> run_bf16(dgpp::IGemm& gemm, const Problem& p,
                               const Device& d) {
  gemm.matmul(d.act, d.w, d.out, p.m, p.n, p.k, dgpp::DType::BF16,
              dgpp::GemmOut::BF16, p.act_stride, nullptr, 0, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), d.out, got.size() * 2);
  return got;
}

std::vector<float> run_f32(dgpp::IGemm& gemm, const Problem& p,
                           const Device& d) {
  gemm.matmul(d.act, d.w, d.out, p.m, p.n, p.k, dgpp::DType::BF16,
              dgpp::GemmOut::F32, p.act_stride, nullptr, 0, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), d.out, got.size() * 4);
  return got;
}

void check_bf16(const Problem& p, const std::vector<uint16_t>& got,
                const char* label) {
  const auto want = oracle(p);
  // 2 bf16 ULP against the fp64 oracle: the fp32 chain's error is far
  // inside one ULP; the budget covers double-rounding boundary cases.
  const auto rep = compare_bf16_vs_oracle(got.data(), want, 2.0, 1e-2);
  if (rep.mismatches != 0)
    throw std::runtime_error(std::string(label) + ": bf16 out off the oracle");
  std::printf("[ OK ] %s bf16: l2_rel=%.3g mismatches=%ld/%ld\n", label,
              rep.l2_rel, rep.mismatches, rep.total);
}

void check_f32(const Problem& p, const std::vector<float>& got,
               const char* label) {
  const auto want = oracle(p);
  double max_abs = 0.0;
  for (double v : want) max_abs = std::max(max_abs, std::abs(v));
  for (size_t i = 0; i < want.size(); ++i) {
    // fp32 chain over k products: k * 2^-24 relative to the magnitude scale.
    const double tol = 1e-6 * max_abs + 4e-6 * std::abs(want[i]) * std::sqrt(p.k);
    if (std::abs(got[i] - want[i]) > tol)
      throw std::runtime_error(std::string(label) + ": f32 out off the oracle");
  }
  std::printf("[ OK ] %s f32 within the fp32-chain budget\n", label);
}

}  // namespace

// Compare both implementations with exact operands, rather than making the
// existing cuBLAS output the definition of accuracy. The output sentinels
// also catch accidental writes of a full prefill into the compact buffer.
DGPP_TEST(compact_bf16_head_matches_fp64) {
  for (int k : {2560, 4096, 5120, 6144, 7168, 8192}) {
    for (int m : {5, 129, 2048}) {
      const Problem p = make_problem(m, 257, k, k, 0x4300 + k + m);
      Device d(p);
      Problem tail{1, p.n, k, size_t(k), {}, p.weight};
      tail.act.assign(p.act.end() - k, p.act.end());
      const auto want = oracle(tail);
      for (int mode = 0; mode < 3; ++mode) {
        dgpp::CublasLtGemm gemm;
        gemm.set_decode_rows(4);
        gemm.set_decode_mma(mode == 1);
        dgpp::Bf12Companions packed;
        if (mode == 2) require(packed.pack(d.w, p.n, k, gemm, nullptr), "BF12 oracle coverage");
        // Wide prefill uses cuBLASLt; compact uses the family's lowering.
        const auto full = run_f32(gemm, p, d);
        auto* out = static_cast<float*>(d.out);
        std::fill(out, out + p.n + 2, -12345.f);
        gemm.matmul(d.act + size_t(m - 1) * k, d.w, out + 1, 1, p.n, k, dgpp::DType::BF16,
                    dgpp::GemmOut::F32, k, nullptr, 0, nullptr);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        require(out[0] == -12345.f && out[p.n + 1] == -12345.f, "compact head output bounds");
        double norm = 0, ef = 0, ec = 0;
        for (int v = 0; v < p.n; ++v) {
          norm += want[v] * want[v];
          ef += std::pow(full[size_t(m - 1) * p.n + v] - want[v], 2);
          ec += std::pow(out[1 + v] - want[v], 2);
          require(std::isfinite(out[1 + v]), "finite compact head output");
        }
        const double lf = std::sqrt(ef / norm), lc = std::sqrt(ec / norm);
        // A long FP32 dot has a length-dependent accumulation error. Keep
        // both below 2e-5 and prohibit degradation beyond 1e-6 absolute
        // relative L2 when changing the selected-row implementation.
        require(lf < 2e-5 && lc < 2e-5 && lc <= lf + 1e-6, "compact/full head FP64 relative L2");
        std::printf("[ .. ] head k=%d m=%d mode=%d FP64 L2 full=%.3g compact=%.3g\n", k, m, mode,
                    lf, lc);
      }
    }
  }
}

DGPP_TEST(bf16_gemv_real_shapes_match_oracle) {
  dgpp::CublasLtGemm gemm;
  // The KDA in_proj shape class (k = hidden), the o_proj class (k = local
  // proj), the DSA indexer wk (n = 128), the lm head slice at m=1.
  for (auto [m, n, k] : std::vector<std::tuple<int, int, int>>{
           {1, 1536, 4096}, {1, 4096, 1024}, {1, 128, 4096}, {1, 3000, 512}}) {
    const Problem p = make_problem(m, n, k, static_cast<size_t>(k), 0x5EED + n);
    Device d(p);
    const std::string label = "bf16_gemv M" + std::to_string(m) + "xN" +
                              std::to_string(n) + "xK" + std::to_string(k);
    check_bf16(p, run_bf16(gemm, p, d), label.c_str());
    // Determinism (graph replay premise).
    const auto again = run_bf16(gemm, p, d);
    const auto first = run_bf16(gemm, p, d);
    require(std::memcmp(again.data(), first.data(), first.size() * 2) == 0,
            "bf16 gemv deterministic");
  }
}

// The Qwen3.8-Flash-Next decode shapes through the interface at every decode row
// count (2026-09-09, the real-checkpoint localizer: the T=5 walk — the
// GEMV family — disagreed with the T=9 walk — cuBLASLt — by 16 % at row 4
// where the python reference had certified the Lt family): every row of
// every shape against the fp64 oracle at m = 1..8, bf16 and f32 outputs.
DGPP_TEST(bf16_gemv_qwen_shapes_every_row_count_matches_oracle) {
  dgpp::CublasLtGemm gemm;
  struct Shape { int n, k; const char* what; };
  const Shape shapes[] = {
      {320, 10240, "GR down (k=W)"},      {10240, 320, "GR up (k=lowrank)"},
      {10240, 2560, "GDN qkv / PLE key"},  {6144, 2560, "GDN z / QSA o-width"},
      {48, 2560, "GDN a/b (n=lv)"},        {2560, 6144, "GDN out / QSA o_proj"},
      {12288, 2560, "QSA q (24 x 512)"},   {512, 2560, "QSA k/v (2 x 256)"},
      {640, 2560, "indexer (5 x 128) / shared gate"}, {2560, 640, "shared down"},
      {2560, 2560, "PLE value / MTP fc"},  {4096, 2560, "lm head slice"},
  };
  uint64_t seed = 0x51ull;
  for (const Shape& sh : shapes) {
    for (int m = 1; m <= 8; ++m) {
      Problem p = make_problem(m, sh.n, sh.k, static_cast<size_t>(sh.k), seed++);
      Device d(p);
      const auto got = run_bf16(gemm, p, d);
      const auto want = oracle(p);
      const auto rep = compare_bf16_vs_oracle(got.data(), want, 2.0, 1e-2);
      if (rep.mismatches != 0)
        throw std::runtime_error(std::string(sh.what) + " m=" + std::to_string(m) +
                                 ": bf16 out off the oracle (" + std::to_string(rep.mismatches) +
                                 " of " + std::to_string(rep.total) + ", l2 " +
                                 std::to_string(rep.l2_rel) + ")");
      if (m == 1 || m == 5 || m == 8)
        std::printf("[ OK ] %s n=%d k=%d m=%d: l2_rel=%.3g\n", sh.what, sh.n, sh.k, m, rep.l2_rel);
    }
    // The f32 output (the head, the shared down) at m = 3.
    Problem p = make_problem(3, sh.n, sh.k, static_cast<size_t>(sh.k), seed++);
    Device d(p);
    check_f32(p, run_f32(gemm, p, d), sh.what);
  }
}

// The other side of the interface at the same shapes: cuBLASLt (m > 8) against
// the fp64 oracle. The real-checkpoint profile (2026-09-09) showed the
// T=5 and T=9 walks bitwise within their families and 0.3-0.8 % apart at
// layer 0 — one family is off; the GEMV passed above.
DGPP_TEST(cublaslt_qwen_shapes_at_prefill_rows_match_oracle) {
  dgpp::CublasLtGemm gemm;
  struct Shape { int n, k; const char* what; };
  const Shape shapes[] = {
      {320, 10240, "GR down (k=W)"},      {10240, 320, "GR up (k=lowrank)"},
      {10240, 2560, "GDN qkv / PLE key"},  {6144, 2560, "GDN z"},
      {48, 2560, "GDN a/b (n=lv)"},        {2560, 6144, "GDN out / QSA o_proj"},
      {12288, 2560, "QSA q"},              {512, 2560, "QSA k/v"},
      {640, 2560, "indexer / shared gate"}, {2560, 640, "shared down"},
      {2560, 2560, "PLE value / MTP fc"},  {4096, 2560, "lm head slice"},
  };
  uint64_t seed = 0x77ull;
  bool ok = true;
  for (const Shape& sh : shapes) {
    for (const int m : {9, 12, 16, 72}) {
      Problem p = make_problem(m, sh.n, sh.k, static_cast<size_t>(sh.k), seed++);
      Device d(p);
      const auto got = run_bf16(gemm, p, d);
      const auto want = oracle(p);
      const auto rep = compare_bf16_vs_oracle(got.data(), want, 2.0, 1e-2);
      std::printf("[ %s ] cuBLASLt %s n=%d k=%d m=%d: l2_rel=%.3g mismatches=%ld/%ld\n",
                  rep.mismatches == 0 ? "OK" : "!!", sh.what, sh.n, sh.k, m, rep.l2_rel, rep.mismatches,
                  rep.total);
      if (rep.mismatches != 0) ok = false;
    }
  }
  if (!ok) throw std::runtime_error("cuBLASLt is off the fp64 oracle at a Qwen prefill shape");
}

DGPP_TEST(bf16_gemv_f32_output_and_strided_activation_view) {
  dgpp::CublasLtGemm gemm;
  // GIVEN the DSA indexer-weights shape (F32 out, n = 32 heads) ...
  {
    const Problem p = make_problem(1, 32, 4096, 4096, 0xF32);
    Device d(p);
    check_f32(p, run_f32(gemm, p, d), "bf16_gemv f32 M1xN32xK4096");
  }
  // ... AND the KDA f_b view: k = head_dim columns read out of a fused
  // projection row of stride n_in (16B-aligned slice, wider stride).
  {
    const Problem p = make_problem(1, 1024, 128, 4352, 0xF33);
    Device d(p);
    check_bf16(p, run_bf16(gemm, p, d), "bf16_gemv strided M1xN1024xK128/4352");
  }
}

DGPP_TEST(bf16_gemv_ragged_n_short_k_and_multi_row) {
  dgpp::CublasLtGemm gemm;
  // GIVEN n not a multiple of the 8-row block and k = one chunk (8 elems):
  {
    const Problem p = make_problem(1, 1003, 8, 8, 0xA9);
    Device d(p);
    check_bf16(p, run_bf16(gemm, p, d), "bf16_gemv M1xN1003xK8");
  }
  // AND k past one batch (16 chunks per lane at 4096; 4224 = 16.5 chunks):
  {
    const Problem p = make_problem(1, 264, 4224, 4224, 0xAA);
    Device d(p);
    check_bf16(p, run_bf16(gemm, p, d), "bf16_gemv M1xN264xK4224");
  }
  // AND m=3: every row's bits equal its m=1 bits (row independence), m=4
  // matches the oracle.
  {
    const Problem p3 = make_problem(3, 520, 2048, 2048, 0xAB);
    Device d3(p3);
    const auto got3 = run_bf16(gemm, p3, d3);
    check_bf16(p3, got3, "bf16_gemv M3xN520xK2048");
    for (int r = 0; r < 3; ++r) {
      Problem p1{1, p3.n, p3.k, p3.act_stride, {}, p3.weight};
      p1.act.assign(p3.act.begin() + static_cast<long>(r) * p3.act_stride,
                    p3.act.begin() + static_cast<long>(r + 1) * p3.act_stride);
      Device d1(p1);
      const auto got1 = run_bf16(gemm, p1, d1);
      require(std::memcmp(got1.data(), got3.data() + static_cast<size_t>(r) * p3.n,
                          static_cast<size_t>(p3.n) * 2) == 0,
              "bf16 gemv row bits independent of m");
    }
    const Problem p4 = make_problem(4, 96, 1024, 1024, 0xAC);
    Device d4(p4);
    check_bf16(p4, run_bf16(gemm, p4, d4), "bf16_gemv M4xN96xK1024");
  }
  // AND the serving ceiling m=8 is lowered to deterministic GEMV chunks.
  // K=4096 means a single eight-row launch would exceed the default 48-KiB
  // smem ceiling, so this also pins output offsets between chunks. Both
  // epilogues must retain the scalar row bits.
  {
    const Problem p8 = make_problem(8, 136, 4096, 4352, 0xAD);
    Device d8(p8);
    const auto got8 = run_bf16(gemm, p8, d8);
    const auto got8f = run_f32(gemm, p8, d8);
    check_bf16(p8, got8, "bf16_gemv chunked M8xN136xK4096");
    check_f32(p8, got8f, "bf16_gemv chunked M8xN136xK4096");
    for (int r = 0; r < p8.m; ++r) {
      Problem p1{1, p8.n, p8.k, p8.act_stride, {}, p8.weight};
      p1.act.assign(p8.act.begin() + static_cast<long>(r) * p8.act_stride,
                    p8.act.begin() + static_cast<long>(r + 1) * p8.act_stride);
      Device d1(p1);
      const auto got1 = run_bf16(gemm, p1, d1);
      const auto got1f = run_f32(gemm, p1, d1);
      require(std::memcmp(got1.data(),
                          got8.data() + static_cast<size_t>(r) * p8.n,
                          static_cast<size_t>(p8.n) * sizeof(uint16_t)) == 0,
              "chunked bf16 row bits independent of m");
      require(std::memcmp(got1f.data(),
                          got8f.data() + static_cast<size_t>(r) * p8.n,
                          static_cast<size_t>(p8.n) * sizeof(float)) == 0,
              "chunked f32 row bits independent of m");
    }
  }
}

// The dual launch (the KDA layer's f_b/g_b pair) must be bitwise the two
// single launches on both outputs, for every row count and both output
// types, including the strided-activation view and a ragged second n.
DGPP_TEST(bf16_gemv_multi_matches_four_single_launches_bitwise) {
  // The GDN's decode shape: four projections off one activation — a wide
  // one, a medium one and two 12-row ones — in one launch, m = 1..4.
  dgpp::CublasLtGemm gemm;
  for (int m = 1; m <= 4; ++m) {
    const Problem a = make_problem(m, 2560, 2560, 2560, 0x0e00 + m);
    const Problem b = make_problem(m, 1536, 2560, 2560, 0x0e10 + m);
    const Problem c = make_problem(m, 12, 2560, 2560, 0x0e20 + m);
    const Problem d = make_problem(m, 12, 2560, 2560, 0x0e30 + m);
    Device da(a), db(b), dc(c), dd(d);
    const std::vector<uint16_t> wa = run_bf16(gemm, a, da), wb = run_bf16(gemm, b, db),
                                wc = run_bf16(gemm, c, dc), wd = run_bf16(gemm, d, dd);
    dgpp::Bf16GemvProblem p[4];
    const Problem* probs[4] = {&a, &b, &c, &d};
    Device* devs[4] = {&da, &db, &dc, &dd};
    for (int i = 0; i < 4; ++i) {
      p[i].act = devs[i]->act; p[i].act_row_stride = probs[i]->act_stride;
      p[i].weight = devs[i]->w; p[i].out = devs[i]->out; p[i].n = probs[i]->n;
    }
    dgpp::launch_bf16_gemv_multi(p, 4, /*out_f32=*/false, m, a.k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    if (std::memcmp(da.out, wa.data(), wa.size() * 2) != 0 || std::memcmp(db.out, wb.data(), wb.size() * 2) != 0 ||
        std::memcmp(dc.out, wc.data(), wc.size() * 2) != 0 || std::memcmp(dd.out, wd.data(), wd.size() * 2) != 0)
      throw std::runtime_error("multi bf16 GEMV differs from four launches (m=" + std::to_string(m) + ")");
  }
  std::printf("[ OK ] multi GEMV: four problems bitwise their single launches at m = 1..4\n");
}

DGPP_TEST(bf16_gemv_dual_matches_two_single_launches_bitwise) {
  dgpp::CublasLtGemm gemm;
  for (int m = 1; m <= 4; ++m) {
    const Problem a = make_problem(m, 1024, 128, 4352, 0x0d0a + m);
    const Problem b = make_problem(m, 1000, 128, 4352, 0x0d0b + m);  // ragged n
    Device da(a), db(b);
    // GIVEN the two single-launch results (through the interface, bf16 and f32)
    const std::vector<uint16_t> want_a = run_bf16(gemm, a, da);
    const std::vector<uint16_t> want_b = run_bf16(gemm, b, db);
    const std::vector<float> want_a32 = run_f32(gemm, a, da);
    const std::vector<float> want_b32 = run_f32(gemm, b, db);
    // WHEN the same pair runs as one dual launch
    dgpp::Bf16GemvProblem p0, p1;
    p0.act = da.act; p0.act_row_stride = a.act_stride; p0.weight = da.w;
    p0.out = da.out; p0.n = a.n;
    p1.act = db.act; p1.act_row_stride = b.act_stride; p1.weight = db.w;
    p1.out = db.out; p1.n = b.n;
    dgpp::launch_bf16_gemv_dual(p0, p1, /*out_f32=*/false, m, a.k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    // THEN both outputs are the single launches' bits
    if (std::memcmp(da.out, want_a.data(), want_a.size() * 2) != 0 ||
        std::memcmp(db.out, want_b.data(), want_b.size() * 2) != 0)
      throw std::runtime_error("dual bf16 GEMV differs from two launches (m=" +
                               std::to_string(m) + ")");
    dgpp::launch_bf16_gemv_dual(p0, p1, /*out_f32=*/true, m, a.k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    if (std::memcmp(da.out, want_a32.data(), want_a32.size() * 4) != 0 ||
        std::memcmp(db.out, want_b32.data(), want_b32.size() * 4) != 0)
      throw std::runtime_error("dual f32 GEMV differs from two launches (m=" +
                               std::to_string(m) + ")");
  }
}

DGPP_TEST(bf16_gemv_contract_rejects_odd_k_and_falls_back) {
  // GIVEN k % 8 != 0: the interface must not take the GEMV (bf16_gemv_accepts
  // says so) — and the launcher refuses it outright.
  require(!dgpp::bf16_gemv_accepts(reinterpret_cast<void*>(16), 1, 12),
          "k=12 outside the contract");
  require(!dgpp::bf16_gemv_accepts(reinterpret_cast<void*>(16), 5, 64),
          "m=5 outside the contract");
  require(!dgpp::bf16_gemv_accepts(reinterpret_cast<void*>(8), 1, 64),
          "8B-aligned weight outside the contract");
  require(dgpp::bf16_gemv_accepts(reinterpret_cast<void*>(16), 4, 64),
          "m=4, k=64, 16B-aligned inside");
  bool threw = false;
  try {
    dgpp::launch_bf16_gemv(reinterpret_cast<const uint16_t*>(16), 12,
                           reinterpret_cast<const uint16_t*>(16),
                           reinterpret_cast<void*>(16), false, 1, 8, 12,
                           nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "launcher rejects k=12");
}

// Timing only (no verdict): the bf16 interface's Lt algorithm and its GEMV
// chunks against the streaming tensor-core form (set_decode_mma) at the
// GLM-4.7 q_proj shape [12288 x 5120] over the row counts a served walk
// sees (decode rows, verify rows, group prefills, prefill chunks): the
// bound above which an opted-in instance hands rows back to Lt is read
// from this table (2026-09-14). Cold weights: four copies rotated.
DGPP_TEST(bf16_interface_lt_vs_mma_timing) {
  const int m_max = 2048, copies = 4;
  const size_t ws_bytes = 64u << 20;
  void* ws;
  DGPP_CUDA_OK(cudaMalloc(&ws, ws_bytes));
  cudaEvent_t e0, e1;
  DGPP_CUDA_OK(cudaEventCreate(&e0)); DGPP_CUDA_OK(cudaEventCreate(&e1));
  // The GLM-4.7 attention shapes: q_proj [12288 x 5120], k/v_proj [1024 x 5120], o_proj [5120 x 12288].
  for (auto [n, k] : std::vector<std::pair<int, int>>{{12288, 5120}, {1024, 5120}, {5120, 12288}}) {
    const size_t wbytes = static_cast<size_t>(n) * k * 2;
    std::vector<uint16_t> host(static_cast<size_t>(n) * k);
    std::mt19937_64 rng(77);
    for (auto& v : host) v = static_cast<uint16_t>(0x3C00 + (rng() % 0x0400));  // ~[1, 2) bf16-ish
    uint16_t* w[copies];
    for (int c = 0; c < copies; ++c) {
      DGPP_CUDA_OK(cudaMalloc(&w[c], wbytes));
      DGPP_CUDA_OK(cudaMemcpy(w[c], host.data(), wbytes, cudaMemcpyHostToDevice));
    }
    uint16_t* act; uint16_t* out;
    DGPP_CUDA_OK(cudaMalloc(&act, static_cast<size_t>(m_max) * k * 2));
    DGPP_CUDA_OK(cudaMemset(act, 0, static_cast<size_t>(m_max) * k * 2));
    DGPP_CUDA_OK(cudaMalloc(&out, static_cast<size_t>(m_max) * n * 2));
    std::printf("[ .. ] bf16 [%d x %d] cold, us per product: m  chunks(<=32)  chunks<=4|Lt  mma\n", n, k);
    for (int m : {1, 2, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 2048}) {
      float t[3];
      for (int path = 0; path < 3; ++path) {
        dgpp::CublasLtGemm gemm;
        gemm.set_decode_rows(path == 0 ? 32 : 4);
        gemm.set_decode_mma(path == 2);
        auto run = [&](int c) {
          gemm.matmul(act, w[c], out, m, n, k, dgpp::DType::BF16, dgpp::GemmOut::BF16,
                      static_cast<size_t>(k), ws, ws_bytes, nullptr);
        };
        const int reps = m > 256 ? 8 : 16;
        for (int i = 0; i < copies; ++i) run(i);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        DGPP_CUDA_OK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i) run(i % copies);
        DGPP_CUDA_OK(cudaEventRecord(e1)); DGPP_CUDA_OK(cudaEventSynchronize(e1));
        float ms = 0; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1)); t[path] = ms * 1000.f / reps;
      }
      std::printf("[ .. ]   m=%5d  %10.1f us  %10.1f us  %10.1f us\n", m, t[0], t[1], t[2]);
    }
    for (int c = 0; c < copies; ++c) cudaFree(w[c]);
    cudaFree(act); cudaFree(out);
  }
  cudaFree(ws);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
