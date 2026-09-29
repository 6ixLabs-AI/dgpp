// Parity tests for the packed-int GEMV core (docs/glm53_plan.md G2): the
// single-matrix launcher against a double oracle across every row
// geometry the core has at both code widths (32 .. 1 lanes per row, one
// to eight passes), the row-count invariance the batch family relies on,
// the f32 epilogue as the unrounded bf16, NaN scale propagation, the
// geometry contract, and — with --checkpoint-dir pointing at the
// GLM-5.3 int4/int8 checkpoint — real slices.
//
// The oracle: w = (code - 2^(bits-1)) x bf16(scale) exactly (the exact
// policy — no weight rounding, none in the engine), fp64 accumulation,
// rounded to bf16 as the launcher's bf16 epilogue rounds. The kernel
// differs from it by fp32 accumulation order only, so the budget is the
// strict FP8 suite's: 2 bf16 ulps with a 1e-3 cancellation floor, zero
// mismatches.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "kernels/packq_gemv.hpp"
#include "models/quant_matrix.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;

struct Problem {
  int bits, m, n, k;
  int sf = 0;                     // the scale format: 0 bf16 per 64, 1 f16 per 128
  std::vector<uint16_t> act;      // [m, k] bf16
  std::vector<uint32_t> packed;   // [n, k*bits/32]
  std::vector<uint16_t> scales;   // [n, k/group] in the format's dtype
  int group() const { return dgpp::packed_scale_group(sf); }
};

// The checkpoint's packing: unsigned codes offset 2^(bits-1), 32/bits per
// word, the low nibble / byte first.
void pack_codes(const std::vector<int>& codes, int bits, std::vector<uint32_t>& out) {
  const int per = 32 / bits;
  out.assign(codes.size() / per, 0u);
  for (size_t i = 0; i < codes.size(); ++i) {
    const uint32_t u = static_cast<uint32_t>(codes[i] + (1 << (bits - 1)));
    out[i / per] |= u << (bits * (i % per));
  }
}

Problem make_problem(int bits, int m, int n, int k, uint64_t seed, int sf = 0) {
  Problem p;
  p.bits = bits;
  p.m = m;
  p.n = n;
  p.k = k;
  p.sf = sf;
  Rng rng(seed);
  p.act.resize(static_cast<size_t>(m) * k);
  fill_act(rng, p.act);
  const int lo = -(1 << (bits - 1)), hi = (1 << (bits - 1)) - 1;
  std::vector<int> codes(static_cast<size_t>(n) * k);
  for (auto& c : codes) c = lo + static_cast<int>(rng.next() % static_cast<uint64_t>(hi - lo + 1));
  pack_codes(codes, bits, p.packed);
  p.scales.resize(static_cast<size_t>(n) * k / p.group());
  for (auto& s : p.scales) {
    const float v = static_cast<float>(std::exp2(rng.unit() * 2.0) * 0.0625);
    s = sf == 0 ? dgpp::float_to_bf16_bits(v) : dgpp::float_to_fp16_bits(v);
  }
  return p;
}

int code_at(const Problem& p, int nn, int kk) {
  const int per = 32 / p.bits;
  const uint32_t word = p.packed[static_cast<size_t>(nn) * (p.k / per) + kk / per];
  const uint32_t u = (word >> (p.bits * (kk % per))) & ((1u << p.bits) - 1u);
  return static_cast<int>(u) - (1 << (p.bits - 1));
}

std::vector<double> oracle(const Problem& p) {
  std::vector<double> out(static_cast<size_t>(p.m) * p.n, 0.0);
  std::vector<double> wrow(p.k);
  for (int nn = 0; nn < p.n; ++nn) {
    const int g = p.group();
    for (int kk = 0; kk < p.k; ++kk) {
      const float s = dgpp::packed_scale_to_float(
          p.scales[static_cast<size_t>(nn) * (p.k / g) + kk / g], p.sf);
      wrow[kk] = static_cast<double>(code_at(p, nn, kk)) * static_cast<double>(s);
    }
    for (int mm = 0; mm < p.m; ++mm) {
      double acc = 0.0;
      const uint16_t* arow = p.act.data() + static_cast<size_t>(mm) * p.k;
      for (int kk = 0; kk < p.k; ++kk) acc += bf16_to_float(arow[kk]) * wrow[kk];
      out[static_cast<size_t>(mm) * p.n + nn] =
          bf16_to_float(dgpp::float_to_bf16_bits(static_cast<float>(acc)));
    }
  }
  return out;
}

struct DevMatrix {
  uint32_t* packed = nullptr;
  uint16_t* scales = nullptr;
  dgpp::GlmPackedMatrix view;
  explicit DevMatrix(const Problem& p) {
    DGPP_CUDA_OK(cudaMallocManaged(&packed, p.packed.size() * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&scales, p.scales.size() * 2));
    std::memcpy(packed, p.packed.data(), p.packed.size() * 4);
    std::memcpy(scales, p.scales.data(), p.scales.size() * 2);
    view = dgpp::GlmPackedMatrix{packed, scales, p.n, p.k, p.bits, p.sf};
  }
  ~DevMatrix() {
    cudaFree(packed);
    cudaFree(scales);
  }
};

std::vector<uint16_t> run_bf16(const Problem& p) {
  DevMatrix w(p);
  uint16_t* act = nullptr;
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 2));
  std::memcpy(act, p.act.data(), p.act.size() * 2);
  dgpp::launch_packq_gemv_bf16(act, p.k, w.view, out, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 2);
  cudaFree(act);
  cudaFree(out);
  return got;
}

std::vector<float> run_f32(const Problem& p) {
  DevMatrix w(p);
  uint16_t* act = nullptr;
  float* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 4));
  std::memcpy(act, p.act.data(), p.act.size() * 2);
  dgpp::launch_packq_gemv_f32(act, p.k, w.view, out, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 4);
  cudaFree(act);
  cudaFree(out);
  return got;
}

void check_oracle(const Problem& p, const std::vector<uint16_t>& got, const char* label) {
  const auto want = oracle(p);
  const auto rep = compare_bf16_vs_oracle(got.data(), want, /*ulp_budget=*/2.0,
                                          /*floor_frac=*/1e-3);
  std::printf("[ OK ] %s: l2_rel=%.3g mismatches=%ld/%zu\n", label, rep.l2_rel,
              rep.mismatches, rep.total);
  require_report(rep, 1e-3, 0, label);
}

Problem row_of(const Problem& p, int r) {
  Problem q;
  q.bits = p.bits;
  q.sf = p.sf;
  q.m = 1;
  q.n = p.n;
  q.k = p.k;
  q.act.assign(p.act.begin() + static_cast<long>(r) * p.k,
               p.act.begin() + static_cast<long>(r + 1) * p.k);
  q.packed = p.packed;
  q.scales = p.scales;
  return q;
}

}  // namespace

DGPP_TEST(packq_gemv_matches_oracle_across_row_geometries) {
  // k picks the row geometry. int4: 64 (one lane, two chunks), 128 (one
  // lane, four), 256 (2 lanes) ... 4096 (32 lanes x 4 chunks), 6144 (32 x
  // 6 in two passes), 192 (2 x 3), 384 (4 x 3), 8192 (32 x 8), 12288 (32
  // x 12), 16384 (32 x 16 in four passes). int8: 64 (one lane, four
  // chunks), 128 (2 lanes x 4) ... 2048 (32 x 4), 4096 (32 x 8), 6144
  // (32 x 12), 8192 (32 x 16), 16384 (32 x 32 in eight passes), 192 (4 x
  // 3), 384 (8 x 3). n leaves partial blocks and dead lane groups.
  struct Shape {
    int bits, m, n, k;
  };
  const Shape shapes[] = {
      {4, 1, 520, 4096}, {4, 3, 264, 1024}, {4, 2, 100, 512},  {4, 4, 40, 256},
      {4, 1, 33, 128},   {4, 2, 17, 64},    {4, 4, 512, 4096}, {4, 2, 300, 2048},
      {4, 1, 390, 6144}, {4, 3, 130, 6144}, {4, 2, 41, 192},   {4, 1, 77, 384},
      {4, 4, 70, 768},   {4, 2, 20, 8192},  {4, 1, 20, 12288}, {4, 2, 12, 16384},
      {8, 1, 520, 2048}, {8, 3, 264, 1024}, {8, 2, 100, 512},  {8, 4, 40, 256},
      {8, 1, 33, 128},   {8, 2, 17, 64},    {8, 4, 300, 6144}, {8, 2, 200, 4096},
      {8, 1, 77, 192},   {8, 3, 41, 384},   {8, 2, 60, 8192},  {8, 1, 12, 16384},
      {8, 4, 70, 1536},  {8, 2, 30, 3072}};
  int i = 0;
  for (const Shape& s : shapes) {
    const Problem p = make_problem(s.bits, s.m, s.n, s.k, 0x9A0 + i++);
    const std::string label = "packq int" + std::to_string(s.bits) + " gemv M" +
                              std::to_string(s.m) + "xN" + std::to_string(s.n) + "xK" +
                              std::to_string(s.k);
    check_oracle(p, run_bf16(p), label.c_str());
  }
}

DGPP_TEST(packq_gemv_g128_f16_matches_oracle) {
  // The f16-per-128 scale format (docs/qwen38_autoround_int4_plan.md D1)
  // at the AutoRound hybrid's widths — int4 640 (4 lanes x 5 chunks) and
  // 2560 (16 x 5), int8 2560 (32 x 5) and 640 (8 x 5) — and the smaller
  // 128-multiples of the compiled set. Same oracle, same budget: the
  // chain is the format-0 chain with a different scale index and widening.
  struct Shape {
    int bits, m, n, k;
  };
  const Shape shapes[] = {
      {4, 1, 520, 640},  {4, 3, 264, 2560}, {4, 2, 100, 128},  {4, 4, 40, 256},
      {4, 1, 77, 384},   {4, 2, 30, 1024},  {4, 4, 17, 512},   {4, 2, 130, 2560},
      {8, 1, 520, 2560}, {8, 3, 264, 640},  {8, 2, 100, 128},  {8, 4, 40, 256},
      {8, 1, 77, 384},   {8, 2, 30, 2048},  {8, 4, 70, 1536},  {8, 1, 300, 640}};
  int i = 0;
  for (const Shape& s : shapes) {
    const Problem p = make_problem(s.bits, s.m, s.n, s.k, 0xF16 + i++, 1);
    const std::string label = "packq int" + std::to_string(s.bits) + " g128/f16 gemv M" +
                              std::to_string(s.m) + "xN" + std::to_string(s.n) + "xK" +
                              std::to_string(s.k);
    check_oracle(p, run_bf16(p), label.c_str());
  }
  // Row independence and the f32 epilogue at the production widths.
  for (int bits : {4, 8})
    for (int k : {640, 2560}) {
      const Problem p3 = make_problem(bits, 3, 296, k, 0xE3 + k + bits, 1);
      const std::vector<uint16_t> got3 = run_bf16(p3);
      for (int r = 0; r < 3; ++r) {
        const std::vector<uint16_t> got1 = run_bf16(row_of(p3, r));
        require(std::memcmp(got1.data(), got3.data() + static_cast<size_t>(r) * p3.n,
                            static_cast<size_t>(p3.n) * 2) == 0,
                "packq g128 row bits independent of m (3)");
      }
      const Problem p8 = make_problem(bits, 8, 136, k, 0xE8 + k + bits, 1);
      const std::vector<uint16_t> got8 = run_bf16(p8);
      const std::vector<float> got8f = run_f32(p8);
      for (int r = 0; r < 8; ++r) {
        const Problem p1 = row_of(p8, r);
        const std::vector<uint16_t> got1 = run_bf16(p1);
        const std::vector<float> got1f = run_f32(p1);
        require(std::memcmp(got1.data(), got8.data() + static_cast<size_t>(r) * p8.n,
                            static_cast<size_t>(p8.n) * 2) == 0,
                "packq g128 chunked bf16 row bits independent of m (8)");
        require(std::memcmp(got1f.data(), got8f.data() + static_cast<size_t>(r) * p8.n,
                            static_cast<size_t>(p8.n) * 4) == 0,
                "packq g128 chunked f32 row bits independent of m (8)");
      }
      for (size_t j = 0; j < got8.size(); ++j)
        require(dgpp::float_to_bf16_bits(got8f[j]) == got8[j], "g128 bf16(out_f32) == out_bf16");
    }
  // An f16 NaN scale poisons exactly its row.
  for (int bits : {4, 8}) {
    Problem p = make_problem(bits, 1, 300, 640, 0xA5 + bits, 1);
    p.scales[static_cast<size_t>(9) * (p.k / 128) + 3] = 0x7E00;
    p.scales[static_cast<size_t>(250) * (p.k / 128) + 0] = 0xFE00;
    const std::vector<uint16_t> got = run_bf16(p);
    for (int nn = 0; nn < p.n; ++nn) {
      const bool got_nan = std::isnan(bf16_to_float(got[static_cast<size_t>(nn)]));
      require(got_nan == (nn == 9 || nn == 250), "g128 f16 NaN scale poisons exactly its row");
    }
  }
  // The contract: a 128-multiple inside the format's compiled set.
  auto rejects = [](int bits, int k, int sf) {
    Problem p = make_problem(bits, 1, 8, k % 128 == 0 ? k : 128, 0xBAD, 1);
    p.k = k;
    p.sf = sf;
    try {
      (void)run_bf16(p);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(rejects(4, 192, 1), "g128: k=192 rejected (not a multiple of 128)");
  require(rejects(4, 320, 1), "g128: k=320 rejected");
  require(rejects(4, 4096, 1), "g128: k=4096 rejected (outside the format's compiled set)");
  require(rejects(8, 640, 2), "an unknown scale format is rejected");
  require(!rejects(4, 640, 1) && !rejects(4, 2560, 1) && !rejects(8, 2560, 1) && !rejects(8, 640, 1),
          "g128: the AutoRound widths accepted");
  std::printf("[ OK ] packq gemv g128/f16: oracle, row independence, NaN, contract\n");
}

DGPP_TEST(packq_gemv_rows_are_independent_of_row_count) {
  // Each row's chain is the same sequence of FMAs whatever m: m=3 against
  // three m=1 runs, and the m=8 chunked launch against eight singles, for
  // both epilogues, both widths and the production geometries.
  for (int bits : {4, 8})
    for (int k : {6144, 512, 2048, 192}) {
      const Problem p3 = make_problem(bits, 3, 296, k, 0xE3 + k + bits);
      const std::vector<uint16_t> got3 = run_bf16(p3);
      for (int r = 0; r < 3; ++r) {
        const std::vector<uint16_t> got1 = run_bf16(row_of(p3, r));
        require(std::memcmp(got1.data(), got3.data() + static_cast<size_t>(r) * p3.n,
                            static_cast<size_t>(p3.n) * 2) == 0,
                "packq row bits independent of m (3)");
      }
      const Problem p8 = make_problem(bits, 8, 136, k, 0xE8 + k + bits);
      const std::vector<uint16_t> got8 = run_bf16(p8);
      const std::vector<float> got8f = run_f32(p8);
      for (int r = 0; r < 8; ++r) {
        const Problem p1 = row_of(p8, r);
        const std::vector<uint16_t> got1 = run_bf16(p1);
        const std::vector<float> got1f = run_f32(p1);
        require(std::memcmp(got1.data(), got8.data() + static_cast<size_t>(r) * p8.n,
                            static_cast<size_t>(p8.n) * 2) == 0,
                "packq chunked bf16 row bits independent of m (8)");
        require(std::memcmp(got1f.data(), got8f.data() + static_cast<size_t>(r) * p8.n,
                            static_cast<size_t>(p8.n) * 4) == 0,
                "packq chunked f32 row bits independent of m (8)");
      }
    }
  std::printf("[ OK ] packq gemv rows independent of m through m=8 chunks\n");
}

DGPP_TEST(packq_gemv_f32_epilogue_is_the_unrounded_bf16) {
  for (int bits : {4, 8}) {
    const Problem p = make_problem(bits, 4, 200, 1024, 0xF32 + bits);
    const std::vector<uint16_t> b = run_bf16(p);
    const std::vector<float> f = run_f32(p);
    for (size_t i = 0; i < b.size(); ++i)
      require(dgpp::float_to_bf16_bits(f[i]) == b[i], "bf16(out_f32) == out_bf16");
  }
  std::printf("[ OK ] packq gemv f32 epilogue rounds to the bf16 epilogue\n");
}

DGPP_TEST(packq_gemv_propagates_nan_scales_exactly) {
  // Two poisoned scales: exactly their rows' outputs NaN.
  for (int bits : {4, 8}) {
    Problem p = make_problem(bits, 1, 300, 1024, 0xA5 + bits);
    p.scales[static_cast<size_t>(9) * (p.k / 64) + 3] = 0x7FC0;
    p.scales[static_cast<size_t>(250) * (p.k / 64) + 0] = 0xFFC0;
    const std::vector<uint16_t> got = run_bf16(p);
    for (int nn = 0; nn < p.n; ++nn) {
      const bool got_nan = std::isnan(bf16_to_float(got[static_cast<size_t>(nn)]));
      if (nn == 9 || nn == 250)
        require(got_nan, "poisoned scale row is NaN");
      else
        require(!got_nan, "unpoisoned row stays finite");
    }
  }
  std::printf("[ OK ] packq gemv nan propagation: rows 9 and 250 NaN\n");
}

DGPP_TEST(packq_gemv_rejects_geometry_outside_the_contract) {
  auto rejects = [](int bits, int k) {
    Problem p = make_problem(bits, 1, 8, k, 0xBAD);
    if (bits != 4 && bits != 8) {
      p.bits = bits;  // an unsupported width on a valid int8 layout
    }
    try {
      (void)run_bf16(p);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(rejects(4, 96), "k=96 rejected (not a multiple of 64)");
  require(rejects(4, 320), "k=320 rejected (not in the compiled set)");
  require(rejects(8, 320), "int8 k=320 rejected (not in the compiled set)");
  require(!rejects(4, 6144), "int4 k=6144 accepted");
  require(!rejects(8, 6144), "int8 k=6144 accepted");
  require(!rejects(4, 16384), "int4 k=16384 accepted (32 lanes x 16 chunks)");
  require(!rejects(8, 16384), "int8 k=16384 accepted (32 lanes x 32 chunks)");
  require(rejects(4, 32768), "k=32768 rejected (not in the compiled set)");
  Problem p = make_problem(8, 1, 8, 1024, 0xBAD);
  p.bits = 3;
  bool bad_width = false;
  try {
    (void)run_bf16(p);
  } catch (const std::invalid_argument&) {
    bad_width = true;
  }
  require(bad_width, "a 3-bit width is rejected");
  std::printf("[ OK ] packq gemv geometry contract enforced\n");
}

// Real-checkpoint slice parity lives in packq_gemv_checkpoint.cpp (host-only
// TU: the safetensors reader's JSON parser does not mix with nvcc).
int run_packq_gemv_checkpoint_parity(const char* checkpoint_dir);

int main(int argc, char** argv) {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  const int rc = dgpp::test::run_all();
  if (rc != 0) return rc;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--checkpoint-dir") == 0 && i + 1 < argc) {
      const int prc = run_packq_gemv_checkpoint_parity(argv[i + 1]);
      if (prc != 0) return prc;
    }
  }
  return 0;
}
