// Parity tests for the scale-aware GEMM (M4 deliverable 3: "block edges,
// saturation, NaN/Inf policy, and real checkpoint slices" — the real-slice
// half runs via --checkpoint-dir, mirroring the dump-parity runner
// pattern). Every synthetic case is checked against both oracles: strict
// (bf16-rounded weights, fp64 accumulation — isolates the kernel) and
// semantic (true dequant — pins the DESIGN §4 scale contract).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "kernels/fp8_gemv.cuh"
#include "kernels/scale_gemm.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;

struct Problem {
  int m, n, k;
  std::vector<uint16_t> act;      // [m, k]
  std::vector<uint8_t> payload;   // [n, k]
  std::vector<float> scales;      // [ceil(n/128), ceil(k/128)]
};

Problem make_problem(int m, int n, int k, uint64_t seed) {
  Problem p;
  p.m = m;
  p.n = n;
  p.k = k;
  Rng rng(seed);
  p.act.resize(static_cast<size_t>(m) * k);
  p.payload.resize(static_cast<size_t>(n) * k);
  p.scales.resize(static_cast<size_t>((n + 127) / 128) * ((k + 127) / 128));
  fill_act(rng, p.act);
  fill_payload(rng, p.payload);
  fill_scales(rng, p.scales);
  return p;
}

// Runs the kernel into managed memory and returns the bf16 output.
std::vector<uint16_t> run_kernel(const Problem& p) {
  uint16_t* act = nullptr;
  uint8_t* w = nullptr;
  float* s = nullptr;
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&w, p.payload.size()));
  DGPP_CUDA_OK(cudaMallocManaged(&s, p.scales.size() * 4));
  DGPP_CUDA_OK(
      cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 2));
  std::memcpy(act, p.act.data(), p.act.size() * 2);
  std::memcpy(w, p.payload.data(), p.payload.size());
  std::memcpy(s, p.scales.data(), p.scales.size() * 4);
  dgpp::launch_scale_gemm_bf16(act, p.k, w, s, out, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 2);
  DGPP_CUDA_OK(cudaFree(act));
  DGPP_CUDA_OK(cudaFree(w));
  DGPP_CUDA_OK(cudaFree(s));
  DGPP_CUDA_OK(cudaFree(out));
  return got;
}

// The same production dispatch with its unrounded f32 epilogue. Used by the
// row-independence gate because the MoE accumulation consumes these bits.
std::vector<float> run_kernel_f32(const Problem& p, int mma_from_rows = 0,
                                  int expected_mma_warps = 0) {
  uint16_t* act = nullptr;
  uint8_t* w = nullptr;
  float* s = nullptr;
  float* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&w, p.payload.size()));
  DGPP_CUDA_OK(cudaMallocManaged(&s, p.scales.size() * 4));
  DGPP_CUDA_OK(
      cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 4));
  std::memcpy(act, p.act.data(), p.act.size() * 2);
  std::memcpy(w, p.payload.data(), p.payload.size());
  std::memcpy(s, p.scales.data(), p.scales.size() * 4);
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  if (expected_mma_warps) {
    DGPP_CUDA_OK(cudaStreamCreate(&stream));
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  }
  dgpp::launch_scale_gemm_f32(act, p.k, w, s, out, p.m, p.n, p.k, stream, 0, mma_from_rows);
  bool width_matches = true;
  if (expected_mma_warps) {
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    size_t count = 0;
    DGPP_CUDA_OK(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    DGPP_CUDA_OK(cudaGraphGetNodes(graph, nodes.data(), &count));
    width_matches = count == 1;
    if (count == 1) {
      cudaKernelNodeParams params{};
      DGPP_CUDA_OK(cudaGraphKernelNodeGetParams(nodes[0], &params));
      const char* name = nullptr;
      DGPP_CUDA_OK(cudaFuncGetName(&name, params.func));
      width_matches = std::string(name).find("mma_gemv_kernel") != std::string::npos &&
                      params.blockDim.x == static_cast<unsigned>(32 * expected_mma_warps);
    }
    DGPP_CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    DGPP_CUDA_OK(cudaGraphLaunch(executable, stream));
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  if (expected_mma_warps) {
    DGPP_CUDA_OK(cudaGraphExecDestroy(executable));
    DGPP_CUDA_OK(cudaGraphDestroy(graph));
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
  }
  std::vector<float> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 4);
  DGPP_CUDA_OK(cudaFree(act));
  DGPP_CUDA_OK(cudaFree(w));
  DGPP_CUDA_OK(cudaFree(s));
  DGPP_CUDA_OK(cudaFree(out));
  require(width_matches, "FP8 head did not select the expected MMA warp width");
  return got;
}

// The tile kernel regardless of m (the routed launcher sends m <= 128 to
// the GEMV core; the grouped tensor-core MoE kernel is bitwise this one).
std::vector<uint16_t> run_tile_kernel(const Problem& p) {
  uint16_t* act = nullptr;
  uint8_t* w = nullptr;
  float* s = nullptr;
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, p.act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&w, p.payload.size()));
  DGPP_CUDA_OK(cudaMallocManaged(&s, p.scales.size() * 4));
  DGPP_CUDA_OK(
      cudaMallocManaged(&out, static_cast<size_t>(p.m) * p.n * 2));
  std::memcpy(act, p.act.data(), p.act.size() * 2);
  std::memcpy(w, p.payload.data(), p.payload.size());
  std::memcpy(s, p.scales.data(), p.scales.size() * 4);
  dgpp::launch_scale_gemm_tile_bf16(act, p.k, w, s, out, p.m, p.n, p.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(p.m) * p.n);
  std::memcpy(got.data(), out, got.size() * 2);
  DGPP_CUDA_OK(cudaFree(act));
  DGPP_CUDA_OK(cudaFree(w));
  DGPP_CUDA_OK(cudaFree(s));
  DGPP_CUDA_OK(cudaFree(out));
  return got;
}

// Runs both oracles and asserts the budgets.
void check_both_oracles(const Problem& p, const std::vector<uint16_t>& got,
                        const char* label) {
  const auto strict = gemm_oracle(p.act, p.k, p.payload, p.scales, p.m, p.n,
                                  p.k, /*strict=*/true);
  auto rep = compare_bf16_vs_oracle(got.data(), strict, /*ulp_budget=*/2.0,
                                    /*floor_frac=*/1e-3);
  std::printf("[ OK ] %s strict: l2_rel=%.3g mismatches=%ld/%zu\n", label,
              rep.l2_rel, rep.mismatches, rep.total);
  require_report(rep, 1e-3, 0, (std::string(label) + " strict").c_str());

  const auto semantic = gemm_oracle(p.act, p.k, p.payload, p.scales, p.m,
                                    p.n, p.k, /*strict=*/false);
  rep = compare_bf16_vs_oracle(got.data(), semantic, /*ulp_budget=*/8.0,
                               /*floor_frac=*/2e-2);
  std::printf("[ OK ] %s semantic: l2_rel=%.3g mismatches=%ld/%zu\n", label,
              rep.l2_rel, rep.mismatches, rep.total);
  require_report(rep, 8e-3, static_cast<long>(rep.total) / 20,
                 (std::string(label) + " semantic").c_str());
}

}  // namespace

// Vocabulary-head dimensions in K, with ragged two- and eight-warp slices. Check
// unrounded FP32 logits against an independent FP64 accumulation of the
// exact BF16-rounded weights, including the GEMV/MMA transition at row 5.
DGPP_TEST(scale_gemm_f32_fp8_head_numerics) {
  // N=4097 selects the same eight-warp specialization as a production
  // TP2 vocabulary shard, while keeping the exhaustive FP64 oracle bounded.
  // Its final block also exercises masked columns in that specialization.
  for (const int n : {257, 4097})
    for (const int m : {4, 5, 8, 16}) {
      const auto p = make_problem(m, n, 2560, 0x12345678 + m + n);
      const auto got = run_kernel_f32(p, 5, n == 4097 && m > 4 ? 8 : 0);
      const auto repeat = run_kernel_f32(p, 5);
      const auto gemv = run_kernel_f32(p);
      require(std::memcmp(got.data(), repeat.data(), got.size() * sizeof(float)) == 0,
              "FP8 head must repeat bitwise at a fixed shape");
      double error2 = 0, reference2 = 0;
      const int scale_cols = (p.k + 127) / 128;
      for (int row = 0; row < p.m; ++row) {
        for (int col = 0; col < p.n; ++col) {
          double reference = 0, magnitude = 0;
          for (int k = 0; k < p.k; ++k) {
            const float decoded = dgpp::fp8_e4m3_bits_to_float(p.payload[col * p.k + k]);
            const float scale = p.scales[(col / 128) * scale_cols + k / 128];
            const double weight = bf16_to_float(dgpp::float_to_bf16_bits(decoded * scale));
            const double product = bf16_to_float(p.act[row * p.k + k]) * weight;
            reference += product;
            magnitude += std::abs(product);
          }
          const size_t i = static_cast<size_t>(row) * p.n + col;
          require(std::isfinite(got[i]) && std::isfinite(gemv[i]), "nonfinite FP8 head logit");
          require(std::abs(got[i] - reference) <= 3e-5 * magnitude,
                  "streaming FP8 head exceeds the FP64 error budget");
          require(std::abs(gemv[i] - reference) <= 3e-5 * magnitude,
                  "reference GEMV exceeds the FP64 error budget");
          error2 += (got[i] - reference) * (got[i] - reference);
          reference2 += reference * reference;
        }
      }
      require(std::sqrt(error2 / reference2) < 1e-5, "FP8 head relative L2 error");
    }
}

DGPP_TEST(scale_gemm_full_blocks_match_both_oracles) {
  // Exact 128-block geometry with n/k tiles straddling block boundaries:
  // N=384 spans block rows 0-2; K=256 crosses the k=128 boundary mid-loop.
  const Problem p = make_problem(29, 384, 256, 0xA1);
  const std::vector<uint16_t> got = run_kernel(p);
  check_both_oracles(p, got, "full-blocks M29xN384xK256");

  // Determinism: a second run must be bitwise identical.
  const std::vector<uint16_t> again = run_kernel(p);
  require(std::memcmp(got.data(), again.data(), got.size() * 2) == 0,
          "second run bitwise identical");
}

DGPP_TEST(scale_gemm_tile_kernel_small_m_matches_both_oracles) {
  // The tile kernel below the routed launcher's GEMV threshold: partial
  // m-tiles (M=1, 5, 17) at the MoE small-case geometry — the prefill's
  // grouped tensor-core kernel is bitwise this kernel per segment, so its
  // accuracy at every segment length is this gate.
  for (const int m : {1, 5, 17}) {
    Problem p = make_problem(m, /*n=*/256, /*k=*/512, 0x7113 + m);
    const auto got = run_tile_kernel(p);
    check_both_oracles(p, got, ("tile M" + std::to_string(m) + "xN256xK512").c_str());
  }
}

DGPP_TEST(scale_gemm_large_m_route_is_bitwise_the_tile_kernel) {
  // m > 128 routes to the 128-row tensor-core kernel (the MoE experts'
  // dense form): bitwise the tile kernel — the same dequantized weights,
  // the same ascending-k16 mma chain — at a ragged n and m (three m-tiles,
  // the last partial), and within both oracles' budgets.
  Problem p = make_problem(/*m=*/300, /*n=*/200, /*k=*/512, 0x1A26E);
  const auto routed = run_kernel(p);
  const auto tile = run_tile_kernel(p);
  require(std::memcmp(routed.data(), tile.data(), routed.size() * 2) == 0,
          "large-m route bitwise the tile kernel");
  check_both_oracles(p, routed, "large-m M300xN200xK512");
}

DGPP_TEST(scale_gemm_last_row_preserves_full_product_bits_and_output_bounds) {
  // Cross the GEMV/dense boundary with the production head's K, including
  // row positions on both sides of an MMA tile. Also cover streaming MMA
  // and the ragged-K fallback, with padded input/output strides. A workspace
  // enables split-K only for groups of at most 32 rows: cover both sides of
  // that boundary, including the final group of products above 128 rows.
  void* workspace = nullptr;
  constexpr size_t workspace_bytes = 4u << 20;
  DGPP_CUDA_OK(cudaMalloc(&workspace, workspace_bytes));
  for (const int k : {1000, 2560}) {
    for (const int m : {1, 4, 5, 16, 17, 32, 33, 64, 127, 128, 129, 144, 160, 161, 255, 256, 257}) {
      const Problem p = make_problem(m, 137, k, 0x43A0 + m + k);
      const size_t act_stride = static_cast<size_t>(k) + 8;
      const size_t out_stride = static_cast<size_t>(p.n) + 7;
      uint16_t* act = nullptr;
      uint8_t* w = nullptr;
      float *scales = nullptr, *full = nullptr, *last = nullptr;
      DGPP_CUDA_OK(cudaMallocManaged(&act, m * act_stride * sizeof(uint16_t)));
      DGPP_CUDA_OK(cudaMallocManaged(&w, p.payload.size()));
      DGPP_CUDA_OK(cudaMallocManaged(&scales, p.scales.size() * sizeof(float)));
      DGPP_CUDA_OK(cudaMallocManaged(&full, m * out_stride * sizeof(float)));
      DGPP_CUDA_OK(cudaMallocManaged(&last, m * out_stride * sizeof(float)));
      for (int row = 0; row < m; ++row)
        std::memcpy(act + row * act_stride, p.act.data() + row * k, k * sizeof(uint16_t));
      std::memcpy(w, p.payload.data(), p.payload.size());
      std::memcpy(scales, p.scales.data(), p.scales.size() * sizeof(float));
      bool equal = true, untouched = true;
      for (const int mma_from : {0, 5}) {
        for (const size_t capacity : {size_t{0}, size_t{128}, workspace_bytes}) {
          void* ws = capacity ? workspace : nullptr;
          constexpr float sentinel = -12345.0f;
          std::fill(last, last + m * out_stride, sentinel);
          dgpp::launch_scale_gemm_f32(act, act_stride, w, scales, full, m, p.n, k,
                                      nullptr, out_stride, mma_from, false, ws, capacity);
          dgpp::launch_scale_gemm_f32(act, act_stride, w, scales, last, m, p.n, k,
                                      nullptr, out_stride, mma_from, true, ws, capacity);
          DGPP_CUDA_OK(cudaDeviceSynchronize());
          const size_t offset = static_cast<size_t>(m - 1) * out_stride;
          equal &= std::memcmp(full + offset, last + offset, p.n * sizeof(float)) == 0;
          for (size_t i = 0; i < m * out_stride; ++i)
            if (i < offset || i >= offset + p.n) untouched &= last[i] == sentinel;
          // Compact selection uses a genuinely one-row allocation, including
          // padding sentinels. Memcheck catches any original-offset write.
          float* compact = nullptr;
          DGPP_CUDA_OK(cudaMallocManaged(&compact, out_stride * sizeof(float)));
          for (const int selected : {0, m / 2, m - 1}) {
            std::fill(compact, compact + out_stride, sentinel);
            dgpp::launch_scale_gemm_f32(act, act_stride, w, scales, compact, m, p.n, k,
                                        nullptr, out_stride, mma_from, false, ws, capacity, selected);
            DGPP_CUDA_OK(cudaDeviceSynchronize());
            equal &= std::memcmp(full + static_cast<size_t>(selected) * out_stride,
                                 compact, p.n * sizeof(float)) == 0;
            for (size_t i = p.n; i < out_stride; ++i) untouched &= compact[i] == sentinel;
          }
          DGPP_CUDA_OK(cudaFree(compact));
        }
      }
      DGPP_CUDA_OK(cudaFree(act));
      DGPP_CUDA_OK(cudaFree(w));
      DGPP_CUDA_OK(cudaFree(scales));
      DGPP_CUDA_OK(cudaFree(full));
      DGPP_CUDA_OK(cudaFree(last));
      require(equal, ("last-row head differs from the full product at M=" + std::to_string(m) +
                      " K=" + std::to_string(k)).c_str());
      require(untouched, "last-row head overwrote another row or output padding");
    }
  }
  DGPP_CUDA_OK(cudaFree(workspace));
}

DGPP_TEST(scale_gemm_ragged_tails_match_both_oracles) {
  // Nothing aligned: N=K=1000 (7 full blocks + 104-wide tail on both axes),
  // M=17 (ragged m-tile), and the final BK stage only 8 k-values wide.
  const Problem p = make_problem(17, 1000, 1000, 0xB2);
  const std::vector<uint16_t> got = run_kernel(p);
  check_both_oracles(p, got, "ragged M17xN1000xK1000");
}

DGPP_TEST(scale_gemm_decode_shape_single_row_real_qa_geometry) {
  // The decode-critical shape: M=1 at the real q_a geometry [1536, 4096].
  const Problem p = make_problem(1, 1536, 4096, 0xC3);
  const std::vector<uint16_t> got = run_kernel(p);
  check_both_oracles(p, got, "decode M1xN1536xK4096");
}

DGPP_TEST(scale_gemm_saturation_extremes) {
  // All-max finite payloads (0x7E = 448) with scales that push products to
  // ~1e35 (bf16/fp32 representable), and a subnormal-minimum case.
  {
    Problem p;
    p.m = 9;
    p.n = 256;
    p.k = 256;
    p.act.resize(static_cast<size_t>(p.m) * p.k);
    p.payload.assign(static_cast<size_t>(p.n) * p.k, 0x7E);
    p.scales.assign(2 * 2, 1e30f);
    Rng rng(0xD4);
    fill_act(rng, p.act);
    const std::vector<uint16_t> got = run_kernel(p);
    check_both_oracles(p, got, "saturation 448x1e30");
  }
  {
    Problem p;
    p.m = 9;
    p.n = 256;
    p.k = 256;
    p.act.resize(static_cast<size_t>(p.m) * p.k);
    // 0x01 = min subnormal (2^-9); scales 2^-20 keep products representable
    // but exercise the bottom of both exponent ranges.
    p.payload.assign(static_cast<size_t>(p.n) * p.k, 0x01);
    p.scales.assign(2 * 2, 9.53674316e-07f);  // 2^-20
    Rng rng(0xD5);
    fill_act(rng, p.act);
    const std::vector<uint16_t> got = run_kernel(p);
    check_both_oracles(p, got, "subnormal min");
  }
}

DGPP_TEST(scale_gemm_propagates_nan_exactly) {
  // e4m3fn 0x7F is NaN. Two poisoned weight entries must NaN exactly the
  // output columns they touch — every m, through the fp32 accumulator —
  // and nothing else (zero-filled tiles and finite neighbors stay finite).
  Problem p = make_problem(13, 512, 384, 0xE6);
  p.payload[static_cast<size_t>(5) * p.k + 7] = 0x7F;
  p.payload[static_cast<size_t>(130) * p.k + 300] = 0x7F;
  const std::vector<uint16_t> got = run_kernel(p);

  const auto oracle = gemm_oracle(p.act, p.k, p.payload, p.scales, p.m, p.n,
                                  p.k, /*strict=*/false);
  // NaN pattern must match element-for-element.
  for (int mm = 0; mm < p.m; ++mm) {
    for (int nn = 0; nn < p.n; ++nn) {
      const size_t i = static_cast<size_t>(mm) * p.n + nn;
      const bool want_nan = std::isnan(oracle[i]);
      const bool got_nan = std::isnan(bf16_to_float(got[i]));
      if (nn == 5 || nn == 130) {
        require(want_nan && got_nan, "poisoned column is NaN");
      } else {
        require(!want_nan && !got_nan, "unpoisoned column stays finite");
      }
    }
  }
  // Finite columns still within the semantic budget.
  auto rep = compare_bf16_vs_oracle(got.data(), oracle, 8.0, 2e-2);
  require(rep.nan_pattern_errors == 0, "nan pattern");
  require(rep.mismatches <= static_cast<long>(rep.total) / 20, "nan test mismatches");
  std::printf("[ OK ] nan propagation: columns 5 and 130 NaN, others finite\n");
}

DGPP_TEST(scale_gemm_degenerate_k_zeroes_output) {
  Problem p = make_problem(4, 64, 0, 0xF7);
  p.payload.clear();
  p.scales.clear();
  p.act.clear();
  p.act.resize(4);  // act unused for k=0
  uint16_t* act = nullptr;
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act, 8));
  DGPP_CUDA_OK(cudaMallocManaged(&out, 4 * 64 * 2));
  std::memset(out, 0xFF, 4 * 64 * 2);  // garbage
  dgpp::launch_scale_gemm_bf16(act, 0, reinterpret_cast<const uint8_t*>(act),
                               reinterpret_cast<const float*>(act), out, 4,
                               64, 0, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (size_t i = 0; i < 4 * 64; ++i)
    if (out[i] != 0) throw std::runtime_error("k=0 must zero outputs");
  DGPP_CUDA_OK(cudaFree(act));
  DGPP_CUDA_OK(cudaFree(out));
}

// ---- the m<=4 GEMV path (fp8_gemv.cuh) ------------------------------------

__global__ void e4m3_convert_all_codes_kernel(float* hw, float* ref) {
  // GIVEN every byte code, converted by the GEMV's hardware path and by the
  // dequant bridge's reference function.
  const int v = threadIdx.x;
  const float2 pair = dgpp::fp8_gemv::e4m3x2_to_float2(
      static_cast<uint16_t>(v | (v << 8)));
  hw[v] = pair.x;
  hw[256 + v] = pair.y;  // the high byte lands in .y
  ref[v] = dgpp::fp8_e4m3_bits_to_float(static_cast<uint8_t>(v));
}

DGPP_TEST(fp8_gemv_hardware_conversion_matches_bridge_on_every_code) {
  float* hw = nullptr;
  float* ref = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&hw, 512 * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&ref, 256 * 4));
  e4m3_convert_all_codes_kernel<<<1, 256>>>(hw, ref);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  // THEN both halves agree with the bridge bitwise on every finite code and
  // are NaN exactly where the bridge is (0x7F / 0xFF).
  for (int v = 0; v < 256; ++v) {
    for (int half = 0; half < 2; ++half) {
      const float got = hw[half * 256 + v];
      if (std::isnan(ref[v])) {
        require(std::isnan(got), "NaN codes convert to NaN");
      } else {
        require(std::memcmp(&got, &ref[v], 4) == 0,
                "finite e4m3 codes convert bit-exactly");
      }
    }
  }
  DGPP_CUDA_OK(cudaFree(hw));
  DGPP_CUDA_OK(cudaFree(ref));
  std::printf("[ OK ] e4m3 hardware conversion == bridge on all 256 codes\n");
}

// The multi-problem GEMV: several [n_i, k] matrices against
// the same rows in one launch — every problem's output bitwise its own
// launch_scale_gemm_bf16, at one to four rows (one chunk) and six (two).
DGPP_TEST(scale_gemm_gemv_multi_is_bitwise_the_single_launches) {
  const int k = 1008;
  const int ns[3] = {100, 24, 8};
  for (const int rows : {1, 2, 4, 6}) {
    Problem ps[3];
    for (int i = 0; i < 3; ++i) ps[i] = make_problem(rows, ns[i], k, 0xF0 + i + rows);
    // The problems share the activations: problem 0's rows for every one.
    for (int i = 1; i < 3; ++i) ps[i].act = ps[0].act;
    std::vector<uint16_t> single[3];
    for (int i = 0; i < 3; ++i) single[i] = run_kernel(ps[i]);
    uint16_t* act = nullptr;
    uint8_t* w[3] = {};
    float* sc[3] = {};
    uint16_t* out[3] = {};
    DGPP_CUDA_OK(cudaMallocManaged(&act, ps[0].act.size() * 2));
    std::memcpy(act, ps[0].act.data(), ps[0].act.size() * 2);
    dgpp::Fp8GemvProblem probs[3];
    for (int i = 0; i < 3; ++i) {
      DGPP_CUDA_OK(cudaMallocManaged(&w[i], ps[i].payload.size()));
      DGPP_CUDA_OK(cudaMallocManaged(&sc[i], ps[i].scales.size() * 4));
      DGPP_CUDA_OK(cudaMallocManaged(&out[i], static_cast<size_t>(rows) * ns[i] * 2));
      std::memcpy(w[i], ps[i].payload.data(), ps[i].payload.size());
      std::memcpy(sc[i], ps[i].scales.data(), ps[i].scales.size() * 4);
      probs[i].payload = w[i];
      probs[i].scales = sc[i];
      probs[i].out = out[i];
      probs[i].n = ns[i];
    }
    dgpp::launch_scale_gemv_multi_bf16(probs, 3, act, static_cast<size_t>(k), rows, k, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    for (int i = 0; i < 3; ++i) {
      require(std::memcmp(out[i], single[i].data(), single[i].size() * 2) == 0,
              "multi GEMV output differs from the single launch");
      DGPP_CUDA_OK(cudaFree(w[i]));
      DGPP_CUDA_OK(cudaFree(sc[i]));
      DGPP_CUDA_OK(cudaFree(out[i]));
    }
    DGPP_CUDA_OK(cudaFree(act));
    std::printf("[ OK ] the multi-problem fp8 GEMV is bitwise the single launches at %d rows\n", rows);
  }
}

DGPP_TEST(scale_gemm_gemv_path_ragged_n_and_k_match_both_oracles) {
  // GIVEN m=1 with k a multiple of 16 but not of 128 (the last chunk sits
  // in a partial scale block) and n not a multiple of the 8-row block:
  const Problem p = make_problem(1, 1003, 1008, 0xE1);
  const std::vector<uint16_t> got = run_kernel(p);
  check_both_oracles(p, got, "gemv M1xN1003xK1008");
  // AND a k shorter than one warp span (512 bytes): every lane but the
  // first few has no chunk at all.
  const Problem q = make_problem(1, 200, 48, 0xE2);
  check_both_oracles(q, run_kernel(q), "gemv M1xN200xK48");
  // Determinism.
  const std::vector<uint16_t> again = run_kernel(p);
  require(std::memcmp(got.data(), again.data(), got.size() * 2) == 0,
          "gemv second run bitwise identical");
}

DGPP_TEST(scale_gemm_gemv_rows_are_independent_of_row_count) {
  // GIVEN the same activation rows run at m=1 each and at m=3 together,
  // THEN every row's bits agree: each row's chain is the same sequence of
  // FMAs whatever the row count (the property glm_moe_test's slot-vs-host
  // bitwise pin relies on).
  const Problem p3 = make_problem(3, 520, 4096, 0xE3);
  const std::vector<uint16_t> got3 = run_kernel(p3);
  check_both_oracles(p3, got3, "gemv M3xN520xK4096");
  for (int r = 0; r < 3; ++r) {
    Problem p1;
    p1.m = 1;
    p1.n = p3.n;
    p1.k = p3.k;
    p1.act.assign(p3.act.begin() + static_cast<long>(r) * p3.k,
                  p3.act.begin() + static_cast<long>(r + 1) * p3.k);
    p1.payload = p3.payload;
    p1.scales = p3.scales;
    const std::vector<uint16_t> got1 = run_kernel(p1);
    require(std::memcmp(got1.data(), got3.data() + static_cast<size_t>(r) * p3.n,
                        static_cast<size_t>(p3.n) * 2) == 0,
            "row bits independent of m");
  }
  // AND m=4 (the largest GEMV row count) still matches the oracles.
  const Problem p4 = make_problem(4, 264, 2048, 0xE4);
  check_both_oracles(p4, run_kernel(p4), "gemv M4xN264xK2048");
  // AND m=8 at real hidden width must take row-independent GEMV chunks,
  // not the numerically different tile kernel. Check both epilogues against
  // eight scalar invocations so chunk boundaries and output offsets are
  // covered directly.
  const Problem p8 = make_problem(8, 136, 4096, 0xE8);
  const std::vector<uint16_t> got8 = run_kernel(p8);
  const std::vector<float> got8f = run_kernel_f32(p8);
  check_both_oracles(p8, got8, "gemv chunked M8xN136xK4096");
  for (int r = 0; r < p8.m; ++r) {
    Problem p1;
    p1.m = 1;
    p1.n = p8.n;
    p1.k = p8.k;
    p1.act.assign(p8.act.begin() + static_cast<long>(r) * p8.k,
                  p8.act.begin() + static_cast<long>(r + 1) * p8.k);
    p1.payload = p8.payload;
    p1.scales = p8.scales;
    const std::vector<uint16_t> got1 = run_kernel(p1);
    const std::vector<float> got1f = run_kernel_f32(p1);
    require(std::memcmp(got1.data(),
                        got8.data() + static_cast<size_t>(r) * p8.n,
                        static_cast<size_t>(p8.n) * sizeof(uint16_t)) == 0,
            "chunked fp8/bf16 row bits independent of m");
    require(std::memcmp(got1f.data(),
                        got8f.data() + static_cast<size_t>(r) * p8.n,
                        static_cast<size_t>(p8.n) * sizeof(float)) == 0,
            "chunked fp8/f32 row bits independent of m");
  }
  std::printf("[ OK ] gemv rows independent of m through m=8 chunks\n");
}

DGPP_TEST(scale_gemm_gemv_path_propagates_nan_exactly) {
  // GIVEN m=1 with two poisoned weights: exactly their output columns NaN.
  Problem p = make_problem(1, 300, 1024, 0xE5);
  p.payload[static_cast<size_t>(9) * p.k + 700] = 0x7F;
  p.payload[static_cast<size_t>(250) * p.k + 3] = 0xFF;
  const std::vector<uint16_t> got = run_kernel(p);
  for (int nn = 0; nn < p.n; ++nn) {
    const bool got_nan = std::isnan(bf16_to_float(got[static_cast<size_t>(nn)]));
    if (nn == 9 || nn == 250)
      require(got_nan, "poisoned column is NaN (gemv)");
    else
      require(!got_nan, "unpoisoned column stays finite (gemv)");
  }
  std::printf("[ OK ] gemv nan propagation: columns 9 and 250 NaN\n");
}

// Real-checkpoint slice parity lives in scale_gemm_checkpoint.cpp (host-only
// TU: minijson does not mix with nvcc).
int run_scale_gemm_checkpoint_parity(const char* checkpoint_dir);

int main(int argc, char** argv) {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  const int rc = dgpp::test::run_all();
  if (rc != 0) return rc;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--checkpoint-dir") == 0 && i + 1 < argc) {
      const int prc = run_scale_gemm_checkpoint_parity(argv[i + 1]);
      if (prc != 0) return prc;
    }
  }
  return 0;
}
