// Independent FP64 oracle for exact offset-code x BF16-scale weights.
// Also pin grouped/dense arithmetic, row maps, ragged tiles, output strides,
// BF16 epilogues, NaN propagation and cold CUDA graph capture.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/packq_gemm.hpp"
#include "kernels/packq_gemv.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {
using scale_gemm_test::require;
template <typename T>
struct Buffer {
  T* p = nullptr;
  explicit Buffer(size_t n) { DGPP_CUDA_OK(cudaMallocManaged(&p, n * sizeof(T))); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { cudaFree(p); }
};

struct Matrix {
  int bits, n, k, sf;
  Buffer<uint32_t> packed;
  Buffer<uint16_t> scales;
  int group() const { return dgpp::packed_scale_group(sf); }
  Matrix(int bits, int n, int k, uint64_t seed, int sf = 0)
      : bits(bits),
        n(n),
        k(k),
        sf(sf),
        packed(static_cast<size_t>(n) * k * bits / 32),
        scales(static_cast<size_t>(n) * k / dgpp::packed_scale_group(sf)) {
    scale_gemm_test::Rng rng(seed);
    for (size_t i = 0; i < static_cast<size_t>(n) * k * bits / 32; ++i)
      packed.p[i] = static_cast<uint32_t>(rng.next());
    for (size_t i = 0; i < static_cast<size_t>(n) * k / group(); ++i) {
      const float v = static_cast<float>(std::exp2(rng.unit() * 4) * 0.0031);
      scales.p[i] = sf == 0 ? dgpp::float_to_bf16_bits(v) : dgpp::float_to_fp16_bits(v);
    }
    scales.p[0] = 0;  // zero scales must also have exact semantics
  }
  dgpp::GlmPackedMatrix view() const { return {packed.p, scales.p, n, k, bits, sf}; }
  double value(int row, int col) const {
    const int per = 32 / bits;
    const int g = group();
    const uint32_t w = packed.p[static_cast<size_t>(row) * (k / per) + col / per];
    const int code =
        static_cast<int>((w >> (bits * (col % per))) & ((1u << bits) - 1)) - (1 << (bits - 1));
    return code * static_cast<double>(dgpp::packed_scale_to_float(
                      scales.p[static_cast<size_t>(row) * (k / g) + col / g], sf));
  }
};

void fill(uint16_t* a, size_t count, uint64_t seed) {
  scale_gemm_test::Rng rng(seed);
  for (size_t i = 0; i < count; ++i)
    a[i] = dgpp::float_to_bf16_bits(static_cast<float>(rng.unit() * std::exp2(rng.unit() * 3)));
}

void oracle_check(const Matrix& w, const uint16_t* a, int stride, const float* got, int os, int m,
                  const float* baseline = nullptr) {
  double error2 = 0, norm2 = 0, max_abs = 0, max_error = 0;
  double baseline_error2 = 0, bf16_error2 = 0, baseline_bf16_error2 = 0;
  int better = 0, worse = 0, changed_bf16 = 0;
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < w.n; ++col) {
      double expected = 0;
      for (int kk = 0; kk < w.k; ++kk)
        expected +=
            dgpp::bf16_bits_to_float(a[static_cast<size_t>(row) * stride + kk]) * w.value(col, kk);
      const double actual = got[static_cast<size_t>(row) * os + col];
      if (std::isnan(expected)) {
        require(std::isnan(actual), "NaN scale must poison exactly its output column");
        continue;
      }
      require(std::isfinite(actual), "finite oracle must produce finite output");
      const double e = std::abs(actual - expected);
      error2 += e * e;
      norm2 += expected * expected;
      max_abs = std::max(max_abs, std::abs(expected));
      max_error = std::max(max_error, e);
      if (baseline) {
        const double old = baseline[static_cast<size_t>(row) * w.n + col];
        require(std::isfinite(old), "finite oracle must produce finite GEMV output");
        baseline_error2 += (old - expected) * (old - expected);
        const uint16_t b = dgpp::float_to_bf16_bits(actual);
        const uint16_t ob = dgpp::float_to_bf16_bits(old);
        const double be = dgpp::bf16_bits_to_float(b) - expected;
        const double obe = dgpp::bf16_bits_to_float(ob) - expected;
        bf16_error2 += be * be;
        baseline_bf16_error2 += obe * obe;
        better += std::abs(be) < std::abs(obe);
        worse += std::abs(be) > std::abs(obe);
        changed_bf16 += b != ob;
      }
    }
  const double relative = std::sqrt(error2 / std::max(norm2, 1e-30));
  std::printf("int%d%s M%d N%d K%d: fp32 l2_rel %.3g max_error/max_abs %.3g\n", w.bits,
              w.sf ? " g128/f16" : "", m, w.n, w.k, relative, max_error / std::max(max_abs, 1e-30));
  require(relative < 5e-6 && max_error < std::max(1e-8, max_abs * 4e-5),
          "packed GEMM differs from exact FP64 oracle");
  if (baseline) {
    const double old_relative = std::sqrt(baseline_error2 / std::max(norm2, 1e-30));
    std::printf(
        "  GEMV fp32 l2_rel %.3g; BF16 vs FP64 GEMV %.9g GEMM %.9g; "
        "changed %d, GEMM closer %d farther %d (remaining changes equidistant)\n",
        old_relative, std::sqrt(baseline_bf16_error2 / std::max(norm2, 1e-30)),
        std::sqrt(bf16_error2 / std::max(norm2, 1e-30)), changed_bf16, better, worse);
    require(old_relative < 5e-6, "packed GEMV differs from exact FP64 oracle");
  }
}
}  // namespace

DGPP_TEST(packq_gemm_exact_weights_and_ragged_tiles) {
  // Format 1 (f16 per 128) at the AutoRound hybrid's widths and the small
  // 128-multiples: the tile's scale is the group of its 64-deep step.
  const std::vector<std::vector<int>> shapes0 = {{1, 1, 64},     {17, 70, 192},  {33, 129, 512},
                                                 {65, 35, 6144}, {32, 67, 2048}, {3, 17, 16384}};
  const std::vector<std::vector<int>> shapes1 = {{1, 1, 128},   {17, 70, 640},   {33, 129, 2560},
                                                 {32, 67, 256}, {65, 35, 1024},  {3, 17, 2560}};
  for (int sf : {0, 1})
  for (int bits : {4, 8})
    for (const auto& shape : sf == 0 ? shapes0 : shapes1) {
      const int m = shape[0], n = shape[1], k = shape[2];
      Matrix w(bits, n, k, 917 + bits + k, sf);
      const int stride = k + (m % 2 ? 3 : 0);
      Buffer<uint16_t> storage(static_cast<size_t>(m) * stride + 1), b(static_cast<size_t>(m) * n);
      uint16_t* a = storage.p + 1;  // explicitly unaligned base
      fill(a, static_cast<size_t>(m) * stride, 519);
      Buffer<float> out(static_cast<size_t>(m) * n), baseline(static_cast<size_t>(m) * n);
      Buffer<uint16_t> aligned(static_cast<size_t>(m) * k);
      for (int row = 0; row < m; ++row)
        std::memcpy(aligned.p + static_cast<size_t>(row) * k, a + static_cast<size_t>(row) * stride,
                    k * sizeof(uint16_t));
      dgpp::launch_packq_gemm_f32(a, stride, w.view(), out.p, m, n, k, nullptr);
      dgpp::launch_packq_gemm_bf16(a, stride, w.view(), b.p, m, n, k, nullptr);
      dgpp::launch_packq_gemv_f32(aligned.p, k, w.view(), baseline.p, m, n, k, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      oracle_check(w, a, stride, out.p, n, m, baseline.p);
      for (int i = 0; i < m * n; ++i)
        require(b.p[i] == dgpp::float_to_bf16_bits(out.p[i]),
                "BF16 epilogue must round FP32 output exactly");
    }
}

DGPP_TEST(packq_gemm_grouped_maps_padding_and_graph) {
  constexpr int n = 70, os = 77, tokens = 137, ns = 9;
  for (int sf : {0, 1})
  for (int bits : {4, 8}) {
    const int k = sf == 0 ? 192 : 640;
    Matrix w(bits, n, k, 611 + bits, sf);
    w.scales.p[9 * (k / w.group()) + 1] = sf == 0 ? 0x7fc0 : 0x7e00;
    Matrix other(bits, n, k, 975 + bits, sf);
    Buffer<dgpp::MoeExpertView> views(ns * 3);
    Buffer<dgpp::MoeSegment> segs(ns);
    const int counts[ns] = {0, 1, 16, 17, 31, 32, 33, 65, 130};
    int total = 0;
    for (int e = 0; e < ns; ++e) {
      segs.p[e] = {total, counts[e], e};
      total += counts[e];
      for (int which = 0; which < 3; ++which)
        views.p[e * 3 + which] = dgpp::MoeExpertView::of(which == 1 ? w.view() : other.view());
    }
    Buffer<int32_t> rows(total);
    Buffer<uint16_t> a(tokens * k), gathered(static_cast<size_t>(total) * k);
    fill(a.p, tokens * k, 617);
    for (int r = 0; r < total; ++r) {
      rows.p[r] = (r * 19) % tokens;  // duplicates and nonmonotonic input rows
      std::memcpy(gathered.p + static_cast<size_t>(r) * k, a.p + rows.p[r] * k, k * 2);
    }
    Buffer<float> out(static_cast<size_t>(total) * os), ref(static_cast<size_t>(total) * n);
    Buffer<uint16_t> bf(static_cast<size_t>(total) * os);
    std::fill(out.p, out.p + static_cast<size_t>(total) * os, -12345.f);
    std::fill(bf.p, bf.p + static_cast<size_t>(total) * os, uint16_t{0x1234});
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, tokens, views.p, 1, out.p, os, n, k,
                                           bits, stream, rows.p, sf);
    dgpp::launch_moe_grouped_mma_packq_bf16(a.p, k, segs.p, ns, tokens, views.p, 1, bf.p, os, n, k,
                                            bits, stream, rows.p, sf);
    DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int repeat = 0; repeat < 2; ++repeat) DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
    dgpp::launch_packq_gemm_f32(gathered.p, k, w.view(), ref.p, total, n, k, stream);
    // The compact tile list (glm_moe_launch.hpp): the same tiles from a 1-D
    // grid, bitwise the segment-major launch; its entries as the host
    // expects them (one 64-row tile per started 64 rows, segment order).
    const int tile_cap = dgpp::moe_tile_list_capacity(ns, total, dgpp::kPackqGemmWideRows);
    Buffer<dgpp::MoeTile> tiles(tile_cap);
    Buffer<int32_t> tile_count(1);
    Buffer<float> listed(static_cast<size_t>(total) * os);
    std::fill(listed.p, listed.p + static_cast<size_t>(total) * os, -12345.f);
    dgpp::launch_moe_tile_list(segs.p, ns, dgpp::kPackqGemmWideRows, tiles.p, tile_count.p, stream);
    if (bits == 4)  // the wide kernel takes the list; int8 rows keep the narrow grid
      dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, /*max_rows=*/1, views.p, 1, listed.p, os,
                                             n, k, bits, stream, rows.p, sf, /*variant=*/1, tiles.p,
                                             tile_count.p, tile_cap);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    {
      int want = 0;
      for (int e = 0; e < ns; ++e) {
        for (int j = 0; j < counts[e]; j += dgpp::kPackqGemmWideRows) {
          require(want < tile_cap, "tile list within its capacity");
          require(tiles.p[want].seg == e && tiles.p[want].m0 == j, "tile list entry");
          ++want;
        }
      }
      require(tile_count.p[0] == want, "tile list count");
      if (bits == 4)
        require(std::memcmp(listed.p, out.p, static_cast<size_t>(total) * os * 4) == 0,
                "the listed launch must be bitwise the segment-major one");
    }
    if (bits == 4) {
      // The paired launch: projection 1 (w) and projection 0 (other) in one
      // launch, each bitwise its own launch.
      Buffer<uint16_t> pa(static_cast<size_t>(total) * os), pb(static_cast<size_t>(total) * os);
      Buffer<uint16_t> sb(static_cast<size_t>(total) * os);
      std::fill(pa.p, pa.p + static_cast<size_t>(total) * os, uint16_t{0x1234});
      std::fill(pb.p, pb.p + static_cast<size_t>(total) * os, uint16_t{0x1234});
      std::fill(sb.p, sb.p + static_cast<size_t>(total) * os, uint16_t{0x1234});
      dgpp::launch_moe_grouped_mma_packq_bf16(a.p, k, segs.p, ns, tokens, views.p, 1, pa.p, os, n, k, bits,
                                              stream, rows.p, sf, /*variant=*/1, tiles.p, tile_count.p, tile_cap,
                                              pb.p, /*which2=*/0);
      dgpp::launch_moe_grouped_mma_packq_bf16(a.p, k, segs.p, ns, tokens, views.p, 0, sb.p, os, n, k, bits,
                                              stream, rows.p, sf, /*variant=*/1);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      require(std::memcmp(pa.p, bf.p, static_cast<size_t>(total) * os * 2) == 0,
              "the paired launch's first projection must be bitwise its own launch");
      require(std::memcmp(pb.p, sb.p, static_cast<size_t>(total) * os * 2) == 0,
              "the paired launch's second projection must be bitwise its own launch");
    }
    if (bits == 4) {
      // The register-decode three-stage form (variant 2), listed and
      // segment-major: bitwise the decoded-tile kernel.
      Buffer<float> reg(static_cast<size_t>(total) * os);
      std::fill(reg.p, reg.p + static_cast<size_t>(total) * os, -12345.f);
      dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, /*max_rows=*/1, views.p, 1, reg.p, os, n, k,
                                             bits, stream, rows.p, sf, /*variant=*/2, tiles.p, tile_count.p,
                                             tile_cap);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      require(std::memcmp(reg.p, out.p, static_cast<size_t>(total) * os * 4) == 0,
              "the register-decode kernel must be bitwise the decoded-tile one (listed)");
      std::fill(reg.p, reg.p + static_cast<size_t>(total) * os, -12345.f);
      dgpp::launch_moe_grouped_mma_packq_f32(a.p, k, segs.p, ns, tokens, views.p, 1, reg.p, os, n, k, bits,
                                             stream, rows.p, sf, /*variant=*/2);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      require(std::memcmp(reg.p, out.p, static_cast<size_t>(total) * os * 4) == 0,
              "the register-decode kernel must be bitwise the decoded-tile one (segment-major)");
    }
    oracle_check(w, gathered.p, k, out.p, os, total);
    for (int row = 0; row < total; ++row) {
      require(std::memcmp(out.p + row * os, ref.p + row * n, n * 4) == 0,
              "grouped mapped FP32 output must be bitwise dense GEMM");
      for (int col = 0; col < n; ++col) {
        const uint16_t want = dgpp::float_to_bf16_bits(out.p[row * os + col]);
        require(bf.p[row * os + col] == want ||
                    (std::isnan(dgpp::bf16_bits_to_float(want)) &&
                     std::isnan(dgpp::bf16_bits_to_float(bf.p[row * os + col]))),
                "grouped BF16 epilogue");
      }
      for (int col = n; col < os; ++col)
        require(out.p[row * os + col] == -12345.f && bf.p[row * os + col] == 0x1234,
                "grouped output padding must remain untouched");
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
  }
}

DGPP_TEST(packq_gemm_rejects_invalid_geometry) {
  Matrix w(4, 16, 64, 919);
  Buffer<uint16_t> a(64), out(16);
  auto rejects = [&](dgpp::GlmPackedMatrix matrix, int stride, int n, int k) {
    try {
      dgpp::launch_packq_gemm_bf16(a.p, stride, matrix, out.p, 1, n, k, nullptr);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  auto matrix = w.view();
  require(rejects(matrix, 63, 16, 64), "short activation stride");
  require(rejects(matrix, 64, 17, 64), "N exceeds matrix rows");
  require(rejects(matrix, 64, 16, 32), "K differs from matrix columns");
  matrix.bits = 3;
  require(rejects(matrix, 64, 16, 64), "invalid code width");
  matrix = w.view();
  matrix.packed += 1;
  require(rejects(matrix, 64, 16, 64), "unaligned packed row");
  matrix = w.view();
  matrix.scale_fmt = 1;
  require(rejects(matrix, 64, 16, 64), "g128 format with K=64 (not a multiple of 128)");
  matrix.scale_fmt = 2;
  require(rejects(matrix, 64, 16, 64), "unknown scale format");
  Matrix g(4, 16, 128, 921, 1);
  Buffer<uint16_t> a128(128);
  bool ok = true;
  try {
    dgpp::launch_packq_gemm_bf16(a128.p, 128, g.view(), out.p, 1, 16, 128, nullptr);
  } catch (const std::invalid_argument&) {
    ok = false;
  }
  require(ok, "g128 format with K=128 accepted");
}

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
