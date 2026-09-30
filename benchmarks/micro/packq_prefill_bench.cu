// Compare packed prefill GEMV and tensor-core kernels on idle hardware.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/packq_gemm.hpp"
#include "kernels/packq_gemv.hpp"

namespace {
template <typename T>
struct Buffer {
  T* p = nullptr;
  explicit Buffer(size_t n) { DGPP_CUDA_OK(cudaMallocManaged(&p, n * sizeof(T))); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { cudaFree(p); }
};
double time_ms(const std::function<void()>& launch, int iters) {
  for (int i = 0; i < 3; ++i) launch();
  cudaEvent_t a, b;
  DGPP_CUDA_OK(cudaEventCreate(&a));
  DGPP_CUDA_OK(cudaEventCreate(&b));
  DGPP_CUDA_OK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) launch();
  DGPP_CUDA_OK(cudaEventRecord(b));
  DGPP_CUDA_OK(cudaEventSynchronize(b));
  float elapsed;
  DGPP_CUDA_OK(cudaEventElapsedTime(&elapsed, a, b));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return elapsed / iters;
}
template <typename T>
double as_float(T x) {
  if constexpr (std::is_same_v<T, float>)
    return x;
  else
    return dgpp::bf16_bits_to_float(x);
}

template <typename Out>
void run(int m, int n, int k, int bits, int experts, int top_k, int iters, bool hot, int sf, int variant,
         const std::string& baseline_mode, const std::string& dist, bool tiles_on, bool gather,
         bool same_weights, bool shuffle_rows, bool device_copy, bool pair) {
  using namespace dgpp;
  std::mt19937 rng(0x6144512);
  const int group = sf == 0 ? 64 : 128;
  const size_t words = static_cast<size_t>(n) * k * bits / 32;
  const size_t factors = static_cast<size_t>(n) * k / group;
  Buffer<uint32_t> packed(words * experts);
  Buffer<uint16_t> scales(factors * experts);
  for (size_t i = 0; i < words * experts; ++i) packed.p[i] = rng();
  for (size_t i = 0; i < factors * experts; ++i)
    scales.p[i] = sf == 0 ? float_to_bf16_bits(0.0001f + (rng() % 1000) * 0.000003f)
                          : float_to_fp16_bits(0.0001f + (rng() % 1000) * 0.000003f);
  Buffer<MoeExpertView> views(experts * 3);
  Buffer<MoeSegment> segs(experts);
  std::vector<int> counts(experts);
  // "zipf": expert e drawn with probability ~ 1 / (1 + e)^0.8 — the real
  // router's skew in shape (a few hot experts, most under 64 rows a chunk).
  std::vector<double> zipf_cdf;
  if (dist == "zipf") {
    double acc = 0;
    for (int e = 0; e < experts; ++e) { acc += 1.0 / std::pow(1.0 + e, 0.8); zipf_cdf.push_back(acc); }
    for (auto& v : zipf_cdf) v /= acc;
  }
  std::mt19937 rng_route(7);
  for (int t = 0; t < m; ++t) {
    std::vector<int> chosen;
    while (static_cast<int>(chosen.size()) < top_k) {
      int e;
      if (dist == "zipf") {
        const double u = (rng_route() % 1000000) / 1000000.0;
        e = static_cast<int>(std::lower_bound(zipf_cdf.begin(), zipf_cdf.end(), u) - zipf_cdf.begin());
        if (e >= experts) e = experts - 1;
      } else {
        e = rng_route() % (hot ? top_k : experts);
      }
      if (std::find(chosen.begin(), chosen.end(), e) != chosen.end()) continue;
      chosen.push_back(e);
      ++counts[e];
    }
  }
  int rows = 0, largest = 0, nonempty = 0, small16 = 0, small32 = 0, small64 = 0;
  for (int e = 0; e < experts; ++e) {
    if (counts[e] > 0 && counts[e] <= 16) ++small16;
    else if (counts[e] > 16 && counts[e] <= 32) ++small32;
    else if (counts[e] > 32 && counts[e] <= 64) ++small64;
    // --same-weights 1: every segment reads expert 0's matrix (an L2-resident
    // weight set under any routing; the products are wrong, the traffic is
    // the experiment's).
    const int we = same_weights ? 0 : e;
    const GlmPackedMatrix w{packed.p + we * words, scales.p + we * factors, n, k, bits, sf};
    views.p[e * 3] = MoeExpertView::of(w);
    segs.p[e] = {rows, counts[e], e};
    rows += counts[e];
    largest = std::max(largest, counts[e]);
    nonempty += counts[e] > 0;
  }
  // --gather 1: A holds the m token rows and act_rows maps every gathered
  // row to its token, as the layer reads the hidden rows through the row
  // map (each token's row is read by its top_k experts' tiles); --gather 0
  // (the default, the earlier records) gives every gathered row its own A.
  const size_t a_rows = gather ? static_cast<size_t>(m) : static_cast<size_t>(rows);
  Buffer<uint16_t> a(a_rows * k);
  for (size_t i = 0; i < a_rows * k; ++i)
    a.p[i] = float_to_bf16_bits((int(rng() % 2001) - 1000) * 0.001f);
  Buffer<int32_t> act_rows(static_cast<size_t>(std::max(rows, 1)));
  Buffer<uint16_t> a_gathered(gather ? static_cast<size_t>(rows) * k : 1);
  if (gather) {
    // Segment order: expert-major, tokens ascending within an expert (the
    // segmentation kernel's order).
    std::vector<int> fill(experts);
    for (int e = 0; e < experts; ++e) fill[e] = segs.p[e].row0;
    std::mt19937 rng2(7);
    for (int t = 0; t < m; ++t) {
      std::vector<int> chosen;
      while (static_cast<int>(chosen.size()) < top_k) {
        int e;
        if (dist == "zipf") {
          const double u = (rng2() % 1000000) / 1000000.0;
          e = static_cast<int>(std::lower_bound(zipf_cdf.begin(), zipf_cdf.end(), u) - zipf_cdf.begin());
          if (e >= experts) e = experts - 1;
        } else {
          e = rng2() % (hot ? top_k : experts);
        }
        if (std::find(chosen.begin(), chosen.end(), e) != chosen.end()) continue;
        chosen.push_back(e);
      }
      for (int e : chosen) act_rows.p[fill[e]++] = t;
    }
    // --shuffle-rows 1: the row map of every segment in a random order (the
    // hot routing's tiles then gather scattered token rows as the real
    // chunk's small segments do).
    if (shuffle_rows)
      for (int e = 0; e < experts; ++e) {
        const int r0 = segs.p[e].row0, cnt = segs.p[e].rows;
        for (int i = cnt - 1; i > 0; --i) std::swap(act_rows.p[r0 + i], act_rows.p[r0 + static_cast<int>(rng2() % (i + 1))]);
      }
    for (int r = 0; r < rows; ++r)
      std::memcpy(a_gathered.p + static_cast<size_t>(r) * k, a.p + static_cast<size_t>(act_rows.p[r]) * k, k * 2);
  }
  Buffer<Out> baseline(static_cast<size_t>(rows) * n), candidate(static_cast<size_t>(rows) * n);
  // --device-copy 1: the timed launches read cudaMalloc copies of the
  // weights, scales, activations, row map, segments and views (the layer's
  // memory), not the managed buffers the host filled.
  struct DeviceCopy {
    void* p = nullptr;
    DeviceCopy(const void* src, size_t bytes) {
      DGPP_CUDA_OK(cudaMalloc(&p, bytes));
      DGPP_CUDA_OK(cudaMemcpy(p, src, bytes, cudaMemcpyDefault));
    }
    ~DeviceCopy() { cudaFree(p); }
  };
  std::vector<std::unique_ptr<DeviceCopy>> copies;
  const uint32_t* packed_p = packed.p;
  const uint16_t* scales_p = scales.p;
  const uint16_t* a_p = a.p;
  const int32_t* act_rows_p = act_rows.p;
  MoeSegment* segs_p = segs.p;
  MoeExpertView* views_p = views.p;
  if (device_copy) {
    auto copy = [&](const void* src, size_t bytes) {
      copies.emplace_back(new DeviceCopy(src, bytes));
      return copies.back()->p;
    };
    packed_p = static_cast<const uint32_t*>(copy(packed.p, words * experts * 4));
    scales_p = static_cast<const uint16_t*>(copy(scales.p, factors * experts * 2));
    a_p = static_cast<const uint16_t*>(copy(a.p, a_rows * k * 2));
    act_rows_p = static_cast<const int32_t*>(copy(act_rows.p, static_cast<size_t>(std::max(rows, 1)) * 4));
    segs_p = static_cast<MoeSegment*>(copy(segs.p, static_cast<size_t>(experts) * sizeof(MoeSegment)));
    for (int e = 0; e < experts; ++e) {
      const int we = same_weights ? 0 : e;
      const GlmPackedMatrix w{packed_p + e * 0 + we * words, scales_p + we * factors, n, k, bits, sf};
      views.p[e * 3] = MoeExpertView::of(w);
    }
    views_p = static_cast<MoeExpertView*>(copy(views.p, static_cast<size_t>(experts) * 3 * sizeof(MoeExpertView)));
  }
  const GlmPackedMatrix dense{packed_p, scales_p, n, k, bits, sf};
  // The compact tile list for the candidate (--tiles 1; the segment-major
  // grid over the longest segment with --tiles 0): built once here, as the
  // layer builds it once per chunk after the segmentation.
  const int tile_cap = moe_tile_list_capacity(experts, rows, kPackqGemmWideRows);
  Buffer<MoeTile> tiles(static_cast<size_t>(tile_cap));
  Buffer<int32_t> tile_count(1);
  if (tiles_on && experts > 1) {
    launch_moe_tile_list(segs_p, experts, kPackqGemmWideRows, tiles.p, tile_count.p, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
  }
  const bool listed = tiles_on && experts > 1;
  // --pair 1 (bf16, timing only): the candidate also runs a second
  // projection of the same view in the launch (the gate+up form); the
  // second output is not compared.
  Buffer<Out> pair_out(pair ? static_cast<size_t>(rows) * n : 1);
  // The baseline: the GEMV core, or (--baseline mma0) the narrow tensor-core
  // kernel; the candidate: the tensor-core kernel at --variant (-1 default).
  auto launch = [&](bool mma) {
    Out* out = mma ? candidate.p : baseline.p;
    const bool tc = mma || baseline_mode == "mma0";
    const int var = mma ? variant : 0;
    if (experts == 1) {
      if constexpr (std::is_same_v<Out, float>) {
        if (tc) launch_packq_gemm_f32_variant(a_p, k, dense, out, m, n, k, nullptr, var);
        else launch_packq_gemv_f32(a_p, k, dense, out, m, n, k, nullptr);
      } else {
        if (tc) launch_packq_gemm_bf16_variant(a_p, k, dense, out, m, n, k, nullptr, var);
        else launch_packq_gemv_bf16(a_p, k, dense, out, m, n, k, nullptr);
      }
    } else if constexpr (std::is_same_v<Out, float>) {
      if (tc)
        launch_moe_grouped_mma_packq_f32(a_p, k, segs_p, experts, m, views_p, 0, out, n, n, k, bits,
                                         nullptr, gather ? act_rows_p : nullptr, sf, var,
                                         mma && listed ? tiles.p : nullptr, tile_count.p, tile_cap);
      else
        launch_moe_grouped_gemv_packq_f32(gather ? a_gathered.p : a_p, k, segs_p, experts, m, 0, views_p, 0,
                                          out, n, n, k, bits, nullptr, sf);
    } else {
      if (tc)
        launch_moe_grouped_mma_packq_bf16(a_p, k, segs_p, experts, m, views_p, 0, out, n, n, k,
                                          bits, nullptr, gather ? act_rows_p : nullptr, sf, var,
                                          mma && listed ? tiles.p : nullptr, tile_count.p, tile_cap,
                                          mma && pair ? pair_out.p : nullptr, 0);
      else
        launch_moe_grouped_gemv_packq_bf16(gather ? a_gathered.p : a_p, k, segs_p, experts, m, 0, views_p, 0,
                                           out, n, n, k, bits, nullptr, sf);
    }
  };
  std::printf(
      "shape m=%d n=%d k=%d bits=%d experts=%d top_k=%d out=%s nonempty=%d max_rows=%d "
      "distribution=%s segments<=16:%d <=32:%d <=64:%d tiles=%d cap=%d count=%d gather=%d\n",
      m, n, k, bits, experts, top_k, std::is_same_v<Out, float> ? "f32" : "bf16", nonempty, largest,
      dist.c_str(), small16, small32, small64, listed ? 1 : 0, tile_cap, listed ? tile_count.p[0] : 0,
      gather ? 1 : 0);
  // Complete managed-memory placement before the measured event pairs.
  launch(false);
  launch(true);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (int trial = 0; trial < 6; ++trial) {
    double ms[2];
    for (int mode : {trial % 2, 1 - trial % 2})
      ms[mode] = time_ms([&] { launch(mode != 0); }, iters);
    std::printf("trial=%d baseline_ms=%.6f candidate_ms=%.6f change_percent=%+.3f\n", trial, ms[0],
                ms[1], 100 * (ms[1] / ms[0] - 1));
    std::fflush(stdout);
  }
  double error = 0, norm = 0;
  for (size_t i = 0; i < static_cast<size_t>(rows) * n; ++i) {
    const double b = as_float(baseline.p[i]), c = as_float(candidate.p[i]);
    if (!std::isfinite(b) || !std::isfinite(c)) throw std::runtime_error("non-finite output");
    error += (b - c) * (b - c);
    norm += b * b;
  }
  const double relative = std::sqrt(error / std::max(norm, 1e-30));
  std::printf("relative_l2=%.9g\n", relative);
  if (baseline_mode == "mma0") {
    const bool same = std::memcmp(baseline.p, candidate.p, static_cast<size_t>(rows) * n * sizeof(Out)) == 0;
    std::printf("bitwise_vs_narrow=%s\n", same ? "yes" : "NO");
    if (!same) throw std::runtime_error("the wide kernel differs from the narrow one");
  } else if (relative > (std::is_same_v<Out, float> ? 5e-6 : 1e-3))
    throw std::runtime_error("GEMM/GEMV numerical discrepancy");
}
}  // namespace

int main(int argc, char** argv) try {
  int m = 256, n = 512, k = 6144, bits = 4, experts = 256, top_k = 8, iters = 5, sf = 0, variant = -1, tiles = 1,
      gather = 0, same_weights = 0, shuffle_rows = 0, device_copy = 0, pair = 0;
  std::string distribution = "uniform", output = "bf16", baseline = "gemv";
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 == argc) throw std::invalid_argument("missing option value");
    const std::string key = argv[i], value = argv[i + 1];
    if (key == "--m")
      m = std::stoi(value);
    else if (key == "--n")
      n = std::stoi(value);
    else if (key == "--k")
      k = std::stoi(value);
    else if (key == "--bits")
      bits = std::stoi(value);
    else if (key == "--experts")
      experts = std::stoi(value);
    else if (key == "--top-k")
      top_k = std::stoi(value);
    else if (key == "--iters")
      iters = std::stoi(value);
    else if (key == "--distribution")
      distribution = value;
    else if (key == "--out")
      output = value;
    else if (key == "--sf")
      sf = std::stoi(value);
    else if (key == "--variant")
      variant = std::stoi(value);
    else if (key == "--baseline")
      baseline = value;
    else if (key == "--tiles")
      tiles = std::stoi(value);
    else if (key == "--gather")
      gather = std::stoi(value);
    else if (key == "--same-weights")
      same_weights = std::stoi(value);
    else if (key == "--shuffle-rows")
      shuffle_rows = std::stoi(value);
    else if (key == "--device-copy")
      device_copy = std::stoi(value);
    else if (key == "--pair")
      pair = std::stoi(value);
    else
      throw std::invalid_argument("unknown option: " + key);
  }
  if (m <= 0 || n <= 0 || k <= 0 || k % 64 || (bits != 4 && bits != 8) || experts < top_k ||
      top_k <= 0 || iters <= 0 || (distribution != "uniform" && distribution != "hot" && distribution != "zipf") ||
      (output != "bf16" && output != "f32"))
    throw std::invalid_argument("invalid shape or options");
  if (output == "f32")
    run<float>(m, n, k, bits, experts, top_k, iters, distribution == "hot", sf, variant, baseline, distribution,
               tiles != 0, gather != 0, same_weights != 0, shuffle_rows != 0, device_copy != 0, pair != 0);
  else
    run<uint16_t>(m, n, k, bits, experts, top_k, iters, distribution == "hot", sf, variant, baseline, distribution,
                  tiles != 0, gather != 0, same_weights != 0, shuffle_rows != 0, device_copy != 0, pair != 0);
} catch (const std::exception& e) {
  std::fprintf(stderr, "packq_prefill_bench: %s\n", e.what());
  return 1;
}
