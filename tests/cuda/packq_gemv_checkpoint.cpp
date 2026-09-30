// Real-checkpoint slice parity for the packed-int GEMV (docs/glm53_plan.md
// G2, the "real slices" half): mmaps four triples out of the GLM-5.3
// int4/int8 checkpoint (`packq_gemv_test --checkpoint-dir <snapshot>`) —
// a routed expert's gate (int4, K 6144) and down (int4, K 2048), the q_a
// projection (int8, K 6144) and kv_b (int8, K 512) — checks each
// weight_shape record against the packed geometry, runs the launcher at
// m = 48 against the double oracle, checks that single rows reproduce the
// batched rows bit for bit, and pins the packing contract on the real
// bytes: the signed codes are unimodal about zero (a two's-complement or
// sign-magnitude reading of the same nibbles would be bimodal at the
// extremes) and the scales are release-like magnitudes. Host-only TU (the
// safetensors reader's JSON parser does not mix with nvcc).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "kernels/packq_gemm.hpp"
#include "kernels/packq_gemv.hpp"
#include "loaders/safetensors.hpp"
#include "models/quant_matrix.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;

struct Slice {
  int bits = 0, n = 0, k = 0;
  std::vector<uint32_t> packed;
  std::vector<uint16_t> scales;
  int64_t shape[2] = {0, 0};
};

int code_at(const Slice& s, int nn, int kk) {
  const int per = 32 / s.bits;
  const uint32_t word = s.packed[static_cast<size_t>(nn) * (s.k / per) + kk / per];
  const uint32_t u = (word >> (s.bits * (kk % per))) & ((1u << s.bits) - 1u);
  return static_cast<int>(u) - (1 << (s.bits - 1));
}

std::vector<double> oracle(const Slice& s, const std::vector<uint16_t>& act, int m,
                           std::vector<double>* unrounded = nullptr) {
  std::vector<double> out(static_cast<size_t>(m) * s.n, 0.0);
  if (unrounded) unrounded->resize(out.size());
  std::vector<double> wrow(s.k);
  for (int nn = 0; nn < s.n; ++nn) {
    for (int kk = 0; kk < s.k; ++kk) {
      const float sc = bf16_to_float(s.scales[static_cast<size_t>(nn) * (s.k / 64) + kk / 64]);
      wrow[kk] = static_cast<double>(code_at(s, nn, kk)) * static_cast<double>(sc);
    }
    for (int mm = 0; mm < m; ++mm) {
      double acc = 0.0;
      const uint16_t* arow = act.data() + static_cast<size_t>(mm) * s.k;
      for (int kk = 0; kk < s.k; ++kk) acc += bf16_to_float(arow[kk]) * wrow[kk];
      if (unrounded) (*unrounded)[static_cast<size_t>(mm) * s.n + nn] = acc;
      out[static_cast<size_t>(mm) * s.n + nn] =
          bf16_to_float(dgpp::float_to_bf16_bits(static_cast<float>(acc)));
    }
  }
  return out;
}

std::vector<uint16_t> run(const Slice& s, const std::vector<uint16_t>& act, int m,
                          bool mma = false) {
  uint32_t* packed = nullptr;
  uint16_t* scales = nullptr;
  uint16_t* d_act = nullptr;
  uint16_t* out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&packed, s.packed.size() * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&scales, s.scales.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_act, act.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&out, static_cast<size_t>(m) * s.n * 2));
  std::memcpy(packed, s.packed.data(), s.packed.size() * 4);
  std::memcpy(scales, s.scales.data(), s.scales.size() * 2);
  std::memcpy(d_act, act.data(), act.size() * 2);
  const dgpp::GlmPackedMatrix w{packed, scales, s.n, s.k, s.bits};
  if (mma)
    dgpp::launch_packq_gemm_bf16(d_act, s.k, w, out, m, s.n, s.k, nullptr);
  else
    dgpp::launch_packq_gemv_bf16(d_act, s.k, w, out, m, s.n, s.k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(static_cast<size_t>(m) * s.n);
  std::memcpy(got.data(), out, got.size() * 2);
  cudaFree(packed);
  cudaFree(scales);
  cudaFree(d_act);
  cudaFree(out);
  return got;
}

// The packing contract on real bytes: the signed codes' mass at {-1, 0, 1}
// against the mass at the four extreme codes, and the scales' median.
void pin_contract(const Slice& s, const std::string& base) {
  const int lo = -(1 << (s.bits - 1)), hi = (1 << (s.bits - 1)) - 1;
  size_t center = 0, tails = 0, total = 0;
  const int rows = std::min(s.n, 64);
  for (int nn = 0; nn < rows; ++nn)
    for (int kk = 0; kk < s.k; ++kk) {
      const int c = code_at(s, nn, kk);
      ++total;
      if (c >= -1 && c <= 1) ++center;
      if (c <= lo + 1 || c >= hi - 1) ++tails;
    }
  std::vector<float> sc;
  for (size_t i = 0; i < s.scales.size(); i += 97) sc.push_back(bf16_to_float(s.scales[i]));
  std::sort(sc.begin(), sc.end());
  const float median = sc[sc.size() / 2];
  std::printf("  contract: codes in [-1,1] %.1f %%, at the four extremes %.1f %%, median scale %.3g\n",
              100.0 * center / total, 100.0 * tails / total, median);
  // int4: the histogram is bell-shaped about 0 (the checkpoint's nibbles
  // are centred on 8); a wrong offset reading puts the mass at the ends.
  // int8's centre bins are narrow by construction, so its pin is the tails.
  if (s.bits == 4)
    require(center > 3 * tails, (base + ": int4 codes are not unimodal about zero (packing offset?)").c_str());
  else
    require(tails * 20 < total, (base + ": int8 codes pile at the extremes (packing offset?)").c_str());
  require(median > 1e-7f && median < 1e-1f, (base + ": scale magnitudes are not release-like").c_str());
}

}  // namespace

int run_packq_gemv_checkpoint_parity(const char* checkpoint_dir) {
  namespace fs = std::filesystem;
  struct Want {
    const char* base;
    const char* shard;
    int bits;
  };
  const Want wants[] = {
      {"model.layers.3.mlp.experts.0.gate_proj.", "layer-003.safetensors", 4},
      {"model.layers.3.mlp.experts.255.down_proj.", "layer-003.safetensors", 4},
      {"model.layers.3.self_attn.q_a_proj.", "layer-003.safetensors", 8},
      {"model.layers.3.self_attn.kv_b_proj.", "layer-003.safetensors", 8},
  };
  const char* suffix[3] = {"weight_packed", "weight_scale", "weight_shape"};
  for (const Want& w : wants) {
    const std::string base = w.base;
    const fs::path shard = fs::path(checkpoint_dir) / w.shard;
    if (!fs::exists(shard)) {
      std::printf("packq checkpoint parity: %s not found under %s\n", w.shard, checkpoint_dir);
      return 2;
    }
    auto f = dgpp::SafetensorsFile::open(shard.string());
    Slice s;
    s.bits = w.bits;
    for (int i = 0; i < 3; ++i) {
      const dgpp::TensorInfo* t = f->find(base + suffix[i]);
      if (!t) {
        std::printf("packq checkpoint parity: %s%s not found\n", base.c_str(), suffix[i]);
        return 2;
      }
      const uint8_t* d = static_cast<const uint8_t*>(t->data);
      if (i == 0) {
        s.n = static_cast<int>(t->shape[0]);
        s.k = static_cast<int>(t->shape[1] * 32 / w.bits);
        s.packed.resize(t->nbytes() / 4);
        std::memcpy(s.packed.data(), d, t->nbytes());
      } else if (i == 1) {
        s.scales.resize(t->nbytes() / 2);
        std::memcpy(s.scales.data(), d, t->nbytes());
      } else {
        std::memcpy(s.shape, d, 16);
      }
    }
    std::printf("packq checkpoint parity: %s int%d [%d x %d], weight_shape [%lld, %lld]\n",
                base.c_str(), s.bits, s.n, s.k, static_cast<long long>(s.shape[0]),
                static_cast<long long>(s.shape[1]));
    require(s.shape[0] == s.n && s.shape[1] == s.k, (base + ": weight_shape disagrees with the packed geometry").c_str());
    require(s.scales.size() == static_cast<size_t>(s.n) * (s.k / 64), (base + ": scale count").c_str());
    pin_contract(s, base);
    const int m = 48;
    Rng rng(0xC0FFEE);
    std::vector<uint16_t> act(static_cast<size_t>(m) * s.k);
    fill_act(rng, act);
    const std::vector<uint16_t> got = run(s, act, m);
    std::vector<double> exact;
    const auto want = oracle(s, act, m, &exact);
    const auto rep = compare_bf16_vs_oracle(got.data(), want, 2.0, 1e-3);
    std::printf("[ OK ] real slice: l2_rel=%.3g mismatches=%ld/%zu\n", rep.l2_rel,
                rep.mismatches, rep.total);
    require_report(rep, 1e-3, 0, "packq real slice vs oracle");
    const std::vector<uint16_t> tiled = run(s, act, m, true);
    const auto tiled_rep = compare_bf16_vs_oracle(tiled.data(), want, 2.0, 1e-3);
    std::printf("[ OK ] real GEMM slice: l2_rel=%.3g mismatches=%ld/%zu\n", tiled_rep.l2_rel,
                tiled_rep.mismatches, tiled_rep.total);
    require_report(tiled_rep, 1e-3, 0, "packq real GEMM slice vs oracle");
    double norm2 = 0, gemv_error2 = 0, gemm_error2 = 0;
    size_t changed = 0, closer = 0, farther = 0;
    for (size_t i = 0; i < exact.size(); ++i) {
      const double old_error = bf16_to_float(got[i]) - exact[i];
      const double new_error = bf16_to_float(tiled[i]) - exact[i];
      norm2 += exact[i] * exact[i];
      gemv_error2 += old_error * old_error;
      gemm_error2 += new_error * new_error;
      changed += got[i] != tiled[i];
      closer += std::abs(new_error) < std::abs(old_error);
      farther += std::abs(new_error) > std::abs(old_error);
    }
    std::printf(
        "  unrounded FP64: GEMV relative L2 %.12g, GEMM %.12g; "
        "changed %zu/%zu, GEMM closer %zu farther %zu (remaining changes equidistant)\n",
        std::sqrt(gemv_error2 / std::max(norm2, 1e-30)),
        std::sqrt(gemm_error2 / std::max(norm2, 1e-30)), changed, exact.size(), closer, farther);
    for (int r : {0, 17, 47}) {
      std::vector<uint16_t> row(act.begin() + static_cast<long>(r) * s.k,
                                act.begin() + static_cast<long>(r + 1) * s.k);
      const std::vector<uint16_t> got1 = run(s, row, 1);
      require(std::memcmp(got1.data(), got.data() + static_cast<size_t>(r) * s.n,
                          static_cast<size_t>(s.n) * 2) == 0,
              "real slice: row bits independent of m");
    }
  }
  std::printf("[ OK ] packq real-checkpoint slice parity (4 tensors)\n");
  return 0;
}
