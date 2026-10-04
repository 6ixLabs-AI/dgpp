// UNVERIFIED DRAFT (2026-10-04) — written without a GPU, never compiled by
// nvcc, never run; built only under -DDGPP_BUILD_MINIMAX_DRAFT=ON.
//
// The MiniMax-M2 draft kernels (kernels/minimax_attn.hpp) against the host
// reference (models/minimax/attn_reference.hpp, itself tested against the
// numpy reference): the per-layer q/k norm finish — every head's elements
// under one RMS, a weight per element, the half-split RoPE, the paged K/V
// append — and the whole attention step through GLM-4.7's unchanged
// split-KV kernels. Modelled on tests/cuda/glm4_attn_test.cu; the bounds
// are that test's (two bf16 ulps for the finish — the device sums the
// squares in fp32, the reference in double).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/glm4_attn.hpp"
#include "kernels/minimax_attn.hpp"
#include "kernels/qsa.hpp"
#include "models/glm4/attn_reference.hpp"
#include "models/minimax/attn_reference.hpp"

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

constexpr int D = dgpp::kGlm4HeadDim;

// The release's attention shape: 48 query heads over 8 K/V heads; the rank
// slice of a two-rank world is 24 over 4.
struct Geo {
  int heads = 48, kv_heads = 8, rotary = 64;
  int head_begin = 0, local_heads = 48, kv_begin = 0, local_kv = 8;
  int block_tokens = 32, blocks_per_request = 12, max_requests = 3;
  double theta = 5e6;
  float eps = 1e-6f;
  int slots() const { return block_tokens * blocks_per_request; }
  int q_width() const { return heads * D; }
  int kv_width() const { return kv_heads * D; }
  int local_kv_width() const { return local_kv * D; }
};

template <class T>
DevBuf up(const std::vector<T>& v) {
  DevBuf b(v.size() * sizeof(T));
  b.upload(v.data(), v.size() * sizeof(T));
  return b;
}
template <class T>
std::vector<T> down(const DevBuf& b, size_t n) {
  std::vector<T> v(n);
  b.download(v.data(), n * sizeof(T));
  return v;
}
template <class T>
const T* ptr(const DevBuf& b) {
  return static_cast<const T*>(b.p);
}
template <class T>
T* mptr(DevBuf& b) {
  return static_cast<T*>(b.p);
}

std::vector<int32_t> tables(const Geo& g) {
  std::vector<int32_t> t(static_cast<size_t>(g.max_requests) * g.blocks_per_request);
  for (int q = 0; q < g.max_requests; ++q)
    for (int b = 0; b < g.blocks_per_request; ++b)
      t[static_cast<size_t>(q) * g.blocks_per_request + b] = q * g.blocks_per_request + b;
  return t;
}
int64_t phys(const Geo& g, int req, int64_t pos) {
  return static_cast<int64_t>(req) * g.blocks_per_request * g.block_tokens + pos;
}

std::vector<float> inv_freq_host(const Geo& g) {
  std::vector<float> f(static_cast<size_t>(g.rotary / 2));
  dgpp::qsa_rope_inv_freq(g.theta, g.rotary, f.data());
  return f;
}

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

std::vector<uint16_t> norm_weights(uint64_t seed, int n) {
  std::vector<uint16_t> w(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    w[static_cast<size_t>(i)] = fast_bf16_rne(0.8f + 0.4f * uniform_01(seed, i, 1.0f));
  return w;
}

// The finish of the rank slice `g` describes, device against reference.
void run_finish(const Geo& g, const char* what) {
  cudaStream_t st = test_stream();
  const int rows = 9;
  const std::vector<int32_t> req_ids = {0, 0, 1, 2, 2, 1, 0, 2, 1};
  const std::vector<int64_t> pos = {0, 1, 17, 63, 64, 200, 383, -1, 5};
  const int qw = g.q_width(), kw = g.kv_width(), lkw = g.local_kv_width();
  // Every head's dots: the norm spans them all, whatever the rank keeps.
  const std::vector<float> qd = random_f32_uniform(11, static_cast<int64_t>(rows) * qw, 2.0f);
  const std::vector<float> kd = random_f32_uniform(12, static_cast<int64_t>(rows) * kw, 2.0f);
  const std::vector<float> vd = random_f32_uniform(13, static_cast<int64_t>(rows) * lkw, 2.0f);
  const std::vector<uint16_t> qn = norm_weights(17, qw), kn = norm_weights(18, kw);
  const std::vector<float> inv = inv_freq_host(g);
  const std::vector<int32_t> tbl = tables(g);
  DevBuf dqd = up(qd), dkd = up(kd), dvd = up(vd), dqn = up(qn), dkn = up(kn), dinv = up(inv),
         dreq = up(req_ids), dpos = up(pos), dtbl = up(tbl);
  DevBuf q_rstd(static_cast<size_t>(rows) * 4), k_rstd(static_cast<size_t>(rows) * 4);
  const int lqw = g.local_heads * D;
  const size_t cache_bytes = static_cast<size_t>(g.max_requests) * g.slots() * lkw * 2;
  DevBuf kc(cache_bytes), vc(cache_bytes), qo(static_cast<size_t>(rows) * lqw * 2);
  DGPP_CUDA_OK(cudaMemset(kc.p, 0x7F, cache_bytes));
  DGPP_CUDA_OK(cudaMemset(vc.p, 0x7F, cache_bytes));
  DGPP_CUDA_OK(cudaMemset(qo.p, 0x7F, static_cast<size_t>(rows) * lqw * 2));
  auto run = [&] {
    dgpp::minimax_qk_rstd(ptr<float>(dqd), qw, qw, ptr<int64_t>(dpos), rows, g.eps,
                          mptr<float>(q_rstd), st);
    dgpp::minimax_qk_rstd(ptr<float>(dkd), kw, kw, ptr<int64_t>(dpos), rows, g.eps,
                          mptr<float>(k_rstd), st);
    dgpp::minimax_qkv_finish(
        ptr<float>(dqd) + g.head_begin * D, qw, ptr<float>(dkd) + g.kv_begin * D, kw,
        ptr<float>(dvd), lkw, ptr<uint16_t>(dqn) + g.head_begin * D,
        ptr<uint16_t>(dkn) + g.kv_begin * D, ptr<float>(q_rstd), ptr<float>(k_rstd),
        ptr<float>(dinv), g.rotary, ptr<int32_t>(dreq), ptr<int64_t>(dpos), rows, g.local_heads,
        g.local_kv, ptr<int32_t>(dtbl), g.blocks_per_request, g.block_tokens, mptr<uint16_t>(qo),
        lqw, mptr<uint16_t>(kc), mptr<uint16_t>(vc), st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
  };
  run();
  const std::vector<uint16_t> got_q = down<uint16_t>(qo, static_cast<size_t>(rows) * lqw);
  const std::vector<uint16_t> got_k = down<uint16_t>(kc, cache_bytes / 2),
                              got_v = down<uint16_t>(vc, cache_bytes / 2);
  int max_ulps = 0;
  long checked = 0;
  std::vector<uint16_t> ref_q(static_cast<size_t>(qw)), ref_k(static_cast<size_t>(kw));
  for (int r = 0; r < rows; ++r) {
    const int64_t p = pos[static_cast<size_t>(r)];
    const uint16_t* gq = got_q.data() + static_cast<size_t>(r) * lqw;
    if (p < 0) {
      for (int i = 0; i < lqw; ++i) require(gq[i] == 0x7F7F, "a padding row writes no q");
      continue;
    }
    // The reference finishes the WHOLE row; the rank's heads are its slice.
    dgpp::minimax_ref::qk_finish(qd.data() + static_cast<size_t>(r) * qw, qn.data(), g.eps,
                                 inv.data(), g.rotary, p, g.heads, D, ref_q.data());
    dgpp::minimax_ref::qk_finish(kd.data() + static_cast<size_t>(r) * kw, kn.data(), g.eps,
                                 inv.data(), g.rotary, p, g.kv_heads, D, ref_k.data());
    for (int i = 0; i < lqw; ++i) {
      max_ulps =
          std::max(max_ulps, bf16_ulps(gq[i], ref_q[static_cast<size_t>(g.head_begin) * D + i]));
      ++checked;
    }
    const int64_t slot = phys(g, req_ids[static_cast<size_t>(r)], p);
    const uint16_t* gk = got_k.data() + slot * lkw;
    const uint16_t* gv = got_v.data() + slot * lkw;
    for (int i = 0; i < lkw; ++i) {
      max_ulps =
          std::max(max_ulps, bf16_ulps(gk[i], ref_k[static_cast<size_t>(g.kv_begin) * D + i]));
      require(gv[i] == dgpp::float_to_bf16_bits(vd[static_cast<size_t>(r) * lkw + i]),
              "v: bf16(dot) bitwise");
      checked += 2;
    }
  }
  std::printf("[ .. ] %s: %ld elements, max %d bf16 ulps\n", what, checked, max_ulps);
  require(max_ulps <= 2, std::string(what) + ": within two bf16 ulps of the reference");
  run();
  require(down<uint16_t>(qo, got_q.size()) == got_q && down<uint16_t>(kc, got_k.size()) == got_k,
          "second run bitwise");
}

}  // namespace

DGPP_TEST(minimax_qk_finish_matches_the_reference_world1) {
  run_finish(Geo{}, "per-layer qk finish, every head");
}

DGPP_TEST(minimax_qk_finish_matches_the_reference_rank_slices) {
  // The two ranks of world 2: each norms the whole row and keeps its half.
  for (int rank = 0; rank < 2; ++rank) {
    Geo g;
    g.local_heads = g.heads / 2;
    g.head_begin = rank * g.local_heads;
    g.local_kv = g.kv_heads / 2;
    g.kv_begin = rank * g.local_kv;
    run_finish(g,
               rank == 0 ? "per-layer qk finish, rank 0 of 2" : "per-layer qk finish, rank 1 of 2");
  }
}

DGPP_TEST(minimax_attention_runs_on_the_glm4_split_kernels) {
  // The attention step after the finish is GLM-4.7's: 48 query heads, six
  // per K/V head. One request of 300 tokens, rows across tile and block
  // boundaries, one padding row — glm4_attn_test's case at this model's
  // head counts, against glm4_ref::attention (the oracle
  // minimax_ref::layer_forward calls).
  Geo g;
  cudaStream_t st = test_stream();
  const int seq = 300, req = 1;
  const int kw = g.kv_width(), qw = g.q_width();
  const std::vector<uint16_t> k = random_bf16_normal(31, static_cast<int64_t>(seq) * kw, 1.0f);
  const std::vector<uint16_t> v = random_bf16_normal(32, static_cast<int64_t>(seq) * kw, 1.0f);
  const size_t cache_bytes = static_cast<size_t>(g.max_requests) * g.slots() * kw * 2;
  std::vector<uint16_t> kh(cache_bytes / 2, 0x7F7F), vh(cache_bytes / 2, 0x7F7F);
  for (int t = 0; t < seq; ++t) {
    std::copy(k.begin() + static_cast<size_t>(t) * kw, k.begin() + static_cast<size_t>(t + 1) * kw,
              kh.begin() + phys(g, req, t) * kw);
    std::copy(v.begin() + static_cast<size_t>(t) * kw, v.begin() + static_cast<size_t>(t + 1) * kw,
              vh.begin() + phys(g, req, t) * kw);
  }
  DevBuf kc = up(kh), vc = up(vh), dtbl = up(tables(g));
  const std::vector<int64_t> pos = {0, 1, 31, 32, 33, 63, 64, 100, 255, 256, 299, -1};
  const int rows = static_cast<int>(pos.size());
  const std::vector<int32_t> req_ids(static_cast<size_t>(rows), req);
  const std::vector<uint16_t> q = random_bf16_normal(33, static_cast<int64_t>(rows) * qw, 1.0f);
  DevBuf dq = up(q), dpos = up(pos), dreq = up(req_ids);
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  for (int n_split : {1, 3, 16}) {
    std::vector<uint16_t> ref(static_cast<size_t>(rows) * qw, 0);
    for (int r = 0; r < rows; ++r) {
      const int64_t p = pos[static_cast<size_t>(r)];
      if (p < 0) continue;
      std::vector<float> c;
      dgpp::glm4_ref::attention(q.data() + static_cast<size_t>(r) * qw, k.data(), v.data(),
                                static_cast<int>(p + 1), g.heads, g.kv_heads, D, scale, c,
                                dgpp::kGlm4AttnTile, n_split);
      for (int i = 0; i < qw; ++i)
        ref[static_cast<size_t>(r) * qw + i] = dgpp::float_to_bf16_bits(c[static_cast<size_t>(i)]);
    }
    const size_t part = static_cast<size_t>(rows) * n_split * g.heads;
    DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * D * 4), out(ref.size() * 2);
    DGPP_CUDA_OK(cudaMemset(out.p, 0x7F, ref.size() * 2));
    dgpp::glm4_attn_partial(ptr<uint16_t>(dq), qw, ptr<uint16_t>(kc), ptr<uint16_t>(vc),
                            ptr<int32_t>(dreq), ptr<int64_t>(dpos), rows, n_split, g.heads,
                            g.kv_heads, g.block_tokens, ptr<int32_t>(dtbl), g.blocks_per_request,
                            scale, mptr<float>(m_ws), mptr<float>(l_ws), mptr<float>(c_ws), st);
    dgpp::glm4_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), rows, n_split,
                            g.heads, mptr<uint16_t>(out), st);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    const std::vector<uint16_t> got = down<uint16_t>(out, ref.size());
    std::vector<float> gf(got.size()), wf(ref.size());
    double rms = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      gf[i] = dgpp::bf16_bits_to_float(got[i]);
      wf[i] = dgpp::bf16_bits_to_float(ref[i]);
      rms += static_cast<double>(wf[i]) * wf[i];
    }
    rms = std::sqrt(rms / static_cast<double>(ref.size()));
    const Stats s = compare_abs_rel(gf.data(), wf.data(), static_cast<long>(got.size()),
                                    2 * std::pow(2.0, -7.0), 0.005 * rms);
    std::printf(
        "[ .. ] attention at 48/8 heads, n_split=%d: max_abs %.3g l2_rel %.3g mismatches %ld/%ld "
        "(rms %.3g)\n",
        n_split, s.max_abs, s.l2_rel, s.mismatches, s.n, rms);
    require_bf16("attention n_split=" + std::to_string(n_split), s, 2e-3, 0.01);
  }
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
