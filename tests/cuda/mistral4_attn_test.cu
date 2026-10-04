// UNVERIFIED DRAFT (2026-10-04) — written without a GPU, never compiled by
// nvcc, never run; built only under -DDGPP_BUILD_MISTRAL_DRAFT=ON.
//
// The Mistral-Small-4 attention step on the device, composed from the
// existing latent-attention kernels (kernels/dsa.hpp) and the one new
// kernel (kernels/mistral4_attn.hpp), against the host reference
// (models/mistral4/mla_reference.hpp, itself tested against the numpy
// reference) at the release's attention geometry: 32 heads, a 1024-wide
// query latent, a 256-wide key/value latent with a 64-wide rope key, 64 +
// 64 query dims and 128 value dims per head, YaRN inverse frequencies,
// positions across the Llama-4 step at 8192.
//
// This test is the draft of the layer's kernel sequence: the order and the
// arguments below are what a Mistral4 attention layer must enqueue. The
// projections (wq_a / wkv_a, wq_b) run on the host here, with the
// reference's own loop; the layer runs them through the GEMM interface.
// What it will tell on a Spark, beyond pass / fail:
//   * whether dsa_attn_dense is compiled for this geometry (kv_lora 256
//     with a 64-wide tail — kernels/dsa.hpp lists 512 / 256 without a tail
//     and 512 with one): printed. When it is not, the list-free prefill and
//     long-context decode path needs that instantiation; the split kernel
//     over an explicit causal list (used here) is the fallback.
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
#include "kernels/dsa.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/mistral4_attn.hpp"
#include "kernels/rope_scaling.hpp"
#include "models/mistral4/mla_reference.hpp"

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

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

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

// out[m, n] = bf16(sum_k act[m, k] * w[n, k]), fp32 accumulation in index
// order — the reference's projection loop (mla_reference.cpp), so the rows
// handed to the device are the reference's own bits.
void gemm_bf16(const uint16_t* act, int64_t act_stride, const uint16_t* w, uint16_t* out, int m,
               int n, int k) {
  for (int mm = 0; mm < m; ++mm)
    for (int nn = 0; nn < n; ++nn) {
      float acc = 0.f;
      for (int kk = 0; kk < k; ++kk)
        acc += dgpp::bf16_bits_to_float(act[mm * act_stride + kk]) *
               dgpp::bf16_bits_to_float(w[static_cast<int64_t>(nn) * k + kk]);
      out[static_cast<int64_t>(mm) * n + nn] = dgpp::float_to_bf16_bits(acc);
    }
}

std::vector<uint16_t> gain(uint64_t seed, int n) {
  std::vector<uint16_t> w(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    w[static_cast<size_t>(i)] = fast_bf16_rne(0.8f + 0.4f * uniform_01(seed, i, 1.0f));
  return w;
}

}  // namespace

DGPP_TEST(mistral4_llama4_scale_is_the_references_bits) {
  cudaStream_t st = test_stream();
  const float beta = 0.1f;
  const int original = 8192, rows = 7, width = 32 * 128;
  const std::vector<int64_t> pos = {0, 8191, 8192, 16384, 100000, -1, 1048575};
  const std::vector<float> table = dgpp::mistral4_llama4_table_host(beta, original, 1048576);
  require(table.size() == 128 && table[0] == 1.0f, "128 steps over a 1M context");
  const std::vector<uint16_t> q = random_bf16_normal(71, static_cast<int64_t>(rows) * width, 1.0f);
  DevBuf dq = up(q), dpos = up(pos), dtab = up(table);
  dgpp::mistral4_llama4_scale(mptr<uint16_t>(dq), width, width, ptr<int64_t>(dpos), rows,
                              ptr<float>(dtab), static_cast<int>(table.size()), original, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> got = down<uint16_t>(dq, q.size());
  for (int r = 0; r < rows; ++r) {
    const float l4 = dgpp::mistral4_ref::llama4_scale(beta, original, pos[static_cast<size_t>(r)]);
    for (int i = 0; i < width; ++i) {
      const size_t at = static_cast<size_t>(r) * width + i;
      const uint16_t want =
          l4 == 1.0f ? q[at] : dgpp::float_to_bf16_bits(dgpp::bf16_bits_to_float(q[at]) * l4);
      require(got[at] == want,
              "row " + std::to_string(r) + ": the scaled query is the reference's bits");
    }
  }
}

DGPP_TEST(mistral4_attention_on_the_latent_kernels_matches_the_reference) {
  cudaStream_t st = test_stream();
  dgpp::dsa_prepare_kernel_smem();
  // The release's attention geometry; a small hidden (the projections are
  // host-side here) and a short request that crosses the Llama-4 step.
  dgpp::mistral4_ref::Geometry g;
  g.hidden = 256;
  g.heads = 32;
  g.q_lora = 1024;
  g.kv_lora = 256;
  g.nope = 64;
  g.rope = 64;
  g.v = 128;
  g.eps = 1e-6f;
  g.scale = 1.0f / std::sqrt(128.0f);
  g.llama4_beta = 0.1f;
  g.llama4_original = 8192;
  const int T = 40;
  const int64_t pos0 = 8192 - 20;
  const int q_rows = g.heads * g.q_head(), v_rows = g.heads * g.v, width = g.row_width();
  const int qkv_cols = g.q_lora + g.kv_lora + g.rope;

  // Weights: BF16, the checkpoint's layouts.
  const std::vector<uint16_t> wq_a =
      random_bf16_normal(1, static_cast<int64_t>(g.q_lora) * g.hidden, 0.06f);
  const std::vector<uint16_t> wkv_a =
      random_bf16_normal(2, static_cast<int64_t>(g.kv_lora + g.rope) * g.hidden, 0.06f);
  const std::vector<uint16_t> wq_b =
      random_bf16_normal(3, static_cast<int64_t>(q_rows) * g.q_lora, 0.06f);
  const std::vector<uint16_t> wkv_b =
      random_bf16_normal(4, static_cast<int64_t>(g.heads) * g.kv_head() * g.kv_lora, 0.06f);
  const std::vector<uint16_t> wo =
      random_bf16_normal(5, static_cast<int64_t>(g.hidden) * v_rows, 0.02f);
  const std::vector<uint16_t> q_a_norm = gain(6, g.q_lora), kv_a_norm = gain(7, g.kv_lora);
  const std::vector<uint16_t> hidden =
      random_bf16_normal(8, static_cast<int64_t>(T) * g.hidden, 1.0f);
  std::vector<float> inv_freq(static_cast<size_t>(g.rope / 2));
  dgpp::yarn_rope_inv_freq_host(g.rope, 10000.0, 8192, 128.0, 32.0, 1.0, inv_freq.data());

  // The reference.
  dgpp::mistral4_ref::HostWeights hw;
  hw.wq_a = wq_a.data();
  hw.q_a_norm = q_a_norm.data();
  hw.wq_b = wq_b.data();
  hw.wkv_a = wkv_a.data();
  hw.kv_a_norm = kv_a_norm.data();
  hw.wkv_b = wkv_b.data();
  hw.wo = wo.data();
  dgpp::mistral4_ref::HostState state;
  state.reset(pos0);
  dgpp::mistral4_ref::Taps taps;
  std::vector<uint16_t> ref_out(static_cast<size_t>(T) * g.hidden);
  dgpp::mistral4_ref::layer_forward<float>(hw, g, inv_freq.data(), hidden.data(), state, T,
                                           ref_out.data(), &taps);

  // ---- the device sequence -------------------------------------------------
  // 1) the fused [q_a | kv_a] rows (host GEMM here), then the two norms.
  std::vector<uint16_t> qkv(static_cast<size_t>(T) * qkv_cols);
  {
    std::vector<uint16_t> qa(static_cast<size_t>(T) * g.q_lora),
        kva(static_cast<size_t>(T) * (g.kv_lora + g.rope));
    gemm_bf16(hidden.data(), g.hidden, wq_a.data(), qa.data(), T, g.q_lora, g.hidden);
    gemm_bf16(hidden.data(), g.hidden, wkv_a.data(), kva.data(), T, g.kv_lora + g.rope, g.hidden);
    for (int t = 0; t < T; ++t) {
      std::copy(qa.begin() + static_cast<int64_t>(t) * g.q_lora,
                qa.begin() + static_cast<int64_t>(t + 1) * g.q_lora,
                qkv.begin() + static_cast<int64_t>(t) * qkv_cols);
      std::copy(kva.begin() + static_cast<int64_t>(t) * (g.kv_lora + g.rope),
                kva.begin() + static_cast<int64_t>(t + 1) * (g.kv_lora + g.rope),
                qkv.begin() + static_cast<int64_t>(t) * qkv_cols + g.q_lora);
    }
  }
  std::vector<int64_t> pos(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t) pos[static_cast<size_t>(t)] = pos0 + t;
  const std::vector<int32_t> req_ids(static_cast<size_t>(T), 0);
  DevBuf dqkv = up(qkv), dqn = up(q_a_norm), dkn = up(kv_a_norm), dpos = up(pos),
         dreq = up(req_ids);
  DevBuf q_c(static_cast<size_t>(T) * g.q_lora * 2), kv_c(static_cast<size_t>(T) * g.kv_lora * 2);
  dgpp::dsa_fused_qkv_rmsnorm(dqkv.p, q_c.p, kv_c.p, g.q_lora, g.kv_lora, T, dqn.p, dkn.p, g.eps,
                              st, qkv_cols);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> got_latent = down<uint16_t>(kv_c, static_cast<size_t>(T) * g.kv_lora);
  int latent_ulps = 0;
  for (size_t i = 0; i < got_latent.size(); ++i)
    latent_ulps = std::max(latent_ulps, bf16_ulps(got_latent[i], taps.latent[i]));
  std::printf("[ .. ] latent norm: max %d bf16 ulps\n", latent_ulps);
  require(latent_ulps <= 1, "the normed latent within one bf16 ulp of the reference");

  // 2) q = wq_b @ q latent (host GEMM here, from the device's normed rows).
  const std::vector<uint16_t> q_c_host = down<uint16_t>(q_c, static_cast<size_t>(T) * g.q_lora);
  std::vector<uint16_t> q(static_cast<size_t>(T) * q_rows);
  gemm_bf16(q_c_host.data(), g.q_lora, wq_b.data(), q.data(), T, q_rows, g.q_lora);
  DevBuf dq = up(q);

  // 3) the rope: each head's rope slice in place, the token's rope key out
  //    of the fused row — the YaRN table, the interleaved pairs.
  const int64_t table_positions = pos0 + T;
  std::vector<uint16_t> table(static_cast<size_t>(table_positions) * g.rope);
  dgpp::mistral4_ref::rope_table(inv_freq.data(), g.rope, table_positions, table.data());
  DevBuf dtable = up(table), k_rot(static_cast<size_t>(T) * g.rope * 2);
  dgpp::dsa_rope_interleave(mptr<uint16_t>(dq) + g.nope, q_rows, g.q_head(), g.heads, g.rope,
                            ptr<int64_t>(dpos), dtable.p, table_positions,
                            mptr<uint16_t>(dq) + g.nope, q_rows, g.q_head(), T, st);
  dgpp::dsa_rope_interleave(ptr<uint16_t>(dqkv) + g.q_lora + g.kv_lora, qkv_cols, g.rope, 1, g.rope,
                            ptr<int64_t>(dpos), dtable.p, table_positions, k_rot.p, g.rope, g.rope,
                            T, st);

  // 4) the Llama-4 query scale.
  const std::vector<float> l4 =
      dgpp::mistral4_llama4_table_host(g.llama4_beta, g.llama4_original, table_positions);
  DevBuf dl4 = up(l4);
  dgpp::mistral4_llama4_scale(mptr<uint16_t>(dq), q_rows, q_rows, ptr<int64_t>(dpos), T,
                              ptr<float>(dl4), static_cast<int>(l4.size()), g.llama4_original, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> got_q = down<uint16_t>(dq, q.size());
  const std::vector<uint16_t> got_k_rot = down<uint16_t>(k_rot, static_cast<size_t>(T) * g.rope);
  int q_ulps = 0, k_ulps = 0;
  for (size_t i = 0; i < got_q.size(); ++i)
    q_ulps = std::max(q_ulps, bf16_ulps(got_q[i], taps.q[i]));
  for (size_t i = 0; i < got_k_rot.size(); ++i)
    k_ulps = std::max(k_ulps, bf16_ulps(got_k_rot[i], taps.k_rope[i]));
  std::printf("[ .. ] query after rope and scale: max %d bf16 ulps; rope key: max %d\n", q_ulps,
              k_ulps);
  require(k_ulps == 0,
          "the rotated rope key is the reference's bits (one table, the same three roundings)");
  require(q_ulps <= 2, "the query within two bf16 ulps (the latent norm's ulp through wq_b)");

  // 5) the cache rows: [latent | rope key] at the row's slot.
  const int block_tokens = 128;
  const int blocks = static_cast<int>((table_positions + block_tokens - 1) / block_tokens);
  std::vector<int32_t> block_table(static_cast<size_t>(blocks));
  for (int b = 0; b < blocks; ++b) block_table[static_cast<size_t>(b)] = b;
  DevBuf dtbl = up(block_table);
  const size_t cache_bytes = static_cast<size_t>(blocks) * block_tokens * width * 2;
  DevBuf cache(cache_bytes);
  DGPP_CUDA_OK(cudaMemset(cache.p, 0, cache_bytes));
  dgpp::dsa_latent_append(kv_c.p, ptr<int32_t>(dreq), ptr<int64_t>(dpos), T, ptr<int32_t>(dtbl),
                          blocks, block_tokens, cache.p, g.kv_lora, st, dgpp::LatentFormat::kBf16,
                          nullptr, k_rot.p, g.rope);

  // 6) attention: the absorbed query against the cache rows of positions
  //    [pos0, pos] (the request's own tokens: an explicit causal list), the
  //    combine, the value projection. Eight rows a tile, as the layer runs.
  std::vector<int32_t> topk(static_cast<size_t>(T) * T, -1), counts(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t) {
    for (int s = 0; s <= t; ++s)
      topk[static_cast<size_t>(t) * T + s] = static_cast<int32_t>(pos0 + s);
    counts[static_cast<size_t>(t)] = t + 1;
  }
  DevBuf dtopk = up(topk), dcounts = up(counts), dkvb = up(wkv_b);
  const int n_split = 4, tile = 8;
  DevBuf q_tilde(static_cast<size_t>(tile) * g.heads * width * 2);
  DevBuf m_ws(static_cast<size_t>(tile) * n_split * g.heads * 4),
      l_ws(static_cast<size_t>(tile) * n_split * g.heads * 4);
  DevBuf c_ws(static_cast<size_t>(tile) * n_split * g.heads * g.kv_lora * 4);
  DevBuf c(static_cast<size_t>(tile) * g.heads * g.kv_lora * 4),
      attn(static_cast<size_t>(T) * v_rows * 2);
  for (int a0 = 0; a0 < T; a0 += tile) {
    const int arows = std::min(tile, T - a0);
    dgpp::dsa_absorb_q(ptr<uint16_t>(dq) + static_cast<size_t>(a0) * q_rows, dkvb.p, q_tilde.p,
                       arows, g.heads, g.nope, g.v, g.kv_lora, st, g.rope, /*tensor_cores=*/false);
    dgpp::dsa_attn_partial(q_tilde.p, cache.p, ptr<int32_t>(dreq) + a0,
                           ptr<int32_t>(dtopk) + static_cast<size_t>(a0) * T, T,
                           ptr<int32_t>(dcounts) + a0, arows, n_split, g.heads, g.kv_lora,
                           block_tokens, ptr<int32_t>(dtbl), blocks, g.scale, mptr<float>(m_ws),
                           mptr<float>(l_ws), mptr<float>(c_ws), st, dgpp::LatentFormat::kBf16,
                           nullptr, g.rope);
    dgpp::dsa_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), arows, n_split,
                           g.heads, g.kv_lora, mptr<float>(c), st);
    dgpp::dsa_vout_gemm(c.p, dkvb.p, mptr<uint16_t>(attn) + static_cast<size_t>(a0) * v_rows, arows,
                        g.heads, g.nope, g.v, g.kv_lora, st, /*tensor_cores=*/false);
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const std::vector<uint16_t> got = down<uint16_t>(attn, static_cast<size_t>(T) * v_rows);
  std::vector<float> gf(got.size()), wf(got.size());
  double rms = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    gf[i] = dgpp::bf16_bits_to_float(got[i]);
    wf[i] = dgpp::bf16_bits_to_float(taps.heads[i]);
    rms += static_cast<double>(wf[i]) * wf[i];
  }
  rms = std::sqrt(rms / static_cast<double>(got.size()));
  const Stats s = compare_abs_rel(gf.data(), wf.data(), static_cast<long>(got.size()),
                                  2 * std::pow(2.0, -7.0), 0.005 * rms);
  std::printf("[ .. ] heads' output: max_abs %.3g l2_rel %.3g mismatches %ld/%ld (rms %.3g)\n",
              s.max_abs, s.l2_rel, s.mismatches, s.n, rms);
  require_bf16("mistral4 attention on the latent kernels", s, 2e-3, 0.01);

  // Whether the list-free dense kernel is compiled for this geometry (it
  // attends [0, pos], whose rows below pos0 are zeros here: only the return
  // value is read).
  const bool dense = dgpp::dsa_attn_dense(
      q_tilde.p, cache.p, ptr<int32_t>(dreq), ptr<int64_t>(dpos), 1, n_split, g.heads, g.kv_lora,
      block_tokens, ptr<int32_t>(dtbl), blocks, g.scale, mptr<float>(m_ws), mptr<float>(l_ws),
      mptr<float>(c_ws), st, dgpp::LatentFormat::kBf16, nullptr, g.rope);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  std::printf("[ .. ] dsa_attn_dense at kv_lora %d + rope %d: %s\n", g.kv_lora, g.rope,
              dense ? "launched (the list-free path exists)"
                    : "NOT compiled for this geometry (the layer needs it, or causal lists)");
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;
  return dgpp::test::run_all();
}
