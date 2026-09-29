#pragma once
// The GPTQ -> packed-core repack (2026-09-28, docs/qwen38_autoround_int4_plan.md
// D2). GPTQ stores a logical [N, K] matrix as qweight I32 [K*bits/32, N]:
// word (i, n) holds codes k = i*(32/bits) + j at bit j*bits, the low nibble
// / byte first, plus scales F16 [K/g, N] and qzeros I32 [K/g, N*bits/32].
// The packed core (kernels/packq_gemv.cuh) reads I32 [N, K*bits/32] and
// scales [N, K/g] — the same words and the same scales, transposed. The
// packing order inside a word is identical (compressed-tensors' and GPTQ's
// agree), and so are the offset-code semantics (GPTQ v1 stores zero - 1 =
// 7 / 127 in qzeros and dequantizes (q - 8) * s, the packed core's fixed
// offset), so the transpose is the whole repack: no code changes value, no
// scale is rounded. The zeros are checked against the symmetric constant
// and never become resident.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace dgpp {

// dst[c * rows + r] = src[r * src_stride + c] for r < rows, c < cols (a
// 32 x 32 tiled transpose; dst is [cols, rows]).
template <typename T>
inline void transpose_tiled(const T* src, int64_t src_stride, int64_t rows, int64_t cols, T* dst) {
  constexpr int64_t kTile = 32;
  for (int64_t r0 = 0; r0 < rows; r0 += kTile) {
    const int64_t r1 = std::min(rows, r0 + kTile);
    for (int64_t c0 = 0; c0 < cols; c0 += kTile) {
      const int64_t c1 = std::min(cols, c0 + kTile);
      for (int64_t r = r0; r < r1; ++r) {
        const T* s = src + r * src_stride;
        for (int64_t c = c0; c < c1; ++c) dst[c * rows + r] = s[c];
      }
    }
  }
}

inline uint32_t gptq_symmetric_zero_word(int bits) { return bits == 4 ? 0x77777777u : 0x7F7F7F7Fu; }

// One GPTQ matrix's slice — the N range [n0, n0 + n) (a row slice of the
// logical matrix) and the K range [word_row0, +word_rows) words / [group0,
// +groups) scale rows (a column slice) — into the packed layout.
struct GptqRepackJob {
  const uint32_t* qweight = nullptr;  // [K*bits/32, N], row stride qw_stride words
  int64_t qw_stride = 0;
  int64_t word_row0 = 0, word_rows = 0;
  int64_t n0 = 0, n = 0;
  const uint16_t* scales = nullptr;   // [K/g, N], row stride sc_stride
  int64_t sc_stride = 0;
  int64_t group0 = 0, groups = 0;
  const uint32_t* qzeros = nullptr;   // [K/g, N*bits/32], row stride qz_stride (null: unchecked)
  int64_t qz_stride = 0;
  int bits = 4;
  uint32_t* dst_words = nullptr;      // [n, word_rows]
  uint16_t* dst_scales = nullptr;     // [n, groups]
};

// Runs one job. Returns false when a zero word in the slice is not the
// symmetric constant (the fixed-offset decode would be wrong).
inline bool gptq_repack_run(const GptqRepackJob& j) {
  transpose_tiled(j.qweight + j.word_row0 * j.qw_stride + j.n0, j.qw_stride, j.word_rows, j.n,
                  j.dst_words);
  transpose_tiled(j.scales + j.group0 * j.sc_stride + j.n0, j.sc_stride, j.groups, j.n, j.dst_scales);
  if (j.qzeros != nullptr) {
    const int per = 32 / j.bits;
    const uint32_t want = gptq_symmetric_zero_word(j.bits);
    const int64_t z0 = j.n0 / per, z1 = (j.n0 + j.n + per - 1) / per;
    for (int64_t g = 0; g < j.groups; ++g) {
      const uint32_t* row = j.qzeros + (j.group0 + g) * j.qz_stride;
      for (int64_t c = z0; c < z1; ++c)
        if (row[c] != want) return false;
    }
  }
  return true;
}

// The same slice as `parts` column ranges (their outputs are contiguous
// [n_i, word_rows] blocks of the job's), for one large matrix across threads.
inline std::vector<GptqRepackJob> gptq_repack_split(const GptqRepackJob& j, int parts) {
  std::vector<GptqRepackJob> out;
  const int64_t step = std::max<int64_t>(1, (j.n + parts - 1) / parts);
  for (int64_t c = 0; c < j.n; c += step) {
    GptqRepackJob p = j;
    p.n0 = j.n0 + c;
    p.n = std::min(step, j.n - c);
    p.dst_words = j.dst_words + c * j.word_rows;
    p.dst_scales = j.dst_scales + c * j.groups;
    out.push_back(p);
  }
  return out;
}

// Runs every job across `threads` workers. Returns the index of the first
// job (in order) whose zeros failed the check, or -1.
inline int64_t gptq_repack_all(const std::vector<GptqRepackJob>& jobs, int threads) {
  if (jobs.empty()) return -1;
  std::atomic<size_t> next{0};
  std::atomic<int64_t> bad{-1};
  auto worker = [&] {
    for (size_t i = next.fetch_add(1); i < jobs.size(); i = next.fetch_add(1)) {
      if (!gptq_repack_run(jobs[i])) {
        int64_t cur = bad.load();
        while ((cur < 0 || static_cast<int64_t>(i) < cur) &&
               !bad.compare_exchange_weak(cur, static_cast<int64_t>(i))) {
        }
      }
    }
  };
  const int n = std::max(1, std::min<int>(threads, static_cast<int>(jobs.size())));
  std::vector<std::thread> pool;
  for (int t = 1; t < n; ++t) pool.emplace_back(worker);
  worker();
  for (auto& t : pool) t.join();
  return bad.load();
}

}  // namespace dgpp
