#pragma once
// The packed-int GEMV core (2026-09-12, docs/glm53_plan.md D2): the full
// GLM-5.3 checkpoint's int4 (routed experts) and int8 (attention
// projections, shared expert) matrices — compressed-tensors
// `pack-quantized`, symmetric, one bf16 scale per 64 elements along K —
// at m = 1..4 activation rows.
//
// SHAPE: the NVFP4 core's (fp4_gemv.cuh), templated on the code width —
// 16-byte weight chunks per lane (32 int4 codes or 16 int8 codes), a
// whole pass of chunks issued before any is consumed, no barrier in the k
// loop, a fixed xor-shuffle tree. A chunk lies inside ONE 64-wide scale
// group (a group is two int4 chunks or four int8 chunks), so a chunk
// carries exactly one scale. The row geometry is a compile-time function
// of (Bits, K): row_chunks = K * Bits / 128, the largest power of two <=
// 32 that divides it on a row, each lane's chunks consumed in passes of at
// most four. Which lane owns which chunk is a function of (Bits, K) alone,
// and every row's chain is the same sequence of FMAs whatever kRows or
// which launcher issued it — the property the slot path, the grouped path
// and the sliced fold are pinned on, bitwise.
//
// NUMERICS (plan D2, the exact policy): the stored value of element k is
// code_k x s_g exactly, and the dot is computed group-factored,
//   dot = sum_chunks s(chunk) x (sum_{k in chunk} code_k x x_k),
// the inner sum an fp32 FMA chain over the chunk's codes from 0 (every
// product code x x is exact in fp32: an 8-bit integer times a bf16), the
// outer an fp32 FMA per chunk into the row's accumulator. No weight is
// ever rounded to bf16 — the tile kernel (prefill) feeds the tensor core
// the exact integer codes as bf16 and scales each 64-deep k-step's
// partial in fp32, so decode and prefill differ by summation order only,
// the fp8 paths' stance. (Rounding code x s to bf16 per weight, what
// compressed-tensors' decompression produces, would cost an extra
// rounding per weight and ~2x the decode instructions; the exact form is
// both cheaper and closer to the quantizer's intent.)
//
// DECODE: the fp16 magic-number trick. A nibble n ORed into 0x6400 is the
// fp16 value 1024 + n exactly (the ulp at 2^10 is 1); subtracting 1032
// leaves n - 8, the signed code, exactly. Two nibbles land in one f16x2
// (positions 0 and 16 of the word) so the extraction is one shift, one
// LOP3 and one HSUB2 per pair, then one conversion per code. Bytes use
// 0x6400 | b = 1024 + b and subtract 1152.
//
// CONTRACT: K a multiple of the scale group with at most 32 chunks per lane
// (int4 K <= 32768, int8 K <= 16384; the compiled set is dispatch_k's and
// sf_compiled's); packed rows 16-byte aligned. Scales are [n, K/group]
// row-major in the scale format's dtype; a NaN scale propagates as NaN.
//
// SCALE FORMATS (2026-09-28, docs/qwen38_autoround_int4_plan.md D1): the
// template parameter SF picks the group and the scale dtype, nothing else
// — a chunk still lies inside ONE group (128 is a multiple of 32 and 16),
// so the chain, the lane geometry and the reduction are the same.
//   kScaleBf16G64 (0): one bf16 scale per 64 codes (compressed-tensors
//     pack-quantized: full GLM-5.3), the form described above.
//   kScaleF16G128 (1): one IEEE f16 scale per 128 codes (GPTQ / AutoRound
//     int4 g128 and int8 g128: the Qwen3.8 AutoRound hybrid). The
//     checkpoint's scales untouched; f16 -> fp32 is exact.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "common/dtypes.hpp"
#include "kernels/gemv_common.cuh"

namespace dgpp {
namespace packq_gemv {

using gemv::kChunkBytes;
using gemv::kMaxRows;
using gemv::kThreads;
using gemv::kWarps;
using gemv::smem_bytes;
using gemv::stage_activations;

constexpr int kGroup = 64;                        // codes per bf16 scale (format 0)
constexpr int kMaxChunksPerLane = 4;              // chunks in flight per lane per pass
constexpr int kSteps = 1;                         // row steps per warp
constexpr int kMaxPasses = 8;                     // chunk passes per lane

constexpr int kScaleBf16G64 = 0;
constexpr int kScaleF16G128 = 1;
template <int SF>
struct ScaleFmt;
template <>
struct ScaleFmt<kScaleBf16G64> {
  static constexpr int group = 64;
  __device__ __forceinline__ static float to_float(uint16_t b) { return bf16_bits_to_float(b); }
};
template <>
struct ScaleFmt<kScaleF16G128> {
  static constexpr int group = 128;
  __device__ __forceinline__ static float to_float(uint16_t b) {
    return __half2float(__ushort_as_half(b));
  }
};
__host__ __device__ constexpr int scale_group_of(int sf) {
  return sf == kScaleF16G128 ? 128 : 64;
}
__host__ __device__ constexpr bool scale_fmt_known(int sf) {
  return sf == kScaleBf16G64 || sf == kScaleF16G128;
}

template <int Bits, int SF = kScaleBf16G64>
struct Fmt {
  static_assert(Bits == 4 || Bits == 8, "packq_gemv: 4- or 8-bit codes");
  static_assert(scale_fmt_known(SF), "packq_gemv: unknown scale format");
  static constexpr int group = ScaleFmt<SF>::group;
  static constexpr int codes_per_chunk = 8 * kChunkBytes / Bits;   // 32 or 16
  static constexpr int codes_per_word = 32 / Bits;                 // 8 or 4
  static constexpr int chunks_per_group = group / codes_per_chunk; // 2 or 4 (g64), 4 or 8 (g128)
  static constexpr int window_vecs = codes_per_chunk / 8;          // uint4 of bf16 per chunk: 4 or 2
  static constexpr int max_k = codes_per_chunk * 32 * kMaxChunksPerLane * kMaxPasses;
};

__host__ __device__ constexpr int pow2_divisor_le32(int r) {
  int l = 1;
  while (l < 32 && r % (l * 2) == 0) l *= 2;
  return l;
}
template <int Bits>
__host__ __device__ constexpr int row_chunks_of(int k) {
  return k / Fmt<Bits>::codes_per_chunk;
}
template <int Bits>
__host__ __device__ constexpr int lanes_per_row_of(int k) {
  const int r = row_chunks_of<Bits>(k);
  const int c0 = r < kMaxChunksPerLane ? r : kMaxChunksPerLane;
  if (c0 > 0 && r % c0 == 0) {
    const int l = r / c0;
    if (l >= 1 && l <= 32 && (l & (l - 1)) == 0) return l;
  }
  return pow2_divisor_le32(r);
}
template <int Bits>
__host__ __device__ constexpr int chunks_of(int k) {
  return row_chunks_of<Bits>(k) / lanes_per_row_of<Bits>(k);
}
template <int Bits, int SF = kScaleBf16G64>
__host__ __device__ constexpr bool k_supported(int k) {
  return k >= ScaleFmt<SF>::group && k % ScaleFmt<SF>::group == 0 &&
         chunks_of<Bits>(k) <= kMaxChunksPerLane * kMaxPasses;
}
__host__ __device__ constexpr bool k_supported_bits(int bits, int k, int sf = kScaleBf16G64) {
  if (sf == kScaleF16G128)
    return bits == 4 ? k_supported<4, kScaleF16G128>(k)
                     : (bits == 8 ? k_supported<8, kScaleF16G128>(k) : false);
  if (sf != kScaleBf16G64) return false;
  return bits == 4 ? k_supported<4>(k) : (bits == 8 ? k_supported<8>(k) : false);
}

__host__ inline bool shape_ok(const void* packed, int bits, int k, int sf = kScaleBf16G64) {
  return k_supported_bits(bits, k, sf) && gemv::aligned16(packed);
}

// The compile-time row geometry.
template <int Bits, int K, int SF = kScaleBf16G64>
struct Geom {
  static_assert(k_supported<Bits, SF>(K), "packq_gemv: K must be a multiple of the scale group with at most 32 chunks per lane");
  using F = Fmt<Bits, SF>;
  static constexpr int row_chunks = row_chunks_of<Bits>(K);
  static constexpr int lanes_per_row = lanes_per_row_of<Bits>(K);   // 1 .. 32
  static constexpr int chunks = row_chunks / lanes_per_row;         // per lane, over every pass
  static constexpr int passes = (chunks + kMaxChunksPerLane - 1) / kMaxChunksPerLane;
  static constexpr int rows_per_step = 32 / lanes_per_row;
  static constexpr int rows_per_warp = kSteps * rows_per_step;
  static constexpr int rows_per_block = kWarps * rows_per_warp;
  static constexpr int row_bytes = K * Bits / 8;
  static constexpr int scale_cols = K / F::group;
  static constexpr int pass_chunks(int p) {
    const int left = chunks - p * kMaxChunksPerLane;
    return left < 1 ? 1 : (left < kMaxChunksPerLane ? left : kMaxChunksPerLane);
  }
};

// The runtime twins for launch arithmetic.
template <int Bits>
__host__ __device__ constexpr int rows_per_block_of(int k) {
  return kWarps * kSteps * (32 / lanes_per_row_of<Bits>(k));
}

// The activation elements a chunk multiplies, per staged row: window_vecs
// uint4 out of shared memory, loaded once per chunk column.
template <int Bits, int kRows>
__device__ __forceinline__ void load_window(const uint16_t* __restrict__ sx, int k, int e0,
                                            uint4 (&xv)[kRows][Fmt<Bits>::window_vecs]) {
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const uint4* xp = reinterpret_cast<const uint4*>(sx + static_cast<size_t>(r) * k + e0);
#pragma unroll
    for (int q = 0; q < Fmt<Bits>::window_vecs; ++q) xv[r][q] = xp[q];
  }
}

__device__ __forceinline__ __half2 as_half2(uint32_t bits) {
  __half2_raw r;
  r.x = static_cast<unsigned short>(bits & 0xFFFFu);
  r.y = static_cast<unsigned short>(bits >> 16);
  return __half2(r);
}
__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

// One 16-byte chunk (codes [e0, e0 + codes_per_chunk) of a row, its group
// scale s) into kRows accumulators: part[r] = the chunk's exact integer
// dot from 0, then acc[r] += s * part[r]. Element order within the chunk
// is the storage order: word q holds codes e0 + q*codes_per_word + j at
// bit position j*Bits (the low nibble / byte first).
template <int Bits, int kRows>
__device__ __forceinline__ void consume_chunk(const uint4& wchunk, float s,
                                              const uint4 (&xv)[kRows][Fmt<Bits>::window_vecs],
                                              float (&acc)[kRows]) {
  const uint32_t words[4] = {wchunk.x, wchunk.y, wchunk.z, wchunk.w};
  float part[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) part[r] = 0.f;
  if constexpr (Bits == 4) {
    const __half2 bias = __float2half2_rn(1032.f);  // 1024 + 8
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t word = words[q];  // codes 8q .. 8q+7; activations xv[r][q]
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        // Nibbles j and j+4 -> {code j, code j+4} as exact fp16 integers.
        const uint32_t t = ((word >> (4 * j)) & 0x000F000Fu) | 0x64006400u;
        const float2 c = __half22float2(__hsub2(as_half2(t), bias));
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
          // element j: pair j/2, half j%2; element j+4: pair j/2 + 2, half j%2.
          const uint32_t xa = (j < 2) ? xv[r][q].x : xv[r][q].y;
          const uint32_t xb = (j < 2) ? xv[r][q].z : xv[r][q].w;
          const float x0 = (j & 1) ? bf16_hi(xa) : bf16_lo(xa);
          const float x1 = (j & 1) ? bf16_hi(xb) : bf16_lo(xb);
          part[r] = __fmaf_rn(c.x, x0, part[r]);
          part[r] = __fmaf_rn(c.y, x1, part[r]);
        }
      }
    }
  } else {
    const __half2 bias = __float2half2_rn(1152.f);  // 1024 + 128
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint32_t word = words[q];  // codes 4q .. 4q+3; activations xv[r][q/2].{x,y} or .{z,w}
      const uint32_t t0 = (word & 0x00FF00FFu) | 0x64006400u;          // {b0, b2}
      const uint32_t t1 = ((word >> 8) & 0x00FF00FFu) | 0x64006400u;   // {b1, b3}
      const float2 c02 = __half22float2(__hsub2(as_half2(t0), bias));
      const float2 c13 = __half22float2(__hsub2(as_half2(t1), bias));
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        const uint4& v = xv[r][q / 2];
        const uint32_t xw0 = (q & 1) ? v.z : v.x;  // elements 4q, 4q+1
        const uint32_t xw1 = (q & 1) ? v.w : v.y;  // elements 4q+2, 4q+3
        part[r] = __fmaf_rn(c02.x, bf16_lo(xw0), part[r]);
        part[r] = __fmaf_rn(c13.x, bf16_hi(xw0), part[r]);
        part[r] = __fmaf_rn(c02.y, bf16_lo(xw1), part[r]);
        part[r] = __fmaf_rn(c13.y, bf16_hi(xw1), part[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) acc[r] = __fmaf_rn(s, part[r], acc[r]);
}

// Reduce kRows accumulators across a lane group of Lanes lanes (a power of
// two, aligned): the fixed xor tree, launch-shape independent.
template <int kRows, int Lanes>
__device__ __forceinline__ void group_reduce(float (&acc)[kRows]) {
#pragma unroll
  for (int r = 0; r < kRows; ++r)
#pragma unroll
    for (int off = Lanes >> 1; off > 0; off >>= 1)
      acc[r] += __shfl_xor_sync(0xFFFFFFFFu, acc[r], off);
}

// The output policy of one dot (the fp8 core's): bf16 or raw fp32.
__device__ __forceinline__ void store_dot(uint16_t* out, float v) {
  *out = float_to_bf16_bits(v);
}
__device__ __forceinline__ void store_dot(float* out, float v) { *out = v; }

// One pass of a warp's row dots: chunks [C0, C0 + NC) of each lane's row
// (rows [n0 + warp*rows_per_warp, +rows_per_warp) of an [n, K] matrix)
// against kRows staged activation rows, accumulated into acc. Every load
// of the pass (the chunk and its scale) is issued before any is consumed;
// the chunks are then consumed column by column, each row's chunks in k
// order into its own accumulator. No warp-uniform early return.
template <int Bits, int K, int kRows, int C0, int NC, int SF>
__device__ __forceinline__ void pass_row_dots(const uint8_t* __restrict__ w,
                                              const uint16_t* __restrict__ scales,
                                              const uint16_t* __restrict__ sx,
                                              int n0, int n,
                                              float (&acc)[kSteps][kRows]) {
  using G = Geom<Bits, K, SF>;
  using F = Fmt<Bits, SF>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group = lane / G::lanes_per_row;
  const int lig = lane % G::lanes_per_row;
  const int row_base = n0 + warp * G::rows_per_warp;
  uint4 wv[kSteps * NC];
  uint16_t sv[kSteps * NC];
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    const int row = row_base + st * G::rows_per_step + group;
    const bool ok = row < n;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      const int ci = (C0 + c) * G::lanes_per_row + lig;  // the chunk's index in the row
      const int i = st * NC + c;
      wv[i] = ok ? *reinterpret_cast<const uint4*>(
                       w + static_cast<size_t>(row) * G::row_bytes + ci * kChunkBytes)
                 : make_uint4(0u, 0u, 0u, 0u);
      sv[i] = ok ? scales[static_cast<size_t>(row) * G::scale_cols + ci / F::chunks_per_group]
                 : static_cast<uint16_t>(0);
    }
  }
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int ci = (C0 + c) * G::lanes_per_row + lig;
    const int e0 = ci * F::codes_per_chunk;
    uint4 xv[kRows][F::window_vecs];
    load_window<Bits, kRows>(sx, K, e0, xv);
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
      const int row = row_base + st * G::rows_per_step + group;
      if (row >= n) continue;  // per lane; no barrier inside
      const int i = st * NC + c;
      consume_chunk<Bits, kRows>(wv[i], ScaleFmt<SF>::to_float(sv[i]), xv, acc[st]);
    }
  }
}

template <int Bits, int K, int kRows, int P, int SF>
__device__ __forceinline__ void pass_chain(const uint8_t* __restrict__ w,
                                           const uint16_t* __restrict__ scales,
                                           const uint16_t* __restrict__ sx,
                                           int n0, int n,
                                           float (&acc)[kSteps][kRows]) {
  using G = Geom<Bits, K, SF>;
  if constexpr (P < G::passes) {
    pass_row_dots<Bits, K, kRows, P * kMaxChunksPerLane, G::pass_chunks(P), SF>(w, scales, sx, n0, n, acc);
    pass_chain<Bits, K, kRows, P + 1, SF>(w, scales, sx, n0, n, acc);
  }
}

// The dots of a warp's rows_per_warp rows against kRows staged activation
// rows, reduced across each row's lane group. acc[st][a] is this lane's
// row of step st (row_base + st * rows_per_step + group); lane `lig == 0`
// of the group holds the reduced value. Each row's chunks are consumed in
// k order into its own accumulator across the passes, so a row's result
// is the same whatever kRows, the pass split or the launcher.
template <int Bits, int K, int kRows, int SF = kScaleBf16G64>
__device__ __forceinline__ void warp_row_dots(const uint8_t* __restrict__ w,
                                              const uint16_t* __restrict__ scales,
                                              const uint16_t* __restrict__ sx,
                                              int n0, int n,
                                              float (&acc)[kSteps][kRows]) {
  using G = Geom<Bits, K, SF>;
#pragma unroll
  for (int st = 0; st < kSteps; ++st)
#pragma unroll
    for (int a = 0; a < kRows; ++a) acc[st][a] = 0.f;
  pass_chain<Bits, K, kRows, 0, SF>(w, scales, sx, n0, n, acc);
#pragma unroll
  for (int st = 0; st < kSteps; ++st) group_reduce<kRows, G::lanes_per_row>(acc[st]);
}

// One pass of two matrices over the same rows and activations (gate and
// up): every load of both issued before either is consumed. Each
// matrix's row chain is pass_row_dots's exactly (bitwise).
template <int Bits, int K, int kRows, int C0, int NC, int SF>
__device__ __forceinline__ void pass_row_dots_pair(
    const uint8_t* __restrict__ w0, const uint16_t* __restrict__ scales0,
    const uint8_t* __restrict__ w1, const uint16_t* __restrict__ scales1,
    const uint16_t* __restrict__ sx, int n0, int n,
    float (&acc0)[kSteps][kRows], float (&acc1)[kSteps][kRows]) {
  using G = Geom<Bits, K, SF>;
  using F = Fmt<Bits, SF>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int group = lane / G::lanes_per_row;
  const int lig = lane % G::lanes_per_row;
  const int row_base = n0 + warp * G::rows_per_warp;
  uint4 wv0[kSteps * NC], wv1[kSteps * NC];
  uint16_t sv0[kSteps * NC], sv1[kSteps * NC];
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    const int row = row_base + st * G::rows_per_step + group;
    const bool ok = row < n;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      const int ci = (C0 + c) * G::lanes_per_row + lig;
      const int i = st * NC + c;
      const size_t wo = static_cast<size_t>(row) * G::row_bytes + ci * kChunkBytes;
      const size_t so = static_cast<size_t>(row) * G::scale_cols + ci / F::chunks_per_group;
      wv0[i] = ok ? *reinterpret_cast<const uint4*>(w0 + wo) : make_uint4(0u, 0u, 0u, 0u);
      wv1[i] = ok ? *reinterpret_cast<const uint4*>(w1 + wo) : make_uint4(0u, 0u, 0u, 0u);
      sv0[i] = ok ? scales0[so] : static_cast<uint16_t>(0);
      sv1[i] = ok ? scales1[so] : static_cast<uint16_t>(0);
    }
  }
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int ci = (C0 + c) * G::lanes_per_row + lig;
    const int e0 = ci * F::codes_per_chunk;
    uint4 xv[kRows][F::window_vecs];
    load_window<Bits, kRows>(sx, K, e0, xv);
#pragma unroll
    for (int st = 0; st < kSteps; ++st) {
      const int row = row_base + st * G::rows_per_step + group;
      if (row >= n) continue;
      const int i = st * NC + c;
      consume_chunk<Bits, kRows>(wv0[i], ScaleFmt<SF>::to_float(sv0[i]), xv, acc0[st]);
      consume_chunk<Bits, kRows>(wv1[i], ScaleFmt<SF>::to_float(sv1[i]), xv, acc1[st]);
    }
  }
}

template <int Bits, int K, int kRows, int P, int SF>
__device__ __forceinline__ void pass_chain_pair(
    const uint8_t* __restrict__ w0, const uint16_t* __restrict__ scales0,
    const uint8_t* __restrict__ w1, const uint16_t* __restrict__ scales1,
    const uint16_t* __restrict__ sx, int n0, int n,
    float (&acc0)[kSteps][kRows], float (&acc1)[kSteps][kRows]) {
  using G = Geom<Bits, K, SF>;
  if constexpr (P < G::passes) {
    pass_row_dots_pair<Bits, K, kRows, P * kMaxChunksPerLane, G::pass_chunks(P), SF>(
        w0, scales0, w1, scales1, sx, n0, n, acc0, acc1);
    pass_chain_pair<Bits, K, kRows, P + 1, SF>(w0, scales0, w1, scales1, sx, n0, n, acc0, acc1);
  }
}

template <int Bits, int K, int kRows, int SF = kScaleBf16G64>
__device__ __forceinline__ void warp_row_dots_pair(
    const uint8_t* __restrict__ w0, const uint16_t* __restrict__ scales0,
    const uint8_t* __restrict__ w1, const uint16_t* __restrict__ scales1,
    const uint16_t* __restrict__ sx, int n0, int n,
    float (&acc0)[kSteps][kRows], float (&acc1)[kSteps][kRows]) {
  using G = Geom<Bits, K, SF>;
#pragma unroll
  for (int st = 0; st < kSteps; ++st)
#pragma unroll
    for (int a = 0; a < kRows; ++a) acc0[st][a] = acc1[st][a] = 0.f;
  pass_chain_pair<Bits, K, kRows, 0, SF>(w0, scales0, w1, scales1, sx, n0, n, acc0, acc1);
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    group_reduce<kRows, G::lanes_per_row>(acc0[st]);
    group_reduce<kRows, G::lanes_per_row>(acc1[st]);
  }
}

// Which lane owns the reduced dot of the warp's step-st row, and that
// row's index: lane `lig == 0` of its group.
template <int Bits, int K, int SF = kScaleBf16G64>
__device__ __forceinline__ int owned_row(int n0, int st, bool& mine) {
  using G = Geom<Bits, K, SF>;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  mine = (lane % G::lanes_per_row) == 0;
  return n0 + warp * G::rows_per_warp + st * G::rows_per_step + lane / G::lanes_per_row;
}

// The block body: kWarps warps x rows_per_warp rows each of an [n, K]
// packed matrix (payload [n, K*Bits/8] bytes, scales [n, K/group] in the
// scale format's dtype):
//   out[a * out_stride + row] = store_dot(dot(row, sx[a]))
template <int Bits, int K, int kRows, int SF = kScaleBf16G64, typename OutT>
__device__ __forceinline__ void block_rows(const uint8_t* __restrict__ w,
                                           const uint16_t* __restrict__ scales,
                                           const uint16_t* __restrict__ sx,
                                           int n0, int n,
                                           OutT* __restrict__ out,
                                           size_t out_stride) {
  float acc[kSteps][kRows];
  warp_row_dots<Bits, K, kRows, SF>(w, scales, sx, n0, n, acc);
#pragma unroll
  for (int st = 0; st < kSteps; ++st) {
    bool mine = false;
    const int row = owned_row<Bits, K, SF>(n0, st, mine);
    if (mine && row < n) {
#pragma unroll
      for (int a = 0; a < kRows; ++a)
        store_dot(out + static_cast<size_t>(a) * out_stride + row, acc[st][a]);
    }
  }
}

// Dispatch over the compiled K set: f(std::integral_constant<int, K>{}).
// Production widths of the full GLM-5.3 (docs/glm53_plan.md D2): 6144
// (q_a, kv_a, gate/up at every world), 2048 (q_b; the expert and shared
// down at world 1), 1024 / 512 (the down at worlds 2 / 4; kv_b), 16384 /
// 8192 / 4096 (o_proj at worlds 1 / 2 / 4), plus the test geometries; the
// Qwen3.8 AutoRound hybrid's 640 (the expert down) and 2560 (gate / up,
// the int8 head) (docs/qwen38_autoround_int4_plan.md).
template <typename F>
__host__ inline void dispatch_k(int k, F&& f) {
  switch (k) {
    case 64: f(std::integral_constant<int, 64>{}); return;
    case 128: f(std::integral_constant<int, 128>{}); return;
    case 192: f(std::integral_constant<int, 192>{}); return;
    case 256: f(std::integral_constant<int, 256>{}); return;
    case 384: f(std::integral_constant<int, 384>{}); return;
    case 512: f(std::integral_constant<int, 512>{}); return;
    case 640: f(std::integral_constant<int, 640>{}); return;
    case 768: f(std::integral_constant<int, 768>{}); return;
    case 1024: f(std::integral_constant<int, 1024>{}); return;
    case 1536: f(std::integral_constant<int, 1536>{}); return;
    case 2048: f(std::integral_constant<int, 2048>{}); return;
    case 2560: f(std::integral_constant<int, 2560>{}); return;
    case 3072: f(std::integral_constant<int, 3072>{}); return;
    case 4096: f(std::integral_constant<int, 4096>{}); return;
    case 6144: f(std::integral_constant<int, 6144>{}); return;
    case 8192: f(std::integral_constant<int, 8192>{}); return;
    case 12288: f(std::integral_constant<int, 12288>{}); return;
    case 16384: f(std::integral_constant<int, 16384>{}); return;
    default:
      throw std::invalid_argument(
          "packq_gemv: K is not in the compiled set (64, 128, 192, 256, 384, 512, 640, 768, "
          "1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192, 12288, 16384)");
  }
}
__host__ __device__ constexpr bool k_compiled(int k) {
  return k == 64 || k == 128 || k == 192 || k == 256 || k == 384 || k == 512 || k == 640 ||
         k == 768 || k == 1024 || k == 1536 || k == 2048 || k == 2560 || k == 3072 ||
         k == 4096 || k == 6144 || k == 8192 || k == 12288 || k == 16384;
}
// The instantiated (scale format, K) pairs: format 0 at every compiled K;
// format 1 (f16 per 128) at the 128-multiples up to 2560 — the AutoRound
// hybrid's widths and the test geometries below them. Every launcher
// checks this before its dispatch; a width outside it throws.
__host__ __device__ constexpr bool sf_compiled(int sf, int k) {
  return sf == kScaleBf16G64 ? k_compiled(k)
                             : (sf == kScaleF16G128 && k_compiled(k) && k % 128 == 0 && k <= 2560);
}
// Dispatch over the scale format: f(std::integral_constant<int, SF>{}).
template <typename F>
__host__ inline void dispatch_sf(int sf, F&& f) {
  if (sf == kScaleBf16G64) f(std::integral_constant<int, kScaleBf16G64>{});
  else if (sf == kScaleF16G128) f(std::integral_constant<int, kScaleF16G128>{});
  else throw std::invalid_argument("packq_gemv: the scale format must be 0 (bf16 per 64) or 1 (f16 per 128)");
}
// Dispatch over the code width: f(std::integral_constant<int, Bits>{}).
template <typename F>
__host__ inline void dispatch_bits(int bits, F&& f) {
  if (bits == 4) f(std::integral_constant<int, 4>{});
  else if (bits == 8) f(std::integral_constant<int, 8>{});
  else throw std::invalid_argument("packq_gemv: the code width must be 4 or 8");
}

}  // namespace packq_gemv
}  // namespace dgpp
