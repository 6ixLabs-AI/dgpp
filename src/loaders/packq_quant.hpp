#pragma once
// The host packed-int encoder (2026-09-12, docs/glm53_plan.md D5): a BF16
// [N, K] matrix into the compressed-tensors `pack-quantized` triple the
// checkpoint carries — symmetric int4 or int8 codes, one bf16 scale per 64
// elements of a row, the codes stored unsigned (offset 2^(bits-1)) 32/bits
// per I32 word with the low nibble / byte the first element — with
// llm-compressor's round-to-nearest recipe (QuantizationModifier,
// memoryless_minmax, symmetric):
//
//   s_g  = bf16( max|w_g| / ((2^bits - 1) / 2) )     (int4: / 7.5, int8: / 127.5)
//   code = clamp( rne(w / s_g), -2^(bits-1), 2^(bits-1) - 1 )
//
// so that the kernels' exact dequant (code x s_g) reproduces w to the
// format's precision. The full GLM-5.3 loader applies it to the draft
// layer's BF16 experts and shared expert (every other packed tensor comes
// quantized from the checkpoint). An all-zero group encodes as zero codes
// under a scale of 1.
#include <cmath>
#include <cstdint>

#include "common/dtypes.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

// The scale of one 64-element group from its absolute maximum.
inline uint16_t packq_group_scale(float amax, int bits) {
  if (!(amax > 0.0f) || !std::isfinite(amax)) return float_to_bf16_bits(1.0f);
  const float half_range = static_cast<float>((1 << bits) - 1) * 0.5f;
  return float_to_bf16_bits(amax / half_range);
}

// Encodes one row of `cols` fp32 values (cols % 64 == 0): `words` receives
// cols * bits / 32 I32 words, `scales` cols / 64 bf16 scales.
inline void packq_encode_row(const float* row, int64_t cols, int bits, uint32_t* words,
                             uint16_t* scales) {
  const int per = 32 / bits;
  const int q_min = -(1 << (bits - 1)), q_max = (1 << (bits - 1)) - 1;
  const uint32_t mask = (1u << bits) - 1u;
  for (int64_t w = 0; w < cols / per; ++w) words[w] = 0u;
  for (int64_t g = 0; g < cols / kPackedGroup; ++g) {
    float amax = 0.0f;
    for (int j = 0; j < kPackedGroup; ++j) {
      const float a = std::fabs(row[g * kPackedGroup + j]);
      amax = a > amax ? a : amax;
    }
    const uint16_t sc = packq_group_scale(amax, bits);
    scales[g] = sc;
    const float inv = 1.0f / bf16_bits_to_float(sc);
    for (int j = 0; j < kPackedGroup; ++j) {
      const int64_t e = g * kPackedGroup + j;
      float q = std::nearbyint(row[e] * inv);
      if (q < static_cast<float>(q_min)) q = static_cast<float>(q_min);
      if (q > static_cast<float>(q_max)) q = static_cast<float>(q_max);
      const uint32_t u = static_cast<uint32_t>(static_cast<int>(q) + (1 << (bits - 1))) & mask;
      words[e / per] |= u << (bits * (e % per));
    }
  }
}

// The absolute maximum of `n` bf16 values.
inline float packq_amax_bf16(const uint16_t* w, int64_t n) {
  float amax = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    const float a = std::fabs(bf16_bits_to_float(w[i]));
    amax = a > amax ? a : amax;
  }
  return amax;
}

// The signed code of element (r, c) of an encoded matrix.
inline int packq_code(const uint32_t* words, int64_t cols, int bits, int64_t r, int64_t c) {
  const int per = 32 / bits;
  const uint32_t word = words[r * (cols / per) + c / per];
  const uint32_t u = (word >> (bits * (c % per))) & ((1u << bits) - 1u);
  return static_cast<int>(u) - (1 << (bits - 1));
}

// Decodes element (r, c) back to the value the kernels compute (code x
// scale, exact in fp32): the tests' oracle and the loader's bf16 bridge.
// scale_fmt selects the group and the scale dtype (kPackedScale*).
inline float packq_decode(const uint32_t* words, const uint16_t* scales, int64_t cols, int bits,
                          int64_t r, int64_t c, int scale_fmt = kPackedScaleBf16G64) {
  const int g = packed_scale_group(scale_fmt);
  return static_cast<float>(packq_code(words, cols, bits, r, c)) *
         packed_scale_to_float(scales[r * (cols / g) + c / g], scale_fmt);
}

}  // namespace dgpp
