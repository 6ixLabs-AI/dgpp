// Dense causal paged attention for the Qwen3.8-27B full-attention layers, in
// two forms over the same paged K/V cache (bf16 [slots, kv_heads * 256]):
//
//  * The row form (decode): one warp per (query row, KV group) — the group's
//    <= 16 query heads are the M rows of m16n8k16 A-fragments, K/V tiles of
//    16 tokens gathered into shared memory with cp.async, S = Q K^T and
//    O += P V on the tensor cores, P reused from S's accumulator layout
//    (FlashAttention-2). A row's keys are split over blockIdx.z ranges so a
//    long context walks in parallel (kMaxSplits warps per (row, group)); each
//    split leaves an unnormalized partial (m, l, acc) and a combine pass
//    merges them. splits = 1 writes the normalized fp32 output directly.
//
//  * The tile form (prefill): one CTA per (16 consecutive query rows of one
//    request, KV group), one warp per query head of the group — 16 queries
//    are the M rows, so every mma row carries a query — over 32-token K/V
//    tiles staged once per CTA and read by all of the group's warps. Rows
//    attend [0, pos[r]] through the per-row mask; the CTA walks to the
//    tile's furthest position.
//
// Numerics (both forms): probabilities rounded to bf16 for P V, the
// denominator summing the unrounded values (the QSA rule); the dots'
// summation order is the tensor cores'. Tolerance-equal to a dense
// reference, not bitwise, and the two forms differ from each other in their
// rescaling points (16- vs 32-token tiles, the split boundaries).
#include "kernels/full_attn.hpp"

#include <cmath>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

constexpr int kD = 256;
constexpr int kRow = kD + 8;  // padded bf16 row: conflict-free ldmatrix

// The row form.
constexpr int kTileTok = 16;
constexpr int kRowStageElems = 2 * kTileTok * kRow;  // K then V
constexpr size_t kRowSmem = size_t(kRowStageElems) * 2;  // bytes, one stage
constexpr int kMaxSplits = 32;
constexpr int kPartialFloats = 16 * kD + 32;  // acc[16][256], m[16], l[16]

// The tile form.
constexpr int kQTile = 16;
constexpr int kKTile = 32;
constexpr int kMaxTileGroup = 8;  // warps per CTA (255 registers per thread)
constexpr int kTileStageElems = 2 * kKTile * kRow;
constexpr size_t kTileSmem = size_t(2) * kTileStageElems * 2;  // two stages, bytes

__device__ __forceinline__ void fldsm_x4(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void fldsm_x4_t(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void fmma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void fcp_async16(void* dst, const void* src, bool valid) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const int bytes = valid ? 16 : 0;  // 0: zero-fill
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(bytes));
}
__device__ __forceinline__ uint32_t fpack_bf16(float lo, float hi) {
  return static_cast<uint32_t>(float_to_bf16_bits(lo)) | (static_cast<uint32_t>(float_to_bf16_bits(hi)) << 16);
}
// exp2 of a masked score against a running max: a fully masked row keeps
// its running max at -inf, and contributes nothing.
__device__ __forceinline__ float fpexp(float s, float n) { return n == -INFINITY ? 0.f : exp2f(s - n); }

// ---- the row form ---------------------------------------------------------------

__global__ __launch_bounds__(32) void full_attn_row_kernel(
    const uint16_t* __restrict__ q, int64_t q_row_stride, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const int32_t* __restrict__ req_ids, const int64_t* __restrict__ pos,
    int local_heads, int kv_heads, int block_tokens, const int32_t* __restrict__ block_tables,
    int blocks_per_request, float scale, float* __restrict__ out, float* __restrict__ partials, int splits) {
  extern __shared__ __align__(16) uint16_t sm[];
  const int64_t r = blockIdx.x;
  const int kvh = blockIdx.y;
  const int split = blockIdx.z;
  const int group = local_heads / kv_heads;
  const int h0 = kvh * group;
  const int lane = threadIdx.x;
  const int g = lane >> 2, t = lane & 3;
  const int width = kv_heads * kD;
  const int64_t cnt = pos[r] + 1;  // dense causal visible set [0, pos[r]]
  const int32_t* bt = block_tables + static_cast<int64_t>(req_ids[r]) * blocks_per_request;
  const float sl2 = scale * 1.4426950408889634f;
  // This split's key range: tile-aligned chunks of ceil(cnt / splits).
  const int64_t chunk = ((cnt + splits - 1) / splits + kTileTok - 1) / kTileTok * kTileTok;
  const int64_t begin = static_cast<int64_t>(split) * chunk;
  const int64_t end = min(cnt, begin + chunk);

  // Q A-fragments, 16 k-steps x 4 registers; rows past the group are zero.
  uint32_t qa[16][4];
  {
    const uint16_t* q0 = q + r * q_row_stride + static_cast<int64_t>(h0) * kD;
    const bool v0 = g < group, v1 = g + 8 < group;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      const int c = ks * 16 + 2 * t;
      qa[ks][0] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * kD + c) : 0u;
      qa[ks][1] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * kD + c) : 0u;
      qa[ks][2] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * kD + c + 8) : 0u;
      qa[ks][3] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * kD + c + 8) : 0u;
    }
  }

  // Stage a tile: 16 tokens x (K, V) x 32 16-byte chunks = 1024 chunks, 32 per lane.
  // Lanes 0..15 resolve the tile's 16 physical rows in one round of loads.
  auto stage = [&](int64_t base) {
    uint16_t* kt = sm;
    uint16_t* vt = kt + kTileTok * kRow;
    int64_t my_off = 0;
    if (lane < kTileTok && base + lane < end) {
      const int64_t tok = base + lane;
      const int64_t phys = static_cast<int64_t>(bt[tok / block_tokens]) * block_tokens + tok % block_tokens;
      my_off = phys * width + static_cast<int64_t>(kvh) * kD;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int64_t off = __shfl_sync(0xffffffffu, my_off, i) + lane * 8;
      const bool valid = base + i < end;
      fcp_async16(kt + i * kRow + lane * 8, k_cache + off, valid);
      fcp_async16(vt + i * kRow + lane * 8, v_cache + off, valid);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  float acc[32][4];
#pragma unroll
  for (int d = 0; d < 32; ++d) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;  // rows g, g + 8 (lane-partial l)

  for (int64_t base = begin; base < end; base += kTileTok) {
    stage(base);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncwarp();
    const uint16_t* kt = sm;
    const uint16_t* vt = kt + kTileTok * kRow;
    float sc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      uint32_t b[4];
      fldsm_x4(b, kt + ((lane & 7) + ((lane >> 4) << 3)) * kRow + ks * 16 + ((lane >> 3) & 1) * 8);
      fmma_bf16(sc[0], qa[ks], b[0], b[1]);
      fmma_bf16(sc[1], qa[ks], b[2], b[3]);
    }
    const int n = min(kTileTok, static_cast<int>(end - base));
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int col = j * 8 + 2 * t + e;
        sc[j][e] = col < n ? sc[j][e] * sl2 : -INFINITY;
        sc[j][2 + e] = col < n ? sc[j][2 + e] * sl2 : -INFINITY;
        mx0 = fmaxf(mx0, sc[j][e]);
        mx1 = fmaxf(mx1, sc[j][2 + e]);
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float a0 = fpexp(m0, n0), a1 = fpexp(m1, n1);  // m = -inf on the first tile: 0
    m0 = n0;
    m1 = n1;
    float p[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      p[j][0] = fpexp(sc[j][0], n0);
      p[j][1] = fpexp(sc[j][1], n0);
      p[j][2] = fpexp(sc[j][2], n1);
      p[j][3] = fpexp(sc[j][3], n1);
    }
    l0 = l0 * a0 + p[0][0] + p[0][1] + p[1][0] + p[1][1];
    l1 = l1 * a1 + p[0][2] + p[0][3] + p[1][2] + p[1][3];
    const uint32_t pa[4] = {fpack_bf16(p[0][0], p[0][1]), fpack_bf16(p[0][2], p[0][3]),
                            fpack_bf16(p[1][0], p[1][1]), fpack_bf16(p[1][2], p[1][3])};
#pragma unroll
    for (int d = 0; d < 32; ++d) {
      acc[d][0] *= a0;
      acc[d][1] *= a0;
      acc[d][2] *= a1;
      acc[d][3] *= a1;
    }
#pragma unroll
    for (int dp = 0; dp < 16; ++dp) {
      uint32_t b[4];
      fldsm_x4_t(b, vt + ((lane & 7) + ((lane >> 3) & 1) * 8) * kRow + dp * 16 + (lane >> 4) * 8);
      fmma_bf16(acc[dp * 2], pa, b[0], b[1]);
      fmma_bf16(acc[dp * 2 + 1], pa, b[2], b[3]);
    }
    __syncwarp();  // this buffer's reads done before the next stage() overwrites it
  }
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  if (partials == nullptr) {
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    float* o0 = out + (r * local_heads + h0 + g) * kD;
    float* o1 = out + (r * local_heads + h0 + g + 8) * kD;
#pragma unroll
    for (int d = 0; d < 32; ++d) {
      const int col = d * 8 + 2 * t;
      if (g < group) *reinterpret_cast<float2*>(o0 + col) = make_float2(acc[d][0] * i0, acc[d][1] * i0);
      if (g + 8 < group) *reinterpret_cast<float2*>(o1 + col) = make_float2(acc[d][2] * i1, acc[d][3] * i1);
    }
    return;
  }
  // The split's unnormalized partial: acc rows g / g + 8 at their running maxima.
  float* part = partials + ((r * kv_heads + kvh) * static_cast<int64_t>(splits) + split) * kPartialFloats;
  float* pa0 = part + g * kD;
  float* pa1 = part + (g + 8) * kD;
#pragma unroll
  for (int d = 0; d < 32; ++d) {
    const int col = d * 8 + 2 * t;
    *reinterpret_cast<float2*>(pa0 + col) = make_float2(acc[d][0], acc[d][1]);
    *reinterpret_cast<float2*>(pa1 + col) = make_float2(acc[d][2], acc[d][3]);
  }
  if (t == 0) {
    part[16 * kD + g] = m0;
    part[16 * kD + g + 8] = m1;
    part[16 * kD + 16 + g] = l0;
    part[16 * kD + 16 + g + 8] = l1;
  }
}

// One block per (row, KV group): every head's 256 outputs from the splits'
// (m, l, acc) — the largest maximum, the rescaled denominators, the sum.
__global__ __launch_bounds__(256) void full_attn_combine_kernel(const float* __restrict__ partials,
                                                                 int splits, int local_heads, int kv_heads,
                                                                 float* __restrict__ out) {
  const int64_t r = blockIdx.x;
  const int kvh = blockIdx.y;
  const int group = local_heads / kv_heads;
  const int d = threadIdx.x;
  const float* base = partials + (r * kv_heads + kvh) * static_cast<int64_t>(splits) * kPartialFloats;
  for (int h = 0; h < group; ++h) {
    float mt = -INFINITY;
    for (int s = 0; s < splits; ++s) mt = fmaxf(mt, base[s * kPartialFloats + 16 * kD + h]);
    float lt = 0.f, o = 0.f;
    if (mt != -INFINITY) {
      for (int s = 0; s < splits; ++s) {
        const float* p = base + s * kPartialFloats;
        const float ms = p[16 * kD + h];
        if (ms == -INFINITY) continue;
        const float w = exp2f(ms - mt);
        lt += p[16 * kD + 16 + h] * w;
        o += p[h * kD + d] * w;
      }
    }
    out[(r * local_heads + kvh * group + h) * kD + d] = lt > 0.f ? o / lt : 0.f;
  }
}

// ---- the tile form -------------------------------------------------------------

__global__ __launch_bounds__(kMaxTileGroup * 32) void full_attn_tile_kernel(
    const uint16_t* __restrict__ q, int64_t q_row_stride, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const int32_t* __restrict__ bt, const int64_t* __restrict__ pos,
    int rows, int local_heads, int kv_heads, int block_tokens, float scale, float* __restrict__ out) {
  extern __shared__ __align__(16) uint16_t sm[];
  __shared__ int64_t s_off[2][kKTile];
  const int r0 = blockIdx.x * kQTile;
  const int kvh = blockIdx.y;
  const int group = local_heads / kv_heads;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int head = kvh * group + warp;
  const int width = kv_heads * kD;
  const int nthreads = blockDim.x;
  const float sl2 = scale * 1.4426950408889634f;
  // The tile's rows and their positions; the walk covers the furthest one.
  const int rg = r0 + g, rg8 = r0 + g + 8;
  const int64_t pg = rg < rows ? pos[rg] : -1, pg8 = rg8 < rows ? pos[rg8] : -1;
  int64_t cnt = 0;
  {
    int64_t p = lane < kQTile && r0 + lane < rows ? pos[r0 + lane] + 1 : 0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) p = max(p, __shfl_xor_sync(0xffffffffu, p, o));
    cnt = p;
  }
  // Q A-fragments: this warp's head, the tile's 16 query rows.
  uint32_t qa[16][4];
  {
    const uint16_t* q0 = q + static_cast<int64_t>(r0) * q_row_stride + static_cast<int64_t>(head) * kD;
    const bool v0 = rg < rows, v1 = rg8 < rows;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      const int c = ks * 16 + 2 * t;
      qa[ks][0] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * q_row_stride + c) : 0u;
      qa[ks][1] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * q_row_stride + c) : 0u;
      qa[ks][2] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * q_row_stride + c + 8) : 0u;
      qa[ks][3] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * q_row_stride + c + 8) : 0u;
    }
  }
  // Stage a 32-token tile into buffer `buf`: warp 0 resolves the physical
  // rows (one lane per token), every thread copies its share of the
  // 2 x 32 x 32 16-byte chunks. Callers bracket with __syncthreads.
  auto resolve = [&](int64_t base, int buf) {
    if (warp == 0) {
      int64_t off = 0;
      const int64_t tok = base + lane;
      if (tok < cnt) {
        const int64_t phys = static_cast<int64_t>(bt[tok / block_tokens]) * block_tokens + tok % block_tokens;
        off = phys * width + static_cast<int64_t>(kvh) * kD;
      }
      s_off[buf][lane] = off;
    }
  };
  auto stage = [&](int64_t base, int buf) {
    uint16_t* kt = sm + buf * kTileStageElems;
    uint16_t* vt = kt + kKTile * kRow;
    for (int c = threadIdx.x; c < 2 * kKTile * 32; c += nthreads) {
      const int plane = c >> 10;        // 0: K, 1: V
      const int i = (c >> 5) & 31;      // token
      const int ch = c & 31;            // 16-byte chunk of the 256-dim row
      const bool valid = base + i < cnt;
      const int64_t off = s_off[buf][i] + ch * 8;
      uint16_t* dst = (plane ? vt : kt) + i * kRow + ch * 8;
      fcp_async16(dst, (plane ? v_cache : k_cache) + off, valid);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  float acc[32][4];
#pragma unroll
  for (int d = 0; d < 32; ++d) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  const int64_t tiles = (cnt + kKTile - 1) / kKTile;
  if (tiles > 0) {
    resolve(0, 0);
    __syncthreads();
    stage(0, 0);
  }
  for (int64_t it = 0; it < tiles; ++it) {
    const int buf = static_cast<int>(it & 1);
    if (it + 1 < tiles) {
      resolve((it + 1) * kKTile, buf ^ 1);
      __syncthreads();
      stage((it + 1) * kKTile, buf ^ 1);
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::);
    }
    __syncthreads();
    const uint16_t* kt = sm + buf * kTileStageElems;
    const uint16_t* vt = kt + kKTile * kRow;
    const int64_t base = it * kKTile;
    // S = Q K^T over the 32 tokens: four n8 tiles.
    float sc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      uint32_t b[4];
      const uint16_t* kp = kt + ((lane & 7) + ((lane >> 4) << 3)) * kRow + ks * 16 + ((lane >> 3) & 1) * 8;
      fldsm_x4(b, kp);
      fmma_bf16(sc[0], qa[ks], b[0], b[1]);
      fmma_bf16(sc[1], qa[ks], b[2], b[3]);
      fldsm_x4(b, kp + 16 * kRow);
      fmma_bf16(sc[2], qa[ks], b[0], b[1]);
      fmma_bf16(sc[3], qa[ks], b[2], b[3]);
    }
    // The causal mask per row (key <= pos[r]) and the tile bound.
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int64_t key = base + j * 8 + 2 * t + e;
        sc[j][e] = key <= pg ? sc[j][e] * sl2 : -INFINITY;
        sc[j][2 + e] = key <= pg8 ? sc[j][2 + e] * sl2 : -INFINITY;
        mx0 = fmaxf(mx0, sc[j][e]);
        mx1 = fmaxf(mx1, sc[j][2 + e]);
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float a0 = fpexp(m0, n0), a1 = fpexp(m1, n1);
    m0 = n0;
    m1 = n1;
    float p[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      p[j][0] = fpexp(sc[j][0], n0);
      p[j][1] = fpexp(sc[j][1], n0);
      p[j][2] = fpexp(sc[j][2], n1);
      p[j][3] = fpexp(sc[j][3], n1);
      l0 = (j == 0 ? l0 * a0 : l0) + p[j][0] + p[j][1];
      l1 = (j == 0 ? l1 * a1 : l1) + p[j][2] + p[j][3];
    }
    const uint32_t pa[4] = {fpack_bf16(p[0][0], p[0][1]), fpack_bf16(p[0][2], p[0][3]),
                            fpack_bf16(p[1][0], p[1][1]), fpack_bf16(p[1][2], p[1][3])};
    const uint32_t pb[4] = {fpack_bf16(p[2][0], p[2][1]), fpack_bf16(p[2][2], p[2][3]),
                            fpack_bf16(p[3][0], p[3][1]), fpack_bf16(p[3][2], p[3][3])};
#pragma unroll
    for (int d = 0; d < 32; ++d) {
      acc[d][0] *= a0;
      acc[d][1] *= a0;
      acc[d][2] *= a1;
      acc[d][3] *= a1;
    }
    // O += P V: tokens 0-15 then 16-31, 16 x4.trans loads each.
#pragma unroll
    for (int dp = 0; dp < 16; ++dp) {
      uint32_t b[4];
      const uint16_t* vp = vt + ((lane & 7) + ((lane >> 3) & 1) * 8) * kRow + dp * 16 + (lane >> 4) * 8;
      fldsm_x4_t(b, vp);
      fmma_bf16(acc[dp * 2], pa, b[0], b[1]);
      fmma_bf16(acc[dp * 2 + 1], pa, b[2], b[3]);
      fldsm_x4_t(b, vp + 16 * kRow);
      fmma_bf16(acc[dp * 2], pb, b[0], b[1]);
      fmma_bf16(acc[dp * 2 + 1], pb, b[2], b[3]);
    }
    __syncthreads();  // this buffer's reads done before the stage after next overwrites it
  }
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
  float* o0 = out + (static_cast<int64_t>(rg) * local_heads + head) * kD;
  float* o1 = out + (static_cast<int64_t>(rg8) * local_heads + head) * kD;
#pragma unroll
  for (int d = 0; d < 32; ++d) {
    const int col = d * 8 + 2 * t;
    if (rg < rows) *reinterpret_cast<float2*>(o0 + col) = make_float2(acc[d][0] * i0, acc[d][1] * i0);
    if (rg8 < rows) *reinterpret_cast<float2*>(o1 + col) = make_float2(acc[d][2] * i1, acc[d][3] * i1);
  }
}

void set_attrs_once() {
  static const bool opted = [] {  // once per process, thread-safe (TP ranks may share one)
    DGPP_CUDA_OK(cudaFuncSetAttribute(full_attn_row_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(kRowSmem)));
    DGPP_CUDA_OK(cudaFuncSetAttribute(full_attn_row_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    DGPP_CUDA_OK(cudaFuncSetAttribute(full_attn_tile_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(kTileSmem)));
    DGPP_CUDA_OK(cudaFuncSetAttribute(full_attn_tile_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    return true;
  }();
  (void)opted;
}

}  // namespace

bool full_attn_supported(int dim, int local_heads, int kv_heads) {
  return dim == kD && kv_heads > 0 && local_heads % kv_heads == 0 && local_heads / kv_heads <= 16;
}

int full_attn_max_splits() { return kMaxSplits; }

size_t full_attn_partials_bytes(int rows, int kv_heads) {
  return static_cast<size_t>(rows) * kv_heads * kMaxSplits * kPartialFloats * sizeof(float);
}

void full_attn_decode(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                      const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos, int rows,
                      int local_heads, int kv_heads, int block_tokens, const int32_t* block_tables,
                      int blocks_per_request, float scale, float* out, float* partials,
                      cudaStream_t stream) {
  if (rows <= 0) return;
  if (!full_attn_supported(kD, local_heads, kv_heads))
    throw std::invalid_argument("full_attn_decode: dim 256, <= 16 query heads per KV head");
  set_attrs_once();
  const int splits = partials != nullptr ? kMaxSplits : 1;
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(kv_heads), static_cast<unsigned>(splits));
  full_attn_row_kernel<<<grid, 32, kRowSmem, stream>>>(q, q_row_stride, k_cache, v_cache, req_ids, pos,
                                                       local_heads, kv_heads, block_tokens, block_tables,
                                                       blocks_per_request, scale, out, partials, splits);
  DGPP_CUDA_OK(cudaGetLastError());
  if (splits > 1) {
    const dim3 cgrid(static_cast<unsigned>(rows), static_cast<unsigned>(kv_heads));
    full_attn_combine_kernel<<<cgrid, 256, 0, stream>>>(partials, splits, local_heads, kv_heads, out);
    DGPP_CUDA_OK(cudaGetLastError());
  }
}

void full_attn_prefill_warp(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                            const uint16_t* v_cache, const int32_t* req_ids, const int64_t* pos,
                            int rows, int local_heads, int kv_heads, int block_tokens,
                            const int32_t* block_tables, int blocks_per_request, float scale,
                            float* out, cudaStream_t stream) {
  full_attn_decode(q, q_row_stride, k_cache, v_cache, req_ids, pos, rows, local_heads, kv_heads, block_tokens,
                   block_tables, blocks_per_request, scale, out, nullptr, stream);
}

void full_attn_prefill(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                       const uint16_t* v_cache, const int32_t* block_table, const int32_t* req_ids,
                       const int64_t* pos, int rows, int local_heads, int kv_heads, int block_tokens,
                       const int32_t* block_tables, int blocks_per_request, float scale, float* out,
                       cudaStream_t stream) {
  if (rows <= 0) return;
  if (!full_attn_supported(kD, local_heads, kv_heads))
    throw std::invalid_argument("full_attn_prefill: dim 256, <= 16 query heads per KV head");
  const int group = local_heads / kv_heads;
  if (group > kMaxTileGroup) {  // wider groups keep the row form
    full_attn_decode(q, q_row_stride, k_cache, v_cache, req_ids, pos, rows, local_heads, kv_heads, block_tokens,
                     block_tables, blocks_per_request, scale, out, nullptr, stream);
    return;
  }
  set_attrs_once();
  const dim3 grid(static_cast<unsigned>((rows + kQTile - 1) / kQTile), static_cast<unsigned>(kv_heads));
  full_attn_tile_kernel<<<grid, group * 32, kTileSmem, stream>>>(q, q_row_stride, k_cache, v_cache, block_table,
                                                                  pos, rows, local_heads, kv_heads, block_tokens,
                                                                  scale, out);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
