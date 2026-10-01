#include "kernels/fp8_gemm.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <stdexcept>
#include <type_traits>

#include "common/cuda_check.hpp"

namespace dgpp {
namespace {

// ---- the quantizer ---------------------------------------------------------
constexpr int kQThreads = 256;  // eight warps: eight (row, group) pairs a block

__device__ __forceinline__ float bf16_bits_to_float_d(uint32_t bits16) {
  return __uint_as_float(bits16 << 16);
}

__global__ __launch_bounds__(kQThreads) void fp8_quantize_rows_kernel(const uint16_t* __restrict__ act,
                                                                     size_t act_stride, int m, int k,
                                                                     uint8_t* __restrict__ q,
                                                                     float* __restrict__ scales) {
  const int groups = k / 128;
  const int gid = blockIdx.x * (kQThreads / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (gid >= m * groups) return;
  const int row = gid / groups, g = gid - row * groups;
  const uint2 raw = *reinterpret_cast<const uint2*>(act + static_cast<size_t>(row) * act_stride + g * 128 + lane * 4);
  float x[4] = {bf16_bits_to_float_d(raw.x & 0xFFFFu), bf16_bits_to_float_d(raw.x >> 16),
                bf16_bits_to_float_d(raw.y & 0xFFFFu), bf16_bits_to_float_d(raw.y >> 16)};
  float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
  const float scale = amax > 0.f ? amax / 448.f : 1.f;
  const __nv_fp8x2_storage_t lo =
      __nv_cvt_float2_to_fp8x2(make_float2(x[0] / scale, x[1] / scale), __NV_SATFINITE, __NV_E4M3);
  const __nv_fp8x2_storage_t hi =
      __nv_cvt_float2_to_fp8x2(make_float2(x[2] / scale, x[3] / scale), __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<uint32_t*>(q + static_cast<size_t>(row) * k + g * 128 + lane * 4) =
      static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
  if (lane == 0) scales[static_cast<size_t>(row) * groups + g] = scale;
}

// ---- the GEMM --------------------------------------------------------------
// 128 x 128 x 64 block tiles on eight warps (2 x 4: a 64 x 32 warp tile of
// four m16 by four n8 mma tiles), a four-stage cp.async pipeline of the
// e4m3 rows (80 KB: one block per SM), both operands k-contiguous in smem
// at an 80-byte row stride (ldmatrix conflict-free), mma m16n8k32 e4m3 into
// a per-group fp32 partial promoted every two k-steps by the row's and the
// column's scale.
constexpr int kBM = 128, kBN = 128, kBK = 64, kGroup = 128;
constexpr int kThreads = 256;
constexpr int kStages = 4;
constexpr int kStride = kBK + 16;                   // smem row bytes
constexpr int kTileBytes = kBM * kStride;           // one operand, one stage
constexpr int kSmem = kStages * 2 * kTileBytes;     // 81,920
static_assert(kBN == kBM, "the copy map covers both operands the same way");
static_assert(kGroup == 2 * kBK, "a scale group is two k-steps");

__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(address), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::); }
template <int N>
__device__ __forceinline__ void wait_pending() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address));
}

template <typename OutT>
__global__ __launch_bounds__(kThreads, 1) void fp8_gemm_kernel(const uint8_t* __restrict__ a,
                                                              const float* __restrict__ a_scales,
                                                              const uint8_t* __restrict__ w,
                                                              const float* __restrict__ w_scales, int sbr,
                                                              OutT* __restrict__ out, int m, int n, int k,
                                                              size_t out_stride) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sa = smem;
  uint8_t* sb = smem + kStages * kTileBytes;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  // The tile order: groups of kGroupM m-tiles walked n-tile by n-tile (a
  // wave's blocks share kGroupM activation tiles from L2 while the weight
  // tiles stream once per group; the plain n-fastest raster streamed a
  // [10240 x 2560] weight once per m-tile — 4.3 ms against 2.2 for cuBLAS).
  constexpr int kGroupM = 8;
  const int m_tiles = (m + kBM - 1) / kBM, n_tiles = (n + kBN - 1) / kBN;
  const int bid = static_cast<int>(blockIdx.x);
  const int group = bid / (kGroupM * n_tiles);
  const int first_m = group * kGroupM;
  const int group_rows = min(kGroupM, m_tiles - first_m);
  const int in_group = bid - group * kGroupM * n_tiles;
  const int m0 = (first_m + in_group % group_rows) * kBM;
  const int n0 = (in_group / group_rows) * kBN;
  const int row_base = (warp / 4) * 64, col_base = (warp % 4) * 32;
  const int steps = k / kBK;
  const int scale_cols = k / kGroup;
  // The copy map: thread -> rows tid / 4 and tid / 4 + 64, 16-byte chunk tid % 4.
  const int crow = tid / 4, cchunk = (tid % 4) * 16;
  const bool av0 = m0 + crow < m, av1 = m0 + crow + 64 < m;
  const bool bv0 = n0 + crow < n, bv1 = n0 + crow + 64 < n;
  const uint8_t* a0 = a + static_cast<size_t>(av0 ? m0 + crow : 0) * k + cchunk;
  const uint8_t* a1 = a + static_cast<size_t>(av1 ? m0 + crow + 64 : 0) * k + cchunk;
  const uint8_t* b0 = w + static_cast<size_t>(bv0 ? n0 + crow : 0) * k + cchunk;
  const uint8_t* b1 = w + static_cast<size_t>(bv1 ? n0 + crow + 64 : 0) * k + cchunk;
  auto issue = [&](int step, int slot) {
    uint8_t* as = sa + slot * kTileBytes;
    uint8_t* bs = sb + slot * kTileBytes;
    const int koff = step * kBK;
    cp16(as + crow * kStride + cchunk, a0 + koff, av0);
    cp16(as + (crow + 64) * kStride + cchunk, a1 + koff, av1);
    cp16(bs + crow * kStride + cchunk, b0 + koff, bv0);
    cp16(bs + (crow + 64) * kStride + cchunk, b1 + koff, bv1);
  };
  const int r = lane / 4, cc = (lane % 4) * 2;
  // The scale rows / columns this thread's accumulators own.
  int arow[4][2];
  bool arow_ok[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = m0 + row_base + i * 16 + r + h * 8;
      arow_ok[i][h] = row < m;
      arow[i][h] = arow_ok[i][h] ? row : 0;
    }
  int wcol[4][2];
  bool wcol_ok[4][2];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int v = 0; v < 2; ++v) {
      const int col = n0 + col_base + j * 8 + cc + v;
      wcol_ok[j][v] = col < n;
      wcol[j][v] = wcol_ok[j][v] ? col / sbr : 0;
    }

  float acc[4][4][4] = {};
  float partial[4][4][4] = {};
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < steps) issue(s, s);
    commit();
  }
  for (int step = 0; step < steps; ++step) {
    wait_pending<kStages - 2>();
    __syncthreads();
    const int nxt = step + kStages - 1;
    if (nxt < steps) issue(nxt, nxt % kStages);
    commit();
    const uint8_t* as = sa + (step % kStages) * kTileBytes;
    const uint8_t* bs = sb + (step % kStages) * kTileBytes;
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 32) {
      uint32_t af[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ldsm_x4(af[i], as + (row_base + i * 16 + lane % 16) * kStride + kk + (lane / 16) * 16);
      uint32_t bf[4][2];
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        // Matrices 0/1 = n8 tile 2jj at k kk..kk+15 / kk+16..kk+31, 2/3 = tile 2jj+1.
        const int tile = col_base + jj * 16 + (lane / 16) * 8 + lane % 8;
        const int kb = kk + ((lane / 8) % 2) * 16;
        uint32_t q4[4];
        ldsm_x4(q4, bs + tile * kStride + kb);
        bf[jj * 2][0] = q4[0];
        bf[jj * 2][1] = q4[1];
        bf[jj * 2 + 1][0] = q4[2];
        bf[jj * 2 + 1][1] = q4[3];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          asm volatile(
              "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(partial[i][j][0]), "+f"(partial[i][j][1]), "+f"(partial[i][j][2]), "+f"(partial[i][j][3])
              : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]));
    }
    if (step % 2 == 1) {
      // A 128-wide group is complete: promote by the row and column scales.
      const int g = step / 2;
      float as_[4][2], ws_[4][2];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          as_[i][h] = arow_ok[i][h] ? a_scales[static_cast<size_t>(arow[i][h]) * scale_cols + g] : 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int v = 0; v < 2; ++v)
          ws_[j][v] = wcol_ok[j][v] ? w_scales[static_cast<size_t>(wcol[j][v]) * scale_cols + g] : 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int v = 0; v < 4; ++v) {
            acc[i][j][v] = __fmaf_rn(as_[i][v / 2] * ws_[j][v % 2], partial[i][j][v], acc[i][j][v]);
            partial[i][j][v] = 0.f;
          }
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int row = m0 + row_base + i * 16 + r + (v / 2) * 8;
        const int col = n0 + col_base + j * 8 + cc + (v % 2);
        if (row < m && col < n) {
          const size_t index = static_cast<size_t>(row) * out_stride + col;
          if constexpr (std::is_same_v<OutT, float>)
            out[index] = acc[i][j][v];
          else
            out[index] = __bfloat16_as_ushort(__float2bfloat16_rn(acc[i][j][v]));
        }
      }
}

template <typename OutT>
void launch(const uint8_t* a, const float* a_scales, const uint8_t* w, const float* w_scales, int sbr, OutT* out,
            int m, int n, int k, cudaStream_t stream, size_t out_stride) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0 || k % kGroup != 0) throw std::invalid_argument("fp8 gemm: k must be a positive multiple of 128");
  if (sbr <= 0) throw std::invalid_argument("fp8 gemm: the weight scale block rows must be positive");
  if (out_stride == 0) out_stride = static_cast<size_t>(n);
  static bool attr_set = false;  // once per OutT
  if (!attr_set) {
    DGPP_CUDA_OK(cudaFuncSetAttribute(fp8_gemm_kernel<OutT>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    attr_set = true;
  }
  const dim3 grid(static_cast<unsigned>(((n + kBN - 1) / kBN) * ((m + kBM - 1) / kBM)));
  fp8_gemm_kernel<OutT><<<grid, kThreads, kSmem, stream>>>(a, a_scales, w, w_scales, sbr, out, m, n, k, out_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace

void launch_fp8_quantize_rows(const uint16_t* act, size_t act_stride, int m, int k, uint8_t* q, float* scales,
                              cudaStream_t stream) {
  if (m <= 0) return;
  if (k <= 0 || k % 128 != 0) throw std::invalid_argument("fp8 quantize: k must be a positive multiple of 128");
  if (act_stride % 4 != 0 || (reinterpret_cast<uintptr_t>(act) % 8) != 0)
    throw std::invalid_argument("fp8 quantize: the activation rows must be 8-byte aligned");
  const int pairs = m * (k / 128);
  const int blocks = (pairs + kQThreads / 32 - 1) / (kQThreads / 32);
  fp8_quantize_rows_kernel<<<blocks, kQThreads, 0, stream>>>(act, act_stride, m, k, q, scales);
  DGPP_CUDA_OK(cudaGetLastError());
}

void launch_fp8_gemm_bf16(const uint8_t* a, const float* a_scales, const uint8_t* w, const float* w_scales,
                          int w_scale_block_rows, uint16_t* out, int m, int n, int k, cudaStream_t stream,
                          size_t out_stride) {
  launch<uint16_t>(a, a_scales, w, w_scales, w_scale_block_rows, out, m, n, k, stream, out_stride);
}
void launch_fp8_gemm_f32(const uint8_t* a, const float* a_scales, const uint8_t* w, const float* w_scales,
                         int w_scale_block_rows, float* out, int m, int n, int k, cudaStream_t stream,
                         size_t out_stride) {
  launch<float>(a, a_scales, w, w_scales, w_scale_block_rows, out, m, n, k, stream, out_stride);
}

}  // namespace dgpp
