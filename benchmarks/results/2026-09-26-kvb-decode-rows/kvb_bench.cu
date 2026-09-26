// kvb_bench: full GLM-5.3's decode absorb_q / vout kernels on the BF16 kv_b bridge (the production
// kernels, libdgpp_kernels.a) against twins that read the checkpoint's int8 g64 kv_b directly and
// rebuild bf16(code x scale) in registers — bitwise the bridge — isolated-cold and behind the layer's
// 16 MiB light-rate prefetch window (dsa_layer.cu: kv_b then o_proj, opened before ~130 us of
// latency-bound kernels; absorb_q runs after the select, vout after the attention core).
// usage: kvb_bench DATA_DIR nlayers [rounds] [g1,g2 us]
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/dsa.hpp"
#include "kernels/l2_prefetch.hpp"
#include "loaders/packq_quant.hpp"

using namespace dgpp;

namespace {
constexpr int HEADS = 16, NOPE = 192, V = 256, KVL = 512, ROPE = 64, HEAD_ROWS = NOPE + V;
constexpr int ROWS_RANK = HEADS * HEAD_ROWS;  // 7168
constexpr int WPR = KVL * 8 / 32;             // 128 I32 words per row
constexpr int SPR = KVL / 64;                 // 8 bf16 scales per row
cudaStream_t g_stream = nullptr;
uint32_t* g_sink = nullptr;

__device__ __forceinline__ float2 bf16x2_to_float2(uint32_t v) {  // dsa.cu's: low half first
  float2 r;
  r.x = __uint_as_float(v << 16);
  r.y = __uint_as_float(v & 0xFFFF0000u);
  return r;
}

// ---- the int8 twins ------------------------------------------------------------------------
// w(r, c) = bf16(float(code) * bf16(scale)), code = byte c%4 of word c/4 as a signed int8
// (loaders/packq_quant.hpp packq_code: byte - 128), scale = scales[r*8 + c/64].
// Hardware RNE (cvt.rn.bf16.f32) in place of the software rounding: the same bits for finite values.
__device__ __forceinline__ float w_bf16(uint32_t word, int j, float s) {
  const int code = static_cast<int>((word >> (8 * j)) & 0xFFu) - 128;  // packq_code: offset-binary bytes
  const __nv_bfloat16 b = __float2bfloat16_rn(static_cast<float>(code) * s);
  return __bfloat162float(b);
}

constexpr int kAbsorbGroups = 8, kAbsorbGroupThreads = 128, kAbsorbThreads = kAbsorbGroups * kAbsorbGroupThreads;
__global__ __launch_bounds__(kAbsorbThreads) void absorb_q_int8_kernel(
    const uint16_t* q, const uint32_t* words, const uint16_t* scales, uint16_t* q_tilde,
    int local_heads, int nope, int v, int kv_lora, int rope) {
  const int64_t r = blockIdx.x;
  const int h = blockIdx.y;
  const int head_rows = nope + v;
  const uint16_t* qh = q + (r * local_heads + h) * (nope + rope);
  const int64_t row0 = int64_t(h) * head_rows;  // W_uk rows of head h
  uint16_t* out = q_tilde + (r * local_heads + h) * (kv_lora + rope);
  __shared__ float qs[256];
  __shared__ __align__(16) float partial[kAbsorbGroups][512];
  for (int d = threadIdx.x; d < nope; d += blockDim.x) qs[d] = bf16_bits_to_float(qh[d]);
  for (int t = threadIdx.x; t < rope; t += blockDim.x) out[kv_lora + t] = qh[nope + t];
  __syncthreads();
  const int group = threadIdx.x / kAbsorbGroupThreads;
  const int col = (threadIdx.x % kAbsorbGroupThreads) * 8;
  const int wpr = kv_lora / 4, spr = kv_lora / 64;
  if (col < kv_lora) {
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
    for (int d = group; d < nope; d += kAbsorbGroups) {
      const float qv = qs[d];
      const int64_t row = row0 + d;
      const uint2 wv = *reinterpret_cast<const uint2*>(words + row * wpr + col / 4);  // 8 codes
      const float s = bf16_bits_to_float(scales[row * spr + col / 64]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[j] += qv * w_bf16(wv.x, j, s);
        acc[4 + j] += qv * w_bf16(wv.y, j, s);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) partial[group][col + j] = acc[j];
  }
  __syncthreads();
  if (group == 0 && col < kv_lora) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float total = partial[0][col + j];
#pragma unroll
      for (int g = 1; g < kAbsorbGroups; ++g) total += partial[g][col + j];
      out[col + j] = float_to_bf16_bits(total);
    }
  }
}

constexpr int kVoutRowsPerBlock = 8, kVoutThreads = 32 * kVoutRowsPerBlock;
__global__ __launch_bounds__(kVoutThreads) void vout_int8_kernel(
    const float* c, const uint32_t* words, const uint16_t* scales, uint16_t* out,
    int local_heads, int nope, int v, int kv_lora) {
  const int64_t r = blockIdx.x;
  const int h = blockIdx.y;
  const int d0 = blockIdx.z * kVoutRowsPerBlock;
  const int head_rows = nope + v;
  __shared__ __align__(16) float cs[512];
  for (int cc = threadIdx.x; cc < kv_lora; cc += blockDim.x) cs[cc] = c[(r * local_heads + h) * kv_lora + cc];
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int d = d0 + warp;
  if (d >= v) return;
  const int64_t row = int64_t(h) * head_rows + nope + d;
  const int wpr = kv_lora / 4, spr = kv_lora / 64;
  const uint32_t* wr = words + row * wpr;
  const uint16_t* sr = scales + row * spr;
  float acc = 0.0f;
  // The bf16 kernel's element order: lane u = lane, lane + 32, ... each eight elements [8u, 8u + 8).
  for (int u = lane; u < kv_lora / 8; u += 32) {
    const uint2 wv = *reinterpret_cast<const uint2*>(wr + 2 * u);
    const float s = bf16_bits_to_float(sr[(8 * u) / 64]);
    const float4 cs4 = *reinterpret_cast<const float4*>(cs + u * 8);
    const float4 cs4b = *reinterpret_cast<const float4*>(cs + u * 8 + 4);
    acc += w_bf16(wv.x, 0, s) * cs4.x;
    acc += w_bf16(wv.x, 1, s) * cs4.y;
    acc += w_bf16(wv.x, 2, s) * cs4.z;
    acc += w_bf16(wv.x, 3, s) * cs4.w;
    acc += w_bf16(wv.y, 0, s) * cs4b.x;
    acc += w_bf16(wv.y, 1, s) * cs4b.y;
    acc += w_bf16(wv.y, 2, s) * cs4b.z;
    acc += w_bf16(wv.y, 3, s) * cs4b.w;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
  if (lane == 0) out[(r * local_heads + h) * v + d] = float_to_bf16_bits(acc);
}


// ---- the row-batched restructure (format-agnostic; the per-row chain is the production kernel's) ----
// absorb: a block owns kTile rows of one head. Thread (group, col) keeps the production kernel's d
// chain (d = group, group + 8, ...) and its 8 columns, but each W_uk load feeds kTile rows' FMAs.
// Every row's accumulator sequence is exactly the one-row kernel's -> bitwise.
template <int kTile>
__global__ __launch_bounds__(kAbsorbThreads) void absorb_q_rows_kernel(
    const uint16_t* q, const uint16_t* kv_b, uint16_t* q_tilde, int rows,
    int local_heads, int nope, int v, int kv_lora, int rope) {
  const int r0 = blockIdx.x * kTile;
  const int h = blockIdx.y;
  const int head_rows = nope + v;
  const int nr = min(kTile, rows - r0);
  const uint16_t* wuk = kv_b + int64_t(h) * head_rows * kv_lora;
  __shared__ float qs[kTile][256];
  __shared__ __align__(16) float run[kTile][512];  // the groups' partials summed in group order
  for (int i = threadIdx.x; i < kTile * nope; i += blockDim.x) {
    const int r = i / nope, d = i - r * nope;
    qs[r][d] = r < nr ? bf16_bits_to_float(q[(int64_t(r0 + r) * local_heads + h) * (nope + rope) + d]) : 0.0f;
  }
  for (int i = threadIdx.x; i < nr * rope; i += blockDim.x) {
    const int r = i / rope, t = i - r * rope;
    q_tilde[(int64_t(r0 + r) * local_heads + h) * (kv_lora + rope) + kv_lora + t] =
        q[(int64_t(r0 + r) * local_heads + h) * (nope + rope) + nope + t];
  }
  __syncthreads();
  const int group = threadIdx.x / kAbsorbGroupThreads;
  const int col = (threadIdx.x % kAbsorbGroupThreads) * 8;
  float acc[kTile][8];
#pragma unroll
  for (int r = 0; r < kTile; ++r)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[r][j] = 0.0f;
  if (col < kv_lora) {
#pragma unroll 4
    for (int d = group; d < nope; d += kAbsorbGroups) {
      const uint4 wv = *reinterpret_cast<const uint4*>(wuk + int64_t(d) * kv_lora + col);
      const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&wv);
      float wf[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) { const float2 f = bf16x2_to_float2(w32[j]); wf[2 * j] = f.x; wf[2 * j + 1] = f.y; }
#pragma unroll
      for (int r = 0; r < kTile; ++r) {
        const float qv = qs[r][d];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r][j] += qv * wf[j];
      }
    }
  }
  // total = partial[0]; total += partial[1]; ... += partial[7] — the one-row kernel's order, one
  // group at a time through the running sum (group 0 seeds it).
  for (int g = 0; g < kAbsorbGroups; ++g) {
    if (group == g && col < kv_lora) {
#pragma unroll
      for (int r = 0; r < kTile; ++r)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (g == 0) run[r][col + j] = acc[r][j];
          else run[r][col + j] = run[r][col + j] + acc[r][j];
        }
    }
    __syncthreads();
  }
  if (group == 0 && col < kv_lora) {
    for (int r = 0; r < nr; ++r) {
      uint16_t* out = q_tilde + (int64_t(r0 + r) * local_heads + h) * (kv_lora + rope);
#pragma unroll
      for (int j = 0; j < 8; ++j) out[col + j] = float_to_bf16_bits(run[r][col + j]);
    }
  }
}

// vout, row-batched: each warp loads its 1 KB weight row ONCE into registers (the lane's two uint4
// = 16 values) and runs the activation rows one after another, each the production chain over the
// staged c row (same lane/element order, same shuffle tree) -> bitwise per row. c rows are staged
// kTileV at a time with float4 loads. kGridTiles = false: one block per (d-slab, head) walks every
// row (512 blocks at any m); true: grid z = the row tiles (2048 blocks at m = 16, the production
// kernel's parallelism at a quarter of its W_uv traffic).
constexpr int kTileV = 4;
template <bool kGridTiles>
__global__ __launch_bounds__(kVoutThreads) void vout_rows_kernel(
    const float* c, const uint16_t* kv_b, uint16_t* out, int rows, int local_heads,
    int nope, int v, int kv_lora) {
  const int h = blockIdx.y;
  const int d0 = blockIdx.x * kVoutRowsPerBlock;
  const int head_rows = nope + v;
  __shared__ __align__(16) float cs[kTileV][512];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int d = d0 + warp;
  const bool live = d < v;
  const uint16_t* wuv = kv_b + (int64_t(h) * head_rows + nope) * kv_lora;
  const uint4* w4 = reinterpret_cast<const uint4*>(wuv + int64_t(d) * kv_lora);
  float wf[2][8];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int u = lane + 32 * i;
    uint4 wv = make_uint4(0u, 0u, 0u, 0u);
    if (live && u < kv_lora / 8) wv = w4[u];
    const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&wv);
#pragma unroll
    for (int j = 0; j < 4; ++j) { const float2 f = bf16x2_to_float2(w32[j]); wf[i][2 * j] = f.x; wf[i][2 * j + 1] = f.y; }
  }
  const int tile_begin = kGridTiles ? blockIdx.z * kTileV : 0;
  const int tile_end = kGridTiles ? min(rows, tile_begin + kTileV) : rows;
  const int vec_per_row = kv_lora / 4;
  for (int t0 = tile_begin; t0 < tile_end; t0 += kTileV) {
    const int nr = min(kTileV, tile_end - t0);
    if (!kGridTiles) __syncthreads();  // the previous tile's readers are done with cs
    for (int i = threadIdx.x; i < nr * vec_per_row; i += blockDim.x) {
      const int r = i / vec_per_row, cc = i - r * vec_per_row;
      reinterpret_cast<float4*>(cs[r])[cc] =
          reinterpret_cast<const float4*>(c + (int64_t(t0 + r) * local_heads + h) * kv_lora)[cc];
    }
    __syncthreads();
    if (!live) continue;
    for (int r = 0; r < nr; ++r) {
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int u = lane + 32 * i;
        if (u < kv_lora / 8) {
          const float4 cs4 = *reinterpret_cast<const float4*>(&cs[r][u * 8]);
          const float4 cs4b = *reinterpret_cast<const float4*>(&cs[r][u * 8 + 4]);
          acc += wf[i][0] * cs4.x;
          acc += wf[i][1] * cs4.y;
          acc += wf[i][2] * cs4.z;
          acc += wf[i][3] * cs4.w;
          acc += wf[i][4] * cs4b.x;
          acc += wf[i][5] * cs4b.y;
          acc += wf[i][6] * cs4b.z;
          acc += wf[i][7] * cs4b.w;
        }
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
      if (lane == 0) out[(int64_t(t0 + r) * local_heads + h) * v + d] = float_to_bf16_bits(acc);
    }
  }
}

// ---- scaffolding ---------------------------------------------------------------------------
__global__ void spin_kernel(uint64_t iters, uint32_t* __restrict__ sink) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uint32_t x = 1u;
  for (uint64_t i = 0; i < iters; ++i) x = x * 1664525u + 1013904223u;
  *sink = x;
}
void spin(uint64_t iters) { if (iters == 0) return; spin_kernel<<<1, 1, 0, g_stream>>>(iters, g_sink); DGPP_CUDA_OK(cudaGetLastError()); }
__global__ void flush_kernel(const uint4* __restrict__ p, size_t n, uint32_t* __restrict__ sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) acc ^= p[i].x ^ p[i].w;
  if (acc == 0x12345678u) *sink = acc;
}
uint4* g_flush = nullptr;
constexpr size_t kFlushBytes = size_t{96} << 20;
void flush_l2() { flush_kernel<<<1024, 256, 0, g_stream>>>(g_flush, kFlushBytes / 16, g_sink); DGPP_CUDA_OK(cudaGetLastError()); }
template <typename T>
T* to_device(const std::vector<T>& v) {
  void* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(16, v.size() * sizeof(T))));
  DGPP_CUDA_OK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return static_cast<T*>(p);
}
template <typename T>
std::vector<T> read_bin(const std::string& path, size_t elems) {
  std::vector<T> w(elems);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f || std::fread(w.data(), sizeof(T), elems, f) != elems) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::fclose(f);
  return w;
}
double elapsed_us(cudaEvent_t a, cudaEvent_t b) { float ms = 0.f; DGPP_CUDA_OK(cudaEventElapsedTime(&ms, a, b)); return 1000.0 * ms; }
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

struct Layer {
  uint16_t* kvb16 = nullptr;     // the bridge [7168, 512] bf16
  uint32_t* words = nullptr;     // [7168, 128]
  uint16_t* scales = nullptr;    // [7168, 8]
  uint8_t* filler = nullptr;     // the window's o_proj share
};
enum Variant { kV16, kV8, kV16R, kV16RZ, kVariants };
const char* kNames[kVariants] = {"bf16 bridge (production)", "int8 g64 twin", "bf16 rows: one block", "bf16 rows: grid tiles"};

void absorb(int v, const Layer& L, const uint16_t* q, uint16_t* q_tilde, int m) {
  if (v == kV16) { dsa_absorb_q(q, L.kvb16, q_tilde, m, HEADS, NOPE, V, KVL, g_stream, ROPE, /*tensor_cores=*/false); return; }
  if (v == kV16R || v == kV16RZ) {
    auto launch = [&](auto tile) {
      constexpr int T = decltype(tile)::value;
      dim3 grid{unsigned((m + T - 1) / T), unsigned(HEADS)};
      absorb_q_rows_kernel<T><<<grid, kAbsorbThreads, 0, g_stream>>>(q, L.kvb16, q_tilde, m, HEADS, NOPE, V, KVL, ROPE);
      DGPP_CUDA_OK(cudaGetLastError());
    };
    switch (m) {
      case 1: launch(std::integral_constant<int, 1>{}); break;
      case 2: launch(std::integral_constant<int, 2>{}); break;
      case 3: launch(std::integral_constant<int, 3>{}); break;
      default: launch(std::integral_constant<int, 4>{}); break;
    }
    return;
  }
  dim3 grid{unsigned(m), unsigned(HEADS)};
  absorb_q_int8_kernel<<<grid, kAbsorbThreads, 0, g_stream>>>(q, L.words, L.scales, q_tilde, HEADS, NOPE, V, KVL, ROPE);
  DGPP_CUDA_OK(cudaGetLastError());
}
void vout(int v, const Layer& L, const float* c, uint16_t* out, int m) {
  if (v == kV16) { dsa_vout_gemm(c, L.kvb16, out, m, HEADS, NOPE, V, KVL, g_stream, /*tensor_cores=*/false); return; }
  if (v == kV16R) {
    dim3 grid{unsigned((V + kVoutRowsPerBlock - 1) / kVoutRowsPerBlock), unsigned(HEADS)};
    vout_rows_kernel<false><<<grid, kVoutThreads, 0, g_stream>>>(c, L.kvb16, out, m, HEADS, NOPE, V, KVL);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  if (v == kV16RZ) {
    dim3 grid{unsigned((V + kVoutRowsPerBlock - 1) / kVoutRowsPerBlock), unsigned(HEADS), unsigned((m + kTileV - 1) / kTileV)};
    vout_rows_kernel<true><<<grid, kVoutThreads, 0, g_stream>>>(c, L.kvb16, out, m, HEADS, NOPE, V, KVL);
    DGPP_CUDA_OK(cudaGetLastError());
    return;
  }
  dim3 grid{unsigned(m), unsigned(HEADS), unsigned((V + kVoutRowsPerBlock - 1) / kVoutRowsPerBlock)};
  vout_int8_kernel<<<grid, kVoutThreads, 0, g_stream>>>(c, L.words, L.scales, out, HEADS, NOPE, V, KVL);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: kvb_bench DATA_DIR nlayers [rounds] [g1,g2 us] [filler_mb]\n"); return 2; }
  const std::string dir = argv[1];
  const int nl = std::atoi(argv[2]);
  const int rounds = argc > 3 ? std::atoi(argv[3]) : 7;
  int g1 = 40, g2 = 70;
  if (argc > 4) { char* a = std::strtok(argv[4], ","); char* b = std::strtok(nullptr, ","); if (a) g1 = std::atoi(a); if (b) g2 = std::atoi(b); }
  const size_t filler_bytes = static_cast<size_t>(argc > 5 ? std::atof(argv[5]) : 9.0) * 1000000;

  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  { void* p = nullptr; DGPP_CUDA_OK(cudaMalloc(&p, 16)); g_sink = static_cast<uint32_t*>(p); }
  { void* p = nullptr; DGPP_CUDA_OK(cudaMalloc(&p, kFlushBytes)); DGPP_CUDA_OK(cudaMemset(p, 1, kFlushBytes)); g_flush = static_cast<uint4*>(p); }

  std::vector<Layer> layers;
  for (int i = 0; i < nl; ++i) {
    const auto words = read_bin<uint32_t>(dir + "/L" + std::to_string(i) + "_packed.bin", size_t{ROWS_RANK} * WPR);
    const auto scales = read_bin<uint16_t>(dir + "/L" + std::to_string(i) + "_scale.bin", size_t{ROWS_RANK} * SPR);
    // The loader's bridge: bf16(code x scale), loaders/packq_quant.hpp packq_decode.
    std::vector<uint16_t> bridge(size_t{ROWS_RANK} * KVL);
    for (int64_t r = 0; r < ROWS_RANK; ++r)
      for (int64_t c = 0; c < KVL; ++c)
        bridge[static_cast<size_t>(r) * KVL + c] = float_to_bf16_bits(packq_decode(words.data(), scales.data(), KVL, 8, r, c));
    Layer L;
    L.kvb16 = to_device(bridge);
    // One allocation for words + scales, as the resident image holds them adjacent (the window's add).
    {
      std::vector<uint8_t> image(words.size() * 4 + scales.size() * 2 + 256);
      std::memcpy(image.data(), words.data(), words.size() * 4);
      std::memcpy(image.data() + words.size() * 4, scales.data(), scales.size() * 2);
      uint8_t* base = to_device(image);
      L.words = reinterpret_cast<uint32_t*>(base);
      L.scales = reinterpret_cast<uint16_t*>(base + words.size() * 4);
    }
    std::vector<uint8_t> fill(filler_bytes + 16, static_cast<uint8_t>(i + 1));
    L.filler = to_device(fill);
    layers.push_back(L);
  }
  std::printf("%d layers, rank-0 kv_b slice [%d x %d]: bf16 bridge %.2f MB, int8 g64 %.2f MB (+ scales %.2f MB); W_uk part %.2f -> %.2f MB, W_uv %.2f -> %.2f MB; window filler %.1f MB\n",
              nl, ROWS_RANK, KVL, ROWS_RANK * KVL * 2 / 1e6, ROWS_RANK * KVL / 1e6, ROWS_RANK * SPR * 2 / 1e6,
              HEADS * NOPE * KVL * 2 / 1e6, HEADS * NOPE * KVL / 1e6, HEADS * V * KVL * 2 / 1e6, HEADS * V * KVL / 1e6, filler_bytes / 1e6);

  // Inputs: 16 rows of q (bf16, [heads x (nope + rope)]) and c (fp32, [heads x kv_lora]).
  const int MAXM = 16;
  std::vector<uint16_t> q(size_t{MAXM} * HEADS * (NOPE + ROPE));
  std::vector<float> c(size_t{MAXM} * HEADS * KVL);
  uint64_t s = 0x9E3779B97F4A7C15ull;
  auto rnd = [&]() { s ^= s << 13, s ^= s >> 7, s ^= s << 17; return s; };
  for (auto& a : q) { const uint64_t x = rnd(); a = static_cast<uint16_t>(((x >> 20) & 0x80FFu) | ((0x7Cu + ((x >> 40) & 7u)) << 7)); }
  for (auto& a : c) a = (static_cast<float>(rnd() & 0xFFFF) / 65536.f - 0.5f) * 4.f;
  uint16_t* dq = to_device(q);
  float* dc = to_device(c);
  uint16_t* qt[kVariants]; uint16_t* ov[kVariants];
  for (int v = 0; v < kVariants; ++v) {
    DGPP_CUDA_OK(cudaMalloc(&qt[v], size_t{MAXM} * HEADS * (KVL + ROPE) * 2));
    DGPP_CUDA_OK(cudaMalloc(&ov[v], size_t{MAXM} * HEADS * V * 2));
  }
  // Bitwise gate: q_tilde and out of the twin == the bridge kernels', every layer, m in {1,2,4,8,16}.
  for (int m : {1, 2, 4, 8, 16}) {
    for (const Layer& L : layers) {
      for (int v = 0; v < kVariants; ++v) { absorb(v, L, dq, qt[v], m); vout(v, L, dc, ov[v], m); }
      DGPP_CUDA_OK(cudaStreamSynchronize(g_stream));
      for (int v = 1; v < kVariants; ++v) {
      auto same = [&](const void* a, const void* b, size_t bytes, const char* what) {
        std::vector<uint8_t> x(bytes), y(bytes);
        DGPP_CUDA_OK(cudaMemcpy(x.data(), a, bytes, cudaMemcpyDeviceToHost));
        DGPP_CUDA_OK(cudaMemcpy(y.data(), b, bytes, cudaMemcpyDeviceToHost));
        if (std::memcmp(x.data(), y.data(), bytes) != 0) { std::printf("BITWISE MISMATCH %s m=%d\n", what, m); std::exit(1); }
      };
      same(qt[0], qt[v], size_t{m} * HEADS * (KVL + ROPE) * 2, "q_tilde");
      same(ov[0], ov[v], size_t{m} * HEADS * V * 2, "vout");
      }
    }
  }
  std::printf("bitwise gate: every variant == the production kernels (q_tilde, out) on every layer, m in {1,2,4,8,16}\n");

  cudaEvent_t e0{}, e1{};
  DGPP_CUDA_OK(cudaEventCreate(&e0)); DGPP_CUDA_OK(cudaEventCreate(&e1));
  double iters_per_us = 0;
  {
    const uint64_t probe = uint64_t{1} << 22;
    std::vector<double> t;
    for (int r = 0; r < 5; ++r) { DGPP_CUDA_OK(cudaEventRecord(e0, g_stream)); spin(probe); DGPP_CUDA_OK(cudaEventRecord(e1, g_stream)); DGPP_CUDA_OK(cudaEventSynchronize(e1)); t.push_back(elapsed_us(e0, e1)); }
    iters_per_us = static_cast<double>(probe) / median(t);
  }
  WeightPrefetcher prefetch;
  const size_t window = size_t{16} << 20;
  const int launches = std::max(96, nl * 16);
  // kind 0: absorb alone, 1: vout alone, 2: both in the layer's context
  // (window: kv_b then the o_proj filler; spin g1; absorb; spin g2; vout).
  auto round = [&](int v, int m, int kind, bool context) {
    if (!context) {
      std::vector<double> t;
      for (int i = 0; i < launches; ++i) {
        const Layer& L = layers[i % nl];
        flush_l2();
        DGPP_CUDA_OK(cudaEventRecord(e0, g_stream));
        if (kind != 1) absorb(v, L, dq, qt[v], m);
        if (kind != 0) vout(v, L, dc, ov[v], m);
        DGPP_CUDA_OK(cudaEventRecord(e1, g_stream));
        DGPP_CUDA_OK(cudaEventSynchronize(e1));
        t.push_back(elapsed_us(e0, e1));
      }
      return median(t);
    }
    const uint64_t i1 = static_cast<uint64_t>(g1 * iters_per_us), i2 = static_cast<uint64_t>(g2 * iters_per_us);
    spin(static_cast<uint64_t>(30000 * iters_per_us));
    DGPP_CUDA_OK(cudaEventRecord(e0, g_stream));
    for (int i = 0; i < launches; ++i) {
      const Layer& L = layers[i % nl];
      prefetch.open_window(g_stream, window, prefetch.layer_rate());
      if (v == kV8) prefetch.add(L.words, size_t{ROWS_RANK} * (WPR * 4 + SPR * 2));
      else prefetch.add(L.kvb16, size_t{ROWS_RANK} * KVL * 2);
      if (filler_bytes) prefetch.add_isolated(L.filler, filler_bytes);
      spin(i1);
      absorb(v, L, dq, qt[v], m);
      spin(i2);
      vout(v, L, dc, ov[v], m);
    }
    DGPP_CUDA_OK(cudaEventRecord(e1, g_stream));
    prefetch.join(g_stream);
    DGPP_CUDA_OK(cudaStreamSynchronize(g_stream));
    return elapsed_us(e0, e1) / launches;
  };
  double spin_us = 0;
  {
    std::vector<double> t;
    for (int r = 0; r < 3; ++r) {
      spin(static_cast<uint64_t>(30000 * iters_per_us));
      DGPP_CUDA_OK(cudaEventRecord(e0, g_stream));
      for (int i = 0; i < launches; ++i) { spin(static_cast<uint64_t>(g1 * iters_per_us)); spin(static_cast<uint64_t>(g2 * iters_per_us)); }
      DGPP_CUDA_OK(cudaEventRecord(e1, g_stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(g_stream));
      t.push_back(elapsed_us(e0, e1) / launches);
    }
    spin_us = median(t);
  }
  std::printf("window %zu MiB at the layer (light) rate, prefetch %s; gaps g1=%d g2=%d us (spins %.1f); %d launches x %d rounds (median)\n",
              window >> 20, prefetch.enabled() ? "on" : "OFF", g1, g2, spin_us, launches, rounds);
  for (int m : {1, 2, 4, 8, 16}) {
    std::printf("\nm=%2d us per layer         absorb cold     vout cold   both cold   both in context\n", m);
    std::vector<std::vector<double>> table(kVariants);
    for (int v = 0; v < kVariants; ++v) {
      for (int kind = 0; kind < 3; ++kind) {
        std::vector<double> t;
        for (int r = 0; r < rounds; ++r) t.push_back(round(v, m, kind, false));
        table[v].push_back(median(t));
      }
      std::vector<double> t;
      for (int r = 0; r < rounds; ++r) t.push_back(round(v, m, 2, true) - spin_us);
      table[v].push_back(median(t));
      std::printf("  %-26s", kNames[v]);
      for (double x : table[v]) std::printf(" %12.1f", x);
      std::printf("\n");
    }
    for (int v = 1; v < kVariants; ++v) {
      std::printf("  d -> %-21s", kNames[v]);
      for (size_t c2 = 0; c2 < table[0].size(); ++c2) std::printf(" %+5.1f (%+5.2f)", table[v][c2] - table[0][c2], 75 * (table[v][c2] - table[0][c2]) / 1000.0);
      std::printf("   us/layer (ms/step x75)\n");
    }
  }
  return 0;
}
