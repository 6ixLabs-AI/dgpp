// Native blockwise-FP8 dense GEMM for Qwen3.5-27B prefill (Path B).
//
// Compiled sm100a ONLY (PTX embedded for driver JIT onto sm_121a). Keep out
// of dgpp_kernels: this TU needs -arch=sm_100a + CUTLASS includes, while the
// rest of the tree builds 121a.
#include "cutlass_fp8_blockwise.hpp"

#include <cuda_bf16.h>

// CUDA 13.0's nvcc does not define __CUDA_ARCH_FEAT_SM100_ALL even under
// -arch=sm_100a, so CUTLASS never enables its SM100A device paths (TMA
// prefetch asserts at runtime). Force it for the DEVICE pass only, mirroring
// what newer toolchains do; the host pass must keep it undefined (host
// fallbacks in HOST_DEVICE helpers call device-only synclog otherwise).
#if defined(__CUDA_ARCH__) && !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)
#define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#endif

// Must precede other CUTLASS headers: the sparse clone's include chain
// reaches cute/arch/mma_sm100_desc.hpp (which calls unqualified
// cast_smem_ptr_to_uint) before cute/arch/util.hpp declares it.
#include <cute/arch/util.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cstdint>

namespace dgpp {
namespace {

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int kAlignmentA = 16;  // 128-bit vectors; K is a multiple of 128
using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;  // (K,N) view over W[N,K] bytes
constexpr int kAlignmentB = 16;
using ElementC = cutlass::bfloat16_t;  // C == D buffer, beta == 0 (never read)
using LayoutC = cutlass::layout::RowMajor;
constexpr int kAlignmentC = 8;
using ElementD = cutlass::bfloat16_t;
using LayoutD = cutlass::layout::RowMajor;
constexpr int kAlignmentD = 8;
using ElementAccumulator = float;
using ElementCompute = float;

using MmaTileShape_MNK = cute::Shape<cute::_128, cute::_64, cute::_64>;
// GB10 (sm_121a) allows ~99KB dynamic smem/block (vs 228KB on sm_100), so the
// 128-cubed config (231KB) cannot run here: use a 128x64x64 CTA tile with 2
// stages. SFVec follows the tile (trivial config); sub-128 scale atoms are
// fed by duplicating checkpoint 128-blocks at format time (exact: same block
// scale reused for each covered atom), and the activation quantizer simply
// quantizes at the finer 128x64 granularity (also exact).
using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;
using ScaleConfig = decltype(cutlass::detail::sm100_trivial_blockwise_scale_config(
    MmaTileShape_MNK{}));
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK,
    ClusterShape_MNK, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute, ElementC, LayoutC, kAlignmentC,
    ElementD, LayoutD, kAlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
    cute::tuple<LayoutA, LayoutSFA>, kAlignmentA, ElementB,
    cute::tuple<LayoutB, LayoutSFB>, kAlignmentB,     ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCount<2>,
    cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
    void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// BF16 [M,K] -> E4M3 payload + F32 SFA grid (one CTA per 128x64 A-atom).
// SFA is written in CUTLASS trivial order: sfa[m_block + k_block * Mb]
// with k_block over 64-wide atoms (Mb = ceil(M/128)).
__global__ void __launch_bounds__(256) bf16_to_fp8_block128x64_kernel(
    const __nv_bfloat16* __restrict__ act, uint8_t* __restrict__ a_fp8,
    float* __restrict__ sfa, int m, int k) {
  const int kb = blockIdx.x;
  const int mb = blockIdx.y;
  const int Mb = gridDim.y;
  constexpr int kCols = 64;
  constexpr int kElems = 128 * kCols;

  float vmax = 0.0f;
  for (int i = threadIdx.x; i < kElems; i += blockDim.x) {
    const int r = mb * 128 + i / kCols;
    const int c = kb * kCols + i % kCols;
    float v = 0.0f;
    if (r < m) {
      v = __bfloat162float(act[r * k + c]);
    }
    vmax = fmaxf(vmax, fabsf(v));
  }

  __shared__ float smax[256];
  smax[threadIdx.x] = vmax;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
    }
    __syncthreads();
  }
  const float amax = smax[0];
  const float inv = (amax > 0.0f) ? (448.0f / amax) : 1.0f;

  for (int i = threadIdx.x; i < kElems; i += blockDim.x) {
    const int r = mb * 128 + i / kCols;
    const int c = kb * kCols + i % kCols;
    if (r < m) {
      const float v = __bfloat162float(act[r * k + c]);
      cutlass::float_e4m3_t q(v * inv);
      a_fp8[r * k + c] = q.storage;
    }
  }
  if (threadIdx.x == 0) {
    sfa[mb + kb * Mb] = (amax > 0.0f) ? (amax / 448.0f) : 1.0f;
  }
}

}  // namespace
void cutlass_fp8_reorder_b_scales(const float* src, float* dst, int n_blocks,
                                  int k_blocks) {
  // Checkpoint: src[(n/2) * (k_blocks/2) + (k/2)] over 128x128 blocks.
  // CUTLASS SFB (64-wide N/K atoms): dst[k * n_blocks + n], each checkpoint
  // scale duplicated over its 2x2 covered atoms (exact: same block scale).
  const int kb128 = k_blocks / 2;
  for (int n = 0; n < n_blocks; ++n) {
    for (int k = 0; k < k_blocks; ++k) {
      dst[k * n_blocks + n] = src[(n / 2) * kb128 + (k / 2)];
    }
  }
}

size_t cutlass_fp8_act_ws_bytes(int m, int k) {
  const int Kb = k / kCutlassFp8AtomK;
  const int Mb = (m + kCutlassFp8AtomM - 1) / kCutlassFp8AtomM;
  return static_cast<size_t>(m) * static_cast<size_t>(k) +
         static_cast<size_t>(Mb) * static_cast<size_t>(Kb) * sizeof(float) +
         kCutlassFp8AdapterWsMax;
}

bool launch_cutlass_fp8_blockwise(const uint16_t* act, const uint8_t* w,
                                  const float* w_scales_cutlass, uint16_t* out,
                                  int m, int n, int k, void* ws,
                                  size_t ws_bytes, cudaStream_t stream) {
  if (!cutlass_fp8_blockwise_supported(m, n, k) || ws == nullptr) {
    return false;
  }
  const int Kb = k / kCutlassFp8AtomK;
  const int Mb = (m + kCutlassFp8AtomM - 1) / kCutlassFp8AtomM;
  const size_t a_bytes = static_cast<size_t>(m) * static_cast<size_t>(k);
  const size_t sfa_bytes =
      static_cast<size_t>(Mb) * static_cast<size_t>(Kb) * sizeof(float);
  if (ws_bytes < a_bytes + sfa_bytes) {
    return false;
  }
  auto* a_data = static_cast<uint8_t*>(ws);
  auto* sfa_data = reinterpret_cast<float*>(a_data + a_bytes);
  auto* adapter_ws = a_data + a_bytes + sfa_bytes;
  const size_t adapter_avail = ws_bytes - (a_bytes + sfa_bytes);

  dim3 grid(static_cast<unsigned>(Kb), static_cast<unsigned>(Mb));
  bf16_to_fp8_block128x64_kernel<<<grid, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(act), a_data, sfa_data, m, k);

  // NOTE: cutlass/util/packed_stride.hpp is absent from the sparse CUTLASS
  // clone, so strides are built by hand. Deduced kernel stride types are
  // (int64, _1, int64): mode0 strided, mode1 contiguous, batch 0.
  auto stride_A = cute::make_stride(static_cast<int64_t>(k), cute::Int<1>{},
                                    int64_t{0});
  // W[N,K] row-major bytes as the (N,K) B-operand tensor: element (n,k) =
  // base[n*K+k] = W[n][k], exactly the convention example 81 itself uses
  // for its B tensor.
  auto stride_B = cute::make_stride(static_cast<int64_t>(k), cute::Int<1>{},
                                    int64_t{0});
  auto stride_C = cute::make_stride(static_cast<int64_t>(n), cute::Int<1>{},
                                     int64_t{0});
  auto stride_D = cute::make_stride(static_cast<int64_t>(n), cute::Int<1>{},
                                     int64_t{0});
  auto layout_SFA =
      ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB =
      ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  auto* out_d = reinterpret_cast<ElementD*>(out);
  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {m, n, k, 1};
  args.mainloop.ptr_A = reinterpret_cast<ElementA const*>(a_data);
  args.mainloop.dA = stride_A;
  args.mainloop.ptr_B = reinterpret_cast<ElementB const*>(w);
  args.mainloop.dB = stride_B;
  args.mainloop.ptr_SFA = sfa_data;
  args.mainloop.layout_SFA = layout_SFA;
  args.mainloop.ptr_SFB = w_scales_cutlass;
  args.mainloop.layout_SFB = layout_SFB;
  args.epilogue.thread.alpha = ElementCompute(1);
  args.epilogue.thread.beta = ElementCompute(0);
  args.epilogue.ptr_C = out_d;
  args.epilogue.dC = stride_C;
  args.epilogue.ptr_D = out_d;
  args.epilogue.dD = stride_D;

  Gemm gemm;
  auto ci = gemm.can_implement(args);
  if (ci != cutlass::Status::kSuccess) {
    fprintf(stderr, "[fp8bw] can_implement=%d\n", (int)ci);
    return false;
  }
  size_t need = Gemm::get_workspace_size(args);
  fprintf(stderr, "[fp8bw] ws_need=%zu avail=%zu\n", need, adapter_avail);
  if (need > adapter_avail) {
    return false;
  }
  auto st = gemm.initialize(args, adapter_ws);
  if (st != cutlass::Status::kSuccess) {
    fprintf(stderr, "[fp8bw] initialize=%d\n", (int)st);
    return false;
  }
  st = gemm.run(stream);
  if (st != cutlass::Status::kSuccess) {
    fprintf(stderr, "[fp8bw] run=%d cuda=%s\n", (int)st,
            cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  return true;
}

// TEMP DEBUG: quantizer-only smoke test (JIT check). Remove before merge.
bool dbg_quant_only(const uint16_t* act, int m, int k, void* ws,
                    cudaStream_t stream) {
  const int Kb = k / kCutlassFp8AtomK;
  const int Mb = (m + kCutlassFp8AtomM - 1) / kCutlassFp8AtomM;
  auto* a_data = static_cast<uint8_t*>(ws);
  auto* sfa_data =
      reinterpret_cast<float*>(a_data + static_cast<size_t>(m) * k);
  dim3 grid(static_cast<unsigned>(Kb), static_cast<unsigned>(Mb));
  bf16_to_fp8_block128x64_kernel<<<grid, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(act), a_data, sfa_data, m, k);
  cudaError_t e = cudaGetLastError();
  fprintf(stderr, "[fp8bw] quant launch: %s\n", cudaGetErrorString(e));
  return e == cudaSuccess;
}

}  // namespace dgpp
