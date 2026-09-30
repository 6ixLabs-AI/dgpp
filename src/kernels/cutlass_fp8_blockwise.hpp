#pragma once
// Path-B native blockwise-FP8 dense GEMM for Qwen3.5-27B prefill.
//
// Replaces the FP8-dequant + cuBLASLt BF16 bridge (gemm_dense in
// src/models/qwen/layers.cpp) with a CUTLASS Sm100 blockwise-scaled
// FP8xFP8 GEMM whose scales are injected in the mainloop (tensor-wise
// dequant cannot express 2D block scales; epilogue-only scaling cannot
// either). Compiled sm100a; PTX is embedded so the driver JITs to sm_121a.
//
// Activation side is quantized per call: BF16 [M,K] row-major ->
// E4M3 payload + F32 per-atom scales (128x64 atoms) written directly in
// CUTLASS trivial SFA order. Weight side reuses the resident checkpoint
// payload W[N,K] viewed ColumnMajor [K,N]; its scales are reformatted
// once at load (host) from checkpoint row-major [Nb][Kb] to CUTLASS
// SFB order [Kb][Nb] (same element count, pure transpose).
//
// Eager-prefill only: Arguments/can_implement/initialize run on the host
// per call. Do NOT invoke under CUDA graph capture.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace dgpp {

// Scale atom sizes the kernel is built for (MMA tile 128x64x64, trivial
// config). M atoms stay 128; N/K atoms are 64 (checkpoint 128-blocks are
// duplicated 2x2 at format time, which is exact).
inline constexpr int kCutlassFp8AtomM = 128;
inline constexpr int kCutlassFp8AtomN = 64;
inline constexpr int kCutlassFp8AtomK = 64;
// Headroom for CUTLASS GemmUniversalAdapter workspace inside the caller
// buffer (actual need is queried per call via get_workspace_size; calls
// that exceed this fall back to the bridge).
inline constexpr size_t kCutlassFp8AdapterWsMax = 1u << 20;

// True when (m,n,k) can take the blockwise path: K and N must be whole
// 64-atoms (scale grid is exact, no ragged tails); M is predicated.
inline bool cutlass_fp8_blockwise_supported(int m, int n, int k) {
  return m > 0 && n >= kCutlassFp8AtomN && k >= kCutlassFp8AtomK &&
         (n % kCutlassFp8AtomN) == 0 && (k % kCutlassFp8AtomK) == 0;
}

// Host-side one-time reorder+expand of weight scales: src is checkpoint
// row-major [n128][k128] over 128x128 blocks; dst is CUTLASS SFB order over
// 64-atoms (offset = k_atom * n_atoms + n_atom, n_atoms = N/64,
// k_atoms = K/64), each checkpoint scale duplicated over its 2x2 atoms.
// n_blocks = N/64, k_blocks = K/64 (both even).
void cutlass_fp8_reorder_b_scales(const float* src, float* dst, int n_blocks,
                                  int k_blocks);

// Device scratch for one GEMM call with the given (m,k): E4M3 activation
// payload (m*k bytes) + SFA F32 grid (ceil(m/128)*(k/64) floats) +
// adapter workspace headroom. Activation row stride is dense (== k).
size_t cutlass_fp8_act_ws_bytes(int m, int k);

// out[M,N] BF16 row-major = act[M,K] BF16 row-major @ W[N,K].
//   w: resident E4M3 payload, n*k bytes, row-major [N,K] (read as ColMajor).
//   w_scales_cutlass: device F32 grid in CUTLASS SFB order (see reorder fn).
//   ws: device scratch of at least cutlass_fp8_act_ws_bytes(m, k) bytes.
// Returns false (leaves `out` untouched) when the shape is unsupported or
// the CUTLASS workspace need exceeds the headroom; caller falls back to
// the dequant bridge.
bool launch_cutlass_fp8_blockwise(const uint16_t* act, const uint8_t* w,
                                  const float* w_scales_cutlass, uint16_t* out,
                                  int m, int n, int k, void* ws, size_t ws_bytes,
                                  cudaStream_t stream);

}  // namespace dgpp
