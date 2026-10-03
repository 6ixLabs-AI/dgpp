#pragma once
// The exact fp8-weight GEMM (2026-10-03, #87): bf16 activations against a
// block-FP8 weight matrix on the bf16 tensor cores, the weights placed
// into bf16 in shared memory by bit arithmetic (no dequant bridge: no
// [n x k] bf16 write and read per matrix). Two scale forms:
//   PerWeight — bf16(e4m3 x scale), one rounding per weight: every product
//     term is bitwise the dequant bridge's / the scale GEMM tile kernel's,
//     the mma chain ascending k16 with fp32 accumulation;
//   PerGroup — the products of the raw codes accumulate in fp32 and the
//     128 x 128 block scale promotes each 128-wide k group once (the fp8
//     GEMM's promotion): finer than the bridge by its per-weight rounding,
//     tolerance-equal, fewer instructions a weight.
// On the GB10 (2026-10-03, m = 2048, [17408 x 5120]): PerWeight 5.9 ms,
// PerGroup 5.5 ms against the bridge's 5.3 (dequant 1.3 + cuBLASLt 4.1)
// and 13.5 ms for the scale GEMM's tile kernel — the bf16 mma pipe issues
// at ~90 TF and the decode's instructions share its issue slots, so the
// kernel replaces the tile kernel's route, not the bridge's.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace dgpp {

// out[m][n] = act[m][k] (bf16, row stride act_stride elements) x w[n][k]^T
// (e4m3, the checkpoint's payload) with f32 scales [ceil(n / 128)][ceil(k /
// 128)] (the 128 x 128 grid). k % 64 == 0; act 16-byte aligned with
// act_stride % 8 == 0 (fp8w_gemm_shape_ok). m and n ragged (zero-filled
// tiles, guarded stores). out_stride 0 = n. The bf16 output is the fp32
// sum rounded once (bitwise bf16(the f32 launch)).
enum class Fp8wScale { PerWeight, PerGroup };
bool fp8w_gemm_shape_ok(const void* act, size_t act_stride, int k);
void launch_fp8w_gemm_bf16(const uint16_t* act, size_t act_stride, const uint8_t* w, const float* w_scales,
                           uint16_t* out, int m, int n, int k, Fp8wScale scale, cudaStream_t stream,
                           size_t out_stride = 0);
void launch_fp8w_gemm_f32(const uint16_t* act, size_t act_stride, const uint8_t* w, const float* w_scales,
                          float* out, int m, int n, int k, Fp8wScale scale, cudaStream_t stream,
                          size_t out_stride = 0);

}  // namespace dgpp
