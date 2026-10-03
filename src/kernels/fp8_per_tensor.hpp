#pragma once
// Per-tensor FP8 helpers (engine.prefill_fp8_per_tensor, the Qwen3.8-27B
// prefill recipe): weights are requantized once at boot and activations per
// call with one scalar scale each, for cuBLASLt's E4M3 x E4M3 scalar-scale
// kernels (the block / outer-vector scale modes are sm100-only there).
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// maxabs over `count` BF16 elements (inf folds to 448, NaN ignored, all-zero
// yields 0 — the quant kernel guards that to scale 1). Multi-CTA vectorized
// reduction, deterministic (max is order-independent).
void launch_fp8_row_maxabs(const uint16_t* data, size_t count, float* out_max,
                           cudaStream_t stream);

// Quantize `count` BF16 elements to E4M3 with the per-tensor map x/mx,
// `dev_max` the same-stream maxabs output (read inside the kernel); the Lt
// scalar scales are then the raw maxabs values.
void launch_fp8_quant_bf16(const uint16_t* in, uint8_t* out, size_t count,
                           const float* dev_max, cudaStream_t stream);

}  // namespace dgpp
