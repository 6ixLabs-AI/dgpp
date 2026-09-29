#pragma once
// Qwen3.8-Flash-Next MoE pieces that are not the shared MoE kernels
// (kernels/glm_moe_launch.hpp carries the router — its SoftmaxTopk mode —
// the expert chain and the accumulation): the BF16 shared expert's scalar
// gate. Deterministic, capturable.
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// w[t] = bf16(sigmoid(bf16(x[t] . g))) as an fp32 value: the reference's
// `sigmoid(shared_expert_gate(x))` — a bf16 Linear (one rounding of the
// logit) then a bf16 sigmoid (one rounding of the gate). x bf16
// [tokens, hidden], g bf16 [hidden]; hidden % 8 == 0, x and g 16-byte
// aligned. One warp per token; the lanes' strided fp32 partials and a
// fixed xor tree (the router dots' reduction shape).
void qwen_moe_shared_gate_bf16(const uint16_t* x, const uint16_t* g, float* w,
                               int tokens, int hidden, cudaStream_t stream);

// The shared expert's decode tail in two launches, bitwise
// the seven-launch chain (the GEMM interface's bf16 GEMVs at m <= 4, the
// swiglu with no clamps, the shared gate, the accumulate's fma, the
// round): act = swiglu(x . gate_w, x . up_w) [tokens, S] and sw[t] =
// sigmoid(x[t] . g), then out = bf16(fma(sw, act . down_w, acc)). x bf16
// [tokens, H] with row stride x_stride; gate_w/up_w bf16 [S, H]; down_w
// bf16 [H, S]; acc fp32 [tokens, H] (the routed chain's); tokens <= 8
// (rows in chunks of four); H, S multiples of 8, 16-byte aligned operands.
void qwen_moe_shared_tail_decode(const uint16_t* x, size_t x_stride, const uint16_t* gate_w,
                                 const uint16_t* up_w, const uint16_t* down_w,
                                 const uint16_t* g, uint16_t* act, float* sw, const float* acc,
                                 uint16_t* out, int tokens, int H, int S, cudaStream_t stream);
// The same tail with the three matrices in block FP8 (2026-09-10,
// engine.dense_weights = "fp8"): E4M3 payloads (gate/up [S, H], down [H,
// S]) with fp32 128 x 128 scales, the dots the scale GEMM's fp8 GEMV chain
// — bitwise the unfused fp8 chain (launch_scale_gemm_bf16 x2, the swiglu,
// launch_scale_gemm_f32, the gate, the accumulate, the round). H and S
// multiples of 16.
// The tail's two launches as separate entries (2026-09-29): the gate/up +
// swiglu half reads only x, so it runs on a side stream beside the routed
// experts (its 21 us ramp hidden behind their stream); the down half needs
// the routed accumulator and runs on the chain after the join. Each half
// is the same kernel the one-call tail launches, so the outputs are
// bitwise the tail's.
// The routed accumulation folded into the down epilogue (2026-09-29): the
// per-slot down outputs and the router weights in place of `acc`, summed
// in the accumulate kernel's fma order — bitwise that kernel + this tail,
// one launch fewer per layer.
void qwen_moe_shared_tail_decode_routed(const uint16_t* x, size_t x_stride, const uint16_t* gate_w,
                                        const uint16_t* up_w, const uint16_t* down_w, const uint16_t* g,
                                        uint16_t* act, float* sw, const float* contrib, const float* weights,
                                        int top_k, uint16_t* out, int tokens, int H, int S, cudaStream_t stream);
void qwen_moe_shared_tail_decode_routed_fp8(const uint16_t* x, size_t x_stride, const uint8_t* gate_p,
                                            const float* gate_s, const uint8_t* up_p, const float* up_s,
                                            const uint8_t* down_p, const float* down_s, const uint16_t* g,
                                            uint16_t* act, float* sw, const float* contrib, const float* weights,
                                            int top_k, uint16_t* out, int tokens, int H, int S, cudaStream_t stream);
void qwen_moe_shared_down_decode_routed(const uint16_t* act, const uint16_t* down_w, const float* sw,
                                        const float* contrib, const float* weights, int top_k, uint16_t* out,
                                        int tokens, int H, int S, cudaStream_t stream);
void qwen_moe_shared_down_decode_routed_fp8(const uint16_t* act, const uint8_t* down_p, const float* down_s,
                                            const float* sw, const float* contrib, const float* weights, int top_k,
                                            uint16_t* out, int tokens, int H, int S, cudaStream_t stream);
void qwen_moe_shared_gate_up_decode(const uint16_t* x, size_t x_stride, const uint16_t* gate_w,
                                    const uint16_t* up_w, const uint16_t* g, uint16_t* act, float* sw,
                                    int tokens, int H, int S, cudaStream_t stream);
void qwen_moe_shared_down_decode(const uint16_t* act, const uint16_t* down_w, const float* sw,
                                 const float* acc, uint16_t* out, int tokens, int H, int S,
                                 cudaStream_t stream);
void qwen_moe_shared_gate_up_decode_fp8(const uint16_t* x, size_t x_stride, const uint8_t* gate_p,
                                        const float* gate_s, const uint8_t* up_p, const float* up_s,
                                        const uint16_t* g, uint16_t* act, float* sw, int tokens, int H,
                                        int S, cudaStream_t stream);
void qwen_moe_shared_down_decode_fp8(const uint16_t* act, const uint8_t* down_p, const float* down_s,
                                     const float* sw, const float* acc, uint16_t* out, int tokens, int H,
                                     int S, cudaStream_t stream);
void qwen_moe_shared_tail_decode_fp8(const uint16_t* x, size_t x_stride, const uint8_t* gate_p,
                                     const float* gate_s, const uint8_t* up_p, const float* up_s,
                                     const uint8_t* down_p, const float* down_s, const uint16_t* g,
                                     uint16_t* act, float* sw, const float* acc, uint16_t* out,
                                     int tokens, int H, int S, cudaStream_t stream);

}  // namespace dgpp
