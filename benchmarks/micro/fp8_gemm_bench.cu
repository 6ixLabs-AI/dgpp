// The opt-in fp8 prefill GEMM (kernels/fp8_gemm, engine.prefill_fp8_gemm)
// against the chain it replaces at the hybrid's dense prefill shapes: the
// block-FP8 dequant into a bf16 bridge plus the cuBLASLt bf16 GEMM, versus
// the per-token e4m3 quantizer plus the fp8 tensor-core GEMM. Prints ms per
// launch pair and the tensor throughput. Usage: fp8_gemm_bench [m]
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/fp8_dequant.hpp"
#include "kernels/fp8w_gemm.hpp"
#include "kernels/fp8_gemm.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_moe_launch.hpp"

int main(int argc, char** argv) {
  const int m = argc > 1 ? std::atoi(argv[1]) : 4096;
  struct Shape {
    int n, k;
    const char* what;
  };
  // --27b: the Qwen3.8-27B dense shapes (the exact prefill's bridge vs the
  // fp8-weight tile GEMM the scale GEMM routes m > 128 to).
  const bool shapes27b = argc > 2 && std::string(argv[2]) == "--27b";
  const Shape flash[] = {{2560, 2560, "q/k/v-like [2560 x 2560]"},   {5120, 2560, "in_proj-like [5120 x 2560]"},
                         {2560, 5120, "out_proj-like [2560 x 5120]"}, {320, 10240, "GR down [320 x 10240]"},
                         {10240, 2560, "PLE key [10240 x 2560]"}};
  const Shape b27[] = {{17408, 5120, "27B gate/up [17408 x 5120]"}, {5120, 17408, "27B down [5120 x 17408]"},
                       {12288, 5120, "27B q [12288 x 5120]"},       {10240, 5120, "27B GDN qkv [10240 x 5120]"},
                       {5120, 6144, "27B out/o [5120 x 6144]"}};
  const Shape* shapes = shapes27b ? b27 : flash;
  const int n_shapes = shapes27b ? 5 : 5;
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.f, 1.f);
  dgpp::CublasLtGemm gemm;
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  cudaEvent_t e0, e1;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  std::printf("m = %d rows; ms per launch pair (median of 20 after 5 warm), TFLOP/s\n", m);
  for (int si = 0; si < n_shapes; ++si) {
    const Shape& sh = shapes[si];
    const int n = sh.n, k = sh.k;
    std::vector<uint16_t> act(static_cast<size_t>(m) * k);
    for (auto& v : act) v = dgpp::float_to_bf16_bits(normal(rng));
    std::vector<uint8_t> w(static_cast<size_t>(n) * k);
    std::uniform_int_distribution<int> code(0, 255);
    for (auto& b : w) {
      int c = code(rng);
      if ((c & 0x7F) == 0x7F) c &= 0x7E;
      b = static_cast<uint8_t>(c);
    }
    std::vector<float> ws(static_cast<size_t>((n + 127) / 128) * (k / 128), 0.01f);
    uint16_t *d_act, *d_bridge, *d_out;
    uint8_t *d_w, *d_q;
    float *d_ws, *d_as;
    DGPP_CUDA_OK(cudaMalloc(&d_act, act.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&d_bridge, w.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&d_out, static_cast<size_t>(m) * n * 2));
    DGPP_CUDA_OK(cudaMalloc(&d_w, w.size()));
    DGPP_CUDA_OK(cudaMalloc(&d_q, act.size()));
    DGPP_CUDA_OK(cudaMalloc(&d_ws, ws.size() * 4));
    DGPP_CUDA_OK(cudaMalloc(&d_as, static_cast<size_t>(m) * (k / 128) * 4));
    DGPP_CUDA_OK(cudaMemcpy(d_act, act.data(), act.size() * 2, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d_ws, ws.data(), ws.size() * 4, cudaMemcpyHostToDevice));
    const size_t ws_bytes = std::max<size_t>(64u << 20, gemm.query_workspace_bytes(m, n, k, dgpp::DType::BF16));
    void* d_gws;
    DGPP_CUDA_OK(cudaMalloc(&d_gws, ws_bytes));
    auto time = [&](auto&& fn) {
      for (int i = 0; i < 5; ++i) fn();
      std::vector<float> ms;
      for (int i = 0; i < 20; ++i) {
        DGPP_CUDA_OK(cudaEventRecord(e0, stream));
        fn();
        DGPP_CUDA_OK(cudaEventRecord(e1, stream));
        DGPP_CUDA_OK(cudaEventSynchronize(e1));
        float t;
        DGPP_CUDA_OK(cudaEventElapsedTime(&t, e0, e1));
        ms.push_back(t);
      }
      std::sort(ms.begin(), ms.end());
      return ms[ms.size() / 2];
    };
    const double flop = 2.0 * m * n * static_cast<double>(k);
    const float bridge_ms = time([&] {
      dgpp::launch_fp8_dequant_blocks(d_w, d_ws, d_bridge, n, k, stream);
      gemm.matmul(d_act, d_bridge, d_out, m, n, k, dgpp::DType::BF16, dgpp::GemmOut::BF16, static_cast<size_t>(k),
                  d_gws, ws_bytes, stream);
    });
    const float cublas_ms = time([&] {
      gemm.matmul(d_act, d_bridge, d_out, m, n, k, dgpp::DType::BF16, dgpp::GemmOut::BF16, static_cast<size_t>(k),
                  d_gws, ws_bytes, stream);
    });
    const float fp8_ms = time([&] {
      dgpp::launch_fp8_quantize_rows(d_act, static_cast<size_t>(k), m, k, d_q, d_as, stream);
      dgpp::launch_fp8_gemm_bf16(d_q, d_as, d_w, d_ws, 128, d_out, m, n, k, stream);
    });
    const float fp8_gemm_ms = time([&] { dgpp::launch_fp8_gemm_bf16(d_q, d_as, d_w, d_ws, 128, d_out, m, n, k, stream); });
    // The exact fp8-weight tile GEMM (bf16 activations, the weights decoded in
    // the kernel): what the scale GEMM runs above 128 rows without a bridge.
    const float dense_ms = time([&] {
      dgpp::launch_dense_mma_bf16(d_act, static_cast<size_t>(k), d_w, d_ws, d_out, m, n, k, stream);
    });
    const float dequant_ms = time([&] { dgpp::launch_fp8_dequant_blocks(d_w, d_ws, d_bridge, n, k, stream); });
    // The exact fp8-weight GEMM (kernels/fp8w_gemm, #87): the bridge's
    // weights decoded in shared memory, the bf16 mma pipeline, no bridge.
    const float fp8w_ms = time([&] {
      dgpp::launch_fp8w_gemm_bf16(d_act, static_cast<size_t>(k), d_w, d_ws, d_out, m, n, k,
                                  dgpp::Fp8wScale::PerWeight, stream);
    });
    const float fp8w_group_ms = time([&] {
      dgpp::launch_fp8w_gemm_bf16(d_act, static_cast<size_t>(k), d_w, d_ws, d_out, m, n, k,
                                  dgpp::Fp8wScale::PerGroup, stream);
    });
    // The pipelined bridge (#87): a chain of kProducts products through two
    // bridge buffers, product i's dequant on a side stream under product
    // i-1's GEMM. Reported per product, for the side stream at the same
    // priority as the main stream and at the device's highest priority (the
    // block scheduler feeds the older grid first, so an equal-priority
    // dequant only gets the GEMM's tail; a higher-priority one interleaves).
    constexpr int kProducts = 6;
    uint16_t* d_bridge2;
    DGPP_CUDA_OK(cudaMalloc(&d_bridge2, w.size() * 2));
    uint16_t* bufs[2] = {d_bridge, d_bridge2};
    int lo, hi;
    DGPP_CUDA_OK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    auto pipelined = [&](int priority) {
      cudaStream_t side;
      DGPP_CUDA_OK(cudaStreamCreateWithPriority(&side, cudaStreamNonBlocking, priority));
      cudaEvent_t freed[2], ready[2];
      for (int i = 0; i < 2; ++i) {
        DGPP_CUDA_OK(cudaEventCreateWithFlags(&freed[i], cudaEventDisableTiming));
        DGPP_CUDA_OK(cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming));
        DGPP_CUDA_OK(cudaEventRecord(freed[i], stream));
      }
      const float ms = time([&] {
        for (int i = 0; i < kProducts; ++i) {
          const int b = i & 1;
          DGPP_CUDA_OK(cudaStreamWaitEvent(side, freed[b], 0));
          dgpp::launch_fp8_dequant_blocks(d_w, d_ws, bufs[b], n, k, side);
          DGPP_CUDA_OK(cudaEventRecord(ready[b], side));
          DGPP_CUDA_OK(cudaStreamWaitEvent(stream, ready[b], 0));
          gemm.matmul(d_act, bufs[b], d_out, m, n, k, dgpp::DType::BF16, dgpp::GemmOut::BF16,
                      static_cast<size_t>(k), d_gws, ws_bytes, stream);
          DGPP_CUDA_OK(cudaEventRecord(freed[b], stream));
        }
      });
      DGPP_CUDA_OK(cudaStreamSynchronize(side));
      for (int i = 0; i < 2; ++i) {
        cudaEventDestroy(freed[i]);
        cudaEventDestroy(ready[i]);
      }
      cudaStreamDestroy(side);
      return ms / kProducts;
    };
    const float serial_chain_ms = time([&] {
      for (int i = 0; i < kProducts; ++i) {
        dgpp::launch_fp8_dequant_blocks(d_w, d_ws, bufs[i & 1], n, k, stream);
        gemm.matmul(d_act, bufs[i & 1], d_out, m, n, k, dgpp::DType::BF16, dgpp::GemmOut::BF16,
                    static_cast<size_t>(k), d_gws, ws_bytes, stream);
      }
    }) / kProducts;
    const float pipe_same_ms = pipelined(lo);
    const float pipe_high_ms = pipelined(hi);
    cudaFree(d_bridge2);
    std::printf("%-28s dequant+cuBLAS %7.3f ms (GEMM alone %7.3f, %5.0f TF; dequant alone %6.3f)  |  quant+fp8 %7.3f ms (GEMM alone %7.3f, %5.0f TF)  %+5.1f %%  |  fp8-weight tile %7.3f ms (%5.0f TF)\n",
                sh.what, bridge_ms, cublas_ms, flop / cublas_ms / 1e9, dequant_ms, fp8_ms, fp8_gemm_ms, flop / fp8_gemm_ms / 1e9,
                100.0 * (fp8_ms - bridge_ms) / bridge_ms, dense_ms, flop / dense_ms / 1e9);
    // The slabbed bridge: dequant a slab of weight rows, GEMM it, next slab —
    // a slab small enough to stay in L2 makes the bridge's write and read L2
    // traffic instead of DRAM traffic (the same two kernels, the same
    // numerics per output element up to the Lt heuristic per slab shape).
    {
      std::printf("%-28s slabbed bridge:", "");
      const int slab_rows[] = {512, 1024, 2048, 4096};
      for (int rows : slab_rows) {
        if (rows > n || n % rows != 0) continue;
        const float ms = time([&] {
          for (int r0 = 0; r0 < n; r0 += rows) {
            dgpp::launch_fp8_dequant_blocks(d_w + static_cast<size_t>(r0) * k, d_ws + static_cast<size_t>(r0 / 128) * (k / 128),
                                            d_bridge, rows, k, stream);
            gemm.matmul(d_act, d_bridge, d_out + static_cast<size_t>(r0) * m, m, rows, k, dgpp::DType::BF16,
                        dgpp::GemmOut::BF16, static_cast<size_t>(k), d_gws, ws_bytes, stream);
          }
        });
        std::printf("  %d rows %7.3f ms (%+5.1f %%)", rows, ms, 100.0 * (ms - bridge_ms) / bridge_ms);
      }
      std::printf("\n");
    }
    std::printf("%-28s fp8w per-weight %7.3f ms (%5.0f TF, %+5.1f %% vs the bridge) | per-group %7.3f ms (%5.0f TF, %+5.1f %%)\n", "",
                fp8w_ms, flop / fp8w_ms / 1e9, 100.0 * (fp8w_ms - bridge_ms) / bridge_ms, fp8w_group_ms,
                flop / fp8w_group_ms / 1e9, 100.0 * (fp8w_group_ms - bridge_ms) / bridge_ms);
    std::printf("%-28s pipelined bridge per product: serial %7.3f ms | side stream same priority %7.3f ms (%+5.1f %%) | highest priority %7.3f ms (%+5.1f %%)\n",
                "", serial_chain_ms, pipe_same_ms, 100.0 * (pipe_same_ms - serial_chain_ms) / serial_chain_ms,
                pipe_high_ms, 100.0 * (pipe_high_ms - serial_chain_ms) / serial_chain_ms);
    cudaFree(d_act); cudaFree(d_bridge); cudaFree(d_out); cudaFree(d_w); cudaFree(d_q); cudaFree(d_ws); cudaFree(d_as); cudaFree(d_gws);
  }
  return 0;
}
