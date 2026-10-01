// The opt-in fp8 prefill GEMM (kernels/fp8_gemm, engine.prefill_fp8_gemm)
// against the chain it replaces at the hybrid's dense prefill shapes: the
// block-FP8 dequant into a bf16 bridge plus the cuBLASLt bf16 GEMM, versus
// the per-token e4m3 quantizer plus the fp8 tensor-core GEMM. Prints ms per
// launch pair and the tensor throughput. Usage: fp8_gemm_bench [m]
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/fp8_dequant.hpp"
#include "kernels/fp8_gemm.hpp"
#include "kernels/gemm.hpp"

int main(int argc, char** argv) {
  const int m = argc > 1 ? std::atoi(argv[1]) : 4096;
  struct Shape {
    int n, k;
    const char* what;
  };
  const Shape shapes[] = {{2560, 2560, "q/k/v-like [2560 x 2560]"},   {5120, 2560, "in_proj-like [5120 x 2560]"},
                          {2560, 5120, "out_proj-like [2560 x 5120]"}, {320, 10240, "GR down [320 x 10240]"},
                          {10240, 2560, "PLE key [10240 x 2560]"}};
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.f, 1.f);
  dgpp::CublasLtGemm gemm;
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  cudaEvent_t e0, e1;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  std::printf("m = %d rows; ms per launch pair (median of 20 after 5 warm), TFLOP/s\n", m);
  for (const Shape& sh : shapes) {
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
    std::printf("%-28s dequant+cuBLAS %7.3f ms (GEMM alone %7.3f, %5.0f TF)  |  quant+fp8 %7.3f ms (GEMM alone %7.3f, %5.0f TF)  %+5.1f %%\n",
                sh.what, bridge_ms, cublas_ms, flop / cublas_ms / 1e9, fp8_ms, fp8_gemm_ms, flop / fp8_gemm_ms / 1e9,
                100.0 * (fp8_ms - bridge_ms) / bridge_ms);
    cudaFree(d_act); cudaFree(d_bridge); cudaFree(d_out); cudaFree(d_w); cudaFree(d_q); cudaFree(d_ws); cudaFree(d_as); cudaFree(d_gws);
  }
  return 0;
}
