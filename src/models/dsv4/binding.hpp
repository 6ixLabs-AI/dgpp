#pragma once
// Expected-tensor table for DeepSeek-V4-Flash (DeepseekV4ForCausalLM, the
// 0731 release; 2026-10-01): every tensor the checkpoint must contain —
// names, dtypes, exact shapes — derived from the parsed config, not
// observed from one file. The table drives the offline validator and the
// resident loader, as the other families' tables do.
//
// Naming is checkpoint truth (deepseek-ai/DeepSeek-V4-Flash-0731, no
// `model.` prefix): backbone layers under `layers.L.`, the DSpark draft
// stages under `mtp.S.`, the globals `embed.weight`, `norm.weight`,
// `head.weight`, `hc_head_{fn,base,scale}`.
//
// Format contract: an fp8 matrix X [N, K] is the pair X.weight F8_E4M3
// [N, K] + X.scale F8_E8M0 [ceil(N/128), ceil(K/128)]; an MXFP4 matrix is
// X.weight I8 [N, K/2] (two e2m1 codes per byte, the low nibble the even
// element) + X.scale F8_E8M0 [N, K/32]. The dequantized value is `code x
// 2^(scale - 127)`, exact in bf16 for both code formats. Everything else
// is BF16, F32 or I64 as listed.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/dsv4/config.hpp"

namespace dgpp {

enum class Dsv4WeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  LayerNorm,      // attn_norm, ffn_norm, q_norm, kv_norm, the compressors' norms, the draft's norms
  Attention,      // wq_a, wq_b, wkv, wo_a, wo_b, attn_sink
  Indexer,        // indexer.wq_b, weights_proj, indexer.compressor.*
  Compressor,     // compressor.wkv, wgate, ape
  Router,         // ffn.gate.weight, .bias, .tid2eid
  SharedExpert,
  RoutedExpert,
  Mhc,            // hc_attn_fn/base/scale, hc_ffn_fn/base/scale, hc_head_fn/base/scale
  Draft,          // main_proj, main_norm, markov_head, confidence_head (the draft's own extras)
};

enum class Dsv4TensorRole : uint8_t {
  Plain,        // BF16 / F32 / I64 as stored
  Fp8Payload,   // F8_E4M3 [N, K]
  Fp8Scale,     // F8_E8M0 [N/128, K/128]
  Fp4Payload,   // I8 [N, K/2]
  Fp4Scale,     // F8_E8M0 [N, K/32]
};

struct Dsv4ExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  Dsv4WeightClass cls = Dsv4WeightClass::Attention;
  int layer = -1;   // layer index (draft stages at num_hidden_layers + S); -1 for globals
  int expert = -1;  // routed-expert id, -1 otherwise
  Dsv4TensorRole role = Dsv4TensorRole::Plain;

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
  bool skipped() const { return false; }
};

// The checkpoint name prefix of a layer: "layers.L." or "mtp.S.".
std::string dsv4_layer_prefix(const Dsv4Config& cfg, int layer);

std::vector<Dsv4ExpectedTensor> dsv4_expected_tensors(const Dsv4Config& cfg);
std::vector<Dsv4ExpectedTensor> dsv4_expected_layer_tensors(const Dsv4Config& cfg, int layer);
std::vector<Dsv4ExpectedTensor> dsv4_expected_global_tensors(const Dsv4Config& cfg);

struct Dsv4TensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct Dsv4BindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t fp4_matrices = 0;      // routed + draft expert matrices
  size_t fp8_matrices = 0;      // dense projections
  std::vector<std::string> errors;
  // Tensors of backbone layers beyond the config's stack (a truncated
  // diagnostic forward over the first layers): counted, not errors.
  int beyond_stack = 0;
  bool ok() const { return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0; }
};

Dsv4BindReport dsv4_validate_binding(const Dsv4Config& cfg,
                                     const std::unordered_map<std::string, Dsv4TensorDesc>& present,
                                     size_t max_errors = 32);

// Tensor-parallel geometry acceptance: throws std::invalid_argument naming
// the dim that does not divide. world must divide the attention heads
// (whole output groups per rank, 16-head attention tiles) and the
// vocabulary, and every sliced axis of an fp8 matrix must land on its
// 128-block grid (wq_b's head rows, wo_a's group rows, wo_b's group
// columns, the shared expert's intermediate slice) with the routed
// experts' intermediate slice on the MXFP4 32-block.
void dsv4_tp_validate_geometry(const Dsv4Config& cfg, int rank, int world);

}  // namespace dgpp
