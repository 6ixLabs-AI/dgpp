#pragma once
// Expected-tensor table for MiniMax-M2.7 (MiniMaxM2ForCausalLM, 2026-10-04,
// docs/minimax_m27_plan.md §2): every tensor the checkpoint must contain —
// names, dtypes, exact shapes — derived from the parsed config, not
// observed from one file. The table drives the offline validator and the
// resident loader, as the other families' tables do.
//
// Naming is checkpoint truth (lukealonso/MiniMax-M2.7-NVFP4 @ db821d7a):
// layers under `model.layers.L.`, the attention as `self_attn.{q_proj,
// k_proj, v_proj, o_proj, q_norm, k_norm}`, the MoE as `block_sparse_moe.
// {gate, e_score_correction_bias, experts.E.{w1, w2, w3}}` (w1 the gate
// projection, w3 the up projection, w2 the down projection), the globals
// `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`.
//
// Format contract (modelopt NVFP4): a routed [N, K] matrix X is the triple
// X.weight U8 [N, K/2] (e2m1 pairs), X.weight_scale F8_E4M3 [N, K/16],
// X.weight_scale_2 F32 [] (a multiplier). Everything else is BF16 — the
// router bias too (GLM-4.7's is F32).
//
// X.input_scale F32 [] — the W4A4 activation scale — is OPTIONAL and
// unused: the release keeps them in model-inputscales.safetensors and has
// none for experts its calibration never routed to (18 of 47,616 are
// absent). A present one must be an F32 scalar; an absent one is not an
// error. The required tensors number 143,471; with the 47,598 activation
// scales the release carries, 191,069 — the count of
// model.safetensors.index.json.
//
// THE FILE LIST IS THE INDEX'S. The repository also holds safetensors files
// that are not the model (calibration by-products: ten
// model-*inputscales*.safetensors variants repeating the input_scale
// names, eight amax*.safetensors under other names, three *.bak copies).
// minimax_shard_files() reads model.safetensors.index.json; a loader that
// globs the directory would bind duplicates.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"
#include "models/minimax/config.hpp"

namespace dgpp {

enum class MinimaxWeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  LayerNorm,  // input_layernorm, post_attention_layernorm
  Attention,  // q/k/v/o projections, the per-layer q/k norms
  Router,     // block_sparse_moe.gate.weight, e_score_correction_bias
  RoutedExpert,
};

enum class MinimaxTensorRole : uint8_t {
  Plain,
  Fp4Payload,  // U8 [N, K/2]
  Fp4Scale,    // F8_E4M3 [N, K/16]
  Fp4Global,   // F32 [] — weight_scale_2, a multiplier
  InputScale,  // F32 [] — the W4A4 activation scale (optional, unused)
};

struct MinimaxExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  MinimaxWeightClass cls = MinimaxWeightClass::Attention;
  int layer = -1;   // layer index; -1 for globals
  int expert = -1;  // routed-expert id, -1 otherwise
  MinimaxTensorRole role = MinimaxTensorRole::Plain;

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
  bool optional() const { return role == MinimaxTensorRole::InputScale; }
  bool unused() const { return role == MinimaxTensorRole::InputScale; }
};

// The checkpoint name prefix of a layer ("model.layers.L.").
std::string minimax_layer_prefix(int layer);

// The table: the required tensors and, per NVFP4 matrix, its optional
// activation scale.
std::vector<MinimaxExpectedTensor> minimax_expected_tensors(const MinimaxTextConfig& cfg);
std::vector<MinimaxExpectedTensor> minimax_expected_layer_tensors(const MinimaxTextConfig& cfg,
                                                                  int layer);
std::vector<MinimaxExpectedTensor> minimax_expected_global_tensors(const MinimaxTextConfig& cfg);

struct MinimaxTensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct MinimaxBindReport {
  size_t expected = 0;  // required tensors of the table
  size_t matched = 0;   // required tensors bound
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t fp4_matrices = 0;
  size_t optional_present = 0;  // activation scales found (and well-formed)
  size_t optional_absent = 0;   // activation scales the file does not have
  size_t bytes = 0;             // matched + optional_present tensors' bytes as stored
  size_t loaded_bytes = 0;      // of those, the bytes a load reads (the required tensors)
  std::vector<std::string> errors;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
  // Every tensor of the file accounted for.
  size_t bound() const { return matched + optional_present; }
};

MinimaxBindReport minimax_validate_binding(
    const MinimaxTextConfig& cfg, const std::unordered_map<std::string, MinimaxTensorDesc>& present,
    size_t max_errors = 32);

// The shard files of a snapshot, from a parsed model.safetensors.index.json
// root: the distinct values of `weight_map`, sorted. Throws
// std::runtime_error when the index has no weight_map.
std::vector<std::string> minimax_shard_files(const minijson::Value& index_root);

// The bytes a rank of `world` holds of the weights AS STORED (no
// re-encoding at load), under the sharding of docs/minimax_m27_plan.md §4:
// the experts' intermediate dimension, v_proj / o_proj by head and the head
// by vocabulary split `world` ways; the embedding, the router, its bias and
// the norms replicated. q_proj and k_proj are head-sharded like
// v_proj when `replicate_qk` is false (plan option B: the per-layer norm's
// sum of squares crosses the bus) and replicated when it is true (option A,
// the plan's choice: every rank computes the whole q / k row). world 1 is
// the whole model.
struct MinimaxWeightBytes {
  size_t embedding = 0;
  size_t head = 0;
  size_t attention = 0;
  size_t routers_and_norms = 0;
  size_t routed_experts = 0;
  size_t total() const { return embedding + head + attention + routers_and_norms + routed_experts; }
};
MinimaxWeightBytes minimax_weight_bytes(const MinimaxTextConfig& cfg, int world = 1,
                                        bool replicate_qk = true);

// The paged K/V cache's bytes per token on one rank: bf16 K and V rows of
// the rank's kv heads, every layer.
size_t minimax_kv_bytes_per_token(const MinimaxTextConfig& cfg, int world = 1);

// Tensor-parallel geometry acceptance: throws std::invalid_argument naming
// the dim that does not divide. world must divide the query heads, the kv
// heads and the vocabulary; a rank's query heads must be whole kv groups;
// the expert intermediate slice must be a multiple of 32 (the NVFP4 GEMV
// core's K and the 16-block column slice).
void minimax_tp_validate_geometry(const MinimaxTextConfig& cfg, int rank, int world);

}  // namespace dgpp
