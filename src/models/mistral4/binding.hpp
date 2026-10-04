#pragma once
// Expected-tensor table for Mistral-Small-4 (2026-10-04,
// docs/mistral_small4_plan.md §2): every tensor the checkpoint must contain
// — names, dtypes, exact shapes — derived from the parsed params.json, not
// observed from one file. The table drives the offline validator and the
// resident loader, as the other families' tables do.
//
// Naming is checkpoint truth (mistralai/Mistral-Small-4-119B-2603-NVFP4 @
// 45331841, 56,313 tensors in 13 shards) and it is Mistral's, not
// transformers': layers under `layers.L.`, the attention as
// `attention.{wq_a, q_a_norm, wq_b, wkv_a_with_mqa, kv_a_norm, wkv_b, wo}`,
// the norms `attention_norm` / `ffn_norm`, the router `gate`, the experts
// `experts.E.{w1, w2, w3}` and `shared_experts.{w1, w2, w3}` (w1 the gate
// projection, w3 the up projection, w2 the down projection), the globals
// `tok_embeddings.weight`, `norm.weight`, `output.weight`.
//
// Format contract (compressed-tensors NVFP4): an [N, K] expert matrix X is
// X.weight_packed U8 [N, K/2], X.weight_scale F8_E4M3 [N, K/16],
// X.weight_global_scale F32 [1] (a divisor) and X.input_global_scale [1],
// the unused activation scale — F32 on the routed experts, BF16 on the
// shared ones (the release's own inconsistency, pinned here because a
// dtype is part of the binding). Everything else is BF16 `.weight`.
//
// The Pixtral tower (`vision_encoder.*`), its projector
// (`vision_language_adapter.*`, `patch_merger.*`, `pre_mm_projector_norm.*`)
// are expected with their shapes and marked unused: a text-only engine
// checks that the file is the release it claims to be and loads none of it.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/mistral4/config.hpp"

namespace dgpp {

enum class Mistral4WeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  LayerNorm,  // attention_norm, ffn_norm
  Attention,  // wq_a, q_a_norm, wq_b, wkv_a_with_mqa, kv_a_norm, wkv_b, wo
  Router,     // gate.weight
  SharedExpert,
  RoutedExpert,
  Vision,  // the tower and its projector (never loaded)
};

enum class Mistral4TensorRole : uint8_t {
  Plain,
  Fp4Payload,  // U8 [N, K/2]
  Fp4Scale,    // F8_E4M3 [N, K/16]
  Fp4Global,   // F32 [1] — weight_global_scale, a divisor
  InputScale,  // [1] — the W4A4 activation scale (unused)
  Unused,      // a vision tensor
};

struct Mistral4ExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  Mistral4WeightClass cls = Mistral4WeightClass::Attention;
  int layer = -1;   // text layer index; -1 for globals and the vision tensors
  int expert = -1;  // routed-expert id, -1 otherwise
  Mistral4TensorRole role = Mistral4TensorRole::Plain;

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
  bool unused() const {
    return role == Mistral4TensorRole::InputScale || role == Mistral4TensorRole::Unused;
  }
};

// The checkpoint name prefix of a text layer ("layers.L.").
std::string mistral4_layer_prefix(int layer);

// Everything the checkpoint holds: globals, every text layer, the vision
// tensors when params.json declares a tower.
std::vector<Mistral4ExpectedTensor> mistral4_expected_tensors(const Mistral4TextConfig& cfg);
std::vector<Mistral4ExpectedTensor> mistral4_expected_layer_tensors(const Mistral4TextConfig& cfg,
                                                                    int layer);
std::vector<Mistral4ExpectedTensor> mistral4_expected_global_tensors(const Mistral4TextConfig& cfg);
std::vector<Mistral4ExpectedTensor> mistral4_expected_vision_tensors(const Mistral4TextConfig& cfg);

struct Mistral4TensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct Mistral4BindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t fp4_matrices = 0;
  size_t unused = 0;        // matched tensors the text engine never loads
  size_t bytes = 0;         // matched tensors' bytes as stored
  size_t loaded_bytes = 0;  // of those, the bytes a text-only load reads (everything not unused)
  std::vector<std::string> errors;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

Mistral4BindReport mistral4_validate_binding(
    const Mistral4TextConfig& cfg,
    const std::unordered_map<std::string, Mistral4TensorDesc>& present, size_t max_errors = 32);

// The bytes a rank of `world` holds of the text weights AS STORED (no
// re-encoding at load): the experts' intermediate dimension, the attention
// heads and the head's vocabulary split `world` ways, the routers, norms,
// the query / key latents' projections and the embedding replicated.
// world 1 is the whole text model. docs/mistral_small4_plan.md §3.
struct Mistral4WeightBytes {
  size_t embedding = 0;
  size_t head = 0;
  size_t attention = 0;
  size_t routers_and_norms = 0;
  size_t shared_experts = 0;
  size_t routed_experts = 0;
  size_t total() const {
    return embedding + head + attention + routers_and_norms + shared_experts + routed_experts;
  }
};
Mistral4WeightBytes mistral4_weight_bytes(const Mistral4TextConfig& cfg, int world = 1);

// Tensor-parallel geometry acceptance: throws std::invalid_argument naming
// the dim that does not divide. world must divide the attention heads, the
// vocabulary and every expert intermediate, and an intermediate slice must
// be a multiple of 32 (the NVFP4 GEMV core's K and the 16-block column
// slice).
void mistral4_tp_validate_geometry(const Mistral4TextConfig& cfg, int rank, int world);

}  // namespace dgpp
