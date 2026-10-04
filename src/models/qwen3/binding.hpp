#pragma once
// Expected-tensor table for the plain Qwen3 family (2026-10-04): every
// tensor the checkpoint must contain — names, dtypes, exact shapes — derived
// from the parsed config, not observed from one file. The table drives the
// offline validator and the resident loader, as the other families' do.
//
// Naming is checkpoint truth, read from the releases' safetensors headers:
//
//   VlMoe  ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4 @ 3c6162d5 — 75,090 tensors.
//          `model.language_model.layers.L.`, `model.language_model.embed_tokens.weight`,
//          `model.language_model.norm.weight`, `lm_head.weight`, and the
//          vision tower under `model.visual.` (351 BF16 tensors: bound here,
//          never loaded). compressed-tensors NVFP4: the attention q/k/v/o and
//          every routed expert's gate/up/down as `weight_packed` U8 [N, K/2],
//          `weight_scale` e4m3 [N, K/16], `weight_global_scale` F32 [1] (a
//          DIVISOR) and `input_global_scale` F32 [1]; the router `mlp.gate`,
//          the head, the norms and the tower BF16. The experts are stored
//          per expert (`mlp.experts.E.{gate,up,down}_proj`) — llm-compressor
//          unpacks transformers' fused `gate_up_proj` / `down_proj` into
//          Linears before it quantizes them.
//   Moe    nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4 @ 1c4ec358 — 145,703
//          tensors. `model.layers.L.`, `model.embed_tokens.weight`,
//          `model.norm.weight`, `lm_head.weight`. modelopt NVFP4: the routed
//          experts and the attention o_proj as `weight` U8 [N, K/2],
//          `weight_scale` e4m3 [N, K/16], `weight_scale_2` F32 [] (a
//          multiplier) and `input_scale` F32 []; q/k/v, the router and the
//          head BF16; the FP8 K/V-cache scales `k_proj.k_scale` /
//          `v_proj.v_scale` F32 [] (bound; nothing reads them).
//   Dense  Qwen/Qwen3-Reranker-0.6B @ e61197ed — 310 tensors, BF16, the Moe
//          names with a dense `mlp.{gate,up,down}_proj.weight`, no head (tied).
//          Qwen/Qwen3-Embedding-0.6B @ 97b0c614 — the same 310 tensors under
//          BARE names (`layers.L.`, `embed_tokens.weight`, `norm.weight`): the
//          base model's state dict. The header says which
//          (qwen3_apply_header_naming).
//
// Shapes: q_proj [heads * head_dim, H], k_proj / v_proj [kv_heads * head_dim, H],
// o_proj [H, heads * head_dim] — heads * head_dim need not equal H (the 0.6B
// models: 16 x 128 over H = 1024); q_norm / k_norm [head_dim].
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/qwen3/config.hpp"

namespace dgpp {

enum class Qwen3WeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  LayerNorm,     // input_layernorm, post_attention_layernorm
  Attention,     // q/k/v/o projections and the q/k head norms
  KvScale,       // modelopt's k_scale / v_scale (unused)
  Router,        // mlp.gate.weight
  DenseMlp,      // the Dense dialect's gate/up/down
  RoutedExpert,
  Vision,        // model.visual.* (bound, never loaded)
};

enum class Qwen3TensorRole : uint8_t {
  Plain,
  Fp4Payload,   // U8 [N, K/2] e2m1 pairs, low nibble = even column
  Fp4Scale,     // F8_E4M3 [N, K/16]
  Fp4Global,    // F32: modelopt's weight_scale_2 [] or compressed-tensors' weight_global_scale [1]
  InputScale,   // F32: the recipe's activation scale (unused: W4A16)
  KvScale,      // F32 []: the FP8 K/V-cache scale (unused)
};

struct Qwen3ExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  Qwen3WeightClass cls = Qwen3WeightClass::Attention;
  int layer = -1;   // layer index; -1 for globals and the tower
  int expert = -1;  // routed-expert id, -1 otherwise
  Qwen3TensorRole role = Qwen3TensorRole::Plain;

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
  bool quantized() const { return role == Qwen3TensorRole::Fp4Payload; }
  // Bound and never read by the text path.
  bool unused() const {
    return role == Qwen3TensorRole::InputScale || role == Qwen3TensorRole::KvScale ||
           cls == Qwen3WeightClass::Vision;
  }
};

// "model.language_model." (VlMoe), "model." (Moe, Dense) or "" (the bare
// Embedding names): the prefix of `layers.L.`, `embed_tokens.weight` and
// `norm.weight`.
std::string qwen3_model_prefix(const Qwen3TextConfig& cfg);
std::string qwen3_layer_prefix(const Qwen3TextConfig& cfg, int layer);

// The names of one NVFP4 matrix in the config's container (`base` is the
// module name without a suffix, e.g. "...self_attn.o_proj").
std::string qwen3_fp4_payload_name(const Qwen3TextConfig& cfg, const std::string& base);
std::string qwen3_fp4_global_name(const Qwen3TextConfig& cfg, const std::string& base);
std::string qwen3_fp4_input_name(const Qwen3TextConfig& cfg, const std::string& base);

// One layer's entries: `layer` in [0, num_hidden_layers).
std::vector<Qwen3ExpectedTensor> qwen3_expected_layer_tensors(const Qwen3TextConfig& cfg, int layer);
// The text globals: embed, the final norm, and lm_head unless tied.
std::vector<Qwen3ExpectedTensor> qwen3_expected_global_tensors(const Qwen3TextConfig& cfg);
// The text model: globals + every layer.
std::vector<Qwen3ExpectedTensor> qwen3_expected_text_tensors(const Qwen3TextConfig& cfg);
// The vision tower's entries (empty without one).
std::vector<Qwen3ExpectedTensor> qwen3_expected_vision_tensors(const Qwen3TextConfig& cfg);
// The whole checkpoint: text + tower. Its size is the header's tensor count.
std::vector<Qwen3ExpectedTensor> qwen3_expected_tensors(const Qwen3TextConfig& cfg);

struct Qwen3TensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};
using Qwen3PresentMap = std::unordered_map<std::string, Qwen3TensorDesc>;

// Sets cfg.bare_names from what the header carries: `embed_tokens.weight`
// without `model.embed_tokens.weight` is the base-model state dict (the
// Dense dialect only; the others keep their prefix).
void qwen3_apply_header_naming(Qwen3TextConfig& cfg, const Qwen3PresentMap& present);

struct Qwen3BindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t out_of_scope = 0;   // layers past a truncated config (the check apps' --layers)
  size_t fp4_matrices = 0;
  size_t vision = 0;         // matched model.visual.* tensors (bound, not served)
  size_t tied_head_copies = 0;  // a stored lm_head.weight beside tied embeddings (accepted, not read)
  size_t bytes = 0;          // the matched tensors' bytes
  std::vector<std::string> errors;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

Qwen3BindReport qwen3_validate_binding(const Qwen3TextConfig& cfg, const Qwen3PresentMap& present,
                                       size_t max_errors = 32);

// Tensor-parallel geometry acceptance (GLM-4.7's rules, the same attention
// and expert kernels): throws std::invalid_argument naming the dim that
// does not divide. world must divide the query heads; either world divides
// the kv heads or the kv heads divide world; every expert / dense
// intermediate slice must be a multiple of 32 (the NVFP4 GEMV core's K and
// the 16-block column slice of down_proj).
void qwen3_tp_validate_geometry(const Qwen3TextConfig& cfg, int rank, int world);

// What one rank of a world holds of the text model, in the resident forms
// the draft loader builds (models/qwen3/loader.hpp) — the sharding plan's
// arithmetic, from the table, without a checkpoint:
//   experts      NVFP4 payload + block scales as shipped, I/world rows of
//                gate/up and columns of down, + one F32 global per matrix
//   attention    q/k/v/o in BF16 (an NVFP4 projection dequantized at load):
//                heads/world rows of q and columns of o, the rank's kv
//                heads' rows of k and v
//   replicated   the router, both layer norms, q/k norms, the final norm,
//                the embedding
//   head         lm_head rows [V * rank / world, V * (rank + 1) / world)
//                (the embedding's bytes are the head's when tied)
struct Qwen3RankBytes {
  size_t experts = 0;
  size_t attention = 0;
  size_t dense_mlp = 0;
  size_t router = 0;
  size_t norms = 0;
  size_t embed = 0;
  size_t head = 0;
  size_t total() const { return experts + attention + dense_mlp + router + norms + embed + head; }
};
Qwen3RankBytes qwen3_rank_resident_bytes(const Qwen3TextConfig& cfg, int rank, int world);

}  // namespace dgpp
