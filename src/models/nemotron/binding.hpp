#pragma once
// Expected-tensor table for Nemotron-3 (NemotronHForCausalLM, 2026-10-04,
// docs/nemotron3_plan.md §2): every tensor the checkpoint must contain —
// names, dtypes, exact shapes — derived from the parsed config and its
// quantization recipe, not observed from one file. The table drives the
// offline validator and, when it lands, the resident loader, as the other
// families' tables do.
//
// Naming is checkpoint truth (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 @
// 6efb4a2a, 24,147 tensors in 5 shards; nvidia/NVIDIA-Nemotron-3-Super-
// 120B-A12B-NVFP4 @ 4f0cf9da, 165,860 tensors in 17 shards): the backbone
// under `backbone.layers.L.` — `norm.weight` and one `mixer.` per layer —
// the globals `backbone.embeddings.weight`, `backbone.norm_f.weight`,
// `lm_head.weight`, and Super's draft block under `mtp.layers.{0,1}.`.
//
//   Mamba2     mixer.{A_log, D, dt_bias} [heads], mixer.conv1d.{weight
//              [conv_dim, 1, K], bias [conv_dim]}, mixer.norm.weight [inner]
//              (the gated norm), mixer.in_proj [inner + conv_dim + heads, H],
//              mixer.out_proj [H, inner]
//   attention  mixer.{q_proj [heads x d, H], k_proj, v_proj [kv x d, H],
//              o_proj [H, heads x d]}; under the MixedPrecision recipe the
//              backbone layers also carry F32 scalars mixer.k_proj.k_scale
//              and mixer.v_proj.v_scale (the recipe's FP8 K/V-cache scales)
//   MoE        mixer.gate.weight [E, H] (F32 in Nano, BF16 in Super),
//              mixer.gate.e_score_correction_bias F32 [E],
//              mixer.experts.E.{up_proj [I, X], down_proj [X, I]} with X the
//              latent width or H, mixer.shared_experts.{up_proj [S, H],
//              down_proj [H, S]}, and with a latent
//              mixer.fc1_latent_proj [X, H], mixer.fc2_latent_proj [H, X]
//   draft      mtp.layers.0.{enorm, hnorm}.weight [H], eh_proj.weight
//              [H, 2H], then an attention layer; mtp.layers.1 a MoE layer,
//              then final_layernorm.weight [H]. All BF16.
//
// Format contract (modelopt), per nn.Linear X as the recipe names it:
//   NVFP4  X.weight U8 [N, K/2] (e2m1 pairs, low nibble = even column),
//          X.weight_scale F8_E4M3 [N, K/16], X.weight_scale_2 F32 [] (a
//          multiplier), X.input_scale F32 [] (the activation scale; unused:
//          the engine runs BF16 activations against the dequantized weight)
//   FP8    X.weight F8_E4M3 [N, K], X.weight_scale F32 [] (a multiplier, one
//          per tensor), X.input_scale F32 []
//   BF16   X.weight BF16 [N, K]
// Every vector (norms, A_log, D, dt_bias, the convolution) is BF16.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/nemotron/config.hpp"

namespace dgpp {

enum class NemotronWeightClass : int {
  Embed,
  LmHead,
  Norm,        // the layers' pre-norms, norm_f
  Mamba,       // A_log, D, dt_bias, conv1d, the gated norm, in_proj, out_proj
  Attention,   // q/k/v/o and the K/V-cache scales
  Router,      // gate.weight, gate.e_score_correction_bias
  LatentProj,  // fc1_latent_proj, fc2_latent_proj
  SharedExpert,
  RoutedExpert,
  MtpHead,  // enorm, hnorm, eh_proj, final_layernorm
};

enum class NemotronTensorRole : uint8_t {
  Plain,
  Fp4Payload,  // U8 [N, K/2]
  Fp4Scale,    // F8_E4M3 [N, K/16]
  Fp4Global,   // F32 [], weight_scale_2
  Fp8Payload,  // F8_E4M3 [N, K]
  Fp8Scale,    // F32 [], the per-tensor weight_scale
  InputScale,  // F32 [], the recipe's activation scale (unused)
  KvScale,     // F32 [], k_scale / v_scale (read only for an fp8 K/V cache)
};

struct NemotronExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  NemotronWeightClass cls = NemotronWeightClass::Norm;
  int layer = -1;   // layer index; -1 for globals; num_hidden_layers + M for the draft block
  int expert = -1;  // routed-expert id, -1 otherwise
  NemotronTensorRole role = NemotronTensorRole::Plain;

  bool quantized() const {
    return role == NemotronTensorRole::Fp4Payload || role == NemotronTensorRole::Fp8Payload;
  }
  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
};

// Full table (backbone layers + the draft block + globals).
std::vector<NemotronExpectedTensor> nemotron_expected_tensors(const NemotronHConfig& cfg);
// One layer's entries: `layer` in [0, num_layers_total()).
std::vector<NemotronExpectedTensor> nemotron_expected_layer_tensors(const NemotronHConfig& cfg,
                                                                    int layer);
// The globals: the embedding, the final norm, the head.
std::vector<NemotronExpectedTensor> nemotron_expected_global_tensors(const NemotronHConfig& cfg);

struct NemotronTensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct NemotronBindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t fp8_matrices = 0;
  size_t fp4_matrices = 0;
  size_t bytes = 0;  // of the matched tensors
  std::vector<std::string> errors;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

NemotronBindReport nemotron_validate_binding(
    const NemotronHConfig& cfg, const std::unordered_map<std::string, NemotronTensorDesc>& present,
    size_t max_errors = 32);

}  // namespace dgpp
