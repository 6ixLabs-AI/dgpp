#pragma once
// Expected-tensor table for the Qwen3.8-27B text model (Qwen3_5ForConditionalGeneration):
// every tensor the checkpoint must contain for the text stack — names,
// dtypes, exact shapes — derived from the parsed config, not observed from
// one file. The table drives the offline validator and (with the loader
// slice) the resident loader.
//
// Naming is checkpoint truth (Qwen/Qwen3.8-27B-FP8): main layers under
// `model.language_model.layers.L.`, the draft layer under `mtp.layers.0.`
// with the head's own tensors under `mtp.`, the globals
// `model.language_model.embed_tokens.weight`, `lm_head.weight` and
// `model.language_model.norm.weight`. Vision (`model.visual.*`) is counted
// and skipped: text-only scope.
//
// Scale contract (the FP8 release): every e4m3 matrix X.weight carries a
// BF16 partner X.weight_scale_inv of shape [ceil(N/128), ceil(K/128)] —
// 128x128 dequant blocks, MULTIPLY on dequant (the loader widens the
// scales to F32 at load). Everything else is BF16.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "models/qwen/binding.hpp"
#include "models/qwen/config35.hpp"

namespace dgpp {

// The Qwen3Next dialect (Qwen3-Next-80B-A3B, nvidia/...-NVFP4 @ 8fb2682f)
// names its tensors flat — `model.layers.L.`, `model.embed_tokens.weight`,
// `model.norm.weight` — fuses the GDN projections (`in_proj_qkvz`,
// `in_proj_ba`, interleaved per key head), carries the routed MoE under
// `mlp.` (`gate`, `shared_expert_gate`, `shared_expert.*`, `experts.E.*`)
// and ships the modelopt NVFP4 set (`weight` U8 [N, K/2], `weight_scale`
// e4m3 [N, K/16], F32 `weight_scale_2`, F32 `input_scale`) for the routed
// experts, the shared expert, the attention o_proj and the GDN out_proj;
// every other matrix is BF16, the draft layer (`mtp.*`) entirely.
//
// The same dialect in its compressed-tensors container (Qwen3-Coder-Next,
// RedHatAI/...-NVFP4 @ 27a8f16f, 296,151 tensors): the NVFP4 set is
// `weight_packed` U8 [N, K/2], `weight_scale` e4m3 [N, K/16], F32 [1]
// `weight_global_scale` and F32 [1] `input_global_scale`, on the routed
// experts, the shared expert and the attention q/k/v/o; the whole GDN
// (out_proj included), the router and the head are BF16; there are no
// K/V-cache scales and no draft layer.

// "model.language_model." (Qwen3.5) or "model." (the flat Qwen3Next names):
// the prefix of `layers.L.`, `embed_tokens.weight` and `norm.weight`.
std::string qwen35_model_prefix(const Qwen35TextConfig& cfg);

// The checkpoint name prefix of a layer ("model.language_model.layers.L."
// or "mtp.layers.0." for the draft layer).
std::string qwen35_layer_prefix(const Qwen35TextConfig& cfg, int layer);

// Full text-model table (main layers + the draft layer + globals).
std::vector<QwenExpectedTensor> qwen35_expected_text_tensors(const Qwen35TextConfig& cfg);
// One layer's entries: `layer` in [0, num_hidden_layers) or mtp_layer().
std::vector<QwenExpectedTensor> qwen35_expected_layer_tensors(const Qwen35TextConfig& cfg,
                                                             int layer);
// The globals: embed, lm_head, the final norm, and the draft head's own
// tensors when the draft layer exists.
std::vector<QwenExpectedTensor> qwen35_expected_global_tensors(const Qwen35TextConfig& cfg);

QwenBindReport qwen35_validate_text_binding(
    const Qwen35TextConfig& cfg,
    const std::unordered_map<std::string, QwenTensorDesc>& present,
    size_t max_errors = 32);

// Tensor-parallel geometry acceptance: throws std::invalid_argument naming
// the dim that does not divide. world must divide the GDN key and value
// heads, the attention query heads, and the dense intermediate size; either
// world divides the kv heads or the kv heads divide world. FP8 scale-grid
// slicing (multiples of 128) is the loader's contract, not this check's.
void qwen35_tp_validate_geometry(const Qwen35TextConfig& cfg, int rank, int world);

}  // namespace dgpp
