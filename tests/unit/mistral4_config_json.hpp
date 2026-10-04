// mistralai/Mistral-Small-4-119B-2603-NVFP4 @ 45331841: params.json verbatim
// (the checkpoint has no config.json), and the helpers the Mistral-Small-4
// config and binding tests share.
#pragma once
#include <stdexcept>
#include <string>

#include "loaders/minijson.hpp"
#include "models/mistral4/config.hpp"

namespace mistral4_test {

inline const char* const kParams = R"PARAMS({
  "dim": 4096,
  "head_dim": 128,
  "hidden_dim": 12288,
  "kv_lora_rank": 256,
  "llama_4_scaling": {
    "beta": 0.1,
    "original_max_position_embeddings": 8192
  },
  "max_position_embeddings": 1048576,
  "moe": {
    "expert_hidden_dim": 2048,
    "expert_model_parallel": 1,
    "expert_parallel": 1,
    "first_k_dense_replace": 0,
    "num_expert_groups": 1,
    "num_expert_groups_per_tok": 1,
    "num_experts": 128,
    "num_experts_per_tok": 4,
    "num_shared_experts": 1,
    "route_every_n": 1,
    "routed_scale": 1.0
  },
  "n_heads": 32,
  "n_kv_heads": 32,
  "n_layers": 36,
  "norm_eps": 1e-06,
  "q_lora_rank": 1024,
  "qk_nope_head_dim": 64,
  "qk_rope_head_dim": 64,
  "quantization_config": {
    "config_groups": {
      "NVFP4A16": {
        "format": "nvfp4-pack-quantized",
        "input_activations": {
          "actorder": null,
          "block_structure": null,
          "dynamic": "local",
          "group_size": 16,
          "num_bits": 4,
          "observer": "static_minmax",
          "observer_kwargs": {},
          "strategy": "tensor_group",
          "symmetric": true,
          "type": "float"
        },
        "output_activations": null,
        "targets": [
          "Linear"
        ],
        "weights": {
          "actorder": null,
          "block_structure": null,
          "dynamic": false,
          "group_size": 16,
          "num_bits": 4,
          "observer": "static_minmax",
          "observer_kwargs": {},
          "scale_dtype": "torch.float8_e4m3fn",
          "strategy": "tensor_group",
          "symmetric": true,
          "type": "float",
          "zp_dtype": null
        }
      }
    },
    "format": "nvfp4-pack-quantized",
    "global_compression_ratio": null,
    "ignore": [
      "model.embed_tokens",
      "re:patch_merger.*",
      "re:vision_encoder.*",
      "re:vision_language_adapter.*",
      "re:.*kv_a_proj_with_mqa$",
      "re:.*q_a_proj$",
      "re:.*gate$",
      "re:.*self_attn.*",
      "re:.*attention.*",
      "lm_head"
    ],
    "kv_cache_scheme": null,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "sparsity_config": {},
    "transform_config": {},
    "version": "0.13.0"
  },
  "rope_theta": 10000.0,
  "tied_embeddings": false,
  "v_head_dim": 128,
  "vision_encoder": {
    "adapter_bias": false,
    "add_pre_mm_projector_layer_norm": true,
    "hidden_size": 1024,
    "image_break_token_id": 12,
    "image_end_token_id": 13,
    "image_size": 1540,
    "image_token_id": 10,
    "intermediate_size": 4096,
    "max_image_size": 1540,
    "mm_projector_id": "patch_merge",
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "rope_theta": 10000.0,
    "spatial_merge_size": 2
  },
  "vocab_size": 131072,
  "yarn": {
    "alpha": 1,
    "apply_scale": false,
    "beta": 32,
    "factor": 128,
    "original_max_position_embeddings": 8192
  }
})PARAMS";

// kParams with the first occurrence of `from` replaced by `to` (an anchor
// that is missing is a test bug, not a pass).
inline std::string patched(const std::string& from, const std::string& to) {
  std::string s = kParams;
  if (from.empty()) return s;
  const size_t at = s.find(from);
  if (at == std::string::npos) throw std::runtime_error("patch anchor missing: " + from);
  s.replace(at, from.size(), to);
  return s;
}

inline dgpp::Mistral4TextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Mistral4TextConfig::parse(t.root);
}

inline dgpp::Mistral4TextConfig release() { return parse(kParams); }

// The refusal's message, or "" when the text parses.
inline std::string refusal(const std::string& text) {
  try {
    (void)parse(text);
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}

}  // namespace mistral4_test
