#pragma once
// The Gemma 4 config.json files as the unit tests build them:
// nvidia/Gemma-4-31B-IT-NVFP4 @ 4135a98a and
// bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16 @ main (transcribed
// 2026-10-04; the vision_config the parser never reads is cut to its
// model_type, and the two long lists — layer_types and the quantization
// ignore list — are generated the way the releases write them). Shared by
// the config and binding tests; `config_json(from, to)` / `config_json_26b
// (from, to)` patch one anchor.
#include <stdexcept>
#include <string>

namespace gemma4_test {

inline const char* kConfig31b = R"JSON({
    "architectures": ["Gemma4ForConditionalGeneration"],
    "audio_config": null,
    "audio_token_id": 258881,
    "boa_token_id": 256000,
    "boi_token_id": 255999,
    "dtype": "bfloat16",
    "eoa_token_id": 258883,
    "eoa_token_index": 258883,
    "eoi_token_id": 258882,
    "eos_token_id": [1, 106],
    "image_token_id": 258880,
    "initializer_range": 0.02,
    "model_type": "gemma4",
    "text_config": {
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attention_k_eq_v": true,
        "bos_token_id": 2,
        "dtype": "bfloat16",
        "enable_moe_block": false,
        "eos_token_id": 1,
        "expert_intermediate_size": null,
        "final_logit_softcapping": 30.0,
        "global_head_dim": 512,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 5376,
        "hidden_size_per_layer_input": 0,
        "initializer_range": 0.02,
        "intermediate_size": 21504,
        "layer_types": [LAYER_TYPES],
        "max_position_embeddings": 262144,
        "model_type": "gemma4_text",
        "num_attention_heads": 32,
        "num_experts": null,
        "num_global_key_value_heads": 4,
        "num_hidden_layers": 60,
        "num_key_value_heads": 16,
        "num_kv_shared_layers": 0,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional"
            },
            "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default"
            }
        },
        "sliding_window": 1024,
        "tie_word_embeddings": true,
        "top_k_experts": null,
        "use_bidirectional_attention": "vision",
        "use_cache": true,
        "use_double_wide_mlp": false,
        "vocab_size": 262144,
        "vocab_size_per_layer_input": 262144
    },
    "tie_word_embeddings": true,
    "transformers_version": "5.5.0.dev0",
    "video_token_id": 258884,
    "vision_config": {"model_type": "gemma4_vision"},
    "vision_soft_tokens_per_image": 280,
    "quantization_config": {
        "config_groups": {
            "group_0": {
                "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                "targets": ["Linear"]
            }
        },
        "ignore": [IGNORE],
        "quant_algo": "NVFP4",
        "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
        "producer": {"name": "modelopt", "version": "0.37.0"},
        "quant_method": "modelopt"
    }
})JSON";

// The 26B-A4B: thirty layers of width 2816, the MoE block on every one (128
// experts, top 8, 704 wide) beside a 2112-wide dense MLP; modelopt 0.43
// weight-only NVFP4 with the routers ignored.
inline const char* kConfig26b = R"JSON({
    "architectures": ["Gemma4ForConditionalGeneration"],
    "audio_config": null,
    "audio_token_id": 258881,
    "boa_token_id": 256000,
    "boi_token_id": 255999,
    "dtype": "bfloat16",
    "eoa_token_id": 258883,
    "eoa_token_index": 258883,
    "eoi_token_id": 258882,
    "eos_token_id": [1, 106],
    "image_token_id": 258880,
    "initializer_range": 0.02,
    "model_type": "gemma4",
    "text_config": {
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attention_k_eq_v": true,
        "bos_token_id": 2,
        "dtype": "bfloat16",
        "enable_moe_block": true,
        "eos_token_id": 1,
        "final_logit_softcapping": 30.0,
        "global_head_dim": 512,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 2816,
        "hidden_size_per_layer_input": 0,
        "initializer_range": 0.02,
        "intermediate_size": 2112,
        "layer_types": [LAYER_TYPES],
        "max_position_embeddings": 262144,
        "model_type": "gemma4_text",
        "moe_intermediate_size": 704,
        "num_attention_heads": 16,
        "num_experts": 128,
        "num_global_key_value_heads": 2,
        "num_hidden_layers": 30,
        "num_key_value_heads": 8,
        "num_kv_shared_layers": 0,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional"
            },
            "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default"
            }
        },
        "sliding_window": 1024,
        "tie_word_embeddings": true,
        "top_k_experts": 8,
        "use_bidirectional_attention": "vision",
        "use_cache": true,
        "use_double_wide_mlp": false,
        "vocab_size": 262144,
        "vocab_size_per_layer_input": 262144
    },
    "tie_word_embeddings": true,
    "transformers_version": "5.5.0",
    "video_token_id": 258884,
    "vision_config": {"model_type": "gemma4_vision"},
    "vision_soft_tokens_per_image": 280,
    "quantization_config": {
        "config_groups": {
            "group_0": {
                "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                "targets": ["Linear"]
            }
        },
        "ignore": [IGNORE],
        "quant_algo": "NVFP4",
        "producer": {"name": "modelopt", "version": "0.43.0rc2.dev57+g87ea8babe"},
        "quant_method": "modelopt"
    }
})JSON";

// "sliding_attention" x 5, "full_attention", repeated.
inline std::string layer_types_json(int layers = 60) {
  std::string s;
  for (int l = 0; l < layers; ++l) {
    if (l) s += ", ";
    s += (l + 1) % 6 != 0 ? "\"sliding_attention\"" : "\"full_attention\"";
  }
  return s;
}

// The 31B release's ignore list: the head, the encoders, every layer's
// attention.
inline std::string ignore_json_31b(int layers = 60) {
  std::string s = "\"lm_head\", \"model.embed_vision*\"";
  for (int l = 0; l < layers; ++l) s += ", \"model.language_model.layers." + std::to_string(l) + ".self_attn*\"";
  s += ", \"model.vision_tower*\"";
  return s;
}

// The 26B-A4B release's ignore list: the head, the encoders, every layer's
// router.
inline std::string ignore_json_26b(int layers = 30) {
  std::string s = "\"lm_head\", \"model.embed_vision*\"";
  for (int l = 0; l < layers; ++l) s += ", \"model.language_model.layers." + std::to_string(l) + ".router*\"";
  s += ", \"model.vision_tower*\"";
  return s;
}

inline std::string patched(std::string s, const std::string& from, const std::string& to) {
  if (from.empty()) return s;
  const size_t at = s.find(from);
  if (at == std::string::npos) throw std::runtime_error("patch anchor missing: " + from);
  s.replace(at, from.size(), to);
  return s;
}

inline std::string config_json(const std::string& patch_from = "", const std::string& patch_to = "",
                               const std::string& ignore = ignore_json_31b(),
                               const std::string& layer_types = layer_types_json()) {
  std::string s = kConfig31b;
  s = patched(s, "[IGNORE]", "[" + ignore + "]");
  s = patched(s, "[LAYER_TYPES]", "[" + layer_types + "]");
  return patched(s, patch_from, patch_to);
}

inline std::string config_json_26b(const std::string& patch_from = "", const std::string& patch_to = "",
                                   const std::string& ignore = ignore_json_26b()) {
  std::string s = kConfig26b;
  s = patched(s, "[IGNORE]", "[" + ignore + "]");
  s = patched(s, "[LAYER_TYPES]", "[" + layer_types_json(30) + "]");
  return patched(s, patch_from, patch_to);
}

}  // namespace gemma4_test
