// lukealonso/MiniMax-M2.7-NVFP4 @ db821d7a: config.json transcribed — every key
// and value of the file; its two generated lists (attn_type_list: 62 ones;
// the ignore list: lm_head, then each layer's router and attention, the
// layer numbers sorted as strings) are rebuilt below — and the helpers the
// MiniMax-M2 config and binding tests share.
#pragma once
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/minimax/config.hpp"

namespace minimax_test {

inline const char* const kConfigHead = R"CONFIG({
  "vocab_size": 200064,
  "max_position_embeddings": 196608,
  "hidden_size": 3072,
  "intermediate_size": 1536,
  "num_hidden_layers": 62,
  "num_attention_heads": 48,
  "sliding_window": null,
  "num_key_value_heads": 8,
  "hidden_act": "silu",
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-06,
  "use_cache": true,
  "rope_theta": 5000000,
  "attention_dropout": 0.0,
  "head_dim": 128,
  "num_experts_per_tok": 8,
  "num_local_experts": 256,
  "output_router_logits": false,
  "router_aux_loss_coef": 0.001,
  "router_jitter_noise": 0.0,
  "use_qk_norm": true,
  "rotary_dim": 64,
  "partial_rotary_factor": 0.5,
  "transformers_version": "5.5.3",
  "architectures": [
    "MiniMaxM2ForCausalLM"
  ],
  "output_hidden_states": false,
  "return_dict": true,
  "dtype": "bfloat16",
  "chunk_size_feed_forward": 0,
  "is_encoder_decoder": false,
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1"
  },
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1
  },
  "problem_type": null,
  "_name_or_path": "/data/models/MiniMax-M2.7-BF16",
  "pad_token_id": null,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "tie_word_embeddings": false,
  "attn_type_list": [ATTN],
  "auto_map": {
    "AutoConfig": "configuration_minimax_m2.MiniMaxM2Config",
    "AutoModelForCausalLM": "modeling_minimax_m2.MiniMaxM2ForCausalLM"
  },
  "model_type": "minimax_m2",
  "mtp_transformer_layers": 1,
  "num_mtp_modules": 3,
  "qk_norm_type": "per_layer",
  "scoring_func": "sigmoid",
  "shared_intermediate_size": 0,
  "use_mtp": true,
  "use_routing_bias": true,
  "rope_parameters": {
    "rope_type": "default",
    "rope_theta": 5000000,
    "partial_rotary_factor": 0.5
  },
  "output_attentions": false,
  "quantization_config": {
    "config_groups": {
      "group_0": {
        "input_activations": {
          "dynamic": false,
          "num_bits": 4,
          "type": "float",
          "group_size": 16
        },
        "weights": {
          "dynamic": false,
          "num_bits": 4,
          "type": "float",
          "group_size": 16
        },
        "targets": [
          "Linear"
        ]
      }
    },
    "ignore": [IGNORE],
    "quant_algo": "NVFP4",
    "producer": {
      "name": "modelopt",
      "version": "0.39.0.dev290+gf9d9a71de.d20260407"
    },
    "quant_method": "modelopt"
  }
})CONFIG";

// The release's ignore list for `layers` layers, in the file's order.
inline std::string ignore_json(int layers = 62) {
  std::vector<std::string> ids;
  for (int l = 0; l < layers; ++l) ids.push_back(std::to_string(l));
  std::sort(ids.begin(), ids.end());
  std::string s = "\"lm_head\"";
  for (const std::string& l : ids)
    s += ", \"model.layers." + l + ".block_sparse_moe.gate\", \"model.layers." + l + ".self_attn*\"";
  return s;
}

inline std::string attn_types_json(int layers = 62) {
  std::string s;
  for (int l = 0; l < layers; ++l) s += l ? ", 1" : "1";
  return s;
}

// The config with the first occurrence of `from` replaced by `to` (an anchor
// that is missing is a test bug, not a pass).
inline std::string config_json(const std::string& from = "", const std::string& to = "",
                               const std::string& ignore = ignore_json(),
                               const std::string& attn_types = attn_types_json()) {
  std::string s = kConfigHead;
  s.replace(s.find("[IGNORE]"), 8, "[" + ignore + "]");
  s.replace(s.find("[ATTN]"), 6, "[" + attn_types + "]");
  if (!from.empty()) {
    const size_t at = s.find(from);
    if (at == std::string::npos) throw std::runtime_error("patch anchor missing: " + from);
    s.replace(at, from.size(), to);
  }
  return s;
}

inline dgpp::MinimaxTextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::MinimaxTextConfig::parse(t.root);
}

inline dgpp::MinimaxTextConfig release() { return parse(config_json()); }

// The refusal's message, or "" when the text parses.
inline std::string refusal(const std::string& text) {
  try {
    (void)parse(text);
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}

}  // namespace minimax_test
