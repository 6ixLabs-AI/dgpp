#pragma once
// The Nemotron-3 config files as the unit tests build them (transcribed
// 2026-10-04): Nano's config.json and hf_quant_config.json verbatim
// (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 @ 6efb4a2a), and Super's
// config.json (nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 @ 4f0cf9da)
// with its 5.7 MB quantization_config rebuilt from the per-class layer
// lists below — the release names 41,100 quantized modules one by one; the
// lists regenerate that map exactly. Shared by the config and binding tests;
// `patched(text, from, to)` rewrites one anchor.
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/nemotron/config.hpp"

namespace nemotron_test {

inline const char* kNanoConfig = R"JSON({
  "architectures": ["NemotronHForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
    "AutoModel": "modeling_nemotron_h.NemotronHForCausalLM",
    "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM"
  },
  "bos_token_id": 1,
  "chunk_size": 128,
  "conv_kernel": 4,
  "eos_token_id": 2,
  "expand": 2,
  "head_dim": 128,
  "hidden_dropout": 0.0,
  "hidden_size": 2688,
  "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
  "initializer_range": 0.02,
  "intermediate_size": 1856,
  "layer_norm_epsilon": 1e-05,
  "mamba_head_dim": 64,
  "mamba_hidden_act": "silu",
  "mamba_num_heads": 64,
  "mamba_proj_bias": false,
  "mamba_ssm_cache_dtype": "float32",
  "max_position_embeddings": 262144,
  "mlp_bias": false,
  "mlp_hidden_act": "relu2",
  "model_type": "nemotron_h",
  "moe_intermediate_size": 1856,
  "moe_shared_expert_intermediate_size": 3712,
  "n_group": 1,
  "n_groups": 8,
  "n_routed_experts": 128,
  "n_shared_experts": 1,
  "norm_eps": 1e-05,
  "norm_topk_prob": true,
  "num_attention_heads": 32,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 52,
  "num_key_value_heads": 2,
  "num_logits_to_keep": 1,
  "pad_token_id": 0,
  "partial_rotary_factor": 1.0,
  "rescale_prenorm_residual": true,
  "residual_in_fp32": false,
  "rope_theta": 10000,
  "routed_scaling_factor": 2.5,
  "sliding_window": null,
  "ssm_state_size": 128,
  "tie_word_embeddings": false,
  "time_step_floor": 0.0001,
  "time_step_max": 0.1,
  "time_step_min": 0.001,
  "topk_group": 1,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.53.2",
  "use_bias": false,
  "use_cache": true,
  "use_conv_bias": true,
  "use_mamba_kernels": true,
  "vocab_size": 131072
})JSON";

// hf_quant_config.json: the head, the six attention layers' projections,
// the Mamba layer in front of each attention layer, and every convolution.
inline const char* kNanoHfQuant = R"JSON({
    "producer": {"name": "modelopt", "version": "0.29.0"},
    "quantization": {
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": "FP8",
        "group_size": 16,
        "exclude_modules": [
            "lm_head",
            "backbone.layers.4.mixer.in_proj", "backbone.layers.4.mixer.out_proj",
            "backbone.layers.5.mixer.q_proj", "backbone.layers.5.mixer.k_proj",
            "backbone.layers.5.mixer.v_proj", "backbone.layers.5.mixer.o_proj",
            "backbone.layers.11.mixer.in_proj", "backbone.layers.11.mixer.out_proj",
            "backbone.layers.12.mixer.q_proj", "backbone.layers.12.mixer.k_proj",
            "backbone.layers.12.mixer.v_proj", "backbone.layers.12.mixer.o_proj",
            "backbone.layers.18.mixer.in_proj", "backbone.layers.18.mixer.out_proj",
            "backbone.layers.19.mixer.q_proj", "backbone.layers.19.mixer.k_proj",
            "backbone.layers.19.mixer.v_proj", "backbone.layers.19.mixer.o_proj",
            "backbone.layers.25.mixer.in_proj", "backbone.layers.25.mixer.out_proj",
            "backbone.layers.26.mixer.q_proj", "backbone.layers.26.mixer.k_proj",
            "backbone.layers.26.mixer.v_proj", "backbone.layers.26.mixer.o_proj",
            "backbone.layers.32.mixer.in_proj", "backbone.layers.32.mixer.out_proj",
            "backbone.layers.33.mixer.q_proj", "backbone.layers.33.mixer.k_proj",
            "backbone.layers.33.mixer.v_proj", "backbone.layers.33.mixer.o_proj",
            "backbone.layers.41.mixer.in_proj", "backbone.layers.41.mixer.out_proj",
            "backbone.layers.42.mixer.q_proj", "backbone.layers.42.mixer.k_proj",
            "backbone.layers.42.mixer.v_proj", "backbone.layers.42.mixer.o_proj",
            "backbone.layers.0.mixer.conv1d", "backbone.layers.2.mixer.conv1d",
            "backbone.layers.4.mixer.conv1d", "backbone.layers.7.mixer.conv1d",
            "backbone.layers.9.mixer.conv1d", "backbone.layers.11.mixer.conv1d",
            "backbone.layers.14.mixer.conv1d", "backbone.layers.16.mixer.conv1d",
            "backbone.layers.18.mixer.conv1d", "backbone.layers.21.mixer.conv1d",
            "backbone.layers.23.mixer.conv1d", "backbone.layers.25.mixer.conv1d",
            "backbone.layers.28.mixer.conv1d", "backbone.layers.30.mixer.conv1d",
            "backbone.layers.32.mixer.conv1d", "backbone.layers.35.mixer.conv1d",
            "backbone.layers.37.mixer.conv1d", "backbone.layers.39.mixer.conv1d",
            "backbone.layers.41.mixer.conv1d", "backbone.layers.44.mixer.conv1d",
            "backbone.layers.46.mixer.conv1d", "backbone.layers.48.mixer.conv1d",
            "backbone.layers.50.mixer.conv1d"
        ]
    }
})JSON";

// Super's config.json; QUANT stands where its quantization_config is.
inline const char* kSuperConfig = R"JSON({
 "architectures": ["NemotronHForCausalLM"],
 "attention_bias": false,
 "attention_dropout": 0.0,
 "auto_map": {
  "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
  "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM"
 },
 "bos_token_id": 1,
 "chunk_size": 128,
 "conv_kernel": 4,
 "dtype": "bfloat16",
 "eos_token_id": 2,
 "expand": 2,
 "head_dim": 128,
 "hidden_dropout": 0.0,
 "hidden_size": 4096,
 "hybrid_override_pattern": "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME",
 "initializer_range": 0.02,
 "intermediate_size": 2688,
 "layer_norm_epsilon": 1e-05,
 "mamba_head_dim": 64,
 "mamba_hidden_act": "silu",
 "mamba_num_heads": 128,
 "mamba_proj_bias": false,
 "mamba_ssm_cache_dtype": "float32",
 "max_position_embeddings": 262144,
 "mlp_bias": false,
 "mlp_hidden_act": "relu2",
 "model_type": "nemotron_h",
 "moe_intermediate_size": 2688,
 "moe_latent_size": 1024,
 "moe_shared_expert_intermediate_size": 5376,
 "moe_shared_expert_overlap": false,
 "mtp_hybrid_override_pattern": "*E",
 "n_group": 1,
 "n_groups": 8,
 "n_routed_experts": 512,
 "n_shared_experts": 1,
 "norm_eps": 1e-05,
 "norm_topk_prob": true,
 "num_attention_heads": 32,
 "num_experts_per_tok": 22,
 "num_hidden_layers": 88,
 "num_key_value_heads": 2,
 "num_logits_to_keep": 1,
 "num_nextn_predict_layers": 1,
 "pad_token_id": 0,
 "partial_rotary_factor": 1.0,
 "rescale_prenorm_residual": true,
 "residual_in_fp32": false,
 "rope_theta": 10000,
 "routed_scaling_factor": 5.0,
 "sliding_window": null,
 "ssm_state_size": 128,
 "tie_word_embeddings": false,
 "time_step_floor": 0.0001,
 "time_step_max": 0.1,
 "time_step_min": 0.001,
 "topk_group": 1,
 "transformers_version": "4.57.6",
 "use_bias": false,
 "use_cache": true,
 "use_conv_bias": true,
 "use_mamba_kernels": true,
 "vocab_size": 131072,
 "quantization_config": QUANT
})JSON";

// Super's quantized modules by class (config.json
// quantization_config.quantized_layers): the layers whose module of that
// name is per-tensor FP8. Every routed expert of the 40 MoE layers is NVFP4,
// and so is layer 1's shared down_proj — the one NVFP4 matrix outside the
// experts. Anything not listed is BF16 (the shared up_proj of layer 41 and
// the shared down_proj of layers 8 and 76 among them).
inline const std::vector<int> kSuperInProjFp8 = {0,  2,  4,  6,  9,  11, 13, 15, 18, 20,
                                                 27, 29, 31, 40, 49, 51, 53, 55, 57, 60,
                                                 62, 66, 68, 71, 73, 82, 84, 86};
inline const std::vector<int> kSuperOutProjFp8 = {0,  2,  4,  9,  11, 13, 18, 38, 40, 42,
                                                  44, 49, 51, 53, 55, 57, 60, 62, 64, 66,
                                                  68, 71, 73, 75, 77, 80, 82, 84, 86};
inline const std::vector<int> kSuperSharedUpFp8 = {
    1,  3,  5,  8,  10, 12, 14, 17, 19, 21, 23, 26, 28, 30, 32, 34, 37, 39, 43, 45,
    48, 50, 52, 54, 56, 59, 61, 63, 65, 67, 70, 72, 74, 76, 79, 81, 83, 85, 87};
inline const std::vector<int> kSuperSharedDownFp8 = {
    3,  5,  10, 12, 14, 17, 19, 21, 23, 26, 28, 30, 32, 34, 37, 39, 41, 43, 45,
    48, 50, 52, 54, 56, 59, 61, 63, 65, 67, 70, 72, 74, 79, 81, 83, 85, 87};
inline const std::vector<int> kSuperFc1Fp8 = {1, 3, 5};
inline const std::vector<int> kSuperFc2Fp8 = {3};
inline const std::vector<int> kSuperOProjFp8 = {69, 78};
inline const int kSuperSharedDownFp4Layer = 1;
inline const std::vector<int> kSuperMoeLayers = {
    1,  3,  5,  8,  10, 12, 14, 17, 19, 21, 23, 26, 28, 30, 32, 34, 37, 39, 41, 43,
    45, 48, 50, 52, 54, 56, 59, 61, 63, 65, 67, 70, 72, 74, 76, 79, 81, 83, 85, 87};

struct SuperModule {
  std::string name;
  bool fp4 = false;
};

// The 41,100 quantized modules: 139 FP8, 40,961 NVFP4.
inline std::vector<SuperModule> super_quantized_modules() {
  std::vector<SuperModule> out;
  const auto fp8 = [&](const std::vector<int>& layers, const char* leaf) {
    for (const int l : layers)
      out.push_back({"backbone.layers." + std::to_string(l) + ".mixer." + leaf, false});
  };
  fp8(kSuperInProjFp8, "in_proj");
  fp8(kSuperOutProjFp8, "out_proj");
  fp8(kSuperSharedUpFp8, "shared_experts.up_proj");
  fp8(kSuperSharedDownFp8, "shared_experts.down_proj");
  fp8(kSuperFc1Fp8, "fc1_latent_proj");
  fp8(kSuperFc2Fp8, "fc2_latent_proj");
  fp8(kSuperOProjFp8, "o_proj");
  out.push_back({"backbone.layers." + std::to_string(kSuperSharedDownFp4Layer) +
                     ".mixer.shared_experts.down_proj",
                 true});
  for (const int l : kSuperMoeLayers)
    for (int e = 0; e < 512; ++e)
      for (const char* leaf : {"up_proj", "down_proj"})
        out.push_back({"backbone.layers." + std::to_string(l) + ".mixer.experts." +
                           std::to_string(e) + "." + leaf,
                       true});
  return out;
}

// {"name": {"quant_algo": ...}, ...} — the body of a quantized_layers object.
inline std::string super_quantized_layers_json() {
  std::string s = "{";
  bool first = true;
  for (const SuperModule& m : super_quantized_modules()) {
    if (!first) s += ",";
    first = false;
    s += "\"" + m.name + "\":" +
         (m.fp4 ? "{\"quant_algo\":\"NVFP4\",\"group_size\":16}" : "{\"quant_algo\":\"FP8\"}");
  }
  return s + "}";
}

// config.json's quantization_config as the release writes it: the two
// config_groups repeat the layer map as target lists.
inline std::string super_quantization_config() {
  std::string t8, t4;
  for (const SuperModule& m : super_quantized_modules()) {
    std::string& t = m.fp4 ? t4 : t8;
    if (!t.empty()) t += ",";
    t += "\"" + m.name + "\"";
  }
  return std::string("{\"config_groups\":{") +
         "\"group_0\":{\"input_activations\":{\"dynamic\":false,\"num_bits\":8,\"type\":\"float\"},"
         "\"weights\":{\"dynamic\":false,\"num_bits\":8,\"type\":\"float\"},\"targets\":[" +
         t8 + "]}," +
         "\"group_1\":{\"input_activations\":{\"dynamic\":false,\"num_bits\":4,\"type\":\"float\","
         "\"group_size\":16},"
         "\"weights\":{\"dynamic\":false,\"num_bits\":4,\"type\":\"float\",\"group_size\":16},"
         "\"targets\":[" +
         t4 + "]}}," + "\"quantized_layers\":" + super_quantized_layers_json() + "," +
         "\"ignore\":[],\"quant_algo\":\"MIXED_PRECISION\","
         "\"kv_cache_scheme\":{\"dynamic\":false,\"num_bits\":8,\"type\":\"float\"},"
         "\"producer\":{\"name\":\"modelopt\",\"version\":\"0.43.0.dev63+g449e700f9\"},\"quant_"
         "method\":\"modelopt\"}";
}

// Super's hf_quant_config.json: the same layer map a second time.
inline std::string super_hf_quant() {
  return "{\"producer\":{\"name\":\"modelopt\",\"version\":\"0.43.0.dev63+g449e700f9\"},"
         "\"quantization\":{\"quant_algo\":\"MIXED_PRECISION\",\"kv_cache_quant_algo\":\"FP8\","
         "\"quantized_layers\":" +
         super_quantized_layers_json() + "}}";
}

// `text` with its one occurrence of `from` replaced by `to`.
inline std::string patched(std::string text, const std::string& from, const std::string& to) {
  const size_t at = text.find(from);
  if (at == std::string::npos) throw std::logic_error("test config: anchor not found: " + from);
  if (text.find(from, at + 1) != std::string::npos)
    throw std::logic_error("test config: anchor not unique: " + from);
  text.replace(at, from.size(), to);
  return text;
}

inline std::string super_config_json() {
  return patched(kSuperConfig, "QUANT", super_quantization_config());
}

// Parses a config with an optional hf_quant_config.json text (empty: none).
inline dgpp::NemotronHConfig parse(const std::string& config, const std::string& hf_quant) {
  const auto c = dgpp::minijson::parse(config);
  if (hf_quant.empty()) return dgpp::NemotronHConfig::parse(c.root, nullptr);
  const auto h = dgpp::minijson::parse(hf_quant);
  return dgpp::NemotronHConfig::parse(c.root, &h.root);
}
inline dgpp::NemotronHConfig nano() {
  return parse(kNanoConfig, kNanoHfQuant);
}
inline dgpp::NemotronHConfig super() {
  return parse(super_config_json(), super_hf_quant());
}

}  // namespace nemotron_test
