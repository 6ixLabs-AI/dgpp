#pragma once
// Synthetic mini-checkpoints for the qwen3_5 stack's later ports (2026-10-04:
// Qwen3-Coder-Next's compressed-tensors NVFP4 container on the Qwen3Next
// dialect; Qwen3.6-35B-A3B's two containers on the Qwen3.5 dialect with the
// routed MoE; Qwen3.5-0.8B's unquantized dense release with tied
// embeddings; Qwen3.5-122B-A10B's two NVFP4 containers): a small config written as config.json and its binding table
// written as one safetensors shard, so fixture and table cannot disagree.
// Values are deterministic per tensor name (glm_rng's scheme); the fixture
// keeps every tensor's bytes so a test can state what the loader must hold.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "glm_rng.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"

namespace qwen35portsfx {

namespace fs = std::filesystem;
using dgpp::Qwen35TextConfig;
using dgpp::QwenExpectedTensor;
using dgpp::QwenTensorRole;

// The kernels' pinned widths (128-wide GDN heads, 256-wide attention heads)
// at counts that slice at worlds 1, 2 and 4, as the 80B's fixture
// (qwen3next_loader_fixture.hpp): 4 key heads x 8 value heads, 4 query heads
// on 2 kv heads, hidden 256, three GDN layers then one attention layer, 4
// experts of intermediate 64 and a shared expert of 512.
//
// Qwen3-Coder-Next: the compressed-tensors recipe, no draft layer.
inline const char* coder_next_config_json() {
  return R"json({
  "architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next",
  "bos_token_id": 1, "eos_token_id": 1, "decoder_sparse_step": 1, "full_attention_interval": 4,
  "head_dim": 256, "hidden_act": "silu", "hidden_size": 256, "intermediate_size": 512,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
  "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 4,
  "linear_num_value_heads": 8, "linear_value_head_dim": 128, "max_position_embeddings": 4096,
  "mlp_only_layers": [], "moe_intermediate_size": 64, "norm_topk_prob": true,
  "num_attention_heads": 4, "num_experts": 4, "num_experts_per_tok": 2, "num_hidden_layers": 4,
  "num_key_value_heads": 2, "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06,
  "rope_scaling": null, "rope_theta": 5000000, "shared_expert_intermediate_size": 512,
  "tie_word_embeddings": false, "vocab_size": 64,
  "quantization_config": {"config_groups": {"group_0": {"format": "nvfp4-pack-quantized",
    "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16,
                "strategy": "tensor_group", "symmetric": true},
    "targets": ["Linear"]}}, "format": "nvfp4-pack-quantized", "kv_cache_scheme": null,
    "quant_method": "compressed-tensors", "quantization_status": "compressed",
    "sparsity_config": {}, "transform_config": {}}
})json";
}

// Qwen3.6-35B-A3B: the nested config with the routed MoE and a draft layer,
// around a quantization_config — the modelopt NVFP4 mixed recipe or the FP8
// release's.
inline std::string qwen36_config_json(const char* quantization_config) {
  return std::string(R"json({
  "architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
  "text_config": {
    "model_type": "qwen3_5_moe_text", "attention_bias": false, "attn_output_gate": true,
    "bos_token_id": 1, "eos_token_id": 1, "full_attention_interval": 4, "head_dim": 256,
    "hidden_act": "silu", "hidden_size": 256,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 4,
    "linear_num_value_heads": 8, "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 4096, "moe_intermediate_size": 64, "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false, "num_attention_heads": 4, "num_experts": 4,
    "num_experts_per_tok": 2, "num_hidden_layers": 4, "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
    "shared_expert_intermediate_size": 512, "tie_word_embeddings": false, "vocab_size": 64},
  "vision_config": {"depth": 2},
  "quantization_config": )json") + quantization_config + "\n}";
}
inline const char* kQuantModeloptMixed = R"json({
    "config_groups": {
      "group_0": {"input_activations": {"dynamic": false, "num_bits": 8, "type": "float"},
                  "weights": {"dynamic": false, "num_bits": 8, "type": "float"}},
      "group_1": {"input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                  "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16}}},
    "ignore": ["mtp.layers.0*", "mtp*"], "quant_algo": "MIXED_PRECISION",
    "producer": {"name": "modelopt", "version": "0.37.0"}, "quant_method": "modelopt"})json";
inline const char* kQuantFp8Block = R"json({
    "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
    "weight_block_size": [128, 128]})json";

// Qwen3.5-122B-A10B's two NVFP4 containers on the same tiny shape, each with
// the ignore list its recipe writes (the config parser holds it to them).
// modelopt `NVFP4`: the routed experts alone quantized.
inline const char* kQuantModeloptExperts = R"json({
    "config_groups": {"group_0": {
      "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
      "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
      "targets": ["Linear"]}},
    "ignore": ["lm_head",
      "model.language_model.layers.0.linear_attn*", "model.language_model.layers.0.mlp.shared_expert*",
      "model.language_model.layers.0.mlp.shared_expert_gate",
      "model.language_model.layers.1.linear_attn*", "model.language_model.layers.1.mlp.shared_expert*",
      "model.language_model.layers.1.mlp.shared_expert_gate",
      "model.language_model.layers.2.linear_attn*", "model.language_model.layers.2.mlp.shared_expert*",
      "model.language_model.layers.2.mlp.shared_expert_gate",
      "model.language_model.layers.3.mlp.shared_expert*",
      "model.language_model.layers.3.mlp.shared_expert_gate", "model.language_model.layers.3.self_attn*",
      "model.visual*", "mtp.layers.0*", "mtp*"],
    "quant_algo": "NVFP4", "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
    "producer": {"name": "modelopt", "version": "0.0.1"}, "quant_method": "modelopt"})json";
// compressed-tensors `nvfp4-pack-quantized`: the attention, the shared
// expert and the routed experts quantized; the GDN and the head ignored.
inline const char* kQuantPackedMoe = R"json({
    "config_groups": {"group_0": {"format": "nvfp4-pack-quantized",
      "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16,
                  "strategy": "tensor_group", "symmetric": true},
      "targets": ["Linear"]}},
    "format": "nvfp4-pack-quantized",
    "ignore": ["lm_head",
      "model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.linear_attn.in_proj_z",
      "model.language_model.layers.0.linear_attn.out_proj", "model.language_model.layers.0.mlp.gate",
      "model.language_model.layers.1.linear_attn.in_proj_qkv", "model.language_model.layers.1.linear_attn.in_proj_z",
      "model.language_model.layers.1.linear_attn.out_proj", "model.language_model.layers.1.mlp.gate",
      "model.language_model.layers.2.linear_attn.in_proj_qkv", "model.language_model.layers.2.linear_attn.in_proj_z",
      "model.language_model.layers.2.linear_attn.out_proj", "model.language_model.layers.2.mlp.gate",
      "model.language_model.layers.3.mlp.gate", "mtp.fc", "re:mtp\\.layers\\.\\d+\\."],
    "kv_cache_scheme": null, "quant_method": "compressed-tensors", "quantization_status": "compressed",
    "sparsity_config": {}, "transform_config": {}})json";

// Qwen3.5-0.8B: the dense Qwen3.5 dialect, no quantization_config, tied
// embeddings, one value head a key head (the release's 16 x 16), a dense
// intermediate of 512 (128 a rank at world 4: the FP8 form's block grid).
inline const char* qwen35_bf16_config_json() {
  return R"json({
  "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
  "text_config": {
    "model_type": "qwen3_5_text", "attention_bias": false, "attn_output_gate": true,
    "eos_token_id": 1, "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu",
    "hidden_size": 256, "intermediate_size": 512,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 4,
    "linear_num_value_heads": 4, "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 4096, "mlp_only_layers": [], "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false, "num_attention_heads": 4, "num_hidden_layers": 4,
    "num_key_value_heads": 2, "rms_norm_eps": 1e-06,
    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
    "tie_word_embeddings": true, "vocab_size": 64},
  "tie_word_embeddings": true, "vision_config": {"depth": 2}
})json";
}

struct Fixture {
  Qwen35TextConfig cfg;
  std::string dir;
  std::vector<QwenExpectedTensor> table;
  std::unordered_map<std::string, std::vector<uint8_t>> src;  // every tensor's bytes, by name

  const std::vector<uint8_t>& bytes(const std::string& name) const {
    const auto it = src.find(name);
    if (it == src.end()) throw std::runtime_error("fixture has no tensor " + name);
    return it->second;
  }
  // A BF16 tensor's bits / an F32 scalar.
  std::vector<uint16_t> bf16(const std::string& name) const {
    const std::vector<uint8_t>& b = bytes(name);
    std::vector<uint16_t> out(b.size() / 2);
    std::memcpy(out.data(), b.data(), out.size() * 2);
    return out;
  }
  float f32(const std::string& name) const {
    float v;
    std::memcpy(&v, bytes(name).data(), 4);
    return v;
  }
};

inline std::vector<uint8_t> tensor_bytes(const QwenExpectedTensor& e, const Qwen35TextConfig& cfg) {
  std::vector<uint8_t> out(e.nbytes());
  glmrng::Rng rng(glmrng::seed_for(e.name));
  const size_t n = e.numel();
  for (size_t i = 0; i < n; ++i) {
    switch (e.dtype) {
      case dgpp::DType::BF16: {
        // A block-FP8 scale partner is a positive scale; the rest are weights.
        const float v = e.role == QwenTensorRole::Fp8Scale ? 0.004f + 0.002f * rng.unit()
                                                           : 0.05f * rng.normal3();
        const uint16_t bits = dgpp::float_to_bf16_bits(v);
        std::memcpy(&out[i * 2], &bits, 2);
        break;
      }
      case dgpp::DType::U8:  // two e2m1 codes: every byte is valid
        out[i] = static_cast<uint8_t>(rng.next() & 0xFFu);
        break;
      case dgpp::DType::F8_E4M3:
        if (e.role == QwenTensorRole::Fp8Payload) {  // an FP8 weight code: any but the two NaN codes
          uint8_t b = static_cast<uint8_t>(rng.next() & 0xFFu);
          if ((b & 0x7Fu) == 0x7Fu) b ^= 0x01u;
          out[i] = b;
        } else {  // an NVFP4 block scale in [2^-3, 2^2): positive, never the NaN code
          out[i] = static_cast<uint8_t>(0x20u + (rng.next() % 0x28u));
        }
        break;
      case dgpp::DType::F32: {
        // The NVFP4 per-tensor scale: modelopt's small multiplier, or
        // compressed-tensors' large divisor. Every other scalar: positive.
        float v = 0.5f + 0.25f * rng.unit();
        if (e.role == QwenTensorRole::Fp8TensorScale) v = 0.004f + 0.002f * rng.unit();
        if (e.role == QwenTensorRole::Fp4Global)
          v = cfg.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Packed ? 400.0f + 100.0f * rng.unit()
                                                                 : 0.002f + 0.001f * rng.unit();
        std::memcpy(&out[i * 4], &v, 4);
        break;
      }
      default:
        throw std::runtime_error("fixture dtype not handled: " + e.name);
    }
  }
  return out;
}

// Writes `dir` (config.json + one safetensors shard) for a tiny config and
// returns what was written. `f32_vectors`: the GDN's A_log and output-norm
// weight stored F32, as Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-122B-A10B-FP8
// store them (the table lists them BF16 and binds either).
inline Fixture write_fixture(const std::string& dir, const char* config_json, bool f32_vectors = false) {
  Fixture fx;
  fx.dir = dir;
  const fs::path root(dir);
  fs::remove_all(root);
  fs::create_directories(root);
  {
    std::FILE* f = std::fopen((root / "config.json").c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write config.json");
    std::fwrite(config_json, 1, std::strlen(config_json), f);
    std::fclose(f);
  }
  fx.cfg = Qwen35TextConfig::from_json_file((root / "config.json").string());
  fx.table = dgpp::qwen35_expected_text_tensors(fx.cfg);
  std::string header = "{";
  std::vector<uint8_t> data;
  bool first = true;
  for (const auto& table_entry : fx.table) {
    QwenExpectedTensor e = table_entry;
    if (f32_vectors && e.role == QwenTensorRole::Bf16OrF32) e.dtype = dgpp::DType::F32;
    std::vector<uint8_t> b = tensor_bytes(e, fx.cfg);
    std::string shape = "[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) shape += ",";
      shape += std::to_string(e.shape[i]);
    }
    shape += "]";
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + std::string(dgpp::dtype_name(e.dtype)) +
              "\",\"shape\":" + shape + ",\"data_offsets\":[" + std::to_string(data.size()) + "," +
              std::to_string(data.size() + b.size()) + "]}";
    data.insert(data.end(), b.begin(), b.end());
    fx.src.emplace(e.name, std::move(b));
  }
  // The Qwen3.5 dialect's checkpoints carry a vision tower the text stack
  // skips: one such tensor beside the table.
  if (!fx.cfg.next()) {
    header += ",\"model.visual.patch_embed.proj.weight\":{\"dtype\":\"BF16\",\"shape\":[4,2],\"data_offsets\":[" +
              std::to_string(data.size()) + "," + std::to_string(data.size() + 16) + "]}";
    data.insert(data.end(), 16, uint8_t{0});
  }
  header += "}";
  std::FILE* f = std::fopen((root / "model.safetensors").c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write shard");
  const uint64_t hlen = header.size();
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header.data(), 1, hlen, f);
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  return fx;
}

}  // namespace qwen35portsfx
