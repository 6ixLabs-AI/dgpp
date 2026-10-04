#pragma once
// Synthetic mini-checkpoint for the Qwen3Next dialect's loader test: the
// binding table of a small flat config written as config.json and one
// safetensors shard, so fixture and table cannot disagree. Values are
// deterministic per tensor name (glm_rng's scheme): BF16 weights, random
// e2m1 code bytes under random positive e4m3 block scales, and positive F32
// scalars (distinct per tensor, so the layer maxima are meaningful).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "glm_rng.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"

namespace qwen3nextfx {

namespace fs = std::filesystem;
using dgpp::Qwen35TextConfig;
using dgpp::QwenExpectedTensor;
using dgpp::QwenTensorRole;

// The tiny release: the kernels' pinned widths (128-wide GDN heads, 256-wide
// attention heads) at counts that slice at worlds 1, 2 and 4 — 4 key heads x
// 8 value heads, 4 query heads on 2 kv heads (a kv head shared by two ranks
// at world 4), hidden 256, three GDN layers then one attention layer, the
// draft layer, 4 experts of intermediate 64 and a shared expert of 512 (128
// a rank at world 4: the FP8 form's block grid).
inline const char* tiny_config_json() {
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
  "rope_scaling": null, "rope_theta": 10000000, "shared_expert_intermediate_size": 512,
  "tie_word_embeddings": false, "vocab_size": 64,
  "quantization_config": {"config_groups": {"group_0": {
    "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
    "targets": ["Linear"]}}}
})json";
}

inline Qwen35TextConfig tiny_config() {
  const auto t = dgpp::minijson::parse(tiny_config_json());
  return Qwen35TextConfig::parse_qwen3_next(t.root);
}

inline std::vector<uint8_t> tensor_bytes(const QwenExpectedTensor& e) {
  std::vector<uint8_t> out(e.nbytes());
  glmrng::Rng rng(glmrng::seed_for(e.name));
  const size_t n = e.numel();
  for (size_t i = 0; i < n; ++i) {
    switch (e.dtype) {
      case dgpp::DType::BF16: {
        const uint16_t bits = dgpp::float_to_bf16_bits(0.05f * rng.normal3());
        std::memcpy(&out[i * 2], &bits, 2);
        break;
      }
      case dgpp::DType::U8:  // two e2m1 codes: every byte is valid
        out[i] = static_cast<uint8_t>(rng.next() & 0xFFu);
        break;
      case dgpp::DType::F8_E4M3:  // a positive block scale in [2^-3, 2^2): never the NaN code
        out[i] = static_cast<uint8_t>(0x20u + (rng.next() % 0x28u));
        break;
      case dgpp::DType::F32: {
        // weight_scale_2, input_scale, k_scale / v_scale: positive.
        const float v = e.role == QwenTensorRole::Fp4Global ? 0.002f + 0.001f * rng.unit()
                                                            : 0.5f + 0.25f * rng.unit();
        std::memcpy(&out[i * 4], &v, 4);
        break;
      }
      default:
        throw std::runtime_error("fixture dtype not handled: " + e.name);
    }
  }
  return out;
}

// Writes `dir` (config.json + one safetensors shard) for the tiny release.
inline void write_fixture(const std::string& dir) {
  const Qwen35TextConfig cfg = tiny_config();
  const fs::path root(dir);
  fs::remove_all(root);
  fs::create_directories(root);
  {
    std::FILE* f = std::fopen((root / "config.json").c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write config.json");
    const char* json = tiny_config_json();
    std::fwrite(json, 1, std::strlen(json), f);
    std::fclose(f);
  }
  std::string header = "{";
  std::vector<uint8_t> data;
  bool first = true;
  for (const auto& e : dgpp::qwen35_expected_text_tensors(cfg)) {
    const auto b = tensor_bytes(e);
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
  }
  header += "}";
  std::FILE* f = std::fopen((root / "model.safetensors").c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write shard");
  const uint64_t hlen = header.size();
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header.data(), 1, hlen, f);
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
}

}  // namespace qwen3nextfx
