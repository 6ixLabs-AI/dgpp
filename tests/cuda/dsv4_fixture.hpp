#pragma once
// Synthetic mini-checkpoint writer for the DeepSeek-V4-Flash tests:
// enumerates the binding table for a small config and writes config.json
// + one safetensors shard, so fixture and table cannot disagree. Values
// are deterministic per tensor NAME (glm_rng's scheme) in the release's
// own formats: fp8 e4m3 payloads with e8m0 scales on 128 x 128 blocks,
// MXFP4 experts (random e2m1 nibbles, e8m0 per 32), the hash layers' token
// tables (distinct experts per token, unsorted as the release stores them),
// bf16 / fp32 for the rest — magnitudes that keep a forward's
// nonlinearities informative.
#include <cmath>
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
#include "models/dsv4/binding.hpp"
#include "models/dsv4/config.hpp"

namespace dsv4fx {

namespace fs = std::filesystem;
using dgpp::Dsv4Config;
using dgpp::Dsv4ExpectedTensor;
using dgpp::Dsv4TensorRole;
using dgpp::Dsv4WeightClass;
using dgpp::float_to_bf16_bits;
using glmrng::Rng;
using glmrng::seed_for;

// The tiny release: hidden 256, 16 heads on the 512-wide latent (4 heads
// per rank at world 4 — the attention head tiling's floor; the listed
// flash path at world 1 —, 4 output groups of 4 heads), q_lora / o_lora
// 128, a 64 x 128 indexer selecting 16 entries, six backbone layers
// [window, window, 4, 128, 4, 128] with a 16-token window (layers 0 and 1
// route by token hash), 8 routed experts of 512 (128 per rank at world 4:
// the shared expert's fp8 block) top-2 sqrtsoftplus, vocab 512, two DSpark
// stages (block 5, targets 3-5, Markov rank 32, noise token 500).
// num_nextn_predict_layers is the release's stale 1: the draft stages are
// the compress_ratios entries past the backbone.
inline const char* tiny_config_json() {
  return R"json({
 "architectures": ["DeepseekV4ForCausalLM"],
 "model_type": "deepseek_v4",
 "bos_token_id": 0, "eos_token_id": 1,
 "expert_dtype": "fp4",
 "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                         "weight_block_size": [128, 128], "scale_fmt": "ue8m0"},
 "vocab_size": 512, "hidden_size": 256, "moe_intermediate_size": 512,
 "num_hidden_layers": 6, "num_attention_heads": 16, "num_key_value_heads": 1,
 "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 128, "o_lora_rank": 128, "o_groups": 4,
 "hidden_act": "silu", "swiglu_limit": 10.0, "rms_norm_eps": 1e-06,
 "max_position_embeddings": 4096, "sliding_window": 16,
 "rope_theta": 10000, "compress_rope_theta": 160000,
 "rope_scaling": {"type": "yarn", "factor": 16, "original_max_position_embeddings": 256,
                  "beta_fast": 32, "beta_slow": 1},
 "num_nextn_predict_layers": 1,
 "compress_ratios": [0, 0, 4, 128, 4, 128, 0, 0],
 "index_n_heads": 64, "index_head_dim": 128, "index_topk": 16,
 "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06,
 "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2, "norm_topk_prob": true,
 "routed_scaling_factor": 1.5, "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
 "num_hash_layers": 2,
 "dspark_block_size": 5, "dspark_noise_token_id": 500, "dspark_target_layer_ids": [3, 4, 5],
 "dspark_markov_rank": 32
})json";
}

inline Dsv4Config tiny_config() {
  const auto t = dgpp::minijson::parse(tiny_config_json());
  return Dsv4Config::parse(t.root);
}

inline bool has(const std::string& name, const char* needle) { return name.find(needle) != std::string::npos; }

// The bytes of one tensor.
inline std::vector<uint8_t> tensor_bytes(const Dsv4Config& cfg, const Dsv4ExpectedTensor& e) {
  std::vector<uint8_t> out(e.nbytes());
  const std::string& name = e.name;
  Rng rng(seed_for(name));
  const size_t n = e.numel();
  switch (e.role) {
    case Dsv4TensorRole::Fp8Payload:
      // e4m3 codes of ~N(0, 0.67): the scales below put the weights near 0.03.
      for (size_t i = 0; i < n; ++i) out[i] = dgpp::float_to_fp8_e4m3_bits(2.0f * rng.normal3());
      return out;
    case Dsv4TensorRole::Fp8Scale:
      for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(121 + rng.next() % 3);  // 2^-6 .. 2^-4
      return out;
    case Dsv4TensorRole::Fp4Payload:
      for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(rng.next());  // every e2m1 code is finite
      return out;
    case Dsv4TensorRole::Fp4Scale:
      for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(118 + rng.next() % 3);  // 2^-9 .. 2^-7
      return out;
    case Dsv4TensorRole::Plain:
      break;
  }
  if (e.dtype == dgpp::DType::I64) {
    // gate.tid2eid [vocab, top_k]: top_k distinct experts per token, unsorted.
    const size_t k = static_cast<size_t>(e.shape[1]);
    const int E = cfg.n_routed_experts;
    for (size_t t = 0; t < static_cast<size_t>(e.shape[0]); ++t) {
      std::vector<int64_t> row;
      while (row.size() < k) {
        const int64_t id = static_cast<int64_t>(rng.next() % static_cast<uint32_t>(E));
        bool dup = false;
        for (const int64_t r : row) dup = dup || r == id;
        if (!dup) row.push_back(id);
      }
      std::memcpy(&out[t * k * 8], row.data(), k * 8);
    }
    return out;
  }
  const bool is_norm = has(name, "norm") && e.shape.size() == 1;  // gains near 1
  const bool is_bias = has(name, ".bias");
  const bool is_hc_scale = e.cls == Dsv4WeightClass::Mhc && has(name, "_scale");
  const bool is_ape = has(name, ".ape");
  const bool is_small = e.cls == Dsv4WeightClass::Mhc || has(name, "attn_sink") || is_bias;
  const bool is_embed = e.cls == Dsv4WeightClass::Embed || e.cls == Dsv4WeightClass::LmHead || has(name, "markov_w1");
  for (size_t i = 0; i < n; ++i) {
    float v;
    if (is_norm) v = 0.9f + 0.2f * (0.5f * (rng.unit() + 1.0f));
    else if (is_hc_scale) v = 1.0f + 0.1f * rng.unit();
    else if (is_ape) v = 0.5f * rng.normal3();
    else if (is_small) v = 0.02f * rng.normal3();
    else if (is_embed) v = 0.3f * rng.normal3();
    else v = 0.05f * rng.normal3();  // projections, routers, indexers, compressors
    if (e.dtype == dgpp::DType::BF16) {
      const uint16_t bits = float_to_bf16_bits(v);
      std::memcpy(&out[i * 2], &bits, 2);
    } else if (e.dtype == dgpp::DType::F32) {
      std::memcpy(&out[i * 4], &v, 4);
    } else {
      throw std::runtime_error("fixture dtype not handled: " + name);
    }
  }
  return out;
}

// Writes `dir` (config.json + one safetensors shard holding every tensor)
// for `cfg`.
inline void write_fixture(const Dsv4Config& cfg, const std::string& dir, const char* config_json) {
  fs::path root(dir);
  fs::remove_all(root);
  fs::create_directories(root);
  {
    const fs::path p = root / "config.json";
    std::FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write config.json");
    std::fwrite(config_json, 1, std::strlen(config_json), f);
    std::fclose(f);
  }
  const auto table = dgpp::dsv4_expected_tensors(cfg);
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  bool first = true;
  for (const auto& e : table) {
    const auto b = tensor_bytes(cfg, e);
    std::string shape = "[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) shape += ",";
      shape += std::to_string(e.shape[i]);
    }
    shape += "]";
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + std::string(dgpp::dtype_name(e.dtype)) + "\",\"shape\":" + shape +
              ",\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + b.size()) + "]}";
    data.insert(data.end(), b.begin(), b.end());
    off += b.size();
  }
  header += "}";
  const fs::path shard = root / "model.safetensors";
  std::FILE* f = std::fopen(shard.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write shard");
  const uint64_t hlen = header.size();
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header.data(), 1, hlen, f);
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  std::printf("dsv4 fixture written: %zu tensors, %.2f MB payload\n", table.size(),
              static_cast<double>(data.size()) / 1048576.0);
}

inline void write_fixture(const Dsv4Config& cfg, const std::string& dir) {
  write_fixture(cfg, dir, tiny_config_json());
}

}  // namespace dsv4fx
