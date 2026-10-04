#pragma once
// Synthetic plain-Qwen3 checkpoints for the host and GPU gates: the C++ twin
// of tools/qwen3_synth.py's generator. Every tensor of the binding table is
// filled from an integer stream seeded by its NAME (splitmix64 over FNV-1a),
// in its STORED form — exact BF16 values, NVFP4 code bytes, e4m3 block
// scales, F32 scales — so no float encoder is involved and the two
// generators produce the same bytes from the same rules. The table here is
// the engine's (qwen3_expected_tensors); the tool's is written separately
// from the releases' headers; the content digest the tool pins
// (qwen3_reference_vectors.hpp) is reached only if both agree on every name,
// dtype, shape and byte.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen3/binding.hpp"
#include "models/qwen3/config.hpp"

namespace qwen3_fixture {

inline uint64_t fnv1a64(std::string_view s, uint64_t h = 0xCBF29CE484222325ull) {
  for (const char c : s) h = (h ^ static_cast<uint8_t>(c)) * 0x100000001B3ull;
  return h;
}

// Output i of the tensor's stream (tools/qwen3_synth.py stream()).
struct Stream {
  uint64_t base;
  Stream(std::string_view name, uint64_t seed) : base(fnv1a64(name) ^ (seed * 0xD6E8FEB86659FD93ull)) {}
  uint64_t at(uint64_t i) const {
    uint64_t z = base + (i + 1) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
};

// A BF16 vector the tool fills as a norm weight (values in [0.75, 1.25]):
// `<something with "norm">.weight`.
inline bool is_norm_weight(const std::string& name) {
  static const std::string_view kSuffix = ".weight";
  if (name.size() <= kSuffix.size() || name.compare(name.size() - kSuffix.size(), kSuffix.size(), kSuffix) != 0)
    return false;
  const std::string stem = name.substr(0, name.size() - kSuffix.size());
  const size_t dot = stem.rfind('.');
  const std::string leaf = dot == std::string::npos ? stem : stem.substr(dot + 1);
  return leaf.find("norm") != std::string::npos;
}

// One tensor's stored bytes (tools/qwen3_synth.py fill()).
inline std::vector<uint8_t> fill(const dgpp::Qwen3ExpectedTensor& e, uint64_t seed, bool packed) {
  const size_t n = e.numel();
  const Stream s(e.name, seed);
  std::vector<uint8_t> out(e.nbytes());
  auto put_f32 = [&](size_t i, float v) { std::memcpy(out.data() + i * 4, &v, 4); };
  auto put_bf16 = [&](size_t i, float v) {
    const uint16_t b = dgpp::float_to_bf16_bits(v);
    std::memcpy(out.data() + i * 2, &b, 2);
  };
  using dgpp::Qwen3TensorRole;
  for (size_t i = 0; i < n; ++i) {
    const uint64_t r = s.at(i);
    switch (e.role) {
      case Qwen3TensorRole::Fp4Payload:
        out[i] = static_cast<uint8_t>((r >> 24) & 0xFFu);
        break;
      case Qwen3TensorRole::Fp4Scale:  // e4m3 in [0.5, 1.875]
        out[i] = static_cast<uint8_t>(((6u + ((r >> 16) & 1u)) << 3) | ((r >> 20) & 7u));
        break;
      case Qwen3TensorRole::Fp4Global: {
        const float k = static_cast<float>((r >> 11) % (packed ? 64u : 16u));
        put_f32(i, packed ? 64.0f + k : (16.0f + k) / 2048.0f);
        break;
      }
      case Qwen3TensorRole::InputScale:
      case Qwen3TensorRole::KvScale:
        put_f32(i, 1.0f);
        break;
      case Qwen3TensorRole::Plain:
        if (e.dtype != dgpp::DType::BF16) throw std::logic_error("qwen3_fixture: unexpected dtype of " + e.name);
        if (is_norm_weight(e.name))
          put_bf16(i, 1.0f + (static_cast<float>((r >> 11) % 65u) - 32.0f) / 128.0f);
        else
          put_bf16(i, (static_cast<float>((r >> 11) % 255u) - 127.0f) / 512.0f);
        break;
    }
  }
  return out;
}

// Writes DIR/config.json and DIR/model.safetensors: every tensor of the
// config's binding table (the vision tower's included). Returns the config
// as parsed (bare_names applied).
inline dgpp::Qwen3TextConfig write_checkpoint(const std::string& dir, const std::string& config_json,
                                              bool bare_names, uint64_t seed) {
  namespace fs = std::filesystem;
  fs::create_directories(dir);
  {
    std::ofstream f(fs::path(dir) / "config.json", std::ios::binary);
    f << config_json;
    if (!f) throw std::runtime_error("qwen3_fixture: cannot write config.json under " + dir);
  }
  const auto parsed = dgpp::minijson::parse(config_json);
  dgpp::Qwen3TextConfig cfg = dgpp::Qwen3TextConfig::parse(parsed.root);
  cfg.bare_names = bare_names;
  const auto table = dgpp::qwen3_expected_tensors(cfg);
  std::string header = "{";
  std::vector<std::vector<uint8_t>> blobs;
  blobs.reserve(table.size());
  uint64_t off = 0;
  for (const auto& e : table) {
    blobs.push_back(fill(e, seed, cfg.fp4_packed()));
    if (header.size() > 1) header += ",";
    header += "\"" + e.name + "\":{\"dtype\":\"" + std::string(dgpp::dtype_name(e.dtype)) + "\",\"shape\":[";
    for (size_t i = 0; i < e.shape.size(); ++i) header += (i ? "," : "") + std::to_string(e.shape[i]);
    header += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + blobs.back().size()) + "]}";
    off += blobs.back().size();
  }
  header += "}";
  while (header.size() % 8 != 0) header += " ";
  std::ofstream f(fs::path(dir) / "model.safetensors", std::ios::binary);
  const uint64_t hlen = header.size();
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const auto& b : blobs) f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
  if (!f) throw std::runtime_error("qwen3_fixture: cannot write model.safetensors under " + dir);
  return cfg;
}

}  // namespace qwen3_fixture
