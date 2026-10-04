#pragma once
// The plain Qwen3 family's device-side fixtures: the unit gate's synthetic
// checkpoints (tests/unit/qwen3_fixture.hpp, held to tools/qwen3_synth.py)
// at GLM-4.7's fixture geometry — hidden 256, 4 query / 2 kv heads of 128,
// 8 experts of 128, a 512-token vocabulary, three layers — in both NVFP4
// containers, plus the small helpers the loader and forward gates share.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "loaders/safetensors.hpp"
#include "models/qwen3/config.hpp"
#include "models/qwen3/reference.hpp"
#include "../unit/qwen3_fixture.hpp"
#include "../unit/qwen3_reference_vectors.hpp"

namespace qwen3_gpu_fixture {

namespace fs = std::filesystem;
namespace v = qwen3_reference_vectors;
using dgpp::Qwen3TextConfig;
using dgpp::qwen3_ref::HostCheckpoint;

inline void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

inline std::string replaced(std::string s, const std::string& from, const std::string& to) {
  const size_t at = s.find(from);
  if (at == std::string::npos) throw std::runtime_error("fixture: '" + from + "' is not in the preset's config");
  return s.replace(at, from.size(), to);
}

// The unit presets' configs at the GPU fixture geometry (the kernels' tiles
// want 256-wide rows and 128-wide expert slices; the unit gate's 64 / 32
// keep its vectors small).
inline std::string gpu_config(const char* preset_json, bool vl) {
  std::string s = preset_json;
  s = replaced(s, "\"hidden_size\": 64", "\"hidden_size\": 256");
  s = replaced(s, "\"moe_intermediate_size\": 32", "\"moe_intermediate_size\": 128");
  s = replaced(s, "\"vocab_size\": 256", "\"vocab_size\": 512");
  if (vl) s = replaced(s, "\"out_hidden_size\": 64", "\"out_hidden_size\": 256");
  return s;
}

struct Fixture {
  std::string dir;
  Qwen3TextConfig cfg;
};
inline Fixture write_fixture(const char* tag, bool vl) {
  Fixture fx;
  fx.dir = (fs::current_path() / (std::string("qwen3_fixture_") + tag)).string();
  fs::remove_all(fx.dir);
  fx.cfg = qwen3_fixture::write_checkpoint(fx.dir, gpu_config(vl ? v::vl::kConfigJson : v::moe::kConfigJson, vl),
                                           /*bare_names=*/false, v::kSeed);
  return fx;
}

inline std::vector<int64_t> ids() { return std::vector<int64_t>(v::kIds, v::kIds + v::kTokens); }

template <class T>
std::vector<T> device(const T* dev, size_t n) {
  std::vector<T> h(n);
  DGPP_CUDA_OK(cudaMemcpy(h.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

// rows [r0, +rn) x columns [c0, +cn) of a row-major [*, full] matrix.
template <class T>
std::vector<T> slice(const std::vector<T>& m, int64_t full, int64_t r0, int64_t rn, int64_t c0, int64_t cn) {
  std::vector<T> out;
  out.reserve(static_cast<size_t>(rn * cn));
  for (int64_t r = r0; r < r0 + rn; ++r)
    out.insert(out.end(), m.begin() + r * full + c0, m.begin() + r * full + c0 + cn);
  return out;
}

inline std::vector<uint8_t> tensor_bytes(const HostCheckpoint& ck, const std::string& name) {
  const dgpp::TensorInfo& t = ck.tensor(name);
  const uint8_t* p = static_cast<const uint8_t*>(t.data);
  return std::vector<uint8_t>(p, p + t.nbytes());
}

}  // namespace qwen3_gpu_fixture
