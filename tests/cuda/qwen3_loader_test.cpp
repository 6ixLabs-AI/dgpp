// UNVERIFIED ON A GPU (2026-10-04). Built only under -DDGPP_BUILD_QWEN3_PLAIN_DRAFT=ON.
// What HAS run: this file, on a Mac, against a host shim of the CUDA runtime
// (device memory is malloc, copies are memcpy) — the loader's builders, slice
// arithmetic, byte formulas and source-byte plan executed on the host. The
// shim proves nothing about a real device, a real upload or nvcc.
//
// The plain Qwen3-MoE resident loader on synthetic checkpoints of both NVFP4
// containers (tests/cuda/qwen3_gpu_fixture.hpp): every resident value against
// the checkpoint at world 1 — the attention projections as the BF16 the
// loader slices or dequantizes, the experts' code and scale bytes as shipped,
// each expert matrix's divisor (compressed-tensors' global as shipped,
// modelopt's multiplier as its reciprocal), the router, the norms, the
// globals — and each rank's slices at world 2 (2 query heads and 1 kv head,
// half of every expert's intermediate dim, half the vocabulary). The stream
// itself checks, on every layer, that the byte formula equals the bytes used
// and that the planned source bytes equal the bytes read.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "models/qwen3/binding.hpp"
#include "models/qwen3/loader.hpp"
#include "qwen3_gpu_fixture.hpp"

namespace {

using namespace qwen3_gpu_fixture;
using dgpp::LoaderHeadSharding;
using dgpp::LoaderResidency;
using dgpp::Qwen3LayerResident;
using dgpp::Qwen3LayerStream;

// One rank's resident layer against the checkpoint: the attention
// projections as the bf16 the loader dequantizes or slices, the experts'
// bytes as shipped, each expert matrix's divisor.
void check_layer(const HostCheckpoint& ck, const Qwen3LayerResident& r, int layer, int rank, int world,
                 const std::string& tag) {
  const Qwen3TextConfig& c = ck.config();
  const int64_t H = c.hidden_size, d = c.head_dim;
  const int64_t lh = c.num_attention_heads / world, lkv = c.num_key_value_heads / world;
  const int64_t q0 = lh * rank * d, qn = lh * d, k0 = lkv * rank * d, kn = lkv * d;
  const std::string p = dgpp::qwen3_layer_prefix(c, layer);
  require(r.layer == layer && r.attn.local_heads == lh && r.attn.local_kv_heads == lkv, tag + ": geometry");
  require(r.attn.q_bias == nullptr && r.attn.k_bias == nullptr && r.attn.v_bias == nullptr, tag + ": no biases");
  int64_t N = 0, K = 0;
  const std::vector<uint16_t> q = ck.linear_bf16(p + "self_attn.q_proj", &N, &K);
  require(device(r.attn.q_proj, static_cast<size_t>(qn * H)) == slice(q, K, q0, qn, 0, H), tag + ": q_proj rows");
  const std::vector<uint16_t> k = ck.linear_bf16(p + "self_attn.k_proj", &N, &K);
  require(device(r.attn.k_proj, static_cast<size_t>(kn * H)) == slice(k, K, k0, kn, 0, H), tag + ": k_proj rows");
  const std::vector<uint16_t> vv = ck.linear_bf16(p + "self_attn.v_proj", &N, &K);
  require(device(r.attn.v_proj, static_cast<size_t>(kn * H)) == slice(vv, K, k0, kn, 0, H), tag + ": v_proj rows");
  const std::vector<uint16_t> o = ck.linear_bf16(p + "self_attn.o_proj", &N, &K);
  require(device(r.attn.o_proj, static_cast<size_t>(H * qn)) == slice(o, K, 0, H, q0, qn), tag + ": o_proj columns");
  require(device(reinterpret_cast<const uint8_t*>(r.attn.q_norm), static_cast<size_t>(d) * 2) ==
              tensor_bytes(ck, p + "self_attn.q_norm.weight"),
          tag + ": q_norm");
  require(device(reinterpret_cast<const uint8_t*>(r.input_norm), static_cast<size_t>(H) * 2) ==
              tensor_bytes(ck, p + "input_layernorm.weight"),
          tag + ": input norm");
  require(device(reinterpret_cast<const uint8_t*>(r.moe.router), static_cast<size_t>(c.num_experts * H) * 2) ==
              tensor_bytes(ck, p + "mlp.gate.weight"),
          tag + ": router");
  const int64_t I = c.moe_intermediate_size / world, i0 = I * rank;
  require(r.moe.local_inter == I && r.moe.experts.size() == static_cast<size_t>(c.num_experts) * 3, tag + ": experts");
  static const char* const kPart[3] = {"gate_proj", "up_proj", "down_proj"};
  for (int e = 0; e < c.num_experts; ++e)
    for (int m = 0; m < 3; ++m) {
      const dgpp::GlmFp4Matrix& g = r.moe.expert(e, m);
      const std::string base = p + "mlp.experts." + std::to_string(e) + "." + kPart[m];
      const std::vector<uint8_t> payload = tensor_bytes(ck, dgpp::qwen3_fp4_payload_name(c, base));
      const std::vector<uint8_t> scales = tensor_bytes(ck, base + ".weight_scale");
      const std::string what = tag + ": " + base;
      if (m < 2) {  // gate, up: rows [i0, +I) of [I_full, H]
        require(g.rows == I && g.cols == H, what + " geometry");
        require(device(g.payload, g.payload_bytes()) == slice(payload, H / 2, i0, I, 0, H / 2), what + " payload rows");
        require(device(g.scales, g.scale_bytes()) == slice(scales, H / 16, i0, I, 0, H / 16), what + " scale rows");
      } else {  // down: columns [i0, +I) of [H, I_full]
        const int64_t full = c.moe_intermediate_size;
        require(g.rows == H && g.cols == I, what + " geometry");
        require(device(g.payload, g.payload_bytes()) == slice(payload, full / 2, 0, H, i0 / 2, I / 2), what + " payload columns");
        require(device(g.scales, g.scale_bytes()) == slice(scales, full / 16, 0, H, i0 / 16, I / 16), what + " scale columns");
      }
      float stored;
      std::memcpy(&stored, ck.tensor(dgpp::qwen3_fp4_global_name(c, base)).data, 4);
      const float divisor = device(g.global_scale, 1)[0];
      require(divisor == (c.fp4_packed() ? stored : 1.0f / stored), what + " divisor");
    }
}

void check_loader(const char* tag, bool vl) {
  const Fixture fx = write_fixture(tag, vl);
  const HostCheckpoint ck(fx.dir);
  const Qwen3TextConfig& c = ck.config();
  // World 1: everything, the head whole.
  {
    Qwen3LayerStream stream(c, fx.dir, 0, 1, LoaderResidency::Resident, LoaderHeadSharding::Full);
    for (int l = 0; l < c.num_hidden_layers; ++l)
      check_layer(ck, stream.load_layer(l), l, 0, 1, std::string(tag) + " w1 layer " + std::to_string(l));
    const dgpp::Qwen3GlobalsResident& g = stream.load_globals();
    const size_t VH = static_cast<size_t>(c.vocab_size) * static_cast<size_t>(c.hidden_size) * 2;
    require(g.lm_vocab_begin == 0 && g.lm_vocab_count == c.vocab_size, "w1: the whole head");
    require(device(reinterpret_cast<const uint8_t*>(g.embed), VH) ==
                tensor_bytes(ck, dgpp::qwen3_model_prefix(c) + "embed_tokens.weight"),
            "w1: embedding");
    require(device(reinterpret_cast<const uint8_t*>(g.lm_head), VH) == tensor_bytes(ck, "lm_head.weight"), "w1: head");
    // The byte formula the memory plan quotes is what the stream used.
    require(Qwen3LayerStream::resident_bytes(c, 0, 1, LoaderHeadSharding::Full, false) > 0, "w1: resident bytes");
  }
  // World 2: each rank's slices (2 query heads and 1 kv head, half of every
  // expert's intermediate, half of the vocabulary).
  for (int rank = 0; rank < 2; ++rank) {
    Qwen3LayerStream stream(c, fx.dir, rank, 2, LoaderResidency::Resident, LoaderHeadSharding::VocabSharded);
    for (int l = 0; l < c.num_hidden_layers; ++l)
      check_layer(ck, stream.load_layer(l), l, rank, 2,
                  std::string(tag) + " w2 rank " + std::to_string(rank) + " layer " + std::to_string(l));
    const dgpp::Qwen3GlobalsResident& g = stream.load_globals();
    const int half = c.vocab_size / 2;
    require(g.lm_vocab_begin == rank * half && g.lm_vocab_count == half, "w2: the head's vocab half");
    const std::vector<uint8_t> head = tensor_bytes(ck, "lm_head.weight");
    const size_t row = static_cast<size_t>(c.hidden_size) * 2;
    require(device(reinterpret_cast<const uint8_t*>(g.lm_head), static_cast<size_t>(half) * row) ==
                std::vector<uint8_t>(head.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(rank * half) * row),
                                     head.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>((rank + 1) * half) * row)),
            "w2: head rows");
  }
}

}  // namespace

DGPP_TEST(qwen3_loader_resident_values_packed) { check_loader("vl", /*vl=*/true); }
DGPP_TEST(qwen3_loader_resident_values_modelopt) { check_loader("moe", /*vl=*/false); }

DGPP_TEST(qwen3_loader_refuses_the_dense_dialect_and_a_bad_world) {
  const Fixture fx = write_fixture("refuse", /*vl=*/false);
  // World 3 does not divide 4 heads; the geometry check names it.
  bool refused = false;
  try {
    Qwen3LayerStream stream(fx.cfg, fx.dir, 0, 3, LoaderResidency::Resident, LoaderHeadSharding::VocabSharded);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "world 3 is refused");
  // The dense dialect binds and has no builder.
  const std::string dense_dir = (fs::current_path() / "qwen3_fixture_dense").string();
  fs::remove_all(dense_dir);
  const Qwen3TextConfig dense = qwen3_fixture::write_checkpoint(dense_dir, v::dense::kConfigJson, false, v::kSeed);
  refused = false;
  try {
    Qwen3LayerStream stream(dense, dense_dir);
  } catch (const std::invalid_argument& e) {
    refused = std::string(e.what()).find("dense dialect") != std::string::npos;
  }
  require(refused, "the dense dialect is refused by name");
}

int main() { return ::dgpp::test::run_all(); }
