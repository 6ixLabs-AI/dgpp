// Qwen3-Coder-Next's container on the Qwen3Next dialect's loader, host side:
// the compressed-tensors NVFP4 dequant (the same codes and block scales as
// modelopt's, the per-tensor scale a divisor), and the counting build over
// the release's table — at world 1 it reads every byte of a layer, the ranks
// of a world tile it, there is no draft layer to count, and the dense stack
// has both forms (BF16, and block FP8 under set_dense_weights_fp8). The copy
// pass is tests/cuda/qwen35_ports_loader_test.cpp.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "loaders/nvfp4_quant.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/loader35.hpp"

namespace {

using dgpp::Qwen35LayerStream;
using dgpp::Qwen35LoaderFamily;
using dgpp::Qwen35TextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The release's shape (RedHatAI/Qwen3-Coder-Next-NVFP4), through the parser.
Qwen35TextConfig release_config() {
  std::string layers;
  for (int i = 0; i < 48; ++i) {
    if (i) layers += ", ";
    layers += (i + 1) % 4 == 0 ? "\"full_attention\"" : "\"linear_attention\"";
  }
  const std::string json = std::string(R"({
  "architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next",
  "bos_token_id": 151643, "eos_token_id": 151645, "decoder_sparse_step": 1,
  "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu", "hidden_size": 2048,
  "intermediate_size": 5120, "layer_types": [)") +
                           layers + R"(],
  "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
  "linear_num_value_heads": 32, "linear_value_head_dim": 128,
  "max_position_embeddings": 262144, "mlp_only_layers": [], "moe_intermediate_size": 512,
  "norm_topk_prob": true, "num_attention_heads": 16, "num_experts": 512,
  "num_experts_per_tok": 10, "num_hidden_layers": 48, "num_key_value_heads": 2,
  "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06, "rope_scaling": null,
  "rope_theta": 5000000, "shared_expert_intermediate_size": 512,
  "tie_word_embeddings": false, "vocab_size": 151936,
  "quantization_config": {"config_groups": {"group_0": {"format": "nvfp4-pack-quantized",
    "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16,
                "strategy": "tensor_group", "symmetric": true},
    "targets": ["Linear"]}}, "format": "nvfp4-pack-quantized", "kv_cache_scheme": null,
    "quant_method": "compressed-tensors", "quantization_status": "compressed",
    "sparsity_config": {}, "transform_config": {}}
})";
  const auto t = dgpp::minijson::parse(json);
  return Qwen35TextConfig::parse_qwen3_next(t.root);
}

// e2m1 and e4m3 from their definitions (not the engine's converters).
float e2m1(uint8_t code) {
  static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return (code & 8u) ? -mag[code & 7u] : mag[code & 7u];
}
float e4m3(uint8_t code) {
  const int e = (code >> 3) & 0xF, m = code & 7;
  const float v = e == 0 ? std::ldexp(static_cast<float>(m), -9)
                         : std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
  return (code & 0x80u) ? -v : v;
}

struct DenseForm {
  bool saved = Qwen35LayerStream::dense_weights_fp8();
  explicit DenseForm(bool fp8) { Qwen35LayerStream::set_dense_weights_fp8(fp8); }
  ~DenseForm() { Qwen35LayerStream::set_dense_weights_fp8(saved); }
};

size_t fp8_saving(int64_t n, int64_t k) {
  const size_t bf16 = dgpp::align_up_256(static_cast<size_t>(n * k) * 2);
  const size_t fp8 =
      dgpp::align_up_256(static_cast<size_t>(n * k)) +
      dgpp::align_up_256(static_cast<size_t>(((n + 127) / 128) * ((k + 127) / 128)) * 4);
  return bf16 - fp8;
}

size_t layer_fp8_saving(const Qwen35TextConfig& c, int layer, int world) {
  const int64_t H = c.hidden_size, S = c.shared_expert_intermediate_size / world;
  size_t n = 2 * fp8_saving(S, H) + fp8_saving(H, S);  // the shared expert
  if (c.layers[layer] == dgpp::Qwen35LayerKind::Gdn) {
    const int64_t K = static_cast<int64_t>(c.gdn_key_heads) * c.gdn_key_head_dim / world;
    const int64_t V = static_cast<int64_t>(c.gdn_value_heads) * c.gdn_value_head_dim / world;
    return n + fp8_saving(2 * K + V, H) + fp8_saving(V, H) + fp8_saving(H, V);  // qkv, z, out
  }
  const int64_t q = static_cast<int64_t>(c.num_attention_heads) / world * c.head_dim;
  const int64_t kv = static_cast<int64_t>(std::max(1, c.num_key_value_heads / world)) * c.head_dim;
  return n + fp8_saving(2 * q, H) + 2 * fp8_saving(kv, H) + fp8_saving(H, q);  // q, k, v, o
}

size_t table_bytes(const Qwen35TextConfig& c, int layer, bool replicated_only) {
  size_t n = 0;
  for (const auto& e : Qwen35LoaderFamily::layer_table(c, layer))
    if (!replicated_only || Qwen35LoaderFamily::digest_included(e)) n += e.nbytes();
  return n;
}

}  // namespace

DGPP_TEST(qwen3codernext_fp4_dequant_is_code_times_block_scale_over_the_global_scale) {
  // Nibble order and the scale's direction on one block: byte 0x21 holds
  // code 1 (0.5) for the even column and code 2 (1.0) for the odd one; 0x38
  // is the e4m3 code of 1.0; weight_global_scale DIVIDES.
  {
    std::vector<uint8_t> payload(8, 0x21), scales(1, 0x38);
    payload[1] = 0xF7;  // +6 (even), -6 (odd)
    std::vector<uint16_t> out(16);
    dgpp::qwen35_fp4_packed_dequant_bf16(payload.data(), 8, scales.data(), 1, 1.0f, 1, 16, out.data());
    require(out[0] == 0x3F00 && out[1] == 0x3F80, "the low nibble is the even column");
    require(out[2] == 0x40C0 && out[3] == 0xC0C0, "sign bit and the largest code");
    dgpp::qwen35_fp4_packed_dequant_bf16(payload.data(), 8, scales.data(), 1, 4.0f, 1, 16, out.data());
    require(out[2] == 0x3FC0, "weight_global_scale divides (6 / 4 = 1.5)");
    // The modelopt convention on the same bytes multiplies: the two differ.
    dgpp::qwen3next_fp4_dequant_bf16(payload.data(), 8, scales.data(), 1, 4.0f, 1, 16, out.data());
    require(out[2] == 0x41C0, "weight_scale_2 multiplies (6 x 4 = 24)");
  }

  // Arbitrary codes against the formula written out: the block scale over
  // the global in fp32, times the code in fp32, rounded to bf16.
  constexpr int64_t rows = 6, cols = 96;
  std::vector<uint8_t> payload(rows * cols / 2), scales(rows * cols / 16);
  uint32_t state = 0x85EBCA6Bu;
  const auto next = [&state] {
    state = state * 1664525u + 1013904223u;
    return static_cast<uint8_t>(state >> 24);
  };
  for (auto& b : payload) b = next();
  for (auto& s : scales) s = static_cast<uint8_t>(next() & 0x77u);  // positive, never the NaN code
  const float global = 731.25f;
  std::vector<uint16_t> full(rows * cols);
  dgpp::qwen35_fp4_packed_dequant_bf16(payload.data(), cols / 2, scales.data(), cols / 16, global, rows,
                                       cols, full.data());
  for (int64_t n = 0; n < rows; ++n)
    for (int64_t k = 0; k < cols; ++k) {
      const uint8_t byte = payload[static_cast<size_t>(n * (cols / 2) + k / 2)];
      const uint8_t code =
          (k & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0xF);
      const float block = e4m3(scales[static_cast<size_t>(n * (cols / 16) + k / 16)]) / global;
      const float value = e2m1(code) * block;
      require(full[static_cast<size_t>(n * cols + k)] == dgpp::float_to_bf16_bits(value),
              "element " + std::to_string(n) + "," + std::to_string(k));
      // The routed experts keep the codes and hand the kernels the global as
      // their divisor: the same value to the format's precision.
      const float kernel = dgpp::nvfp4_decode(payload.data(), scales.data(), global, cols, n, k);
      require(std::fabs(kernel - value) <= std::fabs(value) * 1e-6f,
              "the expert kernels' convention");
    }

  // A slice (rows 1..4, columns 32..80) is the same elements.
  std::vector<uint16_t> part(3 * 48);
  dgpp::qwen35_fp4_packed_dequant_bf16(payload.data() + 1 * (cols / 2) + 32 / 2, cols / 2,
                                       scales.data() + 1 * (cols / 16) + 32 / 16, cols / 16, global,
                                       3, 48, part.data());
  for (int64_t n = 0; n < 3; ++n)
    for (int64_t k = 0; k < 48; ++k)
      require(part[static_cast<size_t>(n * 48 + k)] ==
                  full[static_cast<size_t>((n + 1) * cols + 32 + k)],
              "a slice dequantizes to the same elements");
}

DGPP_TEST(qwen3codernext_loader_counting_build_reads_the_whole_layer_and_ranks_tile_it) {
  const Qwen35TextConfig c = release_config();
  require(c.mtp_layer() == -1 && Qwen35LoaderFamily::max_layer(c) == 48, "no draft layer");
  for (int layer : {0, 3}) {
    const uint64_t full = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 1);
    require(full == table_bytes(c, layer, false),
            "world 1 reads every byte of layer " + std::to_string(layer) + "'s table");
    const uint64_t r0 = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 2);
    const uint64_t r1 = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 1, 2);
    require(r0 == r1, "the ranks of world 2 read alike");
    require(r0 + r1 == full + table_bytes(c, layer, true),
            "world 2 tiles layer " + std::to_string(layer));
  }
  {
    // World 4, an attention layer: each of the two kv heads lives on two
    // ranks, so the NVFP4 k_proj and v_proj rows (codes and block scales)
    // are read twice.
    uint64_t sum = 0;
    for (int r = 0; r < 4; ++r) sum += Qwen35LayerStream::planned_layer_source_bytes(c, 3, r, 4);
    const uint64_t kv_rows = static_cast<uint64_t>(c.num_key_value_heads) * c.head_dim;
    const uint64_t H = static_cast<uint64_t>(c.hidden_size);
    require(sum == table_bytes(c, 3, false) + 3 * table_bytes(c, 3, true) +
                       2 * kv_rows * (H / 2 + H / 16),
            "world 4 shares the kv heads of an attention layer");
  }

  const size_t gdn = Qwen35LayerStream::layer_bytes(c, 0), attn = Qwen35LayerStream::layer_bytes(c, 3);
  for (int l = 0; l < c.num_hidden_layers; ++l)
    require(Qwen35LayerStream::layer_bytes(c, l) ==
                (c.layers[l] == dgpp::Qwen35LayerKind::Gdn ? gdn : attn),
            "layer " + std::to_string(l) + " bytes follow its kind");
  const size_t globals = Qwen35LayerStream::globals_bytes(c);
  const size_t total = Qwen35LayerStream::resident_bytes(c, 0, 1, dgpp::LoaderHeadSharding::Full, true);
  require(total == 36 * gdn + 12 * attn + globals,
          "resident bytes are the layers and the globals: there is no draft layer");
  require(total == Qwen35LayerStream::resident_bytes(c, 0, 1, dgpp::LoaderHeadSharding::Full, false),
          "with or without the draft flag");
  // Globals: embed, head and final norm, nothing of a draft head.
  const size_t V = 151936, H = 2048;
  require(globals == 2 * dgpp::align_up_256(V * H * 2) + dgpp::align_up_256(H * 2), "globals bytes");
}

DGPP_TEST(qwen3codernext_loader_replicated_set) {
  const Qwen35TextConfig c = release_config();
  const auto replicated = [&](int layer, const std::string& suffix) {
    const std::string name = dgpp::qwen35_layer_prefix(c, layer) + suffix;
    for (const auto& e : Qwen35LoaderFamily::layer_table(c, layer))
      if (e.name == name) return Qwen35LoaderFamily::digest_included(e);
    throw std::runtime_error("no tensor " + name);
  };
  require(replicated(0, "input_layernorm.weight") && replicated(0, "linear_attn.norm.weight"), "norms");
  require(replicated(0, "mlp.gate.weight") && replicated(0, "mlp.shared_expert_gate.weight"), "router");
  require(replicated(3, "self_attn.q_proj.weight_global_scale") &&
              replicated(3, "self_attn.q_proj.input_global_scale") &&
              replicated(0, "mlp.experts.7.down_proj.weight_global_scale") &&
              replicated(0, "mlp.experts.7.down_proj.input_global_scale") &&
              replicated(3, "mlp.shared_expert.up_proj.weight_global_scale"),
          "the NVFP4 set's two scalars");
  require(replicated(3, "self_attn.q_norm.weight") && replicated(3, "self_attn.k_norm.weight"),
          "attention norms");
  require(!replicated(0, "linear_attn.in_proj_qkvz.weight") &&
              !replicated(0, "linear_attn.in_proj_ba.weight") &&
              !replicated(0, "linear_attn.out_proj.weight") && !replicated(0, "linear_attn.A_log"),
          "GDN slices (the BF16 out_proj among them)");
  require(!replicated(3, "self_attn.q_proj.weight_packed") &&
              !replicated(3, "self_attn.k_proj.weight_scale") &&
              !replicated(3, "self_attn.o_proj.weight_packed") &&
              !replicated(3, "mlp.shared_expert.down_proj.weight_packed") &&
              !replicated(3, "mlp.experts.0.gate_proj.weight_packed") &&
              !replicated(3, "mlp.experts.0.gate_proj.weight_scale"),
          "attention and MoE slices");
}

DGPP_TEST(qwen3codernext_loader_dense_fp8_counting_build) {
  const Qwen35TextConfig c = release_config();
  require(!Qwen35LayerStream::dense_weights_fp8(), "the dense stack is BF16 by default");
  const size_t bf16_total =
      Qwen35LayerStream::resident_bytes(c, 0, 1, dgpp::LoaderHeadSharding::Full, false);
  {
    const DenseForm fp8(true);
    size_t fp8_total = Qwen35LayerStream::globals_bytes(c);
    for (int layer = 0; layer < c.num_hidden_layers; ++layer) {
      size_t bf16_bytes = 0;
      {
        const DenseForm bf16(false);
        bf16_bytes = Qwen35LayerStream::layer_bytes(c, layer);
      }
      const size_t fp8_bytes = Qwen35LayerStream::layer_bytes(c, layer);
      require(fp8_bytes == bf16_bytes - layer_fp8_saving(c, layer, 1),
              "layer " + std::to_string(layer) + " bytes: every dense matrix is codes plus a scale grid");
      fp8_total += fp8_bytes;
    }
    require(Qwen35LayerStream::resident_bytes(c, 0, 1, dgpp::LoaderHeadSharding::Full, false) ==
                    fp8_total &&
                fp8_total < bf16_total,
            "resident bytes follow the form");
    for (int layer : {0, 3}) {
      const uint64_t full = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 1);
      require(full == table_bytes(c, layer, false), "the FP8 form reads every byte of the layer");
      for (int world : {1, 2, 4})
        for (int rank = 0; rank < world; ++rank) {
          size_t bf16_bytes = 0;
          {
            const DenseForm bf16(false);
            bf16_bytes = Qwen35LayerStream::layer_bytes(c, layer, rank, world);
          }
          require(Qwen35LayerStream::layer_bytes(c, layer, rank, world) ==
                      bf16_bytes - layer_fp8_saving(c, layer, world),
                  "layer " + std::to_string(layer) + " at world " + std::to_string(world) +
                      " rank " + std::to_string(rank));
        }
    }
  }
  require(Qwen35LayerStream::resident_bytes(c, 0, 1, dgpp::LoaderHeadSharding::Full, false) ==
              bf16_total,
          "the BF16 form is back, unchanged");
}
