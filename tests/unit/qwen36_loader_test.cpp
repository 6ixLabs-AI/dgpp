// Qwen3.6-35B-A3B on the qwen3_5 loader family, host side: the counting
// build over both releases' tables — at world 1 it reads every byte of a
// layer (the draft layer's two stacked expert tensors included), the ranks
// of a world tile it, the replicated set is the scalars and the norms, and
// the dense knob moves only what a release ships outside block FP8. The copy
// pass is tests/cuda/qwen35_ports_loader_test.cpp.
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
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

const char* kQuantMixed = R"({"config_groups": {
    "group_0": {"weights": {"dynamic": false, "num_bits": 8, "type": "float"}},
    "group_1": {"weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16}}},
  "quant_algo": "MIXED_PRECISION", "quant_method": "modelopt"})";
const char* kQuantFp8 = R"({"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
  "weight_block_size": [128, 128]})";

// The releases' shape, through the parser.
Qwen35TextConfig release_config(const char* quant) {
  std::string layers;
  for (int i = 0; i < 40; ++i) {
    if (i) layers += ", ";
    layers += (i + 1) % 4 == 0 ? "\"full_attention\"" : "\"linear_attention\"";
  }
  const std::string json = std::string(R"({
  "model_type": "qwen3_5_moe_text", "attention_bias": false, "attn_output_gate": true,
  "bos_token_id": 248044, "eos_token_id": 248044, "full_attention_interval": 4, "head_dim": 256,
  "hidden_act": "silu", "hidden_size": 2048, "layer_types": [)") +
                           layers + R"(],
  "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
  "linear_num_value_heads": 32, "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
  "max_position_embeddings": 262144, "moe_intermediate_size": 512, "mtp_num_hidden_layers": 1,
  "mtp_use_dedicated_embeddings": false, "num_attention_heads": 16, "num_experts": 256,
  "num_experts_per_tok": 8, "num_hidden_layers": 40, "num_key_value_heads": 2, "rms_norm_eps": 1e-06,
  "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                      "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
  "shared_expert_intermediate_size": 512, "tie_word_embeddings": false, "vocab_size": 248320
})";
  const auto t = dgpp::minijson::parse(json);
  const auto q = dgpp::minijson::parse(quant);
  return Qwen35TextConfig::parse(t.root, &q.root);
}

struct DenseForm {
  bool saved = Qwen35LayerStream::dense_weights_fp8();
  explicit DenseForm(bool fp8) { Qwen35LayerStream::set_dense_weights_fp8(fp8); }
  ~DenseForm() { Qwen35LayerStream::set_dense_weights_fp8(saved); }
};

size_t table_bytes(const Qwen35TextConfig& c, int layer, bool replicated_only) {
  size_t n = 0;
  for (const auto& e : Qwen35LoaderFamily::layer_table(c, layer))
    if (!replicated_only || Qwen35LoaderFamily::digest_included(e)) n += e.nbytes();
  return n;
}

size_t fp8_saving(int64_t n, int64_t k) {
  const size_t bf16 = dgpp::align_up_256(static_cast<size_t>(n * k) * 2);
  const size_t fp8 =
      dgpp::align_up_256(static_cast<size_t>(n * k)) +
      dgpp::align_up_256(static_cast<size_t>(((n + 127) / 128) * ((k + 127) / 128)) * 4);
  return bf16 - fp8;
}

}  // namespace

DGPP_TEST(qwen36_loader_counting_build_reads_the_whole_layer_and_ranks_tile_it) {
  for (const char* quant : {kQuantMixed, kQuantFp8}) {
    const Qwen35TextConfig c = release_config(quant);
    const std::string which = quant == kQuantMixed ? "nvfp4 mixed: " : "fp8: ";
    const int draft = c.mtp_layer();
    require(draft == 40 && Qwen35LoaderFamily::max_layer(c) == 41, which + "the draft layer follows the stack");
    for (int layer : {0, 3, draft}) {
      const uint64_t full = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 1);
      require(full == table_bytes(c, layer, false),
              which + "world 1 reads every byte of layer " + std::to_string(layer) + "'s table");
      const uint64_t r0 = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 2);
      const uint64_t r1 = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 1, 2);
      require(r0 == r1, which + "the ranks of world 2 read alike");
      require(r0 + r1 == full + table_bytes(c, layer, true),
              which + "world 2 tiles layer " + std::to_string(layer));
    }
    const size_t gdn = Qwen35LayerStream::layer_bytes(c, 0), attn = Qwen35LayerStream::layer_bytes(c, 3);
    for (int l = 0; l < c.num_hidden_layers; ++l)
      require(Qwen35LayerStream::layer_bytes(c, l) == (c.layers[l] == dgpp::Qwen35LayerKind::Gdn ? gdn : attn),
              which + "layer " + std::to_string(l) + " bytes follow its kind");
    const size_t globals = Qwen35LayerStream::globals_bytes(c);
    require(Qwen35LayerStream::resident_bytes(c, 0, 1, dgpp::LoaderHeadSharding::Full, true) ==
                30 * gdn + 10 * attn + Qwen35LayerStream::layer_bytes(c, draft) + globals,
            which + "resident bytes are the layers, the draft layer and the globals");
    // The head is resident in BF16 under both releases (the NVFP4 one is
    // dequantized at load), beside the embedding, the norm and the draft head.
    const size_t V = 248320, H = 2048;
    require(globals == 2 * dgpp::align_up_256(V * H * 2) + 4 * dgpp::align_up_256(H * 2) +
                           dgpp::align_up_256(H * 2 * H * 2),
            which + "globals bytes");
  }
}

DGPP_TEST(qwen36_loader_replicated_set) {
  const Qwen35TextConfig c = release_config(kQuantMixed);
  const auto replicated = [&](const Qwen35TextConfig& cfg, int layer, const std::string& suffix) {
    const std::string name = dgpp::qwen35_layer_prefix(cfg, layer) + suffix;
    for (const auto& e : Qwen35LoaderFamily::layer_table(cfg, layer))
      if (e.name == name) return Qwen35LoaderFamily::digest_included(e);
    throw std::runtime_error("no tensor " + name);
  };
  // The per-tensor FP8 set: the payload slices, its scale and the activation
  // scale are read by every rank.
  require(replicated(c, 0, "linear_attn.in_proj_qkv.weight_scale") &&
              replicated(c, 0, "linear_attn.in_proj_qkv.input_scale") &&
              replicated(c, 3, "self_attn.o_proj.weight_scale") &&
              replicated(c, 3, "mlp.experts.7.down_proj.weight_scale_2") &&
              replicated(c, 3, "mlp.shared_expert.up_proj.input_scale"),
          "the scalars");
  require(replicated(c, 0, "mlp.gate.weight") && replicated(c, 0, "linear_attn.norm.weight") &&
              replicated(c, 3, "self_attn.q_norm.weight"),
          "router and norms");
  require(!replicated(c, 0, "linear_attn.in_proj_qkv.weight") && !replicated(c, 0, "linear_attn.in_proj_a.weight") &&
              !replicated(c, 3, "self_attn.q_proj.weight") && !replicated(c, 3, "mlp.experts.0.gate_proj.weight") &&
              !replicated(c, 3, "mlp.experts.0.gate_proj.weight_scale") &&
              !replicated(c, c.mtp_layer(), "mlp.experts.gate_up_proj") &&
              !replicated(c, c.mtp_layer(), "mlp.experts.down_proj") &&
              !replicated(c, c.mtp_layer(), "self_attn.q_proj.weight"),
          "the slices: projections, experts, the draft layer's stacked experts");
  const Qwen35TextConfig f = release_config(kQuantFp8);
  require(!replicated(f, 3, "mlp.experts.0.gate_proj.weight") &&
              !replicated(f, 3, "mlp.experts.0.gate_proj.weight_scale_inv") &&
              !replicated(f, 3, "mlp.shared_expert.down_proj.weight_scale_inv"),
          "the FP8 release's block scales slice with their payloads");
}

DGPP_TEST(qwen36_loader_dense_knob_moves_only_what_ships_outside_fp8) {
  const int64_t H = 2048, S = 512, d = 256;
  {
    // The FP8 release: block FP8 throughout — the same bytes either way.
    const Qwen35TextConfig c = release_config(kQuantFp8);
    size_t before[3];
    for (int i = 0; i < 3; ++i) before[i] = Qwen35LayerStream::layer_bytes(c, i == 0 ? 0 : (i == 1 ? 3 : 40));
    const DenseForm fp8(true);
    for (int i = 0; i < 3; ++i)
      require(Qwen35LayerStream::layer_bytes(c, i == 0 ? 0 : (i == 1 ? 3 : 40)) == before[i],
              "the FP8 release's layers are the same bytes under the knob");
  }
  {
    // The NVFP4 mixed release: the backbone's projections ship FP8, so only
    // the shared expert moves there; the BF16 draft layer's attention and
    // shared expert move too, its experts are block FP8 either way.
    const Qwen35TextConfig c = release_config(kQuantMixed);
    const size_t shared = 2 * fp8_saving(S, H) + fp8_saving(H, S);
    const size_t gdn = Qwen35LayerStream::layer_bytes(c, 0), attn = Qwen35LayerStream::layer_bytes(c, 3);
    const size_t draft = Qwen35LayerStream::layer_bytes(c, 40);
    const DenseForm fp8(true);
    require(Qwen35LayerStream::layer_bytes(c, 0) == gdn - shared, "a GDN layer: the shared expert alone");
    require(Qwen35LayerStream::layer_bytes(c, 3) == attn - shared, "an attention layer: the shared expert alone");
    require(Qwen35LayerStream::layer_bytes(c, 40) ==
                draft - shared - fp8_saving(2 * 16 * d, H) - 2 * fp8_saving(2 * d, H) - fp8_saving(H, 16 * d),
            "the draft layer: its attention and its shared expert");
    for (int layer : {0, 3, 40})
      require(Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 1) ==
                  [&] {
                    size_t n = 0;
                    for (const auto& e : Qwen35LoaderFamily::layer_table(c, layer)) n += e.nbytes();
                    return n;
                  }(),
              "the FP8 form reads the same checkpoint bytes");
  }
}
