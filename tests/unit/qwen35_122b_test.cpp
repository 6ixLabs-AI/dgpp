// Qwen3.5-122B-A10B on the qwen3_5 stack (Qwen3_5MoeForConditionalGeneration,
// hidden 3072, 48 layers, 256 experts top-8): three releases bind, each to
// the tensor count of its headers less the 333 vision tensors —
//   nvidia/Qwen3.5-122B-A10B-NVFP4 @ 98915d83  149,309 tensors, 148,976 in the table
//     (modelopt NVFP4 on the routed experts alone, everything else BF16);
//   Sehyo/Qwen3.5-122B-A10B-NVFP4 @ 56a6bdda   149,885 tensors, 149,552 in the table
//     (compressed-tensors NVFP4 on the attention, the shared expert and the
//     routed experts; the GDN and the head BF16);
//   Qwen/Qwen3.5-122B-A10B-FP8 @ a099dee7       76,656 tensors,  76,323 in the table
//     (block FP8 throughout; A_log and the GDN norm weight stored F32);
// Intel's AutoRound int4 release is refused by name. The config parser holds
// each NVFP4 recipe's ignore list to what the table then expects; the
// counting build reads every byte of a layer and the ranks of a world tile it.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/loader35.hpp"

namespace {

using dgpp::DType;
using dgpp::Qwen35LayerStream;
using dgpp::Qwen35LoaderFamily;
using dgpp::Qwen35QuantKind;
using dgpp::Qwen35TextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

constexpr int kLayers = 48;
bool full_layer(int i) { return (i + 1) % 4 == 0; }

// The releases' text_config, transcribed (nvidia's; the FP8 release's leaves
// tie_word_embeddings out: with_tie false).
std::string text_json(bool with_tie = true, const char* model_type = "qwen3_5_moe_text") {
  std::string layers;
  for (int i = 0; i < kLayers; ++i) {
    if (i) layers += ", ";
    layers += full_layer(i) ? "\"full_attention\"" : "\"linear_attention\"";
  }
  return std::string(R"({
  "attention_bias": false, "attention_dropout": 0.0, "attn_output_gate": true, "bos_token_id": null,
  "dtype": "bfloat16", "eos_token_id": 248044, "full_attention_interval": 4, "head_dim": 256,
  "hidden_act": "silu", "hidden_size": 3072, "initializer_range": 0.02,
  "layer_types": [)") + layers + R"(],
  "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
  "linear_num_value_heads": 64, "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
  "max_position_embeddings": 262144, "mlp_only_layers": [], "model_type": ")" + model_type + R"(",
  "moe_intermediate_size": 1024, "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
  "num_attention_heads": 32, "num_experts": 256, "num_experts_per_tok": 8, "num_hidden_layers": 48,
  "num_key_value_heads": 2, "output_router_logits": false, "pad_token_id": null,
  "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06,
  "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                      "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
  "router_aux_loss_coef": 0.001, "shared_expert_intermediate_size": 1024,)" +
         (with_tie ? R"( "tie_word_embeddings": false,)" : "") + R"(
  "use_cache": true, "vocab_size": 248320
})";
}

// nvidia's quantization_config: one 4-bit group and modelopt's ignore
// patterns, written as the release writes them. `drop` leaves one entry out.
std::string modelopt_quant(const std::string& drop = "", const char* algo = "NVFP4") {
  std::string ignore = "\"lm_head\"";
  const auto add = [&](const std::string& entry) {
    if (entry != drop) ignore += ", \"" + entry + "\"";
  };
  if (drop == "lm_head") ignore = "\"model.visual*\"";
  for (int i = 0; i < kLayers; ++i) {
    const std::string lp = "model.language_model.layers." + std::to_string(i) + ".";
    add(lp + (full_layer(i) ? "self_attn*" : "linear_attn*"));
    add(lp + "mlp.shared_expert*");
    add(lp + "mlp.shared_expert_gate");
  }
  add("model.visual*");
  add("mtp.layers.0*");
  add("mtp*");
  return std::string(R"({"config_groups": {"group_0": {
    "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
    "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
    "targets": ["Linear"]}},
  "ignore": [)") + ignore + R"(], "quant_algo": ")" + algo + R"(",
  "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
  "producer": {"name": "modelopt", "version": "0.0.1.dev752+gbd4fc3a65.d20260504"},
  "quant_method": "modelopt"})";
}

// Sehyo's: compressed-tensors, the ignore list expanded to module names.
std::string packed_quant(const std::string& drop = "") {
  std::string ignore = "\"model.fc\"";
  const auto add = [&](const std::string& entry) {
    if (entry != drop) ignore += ", \"" + entry + "\"";
  };
  add("lm_head");
  for (int i = 0; i < kLayers; ++i) {
    const std::string lp = "model.language_model.layers." + std::to_string(i) + ".";
    if (!full_layer(i))
      for (const char* m : {"in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"})
        add(lp + "linear_attn." + m);
    add(lp + "mlp.gate");
    add(lp + "mlp.shared_expert_gate");
  }
  add("mtp.fc");
  return std::string(R"({"config_groups": {"group_0": {"format": "nvfp4-pack-quantized",
    "input_activations": {"dynamic": "local", "group_size": 16, "num_bits": 4, "strategy": "tensor_group",
                          "symmetric": true, "type": "float"},
    "output_activations": null, "targets": ["Linear"],
    "weights": {"dynamic": false, "group_size": 16, "num_bits": 4, "observer": "memoryless_minmax",
                "strategy": "tensor_group", "symmetric": true, "type": "float"}}},
  "format": "nvfp4-pack-quantized", "global_compression_ratio": null,
  "ignore": [)") + ignore + R"(],
  "kv_cache_scheme": null, "quant_method": "compressed-tensors", "quantization_status": "compressed",
  "sparsity_config": {}, "transform_config": {}, "version": "0.13.1.a20260212"})";
}

const char* kQuantFp8 = R"({"quant_method": "fp8", "activation_scheme": "dynamic", "weight_per_tensor": false,
  "act_per_tensor": false, "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head"]})";

Qwen35TextConfig parse(const std::string& text, const std::string& quant) {
  const auto t = dgpp::minijson::parse(text);
  const auto q = dgpp::minijson::parse(quant);
  return Qwen35TextConfig::parse(t.root, &q.root);
}

std::string refusal(const std::string& text, const std::string& quant) {
  try {
    (void)parse(text, quant);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

struct Table {
  std::vector<dgpp::QwenExpectedTensor> all;
  std::unordered_map<std::string, const dgpp::QwenExpectedTensor*> by_name;
  explicit Table(const Qwen35TextConfig& c) : all(dgpp::qwen35_expected_text_tensors(c)) {
    for (const auto& e : all) by_name.emplace(e.name, &e);
  }
  bool is(const std::string& name, DType dt, std::vector<int64_t> shape) const {
    const auto it = by_name.find(name);
    return it != by_name.end() && it->second->dtype == dt && it->second->shape == shape;
  }
  // The table against itself beside a vision tower of 333 tensors.
  dgpp::QwenBindReport bind(const Qwen35TextConfig& c, bool f32_vectors) const {
    std::unordered_map<std::string, dgpp::QwenTensorDesc> present;
    for (const auto& e : all)
      present.emplace(e.name,
                      dgpp::QwenTensorDesc{f32_vectors && e.role == dgpp::QwenTensorRole::Bf16OrF32 ? DType::F32
                                                                                                    : e.dtype,
                                           e.shape});
    for (int i = 0; i < 333; ++i)
      present.emplace("model.visual.blocks." + std::to_string(i) + ".norm1.weight",
                      dgpp::QwenTensorDesc{DType::BF16, {1152}});
    return dgpp::qwen35_validate_text_binding(c, present);
  }
};

const std::string L0 = "model.language_model.layers.0.", L3 = "model.language_model.layers.3.";
const std::string M0 = "mtp.layers.0.";

size_t table_bytes(const Qwen35TextConfig& c, int layer, bool replicated_only) {
  size_t n = 0;
  for (const auto& e : Qwen35LoaderFamily::layer_table(c, layer))
    if (!replicated_only || Qwen35LoaderFamily::digest_included(e)) n += e.nbytes();
  return n;
}

}  // namespace

DGPP_TEST(qwen35_122b_config_parses_the_three_releases) {
  const Qwen35TextConfig n = parse(text_json(), modelopt_quant());
  require(n.quant_kind == Qwen35QuantKind::Nvfp4Experts, "nvidia: modelopt NVFP4 on the experts alone");
  require(!n.next() && n.moe() && n.hidden_size == 3072 && n.num_hidden_layers == 48 &&
              n.num_gdn_layers() == 36 && n.num_full_layers() == 12 && n.mtp_layer() == 48,
          "shape");
  require(n.num_experts == 256 && n.num_experts_per_tok == 8 && n.moe_intermediate_size == 1024 &&
              n.shared_expert_intermediate_size == 1024,
          "the routed MoE");
  require(n.num_attention_heads == 32 && n.num_key_value_heads == 2 && n.head_dim == 256 &&
              n.rotary_dim == 64 && n.gdn_key_heads == 16 && n.gdn_value_heads == 64,
          "attention and GDN geometry");
  require(n.bos_token_id == -1 && n.eos_token_ids.size() == 1 && n.eos_token_ids[0] == 248044 &&
              !n.tie_word_embeddings,
          "tokens");
  require(parse(text_json(), packed_quant()).quant_kind == Qwen35QuantKind::Nvfp4Packed,
          "Sehyo: compressed-tensors NVFP4");
  // The FP8 release's text_config has no tie_word_embeddings: untied.
  const Qwen35TextConfig f = parse(text_json(false), kQuantFp8);
  require(f.quant_kind == Qwen35QuantKind::Fp8Block && !f.tie_word_embeddings, "Qwen: block FP8, untied");
  // (The class's architecture detection is the 35B's: qwen36_config_test.cpp.)
}

DGPP_TEST(qwen35_122b_refusals_name_the_field) {
  // Each NVFP4 recipe is held to its ignore list, by the entry it lacks.
  for (const char* entry : {"lm_head", "model.language_model.layers.5.mlp.shared_expert*",
                            "model.language_model.layers.0.linear_attn*",
                            "model.language_model.layers.47.self_attn*", "mtp*"}) {
    const std::string why = refusal(text_json(), modelopt_quant(entry));
    require(why.find(std::string("quantization_config.ignore: '") + entry + "' missing") != std::string::npos &&
                why.find("nvidia/Qwen3.5-122B-A10B-NVFP4") != std::string::npos,
            std::string("modelopt recipe without ") + entry + ": " + why);
  }
  for (const char* entry : {"lm_head", "model.language_model.layers.0.linear_attn.in_proj_qkv",
                            "model.language_model.layers.46.linear_attn.out_proj",
                            "model.language_model.layers.9.linear_attn.in_proj_z"}) {
    const std::string why = refusal(text_json(), packed_quant(entry));
    require(why.find(std::string("quantization_config.ignore: '") + entry + "' missing") != std::string::npos,
            std::string("packed recipe without ") + entry + ": " + why);
  }
  require(refusal(text_json(), modelopt_quant("", "W4A8_NVFP4_FP8")).find("quant_algo") != std::string::npos,
          "another modelopt algorithm");
  // The compressed-tensors recipe is held to the checks of the Qwen3Next dialect's.
  std::string q = packed_quant();
  q.replace(q.find("\"kv_cache_scheme\": null"), 23, "\"kv_cache_scheme\": {\"num_bits\": 8}");
  require(refusal(text_json(), q).find("kv_cache_scheme must be null") != std::string::npos, "a cache scheme");
  // Intel's AutoRound int4 release.
  const std::string intel = refusal(
      text_json(), R"({"bits": 4, "data_type": "int", "group_size": 128, "packing_format": "auto_round:auto_gptq",
                       "quant_method": "auto-round", "sym": true})");
  require(intel.find("auto-round") != std::string::npos && intel.find("not implemented") != std::string::npos,
          "AutoRound is refused by name: " + intel);
  // A dense model in a single-group NVFP4 recipe (Kbenkhaled/Qwen3.5-27B-NVFP4) has no path.
  std::string dense = text_json(true, "qwen3_5_text");
  dense.replace(dense.find("\"num_experts\": 256,"), 19, "\"intermediate_size\": 17408,");
  require(refusal(dense, packed_quant()).find("routed-MoE models") != std::string::npos, "dense + single group");
}

DGPP_TEST(qwen35_122b_nvfp4_experts_binding_table_is_the_checkpoint) {
  const Qwen35TextConfig c = parse(text_json(), modelopt_quant());
  const Table t(c);
  // 36 GDN layers x (11 + 5 + 256 x 12) + 12 attention layers x (8 + 5 + 3072)
  // + the draft layer's (8 + 5 + 768) + 7 globals.
  require(t.all.size() == 148976, "table size " + std::to_string(t.all.size()));
  require(dgpp::qwen35_expected_layer_tensors(c, 0).size() == 11 + 5 + 3072, "gdn layer");
  require(dgpp::qwen35_expected_layer_tensors(c, 3).size() == 8 + 5 + 3072, "attention layer");
  require(dgpp::qwen35_expected_layer_tensors(c, 48).size() == 8 + 5 + 768, "draft layer");
  require(dgpp::qwen35_expected_global_tensors(c).size() == 7, "globals");
  // Everything but the routed experts is BF16.
  require(t.is(L0 + "linear_attn.in_proj_qkv.weight", DType::BF16, {12288, 3072}) &&
              t.is(L0 + "linear_attn.in_proj_z.weight", DType::BF16, {8192, 3072}) &&
              t.is(L0 + "linear_attn.out_proj.weight", DType::BF16, {3072, 8192}) &&
              t.is(L0 + "linear_attn.conv1d.weight", DType::BF16, {12288, 1, 4}) &&
              t.is(L0 + "linear_attn.A_log", DType::BF16, {64}),
          "the GDN is BF16");
  require(t.is(L3 + "self_attn.q_proj.weight", DType::BF16, {16384, 3072}) &&
              t.is(L3 + "self_attn.k_proj.weight", DType::BF16, {512, 3072}) &&
              t.is(L3 + "self_attn.o_proj.weight", DType::BF16, {3072, 8192}) &&
              t.by_name.count(L3 + "self_attn.k_proj.k_scale") == 0,
          "the attention is BF16, no cache scales");
  require(t.is(L0 + "mlp.gate.weight", DType::BF16, {256, 3072}) &&
              t.is(L0 + "mlp.shared_expert.gate_proj.weight", DType::BF16, {1024, 3072}) &&
              t.is(L0 + "mlp.shared_expert.down_proj.weight", DType::BF16, {3072, 1024}) &&
              t.by_name.count(L0 + "mlp.shared_expert.gate_proj.weight_scale") == 0,
          "the router and the shared expert are BF16");
  require(t.is(L0 + "mlp.experts.255.gate_proj.weight", DType::U8, {1024, 1536}) &&
              t.is(L0 + "mlp.experts.255.gate_proj.weight_scale", DType::F8_E4M3, {1024, 192}) &&
              t.is(L0 + "mlp.experts.255.gate_proj.weight_scale_2", DType::F32, {}) &&
              t.is(L0 + "mlp.experts.255.gate_proj.input_scale", DType::F32, {}) &&
              t.is(L3 + "mlp.experts.0.down_proj.weight", DType::U8, {3072, 512}) &&
              t.is(L3 + "mlp.experts.0.down_proj.weight_scale", DType::F8_E4M3, {3072, 64}),
          "the routed experts are the modelopt NVFP4 set");
  require(t.is(M0 + "mlp.experts.7.up_proj.weight", DType::BF16, {1024, 3072}) &&
              t.is(M0 + "mlp.experts.7.down_proj.weight", DType::BF16, {3072, 1024}) &&
              t.by_name.count(M0 + "mlp.experts.gate_up_proj") == 0 &&
              t.is(M0 + "self_attn.q_proj.weight", DType::BF16, {16384, 3072}),
          "the draft layer is BF16 with per-expert matrices");
  require(t.is("lm_head.weight", DType::BF16, {248320, 3072}) && t.is("mtp.fc.weight", DType::BF16, {3072, 6144}),
          "globals");
  const dgpp::QwenBindReport rep = t.bind(c, false);
  require(rep.ok() && rep.matched == 148976 && rep.vision == 333 && rep.quantized_matrices == 36864,
          "exact binding: 48 x 256 x 3 NVFP4 matrices");
}

DGPP_TEST(qwen35_122b_nvfp4_packed_binding_table_is_the_checkpoint) {
  const Qwen35TextConfig c = parse(text_json(), packed_quant());
  const Table t(c);
  require(t.all.size() == 149552, "table size " + std::to_string(t.all.size()));
  require(t.is(L0 + "linear_attn.in_proj_qkv.weight", DType::BF16, {12288, 3072}) &&
              t.is(L0 + "linear_attn.out_proj.weight", DType::BF16, {3072, 8192}),
          "the GDN is BF16");
  require(t.is(L3 + "self_attn.q_proj.weight_packed", DType::U8, {16384, 1536}) &&
              t.is(L3 + "self_attn.q_proj.weight_scale", DType::F8_E4M3, {16384, 192}) &&
              t.is(L3 + "self_attn.q_proj.weight_global_scale", DType::F32, {1}) &&
              t.is(L3 + "self_attn.q_proj.input_global_scale", DType::F32, {1}) &&
              t.is(L3 + "self_attn.o_proj.weight_packed", DType::U8, {3072, 4096}) &&
              t.by_name.count(L3 + "self_attn.q_proj.weight") == 0,
          "the attention is the compressed-tensors set");
  require(t.is(L0 + "mlp.shared_expert.down_proj.weight_packed", DType::U8, {3072, 512}) &&
              t.is(L0 + "mlp.experts.3.up_proj.weight_packed", DType::U8, {1024, 1536}) &&
              t.is(L0 + "mlp.experts.3.up_proj.weight_global_scale", DType::F32, {1}),
          "the shared expert and the routed experts alike");
  require(t.is(M0 + "self_attn.q_proj.weight", DType::BF16, {16384, 3072}) &&
              t.is(M0 + "mlp.shared_expert.up_proj.weight", DType::BF16, {1024, 3072}) &&
              t.is(M0 + "mlp.experts.255.gate_proj.weight", DType::BF16, {1024, 3072}),
          "the draft layer is BF16");
  const dgpp::QwenBindReport rep = t.bind(c, false);
  require(rep.ok() && rep.matched == 149552 && rep.vision == 333 && rep.quantized_matrices == 37056,
          "exact binding: 12 x 4 attention + 48 x 3 shared + 48 x 768 expert matrices");
}

DGPP_TEST(qwen35_122b_fp8_binding_table_is_the_checkpoint) {
  const Qwen35TextConfig c = parse(text_json(false), kQuantFp8);
  const Table t(c);
  require(t.all.size() == 76323, "table size " + std::to_string(t.all.size()));
  require(t.is(L0 + "linear_attn.in_proj_qkv.weight", DType::F8_E4M3, {12288, 3072}) &&
              t.is(L0 + "linear_attn.in_proj_qkv.weight_scale_inv", DType::BF16, {96, 24}) &&
              t.is(L0 + "mlp.experts.0.gate_proj.weight", DType::F8_E4M3, {1024, 3072}) &&
              t.is(L0 + "mlp.experts.0.gate_proj.weight_scale_inv", DType::BF16, {8, 24}) &&
              t.is(M0 + "mlp.experts.0.down_proj.weight_scale_inv", DType::BF16, {24, 8}),
          "block FP8 throughout, the draft layer included");
  // This release stores A_log and the GDN norm weight F32: bound.
  const dgpp::QwenBindReport rep = t.bind(c, true);
  require(rep.ok() && rep.matched == 76323 && rep.vision == 333 && rep.quantized_matrices == 37939,
          "exact binding with F32 GDN vectors");
  require(t.bind(c, false).ok(), "and with BF16 ones, as the 35B's FP8 release stores them");
}

DGPP_TEST(qwen35_122b_loader_counting_build_reads_the_whole_layer_and_ranks_tile_it) {
  for (int which = 0; which < 3; ++which) {
    const Qwen35TextConfig c = which == 0   ? parse(text_json(), modelopt_quant())
                               : which == 1 ? parse(text_json(), packed_quant())
                                            : parse(text_json(false), kQuantFp8);
    const std::string name = which == 0 ? "nvfp4 experts: " : which == 1 ? "nvfp4 packed: " : "fp8: ";
    require(Qwen35LoaderFamily::max_layer(c) == 49, name + "the draft layer follows the stack");
    for (int layer : {0, 3, 48}) {
      const uint64_t full = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 1);
      require(full == table_bytes(c, layer, false),
              name + "world 1 reads every byte of layer " + std::to_string(layer) + "'s table");
      const uint64_t r0 = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 2);
      const uint64_t r1 = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 1, 2);
      require(r0 == r1 && r0 + r1 == full + table_bytes(c, layer, true),
              name + "world 2 tiles layer " + std::to_string(layer));
    }
    const size_t gdn = Qwen35LayerStream::layer_bytes(c, 0), attn = Qwen35LayerStream::layer_bytes(c, 3);
    for (int l = 0; l < c.num_hidden_layers; ++l)
      require(Qwen35LayerStream::layer_bytes(c, l) == (full_layer(l) ? attn : gdn),
              name + "layer " + std::to_string(l) + " bytes follow its kind");
    const size_t V = 248320, H = 3072;
    require(Qwen35LayerStream::globals_bytes(c) == 2 * dgpp::align_up_256(V * H * 2) +
                                                       4 * dgpp::align_up_256(H * 2) +
                                                       dgpp::align_up_256(H * 2 * H * 2),
            name + "globals: the embedding, a BF16 head, the norm and the draft head");
    require(Qwen35LayerStream::resident_bytes(c, 0, 1, dgpp::LoaderHeadSharding::Full, true) ==
                36 * gdn + 12 * attn + Qwen35LayerStream::layer_bytes(c, 48) + Qwen35LayerStream::globals_bytes(c),
            name + "resident bytes are the layers, the draft layer and the globals");
  }
  // The two NVFP4 containers hold the same matrices — BF16 for everything
  // but the routed experts (the compressed-tensors attention and shared
  // expert are dequantized into the BF16 the modelopt release ships): their
  // layers differ by the containers' scalars alone (the static activation
  // scales of one, the dequantized matrices' input scales of the other).
  const Qwen35TextConfig a = parse(text_json(), modelopt_quant()), b = parse(text_json(), packed_quant());
  for (int layer : {0, 3}) {
    const size_t la = Qwen35LayerStream::layer_bytes(a, layer), lb = Qwen35LayerStream::layer_bytes(b, layer);
    require((la > lb ? la - lb : lb - la) < 4096, "the two NVFP4 containers plan the same matrices");
  }
  require(Qwen35LayerStream::layer_bytes(a, 48) == Qwen35LayerStream::layer_bytes(b, 48),
          "and the same draft layer");
  // 70 GiB of weights fit a 121 GiB Spark beside a K/V pool; the FP8 release's 117 GiB do not.
  const double G = 1024.0 * 1024.0 * 1024.0;
  const double nv = Qwen35LayerStream::resident_bytes(a, 0, 1, dgpp::LoaderHeadSharding::Full, true) / G;
  const Qwen35TextConfig f = parse(text_json(false), kQuantFp8);
  const double fp8 = Qwen35LayerStream::resident_bytes(f, 0, 1, dgpp::LoaderHeadSharding::Full, true) / G;
  require(nv > 74.0 && nv < 75.5, "NVFP4 with BF16 dense matrices: about 74.7 GiB, got " + std::to_string(nv));
  require(fp8 > 117.0 && fp8 < 118.0, "FP8: about 117.6 GiB, got " + std::to_string(fp8));
}
