// Qwen3.6-35B-A3B on the qwen3_5 stack: the Qwen3.5 dialect (nested
// text_config, `model.language_model.*`, split GDN projections) with the
// routed MoE in the dense MLP's place. Its text_config parses under both
// releases' recipes, the unsupported shapes are refused by name, and the
// expected-tensor tables are the checkpoints' — counted from their
// safetensors headers:
//   nvidia/Qwen3.6-35B-A3B-NVFP4 @ 1355db6a: 124,468 tensors, 333 of them
//     the vision tower's (skipped), 124,135 in the table;
//   Qwen/Qwen3.6-35B-A3B-FP8 @ 95a723d0: 64,196 tensors, 333 vision,
//     63,863 in the table.
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::string layers_json(int n) {
  std::string s;
  for (int i = 0; i < n; ++i) {
    if (i) s += ", ";
    s += (i + 1) % 4 == 0 ? "\"full_attention\"" : "\"linear_attention\"";
  }
  return s;
}

// The releases' text_config, transcribed (both carry the same one).
std::string text_json() {
  return std::string(R"({
  "attention_bias": false, "attention_dropout": 0.0, "attn_output_gate": true,
  "bos_token_id": 248044, "dtype": "bfloat16", "eos_token_id": 248044,
  "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu", "hidden_size": 2048,
  "initializer_range": 0.02,
  "layer_types": [)") + layers_json(40) + R"(],
  "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
  "linear_num_value_heads": 32, "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
  "max_position_embeddings": 262144, "model_type": "qwen3_5_moe_text",
  "moe_intermediate_size": 512, "mtp_num_hidden_layers": 1,
  "mtp_use_dedicated_embeddings": false, "num_attention_heads": 16, "num_experts": 256,
  "num_experts_per_tok": 8, "num_hidden_layers": 40, "num_key_value_heads": 2,
  "output_router_logits": false, "pad_token_id": null, "partial_rotary_factor": 0.25,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                      "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
  "router_aux_loss_coef": 0.001, "shared_expert_intermediate_size": 512,
  "tie_word_embeddings": false, "use_cache": true, "vocab_size": 248320
})";
}

// nvidia's quantization_config (modelopt MIXED_PRECISION), its 291 targets
// and quantized_layers entries cut to one of each class.
const char* kQuantMixed = R"({
  "config_groups": {
    "group_0": {"input_activations": {"dynamic": false, "num_bits": 8, "type": "float"},
                "weights": {"dynamic": false, "num_bits": 8, "type": "float"},
                "targets": ["model.language_model.layers.0.linear_attn.in_proj_qkv",
                            "model.language_model.layers.3.self_attn.q_proj"]},
    "group_1": {"input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                "targets": ["lm_head", "model.language_model.layers.0.mlp.experts",
                            "model.language_model.layers.0.mlp.shared_expert.gate_proj"]}},
  "quantized_layers": {"lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
                       "model.language_model.layers.0.linear_attn.out_proj": {"quant_algo": "FP8"}},
  "ignore": ["mtp.layers.0*", "mtp*"], "quant_algo": "MIXED_PRECISION",
  "producer": {"name": "modelopt", "version": "0.37.0"}, "quant_method": "modelopt"})";

// Qwen's FP8 release (its modules_to_not_convert list cut to one).
const char* kQuantFp8 = R"({"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
  "modules_to_not_convert": ["model.visual.blocks.0.attn.proj"], "weight_block_size": [128, 128]})";

dgpp::Qwen35TextConfig parse(const std::string& text, const char* quant) {
  const auto t = dgpp::minijson::parse(text);
  if (quant == nullptr) return dgpp::Qwen35TextConfig::parse(t.root, nullptr);
  const auto q = dgpp::minijson::parse(quant);
  return dgpp::Qwen35TextConfig::parse(t.root, &q.root);
}

std::string refusal(const std::string& text, const char* quant) {
  try {
    (void)parse(text, quant);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

std::string patched(const std::string& from, const std::string& to) {
  std::string s = text_json();
  const size_t at = s.find(from);
  require(at != std::string::npos, "patch anchor missing: " + from);
  s.replace(at, from.size(), to);
  return s;
}

std::unordered_map<std::string, dgpp::QwenTensorDesc> present_from(const dgpp::Qwen35TextConfig& c) {
  std::unordered_map<std::string, dgpp::QwenTensorDesc> present;
  for (const auto& e : dgpp::qwen35_expected_text_tensors(c))
    present.emplace(e.name, dgpp::QwenTensorDesc{e.dtype, e.shape});
  return present;
}

}  // namespace

DGPP_TEST(qwen36_config_parses_both_releases) {
  for (const bool mixed : {true, false}) {
    const dgpp::Qwen35TextConfig c = parse(text_json(), mixed ? kQuantMixed : kQuantFp8);
    require(!c.next() && c.moe(), "the Qwen3.5 dialect with a routed MoE");
    require(c.quant_kind == (mixed ? dgpp::Qwen35QuantKind::Nvfp4Mixed : dgpp::Qwen35QuantKind::Fp8Block),
            "the recipe");
    require(c.hidden_size == 2048 && c.vocab_size == 248320 && c.num_hidden_layers == 40, "shape");
    require(c.num_gdn_layers() == 30 && c.num_full_layers() == 10, "layer kinds");
    require(c.gdn_key_heads == 16 && c.gdn_value_heads == 32 && c.output_gate_type == "swish", "gdn");
    require(c.num_attention_heads == 16 && c.num_key_value_heads == 2 && c.head_dim == 256, "attention");
    require(c.rotary_dim == 64 && c.rope_theta == 1e7 && c.mrope_section == std::vector<int>({11, 11, 10}),
            "rope");
    require(c.num_experts == 256 && c.num_experts_per_tok == 8 && c.norm_topk_prob, "experts");
    require(c.moe_intermediate_size == 512 && c.shared_expert_intermediate_size == 512, "moe widths");
    require(c.intermediate_size == 0, "no dense MLP");
    require(c.mtp_layer() == 40, "the draft layer");
    require(!c.tie_word_embeddings && c.eos_token_ids == std::vector<int64_t>({248044}), "head and eos");
  }
}

DGPP_TEST(qwen36_config_refusals_name_the_field) {
  const auto names = [](const std::string& why, const char* field) {
    return why.find("Qwen3.5") != std::string::npos && why.find(field) != std::string::npos;
  };
  require(names(refusal(patched("\"qwen3_5_moe_text\"", "\"qwen4_exp_text\""), kQuantFp8), "model_type"),
          "model_type");
  // The MoE text config must carry the MoE's fields, and hold them to the
  // router the engine implements.
  require(names(refusal(patched("\"num_experts\": 256,", ""), kQuantFp8), "num_experts"), "experts missing");
  require(names(refusal(patched("\"num_experts_per_tok\": 8", "\"num_experts_per_tok\": 32"), kQuantFp8),
                "num_experts_per_tok"),
          "top-k bound");
  require(names(refusal(patched("\"moe_intermediate_size\": 512", "\"moe_intermediate_size\": 500"), kQuantFp8),
                "moe_intermediate_size"),
          "moe width");
  require(names(refusal(patched("\"use_cache\": true", "\"use_cache\": true, \"norm_topk_prob\": false"),
                        kQuantFp8),
                "norm_topk_prob"),
          "top-k renormalization");
  require(names(refusal(patched("\"use_cache\": true", "\"use_cache\": true, \"decoder_sparse_step\": 2"),
                        kQuantFp8),
                "decoder_sparse_step"),
          "sparse step");
  require(names(refusal(patched("\"use_cache\": true", "\"use_cache\": true, \"mlp_only_layers\": [0]"),
                        kQuantFp8),
                "mlp_only_layers"),
          "dense layers");
  // A dense text config that names experts is not silently a dense model.
  require(names(refusal(patched("\"qwen3_5_moe_text\"", "\"qwen3_5_text\", \"intermediate_size\": 4096"),
                        kQuantFp8),
                "num_experts"),
          "experts under the dense type");
  // The mixed recipe: group_1 NVFP4 beside an 8-bit float group_0.
  require(refusal(text_json(), R"({"config_groups": {"group_0": {"weights": {"num_bits": 16, "type": "float"}},
      "group_1": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}})")
              .find("group_0") != std::string::npos,
          "group_0 must be FP8");
  require(refusal(text_json(), R"({"config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "float"}},
      "group_1": {"weights": {"num_bits": 4, "type": "float", "group_size": 32}}}})")
              .find("NVFP4") != std::string::npos,
          "group_1 must be NVFP4");
  // RedHatAI's compressed-tensors release of this model (one group,
  // nvfp4-pack-quantized) has no binding on this dialect: refused.
  require(refusal(text_json(), R"({"config_groups": {"group_0": {"format": "nvfp4-pack-quantized",
      "weights": {"num_bits": 4, "type": "float", "group_size": 16}}}, "format": "nvfp4-pack-quantized",
      "quant_method": "compressed-tensors"})")
              .find("group_1") != std::string::npos,
          "the single-group compressed-tensors recipe");
}

DGPP_TEST(qwen36_architecture_is_the_qwen3_5_stack) {
  const auto by_class = dgpp::minijson::parse(
      R"({"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe"})");
  require(dgpp::detect_architecture(by_class.root) == dgpp::ModelArchitecture::Qwen3_5, "by class");
  const auto by_type = dgpp::minijson::parse(R"({"model_type": "qwen3_5_moe"})");
  require(dgpp::detect_architecture(by_type.root) == dgpp::ModelArchitecture::Qwen3_5, "by model_type");
}

DGPP_TEST(qwen36_nvfp4_binding_table_is_the_checkpoint) {
  const dgpp::Qwen35TextConfig c = parse(text_json(), kQuantMixed);
  const auto all = dgpp::qwen35_expected_text_tensors(c);
  // 30 GDN layers x 3103 + 10 attention layers x 3102 + the draft layer's 15
  // + 10 globals: the release's headers less its 333 vision tensors.
  require(all.size() == 124135, "table size " + std::to_string(all.size()));
  require(dgpp::qwen35_expected_layer_tensors(c, 0).size() == 3103, "gdn layer");
  require(dgpp::qwen35_expected_layer_tensors(c, 3).size() == 3102, "attention layer");
  require(dgpp::qwen35_expected_layer_tensors(c, c.mtp_layer()).size() == 15, "draft layer");
  require(dgpp::qwen35_expected_global_tensors(c).size() == 10, "globals");
  require(dgpp::qwen35_layer_prefix(c, 5) == "model.language_model.layers.5.", "nested names");

  std::unordered_map<std::string, const dgpp::QwenExpectedTensor*> by_name;
  for (const auto& e : all) by_name.emplace(e.name, &e);
  auto shape_is = [&](const std::string& name, dgpp::DType dt, std::vector<int64_t> shape) {
    const auto it = by_name.find(name);
    return it != by_name.end() && it->second->dtype == dt && it->second->shape == shape;
  };
  using dgpp::DType;
  const std::string L0 = "model.language_model.layers.0.", L3 = "model.language_model.layers.3.";
  // The GDN: split, head-major projections in per-tensor FP8; a and b BF16.
  require(shape_is(L0 + "linear_attn.in_proj_qkv.weight", DType::F8_E4M3, {8192, 2048}), "qkv codes");
  require(shape_is(L0 + "linear_attn.in_proj_qkv.weight_scale", DType::F32, {}), "one scale per tensor");
  require(shape_is(L0 + "linear_attn.in_proj_qkv.input_scale", DType::F32, {}), "qkv input scale");
  require(shape_is(L0 + "linear_attn.in_proj_z.weight", DType::F8_E4M3, {4096, 2048}), "z");
  require(shape_is(L0 + "linear_attn.out_proj.weight", DType::F8_E4M3, {2048, 4096}), "out");
  require(shape_is(L0 + "linear_attn.in_proj_a.weight", DType::BF16, {32, 2048}), "a");
  require(by_name.count(L0 + "linear_attn.in_proj_qkv.weight_scale_inv") == 0, "no block scales");
  require(by_name.count(L0 + "linear_attn.in_proj_qkvz.weight") == 0, "no fused names");
  // Attention: per-tensor FP8 q/k/v/o.
  require(shape_is(L3 + "self_attn.q_proj.weight", DType::F8_E4M3, {8192, 2048}), "q");
  require(shape_is(L3 + "self_attn.o_proj.weight_scale", DType::F32, {}), "o scale");
  require(by_name.count(L3 + "self_attn.k_proj.k_scale") == 0, "no cache scales");
  // The MoE: modelopt NVFP4 experts and shared expert, BF16 router.
  require(shape_is(L3 + "mlp.gate.weight", DType::BF16, {256, 2048}), "router");
  require(shape_is(L3 + "mlp.experts.255.gate_proj.weight", DType::U8, {512, 1024}), "expert codes");
  require(shape_is(L3 + "mlp.experts.255.down_proj.weight_scale", DType::F8_E4M3, {2048, 32}), "expert scales");
  require(shape_is(L3 + "mlp.experts.0.up_proj.weight_scale_2", DType::F32, {}), "expert global");
  require(shape_is(L3 + "mlp.shared_expert.down_proj.weight", DType::U8, {2048, 256}), "shared codes");
  require(by_name.count(L3 + "mlp.experts.256.gate_proj.weight") == 0, "256 experts");
  // The draft layer: BF16, the experts as the module's two stacked parameters.
  require(shape_is("mtp.layers.0.mlp.experts.gate_up_proj", DType::BF16, {256, 1024, 2048}), "stacked gate|up");
  require(shape_is("mtp.layers.0.mlp.experts.down_proj", DType::BF16, {256, 2048, 512}), "stacked down");
  require(by_name.count("mtp.layers.0.mlp.experts.0.gate_proj.weight") == 0, "no per-expert draft names");
  require(shape_is("mtp.layers.0.self_attn.q_proj.weight", DType::BF16, {8192, 2048}), "draft q");
  require(shape_is("mtp.layers.0.mlp.shared_expert.up_proj.weight", DType::BF16, {512, 2048}), "draft shared");
  // Globals: the NVFP4 head.
  require(shape_is("lm_head.weight", DType::U8, {248320, 1024}), "head codes");
  require(shape_is("lm_head.weight_scale", DType::F8_E4M3, {248320, 128}), "head scales");
  require(shape_is("lm_head.weight_scale_2", DType::F32, {}), "head global");
  require(shape_is("model.language_model.embed_tokens.weight", DType::BF16, {248320, 2048}), "embed");
  require(shape_is("mtp.fc.weight", DType::BF16, {2048, 4096}), "mtp fc");

  // Validation: exact, with the vision tower counted and skipped.
  auto present = present_from(c);
  for (int i = 0; i < 333; ++i)
    present.emplace("model.visual.blocks." + std::to_string(i) + ".norm1.weight",
                    dgpp::QwenTensorDesc{DType::BF16, {1152}});
  dgpp::QwenBindReport rep = dgpp::qwen35_validate_text_binding(c, present);
  require(rep.ok() && rep.matched == 124135 && rep.vision == 333, "exact binding beside the vision tower");
  // 40 layers x (256 x 3 routed + 3 shared) + 30 x 3 GDN + 10 x 4 attention + the head.
  require(rep.quantized_matrices == 40 * 771 + 90 + 40 + 1, "quantized " + std::to_string(rep.quantized_matrices));
  // The FP8 release's tensors do not bind under this recipe.
  rep = dgpp::qwen35_validate_text_binding(c, present_from(parse(text_json(), kQuantFp8)), 4);
  require(!rep.ok() && rep.missing > 0 && rep.unexpected > 0, "the other release's names");
}

DGPP_TEST(qwen36_fp8_binding_table_is_the_checkpoint) {
  const dgpp::Qwen35TextConfig c = parse(text_json(), kQuantFp8);
  const auto all = dgpp::qwen35_expected_text_tensors(c);
  // 30 GDN layers x 1558 + 10 attention layers x 1556 + the draft layer's
  // 1556 + 7 globals: the release's headers less its 333 vision tensors.
  require(all.size() == 63863, "table size " + std::to_string(all.size()));
  require(dgpp::qwen35_expected_layer_tensors(c, 0).size() == 1558, "gdn layer");
  require(dgpp::qwen35_expected_layer_tensors(c, 3).size() == 1556, "attention layer");
  require(dgpp::qwen35_expected_layer_tensors(c, c.mtp_layer()).size() == 1556, "draft layer");
  require(dgpp::qwen35_expected_global_tensors(c).size() == 7, "globals");

  std::unordered_map<std::string, const dgpp::QwenExpectedTensor*> by_name;
  for (const auto& e : all) by_name.emplace(e.name, &e);
  auto shape_is = [&](const std::string& name, dgpp::DType dt, std::vector<int64_t> shape) {
    const auto it = by_name.find(name);
    return it != by_name.end() && it->second->dtype == dt && it->second->shape == shape;
  };
  using dgpp::DType;
  const std::string L3 = "model.language_model.layers.3.";
  require(shape_is(L3 + "mlp.experts.7.gate_proj.weight", DType::F8_E4M3, {512, 2048}), "expert gate");
  require(shape_is(L3 + "mlp.experts.7.gate_proj.weight_scale_inv", DType::BF16, {4, 16}), "expert gate scales");
  require(shape_is(L3 + "mlp.experts.7.down_proj.weight_scale_inv", DType::BF16, {16, 4}), "expert down scales");
  require(shape_is(L3 + "mlp.shared_expert.down_proj.weight", DType::F8_E4M3, {2048, 512}), "shared down");
  require(shape_is(L3 + "self_attn.q_proj.weight_scale_inv", DType::BF16, {64, 16}), "q scales");
  require(shape_is("model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv", DType::BF16,
                   {64, 16}),
          "qkv scales");
  // The draft layer is FP8 and per expert, like the backbone.
  require(shape_is("mtp.layers.0.mlp.experts.255.up_proj.weight", DType::F8_E4M3, {512, 2048}), "draft expert");
  require(shape_is("mtp.layers.0.self_attn.o_proj.weight_scale_inv", DType::BF16, {16, 32}), "draft o scales");
  require(by_name.count("mtp.layers.0.mlp.experts.gate_up_proj") == 0, "no stacked draft experts");
  require(shape_is("lm_head.weight", DType::BF16, {248320, 2048}), "the BF16 head");

  int routed = 0;
  for (const auto& e : dgpp::qwen35_expected_layer_tensors(c, 3))
    if (e.cls == dgpp::QwenWeightClass::RoutedExpert && e.role == dgpp::QwenTensorRole::Fp8Payload) {
      require(e.expert >= 0 && e.expert < 256, "expert id");
      ++routed;
    }
  require(routed == 256 * 3, "routed payloads");
  const dgpp::QwenBindReport rep = dgpp::qwen35_validate_text_binding(c, present_from(c));
  require(rep.ok() && rep.matched == 63863, "exact binding");
  // 41 layers x (256 x 3 routed + 3 shared) + 30 x 3 GDN + 11 x 4 attention.
  require(rep.quantized_matrices == 41 * 771 + 90 + 44, "quantized " + std::to_string(rep.quantized_matrices));
}

DGPP_TEST(qwen36_tp_geometry) {
  const dgpp::Qwen35TextConfig c = parse(text_json(), kQuantMixed);
  dgpp::qwen35_tp_validate_geometry(c, 0, 1);
  dgpp::qwen35_tp_validate_geometry(c, 1, 2);
  dgpp::qwen35_tp_validate_geometry(c, 3, 4);
}
