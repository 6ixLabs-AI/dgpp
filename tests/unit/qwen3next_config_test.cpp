// The Qwen3Next dialect of the qwen3_5 stack (Qwen3-Next-80B-A3B): the
// release's flat config parses, the unsupported shapes are refused by
// name, the architecture registry detects it, and the expected-tensor
// table is the checkpoint's — 297,728 tensors for
// nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 @ 8fb2682f, counted from its
// safetensors headers.
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

// The release's config.json, transcribed (the 242-entry ignore list cut to one).
std::string release_json() {
  return std::string(R"({
  "architectures": ["Qwen3NextForCausalLM"], "attention_bias": false,
  "attention_dropout": 0.0, "bos_token_id": 151643, "decoder_sparse_step": 1,
  "dtype": "bfloat16", "eos_token_id": 151645, "full_attention_interval": 4,
  "head_dim": 256, "hidden_act": "silu", "hidden_size": 2048,
  "initializer_range": 0.02, "intermediate_size": 5120,
  "layer_types": [)") + layers_json(48) + R"(],
  "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
  "linear_num_key_heads": 16, "linear_num_value_heads": 32,
  "linear_value_head_dim": 128, "max_position_embeddings": 262144,
  "mlp_only_layers": [], "model_type": "qwen3_next",
  "moe_intermediate_size": 512, "norm_topk_prob": true,
  "num_attention_heads": 16, "num_experts": 512, "num_experts_per_tok": 10,
  "num_hidden_layers": 48, "num_key_value_heads": 2,
  "output_router_logits": false, "partial_rotary_factor": 0.25,
  "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 10000000,
  "router_aux_loss_coef": 0.001, "shared_expert_intermediate_size": 512,
  "tie_word_embeddings": false, "use_cache": true, "use_sliding_window": false,
  "vocab_size": 151936,
  "quantization_config": {
    "config_groups": {"group_0": {
      "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
      "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
      "targets": ["Linear"]}},
    "ignore": ["lm_head"]}
})";
}

dgpp::Qwen35TextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Qwen35TextConfig::parse_qwen3_next(t.root);
}

std::string refusal(const std::string& text) {
  try {
    (void)parse(text);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

std::string patched(const std::string& from, const std::string& to) {
  std::string s = release_json();
  const size_t at = s.find(from);
  require(at != std::string::npos, "patch anchor missing: " + from);
  s.replace(at, from.size(), to);
  return s;
}

bool refuses(const std::string& from, const std::string& to, const std::string& field) {
  const std::string why = refusal(patched(from, to));
  return why.find("Qwen3-Next") != std::string::npos && why.find(field) != std::string::npos;
}

std::unordered_map<std::string, dgpp::QwenTensorDesc> present_from(
    const dgpp::Qwen35TextConfig& c) {
  std::unordered_map<std::string, dgpp::QwenTensorDesc> present;
  for (const auto& e : dgpp::qwen35_expected_text_tensors(c))
    present.emplace(e.name, dgpp::QwenTensorDesc{e.dtype, e.shape});
  return present;
}

}  // namespace

DGPP_TEST(qwen3next_config_parses_the_release) {
  const dgpp::Qwen35TextConfig c = parse(release_json());
  require(c.next() && c.moe(), "dialect");
  require(c.hidden_size == 2048 && c.vocab_size == 151936, "shape");
  require(c.num_hidden_layers == 48, "layers");
  require(c.num_gdn_layers() == 36 && c.num_full_layers() == 12, "layer kinds");
  require(c.gdn_key_heads == 16 && c.gdn_value_heads == 32, "gdn heads");
  require(c.output_gate_type == "swish", "gate");
  require(c.num_attention_heads == 16 && c.num_key_value_heads == 2 && c.head_dim == 256,
          "attention");
  require(c.rotary_dim == 64 && c.rope_theta == 1e7, "rope");
  require(c.mrope_section.empty(), "no mrope");
  require(c.num_experts == 512 && c.num_experts_per_tok == 10, "experts");
  require(c.moe_intermediate_size == 512 && c.shared_expert_intermediate_size == 512, "moe");
  require(c.mtp_layer() == 48, "the release carries the draft layer");
  require(c.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Modelopt, "quant");
  require(c.eos_token_ids == std::vector<int64_t>({151645}), "eos");
  require(c.context_limit() == 262144, "context");
}

DGPP_TEST(qwen3next_config_derives_layer_types_from_the_interval) {
  std::string s = release_json();
  const size_t a = s.find("\"layer_types\"");
  const size_t b = s.find("],", a);
  require(a != std::string::npos && b != std::string::npos, "anchor");
  s.erase(a, b + 2 - a);
  const dgpp::Qwen35TextConfig c = parse(s);
  require(c.num_gdn_layers() == 36 && c.num_full_layers() == 12, "derived kinds");
  require(c.layers[3] == dgpp::Qwen35LayerKind::Full && c.layers[4] == dgpp::Qwen35LayerKind::Gdn,
          "pattern");
}

DGPP_TEST(qwen3next_config_refusals_name_the_field) {
  require(refuses("\"qwen3_next\"", "\"qwen3_5\"", "model_type"), "model_type");
  require(refuses("\"full_attention_interval\": 4", "\"full_attention_interval\": 3",
                  "full_attention_interval"),
          "interval");
  require(refuses("\"head_dim\": 256", "\"head_dim\": 128", "head_dim"), "head dim");
  require(refuses("\"linear_key_head_dim\": 128", "\"linear_key_head_dim\": 64",
                  "linear_key_head_dim"),
          "gdn head dim");
  require(refuses("\"rope_scaling\": null", "\"rope_scaling\": {\"type\": \"yarn\"}", "rope_scaling"),
          "rope scaling");
  require(refuses("\"norm_topk_prob\": true", "\"norm_topk_prob\": false", "norm_topk_prob"),
          "topk norm");
  require(refuses("\"num_experts_per_tok\": 10", "\"num_experts_per_tok\": 32",
                  "num_experts_per_tok"),
          "top-k bound");
  require(refuses("\"moe_intermediate_size\": 512", "\"moe_intermediate_size\": 500",
                  "moe_intermediate_size"),
          "moe width");
  require(refuses("\"decoder_sparse_step\": 1", "\"decoder_sparse_step\": 2", "decoder_sparse_step"),
          "sparse step");
  require(refuses("\"mlp_only_layers\": []", "\"mlp_only_layers\": [0]", "mlp_only_layers"),
          "dense layers");
  require(refuses("\"use_sliding_window\": false", "\"use_sliding_window\": true",
                  "use_sliding_window"),
          "sliding window");
  require(refuses("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true",
                  "tie_word_embeddings"),
          "tied");
  // The quantization recipe: only the modelopt NVFP4 group is implemented.
  require(refusal(patched("\"weights\": {\"dynamic\": false, \"num_bits\": 4",
                          "\"weights\": {\"dynamic\": false, \"num_bits\": 8"))
              .find("NVFP4") != std::string::npos,
          "bits");
  require(refusal(patched("\"group_0\"", "\"group_7\"")).find("group_0") != std::string::npos,
          "group");
}

DGPP_TEST(qwen3next_architecture_detects_the_release) {
  const std::string json = release_json();  // the parsed values view the text
  const auto root = dgpp::minijson::parse(json);
  require(dgpp::detect_architecture(root.root) == dgpp::ModelArchitecture::Qwen3Next, "detect");
  const auto by_type = dgpp::minijson::parse(R"({"model_type": "qwen3_next"})");
  require(dgpp::detect_architecture(by_type.root) == dgpp::ModelArchitecture::Qwen3Next,
          "by model_type");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::Qwen3Next)) ==
              "qwen3_next",
          "name");
  const auto q35 = dgpp::minijson::parse(
      R"({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"})");
  require(dgpp::detect_architecture(q35.root) == dgpp::ModelArchitecture::Qwen3_5,
          "qwen3_5 unchanged");
}

DGPP_TEST(qwen3next_binding_table_is_the_checkpoint) {
  const dgpp::Qwen35TextConfig c = parse(release_json());
  const auto all = dgpp::qwen35_expected_text_tensors(c);
  // 36 GDN layers x 6170 + 12 attention layers x 6171 + the draft layer's
  // 1549 + 7 globals: the count of the release's safetensors headers.
  require(all.size() == 297728, "table size " + std::to_string(all.size()));
  require(dgpp::qwen35_expected_layer_tensors(c, 0).size() == 6170, "gdn layer");
  require(dgpp::qwen35_expected_layer_tensors(c, 3).size() == 6171, "attention layer");
  require(dgpp::qwen35_expected_layer_tensors(c, c.mtp_layer()).size() == 1549, "draft layer");
  require(dgpp::qwen35_expected_global_tensors(c).size() == 7, "globals");
  require(dgpp::qwen35_layer_prefix(c, 5) == "model.layers.5.", "flat names");
  require(dgpp::qwen35_layer_prefix(c, 48) == "mtp.layers.0.", "draft prefix");

  std::unordered_map<std::string, const dgpp::QwenExpectedTensor*> by_name;
  for (const auto& e : all) by_name.emplace(e.name, &e);
  auto shape_is = [&](const std::string& name, dgpp::DType dt, std::vector<int64_t> shape) {
    const auto it = by_name.find(name);
    return it != by_name.end() && it->second->dtype == dt && it->second->shape == shape;
  };
  using dgpp::DType;
  // The fused GDN projections and the NVFP4 output projection.
  require(shape_is("model.layers.0.linear_attn.in_proj_qkvz.weight", DType::BF16, {12288, 2048}),
          "qkvz");
  require(shape_is("model.layers.0.linear_attn.in_proj_ba.weight", DType::BF16, {64, 2048}), "ba");
  require(shape_is("model.layers.0.linear_attn.conv1d.weight", DType::BF16, {8192, 1, 4}), "conv");
  require(shape_is("model.layers.0.linear_attn.out_proj.weight", DType::U8, {2048, 2048}),
          "out_proj codes");
  require(shape_is("model.layers.0.linear_attn.out_proj.weight_scale", DType::F8_E4M3, {2048, 256}),
          "out_proj scales");
  require(shape_is("model.layers.0.linear_attn.out_proj.weight_scale_2", DType::F32, {}),
          "out_proj global");
  require(by_name.count("model.layers.0.linear_attn.in_proj_qkv.weight") == 0, "no split names");
  // Attention: BF16 q/k/v with the cache scales, NVFP4 o_proj.
  require(shape_is("model.layers.3.self_attn.q_proj.weight", DType::BF16, {8192, 2048}), "q");
  require(shape_is("model.layers.3.self_attn.k_proj.k_scale", DType::F32, {}), "k_scale");
  require(shape_is("model.layers.3.self_attn.o_proj.weight", DType::U8, {2048, 2048}), "o codes");
  // The MoE: NVFP4 experts and shared expert, BF16 router.
  require(shape_is("model.layers.3.mlp.gate.weight", DType::BF16, {512, 2048}), "router");
  require(shape_is("model.layers.3.mlp.shared_expert_gate.weight", DType::BF16, {1, 2048}),
          "shared gate");
  require(shape_is("model.layers.3.mlp.experts.511.gate_proj.weight", DType::U8, {512, 1024}),
          "expert gate codes");
  require(shape_is("model.layers.3.mlp.experts.511.down_proj.weight_scale", DType::F8_E4M3,
                   {2048, 32}),
          "expert down scales");
  require(shape_is("model.layers.3.mlp.shared_expert.down_proj.weight", DType::U8, {2048, 256}),
          "shared down codes");
  // The draft layer: everything BF16, per-expert matrices, no cache scales.
  require(shape_is("mtp.layers.0.self_attn.o_proj.weight", DType::BF16, {2048, 4096}), "mtp o");
  require(shape_is("mtp.layers.0.mlp.experts.7.down_proj.weight", DType::BF16, {2048, 512}),
          "mtp expert");
  require(by_name.count("mtp.layers.0.self_attn.k_proj.k_scale") == 0, "mtp has no cache scale");
  require(by_name.count("mtp.layers.0.mlp.experts.7.down_proj.weight_scale") == 0, "mtp is bf16");
  require(shape_is("mtp.fc.weight", DType::BF16, {2048, 4096}), "mtp fc");
  require(shape_is("model.embed_tokens.weight", DType::BF16, {151936, 2048}), "embed");
  require(shape_is("model.norm.weight", DType::BF16, {2048}), "final norm");

  int routed = 0;
  for (const auto& e : dgpp::qwen35_expected_layer_tensors(c, 3))
    if (e.cls == dgpp::QwenWeightClass::RoutedExpert &&
        e.role == dgpp::QwenTensorRole::Fp4Payload) {
      require(e.expert >= 0 && e.expert < 512, "expert id");
      ++routed;
    }
  require(routed == 512 * 3, "routed payloads");
}

DGPP_TEST(qwen3next_binding_validates_and_names_what_is_wrong) {
  dgpp::Qwen35TextConfig c = parse(release_json());
  auto present = present_from(c);
  dgpp::QwenBindReport rep = dgpp::qwen35_validate_text_binding(c, present);
  require(rep.ok() && rep.matched == 297728, "exact binding");
  // 48 layers x (512 x 3 routed + 3 shared + one attention/GDN output projection).
  require(rep.quantized_matrices == 48 * 1540, "quantized " + std::to_string(rep.quantized_matrices));

  // A checkpoint without the draft head is refused by name.
  auto no_mtp = present;
  no_mtp.erase("mtp.fc.weight");
  rep = dgpp::qwen35_validate_text_binding(c, no_mtp);
  require(!rep.ok() && rep.missing == 1 && rep.errors[0].find("mtp.fc.weight") != std::string::npos,
          "missing draft head");

  // The Qwen3.5 split names are unexpected here.
  auto split = present;
  split.emplace("model.layers.0.linear_attn.in_proj_z.weight",
                dgpp::QwenTensorDesc{dgpp::DType::BF16, {4096, 2048}});
  rep = dgpp::qwen35_validate_text_binding(c, split);
  require(!rep.ok() && rep.unexpected == 1, "split name unexpected");

  // A truncated config (the check apps' --layers): later layers are out of
  // scope under the flat prefix, not unexpected.
  dgpp::Qwen35TextConfig cut = c;
  cut.num_hidden_layers = 4;
  cut.layers.resize(4);
  cut.mtp_num_layers = 0;
  auto with_tail = present_from(cut);
  with_tail.emplace("model.layers.9.input_layernorm.weight",
                    dgpp::QwenTensorDesc{dgpp::DType::BF16, {2048}});
  rep = dgpp::qwen35_validate_text_binding(cut, with_tail);
  require(rep.ok() && rep.out_of_scope == 1, "out of scope");
}

DGPP_TEST(qwen3next_tp_geometry) {
  const dgpp::Qwen35TextConfig c = parse(release_json());
  dgpp::qwen35_tp_validate_geometry(c, 0, 1);
  dgpp::qwen35_tp_validate_geometry(c, 1, 2);
  dgpp::qwen35_tp_validate_geometry(c, 3, 4);
  bool threw = false;
  try {
    dgpp::qwen35_tp_validate_geometry(c, 0, 3);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "world 3 does not divide the heads");
}
