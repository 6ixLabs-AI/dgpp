// Qwen3-Coder-Next on the Qwen3Next dialect of the qwen3_5 stack: the
// compressed-tensors NVFP4 release's flat config parses, its recipe is held
// to the one container the loader implements, Intel's AutoRound release of
// the same model is refused by name, and the expected-tensor table is the
// checkpoint's — 296,151 tensors for RedHatAI/Qwen3-Coder-Next-NVFP4 @
// 27a8f16f, counted from its safetensors headers (no `mtp.*` among them).
// The tool-call form is the template's: this checkpoint writes the XML
// tags, the 80B the JSON object, over one tokenizer.
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"
#include "text/chat_template.hpp"
#include "text/tool_parser.hpp"

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

// The release's quantization_config, transcribed (the 205-entry ignore list
// cut to one of each class it names).
const char* kPackedQuant = R"({
    "config_groups": {"group_0": {
      "format": "nvfp4-pack-quantized",
      "input_activations": {"actorder": null, "block_structure": null, "dynamic": "local",
        "group_size": 16, "num_bits": 4, "observer": "static_minmax", "observer_kwargs": {},
        "scale_dtype": "torch.float8_e4m3fn", "strategy": "tensor_group", "symmetric": true,
        "type": "float", "zp_dtype": null},
      "output_activations": null, "targets": ["Linear"],
      "weights": {"actorder": null, "block_structure": null, "dynamic": false,
        "group_size": 16, "num_bits": 4, "observer": "mse", "observer_kwargs": {},
        "scale_dtype": "torch.float8_e4m3fn", "strategy": "tensor_group", "symmetric": true,
        "type": "float", "zp_dtype": null}}},
    "format": "nvfp4-pack-quantized", "global_compression_ratio": null,
    "ignore": ["model.layers.0.linear_attn.in_proj_qkvz", "model.layers.0.linear_attn.in_proj_ba",
               "model.layers.0.linear_attn.out_proj", "model.layers.0.mlp.gate",
               "model.layers.0.mlp.shared_expert_gate", "lm_head"],
    "kv_cache_scheme": null, "quant_method": "compressed-tensors",
    "quantization_status": "compressed", "sparsity_config": {}, "transform_config": {},
    "version": "0.13.1.dev47+gcfc24ef"})";

// Intel/Qwen3-Coder-Next-int4-AutoRound's (its 48 extra_config entries cut to one).
const char* kAutoRoundQuant = R"({
    "bits": 4, "data_type": "int", "group_size": 128, "sym": true,
    "autoround_version": "0.10.0", "quant_method": "auto-round",
    "packing_format": "auto_round:auto_gptq",
    "extra_config": {"model.layers.0.mlp.shared_expert_gate": {"bits": 16, "data_type": "fp"}}})";

// The release's config.json, transcribed, around a quantization_config.
std::string config_json(const std::string& quant) {
  return std::string(R"({
  "architectures": ["Qwen3NextForCausalLM"], "attention_bias": false,
  "attention_dropout": 0, "bos_token_id": 151643, "decoder_sparse_step": 1,
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
  "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 5000000,
  "router_aux_loss_coef": 0.001, "shared_expert_intermediate_size": 512,
  "tie_word_embeddings": false, "use_cache": true, "use_sliding_window": false,
  "vocab_size": 151936,
  "quantization_config": )" + quant + "\n}";
}

std::string release_json() { return config_json(kPackedQuant); }

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

DGPP_TEST(qwen3codernext_config_parses_the_release) {
  const dgpp::Qwen35TextConfig c = parse(release_json());
  require(c.next() && c.moe(), "dialect");
  require(c.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Packed, "the compressed-tensors recipe");
  require(c.hidden_size == 2048 && c.vocab_size == 151936 && c.num_hidden_layers == 48, "shape");
  require(c.num_gdn_layers() == 36 && c.num_full_layers() == 12, "layer kinds");
  require(c.gdn_key_heads == 16 && c.gdn_value_heads == 32, "gdn heads");
  require(c.num_attention_heads == 16 && c.num_key_value_heads == 2 && c.head_dim == 256,
          "attention");
  require(c.rotary_dim == 64 && c.rope_theta == 5e6, "rope (half the 80B's theta)");
  require(c.num_experts == 512 && c.num_experts_per_tok == 10, "experts");
  require(c.moe_intermediate_size == 512 && c.shared_expert_intermediate_size == 512, "moe");
  require(c.mtp_num_layers == 0 && c.mtp_layer() == -1, "the release carries no draft layer");
  require(c.eos_token_ids == std::vector<int64_t>({151645}), "eos");
  require(c.context_limit() == 262144, "context");

  // The same config under the modelopt recipe is the 80B's container and
  // keeps its draft layer; a config that names the field is taken at its word.
  const dgpp::Qwen35TextConfig m = parse(config_json(
      R"({"config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}})"));
  require(m.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Modelopt && m.mtp_layer() == 48, "modelopt");
  const dgpp::Qwen35TextConfig named =
      parse(patched("\"vocab_size\": 151936", "\"vocab_size\": 151936, \"mtp_num_hidden_layers\": 1"));
  require(named.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Packed && named.mtp_layer() == 48,
          "a named draft layer overrides the recipe's default");
}

DGPP_TEST(qwen3codernext_config_refusals_name_the_field) {
  // Intel's AutoRound int4 release of the same model: refused by name.
  const std::string why = refusal(config_json(kAutoRoundQuant));
  require(why.find("quant_method") != std::string::npos && why.find("auto-round") != std::string::npos &&
              why.find("RedHatAI/Qwen3-Coder-Next-NVFP4") != std::string::npos,
          "the AutoRound release is refused and the served one named: " + why);
  // The container: only nvfp4-pack-quantized beside modelopt's.
  require(refuses("\"format\": \"nvfp4-pack-quantized\", \"global_compression_ratio\"",
                  "\"format\": \"int-quantized\", \"global_compression_ratio\"", "format"),
          "format");
  // What would change the meaning of a stored code.
  require(refuses("\"transform_config\": {}",
                  "\"transform_config\": {\"config_groups\": {\"R1\": {\"type\": \"hadamard\"}}}",
                  "transform_config"),
          "transforms");
  require(refuses("\"sparsity_config\": {}", "\"sparsity_config\": {\"format\": \"sparse-bitmask\"}",
                  "sparsity_config"),
          "sparsity");
  require(refuses("\"kv_cache_scheme\": null",
                  "\"kv_cache_scheme\": {\"num_bits\": 8, \"type\": \"float\"}", "kv_cache_scheme"),
          "kv cache scheme");
  require(refuses("\"quantization_status\": \"compressed\"", "\"quantization_status\": \"frozen\"",
                  "quantization_status"),
          "status");
  require(refuses("\"strategy\": \"tensor_group\", \"symmetric\": true,\n        \"type\": \"float\", "
                  "\"zp_dtype\": null}}}",
                  "\"strategy\": \"tensor_group\", \"symmetric\": false,\n        \"type\": \"float\", "
                  "\"zp_dtype\": null}}}",
                  "symmetric"),
          "asymmetric weights");
  require(refusal(patched("\"observer\": \"mse\", \"observer_kwargs\": {},\n        \"scale_dtype\": "
                          "\"torch.float8_e4m3fn\", \"strategy\": \"tensor_group\"",
                          "\"observer\": \"mse\", \"observer_kwargs\": {},\n        \"scale_dtype\": "
                          "\"torch.float8_e4m3fn\", \"strategy\": \"channel\""))
              .find("strategy") != std::string::npos,
          "weight strategy");
  require(refusal(patched("\"dynamic\": false,\n        \"group_size\": 16, \"num_bits\": 4",
                          "\"dynamic\": false,\n        \"group_size\": 32, \"num_bits\": 4"))
              .find("NVFP4") != std::string::npos,
          "group size");
  // The dialect's own refusals hold under this recipe too.
  require(refuses("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true",
                  "tie_word_embeddings"),
          "tied");
  require(refuses("\"rope_scaling\": null", "\"rope_scaling\": {\"type\": \"yarn\"}", "rope_scaling"),
          "rope scaling");
}

DGPP_TEST(qwen3codernext_architecture_is_the_qwen3next_class) {
  const std::string json = release_json();  // the parsed values view the text
  const auto root = dgpp::minijson::parse(json);
  require(dgpp::detect_architecture(root.root) == dgpp::ModelArchitecture::Qwen3Next, "detect");
}

DGPP_TEST(qwen3codernext_binding_table_is_the_checkpoint) {
  const dgpp::Qwen35TextConfig c = parse(release_json());
  const auto all = dgpp::qwen35_expected_text_tensors(c);
  // 36 GDN layers x 6167 + 12 attention layers x 6178 + 3 globals: the count
  // of the release's safetensors headers (10 shards).
  require(all.size() == 296151, "table size " + std::to_string(all.size()));
  require(dgpp::qwen35_expected_layer_tensors(c, 0).size() == 6167, "gdn layer");
  require(dgpp::qwen35_expected_layer_tensors(c, 3).size() == 6178, "attention layer");
  require(dgpp::qwen35_expected_global_tensors(c).size() == 3, "globals");
  bool threw = false;
  try {
    (void)dgpp::qwen35_expected_layer_tensors(c, 48);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "there is no draft layer to describe");

  std::unordered_map<std::string, const dgpp::QwenExpectedTensor*> by_name;
  for (const auto& e : all) by_name.emplace(e.name, &e);
  auto shape_is = [&](const std::string& name, dgpp::DType dt, std::vector<int64_t> shape) {
    const auto it = by_name.find(name);
    return it != by_name.end() && it->second->dtype == dt && it->second->shape == shape;
  };
  using dgpp::DType;
  // The GDN: BF16 throughout, the fused projections and the output one.
  require(shape_is("model.layers.0.linear_attn.in_proj_qkvz.weight", DType::BF16, {12288, 2048}),
          "qkvz");
  require(shape_is("model.layers.0.linear_attn.in_proj_ba.weight", DType::BF16, {64, 2048}), "ba");
  require(shape_is("model.layers.0.linear_attn.out_proj.weight", DType::BF16, {2048, 4096}),
          "out_proj is BF16 here");
  require(by_name.count("model.layers.0.linear_attn.out_proj.weight_scale") == 0 &&
              by_name.count("model.layers.0.linear_attn.out_proj.weight_packed") == 0,
          "no NVFP4 set on the GDN");
  // Attention: all four projections in the packed set, no cache scales.
  require(shape_is("model.layers.3.self_attn.q_proj.weight_packed", DType::U8, {8192, 1024}), "q codes");
  require(shape_is("model.layers.3.self_attn.q_proj.weight_scale", DType::F8_E4M3, {8192, 128}),
          "q scales");
  require(shape_is("model.layers.3.self_attn.k_proj.weight_packed", DType::U8, {512, 1024}), "k codes");
  require(shape_is("model.layers.3.self_attn.v_proj.weight_global_scale", DType::F32, {1}), "v global");
  require(shape_is("model.layers.3.self_attn.o_proj.weight_packed", DType::U8, {2048, 2048}), "o codes");
  require(shape_is("model.layers.3.self_attn.o_proj.input_global_scale", DType::F32, {1}), "o input");
  require(by_name.count("model.layers.3.self_attn.q_proj.weight") == 0, "no BF16 q_proj");
  require(by_name.count("model.layers.3.self_attn.k_proj.k_scale") == 0, "no cache scales");
  // The MoE: BF16 router and shared gate, packed shared expert and experts.
  require(shape_is("model.layers.3.mlp.gate.weight", DType::BF16, {512, 2048}), "router");
  require(shape_is("model.layers.3.mlp.shared_expert_gate.weight", DType::BF16, {1, 2048}),
          "shared gate");
  require(shape_is("model.layers.3.mlp.experts.511.gate_proj.weight_packed", DType::U8, {512, 1024}),
          "expert gate codes");
  require(shape_is("model.layers.3.mlp.experts.511.down_proj.weight_scale", DType::F8_E4M3, {2048, 32}),
          "expert down scales");
  require(shape_is("model.layers.3.mlp.experts.0.up_proj.weight_global_scale", DType::F32, {1}),
          "expert global");
  require(shape_is("model.layers.3.mlp.shared_expert.down_proj.weight_packed", DType::U8, {2048, 256}),
          "shared down codes");
  require(by_name.count("model.layers.3.mlp.experts.0.up_proj.weight_scale_2") == 0 &&
              by_name.count("model.layers.3.mlp.experts.0.up_proj.weight") == 0,
          "no modelopt names");
  // Globals: no draft head.
  require(shape_is("model.embed_tokens.weight", DType::BF16, {151936, 2048}), "embed");
  require(shape_is("lm_head.weight", DType::BF16, {151936, 2048}), "head");
  require(shape_is("model.norm.weight", DType::BF16, {2048}), "final norm");
  for (const auto& e : all) require(e.name.rfind("mtp.", 0) != 0, "no mtp tensor: " + e.name);

  int routed = 0;
  for (const auto& e : dgpp::qwen35_expected_layer_tensors(c, 3))
    if (e.cls == dgpp::QwenWeightClass::RoutedExpert && e.role == dgpp::QwenTensorRole::Fp4Payload) {
      require(e.expert >= 0 && e.expert < 512, "expert id");
      ++routed;
    }
  require(routed == 512 * 3, "routed payloads");
}

DGPP_TEST(qwen3codernext_binding_validates_and_names_what_is_wrong) {
  const dgpp::Qwen35TextConfig c = parse(release_json());
  auto present = present_from(c);
  dgpp::QwenBindReport rep = dgpp::qwen35_validate_text_binding(c, present);
  require(rep.ok() && rep.matched == 296151, "exact binding");
  // 48 layers x (512 x 3 routed + 3 shared) + 12 attention layers x q/k/v/o.
  require(rep.quantized_matrices == 48 * 1539 + 12 * 4,
          "quantized " + std::to_string(rep.quantized_matrices));

  // A draft head is not part of this release: present, it is unexpected.
  auto with_mtp = present;
  with_mtp.emplace("mtp.fc.weight", dgpp::QwenTensorDesc{dgpp::DType::BF16, {2048, 4096}});
  rep = dgpp::qwen35_validate_text_binding(c, with_mtp);
  require(!rep.ok() && rep.unexpected == 1 && rep.errors[0].find("mtp.fc.weight") != std::string::npos,
          "an unexpected draft head is named");

  // The modelopt container's table does not bind this checkpoint (and the
  // reverse): the recipe, not a guess, picks the names.
  const dgpp::Qwen35TextConfig m = parse(config_json(
      R"({"config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}})"));
  rep = dgpp::qwen35_validate_text_binding(m, present, 4);
  require(!rep.ok() && rep.missing > 0 && rep.unexpected > 0, "modelopt table against packed tensors");

  // The global scale is F32 [1] here, not modelopt's F32 [].
  auto scalar = present;
  scalar["model.layers.3.self_attn.o_proj.weight_global_scale"].shape = {};
  rep = dgpp::qwen35_validate_text_binding(c, scalar);
  require(!rep.ok() && rep.shape_mismatch == 1, "global scale shape");
}

DGPP_TEST(qwen3codernext_tool_calls_are_the_templates_form) {
  using dgpp::text::chat_template_writes_json_calls;
  // Qwen3-Next-80B-A3B-Instruct's template: the instruction (a double-quoted
  // Jinja literal) and the rendering of an assistant call (single-quoted).
  const std::string hermes =
      "{{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and "
      "arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": "
      "<function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n"
      "{{- '<tool_call>\\n{\"name\": \"' }}{{- tool_call.name }}{{- '\", \"arguments\": ' }}";
  require(chat_template_writes_json_calls(hermes), "the 80B's template writes JSON calls");
  require(chat_template_writes_json_calls(
              "{{- \"<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args>}\" }}"),
          "the escaped spelling alone");
  // Qwen3-Coder-Next's: the XML tags, in the instruction and in the rendering.
  const std::string xml =
      "{{- '\\n\\nIf you choose to call a function ONLY reply in the following format with NO "
      "suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>"
      "\\nvalue_1\\n</parameter>\\n</function>\\n</tool_call>' }}\n"
      "{{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n"
      "{%- for args_name, args_value in tool_call.arguments|items %}";
  require(!chat_template_writes_json_calls(xml), "Coder-Next's template writes XML calls");
  require(!chat_template_writes_json_calls("{{- messages[0].content }}"), "a template without tools");
  // A template that names both spells the XML tags: the tags decide.
  require(!chat_template_writes_json_calls(hermes + xml), "the tag wins");
}

// The two Jinja constructs this checkpoint's template uses that no earlier
// one did (its render_extra_keys macro walks a schema's extra keys with
// `for key in mapping if key not in handled`): a loop over a mapping's keys
// and the loop filter.
DGPP_TEST(qwen3codernext_template_loops_over_mapping_keys_with_a_filter) {
  const auto render = [](const std::string& source, const std::string& globals) {
    const auto parsed = dgpp::minijson::parse(globals);
    return dgpp::text::ChatTemplate::compile(source).render(
        dgpp::text::Value::from_minijson(parsed.root));
  };
  const std::string globals =
      R"({"d": {"type": "string", "enum": ["x", "y"], "description": "a", "minLength": 2},
          "skip": ["type", "description"], "xs": [3, 1, 4, 1, 5]})";
  // A mapping iterates as its keys, in member order.
  require(render("{% for k in d %}{{ k }};{% endfor %}", globals) == "type;enum;description;minLength;",
          "a mapping iterates as its keys");
  // The filter keeps what its condition accepts; loop.* is the kept items'.
  require(render("{% for k in d if k not in skip %}{{ loop.index }}/{{ loop.length }}:{{ k }}"
                 "{% if not loop.last %},{% endif %}{% endfor %}",
                 globals) == "1/2:enum,2/2:minLength",
          "the filter and the loop's own counters");
  require(render("{% for k in d if k not in skip %}{{ loop.previtem is defined }}|{{ loop.nextitem is defined }};"
                 "{% endfor %}",
                 globals) == "False|True;True|False;",
          "neighbours are the kept items'");
  require(render("{% for x in xs if x != 1 %}{{ x }}{% endfor %}", globals) == "345", "a list under a filter");
  require(render("{% for x in xs if x == 9 %}{{ x }}{% endfor %}!", globals) == "!", "nothing kept: no iteration");
  // A two-name target under a filter, and a conditional inside the filter.
  require(render("{% for k, v in d|items if v is string %}{{ k }}={{ v }};{% endfor %}", globals) ==
              "type=string;description=a;",
          "a tuple target under a filter");
  require(render("{% for x in xs if (x != 1 if skip else false) %}{{ x }}{% endfor %}", globals) == "345",
          "a conditional expression inside the filter");
  // The macro of the template itself, on a parameter's schema.
  const std::string macro =
      "{% macro render_extra_keys(json_dict, handled_keys) %}{%- if json_dict is mapping %}"
      "{%- for json_key in json_dict if json_key not in handled_keys %}"
      "{%- if json_dict[json_key] is string %}"
      "{{-'\\n<' ~ json_key ~ '>' ~ (json_dict[json_key] | string) ~ '</' ~ json_key ~ '>' }}"
      "{%- else %}"
      "{{- '\\n<' ~ json_key ~ '>' ~ (json_dict[json_key] | tojson | safe) ~ '</' ~ json_key ~ '>' }}"
      "{%- endif %}{%- endfor %}{%- endif %}{%- endmacro %}"
      "{{- render_extra_keys(d, ['type', 'description']) }}";
  require(render(macro, globals) == "\n<enum>[\"x\", \"y\"]</enum>\n<minLength>2</minLength>",
          "the template's macro: " + render(macro, globals));
  // A global read only by a filter is still a global the template reads.
  const dgpp::text::ChatTemplate t =
      dgpp::text::ChatTemplate::compile("{% for k in d if k not in skip %}{{ k }}{% endfor %}");
  require(t.reads("skip") && t.reads("d") && !t.reads("k"), "reads() sees the filter's globals");
}

DGPP_TEST(qwen3codernext_tp_geometry) {
  const dgpp::Qwen35TextConfig c = parse(release_json());
  dgpp::qwen35_tp_validate_geometry(c, 0, 1);
  dgpp::qwen35_tp_validate_geometry(c, 1, 2);
  dgpp::qwen35_tp_validate_geometry(c, 3, 4);
}
