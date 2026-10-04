// Qwen3.5-0.8B on the qwen3_5 stack: the dense Qwen3.5 dialect from an
// unquantized checkpoint (no quantization_config: every matrix BF16) with
// tied embeddings (no stored head). Its config parses, the expected-tensor
// table is the checkpoint's — 488 tensors in Qwen/Qwen3.5-0.8B @ 2fc06364's
// headers, 153 of them the vision tower's, 335 in the table — the counting
// build reads every byte of a layer, and the template construct this
// checkpoint's chat template adds (`is sequence`) evaluates as Jinja's.
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/loader35.hpp"
#include "text/chat_template.hpp"

namespace {

using dgpp::Qwen35LayerStream;
using dgpp::Qwen35LoaderFamily;

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

// The release's text_config, transcribed.
std::string text_json() {
  return std::string(R"({
  "attention_bias": false, "attention_dropout": 0.0, "attn_output_gate": true, "dtype": "bfloat16",
  "eos_token_id": 248044, "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu",
  "hidden_size": 1024, "initializer_range": 0.02, "intermediate_size": 3584,
  "layer_types": [)") + layers_json(24) + R"(],
  "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
  "linear_num_value_heads": 16, "linear_value_head_dim": 128, "max_position_embeddings": 262144,
  "mlp_only_layers": [], "model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1,
  "mtp_use_dedicated_embeddings": false, "num_attention_heads": 8, "num_hidden_layers": 24,
  "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "tie_word_embeddings": true, "use_cache": true,
  "vocab_size": 248320, "mamba_ssm_dtype": "float32",
  "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "default",
                      "rope_theta": 10000000, "partial_rotary_factor": 0.25}
})";
}

// The release carries no quantization_config.
dgpp::Qwen35TextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Qwen35TextConfig::parse(t.root, nullptr);
}

std::string patched(const std::string& from, const std::string& to) {
  std::string s = text_json();
  const size_t at = s.find(from);
  require(at != std::string::npos, "patch anchor missing: " + from);
  s.replace(at, from.size(), to);
  return s;
}

struct DenseForm {
  bool saved = Qwen35LayerStream::dense_weights_fp8();
  explicit DenseForm(bool fp8) { Qwen35LayerStream::set_dense_weights_fp8(fp8); }
  ~DenseForm() { Qwen35LayerStream::set_dense_weights_fp8(saved); }
};

size_t table_bytes(const dgpp::Qwen35TextConfig& c, int layer, bool replicated_only) {
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

DGPP_TEST(qwen35_bf16_config_parses_the_release) {
  const dgpp::Qwen35TextConfig c = parse(text_json());
  require(!c.next() && !c.moe(), "the dense Qwen3.5 dialect");
  require(c.quant_kind == dgpp::Qwen35QuantKind::Bf16, "no quantization_config: the unquantized kind");
  require(c.tie_word_embeddings, "tied embeddings");
  require(c.hidden_size == 1024 && c.vocab_size == 248320 && c.num_hidden_layers == 24, "shape");
  require(c.num_gdn_layers() == 18 && c.num_full_layers() == 6, "layer kinds");
  require(c.gdn_key_heads == 16 && c.gdn_value_heads == 16, "one value head a key head");
  require(c.num_attention_heads == 8 && c.num_key_value_heads == 2 && c.head_dim == 256, "attention");
  require(c.intermediate_size == 3584 && c.num_experts == 0, "dense MLP");
  require(c.rotary_dim == 64 && c.mtp_layer() == 24, "rope and the draft layer");
  // A JSON null quantization_config is the same statement.
  const auto t = dgpp::minijson::parse(text_json());
  const auto null_q = dgpp::minijson::parse("null");
  require(dgpp::Qwen35TextConfig::parse(t.root, &null_q.root).quant_kind == dgpp::Qwen35QuantKind::Bf16,
          "a null quantization_config");
  // The Qwen3Next dialect still refuses tied embeddings (qwen3next_config_test);
  // an unknown quantization method is still refused here.
  const auto bad = dgpp::minijson::parse(R"({"quant_method": "linear"})");
  bool refused = false;
  try {
    (void)dgpp::Qwen35TextConfig::parse(t.root, &bad.root);
  } catch (const std::runtime_error& e) {
    refused = std::string(e.what()).find("quant_method") != std::string::npos;
  }
  require(refused, "an unknown quant_method is refused by name");
  const auto root = dgpp::minijson::parse(
      R"({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"})");
  require(dgpp::detect_architecture(root.root) == dgpp::ModelArchitecture::Qwen3_5, "architecture");
}

DGPP_TEST(qwen35_bf16_binding_table_is_the_checkpoint) {
  const dgpp::Qwen35TextConfig c = parse(text_json());
  const auto all = dgpp::qwen35_expected_text_tensors(c);
  // 18 GDN layers x 14 + 6 attention layers x 11 + the draft layer's 11 + 6
  // globals: the release's headers less its 153 vision tensors.
  require(all.size() == 335, "table size " + std::to_string(all.size()));
  require(dgpp::qwen35_expected_layer_tensors(c, 0).size() == 14, "gdn layer");
  require(dgpp::qwen35_expected_layer_tensors(c, 3).size() == 11, "attention layer");
  require(dgpp::qwen35_expected_layer_tensors(c, c.mtp_layer()).size() == 11, "draft layer");
  require(dgpp::qwen35_expected_global_tensors(c).size() == 6, "globals: no lm_head");

  std::unordered_map<std::string, const dgpp::QwenExpectedTensor*> by_name;
  for (const auto& e : all) by_name.emplace(e.name, &e);
  auto shape_is = [&](const std::string& name, dgpp::DType dt, std::vector<int64_t> shape) {
    const auto it = by_name.find(name);
    return it != by_name.end() && it->second->dtype == dt && it->second->shape == shape;
  };
  using dgpp::DType;
  const std::string L0 = "model.language_model.layers.0.", L3 = "model.language_model.layers.3.";
  // The GDN: BF16 split projections; A_log and the norm weight listed BF16
  // and bound in F32 as well (this release stores them F32).
  require(shape_is(L0 + "linear_attn.in_proj_qkv.weight", DType::BF16, {6144, 1024}), "qkv");
  require(shape_is(L0 + "linear_attn.in_proj_z.weight", DType::BF16, {2048, 1024}), "z");
  require(shape_is(L0 + "linear_attn.out_proj.weight", DType::BF16, {1024, 2048}), "out");
  require(shape_is(L0 + "linear_attn.in_proj_a.weight", DType::BF16, {16, 1024}), "a");
  require(shape_is(L0 + "linear_attn.conv1d.weight", DType::BF16, {6144, 1, 4}), "conv");
  require(shape_is(L0 + "linear_attn.A_log", DType::BF16, {16}) &&
              by_name.at(L0 + "linear_attn.A_log")->role == dgpp::QwenTensorRole::Bf16OrF32,
          "A_log: either float dtype");
  require(shape_is(L0 + "linear_attn.norm.weight", DType::BF16, {128}) &&
              by_name.at(L0 + "linear_attn.norm.weight")->role == dgpp::QwenTensorRole::Bf16OrF32,
          "the GDN norm weight: either float dtype");
  require(by_name.at(L0 + "linear_attn.dt_bias")->role == dgpp::QwenTensorRole::Plain, "dt_bias is BF16 alone");
  require(shape_is(L0 + "linear_attn.dt_bias", DType::BF16, {16}), "dt_bias is BF16");
  require(by_name.count(L0 + "linear_attn.in_proj_qkv.weight_scale_inv") == 0, "no scale partners");
  // Attention and the dense MLP: BF16.
  require(shape_is(L3 + "self_attn.q_proj.weight", DType::BF16, {4096, 1024}), "q");
  require(shape_is(L3 + "self_attn.k_proj.weight", DType::BF16, {512, 1024}), "k");
  require(shape_is(L3 + "self_attn.o_proj.weight", DType::BF16, {1024, 2048}), "o");
  require(shape_is(L3 + "mlp.gate_proj.weight", DType::BF16, {3584, 1024}), "mlp gate");
  require(shape_is(L3 + "mlp.down_proj.weight", DType::BF16, {1024, 3584}), "mlp down");
  require(shape_is("mtp.layers.0.mlp.up_proj.weight", DType::BF16, {3584, 1024}), "draft mlp");
  require(shape_is("mtp.fc.weight", DType::BF16, {1024, 2048}), "mtp fc");
  // Globals: the embedding is the head.
  require(shape_is("model.language_model.embed_tokens.weight", DType::BF16, {248320, 1024}), "embed");
  require(by_name.count("lm_head.weight") == 0, "a tied config expects no stored head");

  // The release's headers: the two GDN vectors F32, everything else as listed.
  std::unordered_map<std::string, dgpp::QwenTensorDesc> present;
  for (const auto& e : all)
    present.emplace(e.name, dgpp::QwenTensorDesc{e.role == dgpp::QwenTensorRole::Bf16OrF32 ? DType::F32 : e.dtype,
                                                 e.shape});
  for (int i = 0; i < 153; ++i)
    present.emplace("model.visual.blocks." + std::to_string(i) + ".norm1.weight",
                    dgpp::QwenTensorDesc{DType::BF16, {768}});
  dgpp::QwenBindReport rep = dgpp::qwen35_validate_text_binding(c, present);
  require(rep.ok() && rep.matched == 335 && rep.vision == 153 && rep.quantized_matrices == 0,
          "exact binding beside the vision tower, nothing quantized");
  // A head stored beside a tied config (Hcompany/Holo-3.1-0.8B saves both) is
  // counted and not read: the head is the embedding, as the class ties it.
  present.emplace("lm_head.weight", dgpp::QwenTensorDesc{DType::BF16, {248320, 1024}});
  rep = dgpp::qwen35_validate_text_binding(c, present);
  require(rep.ok() && rep.matched == 335 && rep.out_of_scope == 1 && rep.unexpected == 0,
          "a stored head under a tied config is out of scope");
  present.erase("lm_head.weight");
  // Other releases of this kind store the two vectors BF16 (Holo-3.1-0.8B,
  // Ornith-1.0-9B): bound alike. Any other dtype, or F32 on a vector that
  // has no second dtype, is refused.
  present[L0 + "linear_attn.A_log"].dtype = DType::BF16;
  present[L0 + "linear_attn.norm.weight"].dtype = DType::BF16;
  rep = dgpp::qwen35_validate_text_binding(c, present);
  require(rep.ok() && rep.matched == 335, "BF16 A_log and norm weight bind");
  present[L0 + "linear_attn.A_log"].dtype = DType::F16;
  present[L0 + "linear_attn.dt_bias"].dtype = DType::F32;
  rep = dgpp::qwen35_validate_text_binding(c, present);
  require(!rep.ok() && rep.dtype_mismatch == 2, "an F16 A_log and an F32 dt_bias are refused");
  present[L0 + "linear_attn.A_log"].dtype = DType::F32;
  present[L0 + "linear_attn.dt_bias"].dtype = DType::BF16;
  // A fine-tune saved without its draft layer under a config that names one
  // (deepreinforce-ai/Ornith-1.0-9B): said once, ahead of the missing names.
  for (auto it = present.begin(); it != present.end();)
    it = it->first.rfind("mtp.", 0) == 0 ? present.erase(it) : std::next(it);
  rep = dgpp::qwen35_validate_text_binding(c, present);
  require(!rep.ok() && rep.missing == 15 && rep.errors[0].find("names a draft layer") != std::string::npos,
          "a config naming a draft layer the checkpoint lacks");

  // Untied, the same release would store its head.
  const dgpp::Qwen35TextConfig untied =
      parse(patched("\"tie_word_embeddings\": true", "\"tie_word_embeddings\": false"));
  require(dgpp::qwen35_expected_text_tensors(untied).size() == 336, "the untied table adds lm_head.weight");
}

DGPP_TEST(qwen35_bf16_loader_counting_build) {
  const dgpp::Qwen35TextConfig c = parse(text_json());
  const int draft = c.mtp_layer();
  const int64_t H = 1024, I = 3584, V = 248320;
  for (int layer : {0, 3, draft}) {
    require(Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 1) == table_bytes(c, layer, false),
            "world 1 reads every byte of layer " + std::to_string(layer) + "'s table");
    const uint64_t r0 = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 2);
    const uint64_t r1 = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 1, 2);
    require(r0 == r1 && r0 + r1 == table_bytes(c, layer, false) + table_bytes(c, layer, true),
            "world 2 tiles layer " + std::to_string(layer));
  }
  // The F32 GDN norm weight is replicated like the BF16 one; A_log slices.
  for (const auto& e : Qwen35LoaderFamily::layer_table(c, 0)) {
    if (e.name.find("linear_attn.norm.weight") != std::string::npos)
      require(Qwen35LoaderFamily::digest_included(e), "the GDN norm weight is replicated");
    if (e.name.find("linear_attn.A_log") != std::string::npos)
      require(!Qwen35LoaderFamily::digest_included(e), "A_log slices with the value heads");
  }
  // Globals: the embedding, the final norm and the draft head — no head grant.
  const size_t globals = Qwen35LayerStream::globals_bytes(c);
  require(globals == dgpp::align_up_256(static_cast<size_t>(V * H * 2)) + 4 * dgpp::align_up_256(H * 2) +
                         dgpp::align_up_256(H * 2 * H * 2),
          "tied globals carry no head");
  const dgpp::Qwen35TextConfig untied =
      parse(patched("\"tie_word_embeddings\": true", "\"tie_word_embeddings\": false"));
  require(Qwen35LayerStream::globals_bytes(untied) == globals + dgpp::align_up_256(static_cast<size_t>(V * H * 2)),
          "an untied head is one more grant");
  const size_t gdn = Qwen35LayerStream::layer_bytes(c, 0), attn = Qwen35LayerStream::layer_bytes(c, 3);
  require(Qwen35LayerStream::resident_bytes(c, 0, 1, dgpp::LoaderHeadSharding::Full, true) ==
              18 * gdn + 6 * attn + Qwen35LayerStream::layer_bytes(c, draft) + globals,
          "resident bytes are the layers, the draft layer and the globals");
  // dense_weights fp8: every dense matrix is codes plus a scale grid.
  const size_t mlp = 2 * fp8_saving(I, H) + fp8_saving(H, I);
  const DenseForm fp8(true);
  require(Qwen35LayerStream::layer_bytes(c, 0) ==
              gdn - mlp - fp8_saving(6144, H) - fp8_saving(2048, H) - fp8_saving(H, 2048),
          "a GDN layer under the knob");
  require(Qwen35LayerStream::layer_bytes(c, 3) ==
              attn - mlp - fp8_saving(4096, H) - 2 * fp8_saving(512, H) - fp8_saving(H, 2048),
          "an attention layer under the knob");
  require(Qwen35LayerStream::planned_layer_source_bytes(c, 0, 0, 1) == table_bytes(c, 0, false),
          "the FP8 form reads the same checkpoint bytes");
}

// This checkpoint's chat template tests an argument value with `is sequence`
// (Jinja: anything with a length and an item access — strings, lists and
// mappings alike) before choosing between tojson and str.
DGPP_TEST(qwen35_bf16_template_is_sequence) {
  const auto render = [](const std::string& source, const std::string& globals) {
    const auto parsed = dgpp::minijson::parse(globals);
    return dgpp::text::ChatTemplate::compile(source).render(dgpp::text::Value::from_minijson(parsed.root));
  };
  const std::string globals = R"({"s": "text", "l": [1, 2], "m": {"k": 1}, "n": 3, "b": true})";
  // `u` is undefined. Jinja's test is "len() and __getitem__ exist", and its lenient Undefined has
  // both, so `u is sequence` is True: jinja2 3.1.6's sandboxed environment (the one transformers
  // renders chat templates with) prints True|True|True|False|False|True|True for this line. This
  // case expected False for it until 2026-10-04, from the interpreter's behaviour, not from Jinja's.
  require(render("{{ s is sequence }}|{{ l is sequence }}|{{ m is sequence }}|{{ n is sequence }}|"
                 "{{ b is sequence }}|{{ u is sequence }}|{{ n is not sequence }}",
                 globals) == "True|True|True|False|False|True|True",
          "is sequence");
  // The template's own expression: structured values through tojson, the rest through str.
  const std::string expr =
      "{% for v in [s, l, m, n] %}{{ v | tojson | safe if v is mapping or (v is sequence and v is not string) "
      "else v | string }};{% endfor %}";
  require(render(expr, globals) == "text;[1, 2];{\"k\": 1};3;", "the template's value rendering: " +
                                                                 render(expr, globals));
}
