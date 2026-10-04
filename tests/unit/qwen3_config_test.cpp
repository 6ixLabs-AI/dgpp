// The plain Qwen3 family's config parser on the four real config.json files
// (qwen3_config_json.hpp, read from Hugging Face): every consumed field
// lands on the release's value, the architecture registry dispatches the
// three classes, and each unsupported shape is refused with a message that
// names the field.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen3/config.hpp"
#include "qwen3_config_json.hpp"

namespace {

using dgpp::Qwen3Dialect;
using dgpp::Qwen3QuantKind;
using dgpp::Qwen3TextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

Qwen3TextConfig parse(const std::string& json) {
  const auto parsed = dgpp::minijson::parse(json);
  return Qwen3TextConfig::parse(parsed.root);
}

// `json` with the one occurrence of `from` replaced by `to`.
std::string with(const std::string& json, const std::string& from, const std::string& to) {
  const size_t at = json.find(from);
  if (at == std::string::npos) throw std::runtime_error("test bug: '" + from + "' is not in the config");
  if (json.find(from, at + 1) != std::string::npos)
    throw std::runtime_error("test bug: '" + from + "' is in the config twice");
  return json.substr(0, at) + to + json.substr(at + from.size());
}

// The parse must throw, and the message must name `needle`.
void refused(const std::string& json, const std::string& needle) {
  try {
    (void)parse(json);
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find(needle) == std::string::npos)
      throw std::runtime_error("refusal does not name '" + needle + "': " + e.what());
    return;
  }
  throw std::runtime_error("accepted a config that must be refused (" + needle + ")");
}

}  // namespace

DGPP_TEST(qwen3_vl30b_config_parses) {
  const Qwen3TextConfig c = parse(qwen3_config_json::kVl30b);
  require(c.dialect == Qwen3Dialect::VlMoe && c.vl() && c.moe(), "dialect");
  require(std::string(c.family_name()) == "qwen3_vl_moe", "family name");
  require(c.hidden_size == 2048 && c.vocab_size == 151936 && c.num_hidden_layers == 48, "shape");
  require(c.max_position_embeddings == 262144, "max_position_embeddings");
  require(c.rms_norm_eps == 1e-6f && c.hidden_act == "silu" && !c.tie_word_embeddings, "norm / act / tie");
  require(c.num_attention_heads == 32 && c.num_key_value_heads == 4 && c.head_dim == 128, "attention");
  require(c.q_heads_per_kv() == 8, "query heads per kv head");
  require(c.rope_theta == 5e6, "rope_theta");
  require(c.mrope_section == std::vector<int>({24, 20, 20}), "mrope_section");
  require(c.num_experts == 128 && c.num_experts_per_tok == 8 && c.moe_intermediate_size == 768, "moe");
  require(c.norm_topk_prob, "norm_topk_prob");
  require(c.intermediate_size == 6144, "intermediate_size (unused: no dense layer)");
  require(c.quant_kind == Qwen3QuantKind::Nvfp4Packed && c.fp4() && c.fp4_packed(), "quant kind");
  require(!c.kv_cache_scales && !c.bare_names, "no cache scales, prefixed names");
  require(c.eos_token_ids == std::vector<int64_t>({151645}) && c.bos_token_id == 151643, "eos / bos");
  require(c.image_token_id == 151655 && c.video_token_id == 151656, "image / video placeholder ids");
  require(c.vision_start_token_id == 151652 && c.vision_end_token_id == 151653, "vision delimiters");
  require(c.vision.has_value(), "vision tower parsed");
  const dgpp::Qwen3VisionConfig& v = *c.vision;
  require(v.depth == 27 && v.hidden_size == 1152 && v.intermediate_size == 4304 && v.num_heads == 16, "tower shape");
  require(v.in_channels == 3 && v.patch_size == 16 && v.temporal_patch_size == 2 && v.spatial_merge_size == 2, "patch geometry");
  require(v.num_position_embeddings == 2304 && v.out_hidden_size == 2048, "positions / out width");
  require(v.deepstack_visual_indexes == std::vector<int>({8, 16, 24}), "deepstack indexes");
  require(v.merged_width() == 4608, "merger width");
}

DGPP_TEST(qwen3_235b_config_parses) {
  const Qwen3TextConfig c = parse(qwen3_config_json::kMoe235b);
  require(c.dialect == Qwen3Dialect::Moe && !c.vl() && c.moe(), "dialect");
  require(std::string(c.family_name()) == "qwen3_moe", "family name");
  require(c.hidden_size == 4096 && c.vocab_size == 151936 && c.num_hidden_layers == 94, "shape");
  require(c.max_position_embeddings == 262144 && !c.tie_word_embeddings, "positions / tie");
  require(c.num_attention_heads == 64 && c.num_key_value_heads == 4 && c.head_dim == 128, "attention");
  require(c.q_heads_per_kv() == 16, "query heads per kv head");
  require(c.rope_theta == 5e6 && c.mrope_section.empty(), "rope");
  require(c.num_experts == 128 && c.num_experts_per_tok == 8 && c.moe_intermediate_size == 1536, "moe");
  require(c.quant_kind == Qwen3QuantKind::Nvfp4Modelopt && c.fp4() && !c.fp4_packed(), "quant kind");
  require(c.kv_cache_scales, "the recipe's FP8 K/V-cache scales are expected");
  require(!c.vision.has_value() && c.image_token_id == -1, "no tower");
  require(c.eos_token_ids == std::vector<int64_t>({151645}), "eos");
}

DGPP_TEST(qwen3_dense_retrieval_configs_parse) {
  const Qwen3TextConfig r = parse(qwen3_config_json::kReranker);
  const Qwen3TextConfig e = parse(qwen3_config_json::kEmbedding);
  for (const Qwen3TextConfig* c : {&r, &e}) {
    require(c->dialect == Qwen3Dialect::Dense && !c->moe() && !c->fp4(), "dialect");
    require(std::string(c->family_name()) == "qwen3", "family name");
    require(c->hidden_size == 1024 && c->vocab_size == 151669 && c->num_hidden_layers == 28, "shape");
    require(c->num_attention_heads == 16 && c->num_key_value_heads == 8 && c->head_dim == 128, "attention");
    require(c->rope_theta == 1e6 && c->tie_word_embeddings, "rope / tied embeddings");
    require(c->intermediate_size == 3072 && c->num_experts == 0, "dense MLP");
    require(c->quant_kind == Qwen3QuantKind::Bf16 && !c->kv_cache_scales, "unquantized");
  }
  // The two configs differ in exactly these.
  require(r.max_position_embeddings == 40960 && e.max_position_embeddings == 32768, "max_position_embeddings");
  require(r.eos_token_ids == std::vector<int64_t>({151645}) && e.eos_token_ids == std::vector<int64_t>({151643}), "eos");
}

DGPP_TEST(qwen3_architecture_registry_dispatches) {
  using dgpp::ModelArchitecture;
  const auto arch = [](const char* json) {
    const auto parsed = dgpp::minijson::parse(json);
    return dgpp::detect_architecture(parsed.root);
  };
  require(arch(qwen3_config_json::kVl30b) == ModelArchitecture::Qwen3VlMoe, "VL-MoE");
  require(arch(qwen3_config_json::kMoe235b) == ModelArchitecture::Qwen3Moe, "MoE");
  require(arch(qwen3_config_json::kReranker) == ModelArchitecture::Qwen3, "dense (reranker)");
  require(arch(qwen3_config_json::kEmbedding) == ModelArchitecture::Qwen3, "dense (embedding)");
  require(std::string(dgpp::model_architecture_name(ModelArchitecture::Qwen3VlMoe)) == "qwen3_vl_moe", "name");
  require(std::string(dgpp::model_architecture_name(ModelArchitecture::Qwen3Moe)) == "qwen3_moe", "name");
  require(std::string(dgpp::model_architecture_name(ModelArchitecture::Qwen3)) == "qwen3", "name");
  require(arch(R"({"model_type": "qwen3_moe"})") == ModelArchitecture::Qwen3Moe, "by model_type");
  // The neighbours keep their own entries.
  require(arch(R"({"architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next"})") == ModelArchitecture::Qwen3Next, "qwen3_next");
  require(arch(R"({"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe"})") == ModelArchitecture::Qwen3_5, "qwen3_5");
  // The dense VL class is not one of the three: still refused by name.
  bool threw = false;
  try {
    (void)arch(R"({"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl"})");
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("Qwen3VLForConditionalGeneration") != std::string::npos;
  }
  require(threw, "the dense Qwen3-VL class is refused by name");
}

DGPP_TEST(qwen3_config_refuses_unsupported_text_shapes_by_name) {
  const std::string moe = qwen3_config_json::kMoe235b;
  refused(with(moe, "\"head_dim\":128", "\"head_dim\":64"), "head_dim");
  refused(with(moe, "\"hidden_act\":\"silu\"", "\"hidden_act\":\"gelu\""), "hidden_act");
  refused(with(moe, "\"attention_bias\":false", "\"attention_bias\":true"), "attention_bias");
  refused(with(moe, "\"use_sliding_window\":false", "\"use_sliding_window\":true"), "use_sliding_window");
  refused(with(moe, "\"norm_topk_prob\":true", "\"norm_topk_prob\":false"), "norm_topk_prob");
  refused(with(moe, "\"decoder_sparse_step\":1", "\"decoder_sparse_step\":2"), "decoder_sparse_step");
  refused(with(moe, "\"mlp_only_layers\":[]", "\"mlp_only_layers\":[0]"), "mlp_only_layers");
  refused(with(moe, "\"num_experts_per_tok\":8", "\"num_experts_per_tok\":17"), "num_experts_per_tok");
  refused(with(moe, "\"moe_intermediate_size\":1536", "\"moe_intermediate_size\":1552"), "moe_intermediate_size");
  refused(with(moe, "\"rope_scaling\":null", "\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4.0}"), "rope_scaling");
  refused(with(moe, "\"tie_word_embeddings\":false", "\"tie_word_embeddings\":true"), "tie_word_embeddings");
  refused(with(moe, "\"num_key_value_heads\":4", "\"num_key_value_heads\":1"), "num_attention_heads");  // 64 per kv head
  refused(with(moe, "\"model_type\":\"qwen3_moe\"", "\"model_type\":\"qwen3_vl\""), "model_type");
  refused(with(moe, "\"hidden_size\":4096", "\"hidden_size\":4100"), "hidden_size");

  const std::string vl = qwen3_config_json::kVl30b;
  refused(with(vl, "\"mrope_interleaved\":true", "\"mrope_interleaved\":false"), "mrope_interleaved");
  refused(with(vl, "\"rope_type\":\"default\"", "\"rope_type\":\"yarn\""), "rope_type");
  refused(with(vl, "\"out_hidden_size\":2048", "\"out_hidden_size\":4096"), "out_hidden_size");
  refused(with(vl, "\"image_token_id\":151655", "\"image_token_id\":999999"), "image_token_id");
  refused(with(vl, "\"model_type\":\"qwen3_vl_moe_text\"", "\"model_type\":\"qwen3_5_text\""), "model_type");

  const std::string dense = qwen3_config_json::kReranker;
  refused(with(dense, "\"head_dim\":128", "\"head_dim\":64"), "head_dim");
  refused(with(dense, "\"use_sliding_window\":false", "\"use_sliding_window\":true"), "use_sliding_window");
  refused(with(dense, "\"vocab_size\":151669", "\"vocab_size\":151669,\"num_experts\":8"), "num_experts");
  refused(with(dense, "\"vocab_size\":151669",
               "\"vocab_size\":151669,\"quantization_config\":{\"quant_method\":\"modelopt\"}"),
          "quantization_config");
}

DGPP_TEST(qwen3_config_holds_the_recipe_to_the_containers_the_loader_reads) {
  const std::string moe = qwen3_config_json::kMoe235b;
  const std::string vl = qwen3_config_json::kVl30b;
  // No recipe at all: the BF16 original has no expert path.
  {
    // The recipe is the release's last key: cut it and close the object.
    const size_t at = moe.find("\"quantization_config\":");
    require(at != std::string::npos && moe.find("\"vocab_size\"") < at, "test bug: the recipe is the last key");
    refused(moe.substr(0, moe.rfind(',', at)) + "}", "quantization_config");
  }
  refused(with(moe, "\"quant_method\":\"modelopt\"", "\"quant_method\":\"auto-round\""), "format");
  refused(with(moe, "\"quant_algo\":\"NVFP4\"", "\"quant_algo\":\"FP8\""), "quant_algo");
  // The ignore list is the BF16 modules: one the loader expects quantized, or
  // a missing one, is a tensor class it would bind wrongly.
  refused(with(moe, "\"model.layers.7.mlp.gate\",", ""), "layer 7");
  refused(with(moe, "\"model.layers.7.self_attn.k_proj\",", ""), "layer 7");
  refused(with(moe, "\"ignore\":[\"lm_head\",", "\"ignore\":["), "lm_head");
  refused(with(moe, "\"ignore\":[\"lm_head\",", "\"ignore\":[\"lm_head\",\"model.layers.0.self_attn.o_proj\","), "o_proj");
  refused(with(moe, "\"ignore\":[\"lm_head\",", "\"ignore\":[\"lm_head\",\"model.layers.94.mlp.gate\","), "num_hidden_layers");
  refused(with(moe, "\"ignore\":[\"lm_head\",", "\"ignore\":[\"lm_head\",\"model.visual.merger.linear_fc1\","), "vision module");

  refused(with(vl, "\"format\":\"nvfp4-pack-quantized\",\n\"global_compression_ratio\"",
               "\"format\":\"int-quantized\",\n\"global_compression_ratio\""),
          "format");
  refused(with(vl, "\"kv_cache_scheme\":null", "\"kv_cache_scheme\":{\"num_bits\":8,\"type\":\"float\"}"), "kv_cache_scheme");
  refused(with(vl, "\"quantization_status\":\"compressed\"", "\"quantization_status\":\"frozen\""), "quantization_status");
  refused(with(vl, "\"transform_config\":{}", "\"transform_config\":{\"config_groups\":{\"u\":{}}}"), "transform_config");
  // compressed-tensors quantizes q/k/v: an ignored one would ship BF16.
  refused(with(vl, "\"model.language_model.layers.0.mlp.gate\",",
               "\"model.language_model.layers.0.mlp.gate\",\"model.language_model.layers.0.self_attn.q_proj\","),
          "q_proj");
  refused(with(vl, "\"model.language_model.layers.47.mlp.gate\",", ""), "layer 47");
}

DGPP_TEST(qwen3_config_views_for_the_shared_layers) {
  const Qwen3TextConfig c = parse(qwen3_config_json::kMoe235b);
  // The routed chain: softmax top-k, no shared expert, no scaling, no clamps.
  const dgpp::GlmMoeConfig m = c.moe_config(768);
  require(m.hidden == 4096 && m.inter == 768 && m.n_experts == 128 && m.top_k == 8, "moe geometry");
  require(m.n_shared_experts == 0 && m.router_mode == dgpp::MoeRouterMode::SoftmaxTopk, "routed-only softmax");
  require(m.routed_scaling_factor == 1.0f && m.norm_topk_prob && std::isinf(m.swiglu_limit), "no scaling, no clamps");
  dgpp::GlmMoeConfig::validate_config(m);
  // The GLM-4.7 attention layer's view: unbiased, head-normed, rotary over the whole head.
  const dgpp::Glm4TextConfig a = c.attention_view();
  require(a.hidden_size == 4096 && a.num_attention_heads == 64 && a.num_key_value_heads == 4, "attention geometry");
  require(a.head_dim == 128 && a.rotary_dim == 128, "full rotary: rotary_dim == head_dim");
  require(!a.attention_bias && a.use_qk_norm, "no biases, q/k head norms");
  require(a.rope_theta == 5e6 && a.rms_norm_eps == 1e-6f, "theta / eps");
  require(a.first_k_dense_replace == 0 && a.mtp_layer() == -1 && a.n_shared_experts == 0, "no dense layers, no draft, no shared expert");
}
