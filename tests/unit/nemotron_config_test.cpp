// The Nemotron-H config parser: both releases' files parse (Nano's
// hf_quant_config.json recipe, Super's quantization_config layer map), the
// derived Mamba2 geometry and per-module weight formats, the architecture
// dispatch, and the unsupported shapes and recipes are refused by name.
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/nemotron/config.hpp"
#include "nemotron_config_json.hpp"

namespace {

using dgpp::NemotronLayerKind;
using dgpp::NemotronQuant;
using nemotron_test::patched;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::string refusal(const std::string& config, const std::string& hf_quant) {
  try {
    (void)nemotron_test::parse(config, hf_quant);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}
// A Nano config with one anchor rewritten, under the release's recipe.
std::string nano_refusal(const std::string& from, const std::string& to) {
  return refusal(patched(nemotron_test::kNanoConfig, from, to), nemotron_test::kNanoHfQuant);
}
std::string nano_recipe_refusal(const std::string& from, const std::string& to) {
  return refusal(nemotron_test::kNanoConfig, patched(nemotron_test::kNanoHfQuant, from, to));
}
bool has(const std::string& msg, const char* needle) {
  return msg.find(needle) != std::string::npos;
}

std::vector<int> layers_of(const dgpp::NemotronHConfig& c, NemotronLayerKind k) {
  std::vector<int> out;
  for (int l = 0; l < c.num_hidden_layers; ++l)
    if (c.layers[static_cast<size_t>(l)] == k) out.push_back(l);
  return out;
}

std::filesystem::path landed_snapshot(const char* repo) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root = fs::path(home) / ".cache/huggingface/hub" / repo / "snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (fs::exists(snap.path() / "config.json")) return snap.path();
  return {};
}

}  // namespace

DGPP_TEST(nemotron_config_parses_nano) {
  const dgpp::NemotronHConfig c = nemotron_test::nano();
  require(c.hidden_size == 2688 && c.vocab_size == 131072 && c.num_hidden_layers == 52, "shape");
  require(c.layer_norm_eps == 1e-5f && c.max_position_embeddings == 262144, "eps / positions");
  require(c.count(NemotronLayerKind::Mamba) == 23 && c.count(NemotronLayerKind::Moe) == 23 &&
              c.count(NemotronLayerKind::Attention) == 6,
          "layer kinds");
  require(layers_of(c, NemotronLayerKind::Attention) == std::vector<int>{5, 12, 19, 26, 33, 42},
          "attention layers");
  require(c.kind_of(0) == NemotronLayerKind::Mamba && c.kind_of(1) == NemotronLayerKind::Moe &&
              c.kind_of(51) == NemotronLayerKind::Moe,
          "pattern ends");
  require(c.eos_token_ids == std::vector<int64_t>{2} && c.bos_token_id == 1 && c.pad_token_id == 0,
          "tokens");
  // Mamba2: the inner width is heads x head_dim, not expand x hidden.
  require(c.mamba_num_heads == 64 && c.mamba_head_dim == 64 && c.ssm_state_size == 128 &&
              c.mamba_n_groups == 8,
          "mamba");
  require(c.conv_kernel == 4 && c.conv_bias && c.chunk_size == 128 && c.time_step_min == 0.001,
          "mamba conv / step");
  require(c.mamba_inner() == 4096 && c.mamba_conv_dim() == 6144 && c.mamba_proj_rows() == 10304 &&
              c.mamba_norm_group() == 512,
          "mamba geometry");
  require(c.num_attention_heads == 32 && c.num_key_value_heads == 2 && c.head_dim == 128,
          "attention");
  require(c.n_routed_experts == 128 && c.num_experts_per_tok == 6 &&
              c.moe_intermediate_size == 1856 && c.moe_shared_expert_intermediate_size == 3712,
          "moe");
  require(c.moe_latent_size == 0 && c.expert_width() == 2688, "no latent");
  require(c.routed_scaling_factor == 2.5f && c.norm_topk_prob, "router");
  require(c.num_nextn_predict_layers == 0 && c.mtp_layers.empty() && c.num_layers_total() == 52 &&
              !c.is_mtp_layer(52),
          "no draft block");
  // The recipe: NVFP4 on every backbone linear but the excluded ones.
  require(c.recipe == dgpp::NemotronRecipe::Nvfp4Exclude && c.router_dtype == dgpp::DType::F32 &&
              !c.kv_cache_scales,
          "recipe");
  // 17 Mamba layers x 2 + 23 MoE layers x (128 x 2 + 2).
  require(c.quant.size() == 17 * 2 + 23 * 258,
          "quantized modules " + std::to_string(c.quant.size()));
  require(c.quant_of("backbone.layers.0.mixer.in_proj") == NemotronQuant::Nvfp4, "layer 0 in_proj");
  require(c.quant_of("backbone.layers.4.mixer.in_proj") == NemotronQuant::Bf16 &&
              c.quant_of("backbone.layers.4.mixer.out_proj") == NemotronQuant::Bf16,
          "the Mamba layer before an attention layer is excluded");
  require(c.quant_of("backbone.layers.5.mixer.q_proj") == NemotronQuant::Bf16 &&
              c.quant_of("backbone.layers.5.mixer.o_proj") == NemotronQuant::Bf16,
          "attention is excluded");
  require(c.quant_of("backbone.layers.1.mixer.experts.127.down_proj") == NemotronQuant::Nvfp4 &&
              c.quant_of("backbone.layers.51.mixer.shared_experts.up_proj") == NemotronQuant::Nvfp4,
          "experts");
  require(c.quant_of("lm_head") == NemotronQuant::Bf16, "the head");
}

DGPP_TEST(nemotron_config_parses_super) {
  const dgpp::NemotronHConfig c = nemotron_test::super();
  require(c.hidden_size == 4096 && c.vocab_size == 131072 && c.num_hidden_layers == 88, "shape");
  require(c.count(NemotronLayerKind::Mamba) == 40 && c.count(NemotronLayerKind::Moe) == 40 &&
              c.count(NemotronLayerKind::Attention) == 8,
          "layer kinds");
  require(
      layers_of(c, NemotronLayerKind::Attention) == std::vector<int>{7, 16, 25, 36, 47, 58, 69, 78},
      "attention layers");
  require(layers_of(c, NemotronLayerKind::Moe) == nemotron_test::kSuperMoeLayers, "moe layers");
  require(c.mamba_num_heads == 128 && c.mamba_head_dim == 64 && c.ssm_state_size == 128 &&
              c.mamba_n_groups == 8,
          "mamba");
  require(c.mamba_inner() == 8192 && c.mamba_conv_dim() == 10240 && c.mamba_proj_rows() == 18560 &&
              c.mamba_norm_group() == 1024,
          "mamba geometry");
  require(c.num_attention_heads == 32 && c.num_key_value_heads == 2 && c.head_dim == 128,
          "attention");
  require(c.n_routed_experts == 512 && c.num_experts_per_tok == 22 &&
              c.moe_intermediate_size == 2688 && c.moe_shared_expert_intermediate_size == 5376,
          "moe");
  require(c.moe_latent_size == 1024 && c.expert_width() == 1024, "the latent");
  require(c.routed_scaling_factor == 5.0f, "router scale");
  // The draft block: two layers past the backbone.
  require(c.num_nextn_predict_layers == 1 &&
              c.mtp_layers == std::vector<NemotronLayerKind>{NemotronLayerKind::Attention,
                                                             NemotronLayerKind::Moe},
          "draft block");
  require(c.num_layers_total() == 90 && c.is_mtp_layer(88) && c.is_mtp_layer(89) &&
              !c.is_mtp_layer(87) && !c.is_mtp_layer(90),
          "layer space");
  require(c.kind_of(88) == NemotronLayerKind::Attention && c.kind_of(89) == NemotronLayerKind::Moe,
          "draft kinds");
  require(dgpp::nemotron_layer_prefix(c, 87) == "backbone.layers.87." &&
              dgpp::nemotron_layer_prefix(c, 88) == "mtp.layers.0." &&
              dgpp::nemotron_layer_prefix(c, 89) == "mtp.layers.1.",
          "prefixes");
  // The recipe: a format per module.
  require(c.recipe == dgpp::NemotronRecipe::MixedPrecision && c.router_dtype == dgpp::DType::BF16 &&
              c.kv_cache_scales,
          "recipe");
  size_t fp8 = 0, fp4 = 0;
  for (const auto& [name, q] : c.quant) (q == NemotronQuant::Fp8 ? fp8 : fp4)++;
  require(fp8 == 139 && fp4 == 40961,
          "formats: fp8 " + std::to_string(fp8) + ", nvfp4 " + std::to_string(fp4));
  require(c.quant_of("backbone.layers.0.mixer.in_proj") == NemotronQuant::Fp8 &&
              c.quant_of("backbone.layers.0.mixer.out_proj") == NemotronQuant::Fp8,
          "layer 0");
  // Two Mamba layers differ: layer 6 has an FP8 in_proj and a BF16 out_proj.
  require(c.quant_of("backbone.layers.6.mixer.in_proj") == NemotronQuant::Fp8 &&
              c.quant_of("backbone.layers.6.mixer.out_proj") == NemotronQuant::Bf16,
          "layer 6");
  require(c.quant_of("backbone.layers.22.mixer.in_proj") == NemotronQuant::Bf16, "layer 22");
  require(c.quant_of("backbone.layers.69.mixer.o_proj") == NemotronQuant::Fp8 &&
              c.quant_of("backbone.layers.7.mixer.o_proj") == NemotronQuant::Bf16 &&
              c.quant_of("backbone.layers.69.mixer.q_proj") == NemotronQuant::Bf16,
          "attention");
  require(
      c.quant_of("backbone.layers.1.mixer.shared_experts.down_proj") == NemotronQuant::Nvfp4 &&
          c.quant_of("backbone.layers.3.mixer.shared_experts.down_proj") == NemotronQuant::Fp8 &&
          c.quant_of("backbone.layers.8.mixer.shared_experts.down_proj") == NemotronQuant::Bf16 &&
          c.quant_of("backbone.layers.41.mixer.shared_experts.up_proj") == NemotronQuant::Bf16,
      "the shared expert's three forms");
  require(c.quant_of("backbone.layers.1.mixer.fc1_latent_proj") == NemotronQuant::Fp8 &&
              c.quant_of("backbone.layers.8.mixer.fc1_latent_proj") == NemotronQuant::Bf16 &&
              c.quant_of("backbone.layers.3.mixer.fc2_latent_proj") == NemotronQuant::Fp8,
          "latent projections");
  require(c.quant_of("backbone.layers.87.mixer.experts.511.up_proj") == NemotronQuant::Nvfp4,
          "experts");
  require(c.quant_of("mtp.layers.1.mixer.experts.0.up_proj") == NemotronQuant::Bf16 &&
              c.quant_of("mtp.layers.0.mixer.o_proj") == NemotronQuant::Bf16,
          "the draft block is BF16");
  // The same config without the second copy of the recipe.
  const dgpp::NemotronHConfig alone = nemotron_test::parse(nemotron_test::super_config_json(), "");
  require(alone.quant.size() == c.quant.size() && alone.kv_cache_scales,
          "config.json alone carries the recipe");
}

DGPP_TEST(nemotron_layer_linears_name_the_quantizable_modules) {
  const dgpp::NemotronHConfig n = nemotron_test::nano();
  require(dgpp::nemotron_layer_linears(n, 0) ==
              std::vector<std::string>{"backbone.layers.0.mixer.in_proj",
                                       "backbone.layers.0.mixer.out_proj"},
          "mamba");
  require(dgpp::nemotron_layer_linears(n, 5).size() == 4 &&
              dgpp::nemotron_layer_linears(n, 1).size() == 258,
          "nano counts");
  const dgpp::NemotronHConfig s = nemotron_test::super();
  const auto moe = dgpp::nemotron_layer_linears(s, 1);
  require(moe.size() == 512 * 2 + 4 && moe.back() == "backbone.layers.1.mixer.fc2_latent_proj",
          "super moe");
  require(dgpp::nemotron_layer_linears(s, 88).front() == "mtp.layers.0.mixer.q_proj",
          "draft attention");
  bool refused = false;
  try {
    (void)dgpp::nemotron_layer_linears(s, 90);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "a layer past the draft block");
}

DGPP_TEST(nemotron_config_refuses_unsupported_shapes) {
  require(has(nano_refusal("\"model_type\": \"nemotron_h\"", "\"model_type\": \"mamba2\""),
              "model_type"),
          "model_type");
  // The pattern: a dense-MLP layer, an unknown character, a wrong length.
  require(has(nano_refusal("\"MEMEM*EMEMEM*", "\"ME-EM*EMEMEM*"), "dense-MLP"), "dense MLP layer");
  require(has(nano_refusal("\"MEMEM*EMEMEM*", "\"MEMEA*EMEMEM*"), "hybrid_override_pattern"),
          "unknown layer character");
  require(has(nano_refusal("\"num_hidden_layers\": 52", "\"num_hidden_layers\": 51"),
              "hybrid_override_pattern"),
          "pattern length");
  require(has(nano_refusal("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true"),
              "tie_word_embeddings"),
          "tied");
  require(has(nano_refusal("\"residual_in_fp32\": false", "\"residual_in_fp32\": true"),
              "residual_in_fp32"),
          "fp32 residual");
  require(has(nano_refusal("\"norm_eps\": 1e-05", "\"norm_eps\": 1e-06"), "norm_eps"),
          "two epsilons");
  require(has(nano_refusal("\"layer_norm_epsilon\": 1e-05,", ""), "layer_norm_epsilon"),
          "missing epsilon");
  // Mamba2.
  require(has(nano_refusal("\"n_groups\": 8", "\"n_groups\": 5"), "n_groups"),
          "groups divide heads");
  require(has(nano_refusal("\"conv_kernel\": 4", "\"conv_kernel\": 1"), "conv_kernel"), "kernel");
  require(has(nano_refusal("\"use_conv_bias\": true", "\"use_conv_bias\": false"), "use_conv_bias"),
          "conv bias");
  require(has(nano_refusal("\"mamba_hidden_act\": \"silu\"", "\"mamba_hidden_act\": \"gelu\""),
              "mamba_hidden_act"),
          "mamba act");
  require(has(nano_refusal("\"use_bias\": false", "\"use_bias\": true"), "use_bias"),
          "projection bias");
  require(has(nano_refusal("\"mamba_ssm_cache_dtype\": \"float32\"",
                           "\"mamba_ssm_cache_dtype\": \"float16\""),
              "mamba_ssm_cache_dtype"),
          "state dtype");
  require(has(nano_refusal("\"time_step_min\": 0.001,",
                           "\"time_step_min\": 0.001, \"time_step_limit\": [0.0, 1.0],"),
              "time_step_limit"),
          "a time-step clamp");
  require(has(nano_refusal("\"n_groups\": 8", "\"n_groups\": 8, \"mamba_n_groups\": 4"),
              "mamba_n_groups"),
          "the two names of a Mamba field must agree");
  // Attention.
  require(has(nano_refusal("\"num_key_value_heads\": 2", "\"num_key_value_heads\": 5"),
              "num_key_value_heads"),
          "kv heads");
  require(
      has(nano_refusal("\"attention_bias\": false", "\"attention_bias\": true"), "attention_bias"),
      "attention bias");
  require(
      has(nano_refusal("\"sliding_window\": null", "\"sliding_window\": 4096"), "sliding_window"),
      "sliding window");
  // MoE.
  require(has(nano_refusal("\"mlp_hidden_act\": \"relu2\"", "\"mlp_hidden_act\": \"silu\""),
              "mlp_hidden_act"),
          "expert act");
  require(has(nano_refusal("\"mlp_bias\": false", "\"mlp_bias\": true"), "mlp_bias"),
          "expert bias");
  require(
      has(nano_refusal("\"n_shared_experts\": 1", "\"n_shared_experts\": 2"), "n_shared_experts"),
      "shared experts");
  require(has(nano_refusal("\"n_group\": 1", "\"n_group\": 2"), "n_group"), "group routing");
  require(
      has(nano_refusal("\"norm_topk_prob\": true", "\"norm_topk_prob\": false"), "norm_topk_prob"),
      "renormalization");
  require(has(nano_refusal("\"num_experts_per_tok\": 6", "\"num_experts_per_tok\": 129"),
              "num_experts_per_tok"),
          "top-k");
  // MTP: only the attention + MoE block, and only one.
  const std::string super = nemotron_test::super_config_json();
  require(has(refusal(patched(super, "\"mtp_hybrid_override_pattern\": \"*E\"",
                              "\"mtp_hybrid_override_pattern\": \"ME\""),
                      ""),
              "mtp_hybrid_override_pattern"),
          "draft pattern");
  require(has(refusal(patched(super, "\"num_nextn_predict_layers\": 1",
                              "\"num_nextn_predict_layers\": 2"),
                      ""),
              "num_nextn_predict_layers"),
          "draft count");
  // A re-saved config writes the list form; it parses to the same kinds,
  // and a list that disagrees with the pattern is refused.
  const dgpp::NemotronHConfig nano = nemotron_test::nano();
  std::string list;
  for (const NemotronLayerKind k : nano.layers) {
    if (!list.empty()) list += ", ";
    list += k == NemotronLayerKind::Mamba ? "\"mamba\""
            : k == NemotronLayerKind::Moe ? "\"moe\""
                                          : "\"attention\"";
  }
  const dgpp::NemotronHConfig listed = nemotron_test::parse(
      patched(
          nemotron_test::kNanoConfig,
          "\"hybrid_override_pattern\": \"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME\"",
          "\"layers_block_type\": [" + list + "]"),
      nemotron_test::kNanoHfQuant);
  require(listed.layers == nano.layers && listed.quant == nano.quant, "layers_block_type");
  const dgpp::NemotronHConfig mtp_listed =
      nemotron_test::parse(patched(super, "\"mtp_hybrid_override_pattern\": \"*E\"",
                                   "\"mtp_layers_block_type\": [\"attention\", \"moe\"]"),
                           "");
  require(mtp_listed.mtp_layers == nemotron_test::super().mtp_layers, "mtp_layers_block_type");
  require(has(nano_refusal("\"num_hidden_layers\": 52",
                           "\"num_hidden_layers\": 52, \"layers_block_type\": [\"mamba\"]"),
              "layers_block_type"),
          "list against pattern");
}

DGPP_TEST(nemotron_config_refuses_unsupported_recipes) {
  // No recipe at all: an unquantized checkpoint has no table here.
  require(has(refusal(nemotron_test::kNanoConfig, ""), "quantization_config"), "no recipe");
  // The exclude form.
  require(has(nano_recipe_refusal("\"quant_algo\": \"NVFP4\"", "\"quant_algo\": \"FP8\""),
              "quant_algo"),
          "algo");
  require(has(nano_recipe_refusal("\"group_size\": 16", "\"group_size\": 32"), "group_size"),
          "group size");
  require(has(nano_recipe_refusal("\"kv_cache_quant_algo\": \"FP8\"",
                                  "\"kv_cache_quant_algo\": \"NVFP4\""),
              "kv_cache_quant_algo"),
          "cache algo");
  require(has(nano_recipe_refusal("\"lm_head\",", ""), "lm_head"), "the head must stay BF16");
  require(has(nano_recipe_refusal("\"backbone.layers.4.mixer.in_proj\"",
                                  "\"backbone.layers.4.mixer.in_projection\""),
              "names no module"),
          "an exclude entry that names nothing");
  require(has(nano_recipe_refusal("\"backbone.layers.0.mixer.conv1d\"",
                                  "\"backbone.layers.1.mixer.conv1d\""),
              "names no module"),
          "a convolution on a layer that has none");
  require(has(nano_recipe_refusal("\"name\": \"modelopt\"", "\"name\": \"llm-compressor\""),
              "producer"),
          "producer");
  // The exclude list is what makes a module BF16: one entry fewer, one more NVFP4 module.
  const dgpp::NemotronHConfig more = nemotron_test::parse(
      nemotron_test::kNanoConfig,
      patched(nemotron_test::kNanoHfQuant, "\"backbone.layers.5.mixer.o_proj\",", ""));
  require(more.quant.size() == 17 * 2 + 23 * 258 + 1 &&
              more.quant_of("backbone.layers.5.mixer.o_proj") == NemotronQuant::Nvfp4,
          "the exclude list drives the formats");

  // The layer-map form.
  const std::string super = nemotron_test::super_config_json();
  const auto super_refusal = [&](const std::string& from, const std::string& to,
                                 const std::string& hf = "") {
    return refusal(patched(super, from, to), hf);
  };
  require(has(super_refusal("\"quant_method\":\"modelopt\"", "\"quant_method\":\"fp8\""),
              "quant_method"),
          "method");
  require(has(super_refusal("\"quant_algo\":\"MIXED_PRECISION\"", "\"quant_algo\":\"NVFP4\""),
              "quant_algo"),
          "algo");
  require(has(super_refusal("\"ignore\":[]", "\"ignore\":[\"lm_head\"]"), "ignore"), "ignore list");
  require(has(super_refusal("\"kv_cache_scheme\":{\"dynamic\":false",
                            "\"kv_cache_scheme\":{\"dynamic\":true"),
              "kv_cache_scheme"),
          "dynamic cache scales");
  // A module the table cannot carry a format for: the draft block, the
  // head, a layer of another kind.
  const std::string first =
      "\"quantized_layers\":{\"backbone.layers.0.mixer.in_proj\":{\"quant_algo\":\"FP8\"}";
  const auto with_entry = [&](const std::string& entry) {
    // Drop the config_groups so the layer map is what refuses.
    return refusal(patched(patched(super, first, "\"quantized_layers\":{" + entry),
                           "\"config_groups\":{", "\"unused_groups\":{"),
                   "");
  };
  require(has(with_entry("\"mtp.layers.0.mixer.o_proj\":{\"quant_algo\":\"FP8\"}"),
              "not a backbone linear"),
          "draft module");
  require(has(with_entry("\"lm_head\":{\"quant_algo\":\"FP8\"}"), "not a backbone linear"),
          "the head");
  require(has(with_entry("\"backbone.layers.0.mixer.q_proj\":{\"quant_algo\":\"FP8\"}"),
              "not a backbone linear"),
          "an attention module on a Mamba layer");
  require(has(with_entry("\"backbone.layers.0.mixer.in_proj\":{\"quant_algo\":\"INT8\"}"),
              "only FP8 and NVFP4"),
          "format");
  require(
      has(with_entry(
              "\"backbone.layers.0.mixer.in_proj\":{\"quant_algo\":\"NVFP4\",\"group_size\":32}"),
          "group_size 16"),
      "NVFP4 group");
  // The target lists and the layer map must agree, and so must the second
  // copy of the recipe.
  require(has(refusal(patched(super, first,
                              "\"quantized_layers\":{\"backbone.layers.0.mixer.in_proj\":{\"quant_"
                              "algo\":\"NVFP4\",\"group_size\":16}"),
                      ""),
              "disagrees with quantized_layers"),
          "config_groups against the layer map");
  require(has(refusal(super,
                      patched(nemotron_test::super_hf_quant(),
                              "\"backbone.layers.0.mixer.in_proj\":{\"quant_algo\":\"FP8\"},", "")),
              "hf_quant_config"),
          "hf_quant_config.json with a layer missing");
  require(has(refusal(super, nemotron_test::kNanoHfQuant), "hf_quant_config"),
          "the other release's recipe file");
}

DGPP_TEST(nemotron_architecture_dispatch) {
  for (const std::string& text :
       {std::string(nemotron_test::kNanoConfig), nemotron_test::super_config_json()}) {
    const auto root = dgpp::minijson::parse(text);
    require(dgpp::detect_architecture(root.root) == dgpp::ModelArchitecture::NemotronH, "detect");
  }
  const auto by_type = dgpp::minijson::parse(R"({"model_type": "nemotron_h"})");
  require(dgpp::detect_architecture(by_type.root) == dgpp::ModelArchitecture::NemotronH,
          "by model_type");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::NemotronH)) ==
              "nemotron_h",
          "name");
  const auto next = dgpp::minijson::parse(
      R"({"architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next"})");
  require(dgpp::detect_architecture(next.root) == dgpp::ModelArchitecture::Qwen3Next,
          "qwen3_next unchanged");
}

DGPP_TEST(nemotron_config_reads_the_landed_checkpoints) {
  if (const auto snap = landed_snapshot("models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4");
      !snap.empty()) {
    const std::string cfg = (snap / "config.json").string();
    require(dgpp::detect_architecture_file(cfg) == dgpp::ModelArchitecture::NemotronH, "nano arch");
    const dgpp::NemotronHConfig c = dgpp::NemotronHConfig::from_json_file(cfg);
    require(c.num_hidden_layers == 52 && c.recipe == dgpp::NemotronRecipe::Nvfp4Exclude &&
                c.quant.size() == 5968,
            "nano landed values");
    require(c.quant == nemotron_test::nano().quant,
            "nano: the landed recipe is the transcribed one");
  }
  if (const auto snap = landed_snapshot("models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4");
      !snap.empty()) {
    const std::string cfg = (snap / "config.json").string();
    require(dgpp::detect_architecture_file(cfg) == dgpp::ModelArchitecture::NemotronH,
            "super arch");
    const dgpp::NemotronHConfig c = dgpp::NemotronHConfig::from_json_file(cfg);
    require(c.num_hidden_layers == 88 && c.recipe == dgpp::NemotronRecipe::MixedPrecision &&
                c.quant.size() == 41100 && c.mtp_layers.size() == 2,
            "super landed values");
    require(c.quant == nemotron_test::super().quant,
            "super: the landed layer map is the transcribed one");
  }
}
