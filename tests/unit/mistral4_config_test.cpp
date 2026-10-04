// The Mistral-Small-4 params.json parser: the real file's values parse, the
// architecture registry recognizes a Mistral-native directory, the rope and
// Llama-4 helpers give the references' numbers, and the unsupported shapes
// are refused by name.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "mistral4_config_json.hpp"
#include "models/mistral4/config.hpp"

namespace {

using mistral4_test::patched;
using mistral4_test::refusal;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

bool has(const std::string& msg, const char* needle) {
  return msg.find(needle) != std::string::npos;
}

}  // namespace

DGPP_TEST(mistral4_config_parses_the_release) {
  const dgpp::Mistral4TextConfig c = mistral4_test::release();
  require(c.hidden_size == 4096 && c.vocab_size == 131072 && c.num_hidden_layers == 36, "shape");
  require(c.rms_norm_eps == 1e-6f && !c.tie_word_embeddings && c.max_position_embeddings == 1048576,
          "globals");
  require(c.intermediate_size == 12288, "hidden_dim");
  require(c.bos_token_id == 1 && c.eos_token_id == 2 && c.pad_token_id == 11,
          "the tokenizer's control ids");
  // Latent attention, not GQA: 32 heads over one 256-wide latent and a 64-wide rope key.
  require(c.num_attention_heads == 32 && c.q_lora_rank == 1024 && c.kv_lora_rank == 256, "latents");
  require(c.qk_nope_head_dim == 64 && c.qk_rope_head_dim == 64 && c.v_head_dim == 128, "head dims");
  require(c.qk_head_dim() == 128 && c.kv_a_rows() == 320, "derived dims");
  require(c.rope_theta == 10000.0 && c.yarn_factor == 128.0 &&
              c.yarn_original_max_position_embeddings == 8192,
          "yarn");
  require(c.yarn_beta_fast == 32.0 && c.yarn_beta_slow == 1.0, "yarn band");
  require(c.llama4_scaling_beta == 0.1 && c.llama4_original_max_position_embeddings == 8192,
          "llama 4 scaling");
  require(
      c.moe_intermediate_size == 2048 && c.n_routed_experts == 128 && c.num_experts_per_tok == 4,
      "moe");
  require(
      c.n_shared_experts == 1 && c.shared_expert_inter() == 2048 && c.routed_scaling_factor == 1.0f,
      "shared");
  require(c.fp4_group_size == 16 && c.activation_scales_present, "quantization");
  require(c.vision.present && c.vision.hidden_size == 1024 && c.vision.num_hidden_layers == 24,
          "vision tower");
  require(c.vision.intermediate_size == 4096 && c.vision.patch_size == 14 &&
              c.vision.spatial_merge_size == 2 && c.vision.num_channels == 3 &&
              c.vision.pre_mm_projector_norm,
          "vision shapes");
  const dgpp::GlmMoeConfig m = c.moe_config(2048);
  require(m.hidden == 4096 && m.inter == 2048 && m.n_experts == 128 && m.top_k == 4 &&
              m.n_shared_experts == 1,
          "moe_config");
  require(m.router_mode == dgpp::MoeRouterMode::SoftmaxTopk && m.norm_topk_prob &&
              m.routed_scaling_factor == 1.0f && m.swiglu_limit > 1e30f,
          "moe_config router");
  c.require_kernel_geometry();
}

DGPP_TEST(mistral4_attention_scale_has_no_yarn_magnitude) {
  // apply_scale false: 1 / sqrt(64 + 64) and nothing else. The transformers-main
  // reading would be this times (0.1 ln 128 + 1)^2 = 2.2058.
  const dgpp::Mistral4TextConfig c = mistral4_test::release();
  require(c.attention_scale() == 1.0f / std::sqrt(128.0f), "scale");
  require(std::fabs(c.attention_scale() - 0.08838835f) < 1e-8f, "scale value");
  require(
      has(refusal(patched("\"apply_scale\": false", "\"apply_scale\": true")), "yarn.apply_scale"),
      "apply_scale true is refused");
  // Absent means vLLM's default (true): refused as well.
  require(has(refusal(patched("\"apply_scale\": false,\n", "")), "yarn.apply_scale"),
          "apply_scale absent");
}

DGPP_TEST(mistral4_rope_inv_freq_is_the_yarn_blend) {
  // tools/mistral4_reference.py yarn_inv_freq(64, 1e4, 128, 8192, 32, 1): the band is lanes 12..25.
  const dgpp::Mistral4TextConfig c = mistral4_test::release();
  const std::vector<float> f = c.rope_inv_freq();
  require(f.size() == 32, "32 lanes");
  struct Want {
    int lane;
    float value;
  };
  const Want want[] = {{0, 1.0f},
                       {1, 0x1.7ff222p-1f},
                       {12, 0x1.030dc6p-5f},
                       {13, 0x1.66df6cp-6f},
                       {18, 0x1.8f8aecp-9f},
                       {24, 0x1.60e2dap-14f},
                       {25, 0x1.892918p-18f},
                       {31, 0x1.17a8e4p-20f}};
  for (const Want& w : want) {
    char msg[96];
    std::snprintf(msg, sizeof msg, "lane %d: %a != %a", w.lane,
                  static_cast<double>(f[size_t(w.lane)]), static_cast<double>(w.value));
    require(f[size_t(w.lane)] == w.value, msg);
  }
  // Below the band the plain frequency, above it the frequency / factor.
  const float plain12 =
      1.0f / static_cast<float>(std::pow(10000.0, static_cast<double>(24.0f / 64.0f)));
  require(f[12] == plain12, "lane 12 keeps its frequency");
  const float plain31 =
      1.0f / static_cast<float>(std::pow(10000.0, static_cast<double>(62.0f / 64.0f)));
  require(f[31] == plain31 / 128.0f, "lane 31 is divided by the factor");
}

DGPP_TEST(mistral4_llama4_query_scale) {
  const dgpp::Mistral4TextConfig c = mistral4_test::release();
  require(c.llama4_query_scale(0) == 1.0f && c.llama4_query_scale(8191) == 1.0f,
          "the first 8192 positions are unscaled");
  require(std::fabs(c.llama4_query_scale(8192) - 1.06931471824646f) < 2e-7f, "position 8192");
  require(std::fabs(c.llama4_query_scale(100000) - 1.2564949989318848f) < 2e-7f, "position 100000");
  require(std::fabs(c.llama4_query_scale(1048575) - 1.4852030277252197f) < 2e-7f,
          "the last position");
  require(c.llama4_query_scale(-1) == 1.0f, "a padding row");
}

DGPP_TEST(mistral4_architecture_registry_reads_params_json) {
  const auto p = dgpp::minijson::parse(mistral4_test::kParams);
  require(dgpp::detect_architecture_params(p.root) == dgpp::ModelArchitecture::Mistral4,
          "by shape");
  require(
      std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::Mistral4)) == "mistral4",
      "name");
  // A Mistral-native file of another family (no latent attention) is refused by name.
  const auto dense =
      dgpp::minijson::parse(R"({"dim": 4096, "n_layers": 32, "n_heads": 32, "hidden_dim": 14336})");
  bool refused = false;
  try {
    (void)dgpp::detect_architecture_params(dense.root);
  } catch (const std::exception& e) {
    refused = has(e.what(), "params.json");
  }
  require(refused, "a dense Mistral params.json is refused");
  // The transformers-format release of the same model is refused by name, not guessed at.
  const auto hf = dgpp::minijson::parse(
      R"({"architectures": ["Mistral3ForConditionalGeneration"], "model_type": "mistral3", "text_config": {"model_type": "mistral4"}})");
  refused = false;
  try {
    (void)dgpp::detect_architecture(hf.root);
  } catch (const std::exception& e) {
    refused = has(e.what(), "Mistral-native");
  }
  require(refused, "the transformers-format config names the release to use");
  // The other families are untouched.
  const auto glm = dgpp::minijson::parse(
      R"({"architectures": ["Glm4MoeForCausalLM"], "model_type": "glm4_moe"})");
  require(dgpp::detect_architecture(glm.root) == dgpp::ModelArchitecture::Glm4Moe,
          "glm4 unchanged");

  // On disk: a directory with params.json and no config.json, by directory,
  // by DIR/config.json (the apps' spelling) and by the file itself.
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "dgpp_mistral4_arch_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  {
    std::ofstream f(dir / "params.json");
    f << mistral4_test::kParams;
  }
  require(dgpp::detect_architecture_file(dir.string()) == dgpp::ModelArchitecture::Mistral4,
          "directory");
  require(dgpp::detect_architecture_file((dir / "config.json").string()) ==
              dgpp::ModelArchitecture::Mistral4,
          "DIR/config.json");
  require(dgpp::detect_architecture_file((dir / "params.json").string()) ==
              dgpp::ModelArchitecture::Mistral4,
          "file");
  const dgpp::Mistral4TextConfig c = dgpp::Mistral4TextConfig::from_json_file(dir.string());
  require(c.num_hidden_layers == 36, "from_json_file on a directory");
  // A config.json beside it wins (it is what every other family has).
  {
    std::ofstream f(dir / "config.json");
    f << R"({"architectures": ["Glm4MoeForCausalLM"], "model_type": "glm4_moe"})";
  }
  require(dgpp::detect_architecture_file(dir.string()) == dgpp::ModelArchitecture::Glm4Moe,
          "config.json first");
  fs::remove_all(dir);
  // Neither file: the old error.
  refused = false;
  try {
    (void)dgpp::detect_architecture_file((dir / "config.json").string());
  } catch (const std::exception& e) {
    refused = has(e.what(), "cannot open config");
  }
  require(refused, "a missing config is still a missing config");
}

DGPP_TEST(mistral4_config_refuses_unsupported_shapes) {
  require(has(refusal(patched("\"n_kv_heads\": 32", "\"n_kv_heads\": 8")), "n_kv_heads"),
          "grouped kv heads");
  require(has(refusal(patched("\"head_dim\": 128", "\"head_dim\": 64")), "head_dim"), "head_dim");
  require(has(refusal(patched("\"tied_embeddings\": false", "\"tied_embeddings\": true")),
              "tied_embeddings"),
          "tied");
  require(has(refusal(patched("\"qk_rope_head_dim\": 64", "\"qk_rope_head_dim\": 63")),
              "qk_rope_head_dim"),
          "odd rope");
  require(has(refusal(patched("\"q_lora_rank\": 1024", "\"q_lora_rank\": null")), "q_lora_rank"),
          "unfactored q");
  require(has(refusal(patched("\"first_k_dense_replace\": 0", "\"first_k_dense_replace\": 1")),
              "first_k_dense_replace"),
          "dense layers");
  require(has(refusal(patched("\"route_every_n\": 1", "\"route_every_n\": 2")), "route_every_n"),
          "route_every_n");
  require(has(refusal(patched("\"num_expert_groups\": 1", "\"num_expert_groups\": 4")),
              "num_expert_groups"),
          "groups");
  require(has(refusal(patched("\"num_shared_experts\": 1", "\"num_shared_experts\": 2")),
              "num_shared_experts"),
          "two shared experts");
  require(has(refusal(patched("\"num_experts_per_tok\": 4", "\"num_experts_per_tok\": 200")),
              "num_experts_per_tok"),
          "top-k");
  require(has(refusal(patched("\"expert_hidden_dim\": 2048", "\"expert_hidden_dim\": 2040")),
              "expert_hidden_dim"),
          "an NVFP4 K that is not a multiple of 16");
  // The small closed objects: a key this parser does not know changes the arithmetic.
  require(
      has(refusal(patched("\"alpha\": 1,", "\"alpha\": 1, \"truncate\": false,")), "yarn.truncate"),
      "unknown yarn key");
  require(has(refusal(patched("\"routed_scale\": 1.0",
                              "\"routed_scale\": 1.0, \"renorm_strategy\": \"SCORES\"")),
              "moe.renorm_strategy"),
          "unknown moe key");
  require(has(refusal(patched("\"beta\": 0.1,", "\"beta\": 0.1, \"offset\": 1,")),
              "llama_4_scaling.offset"),
          "unknown llama_4_scaling key");
  require(has(refusal(patched("\"llama_4_scaling\": {", "\"llama_4_scaling_off\": {")),
              "llama_4_scaling"),
          "llama_4_scaling missing");
  require(has(refusal(patched("\"vocab_size\": 131072",
                              "\"vocab_size\": 131072, \"sliding_window\": 4096")),
              "sliding_window"),
          "sliding window");
  require(has(refusal(patched("\"vocab_size\": 131072",
                              "\"vocab_size\": 131072, \"activation\": \"gelu\"")),
              "activation"),
          "activation");
  require(
      has(refusal(patched("\"adapter_bias\": false", "\"adapter_bias\": true")), "adapter_bias"),
      "adapter bias");
  require(
      has(refusal(patched("\"mm_projector_id\": \"patch_merge\"", "\"mm_projector_id\": \"mlp\"")),
          "mm_projector_id"),
      "projector");
  // Text-only checkpoints (no tower) parse.
  const dgpp::Mistral4TextConfig no_vision =
      mistral4_test::parse(patched("\"vision_encoder\": {", "\"vision_encoder_unused\": {"));
  require(!no_vision.vision.present, "no vision tower");
}

DGPP_TEST(mistral4_config_refuses_other_quantization_recipes) {
  require(has(refusal(patched("\"quantization_config\": {", "\"quantization_config_off\": {")),
              "quantization_config"),
          "unquantized");
  require(has(refusal(patched(
                  "\"vocab_size\": 131072",
                  "\"vocab_size\": 131072, \"quantization\": {\"qformat_weight\": \"fp8_e4m3\"}")),
              "FP8 container"),
          "Mistral's FP8 container");
  require(has(refusal(patched("\"quant_method\": \"compressed-tensors\"",
                              "\"quant_method\": \"modelopt\"")),
              "quant_method"),
          "method");
  // The key is parsed whichever of the two spellings the writer used.
  (void)mistral4_test::parse(patched("\"quant_method\": \"compressed-tensors\"",
                                     "\"quant_method\": \"compressed_tensors\""));
  require(has(refusal(patched("\"quantization_status\": \"compressed\"",
                              "\"quantization_status\": \"frozen\"")),
              "quantization_status"),
          "status");
  require(has(refusal(patched("\"transform_config\": {}", "\"transform_config\": {\"h\": 1}")),
              "transform_config"),
          "transforms");
  require(
      has(refusal(patched("\"kv_cache_scheme\": null", "\"kv_cache_scheme\": {\"num_bits\": 8}")),
          "kv_cache_scheme"),
      "kv scheme");
  require(has(refusal(patched("\"Linear\"", "\"re:.*experts.*\"")), "targets"), "module targets");
  // The weights scheme (the second group_size / num_bits of the file).
  require(has(refusal(patched("\"dynamic\": false,\n          \"group_size\": 16",
                              "\"dynamic\": false,\n          \"group_size\": 32")),
              "weights.group_size"),
          "weight group");
  require(has(refusal(patched("\"scale_dtype\": \"torch.float8_e4m3fn\"",
                              "\"scale_dtype\": \"torch.float16\"")),
              "scale_dtype"),
          "scale dtype");
  require(has(refusal(patched("\"dynamic\": \"local\"", "\"dynamic\": true")),
              "input_activations.dynamic"),
          "activation scheme");
  // Weight-only NVFP4 (no activation recipe) parses, and the table then expects no input scales.
  const std::string a16 = patched("\"input_activations\": {", "\"input_activations_off\": {");
  require(!mistral4_test::parse(a16).activation_scales_present, "W4A16");
  // The ignore list decides what is BF16. Dropping the rule that covers the
  // attention projections means they would be NVFP4: not implemented.
  std::string no_attn = patched("\"re:.*self_attn.*\",\n", "");
  require(has(refusal(no_attn), "self_attn.q_b_proj"), "attention must be ignored");
  require(has(refusal(patched("\"re:.*gate$\",\n", "")), "mlp.gate"), "the router must be ignored");
  require(has(refusal(patched("      \"lm_head\"\n", "      \"lm_head_x\"\n")), "lm_head"),
          "the head must be ignored");
  require(has(refusal(patched("\"re:.*gate$\"", "\"re:.*gate.*\"")), "expects NVFP4"),
          "a rule that swallows the experts' gate projections");
  require(has(refusal(patched("\"re:.*gate$\"", "\"re:.*gate$\", \"re:.*shared_experts.*\"")),
              "shared_experts"),
          "a BF16 shared expert");
  require(has(refusal(patched("\"re:.*gate$\"", "\"re:(unclosed\"")), "unparseable regex"),
          "a bad regex");
}

DGPP_TEST(mistral4_kernel_geometry_is_separate_from_the_parse) {
  // The class admits other shapes (the synthetic test checkpoint is one):
  // they parse. The draft device assembly implements the release's.
  std::string small = patched("\"qk_rope_head_dim\": 64", "\"qk_rope_head_dim\": 16");
  const std::string head = "\"head_dim\": 128";
  small.replace(small.find(head), head.size(), "\"head_dim\": 80");
  const dgpp::Mistral4TextConfig t = mistral4_test::parse(small);
  require(t.qk_head_dim() == 80 && t.rope_inv_freq().size() == 8, "a 16-wide rope key parses");
  bool refused = false;
  try {
    t.require_kernel_geometry();
  } catch (const std::invalid_argument& e) {
    refused = has(e.what(), "qk_rope_head_dim");
  }
  require(refused, "a 16-wide rope key has no kernel");
}
