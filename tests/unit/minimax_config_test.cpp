// The MiniMax-M2 config parser: the real file's values parse, the
// architecture registry dispatches on it, and the unsupported shapes are
// refused by name.
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "minimax_config_json.hpp"
#include "models/minimax/config.hpp"

namespace {

using minimax_test::config_json;
using minimax_test::refusal;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

bool has(const std::string& msg, const char* needle) {
  return msg.find(needle) != std::string::npos;
}

}  // namespace

DGPP_TEST(minimax_config_parses_the_release) {
  const dgpp::MinimaxTextConfig c = minimax_test::release();
  require(c.hidden_size == 3072 && c.vocab_size == 200064 && c.num_hidden_layers == 62, "shape");
  require(c.rms_norm_eps == 1e-6f && !c.tie_word_embeddings && c.max_position_embeddings == 196608,
          "globals");
  require(c.num_attention_heads == 48 && c.num_key_value_heads == 8 && c.head_dim == 128,
          "attention");
  require(c.q_heads_per_kv() == 6 && c.q_width() == 6144 && c.kv_width() == 1024, "gqa");
  require(c.rotary_dim == 64 && c.rope_theta == 5e6, "rope");
  require(c.attention_scale() == 1.0f / std::sqrt(128.0f), "scale");
  require(
      c.moe_intermediate_size == 1536 && c.n_routed_experts == 256 && c.num_experts_per_tok == 8,
      "moe");
  require(c.fp4_group_size == 16, "quantization");
  // config.json's token ids are byte tokens, not the tokenizer's control ids.
  require(c.config_bos_token_id == 1 && c.config_eos_token_id == 2,
          "the file's own (stale) token ids");
  // The file declares a draft head; nothing ships one (the binding table has none).
  require(c.mtp_declared && c.mtp_modules_declared == 3, "the declared MTP head is recorded");
  const dgpp::GlmMoeConfig m = c.moe_config(768);
  require(m.hidden == 3072 && m.inter == 768 && m.n_experts == 256 && m.top_k == 8, "moe_config");
  require(m.n_shared_experts == 0 && m.routed_scaling_factor == 1.0f && m.norm_topk_prob,
          "no shared expert, no scale");
  require(m.router_mode == dgpp::MoeRouterMode::SigmoidBias && m.swiglu_limit > 1e30f,
          "moe_config router");
  c.require_kernel_geometry();
  // The rope: 32 lanes, 1 / theta^(i / 32).
  const std::vector<float> f = c.rope_inv_freq();
  require(f.size() == 32 && f[0] == 1.0f, "inv_freq lanes");
  require(std::fabs(f[16] - static_cast<float>(1.0 / std::sqrt(5e6))) < 1e-9f,
          "inv_freq[16] = theta^-0.5");
  require(std::fabs(f[31] - static_cast<float>(std::pow(5e6, -31.0 / 32.0))) < 1e-12f,
          "inv_freq[31]");
}

DGPP_TEST(minimax_architecture_registry_dispatches) {
  const auto g = dgpp::minijson::parse(
      R"({"architectures": ["MiniMaxM2ForCausalLM"], "model_type": "minimax_m2"})");
  require(dgpp::detect_architecture(g.root) == dgpp::ModelArchitecture::MiniMaxM2, "by class");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::MiniMaxM2)) ==
              "minimax_m2",
          "name");
  const auto t = dgpp::minijson::parse(R"({"model_type": "minimax_m2"})");
  require(dgpp::detect_architecture(t.root) == dgpp::ModelArchitecture::MiniMaxM2, "by model_type");
  const auto full = dgpp::minijson::parse(config_json());
  require(dgpp::detect_architecture(full.root) == dgpp::ModelArchitecture::MiniMaxM2,
          "the release's config");
  const auto mimo =
      dgpp::minijson::parse(R"({"architectures": ["MiMoV2ForCausalLM"], "model_type": "mimo_v2"})");
  require(dgpp::detect_architecture(mimo.root) == dgpp::ModelArchitecture::MimoV2,
          "mimo unchanged");
  const auto glm4 = dgpp::minijson::parse(
      R"({"architectures": ["Glm4MoeForCausalLM"], "model_type": "glm4_moe"})");
  require(dgpp::detect_architecture(glm4.root) == dgpp::ModelArchitecture::Glm4Moe,
          "glm4 unchanged");
}

DGPP_TEST(minimax_config_refuses_unsupported_shapes) {
  require(
      has(refusal(config_json("\"qk_norm_type\": \"per_layer\"", "\"qk_norm_type\": \"per_head\"")),
          "qk_norm_type"),
      "per-head norm");
  require(
      has(refusal(config_json("\"use_qk_norm\": true", "\"use_qk_norm\": false")), "use_qk_norm"),
      "no qk norm");
  require(
      has(refusal(config_json("\"scoring_func\": \"sigmoid\"", "\"scoring_func\": \"softmax\"")),
          "scoring_func"),
      "softmax router");
  require(has(refusal(config_json("\"use_routing_bias\": true", "\"use_routing_bias\": false")),
              "use_routing_bias"),
          "no routing bias");
  require(has(refusal(config_json("\"shared_intermediate_size\": 0",
                                  "\"shared_intermediate_size\": 1536")),
              "shared_intermediate_size"),
          "a shared expert");
  require(has(refusal(config_json("\"sliding_window\": null", "\"sliding_window\": 4096")),
              "sliding_window"),
          "sliding window");
  require(has(refusal(config_json("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"")),
              "hidden_act"),
          "act");
  require(
      has(refusal(config_json("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true")),
          "tie_word_embeddings"),
      "tied");
  require(has(refusal(config_json("\"num_key_value_heads\": 8", "\"num_key_value_heads\": 7")),
              "num_key_value_heads"),
          "kv heads");
  require(has(refusal(config_json("\"num_experts_per_tok\": 8", "\"num_experts_per_tok\": 32")),
              "num_experts_per_tok"),
          "top-k");
  require(has(refusal(config_json("\"intermediate_size\": 1536", "\"intermediate_size\": 1530")),
              "intermediate_size"),
          "an NVFP4 K that is not a multiple of 16");
  require(has(refusal(config_json("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"")),
              "rope_type"),
          "rope type");
  // The rotary dim is stated three ways; they must agree.
  require(has(refusal(config_json("\"rotary_dim\": 64", "\"rotary_dim\": 128")), "rotary_dim"),
          "rotary_dim");
  require(has(refusal(config_json("\"rope_theta\": 5000000,\n    \"partial_rotary_factor\": 0.5",
                                  "\"rope_theta\": 5000000,\n    \"partial_rotary_factor\": 1.0")),
              "partial_rotary_factor"),
          "partial rotary factor");
  require(has(refusal(config_json("\"rope_type\": \"default\",\n    \"rope_theta\": 5000000",
                                  "\"rope_type\": \"default\",\n    \"rope_theta\": 1000000")),
              "rope_theta"),
          "rope theta");
  // Every layer is full attention.
  require(has(refusal(config_json("", "", minimax_test::ignore_json(),
                                  minimax_test::attn_types_json(61))),
              "attn_type_list"),
          "attn_type_list length");
  require(has(refusal(config_json("", "", minimax_test::ignore_json(),
                                  minimax_test::attn_types_json(61) + ", 0")),
              "attn_type_list"),
          "a layer of another attention type");
}

DGPP_TEST(minimax_config_refuses_other_quantization_recipes) {
  require(has(refusal(config_json("\"quantization_config\": {", "\"quantization_config_off\": {")),
              "quantization_config"),
          "the unquantized / FP8 base");
  require(has(refusal(config_json("\"quant_method\": \"modelopt\"",
                                  "\"quant_method\": \"compressed-tensors\"")),
              "quant_method"),
          "method");
  require(has(refusal(config_json("\"quant_algo\": \"NVFP4\"", "\"quant_algo\": \"FP8\"")),
              "quant_algo"),
          "algo");
  require(has(refusal(config_json(
                  "\"weights\": {\n          \"dynamic\": false,\n          \"num_bits\": 4",
                  "\"weights\": {\n          \"dynamic\": false,\n          \"num_bits\": 8")),
              "weights.num_bits"),
          "bits");
  require(has(refusal(config_json("\"Linear\"", "\"Conv1d\"")), "targets"), "targets");
  // An ignore list missing one layer's attention, or naming a module the
  // loader does not know, is refused.
  require(has(refusal(config_json("", "", minimax_test::ignore_json(61))), "model.layers.61"),
          "ignore coverage");
  require(has(refusal(config_json("", "",
                                  minimax_test::ignore_json() +
                                      ", \"model.layers.5.block_sparse_moe.experts.3.w1\"")),
              "unexpected ignored"),
          "ignore extra");
  // A weight-only recipe (no activation scheme) parses the same.
  const dgpp::MinimaxTextConfig a16 =
      minimax_test::parse(config_json("\"input_activations\": {", "\"input_activations_off\": {"));
  require(a16.fp4_group_size == 16, "W4A16");
}

DGPP_TEST(minimax_kernel_geometry_is_separate_from_the_parse) {
  // The class admits other shapes (the synthetic test checkpoint is one):
  // they parse. The attention kernels implement 128-wide heads.
  dgpp::MinimaxTextConfig t = minimax_test::release();
  t.head_dim = 16;
  bool refused = false;
  try {
    t.require_kernel_geometry();
  } catch (const std::invalid_argument& e) {
    refused = has(e.what(), "head_dim");
  }
  require(refused, "16-wide heads have no kernel");
}
