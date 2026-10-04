// The Gemma 4 config parser: the 31B release's values parse, the derived
// geometry (the layer pattern, the two attention kinds' head / KV / rotary
// shapes, the bf16 embedding scale), the recipe read off the ignore list,
// and the unsupported shapes are refused by name.
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "gemma4_config_json.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/gemma4/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Gemma4TextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Gemma4TextConfig::parse(t.root);
}

std::string refusal(const std::string& text) {
  try {
    (void)parse(text);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool has(const std::string& msg, const char* needle) { return msg.find(needle) != std::string::npos; }

// One anchor patched in the 31B config must be refused with a message naming `field`.
void refused(const std::string& from, const std::string& to, const char* field) {
  const std::string msg = refusal(gemma4_test::config_json(from, to));
  require(has(msg, field), std::string("expected a refusal naming '") + field + "', got '" + msg + "'");
}

}  // namespace

DGPP_TEST(gemma4_architecture_is_detected) {
  // (the parse tree's strings are views into the text: it must outlive the tree)
  const std::string text = gemma4_test::config_json();
  const auto t = dgpp::minijson::parse(text);
  require(dgpp::detect_architecture(t.root) == dgpp::ModelArchitecture::Gemma4, "Gemma4ForConditionalGeneration");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::Gemma4)) == "gemma4", "name");
  // The text-only class is another container (flat config, other tensor names): refused.
  const auto flat = dgpp::minijson::parse(R"({"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4_text"})");
  bool threw = false;
  try {
    (void)dgpp::detect_architecture(flat.root);
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, "Gemma4ForCausalLM must be refused");
  // No class name: model_type decides.
  const auto typed = dgpp::minijson::parse(R"({"model_type": "gemma4"})");
  require(dgpp::detect_architecture(typed.root) == dgpp::ModelArchitecture::Gemma4, "model_type gemma4");
}

DGPP_TEST(gemma4_config_parses_the_31b_release) {
  const dgpp::Gemma4TextConfig c = parse(gemma4_test::config_json());
  require(c.hidden_size == 5376 && c.vocab_size == 262144 && c.num_hidden_layers == 60, "shape");
  require(c.max_position_embeddings == 262144 && c.rms_norm_eps == 1e-6f, "positions / eps");
  require(c.tie_word_embeddings && c.final_logit_softcapping == 30.0f, "tied head / soft-cap");
  require(c.num_sliding_layers() == 50 && c.num_full_layers() == 10, "layer pattern");
  for (int l = 0; l < 60; ++l)
    require(c.is_sliding_layer(l) == ((l + 1) % 6 != 0), "is_sliding_layer " + std::to_string(l));
  require(c.num_attention_heads == 32 && c.num_key_value_heads == 16 && c.num_global_key_value_heads == 4, "heads");
  require(c.head_dim == 256 && c.global_head_dim == 512 && c.attention_k_eq_v, "head dims");
  require(c.sliding_window == 1024, "window");
  require(c.sliding_rope_theta == 1e4 && c.global_rope_theta == 1e6, "thetas");
  require(c.sliding_rotary_pairs == 128 && c.global_rotary_pairs == 64, "rotary pairs");
  // Per-layer geometry: a sliding layer and a full one.
  require(c.head_dim_of(0) == 256 && c.kv_heads_of(0) == 16 && c.has_v_proj(0) && c.rotary_pairs_of(0) == 128, "sliding kind");
  require(c.head_dim_of(5) == 512 && c.kv_heads_of(5) == 4 && !c.has_v_proj(5) && c.rotary_pairs_of(5) == 64, "full kind");
  require(c.rope_theta_of(0) == 1e4 && c.rope_theta_of(5) == 1e6, "per-layer theta");
  require(c.q_rows(0) == 8192 && c.kv_rows(0) == 4096 && c.q_rows(5) == 16384 && c.kv_rows(5) == 2048, "projection rows");
  require(c.intermediate_size == 21504, "mlp");
  require(c.eos_token_ids.size() == 2 && c.eos_token_ids[0] == 1 && c.eos_token_ids[1] == 106, "eos");
  require(c.bos_token_id == 2 && c.pad_token_id == 0, "bos / pad");
  require(c.image_token_id == 258880 && c.audio_token_id == 258881 && c.video_token_id == 258884, "placeholders");
  require(c.recipe == dgpp::Gemma4Recipe::Nvfp4Mlp && !c.attention_nvfp4() && c.activation_scales(), "recipe");
  require(c.nvfp4_group == 16, "nvfp4 group");
  // sqrt(5376) = 73.3212... rounds to 73.5 in bf16 (7 mantissa bits at 2^6: steps of 0.5).
  require(c.embed_scale() == 73.5f, "embed scale " + std::to_string(c.embed_scale()));
}

DGPP_TEST(gemma4_config_embed_scale_is_the_bf16_of_the_root) {
  // 2816 (the 26B-A4B's width): sqrt = 53.0659..., bf16 steps of 0.25 at 2^5 -> 53.0.
  dgpp::Gemma4TextConfig c;
  c.hidden_size = 2816;
  require(c.embed_scale() == 53.0f, "2816 -> " + std::to_string(c.embed_scale()));
  c.hidden_size = 1024;
  require(c.embed_scale() == 32.0f, "1024");
  c.hidden_size = 64;
  require(c.embed_scale() == 8.0f, "64");
}

DGPP_TEST(gemma4_config_defaults_follow_the_reference) {
  // layer_types absent: every sixth layer full.
  {
    const dgpp::Gemma4TextConfig c = parse(gemma4_test::config_json("\"layer_types\": [" + gemma4_test::layer_types_json() + "],", ""));
    require(c.num_sliding_layers() == 50 && !c.is_sliding_layer(5) && !c.is_sliding_layer(59) && c.is_sliding_layer(58),
            "default layer pattern");
  }
  // rope_parameters absent: the reference's defaults are the release's values.
  {
    std::string s = gemma4_test::config_json();
    const size_t a = s.find("\"rope_parameters\"");
    const size_t b = s.find("\"sliding_window\"");
    require(a != std::string::npos && b != std::string::npos && a < b, "anchors");
    s.erase(a, b - a);
    const dgpp::Gemma4TextConfig c = parse(s);
    require(c.sliding_rope_theta == 1e4 && c.global_rope_theta == 1e6 && c.global_rotary_pairs == 64, "default rope");
  }
  // final_logit_softcapping null: no cap.
  {
    const dgpp::Gemma4TextConfig c = parse(gemma4_test::config_json("\"final_logit_softcapping\": 30.0", "\"final_logit_softcapping\": null"));
    require(c.final_logit_softcapping == 0.0f, "null soft-cap");
  }
  // The proportional rope's pair count is int(factor * head_dim // 2).
  {
    const dgpp::Gemma4TextConfig c = parse(gemma4_test::config_json("\"partial_rotary_factor\": 0.25", "\"partial_rotary_factor\": 0.3"));
    require(c.global_rotary_pairs == 76, "0.3 * 512 // 2 = 76, got " + std::to_string(c.global_rotary_pairs));
  }
  // use_bidirectional_attention null: causal, as "vision" is over text.
  (void)parse(gemma4_test::config_json("\"use_bidirectional_attention\": \"vision\"", "\"use_bidirectional_attention\": null"));
}

DGPP_TEST(gemma4_config_refuses_unsupported_shapes) {
  refused("\"model_type\": \"gemma4\"", "\"model_type\": \"gemma3\"", "model_type");
  refused("\"model_type\": \"gemma4_text\"", "\"model_type\": \"gemma3_text\"", "text_config.model_type");
  refused("\"audio_config\": null", "\"audio_config\": {\"model_type\": \"gemma4_audio\"}", "audio_config");
  refused("\"dtype\": \"bfloat16\"", "\"dtype\": \"float32\"", "dtype");
  refused("\"hidden_activation\": \"gelu_pytorch_tanh\"", "\"hidden_activation\": \"silu\"", "hidden_activation");
  refused("\"hidden_size\": 5376", "\"hidden_size\": 5380", "hidden_size");
  refused("\"intermediate_size\": 21504", "\"intermediate_size\": 21500", "intermediate_size");
  refused("\"hidden_size_per_layer_input\": 0", "\"hidden_size_per_layer_input\": 256", "hidden_size_per_layer_input");
  refused("\"num_kv_shared_layers\": 0", "\"num_kv_shared_layers\": 20", "num_kv_shared_layers");
  refused("\"use_double_wide_mlp\": false", "\"use_double_wide_mlp\": true", "use_double_wide_mlp");
  // The MoE block without its geometry (the 31B's config leaves num_experts null).
  refused("\"enable_moe_block\": false", "\"enable_moe_block\": true", "num_experts");
  refused("\"attention_k_eq_v\": true", "\"attention_k_eq_v\": false", "attention_k_eq_v");
  refused("\"attention_bias\": false", "\"attention_bias\": true", "attention_bias");
  refused("\"use_bidirectional_attention\": \"vision\"", "\"use_bidirectional_attention\": \"all\"", "use_bidirectional_attention");
  refused("\"num_key_value_heads\": 16", "\"num_key_value_heads\": 12", "num_key_value_heads");
  refused("\"num_global_key_value_heads\": 4", "\"num_global_key_value_heads\": 5", "num_global_key_value_heads");
  refused("\"head_dim\": 256", "\"head_dim\": 255", "head_dim");
  refused("\"sliding_window\": 1024", "\"sliding_window\": 0", "sliding_window");
  refused("\"tie_word_embeddings\": true,\n        \"top_k_experts\"", "\"tie_word_embeddings\": false,\n        \"top_k_experts\"",
          "tie_word_embeddings");
  refused("\"rope_type\": \"proportional\"", "\"rope_type\": \"default\"", "full_attention.rope_type");
  refused("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"", "sliding_attention.rope_type");
  refused("\"rope_theta\": 10000.0", "\"rope_theta\": 10000.0, \"factor\": 2.0", "sliding_attention.factor");
  refused("\"partial_rotary_factor\": 0.25", "\"partial_rotary_factor\": 0.0", "partial_rotary_factor");
  refused("\"final_logit_softcapping\": 30.0", "\"final_logit_softcapping\": -1.0", "final_logit_softcapping");
  refused("\"eos_token_id\": [1, 106]", "\"eos_token_id\": [1, 262144]", "eos_token_id");
}

DGPP_TEST(gemma4_config_layer_types_are_checked) {
  auto with_types = [](const std::string& types) {
    return refusal(gemma4_test::config_json("", "", gemma4_test::ignore_json_31b(), types));
  };
  // Wrong length.
  require(has(with_types(gemma4_test::layer_types_json(59)), "layer_types"), "length");
  // An unknown kind.
  {
    std::string t = gemma4_test::layer_types_json();
    t.replace(t.find("\"sliding_attention\""), 19, "\"chunked_attention\"");
    require(has(with_types(t), "chunked_attention"), "unknown kind");
  }
  // The last layer must be full (the reference forces it; a config that says otherwise is refused).
  {
    std::string t = gemma4_test::layer_types_json();
    t.replace(t.rfind("\"full_attention\""), 16, "\"sliding_attention\"");
    require(has(with_types(t), "last layer"), "last layer");
  }
}

DGPP_TEST(gemma4_config_quantization_contract) {
  refused("\"quant_method\": \"modelopt\"", "\"quant_method\": \"compressed-tensors\"", "quant_method");
  refused("\"quant_algo\": \"NVFP4\"", "\"quant_algo\": \"FP8\"", "quant_algo");
  refused("\"weights\": {\"dynamic\": false, \"num_bits\": 4", "\"weights\": {\"dynamic\": false, \"num_bits\": 8", "num_bits");
  refused("\"weights\": {\"dynamic\": false, \"num_bits\": 4, \"type\": \"float\", \"group_size\": 16",
          "\"weights\": {\"dynamic\": false, \"num_bits\": 4, \"type\": \"float\", \"group_size\": 32", "group_size");
  refused("\"targets\": [\"Linear\"]", "\"targets\": [\"Conv2d\"]", "targets");
  // No quantization_config at all: an unquantized checkpoint has no loader here.
  {
    std::string s = gemma4_test::config_json();
    const size_t a = s.find(",\n    \"quantization_config\"");
    require(a != std::string::npos, "anchor");
    s.erase(a);
    s += "\n}";
    require(has(refusal(s), "quantization_config"), "missing quantization_config");
  }
  auto with_ignore = [](const std::string& ignore) { return gemma4_test::config_json("", "", ignore); };
  // The head must be on the list (it is the tied BF16 embedding).
  require(has(refusal(with_ignore("\"model.vision_tower*\"")), "lm_head"), "lm_head not ignored");
  // An ignored class the loader does not expect in BF16.
  require(has(refusal(with_ignore(gemma4_test::ignore_json_31b() + ", \"model.language_model.layers.3.mlp*\"")),
              "layers.3.mlp"),
          "unexpected ignored module");
  require(has(refusal(with_ignore(gemma4_test::ignore_json_31b() + ", \"model.language_model.layers.60.self_attn*\"")),
              "layers.60"),
          "ignored layer out of range");
  // Attention ignored on some layers only: a per-layer mix is refused, naming the first layer that differs.
  {
    std::string ig = "\"lm_head\"";
    for (int l = 0; l < 60; ++l)
      if (l != 7) ig += ", \"model.language_model.layers." + std::to_string(l) + ".self_attn*\"";
    require(has(refusal(with_ignore(ig)), "layers.7.self_attn"), "per-layer mix");
  }
  // No attention entry at all: the weight-only recipe (NVFP4 attention, no activation scales).
  {
    const dgpp::Gemma4TextConfig c = parse(with_ignore("\"lm_head\", \"model.embed_vision*\", \"model.vision_tower*\""));
    require(c.recipe == dgpp::Gemma4Recipe::Nvfp4WeightOnly && c.attention_nvfp4() && !c.activation_scales(),
            "weight-only recipe");
  }
}

// ---- the 26B-A4B (the MoE block) ---------------------------------------------

DGPP_TEST(gemma4_config_parses_the_26b_a4b_release) {
  const dgpp::Gemma4TextConfig c = parse(gemma4_test::config_json_26b());
  require(c.hidden_size == 2816 && c.vocab_size == 262144 && c.num_hidden_layers == 30, "shape");
  require(c.num_sliding_layers() == 25 && c.num_full_layers() == 5 && !c.is_sliding_layer(29), "layer pattern");
  require(c.num_attention_heads == 16 && c.num_key_value_heads == 8 && c.num_global_key_value_heads == 2, "heads");
  require(c.head_dim == 256 && c.global_head_dim == 512 && c.sliding_window == 1024, "head dims / window");
  require(c.q_rows(0) == 4096 && c.kv_rows(0) == 2048 && c.q_rows(5) == 8192 && c.kv_rows(5) == 1024, "projection rows");
  require(c.intermediate_size == 2112, "the dense MLP beside the experts");
  require(c.enable_moe_block && c.num_experts == 128 && c.top_k_experts == 8 && c.moe_intermediate_size == 704, "the MoE block");
  require(c.recipe == dgpp::Gemma4Recipe::Nvfp4WeightOnly && c.attention_nvfp4() && !c.activation_scales(), "recipe");
  require(c.final_logit_softcapping == 30.0f, "soft-cap");
  // sqrt(2816) = 53.07 -> 53.0 in bf16; 2816^-0.5 = 0.0188445 -> 0.018798828125.
  require(c.embed_scale() == 53.0f, "embed scale " + std::to_string(c.embed_scale()));
  require(c.router_input_scale() == 0.018798828125f, "router input scale " + std::to_string(c.router_input_scale()));
}

DGPP_TEST(gemma4_config_moe_block_is_checked) {
  auto refused26 = [](const std::string& from, const std::string& to, const char* field) {
    const std::string msg = refusal(gemma4_test::config_json_26b(from, to));
    require(has(msg, field), std::string("expected a refusal naming '") + field + "', got '" + msg + "'");
  };
  refused26("\"num_experts\": 128", "\"num_experts\": null", "num_experts");
  refused26("\"num_experts\": 128", "\"num_experts\": 5000", "num_experts");
  refused26("\"top_k_experts\": 8", "\"top_k_experts\": 17", "top_k_experts");
  refused26("\"top_k_experts\": 8", "\"top_k_experts\": 0", "top_k_experts");
  refused26("\"moe_intermediate_size\": 704", "\"moe_intermediate_size\": 700", "moe_intermediate_size");
  refused26("\"moe_intermediate_size\": 704,", "", "moe_intermediate_size");
  // The early key name is read too.
  {
    const dgpp::Gemma4TextConfig c =
        parse(gemma4_test::config_json_26b("\"moe_intermediate_size\": 704", "\"expert_intermediate_size\": 704"));
    require(c.moe_intermediate_size == 704, "expert_intermediate_size");
  }
  // Every router must be on the ignore list (BF16); one missing is named.
  {
    std::string ig = "\"lm_head\"";
    for (int l = 0; l < 30; ++l)
      if (l != 12) ig += ", \"model.language_model.layers." + std::to_string(l) + ".router*\"";
    require(has(refusal(gemma4_test::config_json_26b("", "", ig)), "layers.12.router"), "a router that is not ignored");
  }
  // A router on the list of a model without the MoE block is a class the loader does not expect.
  require(has(refusal(gemma4_test::config_json("", "", gemma4_test::ignore_json_31b() +
                                                           ", \"model.language_model.layers.0.router*\"")),
              "layers.0.router"),
          "a router without the MoE block");
  // The MoE block with the 31B's recipe (attention ignored too): accepted — the experts stay NVFP4
  // and carry the activation scales that recipe writes.
  {
    std::string ig = gemma4_test::ignore_json_26b();
    for (int l = 0; l < 30; ++l) ig += ", \"model.language_model.layers." + std::to_string(l) + ".self_attn*\"";
    const dgpp::Gemma4TextConfig c = parse(gemma4_test::config_json_26b("", "", ig));
    require(c.enable_moe_block && c.recipe == dgpp::Gemma4Recipe::Nvfp4Mlp && c.activation_scales(), "MoE under the mlp recipe");
  }
}
