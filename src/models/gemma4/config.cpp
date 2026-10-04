#include "models/gemma4/config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <stdexcept>

#include "common/dtypes.hpp"

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("Gemma-4 config.{}: {}", field, why));
}

// `scope` prefixes the field in messages ("text_config." or "").
struct Scope {
  const minijson::Value& v;
  std::string prefix;

  std::string name(std::string_view field) const { return prefix + std::string(field); }
  const minijson::Value* find(std::string_view field) const {
    const minijson::Value* f = v.find(field);
    return (!f || f->is_null()) ? nullptr : f;
  }
  const minijson::Value& require(std::string_view field) const {
    const minijson::Value* f = find(field);
    if (!f) reject(name(field), "missing");
    return *f;
  }
  int require_int(std::string_view field) const {
    const minijson::Value& f = require(field);
    if (!f.is_number()) reject(name(field), "not a number");
    return static_cast<int>(f.as_int());
  }
  int optional_int(std::string_view field, int dflt) const {
    const minijson::Value* f = find(field);
    if (!f) return dflt;
    if (!f->is_number()) reject(name(field), "not a number");
    return static_cast<int>(f->as_int());
  }
  double optional_double(std::string_view field, double dflt) const {
    const minijson::Value* f = find(field);
    if (!f) return dflt;
    if (!f->is_number()) reject(name(field), "not a number");
    const double d = f->as_double();
    if (!std::isfinite(d)) reject(name(field), "not finite");
    return d;
  }
  bool optional_bool(std::string_view field, bool dflt) const {
    const minijson::Value* f = find(field);
    if (!f) return dflt;
    if (!f->is_bool()) reject(name(field), "not a bool");
    return f->as_bool();
  }
  std::string optional_string(std::string_view field, const std::string& dflt) const {
    const minijson::Value* f = find(field);
    if (!f) return dflt;
    if (!f->is_string()) reject(name(field), "not a string");
    return std::string(f->as_string());
  }
};

std::string read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    throw std::runtime_error(std::format("cannot open config {}: {}", path, std::strerror(errno)));
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  return text;
}

// layer_types: one of the two kinds per layer. Absent, the reference's
// default applies (every sixth layer full). The reference forces the last
// layer to full_attention with a warning; a config that says otherwise is
// refused here rather than silently changed.
std::vector<uint8_t> parse_layer_types(const Scope& t, int layers) {
  std::vector<uint8_t> out;
  const minijson::Value* f = t.find("layer_types");
  if (!f) {
    for (int i = 0; i < layers; ++i) out.push_back((i + 1) % 6 != 0 ? 1 : 0);
  } else {
    if (!f->is_array()) reject(t.name("layer_types"), "not an array");
    for (const auto& item : f->items()) {
      if (!item.is_string()) reject(t.name("layer_types"), "non-string element");
      const std::string_view kind = item.as_string();
      if (kind == "sliding_attention") out.push_back(1);
      else if (kind == "full_attention") out.push_back(0);
      else reject(t.name("layer_types"), "unknown layer type '" + std::string(kind) + "'");
    }
    if (static_cast<int>(out.size()) != layers)
      reject(t.name("layer_types"), "length differs from num_hidden_layers");
  }
  if (out.back() != 0) reject(t.name("layer_types"), "the last layer must be full_attention");
  return out;
}

// rope_parameters: {"sliding_attention": {rope_type default, rope_theta},
// "full_attention": {rope_type proportional, partial_rotary_factor,
// rope_theta}}. Absent, the reference's defaults are these same values.
void parse_rope(const Scope& t, Gemma4TextConfig& c) {
  double sliding_theta = 10000.0, global_theta = 1000000.0, global_factor = 0.25;
  if (const minijson::Value* rp = t.find("rope_parameters")) {
    if (!rp->is_object()) reject(t.name("rope_parameters"), "not an object");
    for (const auto& m : rp->members())
      if (m.key != "sliding_attention" && m.key != "full_attention")
        reject(t.name("rope_parameters"), "unknown layer type '" + m.key + "'");
    if (const minijson::Value* s = rp->find("sliding_attention"); s && !s->is_null()) {
      if (!s->is_object()) reject(t.name("rope_parameters.sliding_attention"), "not an object");
      const Scope sc{*s, t.prefix + "rope_parameters.sliding_attention."};
      if (const std::string type = sc.optional_string("rope_type", "default"); type != "default")
        reject(sc.name("rope_type"), "the sliding layers implement the default rope, got '" + type + "'");
      if (sc.optional_double("partial_rotary_factor", 1.0) != 1.0)
        reject(sc.name("partial_rotary_factor"), "the default rope rotates the whole head");
      if (sc.optional_double("factor", 1.0) != 1.0) reject(sc.name("factor"), "rope scaling is not implemented");
      sliding_theta = sc.optional_double("rope_theta", sliding_theta);
    } else {
      reject(t.name("rope_parameters.sliding_attention"), "missing");
    }
    if (const minijson::Value* g = rp->find("full_attention"); g && !g->is_null()) {
      if (!g->is_object()) reject(t.name("rope_parameters.full_attention"), "not an object");
      const Scope gc{*g, t.prefix + "rope_parameters.full_attention."};
      if (const std::string type = gc.optional_string("rope_type", "default"); type != "proportional")
        reject(gc.name("rope_type"), "the full layers implement the proportional rope, got '" + type + "'");
      if (gc.optional_double("factor", 1.0) != 1.0) reject(gc.name("factor"), "rope scaling is not implemented");
      global_factor = gc.optional_double("partial_rotary_factor", 1.0);
      global_theta = gc.optional_double("rope_theta", global_theta);
    } else {
      reject(t.name("rope_parameters.full_attention"), "missing");
    }
  }
  if (t.find("rope_scaling")) reject(t.name("rope_scaling"), "rope scaling is not implemented");
  if (!(sliding_theta > 0) || !(global_theta > 0)) reject(t.name("rope_parameters"), "rope_theta must be positive");
  if (!(global_factor > 0) || global_factor > 1.0)
    reject(t.name("rope_parameters.full_attention.partial_rotary_factor"), "must be in (0, 1]");
  c.sliding_rope_theta = sliding_theta;
  c.global_rope_theta = global_theta;
  // The default rope: one frequency per pair over the whole head.
  c.sliding_rotary_pairs = c.head_dim / 2;
  // The proportional rope (transformers _compute_proportional_rope_parameters):
  // rope_angles = int(partial_rotary_factor * head_dim // 2) — the product
  // first, then the floor division.
  c.global_rotary_pairs = static_cast<int>(std::floor(global_factor * static_cast<double>(c.global_head_dim) / 2.0));
  if (c.global_rotary_pairs <= 0 || c.global_rotary_pairs > c.global_head_dim / 2)
    reject(t.name("rope_parameters.full_attention.partial_rotary_factor"),
           "the rotated pair count must be in [1, global_head_dim / 2]");
}

// "model.language_model.layers.<L>.<tail>" -> L (and the tail), or -1.
int layer_entry(const std::string& s, std::string* tail) {
  static const std::string kPrefix = "model.language_model.layers.";
  if (s.rfind(kPrefix, 0) != 0) return -1;
  const size_t dot = s.find('.', kPrefix.size());
  if (dot == std::string::npos) return -1;
  const std::string idx = s.substr(kPrefix.size(), dot - kPrefix.size());
  if (idx.empty() || idx.size() > 6 || idx.find_first_not_of("0123456789") != std::string::npos) return -1;
  *tail = s.substr(dot + 1);
  return std::stoi(idx);
}

// The modelopt NVFP4 contract (quantization_config, written by modelopt's
// HF export): NVFP4, 4-bit float weights on groups of 16, every Linear
// targeted, and an ignore list of module-name patterns. The list decides
// the recipe: which classes are BF16 in the file. A module ignored that the
// loader does not expect in BF16 (or the reverse) would be bound wrongly,
// so the list is checked both ways.
void parse_quantization(const minijson::Value& root, Gemma4TextConfig& c) {
  const minijson::Value* qc = root.find("quantization_config");
  if (!qc || qc->is_null())
    reject("quantization_config",
           "missing — the engine implements the modelopt NVFP4 releases; an unquantized Gemma-4 "
           "checkpoint has no loader here");
  if (!qc->is_object()) reject("quantization_config", "not an object");
  const Scope q{*qc, "quantization_config."};
  if (const std::string method = q.optional_string("quant_method", ""); method != "modelopt")
    reject(q.name("quant_method"), "only modelopt is implemented, got '" + method + "'");
  if (const std::string algo = q.optional_string("quant_algo", ""); algo != "NVFP4")
    reject(q.name("quant_algo"), "only NVFP4 is implemented, got '" + algo + "'");
  const minijson::Value* groups = q.find("config_groups");
  if (!groups || !groups->is_object() || groups->members().size() != 1)
    reject(q.name("config_groups"), "expected exactly one group");
  const minijson::Value& group = groups->members()[0].value;
  if (!group.is_object()) reject(q.name("config_groups"), "group is not an object");
  const minijson::Value* weights = group.find("weights");
  if (!weights || !weights->is_object()) reject(q.name("config_groups.weights"), "missing");
  {
    const Scope w{*weights, "quantization_config.config_groups.weights."};
    if (w.require_int("num_bits") != 4) reject(w.name("num_bits"), "the NVFP4 kernels implement 4-bit weights");
    if (const std::string type = w.optional_string("type", "float"); type != "float")
      reject(w.name("type"), "only float (e2m1) codes are implemented, got '" + type + "'");
    if (w.optional_bool("dynamic", false)) reject(w.name("dynamic"), "dynamic weight scales are not implemented");
    c.nvfp4_group = w.optional_int("group_size", 16);
    if (c.nvfp4_group != 16) reject(w.name("group_size"), "the NVFP4 kernels implement groups of 16");
  }
  if (const minijson::Value* targets = group.find("targets"); targets && !targets->is_null()) {
    if (!targets->is_array() || targets->items().size() != 1 || !targets->items()[0].is_string() ||
        targets->items()[0].as_string() != "Linear")
      reject(q.name("config_groups.targets"), "expected [\"Linear\"]");
  }
  // `input_activations` and `kv_cache_scheme` describe activation and cache
  // quantization the engine does not run (BF16 activations, a BF16 cache);
  // they are accepted as the releases write them and consume nothing.

  // The ignore list.
  const minijson::Value* ig = q.find("ignore");
  if (!ig || !ig->is_array()) reject(q.name("ignore"), "missing");
  std::vector<uint8_t> attn_ignored(static_cast<size_t>(c.num_hidden_layers), 0);
  std::vector<uint8_t> router_ignored(static_cast<size_t>(c.num_hidden_layers), 0);
  bool head_ignored = false;
  for (const auto& item : ig->items()) {
    if (!item.is_string()) reject(q.name("ignore"), "non-string entry");
    const std::string s(item.as_string());
    if (s == "lm_head") {
      head_ignored = true;
      continue;
    }
    // The encoders and their projectors: in the file, not served.
    if (s.rfind("model.vision_tower", 0) == 0 || s.rfind("model.embed_vision", 0) == 0 ||
        s.rfind("model.audio_tower", 0) == 0 || s.rfind("model.embed_audio", 0) == 0)
      continue;
    std::string tail;
    const int l = layer_entry(s, &tail);
    if (l < 0 || l >= c.num_hidden_layers)
      reject(q.name("ignore"), "unexpected ignored module '" + s + "' (a BF16 class the loader does not implement)");
    if (tail == "self_attn*") {
      attn_ignored[static_cast<size_t>(l)] = 1;
      continue;
    }
    // The MoE block's router (proj, scale, per_expert_scale): BF16.
    if (tail == "router*" && c.enable_moe_block) {
      router_ignored[static_cast<size_t>(l)] = 1;
      continue;
    }
    reject(q.name("ignore"), "unexpected ignored module '" + s + "' (a BF16 class the loader does not implement)");
  }
  if (c.enable_moe_block)
    for (int l = 0; l < c.num_hidden_layers; ++l)
      if (!router_ignored[static_cast<size_t>(l)])
        reject(q.name("ignore"), std::format("'model.language_model.layers.{}.router*' is not ignored — the engine "
                                             "expects a BF16 router", l));
  // The head is the BF16 embedding (tied); a recipe that quantized it would
  // have written an lm_head the table does not expect.
  if (!head_ignored) reject(q.name("ignore"), "'lm_head' is not ignored — the engine expects the tied BF16 embedding");
  int n_attn = 0;
  for (uint8_t a : attn_ignored) n_attn += a;
  if (n_attn == c.num_hidden_layers) {
    c.recipe = Gemma4Recipe::Nvfp4Mlp;
  } else if (n_attn == 0) {
    c.recipe = Gemma4Recipe::Nvfp4WeightOnly;
  } else {
    for (int l = 0; l < c.num_hidden_layers; ++l)
      if (!attn_ignored[static_cast<size_t>(l)])
        reject(q.name("ignore"), std::format("'model.language_model.layers.{}.self_attn*' is not ignored while other "
                                             "layers' attention is — a per-layer mix is not implemented", l));
  }
}

}  // namespace

Gemma4TextConfig Gemma4TextConfig::parse(const minijson::Value& root_value) {
  if (!root_value.is_object()) reject("", "root is not an object");
  const Scope root{root_value, ""};
  Gemma4TextConfig c;
  if (const std::string type = root.optional_string("model_type", "gemma4"); type != "gemma4")
    reject("model_type", "expected gemma4, got " + type);
  if (const std::string dtype = root.optional_string("dtype", root.optional_string("torch_dtype", "bfloat16"));
      dtype != "bfloat16")
    reject("dtype", "only bfloat16 checkpoints are implemented, got " + dtype);
  if (!root.optional_bool("tie_word_embeddings", true))
    reject("tie_word_embeddings", "an untied head is not implemented (the checkpoint has no lm_head)");
  // The audio tower: the releases served here have none.
  if (root.find("audio_config")) reject("audio_config", "an audio tower is not implemented (expected null)");

  const minijson::Value* tc = root.find("text_config");
  if (!tc || !tc->is_object()) reject("text_config", "missing");
  const Scope t{*tc, "text_config."};
  if (const std::string type = t.optional_string("model_type", "gemma4_text"); type != "gemma4_text")
    reject(t.name("model_type"), "expected gemma4_text, got " + type);
  if (const std::string dtype = t.optional_string("dtype", t.optional_string("torch_dtype", "bfloat16"));
      dtype != "bfloat16")
    reject(t.name("dtype"), "only bfloat16 checkpoints are implemented, got " + dtype);

  // --- model shape ------------------------------------------------------------
  c.hidden_size = t.require_int("hidden_size");
  c.vocab_size = t.require_int("vocab_size");
  c.num_hidden_layers = t.require_int("num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(t.optional_double("rms_norm_eps", 1e-6));
  c.max_position_embeddings = t.require_int("max_position_embeddings");
  c.hidden_activation = t.optional_string("hidden_activation", "gelu_pytorch_tanh");
  c.tie_word_embeddings = t.optional_bool("tie_word_embeddings", true);
  if (c.hidden_size <= 0 || c.hidden_size % 16 != 0)
    reject(t.name("hidden_size"), "must be a positive multiple of 16 (the NVFP4 scale group)");
  if (c.vocab_size <= 0) reject(t.name("vocab_size"), "must be positive");
  if (c.num_hidden_layers <= 0 || c.num_hidden_layers > 1024) reject(t.name("num_hidden_layers"), "must be in [1, 1024]");
  if (!(c.rms_norm_eps > 0)) reject(t.name("rms_norm_eps"), "must be positive");
  if (c.max_position_embeddings <= 0) reject(t.name("max_position_embeddings"), "must be positive");
  if (c.hidden_activation != "gelu_pytorch_tanh")
    reject(t.name("hidden_activation"), "only gelu_pytorch_tanh is implemented, got " + c.hidden_activation);
  if (!c.tie_word_embeddings)
    reject(t.name("tie_word_embeddings"), "an untied head is not implemented (the checkpoint has no lm_head)");
  c.final_logit_softcapping = static_cast<float>(t.optional_double("final_logit_softcapping", 0.0));
  if (c.final_logit_softcapping < 0) reject(t.name("final_logit_softcapping"), "must be positive or null");
  c.sliding_layer = parse_layer_types(t, c.num_hidden_layers);

  // --- the variants of the family the engine does not implement ---------------
  if (t.optional_int("hidden_size_per_layer_input", 0) != 0)
    reject(t.name("hidden_size_per_layer_input"),
           "per-layer input embeddings (the E2B / E4B models) are not implemented (expected 0)");
  if (t.optional_int("num_kv_shared_layers", 0) != 0)
    reject(t.name("num_kv_shared_layers"), "KV-sharing layers are not implemented (expected 0)");
  if (t.optional_bool("use_double_wide_mlp", false))
    reject(t.name("use_double_wide_mlp"), "the double-wide MLP is not implemented");
  if (const std::string bidir = t.optional_string("use_bidirectional_attention", "vision");
      bidir != "vision")
    reject(t.name("use_bidirectional_attention"),
           "only null or \"vision\" (causal over text) is implemented, got '" + bidir + "'");

  // --- tokens -------------------------------------------------------------------
  auto read_ids = [&](const Scope& s, std::vector<int64_t>& out) {
    const minijson::Value* eos = s.find("eos_token_id");
    if (!eos) return;
    if (eos->is_array()) {
      for (const auto& item : eos->items()) {
        if (!item.is_number()) reject(s.name("eos_token_id"), "non-numeric element");
        out.push_back(item.as_int());
      }
    } else if (eos->is_number()) {
      out.push_back(eos->as_int());
    } else {
      reject(s.name("eos_token_id"), "not a number or array");
    }
  };
  read_ids(root, c.eos_token_ids);
  if (c.eos_token_ids.empty()) read_ids(t, c.eos_token_ids);
  if (c.eos_token_ids.empty()) reject("eos_token_id", "missing");
  for (int64_t id : c.eos_token_ids)
    if (id < 0 || id >= c.vocab_size) reject("eos_token_id", "id outside [0, vocab_size)");
  c.bos_token_id = t.optional_int("bos_token_id", 2);
  c.pad_token_id = t.optional_int("pad_token_id", 0);
  if (c.bos_token_id < 0 || c.bos_token_id >= c.vocab_size) reject(t.name("bos_token_id"), "id outside [0, vocab_size)");
  c.image_token_id = root.optional_int("image_token_id", -1);
  c.audio_token_id = root.optional_int("audio_token_id", -1);
  c.video_token_id = root.optional_int("video_token_id", -1);

  // --- attention ----------------------------------------------------------------
  c.num_attention_heads = t.require_int("num_attention_heads");
  c.num_key_value_heads = t.require_int("num_key_value_heads");
  c.head_dim = t.optional_int("head_dim", 256);
  c.global_head_dim = t.optional_int("global_head_dim", 512);
  c.attention_k_eq_v = t.optional_bool("attention_k_eq_v", false);
  if (!c.attention_k_eq_v)
    reject(t.name("attention_k_eq_v"),
           "only true is implemented (the full layers read their value from k_proj and take "
           "num_global_key_value_heads KV heads)");
  // The reference's override: with attention_k_eq_v, a full layer takes
  // num_global_key_value_heads when the config states it, else the sliding count.
  c.num_global_key_value_heads = t.optional_int("num_global_key_value_heads", c.num_key_value_heads);
  if (t.optional_bool("attention_bias", false)) reject(t.name("attention_bias"), "biased projections are not implemented");
  c.sliding_window = t.require_int("sliding_window");
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 || c.num_global_key_value_heads <= 0)
    reject(t.name("num_attention_heads"), "head counts must be positive");
  if (c.num_attention_heads % c.num_key_value_heads != 0)
    reject(t.name("num_key_value_heads"), "must divide num_attention_heads");
  if (c.num_attention_heads % c.num_global_key_value_heads != 0)
    reject(t.name("num_global_key_value_heads"), "must divide num_attention_heads");
  if (c.head_dim <= 0 || c.head_dim % 2 != 0) reject(t.name("head_dim"), "must be a positive even integer");
  if (c.global_head_dim <= 0 || c.global_head_dim % 2 != 0)
    reject(t.name("global_head_dim"), "must be a positive even integer");
  if (c.sliding_window <= 0) reject(t.name("sliding_window"), "must be positive");
  parse_rope(t, c);

  // --- MLP ------------------------------------------------------------------------
  c.intermediate_size = t.require_int("intermediate_size");
  if (c.intermediate_size <= 0 || c.intermediate_size % 16 != 0)
    reject(t.name("intermediate_size"), "must be a positive multiple of 16 (the NVFP4 scale group)");

  // --- the MoE block ------------------------------------------------------------
  c.enable_moe_block = t.optional_bool("enable_moe_block", false);
  if (c.enable_moe_block) {
    c.num_experts = t.require_int("num_experts");
    c.top_k_experts = t.require_int("top_k_experts");
    // (an early config wrote expert_intermediate_size)
    c.moe_intermediate_size = t.optional_int("moe_intermediate_size", t.optional_int("expert_intermediate_size", 0));
    if (c.num_experts <= 0 || c.num_experts > 4096) reject(t.name("num_experts"), "must be in [1, 4096]");
    if (c.top_k_experts <= 0 || c.top_k_experts > 16 || c.top_k_experts > c.num_experts)
      reject(t.name("top_k_experts"), "must be in [1, min(num_experts, 16)]");
    if (c.moe_intermediate_size <= 0 || c.moe_intermediate_size % 16 != 0)
      reject(t.name("moe_intermediate_size"), "must be a positive multiple of 16 (the NVFP4 scale group)");
  }

  parse_quantization(root_value, c);
  // NVFP4 attention: every K the projections see must hold whole groups.
  if (c.attention_nvfp4()) {
    for (int l = 0; l < c.num_hidden_layers; ++l)
      if (c.q_rows(l) % 16 != 0)
        reject(t.name("head_dim"), "heads x head_dim must be a multiple of 16 (the NVFP4 scale group, o_proj's K)");
  }
  return c;
}

Gemma4TextConfig Gemma4TextConfig::from_json_file(const std::string& path) {
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  return parse(parsed.root);
}

float Gemma4TextConfig::router_input_scale() const {
  const float f = static_cast<float>(std::pow(static_cast<double>(hidden_size), -0.5));
  return bf16_bits_to_float(float_to_bf16_bits(f));
}

float Gemma4TextConfig::embed_scale() const {
  // torch.tensor(hidden_size ** 0.5) is an fp32 of the double; the forward
  // casts it to the table's dtype (bf16, round to nearest even).
  const float f = static_cast<float>(std::sqrt(static_cast<double>(hidden_size)));
  return bf16_bits_to_float(float_to_bf16_bits(f));
}

}  // namespace dgpp
