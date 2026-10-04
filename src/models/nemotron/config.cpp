#include "models/nemotron/config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <unordered_set>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("Nemotron-H config.{}: {}", field, why));
}

const minijson::Value& require(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) reject(field, "missing");
  return *f;
}
int require_int(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  return static_cast<int>(f.as_int());
}
int optional_int(const minijson::Value& v, std::string_view field, int dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  return static_cast<int>(f->as_int());
}
double optional_double(const minijson::Value& v, std::string_view field, double dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  const double d = f->as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}
bool optional_bool(const minijson::Value& v, std::string_view field, bool dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_bool()) reject(field, "not a bool");
  return f->as_bool();
}
std::string optional_string(const minijson::Value& v, std::string_view field,
                            const std::string& dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_string()) reject(field, "not a string");
  return std::string(f->as_string());
}
bool present(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  return f != nullptr && !f->is_null();
}

// A Mamba field under the name config.json writes (`n_groups`) or the name
// the config class's constructor takes (`mamba_n_groups`): the released
// files write the first, a re-saved config the second. Both must agree.
int mamba_int(const minijson::Value& v, std::string_view field, std::string_view ctor_name) {
  const bool a = present(v, field), b = present(v, ctor_name);
  if (!a && !b) reject(field, "missing");
  const int x = a ? require_int(v, field) : require_int(v, ctor_name);
  if (a && b && require_int(v, ctor_name) != x)
    reject(ctor_name, std::format("disagrees with {}", field));
  return x;
}

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

// The layer kinds from `hybrid_override_pattern` (one character per layer)
// or `layers_block_type` (the list a re-saved config writes). '-' is the
// reference's dense-MLP block: no release uses it and it has no table here.
std::vector<NemotronLayerKind> parse_pattern(const minijson::Value& root,
                                             std::string_view pattern_field,
                                             std::string_view list_field) {
  std::vector<NemotronLayerKind> from_pattern, from_list;
  if (present(root, pattern_field)) {
    const minijson::Value& p = require(root, pattern_field);
    if (!p.is_string()) reject(pattern_field, "not a string");
    for (const char ch : p.as_string()) {
      if (ch == 'M')
        from_pattern.push_back(NemotronLayerKind::Mamba);
      else if (ch == 'E')
        from_pattern.push_back(NemotronLayerKind::Moe);
      else if (ch == '*')
        from_pattern.push_back(NemotronLayerKind::Attention);
      else if (ch == '-')
        reject(pattern_field, "dense-MLP layers ('-') are not implemented");
      else
        reject(pattern_field, std::format("unsupported layer character '{}'", ch));
    }
  }
  if (present(root, list_field)) {
    const minijson::Value& l = require(root, list_field);
    if (!l.is_array()) reject(list_field, "not an array");
    for (const auto& item : l.items()) {
      const std::string s = item.is_string() ? std::string(item.as_string()) : "";
      if (s == "mamba")
        from_list.push_back(NemotronLayerKind::Mamba);
      else if (s == "moe")
        from_list.push_back(NemotronLayerKind::Moe);
      else if (s == "attention")
        from_list.push_back(NemotronLayerKind::Attention);
      else
        reject(list_field, "unsupported layer type '" + s + "'");
    }
    if (!from_pattern.empty() && from_pattern != from_list)
      reject(list_field, std::format("disagrees with {}", pattern_field));
  }
  return from_pattern.empty() ? from_list : from_pattern;
}

using QuantMap = std::unordered_map<std::string, NemotronQuant>;

// `quantized_layers`: {module: {"quant_algo": "FP8"} | {"quant_algo":
// "NVFP4", "group_size": 16}}. A module that is not a backbone nn.Linear of
// this layer pattern has no table entry to carry its format, so it is
// refused by name rather than bound in BF16 behind the recipe's back.
QuantMap parse_quantized_layers(const minijson::Value& ql, const std::string& field,
                                const std::unordered_set<std::string>& linears) {
  if (!ql.is_object()) reject(field, "not an object");
  QuantMap out;
  out.reserve(ql.members().size());
  for (const auto& m : ql.members()) {
    if (!m.value.is_object()) reject(field, "'" + m.key + "' is not an object");
    if (!linears.count(m.key))
      reject(field, "'" + m.key + "' is not a backbone linear of this layer pattern");
    const std::string algo = optional_string(m.value, "quant_algo", "");
    NemotronQuant q = NemotronQuant::Bf16;
    if (algo == "FP8") {
      q = NemotronQuant::Fp8;
    } else if (algo == "NVFP4") {
      if (optional_int(m.value, "group_size", 0) != 16)
        reject(field, "'" + m.key + "': NVFP4 is implemented at group_size 16");
      q = NemotronQuant::Nvfp4;
    } else {
      reject(field, "'" + m.key + "': only FP8 and NVFP4 are implemented, got '" + algo + "'");
    }
    if (!out.emplace(m.key, q).second) reject(field, "'" + m.key + "' is listed twice");
  }
  return out;
}

std::unordered_set<std::string> backbone_linears(const NemotronHConfig& c) {
  std::unordered_set<std::string> out;
  for (int l = 0; l < c.num_hidden_layers; ++l)
    for (std::string& m : nemotron_layer_linears(c, l)) out.insert(std::move(m));
  return out;
}

// config.json's quantization_config (the Super form). `config_groups`
// repeats the layer map as two target lists (group_0: 8-bit float, group_1:
// 4-bit float per 16); each target must be in the map under that format.
void parse_mixed_precision(const minijson::Value& q, const minijson::Value* hf_quant,
                           NemotronHConfig& c) {
  if (!q.is_object()) reject("quantization_config", "not an object");
  if (const std::string m = optional_string(q, "quant_method", ""); m != "modelopt")
    reject("quantization_config.quant_method", "only modelopt is implemented, got '" + m + "'");
  if (const std::string a = optional_string(q, "quant_algo", ""); a != "MIXED_PRECISION")
    reject("quantization_config.quant_algo",
           "only MIXED_PRECISION is implemented in config.json, got '" + a + "'");
  const std::unordered_set<std::string> linears = backbone_linears(c);
  const minijson::Value* ql = q.find("quantized_layers");
  if (!ql || ql->is_null()) reject("quantization_config.quantized_layers", "missing");
  c.quant = parse_quantized_layers(*ql, "quantization_config.quantized_layers", linears);
  if (const minijson::Value* ig = q.find("ignore"); ig && !ig->is_null())
    if (!ig->is_array() || !ig->items().empty())
      reject("quantization_config.ignore",
             "must be empty: the recipe names its quantized layers one by one");
  if (const minijson::Value* groups = q.find("config_groups"); groups && !groups->is_null()) {
    if (!groups->is_object()) reject("quantization_config.config_groups", "not an object");
    for (const auto& g : groups->members()) {
      const std::string where = "quantization_config.config_groups." + g.key;
      const minijson::Value* w = g.value.find("weights");
      if (!w || !w->is_object()) reject(where, "weights missing");
      if (optional_string(*w, "type", "") != "float") reject(where, "weights.type must be float");
      const int bits = optional_int(*w, "num_bits", 0);
      NemotronQuant want = NemotronQuant::Bf16;
      if (bits == 8 && !present(*w, "group_size"))
        want = NemotronQuant::Fp8;
      else if (bits == 4 && optional_int(*w, "group_size", 0) == 16)
        want = NemotronQuant::Nvfp4;
      else
        reject(where, "only 8-bit float per tensor and 4-bit float per 16 are implemented");
      const minijson::Value* targets = g.value.find("targets");
      if (!targets || !targets->is_array()) reject(where, "targets missing");
      for (const auto& t : targets->items()) {
        const std::string name = t.is_string() ? std::string(t.as_string()) : "";
        const auto it = c.quant.find(name);
        if (it == c.quant.end() || it->second != want)
          reject(where, "target '" + name + "' disagrees with quantized_layers");
      }
    }
  }
  // The K/V-cache scheme: static 8-bit float, written as one F32 scale
  // beside every backbone k_proj and v_proj.
  if (const minijson::Value* kv = q.find("kv_cache_scheme"); kv && !kv->is_null()) {
    if (!kv->is_object()) reject("quantization_config.kv_cache_scheme", "not an object");
    if (optional_int(*kv, "num_bits", 0) != 8 || optional_string(*kv, "type", "") != "float" ||
        optional_bool(*kv, "dynamic", false))
      reject("quantization_config.kv_cache_scheme",
             "only the static 8-bit float cache scales are implemented");
    c.kv_cache_scales = true;
  }
  if (hf_quant != nullptr) {
    // The same recipe a second time: it must name the same layers.
    const minijson::Value* hq = hf_quant->find("quantization");
    if (!hq || !hq->is_object()) reject("hf_quant_config.quantization", "missing");
    if (const std::string a = optional_string(*hq, "quant_algo", ""); a != "MIXED_PRECISION")
      reject("hf_quant_config.quantization.quant_algo",
             "disagrees with config.json's MIXED_PRECISION, got '" + a + "'");
    const minijson::Value* hql = hq->find("quantized_layers");
    if (!hql || hql->is_null()) reject("hf_quant_config.quantization.quantized_layers", "missing");
    const QuantMap again =
        parse_quantized_layers(*hql, "hf_quant_config.quantization.quantized_layers", linears);
    if (again.size() != c.quant.size())
      reject("hf_quant_config.quantization.quantized_layers",
             "disagrees with config.json's quantization_config");
    for (const auto& [name, fmt] : again) {
      const auto it = c.quant.find(name);
      if (it == c.quant.end() || it->second != fmt)
        reject("hf_quant_config.quantization.quantized_layers",
               "'" + name + "' disagrees with config.json's quantization_config");
    }
  }
  c.recipe = NemotronRecipe::MixedPrecision;
  c.router_dtype = DType::BF16;
}

// hf_quant_config.json alone (the Nano form): every backbone nn.Linear is
// NVFP4 unless excluded. The exclude list is checked both ways — an entry
// that names nothing the table knows (a wildcard, a module of another
// model) would leave a matrix bound in the wrong format.
void parse_nvfp4_exclude(const minijson::Value& hf, NemotronHConfig& c) {
  if (!hf.is_object()) reject("hf_quant_config", "not an object");
  if (const minijson::Value* p = hf.find("producer"); p && p->is_object())
    if (const std::string n = optional_string(*p, "name", "modelopt"); n != "modelopt")
      reject("hf_quant_config.producer.name", "only modelopt is implemented, got '" + n + "'");
  const minijson::Value* q = hf.find("quantization");
  if (!q || !q->is_object()) reject("hf_quant_config.quantization", "missing");
  if (const std::string a = optional_string(*q, "quant_algo", ""); a != "NVFP4")
    reject(
        "hf_quant_config.quantization.quant_algo",
        "only NVFP4 is implemented without a quantization_config in config.json, got '" + a + "'");
  if (optional_int(*q, "group_size", 0) != 16)
    reject("hf_quant_config.quantization.group_size", "NVFP4 is implemented at group_size 16");
  if (const std::string kv = optional_string(*q, "kv_cache_quant_algo", "");
      !kv.empty() && kv != "FP8")
    reject("hf_quant_config.quantization.kv_cache_quant_algo",
           "only FP8 is implemented, got '" + kv + "'");
  const std::unordered_set<std::string> linears = backbone_linears(c);
  std::unordered_set<std::string> excluded;
  if (const minijson::Value* ex = q->find("exclude_modules"); ex && !ex->is_null()) {
    if (!ex->is_array()) reject("hf_quant_config.quantization.exclude_modules", "not an array");
    for (const auto& item : ex->items()) {
      const std::string name = item.is_string() ? std::string(item.as_string()) : "";
      // The head and the Mamba convolutions are never quantized; the recipe
      // lists them all the same.
      bool known = name == "lm_head" || linears.count(name) != 0;
      for (int l = 0; !known && l < c.num_hidden_layers; ++l)
        known = c.layers[static_cast<size_t>(l)] == NemotronLayerKind::Mamba &&
                name == nemotron_layer_prefix(c, l) + "mixer.conv1d";
      if (!known)
        reject("hf_quant_config.quantization.exclude_modules",
               "'" + name + "' names no module of this layer pattern");
      excluded.insert(name);
    }
  }
  if (!excluded.count("lm_head"))
    reject("hf_quant_config.quantization.exclude_modules",
           "'lm_head' is not excluded — the engine expects a BF16 head");
  for (const std::string& m : linears)
    if (!excluded.count(m)) c.quant.emplace(m, NemotronQuant::Nvfp4);
  c.recipe = NemotronRecipe::Nvfp4Exclude;
  c.router_dtype = DType::F32;
  c.kv_cache_scales = false;
}

}  // namespace

std::string nemotron_layer_prefix(const NemotronHConfig& cfg, int layer) {
  if (cfg.is_mtp_layer(layer))
    return "mtp.layers." + std::to_string(layer - cfg.num_hidden_layers) + ".";
  return "backbone.layers." + std::to_string(layer) + ".";
}

std::vector<std::string> nemotron_layer_linears(const NemotronHConfig& cfg, int layer) {
  if (layer < 0 || layer >= cfg.num_layers_total())
    throw std::invalid_argument("nemotron_layer_linears: layer out of range");
  const std::string p = nemotron_layer_prefix(cfg, layer) + "mixer.";
  std::vector<std::string> out;
  switch (cfg.kind_of(layer)) {
    case NemotronLayerKind::Mamba:
      out = {p + "in_proj", p + "out_proj"};
      break;
    case NemotronLayerKind::Attention:
      out = {p + "q_proj", p + "k_proj", p + "v_proj", p + "o_proj"};
      break;
    case NemotronLayerKind::Moe:
      out.reserve(static_cast<size_t>(cfg.n_routed_experts) * 2 + 4);
      for (int e = 0; e < cfg.n_routed_experts; ++e) {
        const std::string ep = p + "experts." + std::to_string(e) + ".";
        out.push_back(ep + "up_proj");
        out.push_back(ep + "down_proj");
      }
      out.push_back(p + "shared_experts.up_proj");
      out.push_back(p + "shared_experts.down_proj");
      if (cfg.moe_latent_size > 0) {
        out.push_back(p + "fc1_latent_proj");
        out.push_back(p + "fc2_latent_proj");
      }
      break;
  }
  return out;
}

NemotronHConfig NemotronHConfig::parse(const minijson::Value& root,
                                       const minijson::Value* hf_quant) {
  if (!root.is_object()) reject("", "root is not an object");
  NemotronHConfig c;
  const std::string model_type = optional_string(root, "model_type", "nemotron_h");
  if (model_type != "nemotron_h") reject("model_type", "expected nemotron_h, got " + model_type);

  c.hidden_size = require_int(root, "hidden_size");
  c.vocab_size = require_int(root, "vocab_size");
  c.max_position_embeddings = require_int(root, "max_position_embeddings");
  if (c.hidden_size <= 0) reject("hidden_size", "must be positive");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.max_position_embeddings <= 0) reject("max_position_embeddings", "must be positive");
  if (optional_bool(root, "tie_word_embeddings", false))
    reject("tie_word_embeddings", "tied embeddings are not implemented");
  if (optional_bool(root, "residual_in_fp32", false))
    reject("residual_in_fp32", "an fp32 residual stream is not implemented");
  // NemotronHRMSNorm and MambaRMSNormGated both read layer_norm_epsilon;
  // `norm_eps` is written beside it and read by nothing.
  if (!present(root, "layer_norm_epsilon")) reject("layer_norm_epsilon", "missing");
  c.layer_norm_eps = static_cast<float>(optional_double(root, "layer_norm_epsilon", 1e-5));
  if (!(c.layer_norm_eps > 0)) reject("layer_norm_epsilon", "must be positive");
  if (static_cast<float>(optional_double(root, "norm_eps", c.layer_norm_eps)) != c.layer_norm_eps)
    reject("norm_eps", "differs from layer_norm_epsilon");

  // --- layer kinds ------------------------------------------------------------
  c.layers = parse_pattern(root, "hybrid_override_pattern", "layers_block_type");
  if (c.layers.empty()) reject("hybrid_override_pattern", "missing");
  c.num_hidden_layers = optional_int(root, "num_hidden_layers", static_cast<int>(c.layers.size()));
  if (c.num_hidden_layers != static_cast<int>(c.layers.size()))
    reject("hybrid_override_pattern", "length differs from num_hidden_layers");

  // --- tokens -----------------------------------------------------------------
  if (const minijson::Value* eos = root.find("eos_token_id"); eos && !eos->is_null()) {
    if (eos->is_array()) {
      for (const auto& item : eos->items()) {
        if (!item.is_number()) reject("eos_token_id", "non-numeric element");
        c.eos_token_ids.push_back(item.as_int());
      }
    } else if (eos->is_number()) {
      c.eos_token_ids.push_back(eos->as_int());
    } else {
      reject("eos_token_id", "not a number or array");
    }
    for (int64_t id : c.eos_token_ids)
      if (id < 0 || id >= c.vocab_size) reject("eos_token_id", "id outside [0, vocab_size)");
  }
  if (c.eos_token_ids.empty()) reject("eos_token_id", "missing");
  c.bos_token_id = optional_int(root, "bos_token_id", -1);
  c.pad_token_id = optional_int(root, "pad_token_id", -1);

  // --- Mamba2 -------------------------------------------------------------------
  c.mamba_num_heads = require_int(root, "mamba_num_heads");
  c.mamba_head_dim = require_int(root, "mamba_head_dim");
  c.ssm_state_size = require_int(root, "ssm_state_size");
  c.mamba_n_groups = mamba_int(root, "n_groups", "mamba_n_groups");
  c.conv_kernel = mamba_int(root, "conv_kernel", "mamba_d_conv");
  c.chunk_size = present(root, "chunk_size") || present(root, "mamba_chunk_size")
                     ? mamba_int(root, "chunk_size", "mamba_chunk_size")
                     : 128;
  c.conv_bias = optional_bool(root, "use_conv_bias", optional_bool(root, "mamba_conv_bias", true));
  c.time_step_min =
      optional_double(root, "time_step_min", optional_double(root, "mamba_dt_min", 0.001));
  if (c.mamba_num_heads <= 0 || c.mamba_head_dim <= 0)
    reject("mamba_num_heads", "head geometry must be positive");
  if (c.ssm_state_size <= 0) reject("ssm_state_size", "must be positive");
  if (c.mamba_n_groups <= 0 || c.mamba_num_heads % c.mamba_n_groups != 0)
    reject("n_groups",
           "must divide mamba_num_heads (a group's B and C serve heads/n_groups heads)");
  if (c.conv_kernel < 2 || c.conv_kernel > 8) reject("conv_kernel", "must be in [2, 8]");
  if (c.chunk_size <= 0) reject("chunk_size", "must be positive");
  if (!c.conv_bias)
    reject("use_conv_bias", "the convolution carries a bias in every release (true)");
  if (!(c.time_step_min > 0)) reject("time_step_min", "must be positive");
  if (const std::string a = optional_string(root, "mamba_hidden_act", "silu"); a != "silu")
    reject("mamba_hidden_act", "only silu is implemented, got " + a);
  if (optional_bool(root, "use_bias", false))
    reject("use_bias", "biased Mamba projections are not implemented");
  if (optional_bool(root, "mamba_proj_bias", false))
    reject("mamba_proj_bias", "biased Mamba projections are not implemented");
  if (const std::string dt = optional_string(root, "mamba_ssm_cache_dtype", "float32");
      dt != "float32")
    reject("mamba_ssm_cache_dtype", "the recurrent state is float32, got " + dt);
  // The config class defaults the time-step limit to (0, inf) — no clamp —
  // and neither release writes it; a config that does asks for a clamp.
  for (const char* f : {"time_step_limit", "mamba_dt_limit"})
    if (present(root, f)) reject(f, "only the default unclamped time step (0, inf) is implemented");

  // --- attention ------------------------------------------------------------
  c.num_attention_heads = require_int(root, "num_attention_heads");
  c.num_key_value_heads = optional_int(root, "num_key_value_heads", c.num_attention_heads);
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 ||
      c.num_attention_heads % c.num_key_value_heads != 0)
    reject("num_key_value_heads", "must divide num_attention_heads");
  c.head_dim = optional_int(root, "head_dim", c.hidden_size / c.num_attention_heads);
  if (c.head_dim <= 0) reject("head_dim", "must be positive");
  if (optional_bool(root, "attention_bias", false))
    reject("attention_bias", "biased attention projections are not implemented");
  if (present(root, "sliding_window"))
    reject("sliding_window", "sliding-window attention is not implemented");

  // --- routed MoE -------------------------------------------------------------
  if (const std::string a = optional_string(root, "mlp_hidden_act", "relu2"); a != "relu2")
    reject("mlp_hidden_act", "only relu2 (relu squared) is implemented, got " + a);
  if (optional_bool(root, "mlp_bias", false))
    reject("mlp_bias", "biased expert projections are not implemented");
  c.n_routed_experts = require_int(root, "n_routed_experts");
  c.num_experts_per_tok = require_int(root, "num_experts_per_tok");
  c.moe_intermediate_size = require_int(root, "moe_intermediate_size");
  c.moe_shared_expert_intermediate_size = require_int(root, "moe_shared_expert_intermediate_size");
  c.moe_latent_size = optional_int(root, "moe_latent_size", 0);
  c.routed_scaling_factor = static_cast<float>(optional_double(root, "routed_scaling_factor", 1.0));
  c.norm_topk_prob = optional_bool(root, "norm_topk_prob", true);
  if (c.n_routed_experts <= 0 || c.n_routed_experts > 4096)
    reject("n_routed_experts", "must be in [1, 4096]");
  if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > c.n_routed_experts)
    reject("num_experts_per_tok", "must be in [1, n_routed_experts]");
  if (c.moe_intermediate_size <= 0) reject("moe_intermediate_size", "must be positive");
  if (c.moe_shared_expert_intermediate_size <= 0)
    reject("moe_shared_expert_intermediate_size", "must be positive");
  if (c.moe_latent_size < 0) reject("moe_latent_size", "must be positive or null");
  if (optional_int(root, "n_shared_experts", 1) != 1)
    reject("n_shared_experts", "the MoE block carries exactly one shared expert");
  if (optional_int(root, "n_group", 1) != 1 || optional_int(root, "topk_group", 1) != 1)
    reject("n_group", "group-limited routing (n_group/topk_group != 1) is not implemented");
  if (!c.norm_topk_prob) reject("norm_topk_prob", "the router renormalizes the top-k (true)");
  if (!(c.routed_scaling_factor > 0)) reject("routed_scaling_factor", "must be positive");

  // --- MTP ----------------------------------------------------------------------
  c.num_nextn_predict_layers = optional_int(root, "num_nextn_predict_layers", 0);
  if (c.num_nextn_predict_layers < 0 || c.num_nextn_predict_layers > 1)
    reject("num_nextn_predict_layers", "only one draft block (or none) is implemented");
  if (c.num_nextn_predict_layers == 1) {
    c.mtp_layers = parse_pattern(root, "mtp_hybrid_override_pattern", "mtp_layers_block_type");
    // The config class's default when neither is written.
    if (c.mtp_layers.empty()) c.mtp_layers = {NemotronLayerKind::Attention, NemotronLayerKind::Moe};
    if (c.mtp_layers !=
        std::vector<NemotronLayerKind>{NemotronLayerKind::Attention, NemotronLayerKind::Moe})
      reject("mtp_hybrid_override_pattern",
             "only the attention + MoE draft block (\"*E\") is implemented");
  }

  // --- quantization ---------------------------------------------------------------
  if (present(root, "quantization_config")) {
    parse_mixed_precision(require(root, "quantization_config"), hf_quant, c);
  } else if (hf_quant != nullptr) {
    parse_nvfp4_exclude(*hf_quant, c);
  } else {
    reject("quantization_config",
           "missing, and no hf_quant_config.json beside config.json — the engine implements the "
           "NVFP4 releases as shipped (NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4, "
           "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4)");
  }
  return c;
}

NemotronHConfig NemotronHConfig::from_json_file(const std::string& path) {
  namespace fs = std::filesystem;
  // The parsed values view the text: both files outlive the parse.
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  const fs::path hf = fs::path(path).parent_path() / "hf_quant_config.json";
  if (!fs::exists(hf)) return parse(parsed.root, nullptr);
  const std::string hf_json = read_file(hf.string());
  const auto hf_parsed = minijson::parse(hf_json);
  return parse(parsed.root, &hf_parsed.root);
}

}  // namespace dgpp
