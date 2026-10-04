#include "models/minimax/config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("MiniMax-M2 config.{}: {}", field, why));
}

const minijson::Value& require(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) reject(field, "missing");
  return *f;
}
int require_int(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  const double d = f.as_double();
  if (!std::isfinite(d) || d != std::floor(d) || std::fabs(d) > 2147483647.0)
    reject(field, "not an integer");
  return static_cast<int>(f.as_int());
}
int optional_int(const minijson::Value& v, std::string_view field, int dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  return require_int(v, field);
}
double require_double(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  const double d = f.as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}
double optional_double(const minijson::Value& v, std::string_view field, double dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  return require_double(v, field);
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

// One 4-bit float scheme of the group (weights, or the activations beside them).
void check_fp4_args(const minijson::Value& a, const std::string& field, int* group_out) {
  if (!a.is_object()) reject(field, "not an object");
  const minijson::Value* bits = a.find("num_bits");
  if (!bits || !bits->is_number() || bits->as_int() != 4) reject(field + ".num_bits", "must be 4");
  const minijson::Value* type = a.find("type");
  if (type && !type->is_null() && !(type->is_string() && type->as_string() == "float"))
    reject(field + ".type", "must be float (e2m1)");
  const minijson::Value* group = a.find("group_size");
  if (!group || !group->is_number() || group->as_int() != 16)
    reject(field + ".group_size", "the NVFP4 kernels implement 16");
  if (group_out) *group_out = 16;
}

// The modelopt NVFP4 contract: quant_method modelopt, quant_algo NVFP4, one
// Linear group of 4-bit floats in blocks of 16, and the ignore list naming
// exactly lm_head and every layer's attention and router — the loader
// expects BF16 there and NVFP4 triples on the routed experts, so an ignore
// entry it does not know is a tensor class it would bind wrongly.
void parse_quantization(const minijson::Value& root, MinimaxTextConfig& c) {
  const minijson::Value* qc = root.find("quantization_config");
  if (!qc || qc->is_null())
    reject("quantization_config",
           "missing — the engine implements the NVFP4 release (modelopt); the FP8 base checkpoint "
           "(MiniMaxAI/MiniMax-M2.7) has no expert path here");
  if (!qc->is_object()) reject("quantization_config", "not an object");
  if (const std::string m = optional_string(*qc, "quant_method", ""); m != "modelopt")
    reject("quantization_config.quant_method", "only modelopt is implemented, got '" + m + "'");
  if (const std::string a = optional_string(*qc, "quant_algo", ""); a != "NVFP4")
    reject("quantization_config.quant_algo", "only NVFP4 is implemented, got '" + a + "'");
  if (const minijson::Value* kv = qc->find("kv_cache_scheme"); kv && !kv->is_null())
    reject("quantization_config.kv_cache_scheme",
           "a KV cache scheme is not implemented (the release has none)");
  const minijson::Value* groups = qc->find("config_groups");
  if (!groups || !groups->is_object() || groups->members().empty())
    reject("quantization_config.config_groups", "missing");
  if (groups->members().size() != 1)
    reject("quantization_config.config_groups", "exactly one group is implemented");
  const auto& gm = groups->members().front();
  const std::string gf = "quantization_config.config_groups." + gm.key;
  const minijson::Value& g = gm.value;
  if (!g.is_object()) reject(gf, "group is not an object");
  const minijson::Value* w = g.find("weights");
  if (!w || w->is_null()) reject(gf + ".weights", "missing");
  check_fp4_args(*w, gf + ".weights", &c.fp4_group_size);
  // The recipe's W4A4 activations: accepted and not applied (input_scale is unused).
  if (const minijson::Value* ia = g.find("input_activations"); ia && !ia->is_null())
    check_fp4_args(*ia, gf + ".input_activations", nullptr);
  const minijson::Value* targets = g.find("targets");
  if (!targets || !targets->is_array() || targets->items().size() != 1 ||
      !targets->items()[0].is_string() || targets->items()[0].as_string() != "Linear")
    reject(gf + ".targets", "only the single class target [\"Linear\"] is implemented");

  std::set<std::string> ignore;
  if (const minijson::Value* ig = qc->find("ignore"); ig && ig->is_array())
    for (const auto& item : ig->items()) {
      if (!item.is_string()) reject("quantization_config.ignore", "non-string entry");
      ignore.insert(std::string(item.as_string()));
    }
  std::set<std::string> want;
  want.insert("lm_head");
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    want.insert("model.layers." + std::to_string(l) + ".self_attn*");
    want.insert("model.layers." + std::to_string(l) + ".block_sparse_moe.gate");
  }
  for (const std::string& name : want)
    if (!ignore.count(name))
      reject(
          "quantization_config.ignore",
          "'" + name +
              "' is not ignored — the engine expects BF16 there (attention, the router, the head)");
  for (const std::string& name : ignore)
    if (!want.count(name))
      reject("quantization_config.ignore", "unexpected ignored module '" + name +
                                               "' (a BF16 class the loader does not implement)");
}

}  // namespace

MinimaxTextConfig MinimaxTextConfig::parse(const minijson::Value& root) {
  if (!root.is_object()) reject("", "root is not an object");
  MinimaxTextConfig c;
  if (const std::string mt = optional_string(root, "model_type", "minimax_m2"); mt != "minimax_m2")
    reject("model_type", "expected minimax_m2, got " + mt);

  c.hidden_size = require_int(root, "hidden_size");
  c.vocab_size = require_int(root, "vocab_size");
  c.num_hidden_layers = require_int(root, "num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(require_double(root, "rms_norm_eps"));
  c.tie_word_embeddings = optional_bool(root, "tie_word_embeddings", false);
  c.max_position_embeddings = require_int(root, "max_position_embeddings");
  if (c.hidden_size <= 0 || c.hidden_size % 16 != 0)
    reject("hidden_size", "must be a positive multiple of 16 (the NVFP4 block)");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");
  if (!(c.rms_norm_eps > 0)) reject("rms_norm_eps", "must be positive");
  if (const std::string act = optional_string(root, "hidden_act", "silu"); act != "silu")
    reject("hidden_act", "only silu is implemented, got " + act);
  if (c.tie_word_embeddings) reject("tie_word_embeddings", "tied embeddings are not implemented");
  if (c.max_position_embeddings <= 0) reject("max_position_embeddings", "must be positive");
  c.config_bos_token_id = optional_int(root, "bos_token_id", -1);
  c.config_eos_token_id = optional_int(root, "eos_token_id", -1);

  // --- attention ------------------------------------------------------------
  c.num_attention_heads = require_int(root, "num_attention_heads");
  c.num_key_value_heads = require_int(root, "num_key_value_heads");
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 ||
      c.num_attention_heads % c.num_key_value_heads != 0)
    reject("num_key_value_heads", "must divide num_attention_heads");
  c.head_dim = optional_int(root, "head_dim", c.hidden_size / c.num_attention_heads);
  if (c.head_dim <= 0 || c.head_dim % 2 != 0) reject("head_dim", "must be a positive even number");
  if (optional_bool(root, "attention_bias", false))
    reject("attention_bias",
           "biased attention projections are not implemented (the release has none)");
  if (!optional_bool(root, "use_qk_norm", false))
    reject("use_qk_norm", "must be true (the q/k norm is part of the checkpoint)");
  if (const std::string t = optional_string(root, "qk_norm_type", "per_layer"); t != "per_layer")
    reject("qk_norm_type", "only per_layer is implemented, got '" + t + "'");
  if (const minijson::Value* sw = root.find("sliding_window"); sw && !sw->is_null())
    reject("sliding_window", "sliding-window attention is not implemented");
  if (const minijson::Value* at = root.find("attn_type_list"); at && !at->is_null()) {
    if (!at->is_array() || static_cast<int>(at->items().size()) != c.num_hidden_layers)
      reject("attn_type_list", "length != num_hidden_layers");
    for (const auto& item : at->items())
      if (!item.is_number() || item.as_int() != 1)
        reject("attn_type_list", "only type 1 (full attention) is implemented");
  }
  // The rope: default type, a partial rotary dim stated up to three times.
  c.rope_theta = optional_double(root, "rope_theta", 0.0);
  double factor = optional_double(root, "partial_rotary_factor", 1.0);
  if (const minijson::Value* rs = root.find("rope_scaling"); rs && !rs->is_null())
    reject("rope_scaling", "only the default rope is implemented");
  if (const minijson::Value* rp = root.find("rope_parameters"); rp && !rp->is_null()) {
    if (!rp->is_object()) reject("rope_parameters", "not an object");
    if (const std::string t = optional_string(*rp, "rope_type", "default"); t != "default")
      reject("rope_parameters.rope_type", "only the default rope is implemented, got '" + t + "'");
    const double theta = optional_double(*rp, "rope_theta", c.rope_theta);
    if (c.rope_theta != 0.0 && theta != c.rope_theta)
      reject("rope_parameters.rope_theta", "disagrees with rope_theta");
    c.rope_theta = theta;
    const double f = optional_double(*rp, "partial_rotary_factor", factor);
    if (root.find("partial_rotary_factor") && f != factor)
      reject("rope_parameters.partial_rotary_factor", "disagrees with partial_rotary_factor");
    factor = f;
  }
  if (!(c.rope_theta > 1.0)) reject("rope_theta", "missing or not above 1");
  {
    const double rd = c.head_dim * factor;
    if (!(rd > 0) || rd != std::floor(rd) || static_cast<int>(rd) % 2 != 0 || rd > c.head_dim)
      reject("partial_rotary_factor", "rotary dim must be a positive even integer within the head");
    c.rotary_dim = static_cast<int>(rd);
    if (optional_int(root, "rotary_dim", c.rotary_dim) != c.rotary_dim)
      reject("rotary_dim", "disagrees with head_dim * partial_rotary_factor");
  }

  // --- MoE --------------------------------------------------------------------
  c.moe_intermediate_size = require_int(root, "intermediate_size");
  c.n_routed_experts = require_int(root, "num_local_experts");
  c.num_experts_per_tok = require_int(root, "num_experts_per_tok");
  if (c.moe_intermediate_size <= 0 || c.moe_intermediate_size % 16 != 0)
    reject("intermediate_size", "must be a positive multiple of 16 (the NVFP4 block)");
  if (c.n_routed_experts <= 0 || c.n_routed_experts > 4096)
    reject("num_local_experts", "must be in [1, 4096]");
  if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > 16 ||
      c.num_experts_per_tok > c.n_routed_experts)
    reject("num_experts_per_tok", "must be in [1, min(num_local_experts, 16)]");
  if (const std::string sf = optional_string(root, "scoring_func", "sigmoid"); sf != "sigmoid")
    reject("scoring_func", "only sigmoid is implemented, got '" + sf + "'");
  if (!optional_bool(root, "use_routing_bias", true))
    reject("use_routing_bias", "must be true (e_score_correction_bias is part of the checkpoint)");
  if (optional_int(root, "shared_intermediate_size", 0) != 0)
    reject("shared_intermediate_size", "a shared expert is not implemented (the release has none)");

  // --- the MTP head the config declares and no checkpoint carries -------------
  c.mtp_declared = optional_bool(root, "use_mtp", false);
  c.mtp_modules_declared = optional_int(root, "num_mtp_modules", 0);

  parse_quantization(root, c);
  return c;
}

MinimaxTextConfig MinimaxTextConfig::from_json_file(const std::string& path) {
  namespace fs = std::filesystem;
  const fs::path p = fs::is_directory(path) ? fs::path(path) / "config.json" : fs::path(path);
  const std::string json = read_file(p.string());
  const auto parsed = minijson::parse(json);
  return parse(parsed.root);
}

void MinimaxTextConfig::require_kernel_geometry() const {
  auto fail = [](const std::string& what) {
    throw std::invalid_argument("minimax kernel geometry: " + what);
  };
  if (head_dim != 128) fail("head_dim must be 128 (the paged GQA attention kernels)");
  if (hidden_size % 32 != 0) fail("hidden_size must be a multiple of 32 (the NVFP4 GEMV core's K)");
  if (moe_intermediate_size % 32 != 0) fail("intermediate_size must be a multiple of 32");
  if (fp4_group_size != 16) fail("the NVFP4 block must be 16");
}

float MinimaxTextConfig::attention_scale() const {
  return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

std::vector<float> MinimaxTextConfig::rope_inv_freq() const {
  std::vector<float> out(static_cast<size_t>(rotary_dim / 2));
  for (int i = 0; i < rotary_dim / 2; ++i) {
    // torch: base ** (arange(0, dim, 2, float32) / dim) in fp32, then 1 / it.
    const float e = static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
    out[static_cast<size_t>(i)] =
        1.0f / static_cast<float>(std::pow(rope_theta, static_cast<double>(e)));
  }
  return out;
}

GlmMoeConfig MinimaxTextConfig::moe_config(int local_inter) const {
  GlmMoeConfig m;
  m.hidden = hidden_size;
  m.inter = local_inter;
  m.n_experts = n_routed_experts;
  m.top_k = num_experts_per_tok;
  m.n_shared_experts = 0;
  m.routed_scaling_factor = 1.0f;
  m.norm_topk_prob = true;
  m.swiglu_limit = std::numeric_limits<float>::infinity();  // MiniMaxM2MLP: no clamps
  m.router_mode = MoeRouterMode::SigmoidBias;
  GlmMoeConfig::validate_config(m);
  return m;
}

}  // namespace dgpp
