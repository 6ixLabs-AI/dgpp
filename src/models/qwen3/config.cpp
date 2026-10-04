#include "models/qwen3/config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace dgpp {
namespace {

// The object a refusal names: the flat Qwen3 / Qwen3-MoE config, or the VL
// checkpoint's text_config / vision_config / root while each is parsed.
thread_local const char* g_scope = "Qwen3 config";
struct ScopeGuard {
  const char* saved;
  explicit ScopeGuard(const char* s) : saved(g_scope) { g_scope = s; }
  ~ScopeGuard() { g_scope = saved; }
};

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("{}.{}: {}", g_scope, field, why));
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
int64_t optional_int64(const minijson::Value& v, std::string_view field, int64_t dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_number()) reject(field, "not a number");
  return f->as_int();
}
double require_double(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  const double d = f.as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}
bool optional_bool(const minijson::Value& v, std::string_view field, bool dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_bool()) reject(field, "not a bool");
  return f->as_bool();
}
std::string require_string(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_string()) reject(field, "not a string");
  return std::string(f.as_string());
}
std::string optional_string(const minijson::Value& v, std::string_view field,
                            const std::string& dflt) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) return dflt;
  if (!f->is_string()) reject(field, "not a string");
  return std::string(f->as_string());
}
std::vector<int64_t> require_int_array(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_array()) reject(field, "not an array");
  std::vector<int64_t> out;
  for (const auto& item : f.items()) {
    if (!item.is_number()) reject(field, "non-numeric element");
    out.push_back(item.as_int());
  }
  return out;
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

bool empty_value(const minijson::Value* v) {
  return v == nullptr || v->is_null() || (v->is_object() && v->members().empty());
}

// The text model's fields: `t` is the root of the flat dialects and
// text_config of the VL one. `c.dialect` is set.
void parse_text(const minijson::Value& t, Qwen3TextConfig& c) {
  c.hidden_size = require_int(t, "hidden_size");
  c.vocab_size = require_int(t, "vocab_size");
  c.num_hidden_layers = require_int(t, "num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(require_double(t, "rms_norm_eps"));
  c.hidden_act = require_string(t, "hidden_act");
  c.max_position_embeddings = require_int(t, "max_position_embeddings");
  // The NVFP4 block runs along K, and K is the hidden size on every gate /
  // up / q / k / v matrix: whole 16-blocks, and the kernels' 8-wide lanes.
  if (c.hidden_size <= 0 || c.hidden_size % 16 != 0)
    reject("hidden_size", "must be a positive multiple of 16");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");
  if (c.max_position_embeddings <= 0) reject("max_position_embeddings", "must be positive");
  if (!(c.rms_norm_eps > 0.0f)) reject("rms_norm_eps", "must be positive");
  if (c.hidden_act != "silu") reject("hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (optional_bool(t, "attention_bias", false))
    reject("attention_bias", "biased attention projections are not implemented for this family");
  // Qwen3Config derives `sliding_window` from use_sliding_window (None when
  // false) and `layer_types` from it; every released checkpoint is full
  // attention throughout.
  if (optional_bool(t, "use_sliding_window", false))
    reject("use_sliding_window", "sliding-window attention is not implemented");
  if (const minijson::Value* lt = t.find("layer_types"); lt && !lt->is_null()) {
    if (!lt->is_array()) reject("layer_types", "not an array");
    if (static_cast<int>(lt->items().size()) != c.num_hidden_layers)
      reject("layer_types", "length does not match num_hidden_layers");
    for (const auto& item : lt->items())
      if (!item.is_string() || item.as_string() != "full_attention")
        reject("layer_types", "only full_attention layers are implemented, got '" +
                                  std::string(item.is_string() ? item.as_string() : "?") + "'");
  }

  // --- tokens -----------------------------------------------------------------
  if (const minijson::Value* eos = t.find("eos_token_id"); eos && !eos->is_null()) {
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
  c.bos_token_id = optional_int64(t, "bos_token_id", -1);
  c.pad_token_id = optional_int64(t, "pad_token_id", -1);

  // --- attention --------------------------------------------------------------
  c.num_attention_heads = require_int(t, "num_attention_heads");
  c.num_key_value_heads = require_int(t, "num_key_value_heads");
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 ||
      c.num_attention_heads % c.num_key_value_heads != 0)
    reject("num_key_value_heads", "must divide num_attention_heads");
  // The reference falls back to hidden_size / num_attention_heads when the
  // config names no head_dim (Qwen3Attention's getattr).
  c.head_dim = optional_int(t, "head_dim", c.hidden_size / c.num_attention_heads);
  if (c.head_dim != 128)
    reject("head_dim", "the attention kernels implement 128-wide heads, got " +
                           std::to_string(c.head_dim));
  if (c.q_heads_per_kv() > 32)
    reject("num_attention_heads", "at most 32 query heads per kv head (the attention kernel's group)");
  c.rope_theta = require_double(t, "rope_theta");
  if (!(c.rope_theta > 1.0)) reject("rope_theta", "must be greater than 1");
  {
    // The flat dialects carry `rope_scaling: null`. The VL text model's
    // carries its MROPE layout there with rope_type "default" — no scaling —
    // and the rotary is built over three position axes that a text-only
    // prompt sets equal (config.hpp). A checkpoint that bakes a ramp in
    // (yarn, dynamic, ...) is refused: this family has no rope_scaling path.
    const minijson::Value* rs = t.find("rope_scaling");
    if (c.dialect != Qwen3Dialect::VlMoe) {
      if (rs != nullptr && !rs->is_null())
        reject("rope_scaling", "the checkpoint's rope must be unscaled (null)");
    } else {
      if (rs == nullptr || !rs->is_object())
        reject("rope_scaling", "missing (the VL text model names its MROPE sections there)");
      if (const std::string rt = optional_string(*rs, "rope_type", "default"); rt != "default")
        reject("rope_scaling.rope_type", "only \"default\" (no scaling) is implemented, got " + rt);
      // Qwen3VLMoeTextRotaryEmbedding always interleaves; a config that
      // says otherwise describes another model.
      if (!optional_bool(*rs, "mrope_interleaved", true))
        reject("rope_scaling.mrope_interleaved", "the Qwen3-VL text rotary is interleaved (true)");
      for (const int64_t v : require_int_array(*rs, "mrope_section")) {
        if (v <= 0) reject("rope_scaling.mrope_section", "sections must be positive");
        c.mrope_section.push_back(static_cast<int>(v));
      }
      int64_t sum = 0;
      for (int v : c.mrope_section) sum += v;
      if (c.mrope_section.size() != 3 || sum * 2 != c.head_dim)
        reject("rope_scaling.mrope_section", "must be three sections summing to head_dim / 2");
    }
  }

  // --- MLP / MoE ----------------------------------------------------------------
  c.intermediate_size = optional_int(t, "intermediate_size", 0);
  if (c.dialect == Qwen3Dialect::Dense) {
    if (c.intermediate_size <= 0) reject("intermediate_size", "must be positive");
    // A dense config carries no MoE fields; one that does is another class.
    if (optional_int(t, "num_experts", 0) != 0)
      reject("num_experts", "a routed MoE under model_type qwen3 is not a known checkpoint");
    return;
  }
  c.num_experts = require_int(t, "num_experts");
  c.num_experts_per_tok = require_int(t, "num_experts_per_tok");
  c.moe_intermediate_size = require_int(t, "moe_intermediate_size");
  c.norm_topk_prob = optional_bool(t, "norm_topk_prob", false);  // Qwen3MoeConfig's default
  if (c.num_experts <= 0 || c.num_experts > 4096) reject("num_experts", "must be in [1, 4096]");
  if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > c.num_experts ||
      c.num_experts_per_tok > 16)
    reject("num_experts_per_tok", "must be in [1, min(num_experts, 16)]");
  // The fp4 GEMV core's K and the 16-block column slice of down_proj
  // (GLM-4.7's rule: every expert intermediate slice a multiple of 32).
  if (c.moe_intermediate_size <= 0 || c.moe_intermediate_size % 32 != 0)
    reject("moe_intermediate_size", "must be a positive multiple of 32");
  // Qwen3MoeSparseMoeBlock renormalizes under norm_topk_prob; the VL block
  // renormalizes unconditionally ("all the models use norm_topk_prob"). Both
  // releases set it; the un-normalized router is not a checkpoint that exists.
  if (!c.norm_topk_prob) reject("norm_topk_prob", "the router renormalizes the top-k (true)");
  if (optional_int(t, "decoder_sparse_step", 1) != 1)
    reject("decoder_sparse_step", "every layer is a routed MoE (1)");
  if (const minijson::Value* mo = t.find("mlp_only_layers"); mo && !mo->is_null()) {
    if (!mo->is_array()) reject("mlp_only_layers", "not an array");
    if (!mo->items().empty()) reject("mlp_only_layers", "dense-MLP layers inside the MoE stack are not implemented");
  }
}

Qwen3VisionConfig parse_vision(const minijson::Value& v, int text_hidden) {
  const ScopeGuard scope("Qwen3-VL vision_config");
  if (!v.is_object()) reject("vision_config", "not an object");
  Qwen3VisionConfig c;
  c.depth = require_int(v, "depth");
  c.hidden_size = require_int(v, "hidden_size");
  c.intermediate_size = require_int(v, "intermediate_size");
  c.num_heads = require_int(v, "num_heads");
  c.in_channels = optional_int(v, "in_channels", 3);
  c.patch_size = require_int(v, "patch_size");
  c.temporal_patch_size = require_int(v, "temporal_patch_size");
  c.spatial_merge_size = require_int(v, "spatial_merge_size");
  c.num_position_embeddings = require_int(v, "num_position_embeddings");
  c.out_hidden_size = require_int(v, "out_hidden_size");
  if (c.depth <= 0) reject("depth", "must be positive");
  if (c.hidden_size <= 0 || c.intermediate_size <= 0) reject("hidden_size", "must be positive");
  if (c.num_heads <= 0 || c.hidden_size % c.num_heads != 0) reject("num_heads", "must divide hidden_size");
  if (c.in_channels <= 0 || c.patch_size <= 0 || c.temporal_patch_size <= 0 || c.spatial_merge_size <= 0)
    reject("patch_size", "the patch geometry must be positive");
  if (c.num_position_embeddings <= 0) reject("num_position_embeddings", "must be positive");
  // The merger projects into the text model's residual stream.
  if (c.out_hidden_size != text_hidden)
    reject("out_hidden_size", "must equal text_config.hidden_size");
  for (const int64_t i : require_int_array(v, "deepstack_visual_indexes")) {
    if (i < 0 || i >= c.depth) reject("deepstack_visual_indexes", "index outside [0, depth)");
    c.deepstack_visual_indexes.push_back(static_cast<int>(i));
  }
  return c;
}

// One entry of the recipe's `ignore` list, classified by what the binding
// table does with the module: the head, a layer's router, an attention
// projection (`which` = 'q', 'k' or 'v'), a vision module, or — anything
// else — a module the loader expects quantized.
enum class IgnoreKind { Head, Router, AttnQkv, Vision, Other };
struct IgnoreEntry {
  IgnoreKind kind = IgnoreKind::Other;
  int layer = -1;
  char which = 0;
};
IgnoreEntry classify_ignore(std::string_view name, std::string_view layers_prefix) {
  IgnoreEntry e;
  if (name == "lm_head") {
    e.kind = IgnoreKind::Head;
    return e;
  }
  if (name.rfind("model.visual.", 0) == 0) {
    e.kind = IgnoreKind::Vision;
    return e;
  }
  if (name.rfind(layers_prefix, 0) != 0) return e;
  const std::string_view rest = name.substr(layers_prefix.size());
  const size_t dot = rest.find('.');
  if (dot == std::string_view::npos || dot == 0) return e;
  int layer = 0;
  for (size_t i = 0; i < dot; ++i) {
    if (rest[i] < '0' || rest[i] > '9') return e;
    layer = layer * 10 + (rest[i] - '0');
    if (layer > 100000) return e;
  }
  const std::string_view tail = rest.substr(dot + 1);
  e.layer = layer;
  if (tail == "mlp.gate") {
    e.kind = IgnoreKind::Router;
  } else if (tail == "self_attn.q_proj" || tail == "self_attn.k_proj" || tail == "self_attn.v_proj") {
    e.kind = IgnoreKind::AttnQkv;
    e.which = tail[tail.size() - 6];  // "q_proj": the letter
  } else {
    e.layer = -1;
  }
  return e;
}

// quantization_config of the MoE dialects. Two NVFP4 containers are
// implemented, told apart by what the recipe says of itself: llm-compressor
// writes `format: nvfp4-pack-quantized`; modelopt writes no format and
// `quant_method: modelopt`. The `ignore` list names the modules the release
// left in BF16 — the binding table then expects exactly those in BF16 and
// every other Linear as an NVFP4 set, so an entry this parser does not know
// is a tensor class it would bind wrongly, and is refused by name.
void parse_quantization(const minijson::Value& root, Qwen3TextConfig& c) {
  const ScopeGuard scope(c.vl() ? "Qwen3-VL-MoE quantization_config" : "Qwen3-MoE quantization_config");
  const minijson::Value* q = root.find("quantization_config");
  if (q == nullptr || q->is_null())
    throw std::runtime_error(std::format(
        "{}: missing — the engine implements the NVFP4 releases of this model class "
        "(nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4, ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4); an "
        "unquantized Qwen3-MoE checkpoint has no expert path here",
        g_scope));
  if (!q->is_object()) reject("quantization_config", "not an object");
  const minijson::Value* groups = q->find("config_groups");
  if (groups == nullptr || !groups->is_object() || groups->members().size() != 1)
    reject("config_groups", "exactly one group is implemented");
  const minijson::Value& g0 = groups->members()[0].value;
  const minijson::Value* w = g0.is_object() ? g0.find("weights") : nullptr;
  if (w == nullptr || !w->is_object()) reject("config_groups.weights", "missing");
  if (require_int(*w, "num_bits") != 4 || require_int(*w, "group_size") != 16 ||
      optional_string(*w, "type", "float") != "float")
    reject("config_groups.weights", "only NVFP4 (4-bit float, group 16) is implemented");
  if (const minijson::Value* tg = g0.find("targets"); tg && tg->is_array())
    for (const auto& item : tg->items())
      if (!item.is_string() || item.as_string() != "Linear")
        reject("config_groups.targets", "only the Linear target is implemented");

  const std::string format = optional_string(*q, "format", "");
  const std::string method = optional_string(*q, "quant_method", "");
  if (format == "nvfp4-pack-quantized") {
    if (method != "compressed-tensors")
      reject("quant_method", "nvfp4-pack-quantized is compressed-tensors', got '" + method + "'");
    // Everything that would change what a stored code means is read and
    // held to the one recipe the loader implements (the Coder-Next rule).
    if (!optional_bool(*w, "symmetric", true)) reject("config_groups.weights.symmetric", "must be true");
    if (const std::string st = optional_string(*w, "strategy", "tensor_group"); st != "tensor_group")
      reject("config_groups.weights.strategy", "must be tensor_group, got " + st);
    // A transform (a rotation folded into the weights) or a sparsity mask
    // would make the stored codes something other than the weights.
    if (!empty_value(q->find("transform_config"))) reject("transform_config", "must be empty");
    if (!empty_value(q->find("sparsity_config"))) reject("sparsity_config", "must be empty");
    if (const minijson::Value* kv = q->find("kv_cache_scheme"); kv != nullptr && !kv->is_null())
      reject("kv_cache_scheme", "must be null (this container carries no cache scales)");
    if (const std::string st = optional_string(*q, "quantization_status", "compressed"); st != "compressed")
      reject("quantization_status", "must be compressed, got " + st);
    c.quant_kind = Qwen3QuantKind::Nvfp4Packed;
    c.kv_cache_scales = false;
  } else if (format.empty() && method == "modelopt") {
    if (const std::string algo = optional_string(*q, "quant_algo", ""); algo != "NVFP4")
      reject("quant_algo", "only NVFP4 is implemented, got '" + algo + "'");
    c.quant_kind = Qwen3QuantKind::Nvfp4Modelopt;
    // The recipe's FP8 K/V-cache scheme ships as two F32 scalars per layer;
    // without it the tensors are absent and the table does not expect them.
    const minijson::Value* kv = q->find("kv_cache_scheme");
    c.kv_cache_scales = kv != nullptr && !kv->is_null();
    if (c.kv_cache_scales) {
      if (!kv->is_object() || optional_int(*kv, "num_bits", 8) != 8 ||
          optional_string(*kv, "type", "float") != "float")
        reject("kv_cache_scheme", "only the 8-bit float (FP8) cache scheme is known");
    }
  } else {
    reject("format", "only nvfp4-pack-quantized (compressed-tensors) and the modelopt NVFP4 release "
                     "(quant_method modelopt, no format) are implemented, got format '" +
                         format + "', quant_method '" + method + "'");
  }

  // --- the ignore list ----------------------------------------------------------
  const minijson::Value* ig = q->find("ignore");
  if (ig == nullptr || !ig->is_array()) reject("ignore", "missing (the BF16 modules are named there)");
  const std::string layers_prefix = c.vl() ? "model.language_model.layers." : "model.layers.";
  const bool packed = c.fp4_packed();
  bool head = false;
  size_t vision = 0;
  std::vector<uint8_t> seen(static_cast<size_t>(c.num_hidden_layers), 0);  // bit 0 router, 1 q, 2 k, 3 v
  for (const auto& item : ig->items()) {
    if (!item.is_string()) reject("ignore", "non-string element");
    const std::string_view name = item.as_string();
    const IgnoreEntry e = classify_ignore(name, layers_prefix);
    switch (e.kind) {
      case IgnoreKind::Head:
        head = true;
        break;
      case IgnoreKind::Vision:
        if (!c.vl()) reject("ignore", "'" + std::string(name) + "' names a vision module of a text-only model");
        ++vision;
        break;
      case IgnoreKind::Router:
      case IgnoreKind::AttnQkv: {
        if (e.layer >= c.num_hidden_layers)
          reject("ignore", "'" + std::string(name) + "' names a layer past num_hidden_layers");
        // The compressed-tensors release quantizes q/k/v: an ignored one
        // would be BF16 where the table expects the NVFP4 set.
        if (e.kind == IgnoreKind::AttnQkv && packed)
          reject("ignore", "'" + std::string(name) + "' — this container's loader expects the attention "
                                                     "projections NVFP4");
        const int bit = e.kind == IgnoreKind::Router ? 0 : e.which == 'q' ? 1 : e.which == 'k' ? 2 : 3;
        seen[static_cast<size_t>(e.layer)] |= static_cast<uint8_t>(1u << bit);
        break;
      }
      case IgnoreKind::Other:
        reject("ignore", "'" + std::string(name) + "' — the loader expects this module NVFP4 (only lm_head, "
                                                   "the routers, the modelopt release's q/k/v and the vision "
                                                   "tower ship BF16)");
    }
  }
  // The BF16 modules the table expects: an NVFP4 head or router has no path.
  if (!head) reject("ignore", "lm_head is not listed: an NVFP4 head is not implemented");
  const uint8_t want = packed ? uint8_t{0x1} : uint8_t{0xF};
  for (int l = 0; l < c.num_hidden_layers; ++l)
    if (seen[static_cast<size_t>(l)] != want)
      reject("ignore", std::format("layer {} does not list {} — the loader expects {} BF16 there", l,
                                   packed ? "mlp.gate alone" : "mlp.gate and self_attn.{q,k,v}_proj",
                                   packed ? "the router" : "the router and q/k/v"));
  if (c.vl() && vision == 0)
    reject("ignore", "no model.visual.* module is listed: an NVFP4 vision tower is not what this release ships");
}

}  // namespace

const char* Qwen3TextConfig::scope_name() const {
  return dialect == Qwen3Dialect::Dense ? "Qwen3 config"
         : dialect == Qwen3Dialect::Moe ? "Qwen3-MoE config"
                                        : "Qwen3-VL-MoE config";
}

Qwen3TextConfig Qwen3TextConfig::parse(const minijson::Value& root) {
  if (!root.is_object()) throw std::runtime_error("Qwen3 config: root is not an object");
  Qwen3TextConfig c;
  const std::string model_type = optional_string(root, "model_type", "");
  if (model_type == "qwen3") {
    c.dialect = Qwen3Dialect::Dense;
  } else if (model_type == "qwen3_moe") {
    c.dialect = Qwen3Dialect::Moe;
  } else if (model_type == "qwen3_vl_moe") {
    c.dialect = Qwen3Dialect::VlMoe;
  } else {
    throw std::runtime_error("Qwen3 config.model_type: expected qwen3, qwen3_moe or qwen3_vl_moe, got '" +
                             model_type + "'");
  }

  if (c.dialect == Qwen3Dialect::VlMoe) {
    const minijson::Value* tc = root.find("text_config");
    if (tc == nullptr || !tc->is_object())
      throw std::runtime_error("Qwen3-VL-MoE config.text_config: missing");
    {
      const ScopeGuard scope("Qwen3-VL-MoE text_config");
      if (const std::string tt = optional_string(*tc, "model_type", "qwen3_vl_moe_text"); tt != "qwen3_vl_moe_text")
        reject("model_type", "expected qwen3_vl_moe_text, got " + tt);
      parse_text(*tc, c);
    }
    const ScopeGuard scope("Qwen3-VL-MoE config");
    c.tie_word_embeddings = optional_bool(root, "tie_word_embeddings", false);
    const minijson::Value* vc = root.find("vision_config");
    if (vc == nullptr || vc->is_null()) reject("vision_config", "missing");
    c.vision = parse_vision(*vc, c.hidden_size);
    c.image_token_id = optional_int64(root, "image_token_id", -1);
    c.video_token_id = optional_int64(root, "video_token_id", -1);
    c.vision_start_token_id = optional_int64(root, "vision_start_token_id", -1);
    c.vision_end_token_id = optional_int64(root, "vision_end_token_id", -1);
    // The serving layer refuses a prompt that carries a placeholder: it
    // needs their ids, and the checkpoint is the only place they are named.
    if (c.image_token_id < 0 || c.image_token_id >= c.vocab_size)
      reject("image_token_id", "missing or outside [0, vocab_size)");
    if (c.video_token_id < 0 || c.video_token_id >= c.vocab_size)
      reject("video_token_id", "missing or outside [0, vocab_size)");
  } else {
    const ScopeGuard scope(c.dialect == Qwen3Dialect::Dense ? "Qwen3 config" : "Qwen3-MoE config");
    parse_text(root, c);
    c.tie_word_embeddings = optional_bool(root, "tie_word_embeddings", false);
    if (root.find("vision_config") != nullptr && !root.find("vision_config")->is_null())
      reject("vision_config", "a vision tower under a text-only model_type");
  }

  if (c.dialect == Qwen3Dialect::Dense) {
    // The retrieval models ship unquantized. A quantized dense release
    // would need the dense NVFP4 MLP's binding; none is implemented.
    if (const minijson::Value* q = root.find("quantization_config"); q != nullptr && !q->is_null())
      throw std::runtime_error(
          "Qwen3 config.quantization_config: a quantized dense Qwen3 release is not implemented (the "
          "BF16 retrieval models Qwen3-Reranker / Qwen3-Embedding bind)");
    c.quant_kind = Qwen3QuantKind::Bf16;
  } else {
    // The MoE releases keep an untied head; a tied MoE is not a checkpoint
    // that exists, and the vocab-sharded head has no tied form.
    if (c.tie_word_embeddings)
      throw std::runtime_error(std::string(c.scope_name()) +
                               ".tie_word_embeddings: tied embeddings are not implemented for the MoE dialects");
    parse_quantization(root, c);
  }
  return c;
}

Qwen3TextConfig Qwen3TextConfig::from_json_file(const std::string& path) {
  // The parsed values view the text: it must outlive the parse.
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  return parse(parsed.root);
}

GlmMoeConfig Qwen3TextConfig::moe_config(int local_inter) const {
  GlmMoeConfig m;
  m.hidden = hidden_size;
  m.inter = local_inter;
  m.n_experts = num_experts;
  m.top_k = num_experts_per_tok;
  m.n_shared_experts = 0;
  m.routed_scaling_factor = 1.0f;
  m.norm_topk_prob = norm_topk_prob;
  m.swiglu_limit = std::numeric_limits<float>::infinity();
  m.router_mode = MoeRouterMode::SoftmaxTopk;
  return m;
}

Glm4TextConfig Qwen3TextConfig::attention_view() const {
  Glm4TextConfig g;
  g.hidden_size = hidden_size;
  g.vocab_size = vocab_size;
  g.num_hidden_layers = num_hidden_layers;
  g.rms_norm_eps = rms_norm_eps;
  g.tie_word_embeddings = tie_word_embeddings;
  g.hidden_act = hidden_act;
  g.max_position_embeddings = max_position_embeddings;
  g.first_k_dense_replace = 0;
  g.eos_token_ids = eos_token_ids;
  g.pad_token_id = pad_token_id;
  g.num_attention_heads = num_attention_heads;
  g.num_key_value_heads = num_key_value_heads;
  g.head_dim = head_dim;
  g.attention_bias = false;
  g.use_qk_norm = true;
  g.rotary_dim = head_dim;  // full rotary: pairs (i, i + head_dim / 2)
  g.rope_theta = rope_theta;
  g.intermediate_size = intermediate_size;
  g.moe_intermediate_size = moe_intermediate_size;
  g.n_routed_experts = num_experts;
  g.n_shared_experts = 0;
  g.num_experts_per_tok = num_experts_per_tok;
  g.norm_topk_prob = norm_topk_prob;
  g.routed_scaling_factor = 1.0f;
  g.num_nextn_predict_layers = 0;
  g.kv_scales_present = kv_cache_scales;
  return g;
}

}  // namespace dgpp
