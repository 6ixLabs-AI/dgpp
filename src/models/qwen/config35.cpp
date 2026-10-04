#include "models/qwen/config35.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <stdexcept>
#include <unordered_set>

namespace dgpp {
namespace {

// The object a refusal names: the Qwen3.5 text_config, or the Qwen3-Next
// flat config while parse_qwen3_next() runs (the helpers below are shared).
thread_local const char* g_scope = "Qwen3.5 text_config";
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

// The routed MoE's fields (Qwen3Next's flat config and Qwen3_5Moe's
// text_config spell them alike): softmax top-k routed experts plus a shared
// expert, every layer sparse. `v` is the object that holds them.
void parse_moe_fields(const minijson::Value& v, Qwen35TextConfig& c) {
  c.num_experts = require_int(v, "num_experts");
  c.num_experts_per_tok = require_int(v, "num_experts_per_tok");
  c.moe_intermediate_size = require_int(v, "moe_intermediate_size");
  c.shared_expert_intermediate_size = require_int(v, "shared_expert_intermediate_size");
  c.norm_topk_prob = optional_bool(v, "norm_topk_prob", true);
  if (c.num_experts <= 0 || c.num_experts > 4096) reject("num_experts", "must be in [1, 4096]");
  if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > c.num_experts ||
      c.num_experts_per_tok > 16)
    reject("num_experts_per_tok", "must be in [1, min(num_experts, 16)]");
  if (c.moe_intermediate_size <= 0 || c.moe_intermediate_size % 16 != 0)
    reject("moe_intermediate_size", "must be a positive multiple of 16");
  if (c.shared_expert_intermediate_size <= 0 || c.shared_expert_intermediate_size % 16 != 0)
    reject("shared_expert_intermediate_size", "must be a positive multiple of 16");
  if (!c.norm_topk_prob) reject("norm_topk_prob", "the router renormalizes the top-k (true)");
  if (optional_int(v, "decoder_sparse_step", 1) != 1)
    reject("decoder_sparse_step", "every layer is a routed MoE (1)");
  if (const minijson::Value* mo = v.find("mlp_only_layers"); mo && !mo->is_null()) {
    if (!mo->is_array()) reject("mlp_only_layers", "not an array");
    if (!mo->items().empty()) reject("mlp_only_layers", "dense-MLP layers are not implemented");
  }
}

// compressed-tensors' `nvfp4-pack-quantized`: everything that would change
// what a stored code means is read and held to the one recipe the loader
// implements. `q` is the quantization_config, `groups` its config_groups,
// `w` group_0's weights; `scope` names the config in a refusal.
void require_packed_nvfp4_recipe(const minijson::Value& q, const minijson::Value& groups,
                                 const minijson::Value& w, const char* scope) {
  const auto bad = [scope](const std::string& what) {
    throw std::runtime_error(std::string(scope) +
                             " quantization_config (nvfp4-pack-quantized): " + what);
  };
  if (!optional_bool(w, "symmetric", true)) bad("weights.symmetric must be true");
  if (const std::string st = optional_string(w, "strategy", "tensor_group"); st != "tensor_group")
    bad("weights.strategy must be tensor_group, got " + st);
  if (groups.members().size() != 1) bad("config_groups must hold group_0 alone");
  const auto empty = [](const minijson::Value* v) {
    return v == nullptr || v->is_null() || (v->is_object() && v->members().empty());
  };
  // A transform (a rotation folded into the weights) or a sparsity mask
  // would make the stored codes something other than the weights.
  if (!empty(q.find("transform_config"))) bad("transform_config must be empty");
  if (!empty(q.find("sparsity_config"))) bad("sparsity_config must be empty");
  // The modelopt release carries its K/V-cache scales as tensors; this
  // one names no cache scheme, and one that did would have none to read.
  if (const minijson::Value* kv = q.find("kv_cache_scheme"); kv != nullptr && !kv->is_null())
    bad("kv_cache_scheme must be null");
  if (const std::string st = optional_string(q, "quantization_status", "compressed");
      st != "compressed")
    bad("quantization_status must be compressed, got " + st);
}

// A recipe's `ignore` list as written (module names, or modelopt's trailing-*
// patterns): the entries the single-group NVFP4 recipes of the Qwen3.5
// dialect are held to.
std::unordered_set<std::string> ignore_entries(const minijson::Value& q) {
  std::unordered_set<std::string> out;
  const minijson::Value* ig = q.find("ignore");
  if (ig == nullptr || !ig->is_array()) return out;
  for (const auto& item : ig->items())
    if (item.is_string()) out.emplace(item.as_string());
  return out;
}

std::string read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    throw std::runtime_error(std::format("cannot open config {}: {}", path,
                                         std::strerror(errno)));
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  return text;
}

}  // namespace

Qwen35TextConfig Qwen35TextConfig::parse(const minijson::Value& tc,
                                        const minijson::Value* quantization_config) {
  if (!tc.is_object()) reject("text_config", "not an object");
  Qwen35TextConfig c;
  // qwen3_5_text: the dense family (Qwen3.8-27B). qwen3_5_moe_text: the same
  // stack with the routed MoE in the dense MLP's place
  // (Qwen3_5MoeForConditionalGeneration: Qwen3.6-35B-A3B).
  const std::string model_type = optional_string(tc, "model_type", "qwen3_5_text");
  if (model_type != "qwen3_5_text" && model_type != "qwen3_5_moe_text")
    reject("model_type", "expected qwen3_5_text or qwen3_5_moe_text, got " + model_type);
  const bool moe = model_type == "qwen3_5_moe_text";

  c.hidden_size = require_int(tc, "hidden_size");
  c.vocab_size = require_int(tc, "vocab_size");
  c.num_hidden_layers = require_int(tc, "num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(require_double(tc, "rms_norm_eps"));
  // Absent in some releases' text_config (Qwen/Qwen3.5-122B-A10B-FP8): untied,
  // the class's default — the binding then holds the checkpoint to a stored
  // lm_head.weight.
  c.tie_word_embeddings = optional_bool(tc, "tie_word_embeddings", false);
  c.hidden_act = require_string(tc, "hidden_act");
  c.max_position_embeddings = require_int(tc, "max_position_embeddings");
  if (c.hidden_size <= 0 || c.hidden_size % 8 != 0)
    reject("hidden_size", "must be a positive multiple of 8");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");
  if (c.hidden_act != "silu")
    reject("hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (const minijson::Value* ab = tc.find("attention_bias"))
    if (ab->is_bool() && ab->as_bool())
      reject("attention_bias", "biased attention projections are not implemented");

  // --- layer kinds ------------------------------------------------------------
  {
    const minijson::Value& lt = require(tc, "layer_types");
    if (!lt.is_array()) reject("layer_types", "not an array");
    for (const auto& item : lt.items()) {
      const std::string s = item.is_string() ? std::string(item.as_string()) : "";
      if (s == "linear_attention") c.layers.push_back(Qwen35LayerKind::Gdn);
      else if (s == "full_attention") c.layers.push_back(Qwen35LayerKind::Full);
      else reject("layer_types", "unsupported layer type '" + s + "'");
    }
    if (static_cast<int>(c.layers.size()) != c.num_hidden_layers)
      reject("layer_types", "length does not match num_hidden_layers");
    const int interval = optional_int(tc, "full_attention_interval", 0);
    if (interval > 0)
      for (int i = 0; i < c.num_hidden_layers; ++i)
        if (((i + 1) % interval == 0) != (c.layers[i] == Qwen35LayerKind::Full))
          reject("full_attention_interval",
                 "disagrees with layer_types at layer " + std::to_string(i));
  }

  // --- tokens -----------------------------------------------------------------
  if (const minijson::Value* eos = tc.find("eos_token_id"); eos && !eos->is_null()) {
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
  c.bos_token_id = optional_int64(tc, "bos_token_id", -1);

  // --- Gated DeltaNet ---------------------------------------------------------
  c.gdn_key_heads = require_int(tc, "linear_num_key_heads");
  c.gdn_value_heads = require_int(tc, "linear_num_value_heads");
  c.gdn_key_head_dim = require_int(tc, "linear_key_head_dim");
  c.gdn_value_head_dim = require_int(tc, "linear_value_head_dim");
  c.gdn_conv_width = require_int(tc, "linear_conv_kernel_dim");
  // The swish output gate's kernels land with the layer slice; the config
  // records (and binds) it now so the checkpoint is loadable end to end.
  c.output_gate_type = optional_string(tc, "output_gate_type", "swish");
  if (c.gdn_key_heads <= 0 || c.gdn_value_heads <= 0 ||
      c.gdn_value_heads % c.gdn_key_heads != 0)
    reject("linear_num_value_heads", "must be a positive multiple of linear_num_key_heads");
  if (c.gdn_key_head_dim != 128 || c.gdn_value_head_dim != 128)
    reject("linear_key_head_dim", "the GDN kernels implement 128-wide heads");
  if (c.gdn_conv_width < 2 || c.gdn_conv_width > 8)
    reject("linear_conv_kernel_dim", "must be in [2, 8]");
  if (c.output_gate_type != "swish")
    reject("output_gate_type", "only the swish output gate is implemented, got " +
                                   c.output_gate_type);
  if (const std::string dt = optional_string(tc, "mamba_ssm_dtype", "float32");
      dt != "float32")
    reject("mamba_ssm_dtype", "the recurrent state is float32, got " + dt);

  // --- full attention -----------------------------------------------------------
  c.num_attention_heads = require_int(tc, "num_attention_heads");
  c.num_key_value_heads = require_int(tc, "num_key_value_heads");
  c.head_dim = require_int(tc, "head_dim");
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 ||
      c.num_attention_heads % c.num_key_value_heads != 0)
    reject("num_key_value_heads", "must divide num_attention_heads");
  if (c.head_dim != 256) reject("head_dim", "the full-attention kernels implement 256-wide heads");
  {
    const minijson::Value* rp = tc.find("rope_parameters");
    if (!rp || !rp->is_object()) reject("rope_parameters", "missing");
    c.rope_theta = require_double(*rp, "rope_theta");
    const double factor = require_double(*rp, "partial_rotary_factor");
    const double rd = c.head_dim * factor;
    if (!(rd > 0) || rd != std::floor(rd) || static_cast<int>(rd) % 2 != 0)
      reject("rope_parameters.partial_rotary_factor", "rotary dim must be a positive even integer");
    c.rotary_dim = static_cast<int>(rd);
    if (const std::string rt = optional_string(*rp, "rope_type", "default"); rt != "default")
      // Same checkpoint-vs-engine precedence as Qwen4Exp: the YaRN ramp is
      // the engine's rope_scaling knob, so a checkpoint that bakes one in
      // would be scaled twice.
      reject("rope_parameters.rope_type",
             "the checkpoint's rope must be \"default\": the YaRN ramp is the engine's "
             "rope_scaling knob, got " + rt);
    c.mrope_interleaved = optional_bool(*rp, "mrope_interleaved", false);
    if (const minijson::Value* ms = rp->find("mrope_section"); ms && !ms->is_null()) {
      for (const int64_t v : require_int_array(*rp, "mrope_section"))
        c.mrope_section.push_back(static_cast<int>(v));
      int64_t sum = 0;
      for (int v : c.mrope_section) sum += v;
      if (c.mrope_section.size() != 3 || sum * 2 != c.rotary_dim)
        reject("rope_parameters.mrope_section", "must be three sections summing to rotary_dim / 2");
    }
  }
  c.attn_output_gate = optional_bool(tc, "attn_output_gate", true);
  if (!c.attn_output_gate)
    reject("attn_output_gate", "the checkpoint stacks [q | gate] in q_proj");

  // --- the MLP: dense SwiGLU, or the routed MoE -------------------------------------
  if (moe) {
    // Qwen3_5MoeTopKRouter always renormalizes its top-k and every layer is
    // sparse; the config carries neither switch, and one that did would be
    // held to those values.
    c.intermediate_size = optional_int(tc, "intermediate_size", 0);  // unused: no dense layer
    parse_moe_fields(tc, c);
  } else {
    c.intermediate_size = require_int(tc, "intermediate_size");
    if (c.intermediate_size <= 0) reject("intermediate_size", "must be positive");
    if (optional_int(tc, "num_experts", 0) != 0)
      reject("num_experts", "a routed MoE needs model_type qwen3_5_moe_text");
  }

  // --- MTP ----------------------------------------------------------------------
  c.mtp_num_layers = optional_int(tc, "mtp_num_hidden_layers", 0);
  if (const minijson::Value* m = tc.find("mtp"); m && m->is_object()) {
    const int n = optional_int(*m, "num_hidden_layers", c.mtp_num_layers);
    if (n != c.mtp_num_layers) reject("mtp.num_hidden_layers", "disagrees with mtp_num_hidden_layers");
  }
  if (c.mtp_num_layers != 0 && c.mtp_num_layers != 1)
    reject("mtp_num_hidden_layers", "only the single draft layer is implemented");
  if (optional_bool(tc, "mtp_use_dedicated_embeddings", false))
    reject("mtp_use_dedicated_embeddings", "the draft shares the embeddings");

  // --- quantization ---------------------------------------------------------------
  // No quantization_config: an unquantized release, every matrix BF16.
  if (quantization_config == nullptr || quantization_config->is_null()) {
    c.quant_kind = Qwen35QuantKind::Bf16;
    return c;
  }
  {
    const minijson::Value& q = *quantization_config;
    const std::string method = optional_string(q, "quant_method", "");
    // Intel's AutoRound releases of this model class (`quant_method`
    // "auto-round", GPTQ-packed int4 per 128 on the routed experts and on
    // every GDN and attention projection) have no path here, for the reason
    // the Qwen3Next dialect gives below: the dense projections have no exact
    // resident form.
    if (method == "auto-round" || method == "gptq")
      throw std::runtime_error(
          "Qwen3.5 quantization_config.quant_method: '" + method +
          "' (the AutoRound int4 release) is not implemented — the engine serves this model "
          "class from its FP8, NVFP4 and unquantized releases");
    if (const minijson::Value* groups = q.find("config_groups");
        groups != nullptr && groups->is_object() && groups->find("group_1") == nullptr) {
      // One group: an NVFP4 recipe that says which modules it left alone in
      // its ignore list. Two are implemented, both for the routed-MoE models,
      // and the list is held to what the binding table then expects.
      const minijson::Value* g0 = groups->find("group_0");
      const minijson::Value* w = g0 != nullptr && g0->is_object() ? g0->find("weights") : nullptr;
      if (w == nullptr || !w->is_object())
        throw std::runtime_error("Qwen3.5 quantization_config.config_groups: group_0.weights missing");
      if (require_int(*w, "num_bits") != 4 || require_int(*w, "group_size") != 16 ||
          optional_string(*w, "type", "float") != "float")
        throw std::runtime_error(
            "Qwen3.5 quantization_config.config_groups.group_0.weights: a single group must be "
            "NVFP4 (4-bit float, group 16)");
      if (!moe)
        throw std::runtime_error(
            "Qwen3.5 quantization_config: the single-group NVFP4 recipes are implemented for the "
            "routed-MoE models (qwen3_5_moe_text); a dense model's NVFP4 MLP and GDN have no path");
      const std::unordered_set<std::string> ignored = ignore_entries(q);
      const auto must_ignore = [&](const std::string& entry, const char* release) {
        if (ignored.count(entry) == 0)
          throw std::runtime_error("Qwen3.5 quantization_config.ignore: '" + entry +
                                   "' missing — " + release);
      };
      const std::string format = optional_string(q, "format", "");
      if (format == "nvfp4-pack-quantized") {
        // compressed-tensors (Sehyo/Qwen3.5-122B-A10B-NVFP4): the attention
        // q/k/v/o, the shared expert and the routed experts quantized; the
        // Gated DeltaNet and the head left BF16.
        static const char* kRelease =
            "the engine implements this container with the Gated DeltaNet and the head left "
            "BF16 (Sehyo/Qwen3.5-122B-A10B-NVFP4)";
        require_packed_nvfp4_recipe(q, *groups, *w, "Qwen3.5");
        must_ignore("lm_head", kRelease);
        for (int i = 0; i < c.num_hidden_layers; ++i) {
          if (c.layers[i] != Qwen35LayerKind::Gdn) continue;
          const std::string la = "model.language_model.layers." + std::to_string(i) + ".linear_attn.";
          must_ignore(la + "in_proj_qkv", kRelease);
          must_ignore(la + "in_proj_z", kRelease);
          must_ignore(la + "out_proj", kRelease);
        }
        c.quant_kind = Qwen35QuantKind::Nvfp4Packed;
        return c;
      }
      if (!format.empty())
        throw std::runtime_error(
            "Qwen3.5 quantization_config.format: only nvfp4-pack-quantized and the modelopt "
            "recipes (no format) are implemented, got '" + format + "'");
      if (const std::string algo = optional_string(q, "quant_algo", ""); algo != "NVFP4")
        throw std::runtime_error(
            "Qwen3.5 quantization_config.quant_algo: a single-group modelopt recipe must be "
            "NVFP4, got '" + algo + "'");
      // modelopt NVFP4 (nvidia/Qwen3.5-122B-A10B-NVFP4): the routed experts
      // alone quantized. Its patterns, as modelopt writes them.
      static const char* kRelease =
          "the engine implements modelopt's NVFP4 recipe with the routed experts alone "
          "quantized (nvidia/Qwen3.5-122B-A10B-NVFP4)";
      must_ignore("lm_head", kRelease);
      for (int i = 0; i < c.num_hidden_layers; ++i) {
        const std::string lp = "model.language_model.layers." + std::to_string(i) + ".";
        must_ignore(lp + (c.layers[i] == Qwen35LayerKind::Gdn ? "linear_attn*" : "self_attn*"),
                    kRelease);
        must_ignore(lp + "mlp.shared_expert*", kRelease);
      }
      if (c.mtp_num_layers == 1) must_ignore("mtp*", kRelease);
      c.quant_kind = Qwen35QuantKind::Nvfp4Experts;
      return c;
    }
    if (const minijson::Value* groups = q.find("config_groups");
        groups != nullptr && groups->is_object()) {
      // The NVFP4 mixed release (modelopt MIXED_PRECISION, config35.hpp):
      // group_0 is the 8-bit float class (per-tensor FP8: the GDN and
      // attention projections), group_1 the 4-bit float per 16 (NVFP4: the
      // experts, the shared expert, the head). The binding table holds the
      // checkpoint to which tensor is in which.
      const minijson::Value* g1 = groups->find("group_1");
      const minijson::Value* w = g1 != nullptr ? g1->find("weights") : nullptr;
      if (w == nullptr || !w->is_object())
        throw std::runtime_error("Qwen3.5 quantization_config.config_groups: group_1.weights missing");
      const int64_t bits = require_int(*w, "num_bits");
      const int64_t group = require_int(*w, "group_size");
      if (bits != 4 || group != 16)
        throw std::runtime_error("Qwen3.5 quantization_config.config_groups: only NVFP4 (4-bit, group 16) is implemented");
      const minijson::Value* g0 = groups->find("group_0");
      const minijson::Value* w0 = g0 != nullptr ? g0->find("weights") : nullptr;
      if (w0 == nullptr || !w0->is_object() || require_int(*w0, "num_bits") != 8 ||
          optional_string(*w0, "type", "float") != "float")
        throw std::runtime_error(
            "Qwen3.5 quantization_config.config_groups: group_0.weights must be the 8-bit float "
            "class (FP8) beside group_1's NVFP4");
      c.quant_kind = Qwen35QuantKind::Nvfp4Mixed;
      return c;
    }
    if (method != "fp8")
      throw std::runtime_error("Qwen3.5 quantization_config.quant_method: only fp8 is implemented, got '" + method + "'");
    const std::vector<int64_t> bs = require_int_array(q, "weight_block_size");
    if (bs.size() != 2 || bs[0] != 128 || bs[1] != 128)
      throw std::runtime_error("Qwen3.5 quantization_config.weight_block_size: only [128, 128] is implemented");
    if (const std::string scheme = optional_string(q, "activation_scheme", "dynamic"); scheme != "dynamic")
      throw std::runtime_error("Qwen3.5 quantization_config.activation_scheme: only dynamic is implemented");
    c.quant_kind = Qwen35QuantKind::Fp8Block;
  }
  return c;
}

Qwen35TextConfig Qwen35TextConfig::parse_qwen3_next(const minijson::Value& root) {
  const ScopeGuard scope("Qwen3-Next config");
  if (!root.is_object()) reject("config", "not an object");
  Qwen35TextConfig c;
  c.dialect = Qwen35Dialect::Qwen3Next;
  const std::string model_type = optional_string(root, "model_type", "qwen3_next");
  if (model_type != "qwen3_next") reject("model_type", "expected qwen3_next, got " + model_type);

  c.hidden_size = require_int(root, "hidden_size");
  c.vocab_size = require_int(root, "vocab_size");
  c.num_hidden_layers = require_int(root, "num_hidden_layers");
  c.rms_norm_eps = static_cast<float>(require_double(root, "rms_norm_eps"));
  c.tie_word_embeddings = optional_bool(root, "tie_word_embeddings", false);
  c.hidden_act = require_string(root, "hidden_act");
  c.max_position_embeddings = require_int(root, "max_position_embeddings");
  if (c.hidden_size <= 0 || c.hidden_size % 8 != 0)
    reject("hidden_size", "must be a positive multiple of 8");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("num_hidden_layers", "must be positive");
  if (c.hidden_act != "silu")
    reject("hidden_act", "only silu is implemented, got " + c.hidden_act);
  if (c.tie_word_embeddings)
    reject("tie_word_embeddings", "tied embeddings are not implemented");
  if (optional_bool(root, "attention_bias", false))
    reject("attention_bias", "biased attention projections are not implemented");
  if (optional_bool(root, "use_sliding_window", false))
    reject("use_sliding_window", "sliding-window attention is not implemented");

  // --- layer kinds ------------------------------------------------------------
  // The transformers config derives layer_types from the interval when the
  // list is absent; the released config.json writes both, and they must agree.
  {
    const int interval = optional_int(root, "full_attention_interval", 4);
    if (interval <= 0) reject("full_attention_interval", "must be positive");
    const minijson::Value* lt = root.find("layer_types");
    if (lt != nullptr && !lt->is_null()) {
      if (!lt->is_array()) reject("layer_types", "not an array");
      for (const auto& item : lt->items()) {
        const std::string s = item.is_string() ? std::string(item.as_string()) : "";
        if (s == "linear_attention") c.layers.push_back(Qwen35LayerKind::Gdn);
        else if (s == "full_attention") c.layers.push_back(Qwen35LayerKind::Full);
        else reject("layer_types", "unsupported layer type '" + s + "'");
      }
      if (static_cast<int>(c.layers.size()) != c.num_hidden_layers)
        reject("layer_types", "length does not match num_hidden_layers");
      for (int i = 0; i < c.num_hidden_layers; ++i)
        if (((i + 1) % interval == 0) != (c.layers[i] == Qwen35LayerKind::Full))
          reject("full_attention_interval",
                 "disagrees with layer_types at layer " + std::to_string(i));
    } else {
      for (int i = 0; i < c.num_hidden_layers; ++i)
        c.layers.push_back((i + 1) % interval == 0 ? Qwen35LayerKind::Full : Qwen35LayerKind::Gdn);
    }
  }

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
  c.bos_token_id = optional_int64(root, "bos_token_id", -1);

  // --- Gated DeltaNet ---------------------------------------------------------
  c.gdn_key_heads = require_int(root, "linear_num_key_heads");
  c.gdn_value_heads = require_int(root, "linear_num_value_heads");
  c.gdn_key_head_dim = require_int(root, "linear_key_head_dim");
  c.gdn_value_head_dim = require_int(root, "linear_value_head_dim");
  c.gdn_conv_width = require_int(root, "linear_conv_kernel_dim");
  // Qwen3NextRMSNormGated multiplies by silu(z): the swish gate. The config
  // carries no field for it; one that names another gate is refused.
  c.output_gate_type = optional_string(root, "output_gate_type", "swish");
  if (c.output_gate_type == "silu") c.output_gate_type = "swish";
  if (c.gdn_key_heads <= 0 || c.gdn_value_heads <= 0 ||
      c.gdn_value_heads % c.gdn_key_heads != 0)
    reject("linear_num_value_heads", "must be a positive multiple of linear_num_key_heads");
  if (c.gdn_key_head_dim != 128 || c.gdn_value_head_dim != 128)
    reject("linear_key_head_dim", "the GDN kernels implement 128-wide heads");
  if (c.gdn_conv_width < 2 || c.gdn_conv_width > 8)
    reject("linear_conv_kernel_dim", "must be in [2, 8]");
  if (c.output_gate_type != "swish")
    reject("output_gate_type", "only the swish output gate is implemented, got " +
                                   c.output_gate_type);
  if (const std::string dt = optional_string(root, "mamba_ssm_dtype", "float32");
      dt != "float32")
    reject("mamba_ssm_dtype", "the recurrent state is float32, got " + dt);

  // --- full attention -----------------------------------------------------------
  c.num_attention_heads = require_int(root, "num_attention_heads");
  c.num_key_value_heads = require_int(root, "num_key_value_heads");
  c.head_dim = require_int(root, "head_dim");
  if (c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 ||
      c.num_attention_heads % c.num_key_value_heads != 0)
    reject("num_key_value_heads", "must divide num_attention_heads");
  if (c.head_dim != 256) reject("head_dim", "the full-attention kernels implement 256-wide heads");
  {
    // Flat rope fields (no rope_parameters object, no MROPE: text only).
    c.rope_theta = require_double(root, "rope_theta");
    const double factor = require_double(root, "partial_rotary_factor");
    const double rd = c.head_dim * factor;
    if (!(rd > 0) || rd != std::floor(rd) || static_cast<int>(rd) % 2 != 0)
      reject("partial_rotary_factor", "rotary dim must be a positive even integer");
    c.rotary_dim = static_cast<int>(rd);
    if (const minijson::Value* rs = root.find("rope_scaling"); rs && !rs->is_null())
      reject("rope_scaling", "the checkpoint's rope must be unscaled (null)");
    c.mrope_interleaved = false;
  }
  c.attn_output_gate = true;  // Qwen3NextAttention always stacks [q | gate] in q_proj

  // --- routed MoE ---------------------------------------------------------------
  c.intermediate_size = optional_int(root, "intermediate_size", 0);  // unused: no dense layer
  parse_moe_fields(root, c);

  // --- quantization ---------------------------------------------------------------
  // Two NVFP4 containers of this model class are implemented, told apart by
  // what the recipe says of itself (the binding table then holds the
  // checkpoint to that container's tensor names): modelopt's writes no
  // `format`; llm-compressor's writes "nvfp4-pack-quantized".
  {
    const minijson::Value* q = root.find("quantization_config");
    // Intel's AutoRound release of Qwen3-Coder-Next (`quant_method`
    // "auto-round", GPTQ-packed int4 per 128 on every projection, the router
    // included) has no path here: the routed experts would bind to the packed
    // core, but the GDN and attention projections have no exact resident form
    // (a 4-bit code x an f16 scale does not fit BF16's significand).
    if (q != nullptr && q->is_object()) {
      const std::string method = optional_string(*q, "quant_method", "");
      if (method == "auto-round" || method == "gptq")
        throw std::runtime_error(
            "Qwen3-Next quantization_config.quant_method: '" + method +
            "' (the AutoRound int4 release) is not implemented — the engine serves the NVFP4 "
            "releases of this model class (RedHatAI/Qwen3-Coder-Next-NVFP4, "
            "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4)");
    }
    const minijson::Value* groups = q != nullptr && q->is_object() ? q->find("config_groups") : nullptr;
    const minijson::Value* g0 = groups != nullptr && groups->is_object() ? groups->find("group_0") : nullptr;
    const minijson::Value* w = g0 != nullptr && g0->is_object() ? g0->find("weights") : nullptr;
    if (w == nullptr || !w->is_object())
      throw std::runtime_error(
          "Qwen3-Next quantization_config: config_groups.group_0.weights missing — the engine "
          "implements the NVIDIA NVFP4 release (nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4)");
    const int64_t bits = require_int(*w, "num_bits");
    const int64_t group = require_int(*w, "group_size");
    if (bits != 4 || group != 16 || optional_string(*w, "type", "float") != "float")
      throw std::runtime_error(
          "Qwen3-Next quantization_config.config_groups.group_0.weights: only NVFP4 "
          "(4-bit float, group 16) is implemented");
    const std::string format = optional_string(*q, "format", "");
    if (format.empty()) {
      c.quant_kind = Qwen35QuantKind::Nvfp4Modelopt;
    } else if (format == "nvfp4-pack-quantized") {
      require_packed_nvfp4_recipe(*q, *groups, *w, "Qwen3-Next");
      c.quant_kind = Qwen35QuantKind::Nvfp4Packed;
    } else {
      throw std::runtime_error(
          "Qwen3-Next quantization_config.format: only the modelopt NVFP4 release (no format) and "
          "nvfp4-pack-quantized are implemented, got '" + format + "'");
    }
  }

  // --- MTP ----------------------------------------------------------------------
  // The config has no field for the draft layer: the recipe identifies the
  // release (config35.hpp), and a config that names the field overrides it.
  c.mtp_num_layers = optional_int(root, "mtp_num_hidden_layers",
                                  c.quant_kind == Qwen35QuantKind::Nvfp4Packed ? 0 : 1);
  if (c.mtp_num_layers != 0 && c.mtp_num_layers != 1)
    reject("mtp_num_hidden_layers", "only the single draft layer is implemented");
  if (optional_bool(root, "mtp_use_dedicated_embeddings", false))
    reject("mtp_use_dedicated_embeddings", "the draft shares the embeddings");
  return c;
}

Qwen35TextConfig Qwen35TextConfig::from_json_file(const std::string& path) {
  // The parsed values view the text: it must outlive the parse.
  const std::string json = read_file(path);
  const auto parsed = minijson::parse(json);
  const minijson::Value* tc = parsed.root.find("text_config");
  if (tc == nullptr) {
    // The Qwen3Next dialect is flat: the root is the text config.
    if (const minijson::Value* mt = parsed.root.find("model_type");
        mt != nullptr && mt->is_string() && mt->as_string() == "qwen3_next")
      return parse_qwen3_next(parsed.root);
    throw std::runtime_error("config " + path + ": missing text_config object");
  }
  return parse(*tc, parsed.root.find("quantization_config"));
}

int Qwen35TextConfig::num_gdn_layers() const {
  int n = 0;
  for (auto k : layers) n += k == Qwen35LayerKind::Gdn;
  return n;
}
int Qwen35TextConfig::num_full_layers() const {
  int n = 0;
  for (auto k : layers) n += k == Qwen35LayerKind::Full;
  return n;
}

}  // namespace dgpp
