#include "models/mistral4/config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <initializer_list>
#include <limits>
#include <regex>
#include <set>
#include <stdexcept>
#include <string_view>

#include "kernels/rope_scaling.hpp"

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("Mistral-Small-4 params.{}: {}", field, why));
}

const minijson::Value& require(const minijson::Value& v, std::string_view key,
                               std::string_view field) {
  const minijson::Value* f = v.find(key);
  if (!f || f->is_null()) reject(field, "missing");
  return *f;
}
int require_int(const minijson::Value& v, std::string_view key, std::string_view field) {
  const minijson::Value& f = require(v, key, field);
  if (!f.is_number()) reject(field, "not a number");
  const double d = f.as_double();
  if (!std::isfinite(d) || d != std::floor(d) || std::fabs(d) > 2147483647.0)
    reject(field, "not an integer");
  return static_cast<int>(f.as_int());
}
int optional_int(const minijson::Value& v, std::string_view key, std::string_view field, int dflt) {
  const minijson::Value* f = v.find(key);
  if (!f || f->is_null()) return dflt;
  return require_int(v, key, field);
}
double require_double(const minijson::Value& v, std::string_view key, std::string_view field) {
  const minijson::Value& f = require(v, key, field);
  if (!f.is_number()) reject(field, "not a number");
  const double d = f.as_double();
  if (!std::isfinite(d)) reject(field, "not finite");
  return d;
}
double optional_double(const minijson::Value& v, std::string_view key, std::string_view field,
                       double dflt) {
  const minijson::Value* f = v.find(key);
  if (!f || f->is_null()) return dflt;
  return require_double(v, key, field);
}
bool optional_bool(const minijson::Value& v, std::string_view key, std::string_view field,
                   bool dflt) {
  const minijson::Value* f = v.find(key);
  if (!f || f->is_null()) return dflt;
  if (!f->is_bool()) reject(field, "not a bool");
  return f->as_bool();
}
std::string optional_string(const minijson::Value& v, std::string_view key, std::string_view field,
                            const std::string& dflt) {
  const minijson::Value* f = v.find(key);
  if (!f || f->is_null()) return dflt;
  if (!f->is_string()) reject(field, "not a string");
  return std::string(f->as_string());
}
// An object whose every key changes the arithmetic: a key outside `known`
// is a rule this parser does not implement, not a default to assume
// (vLLM's adapter asserts the same of `yarn`).
void require_known_keys(const minijson::Value& obj, std::string_view field,
                        std::initializer_list<std::string_view> known) {
  for (const auto& m : obj.members()) {
    bool ok = false;
    for (std::string_view k : known) ok = ok || m.key == k;
    if (!ok) reject(std::string(field) + "." + m.key, "unknown key (not implemented)");
  }
}

std::string read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    throw std::runtime_error(std::format("cannot open params {}: {}", path, std::strerror(errno)));
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  return text;
}

// --- the quantization contract ----------------------------------------------

// One compressed-tensors `ignore` rule: a `re:` regex anchored at the start
// of the module name (python's re.match), or an exact module name.
struct IgnoreRule {
  bool is_regex = false;
  std::regex re;
  std::string literal;
  bool matches(const std::string& module) const {
    return is_regex ? std::regex_search(module, re) : module == literal;
  }
};

IgnoreRule make_rule(const std::string& text) {
  IgnoreRule r;
  if (text.rfind("re:", 0) == 0) {
    r.is_regex = true;
    try {
      r.re = std::regex("^(?:" + text.substr(3) + ")", std::regex::ECMAScript);
    } catch (const std::regex_error& e) {
      reject("quantization_config.ignore", "unparseable regex '" + text + "': " + e.what());
    }
  } else {
    r.literal = text;
  }
  return r;
}

// The group's `weights` (and, when present, `input_activations`) scheme:
// 4-bit float in groups of 16 with a global scale — NVFP4.
void check_fp4_args(const minijson::Value& a, const std::string& field, bool activations,
                    Mistral4TextConfig& c) {
  if (!a.is_object()) reject(field, "not an object");
  if (require_int(a, "num_bits", field + ".num_bits") != 4)
    reject(field + ".num_bits", "must be 4");
  if (const std::string t = optional_string(a, "type", field + ".type", "float"); t != "float")
    reject(field + ".type", "must be float (e2m1), got '" + t + "'");
  const int group = require_int(a, "group_size", field + ".group_size");
  if (group != 16) reject(field + ".group_size", "the NVFP4 kernels implement 16");
  if (const std::string s = optional_string(a, "strategy", field + ".strategy", "tensor_group");
      s != "tensor_group")
    reject(field + ".strategy", "only tensor_group is implemented, got '" + s + "'");
  if (!optional_bool(a, "symmetric", field + ".symmetric", true))
    reject(field + ".symmetric", "must be true");
  if (const minijson::Value* bs = a.find("block_structure"); bs && !bs->is_null())
    reject(field + ".block_structure", "block structures are not implemented");
  if (const minijson::Value* ao = a.find("actorder"); ao && !ao->is_null())
    reject(field + ".actorder", "activation ordering is not implemented");
  if (activations) {
    // `dynamic: "local"`: per-16 scales computed at run time under a static
    // global one (the input_global_scale tensor).
    const minijson::Value* d = a.find("dynamic");
    if (!d || !d->is_string() || d->as_string() != "local")
      reject(field + ".dynamic",
             "only \"local\" (dynamic block scales under a stored global scale) is implemented");
  } else {
    const minijson::Value* d = a.find("dynamic");
    if (d && !d->is_null() && !(d->is_bool() && !d->as_bool()))
      reject(field + ".dynamic", "must be false");
    if (const std::string sd =
            optional_string(a, "scale_dtype", field + ".scale_dtype", "torch.float8_e4m3fn");
        sd != "torch.float8_e4m3fn")
      reject(field + ".scale_dtype", "block scales must be e4m3, got '" + sd + "'");
    if (const minijson::Value* zp = a.find("zp_dtype"); zp && !zp->is_null())
      reject(field + ".zp_dtype", "zero points are not implemented");
    c.fp4_group_size = group;
  }
}

void parse_quantization(const minijson::Value& root, Mistral4TextConfig& c) {
  // Mistral's own FP8 container (the base repository's `quantization` key).
  if (const minijson::Value* q = root.find("quantization"); q && !q->is_null())
    reject("quantization",
           "Mistral's FP8 container (qformat_weight / qscale tensors, the base release) is not "
           "implemented; "
           "the engine reads the compressed-tensors NVFP4 release");
  const minijson::Value* qc = root.find("quantization_config");
  if (!qc || qc->is_null())
    reject("quantization_config",
           "missing — the engine implements the NVFP4 release (compressed-tensors); an unquantized "
           "checkpoint has no expert path here");
  if (!qc->is_object()) reject("quantization_config", "not an object");
  std::string method = optional_string(*qc, "quant_method", "quantization_config.quant_method", "");
  for (char& ch : method)
    if (ch == '_') ch = '-';  // vLLM normalizes compressed_tensors to the community spelling
  if (method != "compressed-tensors")
    reject("quantization_config.quant_method",
           "only compressed-tensors is implemented, got '" + method + "'");
  if (const std::string f = optional_string(*qc, "format", "quantization_config.format", "");
      f != "nvfp4-pack-quantized")
    reject("quantization_config.format",
           "only nvfp4-pack-quantized is implemented, got '" + f + "'");
  if (const std::string s = optional_string(
          *qc, "quantization_status", "quantization_config.quantization_status", "compressed");
      s != "compressed")
    reject("quantization_config.quantization_status", "must be compressed, got '" + s + "'");
  if (const minijson::Value* kv = qc->find("kv_cache_scheme"); kv && !kv->is_null())
    reject("quantization_config.kv_cache_scheme",
           "a KV cache scheme is not implemented (the engine picks its own latent format)");
  for (const char* empty : {"sparsity_config", "transform_config"})
    if (const minijson::Value* t = qc->find(empty); t && !t->is_null())
      if (!t->is_object() || !t->members().empty())
        reject(std::string("quantization_config.") + empty,
               "must be empty (sparsity and weight transforms are not implemented)");

  const minijson::Value* groups = qc->find("config_groups");
  if (!groups || !groups->is_object() || groups->members().empty())
    reject("quantization_config.config_groups", "missing");
  if (groups->members().size() != 1)
    reject("quantization_config.config_groups", "exactly one group is implemented");
  const auto& gm = groups->members().front();
  const std::string gf = "quantization_config.config_groups." + gm.key;
  const minijson::Value& g = gm.value;
  if (!g.is_object()) reject(gf, "group is not an object");
  if (const std::string f = optional_string(g, "format", gf + ".format", "nvfp4-pack-quantized");
      f != "nvfp4-pack-quantized")
    reject(gf + ".format", "only nvfp4-pack-quantized is implemented, got '" + f + "'");
  const minijson::Value* targets = g.find("targets");
  if (!targets || !targets->is_array() || targets->items().size() != 1 ||
      !targets->items()[0].is_string() || targets->items()[0].as_string() != "Linear")
    reject(gf + ".targets", "only the single class target [\"Linear\"] is implemented");
  check_fp4_args(require(g, "weights", gf + ".weights"), gf + ".weights", false, c);
  if (const minijson::Value* oa = g.find("output_activations"); oa && !oa->is_null())
    reject(gf + ".output_activations", "output-activation quantization is not implemented");
  c.activation_scales_present = false;
  if (const minijson::Value* ia = g.find("input_activations"); ia && !ia->is_null()) {
    check_fp4_args(*ia, gf + ".input_activations", true, c);
    c.activation_scales_present = true;
  }

  // The ignore list, applied as compressed-tensors applies it to the module
  // names of the runtime that wrote it (vLLM's names; the checkpoint keeps
  // Mistral's). The loader implements one outcome: every attention
  // projection, the router and the head BF16; every expert matrix NVFP4.
  std::vector<IgnoreRule> ignore;
  if (const minijson::Value* ig = qc->find("ignore"); ig && !ig->is_null()) {
    if (!ig->is_array()) reject("quantization_config.ignore", "not an array");
    for (const auto& item : ig->items()) {
      if (!item.is_string()) reject("quantization_config.ignore", "non-string entry");
      ignore.push_back(make_rule(std::string(item.as_string())));
    }
  }
  const auto ignored = [&](const std::string& module) {
    for (const IgnoreRule& r : ignore)
      if (r.matches(module)) return true;
    return false;
  };
  const auto want_bf16 = [&](const std::string& module) {
    if (!ignored(module))
      reject(
          "quantization_config.ignore",
          "module '" + module +
              "' is not ignored — the engine expects BF16 there (attention, the router, the head)");
  };
  const auto want_fp4 = [&](const std::string& module) {
    if (ignored(module))
      reject("quantization_config.ignore",
             "module '" + module +
                 "' is ignored — the engine expects NVFP4 there (the routed and shared experts)");
  };
  want_bf16("lm_head");
  // Every layer's attention, router and shared expert; of the routed experts
  // the first two and the last (a rule naming one expert in between is
  // caught by the binding table, which checks every tensor).
  std::set<int> probe = {0, 1, c.n_routed_experts - 1};
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    for (const char* m : {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"})
      want_bf16(p + "self_attn." + m);
    want_bf16(p + "mlp.gate");
    for (const char* m : {"gate_proj", "up_proj", "down_proj"}) {
      want_fp4(p + "mlp.shared_experts." + m);
      for (int e : probe)
        if (e >= 0 && e < c.n_routed_experts)
          want_fp4(p + "mlp.experts." + std::to_string(e) + "." + m);
    }
  }
}

void parse_vision(const minijson::Value& root, Mistral4TextConfig& c) {
  if (const minijson::Value* mm = root.find("multimodal"); mm && !mm->is_null())
    reject("multimodal",
           "the multimodal block (audio / the older vision layout) is not implemented");
  const minijson::Value* v = root.find("vision_encoder");
  if (!v || v->is_null()) return;
  if (!v->is_object()) reject("vision_encoder", "not an object");
  Mistral4VisionConfig& vc = c.vision;
  vc.present = true;
  vc.hidden_size = require_int(*v, "hidden_size", "vision_encoder.hidden_size");
  vc.num_hidden_layers = require_int(*v, "num_hidden_layers", "vision_encoder.num_hidden_layers");
  vc.intermediate_size = require_int(*v, "intermediate_size", "vision_encoder.intermediate_size");
  vc.num_channels = require_int(*v, "num_channels", "vision_encoder.num_channels");
  vc.patch_size = require_int(*v, "patch_size", "vision_encoder.patch_size");
  vc.spatial_merge_size =
      require_int(*v, "spatial_merge_size", "vision_encoder.spatial_merge_size");
  vc.pre_mm_projector_norm = optional_bool(*v, "add_pre_mm_projector_layer_norm",
                                           "vision_encoder.add_pre_mm_projector_layer_norm", false);
  if (vc.hidden_size <= 0 || vc.num_hidden_layers <= 0 || vc.intermediate_size <= 0 ||
      vc.num_channels <= 0 || vc.patch_size <= 0 || vc.spatial_merge_size <= 0)
    reject("vision_encoder", "sizes must be positive");
  // The tensors the binding table expects follow from these two.
  if (const std::string id =
          optional_string(*v, "mm_projector_id", "vision_encoder.mm_projector_id", "");
      id != "patch_merge")
    reject("vision_encoder.mm_projector_id",
           "only patch_merge is implemented (the binding table), got '" + id + "'");
  if (optional_bool(*v, "adapter_bias", "vision_encoder.adapter_bias", false))
    reject("vision_encoder.adapter_bias",
           "a biased vision-language adapter is not implemented (the binding table)");
}

}  // namespace

Mistral4TextConfig Mistral4TextConfig::parse(const minijson::Value& root) {
  if (!root.is_object()) reject("", "root is not an object");
  Mistral4TextConfig c;
  if (const std::string mt = optional_string(root, "model_type", "model_type", "transformer");
      mt != "transformer")
    reject("model_type", "expected transformer, got " + mt);
  if (const std::string act = optional_string(root, "activation", "activation", "silu");
      act != "silu")
    reject("activation", "only silu is implemented, got " + act);
  if (const minijson::Value* sw = root.find("sliding_window"); sw && !sw->is_null())
    reject("sliding_window", "sliding-window attention is not implemented");

  c.hidden_size = require_int(root, "dim", "dim");
  c.vocab_size = require_int(root, "vocab_size", "vocab_size");
  c.num_hidden_layers = require_int(root, "n_layers", "n_layers");
  c.rms_norm_eps = static_cast<float>(require_double(root, "norm_eps", "norm_eps"));
  c.tie_word_embeddings = optional_bool(root, "tied_embeddings", "tied_embeddings", false);
  c.max_position_embeddings =
      optional_int(root, "max_position_embeddings", "max_position_embeddings",
                   optional_int(root, "max_seq_len", "max_seq_len", 0));
  c.intermediate_size = require_int(root, "hidden_dim", "hidden_dim");
  if (c.hidden_size <= 0 || c.hidden_size % 16 != 0)
    reject("dim", "must be a positive multiple of 16 (the NVFP4 block)");
  if (c.vocab_size <= 0) reject("vocab_size", "must be positive");
  if (c.num_hidden_layers <= 0) reject("n_layers", "must be positive");
  if (!(c.rms_norm_eps > 0)) reject("norm_eps", "must be positive");
  if (c.tie_word_embeddings) reject("tied_embeddings", "tied embeddings are not implemented");
  if (c.max_position_embeddings <= 0) reject("max_position_embeddings", "missing or not positive");
  if (c.intermediate_size <= 0) reject("hidden_dim", "must be positive");
  for (const int64_t id : {c.bos_token_id, c.eos_token_id, c.pad_token_id})
    if (id >= c.vocab_size)
      reject("vocab_size", "smaller than the tokenizer's control ids (<s> 1, </s> 2, <pad> 11)");

  // --- MLA attention -----------------------------------------------------------
  c.num_attention_heads = require_int(root, "n_heads", "n_heads");
  c.q_lora_rank = require_int(root, "q_lora_rank", "q_lora_rank");
  c.kv_lora_rank = require_int(root, "kv_lora_rank", "kv_lora_rank");
  c.qk_nope_head_dim = require_int(root, "qk_nope_head_dim", "qk_nope_head_dim");
  c.qk_rope_head_dim = require_int(root, "qk_rope_head_dim", "qk_rope_head_dim");
  c.v_head_dim = require_int(root, "v_head_dim", "v_head_dim");
  if (c.num_attention_heads <= 0) reject("n_heads", "must be positive");
  // MLA has one latent for every head: there is no grouped K/V to share.
  if (const int kv = optional_int(root, "n_kv_heads", "n_kv_heads", c.num_attention_heads);
      kv != c.num_attention_heads)
    reject("n_kv_heads", "must equal n_heads (latent attention has no grouped K/V heads)");
  if (c.q_lora_rank <= 0)
    reject("q_lora_rank", "must be positive (the unfactored q projection is not implemented)");
  if (c.kv_lora_rank <= 0) reject("kv_lora_rank", "must be positive");
  if (c.qk_nope_head_dim <= 0) reject("qk_nope_head_dim", "must be positive");
  if (c.qk_rope_head_dim <= 0 || c.qk_rope_head_dim % 2 != 0)
    reject("qk_rope_head_dim", "must be a positive even number (rotary pairs)");
  if (c.v_head_dim <= 0) reject("v_head_dim", "must be positive");
  if (const int hd = optional_int(root, "head_dim", "head_dim", c.qk_head_dim());
      hd != c.qk_head_dim())
    reject("head_dim", "must equal qk_nope_head_dim + qk_rope_head_dim");

  // --- rope -----------------------------------------------------------------------
  c.rope_theta = optional_double(root, "rope_theta", "rope_theta", 10000.0);
  if (!(c.rope_theta > 1.0)) reject("rope_theta", "must exceed 1");
  {
    const minijson::Value& y = require(root, "yarn", "yarn");
    if (!y.is_object()) reject("yarn", "not an object");
    require_known_keys(
        y, "yarn", {"factor", "original_max_position_embeddings", "beta", "alpha", "apply_scale"});
    c.yarn_factor = require_double(y, "factor", "yarn.factor");
    c.yarn_original_max_position_embeddings =
        require_int(y, "original_max_position_embeddings", "yarn.original_max_position_embeddings");
    c.yarn_beta_fast = optional_double(y, "beta", "yarn.beta", 32.0);
    c.yarn_beta_slow = optional_double(y, "alpha", "yarn.alpha", 1.0);
    if (!(c.yarn_factor >= 1.0)) reject("yarn.factor", "must be >= 1");
    if (c.yarn_original_max_position_embeddings <= 0)
      reject("yarn.original_max_position_embeddings", "must be positive");
    if (!(c.yarn_beta_fast > c.yarn_beta_slow) || !(c.yarn_beta_slow > 0))
      reject("yarn.beta", "beta (fast) must exceed alpha (slow) > 0");
    // The magnitude correction. `true` is YaRN's 0.1 ln(factor) + 1 on the
    // rotated lanes in vLLM and something else again in transformers; the
    // release ships false and nothing else has been checked.
    if (optional_bool(y, "apply_scale", "yarn.apply_scale", true))
      reject("yarn.apply_scale",
             "only false is implemented (no YaRN magnitude correction; see "
             "models/mistral4/config.hpp)");
  }
  {
    const minijson::Value& s = require(root, "llama_4_scaling", "llama_4_scaling");
    if (!s.is_object()) reject("llama_4_scaling", "not an object");
    require_known_keys(s, "llama_4_scaling", {"beta", "original_max_position_embeddings"});
    c.llama4_scaling_beta = require_double(s, "beta", "llama_4_scaling.beta");
    c.llama4_original_max_position_embeddings = require_int(
        s, "original_max_position_embeddings", "llama_4_scaling.original_max_position_embeddings");
    if (!(c.llama4_scaling_beta >= 0)) reject("llama_4_scaling.beta", "must be >= 0");
    if (c.llama4_original_max_position_embeddings <= 0)
      reject("llama_4_scaling.original_max_position_embeddings", "must be positive");
  }

  // --- MoE --------------------------------------------------------------------------
  {
    const minijson::Value& m = require(root, "moe", "moe");
    if (!m.is_object()) reject("moe", "not an object");
    require_known_keys(
        m, "moe",
        {"expert_hidden_dim", "expert_model_parallel", "expert_parallel", "first_k_dense_replace",
         "num_expert_groups", "num_expert_groups_per_tok", "num_experts", "num_experts_per_tok",
         "num_shared_experts", "route_every_n", "routed_scale"});
    c.moe_intermediate_size = require_int(m, "expert_hidden_dim", "moe.expert_hidden_dim");
    c.n_routed_experts = require_int(m, "num_experts", "moe.num_experts");
    c.num_experts_per_tok = require_int(m, "num_experts_per_tok", "moe.num_experts_per_tok");
    c.n_shared_experts = optional_int(m, "num_shared_experts", "moe.num_shared_experts", 0);
    c.routed_scaling_factor =
        static_cast<float>(optional_double(m, "routed_scale", "moe.routed_scale", 1.0));
    if (c.moe_intermediate_size <= 0 || c.moe_intermediate_size % 16 != 0)
      reject("moe.expert_hidden_dim", "must be a positive multiple of 16 (the NVFP4 block)");
    if (c.n_routed_experts <= 0 || c.n_routed_experts > 4096)
      reject("moe.num_experts", "must be in [1, 4096]");
    if (c.num_experts_per_tok <= 0 || c.num_experts_per_tok > 16 ||
        c.num_experts_per_tok > c.n_routed_experts)
      reject("moe.num_experts_per_tok", "must be in [1, min(num_experts, 16)]");
    if (c.n_shared_experts != 1)
      reject("moe.num_shared_experts", "exactly one shared expert is implemented");
    if (!(c.routed_scaling_factor > 0)) reject("moe.routed_scale", "must be positive");
    if (optional_int(m, "first_k_dense_replace", "moe.first_k_dense_replace", 0) != 0)
      reject("moe.first_k_dense_replace",
             "dense layers are not implemented (every layer carries the MoE)");
    if (optional_int(m, "route_every_n", "moe.route_every_n", 1) != 1)
      reject("moe.route_every_n", "only 1 is implemented (every layer carries the MoE)");
    if (optional_int(m, "num_expert_groups", "moe.num_expert_groups", 1) != 1 ||
        optional_int(m, "num_expert_groups_per_tok", "moe.num_expert_groups_per_tok", 1) != 1)
      reject("moe.num_expert_groups", "group-limited routing is not implemented");
    // Mistral's own serving-time partition hints: 1 = unpartitioned.
    if (optional_int(m, "expert_parallel", "moe.expert_parallel", 1) != 1 ||
        optional_int(m, "expert_model_parallel", "moe.expert_model_parallel", 1) != 1)
      reject("moe.expert_parallel", "only 1 is implemented");
  }

  parse_vision(root, c);
  parse_quantization(root, c);
  return c;
}

Mistral4TextConfig Mistral4TextConfig::from_json_file(const std::string& path) {
  namespace fs = std::filesystem;
  const fs::path p = fs::is_directory(path) ? fs::path(path) / "params.json" : fs::path(path);
  const std::string json = read_file(p.string());
  const auto parsed = minijson::parse(json);
  return parse(parsed.root);
}

void Mistral4TextConfig::require_kernel_geometry() const {
  auto fail = [](const std::string& what) {
    throw std::invalid_argument("mistral4 kernel geometry: " + what);
  };
  // The MLA kernels (kernels/dsa.hpp): a 64-wide bf16 rope tail beside the
  // latent, the fused [q_a | kv_a] projection's alignment, 8-wide loads.
  if (qk_rope_head_dim != 64)
    fail("qk_rope_head_dim must be 64 (the rope tail the attention kernels tile)");
  if (q_lora_rank % 8 != 0)
    fail("q_lora_rank must be a multiple of 8 (the fused projection's alignment)");
  if (kv_lora_rank % 8 != 0)
    fail("kv_lora_rank must be a multiple of 8 (the rope tail's alignment)");
  // The NVFP4 GEMV core's K and the 16-block column slice.
  if (hidden_size % 32 != 0) fail("dim must be a multiple of 32 (the NVFP4 GEMV core's K)");
  if (moe_intermediate_size % 32 != 0) fail("moe.expert_hidden_dim must be a multiple of 32");
  if (fp4_group_size != 16) fail("the NVFP4 block must be 16");
}

float Mistral4TextConfig::attention_scale() const {
  return 1.0f / std::sqrt(static_cast<float>(qk_head_dim()));
}

float Mistral4TextConfig::llama4_query_scale(int64_t pos) const {
  // transformers' get_llama_4_attn_scale / vLLM's _get_llama_4_scaling:
  // 1 + beta * log(1 + floor(pos / original)), the division, the log and the
  // product in fp32.
  if (pos < 0) return 1.0f;
  const float steps = std::floor(static_cast<float>(pos) /
                                 static_cast<float>(llama4_original_max_position_embeddings));
  return 1.0f + static_cast<float>(llama4_scaling_beta) * std::log(1.0f + steps);
}

std::vector<float> Mistral4TextConfig::rope_inv_freq() const {
  std::vector<float> out(static_cast<size_t>(qk_rope_head_dim / 2));
  // The band is computed from the ORIGINAL context (vLLM hands
  // DeepseekScalingRotaryEmbedding original_max_position_embeddings as its
  // max position; transformers' _compute_yarn_parameters uses the same
  // field), and the lanes past it are divided by the factor.
  yarn_rope_inv_freq_host(qk_rope_head_dim, rope_theta, yarn_original_max_position_embeddings,
                          yarn_factor, yarn_beta_fast, yarn_beta_slow, out.data());
  return out;
}

GlmMoeConfig Mistral4TextConfig::moe_config(int local_inter) const {
  GlmMoeConfig m;
  m.hidden = hidden_size;
  m.inter = local_inter;
  m.n_experts = n_routed_experts;
  m.top_k = num_experts_per_tok;
  m.n_shared_experts = n_shared_experts;
  m.routed_scaling_factor = routed_scaling_factor;
  m.norm_topk_prob = true;  // the conversion pins it; params.json has no key
  m.swiglu_limit = std::numeric_limits<float>::infinity();  // Mistral4MLP: no clamps
  m.router_mode = MoeRouterMode::SoftmaxTopk;
  GlmMoeConfig::validate_config(m);
  return m;
}

}  // namespace dgpp
