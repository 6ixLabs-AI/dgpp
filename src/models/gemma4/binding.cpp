#include "models/gemma4/binding.hpp"

#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<Gemma4ExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         Gemma4WeightClass cls, int layer, Gemma4TensorRole role = Gemma4TensorRole::Plain, int expert = -1) {
  out.push_back(Gemma4ExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer, role, expert});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              Gemma4WeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

// The modelopt NVFP4 set of one [rows, cols] matrix (`base` is the name
// without ".weight"): the U8 e2m1 pairs, the e4m3 scales per 16, the F32
// per-tensor scale, and — where the recipe calibrated activations — the
// activation scale the engine does not use.
void add_fp4(TensorList& out, const Gemma4TextConfig& cfg, const std::string& base, int64_t rows,
             int64_t cols, Gemma4WeightClass cls, int layer, int expert = -1) {
  if (cols % 16 != 0)
    throw std::invalid_argument("gemma4 binding: NVFP4 K must be a multiple of 16 on " + base);
  add(out, base + ".weight", DType::U8, {rows, cols / 2}, cls, layer, Gemma4TensorRole::Fp4Payload, expert);
  add(out, base + ".weight_scale", DType::F8_E4M3, {rows, cols / 16}, cls, layer, Gemma4TensorRole::Fp4Scale, expert);
  add(out, base + ".weight_scale_2", DType::F32, {}, cls, layer, Gemma4TensorRole::Fp4Global, expert);
  if (cfg.activation_scales())
    add(out, base + ".input_scale", DType::F32, {}, cls, layer, Gemma4TensorRole::InputScale, expert);
}

// One projection in the format the recipe gives its class.
void add_matrix(TensorList& out, const Gemma4TextConfig& cfg, bool nvfp4, const std::string& base,
                int64_t rows, int64_t cols, Gemma4WeightClass cls, int layer) {
  if (nvfp4)
    add_fp4(out, cfg, base, rows, cols, cls, layer);
  else
    add_bf16(out, base + ".weight", {rows, cols}, cls, layer);
}

void expect_attention(TensorList& out, const std::string& p, const Gemma4TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t hd = cfg.head_dim_of(layer);
  const Gemma4WeightClass c = Gemma4WeightClass::Attention;
  const bool q4 = cfg.attention_nvfp4();
  add_bf16(out, p + "q_norm.weight", {hd}, c, layer);
  add_bf16(out, p + "k_norm.weight", {hd}, c, layer);
  add_matrix(out, cfg, q4, p + "q_proj", cfg.q_rows(layer), H, c, layer);
  add_matrix(out, cfg, q4, p + "k_proj", cfg.kv_rows(layer), H, c, layer);
  if (cfg.has_v_proj(layer)) add_matrix(out, cfg, q4, p + "v_proj", cfg.kv_rows(layer), H, c, layer);
  add_matrix(out, cfg, q4, p + "o_proj", H, cfg.q_rows(layer), c, layer);
}

void expect_dense_mlp(TensorList& out, const std::string& p, const Gemma4TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.intermediate_size;
  const Gemma4WeightClass c = Gemma4WeightClass::DenseMlp;
  add_fp4(out, cfg, p + "gate_proj", I, H, c, layer);
  add_fp4(out, cfg, p + "up_proj", I, H, c, layer);
  add_fp4(out, cfg, p + "down_proj", H, I, c, layer);
}

// The MoE block: three more norms, the BF16 router, and every expert's three
// NVFP4 projections as separate Linears.
void expect_moe(TensorList& out, const std::string& p, const Gemma4TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size, E = cfg.num_experts;
  for (const char* norm : {"post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2"})
    add_bf16(out, p + norm + ".weight", {H}, Gemma4WeightClass::LayerNorm, layer);
  add_bf16(out, p + "router.proj.weight", {E, H}, Gemma4WeightClass::Router, layer);
  add_bf16(out, p + "router.scale", {H}, Gemma4WeightClass::Router, layer);
  add_bf16(out, p + "router.per_expert_scale", {E}, Gemma4WeightClass::Router, layer);
  for (int e = 0; e < cfg.num_experts; ++e) {
    const std::string ep = p + "moe.experts." + std::to_string(e) + ".";
    add_fp4(out, cfg, ep + "gate_proj", I, H, Gemma4WeightClass::RoutedExpert, layer, e);
    add_fp4(out, cfg, ep + "up_proj", I, H, Gemma4WeightClass::RoutedExpert, layer, e);
    add_fp4(out, cfg, ep + "down_proj", H, I, Gemma4WeightClass::RoutedExpert, layer, e);
  }
}

bool starts_with(const std::string& s, const char* prefix) {
  return s.rfind(prefix, 0) == 0;
}

}  // namespace

std::string gemma4_layer_prefix(int layer) {
  return "model.language_model.layers." + std::to_string(layer) + ".";
}

bool gemma4_ignored_tensor(const std::string& name) {
  return starts_with(name, "model.vision_tower.") || starts_with(name, "model.embed_vision.") ||
         starts_with(name, "model.audio_tower.") || starts_with(name, "model.embed_audio.");
}

std::vector<Gemma4ExpectedTensor> gemma4_expected_layer_tensors(const Gemma4TextConfig& cfg, int layer) {
  if (layer < 0 || layer >= cfg.num_hidden_layers)
    throw std::invalid_argument("gemma4_expected_layer_tensors: layer out of range");
  const std::string p = gemma4_layer_prefix(layer);
  const int64_t H = cfg.hidden_size;
  TensorList out;
  for (const char* norm : {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                           "post_feedforward_layernorm"})
    add_bf16(out, p + norm + ".weight", {H}, Gemma4WeightClass::LayerNorm, layer);
  add_bf16(out, p + "layer_scalar", {1}, Gemma4WeightClass::LayerScalar, layer);
  expect_attention(out, p + "self_attn.", cfg, layer);
  expect_dense_mlp(out, p + "mlp.", cfg, layer);
  if (cfg.enable_moe_block) expect_moe(out, p, cfg, layer);
  return out;
}

std::vector<Gemma4ExpectedTensor> gemma4_expected_global_tensors(const Gemma4TextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "model.language_model.embed_tokens.weight", {cfg.vocab_size, H}, Gemma4WeightClass::Embed, -1);
  add_bf16(out, "model.language_model.norm.weight", {H}, Gemma4WeightClass::FinalNorm, -1);
  return out;
}

std::vector<Gemma4ExpectedTensor> gemma4_expected_text_tensors(const Gemma4TextConfig& cfg) {
  TensorList out = gemma4_expected_global_tensors(cfg);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    TensorList layer = gemma4_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()), std::make_move_iterator(layer.end()));
  }
  return out;
}

Gemma4BindReport gemma4_validate_text_binding(
    const Gemma4TextConfig& cfg, const std::unordered_map<std::string, Gemma4TensorDesc>& present,
    size_t max_errors) {
  Gemma4BindReport rep;
  const auto expected = gemma4_expected_text_tensors(cfg);
  rep.expected = expected.size();
  auto push_error = [&](std::string msg) {
    if (rep.errors.size() < max_errors) rep.errors.push_back(std::move(msg));
  };
  auto shape_str = [](const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(s[i]);
    }
    return out + "]";
  };
  std::unordered_map<std::string, int8_t> consumed;
  consumed.reserve(expected.size());
  for (const auto& e : expected) {
    auto it = present.find(e.name);
    if (it == present.end()) {
      ++rep.missing;
      push_error(std::format("missing tensor '{}'", e.name));
      continue;
    }
    consumed.emplace(e.name, 1);
    if (it->second.dtype != e.dtype) {
      ++rep.dtype_mismatch;
      push_error(std::format("'{}' dtype {} != expected {}", e.name, dtype_name(it->second.dtype),
                             dtype_name(e.dtype)));
      continue;
    }
    if (it->second.shape != e.shape) {
      ++rep.shape_mismatch;
      push_error(std::format("'{}' shape {} != expected {}", e.name, shape_str(it->second.shape),
                             shape_str(e.shape)));
      continue;
    }
    ++rep.matched;
    rep.bytes += e.nbytes();
    if (e.role == Gemma4TensorRole::Fp4Payload) ++rep.fp4_matrices;
    if (e.role == Gemma4TensorRole::Plain && e.dtype == DType::BF16 && e.shape.size() == 2) ++rep.bf16_matrices;
    if (e.role == Gemma4TensorRole::Fp4Payload && e.cls == Gemma4WeightClass::RoutedExpert) ++rep.expert_matrices;
  }
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    if (gemma4_ignored_tensor(name)) {
      ++rep.ignored;
      continue;
    }
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

}  // namespace dgpp
