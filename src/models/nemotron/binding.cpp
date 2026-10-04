#include "models/nemotron/binding.hpp"

#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<NemotronExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         NemotronWeightClass cls, int layer, int expert = -1,
         NemotronTensorRole role = NemotronTensorRole::Plain) {
  out.push_back(
      NemotronExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer, expert, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              NemotronWeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

// One nn.Linear [rows, cols] (`base` is the module name, no ".weight") in
// the form the recipe gives it.
void add_linear(TensorList& out, const NemotronHConfig& cfg, const std::string& base, int64_t rows,
                int64_t cols, NemotronWeightClass cls, int layer, int expert = -1) {
  switch (cfg.quant_of(base)) {
    case NemotronQuant::Bf16:
      add(out, base + ".weight", DType::BF16, {rows, cols}, cls, layer, expert);
      return;
    case NemotronQuant::Fp8:
      add(out, base + ".weight", DType::F8_E4M3, {rows, cols}, cls, layer, expert,
          NemotronTensorRole::Fp8Payload);
      add(out, base + ".weight_scale", DType::F32, {}, cls, layer, expert,
          NemotronTensorRole::Fp8Scale);
      add(out, base + ".input_scale", DType::F32, {}, cls, layer, expert,
          NemotronTensorRole::InputScale);
      return;
    case NemotronQuant::Nvfp4:
      if (cols % 16 != 0)
        throw std::invalid_argument("nemotron binding: NVFP4 K must be a multiple of 16 on " +
                                    base);
      add(out, base + ".weight", DType::U8, {rows, cols / 2}, cls, layer, expert,
          NemotronTensorRole::Fp4Payload);
      add(out, base + ".weight_scale", DType::F8_E4M3, {rows, cols / 16}, cls, layer, expert,
          NemotronTensorRole::Fp4Scale);
      add(out, base + ".weight_scale_2", DType::F32, {}, cls, layer, expert,
          NemotronTensorRole::Fp4Global);
      add(out, base + ".input_scale", DType::F32, {}, cls, layer, expert,
          NemotronTensorRole::InputScale);
      return;
  }
}

void expect_mamba(TensorList& out, const std::string& p, const NemotronHConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, heads = cfg.mamba_num_heads;
  const NemotronWeightClass c = NemotronWeightClass::Mamba;
  add_bf16(out, p + "A_log", {heads}, c, layer);
  add_bf16(out, p + "D", {heads}, c, layer);
  add_bf16(out, p + "dt_bias", {heads}, c, layer);
  add_bf16(out, p + "conv1d.weight", {cfg.mamba_conv_dim(), 1, cfg.conv_kernel}, c, layer);
  if (cfg.conv_bias) add_bf16(out, p + "conv1d.bias", {cfg.mamba_conv_dim()}, c, layer);
  add_bf16(out, p + "norm.weight", {cfg.mamba_inner()}, c, layer);
  add_linear(out, cfg, p + "in_proj", cfg.mamba_proj_rows(), H, c, layer);
  add_linear(out, cfg, p + "out_proj", H, cfg.mamba_inner(), c, layer);
}

// The draft block's attention carries no K/V-cache scales (the recipe
// calibrated the backbone only).
void expect_attention(TensorList& out, const std::string& p, const NemotronHConfig& cfg,
                      int layer) {
  const int64_t H = cfg.hidden_size, d = cfg.head_dim;
  const int64_t qh = cfg.num_attention_heads, kvh = cfg.num_key_value_heads;
  const NemotronWeightClass c = NemotronWeightClass::Attention;
  add_linear(out, cfg, p + "q_proj", qh * d, H, c, layer);
  add_linear(out, cfg, p + "k_proj", kvh * d, H, c, layer);
  add_linear(out, cfg, p + "v_proj", kvh * d, H, c, layer);
  add_linear(out, cfg, p + "o_proj", H, qh * d, c, layer);
  if (cfg.kv_cache_scales && !cfg.is_mtp_layer(layer)) {
    add(out, p + "k_proj.k_scale", DType::F32, {}, c, layer, -1, NemotronTensorRole::KvScale);
    add(out, p + "v_proj.v_scale", DType::F32, {}, c, layer, -1, NemotronTensorRole::KvScale);
  }
}

void expect_moe(TensorList& out, const std::string& p, const NemotronHConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size;
  const int64_t S = cfg.moe_shared_expert_intermediate_size, X = cfg.expert_width();
  add(out, p + "gate.weight", cfg.router_dtype, {cfg.n_routed_experts, H},
      NemotronWeightClass::Router, layer);
  add(out, p + "gate.e_score_correction_bias", DType::F32, {cfg.n_routed_experts},
      NemotronWeightClass::Router, layer);
  for (int e = 0; e < cfg.n_routed_experts; ++e) {
    const std::string ep = p + "experts." + std::to_string(e) + ".";
    add_linear(out, cfg, ep + "up_proj", I, X, NemotronWeightClass::RoutedExpert, layer, e);
    add_linear(out, cfg, ep + "down_proj", X, I, NemotronWeightClass::RoutedExpert, layer, e);
  }
  add_linear(out, cfg, p + "shared_experts.up_proj", S, H, NemotronWeightClass::SharedExpert,
             layer);
  add_linear(out, cfg, p + "shared_experts.down_proj", H, S, NemotronWeightClass::SharedExpert,
             layer);
  if (cfg.moe_latent_size > 0) {
    add_linear(out, cfg, p + "fc1_latent_proj", X, H, NemotronWeightClass::LatentProj, layer);
    add_linear(out, cfg, p + "fc2_latent_proj", H, X, NemotronWeightClass::LatentProj, layer);
  }
}

}  // namespace

std::vector<NemotronExpectedTensor> nemotron_expected_layer_tensors(const NemotronHConfig& cfg,
                                                                    int layer) {
  if (layer < 0 || layer >= cfg.num_layers_total())
    throw std::invalid_argument("nemotron_expected_layer_tensors: layer out of range");
  const std::string p = nemotron_layer_prefix(cfg, layer);
  const int64_t H = cfg.hidden_size;
  TensorList out;
  // The fusion sits on the draft block's first layer, the closing norm on
  // its last (checkpoint truth: mtp.layers.0.{enorm,hnorm,eh_proj},
  // mtp.layers.1.final_layernorm).
  if (layer == cfg.num_hidden_layers && cfg.is_mtp_layer(layer)) {
    add_bf16(out, p + "enorm.weight", {H}, NemotronWeightClass::MtpHead, layer);
    add_bf16(out, p + "hnorm.weight", {H}, NemotronWeightClass::MtpHead, layer);
    add_bf16(out, p + "eh_proj.weight", {H, 2 * H}, NemotronWeightClass::MtpHead, layer);
  }
  add_bf16(out, p + "norm.weight", {H}, NemotronWeightClass::Norm, layer);
  switch (cfg.kind_of(layer)) {
    case NemotronLayerKind::Mamba:
      expect_mamba(out, p + "mixer.", cfg, layer);
      break;
    case NemotronLayerKind::Attention:
      expect_attention(out, p + "mixer.", cfg, layer);
      break;
    case NemotronLayerKind::Moe:
      expect_moe(out, p + "mixer.", cfg, layer);
      break;
  }
  if (cfg.is_mtp_layer(layer) && layer == cfg.num_layers_total() - 1)
    add_bf16(out, p + "final_layernorm.weight", {H}, NemotronWeightClass::MtpHead, layer);
  return out;
}

std::vector<NemotronExpectedTensor> nemotron_expected_global_tensors(const NemotronHConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "backbone.embeddings.weight", {cfg.vocab_size, H}, NemotronWeightClass::Embed, -1);
  add_bf16(out, "backbone.norm_f.weight", {H}, NemotronWeightClass::Norm, -1);
  add_bf16(out, "lm_head.weight", {cfg.vocab_size, H}, NemotronWeightClass::LmHead, -1);
  return out;
}

std::vector<NemotronExpectedTensor> nemotron_expected_tensors(const NemotronHConfig& cfg) {
  TensorList out = nemotron_expected_global_tensors(cfg);
  for (int l = 0; l < cfg.num_layers_total(); ++l) {
    TensorList layer = nemotron_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()),
               std::make_move_iterator(layer.end()));
  }
  return out;
}

NemotronBindReport nemotron_validate_binding(
    const NemotronHConfig& cfg, const std::unordered_map<std::string, NemotronTensorDesc>& present,
    size_t max_errors) {
  NemotronBindReport rep;
  const auto expected = nemotron_expected_tensors(cfg);
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
  consumed.reserve(present.size());
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
    if (e.role == NemotronTensorRole::Fp8Payload) ++rep.fp8_matrices;
    if (e.role == NemotronTensorRole::Fp4Payload) ++rep.fp4_matrices;
  }
  // A draft block the config does not declare, or anything else outside the
  // table, is unexpected: nothing in these releases is skipped.
  for (const auto& [name, desc] : present) {
    (void)desc;
    if (consumed.count(name)) continue;
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

}  // namespace dgpp
