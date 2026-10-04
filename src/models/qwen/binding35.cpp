#include "models/qwen/binding35.hpp"

#include <cstdlib>
#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<QwenExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         QwenWeightClass cls, int layer, int expert = -1,
         QwenTensorRole role = QwenTensorRole::Plain) {
  out.push_back(QwenExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer,
                                   expert, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              QwenWeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

// The e4m3 payload + its BF16 block-scale partner, as one call.
void add_fp8(TensorList& out, const std::string& name, int64_t rows, int64_t cols,
             QwenWeightClass cls, int layer) {
  add(out, name, DType::F8_E4M3, {rows, cols}, cls, layer, -1, QwenTensorRole::Fp8Payload);
  add(out, name + "_scale_inv", DType::BF16, qwen_scale_shape({rows, cols}), cls, layer, -1,
      QwenTensorRole::Fp8Scale);
}

// The modelopt NVFP4 set of one matrix (`base` is the name without
// ".weight"): the U8 e2m1 pairs [N, K/2], the e4m3 scales per 16, the F32
// per-tensor scale, and the recipe's activation scale (unused: W4A16).
void add_fp4(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
             QwenWeightClass cls, int layer, int expert = -1) {
  add(out, base + ".weight", DType::U8, {rows, cols / 2}, cls, layer, expert,
      QwenTensorRole::Fp4Payload);
  add(out, base + ".weight_scale", DType::F8_E4M3, {rows, cols / 16}, cls, layer, expert,
      QwenTensorRole::Fp4Scale);
  add(out, base + ".weight_scale_2", DType::F32, {}, cls, layer, expert,
      QwenTensorRole::Fp4Global);
  add(out, base + ".input_scale", DType::F32, {}, cls, layer, expert,
      QwenTensorRole::InputScale);
}

// A Qwen3Next matrix the release quantizes in the backbone and ships in
// BF16 in the draft layer (`mtp.*` is on the recipe's ignore list).
void add_fp4_or_bf16(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
                     QwenWeightClass cls, int layer, int expert, bool bf16) {
  if (bf16)
    add(out, base + ".weight", DType::BF16, {rows, cols}, cls, layer, expert);
  else
    add_fp4(out, base, rows, cols, cls, layer, expert);
}

// --- the Qwen3Next dialect (nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 @ 8fb2682f) ---
// The GDN projections are fused: in_proj_qkvz [2*kdim + 2*vdim, H] rows
// are interleaved per key head ([q 128 | k 128 | v r*128 | z r*128] with
// r = value heads per key head), in_proj_ba [2*vh, H] alike ([b x r | a x r]).
// The loader gathers them into the split, head-major rows the layer binds.
void expect_gdn_next(TensorList& out, const std::string& p, const Qwen35TextConfig& cfg,
                     int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t kdim = static_cast<int64_t>(cfg.gdn_key_heads) * cfg.gdn_key_head_dim;
  const int64_t vdim = static_cast<int64_t>(cfg.gdn_value_heads) * cfg.gdn_value_head_dim;
  const int64_t vh = cfg.gdn_value_heads;
  const QwenWeightClass c = QwenWeightClass::Gdn;
  add_bf16(out, p + "A_log", {vh}, c, layer);
  add_bf16(out, p + "dt_bias", {vh}, c, layer);
  add_bf16(out, p + "conv1d.weight", {2 * kdim + vdim, 1, cfg.gdn_conv_width}, c, layer);
  add_bf16(out, p + "in_proj_ba.weight", {2 * vh, H}, c, layer);
  add_bf16(out, p + "in_proj_qkvz.weight", {2 * kdim + 2 * vdim, H}, c, layer);
  add_bf16(out, p + "norm.weight", {cfg.gdn_value_head_dim}, c, layer);
  add_fp4(out, p + "out_proj", H, vdim, c, layer);
}

// q/k/v BF16 with the recipe's FP8 K/V-cache scales beside k and v (F32
// scalars; read only under engine.kv_dtype fp8), o_proj NVFP4. The draft
// layer: all four BF16, no cache scales.
void expect_full_next(TensorList& out, const std::string& p, const Qwen35TextConfig& cfg,
                      int layer, bool is_mtp) {
  const int64_t H = cfg.hidden_size;
  const int64_t qh = cfg.num_attention_heads, kvh = cfg.num_key_value_heads;
  const int64_t d = cfg.head_dim;
  const QwenWeightClass c = QwenWeightClass::FullAttn;
  add_bf16(out, p + "q_proj.weight", {2 * qh * d, H}, c, layer);
  add_bf16(out, p + "k_proj.weight", {kvh * d, H}, c, layer);
  add_bf16(out, p + "v_proj.weight", {kvh * d, H}, c, layer);
  if (!is_mtp) {
    add(out, p + "k_proj.k_scale", DType::F32, {}, c, layer);
    add(out, p + "v_proj.v_scale", DType::F32, {}, c, layer);
  }
  add_fp4_or_bf16(out, p + "o_proj", H, qh * d, c, layer, -1, is_mtp);
  add_bf16(out, p + "q_norm.weight", {d}, c, layer);
  add_bf16(out, p + "k_norm.weight", {d}, c, layer);
}

// The routed MoE: BF16 router and shared gate, the shared expert and every
// routed expert as gate/up [I, H] and down [H, I] — NVFP4 in the backbone,
// per-expert BF16 in the draft layer.
void expect_moe_next(TensorList& out, const std::string& p, const Qwen35TextConfig& cfg,
                     int layer, bool is_mtp) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.moe_intermediate_size;
  const int64_t S = cfg.shared_expert_intermediate_size;
  add_bf16(out, p + "gate.weight", {cfg.num_experts, H}, QwenWeightClass::Router, layer);
  add_bf16(out, p + "shared_expert_gate.weight", {1, H}, QwenWeightClass::Router, layer);
  const std::string sp = p + "shared_expert.";
  add_fp4_or_bf16(out, sp + "gate_proj", S, H, QwenWeightClass::SharedExpert, layer, -1, is_mtp);
  add_fp4_or_bf16(out, sp + "up_proj", S, H, QwenWeightClass::SharedExpert, layer, -1, is_mtp);
  add_fp4_or_bf16(out, sp + "down_proj", H, S, QwenWeightClass::SharedExpert, layer, -1, is_mtp);
  for (int e = 0; e < cfg.num_experts; ++e) {
    const std::string ep = p + "experts." + std::to_string(e) + ".";
    add_fp4_or_bf16(out, ep + "gate_proj", I, H, QwenWeightClass::RoutedExpert, layer, e, is_mtp);
    add_fp4_or_bf16(out, ep + "up_proj", I, H, QwenWeightClass::RoutedExpert, layer, e, is_mtp);
    add_fp4_or_bf16(out, ep + "down_proj", H, I, QwenWeightClass::RoutedExpert, layer, e, is_mtp);
  }
}

void expect_gdn35(TensorList& out, const std::string& p, const Qwen35TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t kdim = static_cast<int64_t>(cfg.gdn_key_heads) * cfg.gdn_key_head_dim;
  const int64_t vdim = static_cast<int64_t>(cfg.gdn_value_heads) * cfg.gdn_value_head_dim;
  const int64_t vh = cfg.gdn_value_heads;
  const QwenWeightClass c = QwenWeightClass::Gdn;
  add_bf16(out, p + "A_log", {vh}, c, layer);
  add_bf16(out, p + "dt_bias", {vh}, c, layer);
  add_bf16(out, p + "conv1d.weight", {2 * kdim + vdim, 1, cfg.gdn_conv_width}, c, layer);
  add_bf16(out, p + "in_proj_a.weight", {vh, H}, c, layer);
  add_bf16(out, p + "in_proj_b.weight", {vh, H}, c, layer);
  add_fp8(out, p + "in_proj_qkv.weight", 2 * kdim + vdim, H, c, layer);
  add_fp8(out, p + "in_proj_z.weight", vdim, H, c, layer);
  add_bf16(out, p + "norm.weight", {cfg.gdn_value_head_dim}, c, layer);
  add_fp8(out, p + "out_proj.weight", H, vdim, c, layer);
}

void expect_full35(TensorList& out, const std::string& p, const Qwen35TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t qh = cfg.num_attention_heads, kvh = cfg.num_key_value_heads;
  const int64_t d = cfg.head_dim;
  const QwenWeightClass c = QwenWeightClass::FullAttn;
  // The q projection stacks [q | gate] per head (attn_output_gate).
  add_fp8(out, p + "q_proj.weight", 2 * qh * d, H, c, layer);
  add_fp8(out, p + "k_proj.weight", kvh * d, H, c, layer);
  add_fp8(out, p + "v_proj.weight", kvh * d, H, c, layer);
  add_fp8(out, p + "o_proj.weight", H, qh * d, c, layer);
  add_bf16(out, p + "q_norm.weight", {d}, c, layer);
  add_bf16(out, p + "k_norm.weight", {d}, c, layer);
}

void expect_dense_mlp35(TensorList& out, const std::string& p, const Qwen35TextConfig& cfg,
                        int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  const QwenWeightClass c = QwenWeightClass::DenseMlp;
  add_fp8(out, p + "gate_proj.weight", I, H, c, layer);
  add_fp8(out, p + "up_proj.weight", I, H, c, layer);
  add_fp8(out, p + "down_proj.weight", H, I, c, layer);
}

void expect_norm35(TensorList& out, const std::string& p, const Qwen35TextConfig& cfg, int layer) {
  add_bf16(out, p + "input_layernorm.weight", {cfg.hidden_size}, QwenWeightClass::Norm, layer);
  add_bf16(out, p + "post_attention_layernorm.weight", {cfg.hidden_size}, QwenWeightClass::Norm,
           layer);
}

int max_layer35(const Qwen35TextConfig& cfg) {
  return cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
}

}  // namespace

std::string qwen35_model_prefix(const Qwen35TextConfig& cfg) {
  return cfg.next() ? "model." : "model.language_model.";
}

std::string qwen35_layer_prefix(const Qwen35TextConfig& cfg, int layer) {
  if (layer == cfg.mtp_layer()) return "mtp.layers.0.";
  return qwen35_model_prefix(cfg) + "layers." + std::to_string(layer) + ".";
}

std::vector<QwenExpectedTensor> qwen35_expected_layer_tensors(const Qwen35TextConfig& cfg,
                                                             int layer) {
  if (layer < 0 || layer >= max_layer35(cfg))
    throw std::invalid_argument("qwen35_expected_layer_tensors: layer out of range");
  const bool is_mtp = layer == cfg.mtp_layer();
  const Qwen35LayerKind kind = is_mtp ? Qwen35LayerKind::Full : cfg.layers[layer];
  const std::string p = qwen35_layer_prefix(cfg, layer);
  TensorList out;
  // Both RMSNorms (input + post-attention) via one helper so the pair
  // cannot drift apart; the table is order-free (validation by name).
  expect_norm35(out, p, cfg, layer);
  if (cfg.next()) {
    if (kind == Qwen35LayerKind::Gdn)
      expect_gdn_next(out, p + "linear_attn.", cfg, layer);
    else
      expect_full_next(out, p + "self_attn.", cfg, layer, is_mtp);
    expect_moe_next(out, p + "mlp.", cfg, layer, is_mtp);
    return out;
  }
  if (kind == Qwen35LayerKind::Gdn)
    expect_gdn35(out, p + "linear_attn.", cfg, layer);
  else
    expect_full35(out, p + "self_attn.", cfg, layer);
  expect_dense_mlp35(out, p + "mlp.", cfg, layer);
  return out;
}

std::vector<QwenExpectedTensor> qwen35_expected_global_tensors(const Qwen35TextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, qwen35_model_prefix(cfg) + "embed_tokens.weight", {cfg.vocab_size, H},
           QwenWeightClass::Embed, -1);
  add_bf16(out, "lm_head.weight", {cfg.vocab_size, H}, QwenWeightClass::LmHead, -1);
  add_bf16(out, qwen35_model_prefix(cfg) + "norm.weight", {H}, QwenWeightClass::Norm, -1);
  if (cfg.mtp_layer() >= 0) {
    // The fused head projection (embedding + hidden pre-projection).
    add_bf16(out, "mtp.fc.weight", {H, 2 * H}, QwenWeightClass::Mtp, cfg.mtp_layer());
    add_bf16(out, "mtp.norm.weight", {H}, QwenWeightClass::Mtp, cfg.mtp_layer());
    add_bf16(out, "mtp.pre_fc_norm_embedding.weight", {H}, QwenWeightClass::Mtp,
             cfg.mtp_layer());
    add_bf16(out, "mtp.pre_fc_norm_hidden.weight", {H}, QwenWeightClass::Mtp, cfg.mtp_layer());
  }
  return out;
}

std::vector<QwenExpectedTensor> qwen35_expected_text_tensors(const Qwen35TextConfig& cfg) {
  TensorList out = qwen35_expected_global_tensors(cfg);
  for (int l = 0; l < max_layer35(cfg); ++l) {
    TensorList layer = qwen35_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()),
               std::make_move_iterator(layer.end()));
  }
  return out;
}

QwenBindReport qwen35_validate_text_binding(
    const Qwen35TextConfig& cfg,
    const std::unordered_map<std::string, QwenTensorDesc>& present, size_t max_errors) {
  QwenBindReport rep;
  const auto expected = qwen35_expected_text_tensors(cfg);
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
      push_error(std::format("'{}' dtype {} != expected {}", e.name,
                             dtype_name(it->second.dtype), dtype_name(e.dtype)));
      continue;
    }
    if (it->second.shape != e.shape) {
      ++rep.shape_mismatch;
      push_error(std::format("'{}' shape {} != expected {}", e.name,
                             shape_str(it->second.shape), shape_str(e.shape)));
      continue;
    }
    ++rep.matched;
    if (e.quantized()) ++rep.quantized_matrices;
  }
  for (const auto& [name, desc] : present) {
    (void)desc;
    if (consumed.count(name)) continue;
    if (name.rfind("model.visual.", 0) == 0) {
      ++rep.vision;
      continue;
    }
    // A truncated config (the check apps' --layers N): the layers past it
    // are out of scope, not unexpected.
    {
      const std::string kLayers = qwen35_model_prefix(cfg) + "layers.";
      if (name.rfind(kLayers, 0) == 0) {
        const size_t dot = name.find('.', kLayers.size());
        const int64_t idx = dot == std::string::npos ? -1 : std::atoll(name.substr(kLayers.size(), dot - kLayers.size()).c_str());
        if (idx >= cfg.num_hidden_layers) {
          ++rep.out_of_scope;
          continue;
        }
      }
    }
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

void qwen35_tp_validate_geometry(const Qwen35TextConfig& cfg, int rank, int world) {
  auto fail = [](const std::string& what) { throw std::invalid_argument("qwen3.5 tp geometry: " + what); };
  if (world < 1 || rank < 0 || rank >= world) fail("rank/world out of range");
  if (cfg.gdn_key_heads % world != 0) fail("linear_num_key_heads must divide by world");
  if (cfg.gdn_value_heads % world != 0) fail("linear_num_value_heads must divide by world");
  if (cfg.num_attention_heads % world != 0) fail("num_attention_heads must divide by world");
  if (cfg.num_key_value_heads % world != 0 && world % cfg.num_key_value_heads != 0)
    fail("num_key_value_heads must divide world or be divided by it");
  if (cfg.num_attention_heads / cfg.num_key_value_heads * cfg.num_key_value_heads !=
      cfg.num_attention_heads)
    fail("query heads per kv head");
  if (cfg.moe()) {
    // The expert slices keep the NVFP4 scale grid: I/world and S/world are
    // multiples of the 16-wide group.
    if (cfg.moe_intermediate_size % (16 * world) != 0)
      fail("moe_intermediate_size / world must be a multiple of 16");
    if (cfg.shared_expert_intermediate_size % (16 * world) != 0)
      fail("shared_expert_intermediate_size / world must be a multiple of 16");
    return;
  }
  if (cfg.intermediate_size % world != 0) fail("intermediate_size must divide by world");
}

}  // namespace dgpp
