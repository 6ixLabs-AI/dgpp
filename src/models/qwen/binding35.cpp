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

// A routed expert's block-FP8 pair (the payload and its BF16 scale partner,
// both tagged with the expert).
void add_fp8_expert(TensorList& out, const std::string& name, int64_t rows, int64_t cols, int layer,
                    int expert) {
  add(out, name, DType::F8_E4M3, {rows, cols}, QwenWeightClass::RoutedExpert, layer, expert,
      QwenTensorRole::Fp8Payload);
  add(out, name + "_scale_inv", DType::BF16, qwen_scale_shape({rows, cols}),
      QwenWeightClass::RoutedExpert, layer, expert, QwenTensorRole::Fp8Scale);
}

// modelopt's per-tensor FP8 set of one matrix (`base` without ".weight",
// nvidia/Qwen3.6-35B-A3B-NVFP4): the e4m3 payload [N, K], one F32 scale for
// the whole tensor (the value is code x scale) and the recipe's activation
// scale (unused: the activations stay BF16).
void add_fp8_tensor(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
                    QwenWeightClass cls, int layer) {
  add(out, base + ".weight", DType::F8_E4M3, {rows, cols}, cls, layer, -1, QwenTensorRole::Fp8Payload);
  add(out, base + ".weight_scale", DType::F32, {}, cls, layer, -1, QwenTensorRole::Fp8TensorScale);
  add(out, base + ".input_scale", DType::F32, {}, cls, layer, -1, QwenTensorRole::InputScale);
}

// One dense projection of the Qwen3.5 dialect in the form the config's
// recipe ships it: the FP8 release's block form, the NVFP4 mixed release's
// per-tensor FP8 in the backbone and BF16 in the draft layer (`mtp*` is on
// that recipe's ignore list), BF16 in the unquantized release and in the
// experts-only NVFP4 release. Under the compressed-tensors container only
// the GDN's projections come through here (BF16: the recipe ignores them);
// its attention takes the NVFP4 set (expect_full_next).
void add_dense35(TensorList& out, const Qwen35TextConfig& cfg, const std::string& base,
                 int64_t rows, int64_t cols, QwenWeightClass cls, int layer, bool is_mtp) {
  if (cfg.quant_kind == Qwen35QuantKind::Bf16 || cfg.quant_kind == Qwen35QuantKind::Nvfp4Experts ||
      cfg.quant_kind == Qwen35QuantKind::Nvfp4Packed) {
    add_bf16(out, base + ".weight", {rows, cols}, cls, layer);
    return;
  }
  if (cfg.quant_kind == Qwen35QuantKind::Nvfp4Mixed) {
    if (is_mtp)
      add_bf16(out, base + ".weight", {rows, cols}, cls, layer);
    else
      add_fp8_tensor(out, base, rows, cols, cls, layer);
    return;
  }
  add_fp8(out, base + ".weight", rows, cols, cls, layer);
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

// The compressed-tensors NVFP4 set of one matrix (`nvfp4-pack-quantized`,
// RedHatAI/Qwen3-Coder-Next-NVFP4): the same code and block-scale geometry
// under other names — `weight_packed` U8 [N, K/2], `weight_scale` e4m3
// [N, K/16] — with an F32 [1] `weight_global_scale` that DIVIDES (modelopt's
// weight_scale_2 multiplies) and the recipe's `input_global_scale` (unused:
// W4A16).
void add_fp4_packed(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
                    QwenWeightClass cls, int layer, int expert = -1) {
  add(out, base + ".weight_packed", DType::U8, {rows, cols / 2}, cls, layer, expert,
      QwenTensorRole::Fp4Payload);
  add(out, base + ".weight_scale", DType::F8_E4M3, {rows, cols / 16}, cls, layer, expert,
      QwenTensorRole::Fp4Scale);
  add(out, base + ".weight_global_scale", DType::F32, {1}, cls, layer, expert,
      QwenTensorRole::Fp4Global);
  add(out, base + ".input_global_scale", DType::F32, {1}, cls, layer, expert,
      QwenTensorRole::InputScale);
}

// One NVFP4 matrix in the container the config's recipe names.
void add_fp4_set(TensorList& out, const Qwen35TextConfig& cfg, const std::string& base,
                 int64_t rows, int64_t cols, QwenWeightClass cls, int layer, int expert = -1) {
  if (cfg.quant_kind == Qwen35QuantKind::Nvfp4Packed)
    add_fp4_packed(out, base, rows, cols, cls, layer, expert);
  else
    add_fp4(out, base, rows, cols, cls, layer, expert);
}

// A Qwen3Next matrix the release quantizes in the backbone and ships in
// BF16 in the draft layer (`mtp.*` is on the recipe's ignore list).
void add_fp4_or_bf16(TensorList& out, const Qwen35TextConfig& cfg, const std::string& base,
                     int64_t rows, int64_t cols, QwenWeightClass cls, int layer, int expert,
                     bool bf16) {
  if (bf16)
    add(out, base + ".weight", DType::BF16, {rows, cols}, cls, layer, expert);
  else
    add_fp4_set(out, cfg, base, rows, cols, cls, layer, expert);
}

// --- the Qwen3Next dialect (nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 @ 8fb2682f and
// RedHatAI/Qwen3-Coder-Next-NVFP4 @ 27a8f16f) ---
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
  // The compressed-tensors recipe ignores the whole GDN (`re:.*linear_attn.*`):
  // its out_proj ships BF16 where the modelopt release quantizes it.
  add_fp4_or_bf16(out, cfg, p + "out_proj", H, vdim, c, layer, -1,
                  cfg.quant_kind == Qwen35QuantKind::Nvfp4Packed);
}

// The modelopt release: q/k/v BF16 with the recipe's FP8 K/V-cache scales
// beside k and v (F32 scalars; read only under engine.kv_dtype fp8), o_proj
// NVFP4. The compressed-tensors release: all four NVFP4, no cache scales.
// The draft layer: all four BF16, no cache scales.
void expect_full_next(TensorList& out, const std::string& p, const Qwen35TextConfig& cfg,
                      int layer, bool is_mtp) {
  const int64_t H = cfg.hidden_size;
  const int64_t qh = cfg.num_attention_heads, kvh = cfg.num_key_value_heads;
  const int64_t d = cfg.head_dim;
  const QwenWeightClass c = QwenWeightClass::FullAttn;
  const bool qkv_bf16 = is_mtp || cfg.quant_kind != Qwen35QuantKind::Nvfp4Packed;
  add_fp4_or_bf16(out, cfg, p + "q_proj", 2 * qh * d, H, c, layer, -1, qkv_bf16);
  add_fp4_or_bf16(out, cfg, p + "k_proj", kvh * d, H, c, layer, -1, qkv_bf16);
  add_fp4_or_bf16(out, cfg, p + "v_proj", kvh * d, H, c, layer, -1, qkv_bf16);
  if (!is_mtp && cfg.quant_kind == Qwen35QuantKind::Nvfp4Modelopt) {
    add(out, p + "k_proj.k_scale", DType::F32, {}, c, layer);
    add(out, p + "v_proj.v_scale", DType::F32, {}, c, layer);
  }
  add_fp4_or_bf16(out, cfg, p + "o_proj", H, qh * d, c, layer, -1, is_mtp);
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
  add_fp4_or_bf16(out, cfg, sp + "gate_proj", S, H, QwenWeightClass::SharedExpert, layer, -1, is_mtp);
  add_fp4_or_bf16(out, cfg, sp + "up_proj", S, H, QwenWeightClass::SharedExpert, layer, -1, is_mtp);
  add_fp4_or_bf16(out, cfg, sp + "down_proj", H, S, QwenWeightClass::SharedExpert, layer, -1, is_mtp);
  for (int e = 0; e < cfg.num_experts; ++e) {
    const std::string ep = p + "experts." + std::to_string(e) + ".";
    add_fp4_or_bf16(out, cfg, ep + "gate_proj", I, H, QwenWeightClass::RoutedExpert, layer, e,
                    is_mtp);
    add_fp4_or_bf16(out, cfg, ep + "up_proj", I, H, QwenWeightClass::RoutedExpert, layer, e, is_mtp);
    add_fp4_or_bf16(out, cfg, ep + "down_proj", H, I, QwenWeightClass::RoutedExpert, layer, e,
                    is_mtp);
  }
}

// The Qwen3.5 dialect ships the GDN projections split and head-major:
// in_proj_qkv [q all | k all | v all], in_proj_z, in_proj_a, in_proj_b.
void expect_gdn35(TensorList& out, const std::string& p, const Qwen35TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t kdim = static_cast<int64_t>(cfg.gdn_key_heads) * cfg.gdn_key_head_dim;
  const int64_t vdim = static_cast<int64_t>(cfg.gdn_value_heads) * cfg.gdn_value_head_dim;
  const int64_t vh = cfg.gdn_value_heads;
  const QwenWeightClass c = QwenWeightClass::Gdn;
  // A_log and the output norm's weight are BF16 in most releases and F32 in
  // some (Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-122B-A10B-FP8): either is bound.
  add(out, p + "A_log", DType::BF16, {vh}, c, layer, -1, QwenTensorRole::Bf16OrF32);
  add_bf16(out, p + "dt_bias", {vh}, c, layer);
  add_bf16(out, p + "conv1d.weight", {2 * kdim + vdim, 1, cfg.gdn_conv_width}, c, layer);
  add_bf16(out, p + "in_proj_a.weight", {vh, H}, c, layer);
  add_bf16(out, p + "in_proj_b.weight", {vh, H}, c, layer);
  add_dense35(out, cfg, p + "in_proj_qkv", 2 * kdim + vdim, H, c, layer, false);
  add_dense35(out, cfg, p + "in_proj_z", vdim, H, c, layer, false);
  add(out, p + "norm.weight", DType::BF16, {cfg.gdn_value_head_dim}, c, layer, -1,
      QwenTensorRole::Bf16OrF32);
  add_dense35(out, cfg, p + "out_proj", H, vdim, c, layer, false);
}

void expect_full35(TensorList& out, const std::string& p, const Qwen35TextConfig& cfg, int layer,
                   bool is_mtp) {
  const int64_t H = cfg.hidden_size;
  const int64_t qh = cfg.num_attention_heads, kvh = cfg.num_key_value_heads;
  const int64_t d = cfg.head_dim;
  const QwenWeightClass c = QwenWeightClass::FullAttn;
  // The q projection stacks [q | gate] per head (attn_output_gate).
  add_dense35(out, cfg, p + "q_proj", 2 * qh * d, H, c, layer, is_mtp);
  add_dense35(out, cfg, p + "k_proj", kvh * d, H, c, layer, is_mtp);
  add_dense35(out, cfg, p + "v_proj", kvh * d, H, c, layer, is_mtp);
  add_dense35(out, cfg, p + "o_proj", H, qh * d, c, layer, is_mtp);
  add_bf16(out, p + "q_norm.weight", {d}, c, layer);
  add_bf16(out, p + "k_norm.weight", {d}, c, layer);
}

// The routed MoE of the Qwen3.5 dialect (Qwen3_5Moe: Qwen3.6-35B-A3B),
// under `mlp.`: BF16 router and shared gate, then the shared expert and the
// routed experts as the recipe ships them.
//   The FP8 release: every matrix block FP8, per expert, the draft layer too.
//   The NVFP4 mixed release: the modelopt NVFP4 set per matrix in the
//   backbone; in the draft layer a BF16 shared expert and the experts as the
//   module's own two stacked parameters — `experts.gate_up_proj`
//   [E, 2I, H] (an expert's gate rows, then its up rows) and
//   `experts.down_proj` [E, H, I].
//   The experts-only NVFP4 release: a BF16 shared expert in every layer, the
//   modelopt NVFP4 set per routed-expert matrix in the backbone, per-expert
//   BF16 matrices in the draft layer.
// (The compressed-tensors container is the Qwen3Next dialect's table under
// these names: expect_moe_next.)
void expect_moe35(TensorList& out, const std::string& p, const Qwen35TextConfig& cfg, int layer,
                  bool is_mtp) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.moe_intermediate_size;
  const int64_t S = cfg.shared_expert_intermediate_size;
  const int64_t E = cfg.num_experts;
  add_bf16(out, p + "gate.weight", {E, H}, QwenWeightClass::Router, layer);
  add_bf16(out, p + "shared_expert_gate.weight", {1, H}, QwenWeightClass::Router, layer);
  const std::string sp = p + "shared_expert.";
  if (cfg.quant_kind == Qwen35QuantKind::Nvfp4Experts) {
    add_bf16(out, sp + "gate_proj.weight", {S, H}, QwenWeightClass::SharedExpert, layer);
    add_bf16(out, sp + "up_proj.weight", {S, H}, QwenWeightClass::SharedExpert, layer);
    add_bf16(out, sp + "down_proj.weight", {H, S}, QwenWeightClass::SharedExpert, layer);
    for (int e = 0; e < cfg.num_experts; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      add_fp4_or_bf16(out, cfg, ep + "gate_proj", I, H, QwenWeightClass::RoutedExpert, layer, e, is_mtp);
      add_fp4_or_bf16(out, cfg, ep + "up_proj", I, H, QwenWeightClass::RoutedExpert, layer, e, is_mtp);
      add_fp4_or_bf16(out, cfg, ep + "down_proj", H, I, QwenWeightClass::RoutedExpert, layer, e, is_mtp);
    }
    return;
  }
  if (cfg.quant_kind == Qwen35QuantKind::Nvfp4Mixed) {
    add_fp4_or_bf16(out, cfg, sp + "gate_proj", S, H, QwenWeightClass::SharedExpert, layer, -1, is_mtp);
    add_fp4_or_bf16(out, cfg, sp + "up_proj", S, H, QwenWeightClass::SharedExpert, layer, -1, is_mtp);
    add_fp4_or_bf16(out, cfg, sp + "down_proj", H, S, QwenWeightClass::SharedExpert, layer, -1, is_mtp);
    if (is_mtp) {
      add_bf16(out, p + "experts.gate_up_proj", {E, 2 * I, H}, QwenWeightClass::RoutedExpert, layer);
      add_bf16(out, p + "experts.down_proj", {E, H, I}, QwenWeightClass::RoutedExpert, layer);
      return;
    }
    for (int e = 0; e < cfg.num_experts; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      add_fp4(out, ep + "gate_proj", I, H, QwenWeightClass::RoutedExpert, layer, e);
      add_fp4(out, ep + "up_proj", I, H, QwenWeightClass::RoutedExpert, layer, e);
      add_fp4(out, ep + "down_proj", H, I, QwenWeightClass::RoutedExpert, layer, e);
    }
    return;
  }
  add_fp8(out, sp + "gate_proj.weight", S, H, QwenWeightClass::SharedExpert, layer);
  add_fp8(out, sp + "up_proj.weight", S, H, QwenWeightClass::SharedExpert, layer);
  add_fp8(out, sp + "down_proj.weight", H, S, QwenWeightClass::SharedExpert, layer);
  for (int e = 0; e < cfg.num_experts; ++e) {
    const std::string ep = p + "experts." + std::to_string(e) + ".";
    add_fp8_expert(out, ep + "gate_proj.weight", I, H, layer, e);
    add_fp8_expert(out, ep + "up_proj.weight", I, H, layer, e);
    add_fp8_expert(out, ep + "down_proj.weight", H, I, layer, e);
  }
}

void expect_dense_mlp35(TensorList& out, const std::string& p, const Qwen35TextConfig& cfg,
                        int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  const QwenWeightClass c = QwenWeightClass::DenseMlp;
  add_dense35(out, cfg, p + "gate_proj", I, H, c, layer, false);
  add_dense35(out, cfg, p + "up_proj", I, H, c, layer, false);
  add_dense35(out, cfg, p + "down_proj", H, I, c, layer, false);
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
  // The compressed-tensors container quantizes what it does on the Qwen3Next
  // dialect — the attention q/k/v/o, the shared expert, the routed experts,
  // BF16 in the draft layer — under the same names below the layer prefix.
  const bool packed = cfg.quant_kind == Qwen35QuantKind::Nvfp4Packed;
  if (kind == Qwen35LayerKind::Gdn)
    expect_gdn35(out, p + "linear_attn.", cfg, layer);
  else if (packed)
    expect_full_next(out, p + "self_attn.", cfg, layer, is_mtp);
  else
    expect_full35(out, p + "self_attn.", cfg, layer, is_mtp);
  if (cfg.moe() && packed)
    expect_moe_next(out, p + "mlp.", cfg, layer, is_mtp);
  else if (cfg.moe())
    expect_moe35(out, p + "mlp.", cfg, layer, is_mtp);
  else
    expect_dense_mlp35(out, p + "mlp.", cfg, layer);
  return out;
}

std::vector<QwenExpectedTensor> qwen35_expected_global_tensors(const Qwen35TextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, qwen35_model_prefix(cfg) + "embed_tokens.weight", {cfg.vocab_size, H},
           QwenWeightClass::Embed, -1);
  // The NVFP4 mixed release quantizes the head with the experts (the modelopt
  // set over [vocab, H]); every other release ships it BF16 — or not at all
  // when the config ties it to the embedding (the head then reads
  // embed_tokens, and a stored lm_head.weight would be unexpected).
  if (!cfg.tie_word_embeddings) {
    if (cfg.quant_kind == Qwen35QuantKind::Nvfp4Mixed)
      add_fp4(out, "lm_head", cfg.vocab_size, H, QwenWeightClass::LmHead, -1);
    else
      add_bf16(out, "lm_head.weight", {cfg.vocab_size, H}, QwenWeightClass::LmHead, -1);
  }
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
  // A config that names a draft layer the checkpoint does not carry (a
  // fine-tune saved without its `mtp.*` tensors): said once, ahead of the
  // per-tensor lines.
  if (cfg.mtp_layer() >= 0) {
    bool any_mtp = false;
    for (const auto& [name, desc] : present) {
      (void)desc;
      if (name.rfind("mtp.", 0) == 0) {
        any_mtp = true;
        break;
      }
    }
    if (!any_mtp)
      push_error("the config names a draft layer (mtp_num_hidden_layers 1) and the checkpoint "
                 "holds no mtp.* tensor: it cannot be served as described — a config with "
                 "mtp_num_hidden_layers 0 binds it without the draft");
  }
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
    // A Bf16OrF32 vector is listed BF16 and bound in either dtype.
    if (it->second.dtype != e.dtype &&
        !(e.role == QwenTensorRole::Bf16OrF32 && it->second.dtype == DType::F32)) {
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
    // A tied config whose checkpoint was saved with the head written out as
    // well (Hcompany/Holo-3.1-0.8B): the head reads the embedding, as the
    // model class ties it at load; the stored copy is not read.
    if (cfg.tie_word_embeddings && name == "lm_head.weight") {
      ++rep.out_of_scope;
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
