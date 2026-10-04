#include "models/qwen3/binding.hpp"

#include <cstddef>
#include <cstdlib>
#include <format>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace dgpp {
namespace {

using TensorList = std::vector<Qwen3ExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         Qwen3WeightClass cls, int layer, int expert = -1,
         Qwen3TensorRole role = Qwen3TensorRole::Plain) {
  out.push_back(Qwen3ExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer, expert, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              Qwen3WeightClass cls, int layer, int expert = -1) {
  add(out, name, DType::BF16, std::move(shape), cls, layer, expert);
}

// The NVFP4 set of one [rows, cols] matrix in the config's container: the
// same code and block-scale geometry under the two releases' names, with a
// scalar global in modelopt's (shape [], a multiplier) and a one-element
// global in compressed-tensors' (shape [1], a divisor).
void add_fp4(TensorList& out, const Qwen3TextConfig& cfg, const std::string& base, int64_t rows,
             int64_t cols, Qwen3WeightClass cls, int layer, int expert = -1) {
  const std::vector<int64_t> scalar = cfg.fp4_packed() ? std::vector<int64_t>{1} : std::vector<int64_t>{};
  add(out, qwen3_fp4_payload_name(cfg, base), DType::U8, {rows, cols / 2}, cls, layer, expert,
      Qwen3TensorRole::Fp4Payload);
  add(out, base + ".weight_scale", DType::F8_E4M3, {rows, cols / 16}, cls, layer, expert,
      Qwen3TensorRole::Fp4Scale);
  add(out, qwen3_fp4_global_name(cfg, base), DType::F32, scalar, cls, layer, expert,
      Qwen3TensorRole::Fp4Global);
  add(out, qwen3_fp4_input_name(cfg, base), DType::F32, scalar, cls, layer, expert,
      Qwen3TensorRole::InputScale);
}

void add_fp4_or_bf16(TensorList& out, const Qwen3TextConfig& cfg, const std::string& base,
                     int64_t rows, int64_t cols, Qwen3WeightClass cls, int layer, int expert,
                     bool bf16) {
  if (bf16)
    add_bf16(out, base + ".weight", {rows, cols}, cls, layer, expert);
  else
    add_fp4(out, cfg, base, rows, cols, cls, layer, expert);
}

// Attention. Dense: all four BF16. modelopt: q/k/v BF16 with the recipe's
// cache scales beside k and v, o_proj NVFP4. compressed-tensors: all four
// NVFP4, no cache scales.
void expect_attention(TensorList& out, const std::string& p, const Qwen3TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t qh = cfg.num_attention_heads, kvh = cfg.num_key_value_heads;
  const int64_t d = cfg.head_dim;
  const Qwen3WeightClass c = Qwen3WeightClass::Attention;
  const bool qkv_bf16 = !cfg.fp4_packed();
  const bool o_bf16 = !cfg.fp4();
  add_fp4_or_bf16(out, cfg, p + "q_proj", qh * d, H, c, layer, -1, qkv_bf16);
  add_fp4_or_bf16(out, cfg, p + "k_proj", kvh * d, H, c, layer, -1, qkv_bf16);
  add_fp4_or_bf16(out, cfg, p + "v_proj", kvh * d, H, c, layer, -1, qkv_bf16);
  if (cfg.kv_cache_scales) {
    add(out, p + "k_proj.k_scale", DType::F32, {}, Qwen3WeightClass::KvScale, layer, -1,
        Qwen3TensorRole::KvScale);
    add(out, p + "v_proj.v_scale", DType::F32, {}, Qwen3WeightClass::KvScale, layer, -1,
        Qwen3TensorRole::KvScale);
  }
  add_fp4_or_bf16(out, cfg, p + "o_proj", H, qh * d, c, layer, -1, o_bf16);
  add_bf16(out, p + "q_norm.weight", {d}, c, layer);
  add_bf16(out, p + "k_norm.weight", {d}, c, layer);
}

// The routed MoE: a BF16 router and every expert's gate/up [I, H] and
// down [H, I] as NVFP4 sets. No shared expert, no router bias.
void expect_moe(TensorList& out, const std::string& p, const Qwen3TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.moe_intermediate_size;
  add_bf16(out, p + "gate.weight", {cfg.num_experts, H}, Qwen3WeightClass::Router, layer);
  for (int e = 0; e < cfg.num_experts; ++e) {
    const std::string ep = p + "experts." + std::to_string(e) + ".";
    add_fp4(out, cfg, ep + "gate_proj", I, H, Qwen3WeightClass::RoutedExpert, layer, e);
    add_fp4(out, cfg, ep + "up_proj", I, H, Qwen3WeightClass::RoutedExpert, layer, e);
    add_fp4(out, cfg, ep + "down_proj", H, I, Qwen3WeightClass::RoutedExpert, layer, e);
  }
}

void expect_dense_mlp(TensorList& out, const std::string& p, const Qwen3TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  const Qwen3WeightClass c = Qwen3WeightClass::DenseMlp;
  add_bf16(out, p + "gate_proj.weight", {I, H}, c, layer);
  add_bf16(out, p + "up_proj.weight", {I, H}, c, layer);
  add_bf16(out, p + "down_proj.weight", {H, I}, c, layer);
}

// A biased BF16 Linear of the tower.
void add_vis_linear(TensorList& out, const std::string& base, int64_t rows, int64_t cols) {
  add_bf16(out, base + ".weight", {rows, cols}, Qwen3WeightClass::Vision, -1);
  add_bf16(out, base + ".bias", {rows}, Qwen3WeightClass::Vision, -1);
}
// A LayerNorm of the tower (weight and bias).
void add_vis_norm(TensorList& out, const std::string& base, int64_t width) {
  add_bf16(out, base + ".weight", {width}, Qwen3WeightClass::Vision, -1);
  add_bf16(out, base + ".bias", {width}, Qwen3WeightClass::Vision, -1);
}
// Qwen3VLMoeVisionPatchMerger: norm over the unmerged width (the main
// merger) or the merged one (use_postshuffle_norm: the deepstack mergers),
// then fc1 [M, M] and fc2 [out, M] with M the merged width.
void add_vis_merger(TensorList& out, const std::string& base, const Qwen3VisionConfig& v,
                    bool postshuffle_norm) {
  const int64_t M = v.merged_width();
  add_vis_norm(out, base + ".norm", postshuffle_norm ? M : v.hidden_size);
  add_vis_linear(out, base + ".linear_fc1", M, M);
  add_vis_linear(out, base + ".linear_fc2", v.out_hidden_size, M);
}

}  // namespace

std::string qwen3_model_prefix(const Qwen3TextConfig& cfg) {
  if (cfg.vl()) return "model.language_model.";
  return cfg.bare_names ? "" : "model.";
}

std::string qwen3_layer_prefix(const Qwen3TextConfig& cfg, int layer) {
  return qwen3_model_prefix(cfg) + "layers." + std::to_string(layer) + ".";
}

std::string qwen3_fp4_payload_name(const Qwen3TextConfig& cfg, const std::string& base) {
  return base + (cfg.fp4_packed() ? ".weight_packed" : ".weight");
}
std::string qwen3_fp4_global_name(const Qwen3TextConfig& cfg, const std::string& base) {
  return base + (cfg.fp4_packed() ? ".weight_global_scale" : ".weight_scale_2");
}
std::string qwen3_fp4_input_name(const Qwen3TextConfig& cfg, const std::string& base) {
  return base + (cfg.fp4_packed() ? ".input_global_scale" : ".input_scale");
}

std::vector<Qwen3ExpectedTensor> qwen3_expected_layer_tensors(const Qwen3TextConfig& cfg, int layer) {
  if (layer < 0 || layer >= cfg.num_hidden_layers)
    throw std::invalid_argument("qwen3_expected_layer_tensors: layer out of range");
  const std::string p = qwen3_layer_prefix(cfg, layer);
  TensorList out;
  add_bf16(out, p + "input_layernorm.weight", {cfg.hidden_size}, Qwen3WeightClass::LayerNorm, layer);
  add_bf16(out, p + "post_attention_layernorm.weight", {cfg.hidden_size}, Qwen3WeightClass::LayerNorm,
           layer);
  expect_attention(out, p + "self_attn.", cfg, layer);
  if (cfg.moe())
    expect_moe(out, p + "mlp.", cfg, layer);
  else
    expect_dense_mlp(out, p + "mlp.", cfg, layer);
  return out;
}

std::vector<Qwen3ExpectedTensor> qwen3_expected_global_tensors(const Qwen3TextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  const std::string p = qwen3_model_prefix(cfg);
  add_bf16(out, p + "embed_tokens.weight", {cfg.vocab_size, H}, Qwen3WeightClass::Embed, -1);
  add_bf16(out, p + "norm.weight", {H}, Qwen3WeightClass::FinalNorm, -1);
  // Tied embeddings: the head IS embed_tokens and the releases store no
  // lm_head.weight (the validator accepts a stored copy without reading it).
  if (!cfg.tie_word_embeddings)
    add_bf16(out, "lm_head.weight", {cfg.vocab_size, H}, Qwen3WeightClass::LmHead, -1);
  return out;
}

std::vector<Qwen3ExpectedTensor> qwen3_expected_text_tensors(const Qwen3TextConfig& cfg) {
  TensorList out = qwen3_expected_global_tensors(cfg);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    TensorList layer = qwen3_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()), std::make_move_iterator(layer.end()));
  }
  return out;
}

std::vector<Qwen3ExpectedTensor> qwen3_expected_vision_tensors(const Qwen3TextConfig& cfg) {
  TensorList out;
  if (!cfg.vision.has_value()) return out;
  const Qwen3VisionConfig& v = *cfg.vision;
  const std::string p = "model.visual.";
  const int64_t D = v.hidden_size;
  // The Conv3d patch embedding and the learned position table.
  add_bf16(out, p + "patch_embed.proj.weight",
           {D, v.in_channels, v.temporal_patch_size, v.patch_size, v.patch_size}, Qwen3WeightClass::Vision, -1);
  add_bf16(out, p + "patch_embed.proj.bias", {D}, Qwen3WeightClass::Vision, -1);
  add_bf16(out, p + "pos_embed.weight", {v.num_position_embeddings, D}, Qwen3WeightClass::Vision, -1);
  for (int b = 0; b < v.depth; ++b) {
    const std::string bp = p + "blocks." + std::to_string(b) + ".";
    add_vis_norm(out, bp + "norm1", D);
    add_vis_norm(out, bp + "norm2", D);
    add_vis_linear(out, bp + "attn.qkv", 3 * D, D);
    add_vis_linear(out, bp + "attn.proj", D, D);
    add_vis_linear(out, bp + "mlp.linear_fc1", v.intermediate_size, D);
    add_vis_linear(out, bp + "mlp.linear_fc2", D, v.intermediate_size);
  }
  add_vis_merger(out, p + "merger", v, /*postshuffle_norm=*/false);
  for (size_t i = 0; i < v.deepstack_visual_indexes.size(); ++i)
    add_vis_merger(out, p + "deepstack_merger_list." + std::to_string(i), v, /*postshuffle_norm=*/true);
  return out;
}

std::vector<Qwen3ExpectedTensor> qwen3_expected_tensors(const Qwen3TextConfig& cfg) {
  TensorList out = qwen3_expected_text_tensors(cfg);
  TensorList vis = qwen3_expected_vision_tensors(cfg);
  out.insert(out.end(), std::make_move_iterator(vis.begin()), std::make_move_iterator(vis.end()));
  return out;
}

void qwen3_apply_header_naming(Qwen3TextConfig& cfg, const Qwen3PresentMap& present) {
  if (cfg.dialect != Qwen3Dialect::Dense) return;
  cfg.bare_names = present.count("embed_tokens.weight") != 0 &&
                   present.count("model.embed_tokens.weight") == 0;
}

Qwen3BindReport qwen3_validate_binding(const Qwen3TextConfig& cfg, const Qwen3PresentMap& present,
                                       size_t max_errors) {
  Qwen3BindReport rep;
  const auto expected = qwen3_expected_tensors(cfg);
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
    if (e.quantized()) ++rep.fp4_matrices;
    if (e.cls == Qwen3WeightClass::Vision) ++rep.vision;
  }
  const std::string layers_prefix = qwen3_model_prefix(cfg) + "layers.";
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    // A tied checkpoint that also stores its head: the same matrix twice.
    if (cfg.tie_word_embeddings && name == "lm_head.weight" && desc.dtype == DType::BF16 &&
        desc.shape == std::vector<int64_t>{cfg.vocab_size, cfg.hidden_size}) {
      ++rep.tied_head_copies;
      continue;
    }
    // A truncated config (the check apps' --layers N): the layers past it
    // are out of scope, not unexpected.
    if (name.rfind(layers_prefix, 0) == 0) {
      const size_t dot = name.find('.', layers_prefix.size());
      const int64_t idx = dot == std::string::npos
                              ? -1
                              : std::atoll(name.substr(layers_prefix.size(), dot - layers_prefix.size()).c_str());
      if (idx >= cfg.num_hidden_layers) {
        ++rep.out_of_scope;
        continue;
      }
    }
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

void qwen3_tp_validate_geometry(const Qwen3TextConfig& cfg, int rank, int world) {
  auto fail = [](const std::string& what) { throw std::invalid_argument("qwen3 tp geometry: " + what); };
  if (world < 1 || rank < 0 || rank >= world) fail("rank/world out of range");
  if (cfg.num_attention_heads % world != 0) fail("num_attention_heads must divide by world");
  if (cfg.num_key_value_heads % world != 0 && world % cfg.num_key_value_heads != 0)
    fail("num_key_value_heads must divide world or be divided by it");
  // A rank's query heads must all read kv heads the rank holds.
  if (cfg.num_key_value_heads < world && (cfg.num_attention_heads / world) > cfg.q_heads_per_kv())
    fail("query heads per rank exceed one kv head's group");
  if (cfg.moe()) {
    if (cfg.moe_intermediate_size % (32 * world) != 0)
      fail("moe_intermediate_size / world must be a multiple of 32");
  } else if (cfg.intermediate_size % world != 0) {
    fail("intermediate_size must divide by world");
  }
}

Qwen3RankBytes qwen3_rank_resident_bytes(const Qwen3TextConfig& cfg, int rank, int world) {
  qwen3_tp_validate_geometry(cfg, rank, world);
  Qwen3RankBytes b;
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t L = static_cast<size_t>(cfg.num_hidden_layers);
  const size_t d = static_cast<size_t>(cfg.head_dim);
  const size_t lh = static_cast<size_t>(cfg.num_attention_heads / world);
  const size_t lkv = cfg.num_key_value_heads >= world ? static_cast<size_t>(cfg.num_key_value_heads / world) : 1;
  // q [lh*d, H], k / v [lkv*d, H], o [H, lh*d]: BF16 residents.
  b.attention = L * (lh * d * H * 2 + 2 * lkv * d * H * 2 + H * lh * d * 2);
  b.norms = L * (2 * H * 2 + 2 * d * 2) + H * 2;
  if (cfg.moe()) {
    const size_t I = static_cast<size_t>(cfg.moe_intermediate_size / world);
    const size_t E = static_cast<size_t>(cfg.num_experts);
    // gate, up [I, H]; down [H, I]: half a byte a code, an e4m3 scale per
    // 16, one F32 global each.
    const size_t matrix = I * H / 2 + I * H / 16 + 4;
    b.experts = L * E * 3 * matrix;
    b.router = L * E * H * 2;
  } else {
    const size_t I = static_cast<size_t>(cfg.intermediate_size / world);
    b.dense_mlp = L * 3 * I * H * 2;
  }
  const size_t V = static_cast<size_t>(cfg.vocab_size);
  b.embed = V * H * 2;
  if (!cfg.tie_word_embeddings) {
    const size_t begin = V * static_cast<size_t>(rank) / static_cast<size_t>(world);
    const size_t end = V * static_cast<size_t>(rank + 1) / static_cast<size_t>(world);
    b.head = (end - begin) * H * 2;
  }
  return b;
}

}  // namespace dgpp
