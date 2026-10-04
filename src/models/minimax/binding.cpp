#include "models/minimax/binding.hpp"

#include <algorithm>
#include <format>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace dgpp {
namespace {

using TensorList = std::vector<MinimaxExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         MinimaxWeightClass cls, int layer, int expert = -1,
         MinimaxTensorRole role = MinimaxTensorRole::Plain) {
  out.push_back(
      MinimaxExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer, expert, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              MinimaxWeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

// One modelopt NVFP4 [rows, cols] matrix: the packed codes, the per-16 e4m3
// scales, the fp32 per-tensor scale (a multiplier), and the optional
// activation scale.
void add_fp4(TensorList& out, const std::string& base, int64_t rows, int64_t cols, int layer,
             int expert, int group) {
  if (cols % group != 0 || cols % 2 != 0)
    throw std::invalid_argument("minimax binding: NVFP4 K must be a multiple of the group on " +
                                base);
  const MinimaxWeightClass cls = MinimaxWeightClass::RoutedExpert;
  add(out, base + ".weight", DType::U8, {rows, cols / 2}, cls, layer, expert,
      MinimaxTensorRole::Fp4Payload);
  add(out, base + ".weight_scale", DType::F8_E4M3, {rows, cols / group}, cls, layer, expert,
      MinimaxTensorRole::Fp4Scale);
  add(out, base + ".weight_scale_2", DType::F32, {}, cls, layer, expert,
      MinimaxTensorRole::Fp4Global);
  add(out, base + ".input_scale", DType::F32, {}, cls, layer, expert,
      MinimaxTensorRole::InputScale);
}

}  // namespace

std::string minimax_layer_prefix(int layer) {
  return "model.layers." + std::to_string(layer) + ".";
}

std::vector<MinimaxExpectedTensor> minimax_expected_layer_tensors(const MinimaxTextConfig& cfg,
                                                                  int layer) {
  if (layer < 0 || layer >= cfg.num_hidden_layers)
    throw std::invalid_argument("minimax_expected_layer_tensors: layer out of range");
  const std::string p = minimax_layer_prefix(layer);
  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size;
  const int64_t qw = cfg.q_width(), kw = cfg.kv_width();
  const MinimaxWeightClass a = MinimaxWeightClass::Attention;
  TensorList out;
  add_bf16(out, p + "input_layernorm.weight", {H}, MinimaxWeightClass::LayerNorm, layer);
  add_bf16(out, p + "post_attention_layernorm.weight", {H}, MinimaxWeightClass::LayerNorm, layer);
  add_bf16(out, p + "self_attn.q_proj.weight", {qw, H}, a, layer);
  add_bf16(out, p + "self_attn.k_proj.weight", {kw, H}, a, layer);
  add_bf16(out, p + "self_attn.v_proj.weight", {kw, H}, a, layer);
  add_bf16(out, p + "self_attn.o_proj.weight", {H, qw}, a, layer);
  // The per-layer norms: a weight per element of the whole projection.
  add_bf16(out, p + "self_attn.q_norm.weight", {qw}, a, layer);
  add_bf16(out, p + "self_attn.k_norm.weight", {kw}, a, layer);
  add_bf16(out, p + "block_sparse_moe.gate.weight", {cfg.n_routed_experts, H},
           MinimaxWeightClass::Router, layer);
  add_bf16(out, p + "block_sparse_moe.e_score_correction_bias", {cfg.n_routed_experts},
           MinimaxWeightClass::Router, layer);
  for (int e = 0; e < cfg.n_routed_experts; ++e) {
    const std::string ep = p + "block_sparse_moe.experts." + std::to_string(e) + ".";
    add_fp4(out, ep + "w1", I, H, layer, e, cfg.fp4_group_size);
    add_fp4(out, ep + "w2", H, I, layer, e, cfg.fp4_group_size);
    add_fp4(out, ep + "w3", I, H, layer, e, cfg.fp4_group_size);
  }
  return out;
}

std::vector<MinimaxExpectedTensor> minimax_expected_global_tensors(const MinimaxTextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "model.embed_tokens.weight", {cfg.vocab_size, H}, MinimaxWeightClass::Embed, -1);
  add_bf16(out, "model.norm.weight", {H}, MinimaxWeightClass::FinalNorm, -1);
  add_bf16(out, "lm_head.weight", {cfg.vocab_size, H}, MinimaxWeightClass::LmHead, -1);
  return out;
}

std::vector<MinimaxExpectedTensor> minimax_expected_tensors(const MinimaxTextConfig& cfg) {
  TensorList out = minimax_expected_global_tensors(cfg);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    TensorList layer = minimax_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()),
               std::make_move_iterator(layer.end()));
  }
  return out;
}

MinimaxBindReport minimax_validate_binding(
    const MinimaxTextConfig& cfg, const std::unordered_map<std::string, MinimaxTensorDesc>& present,
    size_t max_errors) {
  MinimaxBindReport rep;
  const auto expected = minimax_expected_tensors(cfg);
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
    if (!e.optional()) ++rep.expected;
    auto it = present.find(e.name);
    if (it == present.end()) {
      if (e.optional()) {
        ++rep.optional_absent;
      } else {
        ++rep.missing;
        push_error(std::format("missing tensor '{}'", e.name));
      }
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
    rep.bytes += e.nbytes();
    if (e.optional()) {
      ++rep.optional_present;
      continue;
    }
    ++rep.matched;
    rep.loaded_bytes += e.nbytes();
    if (e.role == MinimaxTensorRole::Fp4Payload) ++rep.fp4_matrices;
  }
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

std::vector<std::string> minimax_shard_files(const minijson::Value& index_root) {
  const minijson::Value* map = index_root.is_object() ? index_root.find("weight_map") : nullptr;
  if (!map || !map->is_object() || map->members().empty())
    throw std::runtime_error("model.safetensors.index.json: no weight_map");
  std::vector<std::string> files;
  for (const auto& m : map->members()) {
    if (!m.value.is_string())
      throw std::runtime_error("model.safetensors.index.json: non-string shard name");
    files.emplace_back(m.value.as_string());
  }
  std::sort(files.begin(), files.end());
  files.erase(std::unique(files.begin(), files.end()), files.end());
  return files;
}

MinimaxWeightBytes minimax_weight_bytes(const MinimaxTextConfig& cfg, int world,
                                        bool replicate_qk) {
  if (world < 1) throw std::invalid_argument("minimax_weight_bytes: world must be >= 1");
  minimax_tp_validate_geometry(cfg, 0, world);
  const size_t w = static_cast<size_t>(world);
  MinimaxWeightBytes b;
  for (const auto& e : minimax_expected_global_tensors(cfg)) {
    // The head is vocabulary-sharded; the embedding is whole on every rank
    // (GLM-4.7's loader keeps it so: a lookup, not a product).
    if (e.cls == MinimaxWeightClass::Embed)
      b.embedding += e.nbytes();
    else if (e.cls == MinimaxWeightClass::LmHead)
      b.head += e.nbytes() / w;
    else
      b.routers_and_norms += e.nbytes();
  }
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const std::string p = minimax_layer_prefix(l) + "self_attn.";
    for (const auto& e : minimax_expected_layer_tensors(cfg, l)) {
      if (e.unused()) continue;
      switch (e.cls) {
        case MinimaxWeightClass::Attention: {
          const bool qk = e.name == p + "q_proj.weight" || e.name == p + "k_proj.weight" ||
                          e.name == p + "q_norm.weight" || e.name == p + "k_norm.weight";
          b.attention += (qk && replicate_qk) ? e.nbytes() : e.nbytes() / w;
          break;
        }
        case MinimaxWeightClass::RoutedExpert:
          // The intermediate dimension is sliced: gate / up rows, down
          // columns — payload and block scales alike; the F32 scalar stays.
          b.routed_experts += e.role == MinimaxTensorRole::Fp4Global ? e.nbytes() : e.nbytes() / w;
          break;
        default:
          b.routers_and_norms += e.nbytes();
      }
    }
  }
  return b;
}

size_t minimax_kv_bytes_per_token(const MinimaxTextConfig& cfg, int world) {
  minimax_tp_validate_geometry(cfg, 0, world);
  const size_t local_kv = static_cast<size_t>(cfg.num_key_value_heads / world);
  return 2 * local_kv * static_cast<size_t>(cfg.head_dim) * 2 *
         static_cast<size_t>(cfg.num_hidden_layers);
}

void minimax_tp_validate_geometry(const MinimaxTextConfig& cfg, int rank, int world) {
  auto fail = [](const std::string& what) {
    throw std::invalid_argument("minimax tp geometry: " + what);
  };
  if (world < 1 || rank < 0 || rank >= world) fail("rank/world out of range");
  if (cfg.num_attention_heads % world != 0) fail("num_attention_heads must divide by world");
  // A kv head lives on one rank with all of its query heads (the paged K/V
  // cache is per rank): world must divide the kv heads.
  if (cfg.num_key_value_heads % world != 0) fail("num_key_value_heads must divide by world");
  if (cfg.vocab_size % world != 0) fail("vocab_size must divide by world");
  if (world == 1) return;
  if (cfg.moe_intermediate_size % world != 0) fail("intermediate_size must divide by world");
  if ((cfg.moe_intermediate_size / world) % 32 != 0)
    fail("the expert intermediate slice must be a multiple of 32 (the NVFP4 GEMV core's K)");
  // o_proj's input columns are sliced by head.
  if ((cfg.num_attention_heads / world) * cfg.head_dim % 16 != 0)
    fail("the o_proj slice must be a multiple of 16");
}

}  // namespace dgpp
