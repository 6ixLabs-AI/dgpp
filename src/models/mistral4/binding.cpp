#include "models/mistral4/binding.hpp"

#include <format>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace dgpp {
namespace {

using TensorList = std::vector<Mistral4ExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         Mistral4WeightClass cls, int layer, int expert = -1,
         Mistral4TensorRole role = Mistral4TensorRole::Plain) {
  out.push_back(
      Mistral4ExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer, expert, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              Mistral4WeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}

void add_vision(TensorList& out, const std::string& name, std::vector<int64_t> shape) {
  add(out, name, DType::BF16, std::move(shape), Mistral4WeightClass::Vision, -1, -1,
      Mistral4TensorRole::Unused);
}

// One compressed-tensors NVFP4 [rows, cols] matrix: the packed codes, the
// per-16 e4m3 scales, the F32 [1] global scale (a divisor) and — when the
// recipe quantized activations — the unused activation scale, whose dtype
// differs between the routed experts (F32) and the shared one (BF16).
void add_fp4(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
             Mistral4WeightClass cls, int layer, int expert, const Mistral4TextConfig& cfg) {
  const int group = cfg.fp4_group_size;
  if (cols % group != 0 || cols % 2 != 0)
    throw std::invalid_argument("mistral4 binding: NVFP4 K must be a multiple of the group on " +
                                base);
  add(out, base + ".weight_packed", DType::U8, {rows, cols / 2}, cls, layer, expert,
      Mistral4TensorRole::Fp4Payload);
  add(out, base + ".weight_scale", DType::F8_E4M3, {rows, cols / group}, cls, layer, expert,
      Mistral4TensorRole::Fp4Scale);
  add(out, base + ".weight_global_scale", DType::F32, {1}, cls, layer, expert,
      Mistral4TensorRole::Fp4Global);
  if (cfg.activation_scales_present)
    add(out, base + ".input_global_scale",
        cls == Mistral4WeightClass::SharedExpert ? DType::BF16 : DType::F32, {1}, cls, layer,
        expert, Mistral4TensorRole::InputScale);
}

// w1: the gate projection [I, H]; w2: the down projection [H, I]; w3: the
// up projection [I, H] (transformers' conversion: w1 -> gate_proj, w2 ->
// down_proj, w3 -> up_proj).
void add_mlp(TensorList& out, const std::string& p, int64_t inter, Mistral4WeightClass cls,
             int layer, int expert, const Mistral4TextConfig& cfg) {
  const int64_t H = cfg.hidden_size;
  add_fp4(out, p + "w1", inter, H, cls, layer, expert, cfg);
  add_fp4(out, p + "w2", H, inter, cls, layer, expert, cfg);
  add_fp4(out, p + "w3", inter, H, cls, layer, expert, cfg);
}

}  // namespace

std::string mistral4_layer_prefix(int layer) {
  return "layers." + std::to_string(layer) + ".";
}

std::vector<Mistral4ExpectedTensor> mistral4_expected_layer_tensors(const Mistral4TextConfig& cfg,
                                                                    int layer) {
  if (layer < 0 || layer >= cfg.num_hidden_layers)
    throw std::invalid_argument("mistral4_expected_layer_tensors: layer out of range");
  const std::string p = mistral4_layer_prefix(layer);
  const int64_t H = cfg.hidden_size, nh = cfg.num_attention_heads;
  const Mistral4WeightClass a = Mistral4WeightClass::Attention;
  TensorList out;
  add_bf16(out, p + "attention_norm.weight", {H}, Mistral4WeightClass::LayerNorm, layer);
  add_bf16(out, p + "ffn_norm.weight", {H}, Mistral4WeightClass::LayerNorm, layer);
  // MLA: the query latent and its expansion to [nope | rope] per head; the
  // fused [kv latent | rope key] projection; the latent's expansion to
  // [k_nope | v] per head; the output projection over the heads' values.
  add_bf16(out, p + "attention.wq_a.weight", {cfg.q_lora_rank, H}, a, layer);
  add_bf16(out, p + "attention.q_a_norm.weight", {cfg.q_lora_rank}, a, layer);
  add_bf16(out, p + "attention.wq_b.weight", {nh * cfg.qk_head_dim(), cfg.q_lora_rank}, a, layer);
  add_bf16(out, p + "attention.wkv_a_with_mqa.weight", {cfg.kv_a_rows(), H}, a, layer);
  add_bf16(out, p + "attention.kv_a_norm.weight", {cfg.kv_lora_rank}, a, layer);
  add_bf16(out, p + "attention.wkv_b.weight",
           {nh * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank}, a, layer);
  add_bf16(out, p + "attention.wo.weight", {H, nh * cfg.v_head_dim}, a, layer);
  add_bf16(out, p + "gate.weight", {cfg.n_routed_experts, H}, Mistral4WeightClass::Router, layer);
  for (int e = 0; e < cfg.n_routed_experts; ++e)
    add_mlp(out, p + "experts." + std::to_string(e) + ".", cfg.moe_intermediate_size,
            Mistral4WeightClass::RoutedExpert, layer, e, cfg);
  add_mlp(out, p + "shared_experts.", cfg.shared_expert_inter(), Mistral4WeightClass::SharedExpert,
          layer, -1, cfg);
  return out;
}

std::vector<Mistral4ExpectedTensor> mistral4_expected_global_tensors(
    const Mistral4TextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "tok_embeddings.weight", {cfg.vocab_size, H}, Mistral4WeightClass::Embed, -1);
  add_bf16(out, "norm.weight", {H}, Mistral4WeightClass::FinalNorm, -1);
  add_bf16(out, "output.weight", {cfg.vocab_size, H}, Mistral4WeightClass::LmHead, -1);
  return out;
}

std::vector<Mistral4ExpectedTensor> mistral4_expected_vision_tensors(
    const Mistral4TextConfig& cfg) {
  TensorList out;
  const Mistral4VisionConfig& v = cfg.vision;
  if (!v.present) return out;
  const int64_t H = cfg.hidden_size, vh = v.hidden_size, vi = v.intermediate_size;
  const int64_t merge = static_cast<int64_t>(v.spatial_merge_size) * v.spatial_merge_size;
  add_vision(out, "patch_merger.merging_layer.weight", {vh, vh * merge});
  if (v.pre_mm_projector_norm) add_vision(out, "pre_mm_projector_norm.weight", {vh});
  add_vision(out, "vision_encoder.ln_pre.weight", {vh});
  add_vision(out, "vision_encoder.patch_conv.weight",
             {vh, v.num_channels, v.patch_size, v.patch_size});
  for (int l = 0; l < v.num_hidden_layers; ++l) {
    const std::string p = "vision_encoder.transformer.layers." + std::to_string(l) + ".";
    for (const char* m : {"wk", "wo", "wq", "wv"})
      add_vision(out, p + "attention." + m + ".weight", {vh, vh});
    add_vision(out, p + "attention_norm.weight", {vh});
    add_vision(out, p + "feed_forward.w1.weight", {vi, vh});
    add_vision(out, p + "feed_forward.w2.weight", {vh, vi});
    add_vision(out, p + "feed_forward.w3.weight", {vi, vh});
    add_vision(out, p + "ffn_norm.weight", {vh});
  }
  add_vision(out, "vision_language_adapter.w_in.weight", {H, vh});
  add_vision(out, "vision_language_adapter.w_out.weight", {H, H});
  return out;
}

std::vector<Mistral4ExpectedTensor> mistral4_expected_tensors(const Mistral4TextConfig& cfg) {
  TensorList out = mistral4_expected_global_tensors(cfg);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    TensorList layer = mistral4_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()),
               std::make_move_iterator(layer.end()));
  }
  TensorList vision = mistral4_expected_vision_tensors(cfg);
  out.insert(out.end(), std::make_move_iterator(vision.begin()),
             std::make_move_iterator(vision.end()));
  return out;
}

Mistral4BindReport mistral4_validate_binding(
    const Mistral4TextConfig& cfg,
    const std::unordered_map<std::string, Mistral4TensorDesc>& present, size_t max_errors) {
  Mistral4BindReport rep;
  const auto expected = mistral4_expected_tensors(cfg);
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
    if (e.unused())
      ++rep.unused;
    else
      rep.loaded_bytes += e.nbytes();
    if (e.role == Mistral4TensorRole::Fp4Payload) ++rep.fp4_matrices;
  }
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

Mistral4WeightBytes mistral4_weight_bytes(const Mistral4TextConfig& cfg, int world) {
  if (world < 1) throw std::invalid_argument("mistral4_weight_bytes: world must be >= 1");
  mistral4_tp_validate_geometry(cfg, 0, world);
  const size_t w = static_cast<size_t>(world);
  Mistral4WeightBytes b;
  for (const auto& e : mistral4_expected_global_tensors(cfg)) {
    // The head is vocabulary-sharded; the embedding is whole on every rank
    // (the other families' loaders keep it so: a lookup, not a product).
    if (e.cls == Mistral4WeightClass::Embed)
      b.embedding += e.nbytes();
    else if (e.cls == Mistral4WeightClass::LmHead)
      b.head += e.nbytes() / w;
    else
      b.routers_and_norms += e.nbytes();
  }
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const std::string p = mistral4_layer_prefix(l) + "attention.";
    for (const auto& e : mistral4_expected_layer_tensors(cfg, l)) {
      if (e.unused()) continue;
      switch (e.cls) {
        case Mistral4WeightClass::Attention: {
          // The latents' own projections and norms are replicated (every
          // rank caches the same latent); the per-head expansions and the
          // output projection are head-sharded.
          const bool head_sharded = e.name == p + "wq_b.weight" || e.name == p + "wkv_b.weight" ||
                                    e.name == p + "wo.weight";
          b.attention += head_sharded ? e.nbytes() / w : e.nbytes();
          break;
        }
        case Mistral4WeightClass::SharedExpert:
        case Mistral4WeightClass::RoutedExpert: {
          // The intermediate dimension is sliced: gate / up rows, down
          // columns — payload and block scales alike; the F32 global stays.
          const size_t n = e.role == Mistral4TensorRole::Fp4Global ? e.nbytes() : e.nbytes() / w;
          (e.cls == Mistral4WeightClass::SharedExpert ? b.shared_experts : b.routed_experts) += n;
          break;
        }
        default:
          b.routers_and_norms += e.nbytes();
      }
    }
  }
  return b;
}

void mistral4_tp_validate_geometry(const Mistral4TextConfig& cfg, int rank, int world) {
  auto fail = [](const std::string& what) {
    throw std::invalid_argument("mistral4 tp geometry: " + what);
  };
  if (world < 1 || rank < 0 || rank >= world) fail("rank/world out of range");
  if (cfg.num_attention_heads % world != 0) fail("n_heads must divide by world");
  if (cfg.vocab_size % world != 0) fail("vocab_size must divide by world");
  if (world == 1) return;
  for (const int inter : {cfg.moe_intermediate_size, cfg.shared_expert_inter()}) {
    if (inter % world != 0) fail("an expert intermediate size must divide by world");
    if ((inter / world) % 32 != 0)
      fail("an expert intermediate slice must be a multiple of 32 (the NVFP4 GEMV core's K)");
  }
  // wo's input columns are sliced by head: a rank's slice is local_heads * v wide.
  if ((cfg.num_attention_heads / world) * cfg.v_head_dim % 16 != 0)
    fail("the wo slice must be a multiple of 16");
}

}  // namespace dgpp
