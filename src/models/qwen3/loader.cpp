// UNVERIFIED DRAFT (2026-10-04): see models/qwen3/loader.hpp. Never compiled
// with nvcc / GCC, never run.
#include "models/qwen3/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "models/qwen/loader35.hpp"

namespace dgpp {
namespace {

bool ends_with(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// The replicated set (rank-invariant reads at world > 1): the layer norms,
// the router, the q/k norms, the embedding and the final norm, and every
// scalar (the NVFP4 global scales). Everything else is a slice: the
// attention projections, every expert matrix, the lm head under
// VocabSharded. The unused roles and the vision tower are never read.
bool is_replicated(const Qwen3ExpectedTensor& e) {
  if (e.role == Qwen3TensorRole::Fp4Global) return true;
  if (e.unused()) return true;
  switch (e.cls) {
    case Qwen3WeightClass::Embed:
    case Qwen3WeightClass::FinalNorm:
    case Qwen3WeightClass::LayerNorm:
    case Qwen3WeightClass::Router:
    case Qwen3WeightClass::KvScale:
    case Qwen3WeightClass::Vision:
      return true;
    case Qwen3WeightClass::Attention:
      return ends_with(e.name, "q_norm.weight") || ends_with(e.name, "k_norm.weight");
    case Qwen3WeightClass::LmHead:
    case Qwen3WeightClass::DenseMlp:
    case Qwen3WeightClass::RoutedExpert:
      return false;
  }
  return false;
}

std::pair<int, int> lm_head_slice(const Qwen3TextConfig& cfg, int rank, int world) {
  const int64_t V = cfg.vocab_size;
  const int64_t begin = V * rank / world;
  const int64_t end = V * (rank + 1) / world;
  return {static_cast<int>(begin), static_cast<int>(end - begin)};
}

std::string& resident_image_dir_storage() {
  static std::string dir;
  return dir;
}

}  // namespace

// The per-class builders (loaders/weight_build.hpp's primitives).
struct Qwen3LoaderFamily::Builder : WeightBuilder<Qwen3ExpectedTensor> {
  const Qwen3TextConfig& cfg;
  const Qwen3LocalGeometry& geo;
  Qwen3LayerResident& out;
  float* globals_ = nullptr;  // the layer's gathered expert divisors (device)
  int globals_used_ = 0;
  int globals_total_ = 0;

  Builder(const Qwen3TextConfig& cfg_, const Qwen3LocalGeometry& geo_,
          const std::vector<Qwen3ExpectedTensor>& table_,
          const std::unordered_map<std::string, const Qwen3ExpectedTensor*>& by_name_,
          LayerBump& bump_, Qwen3LayerResident& out_,
          const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<Qwen3ExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_,
                                           geo_.rank, geo_.world, "qwen3 loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const Qwen3ExpectedTensor& e) const override { return is_replicated(e); }
  bool fp4_global(const Qwen3ExpectedTensor& e) const override {
    return e.role == Qwen3TensorRole::Fp4Global;
  }

  // ---- the gathered expert divisors (GLM-4.7's slots) -------------------------
  void reserve_globals(int n) {
    globals_total_ = n;
    globals_used_ = 0;
    globals_ = static_cast<float*>(bump.alloc(static_cast<size_t>(std::max(n, 1)) * 4));
  }
  // The next slot: the device address the view points at (a real address
  // whenever the bump is not counting); `*host` its staging mirror (copy mode).
  float* next_global(float** host) {
    if (globals_used_ >= globals_total_) fail("global scale slots exhausted (builder bug)");
    const int i = globals_used_++;
    *host = copy ? bump.host(globals_) + i : nullptr;
    return globals_ ? globals_ + i : reinterpret_cast<float*>(static_cast<size_t>(i) * 4);
  }

  // One NVFP4 matrix's three tensors in the config's container, with a
  // slice checked against them: rows are free (each carries its own
  // scales), columns start and span whole 16-blocks.
  struct Fp4Set {
    const Qwen3ExpectedTensor* payload = nullptr;
    const Qwen3ExpectedTensor* scales = nullptr;
    const Qwen3ExpectedTensor* global = nullptr;
    int64_t N = 0, K = 0;
  };
  Fp4Set fp4_set(const std::string& base, int64_t r0, int64_t rn, int64_t c0, int64_t cn) const {
    Fp4Set s;
    s.payload = &expected(qwen3_fp4_payload_name(cfg, base));
    s.scales = &expected(base + ".weight_scale");
    s.global = &expected(qwen3_fp4_global_name(cfg, base));
    if (s.payload->shape.size() != 2) fail(base + ": NVFP4 matrix needs 2 dims");
    s.N = s.payload->shape[0];
    s.K = s.payload->shape[1] * 2;
    fp4_check_cols(s.K, who.c_str());
    if (s.scales->shape.size() != 2 || s.scales->shape[0] != s.N || s.scales->shape[1] != s.K / kFp4Group)
      fail("NVFP4 scale geometry mismatch on " + base);
    check_range(base, r0, rn, s.N);
    check_range(base + " cols", c0, cn, s.K);
    if (c0 % kFp4Group != 0 || cn % kFp4Group != 0)
      fail("NVFP4 column slice of " + base + " is not 16-aligned");
    return s;
  }
  // The per-tensor scale as the container stores it — modelopt's multiplier
  // or compressed-tensors' divisor (the copy pass only).
  float read_fp4_global(const Fp4Set& s) {
    const TensorInfo& t = source(s.global->name);
    float g;
    std::memcpy(&g, t.data, 4);
    if (!(g > 0.0f) || !std::isfinite(g)) fail("'" + s.global->name + "' is not a positive finite scale");
    consumed(t);
    return g;
  }

  // A routed expert's NVFP4 slice in the expert kernels' resident form: the
  // payload and block-scale bytes as shipped, packed contiguous; the
  // per-tensor scale in the next global slot as the DIVISOR the kernels
  // apply to the finished dot — compressed-tensors' as shipped, modelopt's
  // multiplier as its reciprocal (GLM-4.7's and the 80B's rule).
  GlmFp4Matrix load_fp4_expert(const std::string& base, int64_t r0, int64_t rn, int64_t c0, int64_t cn) {
    const Fp4Set s = fp4_set(base, r0, rn, c0, cn);
    const size_t pc_full = static_cast<size_t>(s.K / 2), sc_full = static_cast<size_t>(s.K / kFp4Group);
    const size_t pc = static_cast<size_t>(cn / 2), sc = static_cast<size_t>(cn / kFp4Group);
    GlmFp4Matrix q;
    q.rows = rn;
    q.cols = cn;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rn) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rn) * sc));
    float* host_global = nullptr;
    q.global_scale = next_global(&host_global);
    if (copy) {
      const TensorInfo& tp = source(s.payload->name);
      const TensorInfo& ts = source(s.scales->name);
      const uint8_t* sp = static_cast<const uint8_t*>(tp.data) + static_cast<size_t>(r0) * pc_full + c0 / 2;
      const uint8_t* ss = static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(r0) * sc_full + c0 / kFp4Group;
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      uint8_t* hs = bump.host(const_cast<uint8_t*>(q.scales));
      if (cn == s.K) {
        std::memcpy(hp, sp, static_cast<size_t>(rn) * pc);
        std::memcpy(hs, ss, static_cast<size_t>(rn) * sc);
      } else {
        for (int64_t row = 0; row < rn; ++row) {
          std::memcpy(hp + row * pc, sp + row * pc_full, pc);
          std::memcpy(hs + row * sc, ss + row * sc_full, sc);
        }
      }
      consumed(tp);
      consumed(ts);
      const float stored = read_fp4_global(s);
      *host_global = cfg.fp4_packed() ? stored : 1.0f / stored;
    }
    note_read(*s.payload, static_cast<size_t>(rn) * pc);
    note_read(*s.scales, static_cast<size_t>(rn) * sc);
    note_read(*s.global, 4);
    return q;
  }

  // An attention projection's [rn, cn] slice as the BF16 the GEMM reads:
  // sliced as shipped when the release keeps it BF16, dequantized on the
  // host when it ships NVFP4 (models/qwen/loader35.hpp's two functions: the
  // qwen3_5 stack's resident form of the same containers).
  const uint16_t* load_projection(const std::string& base, int64_t r0, int64_t rn, int64_t c0, int64_t cn,
                                  bool row_slice) {
    // BF16 as shipped: a Plain `.weight`. (modelopt names its NVFP4 CODES
    // `.weight` too — the role tells them apart, not the name.)
    if (const auto it = by_name.find(base + ".weight");
        it != by_name.end() && it->second->role == Qwen3TensorRole::Plain) {
      if (it->second->dtype != DType::BF16) fail("'" + base + ".weight' is not BF16");
      return row_slice ? load_bf16_rows(base + ".weight", r0, rn) : load_bf16_cols(base + ".weight", c0, cn);
    }
    const Fp4Set s = fp4_set(base, r0, rn, c0, cn);
    const size_t pc_full = static_cast<size_t>(s.K / 2), sc_full = static_cast<size_t>(s.K / kFp4Group);
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(rn) * static_cast<size_t>(cn) * 2));
    if (copy) {
      const TensorInfo& tp = source(s.payload->name);
      const TensorInfo& ts = source(s.scales->name);
      const uint8_t* codes = static_cast<const uint8_t*>(tp.data) + static_cast<size_t>(r0) * pc_full + c0 / 2;
      const uint8_t* block_scales =
          static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(r0) * sc_full + c0 / kFp4Group;
      const float global = read_fp4_global(s);
      if (cfg.fp4_packed())
        qwen35_fp4_packed_dequant_bf16(codes, pc_full, block_scales, sc_full, global, rn, cn, bump.host(dst));
      else
        qwen3next_fp4_dequant_bf16(codes, pc_full, block_scales, sc_full, global, rn, cn, bump.host(dst));
      consumed(tp);
      consumed(ts);
    }
    note_read(*s.payload, static_cast<size_t>(rn) * static_cast<size_t>(cn / 2));
    note_read(*s.scales, static_cast<size_t>(rn) * static_cast<size_t>(cn / kFp4Group));
    note_read(*s.global, 4);
    return dst;
  }

  // ---- the classes ---------------------------------------------------------
  void build_attention(const std::string& p) {
    const int64_t d = cfg.head_dim, H = cfg.hidden_size;
    Glm4AttnResident& a = out.attn;
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_kv_heads = geo.local_kv_heads;
    a.kv_head_begin = geo.kv_head_begin;
    const int64_t q0 = static_cast<int64_t>(geo.head_begin) * d, qn = static_cast<int64_t>(geo.local_heads) * d;
    const int64_t k0 = static_cast<int64_t>(geo.kv_head_begin) * d, kn = static_cast<int64_t>(geo.local_kv_heads) * d;
    a.q_proj = load_projection(p + "q_proj", q0, qn, 0, H, /*row_slice=*/true);
    a.k_proj = load_projection(p + "k_proj", k0, kn, 0, H, true);
    a.v_proj = load_projection(p + "v_proj", k0, kn, 0, H, true);
    a.o_proj = load_projection(p + "o_proj", 0, H, q0, qn, /*row_slice=*/false);
    a.q_bias = a.k_bias = a.v_bias = nullptr;  // attention_bias false, by the config parser
    a.q_norm = load_bf16(p + "q_norm.weight");
    a.k_norm = load_bf16(p + "k_norm.weight");
  }

  void build_moe(const std::string& p) {
    Qwen3MoeResident& m = out.moe;
    m.router = load_bf16(p + "gate.weight");
    const int64_t I = geo.local_inter, r = rank, H = cfg.hidden_size;
    m.local_inter = I;
    const int E = cfg.num_experts;
    m.experts.resize(static_cast<size_t>(E) * 3);
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      GlmFp4Matrix* t = m.experts.data() + static_cast<size_t>(e) * 3;
      t[0] = load_fp4_expert(ep + "gate_proj", r * I, I, 0, H);
      t[1] = load_fp4_expert(ep + "up_proj", r * I, I, 0, H);
      t[2] = load_fp4_expert(ep + "down_proj", 0, H, r * I, I);
    }
  }

  void build_layer(int layer) {
    if (layer < 0 || layer >= cfg.num_hidden_layers) fail("layer index out of range: " + std::to_string(layer));
    const std::string p = qwen3_layer_prefix(cfg, layer);
    out.layer = layer;
    // The expert divisors first: one slot per NVFP4 expert matrix.
    reserve_globals(3 * cfg.num_experts);
    out.input_norm = load_bf16(p + "input_layernorm.weight");
    out.post_norm = load_bf16(p + "post_attention_layernorm.weight");
    build_attention(p + "self_attn.");
    build_moe(p + "mlp.");
    if (globals_used_ != globals_total_) fail("global scale slots left unused (builder bug)");
  }
};

// ---------------------------------------------------------------------------

Qwen3LocalGeometry Qwen3LocalGeometry::from_config(const Qwen3TextConfig& cfg, int rank, int world,
                                                   LoaderHeadSharding head) {
  if (!cfg.moe())
    throw std::invalid_argument(
        "qwen3 loader: the dense dialect (Qwen3ForCausalLM: the retrieval models) is bound and has a host "
        "reference, but no GPU path — it is not served");
  if (world < 1 || rank < 0 || rank >= world) throw std::invalid_argument("qwen3 loader: rank/world out of range");
  if (world > 1) qwen3_tp_validate_geometry(cfg, rank, world);
  Qwen3LocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  if (cfg.num_key_value_heads >= world) {
    g.local_kv_heads = cfg.num_key_value_heads / world;
    g.kv_head_begin = g.local_kv_heads * rank;
  } else {
    g.local_kv_heads = 1;
    g.kv_head_begin = rank / (world / cfg.num_key_value_heads);
  }
  g.local_inter = cfg.moe_intermediate_size / world;
  if (head == LoaderHeadSharding::VocabSharded) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.lm_vocab_begin = b;
    g.lm_vocab_count = n;
  } else {
    g.lm_vocab_begin = 0;
    g.lm_vocab_count = cfg.vocab_size;
  }
  return g;
}

// ---- the family hooks (loaders/resident_stream.hpp) -------------------------

// The WHOLE header against the table: the vision tower's tensors are part
// of it (a VL checkpoint without its tower, or with another tower, does not
// load), though the stream never reads them.
void Qwen3LoaderFamily::validate_binding(const Qwen3TextConfig& cfg, const PresentMap& present) {
  const Qwen3BindReport rep = qwen3_validate_binding(cfg, present);
  if (rep.ok()) {
    if (rep.vision != 0)
      DGPP_LOG_INFO("qwen3 loader: {} vision-tower tensors bound and left on disk — this engine serves the "
                    "checkpoint's TEXT path only (image inputs are refused)",
                    rep.vision);
    return;
  }
  std::string msg = "qwen3 loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

bool Qwen3LoaderFamily::digest_included(const Qwen3ExpectedTensor& e) {
  return is_replicated(e) && !e.unused();
}

// No projection is packed after the builder returns: an NVFP4 one is
// dequantized inside it, and a BF16 o_proj column slice (a release that
// ships o_proj in BF16 — none of the two bound) would read its source in
// phase one, so the attention class keeps its sources until then.
bool Qwen3LoaderFamily::discard_after_pack(const Qwen3ExpectedTensor& e) {
  return e.cls == Qwen3WeightClass::Attention && e.role == Qwen3TensorRole::Plain;
}

size_t Qwen3LoaderFamily::globals_bytes(const Qwen3TextConfig& cfg, int rank, int world,
                                        LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(cfg.vocab_size) * H * 2);  // embed
  b += align_up_256(H * 2);                                        // final norm
  b += align_up_256(static_cast<size_t>(Qwen3LocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count) * H * 2);
  return b;
}

void Qwen3LoaderFamily::build_globals(const Qwen3TextConfig& cfg, const Qwen3LocalGeometry& geo,
                                      const LoaderTensorMap& tensors, LayerBump& bump,
                                      Qwen3GlobalsResident& out, uint64_t& source_bytes,
                                      uint64_t& verbatim_bytes, LoaderHeadSharding head) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("qwen3 loader: global tensor missing: " + name);
    return *it->second;
  };
  auto copy_global = [&](const std::string& name) -> uint16_t* {
    const TensorInfo& t = lookup(name);
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(t.nbytes()));
    std::memcpy(bump.host(dst), t.data, t.nbytes());
    source_bytes += t.nbytes();
    verbatim_bytes += t.nbytes();
    return dst;
  };
  const std::string p = qwen3_model_prefix(cfg);
  out.embed = copy_global(p + "embed_tokens.weight");
  out.final_norm = copy_global(p + "norm.weight");
  // The MoE dialects keep an untied head (the config parser refuses a tied one).
  const TensorInfo& t = lookup("lm_head.weight");
  const size_t row_bytes = static_cast<size_t>(cfg.hidden_size) * 2;
  const int begin = geo.lm_vocab_begin, count = geo.lm_vocab_count;
  uint16_t* dst = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(count) * row_bytes));
  std::memcpy(bump.host(dst), static_cast<const uint8_t*>(t.data) + static_cast<size_t>(begin) * row_bytes,
              static_cast<size_t>(count) * row_bytes);
  source_bytes += static_cast<size_t>(count) * row_bytes;
  if (head == LoaderHeadSharding::Full) verbatim_bytes += static_cast<size_t>(count) * row_bytes;
  out.lm_head = dst;
  out.lm_vocab_begin = begin;
  out.lm_vocab_count = count;
}

template class ResidentLayerStream<Qwen3LoaderFamily>;

// ---------------------------------------------------------------------------

Qwen3LayerStream::Qwen3LayerStream(const Qwen3TextConfig& cfg, const std::string& checkpoint_dir,
                                   int rank, int world, LoaderResidency residency,
                                   LoaderHeadSharding head)
    : ResidentLayerStream<Qwen3LoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                             /*resident_mtp=*/false) {
  open_resident_image();
}

void Qwen3LayerStream::set_resident_image_dir(const std::string& dir) {
  resident_image_dir_storage() = dir;
}
const std::string& Qwen3LayerStream::resident_image_dir() { return resident_image_dir_storage(); }

}  // namespace dgpp
