#include "models/dsv4/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {
namespace {

bool contains(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

// One e8m0 scale byte as fp32: 2^(byte - 127); 255 is NaN.
float e8m0_to_float(uint8_t b) {
  if (b == 255) return std::nanf("");
  return std::ldexp(1.0f, static_cast<int>(b) - 127);
}

// The replicated set (rank-invariant reads at world > 1): the norms, the
// routers with their biases and hash tables, the indexers, the
// compressors, the mHC coefficients, the embedding, the final norm, wq_a
// and wkv (every rank's latent), the draft's head tensors. Everything else
// is a slice: wq_b, wo_a, wo_b, the sink, every expert, the lm head.
bool is_replicated(const Dsv4ExpectedTensor& e) {
  switch (e.cls) {
    case Dsv4WeightClass::Embed:
    case Dsv4WeightClass::FinalNorm:
    case Dsv4WeightClass::LayerNorm:
    case Dsv4WeightClass::Indexer:
    case Dsv4WeightClass::Compressor:
    case Dsv4WeightClass::Router:
    case Dsv4WeightClass::Mhc:
    case Dsv4WeightClass::Draft:
      return true;
    case Dsv4WeightClass::Attention:
      return contains(e.name, ".wq_a.") || contains(e.name, ".wkv.");
    case Dsv4WeightClass::LmHead:
    case Dsv4WeightClass::SharedExpert:
    case Dsv4WeightClass::RoutedExpert:
      return false;
  }
  return false;
}

}  // namespace

// The per-class builders (loaders/weight_build.hpp's primitives).
struct Dsv4LoaderFamily::Builder : WeightBuilder<Dsv4ExpectedTensor> {
  const Dsv4Config& cfg;
  const Dsv4LocalGeometry& geo;
  Dsv4LayerResident& out;

  Builder(const Dsv4Config& cfg_, const Dsv4LocalGeometry& geo_, const std::vector<Dsv4ExpectedTensor>& table_,
          const std::unordered_map<std::string, const Dsv4ExpectedTensor*>& by_name_, LayerBump& bump_,
          Dsv4LayerResident& out_, const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<Dsv4ExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_, geo_.rank,
                                          geo_.world, "dsv4 loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const Dsv4ExpectedTensor& e) const override { return is_replicated(e); }

  // ---- fp8 pairs on the 128 x 128 grid (e8m0 -> fp32 scales) ---------------
  struct Fp8Source {
    const Dsv4ExpectedTensor* w;
    const Dsv4ExpectedTensor* s;
    int64_t N, K, SR, SC;
  };
  Fp8Source fp8_source(const std::string& base) {
    Fp8Source f;
    f.w = &expected(base + ".weight");
    f.s = &expected(base + ".scale");
    if (f.w->role != Dsv4TensorRole::Fp8Payload || f.s->role != Dsv4TensorRole::Fp8Scale || f.w->shape.size() != 2 ||
        f.s->shape.size() != 2)
      fail("'" + base + "' is not an fp8 pair");
    f.N = f.w->shape[0];
    f.K = f.w->shape[1];
    f.SR = f.s->shape[0];
    f.SC = f.s->shape[1];
    const int b = cfg.fp8_block_size;
    if (f.SR != (f.N + b - 1) / b || f.SC != (f.K + b - 1) / b) fail("fp8 scale geometry mismatch on " + base);
    return f;
  }
  GlmQuantMatrix alloc_fp8(int64_t rows, int64_t cols) {
    const int b = cfg.fp8_block_size;
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.scale_block_rows = b;
    q.scale_block_cols = b;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols)));
    q.scales = static_cast<const float*>(
        bump.alloc(static_cast<size_t>((rows + b - 1) / b) * static_cast<size_t>((cols + b - 1) / b) * 4));
    return q;
  }
  // Scale entries [col0, col0 + count) of source scale row `src_row` into `dst`.
  void convert_scales(const TensorInfo& ts, const Fp8Source& f, int64_t src_row, int64_t col0, int64_t count,
                      float* dst) const {
    const uint8_t* s = static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(src_row) * f.SC + col0;
    for (int64_t i = 0; i < count; ++i) dst[i] = e8m0_to_float(s[i]);
  }
  // Rows [row_start, +rows): row_start on a block boundary (every row's
  // scale row is then its own block row).
  GlmQuantMatrix load_fp8_rows(const std::string& base, int64_t row_start, int64_t rows) {
    const Fp8Source f = fp8_source(base);
    const int b = cfg.fp8_block_size;
    if (row_start % b != 0) fail("fp8 row slice of '" + base + "' must start on a scale-block boundary");
    check_range(base, row_start, rows, f.N);
    GlmQuantMatrix q = alloc_fp8(rows, f.K);
    const int64_t sr = (rows + b - 1) / b;
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(tw.data) + static_cast<size_t>(row_start) * f.K,
                  static_cast<size_t>(rows) * f.K);
      float* hs = bump.host(const_cast<float*>(q.scales));
      for (int64_t r = 0; r < sr; ++r) convert_scales(ts, f, row_start / b + r, 0, f.SC, hs + r * f.SC);
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(rows) * f.K);
    note_read(*f.s, static_cast<size_t>(sr) * f.SC);
    return q;
  }
  GlmQuantMatrix load_fp8(const std::string& base) {
    const Fp8Source f = fp8_source(base);
    return load_fp8_rows(base, 0, f.N);
  }
  // Columns [col_start, +cols) of every row, packed: whole scale blocks.
  GlmQuantMatrix load_fp8_cols(const std::string& base, int64_t col_start, int64_t cols) {
    const Fp8Source f = fp8_source(base);
    const int b = cfg.fp8_block_size;
    if (col_start % b != 0 || cols % b != 0 || cols <= 0)
      fail("fp8 column slice of '" + base + "' must be whole scale blocks");
    check_range(base, col_start, cols, f.K);
    GlmQuantMatrix q = alloc_fp8(f.N, cols);
    const int64_t sc = cols / b;
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      const uint8_t* sp = static_cast<const uint8_t*>(tw.data);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      float* hs = bump.host(const_cast<float*>(q.scales));
      if (col_start == 0 && cols == f.K) {
        std::memcpy(hp, sp, static_cast<size_t>(f.N) * f.K);
      } else {
        for (int64_t r = 0; r < f.N; ++r)
          std::memcpy(hp + r * cols, sp + r * f.K + col_start, static_cast<size_t>(cols));
      }
      for (int64_t r = 0; r < f.SR; ++r) convert_scales(ts, f, r, col_start / b, sc, hs + r * sc);
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(f.N) * static_cast<size_t>(cols));
    note_read(*f.s, static_cast<size_t>(f.SR) * static_cast<size_t>(sc));
    return q;
  }

  // ---- MXFP4 pairs (e2m1 pairs + e8m0 per 32) ------------------------------
  struct Fp4Source {
    const Dsv4ExpectedTensor* w;
    const Dsv4ExpectedTensor* s;
    int64_t N, K;
  };
  Fp4Source fp4_source(const std::string& base) {
    Fp4Source f;
    f.w = &expected(base + ".weight");
    f.s = &expected(base + ".scale");
    if (f.w->role != Dsv4TensorRole::Fp4Payload || f.s->role != Dsv4TensorRole::Fp4Scale || f.w->shape.size() != 2 ||
        f.s->shape.size() != 2)
      fail("'" + base + "' is not an MXFP4 pair");
    f.N = f.w->shape[0];
    f.K = f.w->shape[1] * 2;
    fp4_check_cols(f.K, who.c_str(), kMxfp4Group);
    if (f.s->shape[0] != f.N || f.s->shape[1] != f.K / kMxfp4Group) fail("MXFP4 scale geometry mismatch on " + base);
    return f;
  }
  GlmFp4Matrix alloc_mxfp4(int64_t rows, int64_t cols) {
    GlmFp4Matrix q;
    q.rows = rows;
    q.cols = cols;
    q.scale_group = kMxfp4Group;
    q.global_scale = nullptr;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols / 2)));
    q.scales =
        static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols / kMxfp4Group)));
    return q;
  }
  GlmFp4Matrix load_mxfp4_rows(const std::string& base, int64_t row_start, int64_t rows) {
    const Fp4Source f = fp4_source(base);
    check_range(base, row_start, rows, f.N);
    const size_t pc = static_cast<size_t>(f.K / 2), sc = static_cast<size_t>(f.K / kMxfp4Group);
    GlmFp4Matrix q = alloc_mxfp4(rows, f.K);
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(tw.data) + static_cast<size_t>(row_start) * pc,
                  static_cast<size_t>(rows) * pc);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.scales)),
                  static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(row_start) * sc,
                  static_cast<size_t>(rows) * sc);
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(rows) * pc);
    note_read(*f.s, static_cast<size_t>(rows) * sc);
    return q;
  }
  GlmFp4Matrix load_mxfp4_cols(const std::string& base, int64_t col_start, int64_t cols) {
    const Fp4Source f = fp4_source(base);
    fp4_check_cols(cols, who.c_str(), kMxfp4Group);
    if (col_start % kMxfp4Group != 0)
      fail("MXFP4 column slice of '" + base + "' must start on a 32-element block boundary");
    check_range(base, col_start, cols, f.K);
    const size_t pc = static_cast<size_t>(cols / 2), pc_full = static_cast<size_t>(f.K / 2);
    const size_t sc = static_cast<size_t>(cols / kMxfp4Group), sc_full = static_cast<size_t>(f.K / kMxfp4Group);
    GlmFp4Matrix q = alloc_mxfp4(f.N, cols);
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      const uint8_t* sp = static_cast<const uint8_t*>(tw.data);
      const uint8_t* ss = static_cast<const uint8_t*>(ts.data);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      uint8_t* hs = bump.host(const_cast<uint8_t*>(q.scales));
      if (cols == f.K) {
        std::memcpy(hp, sp, static_cast<size_t>(f.N) * pc);
        std::memcpy(hs, ss, static_cast<size_t>(f.N) * sc);
      } else {
        for (int64_t r = 0; r < f.N; ++r) {
          std::memcpy(hp + r * pc, sp + r * pc_full + col_start / 2, pc);
          std::memcpy(hs + r * sc, ss + r * sc_full + col_start / kMxfp4Group, sc);
        }
      }
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(f.N) * pc);
    note_read(*f.s, static_cast<size_t>(f.N) * sc);
    return q;
  }

  // An F32 tensor rounded to bf16 at load (the mHC coefficient matrices:
  // RNE, the mHC kernels' bf16 form).
  uint16_t* load_f32_as_bf16(const std::string& name) {
    const Dsv4ExpectedTensor& e = expected(name);
    if (e.dtype != DType::F32) fail("'" + name + "' is not F32");
    const size_t n = e.numel();
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(n * 2));
    if (copy) {
      const TensorInfo& t = source(name);
      const uint8_t* src = static_cast<const uint8_t*>(t.data);
      uint16_t* h = bump.host(dst);
      for (size_t i = 0; i < n; ++i) {
        float v;
        std::memcpy(&v, src + i * 4, 4);
        h[i] = float_to_bf16_bits(v);
      }
      consumed(t);
    }
    note_read(e, n * 4);
    return dst;
  }

  // The head's collapse padded to the mHC kernel's coefficient rows (see
  // Dsv4HcHeadResident): fn rounded to bf16, the rows past hc_mult zero.
  Dsv4HcHeadResident load_hc_head(const std::string& p) {
    const Dsv4ExpectedTensor& ef = expected(p + "hc_head_fn");
    const Dsv4ExpectedTensor& eb = expected(p + "hc_head_base");
    const Dsv4ExpectedTensor& es = expected(p + "hc_head_scale");
    const size_t n = static_cast<size_t>(cfg.hc_mult), rows = static_cast<size_t>(cfg.hc_coeff_rows());
    const size_t width = n * static_cast<size_t>(cfg.hidden_size);
    if (ef.numel() != n * width || eb.numel() != n || es.numel() != 1) fail("'" + p + "hc_head_*' geometry");
    Dsv4HcHeadResident h;
    uint16_t* fn = static_cast<uint16_t*>(bump.alloc(rows * width * 2));
    float* base = static_cast<float*>(bump.alloc(rows * 4));
    float* scale = static_cast<float*>(bump.alloc(3 * 4));
    if (copy) {
      const TensorInfo& tf = source(ef.name);
      const TensorInfo& tb = source(eb.name);
      const TensorInfo& ts = source(es.name);
      uint16_t* hf = bump.host(fn);
      std::memset(hf, 0, rows * width * 2);
      const uint8_t* src = static_cast<const uint8_t*>(tf.data);
      for (size_t i = 0; i < n * width; ++i) {
        float v;
        std::memcpy(&v, src + i * 4, 4);
        hf[i] = float_to_bf16_bits(v);
      }
      float* hb = bump.host(base);
      std::memset(hb, 0, rows * 4);
      std::memcpy(hb, tb.data, n * 4);
      float* hs = bump.host(scale);
      std::memset(hs, 0, 12);
      std::memcpy(hs, ts.data, 4);
      consumed(tf);
      consumed(tb);
      consumed(ts);
    }
    note_read(ef, ef.nbytes());
    note_read(eb, eb.nbytes());
    note_read(es, es.nbytes());
    h.fn = fn;
    h.base = base;
    h.scale = scale;
    return h;
  }

  // The hash layers' token -> expert table: I64 [vocab, top_k] narrowed to
  // int32 with every row sorted ascending (the reference's order is free:
  // the picked scores are normalized over the row and the chain accumulates
  // in ascending expert order, as every routed layer's does).
  int32_t* load_tid2eid(const std::string& name) {
    const Dsv4ExpectedTensor& e = expected(name);
    if (e.dtype != DType::I64 || e.shape.size() != 2) fail("'" + name + "' is not an I64 table");
    const size_t rows = static_cast<size_t>(e.shape[0]), k = static_cast<size_t>(e.shape[1]);
    int32_t* dst = static_cast<int32_t*>(bump.alloc(rows * k * 4));
    if (copy) {
      const TensorInfo& t = source(name);
      const uint8_t* src = static_cast<const uint8_t*>(t.data);
      int32_t* h = bump.host(dst);
      for (size_t r = 0; r < rows; ++r) {
        int32_t* row = h + r * k;
        for (size_t j = 0; j < k; ++j) {
          int64_t v;
          std::memcpy(&v, src + (r * k + j) * 8, 8);
          if (v < 0 || v >= cfg.n_routed_experts)
            fail("'" + name + "' names an expert outside [0, n_routed_experts)");
          row[j] = static_cast<int32_t>(v);
        }
        std::sort(row, row + k);
        for (size_t j = 1; j < k; ++j)
          if (row[j] == row[j - 1]) fail("'" + name + "' repeats an expert within a token's row");
      }
      consumed(t);
    }
    note_read(e, rows * k * 8);
    return dst;
  }

  // ---- the classes ---------------------------------------------------------
  Dsv4CompressorResident build_compressor(const std::string& cp, int layer, int dim) {
    Dsv4CompressorResident c;
    c.ratio = cfg.compress_ratio(layer);
    c.dim = dim;
    c.wide = (cfg.overlapped(layer) ? 2 : 1) * dim;
    c.ape = load_f32(cp + "ape");
    c.wkv = load_bf16(cp + "wkv.weight");
    c.wgate = load_bf16(cp + "wgate.weight");
    c.norm = load_bf16(cp + "norm.weight");
    return c;
  }

  void build_attention(const std::string& p, int layer) {
    Dsv4AttnResident& a = out.attn;
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_groups = geo.local_groups;
    a.group_begin = geo.group_begin;
    const int64_t hd = cfg.head_dim, ol = cfg.o_lora_rank;
    a.q_norm = load_bf16(p + "q_norm.weight");
    a.kv_norm = load_bf16(p + "kv_norm.weight");
    a.attn_sink = load_f32_range(p + "attn_sink", geo.head_begin, geo.local_heads);
    a.wq_a = load_fp8(p + "wq_a");
    a.wkv = load_fp8(p + "wkv");
    a.wq_b = load_fp8_rows(p + "wq_b", static_cast<int64_t>(geo.head_begin) * hd,
                           static_cast<int64_t>(geo.local_heads) * hd);
    a.wo_a = load_fp8_rows(p + "wo_a", static_cast<int64_t>(geo.group_begin) * ol,
                           static_cast<int64_t>(geo.local_groups) * ol);
    a.wo_b = load_fp8_cols(p + "wo_b", static_cast<int64_t>(geo.group_begin) * ol,
                           static_cast<int64_t>(geo.local_groups) * ol);
    if (cfg.compress_ratio(layer) > 0) a.comp = build_compressor(p + "compressor.", layer, cfg.head_dim);
    if (cfg.indexed(layer)) {
      const std::string ip = p + "indexer.";
      a.idx_wq_b = load_fp8(ip + "wq_b");
      a.idx_wp = load_bf16(ip + "weights_proj.weight");
      a.idx_comp = build_compressor(ip + "compressor.", layer, cfg.index_head_dim);
    }
  }

  void build_moe(const std::string& p, int layer) {
    Dsv4MoeResident& m = out.moe;
    m.router = load_bf16(p + "gate.weight");
    if (cfg.hashed(layer)) m.tid2eid = load_tid2eid(p + "gate.tid2eid");
    else m.router_bias = load_f32(p + "gate.bias");
    const int64_t I = geo.local_inter, S = geo.local_shared_inter, r = rank;
    m.local_inter = I;
    m.local_shared_inter = S;
    const int E = cfg.n_routed_experts;
    m.n_experts = E;
    m.experts.resize(static_cast<size_t>(E) * 3);
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      GlmFp4Matrix* t = m.experts.data() + static_cast<size_t>(e) * 3;
      t[0] = load_mxfp4_rows(ep + "w1", r * I, I);  // gate
      t[1] = load_mxfp4_rows(ep + "w3", r * I, I);  // up
      t[2] = load_mxfp4_cols(ep + "w2", r * I, I);  // down
    }
    const std::string sp = p + "shared_experts.";
    m.shared[0] = load_fp8_rows(sp + "w1", r * S, S);
    m.shared[1] = load_fp8_rows(sp + "w3", r * S, S);
    m.shared[2] = load_fp8_cols(sp + "w2", r * S, S);
  }

  void build_mhc(const std::string& p) {
    Dsv4MhcResident& h = out.mhc;
    h.attn_fn = load_f32_as_bf16(p + "hc_attn_fn");
    h.attn_base = load_f32(p + "hc_attn_base");
    h.attn_scale = load_f32(p + "hc_attn_scale");
    h.ffn_fn = load_f32_as_bf16(p + "hc_ffn_fn");
    h.ffn_base = load_f32(p + "hc_ffn_base");
    h.ffn_scale = load_f32(p + "hc_ffn_scale");
  }

  void build_draft(const std::string& p, int layer) {
    Dsv4DraftResident& d = out.draft;
    const int stage = cfg.draft_stage(layer);
    if (stage == 0) {
      d.main_proj = load_fp8(p + "main_proj");
      d.main_norm = load_bf16(p + "main_norm.weight");
    }
    if (stage == cfg.draft_stages - 1) {
      d.norm = load_bf16(p + "norm.weight");
      d.markov_embed = load_bf16(p + "markov_head.markov_w1.weight");
      d.markov_head = load_bf16(p + "markov_head.markov_w2.weight");
      d.confidence = load_bf16_as_f32(p + "confidence_head.proj.weight");
      d.hc_head = load_hc_head(p);
    }
  }

  void build_layer(int layer) {
    if (layer < 0 || layer >= cfg.max_layer()) fail("layer index out of range: " + std::to_string(layer));
    const std::string p = dsv4_layer_prefix(cfg, layer);
    out.layer = layer;
    out.attn_norm = load_bf16(p + "attn_norm.weight");
    out.ffn_norm = load_bf16(p + "ffn_norm.weight");
    build_mhc(p);
    build_attention(p + "attn.", layer);
    build_moe(p + "ffn.", layer);
    if (cfg.is_draft(layer)) build_draft(p, layer);
  }
};

namespace {

std::pair<int, int> lm_head_slice(const Dsv4Config& cfg, int rank, int world) {
  const int64_t V = cfg.vocab_size;
  const int64_t begin = V * rank / world;
  const int64_t end = V * (rank + 1) / world;
  return {static_cast<int>(begin), static_cast<int>(end - begin)};
}

std::string& resident_image_dir_storage() {
  static std::string dir;
  return dir;
}

bool g_embed_vocab_sharded = false;

}  // namespace

// ---------------------------------------------------------------------------

Dsv4LocalGeometry Dsv4LocalGeometry::from_config(const Dsv4Config& cfg, int rank, int world, Dsv4HeadSharding head) {
  if (world > 1) dsv4_tp_validate_geometry(cfg, rank, world);
  if (world < 1 || rank < 0 || rank >= world) throw std::invalid_argument("dsv4 loader: rank/world out of range");
  Dsv4LocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  g.local_groups = cfg.o_groups / world;
  g.group_begin = g.local_groups * rank;
  g.local_inter = cfg.moe_intermediate_size / world;
  g.local_shared_inter = cfg.shared_expert_inter() / world;
  if (head == Dsv4HeadSharding::VocabSharded) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.lm_vocab_begin = b;
    g.lm_vocab_count = n;
  } else {
    g.lm_vocab_begin = 0;
    g.lm_vocab_count = cfg.vocab_size;
  }
  if (world > 1 && g_embed_vocab_sharded) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.embed_vocab_begin = b;
    g.embed_vocab_count = n;
  } else {
    g.embed_vocab_begin = 0;
    g.embed_vocab_count = cfg.vocab_size;
  }
  return g;
}

// ---- the family hooks (loaders/resident_stream.hpp) -------------------------

void Dsv4LoaderFamily::validate_binding(const Dsv4Config& cfg, const PresentMap& present) {
  const Dsv4BindReport rep = dsv4_validate_binding(cfg, present);
  if (rep.ok()) return;
  std::string msg = "dsv4 loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

// The digest covers what a rank reads verbatim: the replicated set.
bool Dsv4LoaderFamily::digest_included(const Dsv4ExpectedTensor& e) { return is_replicated(e); }

size_t Dsv4LoaderFamily::globals_bytes(const Dsv4Config& cfg, int rank, int world, LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t n = static_cast<size_t>(cfg.hc_mult);
  const Dsv4LocalGeometry geo = Dsv4LocalGeometry::from_config(cfg, rank, world, head);
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(geo.embed_vocab_count) * H * 2);
  b += align_up_256(H * 2);
  b += align_up_256(static_cast<size_t>(geo.lm_vocab_count) * H * 2);
  b += align_up_256(static_cast<size_t>(cfg.hc_coeff_rows()) * n * H * 2) +
       align_up_256(static_cast<size_t>(cfg.hc_coeff_rows()) * 4) + align_up_256(12);
  return b;
}

void Dsv4LoaderFamily::build_globals(const Dsv4Config& cfg, const Dsv4LocalGeometry& geo,
                                     const LoaderTensorMap& tensors, LayerBump& bump, Dsv4GlobalsResident& out,
                                     uint64_t& source_bytes, uint64_t& verbatim_bytes, LoaderHeadSharding head) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second) throw std::runtime_error("dsv4 loader: global tensor missing: " + name);
    return *it->second;
  };
  const size_t row_bytes = static_cast<size_t>(cfg.hidden_size) * 2;
  {
    const TensorInfo& e = lookup("embed.weight");
    const int ebegin = geo.embed_vocab_begin, ecount = geo.embed_vocab_count;
    if (ebegin < 0 || ecount <= 0 || static_cast<size_t>(ebegin + ecount) * row_bytes > e.nbytes())
      throw std::runtime_error("dsv4 loader: the embedding slice does not fit the table");
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(ecount) * row_bytes));
    std::memcpy(bump.host(dst), static_cast<const uint8_t*>(e.data) + static_cast<size_t>(ebegin) * row_bytes,
                static_cast<size_t>(ecount) * row_bytes);
    source_bytes += static_cast<size_t>(ecount) * row_bytes;
    if (ecount == cfg.vocab_size) verbatim_bytes += e.nbytes();
    out.embed = dst;
    out.embed_vocab_begin = ebegin;
    out.embed_vocab_count = ecount;
  }
  {
    const TensorInfo& t = lookup("norm.weight");
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(t.nbytes()));
    std::memcpy(bump.host(dst), t.data, t.nbytes());
    source_bytes += t.nbytes();
    verbatim_bytes += t.nbytes();
    out.final_norm = dst;
  }
  {
    const TensorInfo& t = lookup("head.weight");
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
  {
    // The head's collapse in the mHC kernel's shape (Dsv4HcHeadResident): fn
    // rounded to bf16 with the post / comb rows zero, the base and the
    // scale padded likewise.
    const TensorInfo& fn = lookup("hc_head_fn");
    const TensorInfo& tb = lookup("hc_head_base");
    const TensorInfo& ts = lookup("hc_head_scale");
    const size_t n = static_cast<size_t>(cfg.hc_mult), rows = static_cast<size_t>(cfg.hc_coeff_rows());
    const size_t width = n * static_cast<size_t>(cfg.hidden_size);
    if (fn.nbytes() != n * width * 4 || tb.nbytes() != n * 4 || ts.nbytes() != 4)
      throw std::runtime_error("dsv4 loader: hc_head_* geometry");
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(rows * width * 2));
    uint16_t* h = bump.host(dst);
    std::memset(h, 0, rows * width * 2);
    const uint8_t* src = static_cast<const uint8_t*>(fn.data);
    for (size_t i = 0; i < n * width; ++i) {
      float v;
      std::memcpy(&v, src + i * 4, 4);
      h[i] = float_to_bf16_bits(v);
    }
    float* base = static_cast<float*>(bump.alloc(rows * 4));
    std::memset(bump.host(base), 0, rows * 4);
    std::memcpy(bump.host(base), tb.data, n * 4);
    float* scale = static_cast<float*>(bump.alloc(12));
    std::memset(bump.host(scale), 0, 12);
    std::memcpy(bump.host(scale), ts.data, 4);
    for (const TensorInfo* t : {&fn, &tb, &ts}) {
      source_bytes += t->nbytes();
      verbatim_bytes += t->nbytes();
    }
    out.hc_head.fn = dst;
    out.hc_head.base = base;
    out.hc_head.scale = scale;
  }
}

template class ResidentLayerStream<Dsv4LoaderFamily>;

// ---------------------------------------------------------------------------

void Dsv4LayerStream::set_embed_vocab_sharded(bool on) { g_embed_vocab_sharded = on; }
bool Dsv4LayerStream::embed_vocab_sharded() { return g_embed_vocab_sharded; }

Dsv4LayerStream::Dsv4LayerStream(const Dsv4Config& cfg, const std::string& checkpoint_dir, int rank, int world,
                                 Dsv4Residency residency, Dsv4HeadSharding head, bool resident_mtp)
    : ResidentLayerStream<Dsv4LoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head, resident_mtp) {
  open_resident_image();
}

void Dsv4LayerStream::set_resident_image_dir(const std::string& dir) { resident_image_dir_storage() = dir; }
const std::string& Dsv4LayerStream::resident_image_dir() { return resident_image_dir_storage(); }

}  // namespace dgpp
