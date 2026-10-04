// Qwen3.5-27B loader family: FP8-native weights (the checkpoint already holds
// e4m3 payload + BF16 scales — memcpy + BF16→F32 widen, never a BF16→FP8
// re-encode), text-only, dense SwiGLU MLP, standard pre-norm residual.
//
// The same dialect with the routed MoE (Qwen3.6-35B-A3B) binds the Qwen3Next
// dialect's MoE resident: from the FP8 release every matrix as shipped; from
// the NVFP4 mixed release the per-tensor FP8 projections as shipped under a
// uniform scale grid, the modelopt NVFP4 experts, shared expert and head,
// and a BF16 draft layer whose stacked experts are encoded to block FP8.
//
// The Qwen3Next dialect builds the same residents in their BF16 forms from
// the modelopt NVFP4 release: gathered GDN projections, NVFP4 output
// projections and shared expert dequantized on the host, the routed experts
// as Flash-Next's loader holds them (models/qwen/loader.cpp). The same
// dialect's compressed-tensors container (Qwen3-Coder-Next) differs in the
// NVFP4 set's names, in its per-tensor scale (a divisor) and in which
// matrices carry the set: the attention q/k/v as well, the GDN's out_proj not.
#include "models/qwen/loader35.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"

namespace dgpp {
namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
  const size_t n = suffix.size();
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// Per-layer build progress logs, off by default (the sibling qwen loader
// prints nothing). Set DGPP_QWEN35_LOADER_VERBOSE to a non-'0' value to
// trace layer construction at boot.
bool loader_verbose() {
  static const bool v = [] {
    const char* e = std::getenv("DGPP_QWEN35_LOADER_VERBOSE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return v;
}

// Rank-invariant reads at world > 1 (mirrors the qwen loader's rule):
// norms replicate, projections slice. The MTP draft head is BF16 and
// replicated whole (like the norms). The Qwen3Next dialect adds the router
// and the shared gate, the two F32 scalars beside every NVFP4 matrix (its
// payload and block scales slice; the per-tensor scale and the activation
// scale are metadata every rank reads) and the K/V-cache scales.
bool is_replicated_35(const QwenExpectedTensor& e) {
  if (e.role == QwenTensorRole::Fp4Global || e.role == QwenTensorRole::InputScale ||
      e.role == QwenTensorRole::Fp8TensorScale)
    return true;
  switch (e.cls) {
    case QwenWeightClass::Norm:
      return true;
    case QwenWeightClass::Mtp:
      return true;
    case QwenWeightClass::Router:
      return true;
    case QwenWeightClass::FullAttn:
      return ends_with(e.name, "q_norm.weight") || ends_with(e.name, "k_norm.weight") ||
             ends_with(e.name, ".k_scale") || ends_with(e.name, ".v_scale");
    case QwenWeightClass::Gdn:
      return ends_with(e.name, "norm.weight");
    default:
      return false;
  }
}

// The Qwen3Next dialect's dense stack form (Qwen35LayerStream::
// set_dense_weights_fp8): process-wide, set before any stream is built.
bool g_dense_weights_fp8 = false;

void widen_bf16_to_f32(const uint16_t* src, float* dst, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = bf16_bits_to_float(src[i]);
}

}  // namespace

Qwen3NextGdnGather qwen3next_gdn_gather(const Qwen35TextConfig& cfg, int rank, int world) {
  qwen35_tp_validate_geometry(cfg, rank, world);
  const int64_t dk = cfg.gdn_key_head_dim, dv = cfg.gdn_value_head_dim;
  const int64_t ratio = cfg.gdn_value_heads / cfg.gdn_key_heads;  // value heads per key head
  const int64_t lk = cfg.gdn_key_heads / world;
  const int64_t group = 2 * dk + 2 * ratio * dv;       // one key head's rows of in_proj_qkvz
  const int64_t k0 = static_cast<int64_t>(rank) * lk;  // this rank's first key head
  Qwen3NextGdnGather m;
  // Destination order: every q, every k, every v (head-major), as the
  // Qwen3.5 dialect ships in_proj_qkv.
  for (int64_t i = 0; i < lk; ++i) m.qkv.push_back({(k0 + i) * group, dk, i * dk});
  for (int64_t i = 0; i < lk; ++i) m.qkv.push_back({(k0 + i) * group + dk, dk, (lk + i) * dk});
  for (int64_t i = 0; i < lk; ++i)
    m.qkv.push_back({(k0 + i) * group + 2 * dk, ratio * dv, 2 * lk * dk + i * ratio * dv});
  for (int64_t i = 0; i < lk; ++i) {
    m.z.push_back({(k0 + i) * group + 2 * dk + ratio * dv, ratio * dv, i * ratio * dv});
    m.b.push_back({(k0 + i) * 2 * ratio, ratio, i * ratio});
    m.a.push_back({(k0 + i) * 2 * ratio + ratio, ratio, i * ratio});
  }
  return m;
}

void qwen3next_fp4_dequant_bf16(const uint8_t* payload, size_t payload_stride,
                                const uint8_t* scales, size_t scale_stride, float weight_scale_2,
                                int64_t rows, int64_t cols, uint16_t* out) {
  constexpr int64_t kGroup = kFp4Group, kBytes = kFp4Group / 2;
  for (int64_t n = 0; n < rows; ++n) {
    const uint8_t* pr = payload + static_cast<size_t>(n) * payload_stride;
    const uint8_t* sr = scales + static_cast<size_t>(n) * scale_stride;
    uint16_t* o = out + static_cast<size_t>(n) * static_cast<size_t>(cols);
    for (int64_t b = 0; b < cols / kGroup; ++b) {
      // The sixteen codes under this block's scale, then the block's bytes.
      const float s = fp8_e4m3_bits_to_float(sr[b]);
      uint16_t value[16];
      for (int c = 0; c < 16; ++c)
        value[c] = float_to_bf16_bits(fp4_e2m1_bits_to_float(static_cast<uint8_t>(c)) * s *
                                      weight_scale_2);
      for (int64_t j = 0; j < kBytes; ++j) {
        const uint8_t byte = pr[b * kBytes + j];
        o[b * kGroup + 2 * j] = value[byte & 0xFu];
        o[b * kGroup + 2 * j + 1] = value[byte >> 4];
      }
    }
  }
}

void qwen35_fp4_packed_dequant_bf16(const uint8_t* payload, size_t payload_stride,
                                    const uint8_t* scales, size_t scale_stride,
                                    float weight_global_scale, int64_t rows, int64_t cols,
                                    uint16_t* out) {
  constexpr int64_t kGroup = kFp4Group, kBytes = kFp4Group / 2;
  for (int64_t n = 0; n < rows; ++n) {
    const uint8_t* pr = payload + static_cast<size_t>(n) * payload_stride;
    const uint8_t* sr = scales + static_cast<size_t>(n) * scale_stride;
    uint16_t* o = out + static_cast<size_t>(n) * static_cast<size_t>(cols);
    for (int64_t b = 0; b < cols / kGroup; ++b) {
      // The block's scale over the tensor's global scale first (one fp32
      // rounding), then the sixteen codes under it.
      const float s = fp8_e4m3_bits_to_float(sr[b]) / weight_global_scale;
      uint16_t value[16];
      for (int c = 0; c < 16; ++c)
        value[c] = float_to_bf16_bits(fp4_e2m1_bits_to_float(static_cast<uint8_t>(c)) * s);
      for (int64_t j = 0; j < kBytes; ++j) {
        const uint8_t byte = pr[b * kBytes + j];
        o[b * kGroup + 2 * j] = value[byte & 0xFu];
        o[b * kGroup + 2 * j + 1] = value[byte >> 4];
      }
    }
  }
}

Qwen35LocalGeometry Qwen35LocalGeometry::from_config(const Qwen35TextConfig& cfg, int rank,
                                                     int world, LoaderHeadSharding) {
  qwen35_tp_validate_geometry(cfg, rank, world);
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("qwen35 loader: rank/world out of range");
  Qwen35LocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_key_heads = cfg.gdn_key_heads / world;
  g.local_value_heads = cfg.gdn_value_heads / world;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  if (cfg.num_key_value_heads >= world) {
    g.local_kv_heads = cfg.num_key_value_heads / world;
    g.kv_head_begin = g.local_kv_heads * rank;
  } else {
    g.local_kv_heads = 1;
    g.kv_head_begin = rank / (world / cfg.num_key_value_heads);
  }
  // The rank's query heads must belong to its kv head(s).
  const int64_t lh_total = cfg.num_attention_heads, kv_total = cfg.num_key_value_heads;
  for (int h = g.head_begin; h < g.head_begin + g.local_heads; ++h) {
    const int64_t owner = static_cast<int64_t>(h) * kv_total / lh_total;
    if (owner < g.kv_head_begin || owner >= g.kv_head_begin + g.local_kv_heads)
      throw std::invalid_argument("qwen35 loader: query/kv head sharding mismatch");
  }
  g.local_inter = cfg.intermediate_size / world;
  if (cfg.moe()) {
    g.local_moe_inter = cfg.moe_intermediate_size / world;
    g.local_shared_inter = cfg.shared_expert_intermediate_size / world;
  }
  g.lm_vocab_begin =
      static_cast<int>(static_cast<int64_t>(cfg.vocab_size) * rank / world);
  g.lm_vocab_count =
      static_cast<int>(static_cast<int64_t>(cfg.vocab_size) * (rank + 1) / world) - g.lm_vocab_begin;
  return g;
}

struct Qwen35LoaderFamily::Builder : WeightBuilder<QwenExpectedTensor> {
  const Qwen35TextConfig& cfg;
  const Qwen35LocalGeometry& geo;
  Qwen35LayerResident& out;

  Builder(const Qwen35TextConfig& cfg_, const Qwen35LocalGeometry& geo_,
          const std::vector<QwenExpectedTensor>& table_,
          const std::unordered_map<std::string, const QwenExpectedTensor*>& by_name_,
          LayerBump& bump_, Qwen35LayerResident& out_,
          const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<QwenExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_,
                                          geo_.rank, geo_.world, "qwen35 loader"),
        cfg(cfg_),
        geo(geo_),
        out(out_) {}

  bool replicated(const QwenExpectedTensor& e) const override { return is_replicated_35(e); }

  // An FP8 matrix's two checkpoint tensors: e4m3 [N, K] payload + BF16
  // [N/128, K/128] scales (the binding table registers both; the suffix is
  // part of the contract).
  struct Fp8Native {
    const TensorInfo* payload = nullptr;
    const TensorInfo* scales = nullptr;
    const QwenExpectedTensor* payload_entry = nullptr;
    const QwenExpectedTensor* scales_entry = nullptr;
    int64_t N = 0, K = 0, SN = 0, SK = 0;
  };

  Fp8Native fp8_source(const std::string& name) {
    const QwenExpectedTensor& e = expected(name);
    if (e.shape.size() != 2) fail(name + ": fp8 matrix needs 2 dims");
    const int64_t N = e.shape[0], K = e.shape[1];
    const std::string sname = name + "_scale_inv";
    const QwenExpectedTensor& se = expected(sname);
    const int64_t SN = (N + 127) / 128, SK = (K + 127) / 128;
    if (se.shape.size() != 2 || se.shape[0] != SN || se.shape[1] != SK)
      fail(sname + ": BF16 scale grid must be [N/128, K/128]");
    Fp8Native s;
    // source() needs the checkpoint map: only the copy build has it. The
    // counting build (copy=false) runs on an empty map, so it must never
    // touch sources — every dereference below is already inside if (copy).
    if (copy) {
      s.payload = &source(name);
      s.scales = &source(sname);
    } else {
      s.payload = nullptr;
      s.scales = nullptr;
    }
    s.payload_entry = &e;
    s.scales_entry = &se;
    s.N = N;
    s.K = K;
    s.SN = SN;
    s.SK = SK;
    return s;
  }

  // FP8 row range into the bump (128-aligned bounds: whole scale rows, so no
  // re-blocking — the scales concatenate exactly like the payload).
  GlmQuantMatrix load_fp8_native_rows(const std::string& name, int64_t r0, int64_t rn) {
    Fp8Native s = fp8_source(name);
    check_range(name, r0, rn, s.N);
    if (r0 % 128 != 0 || rn % 128 != 0)
      fail(name + ": fp8 row slice needs 128-aligned bounds");
    GlmQuantMatrix q;
    q.rows = rn;
    q.cols = s.K;
    q.scale_block_rows = 128;
    q.scale_block_cols = 128;
    const int64_t sn = rn / 128;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rn) * s.K));
    q.scales = static_cast<const float*>(bump.alloc(static_cast<size_t>(sn) * s.SK * 4));
    if (copy) {
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(s.payload->data) + static_cast<size_t>(r0) * s.K,
                  static_cast<size_t>(rn) * s.K);
      widen_bf16_to_f32(static_cast<const uint16_t*>(s.scales->data) +
                            static_cast<size_t>(r0 / 128) * s.SK,
                        static_cast<float*>(bump.host(const_cast<float*>(q.scales))),
                        static_cast<size_t>(sn) * s.SK);
    }
    note_read(*s.payload_entry, static_cast<size_t>(rn) * s.K);
    note_read(*s.scales_entry, static_cast<size_t>(sn) * s.SK * 2);
    if (copy) {
      consumed(*s.payload);
      consumed(*s.scales);
    }
    return q;
  }

  // FP8 column range into the bump (128-aligned; payload rows packed, scale
  // rows packed + widened).
  GlmQuantMatrix load_fp8_native_cols(const std::string& name, int64_t c0, int64_t cn) {
    Fp8Native s = fp8_source(name);
    check_range(name + " cols", c0, cn, s.K);
    if (c0 % 128 != 0 || cn % 128 != 0)
      fail(name + ": fp8 column slice needs 128-aligned bounds");
    GlmQuantMatrix q;
    q.rows = s.N;
    q.cols = cn;
    q.scale_block_rows = 128;
    q.scale_block_cols = 128;
    const int64_t sk = cn / 128;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(s.N) * cn));
    q.scales = static_cast<const float*>(bump.alloc(static_cast<size_t>(s.SN) * sk * 4));
    if (copy) {
      const uint8_t* sp = static_cast<const uint8_t*>(s.payload->data);
      uint8_t* dp = static_cast<uint8_t*>(bump.host(const_cast<uint8_t*>(q.payload)));
      for (int64_t r = 0; r < s.N; ++r)
        std::memcpy(dp + static_cast<size_t>(r) * cn, sp + static_cast<size_t>(r) * s.K + c0,
                    static_cast<size_t>(cn));
      const uint16_t* ss = static_cast<const uint16_t*>(s.scales->data);
      float* ds = const_cast<float*>(q.scales);
      float* dh = static_cast<float*>(bump.host(ds));
      for (int64_t sr = 0; sr < s.SN; ++sr)
        widen_bf16_to_f32(ss + static_cast<size_t>(sr) * s.SK + c0 / 128,
                          dh + static_cast<size_t>(sr) * sk, static_cast<size_t>(sk));
    }
    note_read(*s.payload_entry, static_cast<size_t>(s.N) * cn);
    note_read(*s.scales_entry, static_cast<size_t>(s.SN) * sk * 2);
    if (copy) {
      consumed(*s.payload);
      consumed(*s.scales);
    }
    return q;
  }

  // Contiguous BF16 row ranges of several source row spans concatenated into
  // one bump buffer (the GDN's segmented qkv/conv assembly + lm head shard).
  void copy_rows_into(const std::string& name, int64_t src_row, int64_t rows, uint16_t* dst,
                      int64_t dst_row, int64_t width) {
    const QwenExpectedTensor& e = expected(name);
    check_range(name, src_row, rows, e.shape[0]);
    if (copy) {
      const TensorInfo& t = source(name);
      std::memcpy(bump.host(dst) + static_cast<size_t>(dst_row) * width,
                  static_cast<const uint8_t*>(t.data) + static_cast<size_t>(src_row) * width * 2,
                  static_cast<size_t>(rows) * width * 2);
    }
    note_read(e, static_cast<size_t>(rows) * width * 2);
  }

  // FP8 row segments (each 128-aligned) concatenated into one bump matrix
  // (the GDN's merged in_proj_qkv): payload and scale rows both concatenate.
  GlmQuantMatrix merge_fp8_segments(const std::string& name, int64_t total_rows, int64_t K,
                                    const std::vector<std::array<int64_t, 3>>& segs) {
    Fp8Native s = fp8_source(name);
    if (K != s.K) fail(name + ": merged width disagrees with the checkpoint");
    GlmQuantMatrix q;
    q.rows = total_rows;
    q.cols = K;
    q.scale_block_rows = 128;
    q.scale_block_cols = 128;
    const int64_t SK = s.SK;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(total_rows) * K));
    q.scales = static_cast<const float*>(
        bump.alloc(static_cast<size_t>(total_rows / 128) * SK * 4));
    if (copy) {
      uint8_t* dp = reinterpret_cast<uint8_t*>(bump.host(const_cast<uint8_t*>(q.payload)));
      float* ds = static_cast<float*>(bump.host(const_cast<float*>(q.scales)));
      const uint8_t* sp = static_cast<const uint8_t*>(s.payload->data);
      const uint16_t* ss = static_cast<const uint16_t*>(s.scales->data);
      for (const auto& sg : segs) {
        const int64_t src_row = sg[0], rows = sg[1], dst_row = sg[2];
        check_range(name, src_row, rows, s.N);
        if (src_row % 128 != 0 || rows % 128 != 0 || dst_row % 128 != 0)
          fail(name + ": merged fp8 segments need 128-aligned bounds");
        std::memcpy(dp + static_cast<size_t>(dst_row) * K,
                    sp + static_cast<size_t>(src_row) * K, static_cast<size_t>(rows) * K);
        widen_bf16_to_f32(ss + static_cast<size_t>(src_row / 128) * SK,
                          ds + static_cast<size_t>(dst_row / 128) * SK,
                          static_cast<size_t>(rows / 128) * SK);
      }
      consumed(*s.payload);
      consumed(*s.scales);
    }
    // Accounting runs in both modes (the counting pass plans source bytes).
    for (const auto& sg : segs) {
      note_read(*s.payload_entry, static_cast<size_t>(sg[1]) * K);
      note_read(*s.scales_entry, static_cast<size_t>(sg[1] / 128) * SK * 2);
    }
    return q;
  }

  // ---- modelopt's per-tensor FP8 (the NVFP4 mixed release's dense projections) ----
  // `base.weight` e4m3 [N, K] under ONE F32 scale `base.weight_scale` (the
  // value is code x scale) and the recipe's `base.input_scale`. The resident
  // is the block form the fp8 kernels read: the payload bytes as shipped,
  // every block of the 128 x 128 scale grid holding the tensor's scale — the
  // same value for every element, so nothing is re-encoded.
  struct Fp8Tensor {
    const QwenExpectedTensor* payload = nullptr;
    const QwenExpectedTensor* scale = nullptr;
    int64_t N = 0, K = 0;
  };
  Fp8Tensor fp8_tensor(const std::string& base) const {
    Fp8Tensor s;
    s.payload = &expected(base + ".weight");
    s.scale = &expected(base + ".weight_scale");
    if (s.payload->shape.size() != 2) fail(base + ": fp8 matrix needs 2 dims");
    if (!s.scale->shape.empty()) fail(base + ".weight_scale: the per-tensor scale is a scalar");
    s.N = s.payload->shape[0];
    s.K = s.payload->shape[1];
    return s;
  }
  // The tensor's scale (the copy pass only: the counting pass has no sources).
  float read_fp8_tensor_scale(const Fp8Tensor& s) {
    const TensorInfo& t = source(s.scale->name);
    float v;
    std::memcpy(&v, t.data, 4);
    if (!(v > 0.0f) || !std::isfinite(v))
      fail("'" + s.scale->name + "' is not a positive finite scale");
    consumed(t);
    return v;
  }
  // The block-form grant of a [rows, cols] slice, its scale grid filled with
  // the tensor's scale in the copy pass; the set's scalars accounted.
  GlmQuantMatrix grant_fp8_tensor(const std::string& base, const Fp8Tensor& s, int64_t rows,
                                  int64_t cols) {
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.scale_block_rows = 128;
    q.scale_block_cols = 128;
    const size_t blocks = static_cast<size_t>(q.scale_rows()) * static_cast<size_t>(q.scale_cols());
    q.payload = static_cast<const uint8_t*>(
        bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols)));
    q.scales = static_cast<const float*>(bump.alloc(blocks * 4));
    if (copy) {
      const float scale = read_fp8_tensor_scale(s);
      float* hs = bump.host(const_cast<float*>(q.scales));
      std::fill(hs, hs + blocks, scale);
    }
    note_read(*s.scale, 4);
    (void)load_raw(base + ".input_scale");
    return q;
  }
  // Row segments {source row, rows, destination row} of the matrix (128-
  // aligned, as the block form's are) into one resident of `total_rows` rows.
  GlmQuantMatrix load_fp8_tensor_rows(const std::string& base,
                                      const std::vector<std::array<int64_t, 3>>& segs,
                                      int64_t total_rows) {
    const Fp8Tensor s = fp8_tensor(base);
    int64_t covered = 0;
    for (const auto& sg : segs) {
      check_range(base, sg[0], sg[1], s.N);
      check_range(base + " (gathered)", sg[2], sg[1], total_rows);
      if (sg[0] % 128 != 0 || sg[1] % 128 != 0 || sg[2] % 128 != 0)
        fail(base + ": fp8 row segments need 128-aligned bounds");
      covered += sg[1];
    }
    if (covered != total_rows) fail("the segments of '" + base + "' do not fill its destination");
    const GlmQuantMatrix q = grant_fp8_tensor(base, s, total_rows, s.K);
    if (copy) {
      const TensorInfo& tp = source(s.payload->name);
      const uint8_t* sp = static_cast<const uint8_t*>(tp.data);
      uint8_t* dp = bump.host(const_cast<uint8_t*>(q.payload));
      for (const auto& sg : segs)
        std::memcpy(dp + static_cast<size_t>(sg[2]) * s.K, sp + static_cast<size_t>(sg[0]) * s.K,
                    static_cast<size_t>(sg[1]) * s.K);
      consumed(tp);
    }
    for (const auto& sg : segs) note_read(*s.payload, static_cast<size_t>(sg[1]) * s.K);
    return q;
  }
  // A column range of the matrix (128-aligned), rows packed contiguous.
  GlmQuantMatrix load_fp8_tensor_cols(const std::string& base, int64_t c0, int64_t cn) {
    const Fp8Tensor s = fp8_tensor(base);
    check_range(base + " cols", c0, cn, s.K);
    if (c0 % 128 != 0 || cn % 128 != 0) fail(base + ": fp8 column slice needs 128-aligned bounds");
    const GlmQuantMatrix q = grant_fp8_tensor(base, s, s.N, cn);
    if (copy) {
      const TensorInfo& tp = source(s.payload->name);
      const uint8_t* sp = static_cast<const uint8_t*>(tp.data);
      uint8_t* dp = bump.host(const_cast<uint8_t*>(q.payload));
      for (int64_t r = 0; r < s.N; ++r)
        std::memcpy(dp + static_cast<size_t>(r) * cn, sp + static_cast<size_t>(r) * s.K + c0,
                    static_cast<size_t>(cn));
      consumed(tp);
    }
    note_read(*s.payload, static_cast<size_t>(s.N) * static_cast<size_t>(cn));
    return q;
  }

  // A dense projection of the Qwen3.5 dialect that ships FP8, in whichever
  // form the recipe has it (`base` without ".weight"): the FP8 release's
  // block scales, or the NVFP4 mixed release's one scale per tensor.
  bool fp8_per_tensor() const { return cfg.quant_kind == Qwen35QuantKind::Nvfp4Mixed; }
  GlmQuantMatrix load_fp8_rows35(const std::string& base, int64_t r0, int64_t rn) {
    if (fp8_per_tensor()) return load_fp8_tensor_rows(base, {{r0, rn, 0}}, rn);
    return load_fp8_native_rows(base + ".weight", r0, rn);
  }
  GlmQuantMatrix load_fp8_cols35(const std::string& base, int64_t c0, int64_t cn) {
    if (fp8_per_tensor()) return load_fp8_tensor_cols(base, c0, cn);
    return load_fp8_native_cols(base + ".weight", c0, cn);
  }

  // `is_mtp`: the NVFP4 mixed release ships its draft layer BF16 (`mtp*` is on
  // the recipe's ignore list) — the Qwen3Next draft builder's case, under the
  // same names; the FP8 release's draft layer is FP8 like the rest.
  void build_full(const std::string& p, bool is_mtp) {
    if (is_mtp && fp8_per_tensor()) {
      build_full_next(p, true);
      return;
    }
    const int64_t d = cfg.head_dim;
    QwenFullAttnResident& a = out.full;
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_kv_heads = geo.local_kv_heads;
    a.kv_head_begin = geo.kv_head_begin;
    const int64_t q0 = static_cast<int64_t>(geo.head_begin) * 2 * d;
    const int64_t qn = static_cast<int64_t>(geo.local_heads) * 2 * d;
    const int64_t kv0 = static_cast<int64_t>(geo.kv_head_begin) * d;
    const int64_t kvn = static_cast<int64_t>(geo.local_kv_heads) * d;
    const int64_t o0 = static_cast<int64_t>(geo.head_begin) * d;
    const int64_t on = static_cast<int64_t>(geo.local_heads) * d;
    a.q_proj_fp8 = load_fp8_rows35(p + "self_attn.q_proj", q0, qn);
    a.k_proj_fp8 = load_fp8_rows35(p + "self_attn.k_proj", kv0, kvn);
    a.v_proj_fp8 = load_fp8_rows35(p + "self_attn.v_proj", kv0, kvn);
    a.o_proj_fp8 = load_fp8_cols35(p + "self_attn.o_proj", o0, on);
    a.q_norm = load_bf16(p + "self_attn.q_norm.weight");
    a.k_norm = load_bf16(p + "self_attn.k_norm.weight");
  }

  // The conv channels at the local geometry: the same three [q | k | v]
  // segments as in_proj_qkv ([C, 1, w] rows, head-major in both dialects).
  uint16_t* load_gdn_conv(const std::string& p) {
    const int64_t dk = cfg.gdn_key_head_dim, dv = cfg.gdn_value_head_dim;
    const int64_t K = static_cast<int64_t>(cfg.gdn_key_heads) * dk;
    const int64_t lk = geo.local_key_heads, lv = geo.local_value_heads;
    const int64_t r = geo.rank;
    const int64_t local_rows = 2 * lk * dk + lv * dv;
    const int64_t w = cfg.gdn_conv_width;
    uint16_t* conv = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(local_rows) * static_cast<size_t>(w) * 2));
    const std::string conv_name = p + "linear_attn.conv1d.weight";
    copy_rows_into(conv_name, r * lk * dk, lk * dk, conv, 0, w);
    copy_rows_into(conv_name, K + r * lk * dk, lk * dk, conv, lk * dk, w);
    copy_rows_into(conv_name, 2 * K + r * lv * dv, lv * dv, conv, 2 * lk * dk, w);
    if (copy) consumed(source(conv_name));
    return conv;
  }

  void build_gdn(const std::string& p) {
    const int64_t H = cfg.hidden_size;
    const int64_t dk = cfg.gdn_key_head_dim, dv = cfg.gdn_value_head_dim;
    const int64_t K = static_cast<int64_t>(cfg.gdn_key_heads) * dk;
    const int64_t lk = geo.local_key_heads, lv = geo.local_value_heads;
    const int64_t r = geo.rank;
    QwenGdnResident& g = out.gdn;
    g.local_key_heads = static_cast<int>(lk);
    g.local_value_heads = static_cast<int>(lv);
    // in_proj_qkv rows [q | k | v] at the local geometry (world=1: whole).
    const int64_t local_rows = 2 * lk * dk + lv * dv;
    const std::vector<std::array<int64_t, 3>> qkv_segs = {
        {r * lk * dk, lk * dk, 0},
        {K + r * lk * dk, lk * dk, lk * dk},
        {2 * K + r * lv * dv, lv * dv, 2 * lk * dk}};
    if (fp8_per_tensor()) {
      g.in_proj_qkv_fp8 = load_fp8_tensor_rows(p + "linear_attn.in_proj_qkv", qkv_segs, local_rows);
    } else {
      const std::string qkv_name = p + "linear_attn.in_proj_qkv.weight";
      g.in_proj_qkv_fp8 = merge_fp8_segments(qkv_name, local_rows, H, qkv_segs);
      if (copy) consumed(source(qkv_name));
    }
    g.conv = load_gdn_conv(p);
    g.in_proj_z_fp8 = load_fp8_rows35(p + "linear_attn.in_proj_z", r * lv * dv, lv * dv);
    g.in_proj_a = load_bf16_rows(p + "linear_attn.in_proj_a.weight", r * lv, lv);
    g.in_proj_b = load_bf16_rows(p + "linear_attn.in_proj_b.weight", r * lv, lv);
    g.a_log = load_bf16_as_f32(p + "linear_attn.A_log", r * lv, lv);
    g.dt_bias = load_bf16_as_f32(p + "linear_attn.dt_bias", r * lv, lv);
    g.norm = load_bf16(p + "linear_attn.norm.weight");
    g.out_proj_fp8 = load_fp8_cols35(p + "linear_attn.out_proj", r * lv * dv, lv * dv);
  }

  void build_mlp(const std::string& p) {
    const int64_t li = geo.local_inter;
    const int64_t r0 = static_cast<int64_t>(geo.rank) * li;
    Qwen35DenseMlpResident& m = out.mlp;
    m.gate_fp8 = load_fp8_native_rows(p + "mlp.gate_proj.weight", r0, li);
    m.up_fp8 = load_fp8_native_rows(p + "mlp.up_proj.weight", r0, li);
    m.down_fp8 = load_fp8_native_cols(p + "mlp.down_proj.weight", r0, li);
  }

  // ---- the Qwen3Next dialect ---------------------------------------------------

  // Row runs of a BF16 matrix gathered into one grant of `total_rows` rows
  // (the interleaved GDN projections). The source is not consumed here: a
  // second gather of the same tensor follows.
  uint16_t* gather_bf16_rows(const std::string& name, const std::vector<Qwen35RowRun>& runs,
                             int64_t total_rows) {
    const QwenExpectedTensor& e = expected(name);
    if (e.shape.size() != 2) fail("'" + name + "' is not a matrix");
    const int64_t width = e.shape[1];
    int64_t covered = 0;
    for (const Qwen35RowRun& run : runs) {
      check_range(name + " (gathered)", run.dst_row, run.rows, total_rows);
      covered += run.rows;
    }
    if (covered != total_rows) fail("the gather of '" + name + "' does not fill its destination");
    uint16_t* dst = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(total_rows) * static_cast<size_t>(width) * 2));
    for (const Qwen35RowRun& run : runs)
      copy_rows_into(name, run.src_row, run.rows, dst, run.dst_row, width);
    return dst;
  }

  // ---- the dense stack as block FP8 (dense_weights fp8) ------------------------
  // The fp8 kernels read whole 128-blocks on a sliced axis (the Qwen3.5
  // release's contract): a slice that starts or ends inside a block is
  // refused by name.
  void require_fp8_aligned(const std::string& name, int64_t start, int64_t count) const {
    if (start % fp8_quant::kBlock != 0 || count % fp8_quant::kBlock != 0)
      fail(name + ": fp8 slice [" + std::to_string(start) + ", +" + std::to_string(count) +
           ") needs 128-aligned bounds (dense_weights fp8 does not serve this geometry)");
  }

  // The gather of gather_bf16_rows assembled on the host and encoded into
  // the bump: every run is whole 128-row blocks, so each head's rows sit on
  // the scale grid as they do in the Qwen3.5 release's merged matrix.
  GlmQuantMatrix gather_bf16_rows_fp8(const std::string& name,
                                      const std::vector<Qwen35RowRun>& runs, int64_t total_rows) {
    const QwenExpectedTensor& e = expected(name);
    if (e.shape.size() != 2) fail("'" + name + "' is not a matrix");
    const int64_t width = e.shape[1];
    int64_t covered = 0;
    for (const Qwen35RowRun& run : runs) {
      check_range(name, run.src_row, run.rows, e.shape[0]);
      check_range(name + " (gathered)", run.dst_row, run.rows, total_rows);
      require_fp8_aligned(name + " (gathered)", run.dst_row, run.rows);
      covered += run.rows;
    }
    if (covered != total_rows) fail("the gather of '" + name + "' does not fill its destination");
    std::vector<uint16_t> merged;
    if (copy) {
      merged.resize(static_cast<size_t>(total_rows) * static_cast<size_t>(width));
      const uint8_t* src = static_cast<const uint8_t*>(source(name).data);
      for (const Qwen35RowRun& run : runs)
        std::memcpy(merged.data() + static_cast<size_t>(run.dst_row) * width,
                    src + static_cast<size_t>(run.src_row) * width * 2,
                    static_cast<size_t>(run.rows) * width * 2);
    }
    for (const Qwen35RowRun& run : runs) note_read(e, static_cast<size_t>(run.rows) * width * 2);
    return encode_fp8(merged.data(), static_cast<size_t>(width), total_rows, width);
  }

  // A BF16 matrix's row / column slice encoded to block FP8 (the builder's
  // encoders, behind the alignment check).
  GlmQuantMatrix load_dense_rows_fp8(const std::string& name, int64_t row_start, int64_t rows) {
    require_fp8_aligned(name, row_start, rows);
    return load_bf16_rows_fp8(name, row_start, rows);
  }
  GlmQuantMatrix load_dense_cols_fp8(const std::string& name, int64_t col_start, int64_t cols) {
    require_fp8_aligned(name + " cols", col_start, cols);
    return load_bf16_cols_fp8(name, col_start, cols);
  }

  // The NVFP4 set of the matrix `base` ("...out_proj") in the container the
  // recipe names. modelopt: base.weight U8 [N, K/2] (two e2m1 codes a byte),
  // base.weight_scale e4m3 [N, K/16], base.weight_scale_2 F32 [] (the
  // per-tensor scale, a multiplier) and base.input_scale F32 [] (the
  // recipe's activation scale). compressed-tensors: base.weight_packed and
  // base.weight_scale in the same geometry, base.weight_global_scale F32 [1]
  // (the per-tensor scale, a divisor) and base.input_global_scale F32 [1].
  bool fp4_packed() const { return cfg.quant_kind == Qwen35QuantKind::Nvfp4Packed; }
  std::string fp4_payload_name(const std::string& base) const {
    return base + (fp4_packed() ? ".weight_packed" : ".weight");
  }
  std::string fp4_global_name(const std::string& base) const {
    return base + (fp4_packed() ? ".weight_global_scale" : ".weight_scale_2");
  }
  std::string fp4_input_name(const std::string& base) const {
    return base + (fp4_packed() ? ".input_global_scale" : ".input_scale");
  }
  struct Fp4Set {
    const QwenExpectedTensor* payload = nullptr;
    const QwenExpectedTensor* scales = nullptr;
    const QwenExpectedTensor* global = nullptr;
    int64_t N = 0, K = 0;
  };

  // The set's entries with a slice checked against them: rows are free
  // (each carries its own scales), columns start and span whole 16-blocks.
  Fp4Set fp4_set(const std::string& base, int64_t r0, int64_t rn, int64_t c0, int64_t cn) const {
    Fp4Set s;
    s.payload = &expected(fp4_payload_name(base));
    s.scales = &expected(base + ".weight_scale");
    s.global = &expected(fp4_global_name(base));
    if (s.payload->shape.size() != 2) fail(base + ": NVFP4 matrix needs 2 dims");
    s.N = s.payload->shape[0];
    s.K = s.payload->shape[1] * 2;
    fp4_check_cols(s.K, who.c_str());
    if (s.scales->shape.size() != 2 || s.scales->shape[0] != s.N ||
        s.scales->shape[1] != s.K / kFp4Group)
      fail("NVFP4 scale geometry mismatch on " + base);
    check_range(base, r0, rn, s.N);
    check_range(base + " cols", c0, cn, s.K);
    if (c0 % kFp4Group != 0 || cn % kFp4Group != 0)
      fail("NVFP4 column slice of " + base + " is not 16-aligned");
    return s;
  }

  // The per-tensor scale as the container stores it — modelopt's multiplier
  // or compressed-tensors' divisor (the copy pass only: the counting pass has
  // no sources).
  float read_fp4_global(const Fp4Set& s) {
    const TensorInfo& t = source(s.global->name);
    float g;
    std::memcpy(&g, t.data, 4);
    if (!(g > 0.0f) || !std::isfinite(g))
      fail("'" + s.global->name + "' is not a positive finite scale");
    consumed(t);
    return g;
  }

  // The copy pass's dequant of rows [r0, +rn) x columns [c0, +cn) of an NVFP4
  // matrix into `host` ([rn, cn] BF16; qwen3next_fp4_dequant_bf16, or
  // qwen35_fp4_packed_dequant_bf16 for the compressed-tensors container), and
  // the set's byte accounting (both passes).
  void read_fp4_dequantized(const Fp4Set& s, int64_t r0, int64_t rn, int64_t c0, int64_t cn,
                            uint16_t* host) {
    const size_t pc_full = static_cast<size_t>(s.K / 2),
                 sc_full = static_cast<size_t>(s.K / kFp4Group);
    if (copy) {
      const TensorInfo& tp = source(s.payload->name);
      const TensorInfo& ts = source(s.scales->name);
      const uint8_t* codes =
          static_cast<const uint8_t*>(tp.data) + static_cast<size_t>(r0) * pc_full + c0 / 2;
      const uint8_t* block_scales =
          static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(r0) * sc_full + c0 / kFp4Group;
      const float global = read_fp4_global(s);
      if (fp4_packed())
        qwen35_fp4_packed_dequant_bf16(codes, pc_full, block_scales, sc_full, global, rn, cn, host);
      else
        qwen3next_fp4_dequant_bf16(codes, pc_full, block_scales, sc_full, global, rn, cn, host);
      consumed(tp);
      consumed(ts);
    }
    note_read(*s.payload, static_cast<size_t>(rn) * static_cast<size_t>(cn / 2));
    note_read(*s.scales, static_cast<size_t>(rn) * static_cast<size_t>(cn / kFp4Group));
    note_read(*s.global, 4);
  }

  // An NVFP4 slice dequantized on the host into a BF16 grant: the dense
  // kernels' form of out_proj, o_proj and the shared expert. The activation
  // scale has no reader in this form (W4A16); it is read so the build
  // accounts for the whole set, as Flash-Next's loader reads its experts'.
  uint16_t* load_fp4_as_bf16(const std::string& base, int64_t r0, int64_t rn, int64_t c0,
                             int64_t cn) {
    const Fp4Set s = fp4_set(base, r0, rn, c0, cn);
    uint16_t* dst =
        static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(rn) * static_cast<size_t>(cn) * 2));
    read_fp4_dequantized(s, r0, rn, c0, cn, copy ? bump.host(dst) : nullptr);
    (void)load_raw(fp4_input_name(base));
    return dst;
  }

  // The same slice under dense_weights fp8: dequantized to its BF16 values
  // on the host, then encoded to block FP8 into the bump — the matrix the
  // fp8 kernels read is the encode of exactly what the BF16 form holds.
  GlmQuantMatrix load_fp4_as_fp8(const std::string& base, int64_t r0, int64_t rn, int64_t c0,
                                 int64_t cn) {
    const Fp4Set s = fp4_set(base, r0, rn, c0, cn);
    std::vector<uint16_t> bf16;
    if (copy) bf16.resize(static_cast<size_t>(rn) * static_cast<size_t>(cn));
    read_fp4_dequantized(s, r0, rn, c0, cn, bf16.data());
    const GlmQuantMatrix q = encode_fp8(bf16.data(), static_cast<size_t>(cn), rn, cn);
    (void)load_raw(fp4_input_name(base));
    return q;
  }
  GlmQuantMatrix load_fp4_rows_as_fp8(const std::string& base, int64_t row_start, int64_t rows) {
    require_fp8_aligned(base, row_start, rows);
    const QwenExpectedTensor& ep = expected(fp4_payload_name(base));
    return load_fp4_as_fp8(base, row_start, rows, 0, ep.shape.size() == 2 ? ep.shape[1] * 2 : 0);
  }
  GlmQuantMatrix load_fp4_cols_as_fp8(const std::string& base, int64_t col_start, int64_t cols) {
    require_fp8_aligned(base + " cols", col_start, cols);
    const QwenExpectedTensor& ep = expected(fp4_payload_name(base));
    return load_fp4_as_fp8(base, 0, ep.shape.empty() ? 0 : ep.shape[0], col_start, cols);
  }

  // A routed expert's NVFP4 slice in Flash-Next's resident form (the expert
  // kernels are shared): the payload and block-scale bytes as shipped,
  // packed contiguous; the per-tensor scale stored in `global` as the
  // divisor the kernels apply to the finished dot once — modelopt's
  // multiplier as its reciprocal, compressed-tensors' global scale as
  // shipped; the activation scale read into the image and, in the modelopt
  // container, kept for the layer's maximum.
  float last_input_scale_ = 0.0f;  // the last NVFP4 expert matrix's input_scale (0: not read)
  GlmFp4Matrix load_fp4_slice(const std::string& base, int64_t r0, int64_t rn, int64_t c0,
                              int64_t cn, float* global) {
    const Fp4Set s = fp4_set(base, r0, rn, c0, cn);
    const size_t pc_full = static_cast<size_t>(s.K / 2),
                 sc_full = static_cast<size_t>(s.K / kFp4Group);
    const size_t pc = static_cast<size_t>(cn / 2), sc = static_cast<size_t>(cn / kFp4Group);
    GlmFp4Matrix q;
    q.rows = rn;
    q.cols = cn;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rn) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rn) * sc));
    if (copy) {
      const TensorInfo& tp = source(s.payload->name);
      const TensorInfo& ts = source(s.scales->name);
      const uint8_t* sp =
          static_cast<const uint8_t*>(tp.data) + static_cast<size_t>(r0) * pc_full + c0 / 2;
      const uint8_t* ss =
          static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(r0) * sc_full + c0 / kFp4Group;
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
      const float divisor = fp4_packed() ? stored : 1.0f / stored;
      std::memcpy(bump.host(global), &divisor, 4);
    }
    note_read(*s.payload, static_cast<size_t>(rn) * pc);
    note_read(*s.scales, static_cast<size_t>(rn) * sc);
    note_read(*s.global, 4);
    // modelopt's input_scale is the W4A4 prefill path's static activation
    // scale. compressed-tensors' input_global_scale is another convention
    // (the global of a locally dynamic activation scale) that nothing here
    // has been checked against: it is read into the image so the build
    // accounts for the whole set, and the layer keeps no static scale (0).
    last_input_scale_ = 0.0f;
    if (copy && !fp4_packed()) std::memcpy(&last_input_scale_, source(fp4_input_name(base)).data, 4);
    (void)load_raw(fp4_input_name(base));
    q.global_scale = global;
    return q;
  }

  void build_gdn_next(const std::string& p) {
    const int64_t H = cfg.hidden_size;
    const int64_t dk = cfg.gdn_key_head_dim, dv = cfg.gdn_value_head_dim;
    const int64_t lk = geo.local_key_heads, lv = geo.local_value_heads;
    const int64_t r = geo.rank;
    QwenGdnResident& g = out.gdn;
    g.local_key_heads = static_cast<int>(lk);
    g.local_value_heads = static_cast<int>(lv);
    // The fused projections, gathered out of the per-key-head interleave
    // into the split head-major matrices at the local geometry.
    const Qwen3NextGdnGather map = qwen3next_gdn_gather(cfg, geo.rank, geo.world);
    const bool fp8 = g_dense_weights_fp8;
    const std::string qkvz_name = p + "linear_attn.in_proj_qkvz.weight";
    if (fp8)
      g.in_proj_qkv_fp8 = gather_bf16_rows_fp8(qkvz_name, map.qkv, 2 * lk * dk + lv * dv);
    else
      g.in_proj_qkv = gather_bf16_rows(qkvz_name, map.qkv, 2 * lk * dk + lv * dv);
    g.conv = load_gdn_conv(p);
    if (fp8)
      g.in_proj_z_fp8 = gather_bf16_rows_fp8(qkvz_name, map.z, lv * dv);
    else
      g.in_proj_z = gather_bf16_rows(qkvz_name, map.z, lv * dv);
    if (copy) consumed(source(qkvz_name));
    const std::string ba_name = p + "linear_attn.in_proj_ba.weight";
    g.in_proj_a = gather_bf16_rows(ba_name, map.a, lv);
    g.in_proj_b = gather_bf16_rows(ba_name, map.b, lv);
    if (copy) consumed(source(ba_name));
    g.a_log = load_bf16_as_f32(p + "linear_attn.A_log", r * lv, lv);
    g.dt_bias = load_bf16_as_f32(p + "linear_attn.dt_bias", r * lv, lv);
    g.norm = load_bf16(p + "linear_attn.norm.weight");
    if (fp4_packed()) {
      // The compressed-tensors release leaves the GDN whole in BF16.
      if (fp8)
        g.out_proj_fp8 =
            load_dense_cols_fp8(p + "linear_attn.out_proj.weight", r * lv * dv, lv * dv);
      else
        g.out_proj = load_bf16_cols(p + "linear_attn.out_proj.weight", r * lv * dv, lv * dv);
    } else if (fp8) {
      g.out_proj_fp8 = load_fp4_cols_as_fp8(p + "linear_attn.out_proj", r * lv * dv, lv * dv);
    } else {
      g.out_proj = load_fp4_as_bf16(p + "linear_attn.out_proj", 0, H, r * lv * dv, lv * dv);
    }
  }

  void build_full_next(const std::string& p, bool is_mtp) {
    const int64_t H = cfg.hidden_size;
    const int64_t d = cfg.head_dim;
    QwenFullAttnResident& a = out.full;
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_kv_heads = geo.local_kv_heads;
    a.kv_head_begin = geo.kv_head_begin;
    const int64_t q0 = static_cast<int64_t>(geo.head_begin) * 2 * d;
    const int64_t qn = static_cast<int64_t>(geo.local_heads) * 2 * d;
    const int64_t kv0 = static_cast<int64_t>(geo.kv_head_begin) * d;
    const int64_t kvn = static_cast<int64_t>(geo.local_kv_heads) * d;
    const int64_t o0 = static_cast<int64_t>(geo.head_begin) * d;
    const int64_t on = static_cast<int64_t>(geo.local_heads) * d;
    const bool fp8 = g_dense_weights_fp8;
    if (!is_mtp && fp4_packed()) {
      // The compressed-tensors release quantizes q, k and v as well:
      // dequantized on the host like o_proj below.
      if (fp8) {
        a.q_proj_fp8 = load_fp4_rows_as_fp8(p + "self_attn.q_proj", q0, qn);
        a.k_proj_fp8 = load_fp4_rows_as_fp8(p + "self_attn.k_proj", kv0, kvn);
        a.v_proj_fp8 = load_fp4_rows_as_fp8(p + "self_attn.v_proj", kv0, kvn);
      } else {
        a.q_proj = load_fp4_as_bf16(p + "self_attn.q_proj", q0, qn, 0, H);
        a.k_proj = load_fp4_as_bf16(p + "self_attn.k_proj", kv0, kvn, 0, H);
        a.v_proj = load_fp4_as_bf16(p + "self_attn.v_proj", kv0, kvn, 0, H);
      }
    } else if (fp8) {
      a.q_proj_fp8 = load_dense_rows_fp8(p + "self_attn.q_proj.weight", q0, qn);
      a.k_proj_fp8 = load_dense_rows_fp8(p + "self_attn.k_proj.weight", kv0, kvn);
      a.v_proj_fp8 = load_dense_rows_fp8(p + "self_attn.v_proj.weight", kv0, kvn);
    } else {
      a.q_proj = load_bf16_rows(p + "self_attn.q_proj.weight", q0, qn);
      a.k_proj = load_bf16_rows(p + "self_attn.k_proj.weight", kv0, kvn);
      a.v_proj = load_bf16_rows(p + "self_attn.v_proj.weight", kv0, kvn);
    }
    if (is_mtp) {
      if (fp8)
        a.o_proj_fp8 = load_dense_cols_fp8(p + "self_attn.o_proj.weight", o0, on);
      else
        a.o_proj = load_bf16_cols(p + "self_attn.o_proj.weight", o0, on);
    } else {
      if (fp8)
        a.o_proj_fp8 = load_fp4_cols_as_fp8(p + "self_attn.o_proj", o0, on);
      else
        a.o_proj = load_fp4_as_bf16(p + "self_attn.o_proj", 0, H, o0, on);
    }
    if (!is_mtp && cfg.quant_kind == Qwen35QuantKind::Nvfp4Modelopt) {
      // The modelopt recipe's FP8 K/V-cache scales ride along: [k, v] in the
      // image and on the host (after_restore re-reads the host copies).
      const QwenExpectedTensor& ek = expected(p + "self_attn.k_proj.k_scale");
      const QwenExpectedTensor& ev = expected(p + "self_attn.v_proj.v_scale");
      float* kv = static_cast<float*>(bump.alloc(2 * sizeof(float)));
      if (copy) {
        const TensorInfo& tk = source(ek.name);
        const TensorInfo& tv = source(ev.name);
        float v[2];
        std::memcpy(&v[0], tk.data, 4);
        std::memcpy(&v[1], tv.data, 4);
        std::memcpy(bump.host(kv), v, sizeof(v));
        out.k_cache_scale = v[0];
        out.v_cache_scale = v[1];
        consumed(tk);
        consumed(tv);
      }
      note_read(ek, 4);
      note_read(ev, 4);
      out.kv_cache_scales = kv;
    }
    a.q_norm = load_bf16(p + "self_attn.q_norm.weight");
    a.k_norm = load_bf16(p + "self_attn.k_norm.weight");
  }

  // The routed MoE in Flash-Next's resident (models/qwen/loader.cpp's
  // build_moe is the model): router and shared gate replicated, the shared
  // expert at S/W (BF16, or block FP8 under dense_weights fp8), every routed
  // expert at I/W — rows of gate/up, columns of down.
  void build_moe_next(const std::string& p, bool is_mtp) {
    QwenMoeResident& m = out.moe;
    m.router = load_bf16(p + "mlp.gate.weight");
    m.shared_gate = load_bf16(p + "mlp.shared_expert_gate.weight");
    const int64_t H = cfg.hidden_size;
    const int64_t S = geo.local_shared_inter, I = geo.local_moe_inter;
    const int64_t r = geo.rank;
    m.local_inter = I;
    m.local_shared_inter = S;
    m.scale_block = std::gcd(128, static_cast<int>(I));
    const std::string sp = p + "mlp.shared_expert.";
    const int E = cfg.num_experts;
    const bool fp8 = g_dense_weights_fp8;
    if (is_mtp) {
      // The draft layer is BF16 throughout. Its shared expert loads as is;
      // its per-expert matrices are encoded to block FP8 at load
      // (loaders/fp8_quant.hpp) — the form the routed kernels read, and
      // what Flash-Next's loader does with a BF16 draft layer.
      if (fp8) {
        m.shared_fp8[0] = load_dense_rows_fp8(sp + "gate_proj.weight", r * S, S);
        m.shared_fp8[1] = load_dense_rows_fp8(sp + "up_proj.weight", r * S, S);
        m.shared_fp8[2] = load_dense_cols_fp8(sp + "down_proj.weight", r * S, S);
      } else {
        m.shared[0] = load_bf16_rows(sp + "gate_proj.weight", r * S, S);
        m.shared[1] = load_bf16_rows(sp + "up_proj.weight", r * S, S);
        m.shared[2] = load_bf16_cols(sp + "down_proj.weight", r * S, S);
      }
      m.experts.resize(static_cast<size_t>(E) * 3);
      for (int e = 0; e < E; ++e) {
        const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
        m.experts[static_cast<size_t>(e) * 3 + 0] =
            load_bf16_rows_fp8(ep + "gate_proj.weight", r * I, I);
        m.experts[static_cast<size_t>(e) * 3 + 1] =
            load_bf16_rows_fp8(ep + "up_proj.weight", r * I, I);
        m.experts[static_cast<size_t>(e) * 3 + 2] =
            load_bf16_cols_fp8(ep + "down_proj.weight", r * I, I);
      }
      return;
    }
    if (fp8) {
      m.shared_fp8[0] = load_fp4_rows_as_fp8(sp + "gate_proj", r * S, S);
      m.shared_fp8[1] = load_fp4_rows_as_fp8(sp + "up_proj", r * S, S);
      m.shared_fp8[2] = load_fp4_cols_as_fp8(sp + "down_proj", r * S, S);
    } else {
      m.shared[0] = load_fp4_as_bf16(sp + "gate_proj", r * S, S, 0, H);
      m.shared[1] = load_fp4_as_bf16(sp + "up_proj", r * S, S, 0, H);
      m.shared[2] = load_fp4_as_bf16(sp + "down_proj", 0, H, r * S, S);
    }
    // The backbone's routed experts stay NVFP4: the modelopt set per matrix,
    // sliced on the intermediate axis.
    m.experts_fp4.resize(static_cast<size_t>(E) * 3);
    m.expert_globals = static_cast<float*>(bump.alloc(static_cast<size_t>(E) * 3 * sizeof(float)));
    // The layer's static activation scales exist in the modelopt container
    // alone (load_fp4_slice): without them the W4A4 opt-in quantizes its
    // activations dynamically.
    const bool static_act = !fp4_packed();
    if (static_act) m.act_scales = static_cast<float*>(bump.alloc(2 * sizeof(float)));
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
      float* g = m.expert_globals + static_cast<size_t>(e) * 3;
      m.experts_fp4[static_cast<size_t>(e) * 3 + 0] =
          load_fp4_slice(ep + "gate_proj", r * I, I, 0, H, g + 0);
      m.act_scale_w13 = std::max(m.act_scale_w13, last_input_scale_);
      m.experts_fp4[static_cast<size_t>(e) * 3 + 1] =
          load_fp4_slice(ep + "up_proj", r * I, I, 0, H, g + 1);
      m.act_scale_w13 = std::max(m.act_scale_w13, last_input_scale_);
      m.experts_fp4[static_cast<size_t>(e) * 3 + 2] =
          load_fp4_slice(ep + "down_proj", 0, H, r * I, I, g + 2);
      m.act_scale_w2 = std::max(m.act_scale_w2, last_input_scale_);
    }
    // The layer's activation scales go into the image with the weights: a
    // resident-image restore lays the views out without the copy pass, so
    // host-side values would come back as 0.
    if (copy && static_act) {
      const float v[2] = {m.act_scale_w13, m.act_scale_w2};
      std::memcpy(bump.host(m.act_scales), v, sizeof(v));
    }
  }

  // The routed MoE of the Qwen3.5 dialect (Qwen3.6-35B-A3B), in the same
  // resident as the Qwen3Next dialect's.
  //   The NVFP4 mixed release's backbone is the modelopt NVFP4 set under the
  //   same names (the 80B's builder); its draft layer is BF16 — the shared
  //   expert as is (or block FP8 under dense_weights fp8) and the experts in
  //   the module's two stacked parameters, each expert's slice encoded to
  //   block FP8 at load as the 80B's per-expert draft matrices are.
  //   The FP8 release ships every matrix block FP8, the draft layer included:
  //   bound as shipped, the experts' sliced axis re-blocked at gcd(128, I/W).
  void build_moe35(const std::string& p, bool is_mtp) {
    if (fp8_per_tensor() && !is_mtp) {
      build_moe_next(p, false);
      return;
    }
    QwenMoeResident& m = out.moe;
    m.router = load_bf16(p + "mlp.gate.weight");
    m.shared_gate = load_bf16(p + "mlp.shared_expert_gate.weight");
    const int64_t H = cfg.hidden_size;
    const int64_t S = geo.local_shared_inter, I = geo.local_moe_inter;
    const int64_t r = geo.rank;
    m.local_inter = I;
    m.local_shared_inter = S;
    m.scale_block = std::gcd(128, static_cast<int>(I));
    const std::string sp = p + "mlp.shared_expert.";
    const int E = cfg.num_experts;
    m.experts.resize(static_cast<size_t>(E) * 3);
    if (fp8_per_tensor()) {
      if (g_dense_weights_fp8) {
        m.shared_fp8[0] = load_dense_rows_fp8(sp + "gate_proj.weight", r * S, S);
        m.shared_fp8[1] = load_dense_rows_fp8(sp + "up_proj.weight", r * S, S);
        m.shared_fp8[2] = load_dense_cols_fp8(sp + "down_proj.weight", r * S, S);
      } else {
        m.shared[0] = load_bf16_rows(sp + "gate_proj.weight", r * S, S);
        m.shared[1] = load_bf16_rows(sp + "up_proj.weight", r * S, S);
        m.shared[2] = load_bf16_cols(sp + "down_proj.weight", r * S, S);
      }
      // experts.gate_up_proj [E, 2 * I_full, H]: expert e's gate rows, then
      // its up rows; experts.down_proj [E, H, I_full]. Each tensor is read
      // once (this rank's slice of every expert) and accounted once.
      const int64_t I_full = cfg.moe_intermediate_size;
      const std::string gup = p + "mlp.experts.gate_up_proj", dwn = p + "mlp.experts.down_proj";
      const QwenExpectedTensor& eg = expected(gup);
      const QwenExpectedTensor& ed = expected(dwn);
      const uint16_t* gup_src = nullptr;
      const uint16_t* dwn_src = nullptr;
      if (copy) {
        gup_src = static_cast<const uint16_t*>(source(gup).data);
        dwn_src = static_cast<const uint16_t*>(source(dwn).data);
      }
      for (int e = 0; e < E; ++e) {
        const size_t eo = static_cast<size_t>(e) * 2 * static_cast<size_t>(I_full) * H;
        const size_t gate0 = eo + static_cast<size_t>(r * I) * H;
        const size_t up0 = eo + static_cast<size_t>(I_full + r * I) * H;
        const size_t down0 = static_cast<size_t>(e) * H * I_full + static_cast<size_t>(r * I);
        m.experts[static_cast<size_t>(e) * 3 + 0] =
            encode_fp8(copy ? gup_src + gate0 : nullptr, static_cast<size_t>(H), I, H);
        m.experts[static_cast<size_t>(e) * 3 + 1] =
            encode_fp8(copy ? gup_src + up0 : nullptr, static_cast<size_t>(H), I, H);
        m.experts[static_cast<size_t>(e) * 3 + 2] =
            encode_fp8(copy ? dwn_src + down0 : nullptr, static_cast<size_t>(I_full), H, I);
      }
      if (copy) {
        consumed(source(gup));
        consumed(source(dwn));
      }
      note_read(eg, static_cast<size_t>(E) * 2 * static_cast<size_t>(I) * H * 2);
      note_read(ed, static_cast<size_t>(E) * H * static_cast<size_t>(I) * 2);
      return;
    }
    m.shared_fp8[0] = load_fp8_native_rows(sp + "gate_proj.weight", r * S, S);
    m.shared_fp8[1] = load_fp8_native_rows(sp + "up_proj.weight", r * S, S);
    m.shared_fp8[2] = load_fp8_native_cols(sp + "down_proj.weight", r * S, S);
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
      m.experts[static_cast<size_t>(e) * 3 + 0] =
          load_quant_rows(ep + "gate_proj.weight", r * I, I, m.scale_block);
      m.experts[static_cast<size_t>(e) * 3 + 1] =
          load_quant_rows(ep + "up_proj.weight", r * I, I, m.scale_block);
      m.experts[static_cast<size_t>(e) * 3 + 2] =
          load_quant_cols(ep + "down_proj.weight", r * I, I, m.scale_block);
    }
  }

  void build_layer(int layer) {
    const int mtp_layer = cfg.mtp_layer();
    if (layer < 0 || layer >= cfg.num_hidden_layers + (mtp_layer >= 0 ? 1 : 0))
      fail("build_layer: layer out of range");
    const bool is_mtp = layer == mtp_layer;
    const Qwen35LayerKind kind = is_mtp ? Qwen35LayerKind::Full : cfg.layers[layer];
    const std::string p = qwen35_layer_prefix(cfg, layer);
    if (loader_verbose())
      std::fprintf(stderr, "[qwen35] build_layer %d kind=%d prefix=%s\n", layer, (int)kind, p.c_str());
    Qwen35LayerResident& o = out;
    o.kind = kind;
    o.layer = layer;
    o.input_norm = load_bf16(p + "input_layernorm.weight");
    o.post_norm = load_bf16(p + "post_attention_layernorm.weight");
    if (cfg.next()) {
      if (kind == Qwen35LayerKind::Gdn)
        build_gdn_next(p);
      else
        build_full_next(p, is_mtp);
      build_moe_next(p, is_mtp);
    } else {
      if (kind == Qwen35LayerKind::Gdn)
        build_gdn(p);
      else
        build_full(p, is_mtp);
      if (cfg.moe())
        build_moe35(p, is_mtp);
      else
        build_mlp(p);
    }
    if (loader_verbose())
      std::fprintf(stderr, "[qwen35] layer %d done (kind=%d)\n", layer, (int)kind);
  }
};

const char* Qwen35LoaderFamily::who() { return "qwen35 loader"; }

// The resident layout's version; the Qwen3Next dense stack's form is part of
// it (bit 2), as Flash-Next's family folds its own into loader_format(): the
// resident image key mixes this value, so a BF16-form image is never
// restored into an FP8-form stream or the reverse — the other form builds
// from the checkpoint into its own image file.
uint64_t Qwen35LoaderFamily::loader_format() {
  return 2 | (g_dense_weights_fp8 ? 4 : 0);
}

int Qwen35LoaderFamily::max_layer(const Config& c) {
  return c.num_hidden_layers + (c.mtp_layer() >= 0 ? 1 : 0);
}

int Qwen35LoaderFamily::main_layers(const Config& c) { return c.num_hidden_layers; }

std::vector<Qwen35LoaderFamily::Expected> Qwen35LoaderFamily::layer_table(const Config& c,
                                                                          int layer) {
  return qwen35_expected_layer_tensors(c, layer);
}

std::vector<Qwen35LoaderFamily::Expected> Qwen35LoaderFamily::global_table(const Config& c) {
  return qwen35_expected_global_tensors(c);
}

void Qwen35LoaderFamily::validate_binding(const Config& c, const PresentMap& present) {
  const QwenBindReport rep = qwen35_validate_text_binding(c, present);
  if (rep.ok()) return;
  std::string msg = "qwen35 loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

void Qwen35LoaderFamily::check_sources(const Config&, const LoaderTensorMap&) {
  // No PLE n-gram hash buffers on this family: nothing to check beyond the
  // binding table (names, dtypes, shapes), already gated above.
}

bool Qwen35LoaderFamily::digest_included(const Expected& e) { return is_replicated_35(e); }

// The packed column slices read their sources after the builder returns:
// the Qwen3Next draft layer's BF16 o_proj and shared down projection among
// them.
bool Qwen35LoaderFamily::discard_after_pack(const Expected& e) {
  return e.cls == QwenWeightClass::Gdn || e.cls == QwenWeightClass::FullAttn ||
         e.cls == QwenWeightClass::DenseMlp || e.cls == QwenWeightClass::SharedExpert;
}

size_t Qwen35LoaderFamily::globals_bytes(const Config& c, int rank, int world,
                                         LoaderHeadSharding) {
  const size_t H = static_cast<size_t>(c.hidden_size);
  const size_t V = static_cast<size_t>(c.vocab_size);
  const size_t Vn = static_cast<size_t>(V * (rank + 1) / world) - static_cast<size_t>(V * rank / world);
  size_t b = 0;
  b += align_up_256(V * H * 2);   // embed
  b += align_up_256(Vn * H * 2);  // lm head shard
  b += align_up_256(H * 2);       // final norm
  if (c.mtp_layer() >= 0) {
    b += align_up_256(H * 2 * H * 2);  // mtp.fc [H, 2H] BF16
    b += align_up_256(H * 2);          // mtp.norm
    b += align_up_256(H * 2);          // mtp.pre_fc_norm_embedding
    b += align_up_256(H * 2);          // mtp.pre_fc_norm_hidden
  }
  return b;
}

size_t Qwen35LoaderFamily::extra_resident_bytes(const Config&, int, int) { return 0; }

// The Qwen3.5 dialect keeps nothing on the host. A restored Qwen3Next
// attention layer's K/V-cache scales have host copies the layout pass
// cannot fill: from the source when mapped (the image carries the device
// pair). The MoE's activation scales stay device-only after a restore, as
// in Flash-Next's family.
void Qwen35LoaderFamily::after_restore(const Config& c, int layer, const LoaderTensorMap& tensors,
                                       LayerResident& out) {
  if (out.kv_cache_scales == nullptr || tensors.empty()) return;
  const std::string p = qwen35_layer_prefix(c, layer) + "self_attn.";
  std::memcpy(&out.k_cache_scale, tensors.at(p + "k_proj.k_scale")->data, 4);
  std::memcpy(&out.v_cache_scale, tensors.at(p + "v_proj.v_scale")->data, 4);
}

void Qwen35LoaderFamily::build_globals(const Config& c, const Geometry& geo,
                                       const LoaderTensorMap& tensors, LayerBump& bump,
                                       GlobalsResident& out, uint64_t& source_bytes,
                                       uint64_t& verbatim_bytes, LoaderHeadSharding) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("qwen35 loader: global tensor missing: " + name);
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
  const std::string mp = qwen35_model_prefix(c);  // flat names in the Qwen3Next dialect
  out.embed = copy_global(mp + "embed_tokens.weight");
  const int64_t V = c.vocab_size;
  const int64_t V0 = V * geo.rank / geo.world;
  const int64_t Vn = V * (geo.rank + 1) / geo.world - V0;
  const size_t H = static_cast<size_t>(c.hidden_size);
  const TensorInfo& lm = lookup("lm_head.weight");
  uint16_t* head = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(Vn) * H * 2));
  if (c.quant_kind == Qwen35QuantKind::Nvfp4Mixed) {
    // The NVFP4 mixed release quantizes the head (the modelopt set over
    // [vocab, H]): this rank's vocab rows dequantized on the host into the
    // BF16 head every head path reads, as the dense NVFP4 matrices are.
    if (H % kFp4Group != 0)
      throw std::runtime_error("qwen35 loader: an NVFP4 lm_head needs hidden_size % 16 == 0");
    const TensorInfo& ls = lookup("lm_head.weight_scale");
    float ws2;
    std::memcpy(&ws2, lookup("lm_head.weight_scale_2").data, 4);
    if (!(ws2 > 0.0f) || !std::isfinite(ws2))
      throw std::runtime_error("qwen35 loader: 'lm_head.weight_scale_2' is not a positive finite scale");
    const size_t pc = H / 2, sc = H / static_cast<size_t>(kFp4Group);
    qwen3next_fp4_dequant_bf16(static_cast<const uint8_t*>(lm.data) + static_cast<size_t>(V0) * pc, pc,
                               static_cast<const uint8_t*>(ls.data) + static_cast<size_t>(V0) * sc, sc,
                               ws2, Vn, static_cast<int64_t>(H), bump.host(head));
    source_bytes += static_cast<uint64_t>(Vn) * (pc + sc) + 4;
    verbatim_bytes += static_cast<uint64_t>(Vn) * (pc + sc) + 4;
  } else {
    std::memcpy(reinterpret_cast<uint8_t*>(bump.host(head)),
                static_cast<const uint8_t*>(lm.data) + static_cast<size_t>(V0) * H * 2,
                static_cast<size_t>(Vn) * H * 2);
    source_bytes += static_cast<uint64_t>(Vn) * H * 2;
    verbatim_bytes += static_cast<uint64_t>(Vn) * H * 2;
  }
  out.lm_head = head;
  out.lm_vocab_begin = geo.lm_vocab_begin;
  out.lm_vocab_count = geo.lm_vocab_count;
  out.final_norm = copy_global(mp + "norm.weight");
  if (c.mtp_layer() >= 0) {
    out.mtp_fc = copy_global("mtp.fc.weight");
    out.mtp_norm = copy_global("mtp.norm.weight");
    out.mtp_pre_fc_norm_embedding = copy_global("mtp.pre_fc_norm_embedding.weight");
    out.mtp_pre_fc_norm_hidden = copy_global("mtp.pre_fc_norm_hidden.weight");
  } else {
    out.mtp_fc = nullptr;
    out.mtp_norm = nullptr;
    out.mtp_pre_fc_norm_embedding = nullptr;
    out.mtp_pre_fc_norm_hidden = nullptr;
  }
}

std::string& resident_image_dir_storage_35() {
  static std::string dir;
  return dir;
}

Qwen35LayerStream::Qwen35LayerStream(const Qwen35TextConfig& cfg, const std::string& checkpoint_dir,
                                     int rank, int world, LoaderResidency residency,
                                     LoaderHeadSharding head, bool resident_mtp)
    : ResidentLayerStream<Qwen35LoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                              resident_mtp) {
  if (resident_image_dir().empty()) resident_image_dir_storage_35() = checkpoint_dir + "/resident_qwen35";
  open_resident_image();
}

void Qwen35LayerStream::set_resident_image_dir(const std::string& dir) {
  resident_image_dir_storage_35() = dir;
}

const std::string& Qwen35LayerStream::resident_image_dir() { return resident_image_dir_storage_35(); }

const std::string& Qwen35LayerStream::image_dir() const { return resident_image_dir(); }

void Qwen35LayerStream::set_dense_weights_fp8(bool on) {
  g_dense_weights_fp8 = on;
}
bool Qwen35LayerStream::dense_weights_fp8() {
  return g_dense_weights_fp8;
}

template class ResidentLayerStream<Qwen35LoaderFamily>;

}  // namespace dgpp
