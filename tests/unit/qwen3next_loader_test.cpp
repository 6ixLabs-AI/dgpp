// The Qwen3Next dialect of the qwen3_5 loader family, host side: the GDN
// gather (the checkpoint's per-key-head interleave to the head-major rows
// the layer binds), the local geometry at worlds 1, 2 and 4, the NVFP4 to
// BF16 dequant, and the counting build — at world 1 it reads every byte of
// a layer's table, and the ranks of a world tile it — in both forms of the
// dense stack (BF16, and block FP8 under set_dense_weights_fp8). The copy
// pass against the release is tests/cuda/qwen3next_loader_smoke.cpp.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "loaders/nvfp4_quant.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/loader35.hpp"

namespace {

using dgpp::Qwen35LayerStream;
using dgpp::Qwen35LoaderFamily;
using dgpp::Qwen35LocalGeometry;
using dgpp::Qwen35RowRun;
using dgpp::Qwen35TextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The release's shape (nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4), through
// the parser.
Qwen35TextConfig release_config() {
  std::string layers;
  for (int i = 0; i < 48; ++i) {
    if (i) layers += ", ";
    layers += (i + 1) % 4 == 0 ? "\"full_attention\"" : "\"linear_attention\"";
  }
  const std::string json = std::string(R"({
  "architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next",
  "bos_token_id": 151643, "eos_token_id": 151645, "decoder_sparse_step": 1,
  "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu", "hidden_size": 2048,
  "intermediate_size": 5120, "layer_types": [)") +
                           layers + R"(],
  "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
  "linear_num_value_heads": 32, "linear_value_head_dim": 128,
  "max_position_embeddings": 262144, "mlp_only_layers": [], "moe_intermediate_size": 512,
  "norm_topk_prob": true, "num_attention_heads": 16, "num_experts": 512,
  "num_experts_per_tok": 10, "num_hidden_layers": 48, "num_key_value_heads": 2,
  "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06, "rope_scaling": null,
  "rope_theta": 10000000, "shared_expert_intermediate_size": 512,
  "tie_word_embeddings": false, "vocab_size": 151936,
  "quantization_config": {"config_groups": {"group_0": {
    "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
    "targets": ["Linear"]}}}
})";
  const auto t = dgpp::minijson::parse(json);
  return Qwen35TextConfig::parse_qwen3_next(t.root);
}

// The source row of every destination row of a gathered matrix; -1 where
// no run lands, and a throw where two do.
std::vector<int64_t> sources(const std::vector<Qwen35RowRun>& runs, int64_t dst_rows) {
  std::vector<int64_t> src(static_cast<size_t>(dst_rows), -1);
  for (const Qwen35RowRun& run : runs)
    for (int64_t i = 0; i < run.rows; ++i) {
      require(run.dst_row + i >= 0 && run.dst_row + i < dst_rows, "a run lands outside its matrix");
      int64_t& slot = src[static_cast<size_t>(run.dst_row + i)];
      require(slot == -1, "two runs land on one destination row");
      slot = run.src_row + i;
    }
  return src;
}

// The interleave, written from the checkpoint's layout: key head g owns
// rows [g * G, +G) of in_proj_qkvz as [q dk | k dk | v r*dv | z r*dv] and
// rows [g * 2r, +2r) of in_proj_ba as [b x r | a x r], r value heads a key
// head. Returns false at the first destination row whose source differs.
bool gather_is_the_interleave(const Qwen35TextConfig& c, int rank, int world) {
  const int64_t nk = c.gdn_key_heads, nv = c.gdn_value_heads;
  const int64_t dk = c.gdn_key_head_dim, dv = c.gdn_value_head_dim;
  const int64_t r = nv / nk, G = 2 * dk + 2 * r * dv;
  const int64_t lk = nk / world, lv = nv / world;
  const int64_t k0 = rank * lk, v0 = rank * lv;
  const dgpp::Qwen3NextGdnGather m = dgpp::qwen3next_gdn_gather(c, rank, world);
  const std::vector<int64_t> qkv = sources(m.qkv, 2 * lk * dk + lv * dv);
  const std::vector<int64_t> z = sources(m.z, lv * dv);
  const std::vector<int64_t> a = sources(m.a, lv), b = sources(m.b, lv);
  for (int64_t h = 0; h < lk; ++h)
    for (int64_t o = 0; o < dk; ++o) {
      if (qkv[static_cast<size_t>(h * dk + o)] != (k0 + h) * G + o) return false;
      if (qkv[static_cast<size_t>(lk * dk + h * dk + o)] != (k0 + h) * G + dk + o) return false;
    }
  for (int64_t h = 0; h < lv; ++h) {
    const int64_t v = v0 + h, g = v / r, i = v % r;  // value head v is key head g's i-th
    for (int64_t o = 0; o < dv; ++o) {
      if (qkv[static_cast<size_t>(2 * lk * dk + h * dv + o)] != g * G + 2 * dk + i * dv + o)
        return false;
      if (z[static_cast<size_t>(h * dv + o)] != g * G + 2 * dk + r * dv + i * dv + o) return false;
    }
    if (b[static_cast<size_t>(h)] != g * 2 * r + i) return false;
    if (a[static_cast<size_t>(h)] != g * 2 * r + r + i) return false;
  }
  return true;
}

// e2m1 and e4m3 from their definitions (not the engine's converters).
float e2m1(uint8_t code) {
  static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return (code & 8u) ? -mag[code & 7u] : mag[code & 7u];
}
float e4m3(uint8_t code) {
  const int e = (code >> 3) & 0xF, m = code & 7;
  const float v = e == 0 ? std::ldexp(static_cast<float>(m), -9)
                         : std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
  return (code & 0x80u) ? -v : v;
}

// The Qwen3Next dense stack's form for the duration of a scope.
struct DenseForm {
  bool saved = Qwen35LayerStream::dense_weights_fp8();
  explicit DenseForm(bool fp8) { Qwen35LayerStream::set_dense_weights_fp8(fp8); }
  ~DenseForm() { Qwen35LayerStream::set_dense_weights_fp8(saved); }
};

// What a dense [n, k] matrix gives back when its BF16 grant becomes block
// FP8: one byte a code and an fp32 scale per 128 x 128 block.
size_t fp8_saving(int64_t n, int64_t k) {
  const size_t bf16 = dgpp::align_up_256(static_cast<size_t>(n * k) * 2);
  const size_t fp8 =
      dgpp::align_up_256(static_cast<size_t>(n * k)) +
      dgpp::align_up_256(static_cast<size_t>(((n + 127) / 128) * ((k + 127) / 128)) * 4);
  return bf16 - fp8;
}

// The dense projections of a layer at world `world` and what they save.
size_t layer_fp8_saving(const Qwen35TextConfig& c, int layer, int world) {
  const int64_t H = c.hidden_size, S = c.shared_expert_intermediate_size / world;
  size_t n = 2 * fp8_saving(S, H) + fp8_saving(H, S);  // the shared expert
  if (layer != c.mtp_layer() && c.layers[layer] == dgpp::Qwen35LayerKind::Gdn) {
    const int64_t K = static_cast<int64_t>(c.gdn_key_heads) * c.gdn_key_head_dim / world;
    const int64_t V = static_cast<int64_t>(c.gdn_value_heads) * c.gdn_value_head_dim / world;
    return n + fp8_saving(2 * K + V, H) + fp8_saving(V, H) + fp8_saving(H, V);  // qkv, z, out
  }
  const int64_t q = static_cast<int64_t>(c.num_attention_heads) / world * c.head_dim;
  const int64_t kv = static_cast<int64_t>(std::max(1, c.num_key_value_heads / world)) * c.head_dim;
  return n + fp8_saving(2 * q, H) + 2 * fp8_saving(kv, H) + fp8_saving(H, q);  // q, k, v, o
}

size_t table_bytes(const Qwen35TextConfig& c, int layer, bool replicated_only) {
  size_t n = 0;
  for (const auto& e : Qwen35LoaderFamily::layer_table(c, layer))
    if (!replicated_only || Qwen35LoaderFamily::digest_included(e)) n += e.nbytes();
  return n;
}

}  // namespace

DGPP_TEST(qwen3next_gdn_gather_maps_the_interleave_to_head_major_rows) {
  const Qwen35TextConfig c = release_config();
  for (int world : {1, 2, 4, 8, 16})
    for (int rank = 0; rank < world; ++rank)
      require(gather_is_the_interleave(c, rank, world),
              "gather at world " + std::to_string(world) + " rank " + std::to_string(rank));

  // The release's numbers, spelled out: 768-row groups, two value heads a key head.
  const dgpp::Qwen3NextGdnGather m = dgpp::qwen3next_gdn_gather(c, 0, 1);
  const std::vector<int64_t> qkv = sources(m.qkv, 8192), z = sources(m.z, 4096);
  const std::vector<int64_t> a = sources(m.a, 32), b = sources(m.b, 32);
  require(qkv[3 * 128 + 5] == 3 * 768 + 5, "q of key head 3");
  require(qkv[2048 + 3 * 128 + 5] == 3 * 768 + 128 + 5, "k of key head 3");
  require(qkv[4096 + 7 * 128 + 9] == 3 * 768 + 256 + 128 + 9, "v of value head 7");
  require(z[7 * 128 + 9] == 3 * 768 + 512 + 128 + 9, "z of value head 7");
  require(b[7] == 3 * 4 + 1 && a[7] == 3 * 4 + 3, "b and a of value head 7");

  // World 1 reads every source row exactly once: q | k | v | z cover
  // in_proj_qkvz, b | a cover in_proj_ba.
  std::vector<int> seen(12288, 0), seen_ba(64, 0);
  for (int64_t s : qkv) ++seen[static_cast<size_t>(s)];
  for (int64_t s : z) ++seen[static_cast<size_t>(s)];
  for (int64_t s : a) ++seen_ba[static_cast<size_t>(s)];
  for (int64_t s : b) ++seen_ba[static_cast<size_t>(s)];
  for (int n : seen) require(n == 1, "in_proj_qkvz row read once");
  for (int n : seen_ba) require(n == 1, "in_proj_ba row read once");

  // The ranks of a world tile the world-1 matrices: rank r's local rows are
  // the global head-major rows of its heads.
  for (int world : {2, 4}) {
    const int64_t lkd = 2048 / world, lvd = 4096 / world, lv = 32 / world;
    for (int rank = 0; rank < world; ++rank) {
      const dgpp::Qwen3NextGdnGather mr = dgpp::qwen3next_gdn_gather(c, rank, world);
      const std::vector<int64_t> rq = sources(mr.qkv, 2 * lkd + lvd), rz = sources(mr.z, lvd);
      const std::vector<int64_t> ra = sources(mr.a, lv), rb = sources(mr.b, lv);
      for (int64_t i = 0; i < lkd; ++i) {
        require(rq[static_cast<size_t>(i)] == qkv[static_cast<size_t>(rank * lkd + i)], "q tiles");
        require(rq[static_cast<size_t>(lkd + i)] == qkv[static_cast<size_t>(2048 + rank * lkd + i)],
                "k tiles");
      }
      for (int64_t i = 0; i < lvd; ++i) {
        require(
            rq[static_cast<size_t>(2 * lkd + i)] == qkv[static_cast<size_t>(4096 + rank * lvd + i)],
            "v tiles");
        require(rz[static_cast<size_t>(i)] == z[static_cast<size_t>(rank * lvd + i)], "z tiles");
      }
      for (int64_t i = 0; i < lv; ++i)
        require(ra[static_cast<size_t>(i)] == a[static_cast<size_t>(rank * lv + i)] &&
                    rb[static_cast<size_t>(i)] == b[static_cast<size_t>(rank * lv + i)],
                "a and b tile");
    }
  }

  // Other head ratios follow the same rule (one and three value heads a key head).
  Qwen35TextConfig one = c, three = c;
  one.gdn_value_heads = 16;
  three.gdn_key_heads = 4;
  three.gdn_value_heads = 12;
  for (int world : {1, 2, 4})
    for (int rank = 0; rank < world; ++rank)
      require(gather_is_the_interleave(one, rank, world) &&
                  gather_is_the_interleave(three, rank, world),
              "gather at other head ratios");

  bool refused = false;
  try {
    (void)dgpp::qwen3next_gdn_gather(c, 0, 3);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "a world that does not divide the heads is refused");
}

DGPP_TEST(qwen3next_loader_geometry_at_worlds_1_2_4) {
  const Qwen35TextConfig c = release_config();
  const auto geo = [&](int rank, int world) {
    return Qwen35LocalGeometry::from_config(c, rank, world, dgpp::LoaderHeadSharding::Full);
  };
  {
    const Qwen35LocalGeometry g = geo(0, 1);
    require(g.local_key_heads == 16 && g.local_value_heads == 32, "world 1 gdn heads");
    require(g.local_heads == 16 && g.head_begin == 0, "world 1 query heads");
    require(g.local_kv_heads == 2 && g.kv_head_begin == 0, "world 1 kv heads");
    require(g.local_moe_inter == 512 && g.local_shared_inter == 512, "world 1 moe slices");
    require(g.lm_vocab_begin == 0 && g.lm_vocab_count == 151936, "world 1 vocab");
  }
  {
    const Qwen35LocalGeometry g = geo(1, 2);
    require(g.local_key_heads == 8 && g.local_value_heads == 16, "world 2 gdn heads");
    require(g.local_heads == 8 && g.head_begin == 8, "world 2 query heads");
    require(g.local_kv_heads == 1 && g.kv_head_begin == 1, "world 2 kv head");
    require(g.local_moe_inter == 256 && g.local_shared_inter == 256, "world 2 moe slices");
    require(g.lm_vocab_begin == 75968 && g.lm_vocab_count == 75968, "world 2 vocab");
  }
  {
    // Two kv heads on four ranks: a kv head lives on two of them.
    const Qwen35LocalGeometry g1 = geo(1, 4), g3 = geo(3, 4);
    require(g3.local_key_heads == 4 && g3.local_value_heads == 8, "world 4 gdn heads");
    require(g1.local_heads == 4 && g1.head_begin == 4 && g3.head_begin == 12,
            "world 4 query heads");
    require(g1.local_kv_heads == 1 && g1.kv_head_begin == 0, "world 4 rank 1 shares kv head 0");
    require(g3.local_kv_heads == 1 && g3.kv_head_begin == 1, "world 4 rank 3 shares kv head 1");
    require(g3.local_moe_inter == 128 && g3.local_shared_inter == 128, "world 4 moe slices");
    require(g3.lm_vocab_begin == 113952 && g3.lm_vocab_count == 37984, "world 4 vocab");
  }
  bool refused = false;
  try {
    (void)geo(0, 3);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "world 3 is refused");

  // The dense dialect has no MoE slices.
  Qwen35TextConfig dense;
  const Qwen35LocalGeometry d =
      Qwen35LocalGeometry::from_config(dense, 1, 2, dgpp::LoaderHeadSharding::Full);
  require(d.local_moe_inter == 0 && d.local_shared_inter == 0, "qwen3.5 moe slices stay empty");
  require(d.local_inter == dense.intermediate_size / 2, "qwen3.5 dense slice unchanged");
}

DGPP_TEST(qwen3next_fp4_dequant_is_code_times_block_scale_times_tensor_scale) {
  // Nibble order and scale direction on one block: byte 0x21 holds code 1
  // (0.5) for the even column and code 2 (1.0) for the odd one; 0x38 is the
  // e4m3 code of 1.0; weight_scale_2 multiplies.
  {
    std::vector<uint8_t> payload(8, 0x21), scales(1, 0x38);
    payload[1] = 0xF7;  // +6 (even), -6 (odd)
    std::vector<uint16_t> out(16);
    dgpp::qwen3next_fp4_dequant_bf16(payload.data(), 8, scales.data(), 1, 1.0f, 1, 16, out.data());
    require(out[0] == 0x3F00 && out[1] == 0x3F80, "the low nibble is the even column");
    require(out[2] == 0x40C0 && out[3] == 0xC0C0, "sign bit and the largest code");
    dgpp::qwen3next_fp4_dequant_bf16(payload.data(), 8, scales.data(), 1, 0.25f, 1, 16, out.data());
    require(out[2] == 0x3FC0, "weight_scale_2 multiplies (6 x 0.25 = 1.5)");
  }

  // A matrix of arbitrary codes against the formula written out: the exact
  // fp32 product code x block scale, one fp32 multiply by the tensor scale,
  // rounded to bf16.
  constexpr int64_t rows = 6, cols = 96;
  std::vector<uint8_t> payload(rows * cols / 2), scales(rows * cols / 16);
  uint32_t state = 0x9E3779B9u;
  const auto next = [&state] {
    state = state * 1664525u + 1013904223u;
    return static_cast<uint8_t>(state >> 24);
  };
  for (auto& b : payload) b = next();
  for (auto& s : scales) s = static_cast<uint8_t>(next() & 0x77u);  // positive, never the NaN code
  const float ws2 = 0.00123456f;
  std::vector<uint16_t> full(rows * cols);
  dgpp::qwen3next_fp4_dequant_bf16(payload.data(), cols / 2, scales.data(), cols / 16, ws2, rows,
                                   cols, full.data());
  for (int64_t n = 0; n < rows; ++n)
    for (int64_t k = 0; k < cols; ++k) {
      const uint8_t byte = payload[static_cast<size_t>(n * (cols / 2) + k / 2)];
      const uint8_t code =
          (k & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0xF);
      const float block = e2m1(code) * e4m3(scales[static_cast<size_t>(n * (cols / 16) + k / 16)]);
      const float value = block * ws2;
      require(full[static_cast<size_t>(n * cols + k)] == dgpp::float_to_bf16_bits(value),
              "element " + std::to_string(n) + "," + std::to_string(k));
      // The reciprocal form the expert kernels use agrees to the format's precision.
      const float kernel =
          dgpp::nvfp4_decode(payload.data(), scales.data(), 1.0f / ws2, cols, n, k);
      require(std::fabs(kernel - value) <= std::fabs(value) * 1e-6f,
              "the expert kernels' convention");
    }

  // A slice (rows 1..4, columns 32..80) is the same elements.
  std::vector<uint16_t> part(3 * 48);
  dgpp::qwen3next_fp4_dequant_bf16(payload.data() + 1 * (cols / 2) + 32 / 2, cols / 2,
                                   scales.data() + 1 * (cols / 16) + 32 / 16, cols / 16, ws2, 3, 48,
                                   part.data());
  for (int64_t n = 0; n < 3; ++n)
    for (int64_t k = 0; k < 48; ++k)
      require(part[static_cast<size_t>(n * 48 + k)] ==
                  full[static_cast<size_t>((n + 1) * cols + 32 + k)],
              "a slice dequantizes to the same elements");

  // The engine's NVFP4 encoder round-trips through it: every element lands
  // within half the widest e2m1 step (4 -> 6, in units of the block scale).
  std::vector<float> row(cols);
  float amax = 0.0f;
  for (int64_t k = 0; k < cols; ++k) {
    row[static_cast<size_t>(k)] = (static_cast<float>(next()) - 127.5f) * 3e-4f;
    amax = std::max(amax, std::fabs(row[static_cast<size_t>(k)]));
  }
  const float enc_ws2 = dgpp::nvfp4_tensor_scale(amax);
  std::vector<uint8_t> ep(cols / 2), es(cols / 16);
  dgpp::nvfp4_encode_row(row.data(), cols, enc_ws2, ep.data(), es.data());
  std::vector<uint16_t> back(cols);
  dgpp::qwen3next_fp4_dequant_bf16(ep.data(), cols / 2, es.data(), cols / 16, enc_ws2, 1, cols,
                                   back.data());
  for (int64_t k = 0; k < cols; ++k) {
    const float S = e4m3(es[static_cast<size_t>(k / 16)]) * enc_ws2;
    const float got = dgpp::bf16_bits_to_float(back[static_cast<size_t>(k)]);
    require(std::fabs(got - row[static_cast<size_t>(k)]) <= S * 1.02f, "encoder round trip");
  }
}

DGPP_TEST(qwen3next_loader_counting_build_reads_the_whole_layer_and_ranks_tile_it) {
  const Qwen35TextConfig c = release_config();
  const int draft = c.mtp_layer();
  for (int layer : {0, 3, draft}) {
    const uint64_t full = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 1);
    require(full == table_bytes(c, layer, false),
            "world 1 reads every byte of layer " + std::to_string(layer) + "'s table");
    // Two ranks read the layer once between them, plus the replicated set
    // once more.
    const uint64_t r0 = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 2);
    const uint64_t r1 = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 1, 2);
    require(r0 == r1, "the ranks of world 2 read alike");
    require(r0 + r1 == full + table_bytes(c, layer, true),
            "world 2 tiles layer " + std::to_string(layer));
  }
  {
    // World 4, a GDN layer: four slices and the replicated set four times.
    uint64_t sum = 0;
    for (int r = 0; r < 4; ++r) sum += Qwen35LayerStream::planned_layer_source_bytes(c, 0, r, 4);
    require(sum == table_bytes(c, 0, false) + 3 * table_bytes(c, 0, true),
            "world 4 tiles a GDN layer");
    // An attention layer: each of the two kv heads lives on two ranks, so
    // k_proj and v_proj are read twice.
    sum = 0;
    for (int r = 0; r < 4; ++r) sum += Qwen35LayerStream::planned_layer_source_bytes(c, 3, r, 4);
    const uint64_t kv_rows = static_cast<uint64_t>(c.num_key_value_heads) * c.head_dim;
    require(sum == table_bytes(c, 3, false) + 3 * table_bytes(c, 3, true) +
                       2 * kv_rows * static_cast<uint64_t>(c.hidden_size) * 2,
            "world 4 shares the kv heads of an attention layer");
  }

  // The layout: one byte count per layer kind, equal across the ranks of a
  // world, and the stream's total is their sum.
  const size_t gdn = Qwen35LayerStream::layer_bytes(c, 0),
               attn = Qwen35LayerStream::layer_bytes(c, 3);
  for (int l = 0; l < c.num_hidden_layers; ++l)
    require(Qwen35LayerStream::layer_bytes(c, l) ==
                (c.layers[l] == dgpp::Qwen35LayerKind::Gdn ? gdn : attn),
            "layer " + std::to_string(l) + " bytes follow its kind");
  for (int layer : {0, 3, draft})
    require(Qwen35LayerStream::layer_bytes(c, layer, 0, 2) ==
                    Qwen35LayerStream::layer_bytes(c, layer, 1, 2) &&
                Qwen35LayerStream::layer_bytes(c, layer, 0, 2) <
                    Qwen35LayerStream::layer_bytes(c, layer),
            "world 2 halves are equal");
  const size_t globals = Qwen35LayerStream::globals_bytes(c);
  require(Qwen35LayerStream::resident_bytes(c, 0, 1, dgpp::LoaderHeadSharding::Full,
                                            /*with_mtp=*/true) ==
              36 * gdn + 12 * attn + Qwen35LayerStream::layer_bytes(c, draft) + globals,
          "resident bytes are the layers, the draft layer and the globals");
}

DGPP_TEST(qwen3next_loader_replicated_set) {
  const Qwen35TextConfig c = release_config();
  const auto replicated = [&](int layer, const std::string& suffix) {
    const std::string name = dgpp::qwen35_layer_prefix(c, layer) + suffix;
    for (const auto& e : Qwen35LoaderFamily::layer_table(c, layer))
      if (e.name == name) return Qwen35LoaderFamily::digest_included(e);
    throw std::runtime_error("no tensor " + name);
  };
  // Replicated: norms, router and shared gate, every NVFP4 matrix's two
  // scalars, the K/V-cache scales.
  require(replicated(0, "input_layernorm.weight") && replicated(0, "linear_attn.norm.weight"),
          "norms");
  require(replicated(0, "mlp.gate.weight") && replicated(0, "mlp.shared_expert_gate.weight"),
          "router");
  require(replicated(0, "linear_attn.out_proj.weight_scale_2") &&
              replicated(0, "linear_attn.out_proj.input_scale") &&
              replicated(0, "mlp.experts.7.down_proj.weight_scale_2") &&
              replicated(0, "mlp.experts.7.down_proj.input_scale") &&
              replicated(3, "mlp.shared_expert.up_proj.weight_scale_2"),
          "NVFP4 scalars");
  require(replicated(3, "self_attn.k_proj.k_scale") && replicated(3, "self_attn.v_proj.v_scale") &&
              replicated(3, "self_attn.q_norm.weight"),
          "attention scalars and norms");
  // Sliced: every projection, the NVFP4 payloads and block scales.
  require(!replicated(0, "linear_attn.in_proj_qkvz.weight") &&
              !replicated(0, "linear_attn.in_proj_ba.weight") &&
              !replicated(0, "linear_attn.conv1d.weight") && !replicated(0, "linear_attn.A_log") &&
              !replicated(0, "linear_attn.out_proj.weight") &&
              !replicated(0, "linear_attn.out_proj.weight_scale"),
          "GDN slices");
  require(!replicated(3, "self_attn.q_proj.weight") && !replicated(3, "self_attn.k_proj.weight") &&
              !replicated(3, "self_attn.o_proj.weight") &&
              !replicated(3, "mlp.shared_expert.down_proj.weight") &&
              !replicated(3, "mlp.experts.0.gate_proj.weight") &&
              !replicated(3, "mlp.experts.0.gate_proj.weight_scale"),
          "attention and MoE slices");
  require(!replicated(c.mtp_layer(), "mlp.experts.0.gate_proj.weight") &&
              !replicated(c.mtp_layer(), "self_attn.o_proj.weight"),
          "draft layer slices");
}

DGPP_TEST(qwen3next_loader_dense_fp8_counting_build) {
  const Qwen35TextConfig c = release_config();
  const int draft = c.mtp_layer();
  const auto fmt = [] { return Qwen35LoaderFamily::loader_format(); };
  require(!Qwen35LayerStream::dense_weights_fp8(), "the dense stack is BF16 by default");
  const uint64_t bf16_format = fmt();
  require(bf16_format == 2, "the BF16 form keeps the family's image identity");
  const size_t bf16_total =
      Qwen35LayerStream::resident_bytes(c, 0, 1, dgpp::LoaderHeadSharding::Full, true);
  {
    const DenseForm fp8(true);
    require(Qwen35LayerStream::dense_weights_fp8(), "the knob reads back");
    // The image identity: loader_format() feeds the resident image key.
    require(fmt() != bf16_format, "the FP8 form has its own resident image identity");
    size_t fp8_total = Qwen35LayerStream::globals_bytes(c);
    for (int layer = 0; layer <= draft; ++layer) {
      size_t bf16_bytes = 0;
      {
        const DenseForm bf16(false);
        bf16_bytes = Qwen35LayerStream::layer_bytes(c, layer);
      }
      const size_t fp8_bytes = Qwen35LayerStream::layer_bytes(c, layer);
      require(fp8_bytes == bf16_bytes - layer_fp8_saving(c, layer, 1),
              "layer " + std::to_string(layer) +
                  " bytes: every dense matrix is codes plus a scale grid");
      fp8_total += fp8_bytes;
    }
    require(Qwen35LayerStream::resident_bytes(c, 0, 1, dgpp::LoaderHeadSharding::Full, true) ==
                    fp8_total &&
                fp8_total < bf16_total,
            "resident bytes follow the form");
    for (int layer : {0, 3, draft}) {
      // The same checkpoint bytes are read in either form: all of the layer
      // at world 1, and two ranks tile it.
      const uint64_t full = Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 1);
      require(full == table_bytes(c, layer, false), "the FP8 form reads every byte of the layer");
      require(Qwen35LayerStream::planned_layer_source_bytes(c, layer, 0, 2) +
                      Qwen35LayerStream::planned_layer_source_bytes(c, layer, 1, 2) ==
                  full + table_bytes(c, layer, true),
              "world 2 tiles the layer in the FP8 form");
      // Every slice the geometry produces at worlds 1, 2 and 4 sits on the
      // 128 grid: the counting build accepts every rank, and the bytes are
      // the BF16 form's less the dense savings.
      for (int world : {1, 2, 4})
        for (int rank = 0; rank < world; ++rank) {
          size_t bf16_bytes = 0;
          {
            const DenseForm bf16(false);
            bf16_bytes = Qwen35LayerStream::layer_bytes(c, layer, rank, world);
          }
          require(Qwen35LayerStream::layer_bytes(c, layer, rank, world) ==
                      bf16_bytes - layer_fp8_saving(c, layer, world),
                  "layer " + std::to_string(layer) + " at world " + std::to_string(world) +
                      " rank " + std::to_string(rank));
        }
    }
    // World 8 slices the shared expert at 64: served in BF16, refused by
    // name in the FP8 form.
    std::string why;
    try {
      (void)Qwen35LayerStream::layer_bytes(c, 0, 0, 8);
    } catch (const std::runtime_error& e) {
      why = e.what();
    }
    require(why.find("mlp.shared_expert.gate_proj") != std::string::npos &&
                why.find("128-aligned") != std::string::npos,
            "a misaligned FP8 slice is refused by name: " + why);
    {
      const DenseForm bf16(false);
      require(Qwen35LayerStream::layer_bytes(c, 0, 0, 8) > 0, "the BF16 form serves world 8");
    }
  }
  require(!Qwen35LayerStream::dense_weights_fp8() && fmt() == bf16_format &&
              Qwen35LayerStream::resident_bytes(c, 0, 1, dgpp::LoaderHeadSharding::Full, true) ==
                  bf16_total,
          "the BF16 form is back, unchanged");

  // The Qwen3.5 dialect ships FP8: the knob does not move its layout.
  Qwen35TextConfig dense;
  dense.num_hidden_layers = 4;
  dense.layers = {dgpp::Qwen35LayerKind::Gdn, dgpp::Qwen35LayerKind::Gdn,
                  dgpp::Qwen35LayerKind::Gdn, dgpp::Qwen35LayerKind::Full};
  const size_t gdn35 = Qwen35LayerStream::layer_bytes(dense, 0),
               attn35 = Qwen35LayerStream::layer_bytes(dense, 3);
  {
    const DenseForm fp8(true);
    require(Qwen35LayerStream::layer_bytes(dense, 0) == gdn35 &&
                Qwen35LayerStream::layer_bytes(dense, 3) == attn35,
            "qwen3.5 layers are the same bytes under the knob");
  }
}
