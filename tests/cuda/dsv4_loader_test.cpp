// The DeepSeek-V4-Flash resident loader on the synthetic fixture: every
// class lands byte-exact at worlds 1, 2 and 4 as the placement formulas of
// models/dsv4/loader.hpp say (wq_b head blocks, wo_a group rows, wo_b
// packed group columns, the sink per head, the fp8 128 x 128 scale grids
// converted e8m0 -> fp32 alongside their slices, the MXFP4 experts sliced
// on the intermediate dim with their per-row e8m0 scales, the compressors
// and the indexers replicated, the hash layers' token tables narrowed to
// int32 with every row ascending, the mHC coefficients rounded fp32 ->
// bf16 once, the head collapses padded to the mHC kernel's rows, the
// draft's head tensors whole); the byte formulas equal actual usage and
// the source-byte plan equals the bytes read; the globals; resident cache
// hits and the image round trip; a geometry the world does not divide and
// a checkpoint that does not bind are refused by name.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "dsv4_fixture.hpp"
#include "glm_rng.hpp"
#include "models/dsv4/binding.hpp"
#include "models/dsv4/config.hpp"
#include "models/dsv4/loader.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::Dsv4ExpectedTensor;
using dgpp::Dsv4LayerStream;
using dgpp::Dsv4Config;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  Dsv4Config cfg;
  std::string dir;
  std::vector<Dsv4ExpectedTensor> table;
  const Dsv4ExpectedTensor& expected(const std::string& name) const {
    for (const auto& e : table)
      if (e.name == name) return e;
    throw std::runtime_error("fixture: no expected tensor " + name);
  }
  std::vector<uint8_t> bytes(const std::string& name) const { return dsv4fx::tensor_bytes(cfg, expected(name)); }
};

Fixture write_fixture() {
  Fixture fx;
  fx.cfg = dsv4fx::tiny_config();
  fx.dir = (fs::current_path() / "dsv4_loader_fixture").string();
  dsv4fx::write_fixture(fx.cfg, fx.dir);
  fx.table = dgpp::dsv4_expected_tensors(fx.cfg);
  return fx;
}

std::vector<uint8_t> device_bytes(const void* dev, size_t n) {
  std::vector<uint8_t> h(n);
  DGPP_CUDA_OK(cudaMemcpy(h.data(), dev, n, cudaMemcpyDeviceToHost));
  return h;
}
std::vector<uint8_t> host_rows(const std::vector<uint8_t>& t, size_t row_bytes, int64_t r0, int64_t rows) {
  return std::vector<uint8_t>(t.begin() + r0 * row_bytes, t.begin() + (r0 + rows) * row_bytes);
}
std::vector<uint8_t> host_cols(const std::vector<uint8_t>& t, int64_t rows, size_t full, size_t c0, size_t w) {
  std::vector<uint8_t>
      out;
  out.reserve(rows * w);
  for (int64_t r = 0; r < rows; ++r)
    out.insert(out.end(), t.begin() + r * full + c0, t.begin() + r * full + c0 + w);
  return out;
}
void expect_device_equals(const void* dev, const std::vector<uint8_t>& want, const std::string& what) {
  require(dev != nullptr, what + ": null resident pointer");
  const std::vector<uint8_t> got = device_bytes(dev, want.size());
  require(got == want, what + ": resident bytes differ from the checkpoint slice");
}
float e8m0_value(uint8_t b) { return std::ldexp(1.0f, static_cast<int>(b) - 127); }
std::vector<uint8_t> f32_bytes(const std::vector<float>& v) {
  std::vector<uint8_t> out(v.size() * 4);
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

// An fp8 pair against the checkpoint: rows [row0, +rows) of every column
// (`ranges` empty), or the packed column ranges of every row. The scale
// grid: the e8m0 bytes of the slice's blocks as fp32.
using Ranges = std::vector<std::pair<int64_t, int64_t>>;
void check_fp8(const Fixture& fx, const dgpp::GlmQuantMatrix& q, const std::string& base, int64_t row0, int64_t rows,
               const Ranges& ranges, const std::string& tag) {
  const auto& ew = fx.expected(base + ".weight");
  const auto& es = fx.expected(base + ".scale");
  const int64_t N = ew.shape[0], K = ew.shape[1], SR = es.shape[0], SC = es.shape[1];
  const std::vector<uint8_t> w = fx.bytes(base + ".weight");
  const std::vector<uint8_t> s = fx.bytes(base + ".scale");
  require(q.scale_block_rows == 128 && q.scale_block_cols == 128, tag + " scale grid");
  std::vector<uint8_t> want_w;
  std::vector<float> want_s;
  if (ranges.empty()) {
    require(q.rows == rows && q.cols == K, tag + " geometry");
    want_w = host_rows(w, static_cast<size_t>(K), row0, rows);
    const int64_t sr = (rows + 127) / 128;
    for (int64_t r = 0; r < sr; ++r)
      for (int64_t c = 0; c < SC; ++c) want_s.push_back(e8m0_value(s[static_cast<size_t>((row0 / 128 + r) * SC + c)]));
  } else {
    int64_t cols = 0;
    for (const auto& [start, count] : ranges) cols += count;
    require(q.rows == N && q.cols == cols, tag + " geometry");
    for (int64_t r = 0; r < N; ++r)
      for (const auto& [start, count] : ranges)
        want_w.insert(want_w.end(), w.begin() + r * K + start, w.begin() + r * K + start + count);
    for (int64_t r = 0; r < SR; ++r)
      for (const auto& [start, count] : ranges)
        for (int64_t c = 0; c < count / 128; ++c) want_s.push_back(e8m0_value(s[static_cast<size_t>(r * SC + start / 128 + c)]));
  }
  expect_device_equals(q.payload, want_w, tag + " payload");
  expect_device_equals(q.scales, f32_bytes(want_s), tag + " scales");
}
void check_fp8_full(const Fixture& fx, const dgpp::GlmQuantMatrix& q, const std::string& base, const std::string& tag) {
  check_fp8(fx, q, base, 0, fx.expected(base + ".weight").shape[0], {}, tag);
}

// An MXFP4 matrix against the checkpoint's row / column slice.
void check_mxfp4(const Fixture& fx, const dgpp::GlmFp4Matrix& m, const std::string& base, bool rows_slice, int64_t start,
                 int64_t count, const std::string& tag) {
  const auto& ew = fx.expected(base + ".weight");
  const int64_t N = ew.shape[0], K = ew.shape[1] * 2;
  const size_t pr = static_cast<size_t>(K / 2), sr = static_cast<size_t>(K / 32);
  require(m.scale_group == 32 && m.mxfp4() && m.global_scale == nullptr, tag + " format");
  if (rows_slice) {
    require(m.rows == count && m.cols == K, tag + " geometry");
    expect_device_equals(m.payload, host_rows(fx.bytes(base + ".weight"), pr, start, count), tag + " payload rows");
    expect_device_equals(m.scales, host_rows(fx.bytes(base + ".scale"), sr, start, count), tag + " scale rows");
  } else {
    require(m.rows == N && m.cols == count, tag + " geometry");
    expect_device_equals(m.payload, host_cols(fx.bytes(base + ".weight"), N, pr, start / 2, count / 2), tag + " payload cols");
    expect_device_equals(m.scales, host_cols(fx.bytes(base + ".scale"), N, sr, start / 32, count / 32), tag + " scale cols");
  }
}

std::vector<uint8_t> f32_as_bf16(const std::vector<uint8_t>& f32) {
  std::vector<uint8_t> out(f32.size() / 2);
  for (size_t i = 0; i < f32.size() / 4; ++i) {
    float v;
    std::memcpy(&v, f32.data() + i * 4, 4);
    const uint16_t b = dgpp::float_to_bf16_bits(v);
    std::memcpy(out.data() + i * 2, &b, 2);
  }
  return out;
}
std::vector<uint8_t> bf16_as_f32(const std::vector<uint8_t>& bf16) {
  std::vector<uint8_t> out(bf16.size() * 2);
  for (size_t i = 0; i < bf16.size() / 2; ++i) {
    uint16_t b;
    std::memcpy(&b, bf16.data() + i * 2, 2);
    const float v = dgpp::bf16_bits_to_float(b);
    std::memcpy(out.data() + i * 4, &v, 4);
  }
  return out;
}

// A compressor (the main cache's or the indexer's own) against the checkpoint.
void check_compressor(const Fixture& fx, const dgpp::Dsv4CompressorResident& c, const std::string& cp, int ratio, int dim,
                      const std::string& tag) {
  require(c.present() && c.ratio == ratio && c.dim == dim && c.wide == (ratio == 4 ? 2 * dim : dim), tag + " geometry");
  expect_device_equals(c.wkv, fx.bytes(cp + "wkv.weight"), tag + " wkv");
  expect_device_equals(c.wgate, fx.bytes(cp + "wgate.weight"), tag + " wgate");
  expect_device_equals(c.ape, fx.bytes(cp + "ape"), tag + " ape");
  expect_device_equals(c.norm, fx.bytes(cp + "norm.weight"), tag + " norm");
}

// The head collapse in the mHC kernel's shape: fn bf16 [24, 4H] with the
// rows past hc_mult zero, base f32 [24] likewise, scale (s, 0, 0).
void check_hc_head(const Fixture& fx, const dgpp::Dsv4HcHeadResident& h, const std::string& p, const std::string& tag) {
  const size_t n = static_cast<size_t>(fx.cfg.hc_mult), rows = static_cast<size_t>(fx.cfg.hc_coeff_rows());
  const size_t width = n * static_cast<size_t>(fx.cfg.hidden_size);
  std::vector<uint8_t> fn = f32_as_bf16(fx.bytes(p + "hc_head_fn"));
  fn.resize(rows * width * 2, 0);
  std::vector<uint8_t> base = fx.bytes(p + "hc_head_base");
  base.resize(rows * 4, 0);
  std::vector<uint8_t> scale = fx.bytes(p + "hc_head_scale");
  scale.resize(12, 0);
  expect_device_equals(h.fn, fn, tag + " fn");
  expect_device_equals(h.base, base, tag + " base");
  expect_device_equals(h.scale, scale, tag + " scale");
}

void check_layer(const Fixture& fx, const Dsv4LayerStream& s, const dgpp::Dsv4LayerResident& r) {
  const Dsv4Config& cfg = fx.cfg;
  const dgpp::Dsv4LocalGeometry& g = s.geometry();
  const int world = s.world(), rank = s.rank();
  const std::string p = dgpp::dsv4_layer_prefix(cfg, r.layer);
  const std::string tag = "world " + std::to_string(world) + " rank " + std::to_string(rank) + " layer " +
                          std::to_string(r.layer) + " ";
  const bool draft = cfg.is_draft(r.layer);
  expect_device_equals(r.attn_norm, fx.bytes(p + "attn_norm.weight"), tag + "attn norm");
  expect_device_equals(r.ffn_norm, fx.bytes(p + "ffn_norm.weight"), tag + "ffn norm");
  // mHC: fn rounded once to bf16, base / scale fp32 verbatim.
  expect_device_equals(r.mhc.attn_fn, f32_as_bf16(fx.bytes(p + "hc_attn_fn")), tag + "hc attn fn");
  expect_device_equals(r.mhc.attn_base, fx.bytes(p + "hc_attn_base"), tag + "hc attn base");
  expect_device_equals(r.mhc.attn_scale, fx.bytes(p + "hc_attn_scale"), tag + "hc attn scale");
  expect_device_equals(r.mhc.ffn_fn, f32_as_bf16(fx.bytes(p + "hc_ffn_fn")), tag + "hc ffn fn");
  expect_device_equals(r.mhc.ffn_base, fx.bytes(p + "hc_ffn_base"), tag + "hc ffn base");
  expect_device_equals(r.mhc.ffn_scale, fx.bytes(p + "hc_ffn_scale"), tag + "hc ffn scale");
  // Attention.
  const std::string ap = p + "attn.";
  const int64_t hd = cfg.head_dim, ol = cfg.o_lora_rank;
  require(r.attn.local_heads == g.local_heads && r.attn.head_begin == g.head_begin, tag + "head geometry");
  require(r.attn.local_groups == g.local_groups && r.attn.group_begin == g.group_begin, tag + "group geometry");
  require(g.local_heads == cfg.num_attention_heads / world && g.local_groups == cfg.o_groups / world, tag + "geometry formulas");
  expect_device_equals(r.attn.q_norm, fx.bytes(ap + "q_norm.weight"), tag + "q norm");
  expect_device_equals(r.attn.kv_norm, fx.bytes(ap + "kv_norm.weight"), tag + "kv norm");
  expect_device_equals(r.attn.attn_sink, host_rows(fx.bytes(ap + "attn_sink"), 4, g.head_begin, g.local_heads), tag + "sink");
  check_fp8_full(fx, r.attn.wq_a, ap + "wq_a", tag + "wq_a");
  check_fp8_full(fx, r.attn.wkv, ap + "wkv", tag + "wkv");
  check_fp8(fx, r.attn.wq_b, ap + "wq_b", g.head_begin * hd, g.local_heads * hd, {}, tag + "wq_b");
  check_fp8(fx, r.attn.wo_a, ap + "wo_a", g.group_begin * ol, g.local_groups * ol, {}, tag + "wo_a");
  check_fp8(fx, r.attn.wo_b, ap + "wo_b", 0, 0, {{g.group_begin * ol, g.local_groups * ol}}, tag + "wo_b");
  const int ratio = cfg.compress_ratio(r.layer);
  if (ratio > 0)
    check_compressor(fx, r.attn.comp, ap + "compressor.", ratio, cfg.head_dim, tag + "compressor");
  else
    require(!r.attn.comp.present(), tag + "no compressor on a window layer");
  if (cfg.indexed(r.layer)) {
    check_fp8_full(fx, r.attn.idx_wq_b, ap + "indexer.wq_b", tag + "indexer wq_b");
    expect_device_equals(r.attn.idx_wp, fx.bytes(ap + "indexer.weights_proj.weight"), tag + "indexer weights_proj");
    check_compressor(fx, r.attn.idx_comp, ap + "indexer.compressor.", ratio, cfg.index_head_dim, tag + "indexer compressor");
  } else {
    require(r.attn.idx_wp == nullptr && r.attn.idx_wq_b.payload == nullptr && !r.attn.idx_comp.present(), tag + "no indexer");
  }
  // MoE.
  const std::string mp = p + "ffn.";
  expect_device_equals(r.moe.router, fx.bytes(mp + "gate.weight"), tag + "router");
  if (cfg.hashed(r.layer)) {
    require(r.moe.router_bias == nullptr && r.moe.tid2eid != nullptr, tag + "a hash layer carries its table, no bias");
    // The table: int64 -> int32, every row ascending.
    const std::vector<uint8_t> raw = fx.bytes(mp + "gate.tid2eid");
    const size_t k = static_cast<size_t>(cfg.num_experts_per_tok), rows = static_cast<size_t>(cfg.vocab_size);
    std::vector<int32_t> want(rows * k);
    for (size_t t = 0; t < rows; ++t) {
      for (size_t j = 0; j < k; ++j) {
        int64_t v;
        std::memcpy(&v, raw.data() + (t * k + j) * 8, 8);
        want[t * k + j] = static_cast<int32_t>(v);
      }
      std::sort(want.begin() + static_cast<std::ptrdiff_t>(t * k), want.begin() + static_cast<std::ptrdiff_t>((t + 1) * k));
    }
    std::vector<uint8_t> wb(want.size() * 4);
    std::memcpy(wb.data(), want.data(), wb.size());
    expect_device_equals(r.moe.tid2eid, wb, tag + "tid2eid");
  } else {
    require(r.moe.tid2eid == nullptr, tag + "no token table past the hash layers");
    expect_device_equals(r.moe.router_bias, fx.bytes(mp + "gate.bias"), tag + "router bias");
  }
  const int64_t I = g.local_inter, S = g.local_shared_inter;
  require(I == cfg.moe_intermediate_size / world && S == cfg.shared_expert_inter() / world, tag + "inter formulas");
  require(r.moe.local_inter == I && r.moe.local_shared_inter == S, tag + "moe inter");
  const int E = cfg.n_routed_experts;
  require(r.moe.n_experts == E && static_cast<int>(r.moe.experts.size()) == E * 3, tag + "expert count");
  for (int e = 0; e < E; ++e) {
    const std::string ep = mp + "experts." + std::to_string(e) + ".";
    check_mxfp4(fx, r.moe.experts[static_cast<size_t>(e) * 3 + 0], ep + "w1", true, rank * I, I, tag + "expert gate");
    check_mxfp4(fx, r.moe.experts[static_cast<size_t>(e) * 3 + 1], ep + "w3", true, rank * I, I, tag + "expert up");
    check_mxfp4(fx, r.moe.experts[static_cast<size_t>(e) * 3 + 2], ep + "w2", false, rank * I, I, tag + "expert down");
  }
  const std::string sp = mp + "shared_experts.";
  check_fp8(fx, r.moe.shared[0], sp + "w1", rank * S, S, {}, tag + "shared gate");
  check_fp8(fx, r.moe.shared[1], sp + "w3", rank * S, S, {}, tag + "shared up");
  check_fp8(fx, r.moe.shared[2], sp + "w2", 0, 0, {{rank * S, S}}, tag + "shared down");
  // Draft.
  if (draft) {
    const int stage = cfg.draft_stage(r.layer);
    if (stage == 0) {
      check_fp8_full(fx, r.draft.main_proj, p + "main_proj", tag + "main_proj");
      expect_device_equals(r.draft.main_norm, fx.bytes(p + "main_norm.weight"), tag + "main norm");
    } else {
      require(r.draft.main_proj.payload == nullptr && r.draft.main_norm == nullptr, tag + "no main_proj past stage 0");
    }
    if (stage == cfg.draft_stages - 1) {
      expect_device_equals(r.draft.norm, fx.bytes(p + "norm.weight"), tag + "draft norm");
      expect_device_equals(r.draft.markov_embed, fx.bytes(p + "markov_head.markov_w1.weight"), tag + "markov embed");
      expect_device_equals(r.draft.markov_head, fx.bytes(p + "markov_head.markov_w2.weight"), tag + "markov head");
      expect_device_equals(r.draft.confidence, bf16_as_f32(fx.bytes(p + "confidence_head.proj.weight")), tag + "confidence");
      check_hc_head(fx, r.draft.hc_head, p, tag + "draft hc_head");
    } else {
      require(r.draft.norm == nullptr && r.draft.markov_head == nullptr && r.draft.hc_head.fn == nullptr,
              tag + "no head before the last stage");
    }
  } else {
    require(r.draft.main_proj.payload == nullptr && r.draft.markov_head == nullptr, tag + "no draft head on a main layer");
  }
  require(r.bytes == Dsv4LayerStream::layer_bytes(cfg, r.layer, rank, world), tag + "layer bytes formula");
}

}  // namespace

DGPP_TEST(dsv4_loader_slices_every_class_byte_exact_at_worlds_1_2_4) {
  const Fixture fx = write_fixture();
  const int layers = fx.cfg.max_layer();
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      const auto head = world > 1 ? dgpp::Dsv4HeadSharding::VocabSharded : dgpp::Dsv4HeadSharding::Full;
      Dsv4LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::Dsv4Residency::Streaming, head, /*resident_mtp=*/true);
      for (int l = 0; l < layers; ++l) {
        const uint64_t before = s.source_bytes_read();
        const auto& r = s.load_layer(l);
        require(r.layer == l, "layer index");
        check_layer(fx, s, r);
        require(s.source_bytes_read() - before == Dsv4LayerStream::planned_layer_source_bytes(fx.cfg, l, rank, world),
                "the source-byte plan equals the bytes read");
        s.release_layer();
      }
    }
  }
}

DGPP_TEST(dsv4_loader_globals_slices) {
  const Fixture fx = write_fixture();
  const size_t H2 = static_cast<size_t>(fx.cfg.hidden_size) * 2;
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      const auto head = world > 1 ? dgpp::Dsv4HeadSharding::VocabSharded : dgpp::Dsv4HeadSharding::Full;
      Dsv4LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::Dsv4Residency::Streaming, head);
      const auto& g = s.load_globals();
      expect_device_equals(g.embed, fx.bytes("embed.weight"), "embed");
      expect_device_equals(g.final_norm, fx.bytes("norm.weight"), "final norm");
      check_hc_head(fx, g.hc_head, "", "the head collapse");
      expect_device_equals(g.lm_head, host_rows(fx.bytes("head.weight"), H2, g.lm_vocab_begin, g.lm_vocab_count), "lm head slice");
      require(g.lm_vocab_count == Dsv4LayerStream::lm_vocab_count(fx.cfg, rank, world, head), "lm head count");
      require(g.lm_vocab_count == (world > 1 ? fx.cfg.vocab_size / world : fx.cfg.vocab_size), "lm head formula");
      require(g.embed_vocab_begin == 0 && g.embed_vocab_count == fx.cfg.vocab_size, "the embedding is whole by default");
      require(g.bytes == Dsv4LayerStream::globals_bytes(fx.cfg, rank, world, head), "globals formula");
    }
  }
  Dsv4LayerStream::set_embed_vocab_sharded(true);
  for (const int world : {1, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      const auto head = world > 1 ? dgpp::Dsv4HeadSharding::VocabSharded : dgpp::Dsv4HeadSharding::Full;
      Dsv4LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::Dsv4Residency::Streaming, head);
      const auto& g = s.load_globals();
      if (world == 1)
        require(g.embed_vocab_begin == 0 && g.embed_vocab_count == fx.cfg.vocab_size, "world 1 keeps the whole table");
      else
        require(g.embed_vocab_begin == g.lm_vocab_begin && g.embed_vocab_count == g.lm_vocab_count, "the embedding slice is the lm head's");
      expect_device_equals(g.embed, host_rows(fx.bytes("embed.weight"), H2, g.embed_vocab_begin, g.embed_vocab_count), "embed slice");
      require(g.bytes == Dsv4LayerStream::globals_bytes(fx.cfg, rank, world, head), "globals formula (sharded)");
    }
  }
  Dsv4LayerStream::set_embed_vocab_sharded(false);
}

DGPP_TEST(dsv4_loader_resident_mode_and_image_round_trip) {
  const Fixture fx = write_fixture();
  const fs::path cache = fs::current_path() / "dsv4_loader_image_cache";
  fs::remove_all(cache);
  const std::string saved = Dsv4LayerStream::resident_image_dir();
  Dsv4LayerStream::set_resident_image_dir(cache.string());
  const int layers = fx.cfg.max_layer();
  std::vector<std::vector<uint8_t>> built(static_cast<size_t>(layers));
  {
    Dsv4LayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::Dsv4Residency::Resident, dgpp::Dsv4HeadSharding::VocabSharded,
                       /*resident_mtp=*/true);
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      check_layer(fx, s, r);
      const auto span = s.resident_layer_span(l);
      require(span.first != nullptr && span.second == r.bytes, "resident span");
      built[static_cast<size_t>(l)] = device_bytes(span.first, span.second);
    }
    (void)s.load_globals();
    require(s.image_layers_captured() == layers, "every layer captured");
    const uint64_t before = s.source_bytes_read();
    for (int l = 0; l < layers; ++l) (void)s.load_layer(l);
    require(s.source_bytes_read() == before, "cache hits read no storage");
    require(Dsv4LayerStream::resident_bytes(fx.cfg, 1, 2, dgpp::Dsv4HeadSharding::VocabSharded, true) ==
                [&] { size_t t = s.load_globals().bytes; for (int l = 0; l < layers; ++l) t += s.load_layer(l).bytes; return t; }(),
            "resident bytes formula");
    s.release_sources();
    require(s.sources_released() && s.staging_released(), "the mappings and the staging mirror go together");
  }
  {
    Dsv4LayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::Dsv4Residency::Resident, dgpp::Dsv4HeadSharding::VocabSharded,
                       /*resident_mtp=*/true);
    const uint64_t before = s.source_bytes_read();
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      const auto span = s.resident_layer_span(l);
      require(device_bytes(span.first, span.second) == built[static_cast<size_t>(l)], "restored layer bitwise the built one");
      check_layer(fx, s, r);
    }
    require(s.image_layers_restored() == layers && s.source_bytes_read() == before,
            "every layer restored from the image, no source reads");
    const dgpp::ReplicatedDigest d = s.hash_replicated();
    require(d.layer.size() == static_cast<size_t>(layers) && d.tensors > 0, "digest shape");
    // The digest is rank-invariant: rank 0's equals rank 1's.
    Dsv4LayerStream s0(fx.cfg, fx.dir, 0, 2, dgpp::Dsv4Residency::Streaming, dgpp::Dsv4HeadSharding::VocabSharded, true);
    const dgpp::ReplicatedDigest d0 = s0.hash_replicated();
    require(d0.globals == d.globals && d0.layer == d.layer && d0.bytes == d.bytes, "the replicated digest is rank-invariant");
  }
  Dsv4LayerStream::set_resident_image_dir(saved);
  fs::remove_all(cache);
}

DGPP_TEST(dsv4_loader_refuses_bad_geometry_and_binding_by_name) {
  const Fixture fx = write_fixture();
  {
    bool refused = false;
    try {
      Dsv4LayerStream s(fx.cfg, fx.dir, 0, 3, dgpp::Dsv4Residency::Streaming, dgpp::Dsv4HeadSharding::VocabSharded);
    } catch (const std::invalid_argument& e) {
      refused = std::string(e.what()).find("world") != std::string::npos;
    }
    require(refused, "a world that does not divide the heads is refused by name");
  }
  {
    Dsv4Config cfg = fx.cfg;
    cfg.n_routed_experts = 16;  // experts 8..15 are not in the shard
    bool refused = false;
    try {
      Dsv4LayerStream s(cfg, fx.dir, 0, 1);
    } catch (const std::runtime_error& e) {
      refused = std::string(e.what()).find("binding failed") != std::string::npos;
    }
    require(refused, "a checkpoint that does not bind is refused by name");
  }
}

int main() { return dgpp::test::run_all(); }
