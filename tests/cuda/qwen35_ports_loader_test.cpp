// The qwen3_5 stack's resident loader on the later ports' synthetic
// fixtures (tests/cuda/qwen35_ports_fixture.hpp), the copy pass.
//
// Qwen3-Coder-Next (the Qwen3Next dialect, compressed-tensors NVFP4):
//   every resident value is the checkpoint's — the GDN's BF16 projections
//   gathered out of the per-key-head interleave, the attention q/k/v/o and
//   the shared expert dequantized by the container's formula (written out
//   here, not called), the routed experts' codes, block scales and global
//   scales as shipped, no cache scales, no static activation scales, no
//   draft head;
//   the rank slices at worlds 2 and 4 tile the world-1 layers;
//   the FP8 form of the dense stack is the block-FP8 encode of the BF16
//   form's matrices and leaves the rest of the layer alone.
// Qwen3.6-35B-A3B (the Qwen3.5 dialect with the routed MoE), both containers:
//   the NVFP4 mixed release — the per-tensor FP8 projections' codes as
//   shipped under a scale grid that holds the tensor's one scale everywhere,
//   the modelopt NVFP4 experts as shipped with the reciprocal of their
//   multiplier, the shared expert and the lm head dequantized by the modelopt
//   formula, the BF16 draft layer with each expert's slice of the two stacked
//   tensors encoded to block FP8, the vision tensor skipped;
//   the FP8 release — every matrix's codes as shipped and its BF16 block
//   scales widened, the experts and the draft layer included.
// Qwen3.5-0.8B (the dense Qwen3.5 dialect, unquantized, tied embeddings):
//   every BF16 matrix as shipped — the dense MLP through its BF16 pointers —
//   A_log in F32 as stored, the F32 GDN norm weight rounded to BF16, the head
//   the embedding's own rows (no grant, no stored tensor); under
//   dense_weights fp8 every dense matrix the block-FP8 encode of those.
// The stream throws on any drift between its byte formulas, its planned
// source bytes and what a build used, so every load here is that check too.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "loaders/fp8_quant.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/loader35.hpp"
#include "qwen35_ports_fixture.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::GlmFp4Matrix;
using dgpp::GlmQuantMatrix;
using dgpp::Qwen35LayerResident;
using dgpp::Qwen35LayerStream;
using dgpp::Qwen35LocalGeometry;
using dgpp::Qwen35TextConfig;
using qwen35portsfx::Fixture;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

Fixture coder_next_fixture() {
  return qwen35portsfx::write_fixture((fs::current_path() / "qwen3codernext_loader_fixture").string(),
                                      qwen35portsfx::coder_next_config_json());
}

// The dense stack's form for the duration of a scope.
struct DenseForm {
  bool saved = Qwen35LayerStream::dense_weights_fp8();
  explicit DenseForm(bool fp8) { Qwen35LayerStream::set_dense_weights_fp8(fp8); }
  ~DenseForm() { Qwen35LayerStream::set_dense_weights_fp8(saved); }
};

template <class T>
std::vector<T> down(const T* dev, size_t n, const std::string& what) {
  require(dev != nullptr, what + ": null resident pointer");
  std::vector<T> h(n);
  DGPP_CUDA_OK(cudaMemcpy(h.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
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

// The compressed-tensors NVFP4 matrix `base` [N, K] of the fixture as BF16,
// by the container's formula: bf16(code x (block scale / global scale)).
std::vector<uint16_t> packed_matrix(const Fixture& fx, const std::string& base, int64_t N, int64_t K) {
  const std::vector<uint8_t>& codes = fx.bytes(base + ".weight_packed");
  const std::vector<uint8_t>& scales = fx.bytes(base + ".weight_scale");
  const float global = fx.f32(base + ".weight_global_scale");
  require(static_cast<int64_t>(codes.size()) == N * K / 2 &&
              static_cast<int64_t>(scales.size()) == N * K / 16,
          base + ": fixture geometry");
  std::vector<uint16_t> out(static_cast<size_t>(N * K));
  for (int64_t n = 0; n < N; ++n)
    for (int64_t k = 0; k < K; ++k) {
      const uint8_t byte = codes[static_cast<size_t>(n * (K / 2) + k / 2)];
      const uint8_t code = (k & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0xF);
      const float block = e4m3(scales[static_cast<size_t>(n * (K / 16) + k / 16)]) / global;
      out[static_cast<size_t>(n * K + k)] = dgpp::float_to_bf16_bits(e2m1(code) * block);
    }
  return out;
}

// Rows [r0, +rn) of a [*, width] matrix; columns [c0, +cn) of a [*, full] one.
template <class T>
std::vector<T> rows_of(const std::vector<T>& m, int64_t width, int64_t r0, int64_t rn) {
  return std::vector<T>(m.begin() + r0 * width, m.begin() + (r0 + rn) * width);
}
template <class T>
std::vector<T> cols_of(const std::vector<T>& m, int64_t full, int64_t c0, int64_t cn) {
  std::vector<T> out;
  for (int64_t r = 0; r < static_cast<int64_t>(m.size()) / full; ++r)
    out.insert(out.end(), m.begin() + r * full + c0, m.begin() + r * full + c0 + cn);
  return out;
}
template <class T>
void append(std::vector<T>& to, const std::vector<T>& from) {
  to.insert(to.end(), from.begin(), from.end());
}

// The Qwen3Next dialect's GDN projections at world 1, head-major, from the
// checkpoint's per-key-head interleave: in_proj_qkvz rows are groups of
// [q dk | k dk | v r*dv | z r*dv], in_proj_ba rows groups of [b x r | a x r].
struct GdnWant {
  std::vector<uint16_t> qkv, z, a, b;
};
GdnWant gdn_from_interleave(const Fixture& fx, const std::string& p) {
  const Qwen35TextConfig& c = fx.cfg;
  const int64_t H = c.hidden_size, nk = c.gdn_key_heads, dk = c.gdn_key_head_dim;
  const int64_t dv = c.gdn_value_head_dim, r = c.gdn_value_heads / nk, G = 2 * dk + 2 * r * dv;
  const std::vector<uint16_t> qkvz = fx.bf16(p + "in_proj_qkvz.weight");
  const std::vector<uint16_t> ba = fx.bf16(p + "in_proj_ba.weight");
  GdnWant w;
  for (int64_t g = 0; g < nk; ++g) append(w.qkv, rows_of(qkvz, H, g * G, dk));
  for (int64_t g = 0; g < nk; ++g) append(w.qkv, rows_of(qkvz, H, g * G + dk, dk));
  for (int64_t g = 0; g < nk; ++g) append(w.qkv, rows_of(qkvz, H, g * G + 2 * dk, r * dv));
  for (int64_t g = 0; g < nk; ++g) {
    append(w.z, rows_of(qkvz, H, g * G + 2 * dk + r * dv, r * dv));
    append(w.b, rows_of(ba, H, g * 2 * r, r));
    append(w.a, rows_of(ba, H, g * 2 * r + r, r));
  }
  return w;
}

// One dense projection of a resident layer of the MoE dialects: both views
// and its shape at the stream's geometry.
struct Dense {
  std::string name;
  const uint16_t* bf16;
  const GlmQuantMatrix* fp8;
  int64_t rows, cols;
};
std::vector<Dense> dense_of(const Qwen35TextConfig& c, const Qwen35LocalGeometry& g,
                            const Qwen35LayerResident& r) {
  const int64_t H = c.hidden_size, d = c.head_dim, S = g.local_shared_inter;
  std::vector<Dense> out;
  if (r.kind == dgpp::Qwen35LayerKind::Gdn) {
    const int64_t kd = static_cast<int64_t>(g.local_key_heads) * c.gdn_key_head_dim;
    const int64_t vd = static_cast<int64_t>(g.local_value_heads) * c.gdn_value_head_dim;
    out.push_back({"in_proj_qkv", r.gdn.in_proj_qkv, &r.gdn.in_proj_qkv_fp8, 2 * kd + vd, H});
    out.push_back({"in_proj_z", r.gdn.in_proj_z, &r.gdn.in_proj_z_fp8, vd, H});
    out.push_back({"out_proj", r.gdn.out_proj, &r.gdn.out_proj_fp8, H, vd});
  } else {
    out.push_back({"q_proj", r.full.q_proj, &r.full.q_proj_fp8, g.local_heads * 2 * d, H});
    out.push_back({"k_proj", r.full.k_proj, &r.full.k_proj_fp8, g.local_kv_heads * d, H});
    out.push_back({"v_proj", r.full.v_proj, &r.full.v_proj_fp8, g.local_kv_heads * d, H});
    out.push_back({"o_proj", r.full.o_proj, &r.full.o_proj_fp8, H, g.local_heads * d});
  }
  out.push_back({"shared gate_proj", r.moe.shared[0], &r.moe.shared_fp8[0], S, H});
  out.push_back({"shared up_proj", r.moe.shared[1], &r.moe.shared_fp8[1], S, H});
  out.push_back({"shared down_proj", r.moe.shared[2], &r.moe.shared_fp8[2], H, S});
  return out;
}

// What the dense form does not touch, as bytes: the norms, the GDN's a/b,
// conv, A_log, dt_bias and head norm (or the q/k norms), the router and the
// shared gate, and the NVFP4 routed experts with their global scales.
std::vector<uint8_t> rest_of(const Qwen35TextConfig& c, const Qwen35LocalGeometry& g,
                             const Qwen35LayerResident& r) {
  const size_t H = static_cast<size_t>(c.hidden_size);
  std::vector<uint8_t> out;
  const auto add = [&](const void* dev, size_t bytes, const char* what) {
    append(out, down(static_cast<const uint8_t*>(dev), bytes, what));
  };
  add(r.input_norm, H * 2, "input_norm");
  add(r.post_norm, H * 2, "post_norm");
  if (r.kind == dgpp::Qwen35LayerKind::Gdn) {
    const size_t lv = static_cast<size_t>(g.local_value_heads);
    const size_t rows = 2 * static_cast<size_t>(g.local_key_heads) * c.gdn_key_head_dim +
                        lv * static_cast<size_t>(c.gdn_value_head_dim);
    add(r.gdn.in_proj_a, lv * H * 2, "in_proj_a");
    add(r.gdn.in_proj_b, lv * H * 2, "in_proj_b");
    add(r.gdn.conv, rows * static_cast<size_t>(c.gdn_conv_width) * 2, "conv");
    add(r.gdn.a_log, lv * 4, "a_log");
    add(r.gdn.dt_bias, lv * 4, "dt_bias");
    add(r.gdn.norm, static_cast<size_t>(c.gdn_value_head_dim) * 2, "gdn norm");
  } else {
    add(r.full.q_norm, static_cast<size_t>(c.head_dim) * 2, "q_norm");
    add(r.full.k_norm, static_cast<size_t>(c.head_dim) * 2, "k_norm");
  }
  add(r.moe.router, static_cast<size_t>(c.num_experts) * H * 2, "router");
  add(r.moe.shared_gate, H * 2, "shared gate");
  const size_t matrices = static_cast<size_t>(c.num_experts) * 3;
  require(r.moe.experts_fp4.size() == matrices && r.moe.experts.empty(), "the experts are NVFP4");
  for (const GlmFp4Matrix& q : r.moe.experts_fp4) {
    add(q.payload, q.payload_bytes(), "expert payload");
    add(q.scales, q.scale_bytes(), "expert scales");
  }
  add(r.moe.expert_globals, matrices * 4, "expert globals");
  return out;
}

struct LayerCopy {
  std::vector<std::vector<uint16_t>> dense;  // dense_of's order, BF16
  std::vector<uint8_t> rest;
  // The routed experts, per matrix (gate, up, down per expert).
  std::vector<std::vector<uint8_t>> expert_codes, expert_scales;
  std::vector<float> expert_globals;
  size_t bytes = 0;
};

// Every layer of the BF16 form at (rank, world).
std::vector<LayerCopy> load_bf16_form(const Fixture& fx, int rank, int world) {
  const DenseForm bf16(false);
  Qwen35LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::LoaderResidency::Streaming,
                      dgpp::LoaderHeadSharding::VocabSharded);
  std::vector<LayerCopy> out;
  for (int l = 0; l < s.max_layer(); ++l) {
    const Qwen35LayerResident& r = s.load_layer(l);
    LayerCopy lc;
    for (const Dense& d : dense_of(fx.cfg, s.geometry(), r)) {
      require(d.fp8->payload == nullptr, d.name + ": the BF16 form has no fp8 view");
      lc.dense.push_back(down(d.bf16, static_cast<size_t>(d.rows * d.cols), d.name));
    }
    lc.rest = rest_of(fx.cfg, s.geometry(), r);
    for (const GlmFp4Matrix& q : r.moe.experts_fp4) {
      lc.expert_codes.push_back(down(q.payload, q.payload_bytes(), "expert payload"));
      lc.expert_scales.push_back(down(q.scales, q.scale_bytes(), "expert scales"));
    }
    lc.expert_globals = down(r.moe.expert_globals, r.moe.experts_fp4.size(), "expert globals");
    lc.bytes = r.bytes;
    out.push_back(std::move(lc));
  }
  return out;
}

}  // namespace

DGPP_TEST(qwen3codernext_loader_resident_values_are_the_checkpoints) {
  const Fixture fx = coder_next_fixture();
  const Qwen35TextConfig& c = fx.cfg;
  require(c.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Packed && c.mtp_layer() == -1, "the fixture's recipe");
  const int64_t H = c.hidden_size, d = c.head_dim, E = c.num_experts;
  const int64_t I = c.moe_intermediate_size, S = c.shared_expert_intermediate_size;
  const int64_t vdim = static_cast<int64_t>(c.gdn_value_heads) * c.gdn_value_head_dim;
  const DenseForm bf16(false);
  Qwen35LayerStream s(c, fx.dir, 0, 1, dgpp::LoaderResidency::Streaming, dgpp::LoaderHeadSharding::Full);
  require(s.max_layer() == c.num_hidden_layers, "no draft layer to load");
  for (int l = 0; l < s.max_layer(); ++l) {
    const Qwen35LayerResident& r = s.load_layer(l);
    const std::string p = dgpp::qwen35_layer_prefix(c, l);
    const std::string at = "layer " + std::to_string(l) + ": ";
    require(down(r.input_norm, static_cast<size_t>(H), "input norm") == fx.bf16(p + "input_layernorm.weight"),
            at + "input norm");
    require(down(r.post_norm, static_cast<size_t>(H), "post norm") ==
                fx.bf16(p + "post_attention_layernorm.weight"),
            at + "post norm");
    require(r.kv_cache_scales == nullptr && r.k_cache_scale == 0.0f, at + "no K/V-cache scales in this release");
    if (r.kind == dgpp::Qwen35LayerKind::Gdn) {
      const std::string g = p + "linear_attn.";
      const GdnWant want = gdn_from_interleave(fx, g);
      require(down(r.gdn.in_proj_qkv, want.qkv.size(), "qkv") == want.qkv, at + "in_proj_qkv is the gather");
      require(down(r.gdn.in_proj_z, want.z.size(), "z") == want.z, at + "in_proj_z is the gather");
      require(down(r.gdn.in_proj_a, want.a.size(), "a") == want.a, at + "in_proj_a is the gather");
      require(down(r.gdn.in_proj_b, want.b.size(), "b") == want.b, at + "in_proj_b is the gather");
      // The BF16 out_proj as it ships (the modelopt release quantizes it).
      require(down(r.gdn.out_proj, static_cast<size_t>(H * vdim), "out") == fx.bf16(g + "out_proj.weight"),
              at + "out_proj is the checkpoint's BF16 matrix");
      require(down(r.gdn.conv, fx.bytes(g + "conv1d.weight").size() / 2, "conv") ==
                  fx.bf16(g + "conv1d.weight"),
              at + "conv");
      require(down(r.gdn.norm, static_cast<size_t>(c.gdn_value_head_dim), "norm") == fx.bf16(g + "norm.weight"),
              at + "gdn norm");
      const std::vector<uint16_t> a_log = fx.bf16(g + "A_log"), dt = fx.bf16(g + "dt_bias");
      const std::vector<float> got_a = down(r.gdn.a_log, a_log.size(), "a_log");
      const std::vector<float> got_d = down(r.gdn.dt_bias, dt.size(), "dt_bias");
      for (size_t i = 0; i < a_log.size(); ++i)
        require(got_a[i] == dgpp::bf16_bits_to_float(a_log[i]) && got_d[i] == dgpp::bf16_bits_to_float(dt[i]),
                at + "A_log and dt_bias widened");
    } else {
      const std::string a = p + "self_attn.";
      const int64_t qh = c.num_attention_heads, kvh = c.num_key_value_heads;
      require(down(r.full.q_proj, static_cast<size_t>(2 * qh * d * H), "q") ==
                  packed_matrix(fx, a + "q_proj", 2 * qh * d, H),
              at + "q_proj is the container's dequant");
      require(down(r.full.k_proj, static_cast<size_t>(kvh * d * H), "k") ==
                  packed_matrix(fx, a + "k_proj", kvh * d, H),
              at + "k_proj is the container's dequant");
      require(down(r.full.v_proj, static_cast<size_t>(kvh * d * H), "v") ==
                  packed_matrix(fx, a + "v_proj", kvh * d, H),
              at + "v_proj is the container's dequant");
      require(down(r.full.o_proj, static_cast<size_t>(H * qh * d), "o") ==
                  packed_matrix(fx, a + "o_proj", H, qh * d),
              at + "o_proj is the container's dequant");
      require(down(r.full.q_norm, static_cast<size_t>(d), "q_norm") == fx.bf16(a + "q_norm.weight") &&
                  down(r.full.k_norm, static_cast<size_t>(d), "k_norm") == fx.bf16(a + "k_norm.weight"),
              at + "q/k norms");
    }
    const std::string m = p + "mlp.";
    require(down(r.moe.router, static_cast<size_t>(E * H), "router") == fx.bf16(m + "gate.weight"),
            at + "router");
    require(down(r.moe.shared_gate, static_cast<size_t>(H), "shared gate") ==
                fx.bf16(m + "shared_expert_gate.weight"),
            at + "shared gate");
    require(down(r.moe.shared[0], static_cast<size_t>(S * H), "shared gate_proj") ==
                    packed_matrix(fx, m + "shared_expert.gate_proj", S, H) &&
                down(r.moe.shared[1], static_cast<size_t>(S * H), "shared up_proj") ==
                    packed_matrix(fx, m + "shared_expert.up_proj", S, H) &&
                down(r.moe.shared[2], static_cast<size_t>(H * S), "shared down_proj") ==
                    packed_matrix(fx, m + "shared_expert.down_proj", H, S),
            at + "the shared expert is the container's dequant");
    // The routed experts: the checkpoint's bytes, and the global scale as
    // shipped — the divisor the kernels apply (modelopt's is stored inverted).
    require(r.moe.experts_fp4.size() == static_cast<size_t>(E) * 3 && r.moe.experts.empty() &&
                !r.moe.packq(),
            at + "NVFP4 experts");
    require(r.moe.local_inter == I && r.moe.local_shared_inter == S, at + "moe slices");
    const std::vector<float> globals = down(r.moe.expert_globals, static_cast<size_t>(E) * 3, "globals");
    static const char* kPart[3] = {"gate_proj", "up_proj", "down_proj"};
    for (int64_t e = 0; e < E; ++e)
      for (int i = 0; i < 3; ++i) {
        const std::string base = m + "experts." + std::to_string(e) + "." + kPart[i];
        const GlmFp4Matrix& q = r.moe.experts_fp4[static_cast<size_t>(e * 3 + i)];
        require(q.rows == (i == 2 ? H : I) && q.cols == (i == 2 ? I : H) && q.scale_group == 16,
                at + base + " shape");
        require(down(q.payload, q.payload_bytes(), base) == fx.bytes(base + ".weight_packed"),
                at + base + " codes as shipped");
        require(down(q.scales, q.scale_bytes(), base) == fx.bytes(base + ".weight_scale"),
                at + base + " block scales as shipped");
        require(q.global_scale == r.moe.expert_globals + e * 3 + i &&
                    globals[static_cast<size_t>(e * 3 + i)] == fx.f32(base + ".weight_global_scale"),
                at + base + " global scale as shipped (the kernels' divisor)");
      }
    require(r.moe.act_scales == nullptr && r.moe.act_scale_w13 == 0.0f && r.moe.act_scale_w2 == 0.0f,
            at + "no static activation scales in this container");
    require(!r.mlp.gate && !r.mlp.gate_fp8.payload, at + "the dense mlp stays empty");
  }
  const dgpp::Qwen35GlobalsResident& g = s.load_globals();
  const size_t V = static_cast<size_t>(c.vocab_size);
  require(down(g.embed, V * H, "embed") == fx.bf16("model.embed_tokens.weight"), "embed");
  require(down(g.lm_head, V * H, "head") == fx.bf16("lm_head.weight"), "lm head");
  require(down(g.final_norm, static_cast<size_t>(H), "norm") == fx.bf16("model.norm.weight"), "final norm");
  require(g.mtp_fc == nullptr && g.mtp_norm == nullptr, "no draft head");
  require(s.source_bytes_read() > 0, "the copy pass read the checkpoint");
}

DGPP_TEST(qwen3codernext_loader_rank_slices_tile_world_one) {
  const Fixture fx = coder_next_fixture();
  const Qwen35TextConfig& c = fx.cfg;
  const int64_t H = c.hidden_size, d = c.head_dim, E = c.num_experts, I = c.moe_intermediate_size;
  const std::vector<LayerCopy> one = load_bf16_form(fx, 0, 1);
  for (int world : {2, 4})
    for (int rank = 0; rank < world; ++rank) {
      const std::vector<LayerCopy> part = load_bf16_form(fx, rank, world);
      const Qwen35LocalGeometry g =
          Qwen35LocalGeometry::from_config(c, rank, world, dgpp::LoaderHeadSharding::VocabSharded);
      const int64_t K = static_cast<int64_t>(c.gdn_key_heads) * c.gdn_key_head_dim;
      const int64_t V = static_cast<int64_t>(c.gdn_value_heads) * c.gdn_value_head_dim;
      const int64_t kd = K / world, vd = V / world, S = c.shared_expert_intermediate_size;
      const int64_t ls = S / world, li = I / world;
      for (int l = 0; l < c.num_hidden_layers; ++l) {
        const std::string at = "layer " + std::to_string(l) + " at world " + std::to_string(world) +
                               " rank " + std::to_string(rank);
        const auto& w1 = one[static_cast<size_t>(l)].dense;
        const auto& p = part[static_cast<size_t>(l)].dense;
        const bool gdn = c.layers[l] == dgpp::Qwen35LayerKind::Gdn;
        if (gdn) {
          std::vector<uint16_t> qkv = rows_of(w1[0], H, rank * kd, kd);
          append(qkv, rows_of(w1[0], H, K + rank * kd, kd));
          append(qkv, rows_of(w1[0], H, 2 * K + rank * vd, vd));
          require(p[0] == qkv, at + ": in_proj_qkv");
          require(p[1] == rows_of(w1[1], H, rank * vd, vd), at + ": in_proj_z");
          require(p[2] == cols_of(w1[2], V, rank * vd, vd), at + ": out_proj");
        } else {
          require(p[0] == rows_of(w1[0], H, g.head_begin * 2 * d, g.local_heads * 2 * d), at + ": q_proj");
          require(p[1] == rows_of(w1[1], H, g.kv_head_begin * d, g.local_kv_heads * d), at + ": k_proj");
          require(p[2] == rows_of(w1[2], H, g.kv_head_begin * d, g.local_kv_heads * d), at + ": v_proj");
          require(p[3] == cols_of(w1[3], c.num_attention_heads * d, g.head_begin * d, g.local_heads * d),
                  at + ": o_proj");
        }
        const size_t s0 = gdn ? 3 : 4;
        require(p[s0] == rows_of(w1[s0], H, rank * ls, ls), at + ": shared gate_proj");
        require(p[s0 + 1] == rows_of(w1[s0 + 1], H, rank * ls, ls), at + ": shared up_proj");
        require(p[s0 + 2] == cols_of(w1[s0 + 2], S, rank * ls, ls), at + ": shared down_proj");
        // The routed experts: rows of gate/up, whole 16-blocks of columns of
        // down; every rank holds every global scale.
        const LayerCopy& a = one[static_cast<size_t>(l)];
        const LayerCopy& b = part[static_cast<size_t>(l)];
        for (int64_t e = 0; e < E; ++e) {
          const size_t m = static_cast<size_t>(e) * 3;
          for (size_t i = 0; i < 2; ++i) {
            require(b.expert_codes[m + i] == rows_of(a.expert_codes[m + i], H / 2, rank * li, li),
                    at + ": expert gate/up codes");
            require(b.expert_scales[m + i] == rows_of(a.expert_scales[m + i], H / 16, rank * li, li),
                    at + ": expert gate/up block scales");
          }
          require(b.expert_codes[m + 2] == cols_of(a.expert_codes[m + 2], I / 2, rank * li / 2, li / 2),
                  at + ": expert down codes");
          require(b.expert_scales[m + 2] == cols_of(a.expert_scales[m + 2], I / 16, rank * li / 16, li / 16),
                  at + ": expert down block scales");
        }
        require(b.expert_globals == a.expert_globals, at + ": expert global scales are replicated");
      }
    }
}

DGPP_TEST(qwen3codernext_loader_fp8_form_is_the_encode_of_the_bf16_form) {
  const Fixture fx = coder_next_fixture();
  for (int world : {1, 2, 4})
    for (int rank = 0; rank < world; ++rank) {
      const std::string at = " at world " + std::to_string(world) + " rank " + std::to_string(rank);
      const std::vector<LayerCopy> bf16 = load_bf16_form(fx, rank, world);
      const DenseForm fp8(true);
      Qwen35LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::LoaderResidency::Streaming,
                          dgpp::LoaderHeadSharding::VocabSharded);
      for (int l = 0; l < s.max_layer(); ++l) {
        const Qwen35LayerResident& r = s.load_layer(l);
        const LayerCopy& want = bf16[static_cast<size_t>(l)];
        const std::vector<Dense> dense = dense_of(fx.cfg, s.geometry(), r);
        require(dense.size() == want.dense.size(), "dense count");
        for (size_t i = 0; i < dense.size(); ++i) {
          const Dense& dn = dense[i];
          const std::string what = "layer " + std::to_string(l) + " " + dn.name + at;
          require(dn.bf16 == nullptr, what + ": the BF16 pointer is null in the FP8 form");
          const GlmQuantMatrix& q = *dn.fp8;
          require(q.rows == dn.rows && q.cols == dn.cols && q.scale_block_rows == 128 &&
                      q.scale_block_cols == 128 && dn.rows % 128 == 0 && dn.cols % 128 == 0,
                  what + ": shape and scale grid");
          const size_t n = static_cast<size_t>(dn.rows * dn.cols);
          const size_t blocks = static_cast<size_t>((dn.rows / 128) * (dn.cols / 128));
          std::vector<uint8_t> codes(n);
          std::vector<float> scales(blocks);
          dgpp::fp8_quant::encode_block128(want.dense[i].data(), static_cast<size_t>(dn.cols), dn.rows,
                                           dn.cols, codes.data(), scales.data(), /*threads=*/1);
          require(down(q.payload, n, what) == codes, what + ": codes are the encode of the BF16 form");
          const std::vector<float> got = down(q.scales, blocks, what);
          require(std::memcmp(got.data(), scales.data(), blocks * 4) == 0,
                  what + ": block scales are the encode of the BF16 form");
        }
        require(rest_of(fx.cfg, s.geometry(), r) == want.rest,
                "layer " + std::to_string(l) + at + ": everything but the dense stack is unchanged");
        require(r.bytes < want.bytes, "layer " + std::to_string(l) + at + ": the FP8 form is smaller");
      }
    }
}

// ---- Qwen3.6-35B-A3B -------------------------------------------------------------

namespace {

Fixture qwen36_fixture(const char* quant, const char* dir) {
  return qwen35portsfx::write_fixture((fs::current_path() / dir).string(),
                                      qwen35portsfx::qwen36_config_json(quant).c_str());
}

// The modelopt NVFP4 matrix `base` [N, K] of the fixture as BF16:
// bf16((code x block scale) x weight_scale_2).
std::vector<uint16_t> modelopt_matrix(const Fixture& fx, const std::string& base, int64_t N, int64_t K) {
  const std::vector<uint8_t>& codes = fx.bytes(base + ".weight");
  const std::vector<uint8_t>& scales = fx.bytes(base + ".weight_scale");
  const float ws2 = fx.f32(base + ".weight_scale_2");
  require(static_cast<int64_t>(codes.size()) == N * K / 2, base + ": fixture geometry");
  std::vector<uint16_t> out(static_cast<size_t>(N * K));
  for (int64_t n = 0; n < N; ++n)
    for (int64_t k = 0; k < K; ++k) {
      const uint8_t byte = codes[static_cast<size_t>(n * (K / 2) + k / 2)];
      const uint8_t code = (k & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0xF);
      const float block = e2m1(code) * e4m3(scales[static_cast<size_t>(n * (K / 16) + k / 16)]);
      out[static_cast<size_t>(n * K + k)] = dgpp::float_to_bf16_bits(block * ws2);
    }
  return out;
}

// A block-form FP8 resident against its expectation: the codes, and the
// scale grid as F32.
void require_fp8(const GlmQuantMatrix& q, int64_t rows, int64_t cols, const std::vector<uint8_t>& codes,
                 const std::vector<float>& scales, const std::string& what) {
  require(q.rows == rows && q.cols == cols, what + ": shape");
  require(down(q.payload, static_cast<size_t>(rows * cols), what) == codes, what + ": codes");
  const size_t blocks = static_cast<size_t>(q.scale_rows() * q.scale_cols());
  require(blocks == scales.size(), what + ": scale grid size " + std::to_string(blocks));
  const std::vector<float> got = down(q.scales, blocks, what);
  require(std::memcmp(got.data(), scales.data(), blocks * 4) == 0, what + ": scales");
}

// A per-tensor FP8 projection: the codes as shipped, every block the tensor's scale.
void require_fp8_tensor(const Fixture& fx, const GlmQuantMatrix& q, const std::string& base, int64_t rows,
                        int64_t cols, const std::string& what) {
  const std::vector<float> scales(static_cast<size_t>(((rows + 127) / 128) * ((cols + 127) / 128)),
                                  fx.f32(base + ".weight_scale"));
  require(q.scale_block_rows == 128 && q.scale_block_cols == 128, what + ": the 128 x 128 grid");
  require_fp8(q, rows, cols, fx.bytes(base + ".weight"), scales, what);
}

// A block-FP8 matrix of the FP8 release: the codes as shipped, the BF16 scales widened.
void require_fp8_block(const Fixture& fx, const GlmQuantMatrix& q, const std::string& name, int64_t rows,
                       int64_t cols, const std::string& what) {
  std::vector<float> scales;
  for (const uint16_t b : fx.bf16(name + "_scale_inv")) scales.push_back(dgpp::bf16_bits_to_float(b));
  require_fp8(q, rows, cols, fx.bytes(name), scales, what);
}

// The block-FP8 encode of a BF16 [rows, cols] matrix (loaders/fp8_quant.hpp).
void require_fp8_encode(const GlmQuantMatrix& q, const std::vector<uint16_t>& bf16, int64_t rows, int64_t cols,
                        const std::string& what) {
  std::vector<uint8_t> codes(static_cast<size_t>(rows * cols));
  std::vector<float> scales(static_cast<size_t>(((rows + 127) / 128) * ((cols + 127) / 128)));
  dgpp::fp8_quant::encode_block128(bf16.data(), static_cast<size_t>(cols), rows, cols, codes.data(),
                                   scales.data(), /*threads=*/1);
  require_fp8(q, rows, cols, codes, scales, what);
}

// The GDN and attention norms and BF16 leftovers every container shares.
void require_gdn_bf16_parts(const Fixture& fx, const Qwen35LayerResident& r, const std::string& g,
                            const std::string& at) {
  const Qwen35TextConfig& c = fx.cfg;
  const size_t H = static_cast<size_t>(c.hidden_size), vh = static_cast<size_t>(c.gdn_value_heads);
  require(down(r.gdn.in_proj_a, vh * H, "a") == fx.bf16(g + "in_proj_a.weight") &&
              down(r.gdn.in_proj_b, vh * H, "b") == fx.bf16(g + "in_proj_b.weight"),
          at + "in_proj_a / in_proj_b");
  require(down(r.gdn.conv, fx.bytes(g + "conv1d.weight").size() / 2, "conv") == fx.bf16(g + "conv1d.weight"),
          at + "conv");
  require(down(r.gdn.norm, static_cast<size_t>(c.gdn_value_head_dim), "norm") == fx.bf16(g + "norm.weight"),
          at + "gdn norm");
  const std::vector<uint16_t> a_log = fx.bf16(g + "A_log"), dt = fx.bf16(g + "dt_bias");
  const std::vector<float> got_a = down(r.gdn.a_log, a_log.size(), "a_log");
  const std::vector<float> got_d = down(r.gdn.dt_bias, dt.size(), "dt_bias");
  for (size_t i = 0; i < a_log.size(); ++i)
    require(got_a[i] == dgpp::bf16_bits_to_float(a_log[i]) && got_d[i] == dgpp::bf16_bits_to_float(dt[i]),
            at + "A_log and dt_bias widened");
}

}  // namespace

DGPP_TEST(qwen36_nvfp4_mixed_loader_resident_values_are_the_checkpoints) {
  const Fixture fx = qwen36_fixture(qwen35portsfx::kQuantModeloptMixed, "qwen36_nvfp4_loader_fixture");
  const Qwen35TextConfig& c = fx.cfg;
  require(!c.next() && c.moe() && c.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Mixed && c.mtp_layer() == 4,
          "the fixture's dialect and recipe");
  const int64_t H = c.hidden_size, d = c.head_dim, E = c.num_experts, I = c.moe_intermediate_size;
  const int64_t S = c.shared_expert_intermediate_size, qh = c.num_attention_heads, kvh = c.num_key_value_heads;
  const int64_t kdim = static_cast<int64_t>(c.gdn_key_heads) * c.gdn_key_head_dim;
  const int64_t vdim = static_cast<int64_t>(c.gdn_value_heads) * c.gdn_value_head_dim;
  const DenseForm bf16(false);
  Qwen35LayerStream s(c, fx.dir, 0, 1, dgpp::LoaderResidency::Streaming, dgpp::LoaderHeadSharding::Full);
  require(s.max_layer() == 5, "four layers and the draft layer");
  static const char* kPart[3] = {"gate_proj", "up_proj", "down_proj"};
  for (int l = 0; l < s.max_layer(); ++l) {
    const Qwen35LayerResident& r = s.load_layer(l);
    const bool is_mtp = l == c.mtp_layer();
    const std::string p = dgpp::qwen35_layer_prefix(c, l);
    const std::string at = "layer " + std::to_string(l) + ": ";
    require(p == (is_mtp ? "mtp.layers.0." : "model.language_model.layers." + std::to_string(l) + "."),
            at + "the nested prefix");
    require(down(r.input_norm, static_cast<size_t>(H), "norm") == fx.bf16(p + "input_layernorm.weight"),
            at + "input norm");
    require(r.kv_cache_scales == nullptr, at + "no K/V-cache scales in this release");
    if (!is_mtp && r.kind == dgpp::Qwen35LayerKind::Gdn) {
      const std::string g = p + "linear_attn.";
      // The split, head-major projections: the codes as shipped (world 1
      // takes every row), one scale for the whole tensor.
      require(r.gdn.in_proj_qkv == nullptr && r.gdn.in_proj_z == nullptr && r.gdn.out_proj == nullptr,
              at + "the FP8 projections have no BF16 view");
      require_fp8_tensor(fx, r.gdn.in_proj_qkv_fp8, g + "in_proj_qkv", 2 * kdim + vdim, H, at + "in_proj_qkv");
      require_fp8_tensor(fx, r.gdn.in_proj_z_fp8, g + "in_proj_z", vdim, H, at + "in_proj_z");
      require_fp8_tensor(fx, r.gdn.out_proj_fp8, g + "out_proj", H, vdim, at + "out_proj");
      require_gdn_bf16_parts(fx, r, g, at);
    } else if (!is_mtp) {
      const std::string a = p + "self_attn.";
      require(r.full.q_proj == nullptr && r.full.o_proj == nullptr, at + "no BF16 view of the FP8 attention");
      require_fp8_tensor(fx, r.full.q_proj_fp8, a + "q_proj", 2 * qh * d, H, at + "q_proj");
      require_fp8_tensor(fx, r.full.k_proj_fp8, a + "k_proj", kvh * d, H, at + "k_proj");
      require_fp8_tensor(fx, r.full.v_proj_fp8, a + "v_proj", kvh * d, H, at + "v_proj");
      require_fp8_tensor(fx, r.full.o_proj_fp8, a + "o_proj", H, qh * d, at + "o_proj");
      require(down(r.full.q_norm, static_cast<size_t>(d), "qn") == fx.bf16(a + "q_norm.weight"), at + "q norm");
    } else {
      // The draft layer: BF16 as shipped.
      const std::string a = p + "self_attn.";
      require(r.kind == dgpp::Qwen35LayerKind::Full && r.full.q_proj_fp8.payload == nullptr,
              at + "the draft layer is a BF16 attention layer");
      require(down(r.full.q_proj, static_cast<size_t>(2 * qh * d * H), "q") == fx.bf16(a + "q_proj.weight") &&
                  down(r.full.k_proj, static_cast<size_t>(kvh * d * H), "k") == fx.bf16(a + "k_proj.weight") &&
                  down(r.full.v_proj, static_cast<size_t>(kvh * d * H), "v") == fx.bf16(a + "v_proj.weight") &&
                  down(r.full.o_proj, static_cast<size_t>(H * qh * d), "o") == fx.bf16(a + "o_proj.weight"),
              at + "the draft attention is the checkpoint's BF16");
    }
    const std::string m = p + "mlp.";
    require(down(r.moe.router, static_cast<size_t>(E * H), "router") == fx.bf16(m + "gate.weight") &&
                down(r.moe.shared_gate, static_cast<size_t>(H), "gate") == fx.bf16(m + "shared_expert_gate.weight"),
            at + "router and shared gate");
    require(!r.mlp.gate && !r.mlp.gate_fp8.payload, at + "the dense mlp stays empty");
    if (!is_mtp) {
      require(down(r.moe.shared[0], static_cast<size_t>(S * H), "sg") ==
                      modelopt_matrix(fx, m + "shared_expert.gate_proj", S, H) &&
                  down(r.moe.shared[1], static_cast<size_t>(S * H), "su") ==
                      modelopt_matrix(fx, m + "shared_expert.up_proj", S, H) &&
                  down(r.moe.shared[2], static_cast<size_t>(H * S), "sd") ==
                      modelopt_matrix(fx, m + "shared_expert.down_proj", H, S),
              at + "the shared expert is the modelopt dequant");
      require(r.moe.experts_fp4.size() == static_cast<size_t>(E) * 3 && r.moe.experts.empty(),
              at + "NVFP4 backbone experts");
      const std::vector<float> globals = down(r.moe.expert_globals, static_cast<size_t>(E) * 3, "globals");
      float w13 = 0.0f, w2 = 0.0f;
      for (int64_t e = 0; e < E; ++e)
        for (int i = 0; i < 3; ++i) {
          const std::string base = m + "experts." + std::to_string(e) + "." + kPart[i];
          const GlmFp4Matrix& q = r.moe.experts_fp4[static_cast<size_t>(e * 3 + i)];
          require(down(q.payload, q.payload_bytes(), base) == fx.bytes(base + ".weight") &&
                      down(q.scales, q.scale_bytes(), base) == fx.bytes(base + ".weight_scale"),
                  at + base + " as shipped");
          require(globals[static_cast<size_t>(e * 3 + i)] == 1.0f / fx.f32(base + ".weight_scale_2"),
                  at + base + ": the reciprocal of the multiplier");
          (i == 2 ? w2 : w13) = std::max(i == 2 ? w2 : w13, fx.f32(base + ".input_scale"));
        }
      const std::vector<float> act = down(r.moe.act_scales, 2, "act scales");
      require(act[0] == w13 && act[1] == w2, at + "the layer's static activation scales");
    } else {
      require(down(r.moe.shared[0], static_cast<size_t>(S * H), "sg") == fx.bf16(m + "shared_expert.gate_proj.weight") &&
                  down(r.moe.shared[2], static_cast<size_t>(H * S), "sd") ==
                      fx.bf16(m + "shared_expert.down_proj.weight"),
              at + "the draft shared expert is the checkpoint's BF16");
      // The stacked draft experts: expert e's gate rows, then its up rows, of
      // gate_up_proj [E, 2I, H]; its [H, I] block of down_proj.
      require(r.moe.experts.size() == static_cast<size_t>(E) * 3 && r.moe.experts_fp4.empty(),
              at + "the draft experts are block FP8");
      const std::vector<uint16_t> gup = fx.bf16(m + "experts.gate_up_proj");
      const std::vector<uint16_t> dwn = fx.bf16(m + "experts.down_proj");
      require(static_cast<int64_t>(gup.size()) == E * 2 * I * H && static_cast<int64_t>(dwn.size()) == E * H * I,
              "stacked fixture geometry");
      for (int64_t e = 0; e < E; ++e) {
        const std::string who = at + "draft expert " + std::to_string(e);
        require_fp8_encode(r.moe.experts[static_cast<size_t>(e * 3)], rows_of(gup, H, e * 2 * I, I), I, H,
                           who + " gate");
        require_fp8_encode(r.moe.experts[static_cast<size_t>(e * 3 + 1)], rows_of(gup, H, e * 2 * I + I, I), I,
                           H, who + " up");
        require_fp8_encode(r.moe.experts[static_cast<size_t>(e * 3 + 2)], rows_of(dwn, I, e * H, H), H, I,
                           who + " down");
      }
    }
  }
  // Globals: the NVFP4 head dequantized into the BF16 head, the draft head, the names nested.
  const dgpp::Qwen35GlobalsResident& g = s.load_globals();
  const size_t V = static_cast<size_t>(c.vocab_size);
  require(down(g.embed, V * H, "embed") == fx.bf16("model.language_model.embed_tokens.weight"), "embed");
  require(down(g.lm_head, V * H, "head") == modelopt_matrix(fx, "lm_head", c.vocab_size, H),
          "the lm head is the modelopt dequant of its NVFP4 set");
  require(down(g.final_norm, static_cast<size_t>(H), "norm") == fx.bf16("model.language_model.norm.weight"),
          "final norm");
  require(down(g.mtp_fc, static_cast<size_t>(H * 2 * H), "fc") == fx.bf16("mtp.fc.weight"), "mtp fc");

  // Rank slices at world 2: the per-tensor projections' rows and columns,
  // and the draft experts' slices of the stacked tensors.
  for (int rank = 0; rank < 2; ++rank) {
    Qwen35LayerStream s2(c, fx.dir, rank, 2, dgpp::LoaderResidency::Streaming,
                         dgpp::LoaderHeadSharding::VocabSharded);
    const std::string at = "world 2 rank " + std::to_string(rank) + ": ";
    {
      const Qwen35LayerResident& r = s2.load_layer(0);
      const std::string g0 = "model.language_model.layers.0.linear_attn.";
      const std::vector<uint8_t>& qkv = fx.bytes(g0 + "in_proj_qkv.weight");
      std::vector<uint8_t> want = rows_of(qkv, H, rank * kdim / 2, kdim / 2);
      append(want, rows_of(qkv, H, kdim + rank * kdim / 2, kdim / 2));
      append(want, rows_of(qkv, H, 2 * kdim + rank * vdim / 2, vdim / 2));
      require(down(r.gdn.in_proj_qkv_fp8.payload, want.size(), "qkv") == want, at + "in_proj_qkv segments");
      require(down(r.gdn.out_proj_fp8.payload, static_cast<size_t>(H * vdim / 2), "out") ==
                  cols_of(fx.bytes(g0 + "out_proj.weight"), vdim, rank * vdim / 2, vdim / 2),
              at + "out_proj columns");
      const std::vector<float> sc = down(r.gdn.out_proj_fp8.scales, 2, "scales");
      require(sc[0] == fx.f32(g0 + "out_proj.weight_scale") && sc[1] == sc[0], at + "the slice keeps the scale");
    }
    {
      const Qwen35LayerResident& r = s2.load_layer(4);
      const std::vector<uint16_t> gup = fx.bf16("mtp.layers.0.mlp.experts.gate_up_proj");
      const std::vector<uint16_t> dwn = fx.bf16("mtp.layers.0.mlp.experts.down_proj");
      const int64_t li = I / 2;
      for (int64_t e = 0; e < E; ++e) {
        require_fp8_encode(r.moe.experts[static_cast<size_t>(e * 3)],
                           rows_of(gup, H, e * 2 * I + rank * li, li), li, H, at + "draft gate slice");
        require_fp8_encode(r.moe.experts[static_cast<size_t>(e * 3 + 1)],
                           rows_of(gup, H, e * 2 * I + I + rank * li, li), li, H, at + "draft up slice");
        require_fp8_encode(r.moe.experts[static_cast<size_t>(e * 3 + 2)],
                           cols_of(rows_of(dwn, I, e * H, H), I, rank * li, li), H, li,
                           at + "draft down slice");
      }
    }
    const dgpp::Qwen35GlobalsResident& g2 = s2.load_globals();
    const size_t v0 = V * static_cast<size_t>(rank) / 2, vn = V / 2;
    require(down(g2.lm_head, vn * H, "head") ==
                rows_of(modelopt_matrix(fx, "lm_head", c.vocab_size, H), H, static_cast<int64_t>(v0),
                        static_cast<int64_t>(vn)),
            at + "the head's vocab shard");
  }

  // dense_weights fp8: what ships BF16 or NVFP4 among the dense matrices —
  // the shared expert, the draft layer — is encoded; the FP8 projections and
  // the experts do not move.
  {
    const DenseForm fp8(true);
    Qwen35LayerStream s8(c, fx.dir, 0, 1, dgpp::LoaderResidency::Streaming, dgpp::LoaderHeadSharding::Full);
    const std::string m0 = "model.language_model.layers.0.mlp.";
    const Qwen35LayerResident& r0 = s8.load_layer(0);
    require(r0.moe.shared[0] == nullptr, "the FP8 form has no BF16 shared expert");
    require_fp8_encode(r0.moe.shared_fp8[0], modelopt_matrix(fx, m0 + "shared_expert.gate_proj", S, H), S, H,
                       "fp8 form: shared gate_proj");
    require_fp8_encode(r0.moe.shared_fp8[2], modelopt_matrix(fx, m0 + "shared_expert.down_proj", H, S), H, S,
                       "fp8 form: shared down_proj");
    require_fp8_tensor(fx, r0.gdn.in_proj_z_fp8, "model.language_model.layers.0.linear_attn.in_proj_z", vdim, H,
                       "fp8 form: the per-tensor projection is unchanged");
    const Qwen35LayerResident& rd = s8.load_layer(4);
    require(rd.full.q_proj == nullptr, "the FP8 form has no BF16 draft attention");
    require_fp8_encode(rd.full.q_proj_fp8, fx.bf16("mtp.layers.0.self_attn.q_proj.weight"), 2 * qh * d, H,
                       "fp8 form: draft q_proj");
    require_fp8_encode(rd.full.o_proj_fp8, fx.bf16("mtp.layers.0.self_attn.o_proj.weight"), H, qh * d,
                       "fp8 form: draft o_proj");
    require_fp8_encode(rd.moe.shared_fp8[1], fx.bf16("mtp.layers.0.mlp.shared_expert.up_proj.weight"), S, H,
                       "fp8 form: draft shared up_proj");
  }
}

DGPP_TEST(qwen36_fp8_loader_resident_values_are_the_checkpoints) {
  const Fixture fx = qwen36_fixture(qwen35portsfx::kQuantFp8Block, "qwen36_fp8_loader_fixture");
  const Qwen35TextConfig& c = fx.cfg;
  require(!c.next() && c.moe() && c.quant_kind == dgpp::Qwen35QuantKind::Fp8Block, "the fixture's recipe");
  const int64_t H = c.hidden_size, d = c.head_dim, E = c.num_experts, I = c.moe_intermediate_size;
  const int64_t S = c.shared_expert_intermediate_size, qh = c.num_attention_heads, kvh = c.num_key_value_heads;
  const int64_t kdim = static_cast<int64_t>(c.gdn_key_heads) * c.gdn_key_head_dim;
  const int64_t vdim = static_cast<int64_t>(c.gdn_value_heads) * c.gdn_value_head_dim;
  static const char* kPart[3] = {"gate_proj", "up_proj", "down_proj"};
  for (const bool knob : {false, true}) {  // the dense knob does not move a checkpoint that ships FP8
    const DenseForm form(knob);
    Qwen35LayerStream s(c, fx.dir, 0, 1, dgpp::LoaderResidency::Streaming, dgpp::LoaderHeadSharding::Full);
    for (int l = 0; l < s.max_layer(); ++l) {
      const Qwen35LayerResident& r = s.load_layer(l);
      const bool is_mtp = l == c.mtp_layer();
      const std::string p = dgpp::qwen35_layer_prefix(c, l);
      const std::string at = "layer " + std::to_string(l) + ": ";
      if (!is_mtp && r.kind == dgpp::Qwen35LayerKind::Gdn) {
        const std::string g = p + "linear_attn.";
        require_fp8_block(fx, r.gdn.in_proj_qkv_fp8, g + "in_proj_qkv.weight", 2 * kdim + vdim, H, at + "qkv");
        require_fp8_block(fx, r.gdn.in_proj_z_fp8, g + "in_proj_z.weight", vdim, H, at + "z");
        require_fp8_block(fx, r.gdn.out_proj_fp8, g + "out_proj.weight", H, vdim, at + "out");
        require_gdn_bf16_parts(fx, r, g, at);
      } else {
        const std::string a = p + "self_attn.";
        require_fp8_block(fx, r.full.q_proj_fp8, a + "q_proj.weight", 2 * qh * d, H, at + "q");
        require_fp8_block(fx, r.full.k_proj_fp8, a + "k_proj.weight", kvh * d, H, at + "k");
        require_fp8_block(fx, r.full.v_proj_fp8, a + "v_proj.weight", kvh * d, H, at + "v");
        require_fp8_block(fx, r.full.o_proj_fp8, a + "o_proj.weight", H, qh * d, at + "o");
      }
      const std::string m = p + "mlp.";
      require(down(r.moe.router, static_cast<size_t>(E * H), "router") == fx.bf16(m + "gate.weight"),
              at + "router");
      require(r.moe.shared[0] == nullptr && r.moe.experts_fp4.empty() &&
                  r.moe.experts.size() == static_cast<size_t>(E) * 3 && r.moe.scale_block == 64,
              at + "FP8 shared expert and experts, the sliced axis re-blocked at gcd(128, 64)");
      require_fp8_block(fx, r.moe.shared_fp8[0], m + "shared_expert.gate_proj.weight", S, H, at + "shared gate");
      require_fp8_block(fx, r.moe.shared_fp8[1], m + "shared_expert.up_proj.weight", S, H, at + "shared up");
      require_fp8_block(fx, r.moe.shared_fp8[2], m + "shared_expert.down_proj.weight", H, S, at + "shared down");
      for (int64_t e = 0; e < E; ++e)
        for (int i = 0; i < 3; ++i) {
          const std::string name = m + "experts." + std::to_string(e) + "." + kPart[i] + ".weight";
          require_fp8_block(fx, r.moe.experts[static_cast<size_t>(e * 3 + i)], name, i == 2 ? H : I,
                            i == 2 ? I : H, at + name);
        }
    }
    const dgpp::Qwen35GlobalsResident& g = s.load_globals();
    require(down(g.lm_head, static_cast<size_t>(c.vocab_size) * H, "head") == fx.bf16("lm_head.weight"),
            "the BF16 head as shipped");
  }
  // World 2: an expert's rows and columns, its scale grid re-blocked at 32.
  for (int rank = 0; rank < 2; ++rank) {
    const DenseForm form(false);
    Qwen35LayerStream s2(c, fx.dir, rank, 2, dgpp::LoaderResidency::Streaming,
                         dgpp::LoaderHeadSharding::VocabSharded);
    const Qwen35LayerResident& r = s2.load_layer(3);
    const std::string ep = "model.language_model.layers.3.mlp.experts.1.";
    const int64_t li = I / 2;
    const GlmQuantMatrix& gate = r.moe.experts[3];
    const GlmQuantMatrix& dn = r.moe.experts[5];
    require(r.moe.scale_block == 32 && gate.rows == li && gate.scale_block_rows == 32 && dn.cols == li &&
                dn.scale_block_cols == 32,
            "world 2 expert slices");
    require(down(gate.payload, static_cast<size_t>(li * H), "gate") ==
                rows_of(fx.bytes(ep + "gate_proj.weight"), H, rank * li, li),
            "world 2: expert gate rows");
    require(down(dn.payload, static_cast<size_t>(H * li), "down") ==
                cols_of(fx.bytes(ep + "down_proj.weight"), I, rank * li, li),
            "world 2: expert down columns");
    // The parent 128-block's scale stands for every sub-block.
    const std::vector<uint16_t> gs = fx.bf16(ep + "gate_proj.weight_scale_inv");  // [1, 2]
    const std::vector<float> got = down(gate.scales, 2, "gate scales");
    require(got[0] == dgpp::bf16_bits_to_float(gs[0]) && got[1] == dgpp::bf16_bits_to_float(gs[1]),
            "world 2: expert gate scales");
  }
}

// ---- Qwen3.5-0.8B -----------------------------------------------------------------

DGPP_TEST(qwen35_bf16_loader_resident_values_are_the_checkpoints) {
  const Fixture fx = qwen35portsfx::write_fixture((fs::current_path() / "qwen35_bf16_loader_fixture").string(),
                                                  qwen35portsfx::qwen35_bf16_config_json());
  const Qwen35TextConfig& c = fx.cfg;
  require(!c.next() && !c.moe() && c.quant_kind == dgpp::Qwen35QuantKind::Bf16 && c.tie_word_embeddings &&
              c.mtp_layer() == 4,
          "the fixture's dialect and recipe");
  require(fx.src.count("lm_head.weight") == 0, "a tied checkpoint stores no head");
  const int64_t H = c.hidden_size, d = c.head_dim, I = c.intermediate_size;
  const int64_t qh = c.num_attention_heads, kvh = c.num_key_value_heads;
  const int64_t kdim = static_cast<int64_t>(c.gdn_key_heads) * c.gdn_key_head_dim;
  const int64_t vdim = static_cast<int64_t>(c.gdn_value_heads) * c.gdn_value_head_dim;
  const size_t V = static_cast<size_t>(c.vocab_size);
  {
    const DenseForm bf16(false);
    Qwen35LayerStream s(c, fx.dir, 0, 1, dgpp::LoaderResidency::Streaming, dgpp::LoaderHeadSharding::Full);
    for (int l = 0; l < s.max_layer(); ++l) {
      const Qwen35LayerResident& r = s.load_layer(l);
      const bool is_mtp = l == c.mtp_layer();
      const std::string p = dgpp::qwen35_layer_prefix(c, l);
      const std::string at = "layer " + std::to_string(l) + ": ";
      if (!is_mtp && r.kind == dgpp::Qwen35LayerKind::Gdn) {
        const std::string g = p + "linear_attn.";
        require(r.gdn.in_proj_qkv_fp8.payload == nullptr && r.gdn.out_proj_fp8.payload == nullptr,
                at + "no fp8 view in the BF16 form");
        // Split and head-major as shipped: world 1 takes every row.
        require(down(r.gdn.in_proj_qkv, static_cast<size_t>((2 * kdim + vdim) * H), "qkv") ==
                    fx.bf16(g + "in_proj_qkv.weight"),
                at + "in_proj_qkv");
        require(down(r.gdn.in_proj_z, static_cast<size_t>(vdim * H), "z") == fx.bf16(g + "in_proj_z.weight"),
                at + "in_proj_z");
        require(down(r.gdn.out_proj, static_cast<size_t>(H * vdim), "out") == fx.bf16(g + "out_proj.weight"),
                at + "out_proj");
        require(down(r.gdn.in_proj_a, static_cast<size_t>(c.gdn_value_heads * H), "a") ==
                    fx.bf16(g + "in_proj_a.weight"),
                at + "in_proj_a");
        // A_log: F32 in this release, as stored. The norm weight: F32,
        // rounded to the BF16 the gated norm reads. dt_bias: BF16, widened.
        const std::vector<uint8_t>& a_log = fx.bytes(g + "A_log");
        require(a_log.size() == static_cast<size_t>(c.gdn_value_heads) * 4 &&
                    down(reinterpret_cast<const uint8_t*>(r.gdn.a_log), a_log.size(), "a_log") == a_log,
                at + "A_log is the stored F32");
        const std::vector<uint8_t>& norm = fx.bytes(g + "norm.weight");
        const std::vector<uint16_t> got_norm = down(r.gdn.norm, static_cast<size_t>(c.gdn_value_head_dim), "norm");
        bool inexact = false;
        for (size_t i = 0; i < got_norm.size(); ++i) {
          float v;
          std::memcpy(&v, &norm[i * 4], 4);
          require(got_norm[i] == dgpp::float_to_bf16_bits(v), at + "the norm weight rounded to BF16");
          inexact = inexact || dgpp::bf16_bits_to_float(got_norm[i]) != v;
        }
        require(inexact, "the fixture's F32 norm weights are not BF16 values (the rounding is exercised)");
        const std::vector<uint16_t> dt = fx.bf16(g + "dt_bias");
        const std::vector<float> got_d = down(r.gdn.dt_bias, dt.size(), "dt_bias");
        for (size_t i = 0; i < dt.size(); ++i)
          require(got_d[i] == dgpp::bf16_bits_to_float(dt[i]), at + "dt_bias widened");
      } else {
        const std::string a = p + "self_attn.";
        require(r.full.q_proj_fp8.payload == nullptr, at + "no fp8 view in the BF16 form");
        require(down(r.full.q_proj, static_cast<size_t>(2 * qh * d * H), "q") == fx.bf16(a + "q_proj.weight") &&
                    down(r.full.k_proj, static_cast<size_t>(kvh * d * H), "k") == fx.bf16(a + "k_proj.weight") &&
                    down(r.full.v_proj, static_cast<size_t>(kvh * d * H), "v") == fx.bf16(a + "v_proj.weight") &&
                    down(r.full.o_proj, static_cast<size_t>(H * qh * d), "o") == fx.bf16(a + "o_proj.weight"),
                at + "the attention projections as shipped");
        require(r.kv_cache_scales == nullptr, at + "no cache scales");
      }
      // The dense MLP through its BF16 pointers.
      require(r.mlp.gate_fp8.payload == nullptr && r.moe.router == nullptr, at + "a BF16 dense MLP, no MoE");
      require(down(r.mlp.gate, static_cast<size_t>(I * H), "gate") == fx.bf16(p + "mlp.gate_proj.weight") &&
                  down(r.mlp.up, static_cast<size_t>(I * H), "up") == fx.bf16(p + "mlp.up_proj.weight") &&
                  down(r.mlp.down, static_cast<size_t>(H * I), "down") == fx.bf16(p + "mlp.down_proj.weight"),
              at + "the dense MLP as shipped");
    }
    const dgpp::Qwen35GlobalsResident& g = s.load_globals();
    require(down(g.embed, V * H, "embed") == fx.bf16("model.language_model.embed_tokens.weight"), "embed");
    require(g.lm_head == g.embed && g.lm_vocab_begin == 0 && g.lm_vocab_count == c.vocab_size,
            "tied: the head is the embedding itself");
    require(g.bytes == Qwen35LayerStream::globals_bytes(c) &&
                g.bytes == dgpp::align_up_256(V * H * 2) + 4 * dgpp::align_up_256(static_cast<size_t>(H) * 2) +
                               dgpp::align_up_256(static_cast<size_t>(H) * 2 * H * 2),
            "no head grant among the globals");
    require(down(g.mtp_fc, static_cast<size_t>(H * 2 * H), "fc") == fx.bf16("mtp.fc.weight"), "mtp fc");
  }
  // World 2: the dense MLP's rows and columns, and the head shard as rows of the embedding.
  for (int rank = 0; rank < 2; ++rank) {
    const DenseForm bf16(false);
    Qwen35LayerStream s2(c, fx.dir, rank, 2, dgpp::LoaderResidency::Streaming,
                         dgpp::LoaderHeadSharding::VocabSharded);
    const Qwen35LayerResident& r = s2.load_layer(3);
    const std::string p = "model.language_model.layers.3.";
    const int64_t li = I / 2;
    require(down(r.mlp.gate, static_cast<size_t>(li * H), "gate") ==
                    rows_of(fx.bf16(p + "mlp.gate_proj.weight"), H, rank * li, li) &&
                down(r.mlp.down, static_cast<size_t>(H * li), "down") ==
                    cols_of(fx.bf16(p + "mlp.down_proj.weight"), I, rank * li, li),
            "world 2: the dense MLP slices");
    const dgpp::Qwen35GlobalsResident& g2 = s2.load_globals();
    const size_t v0 = V * static_cast<size_t>(rank) / 2, vn = V / 2;
    require(g2.lm_head == g2.embed + v0 * H && g2.lm_vocab_begin == static_cast<int>(v0) &&
                g2.lm_vocab_count == static_cast<int>(vn),
            "world 2: the head shard is this rank's rows of the embedding");
  }
  // dense_weights fp8: every dense matrix is the block-FP8 encode of what ships.
  {
    const DenseForm fp8(true);
    Qwen35LayerStream s8(c, fx.dir, 0, 1, dgpp::LoaderResidency::Streaming, dgpp::LoaderHeadSharding::Full);
    {
      const Qwen35LayerResident& r = s8.load_layer(0);
      const std::string p = "model.language_model.layers.0.";
      require(r.gdn.in_proj_qkv == nullptr && r.mlp.gate == nullptr, "the FP8 form has no BF16 views");
      require_fp8_encode(r.gdn.in_proj_qkv_fp8, fx.bf16(p + "linear_attn.in_proj_qkv.weight"), 2 * kdim + vdim, H,
                         "fp8 form: in_proj_qkv");
      require_fp8_encode(r.gdn.in_proj_z_fp8, fx.bf16(p + "linear_attn.in_proj_z.weight"), vdim, H,
                         "fp8 form: in_proj_z");
      require_fp8_encode(r.gdn.out_proj_fp8, fx.bf16(p + "linear_attn.out_proj.weight"), H, vdim,
                         "fp8 form: out_proj");
      require_fp8_encode(r.mlp.gate_fp8, fx.bf16(p + "mlp.gate_proj.weight"), I, H, "fp8 form: mlp gate");
      require_fp8_encode(r.mlp.up_fp8, fx.bf16(p + "mlp.up_proj.weight"), I, H, "fp8 form: mlp up");
      require_fp8_encode(r.mlp.down_fp8, fx.bf16(p + "mlp.down_proj.weight"), H, I, "fp8 form: mlp down");
    }
    {
      const Qwen35LayerResident& r = s8.load_layer(3);
      const std::string a = "model.language_model.layers.3.self_attn.";
      require_fp8_encode(r.full.q_proj_fp8, fx.bf16(a + "q_proj.weight"), 2 * qh * d, H, "fp8 form: q_proj");
      require_fp8_encode(r.full.o_proj_fp8, fx.bf16(a + "o_proj.weight"), H, qh * d, "fp8 form: o_proj");
    }
  }
}

int main() {
  return dgpp::test::run_all();
}
