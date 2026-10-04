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

int main() {
  return dgpp::test::run_all();
}
