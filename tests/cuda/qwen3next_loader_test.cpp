// The Qwen3Next dialect's resident loader on the synthetic fixture, in both
// forms of the dense stack (Qwen35LayerStream::set_dense_weights_fp8):
//   the BF16 form's rank slices at worlds 2 and 4 tile the world-1 layers
//   (a kv head shared by two ranks at world 4);
//   the FP8 form binds every dense projection through its `_fp8` member, as
//   the block-FP8 encode (loaders/fp8_quant.hpp) of exactly the matrix the
//   BF16 form holds at that rank, and leaves the rest of the layer alone;
//   the byte formulas equal actual usage (the stream throws otherwise), and
//   a resident image round trip restores each form bitwise and never
//   restores one form's image for the other.
// The release checkpoint's values are tests/cuda/qwen3next_loader_smoke.cpp's.
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
#include "qwen3next_loader_fixture.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::GlmQuantMatrix;
using dgpp::Qwen35LayerResident;
using dgpp::Qwen35LayerStream;
using dgpp::Qwen35LocalGeometry;
using dgpp::Qwen35TextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  Qwen35TextConfig cfg;
  std::string dir;
};

Fixture write_fixture() {
  Fixture fx;
  fx.cfg = qwen3nextfx::tiny_config();
  fx.dir = (fs::current_path() / "qwen3next_loader_fixture").string();
  qwen3nextfx::write_fixture(fx.dir);
  return fx;
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

// One dense projection of a resident layer: both views and its shape at
// the stream's geometry.
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
  if (r.layer != c.mtp_layer() && r.kind == dgpp::Qwen35LayerKind::Gdn) {
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

// The parts of a layer the dense form does not touch, as bytes: the norms,
// the GDN's a/b, conv, A_log, dt_bias and head norm (or the q/k norms and
// the K/V-cache scales), the router and shared gate, and the routed experts.
std::vector<uint8_t> rest_of(const Qwen35TextConfig& c, const Qwen35LocalGeometry& g,
                             const Qwen35LayerResident& r) {
  const size_t H = static_cast<size_t>(c.hidden_size);
  std::vector<uint8_t> out;
  const auto add = [&](const void* dev, size_t bytes, const char* what) {
    const std::vector<uint8_t> b = down(static_cast<const uint8_t*>(dev), bytes, what);
    out.insert(out.end(), b.begin(), b.end());
  };
  add(r.input_norm, H * 2, "input_norm");
  add(r.post_norm, H * 2, "post_norm");
  const bool is_mtp = r.layer == c.mtp_layer();
  if (!is_mtp && r.kind == dgpp::Qwen35LayerKind::Gdn) {
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
    if (!is_mtp) add(r.kv_cache_scales, 8, "kv cache scales");
  }
  add(r.moe.router, static_cast<size_t>(c.num_experts) * H * 2, "router");
  add(r.moe.shared_gate, H * 2, "shared gate");
  const size_t matrices = static_cast<size_t>(c.num_experts) * 3;
  if (is_mtp) {
    require(r.moe.experts.size() == matrices && !r.moe.nvfp4(), "the draft experts are block FP8");
    for (const GlmQuantMatrix& q : r.moe.experts) {
      add(q.payload, static_cast<size_t>(q.rows * q.cols), "draft expert codes");
      add(q.scales, q.scale_bytes(), "draft expert scales");
    }
  } else {
    require(r.moe.experts_fp4.size() == matrices && r.moe.experts.empty(),
            "the backbone experts are NVFP4");
    for (const dgpp::GlmFp4Matrix& q : r.moe.experts_fp4) {
      add(q.payload, q.payload_bytes(), "expert payload");
      add(q.scales, q.scale_bytes(), "expert scales");
    }
    add(r.moe.expert_globals, matrices * 4, "expert globals");
    add(r.moe.act_scales, 8, "activation scales");
  }
  return out;
}

struct LayerCopy {
  std::vector<std::vector<uint16_t>> dense;  // dense_of's order, BF16
  std::vector<uint8_t> rest;
  size_t bytes = 0;
};

// Every layer (the draft layer last) of the BF16 form at (rank, world).
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
    lc.bytes = r.bytes;
    out.push_back(std::move(lc));
  }
  return out;
}

std::vector<uint8_t> resident_bytes_of(const Qwen35LayerStream& s, int layer, size_t bytes) {
  std::vector<uint8_t> out(bytes);
  s.copy_resident_layer(layer, out.data());
  return out;
}

}  // namespace

DGPP_TEST(qwen3next_loader_fp8_form_is_the_encode_of_the_bf16_form) {
  const Fixture fx = write_fixture();
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
          const Dense& d = dense[i];
          const std::string what = "layer " + std::to_string(l) + " " + d.name + at;
          require(d.bf16 == nullptr, what + ": the BF16 pointer is null in the FP8 form");
          const GlmQuantMatrix& q = *d.fp8;
          require(q.rows == d.rows && q.cols == d.cols && q.scale_block_rows == 128 &&
                      q.scale_block_cols == 128 && d.rows % 128 == 0 && d.cols % 128 == 0,
                  what + ": shape and scale grid");
          // The repo's encoder over the BF16 form's matrix: codes and scales.
          const size_t n = static_cast<size_t>(d.rows * d.cols);
          const size_t blocks = static_cast<size_t>((d.rows / 128) * (d.cols / 128));
          std::vector<uint8_t> codes(n);
          std::vector<float> scales(blocks);
          dgpp::fp8_quant::encode_block128(want.dense[i].data(), static_cast<size_t>(d.cols),
                                           d.rows, d.cols, codes.data(), scales.data(),
                                           /*threads=*/1);
          require(down(q.payload, n, what) == codes,
                  what + ": codes are the encode of the BF16 form");
          const std::vector<float> got = down(q.scales, blocks, what);
          require(std::memcmp(got.data(), scales.data(), blocks * 4) == 0,
                  what + ": block scales are the encode of the BF16 form");
        }
        require(
            rest_of(fx.cfg, s.geometry(), r) == want.rest,
            "layer " + std::to_string(l) + at + ": everything but the dense stack is unchanged");
        require(r.bytes < want.bytes,
                "layer " + std::to_string(l) + at + ": the FP8 form is smaller");
        require(!r.mlp.gate && !r.mlp.gate_fp8.payload, "the dense mlp stays empty");
      }
    }
}

DGPP_TEST(qwen3next_loader_rank_slices_tile_world_one) {
  const Fixture fx = write_fixture();
  const Qwen35TextConfig& c = fx.cfg;
  const int64_t H = c.hidden_size, d = c.head_dim;
  const std::vector<LayerCopy> one = load_bf16_form(fx, 0, 1);
  // Rows [r0, +rn) of a [*, width] matrix; columns [c0, +cn) of a [*, full] one.
  const auto rows_of = [](const std::vector<uint16_t>& m, int64_t width, int64_t r0, int64_t rn) {
    return std::vector<uint16_t>(m.begin() + r0 * width, m.begin() + (r0 + rn) * width);
  };
  const auto cols_of = [](const std::vector<uint16_t>& m, int64_t full, int64_t c0, int64_t cn) {
    std::vector<uint16_t> out;
    for (int64_t r = 0; r < static_cast<int64_t>(m.size()) / full; ++r)
      out.insert(out.end(), m.begin() + r * full + c0, m.begin() + r * full + c0 + cn);
    return out;
  };
  for (int world : {2, 4})
    for (int rank = 0; rank < world; ++rank) {
      const std::vector<LayerCopy> part = load_bf16_form(fx, rank, world);
      const Qwen35LocalGeometry g =
          Qwen35LocalGeometry::from_config(c, rank, world, dgpp::LoaderHeadSharding::VocabSharded);
      const int64_t K = static_cast<int64_t>(c.gdn_key_heads) * c.gdn_key_head_dim;
      const int64_t V = static_cast<int64_t>(c.gdn_value_heads) * c.gdn_value_head_dim;
      const int64_t kd = K / world, vd = V / world, S = c.shared_expert_intermediate_size;
      const int64_t ls = S / world;
      for (int l = 0; l <= c.mtp_layer(); ++l) {
        const std::string at = "layer " + std::to_string(l) + " at world " + std::to_string(world) +
                               " rank " + std::to_string(rank);
        const auto& w1 = one[static_cast<size_t>(l)].dense;
        const auto& p = part[static_cast<size_t>(l)].dense;
        const bool gdn = l != c.mtp_layer() && c.layers[l] == dgpp::Qwen35LayerKind::Gdn;
        if (gdn) {
          // in_proj_qkv: this rank's q, k and v segments of the head-major rows.
          std::vector<uint16_t> qkv = rows_of(w1[0], H, rank * kd, kd);
          const std::vector<uint16_t> k = rows_of(w1[0], H, K + rank * kd, kd);
          const std::vector<uint16_t> v = rows_of(w1[0], H, 2 * K + rank * vd, vd);
          qkv.insert(qkv.end(), k.begin(), k.end());
          qkv.insert(qkv.end(), v.begin(), v.end());
          require(p[0] == qkv, at + ": in_proj_qkv");
          require(p[1] == rows_of(w1[1], H, rank * vd, vd), at + ": in_proj_z");
          require(p[2] == cols_of(w1[2], V, rank * vd, vd), at + ": out_proj");
        } else {
          require(p[0] == rows_of(w1[0], H, g.head_begin * 2 * d, g.local_heads * 2 * d),
                  at + ": q_proj");
          require(p[1] == rows_of(w1[1], H, g.kv_head_begin * d, g.local_kv_heads * d),
                  at + ": k_proj");
          require(p[2] == rows_of(w1[2], H, g.kv_head_begin * d, g.local_kv_heads * d),
                  at + ": v_proj");
          require(p[3] == cols_of(w1[3], c.num_attention_heads * d, g.head_begin * d,
                                  g.local_heads * d),
                  at + ": o_proj");
        }
        const size_t s0 = gdn ? 3 : 4;
        require(p[s0] == rows_of(w1[s0], H, rank * ls, ls), at + ": shared gate_proj");
        require(p[s0 + 1] == rows_of(w1[s0 + 1], H, rank * ls, ls), at + ": shared up_proj");
        require(p[s0 + 2] == cols_of(w1[s0 + 2], S, rank * ls, ls), at + ": shared down_proj");
      }
    }
}

DGPP_TEST(qwen3next_loader_image_round_trip_keeps_the_forms_apart) {
  const Fixture fx = write_fixture();
  const fs::path cache = fs::current_path() / "qwen3next_loader_image_cache";
  fs::remove_all(cache);
  const std::string saved = Qwen35LayerStream::resident_image_dir();
  Qwen35LayerStream::set_resident_image_dir(cache.string());
  const int layers = fx.cfg.num_hidden_layers + 1;
  const int rank = 1, world = 2;
  std::vector<std::vector<uint8_t>> built[2];  // [bf16, fp8][layer]
  for (int form = 1; form >= 0; --form) {  // the FP8 form first: its image exists when BF16 opens
    const DenseForm dense(form == 1);
    const std::string name = form == 1 ? "FP8" : "BF16";
    for (int pass = 0; pass < 2; ++pass) {
      Qwen35LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::LoaderResidency::Resident,
                          dgpp::LoaderHeadSharding::VocabSharded, /*resident_mtp=*/true);
      for (int l = 0; l < layers; ++l) {
        const Qwen35LayerResident& r = s.load_layer(l);
        const std::vector<uint8_t> bytes = resident_bytes_of(s, l, r.bytes);
        const bool fp8_bound =
            (r.kind == dgpp::Qwen35LayerKind::Gdn ? r.gdn.in_proj_qkv_fp8.payload
                                                  : r.full.q_proj_fp8.payload) != nullptr;
        require(fp8_bound == (form == 1), name + ": the layer is bound in its own form");
        if (pass == 0) {
          built[form].push_back(bytes);
        } else {
          require(bytes == built[form][static_cast<size_t>(l)],
                  name + " layer " + std::to_string(l) + ": restored bitwise the built layer");
          if (l == 3)
            require(
                r.kv_cache_scales != nullptr && r.k_cache_scale > 0.0f && r.v_cache_scale > 0.0f,
                name + ": the K/V-cache scales' host copies come back with a restore");
        }
      }
      // A cold load builds and captures; the next restores without a source
      // read. The BF16 stream opens beside the FP8 image and restores none of
      // it: the form is part of the image key.
      if (pass == 0)
        require(s.image_layers_captured() == layers && s.image_layers_restored() == 0,
                name + ": the cold load built every layer (no other form's image was restored)");
      else
        require(s.image_layers_restored() == layers && s.source_bytes_read() == 0,
                name + ": every layer restored from its own image");
    }
  }
  for (int l = 0; l < layers; ++l)
    require(built[0][static_cast<size_t>(l)].size() > built[1][static_cast<size_t>(l)].size(),
            "the two forms are different layouts");
  Qwen35LayerStream::set_resident_image_dir(saved);
  fs::remove_all(cache);
}

int main() {
  return dgpp::test::run_all();
}
