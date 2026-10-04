// Loader smoke test for the Qwen3Next dialect: Qwen35LayerStream against
// the release checkpoint (nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4). At
// world 1 it loads layer 0 (GDN), layer 3 (attention), the draft layer and
// the globals in streaming mode and compares every resident form with a
// host computation made here, straight from the shards mapped a second
// time — the format decoders below are written from the format definitions,
// not taken from the loader:
//   (a) the gathered GDN projections, row by row, against the interleaved
//       source rows;
//   (b) out_proj, o_proj and the shared expert against the NVFP4 dequant
//       bf16(e2m1(code) x e4m3(block scale) x weight_scale_2);
//   (c) routed experts 0, 255 and 511 against the shipped bytes, the global
//       being 1 / weight_scale_2;
//   (d) the draft layer's experts 0 and 511 against the block-FP8 recipe;
//   (e) everything else bitwise.
// It then repeats (a), (b) and (c) for both ranks of world 2, checks that the
// two ranks tile the world-1 matrices and agree on the replicated digest, and
// takes an attention layer and the draft layer of rank 1 through resident
// mode and a resident-image round trip (the image in a scratch directory
// under the system's temp dir, removed afterwards). One PASS/FAIL line per
// check; exit 1 on any failure, 3 when the node has too little memory.
//
// With `--dense fp8` it checks the block-FP8 form of the dense stack instead
// (Qwen35LayerStream::set_dense_weights_fp8): the BF16-form loader's dense
// matrices of layers 0 and 3 are first shown bitwise equal to the host
// values, then the FP8 form of layers 0, 3 and the draft layer at world 1 is
// checked against those BF16 values with the (d) recipe — block scales
// amax / 448, the quantization bound, nearest codes — the rest of each layer
// being what it was; and rank 1 of world 2 goes through resident mode and an
// image round trip in the FP8 form. Without the flag the dense stack is BF16
// and the checks are the ones above.
//
// `--plan` prints the world-1 resident bytes of both forms from the counting
// builds alone and exits: no checkpoint byte is read and nothing is loaded.
//
//   qwen3next_loader_smoke (--model ORG/NAME | --checkpoint-dir <dir>) [--dense bf16|fp8] [--plan]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "loaders/hf_cache.hpp"
#include "loaders/safetensors.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/loader35.hpp"

namespace {

using dgpp::GlmFp4Matrix;
using dgpp::GlmQuantMatrix;
using dgpp::Qwen35LayerResident;
using dgpp::Qwen35LayerStream;
using dgpp::Qwen35TextConfig;

int g_passed = 0, g_failed = 0;

void report(bool ok, const std::string& what, const std::string& detail = "") {
  std::printf("%s  %s%s%s\n", ok ? "PASS" : "FAIL", what.c_str(), detail.empty() ? "" : " — ",
              detail.c_str());
  std::fflush(stdout);
  ++(ok ? g_passed : g_failed);
}

// ---- the independent source: the shards, mapped here -------------------------

struct Source {
  std::vector<std::unique_ptr<dgpp::SafetensorsFile>> shards;
  std::unordered_map<std::string, const dgpp::TensorInfo*> tensors;

  explicit Source(const std::string& dir) {
    namespace fs = std::filesystem;
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir))
      if (entry.path().extension() == ".safetensors") paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    for (const auto& path : paths) {
      auto f = dgpp::SafetensorsFile::open(path.string());
      f->for_each([&](const dgpp::TensorInfo& t) { tensors.emplace(t.name, &t); });
      shards.push_back(std::move(f));
    }
  }
  const dgpp::TensorInfo& at(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("not in the checkpoint: " + name);
    return *it->second;
  }
  const uint8_t* bytes(const std::string& name) const {
    return static_cast<const uint8_t*>(at(name).data);
  }
  size_t nbytes(const std::string& name) const { return at(name).nbytes(); }
  // A BF16 tensor's bits (copied: safetensors does not align tensors).
  std::vector<uint16_t> bf16(const std::string& name) const {
    const dgpp::TensorInfo& t = at(name);
    if (t.dtype != dgpp::DType::BF16) throw std::runtime_error("not BF16: " + name);
    std::vector<uint16_t> v(t.numel());
    std::memcpy(v.data(), t.data, v.size() * 2);
    return v;
  }
  float f32(const std::string& name) const {
    const dgpp::TensorInfo& t = at(name);
    if (t.dtype != dgpp::DType::F32 || t.nbytes() != 4)
      throw std::runtime_error("not an F32 scalar: " + name);
    float v;
    std::memcpy(&v, t.data, 4);
    return v;
  }
};

// ---- the formats, from their definitions -------------------------------------

// e2m1: bit 3 the sign, magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6.
float e2m1(uint8_t code) {
  static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  return (code & 8u) ? -mag[code & 7u] : mag[code & 7u];
}

// e4m3 (OCP): sign, four exponent bits (bias 7), three mantissa bits;
// exponent 0 is subnormal (m x 2^-9); 0x7F / 0xFF are NaN.
const float* e4m3_table() {
  static float table[256];
  static bool built = false;
  if (!built) {
    for (int c = 0; c < 256; ++c) {
      const int e = (c >> 3) & 0xF, m = c & 7;
      float v = e == 0 ? std::ldexp(static_cast<float>(m), -9)
                       : std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
      if (e == 15 && m == 7) v = std::nanf("");
      table[c] = (c & 0x80) ? -v : v;
    }
    built = true;
  }
  return table;
}

float bf16_value(uint16_t bits) {
  const uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// fp32 -> bf16, round to nearest, ties to even (finite inputs).
uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  uint32_t hi = u >> 16;
  const uint32_t low = u & 0xFFFFu;
  if (low > 0x8000u || (low == 0x8000u && (hi & 1u))) ++hi;
  return static_cast<uint16_t>(hi);
}

// Rows [r0, +rn) x columns [c0, +cn) of the NVFP4 matrix `base`, dequantized:
// element (n, k) = bf16((e2m1(code) * e4m3(scale[n][k / 16])) * weight_scale_2),
// the low nibble of a payload byte being the even column.
std::vector<uint16_t> host_dequant(const Source& src, const std::string& base, int64_t r0,
                                   int64_t rn, int64_t c0, int64_t cn) {
  const dgpp::TensorInfo& tp = src.at(base + ".weight");
  const dgpp::TensorInfo& ts = src.at(base + ".weight_scale");
  const int64_t K = tp.shape[1] * 2;
  if (tp.dtype != dgpp::DType::U8 || ts.dtype != dgpp::DType::F8_E4M3 ||
      ts.shape[0] != tp.shape[0] || ts.shape[1] != K / 16 || r0 + rn > tp.shape[0] || c0 + cn > K)
    throw std::runtime_error("unexpected NVFP4 set: " + base);
  const uint8_t* payload = static_cast<const uint8_t*>(tp.data);
  const uint8_t* scales = static_cast<const uint8_t*>(ts.data);
  const float ws2 = src.f32(base + ".weight_scale_2");
  const float* e4m3 = e4m3_table();
  std::vector<uint16_t> out(static_cast<size_t>(rn) * static_cast<size_t>(cn));
  for (int64_t n = 0; n < rn; ++n)
    for (int64_t k = 0; k < cn; ++k) {
      const int64_t row = r0 + n, col = c0 + k;
      const uint8_t byte = payload[row * (K / 2) + col / 2];
      const uint8_t code =
          (col & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0xFu);
      const float block = e2m1(code) * e4m3[scales[row * (K / 16) + col / 16]];
      out[static_cast<size_t>(n * cn + k)] = to_bf16(block * ws2);
    }
  return out;
}

// ---- device reads and comparisons --------------------------------------------

template <class T>
std::vector<T> down(const T* dev, size_t n) {
  std::vector<T> h(n);
  if (dev == nullptr) throw std::runtime_error("a resident pointer is null");
  const cudaError_t err = cudaMemcpy(h.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("cudaMemcpy: ") + cudaGetErrorString(err));
  return h;
}

// Elements that differ (0: bitwise equal).
template <class T>
size_t differing(const std::vector<T>& got, const std::vector<T>& want) {
  if (got.size() != want.size()) return std::max(got.size(), want.size());
  if (std::memcmp(got.data(), want.data(), got.size() * sizeof(T)) == 0) return 0;
  size_t n = 0;
  for (size_t i = 0; i < got.size(); ++i) n += std::memcmp(&got[i], &want[i], sizeof(T)) != 0;
  return n;
}

// A resident BF16 matrix against the expected bits: every row compared.
void check_rows(const std::string& what, const uint16_t* dev, const std::vector<uint16_t>& want,
                int64_t width) {
  const std::vector<uint16_t> got = down(dev, want.size());
  const int64_t rows = static_cast<int64_t>(want.size()) / width;
  int64_t bad = 0, first = -1;
  for (int64_t r = 0; r < rows; ++r)
    if (std::memcmp(&got[static_cast<size_t>(r * width)], &want[static_cast<size_t>(r * width)],
                    static_cast<size_t>(width) * 2) != 0) {
      if (first < 0) first = r;
      ++bad;
    }
  report(bad == 0, what + ": " + std::to_string(rows) + " rows x " + std::to_string(width),
         bad == 0 ? "" : std::to_string(bad) + " rows differ, first " + std::to_string(first));
}

// A resident BF16 matrix against the expected bits: every element compared.
void check_elems(const std::string& what, const uint16_t* dev, const std::vector<uint16_t>& want) {
  const size_t bad = differing(down(dev, want.size()), want);
  report(bad == 0, what + ": " + std::to_string(want.size()) + " elements",
         bad == 0 ? "" : std::to_string(bad) + " elements differ");
}

// Resident bytes against a source tensor's bytes, whole.
void check_bytes(const std::string& what, const void* dev, const Source& src,
                 const std::string& name) {
  const size_t n = src.nbytes(name);
  const std::vector<uint8_t> got = down(static_cast<const uint8_t*>(dev), n);
  report(std::memcmp(got.data(), src.bytes(name), n) == 0,
         what + ": " + std::to_string(n) + " bytes of " + name);
}

// ---- (a) the GDN gather --------------------------------------------------------

// The checkpoint's interleave at (rank, world), written from its layout:
// key head g owns rows [g * G, +G) of in_proj_qkvz as [q dk | k dk | v r*dv
// | z r*dv] and rows [g * 2r, +2r) of in_proj_ba as [b x r | a x r].
struct GdnMap {
  int64_t dk, dv, ratio, G, lk, lv, k0, v0;
  GdnMap(const Qwen35TextConfig& c, int rank, int world)
      : dk(c.gdn_key_head_dim),
        dv(c.gdn_value_head_dim),
        ratio(c.gdn_value_heads / c.gdn_key_heads),
        G(2 * dk + 2 * ratio * dv),
        lk(c.gdn_key_heads / world),
        lv(c.gdn_value_heads / world),
        k0(rank * lk),
        v0(rank * lv) {}
  // Source rows of the local in_proj_qkv [lk*dk | lk*dk | lv*dv], in_proj_z, a, b.
  std::vector<int64_t> qkv() const {
    std::vector<int64_t> s;
    for (int64_t h = 0; h < lk; ++h)
      for (int64_t o = 0; o < dk; ++o) s.push_back((k0 + h) * G + o);
    for (int64_t h = 0; h < lk; ++h)
      for (int64_t o = 0; o < dk; ++o) s.push_back((k0 + h) * G + dk + o);
    for (int64_t h = 0; h < lv; ++h)
      for (int64_t o = 0; o < dv; ++o)
        s.push_back(((v0 + h) / ratio) * G + 2 * dk + ((v0 + h) % ratio) * dv + o);
    return s;
  }
  std::vector<int64_t> z() const {
    std::vector<int64_t> s;
    for (int64_t h = 0; h < lv; ++h)
      for (int64_t o = 0; o < dv; ++o)
        s.push_back(((v0 + h) / ratio) * G + 2 * dk + ratio * dv + ((v0 + h) % ratio) * dv + o);
    return s;
  }
  std::vector<int64_t> b() const {
    std::vector<int64_t> s;
    for (int64_t h = 0; h < lv; ++h) s.push_back(((v0 + h) / ratio) * 2 * ratio + (v0 + h) % ratio);
    return s;
  }
  std::vector<int64_t> a() const {
    std::vector<int64_t> s;
    for (int64_t h = 0; h < lv; ++h)
      s.push_back(((v0 + h) / ratio) * 2 * ratio + ratio + (v0 + h) % ratio);
    return s;
  }
};

std::vector<uint16_t> gather(const std::vector<uint16_t>& src, int64_t width,
                             const std::vector<int64_t>& rows) {
  std::vector<uint16_t> out(rows.size() * static_cast<size_t>(width));
  for (size_t i = 0; i < rows.size(); ++i)
    std::memcpy(&out[i * static_cast<size_t>(width)], &src[static_cast<size_t>(rows[i] * width)],
                static_cast<size_t>(width) * 2);
  return out;
}

// Rows [r0, +rn) of a [*, width] matrix; columns [c0, +cn) of a [rows, full] matrix.
std::vector<uint16_t> row_slice(const std::vector<uint16_t>& m, int64_t width, int64_t r0,
                                int64_t rn) {
  return std::vector<uint16_t>(m.begin() + r0 * width, m.begin() + (r0 + rn) * width);
}
std::vector<uint16_t> col_slice(const std::vector<uint16_t>& m, int64_t full, int64_t c0,
                                int64_t cn) {
  const int64_t rows = static_cast<int64_t>(m.size()) / full;
  std::vector<uint16_t> out(static_cast<size_t>(rows * cn));
  for (int64_t r = 0; r < rows; ++r)
    std::memcpy(&out[static_cast<size_t>(r * cn)], &m[static_cast<size_t>(r * full + c0)],
                static_cast<size_t>(cn) * 2);
  return out;
}

// ---- (c) the NVFP4 routed experts ----------------------------------------------

// Flash-Next's resident form: the payload and the block scales are the
// shipped bytes of this rank's intermediate slice — rows [rank * I, +I) of
// gate/up, columns [rank * I, +I) of down, packed — and the global is the
// fp32 reciprocal of weight_scale_2.
void check_fp4_expert(const std::string& tag, const Source& src, const Qwen35TextConfig& c,
                      const std::string& layer_prefix, const Qwen35LayerResident& r, int e,
                      int rank, int world) {
  static const char* part[3] = {"gate_proj", "up_proj", "down_proj"};
  const int64_t H = c.hidden_size, I_full = c.moe_intermediate_size, I = I_full / world;
  std::string bad;
  for (int i = 0; i < 3; ++i) {
    const size_t idx = static_cast<size_t>(e) * 3 + static_cast<size_t>(i);
    const GlmFp4Matrix& q = r.moe.experts_fp4.at(idx);
    const std::string base = layer_prefix + "mlp.experts." + std::to_string(e) + "." + part[i];
    const dgpp::TensorInfo& tp = src.at(base + ".weight");
    const dgpp::TensorInfo& ts = src.at(base + ".weight_scale");
    const uint8_t* sp = static_cast<const uint8_t*>(tp.data);
    const uint8_t* ss = static_cast<const uint8_t*>(ts.data);
    const int64_t rows = i < 2 ? I : H, cols = i < 2 ? H : I;
    const int64_t K = tp.shape[1] * 2;  // the source's columns
    const int64_t r0 = i < 2 ? rank * I : 0, c0 = i < 2 ? 0 : rank * I;
    if (q.rows != rows || q.cols != cols || q.scale_group != 16) {
      bad += std::string(part[i]) + " shape; ";
      continue;
    }
    std::vector<uint8_t> want_p(static_cast<size_t>(rows * cols / 2)),
        want_s(static_cast<size_t>(rows * cols / 16));
    for (int64_t n = 0; n < rows; ++n) {
      std::memcpy(&want_p[static_cast<size_t>(n * (cols / 2))], sp + (r0 + n) * (K / 2) + c0 / 2,
                  static_cast<size_t>(cols / 2));
      std::memcpy(&want_s[static_cast<size_t>(n * (cols / 16))], ss + (r0 + n) * (K / 16) + c0 / 16,
                  static_cast<size_t>(cols / 16));
    }
    if (differing(down(q.payload, want_p.size()), want_p) != 0)
      bad += std::string(part[i]) + " payload; ";
    if (differing(down(q.scales, want_s.size()), want_s) != 0)
      bad += std::string(part[i]) + " block scales; ";
    if (q.global_scale != r.moe.expert_globals + idx)
      bad += std::string(part[i]) + " global slot; ";
    const float want = 1.0f / src.f32(base + ".weight_scale_2");
    const float got = down(q.global_scale, 1)[0];
    if (std::memcmp(&got, &want, 4) != 0) bad += std::string(part[i]) + " global value; ";
  }
  report(bad.empty(),
         tag + " (c) expert " + std::to_string(e) +
             ": gate/up/down payload and block scales are the shipped bytes, global = 1 / "
             "weight_scale_2",
         bad);
}

// The layer's activation scales: the maximum input_scale over every
// expert's gate and up, and over every down.
void check_act_scales(const std::string& tag, const Source& src, const std::string& layer_prefix,
                      const Qwen35TextConfig& c, const Qwen35LayerResident& r) {
  float w13 = 0.0f, w2 = 0.0f;
  for (int e = 0; e < c.num_experts; ++e) {
    const std::string ep = layer_prefix + "mlp.experts." + std::to_string(e) + ".";
    w13 =
        std::max({w13, src.f32(ep + "gate_proj.input_scale"), src.f32(ep + "up_proj.input_scale")});
    w2 = std::max(w2, src.f32(ep + "down_proj.input_scale"));
  }
  const std::vector<float> got = down(r.moe.act_scales, 2);
  char detail[160];
  std::snprintf(detail, sizeof(detail), "device [%g, %g] host [%g, %g] want [%g, %g]", got[0],
                got[1], r.moe.act_scale_w13, r.moe.act_scale_w2, w13, w2);
  const bool ok =
      got[0] == w13 && got[1] == w2 && r.moe.act_scale_w13 == w13 && r.moe.act_scale_w2 == w2;
  report(ok,
         tag + " (c) act_scales = max input_scale over the " + std::to_string(c.num_experts) +
             " experts (gate/up, down): [" + std::to_string(w13) + ", " + std::to_string(w2) + "]",
         ok ? "" : detail);
}

// ---- (d) the draft layer's block-FP8 experts -----------------------------------

// The recipe (loaders/fp8_quant.hpp): per 128 x 128 block, scale = amax / 448
// (1 for an all-zero block) and code = e4m3(w / scale), round to nearest
// even, saturating. Checked three ways against the BF16 source:
//   the scale grid, bitwise;
//   the bound |e4m3(code) * scale - w| <= max(|w| / 16, scale * 2^-10) —
//     e4m3 keeps 4 significant bits, so the nearest code is within half a
//     step, at most 1/16 of the value (the bottom of a binade), and within
//     half the subnormal step 2^-9 below 2^-6; the 1e-5 slack covers the
//     two fp32 roundings (the division, the product);
//   the code is a nearest grid point to w / scale (no neighbour is closer).
struct Fp8Stats {
  double max_rel = 0.0;  // largest |decoded - w| / |w| over the normal range (|w| >= scale * 2^-6)
  size_t elems = 0;
  size_t subnormal = 0;  // elements below it (the absolute bound applies)
};
// Experts at (rank, world): rows [rank * I, +I) of gate/up, columns of down,
// the 128 x 128 scale grid anchored at the slice's origin.
struct Fp8Faults {
  size_t shape = 0, scales = 0, bound = 0, nearest = 0, elems = 0;
  std::string text() const {
    std::string bad;
    if (shape) bad += std::to_string(shape) + " matrices of the wrong shape or scale grid; ";
    if (scales) bad += std::to_string(scales) + " block scales differ from amax / 448; ";
    if (bound) bad += std::to_string(bound) + " elements outside the bound; ";
    if (nearest) bad += std::to_string(nearest) + " codes are not a nearest grid point; ";
    return bad;
  }
};

// One resident block-FP8 matrix against the BF16 values `w` [rows, cols] it
// encodes: the three checks above, added to the tallies.
void fp8_matrix_faults(const GlmQuantMatrix& q, const std::vector<uint16_t>& w, int64_t rows,
                       int64_t cols, Fp8Stats& stats, Fp8Faults& f) {
  const float* e4m3 = e4m3_table();
  const size_t before = f.elems;
  if (q.rows != rows || q.cols != cols || q.scale_block_rows != 128 || q.scale_block_cols != 128 ||
      static_cast<int64_t>(w.size()) != rows * cols) {
    ++f.shape;
    return;
  }
  const int64_t sr = (rows + 127) / 128, sc = (cols + 127) / 128;
  const std::vector<uint8_t> codes = down(q.payload, static_cast<size_t>(rows * cols));
  const std::vector<float> scales = down(q.scales, static_cast<size_t>(sr * sc));
  for (int64_t br = 0; br < sr; ++br)
    for (int64_t bc = 0; bc < sc; ++bc) {
      const int64_t r1 = std::min(rows, (br + 1) * 128), c1 = std::min(cols, (bc + 1) * 128);
      float amax = 0.0f;
      for (int64_t rr = br * 128; rr < r1; ++rr)
        for (int64_t cc = bc * 128; cc < c1; ++cc)
          amax = std::max(amax, std::fabs(bf16_value(w[static_cast<size_t>(rr * cols + cc)])));
      const float want = amax > 0.0f ? amax / 448.0f : 1.0f;
      const float s = scales[static_cast<size_t>(br * sc + bc)];
      if (std::memcmp(&s, &want, 4) != 0) {
        ++f.scales;
        continue;
      }
      for (int64_t rr = br * 128; rr < r1; ++rr)
        for (int64_t cc = bc * 128; cc < c1; ++cc) {
          const float v = bf16_value(w[static_cast<size_t>(rr * cols + cc)]);
          const uint8_t code = codes[static_cast<size_t>(rr * cols + cc)];
          const float dec = e4m3[code] * s;
          const float err = std::fabs(dec - v);
          const float bound = std::max(std::fabs(v) / 16.0f, s * 0x1p-10f) * (1.0f + 1e-5f);
          if (!(err <= bound)) ++f.bound;
          if (std::fabs(v) >= s * 0x1p-6f)
            stats.max_rel = std::max(stats.max_rel, static_cast<double>(err) / std::fabs(v));
          else
            ++stats.subnormal;
          // Nearest: the magnitude codes 0x00 .. 0x7E ascend with their values.
          const uint8_t m = code & 0x7Fu;
          const float x = std::fabs(v) / s;
          const float d = std::fabs(e4m3[m] - x);
          const bool sign_ok = e4m3[m] == 0.0f || ((code >> 7) != 0) == (v < 0.0f);
          const bool lower_ok = m == 0 || d <= std::fabs(e4m3[m - 1] - x);
          const bool upper_ok = m == 0x7E || d <= std::fabs(e4m3[m + 1] - x);
          if (m == 0x7F || !sign_ok || !lower_ok || !upper_ok) ++f.nearest;
          ++f.elems;
        }
    }
  stats.elems += f.elems - before;
}

// Experts at (rank, world): rows [rank * I, +I) of gate/up, columns of down,
// the 128 x 128 scale grid anchored at the slice's origin.
void check_fp8_expert(const std::string& tag, const Source& src, const Qwen35TextConfig& c,
                      const Qwen35LayerResident& r, int e, int rank, int world, Fp8Stats& stats) {
  static const char* part[3] = {"gate_proj", "up_proj", "down_proj"};
  const int64_t H = c.hidden_size, I_full = c.moe_intermediate_size, I = I_full / world;
  Fp8Faults f;
  for (int i = 0; i < 3; ++i) {
    const GlmQuantMatrix& q = r.moe.experts.at(static_cast<size_t>(e) * 3 + static_cast<size_t>(i));
    const std::string name =
        "mtp.layers.0.mlp.experts." + std::to_string(e) + "." + part[i] + ".weight";
    const std::vector<uint16_t> full = src.bf16(name);
    const std::vector<uint16_t> w =
        i < 2 ? row_slice(full, H, rank * I, I) : col_slice(full, I_full, rank * I, I);
    fp8_matrix_faults(q, w, i < 2 ? I : H, i < 2 ? H : I, stats, f);
  }
  report(f.text().empty(),
         tag + " (d) draft expert " + std::to_string(e) + ": " + std::to_string(f.elems) +
             " elements — block scales = amax / 448 bitwise, |decoded - w| <= max(|w| / 16, scale "
             "* 2^-10), "
             "every code a nearest e4m3 point",
         f.text());
}

void print_fp8_stats(const Fp8Stats& stats, const char* what = "draft experts") {
  std::printf(
      "   %s: %zu elements; largest |decoded - w| / |w| in the normal range (|w| >= "
      "scale * 2^-6) = "
      "%.5f (bound 0.06250); %zu elements below it, held to the absolute bound scale * 2^-10\n",
      what, stats.elems, stats.max_rel, stats.subnormal);
}

// ---- the layers ------------------------------------------------------------------

// What world 1 keeps for the world-2 tiling.
struct Kept {
  std::vector<uint16_t> qkv, z, a, b, out_proj;  // layer 0
  std::vector<uint16_t> o_proj;                  // layer 3
  std::vector<uint16_t> shared[2][3];            // [layer 0, layer 3][gate, up, down]
};

void check_norms(const std::string& tag, const Source& src, const std::string& p,
                 const Qwen35LayerResident& r) {
  check_bytes(tag + " (e) input_layernorm", r.input_norm, src, p + "input_layernorm.weight");
  check_bytes(tag + " (e) post_attention_layernorm", r.post_norm, src,
              p + "post_attention_layernorm.weight");
}

void check_router(const std::string& tag, const Source& src, const std::string& p,
                  const Qwen35LayerResident& r) {
  check_bytes(tag + " (e) router", r.moe.router, src, p + "mlp.gate.weight");
  check_bytes(tag + " (e) shared gate", r.moe.shared_gate, src,
              p + "mlp.shared_expert_gate.weight");
}

// (b) for the shared expert at (rank, world): gate/up rows, down columns.
void check_shared(const std::string& tag, const Source& src, const std::string& p,
                  const Qwen35TextConfig& c, const Qwen35LayerResident& r, int rank, int world,
                  std::vector<uint16_t>* keep) {
  static const char* part[3] = {"gate_proj", "up_proj", "down_proj"};
  const int64_t H = c.hidden_size, S = c.shared_expert_intermediate_size / world;
  report(r.moe.local_shared_inter == S && r.moe.local_inter == c.moe_intermediate_size / world &&
             r.moe.scale_block == std::min<int64_t>(128, c.moe_intermediate_size / world),
         tag + " moe geometry: local_inter " + std::to_string(r.moe.local_inter) +
             ", local_shared_inter " + std::to_string(r.moe.local_shared_inter) + ", scale_block " +
             std::to_string(r.moe.scale_block));
  for (int i = 0; i < 3; ++i) {
    const std::string base = p + "mlp.shared_expert." + part[i];
    const std::vector<uint16_t> want = i < 2 ? host_dequant(src, base, rank * S, S, 0, H)
                                             : host_dequant(src, base, 0, H, rank * S, S);
    check_elems(tag + " (b) shared " + part[i] + " = host dequant", r.moe.shared[i], want);
    if (keep) keep[i] = want;
  }
}

void check_forms(const std::string& tag, const Qwen35TextConfig& c, const Qwen35LayerResident& r,
                 bool backbone) {
  const size_t matrices = static_cast<size_t>(c.num_experts) * 3;
  const bool fp8_empty = !r.gdn.in_proj_qkv_fp8.payload && !r.gdn.in_proj_z_fp8.payload &&
                         !r.gdn.out_proj_fp8.payload && !r.full.q_proj_fp8.payload &&
                         !r.full.k_proj_fp8.payload && !r.full.v_proj_fp8.payload &&
                         !r.full.o_proj_fp8.payload && !r.moe.shared_fp8[0].payload &&
                         !r.moe.shared_fp8[1].payload && !r.moe.shared_fp8[2].payload;
  const bool mlp_empty = !r.mlp.gate && !r.mlp.up && !r.mlp.down && !r.mlp.gate_fp8.payload &&
                         !r.mlp.up_fp8.payload && !r.mlp.down_fp8.payload;
  const bool experts =
      backbone ? (r.moe.nvfp4() && r.moe.experts.empty() && r.moe.experts_fp4.size() == matrices &&
                  r.moe.expert_globals && r.moe.act_scales)
               : (!r.moe.nvfp4() && r.moe.experts.size() == matrices && !r.moe.expert_globals &&
                  !r.moe.act_scales);
  report(fp8_empty && mlp_empty && experts && !r.moe.packq(),
         tag + " forms: every _fp8 member and the dense mlp empty, routed experts " +
             (backbone ? "NVFP4 (experts_fp4 + expert_globals)" : "block FP8 (experts)"));
}

void check_gdn_layer(const std::string& tag, const Source& src, const Qwen35TextConfig& c,
                     const Qwen35LayerResident& r, int layer, int rank, int world, Kept* keep) {
  const std::string p = dgpp::qwen35_layer_prefix(c, layer);
  const int64_t H = c.hidden_size;
  const GdnMap map(c, rank, world);
  report(r.kind == dgpp::Qwen35LayerKind::Gdn && r.layer == layer &&
             r.gdn.local_key_heads == map.lk && r.gdn.local_value_heads == map.lv,
         tag + " kind GDN, " + std::to_string(r.gdn.local_key_heads) + " key heads, " +
             std::to_string(r.gdn.local_value_heads) + " value heads");
  const std::vector<uint16_t> qkvz = src.bf16(p + "linear_attn.in_proj_qkvz.weight");
  const std::vector<uint16_t> ba = src.bf16(p + "linear_attn.in_proj_ba.weight");
  const std::vector<uint16_t> want_qkv = gather(qkvz, H, map.qkv()),
                              want_z = gather(qkvz, H, map.z());
  const std::vector<uint16_t> want_a = gather(ba, H, map.a()), want_b = gather(ba, H, map.b());
  check_rows(tag + " (a) in_proj_qkv = interleaved in_proj_qkvz rows", r.gdn.in_proj_qkv, want_qkv,
             H);
  check_rows(tag + " (a) in_proj_z = interleaved in_proj_qkvz rows", r.gdn.in_proj_z, want_z, H);
  check_rows(tag + " (a) in_proj_a = interleaved in_proj_ba rows", r.gdn.in_proj_a, want_a, H);
  check_rows(tag + " (a) in_proj_b = interleaved in_proj_ba rows", r.gdn.in_proj_b, want_b, H);
  const int64_t c0 = map.v0 * map.dv, cn = map.lv * map.dv;
  const std::vector<uint16_t> want_out =
      host_dequant(src, p + "linear_attn.out_proj", 0, H, c0, cn);
  check_elems(tag + " (b) out_proj = host dequant", r.gdn.out_proj, want_out);
  if (keep) {
    keep->qkv = want_qkv;
    keep->z = want_z;
    keep->a = want_a;
    keep->b = want_b;
    keep->out_proj = want_out;
  }
  check_shared(tag, src, p, c, r, rank, world, keep ? keep->shared[0] : nullptr);
}

void check_attention_layer(const std::string& tag, const Source& src, const Qwen35TextConfig& c,
                           const Qwen35LayerResident& r, int layer, int rank, int world,
                           Kept* keep) {
  const std::string p = dgpp::qwen35_layer_prefix(c, layer);
  const int64_t H = c.hidden_size, d = c.head_dim;
  const int64_t lh = c.num_attention_heads / world, h0 = rank * lh;
  report(r.kind == dgpp::Qwen35LayerKind::Full && r.layer == layer && r.full.local_heads == lh &&
             r.full.head_begin == h0,
         tag + " kind attention, query heads [" + std::to_string(r.full.head_begin) + ", +" +
             std::to_string(r.full.local_heads) + "), kv heads [" +
             std::to_string(r.full.kv_head_begin) + ", +" + std::to_string(r.full.local_kv_heads) +
             ")");
  const std::vector<uint16_t> want_o =
      host_dequant(src, p + "self_attn.o_proj", 0, H, h0 * d, lh * d);
  check_elems(tag + " (b) o_proj = host dequant", r.full.o_proj, want_o);
  if (keep) keep->o_proj = want_o;
  check_shared(tag, src, p, c, r, rank, world, keep ? keep->shared[1] : nullptr);
}

// Rank slices reassembled: rows (or columns) [begin, +count) of each rank
// written once into a matrix that must then equal world 1's.
struct Tiling {
  std::vector<uint16_t> m;
  std::vector<int> hits;  // per row (or column)
  int64_t width = 0;      // elements per row
  bool by_cols = false;
  Tiling(size_t elems, int64_t width_, bool by_cols_)
      : m(elems, 0xDEAD),
        hits(static_cast<size_t>(by_cols_ ? width_ : elems / width_), 0),
        width(width_),
        by_cols(by_cols_) {}
  void put(const std::vector<uint16_t>& part, int64_t begin, int64_t count) {
    if (!by_cols) {
      std::memcpy(&m[static_cast<size_t>(begin * width)], part.data(), part.size() * 2);
    } else {
      const int64_t rows = static_cast<int64_t>(m.size()) / width;
      for (int64_t r = 0; r < rows; ++r)
        std::memcpy(&m[static_cast<size_t>(r * width + begin)],
                    &part[static_cast<size_t>(r * count)], static_cast<size_t>(count) * 2);
    }
    for (int64_t i = 0; i < count; ++i) ++hits[static_cast<size_t>(begin + i)];
  }
  void check(const std::string& what, const std::vector<uint16_t>& world1) const {
    const bool once = std::all_of(hits.begin(), hits.end(), [](int n) { return n == 1; });
    report(once && differing(m, world1) == 0,
           what + ": the two ranks' slices tile the world-1 matrix (" +
               std::to_string(world1.size()) + " elements)",
           once ? "" : "a row or column was written zero or several times");
  }
};

// ---- the dense stack as block FP8 (--dense fp8) --------------------------------

// A layer's dense projections at (rank, world) as the BF16 form holds them,
// computed on the host: the gathered GDN projections, the attention row
// slices, the NVFP4 dequant (the draft layer's BF16 column slices).
struct DenseWant {
  std::string name;
  std::vector<uint16_t> m;
  int64_t rows = 0, cols = 0;
};

std::vector<DenseWant> shared_want(const Source& src, const Qwen35TextConfig& c, int layer,
                                   int rank, int world) {
  static const char* part[3] = {"gate_proj", "up_proj", "down_proj"};
  const std::string p = dgpp::qwen35_layer_prefix(c, layer) + "mlp.shared_expert.";
  const int64_t H = c.hidden_size, S_full = c.shared_expert_intermediate_size, S = S_full / world;
  const bool is_mtp = layer == c.mtp_layer();
  std::vector<DenseWant> w;
  for (int i = 0; i < 3; ++i) {
    const std::string base = p + part[i];
    DenseWant d;
    d.name = std::string("shared ") + part[i];
    d.rows = i < 2 ? S : H;
    d.cols = i < 2 ? H : S;
    if (is_mtp)
      d.m = i < 2 ? row_slice(src.bf16(base + ".weight"), H, rank * S, S)
                  : col_slice(src.bf16(base + ".weight"), S_full, rank * S, S);
    else
      d.m = i < 2 ? host_dequant(src, base, rank * S, S, 0, H)
                  : host_dequant(src, base, 0, H, rank * S, S);
    w.push_back(std::move(d));
  }
  return w;
}

// [in_proj_qkv, in_proj_z, out_proj, shared gate/up/down].
std::vector<DenseWant> gdn_want(const Source& src, const Qwen35TextConfig& c, int layer, int rank,
                                int world) {
  const std::string p = dgpp::qwen35_layer_prefix(c, layer);
  const int64_t H = c.hidden_size;
  const GdnMap map(c, rank, world);
  const std::vector<uint16_t> qkvz = src.bf16(p + "linear_attn.in_proj_qkvz.weight");
  std::vector<DenseWant> w;
  w.push_back(
      {"in_proj_qkv", gather(qkvz, H, map.qkv()), 2 * map.lk * map.dk + map.lv * map.dv, H});
  w.push_back({"in_proj_z", gather(qkvz, H, map.z()), map.lv * map.dv, H});
  w.push_back(
      {"out_proj",
       host_dequant(src, p + "linear_attn.out_proj", 0, H, map.v0 * map.dv, map.lv * map.dv), H,
       map.lv * map.dv});
  for (auto& d : shared_want(src, c, layer, rank, world)) w.push_back(std::move(d));
  return w;
}

// [q_proj, k_proj, v_proj, o_proj, shared gate/up/down]; world <= kv heads.
std::vector<DenseWant> attention_want(const Source& src, const Qwen35TextConfig& c, int layer,
                                      int rank, int world) {
  const std::string p = dgpp::qwen35_layer_prefix(c, layer) + "self_attn.";
  const int64_t H = c.hidden_size, d = c.head_dim;
  const int64_t lh = c.num_attention_heads / world, lkv = c.num_key_value_heads / world;
  std::vector<DenseWant> w;
  w.push_back({"q_proj", row_slice(src.bf16(p + "q_proj.weight"), H, rank * lh * 2 * d, lh * 2 * d),
               lh * 2 * d, H});
  w.push_back(
      {"k_proj", row_slice(src.bf16(p + "k_proj.weight"), H, rank * lkv * d, lkv * d), lkv * d, H});
  w.push_back(
      {"v_proj", row_slice(src.bf16(p + "v_proj.weight"), H, rank * lkv * d, lkv * d), lkv * d, H});
  if (layer == c.mtp_layer())
    w.push_back(
        {"o_proj",
         col_slice(src.bf16(p + "o_proj.weight"), c.num_attention_heads * d, rank * lh * d, lh * d),
         H, lh * d});
  else
    w.push_back(
        {"o_proj", host_dequant(src, p + "o_proj", 0, H, rank * lh * d, lh * d), H, lh * d});
  for (auto& s : shared_want(src, c, layer, rank, world)) w.push_back(std::move(s));
  return w;
}

std::vector<DenseWant> dense_want(const Source& src, const Qwen35TextConfig& c, int layer, int rank,
                                  int world) {
  const bool gdn = layer != c.mtp_layer() && c.layers[layer] == dgpp::Qwen35LayerKind::Gdn;
  return gdn ? gdn_want(src, c, layer, rank, world) : attention_want(src, c, layer, rank, world);
}

// The resident's BF16 pointers and FP8 matrices, in dense_want's order.
std::vector<const uint16_t*> dense_bf16(const Qwen35TextConfig& c, const Qwen35LayerResident& r) {
  if (r.layer != c.mtp_layer() && r.kind == dgpp::Qwen35LayerKind::Gdn)
    return {r.gdn.in_proj_qkv, r.gdn.in_proj_z, r.gdn.out_proj,
            r.moe.shared[0],   r.moe.shared[1], r.moe.shared[2]};
  return {r.full.q_proj,   r.full.k_proj,   r.full.v_proj,  r.full.o_proj,
          r.moe.shared[0], r.moe.shared[1], r.moe.shared[2]};
}
std::vector<const GlmQuantMatrix*> dense_fp8(const Qwen35TextConfig& c,
                                             const Qwen35LayerResident& r) {
  if (r.layer != c.mtp_layer() && r.kind == dgpp::Qwen35LayerKind::Gdn)
    return {&r.gdn.in_proj_qkv_fp8, &r.gdn.in_proj_z_fp8, &r.gdn.out_proj_fp8,
            &r.moe.shared_fp8[0],   &r.moe.shared_fp8[1], &r.moe.shared_fp8[2]};
  return {&r.full.q_proj_fp8,   &r.full.k_proj_fp8,   &r.full.v_proj_fp8,  &r.full.o_proj_fp8,
          &r.moe.shared_fp8[0], &r.moe.shared_fp8[1], &r.moe.shared_fp8[2]};
}

// The BF16 form: the loader's dense matrices are the host values, bitwise.
void check_bf16_form(const std::string& tag, const Qwen35TextConfig& c,
                     const Qwen35LayerResident& r, const std::vector<DenseWant>& want) {
  const std::vector<const uint16_t*> got = dense_bf16(c, r);
  const std::vector<const GlmQuantMatrix*> fp8 = dense_fp8(c, r);
  for (size_t i = 0; i < want.size(); ++i) {
    report(fp8[i]->payload == nullptr, tag + " " + want[i].name + ": BF16 form, no fp8 view");
    check_elems(tag + " " + want[i].name + ": the BF16-form loader's matrix is the host value",
                got[i], want[i].m);
  }
}

// The FP8 form: every dense matrix bound through its `_fp8` member (the
// BF16 pointer null) and the block-FP8 encode of the BF16 form's values.
void check_fp8_form(const std::string& tag, const Qwen35TextConfig& c, const Qwen35LayerResident& r,
                    const std::vector<DenseWant>& want, Fp8Stats& stats) {
  const std::vector<const uint16_t*> bf16 = dense_bf16(c, r);
  const std::vector<const GlmQuantMatrix*> got = dense_fp8(c, r);
  bool bf16_null = true;
  for (const uint16_t* ptr : bf16) bf16_null = bf16_null && ptr == nullptr;
  report(bf16_null && !r.mlp.gate && !r.mlp.gate_fp8.payload,
         tag +
             " forms: every dense BF16 pointer null (bound through the _fp8 members), dense mlp "
             "empty");
  for (size_t i = 0; i < want.size(); ++i) {
    Fp8Faults f;
    fp8_matrix_faults(*got[i], want[i].m, want[i].rows, want[i].cols, stats, f);
    report(f.text().empty() && f.elems == want[i].m.size(),
           tag + " " + want[i].name + "_fp8 [" + std::to_string(want[i].rows) + " x " +
               std::to_string(want[i].cols) + "]: " + std::to_string(f.elems) +
               " elements — block scales = amax / 448 bitwise, |decoded - w| <= max(|w| / 16, "
               "scale * 2^-10), "
               "every code a nearest e4m3 point of the BF16-form value",
           f.text());
  }
}

// What the FP8 form leaves alone, against the source (world 1).
void check_fp8_form_rest(const std::string& tag, const Source& src, const Qwen35TextConfig& c,
                         const Qwen35LayerResident& r) {
  const std::string p = dgpp::qwen35_layer_prefix(c, r.layer);
  const int64_t H = c.hidden_size;
  const bool is_mtp = r.layer == c.mtp_layer();
  check_norms(tag, src, p, r);
  check_router(tag, src, p, r);
  if (!is_mtp && r.kind == dgpp::Qwen35LayerKind::Gdn) {
    const GdnMap map(c, 0, 1);
    const std::vector<uint16_t> ba = src.bf16(p + "linear_attn.in_proj_ba.weight");
    check_rows(tag + " in_proj_a stays BF16 (gathered rows)", r.gdn.in_proj_a,
               gather(ba, H, map.a()), H);
    check_rows(tag + " in_proj_b stays BF16 (gathered rows)", r.gdn.in_proj_b,
               gather(ba, H, map.b()), H);
    check_bytes(tag + " conv", r.gdn.conv, src, p + "linear_attn.conv1d.weight");
    check_bytes(tag + " gdn norm", r.gdn.norm, src, p + "linear_attn.norm.weight");
    report(r.gdn.a_log != nullptr && r.gdn.dt_bias != nullptr,
           tag + " A_log and dt_bias present (F32)");
  } else {
    check_bytes(tag + " q_norm", r.full.q_norm, src, p + "self_attn.q_norm.weight");
    check_bytes(tag + " k_norm", r.full.k_norm, src, p + "self_attn.k_norm.weight");
    if (!is_mtp) {
      const std::vector<float> kv = down(r.kv_cache_scales, 2);
      report(kv[0] == src.f32(p + "self_attn.k_proj.k_scale") &&
                 kv[1] == src.f32(p + "self_attn.v_proj.v_scale") && r.k_cache_scale == kv[0] &&
                 r.v_cache_scale == kv[1],
             tag + " K/V-cache scales kept");
    }
  }
  if (is_mtp) {
    report(r.moe.experts.size() == static_cast<size_t>(c.num_experts) * 3 && !r.moe.nvfp4(),
           tag + " routed experts unchanged: block FP8");
  } else {
    report(r.moe.nvfp4() && r.moe.experts.empty(), tag + " routed experts unchanged: NVFP4");
    for (int e : {0, 511}) check_fp4_expert(tag, src, c, p, r, e, 0, 1);
    check_act_scales(tag, src, p, c, r);
  }
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// --plan: the world-1 resident bytes of both forms, counted (nothing loaded).
int run_plan(const Qwen35TextConfig& cfg) {
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  const int draft = cfg.mtp_layer();
  size_t layer[2][3], total[2], staging[2];
  for (int form = 0; form < 2; ++form) {
    Qwen35LayerStream::set_dense_weights_fp8(form == 1);
    const int layers[3] = {0, 3, draft};
    for (int i = 0; i < 3; ++i) layer[form][i] = Qwen35LayerStream::layer_bytes(cfg, layers[i]);
    total[form] =
        Qwen35LayerStream::resident_bytes(cfg, 0, 1, dgpp::LoaderHeadSharding::Full, true);
    staging[form] =
        Qwen35LayerStream::staging_plan_bytes(cfg, 0, 1, dgpp::LoaderHeadSharding::Full, true);
  }
  Qwen35LayerStream::set_dense_weights_fp8(false);
  const size_t globals = Qwen35LayerStream::globals_bytes(cfg);
  std::printf("resident bytes, world 1 (counting builds):   BF16 form         FP8 form\n");
  std::printf("  layer 0 (GDN)                          %12zu     %12zu\n", layer[0][0],
              layer[1][0]);
  std::printf("  layer 3 (attention)                    %12zu     %12zu\n", layer[0][1],
              layer[1][1]);
  std::printf("  layer %d (draft)                       %12zu     %12zu\n", draft, layer[0][2],
              layer[1][2]);
  std::printf("  globals                                %12zu     %12zu\n", globals, globals);
  std::printf("  all layers + draft + globals           %12zu     %12zu   (%.3f GiB -> %.3f GiB)\n",
              total[0], total[1], total[0] / kGiB, total[1] / kGiB);
  std::printf("  without the draft layer                %12zu     %12zu   (%.3f GiB -> %.3f GiB)\n",
              total[0] - layer[0][2], total[1] - layer[1][2], (total[0] - layer[0][2]) / kGiB,
              (total[1] - layer[1][2]) / kGiB);
  std::printf("  pinned staging mirror during the load  %12zu     %12zu\n", staging[0], staging[1]);
  return 0;
}

// --dense fp8: see the header.
int run_dense_fp8(const Qwen35TextConfig& cfg, const std::string& dir, const Source& src) {
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  const int draft = cfg.mtp_layer();
  const std::vector<int> layers = {0, 3, draft};
  std::vector<std::vector<DenseWant>> want;
  for (int layer : layers) want.push_back(dense_want(src, cfg, layer, 0, 1));

  // The BF16 form (the knob off): the loader's dense matrices of layers 0
  // and 3 are the host values. The draft layer's are the shipped BF16 bytes
  // (the default run's (e) lines); it is not loaded twice here.
  size_t bf16_bytes[3] = {};
  {
    Qwen35LayerStream::set_dense_weights_fp8(false);
    Qwen35LayerStream stream(cfg, dir, 0, 1, dgpp::LoaderResidency::Streaming,
                             dgpp::LoaderHeadSharding::Full);
    for (size_t i = 0; i < 2; ++i) {
      const Qwen35LayerResident& r = stream.load_layer(layers[i]);
      bf16_bytes[i] = r.bytes;
      check_bf16_form("bf16 w1 L" + std::to_string(layers[i]), cfg, r, want[i]);
    }
    bf16_bytes[2] = Qwen35LayerStream::layer_bytes(cfg, draft);
  }
  const size_t bf16_total =
      Qwen35LayerStream::resident_bytes(cfg, 0, 1, dgpp::LoaderHeadSharding::Full, true);

  // The FP8 form at world 1.
  Qwen35LayerStream::set_dense_weights_fp8(true);
  size_t fp8_bytes[3] = {}, globals_bytes = 0;
  {
    Qwen35LayerStream stream(cfg, dir, 0, 1, dgpp::LoaderResidency::Streaming,
                             dgpp::LoaderHeadSharding::Full);
    Fp8Stats stats;
    for (size_t i = 0; i < layers.size(); ++i) {
      const std::string tag = "fp8 w1 L" + std::to_string(layers[i]);
      const auto t0 = std::chrono::steady_clock::now();
      const Qwen35LayerResident& r = stream.load_layer(layers[i]);
      std::printf("-- layer %d loaded in the FP8 form in %.2f s, %zu bytes\n", layers[i],
                  seconds_since(t0), r.bytes);
      fp8_bytes[i] = r.bytes;
      check_fp8_form(tag, cfg, r, want[i], stats);
      check_fp8_form_rest(tag, src, cfg, r);
    }
    print_fp8_stats(stats, "dense fp8");
    globals_bytes = Qwen35LayerStream::globals_bytes(cfg);
  }

  // Resident mode and the image round trip in the FP8 form (world 2, rank 1,
  // a GDN and an attention layer): built layers against the host values,
  // restored bytes against the built ones.
  {
    namespace fs = std::filesystem;
    const fs::path cache =
        fs::temp_directory_path() / ("qwen3next_loader_smoke_image." + std::to_string(getpid()));
    fs::remove_all(cache);
    const std::string saved = Qwen35LayerStream::resident_image_dir();
    Qwen35LayerStream::set_resident_image_dir(cache.string());
    const int rank = 1, world = 2;
    std::vector<uint8_t> built[2];
    for (int pass = 0; pass < 2; ++pass) {
      Qwen35LayerStream stream(cfg, dir, rank, world, dgpp::LoaderResidency::Resident,
                               dgpp::LoaderHeadSharding::VocabSharded);
      for (int i = 0; i < 2; ++i) {
        const std::string tag = std::string("fp8 w2 r1 resident ") +
                                (pass == 0 ? "built" : "restored") + " L" +
                                std::to_string(layers[i]);
        const Qwen35LayerResident& r = stream.load_layer(layers[i]);
        std::vector<uint8_t> bytes(r.bytes);
        stream.copy_resident_layer(layers[i], bytes.data());
        if (pass == 0) {
          Fp8Stats stats;
          check_fp8_form(tag, cfg, r, dense_want(src, cfg, layers[i], rank, world), stats);
          built[i] = std::move(bytes);
        } else {
          report(bytes == built[i] && r.gdn.in_proj_qkv == nullptr && r.full.q_proj == nullptr &&
                     r.moe.shared_fp8[0].payload != nullptr,
                 tag + ": the restored FP8-form layer is bitwise the built one (" +
                     std::to_string(bytes.size()) + " bytes)");
        }
      }
      if (pass == 0) {
        report(stream.image_layers_captured() == 2 && stream.image_layers_restored() == 0,
               "fp8 w2 r1 resident: the cold load captured both layers to the image");
        // A BF16-form stream must not see this image: its key differs.
        Qwen35LayerStream::set_dense_weights_fp8(false);
        {
          Qwen35LayerStream other(cfg, dir, rank, world, dgpp::LoaderResidency::Resident,
                                  dgpp::LoaderHeadSharding::VocabSharded);
          const Qwen35LayerResident& r = other.load_layer(3);
          report(other.image_layers_restored() == 0 && r.full.q_proj != nullptr &&
                     r.full.q_proj_fp8.payload == nullptr,
                 "fp8 w2 r1 resident: a BF16-form stream does not restore the FP8-form image (it "
                 "builds layer 3 in BF16)");
        }
        Qwen35LayerStream::set_dense_weights_fp8(true);
      } else {
        report(stream.image_layers_restored() == 2 && stream.source_bytes_read() == 0,
               "fp8 w2 r1 resident: the second load restored both layers and read no checkpoint "
               "bytes");
      }
    }
    Qwen35LayerStream::set_resident_image_dir(saved);
    fs::remove_all(cache);
  }

  // The byte accounting, both forms (world 1).
  const int n_gdn = cfg.num_gdn_layers(), n_attn = cfg.num_full_layers();
  const size_t fp8_total = static_cast<size_t>(n_gdn) * fp8_bytes[0] +
                           static_cast<size_t>(n_attn) * fp8_bytes[1] + fp8_bytes[2] +
                           globals_bytes;
  const size_t fp8_planned =
      Qwen35LayerStream::resident_bytes(cfg, 0, 1, dgpp::LoaderHeadSharding::Full, true);
  Qwen35LayerStream::set_dense_weights_fp8(false);
  std::printf("resident bytes, world 1:            BF16 form         FP8 form\n");
  std::printf("  layer 0 (GDN)                 %12zu     %12zu\n", bf16_bytes[0], fp8_bytes[0]);
  std::printf("  layer 3 (attention)           %12zu     %12zu\n", bf16_bytes[1], fp8_bytes[1]);
  std::printf("  layer %d (draft)              %12zu     %12zu\n", draft, bf16_bytes[2],
              fp8_bytes[2]);
  std::printf("  globals                       %12zu     %12zu\n", globals_bytes, globals_bytes);
  std::printf("  %d x GDN + %d x attention + draft + globals:\n", n_gdn, n_attn);
  std::printf("                                %12zu     %12zu   (%.3f GiB -> %.3f GiB)\n",
              bf16_total, fp8_total, bf16_total / kGiB, fp8_total / kGiB);
  std::printf("  without the draft layer       %12zu     %12zu   (%.3f GiB -> %.3f GiB)\n",
              bf16_total - bf16_bytes[2], fp8_total - fp8_bytes[2],
              (bf16_total - bf16_bytes[2]) / kGiB, (fp8_total - fp8_bytes[2]) / kGiB);
  report(fp8_total == fp8_planned && fp8_total < bf16_total,
         "byte accounting: the FP8-form projection equals the stream's resident_bytes plan (" +
             std::to_string(fp8_planned) + ")");

  std::printf("%d checks, %d failed\n", g_passed + g_failed, g_failed);
  std::printf("%s\n", g_failed == 0 ? "SMOKE PASS" : "SMOKE FAIL");
  return g_failed == 0 ? 0 : 1;
}

int run(int argc, char** argv) {
  std::string dir, model, dense = "bf16";
  bool plan = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--model" && i + 1 < argc)
      model = argv[++i];
    else if (a == "--checkpoint-dir" && i + 1 < argc)
      dir = argv[++i];
    else if (a == "--dense" && i + 1 < argc)
      dense = argv[++i];
    else if (a == "--plan")
      plan = true;
    else
      throw std::runtime_error(
          "usage: qwen3next_loader_smoke (--model ORG/NAME | --checkpoint-dir "
          "<dir>) [--dense bf16|fp8] [--plan]");
  }
  if (dense != "bf16" && dense != "fp8") throw std::runtime_error("--dense takes bf16 or fp8");
  if (!model.empty()) {
    std::string err;
    dir = dgpp::hf::model_dir(model, &err);
    if (dir.empty()) throw std::runtime_error("--model " + model + ": " + err);
  }
  if (dir.empty())
    throw std::runtime_error(
        "usage: qwen3next_loader_smoke (--model ORG/NAME | --checkpoint-dir "
        "<dir>) [--dense bf16|fp8] [--plan]");

  if (plan) return run_plan(Qwen35TextConfig::from_json_file(dir + "/config.json"));

  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  const size_t available = dgpp::host_mem_available_bytes();
  std::printf("checkpoint: %s\nhost memory available: %.1f GiB\n", dir.c_str(), available / kGiB);
  if (available < static_cast<size_t>(40 * kGiB)) {
    std::printf("STOP: under 40 GiB available — not loading\n");
    return 3;
  }

  const Qwen35TextConfig cfg = Qwen35TextConfig::from_json_file(dir + "/config.json");
  if (!cfg.next() || !cfg.moe() || cfg.mtp_layer() < 0)
    throw std::runtime_error("not a Qwen3-Next checkpoint with a draft layer");
  std::printf(
      "config: hidden %d, %d layers + draft, GDN %d/%d heads, attention %d/%d heads, %d experts x "
      "%d, "
      "shared %d\n",
      cfg.hidden_size, cfg.num_hidden_layers, cfg.gdn_key_heads, cfg.gdn_value_heads,
      cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_experts, cfg.moe_intermediate_size,
      cfg.shared_expert_intermediate_size);
  const Source src(dir);
  std::printf("source: %zu shards, %zu tensors mapped for the host computation\n",
              src.shards.size(), src.tensors.size());
  if (dense == "fp8") return run_dense_fp8(cfg, dir, src);
  const int64_t H = cfg.hidden_size, d = cfg.head_dim;
  const int draft = cfg.mtp_layer();
  Kept kept;

  // ---- world 1 -------------------------------------------------------------------
  size_t bytes_gdn = 0, bytes_attn = 0, bytes_draft = 0, bytes_globals = 0;
  {
    Qwen35LayerStream stream(cfg, dir, 0, 1, dgpp::LoaderResidency::Streaming,
                             dgpp::LoaderHeadSharding::Full);
    {
      const std::string tag = "w1 L0";
      const auto t0 = std::chrono::steady_clock::now();
      const Qwen35LayerResident& r = stream.load_layer(0);
      std::printf("-- layer 0 (GDN) loaded in %.2f s, %zu bytes\n", seconds_since(t0), r.bytes);
      bytes_gdn = r.bytes;
      const std::string p = dgpp::qwen35_layer_prefix(cfg, 0);
      check_forms(tag, cfg, r, /*backbone=*/true);
      check_gdn_layer(tag, src, cfg, r, 0, 0, 1, &kept);
      for (int e : {0, 255, 511}) check_fp4_expert(tag, src, cfg, p, r, e, 0, 1);
      check_act_scales(tag, src, p, cfg, r);
      check_bytes(tag + " (e) conv", r.gdn.conv, src, p + "linear_attn.conv1d.weight");
      for (const char* name : {"A_log", "dt_bias"}) {
        // F32 widening: the bf16 bits are the high half of the fp32.
        const std::vector<uint16_t> bits = src.bf16(p + "linear_attn." + name);
        std::vector<float> want(bits.size());
        for (size_t i = 0; i < bits.size(); ++i) want[i] = bf16_value(bits[i]);
        const std::vector<float> got =
            down(std::string(name) == "A_log" ? r.gdn.a_log : r.gdn.dt_bias, want.size());
        report(differing(got, want) == 0, tag + " (e) " + name + " widened to F32: " +
                                              std::to_string(want.size()) + " values");
      }
      check_bytes(tag + " (e) gdn norm", r.gdn.norm, src, p + "linear_attn.norm.weight");
      check_norms(tag, src, p, r);
      check_router(tag, src, p, r);
      report(r.kv_cache_scales == nullptr, tag + " no K/V-cache scales on a GDN layer");
    }
    {
      const std::string tag = "w1 L3";
      const auto t0 = std::chrono::steady_clock::now();
      const Qwen35LayerResident& r = stream.load_layer(3);
      std::printf("-- layer 3 (attention) loaded in %.2f s, %zu bytes\n", seconds_since(t0),
                  r.bytes);
      bytes_attn = r.bytes;
      const std::string p = dgpp::qwen35_layer_prefix(cfg, 3);
      check_forms(tag, cfg, r, /*backbone=*/true);
      check_attention_layer(tag, src, cfg, r, 3, 0, 1, &kept);
      for (int e : {0, 255, 511}) check_fp4_expert(tag, src, cfg, p, r, e, 0, 1);
      check_act_scales(tag, src, p, cfg, r);
      check_bytes(tag + " (e) q_proj", r.full.q_proj, src, p + "self_attn.q_proj.weight");
      check_bytes(tag + " (e) k_proj", r.full.k_proj, src, p + "self_attn.k_proj.weight");
      check_bytes(tag + " (e) v_proj", r.full.v_proj, src, p + "self_attn.v_proj.weight");
      check_bytes(tag + " (e) q_norm", r.full.q_norm, src, p + "self_attn.q_norm.weight");
      check_bytes(tag + " (e) k_norm", r.full.k_norm, src, p + "self_attn.k_norm.weight");
      check_norms(tag, src, p, r);
      check_router(tag, src, p, r);
      const float ks = src.f32(p + "self_attn.k_proj.k_scale"),
                  vs = src.f32(p + "self_attn.v_proj.v_scale");
      const std::vector<float> kv = down(r.kv_cache_scales, 2);
      report(kv[0] == ks && kv[1] == vs && r.k_cache_scale == ks && r.v_cache_scale == vs,
             tag + " (e) K/V-cache scales kept: k_scale " + std::to_string(ks) + ", v_scale " +
                 std::to_string(vs));
    }
    {
      const std::string tag = "w1 L" + std::to_string(draft);
      const auto t0 = std::chrono::steady_clock::now();
      const Qwen35LayerResident& r = stream.load_layer(draft);
      std::printf("-- layer %d (draft) loaded in %.2f s, %zu bytes\n", draft, seconds_since(t0),
                  r.bytes);
      bytes_draft = r.bytes;
      const std::string p = dgpp::qwen35_layer_prefix(cfg, draft);
      check_forms(tag, cfg, r, /*backbone=*/false);
      report(
          r.kind == dgpp::Qwen35LayerKind::Full && r.layer == draft && r.kv_cache_scales == nullptr,
          tag + " kind attention (the draft layer), no K/V-cache scales");
      check_bytes(tag + " (e) q_proj", r.full.q_proj, src, p + "self_attn.q_proj.weight");
      check_bytes(tag + " (e) k_proj", r.full.k_proj, src, p + "self_attn.k_proj.weight");
      check_bytes(tag + " (e) v_proj", r.full.v_proj, src, p + "self_attn.v_proj.weight");
      check_bytes(tag + " (e) o_proj (BF16 as shipped)", r.full.o_proj, src,
                  p + "self_attn.o_proj.weight");
      check_bytes(tag + " (e) q_norm", r.full.q_norm, src, p + "self_attn.q_norm.weight");
      check_bytes(tag + " (e) k_norm", r.full.k_norm, src, p + "self_attn.k_norm.weight");
      check_norms(tag, src, p, r);
      check_router(tag, src, p, r);
      check_bytes(tag + " (e) shared gate_proj (BF16 as shipped)", r.moe.shared[0], src,
                  p + "mlp.shared_expert.gate_proj.weight");
      check_bytes(tag + " (e) shared up_proj (BF16 as shipped)", r.moe.shared[1], src,
                  p + "mlp.shared_expert.up_proj.weight");
      check_bytes(tag + " (e) shared down_proj (BF16 as shipped)", r.moe.shared[2], src,
                  p + "mlp.shared_expert.down_proj.weight");
      Fp8Stats stats;
      for (int e : {0, 511}) check_fp8_expert(tag, src, cfg, r, e, 0, 1, stats);
      print_fp8_stats(stats);
    }
    {
      const auto t0 = std::chrono::steady_clock::now();
      const dgpp::Qwen35GlobalsResident& g = stream.load_globals();
      std::printf("-- globals loaded in %.2f s, %zu bytes\n", seconds_since(t0), g.bytes);
      bytes_globals = g.bytes;
      const std::string tag = "w1 globals";
      check_bytes(tag + " (e) embed", g.embed, src, "model.embed_tokens.weight");
      check_bytes(tag + " (e) lm_head", g.lm_head, src, "lm_head.weight");
      report(g.lm_vocab_begin == 0 && g.lm_vocab_count == cfg.vocab_size,
             tag + " lm head rows [0, vocab)");
      check_bytes(tag + " (e) final norm", g.final_norm, src, "model.norm.weight");
      check_bytes(tag + " (e) mtp.fc", g.mtp_fc, src, "mtp.fc.weight");
      check_bytes(tag + " (e) mtp.norm", g.mtp_norm, src, "mtp.norm.weight");
      check_bytes(tag + " (e) mtp.pre_fc_norm_embedding", g.mtp_pre_fc_norm_embedding, src,
                  "mtp.pre_fc_norm_embedding.weight");
      check_bytes(tag + " (e) mtp.pre_fc_norm_hidden", g.mtp_pre_fc_norm_hidden, src,
                  "mtp.pre_fc_norm_hidden.weight");
    }
    std::printf("-- world 1: %.2f GiB of checkpoint bytes read\n",
                stream.source_bytes_read() / kGiB);
  }

  // ---- world 2: (a) and (b) on both ranks, and the tiling ---------------------------
  {
    const int64_t K = static_cast<int64_t>(cfg.gdn_key_heads) * cfg.gdn_key_head_dim;
    const int64_t V = static_cast<int64_t>(cfg.gdn_value_heads) * cfg.gdn_value_head_dim;
    const int64_t S = cfg.shared_expert_intermediate_size;
    Tiling t_qkv(kept.qkv.size(), H, false), t_z(kept.z.size(), H, false),
        t_a(kept.a.size(), H, false), t_b(kept.b.size(), H, false),
        t_out(kept.out_proj.size(), V, true),
        t_o(kept.o_proj.size(), cfg.num_attention_heads * d, true);
    std::vector<Tiling> t_shared;
    for (int l = 0; l < 2; ++l) {
      t_shared.emplace_back(kept.shared[l][0].size(), H, false);
      t_shared.emplace_back(kept.shared[l][1].size(), H, false);
      t_shared.emplace_back(kept.shared[l][2].size(), S, true);
    }
    size_t w2_bytes[2][2] = {};
    dgpp::ReplicatedDigest digest[2];
    for (int rank = 0; rank < 2; ++rank) {
      Qwen35LayerStream stream(cfg, dir, rank, 2, dgpp::LoaderResidency::Streaming,
                               dgpp::LoaderHeadSharding::VocabSharded);
      digest[rank] = stream.hash_replicated();
      const int64_t lkd = K / 2, lvd = V / 2, lv = cfg.gdn_value_heads / 2, ls = S / 2;
      {
        const std::string tag = "w2 r" + std::to_string(rank) + " L0";
        const Qwen35LayerResident& r = stream.load_layer(0);
        w2_bytes[rank][0] = r.bytes;
        check_gdn_layer(tag, src, cfg, r, 0, rank, 2, nullptr);
        for (int e : {0, 255, 511})
          check_fp4_expert(tag, src, cfg, dgpp::qwen35_layer_prefix(cfg, 0), r, e, rank, 2);
        // The resident slices into the tiling: q | k | v segments of
        // in_proj_qkv to their world-1 rows, the rest to their head ranges.
        const std::vector<uint16_t> qkv =
            down(r.gdn.in_proj_qkv, static_cast<size_t>((2 * lkd + lvd) * H));
        t_qkv.put(row_slice(qkv, H, 0, lkd), rank * lkd, lkd);
        t_qkv.put(row_slice(qkv, H, lkd, lkd), K + rank * lkd, lkd);
        t_qkv.put(row_slice(qkv, H, 2 * lkd, lvd), 2 * K + rank * lvd, lvd);
        t_z.put(down(r.gdn.in_proj_z, static_cast<size_t>(lvd * H)), rank * lvd, lvd);
        t_a.put(down(r.gdn.in_proj_a, static_cast<size_t>(lv * H)), rank * lv, lv);
        t_b.put(down(r.gdn.in_proj_b, static_cast<size_t>(lv * H)), rank * lv, lv);
        t_out.put(down(r.gdn.out_proj, static_cast<size_t>(H * lvd)), rank * lvd, lvd);
        t_shared[0].put(down(r.moe.shared[0], static_cast<size_t>(ls * H)), rank * ls, ls);
        t_shared[1].put(down(r.moe.shared[1], static_cast<size_t>(ls * H)), rank * ls, ls);
        t_shared[2].put(down(r.moe.shared[2], static_cast<size_t>(H * ls)), rank * ls, ls);
      }
      {
        const std::string tag = "w2 r" + std::to_string(rank) + " L3";
        const Qwen35LayerResident& r = stream.load_layer(3);
        w2_bytes[rank][1] = r.bytes;
        check_attention_layer(tag, src, cfg, r, 3, rank, 2, nullptr);
        for (int e : {0, 255, 511})
          check_fp4_expert(tag, src, cfg, dgpp::qwen35_layer_prefix(cfg, 3), r, e, rank, 2);
        const int64_t on = cfg.num_attention_heads / 2 * d;
        t_o.put(down(r.full.o_proj, static_cast<size_t>(H * on)), rank * on, on);
        t_shared[3].put(down(r.moe.shared[0], static_cast<size_t>(ls * H)), rank * ls, ls);
        t_shared[4].put(down(r.moe.shared[1], static_cast<size_t>(ls * H)), rank * ls, ls);
        t_shared[5].put(down(r.moe.shared[2], static_cast<size_t>(H * ls)), rank * ls, ls);
      }
    }
    t_qkv.check("w2 L0 tile in_proj_qkv", kept.qkv);
    t_z.check("w2 L0 tile in_proj_z", kept.z);
    t_a.check("w2 L0 tile in_proj_a", kept.a);
    t_b.check("w2 L0 tile in_proj_b", kept.b);
    t_out.check("w2 L0 tile out_proj", kept.out_proj);
    t_o.check("w2 L3 tile o_proj", kept.o_proj);
    static const char* part[3] = {"gate_proj", "up_proj", "down_proj"};
    for (int l = 0; l < 2; ++l)
      for (int i = 0; i < 3; ++i)
        t_shared[static_cast<size_t>(l * 3 + i)].check(
            std::string("w2 L") + (l == 0 ? "0" : "3") + " tile shared " + part[i],
            kept.shared[l][i]);
    std::printf(
        "-- world 2 resident bytes: layer 0 rank 0/1 = %zu / %zu, layer 3 rank 0/1 = %zu / %zu\n",
        w2_bytes[0][0], w2_bytes[1][0], w2_bytes[0][1], w2_bytes[1][1]);
    report(digest[0].tensors > 0 && digest[0].tensors == digest[1].tensors &&
               digest[0].bytes == digest[1].bytes && digest[0].globals == digest[1].globals &&
               digest[0].layer == digest[1].layer &&
               digest[0].layer.size() == static_cast<size_t>(draft + 1),
           "w2 replicated digest: the two ranks agree on " + std::to_string(digest[0].tensors) +
               " tensors, " + std::to_string(digest[0].bytes) + " bytes, " +
               std::to_string(digest[0].layer.size()) + " layers");
  }

  // ---- resident mode and the image round trip (world 2, rank 1) ----------------------
  // A resident build reads each source once (prefetch, then discard) and
  // captures the layer; the next stream restores it from the image with no
  // copy pass. The built layers are checked against the source, the restored
  // bytes against the built ones, and the host-side scales must come back.
  {
    namespace fs = std::filesystem;
    const fs::path cache =
        fs::temp_directory_path() / ("qwen3next_loader_smoke_image." + std::to_string(getpid()));
    fs::remove_all(cache);
    const std::string saved = Qwen35LayerStream::resident_image_dir();
    Qwen35LayerStream::set_resident_image_dir(cache.string());
    const int rank = 1, world = 2;
    const int layers[2] = {3, draft};
    std::vector<uint8_t> built[2];
    const std::string p3 = dgpp::qwen35_layer_prefix(cfg, 3),
                      pd = dgpp::qwen35_layer_prefix(cfg, draft);
    const float ks = src.f32(p3 + "self_attn.k_proj.k_scale"),
                vs = src.f32(p3 + "self_attn.v_proj.v_scale");
    for (int pass = 0; pass < 2; ++pass) {
      Qwen35LayerStream stream(cfg, dir, rank, world, dgpp::LoaderResidency::Resident,
                               dgpp::LoaderHeadSharding::VocabSharded, /*resident_mtp=*/true);
      const std::string mode = pass == 0 ? "built" : "restored";
      for (int i = 0; i < 2; ++i) {
        const std::string tag = "w2 r1 resident " + mode + " L" + std::to_string(layers[i]);
        const Qwen35LayerResident& r = stream.load_layer(layers[i]);
        std::vector<uint8_t> bytes(r.bytes);
        stream.copy_resident_layer(layers[i], bytes.data());
        if (pass == 0) {
          if (i == 0) {
            check_attention_layer(tag, src, cfg, r, 3, rank, world, nullptr);
          } else {
            // The draft layer's BF16 column slices (packed after the build)
            // and its experts at this rank's intermediate slice.
            const int64_t on = cfg.num_attention_heads / world * d,
                          ls = cfg.shared_expert_intermediate_size / world;
            check_elems(tag + " (e) o_proj columns (BF16 as shipped)", r.full.o_proj,
                        col_slice(src.bf16(pd + "self_attn.o_proj.weight"),
                                  cfg.num_attention_heads * d, rank * on, on));
            check_elems(
                tag + " (e) shared gate_proj rows", r.moe.shared[0],
                row_slice(src.bf16(pd + "mlp.shared_expert.gate_proj.weight"), H, rank * ls, ls));
            check_elems(tag + " (e) shared down_proj columns", r.moe.shared[2],
                        col_slice(src.bf16(pd + "mlp.shared_expert.down_proj.weight"),
                                  cfg.shared_expert_intermediate_size, rank * ls, ls));
            Fp8Stats stats;
            for (int e : {0, 511}) check_fp8_expert(tag, src, cfg, r, e, rank, world, stats);
            print_fp8_stats(stats);
          }
          built[i] = std::move(bytes);
        } else {
          report(bytes == built[i], tag + ": the restored layer is bitwise the built one (" +
                                        std::to_string(bytes.size()) + " bytes)");
        }
        if (i == 0) {
          const std::vector<float> kv = down(r.kv_cache_scales, 2);
          const std::vector<float> act = down(r.moe.act_scales, 2);
          report(
              kv[0] == ks && kv[1] == vs && r.k_cache_scale == ks && r.v_cache_scale == vs &&
                  act[0] > 0.0f && act[1] > 0.0f,
              tag +
                  ": K/V-cache scales on the device and the host, activation scales on the device");
        }
      }
      if (pass == 0)
        report(stream.image_layers_captured() == 2 && stream.image_layers_restored() == 0,
               "w2 r1 resident: the cold load captured both layers to the image");
      else
        report(stream.image_layers_restored() == 2 && stream.source_bytes_read() == 0,
               "w2 r1 resident: the second load restored both layers and read no checkpoint bytes");
    }
    Qwen35LayerStream::set_resident_image_dir(saved);
    fs::remove_all(cache);
  }

  // ---- the byte accounting (world 1) ------------------------------------------------
  const int n_gdn = cfg.num_gdn_layers(), n_attn = cfg.num_full_layers();
  const size_t projected = static_cast<size_t>(n_gdn) * bytes_gdn +
                           static_cast<size_t>(n_attn) * bytes_attn + bytes_draft + bytes_globals;
  const size_t planned = Qwen35LayerStream::resident_bytes(
      cfg, 0, 1, dgpp::LoaderHeadSharding::Full, /*with_mtp=*/true);
  std::printf("resident bytes, world 1:\n");
  std::printf("  layer 0 (GDN)        %12zu  (%.4f GiB)\n", bytes_gdn, bytes_gdn / kGiB);
  std::printf("  layer 3 (attention)  %12zu  (%.4f GiB)\n", bytes_attn, bytes_attn / kGiB);
  std::printf("  layer %d (draft)     %12zu  (%.4f GiB)\n", draft, bytes_draft, bytes_draft / kGiB);
  std::printf("  globals              %12zu  (%.4f GiB)\n", bytes_globals, bytes_globals / kGiB);
  std::printf("  projected total: %d x GDN + %d x attention + draft + globals = %zu  (%.3f GiB)\n",
              n_gdn, n_attn, projected, projected / kGiB);
  std::printf("  without the draft layer: %zu  (%.3f GiB)\n", projected - bytes_draft,
              (projected - bytes_draft) / kGiB);
  std::printf(
      "  pinned staging mirror during the load: %zu  (%.3f GiB)\n",
      Qwen35LayerStream::staging_plan_bytes(cfg, 0, 1, dgpp::LoaderHeadSharding::Full, true),
      Qwen35LayerStream::staging_plan_bytes(cfg, 0, 1, dgpp::LoaderHeadSharding::Full, true) /
          kGiB);
  report(projected == planned,
         "byte accounting: the projection equals the stream's resident_bytes plan (" +
             std::to_string(planned) + ")");

  std::printf("%d checks, %d failed\n", g_passed + g_failed, g_failed);
  std::printf("%s\n", g_failed == 0 ? "SMOKE PASS" : "SMOKE FAIL");
  return g_failed == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::printf("FAIL  exception: %s\n", e.what());
    return 1;
  }
}
