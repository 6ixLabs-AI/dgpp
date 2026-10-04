// UNVERIFIED DRAFT (2026-10-04) — see models/gemma4/forward_draft.hpp for what
// has and has not run. Built only with -DDGPP_BUILD_GEMMA_DRAFT=ON.
//
// Every kernel here is ONE THREAD PER OUTPUT ELEMENT: thread i reads its
// inputs, loops serially over whatever it reduces, writes element i and
// touches nothing another thread writes. No shared memory, no barriers, a
// 1-D grid of 256-thread blocks with a bounds check. That is what lets the
// same source run as plain C++ (G4_LAUNCH below becomes a loop over thread
// indices) for the host test, and it keeps the first GPU run about one
// question — does nvcc build it and do the numbers match — rather than
// about a reduction tree.
//
// The arithmetic mirrors the host reference (models/gemma4/reference.cpp)
// line for line: dot products and norm sums accumulated in double, fp32
// values, the rotary angle in double, no rounding to bf16.
#include "models/gemma4/forward_draft.hpp"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"
#include "loaders/safetensors.hpp"
#include "models/gemma4/binding.hpp"

#if defined(__CUDACC__)
#define G4_KERNEL __global__ void
#define G4_THREAD (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x)
#define G4_LAUNCH(kernel, n, ...)                                                               \
  do {                                                                                          \
    kernel<<<static_cast<unsigned>((static_cast<int64_t>(n) + 255) / 256), 256, 0, nullptr>>>(  \
        __VA_ARGS__);                                                                           \
    DGPP_CUDA_OK(cudaGetLastError());                                                           \
  } while (0)
#else
// Host emulation (no nvcc): a "launch" runs every thread index of the grid
// in turn, the padding threads past n included.
namespace {
thread_local int64_t g4_thread = 0;
}
#define G4_KERNEL static void
#define G4_THREAD g4_thread
#define G4_LAUNCH(kernel, n, ...)                                                               \
  do {                                                                                          \
    const int64_t g4_threads = ((static_cast<int64_t>(n) + 255) / 256) * 256;                   \
    for (int64_t g4_i = 0; g4_i < g4_threads; ++g4_i) {                                         \
      g4_thread = g4_i;                                                                         \
      kernel(__VA_ARGS__);                                                                      \
    }                                                                                           \
  } while (0)
#endif

namespace dgpp {
namespace {

// ---- kernels -------------------------------------------------------------------

// x[t, d] = bf16(table[ids[t], d]) * scale.            n = T * H
G4_KERNEL g4_embed(const uint16_t* table, const int64_t* ids, float scale, float* x, int H, int64_t n) {
  const int64_t i = G4_THREAD;
  if (i >= n) return;
  const int64_t t = i / H, d = i % H;
  x[i] = bf16_bits_to_float(table[ids[t] * H + d]) * scale;
}

// y[t, r] = W[r, :] . x[t, :], W bf16 [N, K].          n = T * N
G4_KERNEL g4_gemv_bf16(const uint16_t* w, const float* x, float* y, int N, int K, int64_t n) {
  const int64_t i = G4_THREAD;
  if (i >= n) return;
  const int64_t t = i / N, r = i % N;
  const uint16_t* row = w + r * K;
  const float* xr = x + t * K;
  double acc = 0.0;
  for (int k = 0; k < K; ++k) acc += static_cast<double>(bf16_bits_to_float(row[k])) * static_cast<double>(xr[k]);
  y[i] = static_cast<float>(acc);
}

// One row of a modelopt NVFP4 matrix against x [K]: prow U8 [K/2] (low
// nibble = even column), srow e4m3 [K/16], weight_scale_2 a multiplier.
DGPP_HD inline float g4_nvfp4_row_dot(const uint8_t* prow, const uint8_t* srow, float ws2, const float* xr, int K) {
  double acc = 0.0;
  for (int g = 0; g < K / 16; ++g) {
    const float s = fp8_e4m3_bits_to_float(srow[g]) * ws2;
    for (int j = 0; j < 8; ++j) {
      const uint8_t byte = prow[g * 8 + j];
      const float lo = fp4_e2m1_bits_to_float(static_cast<uint8_t>(byte & 0xFu)) * s;
      const float hi = fp4_e2m1_bits_to_float(static_cast<uint8_t>(byte >> 4)) * s;
      acc += static_cast<double>(lo) * static_cast<double>(xr[g * 16 + 2 * j]);
      acc += static_cast<double>(hi) * static_cast<double>(xr[g * 16 + 2 * j + 1]);
    }
  }
  return static_cast<float>(acc);
}

DGPP_HD inline float g4_gelu_tanh(float x) {
  return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// y[t, r] = W[r, :] . x[t, :] over a modelopt NVFP4 matrix [N, K].   n = T * N
G4_KERNEL g4_gemv_nvfp4(const uint8_t* payload, const uint8_t* scales, float ws2, const float* x, float* y, int N,
                        int K, int64_t n) {
  const int64_t i = G4_THREAD;
  if (i >= n) return;
  const int64_t t = i / N, r = i % N;
  y[i] = g4_nvfp4_row_dot(payload + r * (K / 2), scales + r * (K / 16), ws2, x + t * K, K);
}

// Gemma4RMSNorm of row i: y = x * (mean(x^2) + eps)^-0.5 [* w], w bf16 [dim]
// or null (the value norm). y may be x.                 n = rows
G4_KERNEL g4_rms_norm(const float* x, const uint16_t* w, float eps, float* y, int dim, int64_t n) {
  const int64_t i = G4_THREAD;
  if (i >= n) return;
  const float* xr = x + i * dim;
  float* yr = y + i * dim;
  double ss = 0.0;
  for (int d = 0; d < dim; ++d) ss += static_cast<double>(xr[d]) * static_cast<double>(xr[d]);
  const float ms = static_cast<float>(ss / static_cast<double>(dim)) + eps;
  const float r = powf(ms, -0.5f);
  for (int d = 0; d < dim; ++d) yr[d] = w != nullptr ? xr[d] * r * bf16_bits_to_float(w[d]) : xr[d] * r;
}

// Rotates head i of x [T, heads, hd] in place at position pos0 + t: pair p
// couples (p, p + hd/2) with angle pos * inv_freq[p], p < pairs.   n = T * heads
G4_KERNEL g4_rope(float* x, const double* inv_freq, int pairs, int hd, int heads, int64_t pos0, int64_t n) {
  const int64_t i = G4_THREAD;
  if (i >= n) return;
  const int64_t pos = pos0 + i / heads;
  float* h = x + i * hd;
  const int half = hd / 2;
  for (int p = 0; p < pairs; ++p) {
    const double ang = static_cast<double>(pos) * inv_freq[p];
    const float c = static_cast<float>(cos(ang)), s = static_cast<float>(sin(ang));
    const float a = h[p], b = h[p + half];
    h[p] = a * c - b * s;
    h[p + half] = b * c + a * s;
  }
}

// One (position, query head): scores over the visible keys of the layer's
// cache — rows [first, pos] of kc / vc ([rows, kvh x hd]) — a softmax, the
// value sum. Plain dot products (no 1/sqrt(d)); head h reads KV head
// h / (heads / kvh). `scores` is scratch, [T x heads, stride].   n = T * heads
G4_KERNEL g4_attention(const float* q, const float* kc, const float* vc, float* scores, float* ctx, int heads,
                       int kvh, int hd, int64_t pos0, int window, int64_t stride, int64_t n) {
  const int64_t i = G4_THREAD;
  if (i >= n) return;
  const int64_t t = i / heads;
  const int h = static_cast<int>(i % heads);
  const int kv = h / (heads / kvh);
  const int64_t pos = pos0 + t;
  int64_t first = 0;
  if (window > 0 && pos - window + 1 > 0) first = pos - window + 1;  // kv_idx > q_idx - sliding_window
  const int64_t count = pos - first + 1;
  const int64_t width = static_cast<int64_t>(kvh) * hd;
  const float* qh = q + i * hd;
  float* sc = scores + i * stride;
  float m = -INFINITY;
  for (int64_t s = 0; s < count; ++s) {
    const float* kr = kc + (first + s) * width + static_cast<int64_t>(kv) * hd;
    double dot = 0.0;
    for (int d = 0; d < hd; ++d) dot += static_cast<double>(qh[d]) * static_cast<double>(kr[d]);
    sc[s] = static_cast<float>(dot);
    if (sc[s] > m) m = sc[s];
  }
  double denom = 0.0;
  for (int64_t s = 0; s < count; ++s) {
    sc[s] = expf(sc[s] - m);
    denom += static_cast<double>(sc[s]);
  }
  float* out = ctx + i * hd;
  for (int d = 0; d < hd; ++d) {
    double acc = 0.0;
    for (int64_t s = 0; s < count; ++s)
      acc += (static_cast<double>(sc[s]) / denom) *
             static_cast<double>(vc[(first + s) * width + static_cast<int64_t>(kv) * hd + d]);
    out[d] = static_cast<float>(acc);
  }
}

// gate[i] = gelu_tanh(gate[i]) * up[i].
G4_KERNEL g4_geglu(float* gate, const float* up, int64_t n) {
  const int64_t i = G4_THREAD;
  if (i >= n) return;
  gate[i] = g4_gelu_tanh(gate[i]) * up[i];
}

// x[i] = (x[i] + y[i]) * scalar.
G4_KERNEL g4_residual(float* x, const float* y, float scalar, int64_t n) {
  const int64_t i = G4_THREAD;
  if (i >= n) return;
  x[i] = (x[i] + y[i]) * scalar;
}

// The MoE block's router for token i (models/gemma4/reference.cpp route(),
// operation for operation): r = rms_noscale(x) * scale * in_scale; an fp32
// softmax of proj @ r; the K largest, best first, a tie to the lower id;
// renormalized, times per_expert_scale. r [T, H] and p [T, E] are scratch.
// n = T
G4_KERNEL g4_route(const float* x, const uint16_t* proj, const uint16_t* scale, const uint16_t* pes, float in_scale,
                   float eps, float* r, float* p, int* ids, float* w, int H, int E, int K, int64_t n) {
  const int64_t i = G4_THREAD;
  if (i >= n) return;
  const float* xr = x + i * H;
  float* rr = r + i * H;
  float* pp = p + i * E;
  double ss = 0.0;
  for (int d = 0; d < H; ++d) ss += static_cast<double>(xr[d]) * static_cast<double>(xr[d]);
  const float rinv = powf(static_cast<float>(ss / static_cast<double>(H)) + eps, -0.5f);
  for (int d = 0; d < H; ++d) rr[d] = xr[d] * rinv * bf16_bits_to_float(scale[d]) * in_scale;
  float m = -INFINITY;
  for (int e = 0; e < E; ++e) {
    const uint16_t* row = proj + static_cast<int64_t>(e) * H;
    double dot = 0.0;
    for (int d = 0; d < H; ++d) dot += static_cast<double>(bf16_bits_to_float(row[d])) * static_cast<double>(rr[d]);
    pp[e] = static_cast<float>(dot);
    if (pp[e] > m) m = pp[e];
  }
  double denom = 0.0;
  for (int e = 0; e < E; ++e) {
    pp[e] = expf(pp[e] - m);
    denom += static_cast<double>(pp[e]);
  }
  for (int e = 0; e < E; ++e) pp[e] = static_cast<float>(static_cast<double>(pp[e]) / denom);
  int* id = ids + i * K;
  float* wt = w + i * K;
  double sum = 0.0;
  for (int k = 0; k < K; ++k) {
    int best = -1;
    for (int e = 0; e < E; ++e)
      if (pp[e] >= 0.0f && (best < 0 || pp[e] > pp[best])) best = e;  // a taken slot holds -1
    id[k] = best;
    wt[k] = pp[best];
    sum += static_cast<double>(wt[k]);
    pp[best] = -1.0f;
  }
  for (int k = 0; k < K; ++k) wt[k] = static_cast<float>(static_cast<double>(wt[k]) / sum) * bf16_bits_to_float(pes[id[k]]);
}

// act[t, k, row] = gelu_tanh(gate_e[row] . h[t]) * (up_e[row] . h[t]) for
// the token's k-th expert e. The experts of a layer lie contiguous: payload
// [E, Im, H/2], scales [E, Im, H/16], weight_scale_2 [E].   n = T * K * Im
G4_KERNEL g4_expert_act(const uint8_t* gp, const uint8_t* gs, const float* gws2, const uint8_t* up, const uint8_t* us,
                        const float* uws2, const int* ids, const float* h, float* act, int H, int Im, int K, int64_t n) {
  const int64_t i = G4_THREAD;
  if (i >= n) return;
  const int64_t t = i / (static_cast<int64_t>(K) * Im);
  const int64_t k = (i / Im) % K, row = i % Im;
  const int64_t e = ids[t * K + k];
  const int64_t at = e * Im + row;
  const float g = g4_nvfp4_row_dot(gp + at * (H / 2), gs + at * (H / 16), gws2[e], h + t * H, H);
  const float u = g4_nvfp4_row_dot(up + at * (H / 2), us + at * (H / 16), uws2[e], h + t * H, H);
  act[i] = g4_gelu_tanh(g) * u;
}

// out[t, d] = sum_k w[t, k] * (down_e[d] . act[t, k]): payload [E, H, Im/2],
// scales [E, H, Im/16].   n = T * H
G4_KERNEL g4_expert_down(const uint8_t* dp, const uint8_t* ds, const float* dws2, const int* ids, const float* w,
                         const float* act, float* out, int H, int Im, int K, int64_t n) {
  const int64_t i = G4_THREAD;
  if (i >= n) return;
  const int64_t t = i / H, d = i % H;
  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    const int64_t e = ids[t * K + k];
    const int64_t at = e * H + d;
    const float y = g4_nvfp4_row_dot(dp + at * (Im / 2), ds + at * (Im / 16), dws2[e], act + (t * K + k) * Im, Im);
    sum += w[t * K + k] * y;
  }
  out[i] = sum;
}

// ---- device tensors ----------------------------------------------------------------

struct DevMatrix {
  int64_t rows = 0, cols = 0;
  bool nvfp4 = false;
  void* w = nullptr;       // bf16 [rows, cols], or the NVFP4 payload [rows, cols / 2]
  void* scales = nullptr;  // NVFP4: e4m3 [rows, cols / 16]
  float ws2 = 1.0f;
  bool empty() const { return w == nullptr; }
};

struct DevLayer {
  uint16_t* norms[4] = {nullptr, nullptr, nullptr, nullptr};  // input, post-attention, pre-ffn, post-ffn
  float scalar = 1.0f;
  uint16_t* q_norm = nullptr;
  uint16_t* k_norm = nullptr;
  double* inv_freq = nullptr;
  DevMatrix q, k, v, o, gate, up, down;
  float* kc = nullptr;  // [max_tokens, kvh x hd]
  float* vc = nullptr;
  // The MoE block: three more norms, the BF16 router, and the experts'
  // NVFP4 sets laid contiguous by expert id (gate, up, down).
  uint16_t* moe_norms[3] = {nullptr, nullptr, nullptr};  // post-ffn 1, post-ffn 2, pre-ffn 2
  uint16_t* router_proj = nullptr;
  uint16_t* router_scale = nullptr;
  uint16_t* per_expert_scale = nullptr;
  uint8_t* ex_payload[3] = {nullptr, nullptr, nullptr};
  uint8_t* ex_scales[3] = {nullptr, nullptr, nullptr};
  float* ex_ws2[3] = {nullptr, nullptr, nullptr};
};

// Every shard of a checkpoint directory, tensors found by name.
class Shards {
 public:
  explicit Shards(const std::string& dir) {
    namespace fs = std::filesystem;
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir))
      if (entry.path().extension() == ".safetensors") paths.push_back(entry.path());
    if (paths.empty()) throw std::runtime_error("gemma4 draft: no .safetensors shards in " + dir);
    std::sort(paths.begin(), paths.end());
    for (const auto& p : paths) files_.push_back(SafetensorsFile::open(p.string()));
  }
  const TensorInfo& expect(const std::string& name, DType dtype, const std::vector<int64_t>& shape) const {
    for (const auto& f : files_)
      if (const TensorInfo* t = f->find(name)) {
        if (t->dtype != dtype || t->shape != shape)
          throw std::runtime_error("gemma4 draft: '" + name + "' has another dtype or shape than the table");
        return *t;
      }
    throw std::runtime_error("gemma4 draft: tensor not found: " + name);
  }

 private:
  std::vector<std::unique_ptr<SafetensorsFile>> files_;
};

}  // namespace

struct Gemma4DraftForward::Impl {
  Gemma4TextConfig cfg;
  int64_t max_tokens = 0;
  int64_t pos = 0;
  size_t bytes = 0;
  std::vector<void*> owned;  // every device allocation, freed in the destructor
  uint16_t* embed = nullptr;
  uint16_t* final_norm = nullptr;
  std::vector<DevLayer> layers;

  void* alloc(size_t n) {
    void* p = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&p, n ? n : 1));
    owned.push_back(p);
    bytes += n;
    return p;
  }
  void* upload(const void* host, size_t n) {
    void* p = alloc(n);
    DGPP_CUDA_OK(cudaMemcpy(p, host, n, cudaMemcpyHostToDevice));
    return p;
  }
  uint16_t* upload_bf16(const Shards& s, const std::string& name, const std::vector<int64_t>& shape) {
    const TensorInfo& t = s.expect(name, DType::BF16, shape);
    return static_cast<uint16_t*>(upload(t.data, t.nbytes()));
  }
  DevMatrix upload_matrix(const Shards& s, const std::string& base, int64_t rows, int64_t cols, bool nvfp4) {
    DevMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.nvfp4 = nvfp4;
    if (!nvfp4) {
      m.w = upload_bf16(s, base + ".weight", {rows, cols});
      return m;
    }
    const TensorInfo& payload = s.expect(base + ".weight", DType::U8, {rows, cols / 2});
    const TensorInfo& scales = s.expect(base + ".weight_scale", DType::F8_E4M3, {rows, cols / kLatentFp4Block});
    const TensorInfo& global = s.expect(base + ".weight_scale_2", DType::F32, {});
    std::memcpy(&m.ws2, global.data, sizeof m.ws2);
    if (!(m.ws2 > 0.0f) || !std::isfinite(m.ws2))
      throw std::runtime_error("gemma4 draft: '" + base + ".weight_scale_2' is not a positive finite number");
    m.w = upload(payload.data, payload.nbytes());
    m.scales = upload(scales.data, scales.nbytes());
    return m;
  }
  // One of an MoE layer's three expert projections ([rows, cols] each), all
  // experts into three contiguous device buffers.
  void upload_experts(const Shards& s, const std::string& layer_prefix, const char* which, int64_t rows, int64_t cols,
                      int n_experts, uint8_t** payload, uint8_t** scales, float** ws2) {
    const size_t pbytes = static_cast<size_t>(rows * (cols / 2)), sbytes = static_cast<size_t>(rows * (cols / kLatentFp4Block));
    *payload = static_cast<uint8_t*>(alloc(pbytes * static_cast<size_t>(n_experts)));
    *scales = static_cast<uint8_t*>(alloc(sbytes * static_cast<size_t>(n_experts)));
    std::vector<float> globals(static_cast<size_t>(n_experts));
    for (int e = 0; e < n_experts; ++e) {
      const std::string base = layer_prefix + "moe.experts." + std::to_string(e) + "." + which;
      const TensorInfo& p = s.expect(base + ".weight", DType::U8, {rows, cols / 2});
      const TensorInfo& sc = s.expect(base + ".weight_scale", DType::F8_E4M3, {rows, cols / kLatentFp4Block});
      const TensorInfo& g = s.expect(base + ".weight_scale_2", DType::F32, {});
      std::memcpy(&globals[static_cast<size_t>(e)], g.data, sizeof(float));
      if (!(globals[static_cast<size_t>(e)] > 0.0f) || !std::isfinite(globals[static_cast<size_t>(e)]))
        throw std::runtime_error("gemma4 draft: '" + base + ".weight_scale_2' is not a positive finite number");
      DGPP_CUDA_OK(cudaMemcpy(*payload + pbytes * static_cast<size_t>(e), p.data, pbytes, cudaMemcpyHostToDevice));
      DGPP_CUDA_OK(cudaMemcpy(*scales + sbytes * static_cast<size_t>(e), sc.data, sbytes, cudaMemcpyHostToDevice));
    }
    *ws2 = static_cast<float*>(upload(globals.data(), globals.size() * sizeof(float)));
  }
  // y [T, rows] = W @ x [T, cols].
  void gemv(const DevMatrix& m, const float* x, float* y, int64_t T) {
    const int N = static_cast<int>(m.rows), K = static_cast<int>(m.cols);
    if (m.nvfp4)
      G4_LAUNCH(g4_gemv_nvfp4, T * N, static_cast<const uint8_t*>(m.w), static_cast<const uint8_t*>(m.scales), m.ws2, x,
                y, N, K, T * N);
    else
      G4_LAUNCH(g4_gemv_bf16, T * N, static_cast<const uint16_t*>(m.w), x, y, N, K, T * N);
  }
};

Gemma4DraftForward::Gemma4DraftForward(const Gemma4TextConfig& cfg, const std::string& checkpoint_dir, int layers,
                                       int64_t max_tokens)
    : impl_(new Impl) {
  try {
    Impl& m = *impl_;
    m.cfg = cfg;
    m.max_tokens = max_tokens;
    if (max_tokens <= 0) throw std::invalid_argument("gemma4 draft: max_tokens must be positive");
    const Shards s(checkpoint_dir);
    const int n_layers = layers < 0 ? cfg.num_hidden_layers : std::min(layers, cfg.num_hidden_layers);
    const int64_t H = cfg.hidden_size, I = cfg.intermediate_size;
    m.embed = m.upload_bf16(s, "model.language_model.embed_tokens.weight", {cfg.vocab_size, H});
    m.final_norm = m.upload_bf16(s, "model.language_model.norm.weight", {H});
    m.layers.resize(static_cast<size_t>(n_layers));
    const bool attn4 = cfg.attention_nvfp4();
    for (int l = 0; l < n_layers; ++l) {
      const std::string p = gemma4_layer_prefix(l);
      DevLayer& d = m.layers[static_cast<size_t>(l)];
      const char* norms[4] = {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                              "post_feedforward_layernorm"};
      for (int k = 0; k < 4; ++k) d.norms[k] = m.upload_bf16(s, p + norms[k] + ".weight", {H});
      {
        const TensorInfo& t = s.expect(p + "layer_scalar", DType::BF16, {1});
        d.scalar = bf16_bits_to_float(*static_cast<const uint16_t*>(t.data));
      }
      const int64_t hd = cfg.head_dim_of(l);
      d.q_norm = m.upload_bf16(s, p + "self_attn.q_norm.weight", {hd});
      d.k_norm = m.upload_bf16(s, p + "self_attn.k_norm.weight", {hd});
      d.q = m.upload_matrix(s, p + "self_attn.q_proj", cfg.q_rows(l), H, attn4);
      d.k = m.upload_matrix(s, p + "self_attn.k_proj", cfg.kv_rows(l), H, attn4);
      if (cfg.has_v_proj(l)) d.v = m.upload_matrix(s, p + "self_attn.v_proj", cfg.kv_rows(l), H, attn4);
      d.o = m.upload_matrix(s, p + "self_attn.o_proj", H, cfg.q_rows(l), attn4);
      d.gate = m.upload_matrix(s, p + "mlp.gate_proj", I, H, true);
      d.up = m.upload_matrix(s, p + "mlp.up_proj", I, H, true);
      d.down = m.upload_matrix(s, p + "mlp.down_proj", H, I, true);
      if (cfg.enable_moe_block) {
        const int64_t E = cfg.num_experts, Im = cfg.moe_intermediate_size;
        const char* moe_norms[3] = {"post_feedforward_layernorm_1", "post_feedforward_layernorm_2",
                                    "pre_feedforward_layernorm_2"};
        for (int k = 0; k < 3; ++k) d.moe_norms[k] = m.upload_bf16(s, p + moe_norms[k] + ".weight", {H});
        d.router_proj = m.upload_bf16(s, p + "router.proj.weight", {E, H});
        d.router_scale = m.upload_bf16(s, p + "router.scale", {H});
        d.per_expert_scale = m.upload_bf16(s, p + "router.per_expert_scale", {E});
        m.upload_experts(s, p, "gate_proj", Im, H, cfg.num_experts, &d.ex_payload[0], &d.ex_scales[0], &d.ex_ws2[0]);
        m.upload_experts(s, p, "up_proj", Im, H, cfg.num_experts, &d.ex_payload[1], &d.ex_scales[1], &d.ex_ws2[1]);
        m.upload_experts(s, p, "down_proj", H, Im, cfg.num_experts, &d.ex_payload[2], &d.ex_scales[2], &d.ex_ws2[2]);
      }
      // The rotated pairs' inverse frequencies: theta^(-2p / head_dim).
      const int pairs = cfg.rotary_pairs_of(l);
      std::vector<double> inv(static_cast<size_t>(pairs));
      for (int k = 0; k < pairs; ++k)
        inv[static_cast<size_t>(k)] = pow(cfg.rope_theta_of(l), -2.0 * static_cast<double>(k) / static_cast<double>(hd));
      d.inv_freq = static_cast<double*>(m.upload(inv.data(), inv.size() * sizeof(double)));
      const size_t cache = static_cast<size_t>(max_tokens) * static_cast<size_t>(cfg.kv_rows(l)) * sizeof(float);
      d.kc = static_cast<float*>(m.alloc(cache));
      d.vc = static_cast<float*>(m.alloc(cache));
    }
  } catch (...) {
    for (void* p : impl_->owned) cudaFree(p);
    delete impl_;
    throw;
  }
}

Gemma4DraftForward::~Gemma4DraftForward() {
  for (void* p : impl_->owned) cudaFree(p);
  delete impl_;
}

int Gemma4DraftForward::layers() const { return static_cast<int>(impl_->layers.size()); }
int64_t Gemma4DraftForward::pos() const { return impl_->pos; }
size_t Gemma4DraftForward::device_bytes() const { return impl_->bytes; }

namespace {

// A scratch device buffer for one call.
struct Scratch {
  explicit Scratch(size_t n) { DGPP_CUDA_OK(cudaMalloc(&p, n ? n : 1)); }
  ~Scratch() { cudaFree(p); }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  float* f() const { return static_cast<float*>(p); }
  void* p = nullptr;
};

}  // namespace

std::vector<float> Gemma4DraftForward::forward(const std::vector<int64_t>& ids, std::vector<std::vector<float>>* trace) {
  Impl& m = *impl_;
  const Gemma4TextConfig& cfg = m.cfg;
  const int64_t T = static_cast<int64_t>(ids.size());
  if (T == 0) return {};
  if (m.pos + T > m.max_tokens) throw std::runtime_error("gemma4 draft: the sequence outgrows the K/V cache (max_tokens)");
  for (const int64_t id : ids)
    if (id < 0 || id >= cfg.vocab_size) throw std::out_of_range("gemma4 draft: token id outside the vocabulary");
  const int H = cfg.hidden_size, I = cfg.intermediate_size, heads = cfg.num_attention_heads;
  const int64_t max_q = static_cast<int64_t>(heads) * std::max(cfg.head_dim, cfg.global_head_dim);
  const int64_t max_kv = std::max(static_cast<int64_t>(cfg.num_key_value_heads) * cfg.head_dim,
                                  static_cast<int64_t>(cfg.num_global_key_value_heads) * cfg.global_head_dim);
  const int64_t stride = m.pos + T;  // the most keys a row of this call can see
  const size_t f = sizeof(float);
  Scratch d_ids(static_cast<size_t>(T) * sizeof(int64_t));
  Scratch x(static_cast<size_t>(T * H) * f), h(static_cast<size_t>(T * H) * f), a(static_cast<size_t>(T * H) * f);
  Scratch q(static_cast<size_t>(T * max_q) * f), ctx(static_cast<size_t>(T * max_q) * f);
  Scratch k(static_cast<size_t>(T * max_kv) * f), v(static_cast<size_t>(T * max_kv) * f);
  Scratch gate(static_cast<size_t>(T * I) * f), up(static_cast<size_t>(T * I) * f);
  Scratch scores(static_cast<size_t>(T * heads * stride) * f);
  // The MoE block's scratch: the router's input and probabilities, the
  // picks, the picked experts' activations and their weighted sum.
  const bool moe = cfg.enable_moe_block;
  const int E = moe ? cfg.num_experts : 0, K = moe ? cfg.top_k_experts : 0, Im = moe ? cfg.moe_intermediate_size : 0;
  Scratch route_r(static_cast<size_t>(moe ? T * H : 0) * f), route_p(static_cast<size_t>(T * E) * f);
  Scratch route_ids(static_cast<size_t>(T * K) * sizeof(int)), route_w(static_cast<size_t>(T * K) * f);
  Scratch ex_act(static_cast<size_t>(T * K * Im) * f), ex_out(static_cast<size_t>(moe ? T * H : 0) * f);
  DGPP_CUDA_OK(cudaMemcpy(d_ids.p, ids.data(), static_cast<size_t>(T) * sizeof(int64_t), cudaMemcpyHostToDevice));

  auto snapshot = [&] {
    std::vector<float> out(static_cast<size_t>(T * H));
    DGPP_CUDA_OK(cudaMemcpy(out.data(), x.p, out.size() * f, cudaMemcpyDeviceToHost));
    return out;
  };
  G4_LAUNCH(g4_embed, T * H, m.embed, static_cast<const int64_t*>(d_ids.p), cfg.embed_scale(), x.f(), H, T * H);
  if (trace != nullptr) trace->push_back(snapshot());

  for (size_t li = 0; li < m.layers.size(); ++li) {
    const int l = static_cast<int>(li);
    DevLayer& d = m.layers[li];
    const int hd = cfg.head_dim_of(l), kvh = cfg.kv_heads_of(l);
    const int64_t kw = static_cast<int64_t>(kvh) * hd;
    const int window = cfg.is_sliding_layer(l) ? cfg.sliding_window : 0;
    const int pairs = cfg.rotary_pairs_of(l);
    // --- attention: x = x + rms(attn(rms(x, input_layernorm)), post_attention_layernorm)
    G4_LAUNCH(g4_rms_norm, T, x.f(), d.norms[0], cfg.rms_norm_eps, h.f(), H, T);
    m.gemv(d.q, h.f(), q.f(), T);
    m.gemv(d.k, h.f(), k.f(), T);
    // The value: its own projection, or (a full layer) the k_proj output as
    // it stands — before k_norm and before the rotation.
    if (!d.v.empty())
      m.gemv(d.v, h.f(), v.f(), T);
    else
      DGPP_CUDA_OK(cudaMemcpy(v.p, k.p, static_cast<size_t>(T * kw) * f, cudaMemcpyDeviceToDevice));
    G4_LAUNCH(g4_rms_norm, T * heads, q.f(), d.q_norm, cfg.rms_norm_eps, q.f(), hd, T * heads);
    G4_LAUNCH(g4_rope, T * heads, q.f(), d.inv_freq, pairs, hd, heads, m.pos, T * heads);
    G4_LAUNCH(g4_rms_norm, T * kvh, k.f(), d.k_norm, cfg.rms_norm_eps, k.f(), hd, T * kvh);
    G4_LAUNCH(g4_rope, T * kvh, k.f(), d.inv_freq, pairs, hd, kvh, m.pos, T * kvh);
    G4_LAUNCH(g4_rms_norm, T * kvh, v.f(), static_cast<const uint16_t*>(nullptr), cfg.rms_norm_eps, v.f(), hd, T * kvh);
    // Append this chunk's rows to the layer's cache, then attend.
    DGPP_CUDA_OK(cudaMemcpy(d.kc + m.pos * kw, k.p, static_cast<size_t>(T * kw) * f, cudaMemcpyDeviceToDevice));
    DGPP_CUDA_OK(cudaMemcpy(d.vc + m.pos * kw, v.p, static_cast<size_t>(T * kw) * f, cudaMemcpyDeviceToDevice));
    G4_LAUNCH(g4_attention, T * heads, q.f(), d.kc, d.vc, scores.f(), ctx.f(), heads, kvh, hd, m.pos, window, stride,
              T * heads);
    m.gemv(d.o, ctx.f(), a.f(), T);
    G4_LAUNCH(g4_rms_norm, T, a.f(), d.norms[1], cfg.rms_norm_eps, a.f(), H, T);
    G4_LAUNCH(g4_residual, T * H, x.f(), a.f(), 1.0f, T * H);
    // --- MLP: x = (x + rms(mlp(rms(x, pre_feedforward_layernorm)), post_feedforward_layernorm)) * layer_scalar
    G4_LAUNCH(g4_rms_norm, T, x.f(), d.norms[2], cfg.rms_norm_eps, h.f(), H, T);
    m.gemv(d.gate, h.f(), gate.f(), T);
    m.gemv(d.up, h.f(), up.f(), T);
    G4_LAUNCH(g4_geglu, T * I, gate.f(), up.f(), T * I);
    m.gemv(d.down, gate.f(), a.f(), T);
    if (moe) {
      // a = rms(mlp, post_feedforward_layernorm_1) + rms(moe(x), post_feedforward_layernorm_2): the
      // router reads the residual stream itself, the experts their own norm of it.
      G4_LAUNCH(g4_route, T, x.f(), d.router_proj, d.router_scale, d.per_expert_scale, cfg.router_input_scale(),
                cfg.rms_norm_eps, route_r.f(), route_p.f(), static_cast<int*>(route_ids.p), route_w.f(), H, E, K, T);
      G4_LAUNCH(g4_rms_norm, T, x.f(), d.moe_norms[2], cfg.rms_norm_eps, h.f(), H, T);
      G4_LAUNCH(g4_expert_act, T * K * Im, d.ex_payload[0], d.ex_scales[0], d.ex_ws2[0], d.ex_payload[1], d.ex_scales[1],
                d.ex_ws2[1], static_cast<const int*>(route_ids.p), h.f(), ex_act.f(), H, Im, K, T * K * Im);
      G4_LAUNCH(g4_expert_down, T * H, d.ex_payload[2], d.ex_scales[2], d.ex_ws2[2], static_cast<const int*>(route_ids.p),
                route_w.f(), ex_act.f(), ex_out.f(), H, Im, K, T * H);
      G4_LAUNCH(g4_rms_norm, T, a.f(), d.moe_norms[0], cfg.rms_norm_eps, a.f(), H, T);
      G4_LAUNCH(g4_rms_norm, T, ex_out.f(), d.moe_norms[1], cfg.rms_norm_eps, ex_out.f(), H, T);
      G4_LAUNCH(g4_residual, T * H, a.f(), ex_out.f(), 1.0f, T * H);
    }
    G4_LAUNCH(g4_rms_norm, T, a.f(), d.norms[3], cfg.rms_norm_eps, a.f(), H, T);
    G4_LAUNCH(g4_residual, T * H, x.f(), a.f(), d.scalar, T * H);
    if (trace != nullptr) trace->push_back(snapshot());
  }
  m.pos += T;
  return snapshot();
}

std::vector<float> Gemma4DraftForward::logits(const std::vector<float>& residual_rows) {
  Impl& m = *impl_;
  const Gemma4TextConfig& cfg = m.cfg;
  const int H = cfg.hidden_size, V = cfg.vocab_size;
  const int64_t T = static_cast<int64_t>(residual_rows.size()) / H;
  if (T == 0) return {};
  const size_t f = sizeof(float);
  Scratch x(static_cast<size_t>(T * H) * f), out(static_cast<size_t>(T) * static_cast<size_t>(V) * f);
  DGPP_CUDA_OK(cudaMemcpy(x.p, residual_rows.data(), static_cast<size_t>(T * H) * f, cudaMemcpyHostToDevice));
  G4_LAUNCH(g4_rms_norm, T, x.f(), m.final_norm, cfg.rms_norm_eps, x.f(), H, T);
  // The head is the embedding (tied).
  G4_LAUNCH(g4_gemv_bf16, T * V, m.embed, x.f(), out.f(), V, H, T * V);
  std::vector<float> host(static_cast<size_t>(T) * static_cast<size_t>(V));
  DGPP_CUDA_OK(cudaMemcpy(host.data(), out.p, host.size() * f, cudaMemcpyDeviceToHost));
  // The soft-cap on the host: tanh(x / cap) * cap.
  const float cap = cfg.final_logit_softcapping;
  if (cap > 0.0f)
    for (float& o : host) o = tanhf(o / cap) * cap;
  return host;
}

}  // namespace dgpp
