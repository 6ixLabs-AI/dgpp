#include "models/qwen3/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"
#include "models/glm4/attn_reference.hpp"

namespace dgpp::qwen3_ref {
namespace {

namespace fs = std::filesystem;

float rb(float v) { return bf16_bits_to_float(float_to_bf16_bits(v)); }

uint64_t fnv1a64(const uint8_t* data, size_t n, uint64_t h = 0xCBF29CE484222325ull) {
  for (size_t i = 0; i < n; ++i) h = (h ^ data[i]) * 0x100000001B3ull;
  return h;
}

// y[n, rows] = x[n, cols] @ W[rows, cols]^T, double accumulation.
template <typename X, typename W>
std::vector<double> matmul(const std::vector<X>& x, int n, const std::vector<W>& w, int64_t rows,
                           int64_t cols) {
  std::vector<double> y(static_cast<size_t>(n) * static_cast<size_t>(rows));
  for (int t = 0; t < n; ++t) {
    const X* xr = x.data() + static_cast<size_t>(t) * static_cast<size_t>(cols);
    for (int64_t r = 0; r < rows; ++r) {
      const W* wr = w.data() + static_cast<size_t>(r) * static_cast<size_t>(cols);
      double acc = 0.0;
      for (int64_t c = 0; c < cols; ++c) acc += static_cast<double>(xr[c]) * static_cast<double>(wr[c]);
      y[static_cast<size_t>(t) * static_cast<size_t>(rows) + static_cast<size_t>(r)] = acc;
    }
  }
  return y;
}

double silu(double x) { return x / (1.0 + std::exp(-x)); }

// ---- Exact: the numpy reference in double --------------------------------------------

// x * rsqrt(mean(x^2) + eps) * w over the last `dim` elements of every row.
void norm_exact(std::vector<double>& x, size_t rows, size_t dim, const std::vector<float>& w, double eps) {
  for (size_t r = 0; r < rows; ++r) {
    double* v = x.data() + r * dim;
    double ss = 0.0;
    for (size_t d = 0; d < dim; ++d) ss += v[d] * v[d];
    const double rstd = 1.0 / std::sqrt(ss / static_cast<double>(dim) + eps);
    for (size_t d = 0; d < dim; ++d) v[d] = v[d] * rstd * static_cast<double>(w[d]);
  }
}

// Full rotary over all `dim` dims of every head of row t: pairs
// (i, i + dim / 2), angle pos * theta^(-i / (dim / 2)).
void rope_exact(std::vector<double>& x, int n, int heads, int dim, double theta) {
  const int half = dim / 2;
  for (int t = 0; t < n; ++t)
    for (int i = 0; i < half; ++i) {
      const double inv = std::pow(theta, -static_cast<double>(i) / static_cast<double>(half));
      const double ang = static_cast<double>(t) * inv;
      const double c = std::cos(ang), s = std::sin(ang);
      for (int h = 0; h < heads; ++h) {
        double* v = x.data() + (static_cast<size_t>(t) * static_cast<size_t>(heads) + static_cast<size_t>(h)) * static_cast<size_t>(dim);
        const double x1 = v[i], x2 = v[i + half];
        v[i] = x1 * c - x2 * s;
        v[i + half] = x2 * c + x1 * s;
      }
    }
}

std::vector<double> attention_exact(const HostCheckpoint& ck, const std::string& p, const std::vector<double>& h, int T) {
  const Qwen3TextConfig& c = ck.config();
  const int H = c.hidden_size, nh = c.num_attention_heads, nkv = c.num_key_value_heads, d = c.head_dim;
  std::vector<double> q = matmul(h, T, ck.linear(p + "q_proj"), static_cast<int64_t>(nh) * d, H);
  std::vector<double> k = matmul(h, T, ck.linear(p + "k_proj"), static_cast<int64_t>(nkv) * d, H);
  const std::vector<double> v = matmul(h, T, ck.linear(p + "v_proj"), static_cast<int64_t>(nkv) * d, H);
  norm_exact(q, static_cast<size_t>(T) * static_cast<size_t>(nh), static_cast<size_t>(d), ck.dense(p + "q_norm.weight"), c.rms_norm_eps);
  norm_exact(k, static_cast<size_t>(T) * static_cast<size_t>(nkv), static_cast<size_t>(d), ck.dense(p + "k_norm.weight"), c.rms_norm_eps);
  rope_exact(q, T, nh, d, c.rope_theta);
  rope_exact(k, T, nkv, d, c.rope_theta);
  const double scale = std::pow(static_cast<double>(d), -0.5);
  const int rep = nh / nkv;
  std::vector<double> ctx(static_cast<size_t>(T) * static_cast<size_t>(nh) * static_cast<size_t>(d), 0.0);
  std::vector<double> s(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t)
    for (int head = 0; head < nh; ++head) {
      const int g = head / rep;
      const double* qv = q.data() + (static_cast<size_t>(t) * static_cast<size_t>(nh) + static_cast<size_t>(head)) * static_cast<size_t>(d);
      double m = -std::numeric_limits<double>::infinity();
      for (int j = 0; j <= t; ++j) {
        const double* kv = k.data() + (static_cast<size_t>(j) * static_cast<size_t>(nkv) + static_cast<size_t>(g)) * static_cast<size_t>(d);
        double acc = 0.0;
        for (int e = 0; e < d; ++e) acc += qv[e] * kv[e];
        s[static_cast<size_t>(j)] = acc * scale;
        m = std::max(m, s[static_cast<size_t>(j)]);
      }
      double sum = 0.0;
      for (int j = 0; j <= t; ++j) {
        s[static_cast<size_t>(j)] = std::exp(s[static_cast<size_t>(j)] - m);
        sum += s[static_cast<size_t>(j)];
      }
      double* out = ctx.data() + (static_cast<size_t>(t) * static_cast<size_t>(nh) + static_cast<size_t>(head)) * static_cast<size_t>(d);
      for (int j = 0; j <= t; ++j) {
        const double pj = s[static_cast<size_t>(j)] / sum;
        const double* vv = v.data() + (static_cast<size_t>(j) * static_cast<size_t>(nkv) + static_cast<size_t>(g)) * static_cast<size_t>(d);
        for (int e = 0; e < d; ++e) out[e] += pj * vv[e];
      }
    }
  return matmul(ctx, T, ck.linear(p + "o_proj"), H, static_cast<int64_t>(nh) * d);
}

// One SwiGLU MLP's three matrices as fp32 (gate, up [I, H]; down [H, I]).
struct Mlp {
  std::vector<float> gate, up, down;
  int64_t I = 0, H = 0;
};
Mlp load_mlp(const HostCheckpoint& ck, const std::string& p) {
  Mlp w;
  w.gate = ck.linear(p + "gate_proj", &w.I, &w.H);
  w.up = ck.linear(p + "up_proj");
  w.down = ck.linear(p + "down_proj");
  return w;
}
// down(silu(gate(h)) * up(h)) of `n` rows.
std::vector<double> mlp_exact(const Mlp& w, const std::vector<double>& h, int n) {
  const std::vector<double> g = matmul(h, n, w.gate, w.I, w.H);
  const std::vector<double> u = matmul(h, n, w.up, w.I, w.H);
  std::vector<double> act(g.size());
  for (size_t i = 0; i < g.size(); ++i) act[i] = silu(g[i]) * u[i];
  return matmul(act, n, w.down, w.H, w.I);
}

std::vector<double> moe_exact(const HostCheckpoint& ck, const std::string& p, const std::vector<double>& h, int T,
                              std::vector<int32_t>* ids_out) {
  const Qwen3TextConfig& c = ck.config();
  const int H = c.hidden_size, E = c.num_experts, K = c.num_experts_per_tok;
  const std::vector<double> logits = matmul(h, T, ck.dense(p + "gate.weight"), E, H);
  std::vector<double> out(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0);
  std::vector<double> prob(static_cast<size_t>(E));
  std::unordered_map<int, Mlp> experts;  // the selected experts, dequantized on first use
  for (int t = 0; t < T; ++t) {
    const double* lg = logits.data() + static_cast<size_t>(t) * static_cast<size_t>(E);
    const double m = *std::max_element(lg, lg + E);
    double sum = 0.0;
    for (int e = 0; e < E; ++e) {
      prob[static_cast<size_t>(e)] = std::exp(lg[e] - m);
      sum += prob[static_cast<size_t>(e)];
    }
    for (int e = 0; e < E; ++e) prob[static_cast<size_t>(e)] /= sum;
    // The k largest probabilities; a tie goes to the lower expert id.
    std::vector<int> sel;
    std::vector<char> taken(static_cast<size_t>(E), 0);
    for (int r = 0; r < K; ++r) {
      int best = -1;
      for (int e = 0; e < E; ++e)
        if (!taken[static_cast<size_t>(e)] && (best < 0 || prob[static_cast<size_t>(e)] > prob[static_cast<size_t>(best)])) best = e;
      taken[static_cast<size_t>(best)] = 1;
      sel.push_back(best);
    }
    double denom = 0.0;
    for (int e : sel) denom += prob[static_cast<size_t>(e)];
    std::sort(sel.begin(), sel.end());
    const std::vector<double> row(h.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(t) * static_cast<size_t>(H)),
                                  h.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(t + 1) * static_cast<size_t>(H)));
    for (int e : sel) {
      const double w = c.norm_topk_prob ? prob[static_cast<size_t>(e)] / denom : prob[static_cast<size_t>(e)];
      auto it = experts.find(e);
      if (it == experts.end()) it = experts.emplace(e, load_mlp(ck, p + "experts." + std::to_string(e) + ".")).first;
      const std::vector<double> y = mlp_exact(it->second, row, 1);
      for (int d = 0; d < H; ++d) out[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(d)] += w * y[static_cast<size_t>(d)];
      if (ids_out) ids_out->push_back(e);
    }
  }
  return out;
}

// ---- Engine: the draft walk's rounding points -------------------------------------------

// The two-rounding RMSNorm (kernels/glm_norm.hpp): u = bf16(x * rstd),
// y = bf16(w * u), fp32 interior.
std::vector<uint16_t> norm_engine(const std::vector<uint16_t>& x, int rows, int dim, const std::vector<float>& w, float eps) {
  std::vector<uint16_t> y(x.size());
  for (int r = 0; r < rows; ++r) {
    const uint16_t* v = x.data() + static_cast<size_t>(r) * static_cast<size_t>(dim);
    double ss = 0.0;
    for (int d = 0; d < dim; ++d) {
      const double f = static_cast<double>(bf16_bits_to_float(v[d]));
      ss += f * f;
    }
    const float rstd = 1.0f / std::sqrt(static_cast<float>(ss / dim) + eps);
    for (int d = 0; d < dim; ++d) {
      const float u = rb(bf16_bits_to_float(v[d]) * rstd);
      y[static_cast<size_t>(r) * static_cast<size_t>(dim) + static_cast<size_t>(d)] = float_to_bf16_bits(w[static_cast<size_t>(d)] * u);
    }
  }
  return y;
}

// fp32 dots of bf16 rows against a bf16 matrix (the GEMM's F32 output).
std::vector<float> dots_bf16(const std::vector<uint16_t>& x, int n, const std::vector<uint16_t>& w, int64_t rows, int64_t cols) {
  std::vector<float> y(static_cast<size_t>(n) * static_cast<size_t>(rows));
  for (int t = 0; t < n; ++t)
    for (int64_t r = 0; r < rows; ++r) {
      double acc = 0.0;
      for (int64_t c = 0; c < cols; ++c)
        acc += static_cast<double>(bf16_bits_to_float(x[static_cast<size_t>(t) * static_cast<size_t>(cols) + static_cast<size_t>(c)])) *
               static_cast<double>(bf16_bits_to_float(w[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)]));
      y[static_cast<size_t>(t) * static_cast<size_t>(rows) + static_cast<size_t>(r)] = static_cast<float>(acc);
    }
  return y;
}

std::vector<uint16_t> attention_engine(const HostCheckpoint& ck, const std::string& p, const std::vector<uint16_t>& h, int T) {
  const Qwen3TextConfig& c = ck.config();
  const int H = c.hidden_size, nh = c.num_attention_heads, nkv = c.num_key_value_heads, d = c.head_dim;
  const int Q = nh * d, KV = nkv * d;
  const std::vector<float> qd = dots_bf16(h, T, ck.linear_bf16(p + "q_proj"), Q, H);
  const std::vector<float> kd = dots_bf16(h, T, ck.linear_bf16(p + "k_proj"), KV, H);
  const std::vector<float> vd = dots_bf16(h, T, ck.linear_bf16(p + "v_proj"), KV, H);
  // The kernel's table (kernels/qsa.cu qsa_rope_inv_freq): fp32 throughout.
  std::vector<float> inv(static_cast<size_t>(d / 2));
  for (int i = 0; i < d / 2; ++i)
    inv[static_cast<size_t>(i)] = 1.0f / std::pow(static_cast<float>(c.rope_theta), static_cast<float>(2 * i) / static_cast<float>(d));
  const std::vector<float> qn = ck.dense(p + "q_norm.weight"), kn = ck.dense(p + "k_norm.weight");
  std::vector<uint16_t> qnb(qn.size()), knb(kn.size());
  for (size_t i = 0; i < qn.size(); ++i) qnb[i] = float_to_bf16_bits(qn[i]);
  for (size_t i = 0; i < kn.size(); ++i) knb[i] = float_to_bf16_bits(kn[i]);
  std::vector<uint16_t> q(static_cast<size_t>(T) * static_cast<size_t>(Q)), k(static_cast<size_t>(T) * static_cast<size_t>(KV)), v(k.size());
  for (int t = 0; t < T; ++t) {
    for (int head = 0; head < nh; ++head)
      glm4_ref::qkv_finish(qd.data() + static_cast<size_t>(t) * Q + static_cast<size_t>(head) * d, nullptr, qnb.data(), c.rms_norm_eps,
                           inv.data(), d, t, q.data() + static_cast<size_t>(t) * Q + static_cast<size_t>(head) * d, d);
    for (int head = 0; head < nkv; ++head) {
      glm4_ref::qkv_finish(kd.data() + static_cast<size_t>(t) * KV + static_cast<size_t>(head) * d, nullptr, knb.data(), c.rms_norm_eps,
                           inv.data(), d, t, k.data() + static_cast<size_t>(t) * KV + static_cast<size_t>(head) * d, d);
      // v: the Linear's one rounding, no norm, no rotary.
      glm4_ref::qkv_finish(vd.data() + static_cast<size_t>(t) * KV + static_cast<size_t>(head) * d, nullptr, nullptr, c.rms_norm_eps,
                           nullptr, 0, t, v.data() + static_cast<size_t>(t) * KV + static_cast<size_t>(head) * d, d);
    }
  }
  const float scale = static_cast<float>(std::pow(static_cast<double>(d), -0.5));
  std::vector<uint16_t> o(static_cast<size_t>(T) * static_cast<size_t>(Q));
  std::vector<float> ctx;
  for (int t = 0; t < T; ++t) {
    glm4_ref::attention(q.data() + static_cast<size_t>(t) * Q, k.data(), v.data(), t + 1, nh, nkv, d, scale, ctx);
    for (int i = 0; i < Q; ++i) o[static_cast<size_t>(t) * Q + static_cast<size_t>(i)] = float_to_bf16_bits(ctx[static_cast<size_t>(i)]);
  }
  const std::vector<float> y = dots_bf16(o, T, ck.linear_bf16(p + "o_proj"), H, Q);
  std::vector<uint16_t> out(y.size());
  for (size_t i = 0; i < y.size(); ++i) out[i] = float_to_bf16_bits(y[i]);
  return out;
}

void residual_add_engine(std::vector<uint16_t>& x, const std::vector<uint16_t>& y) {
  for (size_t i = 0; i < x.size(); ++i) x[i] = float_to_bf16_bits(bf16_bits_to_float(x[i]) + bf16_bits_to_float(y[i]));
}

std::vector<double> widen(const std::vector<uint16_t>& v) {
  std::vector<double> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<double>(bf16_bits_to_float(v[i]));
  return out;
}

void check_ids(const Qwen3TextConfig& c, const std::vector<int64_t>& ids) {
  if (ids.empty()) throw std::invalid_argument("qwen3_ref::forward: empty token batch");
  for (int64_t id : ids) {
    if (id < 0 || id >= c.vocab_size) throw std::invalid_argument("qwen3_ref::forward: token id out of range");
    if (c.vl() && (id == c.image_token_id || id == c.video_token_id))
      throw std::invalid_argument(
          "qwen3_ref::forward: the prompt carries an image / video placeholder — the vision tower is "
          "not served (text path only)");
  }
}

}  // namespace

void fp4_dequant_f32(const uint8_t* payload, size_t payload_stride, const uint8_t* scales,
                     size_t scale_stride, float global, bool global_divides, int64_t rows,
                     int64_t cols, float* out) {
  for (int64_t n = 0; n < rows; ++n) {
    const uint8_t* pr = payload + static_cast<size_t>(n) * payload_stride;
    const uint8_t* sr = scales + static_cast<size_t>(n) * scale_stride;
    float* o = out + static_cast<size_t>(n) * static_cast<size_t>(cols);
    for (int64_t b = 0; b < cols / kFp4Group; ++b) {
      const float e = fp8_e4m3_bits_to_float(sr[b]);
      const float s = global_divides ? e / global : e * global;
      for (int64_t j = 0; j < kFp4Group / 2; ++j) {
        const uint8_t byte = pr[b * (kFp4Group / 2) + j];
        o[b * kFp4Group + 2 * j] = fp4_e2m1_bits_to_float(static_cast<uint8_t>(byte & 0xFu)) * s;
        o[b * kFp4Group + 2 * j + 1] = fp4_e2m1_bits_to_float(static_cast<uint8_t>(byte >> 4)) * s;
      }
    }
  }
}

void fp4_dequant_bf16(const uint8_t* payload, size_t payload_stride, const uint8_t* scales,
                      size_t scale_stride, float global, bool global_divides, int64_t rows,
                      int64_t cols, uint16_t* out) {
  for (int64_t n = 0; n < rows; ++n) {
    const uint8_t* pr = payload + static_cast<size_t>(n) * payload_stride;
    const uint8_t* sr = scales + static_cast<size_t>(n) * scale_stride;
    uint16_t* o = out + static_cast<size_t>(n) * static_cast<size_t>(cols);
    for (int64_t b = 0; b < cols / kFp4Group; ++b) {
      const float e = fp8_e4m3_bits_to_float(sr[b]);
      uint16_t value[16];
      for (int c = 0; c < 16; ++c) {
        const float code = fp4_e2m1_bits_to_float(static_cast<uint8_t>(c));
        // qwen35_fp4_packed_dequant_bf16: code * (e / g); qwen3next_fp4_dequant_bf16: (code * e) * g.
        value[c] = float_to_bf16_bits(global_divides ? code * (e / global) : code * e * global);
      }
      for (int64_t j = 0; j < kFp4Group / 2; ++j) {
        const uint8_t byte = pr[b * (kFp4Group / 2) + j];
        o[b * kFp4Group + 2 * j] = value[byte & 0xFu];
        o[b * kFp4Group + 2 * j + 1] = value[byte >> 4];
      }
    }
  }
}

HostCheckpoint::HostCheckpoint(const std::string& dir, int layers) {
  cfg_ = Qwen3TextConfig::from_json_file((fs::path(dir) / "config.json").string());
  std::vector<std::string> paths;
  for (const auto& entry : fs::directory_iterator(dir))
    if (entry.is_regular_file() && entry.path().extension() == ".safetensors") paths.push_back(entry.path().string());
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) throw std::runtime_error("qwen3 host checkpoint: no .safetensors under " + dir);
  Qwen3PresentMap present;
  for (const std::string& path : paths) {
    shards_.push_back(SafetensorsFile::open(path));
    shards_.back()->for_each([&](const TensorInfo& t) {
      if (!tensors_.emplace(t.name, &t).second)
        throw std::runtime_error("qwen3 host checkpoint: tensor '" + t.name + "' appears in two shards");
      present.emplace(t.name, Qwen3TensorDesc{t.dtype, t.shape});
    });
  }
  qwen3_apply_header_naming(cfg_, present);
  if (layers > 0 && layers < cfg_.num_hidden_layers) cfg_.num_hidden_layers = layers;
  report_ = qwen3_validate_binding(cfg_, present);
  if (!report_.ok()) {
    std::string msg = "qwen3 host checkpoint: the header does not match the binding table (" +
                      std::to_string(report_.missing) + " missing, " + std::to_string(report_.dtype_mismatch) +
                      " dtype, " + std::to_string(report_.shape_mismatch) + " shape, " +
                      std::to_string(report_.unexpected) + " unexpected)";
    for (size_t i = 0; i < report_.errors.size() && i < 8; ++i) msg += "\n  " + report_.errors[i];
    throw std::runtime_error(msg);
  }
}

const TensorInfo& HostCheckpoint::tensor(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("qwen3 host checkpoint: tensor not found: " + name);
  return *it->second;
}

uint64_t HostCheckpoint::content_digest() const {
  uint64_t total = 0;
  for (const auto& [name, t] : tensors_) {
    std::string head = name + "|" + std::string(dtype_name(t->dtype)) + "|";
    for (size_t i = 0; i < t->shape.size(); ++i) head += (i ? "," : "") + std::to_string(t->shape[i]);
    head += "|";
    const uint64_t h = fnv1a64(reinterpret_cast<const uint8_t*>(head.data()), head.size());
    total += fnv1a64(static_cast<const uint8_t*>(t->data), t->nbytes(), h);
  }
  return total;
}

std::vector<float> HostCheckpoint::dense(const std::string& name) const {
  const TensorInfo& t = tensor(name);
  std::vector<float> out(t.numel());
  if (t.dtype == DType::BF16) {
    const uint16_t* p = static_cast<const uint16_t*>(t.data);
    for (size_t i = 0; i < out.size(); ++i) {
      uint16_t bits;
      std::memcpy(&bits, p + i, 2);
      out[i] = bf16_bits_to_float(bits);
    }
  } else if (t.dtype == DType::F32) {
    std::memcpy(out.data(), t.data, out.size() * 4);
  } else {
    throw std::runtime_error("qwen3 host checkpoint: '" + name + "' is neither BF16 nor F32");
  }
  return out;
}

bool HostCheckpoint::is_fp4(const std::string& base) const { return has(base + ".weight_scale"); }

std::vector<float> HostCheckpoint::linear(const std::string& base, int64_t* rows, int64_t* cols) const {
  if (!is_fp4(base)) {
    const TensorInfo& t = tensor(base + ".weight");
    if (t.shape.size() != 2) throw std::runtime_error("qwen3 host checkpoint: '" + base + ".weight' is not a matrix");
    if (rows) *rows = t.shape[0];
    if (cols) *cols = t.shape[1];
    return dense(base + ".weight");
  }
  const TensorInfo& tp = tensor(qwen3_fp4_payload_name(cfg_, base));
  const TensorInfo& ts = tensor(base + ".weight_scale");
  const TensorInfo& tg = tensor(qwen3_fp4_global_name(cfg_, base));
  const int64_t N = tp.shape[0], K = tp.shape[1] * 2;
  float g;
  std::memcpy(&g, tg.data, 4);
  std::vector<float> out(static_cast<size_t>(N) * static_cast<size_t>(K));
  fp4_dequant_f32(static_cast<const uint8_t*>(tp.data), static_cast<size_t>(K / 2), static_cast<const uint8_t*>(ts.data),
                  static_cast<size_t>(K / kFp4Group), g, cfg_.fp4_packed(), N, K, out.data());
  if (rows) *rows = N;
  if (cols) *cols = K;
  return out;
}

std::vector<uint16_t> HostCheckpoint::linear_bf16(const std::string& base, int64_t* rows, int64_t* cols) const {
  if (!is_fp4(base)) {
    const TensorInfo& t = tensor(base + ".weight");
    if (t.dtype != DType::BF16 || t.shape.size() != 2)
      throw std::runtime_error("qwen3 host checkpoint: '" + base + ".weight' is not a BF16 matrix");
    std::vector<uint16_t> out(t.numel());
    std::memcpy(out.data(), t.data, out.size() * 2);
    if (rows) *rows = t.shape[0];
    if (cols) *cols = t.shape[1];
    return out;
  }
  const TensorInfo& tp = tensor(qwen3_fp4_payload_name(cfg_, base));
  const TensorInfo& ts = tensor(base + ".weight_scale");
  const TensorInfo& tg = tensor(qwen3_fp4_global_name(cfg_, base));
  const int64_t N = tp.shape[0], K = tp.shape[1] * 2;
  float g;
  std::memcpy(&g, tg.data, 4);
  std::vector<uint16_t> out(static_cast<size_t>(N) * static_cast<size_t>(K));
  fp4_dequant_bf16(static_cast<const uint8_t*>(tp.data), static_cast<size_t>(K / 2), static_cast<const uint8_t*>(ts.data),
                   static_cast<size_t>(K / kFp4Group), g, cfg_.fp4_packed(), N, K, out.data());
  if (rows) *rows = N;
  if (cols) *cols = K;
  return out;
}

Qwen3MoeHostWeights HostCheckpoint::experts(int layer) const {
  if (!cfg_.moe()) throw std::logic_error("qwen3 host checkpoint: the dense dialect has no experts");
  Qwen3MoeHostWeights w;
  w.hidden = cfg_.hidden_size;
  w.inter = cfg_.moe_intermediate_size;
  w.n_experts = cfg_.num_experts;
  w.allocate();
  const std::string p = qwen3_layer_prefix(cfg_, layer) + "mlp.";
  {
    const TensorInfo& t = tensor(p + "gate.weight");
    std::memcpy(w.router.data(), t.data, w.router.size() * 2);
  }
  static const char* const kPart[3] = {"gate_proj", "up_proj", "down_proj"};
  for (int e = 0; e < cfg_.num_experts; ++e)
    for (int m = 0; m < 3; ++m) {
      const std::string base = p + "experts." + std::to_string(e) + "." + kPart[m];
      const int index = e * 3 + m;
      const TensorInfo& tp = tensor(qwen3_fp4_payload_name(cfg_, base));
      const TensorInfo& ts = tensor(base + ".weight_scale");
      const TensorInfo& tg = tensor(qwen3_fp4_global_name(cfg_, base));
      std::memcpy(w.payloads.data() + w.payload_offset(index), tp.data, w.payload_bytes(m));
      std::memcpy(w.scales.data() + w.scale_offset(index), ts.data, w.scale_bytes(m));
      float g;
      std::memcpy(&g, tg.data, 4);
      // The kernels' divisor: compressed-tensors' global as shipped,
      // modelopt's multiplier as its reciprocal (the loader's rule).
      w.globals[static_cast<size_t>(index)] = cfg_.fp4_packed() ? g : 1.0f / g;
    }
  return w;
}

std::string HostCheckpoint::head_name() const {
  return cfg_.tie_word_embeddings ? qwen3_model_prefix(cfg_) + "embed_tokens.weight" : "lm_head.weight";
}

std::vector<double> forward(const HostCheckpoint& ck, const std::vector<int64_t>& ids, Arithmetic arithmetic, Trace* trace) {
  const Qwen3TextConfig& c = ck.config();
  check_ids(c, ids);
  const int T = static_cast<int>(ids.size());
  const int H = c.hidden_size;
  const std::string P = qwen3_model_prefix(c);
  const TensorInfo& embed = ck.tensor(P + "embed_tokens.weight");
  if (trace) *trace = Trace{};

  if (arithmetic == Arithmetic::Exact) {
    std::vector<double> x(static_cast<size_t>(T) * static_cast<size_t>(H));
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < H; ++d) {
        uint16_t bits;
        std::memcpy(&bits, static_cast<const uint16_t*>(embed.data) + static_cast<size_t>(ids[static_cast<size_t>(t)]) * static_cast<size_t>(H) + static_cast<size_t>(d), 2);
        x[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(d)] = static_cast<double>(bf16_bits_to_float(bits));
      }
    if (trace) trace->states.push_back(x);
    for (int L = 0; L < c.num_hidden_layers; ++L) {
      const std::string p = qwen3_layer_prefix(c, L);
      std::vector<double> h = x;
      norm_exact(h, static_cast<size_t>(T), static_cast<size_t>(H), ck.dense(p + "input_layernorm.weight"), c.rms_norm_eps);
      const std::vector<double> y = attention_exact(ck, p + "self_attn.", h, T);
      for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
      h = x;
      norm_exact(h, static_cast<size_t>(T), static_cast<size_t>(H), ck.dense(p + "post_attention_layernorm.weight"), c.rms_norm_eps);
      std::vector<int32_t> route;
      const std::vector<double> m = c.moe() ? moe_exact(ck, p + "mlp.", h, T, trace ? &route : nullptr)
                                            : mlp_exact(load_mlp(ck, p + "mlp."), h, T);
      for (size_t i = 0; i < x.size(); ++i) x[i] += m[i];
      if (trace) {
        trace->states.push_back(x);
        if (c.moe()) trace->router_ids.push_back(std::move(route));
      }
    }
    norm_exact(x, static_cast<size_t>(T), static_cast<size_t>(H), ck.dense(P + "norm.weight"), c.rms_norm_eps);
    if (trace) trace->final_hidden = x;
    return matmul(x, T, ck.dense(ck.head_name()), c.vocab_size, H);
  }

  // ---- Engine ---------------------------------------------------------------------------
  if (!c.moe()) throw std::invalid_argument("qwen3_ref::forward: the dense dialect has no engine path (Exact only)");
  const GlmMoeConfig moe_cfg = c.moe_config(c.moe_intermediate_size);
  std::vector<uint16_t> x(static_cast<size_t>(T) * static_cast<size_t>(H));
  for (int t = 0; t < T; ++t)
    std::memcpy(x.data() + static_cast<size_t>(t) * static_cast<size_t>(H),
                static_cast<const uint16_t*>(embed.data) + static_cast<size_t>(ids[static_cast<size_t>(t)]) * static_cast<size_t>(H), static_cast<size_t>(H) * 2);
  if (trace) trace->states.push_back(widen(x));
  for (int L = 0; L < c.num_hidden_layers; ++L) {
    const std::string p = qwen3_layer_prefix(c, L);
    std::vector<uint16_t> h = norm_engine(x, T, H, ck.dense(p + "input_layernorm.weight"), c.rms_norm_eps);
    residual_add_engine(x, attention_engine(ck, p + "self_attn.", h, T));
    h = norm_engine(x, T, H, ck.dense(p + "post_attention_layernorm.weight"), c.rms_norm_eps);
    std::vector<uint16_t> m;
    GlmMoeRouterRef route;
    qwen3_moe_ref_forward(h.data(), ck.experts(L), moe_cfg, T, m, &route);
    residual_add_engine(x, m);
    if (trace) {
      trace->states.push_back(widen(x));
      trace->router_ids.push_back(route.ids);
    }
  }
  const std::vector<uint16_t> hn = norm_engine(x, T, H, ck.dense(P + "norm.weight"), c.rms_norm_eps);
  if (trace) trace->final_hidden = widen(hn);
  int64_t V = 0, HH = 0;
  const std::vector<uint16_t> head = ck.linear_bf16(ck.head_name().substr(0, ck.head_name().size() - std::strlen(".weight")), &V, &HH);
  const std::vector<float> lg = dots_bf16(hn, T, head, V, HH);
  return std::vector<double>(lg.begin(), lg.end());
}

std::vector<double> teacher_forced_logprobs(const std::vector<double>& logits, const std::vector<int64_t>& ids, int vocab) {
  std::vector<double> out;
  for (size_t i = 1; i < ids.size(); ++i) {
    const double* row = logits.data() + (i - 1) * static_cast<size_t>(vocab);
    const double m = *std::max_element(row, row + vocab);
    double sum = 0.0;
    for (int v = 0; v < vocab; ++v) sum += std::exp(row[v] - m);
    out.push_back(row[ids[i]] - m - std::log(sum));
  }
  return out;
}

}  // namespace dgpp::qwen3_ref
