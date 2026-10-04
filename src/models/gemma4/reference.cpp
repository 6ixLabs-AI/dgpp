#include "models/gemma4/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>

#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"
#include "loaders/safetensors.hpp"
#include "models/gemma4/binding.hpp"

namespace dgpp::gemma4_ref {

// ---- operators ---------------------------------------------------------------

float gelu_tanh(float x) {
  // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
  const float k = 0.7978845608028654f;
  return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

float softcap(float x, float cap) {
  if (!(cap > 0.0f)) return x;
  return std::tanh(x / cap) * cap;
}

void rms_norm(const float* x, const float* w, int n, float eps, float* out) {
  double ss = 0.0;
  for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * static_cast<double>(x[i]);
  // The reference: mean_squared = x.pow(2).mean(-1) + eps; x * pow(mean_squared, -0.5).
  const float ms = static_cast<float>(ss / static_cast<double>(n)) + eps;
  const float r = std::pow(ms, -0.5f);
  for (int i = 0; i < n; ++i) out[i] = w ? x[i] * r * w[i] : x[i] * r;
}

std::vector<double> rope_inv_freq(int head_dim, int pairs, double theta) {
  std::vector<double> inv(static_cast<size_t>(pairs));
  for (int i = 0; i < pairs; ++i)
    inv[static_cast<size_t>(i)] = std::pow(theta, -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
  return inv;
}

void rope_apply(float* x, int head_dim, const std::vector<double>& inv_freq, int64_t pos) {
  // transformers apply_rotary_pos_emb: x * cos + rotate_half(x) * sin with
  // cos / sin = cat(freqs, freqs) — dim d < half pairs with d + half.
  const int half = head_dim / 2;
  for (size_t i = 0; i < inv_freq.size(); ++i) {
    const double ang = static_cast<double>(pos) * inv_freq[i];
    const float c = static_cast<float>(std::cos(ang)), s = static_cast<float>(std::sin(ang));
    const float a = x[i], b = x[i + static_cast<size_t>(half)];
    x[i] = a * c - b * s;
    x[i + static_cast<size_t>(half)] = b * c + a * s;
  }
}

int64_t first_visible(int64_t pos, int window) {
  if (window <= 0) return 0;
  return std::max<int64_t>(0, pos - window + 1);
}

void attention(const float* q, const float* k_rows, const float* v_rows, int64_t n, int heads,
               int kv_heads, int hd, float* ctx) {
  const int groups = heads / kv_heads;
  const int64_t width = static_cast<int64_t>(kv_heads) * hd;
  std::vector<float> s(static_cast<size_t>(n));
  std::vector<double> acc(static_cast<size_t>(hd));
  for (int h = 0; h < heads; ++h) {
    const int kvh = h / groups;
    float m = -INFINITY;
    for (int64_t t = 0; t < n; ++t) {
      double dot = 0.0;
      const float* kr = k_rows + t * width + static_cast<int64_t>(kvh) * hd;
      for (int d = 0; d < hd; ++d) dot += static_cast<double>(q[h * hd + d]) * static_cast<double>(kr[d]);
      s[static_cast<size_t>(t)] = static_cast<float>(dot);  // scaling 1.0: no 1/sqrt(d)
      m = std::max(m, s[static_cast<size_t>(t)]);
    }
    double denom = 0.0;
    for (int64_t t = 0; t < n; ++t) {
      s[static_cast<size_t>(t)] = std::exp(s[static_cast<size_t>(t)] - m);
      denom += static_cast<double>(s[static_cast<size_t>(t)]);
    }
    std::fill(acc.begin(), acc.end(), 0.0);
    for (int64_t t = 0; t < n; ++t) {
      const double p = static_cast<double>(s[static_cast<size_t>(t)]) / denom;
      const float* vr = v_rows + t * width + static_cast<int64_t>(kvh) * hd;
      for (int d = 0; d < hd; ++d) acc[static_cast<size_t>(d)] += p * static_cast<double>(vr[d]);
    }
    for (int d = 0; d < hd; ++d) ctx[h * hd + d] = static_cast<float>(acc[static_cast<size_t>(d)]);
  }
}

void route(const float* x, const float* proj, const float* scale, const float* per_expert_scale,
           float input_scale, int H, int E, int top_k, float eps, int* ids, float* weights) {
  // Gemma4TextRouter: norm (no weight), * scale * hidden^-0.5, proj, fp32 softmax, top-k,
  // renormalize, * per_expert_scale.
  std::vector<float> r(static_cast<size_t>(H));
  rms_norm(x, nullptr, H, eps, r.data());
  for (int d = 0; d < H; ++d) r[static_cast<size_t>(d)] = r[static_cast<size_t>(d)] * scale[d] * input_scale;
  std::vector<float> p(static_cast<size_t>(E));
  float m = -INFINITY;
  for (int e = 0; e < E; ++e) {
    double dot = 0.0;
    for (int d = 0; d < H; ++d) dot += static_cast<double>(proj[static_cast<int64_t>(e) * H + d]) * static_cast<double>(r[static_cast<size_t>(d)]);
    p[static_cast<size_t>(e)] = static_cast<float>(dot);
    m = std::max(m, p[static_cast<size_t>(e)]);
  }
  double denom = 0.0;
  for (int e = 0; e < E; ++e) {
    p[static_cast<size_t>(e)] = std::exp(p[static_cast<size_t>(e)] - m);
    denom += static_cast<double>(p[static_cast<size_t>(e)]);
  }
  for (int e = 0; e < E; ++e) p[static_cast<size_t>(e)] = static_cast<float>(static_cast<double>(p[static_cast<size_t>(e)]) / denom);
  // The k largest, best first; a tie goes to the lower id.
  std::vector<char> taken(static_cast<size_t>(E), 0);
  double sum = 0.0;
  for (int k = 0; k < top_k; ++k) {
    int best = -1;
    for (int e = 0; e < E; ++e)
      if (!taken[static_cast<size_t>(e)] && (best < 0 || p[static_cast<size_t>(e)] > p[static_cast<size_t>(best)])) best = e;
    taken[static_cast<size_t>(best)] = 1;
    ids[k] = best;
    weights[k] = p[static_cast<size_t>(best)];
    sum += static_cast<double>(weights[k]);
  }
  for (int k = 0; k < top_k; ++k)
    weights[k] = static_cast<float>(static_cast<double>(weights[k]) / sum) * per_expert_scale[ids[k]];
}

void decode_nvfp4(const uint8_t* payload, const uint8_t* scales, float ws2, int64_t rows,
                  int64_t cols, float* out) {
  const int64_t half = cols / 2, groups = cols / kLatentFp4Block;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      const uint8_t byte = payload[r * half + c / 2];
      const uint8_t code = (c & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0xFu);
      const float s = fp8_e4m3_bits_to_float(scales[r * groups + c / kLatentFp4Block]) * ws2;
      out[r * cols + c] = fp4_e2m1_bits_to_float(code) * s;
    }
  }
}

// ---- weights -------------------------------------------------------------------

namespace {

// Every shard of a checkpoint directory, tensors found by name.
class Shards {
 public:
  explicit Shards(const std::string& dir) {
    namespace fs = std::filesystem;
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir))
      if (entry.path().extension() == ".safetensors") paths.push_back(entry.path());
    if (paths.empty()) throw std::runtime_error("gemma4 reference: no .safetensors shards in " + dir);
    std::sort(paths.begin(), paths.end());
    for (const auto& p : paths) files_.push_back(SafetensorsFile::open(p.string()));
  }
  const TensorInfo& at(const std::string& name) const {
    for (const auto& f : files_)
      if (const TensorInfo* t = f->find(name)) return *t;
    throw std::runtime_error("gemma4 reference: tensor not found: " + name);
  }
  bool has(const std::string& name) const {
    for (const auto& f : files_)
      if (f->find(name)) return true;
    return false;
  }

 private:
  std::vector<std::unique_ptr<SafetensorsFile>> files_;
};

const TensorInfo& expect(const Shards& s, const std::string& name, DType dtype, const std::vector<int64_t>& shape) {
  const TensorInfo& t = s.at(name);
  if (t.dtype != dtype)
    throw std::runtime_error(std::format("gemma4 reference: '{}' dtype {} != expected {}", name, dtype_name(t.dtype),
                                         dtype_name(dtype)));
  if (t.shape != shape) throw std::runtime_error("gemma4 reference: '" + name + "' has another shape than the table");
  return t;
}

std::vector<float> bf16_vector(const Shards& s, const std::string& name, const std::vector<int64_t>& shape) {
  const TensorInfo& t = expect(s, name, DType::BF16, shape);
  const uint16_t* p = static_cast<const uint16_t*>(t.data);
  std::vector<float> out(t.numel());
  for (size_t i = 0; i < out.size(); ++i) out[i] = bf16_bits_to_float(p[i]);
  return out;
}

// One projection, in whichever format the recipe gives it.
Matrix matrix(const Shards& s, const std::string& base, int64_t rows, int64_t cols, bool nvfp4) {
  Matrix m;
  m.rows = rows;
  m.cols = cols;
  if (!nvfp4) {
    m.w = bf16_vector(s, base + ".weight", {rows, cols});
    return m;
  }
  const TensorInfo& payload = expect(s, base + ".weight", DType::U8, {rows, cols / 2});
  const TensorInfo& scales = expect(s, base + ".weight_scale", DType::F8_E4M3, {rows, cols / kLatentFp4Block});
  const TensorInfo& global = expect(s, base + ".weight_scale_2", DType::F32, {});
  float ws2;
  std::memcpy(&ws2, global.data, sizeof ws2);
  if (!(ws2 > 0.0f) || !std::isfinite(ws2))
    throw std::runtime_error("gemma4 reference: '" + base + ".weight_scale_2' is not a positive finite number");
  m.w.resize(static_cast<size_t>(rows * cols));
  decode_nvfp4(static_cast<const uint8_t*>(payload.data), static_cast<const uint8_t*>(scales.data), ws2, rows, cols,
               m.w.data());
  return m;
}

// y[rows] = W [rows, cols] @ x[cols], accumulated in double.
void matvec(const Matrix& m, const float* x, float* y) {
  for (int64_t r = 0; r < m.rows; ++r) {
    const float* row = m.w.data() + r * m.cols;
    double acc = 0.0;
    for (int64_t c = 0; c < m.cols; ++c) acc += static_cast<double>(row[c]) * static_cast<double>(x[c]);
    y[r] = static_cast<float>(acc);
  }
}

std::string tap_name(int layer, const char* what) { return std::format("L{:02d}_{}", layer, what); }

}  // namespace

Weights load_weights(const Gemma4TextConfig& cfg, const std::string& checkpoint_dir, int layers) {
  const Shards s(checkpoint_dir);
  const int n_layers = layers < 0 ? cfg.num_hidden_layers : std::min(layers, cfg.num_hidden_layers);
  const int64_t H = cfg.hidden_size, I = cfg.intermediate_size;
  Weights w;
  w.embed.rows = cfg.vocab_size;
  w.embed.cols = H;
  w.embed.w = bf16_vector(s, "model.language_model.embed_tokens.weight", {cfg.vocab_size, H});
  w.final_norm = bf16_vector(s, "model.language_model.norm.weight", {H});
  w.layers.resize(static_cast<size_t>(n_layers));
  const bool attn4 = cfg.attention_nvfp4();
  for (int l = 0; l < n_layers; ++l) {
    const std::string p = gemma4_layer_prefix(l);
    LayerWeights& lw = w.layers[static_cast<size_t>(l)];
    lw.input_norm = bf16_vector(s, p + "input_layernorm.weight", {H});
    lw.post_attn_norm = bf16_vector(s, p + "post_attention_layernorm.weight", {H});
    lw.pre_ffn_norm = bf16_vector(s, p + "pre_feedforward_layernorm.weight", {H});
    lw.post_ffn_norm = bf16_vector(s, p + "post_feedforward_layernorm.weight", {H});
    lw.layer_scalar = bf16_vector(s, p + "layer_scalar", {1})[0];
    const int64_t hd = cfg.head_dim_of(l);
    lw.q_norm = bf16_vector(s, p + "self_attn.q_norm.weight", {hd});
    lw.k_norm = bf16_vector(s, p + "self_attn.k_norm.weight", {hd});
    lw.q_proj = matrix(s, p + "self_attn.q_proj", cfg.q_rows(l), H, attn4);
    lw.k_proj = matrix(s, p + "self_attn.k_proj", cfg.kv_rows(l), H, attn4);
    if (cfg.has_v_proj(l)) lw.v_proj = matrix(s, p + "self_attn.v_proj", cfg.kv_rows(l), H, attn4);
    lw.o_proj = matrix(s, p + "self_attn.o_proj", H, cfg.q_rows(l), attn4);
    lw.gate_proj = matrix(s, p + "mlp.gate_proj", I, H, true);
    lw.up_proj = matrix(s, p + "mlp.up_proj", I, H, true);
    lw.down_proj = matrix(s, p + "mlp.down_proj", H, I, true);
    if (cfg.enable_moe_block) {
      const int64_t E = cfg.num_experts, Im = cfg.moe_intermediate_size;
      lw.post_ffn_norm_1 = bf16_vector(s, p + "post_feedforward_layernorm_1.weight", {H});
      lw.post_ffn_norm_2 = bf16_vector(s, p + "post_feedforward_layernorm_2.weight", {H});
      lw.pre_ffn_norm_2 = bf16_vector(s, p + "pre_feedforward_layernorm_2.weight", {H});
      lw.router_proj = matrix(s, p + "router.proj", E, H, false);  // the router is BF16 under every recipe
      lw.router_scale = bf16_vector(s, p + "router.scale", {H});
      lw.per_expert_scale = bf16_vector(s, p + "router.per_expert_scale", {E});
      lw.experts.resize(static_cast<size_t>(E));
      for (int64_t e = 0; e < E; ++e) {
        const std::string ep = p + "moe.experts." + std::to_string(e) + ".";
        LayerWeights::Expert& x = lw.experts[static_cast<size_t>(e)];
        x.gate_proj = matrix(s, ep + "gate_proj", Im, H, true);
        x.up_proj = matrix(s, ep + "up_proj", Im, H, true);
        x.down_proj = matrix(s, ep + "down_proj", H, Im, true);
      }
    }
  }
  return w;
}

// ---- the model -------------------------------------------------------------------

Model::Model(const Gemma4TextConfig& cfg, Weights weights) : cfg_(cfg), w_(std::move(weights)) {
  if (w_.embed.rows != cfg_.vocab_size || w_.embed.cols != cfg_.hidden_size)
    throw std::invalid_argument("gemma4 reference: the embedding does not match the config");
  if (static_cast<int>(w_.layers.size()) > cfg_.num_hidden_layers)
    throw std::invalid_argument("gemma4 reference: more layers than the config declares");
}

void Model::attention_layer(int l, const std::vector<float>& h, int64_t T, State& st, std::vector<float>& out,
                            const Tap& tap) const {
  const LayerWeights& lw = w_.layers[static_cast<size_t>(l)];
  const int H = cfg_.hidden_size, heads = cfg_.num_attention_heads;
  const int hd = cfg_.head_dim_of(l), kvh = cfg_.kv_heads_of(l);
  const int window = cfg_.is_sliding_layer(l) ? cfg_.sliding_window : 0;
  const std::vector<double> inv = rope_inv_freq(hd, cfg_.rotary_pairs_of(l), cfg_.rope_theta_of(l));
  const int64_t qw = static_cast<int64_t>(heads) * hd, kw = static_cast<int64_t>(kvh) * hd;
  LayerCache& cache = st.layers[static_cast<size_t>(l)];
  std::vector<float> q(static_cast<size_t>(T * qw)), k(static_cast<size_t>(T * kw)), v(static_cast<size_t>(T * kw));
  for (int64_t t = 0; t < T; ++t) {
    const float* x = h.data() + t * H;
    const int64_t pos = st.pos + t;
    float* qt = q.data() + t * qw;
    float* kt = k.data() + t * kw;
    float* vt = v.data() + t * kw;
    matvec(lw.q_proj, x, qt);
    matvec(lw.k_proj, x, kt);
    // The value: its own projection, or (a full layer) the k_proj output as
    // it stands — before k_norm and before the rotation.
    if (!lw.v_proj.empty())
      matvec(lw.v_proj, x, vt);
    else
      std::copy(kt, kt + kw, vt);
    for (int hh = 0; hh < heads; ++hh) {
      rms_norm(qt + hh * hd, lw.q_norm.data(), hd, cfg_.rms_norm_eps, qt + hh * hd);
      rope_apply(qt + hh * hd, hd, inv, pos);
    }
    for (int g = 0; g < kvh; ++g) {
      rms_norm(kt + g * hd, lw.k_norm.data(), hd, cfg_.rms_norm_eps, kt + g * hd);
      rope_apply(kt + g * hd, hd, inv, pos);
      rms_norm(vt + g * hd, nullptr, hd, cfg_.rms_norm_eps, vt + g * hd);
    }
  }
  if (tap) {
    tap(tap_name(l, "attn_q"), q);
    tap(tap_name(l, "attn_k"), k);
    tap(tap_name(l, "attn_v"), v);
  }
  // The cache keeps every row; a sliding layer reads the window's tail of it.
  cache.k.insert(cache.k.end(), k.begin(), k.end());
  cache.v.insert(cache.v.end(), v.begin(), v.end());
  cache.rows += T;
  std::vector<float> ctx(static_cast<size_t>(qw));
  out.assign(static_cast<size_t>(T * H), 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    const int64_t pos = st.pos + t;
    const int64_t first = first_visible(pos, window);
    attention(q.data() + t * qw, cache.k.data() + first * kw, cache.v.data() + first * kw, pos - first + 1, heads, kvh,
              hd, ctx.data());
    matvec(lw.o_proj, ctx.data(), out.data() + t * H);
  }
}

std::vector<float> Model::forward(const std::vector<int64_t>& ids, State& st, const Tap& tap) const {
  const int H = cfg_.hidden_size, I = cfg_.intermediate_size;
  const int64_t T = static_cast<int64_t>(ids.size());
  if (st.layers.size() != w_.layers.size()) {
    if (st.pos != 0) throw std::invalid_argument("gemma4 reference: a state from another model");
    st.layers.assign(w_.layers.size(), LayerCache{});
  }
  const float scale = cfg_.embed_scale();
  std::vector<float> x(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    const int64_t id = ids[static_cast<size_t>(t)];
    if (id < 0 || id >= cfg_.vocab_size) throw std::out_of_range("gemma4 reference: token id outside the vocabulary");
    const float* row = w_.embed.w.data() + id * H;
    for (int d = 0; d < H; ++d) x[static_cast<size_t>(t * H + d)] = row[d] * scale;
  }
  if (tap) tap("h_00", x);
  std::vector<float> h(static_cast<size_t>(T * H)), a, m(static_cast<size_t>(T * H));
  std::vector<float> gate(static_cast<size_t>(I)), up(static_cast<size_t>(I)), normed(static_cast<size_t>(H));
  for (int l = 0; l < layers(); ++l) {
    const LayerWeights& lw = w_.layers[static_cast<size_t>(l)];
    // x = x + rms(attn(rms(x, input_layernorm)), post_attention_layernorm)
    for (int64_t t = 0; t < T; ++t)
      rms_norm(x.data() + t * H, lw.input_norm.data(), H, cfg_.rms_norm_eps, h.data() + t * H);
    attention_layer(l, h, T, st, a, tap);
    if (tap) tap(tap_name(l, "attn_out"), a);
    for (int64_t t = 0; t < T; ++t) {
      rms_norm(a.data() + t * H, lw.post_attn_norm.data(), H, cfg_.rms_norm_eps, normed.data());
      for (int d = 0; d < H; ++d) x[static_cast<size_t>(t * H + d)] += normed[static_cast<size_t>(d)];
    }
    // x = x + rms(mlp(rms(x, pre_feedforward_layernorm)), post_feedforward_layernorm)
    for (int64_t t = 0; t < T; ++t) {
      rms_norm(x.data() + t * H, lw.pre_ffn_norm.data(), H, cfg_.rms_norm_eps, h.data() + t * H);
      matvec(lw.gate_proj, h.data() + t * H, gate.data());
      matvec(lw.up_proj, h.data() + t * H, up.data());
      for (int i = 0; i < I; ++i) gate[static_cast<size_t>(i)] = gelu_tanh(gate[static_cast<size_t>(i)]) * up[static_cast<size_t>(i)];
      matvec(lw.down_proj, gate.data(), m.data() + t * H);
    }
    if (tap) tap(tap_name(l, "mlp_out"), m);
    if (cfg_.enable_moe_block) {
      // m = rms(mlp, post_feedforward_layernorm_1) + rms(moe(x), post_feedforward_layernorm_2):
      // the router reads the residual stream itself, the experts their own norm of it.
      const int E = cfg_.num_experts, K = cfg_.top_k_experts, Im = cfg_.moe_intermediate_size;
      std::vector<float> moe(static_cast<size_t>(T * H), 0.0f), router_ids, router_w;
      std::vector<int> ids(static_cast<size_t>(K));
      std::vector<float> w(static_cast<size_t>(K)), h2(static_cast<size_t>(H)), eg(static_cast<size_t>(Im)),
          eu(static_cast<size_t>(Im)), y(static_cast<size_t>(H));
      for (int64_t t = 0; t < T; ++t) {
        const float* xt = x.data() + t * H;
        route(xt, lw.router_proj.w.data(), lw.router_scale.data(), lw.per_expert_scale.data(), cfg_.router_input_scale(), H,
              E, K, cfg_.rms_norm_eps, ids.data(), w.data());
        rms_norm(xt, lw.pre_ffn_norm_2.data(), H, cfg_.rms_norm_eps, h2.data());
        for (int k = 0; k < K; ++k) {
          const LayerWeights::Expert& ex = lw.experts[static_cast<size_t>(ids[static_cast<size_t>(k)])];
          matvec(ex.gate_proj, h2.data(), eg.data());
          matvec(ex.up_proj, h2.data(), eu.data());
          for (int i = 0; i < Im; ++i) eg[static_cast<size_t>(i)] = gelu_tanh(eg[static_cast<size_t>(i)]) * eu[static_cast<size_t>(i)];
          matvec(ex.down_proj, eg.data(), y.data());
          for (int d = 0; d < H; ++d) moe[static_cast<size_t>(t * H + d)] += w[static_cast<size_t>(k)] * y[static_cast<size_t>(d)];
          router_ids.push_back(static_cast<float>(ids[static_cast<size_t>(k)]));
          router_w.push_back(w[static_cast<size_t>(k)]);
        }
      }
      if (tap) {
        tap(tap_name(l, "router_ids"), router_ids);
        tap(tap_name(l, "router_w"), router_w);
        tap(tap_name(l, "moe_out"), moe);
      }
      std::vector<float> n2(static_cast<size_t>(H));
      for (int64_t t = 0; t < T; ++t) {
        rms_norm(m.data() + t * H, lw.post_ffn_norm_1.data(), H, cfg_.rms_norm_eps, m.data() + t * H);
        rms_norm(moe.data() + t * H, lw.post_ffn_norm_2.data(), H, cfg_.rms_norm_eps, n2.data());
        for (int d = 0; d < H; ++d) m[static_cast<size_t>(t * H + d)] += n2[static_cast<size_t>(d)];
      }
    }
    for (int64_t t = 0; t < T; ++t) {
      rms_norm(m.data() + t * H, lw.post_ffn_norm.data(), H, cfg_.rms_norm_eps, normed.data());
      for (int d = 0; d < H; ++d) {
        float& xv = x[static_cast<size_t>(t * H + d)];
        xv = (xv + normed[static_cast<size_t>(d)]) * lw.layer_scalar;
      }
    }
    if (tap) tap(std::format("h_{:02d}", l + 1), x);
  }
  st.pos += T;
  return x;
}

std::vector<float> Model::final_norm(const std::vector<float>& x) const {
  const int H = cfg_.hidden_size;
  const int64_t T = static_cast<int64_t>(x.size()) / H;
  std::vector<float> out(x.size());
  for (int64_t t = 0; t < T; ++t)
    rms_norm(x.data() + t * H, w_.final_norm.data(), H, cfg_.rms_norm_eps, out.data() + t * H);
  return out;
}

std::vector<float> Model::logits(const std::vector<float>& x) const {
  const int H = cfg_.hidden_size, V = cfg_.vocab_size;
  const int64_t T = static_cast<int64_t>(x.size()) / H;
  const std::vector<float> hn = final_norm(x);
  std::vector<float> out(static_cast<size_t>(T * V));
  for (int64_t t = 0; t < T; ++t) {
    matvec(w_.embed, hn.data() + t * H, out.data() + t * V);  // the head is the embedding
    for (int v = 0; v < V; ++v) {
      float& o = out[static_cast<size_t>(t * V + v)];
      o = softcap(o, cfg_.final_logit_softcapping);
    }
  }
  return out;
}

std::vector<float> Model::logprobs(const std::vector<float>& x) const {
  const int V = cfg_.vocab_size;
  std::vector<float> lg = logits(x);
  const int64_t T = static_cast<int64_t>(lg.size()) / V;
  for (int64_t t = 0; t < T; ++t) {
    float* row = lg.data() + t * V;
    double m = -INFINITY;
    for (int v = 0; v < V; ++v) m = std::max(m, static_cast<double>(row[v]));
    double sum = 0.0;
    for (int v = 0; v < V; ++v) sum += std::exp(static_cast<double>(row[v]) - m);
    const double lse = m + std::log(sum);
    for (int v = 0; v < V; ++v) row[v] = static_cast<float>(static_cast<double>(row[v]) - lse);
  }
  return lg;
}

}  // namespace dgpp::gemma4_ref
