#pragma once
// Host reference of the Gemma 4 text model (2026-10-04): every operator the
// family adds to the engine — the plain-weight RMS norm and its weightless
// form, GeGLU, the two rotary kinds (whole-head and "proportional"), the
// q/k/v-normed GQA attention with a sliding window or full causal reach and
// the full layers' k_proj-as-value, the sandwich norms, layer_scalar, the
// bf16 embedding scale, the tied head and the logit soft-cap, and (the
// 26B-A4B) the MoE block's router and experts beside the dense MLP — and
// the model assembled from them, reading a checkpoint through the family's
// tensor table.
//
// It is the C++ twin of tools/gemma4_reference.py (which is checked against
// the transformers class by tools/gemma4_torch_check.py): fp32 values, the
// dot products and norm sums accumulated in double, the rotary angles in
// double. No rounding to bf16 anywhere — this is the mathematical model,
// what a kernel's output is compared with to within its rounding, not a
// model of any kernel's rounding chain. It is slow by design (a triple loop
// per matrix) and meant for synthetic checkpoints and single layers.
//
// Host-only: no CUDA.
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "models/gemma4/config.hpp"

namespace dgpp::gemma4_ref {

// ---- operators ---------------------------------------------------------------

// torch.nn.functional.gelu(x, approximate="tanh").
float gelu_tanh(float x);
// tanh(x / cap) * cap; cap <= 0 returns x.
float softcap(float x, float cap);
// Gemma4RMSNorm over n values: out = x * (mean(x^2) + eps)^-0.5 [* w].
// `w` null is the weightless form (the value norm). out may alias x.
void rms_norm(const float* x, const float* w, int n, float eps, float* out);
// The rotated pairs' inverse frequencies: pair i -> theta^(-2i / head_dim),
// i < pairs (the exponent's divisor is the whole head for both kinds).
std::vector<double> rope_inv_freq(int head_dim, int pairs, double theta);
// Rotates one head in place at `pos`: pair i couples (i, i + head_dim/2);
// dims past the rotated pairs are untouched.
void rope_apply(float* x, int head_dim, const std::vector<double>& inv_freq, int64_t pos);
// The first key position a query at `pos` reads: pos - window + 1 clamped
// at 0 for a sliding layer (`kv_idx > q_idx - sliding_window`), 0 for a
// full layer (window 0).
int64_t first_visible(int64_t pos, int window);
// One position's attention: q [heads x hd] over the n visible K / V rows
// ([n, kv_heads x hd] each, in position order); ctx [heads x hd]. Scores
// are plain dot products (no 1/sqrt(d)); query head h reads KV head
// h / (heads / kv_heads).
void attention(const float* q, const float* k_rows, const float* v_rows, int64_t n, int heads,
               int kv_heads, int hd, float* ctx);
// The MoE block's router for one token. x [H] is the residual stream:
//   r = rms_noscale(x) * scale * input_scale;  p = softmax(proj @ r)  (fp32)
//   the top_k probabilities, best first (a tie goes to the lower id — the
//   engine's rule; torch's topk leaves ties unspecified), renormalized to
//   sum 1, each then times per_expert_scale[id].
// proj [E, H] row-major; ids / weights [top_k].
void route(const float* x, const float* proj, const float* scale, const float* per_expert_scale,
           float input_scale, int H, int E, int top_k, float eps, int* ids, float* weights);
// A modelopt NVFP4 matrix to fp32: w[r, c] = e2m1(code) * (e4m3(scale[r, c/16]) * ws2),
// low nibble = even column.
void decode_nvfp4(const uint8_t* payload, const uint8_t* scales, float ws2, int64_t rows,
                  int64_t cols, float* out);

// ---- weights -------------------------------------------------------------------

struct Matrix {
  int64_t rows = 0, cols = 0;
  std::vector<float> w;  // row-major [rows, cols]
  bool empty() const { return w.empty(); }
};

struct LayerWeights {
  std::vector<float> input_norm, post_attn_norm, pre_ffn_norm, post_ffn_norm;  // [H]
  float layer_scalar = 1.0f;
  std::vector<float> q_norm, k_norm;  // [head_dim of the layer]
  Matrix q_proj, k_proj, v_proj, o_proj;  // v_proj empty on a full layer
  Matrix gate_proj, up_proj, down_proj;
  // The MoE block (empty without it).
  std::vector<float> post_ffn_norm_1, post_ffn_norm_2, pre_ffn_norm_2;  // [H]
  Matrix router_proj;                       // [E, H]
  std::vector<float> router_scale;          // [H]
  std::vector<float> per_expert_scale;      // [E]
  struct Expert {
    Matrix gate_proj, up_proj, down_proj;   // [Im, H], [Im, H], [H, Im]
  };
  std::vector<Expert> experts;
};

struct Weights {
  Matrix embed;                 // [vocab, H]; also the head
  std::vector<float> final_norm;
  std::vector<LayerWeights> layers;
};

// Reads the first `layers` layers (all when < 0) of the checkpoint in
// `checkpoint_dir` (every *.safetensors there) through the family's tensor
// table: BF16 widened, NVFP4 dequantized. Throws naming a tensor that is
// missing or has another dtype / shape than the table says.
Weights load_weights(const Gemma4TextConfig& cfg, const std::string& checkpoint_dir, int layers = -1);

// ---- the model -------------------------------------------------------------------

struct LayerCache {
  std::vector<float> k, v;  // [rows, kv_heads x hd] each
  int64_t rows = 0;
};

// Per-sequence state: a sequence can be fed in chunks or token by token.
struct State {
  int64_t pos = 0;
  std::vector<LayerCache> layers;
};

// Named intermediate values (the numpy reference's `dump` names: h_NN,
// LNN_attn_q, LNN_attn_k, LNN_attn_v, LNN_attn_out, LNN_mlp_out and, with
// the MoE block, LNN_router_ids (the ids as floats), LNN_router_w,
// LNN_moe_out).
using Tap = std::function<void(std::string_view name, const std::vector<float>& values)>;

class Model {
 public:
  Model(const Gemma4TextConfig& cfg, Weights weights);

  int layers() const { return static_cast<int>(w_.layers.size()); }
  // Feeds `ids` at positions st.pos..; returns the residual stream [T, H]
  // after the last loaded layer, before the final norm.
  std::vector<float> forward(const std::vector<int64_t>& ids, State& st, const Tap& tap = {}) const;
  // rms(x, norm) [T, H].
  std::vector<float> final_norm(const std::vector<float>& x) const;
  // The tied head and the soft-cap on [T, H] residual rows: [T, vocab].
  std::vector<float> logits(const std::vector<float>& x) const;
  // log-softmax of logits(x), per row.
  std::vector<float> logprobs(const std::vector<float>& x) const;

 private:
  void attention_layer(int l, const std::vector<float>& h, int64_t T, State& st, std::vector<float>& out,
                       const Tap& tap) const;

  Gemma4TextConfig cfg_;
  Weights w_;
};

}  // namespace dgpp::gemma4_ref
