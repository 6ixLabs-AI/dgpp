// The Gemma 4 host reference (models/gemma4/reference.hpp) against the numpy
// reference (tools/gemma4_reference.py) on the synthetic checkpoints
// "Gemma4Synth v1" — one in the 31B release's shape (BF16 attention, NVFP4
// MLP), one in the 26B-A4B's (NVFP4 attention, the MoE block with unfused
// NVFP4 experts and BF16 routers): the operators one by one, then the model — the C++
// generator's bytes (content hash), the tensor table over the file it
// writes, the loader, and every layer's residual stream, the attention
// taps, the final norm, the soft-capped logits and the log-probabilities.
// The expected values (gemma4_reference_vectors.hpp) are written by
// `tools/gemma4_synth.py vectors`; that numpy reference is itself compared
// with the transformers class by tools/gemma4_torch_check.py.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "gemma4_reference_vectors.hpp"
#include "gemma4_synth.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "models/gemma4/binding.hpp"
#include "models/gemma4/config.hpp"
#include "models/gemma4/reference.hpp"

namespace {

namespace ref = dgpp::gemma4_ref;
namespace vec = gemma4_vectors;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// max |a - b| over n values, as an error message when above tol.
void close(const float* got, const float* want, size_t n, float tol, const std::string& what) {
  float worst = 0.0f;
  size_t at = 0;
  for (size_t i = 0; i < n; ++i) {
    const float d = std::fabs(got[i] - want[i]);
    if (!(d <= worst)) {
      worst = d;
      at = i;
    }
  }
  if (!(worst <= tol))
    throw std::runtime_error(what + ": max |diff| " + std::to_string(worst) + " at " + std::to_string(at) + " (got " +
                             std::to_string(got[at]) + ", want " + std::to_string(want[at]) + ")");
}

// A scratch directory holding the synthetic checkpoint, removed on exit.
struct SynthCheckpoint {
  std::filesystem::path dir;
  dgpp::Gemma4TextConfig cfg;
  std::map<std::string, gemma4_synth::SynthTensor> tensors;

  // `config_json`: the recipe's config.json as the Python tool wrote it.
  explicit SynthCheckpoint(const char* config_json = vec::kConfigJsonMlp) {
    const std::string text = config_json;
    const auto parsed = dgpp::minijson::parse(text);
    cfg = dgpp::Gemma4TextConfig::parse(parsed.root);
    tensors = gemma4_synth::tensors(cfg, vec::kSeed);
    static int serial = 0;
    dir = std::filesystem::temp_directory_path() /
          ("gemma4_synth_" + std::to_string(::getpid()) + "_" + std::to_string(serial++));
    gemma4_synth::write_checkpoint(dir, config_json, tensors);
  }
  ~SynthCheckpoint() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

std::vector<int64_t> ids() { return std::vector<int64_t>(vec::kIds, vec::kIds + vec::kTokens); }

constexpr float kTol = 2e-4f;  // fp32 summation order against numpy's BLAS

}  // namespace

DGPP_TEST(gemma4_ref_gelu_tanh_and_softcap) {
  // torch.nn.functional.gelu(x, approximate="tanh") at a few points (float64 values).
  const float xs[] = {-3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f};
  const float want[] = {-0.0036373920817729943f, -0.1588080093917233f, -0.15428599017485606f, 0.0f,
                        0.34571400982514394f,    0.8411919906082768f,  2.996362607918227f};
  for (int i = 0; i < 7; ++i)
    require(std::fabs(ref::gelu_tanh(xs[i]) - want[i]) < 2e-6f, "gelu_tanh(" + std::to_string(xs[i]) + ")");
  // tanh(x / cap) * cap.
  require(std::fabs(ref::softcap(30.0f, 30.0f) - 30.0f * 0.7615941559557649f) < 1e-4f, "softcap at the cap");
  require(std::fabs(ref::softcap(-60.0f, 30.0f) + 30.0f * 0.9640275800758169f) < 1e-4f, "softcap is odd");
  require(std::fabs(ref::softcap(0.3f, 30.0f) - 0.29999f) < 1e-4f, "softcap is the identity near 0");
  require(ref::softcap(123.0f, 0.0f) == 123.0f, "no cap");
}

DGPP_TEST(gemma4_ref_rms_norm_is_plain_weight) {
  // mean(x^2) = (1 + 4 + 4 + 16) / 4 = 6.25 -> x / 2.5 (eps negligible), times w — NOT (1 + w).
  const float x[4] = {1.0f, -2.0f, 2.0f, 4.0f};
  const float w[4] = {1.0f, 0.5f, 2.0f, 0.0f};
  float out[4];
  ref::rms_norm(x, w, 4, 1e-6f, out);
  const float want[4] = {0.4f, -0.4f, 1.6f, 0.0f};
  close(out, want, 4, 1e-6f, "weighted");
  ref::rms_norm(x, nullptr, 4, 1e-6f, out);
  const float bare[4] = {0.4f, -0.8f, 0.8f, 1.6f};
  close(out, bare, 4, 1e-6f, "weightless");
  // eps sits inside the root: a zero vector stays zero.
  const float z[4] = {0, 0, 0, 0};
  ref::rms_norm(z, w, 4, 1e-6f, out);
  require(out[0] == 0.0f && out[3] == 0.0f, "zero in, zero out");
  // In place.
  float y[4] = {1.0f, -2.0f, 2.0f, 4.0f};
  ref::rms_norm(y, nullptr, 4, 1e-6f, y);
  close(y, bare, 4, 1e-6f, "in place");
}

DGPP_TEST(gemma4_ref_rope_kinds) {
  // The whole-head kind: head_dim 8 -> 4 pairs (i, i + 4), inv_freq theta^(-2i/8).
  {
    const auto inv = ref::rope_inv_freq(8, 4, 10000.0);
    require(inv.size() == 4 && inv[0] == 1.0 && std::fabs(inv[1] - 0.1) < 1e-12 && std::fabs(inv[3] - 1e-3) < 1e-15,
            "default inv_freq");
    float x[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    ref::rope_apply(x, 8, inv, 2);
    // pair 0: (x0, x4) = (1, 0) rotated by 2 rad -> (cos 2, sin 2); pair 1: (x1, x5) = (0, 1) by 0.2 -> (-sin, cos).
    const float want[8] = {std::cos(2.0f), -std::sin(0.2f), 0, 0, std::sin(2.0f), std::cos(0.2f), 0, 0};
    close(x, want, 8, 1e-6f, "default rope");
    float y[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    ref::rope_apply(y, 8, inv, 0);
    const float same[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    close(y, same, 8, 0.0f, "position 0 is the identity");
  }
  // The proportional kind: head_dim 16, factor 0.25 -> 2 rotated pairs (i, i + 8) with inv_freq
  // theta^(-2i/16) — the divisor is the whole head — and 6 pairs left alone.
  {
    const auto inv = ref::rope_inv_freq(16, 2, 1000000.0);
    require(inv.size() == 2 && inv[0] == 1.0 && std::fabs(inv[1] - std::pow(1e6, -0.125)) < 1e-15, "proportional inv_freq");
    float x[16];
    for (int i = 0; i < 16; ++i) x[i] = static_cast<float>(i + 1);
    ref::rope_apply(x, 16, inv, 3);
    const double a0 = 3.0, a1 = 3.0 * std::pow(1e6, -0.125);
    require(std::fabs(x[0] - static_cast<float>(1 * std::cos(a0) - 9 * std::sin(a0))) < 1e-5f, "pair 0 low");
    require(std::fabs(x[8] - static_cast<float>(9 * std::cos(a0) + 1 * std::sin(a0))) < 1e-5f, "pair 0 high");
    require(std::fabs(x[1] - static_cast<float>(2 * std::cos(a1) - 10 * std::sin(a1))) < 1e-5f, "pair 1 low");
    require(std::fabs(x[9] - static_cast<float>(10 * std::cos(a1) + 2 * std::sin(a1))) < 1e-5f, "pair 1 high");
    for (int i : {2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15})
      require(x[i] == static_cast<float>(i + 1), "dim " + std::to_string(i) + " must not rotate");
  }
  // The releases' geometry: 128 pairs over a 256 head, 64 over a 512 head.
  require(ref::rope_inv_freq(256, 128, 1e4).back() == std::pow(1e4, -254.0 / 256.0), "sliding last frequency");
  require(ref::rope_inv_freq(512, 64, 1e6).back() == std::pow(1e6, -126.0 / 512.0), "full last frequency");
}

DGPP_TEST(gemma4_ref_window_is_kv_greater_than_q_minus_w) {
  // kv_idx > q_idx - sliding_window: W keys including the query's own position.
  require(ref::first_visible(0, 1024) == 0 && ref::first_visible(1023, 1024) == 0, "inside the first window");
  require(ref::first_visible(1024, 1024) == 1 && ref::first_visible(5000, 1024) == 3977, "sliding");
  require(ref::first_visible(7, 4) == 4 && ref::first_visible(3, 4) == 0, "window 4");
  require(ref::first_visible(123456, 0) == 0, "a full layer reads everything");
}

DGPP_TEST(gemma4_ref_attention_is_an_unscaled_gqa_softmax) {
  // 2 query heads on 1 KV head, head_dim 2, three rows. Scores are plain dot products.
  const float q[4] = {1.0f, 0.0f, 0.0f, 2.0f};
  const float k[6] = {0.0f, 0.0f, std::log(2.0f), 0.0f, 0.0f, std::log(3.0f) / 2.0f};
  const float v[6] = {1.0f, 10.0f, 2.0f, 20.0f, 4.0f, 40.0f};
  float ctx[4];
  ref::attention(q, k, v, 3, 2, 1, 2, ctx);
  // head 0: scores (0, ln 2, 0) -> p = (1/4, 1/2, 1/4); head 1: (0, 0, ln 3) -> (1/5, 1/5, 3/5).
  const float want[4] = {0.25f * 1 + 0.5f * 2 + 0.25f * 4, 0.25f * 10 + 0.5f * 20 + 0.25f * 40,
                         0.2f * 1 + 0.2f * 2 + 0.6f * 4,   0.2f * 10 + 0.2f * 20 + 0.6f * 40};
  close(ctx, want, 4, 1e-5f, "attention");
  // 4 query heads on 2 KV heads: heads 0, 1 read KV head 0; heads 2, 3 read KV head 1.
  const float q4[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  const float k2[2] = {0.0f, 0.0f}, v2[2] = {5.0f, 7.0f};
  float c4[4];
  ref::attention(q4, k2, v2, 1, 4, 2, 1, c4);
  require(c4[0] == 5.0f && c4[1] == 5.0f && c4[2] == 7.0f && c4[3] == 7.0f, "h -> h / groups");
}

DGPP_TEST(gemma4_ref_nvfp4_decode) {
  // Two rows of 16: byte j holds columns 2j (low nibble) and 2j + 1 (high nibble).
  uint8_t payload[16], scales[2] = {0x38, 0x40};  // e4m3 1.0 and 2.0
  for (int j = 0; j < 8; ++j) {
    payload[j] = static_cast<uint8_t>(j | ((j + 8) << 4));  // codes j and -(same magnitude)
    payload[8 + j] = static_cast<uint8_t>(0x7 | (0x1 << 4));  // 6.0 and 0.5
  }
  float out[32];
  ref::decode_nvfp4(payload, scales, 0.25f, 2, 16, out);
  const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (int j = 0; j < 8; ++j) {
    require(out[2 * j] == mag[j] * 0.25f, "row 0 even column " + std::to_string(j));
    require(out[2 * j + 1] == -mag[j] * 0.25f, "row 0 odd column " + std::to_string(j));
    require(out[16 + 2 * j] == 6.0f * 2.0f * 0.25f && out[16 + 2 * j + 1] == 0.5f * 2.0f * 0.25f, "row 1");
  }
}

DGPP_TEST(gemma4_ref_synthetic_checkpoint_is_the_python_one) {
  const SynthCheckpoint ck;
  // The same bytes as tools/gemma4_synth.py wrote: one hash over every tensor.
  require(static_cast<int>(ck.tensors.size()) == vec::kTensorsMlp, "tensor count " + std::to_string(ck.tensors.size()));
  require(gemma4_synth::content_hash(ck.tensors) == vec::kContentHashMlp, "the C++ and Python generators differ");
  // The table binds the file it describes: every tensor, the encoder's one ignored.
  auto f = dgpp::SafetensorsFile::open((ck.dir / "model.safetensors").string());
  std::unordered_map<std::string, dgpp::Gemma4TensorDesc> present;
  f->for_each([&](const dgpp::TensorInfo& t) { present.emplace(t.name, dgpp::Gemma4TensorDesc{t.dtype, t.shape}); });
  const auto rep = dgpp::gemma4_validate_text_binding(ck.cfg, present);
  std::string errors;
  for (const auto& e : rep.errors) errors += "\n  " + e;
  require(rep.ok(), "binding:" + errors);
  require(rep.expected == 138 && rep.ignored == 1 && rep.fp4_matrices == 18, "report counts");
  // The synthetic geometry.
  require(ck.cfg.num_hidden_layers == 6 && ck.cfg.num_sliding_layers() == 4 && ck.cfg.sliding_window == 4, "layers / window");
  require(ck.cfg.head_dim == 8 && ck.cfg.global_head_dim == 16 && ck.cfg.global_rotary_pairs == 2, "heads / rotary");
  require(ck.cfg.embed_scale() == 5.65625f, "bf16(sqrt(32))");
}

DGPP_TEST(gemma4_ref_loader_reads_through_the_table) {
  const SynthCheckpoint ck;
  const ref::Weights w = ref::load_weights(ck.cfg, ck.dir.string());
  require(w.layers.size() == 6 && w.embed.rows == 96 && w.embed.cols == 32, "shapes");
  require(!w.layers[0].v_proj.empty() && w.layers[2].v_proj.empty(), "a full layer has no v_proj");
  require(w.layers[0].q_proj.rows == 32 && w.layers[2].q_proj.rows == 64 && w.layers[2].k_proj.rows == 16, "projection rows");
  require(w.layers[0].q_norm.size() == 8 && w.layers[2].q_norm.size() == 16, "norm widths");
  require(w.layers[0].gate_proj.rows == 64 && w.layers[0].gate_proj.cols == 32 && w.layers[0].down_proj.cols == 64, "mlp");
  // A decoded NVFP4 value is code x scale x weight_scale_2 with the generator's draws.
  const auto& payload = ck.tensors.at("model.language_model.layers.0.mlp.gate_proj.weight").bytes;
  const auto& scales = ck.tensors.at("model.language_model.layers.0.mlp.gate_proj.weight_scale").bytes;
  float ws2;
  std::memcpy(&ws2, ck.tensors.at("model.language_model.layers.0.mlp.gate_proj.weight_scale_2").bytes.data(), 4);
  const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (int c : {0, 1, 17, 31}) {
    const uint8_t code = (c & 1) ? payload[static_cast<size_t>(c / 2)] >> 4 : payload[static_cast<size_t>(c / 2)] & 0xF;
    const float want = ((code & 8) ? -1.0f : 1.0f) * mag[code & 7] * dgpp::fp8_e4m3_bits_to_float(scales[static_cast<size_t>(c / 16)]) * ws2;
    require(w.layers[0].gate_proj.w[static_cast<size_t>(c)] == want, "decoded column " + std::to_string(c));
  }
  // --layers: the first N only.
  require(ref::load_weights(ck.cfg, ck.dir.string(), 2).layers.size() == 2, "partial load");
  // A tensor with another shape than the table says is refused by name.
  {
    dgpp::Gemma4TextConfig other = ck.cfg;
    other.intermediate_size = 48;
    bool threw = false;
    try {
      (void)ref::load_weights(other, ck.dir.string());
    } catch (const std::exception& e) {
      threw = std::string(e.what()).find("mlp.gate_proj.weight") != std::string::npos;
    }
    require(threw, "a shape mismatch must name the tensor");
  }
}

DGPP_TEST(gemma4_ref_forward_matches_the_numpy_reference) {
  const SynthCheckpoint ck;
  const ref::Model model(ck.cfg, ref::load_weights(ck.cfg, ck.dir.string()));
  std::unordered_map<std::string, std::vector<float>> taps;
  ref::State st;
  const std::vector<float> x = model.forward(ids(), st, [&](std::string_view name, const std::vector<float>& v) {
    taps[std::string(name)] = v;
  });
  require(st.pos == vec::kTokens && x.size() == 12 * 32, "state / shape");
  const size_t last = static_cast<size_t>(vec::kTokens - 1);
  close(taps.at("h_00").data() + last * 32, vec::kEmbedLastMlp, 32, 1e-6f, "scaled embedding");
  // A sliding layer (0): 4 heads x 8, 2 KV heads; a full layer (2): 4 heads x 16, 1 KV head, value = k_proj.
  close(taps.at("L00_attn_q").data() + last * 32, vec::kL0AttnQLastMlp, 32, kTol, "L0 q");
  close(taps.at("L00_attn_k").data() + last * 16, vec::kL0AttnKLastMlp, 16, kTol, "L0 k");
  close(taps.at("L00_attn_v").data() + last * 16, vec::kL0AttnVLastMlp, 16, kTol, "L0 v");
  close(taps.at("L00_attn_out").data(), vec::kL0AttnOutMlp, 384, kTol, "L0 attention output");
  close(taps.at("L00_mlp_out").data(), vec::kL0MlpOutMlp, 384, kTol, "L0 mlp output");
  close(taps.at("L02_attn_q").data() + last * 64, vec::kL2AttnQLastMlp, 64, kTol, "L2 q");
  close(taps.at("L02_attn_k").data() + last * 16, vec::kL2AttnKLastMlp, 16, kTol, "L2 k");
  close(taps.at("L02_attn_v").data() + last * 16, vec::kL2AttnVLastMlp, 16, kTol, "L2 v");
  close(taps.at("L02_attn_out").data(), vec::kL2AttnOutMlp, 384, kTol, "L2 attention output");
  close(taps.at("L02_mlp_out").data(), vec::kL2MlpOutMlp, 384, kTol, "L2 mlp output");
  const float* hidden[7] = {vec::kHidden0Mlp, vec::kHidden1Mlp, vec::kHidden2Mlp, vec::kHidden3Mlp,
                            vec::kHidden4Mlp, vec::kHidden5Mlp, vec::kHidden6Mlp};
  for (int l = 0; l <= 6; ++l) {
    char name[8];
    std::snprintf(name, sizeof name, "h_%02d", l);
    close(taps.at(name).data(), hidden[l], 384, kTol, std::string("residual stream ") + name);
  }
  close(x.data(), vec::kHidden6Mlp, 384, kTol, "returned stream");
  close(model.final_norm(x).data(), vec::kFinalNormMlp, 384, kTol, "final norm");
  const std::vector<float> logits = model.logits(x);
  require(logits.size() == 12 * 96, "logits shape");
  close(logits.data(), vec::kLogitsMlp, 1152, kTol, "soft-capped logits");
  // The cap (1.5 on this fixture) bites: no logit reaches it, some come close.
  float amax = 0.0f;
  for (float v : logits) amax = std::max(amax, std::fabs(v));
  require(amax < 1.5f && amax > 1.0f, "soft-cap range " + std::to_string(amax));
  const std::vector<float> lp = model.logprobs(x);
  close(lp.data() + last * 96, vec::kLogprobsLastMlp, 96, kTol, "log-probabilities");
}

DGPP_TEST(gemma4_ref_incremental_forward_equals_whole_sequence) {
  const SynthCheckpoint ck;
  const ref::Model model(ck.cfg, ref::load_weights(ck.cfg, ck.dir.string()));
  // One token at a time, and in chunks of 5: the cached keys / values and the window's tail.
  for (int chunk : {1, 5}) {
    ref::State st;
    std::vector<float> x;
    const std::vector<int64_t> all = ids();
    for (size_t s = 0; s < all.size(); s += static_cast<size_t>(chunk)) {
      const std::vector<int64_t> part(all.begin() + static_cast<long>(s),
                                      all.begin() + static_cast<long>(std::min(all.size(), s + static_cast<size_t>(chunk))));
      const std::vector<float> y = model.forward(part, st);
      x.insert(x.end(), y.begin(), y.end());
    }
    require(st.pos == 12, "position");
    close(x.data(), vec::kHidden6Mlp, 384, kTol, "chunk " + std::to_string(chunk));
  }
  // A token id outside the vocabulary is refused.
  ref::State st;
  bool threw = false;
  try {
    (void)model.forward({96}, st);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  require(threw, "out-of-range id");
}

DGPP_TEST(gemma4_ref_first_layers_only) {
  // A model loaded with --layers 3 stops after layer 2: the residual stream is h_03.
  const SynthCheckpoint ck;
  const ref::Model model(ck.cfg, ref::load_weights(ck.cfg, ck.dir.string(), 3));
  ref::State st;
  const std::vector<float> x = model.forward(ids(), st);
  close(x.data(), vec::kHidden3Mlp, 384, kTol, "three layers");
}

// ---- the MoE block (the 26B-A4B's shape) ------------------------------------

DGPP_TEST(gemma4_ref_route_is_softmax_topk_renormalized_and_scaled) {
  // H = 2, three experts. x = (3, 4): the weightless norm gives (3, 4) / sqrt(12.5); with scale
  // (1, 1) and input_scale sqrt(12.5) the router sees (3, 4) again.
  const float x[2] = {3.0f, 4.0f};
  const float scale[2] = {1.0f, 1.0f};
  const float in_scale = std::sqrt(12.5f);
  // logits = proj @ (3, 4): ln 1, ln 2, ln 5 -> p = (1/8, 2/8, 5/8).
  const float l2 = std::log(2.0f), l5 = std::log(5.0f);
  const float proj[6] = {0.0f, 0.0f, l2 / 3.0f, 0.0f, 0.0f, l5 / 4.0f};
  const float pes[3] = {10.0f, 100.0f, 1000.0f};
  int ids[2];
  float w[2];
  ref::route(x, proj, scale, pes, in_scale, 2, 3, 2, 1e-6f, ids, w);
  // Top 2: expert 2 (5/8) then expert 1 (2/8); renormalized 5/7 and 2/7; times per_expert_scale.
  require(ids[0] == 2 && ids[1] == 1, "top-k order");
  require(std::fabs(w[0] - 1000.0f * 5.0f / 7.0f) < 0.05f && std::fabs(w[1] - 100.0f * 2.0f / 7.0f) < 0.005f,
          "weights " + std::to_string(w[0]) + ", " + std::to_string(w[1]));
  // The per-dimension scale multiplies the router's input.
  const float scale2[2] = {1.0f, 0.0f};  // only x[0] reaches the router: logits (0, ln 2, 0)
  int ids3[3];
  float w3[3];
  ref::route(x, proj, scale2, pes, in_scale, 2, 3, 3, 1e-6f, ids3, w3);
  // p = (1/4, 1/2, 1/4): expert 1 first, then the tie by id.
  require(ids3[0] == 1 && ids3[1] == 0 && ids3[2] == 2, "the scaled input routes to expert 1");
  require(std::fabs(w3[0] - 100.0f * 0.5f) < 1e-3f && std::fabs(w3[1] - 10.0f * 0.25f) < 1e-4f, "its weights");
  // A tie goes to the lower id: all logits equal.
  const float zero[6] = {0, 0, 0, 0, 0, 0};
  int tie[3];
  float tw[3];
  ref::route(x, zero, scale, pes, in_scale, 2, 3, 3, 1e-6f, tie, tw);
  require(tie[0] == 0 && tie[1] == 1 && tie[2] == 2, "ties by id");
  require(std::fabs(tw[0] - 10.0f / 3.0f) < 1e-4f && std::fabs(tw[2] - 1000.0f / 3.0f) < 1e-2f, "tied weights");
}

DGPP_TEST(gemma4_ref_moe_checkpoint_is_the_python_one) {
  const SynthCheckpoint ck(vec::kConfigJsonMoe);
  require(ck.cfg.enable_moe_block && ck.cfg.num_experts == 8 && ck.cfg.top_k_experts == 3 && ck.cfg.moe_intermediate_size == 16,
          "the MoE geometry");
  require(ck.cfg.recipe == dgpp::Gemma4Recipe::Nvfp4WeightOnly && ck.cfg.attention_nvfp4() && !ck.cfg.activation_scales(),
          "the weight-only recipe");
  require(ck.cfg.router_input_scale() == 0.1767578125f, "bf16(32^-0.5)");
  require(static_cast<int>(ck.tensors.size()) == vec::kTensorsMoe, "tensor count " + std::to_string(ck.tensors.size()));
  require(gemma4_synth::content_hash(ck.tensors) == vec::kContentHashMoe, "the C++ and Python generators differ");
  auto f = dgpp::SafetensorsFile::open((ck.dir / "model.safetensors").string());
  std::unordered_map<std::string, dgpp::Gemma4TensorDesc> present;
  f->for_each([&](const dgpp::TensorInfo& t) { present.emplace(t.name, dgpp::Gemma4TensorDesc{t.dtype, t.shape}); });
  const auto rep = dgpp::gemma4_validate_text_binding(ck.cfg, present);
  std::string errors;
  for (const auto& e : rep.errors) errors += "\n  " + e;
  require(rep.ok(), "binding:" + errors);
  // 22 attention + 18 MLP + 6 x 8 x 3 expert matrices, all NVFP4; 6 routers and the embedding BF16.
  require(rep.expected == 632 && rep.ignored == 1 && rep.fp4_matrices == 184 && rep.expert_matrices == 144 &&
              rep.bf16_matrices == 7,
          "report counts: fp4 " + std::to_string(rep.fp4_matrices) + " bf16 " + std::to_string(rep.bf16_matrices));
  const ref::Weights w = ref::load_weights(ck.cfg, ck.dir.string());
  require(w.layers[0].experts.size() == 8 && w.layers[0].experts[7].gate_proj.rows == 16 &&
              w.layers[0].experts[7].down_proj.cols == 16 && w.layers[0].router_proj.rows == 8 &&
              w.layers[0].per_expert_scale.size() == 8 && w.layers[0].pre_ffn_norm_2.size() == 32,
          "the MoE weights' shapes");
}

DGPP_TEST(gemma4_ref_moe_forward_matches_the_numpy_reference) {
  const SynthCheckpoint ck(vec::kConfigJsonMoe);
  const ref::Model model(ck.cfg, ref::load_weights(ck.cfg, ck.dir.string()));
  std::unordered_map<std::string, std::vector<float>> taps;
  ref::State st;
  const std::vector<float> x = model.forward(ids(), st, [&](std::string_view name, const std::vector<float>& v) {
    taps[std::string(name)] = v;
  });
  // Layer 0's routing: the same experts in the same order, the same weights.
  const std::vector<float>& rid = taps.at("L00_router_ids");
  require(rid.size() == 12 * static_cast<size_t>(vec::kTopKMoe), "router ids shape");
  for (size_t i = 0; i < rid.size(); ++i)
    require(static_cast<int>(rid[i]) == vec::kL0RouterIdsMoe[i],
            "router id " + std::to_string(i) + ": " + std::to_string(rid[i]) + " != " + std::to_string(vec::kL0RouterIdsMoe[i]));
  close(taps.at("L00_router_w").data(), vec::kL0RouterWMoe, 36, kTol, "router weights");
  close(taps.at("L00_moe_out").data(), vec::kL0MoeOutMoe, 384, kTol, "L0 experts' sum");
  close(taps.at("L02_attn_out").data(), vec::kL2AttnOutMoe, 384, kTol, "L2 attention output (NVFP4 projections)");
  const float* hidden[7] = {vec::kHidden0Moe, vec::kHidden1Moe, vec::kHidden2Moe, vec::kHidden3Moe,
                            vec::kHidden4Moe, vec::kHidden5Moe, vec::kHidden6Moe};
  for (int l = 0; l <= 6; ++l) {
    char name[8];
    std::snprintf(name, sizeof name, "h_%02d", l);
    close(taps.at(name).data(), hidden[l], 384, kTol, std::string("residual stream ") + name);
  }
  close(model.final_norm(x).data(), vec::kFinalNormMoe, 384, kTol, "final norm");
  close(model.logits(x).data(), vec::kLogitsMoe, 1152, kTol, "soft-capped logits");
  const std::vector<float> lp = model.logprobs(x);
  close(lp.data() + 11 * 96, vec::kLogprobsLastMoe, 96, kTol, "log-probabilities");
  // Token by token through the caches.
  ref::State st1;
  std::vector<float> x1;
  for (const int64_t id : ids()) {
    const std::vector<float> y = model.forward({id}, st1);
    x1.insert(x1.end(), y.begin(), y.end());
  }
  close(x1.data(), vec::kHidden6Moe, 384, kTol, "one token at a time");
}
