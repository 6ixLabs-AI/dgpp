#pragma once
// NVIDIA Nemotron-3 (NemotronHForCausalLM, model_type nemotron_h)
// configuration, parsed from the checkpoint's config.json root and its
// quantization recipe (2026-10-04, docs/nemotron3_plan.md §1). The same
// policy as the other families: every field the assembly consumes is parsed
// into a known-supported value or rejected with a message naming the field,
// at load time. The reference is the release's own modeling_nemotron_h.py
// (remote code in the snapshot directory).
//
// The architecture: a pre-norm residual stack of ONE mixer per layer (not an
// attention + MLP pair), the kind read off `hybrid_override_pattern`, one
// character per layer:
//   M  Mamba2 (SSD): in_proj -> [gate | x B C | dt]; a depthwise causal
//      conv (kernel 4, bias) + SiLU over [x B C]; per head a
//      [head_dim, ssm_state_size] recurrent state
//          s = s * exp(-exp(A_log) * dt) + (dt * x) (x) B,   y = s C + D x
//      with dt = softplus(dt_raw + dt_bias) and B, C shared by the
//      heads/n_groups heads of a group; y * silu(gate) through an RMS norm
//      over n_groups channel groups; out_proj.
//   E  a routed MoE: sigmoid router with a selection bias (the GLM /
//      DeepSeek rule: the bias on the selection key, the picked scores
//      normalized, x routed_scaling_factor), experts of TWO matrices
//      down(relu(up x)^2) — no gate_proj, no SwiGLU — plus one shared expert
//      added with weight 1. Under `moe_latent_size` (Super) the routed
//      experts run in a latent space: fc1_latent_proj before them,
//      fc2_latent_proj after their weighted sum; the router and the shared
//      expert read the full hidden.
//   *  GQA attention with NO positional encoding: the reference projects
//      q/k/v and calls the softmax directly. `rope_theta` and
//      `partial_rotary_factor` are in config.json and are never read by the
//      modeling code — position comes from the Mamba layers.
// The checkpoints: Nano 30B-A3B (52 layers: 23 M, 23 E, 6 *; hidden 2688;
// 128 experts top-6) and Super 120B-A12B (88 layers: 40 M, 40 E, 8 *; hidden
// 4096; 512 experts top-22 in a 1024-wide latent; a two-layer `*E` draft
// block under `mtp.`).
//
// The quantization recipe comes in two forms, and the form identifies the
// release (as the Qwen3-Next recipe does, models/qwen/config35.hpp):
//   Nvfp4Exclude   (Nano, modelopt 0.29): config.json carries no
//       quantization_config; hf_quant_config.json beside it says
//       quant_algo NVFP4, group_size 16 and lists `exclude_modules`. Every
//       nn.Linear of the backbone is NVFP4 unless excluded. The router gate
//       is an F32 parameter in this release and there are no K/V-cache
//       scale tensors (the recipe names an FP8 cache and writes none).
//   MixedPrecision (Super, modelopt 0.43): config.json's
//       quantization_config.quantized_layers names every quantized module
//       with its own format — NVFP4 (group 16) or per-tensor FP8 — chosen
//       per layer by the exporter's search, so two layers of the same kind
//       differ. Everything else is BF16, the router gate included, and the
//       backbone attention layers carry `k_proj.k_scale` / `v_proj.v_scale`.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"

namespace dgpp {

enum class NemotronLayerKind : uint8_t { Mamba, Moe, Attention };

// The stored form of one nn.Linear weight.
enum class NemotronQuant : uint8_t {
  Bf16,   // X.weight BF16 [N, K]
  Fp8,    // X.weight F8_E4M3 [N, K] x X.weight_scale F32 [] (per tensor)
  Nvfp4,  // X.weight U8 [N, K/2] e2m1 pairs x X.weight_scale e4m3 [N, K/16] x X.weight_scale_2 F32
          // []
};

enum class NemotronRecipe : int { Nvfp4Exclude, MixedPrecision };

struct NemotronHConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 2688;
  int vocab_size = 131072;
  int num_hidden_layers = 52;
  float layer_norm_eps = 1e-5f;  // every RMSNorm, the Mamba gated norm included
  int max_position_embeddings = 262144;
  std::vector<NemotronLayerKind> layers;  // size == num_hidden_layers
  std::vector<int64_t>
      eos_token_ids;  // config.json's; serving resolves generation stop IDs separately
  int64_t bos_token_id = -1;
  int64_t pad_token_id = -1;

  // --- Mamba2 --------------------------------------------------------------
  int mamba_num_heads = 64;
  int mamba_head_dim = 64;
  int ssm_state_size = 128;
  int mamba_n_groups = 8;  // config `n_groups`: the B/C groups and the gated norm's groups
  int conv_kernel = 4;
  bool conv_bias = true;
  int chunk_size = 128;  // the reference's SSD chunk; the recurrence is the same at any chunking
  // The CUDA path the reference serves with (mamba_ssm, and vLLM) applies no
  // time-step clamp (dt_limit (0, inf)); Super's CPU fallback clamps dt from
  // below at this value instead. Recorded so the two can be told apart.
  double time_step_min = 0.001;

  // --- attention (GQA, no positions) ---------------------------------------
  int num_attention_heads = 32;
  int num_key_value_heads = 2;
  int head_dim = 128;

  // --- routed MoE ------------------------------------------------------------
  int n_routed_experts = 128;
  int num_experts_per_tok = 6;
  int moe_intermediate_size = 1856;
  int moe_shared_expert_intermediate_size = 3712;
  int moe_latent_size = 0;  // 0: the experts read the hidden itself
  float routed_scaling_factor = 2.5f;
  bool norm_topk_prob = true;

  // --- MTP -------------------------------------------------------------------
  // Super's draft block: `mtp.layers.0` an attention layer carrying the
  // fusion (enorm, hnorm, eh_proj), `mtp.layers.1` a MoE layer carrying
  // final_layernorm. The release's modeling code does not run it (it lists
  // `mtp.*` as ignored on load); the table binds it for the engine's draft.
  int num_nextn_predict_layers = 0;
  std::vector<NemotronLayerKind> mtp_layers;  // empty, or {Attention, Moe}

  // --- weight formats -------------------------------------------------------
  NemotronRecipe recipe = NemotronRecipe::Nvfp4Exclude;
  DType router_dtype = DType::F32;  // gate.weight: F32 (Nvfp4Exclude) or BF16 (MixedPrecision)
  bool kv_cache_scales = false;  // k_proj.k_scale / v_proj.v_scale on the backbone attention layers
  // Module name (no ".weight") -> Fp8 or Nvfp4; a module not here is BF16.
  std::unordered_map<std::string, NemotronQuant> quant;

  // `root` is config.json; `hf_quant` is hf_quant_config.json's root, or
  // null when the snapshot has none. The recipe is read from
  // root.quantization_config when present (MixedPrecision; an hf_quant that
  // is also given must name the same layers), else from hf_quant
  // (Nvfp4Exclude). A checkpoint with neither is refused.
  static NemotronHConfig parse(const minijson::Value& root, const minijson::Value* hf_quant);
  // Reads config.json and, when it exists, hf_quant_config.json beside it.
  static NemotronHConfig from_json_file(const std::string& path);

  // Layer space: [0, num_hidden_layers) the backbone, then the draft block.
  int num_layers_total() const { return num_hidden_layers + static_cast<int>(mtp_layers.size()); }
  bool is_mtp_layer(int l) const { return l >= num_hidden_layers && l < num_layers_total(); }
  NemotronLayerKind kind_of(int l) const {
    return is_mtp_layer(l) ? mtp_layers[static_cast<size_t>(l - num_hidden_layers)]
                           : layers[static_cast<size_t>(l)];
  }
  int count(NemotronLayerKind k) const {
    int n = 0;
    for (NemotronLayerKind x : layers) n += x == k;
    return n;
  }

  // Mamba2 geometry (NemotronHMamba2Mixer): `expand` is in config.json and
  // unused — the inner width is heads x head_dim (Nano: 4096, not 2 x 2688).
  int64_t mamba_inner() const { return static_cast<int64_t>(mamba_num_heads) * mamba_head_dim; }
  int64_t mamba_conv_dim() const {
    return mamba_inner() + 2 * static_cast<int64_t>(mamba_n_groups) * ssm_state_size;
  }
  // in_proj rows: [gate (inner) | x B C (conv_dim) | dt (heads)].
  int64_t mamba_proj_rows() const { return mamba_inner() + mamba_conv_dim() + mamba_num_heads; }
  int64_t mamba_norm_group() const { return mamba_inner() / mamba_n_groups; }
  // The routed experts' input width: the latent, or the hidden.
  int64_t expert_width() const { return moe_latent_size > 0 ? moe_latent_size : hidden_size; }

  NemotronQuant quant_of(const std::string& module) const {
    const auto it = quant.find(module);
    return it == quant.end() ? NemotronQuant::Bf16 : it->second;
  }
};

// "backbone.layers.L." for the backbone, "mtp.layers.M." for the draft block.
std::string nemotron_layer_prefix(const NemotronHConfig& cfg, int layer);
// Every nn.Linear of layer `layer` by module name (no ".weight"): the
// modules a recipe may quantize. Mamba: mixer.in_proj, mixer.out_proj;
// attention: mixer.{q,k,v,o}_proj; MoE: mixer.experts.E.{up,down}_proj,
// mixer.shared_experts.{up,down}_proj and, with a latent,
// mixer.fc{1,2}_latent_proj.
std::vector<std::string> nemotron_layer_linears(const NemotronHConfig& cfg, int layer);

}  // namespace dgpp
