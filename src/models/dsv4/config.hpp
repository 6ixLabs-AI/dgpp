#pragma once
// DeepSeek-V4-Flash (DeepseekV4ForCausalLM, model_type deepseek_v4; the
// 0731 release) configuration, parsed from the checkpoint's flat
// config.json (2026-10-01). The same policy as the other parsers: every
// field the assembly consumes is parsed into a known-supported value or
// rejected with a message naming the field, at load time. The reference is
// the checkpoint's own `inference/model.py` (ModelArgs / Transformer).
//
// The quantization contract (the release as DeepSeek ships it): `fp8` with
// `weight_block_size` [128, 128] and `scale_fmt` ue8m0 for every dense
// projection (`X.weight` F8_E4M3 [N, K] + `X.scale` F8_E8M0 [N/128,
// K/128]), `expert_dtype` fp4 for the routed and draft experts (`X.weight`
// I8 [N, K/2] — two e2m1 codes per byte, the low nibble first — +
// `X.scale` F8_E8M0 [N, K/32]), everything else BF16 / F32 / I64. Nothing
// is requantized: the engine consumes these bytes as they are.
//
// What differs from DeepSeek-V4.1 (models/dsv41): every compressing layer
// owns its cache (no shared kv sources, no candidate pool, no encoder /
// decoder split, no Engram); ratio 4 pools OVERLAPPING groups and selects
// with its own indexer, ratio 128 pools plain groups and attends to every
// compressed entry; the mHC is the two-pass form (a sublayer collapses
// with its own coefficients); the first `num_hash_layers` layers route by
// token id.
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm/moe.hpp"

namespace dgpp {

struct Dsv4Config {
  // --- model shape -------------------------------------------------------
  int hidden_size = 4096;
  int vocab_size = 129280;
  int num_hidden_layers = 43;
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  int max_position_embeddings = 1048576;
  int64_t bos_token_id = 0;
  int64_t eos_token_id = 1;

  // --- attention -----------------------------------------------------------
  int num_attention_heads = 64;
  int num_key_value_heads = 1;   // one 512-wide latent per token, K == V
  int head_dim = 512;            // the latent width, rope tail included
  int qk_rope_head_dim = 64;
  int q_lora_rank = 1024;
  int o_lora_rank = 1024;
  int o_groups = 8;
  bool attention_bias = false;
  int sliding_window = 128;
  double rope_theta = 10000.0;           // window-only layers, no scaling
  double compress_rope_theta = 160000.0; // every layer with compress_ratio > 0, with YaRN
  double rope_factor = 16.0;
  int original_max_position_embeddings = 65536;
  double beta_fast = 32.0;
  double beta_slow = 1.0;

  // --- compression schedule ------------------------------------------------
  // One entry per layer, the draft stages included: 0 = window only, r =
  // the layer's own KV compressed r-to-1 (4: overlapping groups, indexed;
  // 128: plain groups, every entry attended).
  std::vector<int> compress_ratios;
  int index_n_heads = 64;
  int index_head_dim = 128;
  int index_topk = 512;

  // --- mHC (two-pass) --------------------------------------------------
  int hc_mult = 4;
  int hc_sinkhorn_iters = 20;
  float hc_eps = 1e-6f;

  // --- MoE ------------------------------------------------------------------
  int moe_intermediate_size = 2048;
  int n_routed_experts = 256;
  int n_shared_experts = 1;
  int num_experts_per_tok = 6;
  bool norm_topk_prob = true;
  float routed_scaling_factor = 1.5f;
  float swiglu_limit = 10.0f;
  std::string scoring_func = "sqrtsoftplus";
  std::string topk_method = "noaux_tc";
  int num_hash_layers = 3;  // layers [0, n): expert ids from gate.tid2eid[token]

  // --- DSpark -------------------------------------------------------------
  // The draft stages (checkpoint prefix `mtp.S.`): the entries of
  // compress_ratios past the backbone (the config's own
  // num_nextn_predict_layers is a stale 1 in the 0731 release; the
  // reference reads `n_mtp_layers` 3 from inference/config.json).
  int draft_stages = 0;
  int dspark_block_size = 0;
  int64_t dspark_noise_token_id = -1;
  std::vector<int> dspark_target_layer_ids;
  int dspark_markov_rank = 0;

  // --- weight formats ----------------------------------------------------
  int fp8_block_size = 128;  // one e8m0 scale per 128x128 block of a dense fp8 matrix
  int fp4_block_size = 32;   // one e8m0 scale per 32 codes along K of an fp4 matrix

  // Parses config.json's root object. Throws std::runtime_error naming the
  // offending field on anything unsupported.
  static Dsv4Config parse(const minijson::Value& root);
  static Dsv4Config from_json_file(const std::string& path);

  // --- derived layer facts ------------------------------------------------
  // Layers are indexed 0..num_hidden_layers-1 (the backbone) and
  // num_hidden_layers..max_layer()-1 (the DSpark draft stages).
  int max_layer() const { return num_hidden_layers + draft_stages; }
  bool is_draft(int l) const { return l >= num_hidden_layers && l < max_layer(); }
  int draft_stage(int l) const { return is_draft(l) ? l - num_hidden_layers : -1; }
  int compress_ratio(int l) const { return compress_ratios[static_cast<size_t>(l)]; }
  // The overlapping compressor and the indexer both belong to ratio 4 (the
  // reference: `overlap = compress_ratio == 4`, `if compress_ratio == 4:
  // indexer`).
  bool overlapped(int l) const { return compress_ratio(l) == 4; }
  bool indexed(int l) const { return compress_ratio(l) == 4; }
  bool hashed(int l) const { return l >= 0 && l < num_hash_layers; }
  // The compressed-cache ordinal of a layer (-1: window only) and the index
  // (key) cache ordinal among the indexed layers.
  int cache_ordinal(int l) const;
  int index_ordinal(int l) const;
  int num_caches() const;
  int num_index_caches() const;
  bool is_dspark_target(int l) const;
  int shared_expert_inter() const { return n_shared_experts * moe_intermediate_size; }
  int qk_nope_head_dim() const { return head_dim - qk_rope_head_dim; }
  int heads_per_group() const { return num_attention_heads / o_groups; }
  int hc_coeff_rows() const { return (2 + hc_mult) * hc_mult; }

  // The routed chain's configuration (models/glm/moe.hpp): the sqrtsoftplus
  // router with its bias, the shared expert in the chain, the clamped
  // SwiGLU. `local_inter` is this rank's slice of moe_intermediate_size.
  GlmMoeConfig moe_config(int local_inter) const;
};

}  // namespace dgpp
