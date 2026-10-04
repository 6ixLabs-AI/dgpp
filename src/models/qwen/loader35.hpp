#pragma once
// Qwen3.5-27B loader family: FP8-native weights (the checkpoint already holds
// e4m3 payload + BF16 scales — no BF16→FP8 re-encode at load), text-only,
// dense SwiGLU MLP, standard pre-norm residual
// (input_layernorm/post_attention_layernorm).
//
// The Qwen3Next dialect (Qwen3-Next-80B-A3B, the modelopt NVFP4 release)
// fills the same residents in their BF16 forms, every `_fp8` member empty:
//   GDN:  the fused, per-key-head interleaved in_proj_qkvz / in_proj_ba
//         gathered into the split head-major in_proj_qkv, in_proj_z,
//         in_proj_a and in_proj_b; the NVFP4 out_proj dequantized to BF16.
//   Full: q/k/v BF16 row slices; the NVFP4 o_proj dequantized to BF16 (the
//         draft layer's is BF16 already); the recipe's K/V-cache scales kept.
//   MoE:  Flash-Next's resident (`moe`, `mlp` empty) — router and shared
//         gate replicated, the shared expert BF16 (dequantized in the
//         backbone), the routed experts NVFP4 as shipped in the backbone
//         and block FP8 encoded from BF16 in the draft layer.
// Its slices follow the family's rules at every world: key and value heads,
// query heads and their kv head(s), I/W rows of gate/up and columns of down.
// Under Qwen35LayerStream::set_dense_weights_fp8 the same dialect binds its
// dense projections through the `_fp8` members instead (the BF16 pointers
// null), encoded to block FP8 at load from the BF16 form's values: the GDN
// in_proj_qkv / in_proj_z / out_proj, the attention q/k/v/o and the shared
// expert. Everything else is unchanged.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "loaders/resident_stream.hpp"
#include "loaders/weight_build.hpp"
#include "models/quant_matrix.hpp"
#include "models/qwen/binding.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/loader.hpp"

namespace dgpp {

// Dense SwiGLU MLP resident (BF16 or block-FP8 weights).
struct Qwen35DenseMlpResident {
  const uint16_t* gate = nullptr;  // BF16 [I, H]
  const uint16_t* up = nullptr;    // BF16 [I, H]
  const uint16_t* down = nullptr;  // BF16 [H, I]
  GlmQuantMatrix gate_fp8, up_fp8, down_fp8;  // dense_weights fp8
};

struct Qwen35LayerResident {
  Qwen35LayerKind kind = Qwen35LayerKind::Gdn;
  int layer = -1;  // set by Builder::build_layer (the stream checks slot.layer)
  const uint16_t* input_norm = nullptr;  // BF16 [H]
  const uint16_t* post_norm = nullptr;   // BF16 [H]
  QwenGdnResident gdn;                   // kind == Gdn
  QwenFullAttnResident full;             // kind == Full
  Qwen35DenseMlpResident mlp;            // every layer (the dense dialect)
  // The Qwen3Next dialect's routed MoE (cfg.moe()): the Flash-Next block's
  // resident, `mlp` left empty. Backbone layers carry the NVFP4 experts
  // (experts_fp4); the draft layer's BF16 experts are encoded to block FP8
  // at load (experts), as Flash-Next's bf16_fused draft experts are.
  QwenMoeResident moe;
  // The Qwen3Next recipe's FP8 K/V-cache scales of a backbone attention
  // layer (`k_proj.k_scale`, `v_proj.v_scale`): [k, v] in the layer image
  // and their host copies. Nothing reads them yet. Null and 0 on GDN
  // layers, the draft layer and the Qwen3.5 dialect.
  const float* kv_cache_scales = nullptr;  // F32 [2], device
  float k_cache_scale = 0.0f, v_cache_scale = 0.0f;
  size_t bytes = 0;  // set by the stream (bump cursor after build)
};

struct Qwen35GlobalsResident {
  const uint16_t* embed = nullptr;       // BF16 [vocab, H]
  const uint16_t* final_norm = nullptr;  // BF16 [H]
  const uint16_t* lm_head = nullptr;     // BF16 [V/W, H] vocab shard
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  // MTP draft head (BF16, replicated): fused fc [H, 2H] over
  // cat(pre_fc_norm_embedding(embed), pre_fc_norm_hidden(hidden)) plus the
  // draft layer's final norm. Null when the config has no draft layer.
  const uint16_t* mtp_fc = nullptr;                 // BF16 [H, 2H]
  const uint16_t* mtp_norm = nullptr;               // BF16 [H]
  const uint16_t* mtp_pre_fc_norm_embedding = nullptr;  // BF16 [H]
  const uint16_t* mtp_pre_fc_norm_hidden = nullptr;     // BF16 [H]
  size_t bytes = 0;  // set by the stream (globals bump cursor after build)
};

// The local TP geometry at (rank, world): every slice bound the builders use.
struct Qwen35LocalGeometry {
  int world = 1, rank = 0;
  int local_key_heads = 0, local_value_heads = 0;  // GDN
  int local_heads = 0, head_begin = 0;             // Full query heads
  int local_kv_heads = 0, kv_head_begin = 0;       // Full kv heads
  int64_t local_inter = 0;                         // MLP I/W
  int64_t local_moe_inter = 0;                     // routed experts' I/W (the MoE dialect)
  int64_t local_shared_inter = 0;                  // shared expert's S/W
  int lm_vocab_begin = 0, lm_vocab_count = 0;      // lm head slice
  static Qwen35LocalGeometry from_config(const Qwen35TextConfig& cfg, int rank, int world,
                                          LoaderHeadSharding head);
};

// The Qwen3Next dialect's GDN gather. The checkpoint fuses the input
// projections and interleaves their rows per key head: in_proj_qkvz is one
// group per key head g of [q(g) dk | k(g) dk | v(g*r ..) r*dv | z(g*r ..) r*dv]
// rows (r = value heads per key head), in_proj_ba one group of
// [b(g*r ..) r | a(g*r ..) r] rows. The layer binds the split, head-major
// matrices, so the loader gathers them: each run is a contiguous source row
// range and the destination row it lands on, for this rank's key heads
// [rank * lk, +lk) and their value heads.
struct Qwen35RowRun {
  int64_t src_row = 0, rows = 0, dst_row = 0;
};
struct Qwen3NextGdnGather {
  std::vector<Qwen35RowRun> qkv;  // in_proj_qkvz -> in_proj_qkv [lk*dk | lk*dk | lv*dv]
  std::vector<Qwen35RowRun> z;    // in_proj_qkvz -> in_proj_z [lv*dv]
  std::vector<Qwen35RowRun> a;    // in_proj_ba -> in_proj_a [lv]
  std::vector<Qwen35RowRun> b;    // in_proj_ba -> in_proj_b [lv]
};
Qwen3NextGdnGather qwen3next_gdn_gather(const Qwen35TextConfig& cfg, int rank, int world);

// A [rows, cols] block of a modelopt NVFP4 matrix dequantized to BF16 (the
// Qwen3Next dialect's out_proj, o_proj and shared expert, which the dense
// BF16 kernels read). `payload` and `scales` point at the block's first
// byte — two e2m1 codes a byte, the low nibble the even column; one e4m3
// scale per 16 columns — with the source's row strides in bytes; the block
// starts on a 16-column boundary and cols is a multiple of 16. Element
// (n, k) is bf16(e2m1(code) * e4m3(scale[n][k / 16]) * weight_scale_2): the
// code x block-scale product is exact in fp32, the per-tensor scale
// multiplies it (one fp32 rounding), and the result rounds to bf16, both to
// nearest even.
void qwen3next_fp4_dequant_bf16(const uint8_t* payload, size_t payload_stride,
                                const uint8_t* scales, size_t scale_stride, float weight_scale_2,
                                int64_t rows, int64_t cols, uint16_t* out);

// The family behind the shared stream (loaders/resident_stream.hpp).
struct Qwen35LoaderFamily {
  using Config = Qwen35TextConfig;
  using Expected = QwenExpectedTensor;
  using LayerResident = Qwen35LayerResident;
  using GlobalsResident = Qwen35GlobalsResident;
  using Geometry = Qwen35LocalGeometry;
  using PresentMap = std::unordered_map<std::string, QwenTensorDesc>;
  struct Builder;  // models/qwen/loader35.cpp
  static const char* who();
  // Fresh family: layout version 1 (no old images to be compatible with).
  static uint64_t loader_format();
  static int max_layer(const Config& c);
  static int main_layers(const Config& c);
  static std::vector<Expected> layer_table(const Config& c, int layer);
  static std::vector<Expected> global_table(const Config& c);
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config& c, const LoaderTensorMap& tensors);
  static bool digest_included(const Expected& e);
  static bool discard_after_pack(const Expected& e);
  static void build_globals(const Config& c, const Geometry& geo, const LoaderTensorMap& tensors,
                            LayerBump& bump, GlobalsResident& out, uint64_t& source_bytes,
                            uint64_t& verbatim_bytes, LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t extra_resident_bytes(const Config& c, int rank, int world);
  static size_t min_staging_bytes() { return size_t{256} << 20; }
  static void after_restore(const Config& c, int layer, const LoaderTensorMap& tensors,
                            LayerResident& out);
};

struct Qwen35LayerStream : ResidentLayerStream<Qwen35LoaderFamily> {
  Qwen35LayerStream(const Qwen35TextConfig& cfg, const std::string& checkpoint_dir,
                    int rank = 0, int world = 1,
                    LoaderResidency residency = LoaderResidency::Streaming,
                    LoaderHeadSharding head = LoaderHeadSharding::Full,
                    bool resident_mtp = false);
  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();
  const std::string& image_dir() const override;
  // The Qwen3Next dialect's dense stack form (engine.dense_weights): false =
  // BF16 (the default); true = every dense projection (GDN qkv/z/out, the
  // attention q/k/v/o, the shared expert) encoded to block FP8 at load, the
  // form the Qwen3.5 release ships and the model's fp8 paths read. Process-
  // wide; set before the stream is built — the byte formulas follow it, and
  // it is part of loader_format(), so a resident image of one form is never
  // restored for the other. The Qwen3.5 dialect is FP8 as shipped either way.
  static void set_dense_weights_fp8(bool on);
  static bool dense_weights_fp8();
};

extern template class ResidentLayerStream<Qwen35LoaderFamily>;

}  // namespace dgpp
