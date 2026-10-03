#pragma once
// DeepSeek-V4-Flash weight loader using ResidentLayerStream (2026-10-01).
// The family supplies tensor bindings, TP geometry and the builders for the
// release's own formats — fp8 e4m3 with e8m0 scales on 128 x 128 blocks
// (converted to the fp32 grid the fp8 core reads; exact), MXFP4 experts
// (e2m1 + e8m0 per 32, kept as they are), the mHC coefficient matrices
// rounded fp32 -> bf16 at load (the mHC kernels' form, as for V4.1), the
// hash layers' token -> expert table narrowed to int32 with every row
// sorted ascending (the chain's accumulation order). The shared stream
// owns allocations, staging, resident images, byte accounting and digests.
//
// Placement (every slice a formula in the world size W):
//   attention: wq_a and wkv replicated; wq_b rows per head block (64/W
//              heads); wo_a rows per output group (8/W groups); wo_b
//              packed columns of those groups; attn_sink per local head;
//              the norms replicated.
//   compressor, indexer (its wq_b, weights_proj and compressor): replicated
//              — every rank's cache rows, index keys and selections are
//              bitwise the others'.
//   MoE:       router, its bias and the hash table replicated; every routed
//              expert and the shared expert sliced on the intermediate dim
//              at 2048/W (w1/w3 rows, w2 columns).
//   mHC:       replicated (fn as bf16, base / scale fp32).
//   draft:     as a main layer; main_proj, main_norm, norm, the Markov
//              embedding, the Markov head, the confidence head and the
//              draft's head collapse replicated.
//   globals:   embed replicated (a row gather) or vocab-sharded, the final
//              norm and head collapse, head vocab-sharded under VocabSharded.
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/resident_stream.hpp"
#include "loaders/safetensors.hpp"
#include "loaders/weight_build.hpp"
#include "models/dsv4/binding.hpp"
#include "models/dsv4/compressor.hpp"
#include "models/dsv4/config.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

using Dsv4Residency = LoaderResidency;
using Dsv4HeadSharding = LoaderHeadSharding;

struct Dsv4AttnResident {
  GlmQuantMatrix wq_a;   // fp8 [q_lora, hidden], replicated
  GlmQuantMatrix wkv;    // fp8 [head_dim, hidden], replicated
  GlmQuantMatrix wq_b;   // fp8 [local_heads * head_dim, q_lora]
  GlmQuantMatrix wo_a;   // fp8 [local_groups * o_lora, heads_per_group * head_dim]
  GlmQuantMatrix wo_b;   // fp8 [hidden, local_groups * o_lora] (packed columns)
  const uint16_t* q_norm = nullptr;   // bf16 [q_lora]
  const uint16_t* kv_norm = nullptr;  // bf16 [head_dim]
  const float* attn_sink = nullptr;   // f32 [local_heads]
  Dsv4CompressorResident comp;        // ratio > 0
  // The indexer (ratio 4).
  GlmQuantMatrix idx_wq_b;            // fp8 [index_heads * 128, q_lora]
  const uint16_t* idx_wp = nullptr;   // bf16 [index_heads, hidden]
  Dsv4CompressorResident idx_comp;
  int local_heads = 0;
  int head_begin = 0;
  int local_groups = 0;
  int group_begin = 0;
};

struct Dsv4MoeResident {
  const uint16_t* router = nullptr;     // bf16 [E, hidden]
  const float* router_bias = nullptr;   // f32 [E] (null on a hash layer)
  const int32_t* tid2eid = nullptr;     // i32 [vocab, top_k], rows ascending (hash layers)
  GlmQuantMatrix shared[3];             // fp8: w1, w3 [S/W, hidden]; w2 [hidden, S/W]
  std::vector<GlmFp4Matrix> experts;    // [E * 3]: w1, w3, w2 per expert (MXFP4, inter-sliced)
  int n_experts = 0;
  int64_t local_inter = 0;              // I/W
  int64_t local_shared_inter = 0;       // S/W
};

struct Dsv4MhcResident {
  const uint16_t* attn_fn = nullptr;  // bf16 [24, 4 * hidden] (fp32 in the file, rounded at load)
  const float* attn_base = nullptr;   // f32 [24]
  const float* attn_scale = nullptr;  // f32 [3]
  const uint16_t* ffn_fn = nullptr;
  const float* ffn_base = nullptr;
  const float* ffn_scale = nullptr;
};

// The head's collapse (hc_head): pre = sigmoid(mix * scale + base) + eps —
// the mHC site's `pre` rule exactly, so it is held in the mHC kernel's
// shape with the post / comb rows zero: fn bf16 [24, 4 * hidden] (rows 4..23
// zero), base f32 [24] (4..23 zero), scale f32 [3] = (scale, 0, 0). The
// kernel's collapse output is the head's; its post / comb are unused.
struct Dsv4HcHeadResident {
  const uint16_t* fn = nullptr;
  const float* base = nullptr;
  const float* scale = nullptr;
};

struct Dsv4DraftResident {
  GlmQuantMatrix main_proj;                 // stage 0: fp8 [hidden, targets * hidden]
  const uint16_t* main_norm = nullptr;      // stage 0: bf16 [hidden]
  const uint16_t* norm = nullptr;           // last stage: bf16 [hidden]
  const uint16_t* markov_embed = nullptr;   // last stage: bf16 [vocab, rank], replicated
  const uint16_t* markov_head = nullptr;    // last stage: bf16 [vocab, rank], replicated
  const float* confidence = nullptr;        // last stage: f32 [hidden + rank]
  Dsv4HcHeadResident hc_head;               // last stage
};

struct Dsv4LayerResident {
  int layer = -1;
  const uint16_t* attn_norm = nullptr;  // bf16 [hidden]
  const uint16_t* ffn_norm = nullptr;   // bf16 [hidden]
  Dsv4AttnResident attn;
  Dsv4MoeResident moe;
  Dsv4MhcResident mhc;
  Dsv4DraftResident draft;
  size_t bytes = 0;
};

struct Dsv4GlobalsResident {
  const uint16_t* embed = nullptr;       // bf16 [embed_vocab_count, hidden]
  int embed_vocab_begin = 0;
  int embed_vocab_count = 0;
  const uint16_t* final_norm = nullptr;  // bf16 [hidden]
  const uint16_t* lm_head = nullptr;     // bf16 [lm_vocab_count, hidden]
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  Dsv4HcHeadResident hc_head;
  size_t bytes = 0;
};

// The local TP geometry at (rank, world): every slice bound the builders
// and the views use, in one place.
struct Dsv4LocalGeometry {
  int world = 1, rank = 0;
  int local_heads = 0, head_begin = 0;      // attention heads
  int local_groups = 0, group_begin = 0;    // wo_a / wo_b output groups
  int64_t local_inter = 0;                  // moe_intermediate_size / W
  int64_t local_shared_inter = 0;           // shared_expert_inter() / W
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  int embed_vocab_begin = 0, embed_vocab_count = 0;
  static Dsv4LocalGeometry from_config(const Dsv4Config& cfg, int rank, int world, Dsv4HeadSharding head);
};

// The family behind the shared stream (loaders/resident_stream.hpp).
struct Dsv4LoaderFamily {
  using Config = Dsv4Config;
  using Expected = Dsv4ExpectedTensor;
  using LayerResident = Dsv4LayerResident;
  using GlobalsResident = Dsv4GlobalsResident;
  using Geometry = Dsv4LocalGeometry;
  using PresentMap = std::unordered_map<std::string, Dsv4TensorDesc>;
  struct Builder;  // models/dsv4/loader.cpp
  static const char* who() { return "dsv4 loader"; }
  static uint64_t loader_format() { return 1; }
  static int max_layer(const Config& c) { return c.max_layer(); }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer) { return dsv4_expected_layer_tensors(c, layer); }
  static std::vector<Expected> global_table(const Config& c) { return dsv4_expected_global_tensors(c); }
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config&, const LoaderTensorMap&) {}
  static bool digest_included(const Expected& e);
  static bool discard_after_pack(const Expected&) { return false; }
  static void build_globals(const Config& c, const Geometry& geo, const LoaderTensorMap& tensors, LayerBump& bump,
                            GlobalsResident& out, uint64_t& source_bytes, uint64_t& verbatim_bytes,
                            LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t extra_resident_bytes(const Config&, int, int) { return 0; }
  static size_t min_staging_bytes() { return 0; }
  static void after_restore(const Config&, int, const LoaderTensorMap&, LayerResident&) {}
};

extern template class ResidentLayerStream<Dsv4LoaderFamily>;

class Dsv4LayerStream : public ResidentLayerStream<Dsv4LoaderFamily> {
 public:
  Dsv4LayerStream(const Dsv4Config& cfg, const std::string& checkpoint_dir, int rank = 0, int world = 1,
                  Dsv4Residency residency = Dsv4Residency::Streaming, Dsv4HeadSharding head = Dsv4HeadSharding::Full,
                  bool resident_mtp = false);
  ~Dsv4LayerStream() override = default;

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();
  // The deployment's `engine.embed_sharding` ("vocab": each rank holds its
  // lm-head slice of the embedding; "replicated": the whole table).
  static void set_embed_vocab_sharded(bool on);
  static bool embed_vocab_sharded();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }
};

}  // namespace dgpp
