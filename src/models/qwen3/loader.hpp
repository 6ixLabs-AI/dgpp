#pragma once
// UNVERIFIED DRAFT (2026-10-04): written without access to a GPU or to nvcc.
// It has been syntax-checked against a CUDA header shim on a Mac and has
// never been compiled for a device, loaded a weight or run. Built only under
// -DDGPP_BUILD_QWEN3_PLAIN_DRAFT=ON (CMakeLists.txt);
// sixlabs/ports/qwen3-vl-30b-a3b/gpu-steps.md is the sequence that would
// verify it.
//
// Plain Qwen3-MoE weight loader on the shared ResidentLayerStream: the
// family supplies its tensor bindings, TP geometry and builders; the stream
// owns allocations, staging, resident images, byte accounting and digests.
// The resident forms are the ones the reused layer objects read:
//
//   attention  Glm4AttnResident (models/glm4/loader.hpp), no biases: q/k/v
//              row slices and o_proj column slices in BF16. A projection the
//              release ships in BF16 is sliced as it ships; one it ships in
//              NVFP4 (the modelopt release's o_proj, the compressed-tensors
//              release's q/k/v/o) is dequantized on the host to BF16 with the
//              qwen3_5 loader's two functions (qwen3next_fp4_dequant_bf16,
//              qwen35_fp4_packed_dequant_bf16 — models/qwen/loader35.hpp).
//   MoE        the BF16 router replicated; every routed expert's NVFP4
//              payload and block scales as shipped, sliced on the
//              intermediate dim (gate/up rows, down columns, 16-block
//              aligned), with each matrix's per-tensor scale stored as the
//              DIVISOR the expert kernels apply to the finished dot —
//              compressed-tensors' weight_global_scale as shipped, modelopt's
//              weight_scale_2 as its reciprocal. No shared expert, no bias.
//   globals    embed replicated, the final norm, lm_head vocab-sharded under
//              VocabSharded.
// Never read: the recipes' activation scales and K/V-cache scales, and the
// whole vision tower (`model.visual.*`: validated with the header, then left
// on disk).
//
// The Dense dialect (the retrieval models) has no builder here: the stream
// refuses it by name.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "loaders/resident_stream.hpp"
#include "loaders/safetensors.hpp"
#include "loaders/weight_build.hpp"
#include "models/glm4/loader.hpp"
#include "models/quant_matrix.hpp"
#include "models/qwen3/binding.hpp"
#include "models/qwen3/config.hpp"

namespace dgpp {

struct Qwen3MoeResident {
  const uint16_t* router = nullptr;     // BF16 [E, hidden]
  std::vector<GlmFp4Matrix> experts;    // [E * 3]: gate, up, down per expert (inter-sliced)
  int64_t local_inter = 0;              // I / W
  const GlmFp4Matrix& expert(int e, int i) const {
    return experts[static_cast<size_t>(e) * 3 + static_cast<size_t>(i)];
  }
};

struct Qwen3LayerResident {
  int layer = -1;
  const uint16_t* input_norm = nullptr;  // BF16 [hidden]
  const uint16_t* post_norm = nullptr;   // BF16 [hidden]
  Glm4AttnResident attn;                 // biases null
  Qwen3MoeResident moe;
  size_t bytes = 0;
};

struct Qwen3GlobalsResident {
  const uint16_t* embed = nullptr;       // BF16 [vocab, hidden]
  const uint16_t* final_norm = nullptr;  // BF16 [hidden]
  const uint16_t* lm_head = nullptr;     // BF16 [lm_vocab_count, hidden]
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  size_t bytes = 0;
};

// The local TP geometry at (rank, world): every slice bound the builders
// and the views use, in one place (GLM-4.7's rules).
struct Qwen3LocalGeometry {
  int world = 1, rank = 0;
  int local_heads = 0, head_begin = 0;
  int local_kv_heads = 0, kv_head_begin = 0;
  int64_t local_inter = 0;  // moe_intermediate_size / W
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  static Qwen3LocalGeometry from_config(const Qwen3TextConfig& cfg, int rank, int world,
                                        LoaderHeadSharding head);
};

// The family behind the shared stream (loaders/resident_stream.hpp).
struct Qwen3LoaderFamily {
  using Config = Qwen3TextConfig;
  using Expected = Qwen3ExpectedTensor;
  using LayerResident = Qwen3LayerResident;
  using GlobalsResident = Qwen3GlobalsResident;
  using Geometry = Qwen3LocalGeometry;
  using PresentMap = Qwen3PresentMap;
  struct Builder;  // models/qwen3/loader.cpp
  static const char* who() { return "qwen3 loader"; }
  static uint64_t loader_format() { return 1; }
  static int max_layer(const Config& c) { return c.num_hidden_layers; }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer) {
    return qwen3_expected_layer_tensors(c, layer);
  }
  static std::vector<Expected> global_table(const Config& c) { return qwen3_expected_global_tensors(c); }
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config&, const LoaderTensorMap&) {}
  static bool digest_included(const Expected& e);
  static bool discard_after_pack(const Expected& e);
  static void build_globals(const Config& c, const Geometry& geo, const LoaderTensorMap& tensors,
                            LayerBump& bump, GlobalsResident& out, uint64_t& source_bytes,
                            uint64_t& verbatim_bytes, LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t extra_resident_bytes(const Config&, int, int) { return 0; }
  static size_t min_staging_bytes() { return 0; }
  static void after_restore(const Config&, int, const LoaderTensorMap&, LayerResident&) {}
};

extern template class ResidentLayerStream<Qwen3LoaderFamily>;

class Qwen3LayerStream : public ResidentLayerStream<Qwen3LoaderFamily> {
 public:
  Qwen3LayerStream(const Qwen3TextConfig& cfg, const std::string& checkpoint_dir, int rank = 0,
                   int world = 1, LoaderResidency residency = LoaderResidency::Streaming,
                   LoaderHeadSharding head = LoaderHeadSharding::Full);
  ~Qwen3LayerStream() override = default;

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }
};

}  // namespace dgpp
