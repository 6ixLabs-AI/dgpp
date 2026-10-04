#pragma once
// Host reference of the plain Qwen3 family (2026-10-04): a checkpoint read
// on the host through the family's own config parser and binding table, and
// the whole forward pass over it — the C++ counterpart of
// tools/qwen3_reference.py, with no CUDA anywhere.
//
// Two arithmetics:
//   Exact   everything in double from the checkpoint's stored values: what
//           tools/qwen3_reference.py computes with `--dtype float64
//           --variant ropeangle=f64` (and, through tools/qwen3_synth.py
//           torch-check, what transformers' Qwen3 classes compute once their
//           float32 islands are taken in float64). The unit gate compares
//           the two at rounding-noise tolerance: it holds the binding table,
//           both NVFP4 containers' readers and every layout convention to
//           the numpy reference.
//   Engine  the rounding points the draft GPU walk has (models/qwen3/
//           model.hpp): a bf16 residual stream, the two-rounding RMSNorm,
//           glm4_ref's qkv finish (bf16 head norm and rotary from fp32
//           angles) and bf16-probability attention, NVFP4 attention
//           projections dequantized to bf16 as the loader stores them, the
//           routed chain of qwen3_moe_ref_forward. Tolerance-equal to Exact;
//           the oracle a fixture test of the GPU walk compares against. The
//           Dense dialect has no engine path and is refused.
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "loaders/safetensors.hpp"
#include "models/qwen3/binding.hpp"
#include "models/qwen3/config.hpp"
#include "models/qwen3/moe_reference.hpp"

namespace dgpp::qwen3_ref {

// A [rows, cols] block of an NVFP4 matrix as fp32 values, in the numpy
// reference's order of operations (each one fp32 rounding at most):
//   modelopt            code * (e4m3 * weight_scale_2)
//   compressed-tensors  code * (e4m3 / weight_global_scale)
// `payload` and `scales` point at the block's first byte with the source's
// row strides; cols is a multiple of 16; the low nibble is the even column.
void fp4_dequant_f32(const uint8_t* payload, size_t payload_stride, const uint8_t* scales,
                     size_t scale_stride, float global, bool global_divides, int64_t rows,
                     int64_t cols, float* out);

// The same block as the bf16 the loader makes resident for a dense
// projection (models/qwen/loader35.hpp's two functions, restated for the
// host library): modelopt bf16((code * e4m3) * weight_scale_2),
// compressed-tensors bf16(code * (e4m3 / weight_global_scale)).
void fp4_dequant_bf16(const uint8_t* payload, size_t payload_stride, const uint8_t* scales,
                      size_t scale_stride, float global, bool global_divides, int64_t rows,
                      int64_t cols, uint16_t* out);

class HostCheckpoint {
 public:
  // Parses DIR/config.json, maps every DIR/*.safetensors shard, takes the
  // names' prefix from the header and validates the whole header against
  // the binding table (throws std::runtime_error listing the first errors).
  // layers > 0 truncates the config to the first `layers` layers.
  explicit HostCheckpoint(const std::string& dir, int layers = -1);

  const Qwen3TextConfig& config() const { return cfg_; }
  const Qwen3BindReport& report() const { return report_; }
  size_t tensor_count() const { return tensors_.size(); }
  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const TensorInfo& tensor(const std::string& name) const;
  // The synthetic fixtures' order-independent content digest (tools/
  // qwen3_synth.py digest): the sum mod 2^64 over tensors of FNV-1a over
  // "name|DTYPE|d0,d1,..|" and the tensor's bytes.
  uint64_t content_digest() const;

  // A BF16 or F32 tensor widened to fp32.
  std::vector<float> dense(const std::string& name) const;
  // A Linear's [N, K] weight as fp32 whichever way the checkpoint stores it
  // (`base` is the module name): BF16 verbatim or NVFP4 (fp4_dequant_f32).
  std::vector<float> linear(const std::string& base, int64_t* rows = nullptr, int64_t* cols = nullptr) const;
  // The same weight as the bf16 bits the loader makes resident.
  std::vector<uint16_t> linear_bf16(const std::string& base, int64_t* rows = nullptr, int64_t* cols = nullptr) const;
  // One layer's routed experts in the oracle's form (the code and scale
  // bytes as shipped, each matrix's divisor).
  Qwen3MoeHostWeights experts(int layer) const;
  // The head's matrix name: lm_head.weight, or the embedding when tied.
  std::string head_name() const;

 private:
  bool is_fp4(const std::string& base) const;
  Qwen3TextConfig cfg_;
  Qwen3BindReport report_;
  std::vector<std::unique_ptr<SafetensorsFile>> shards_;
  std::unordered_map<std::string, const TensorInfo*> tensors_;
};

enum class Arithmetic { Exact, Engine };

struct Trace {
  // The residual stream [T * H] after the embedding and after every layer.
  std::vector<std::vector<double>> states;
  // Per MoE layer, every token's selected experts [T * top_k], ascending.
  std::vector<std::vector<int32_t>> router_ids;
  // The final-norm hidden state [T * H] (the head's input).
  std::vector<double> final_hidden;
};

// The cold forward over `ids`: logits [T * vocab_size]. Throws on a token
// id outside the vocabulary and — the VL dialect — on an image or video
// placeholder (the tower is not served).
std::vector<double> forward(const HostCheckpoint& ck, const std::vector<int64_t>& ids,
                            Arithmetic arithmetic = Arithmetic::Exact, Trace* trace = nullptr);

// log P(ids[i] | ids[<i]) for i in [1, T) from logits [T * vocab].
std::vector<double> teacher_forced_logprobs(const std::vector<double>& logits,
                                            const std::vector<int64_t>& ids, int vocab);

}  // namespace dgpp::qwen3_ref
