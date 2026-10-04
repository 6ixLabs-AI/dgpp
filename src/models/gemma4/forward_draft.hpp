#pragma once
// UNVERIFIED DRAFT (2026-10-04) — built only with -DDGPP_BUILD_GEMMA_DRAFT=ON.
//
// A plain GPU forward of the Gemma 4 text model for ONE sequence: the first
// numerical gate of the family on a Spark (apps/gemma4_forward_check), not
// an engine. Written with no GPU at hand: it has never been compiled by
// nvcc, has loaded no weight on a device and has produced no token there.
//
// What HAS run: the same source compiled as plain C++ (every kernel below is
// one independent thread per output element — no shared memory, no barrier —
// so a loop over thread indices executes it exactly), against the numpy
// reference's vectors on the synthetic checkpoint
// (tests/cuda/gemma4_forward_draft_test.cpp, on a Mac). That checks the
// kernels' arithmetic and the assembly; it says nothing about nvcc, launch
// geometry at real sizes, device memory, or speed.
//
// Shape of the thing: the checkpoint's own bytes resident — BF16 tensors as
// they are, the NVFP4 payloads and scales as they are, decoded inside the
// GEMV — fp32 activations, an fp32 K/V cache of every layer for the whole
// sequence (about 1.8 MiB a token for the 31B: this is a checker's cache,
// not a pool). The 26B-A4B's MoE block is in: the router, and each token's
// picked experts read straight from their NVFP4 sets (a layer's experts laid
// contiguous). No batching, no graph, no paging, no sampling, no MTP. One
// thread per output element means a GEMV row costs a serial K-long loop:
// correct first, slow by design (see gpu-steps.md for what to expect).
#include <cstdint>
#include <string>
#include <vector>

#include "models/gemma4/config.hpp"

namespace dgpp {

class Gemma4DraftForward {
 public:
  // Uploads the first `layers` layers (all when < 0) of the checkpoint in
  // `checkpoint_dir` and sizes the K/V cache for `max_tokens` positions.
  Gemma4DraftForward(const Gemma4TextConfig& cfg, const std::string& checkpoint_dir, int layers = -1,
                     int64_t max_tokens = 2048);
  ~Gemma4DraftForward();
  Gemma4DraftForward(const Gemma4DraftForward&) = delete;
  Gemma4DraftForward& operator=(const Gemma4DraftForward&) = delete;

  int layers() const;
  int64_t pos() const;             // tokens consumed so far
  size_t device_bytes() const;     // weights + K/V cache

  // Feeds `ids` at positions pos()..; returns the residual stream [T, H]
  // (fp32, host) after the last loaded layer, before the final norm. With
  // `trace`, the stream after the embedding and after every layer is
  // appended to it (layers() + 1 entries of [T, H]).
  std::vector<float> forward(const std::vector<int64_t>& ids, std::vector<std::vector<float>>* trace = nullptr);
  // The final norm, the tied head and the soft-cap on [T, H] residual rows:
  // [T, vocab] (fp32, host).
  std::vector<float> logits(const std::vector<float>& residual_rows);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace dgpp
