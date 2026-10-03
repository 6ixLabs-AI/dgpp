#pragma once
// A gated-pooling compressor's weights (the reference `Compressor`):
// device-visible pointers, CUDA-free so the loader and the attention layer
// share the type.
#include <cstdint>

namespace dgpp {

struct Dsv4CompressorResident {
  const uint16_t* wkv = nullptr;    // bf16 [wide, hidden]
  const uint16_t* wgate = nullptr;  // bf16 [wide, hidden]
  const float* ape = nullptr;       // f32 [ratio, wide]
  const uint16_t* norm = nullptr;   // bf16 [dim]
  int ratio = 0;
  int dim = 0;   // the pooled entry's width (512 main, 128 index)
  int wide = 0;  // the projection width: 2 * dim when the groups overlap
  bool present() const { return wkv != nullptr; }
};

}  // namespace dgpp
