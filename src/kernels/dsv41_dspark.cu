#include "kernels/dsv41_dspark.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {

__global__ void stream_mean_kernel(const uint16_t* __restrict__ streams, int hc_mult, int hidden, int rows,
                                   uint16_t* __restrict__ out, int64_t out_stride) {
  const int t = blockIdx.x;
  if (t >= rows) return;
  const uint16_t* x = streams + static_cast<size_t>(t) * static_cast<size_t>(hc_mult) * hidden;
  const float inv = 1.0f / static_cast<float>(hc_mult);
  for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
    float s = 0.f;
    for (int j = 0; j < hc_mult; ++j) s += bf16_bits_to_float(x[static_cast<size_t>(j) * hidden + d]);
    out[static_cast<size_t>(t) * out_stride + d] = float_to_bf16_bits(s * inv);
  }
}

__global__ void block_rows_kernel(const int64_t* __restrict__ step_pos, const int64_t* __restrict__ tokens,
                                  const int32_t* __restrict__ req_ids, int groups, int rows_per_group, int block,
                                  int64_t noise_id, int64_t* __restrict__ pos_out, int64_t* __restrict__ tok_out,
                                  int32_t* __restrict__ req_out, int32_t* __restrict__ spans_out) {
  const int g = blockIdx.x;
  if (g >= groups) return;
  // One block per group; thread 0 scans the group's rows (rows_per_group <= 32).
  __shared__ int64_t s_p;
  __shared__ int64_t s_next;
  if (threadIdx.x == 0) {
    int64_t best = -1;
    int64_t next = noise_id;
    for (int r = 0; r < rows_per_group; ++r) {
      const int64_t p = step_pos[g * rows_per_group + r];
      if (p > best) {
        best = p;
        next = tokens[g * rows_per_group + r];
      }
    }
    s_p = best < 0 ? -1 : best + 1;
    s_next = next;
    spans_out[g] = g * block;
    if (g == 0) spans_out[groups] = groups * block;
  }
  __syncthreads();
  for (int k = threadIdx.x; k < block; k += blockDim.x) {
    const int row = g * block + k;
    pos_out[row] = s_p < 0 ? -1 : s_p + k;
    tok_out[row] = k == 0 ? s_next : noise_id;
    req_out[row] = req_ids[g * rows_per_group];
  }
}

constexpr int kMaxRank = 512;

__global__ void markov_bias_kernel(const float* __restrict__ base, int64_t base_group_stride, int block_row,
                                   const uint16_t* __restrict__ markov_embed, const uint16_t* __restrict__ markov_head,
                                   int rank, int vocab_begin, int count, const int64_t* __restrict__ tok,
                                   int tok_stride, float* __restrict__ out, int64_t out_group_stride, int rows_out,
                                   float* __restrict__ block_max) {
  const int g = blockIdx.y;
  __shared__ float e[kMaxRank];
  __shared__ float s_max[8];
  // A padding group's token (-1: a closed slot) biases nothing.
  const int64_t t = tok[static_cast<size_t>(g) * tok_stride];
  for (int r = threadIdx.x; r < rank; r += blockDim.x)
    e[r] = t < 0 ? 0.f : bf16_bits_to_float(markov_embed[static_cast<size_t>(t) * rank + r]);
  __syncthreads();
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  const bool live = v < count;
  if (!live && block_max == nullptr) return;
  const uint16_t* h = markov_head + static_cast<size_t>(vocab_begin + (live ? v : 0)) * rank;
  float acc = 0.f;
  if (t >= 0 && live) {
    // The row's chain in r order, its loads as 16-byte vectors with a whole
    // 64-element span in flight (until 2026-10-01 one bf16 load at a time:
    // 100 us a launch for a 16.5 MB slice the second and third chain rows
    // read from L2). The same FMAs in the same order: bitwise.
    int r = 0;
    if ((rank % 8) == 0 && (reinterpret_cast<uintptr_t>(h) & 15u) == 0) {
      const uint4* hv = reinterpret_cast<const uint4*>(h);
      constexpr int kVecs = 8;
      for (; r + 8 * kVecs <= rank; r += 8 * kVecs) {
        uint4 w[kVecs];
#pragma unroll
        for (int b = 0; b < kVecs; ++b) w[b] = hv[r / 8 + b];
#pragma unroll
        for (int b = 0; b < kVecs; ++b) {
          const uint32_t q[4] = {w[b].x, w[b].y, w[b].z, w[b].w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            acc = fmaf(e[r + 8 * b + 2 * i], bf16_bits_to_float(static_cast<uint16_t>(q[i] & 0xFFFFu)), acc);
            acc = fmaf(e[r + 8 * b + 2 * i + 1], bf16_bits_to_float(static_cast<uint16_t>(q[i] >> 16)), acc);
          }
        }
      }
    }
    for (; r < rank; ++r) acc = fmaf(e[r], bf16_bits_to_float(h[r]), acc);
  }
  float val = -INFINITY;
  if (live) {
    const float b = base[static_cast<size_t>(g) * base_group_stride + static_cast<size_t>(block_row) * count + v];
    val = b + acc;
    float* o = out + static_cast<size_t>(g) * out_group_stride + v;
    for (int j = 0; j < rows_out; ++j) o[static_cast<size_t>(j) * count] = val;
  }
  if (block_max != nullptr) {
    float m = val;
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if ((threadIdx.x & 31) == 0) s_max[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
      float bm = s_max[0];
#pragma unroll
      for (int w = 1; w < 8; ++w) bm = fmaxf(bm, s_max[w]);
      block_max[static_cast<size_t>(g) * gridDim.x + blockIdx.x] = bm;
    }
  }
}

// Block (x, y): entries [256 x, 256 x + 256) of group y's biased row. The
// row maximum from the per-block maxima; the entries within delta of it,
// compacted in id order (ballots and prefix counts: the same list every
// run), each recomputed by the whole block — thread t owns elements
// [8 t + 2048 j, ...) of the dot, the partials folded in a fixed order.
__global__ void rescore_kernel(float* __restrict__ biased, int64_t out_group_stride, int rows_out,
                               const float* __restrict__ base, int64_t base_group_stride, int block_row, int count,
                               const uint16_t* __restrict__ h, int64_t h_group_stride, int hidden,
                               const uint16_t* __restrict__ head, const float* __restrict__ block_max, float delta) {
  __shared__ int s_cnt[8];
  __shared__ int s_list[kDsparkRescoreCap];
  __shared__ float s_red[8];
  __shared__ float s_val[256];
  const int g = blockIdx.y;
  const int v = blockIdx.x * 256 + threadIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < static_cast<int>(gridDim.x); i += 256)
    m = fmaxf(m, block_max[static_cast<size_t>(g) * gridDim.x + i]);
  for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  if (lane == 0) s_red[warp] = m;
  __syncthreads();
  float row_max = s_red[0];
#pragma unroll
  for (int w = 1; w < 8; ++w) row_max = fmaxf(row_max, s_red[w]);
  float* brow = biased + static_cast<size_t>(g) * out_group_stride;
  const float val = v < count ? brow[v] : -INFINITY;
  const bool hit = v < count && val >= row_max - delta;
  const unsigned mask = __ballot_sync(0xffffffffu, hit);
  __syncthreads();  // every thread has read s_red
  if (lane == 0) s_cnt[warp] = __popc(mask);
  s_val[threadIdx.x] = hit ? val : -INFINITY;
  __syncthreads();
  int before = 0, total = 0;
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    before += w < warp ? s_cnt[w] : 0;
    total += s_cnt[w];
  }
  if (total == 0) return;
  if (hit) {
    // The tile's candidates in id order; a flat row with more of them than
    // the cap keeps the HIGHEST ones (rank by value, ties to the lower id),
    // so the row's maximum is always among the recomputed.
    int i = before + __popc(mask & ((1u << lane) - 1u));
    if (total > kDsparkRescoreCap) {
      i = 0;
      for (int j = 0; j < 256; ++j) {
        const float o = s_val[j];
        i += (o > val || (o == val && j < static_cast<int>(threadIdx.x))) ? 1 : 0;
      }
    }
    if (i < kDsparkRescoreCap) s_list[i] = v;
  }
  __syncthreads();
  const int n = total < kDsparkRescoreCap ? total : kDsparkRescoreCap;
  const uint16_t* hr = h + static_cast<size_t>(g) * h_group_stride + static_cast<size_t>(block_row) * hidden;
  const float* base_row = base + static_cast<size_t>(g) * base_group_stride + static_cast<size_t>(block_row) * count;
  for (int i = 0; i < n; ++i) {
    const int tok = s_list[i];
    const uint16_t* w = head + static_cast<size_t>(tok) * hidden;
    float acc = 0.f;
    for (int e = threadIdx.x * 8; e < hidden; e += 256 * 8) {
      const uint4 wv = *reinterpret_cast<const uint4*>(w + e);
      const uint4 hv = *reinterpret_cast<const uint4*>(hr + e);
      const uint32_t wq[4] = {wv.x, wv.y, wv.z, wv.w};
      const uint32_t hq[4] = {hv.x, hv.y, hv.z, hv.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        acc = fmaf(bf16_bits_to_float(static_cast<uint16_t>(hq[q] & 0xFFFFu)),
                   bf16_bits_to_float(static_cast<uint16_t>(wq[q] & 0xFFFFu)), acc);
        acc = fmaf(bf16_bits_to_float(static_cast<uint16_t>(hq[q] >> 16)),
                   bf16_bits_to_float(static_cast<uint16_t>(wq[q] >> 16)), acc);
      }
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) s_red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
      float exact = 0.f;
#pragma unroll
      for (int w2 = 0; w2 < 8; ++w2) exact += s_red[w2];
      const float fixed = brow[tok] - base_row[tok] + exact;
      for (int j = 0; j < rows_out; ++j) brow[static_cast<size_t>(j) * count + tok] = fixed;
    }
    __syncthreads();
  }
}

__global__ void confidence_kernel(const uint16_t* __restrict__ x, int64_t x_group_stride, int block_row, int hidden,
                                  const uint16_t* __restrict__ markov_embed, int rank, const int64_t* __restrict__ tok,
                                  int tok_stride, const float* __restrict__ w, float* __restrict__ conf_out,
                                  int conf_stride) {
  const int g = blockIdx.x;
  const uint16_t* xr = x + static_cast<size_t>(g) * x_group_stride + static_cast<size_t>(block_row) * hidden;
  const int64_t t = tok[static_cast<size_t>(g) * tok_stride];
  const uint16_t* e = markov_embed + static_cast<size_t>(t < 0 ? 0 : t) * rank;
  float acc = 0.f;
  if (t >= 0) {
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) acc = fmaf(w[d], bf16_bits_to_float(xr[d]), acc);
    for (int r = threadIdx.x; r < rank; r += blockDim.x) acc = fmaf(w[hidden + r], bf16_bits_to_float(e[r]), acc);
  }
  __shared__ float red[256];
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) conf_out[static_cast<size_t>(g) * conf_stride] = red[0];
}

}  // namespace

void dsv41_stream_mean_bf16(const uint16_t* streams, int hc_mult, int hidden, int rows, uint16_t* out,
                            int64_t out_stride, cudaStream_t stream) {
  if (rows <= 0) return;
  if (!streams || !out || hc_mult <= 0 || hidden <= 0 || out_stride < hidden)
    throw std::invalid_argument("dsv41_stream_mean_bf16: arguments");
  stream_mean_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(streams, hc_mult, hidden, rows, out, out_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv41_dspark_block_rows(const int64_t* step_pos, const int64_t* tokens, const int32_t* req_ids, int groups,
                             int rows_per_group, int block, int64_t noise_id, int64_t* pos_out, int64_t* tok_out,
                             int32_t* req_out, int32_t* spans_out, cudaStream_t stream) {
  if (!step_pos || !tokens || !req_ids || !pos_out || !tok_out || !req_out || !spans_out)
    throw std::invalid_argument("dsv41_dspark_block_rows: null buffer");
  if (groups <= 0 || rows_per_group <= 0 || rows_per_group > 32 || block <= 0 || block > 32)
    throw std::invalid_argument("dsv41_dspark_block_rows: shape");
  block_rows_kernel<<<static_cast<unsigned>(groups), 32, 0, stream>>>(step_pos, tokens, req_ids, groups, rows_per_group,
                                                                       block, noise_id, pos_out, tok_out, req_out,
                                                                       spans_out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv41_dspark_markov_bias(const float* base, int64_t base_group_stride, int block_row, const uint16_t* markov_embed,
                              const uint16_t* markov_head, int rank, int vocab_begin, int count, const int64_t* tok,
                              int tok_stride, int groups, float* out, int64_t out_group_stride, int rows_out,
                              cudaStream_t stream, float* block_max) {
  if (!base || !markov_embed || !markov_head || !tok || !out) throw std::invalid_argument("dsv41_dspark_markov_bias: null buffer");
  if (rank <= 0 || rank > kMaxRank || count <= 0 || groups <= 0 || rows_out <= 0 || block_row < 0 || tok_stride <= 0)
    throw std::invalid_argument("dsv41_dspark_markov_bias: shape");
  const dim3 grid(static_cast<unsigned>((count + 255) / 256), static_cast<unsigned>(groups));
  markov_bias_kernel<<<grid, 256, 0, stream>>>(base, base_group_stride, block_row, markov_embed, markov_head, rank,
                                               vocab_begin, count, tok, tok_stride, out, out_group_stride, rows_out,
                                               block_max);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv41_dspark_rescore(float* biased, int64_t out_group_stride, int rows_out, const float* base,
                          int64_t base_group_stride, int block_row, int count, const uint16_t* h,
                          int64_t h_group_stride, int hidden, const uint16_t* head, const float* block_max,
                          float delta, int groups, cudaStream_t stream) {
  if (groups <= 0) return;
  if (!biased || !base || !h || !head || !block_max) throw std::invalid_argument("dsv41_dspark_rescore: null buffer");
  if (count <= 0 || rows_out <= 0 || block_row < 0 || hidden <= 0 || hidden % 8 != 0 || h_group_stride % 8 != 0 ||
      !(delta >= 0.f) || (reinterpret_cast<uintptr_t>(h) & 15u) != 0 || (reinterpret_cast<uintptr_t>(head) & 15u) != 0)
    throw std::invalid_argument("dsv41_dspark_rescore: aligned rows of a multiple of 8 elements");
  const dim3 grid(static_cast<unsigned>((count + 255) / 256), static_cast<unsigned>(groups));
  rescore_kernel<<<grid, 256, 0, stream>>>(biased, out_group_stride, rows_out, base, base_group_stride, block_row, count,
                                           h, h_group_stride, hidden, head, block_max, delta);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv41_dspark_confidence(const uint16_t* x, int64_t x_group_stride, int block_row, int hidden,
                             const uint16_t* markov_embed, int rank, const int64_t* tok, int tok_stride,
                             const float* w, int groups, float* conf_out, int conf_stride, cudaStream_t stream) {
  if (!x || !markov_embed || !tok || !w || !conf_out) throw std::invalid_argument("dsv41_dspark_confidence: null buffer");
  if (hidden <= 0 || rank <= 0 || groups <= 0 || block_row < 0 || tok_stride <= 0 || conf_stride <= 0)
    throw std::invalid_argument("dsv41_dspark_confidence: shape");
  confidence_kernel<<<static_cast<unsigned>(groups), 256, 0, stream>>>(x, x_group_stride, block_row, hidden, markov_embed,
                                                                        rank, tok, tok_stride, w, conf_out, conf_stride);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
