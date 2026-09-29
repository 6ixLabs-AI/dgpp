#include "kernels/l2_prefetch.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {
namespace {

constexpr int kThreads = 256;
// Full: 48 blocks x 4 loads x 16 B x 256 threads = 768 KB in flight — line
// rate (~205 GB/s measured) at ~3 us of added latency for other traffic.
// Light: 16 x 2 = 128 KB in flight — ~165 GB/s, well under a microsecond
// of queue. (96 x 8 = 3 MB reached 215 GB/s and made every memory access
// on the box wait ~12 us: the bus handshake went 9 -> 26 us.)
constexpr int kFullBlocks = 48;
constexpr int kFullUnroll = 4;
constexpr int kLightUnroll = 2;
constexpr int kLightBlocksDefault = 16;
// DGPP_L2_PREFETCH_LIGHT_BLOCKS: the Light rate's grid (tuning knob).
int light_blocks() {
  static const int blocks = [] {
    const char* v = std::getenv("DGPP_L2_PREFETCH_LIGHT_BLOCKS");
    if (v == nullptr) return kLightBlocksDefault;
    const long b = std::strtol(v, nullptr, 10);
    return (b >= 1 && b <= 96) ? static_cast<int>(b) : kLightBlocksDefault;
  }();
  return blocks;
}

// A device word the kernel compares its fold against. Nothing sets it and
// the fold is arbitrary, so the compare (almost) never matches — and when
// it does, the store is a harmless write to a sink. The compiler cannot
// prove either, so the loads survive optimization.
__device__ uint32_t g_prefetch_sink[2];

template <int kUnroll>
__global__ __launch_bounds__(kThreads) void l2_prefetch_kernel(
    const uint4* __restrict__ p, size_t vec_count) {
  const size_t stride = static_cast<size_t>(gridDim.x) * kThreads;
  size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x;
  uint32_t fold = 0;
  for (; i + (kUnroll - 1) * stride < vec_count; i += kUnroll * stride) {
    uint4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) v[u] = __ldcg(p + i + u * stride);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) fold ^= v[u].x ^ v[u].w;
  }
  for (; i < vec_count; i += stride) {
    const uint4 v = __ldcg(p + i);
    fold ^= v.x ^ v.w;
  }
  if (fold == g_prefetch_sink[0] + 0x9E3779B9u) g_prefetch_sink[1] = fold;
}

// The prefetch-instruction form (2026-09-10, DGPP_L2_PREFETCH_FORM=prefetch):
// one `prefetch.global.L2` per 128-byte line per thread, grid-strided —
// no destination register, no scoreboard wait, so a warp keeps issuing
// lines instead of folding what the loads above brought back. The bytes
// in flight are then the memory system's to bound, not the fold's; the
// sweep says whether that lifts the sustainable rate beside the
// collectives or just their latency (the load form's light rate was
// tuned to exactly that trade).
__global__ __launch_bounds__(kThreads) void l2_prefetch_lines_kernel(
    const uint8_t* __restrict__ p, size_t lines) {
  const size_t stride = static_cast<size_t>(gridDim.x) * kThreads;
  for (size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x; i < lines; i += stride)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p + i * 128));
}

bool env_prefetch_form_lines() {
  static const bool lines = [] {
    const char* v = std::getenv("DGPP_L2_PREFETCH_FORM");
    return v != nullptr && std::string(v) == "prefetch";
  }();
  return lines;
}

// The touch form (2026-09-29, the Qwen depth-3 profiles with the prefetch on
// against off): of the ~3.9 GB a pass prefetched, the consumers found
// 96 MB in L2 — yet they ran ~10 % slower without it. What the prefetch
// delivered was the TRANSLATIONS: a 65 GB working set walks 32K 2 MB
// pages per pass through a walker every SM shares, and the consumer that
// finds its pages already walked runs at the isolated rate. This form
// touches one 128-byte line per DGPP_L2_PREFETCH_TOUCH_KB (default 64 KB:
// 1/512 of the bytes) so the walks happen ahead of the consumer while the
// DRAM stays free for it. DGPP_L2_PREFETCH_FORM=touch.
bool env_prefetch_form_touch() {
  static const bool touch = [] {
    const char* v = std::getenv("DGPP_L2_PREFETCH_FORM");
    return v != nullptr && std::string(v) == "touch";
  }();
  return touch;
}
size_t env_touch_stride_bytes() {
  static const size_t stride = [] {
    const char* v = std::getenv("DGPP_L2_PREFETCH_TOUCH_KB");
    if (v == nullptr) return size_t{64} << 10;
    const long kb = std::strtol(v, nullptr, 10);
    return (kb >= 1 && kb <= (1 << 20)) ? static_cast<size_t>(kb) << 10 : size_t{64} << 10;
  }();
  return stride;
}
__global__ __launch_bounds__(kThreads) void l2_touch_pages_kernel(const uint8_t* __restrict__ p, size_t touches,
                                                                 size_t stride_bytes) {
  const size_t stride = static_cast<size_t>(gridDim.x) * kThreads;
  for (size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x; i < touches; i += stride)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p + i * stride_bytes));
}

bool env_is_off(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && std::string(v) == "off";
}

PrefetchRate env_rate(const char* name, PrefetchRate fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr) return fallback;
  const std::string s(v);
  if (s == "off") return PrefetchRate::Off;
  if (s == "light") return PrefetchRate::Light;
  if (s == "full") return PrefetchRate::Full;
  DGPP_LOG_WARN("{}={} ignored (off|light|full)", name, s);
  return fallback;
}

const char* rate_name(PrefetchRate r) {
  switch (r) {
    case PrefetchRate::Off: return "off";
    case PrefetchRate::Light: return "light";
    case PrefetchRate::Full: return "full";
  }
  return "?";
}

size_t env_window_bytes(size_t fallback) {
  const char* v = std::getenv("DGPP_L2_PREFETCH_MB");
  if (v == nullptr) return fallback;
  const long mb = std::strtol(v, nullptr, 10);
  if (mb <= 0 || mb > 64) {
    DGPP_LOG_WARN("DGPP_L2_PREFETCH_MB={} ignored (want 1..64)", v);
    return fallback;
  }
  return static_cast<size_t>(mb) << 20;
}

}  // namespace

// The kernel launch with the persisting access-policy window over
// [begin, end) (cudaLaunchKernelEx: the attribute is captured into the
// graph node).
template <int kUnroll>
void launch_persisting(const uint4* p, size_t vecs, int blocks, cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(blocks));
  cfg.blockDim = dim3(kThreads);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeAccessPolicyWindow;
  attr[0].val.accessPolicyWindow.base_ptr = const_cast<uint4*>(p);
  attr[0].val.accessPolicyWindow.num_bytes = vecs * 16;
  attr[0].val.accessPolicyWindow.hitRatio = 1.0f;
  attr[0].val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr[0].val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  DGPP_CUDA_OK(cudaLaunchKernelEx(&cfg, l2_prefetch_kernel<kUnroll>, p, vecs));
}

void launch_l2_prefetch(const void* ptr, size_t bytes, PrefetchRate rate,
                        cudaStream_t stream, bool persisting) {
  if (ptr == nullptr || bytes == 0 || rate == PrefetchRate::Off) return;
  // Widen to 16-byte boundaries: the read may cover a few bytes past either
  // end of the request, which is safe for any cudaMalloc'd range (256-byte
  // aligned and sized).
  const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr) & ~uintptr_t{15};
  const uintptr_t end =
      (reinterpret_cast<uintptr_t>(ptr) + bytes + 15) & ~uintptr_t{15};
  const uint4* p = reinterpret_cast<const uint4*>(begin);
  const size_t vecs = (end - begin) / 16;
  if (persisting) {
    if (rate == PrefetchRate::Full)
      launch_persisting<kFullUnroll>(p, vecs, kFullBlocks, stream);
    else
      launch_persisting<kLightUnroll>(p, vecs, light_blocks(), stream);
    return;
  }
  if (rate == PrefetchRate::Full) {
    l2_prefetch_kernel<kFullUnroll><<<kFullBlocks, kThreads, 0, stream>>>(p, vecs);
  } else if (env_prefetch_form_touch()) {
    const size_t sb = env_touch_stride_bytes();
    const size_t touches = (end - begin + sb - 1) / sb;
    l2_touch_pages_kernel<<<static_cast<unsigned>(std::min<size_t>(light_blocks(), (touches + kThreads - 1) / kThreads)),
                            kThreads, 0, stream>>>(reinterpret_cast<const uint8_t*>(begin), touches, sb);
  } else if (env_prefetch_form_lines()) {
    const size_t lines = (end - begin + 127) / 128;
    l2_prefetch_lines_kernel<<<light_blocks(), kThreads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(begin), lines);
  } else {
    l2_prefetch_kernel<kLightUnroll><<<light_blocks(), kThreads, 0, stream>>>(p, vecs);
  }
  DGPP_CUDA_OK(cudaGetLastError());
}

__global__ __launch_bounds__(kThreads) void l2_release_kernel(const uint8_t* p, size_t lines) {
  const size_t stride = static_cast<size_t>(gridDim.x) * kThreads;
  for (size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x; i < lines; i += stride)
    asm volatile("applypriority.global.L2::evict_normal [%0], 128;" ::"l"(p + i * 128) : "memory");
}

void launch_l2_release(const void* ptr, size_t bytes, cudaStream_t stream) {
  if (ptr == nullptr || bytes == 0) return;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr) & ~uintptr_t{127};
  const uintptr_t end = (reinterpret_cast<uintptr_t>(ptr) + bytes + 127) & ~uintptr_t{127};
  const size_t lines = (end - begin) / 128;
  l2_release_kernel<<<kFullBlocks, kThreads, 0, stream>>>(reinterpret_cast<const uint8_t*>(begin), lines);
  DGPP_CUDA_OK(cudaGetLastError());
}

size_t l2_set_persisting_limit(size_t bytes) {
  int dev = 0;
  DGPP_CUDA_OK(cudaGetDevice(&dev));
  int max_bytes = 0;
  DGPP_CUDA_OK(cudaDeviceGetAttribute(&max_bytes, cudaDevAttrMaxPersistingL2CacheSize, dev));
  if (max_bytes <= 0) return 0;
  const size_t want = std::min(bytes, static_cast<size_t>(max_bytes));
  DGPP_CUDA_OK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want));
  size_t got = 0;
  DGPP_CUDA_OK(cudaDeviceGetLimit(&got, cudaLimitPersistingL2CacheSize));
  return got;
}

void WeightPrefetcher::note_persisting(const void* ptr, size_t bytes) {
  persisted_[persisted_next_] = Range{reinterpret_cast<uintptr_t>(ptr), bytes};
  persisted_next_ = (persisted_next_ + 1) % kPersistRing;
}

void WeightPrefetcher::release_persisting(cudaStream_t main) {
  for (Range& r : persisted_) {
    if (r.bytes == 0) continue;
    launch_l2_release(reinterpret_cast<const void*>(r.begin), r.bytes, main);
    r = Range{};
  }
}

WeightPrefetcher::WeightPrefetcher() {
  enabled_ = !env_is_off("DGPP_L2_PREFETCH");
  merge_ = !env_is_off("DGPP_L2_PREFETCH_MERGE");
  window_bytes_ = env_window_bytes(kDefaultWindowBytes);
  boundary_rate_ = env_rate("DGPP_L2_PREFETCH_BOUNDARY", PrefetchRate::Light);
  layer_rate_ = env_rate("DGPP_L2_PREFETCH_LAYER", PrefetchRate::Light);
  if (!enabled_) {
    DGPP_LOG_INFO("l2 prefetch: off (DGPP_L2_PREFETCH=off)");
    return;
  }
  // Lowest priority: the chain's kernels (on a higher-priority stream)
  // take SM slots first; the prefetch fills what is left, which during a
  // latency-bound phase is nearly everything.
  int least = 0, greatest = 0;
  DGPP_CUDA_OK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  DGPP_CUDA_OK(cudaStreamCreateWithPriority(&side_, cudaStreamNonBlocking,
                                            least));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&join_, cudaEventDisableTiming));
  DGPP_LOG_INFO("l2 prefetch: on, window budget {} MiB, boundary rate {}, "
                "layer rate {}",
                window_bytes_ >> 20, rate_name(boundary_rate_),
                rate_name(layer_rate_));
}

WeightPrefetcher::~WeightPrefetcher() {
  if (join_) cudaEventDestroy(join_);
  if (fork_) cudaEventDestroy(fork_);
  if (side_) cudaStreamDestroy(side_);
}

void WeightPrefetcher::flush_pending() {
  if (pending_end_ > pending_begin_) {
    launch_l2_prefetch(reinterpret_cast<const void*>(pending_begin_),
                       pending_end_ - pending_begin_, rate_, side_, persisting_);
    if (persisting_) note_persisting(reinterpret_cast<const void*>(pending_begin_), pending_end_ - pending_begin_);
    ++launches_;
  }
  pending_begin_ = pending_end_ = 0;
}

void WeightPrefetcher::open_window(cudaStream_t main, size_t budget_bytes,
                                   PrefetchRate rate, bool persisting) {
  if (!enabled_) return;
  flush_pending();  // the previous window's tail, at its own rate and policy
  rate_ = rate;
  persisting_ = persisting;
  remaining_ = budget_bytes > 0 ? budget_bytes : window_bytes_;
  window_open_ = true;
  if (rate == PrefetchRate::Off) return;  // no fork: nothing will launch
  DGPP_CUDA_OK(cudaEventRecord(fork_, main));
  DGPP_CUDA_OK(cudaStreamWaitEvent(side_, fork_, 0));
  forked_ = true;
}

void WeightPrefetcher::add(const void* ptr, size_t bytes) {
  if (!enabled_ || !window_open_ || rate_ == PrefetchRate::Off ||
      ptr == nullptr || bytes == 0)
    return;
  if (!merge_) {
    const size_t take = std::min(bytes, remaining_);
    if (take == 0) return;
    remaining_ -= take;
    launch_l2_prefetch(ptr, take, rate_, side_, persisting_);
    if (persisting_) note_persisting(ptr, take);
    ++launches_;
    return;
  }
  const uintptr_t b = reinterpret_cast<uintptr_t>(ptr);
  if (pending_end_ > pending_begin_ && b >= pending_begin_ &&
      b <= pending_end_ + kMergeGap) {
    // Extend the pending range; the bridged gap (a neighbouring tensor of
    // the same layer, read soon anyway) is charged to the budget too.
    const size_t gap = b > pending_end_ ? b - pending_end_ : 0;
    const size_t take = std::min(bytes, remaining_ > gap ? remaining_ - gap : 0);
    if (take == 0) return;
    remaining_ -= gap + take;
    pending_end_ = std::max(pending_end_, b + take);
    return;
  }
  const size_t take = std::min(bytes, remaining_);
  if (take == 0) return;
  remaining_ -= take;
  flush_pending();
  pending_begin_ = b;
  pending_end_ = b + take;
}

void WeightPrefetcher::add_isolated(const void* ptr, size_t bytes) {
  if (!enabled_ || !window_open_ || rate_ == PrefetchRate::Off || ptr == nullptr || bytes == 0)
    return;
  const size_t take = std::min(bytes, remaining_);
  if (take == 0) return;
  remaining_ -= take;
  flush_pending();  // nothing pending may grow into this range, nor this range into the next add
  launch_l2_prefetch(ptr, take, rate_, side_, persisting_);
  if (persisting_) note_persisting(ptr, take);
  ++launches_;
}

void WeightPrefetcher::prefetch_after(cudaStream_t main, const void* ptr,
                                      size_t bytes, size_t budget_bytes,
                                      PrefetchRate rate) {
  open_window(main, budget_bytes, rate);
  add(ptr, bytes);
}

void WeightPrefetcher::join(cudaStream_t main) {
  // Only a forked side stream may be joined: under capture, waiting on an
  // event the capture never recorded is a capture-isolation error.
  if (!enabled_ || !forked_) return;
  flush_pending();
  DGPP_CUDA_OK(cudaEventRecord(join_, side_));
  DGPP_CUDA_OK(cudaStreamWaitEvent(main, join_, 0));
  window_open_ = false;
  forked_ = false;
}

}  // namespace dgpp
