// QSA decode score/select timings with production paging and an exact host
// oracle. Run only on idle hardware; cold samples evict L2 outside the timer.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/qsa.hpp"
#include "models/qwen/qsa_reference.hpp"

namespace {
template <class T>
struct Buffer {
  T* p = nullptr;
  size_t size;
  explicit Buffer(size_t n) : size(n) { DGPP_CUDA_OK(cudaMalloc(&p, n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void upload(const std::vector<T>& v) {
    DGPP_CUDA_OK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> v(size);
    DGPP_CUDA_OK(cudaMemcpy(v.data(), p, size * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};

__global__ void evict_l2(uint4* data, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    data[i] = make_uint4(i, i + 1, i + 2, i + 3);
}
__global__ void empty_kernel() {}

struct Launch {
  std::function<void()> fn;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  Launch(std::function<void()> f, bool capture, cudaStream_t stream) : fn(std::move(f)) {
    if (capture) {
      DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      fn();
      DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
      DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
  }
  ~Launch() {
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
  }
  void run(cudaStream_t stream) const {
    if (exec)
      DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
    else
      fn();
  }
};

std::vector<double> measure(const Launch& launch, bool cold, Buffer<uint4>& eviction, int iters,
                            int warmup, cudaStream_t stream) {
  for (int i = 0; i < warmup; ++i) launch.run(stream);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  cudaEvent_t start, end;
  DGPP_CUDA_OK(cudaEventCreate(&start));
  DGPP_CUDA_OK(cudaEventCreate(&end));
  std::vector<double> times;
  for (int i = 0; i < iters; ++i) {
    if (cold) evict_l2<<<1024, 256, 0, stream>>>(eviction.p, eviction.size);
    DGPP_CUDA_OK(cudaEventRecord(start, stream));
    launch.run(stream);
    DGPP_CUDA_OK(cudaEventRecord(end, stream));
    DGPP_CUDA_OK(cudaEventSynchronize(end));
    float elapsed;
    DGPP_CUDA_OK(cudaEventElapsedTime(&elapsed, start, end));
    times.push_back(elapsed * 1000.0);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  std::sort(times.begin(), times.end());
  return times;
}

void require(bool condition, const std::string& what) {
  if (!condition) throw std::runtime_error(what);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    int rows = 2, iters = 30, warmup = 5, capacity = 0;
    std::vector<int> pools{16384, 65322, 131072};
    bool graph = false, visible_bound = false, skip_select_all = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--graph") {
        graph = true;
        continue;
      }
      if (arg == "--visible-bound") {
        visible_bound = true;
        continue;
      }
      if (arg == "--skip-select-all") {
        skip_select_all = true;
        continue;
      }
      if (i + 1 == argc) throw std::invalid_argument("missing value for " + arg);
      if (arg == "--pools") {
        pools.clear();
        std::string values = argv[++i];
        size_t start = 0;
        for (;;) {
          const size_t end = values.find(',', start);
          pools.push_back(std::stoi(values.substr(start, end - start)));
          if (end == std::string::npos) break;
          start = end + 1;
        }
      } else {
        const int value = std::stoi(argv[++i]);
        if (arg == "--rows")
          rows = value;
        else if (arg == "--iters")
          iters = value;
        else if (arg == "--warmup")
          warmup = value;
        else if (arg == "--capacity")
          capacity = value;
        else
          throw std::invalid_argument("unknown option " + arg);
      }
    }
    require(rows > 0 && rows <= 2048 && iters > 0 && warmup >= 0, "invalid rows/iterations");
    require(capacity >= 0 && capacity <= (1 << 21), "invalid workspace capacity");
    for (int n : pools) require(n > 0 && n < (1 << 21) && 4 * n >= rows, "invalid pool count");
    int device;
    DGPP_CUDA_OK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    DGPP_CUDA_OK(cudaGetDeviceProperties(&prop, device));
    const size_t eviction_bytes = std::max<size_t>(64 << 20, 4ull * prop.l2CacheSize);
    Buffer<uint4> eviction(eviction_bytes / sizeof(uint4));
    std::fprintf(stderr, "%s: %d SMs, L2 %d bytes, eviction %zu bytes\n", prop.name,
                 prop.multiProcessorCount, prop.l2CacheSize, eviction_bytes);
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreate(&stream));
    constexpr int heads = 4, dim = 128, kpool = 4, ppb = 16, select_k = 512;
    constexpr int max_selected = select_k * kpool + kpool - 1;
    for (int n : pools) {
      const int blocks = (n + ppb - 1) / ppb;
      const int stride = capacity ? capacity : blocks * ppb;
      require(stride >= n, "workspace capacity smaller than visible pools");
      std::vector<uint16_t> q(static_cast<size_t>(rows) * heads * dim),
          cache(static_cast<size_t>(stride) * dim);
      std::mt19937 rng(20260921);
      std::uniform_real_distribution<float> dist(-1.f, 1.f);
      for (auto* v : {&q, &cache})
        for (auto& x : *v) x = dgpp::float_to_bf16_bits(dist(rng));
      std::vector<int32_t> table(blocks), req(rows, 0);
      std::iota(table.begin(), table.end(), 0);
      std::shuffle(table.begin(), table.end(), rng);
      std::vector<int64_t> pos(rows);
      for (int r = 0; r < rows; ++r) pos[r] = int64_t(n) * kpool - rows + r;
      Buffer<uint16_t> dq(q.size()), dc(cache.size());
      Buffer<int32_t> dt(table.size()), dr(rows),
          selected(static_cast<size_t>(rows) * max_selected), counts(rows);
      Buffer<int64_t> dp(rows);
      Buffer<uint64_t> keys(static_cast<size_t>(rows) * stride);
      dq.upload(q);
      dc.upload(cache);
      dt.upload(table);
      dr.upload(req);
      dp.upload(pos);
      auto score = [&] {
        dgpp::qsa_index_score(dq.p, heads * dim, dr.p, dp.p, rows, dt.p, blocks, dc.p, ppb, heads,
                              dim, kpool, keys.p, stride, stream, visible_bound ? n : -1,
                              skip_select_all ? select_k : 0);
      };
      auto select = [&] {
        dgpp::qsa_select_from_keys(keys.p, stride, dp.p, rows, select_k, kpool, max_selected,
                                   selected.p, counts.p, stream);
      };
      score();
      select();
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      const auto got_keys = keys.download();
      const auto got_selected = selected.download(), got_counts = counts.download();
      size_t checked_keys = 0, checked_tokens = 0;
      for (int r = 0; r < rows; ++r) {
        const int visible = static_cast<int>((pos[r] + 1) / kpool);
        std::vector<float> scores(visible);
        for (int p = 0; p < visible; ++p) {
          const int slot = table[p / ppb] * ppb + p % ppb;
          const float value = dgpp::qwen_ref::qsa_index_score(q.data() + r * heads * dim,
                                                              cache.data() + slot * dim);
          scores[p] = value;
          uint32_t bits;
          std::memcpy(&bits, &value, sizeof(bits));
          const uint32_t sortable = bits >> 31 ? ~bits : bits | 0x80000000u;
          const uint64_t key = (static_cast<uint64_t>(~sortable) << 21) | p;
          if (!skip_select_all || visible > select_k) {
            require(got_keys[static_cast<size_t>(r) * stride + p] == key,
                    "score key differs from host oracle");
            ++checked_keys;
          }
        }
        std::vector<int32_t> ids, tokens;
        dgpp::qwen_ref::qsa_select(scores, select_k, ids);
        dgpp::qwen_ref::qsa_expand(ids, pos[r], kpool, tokens);
        require(got_counts[r] == static_cast<int>(tokens.size()), "selected count differs");
        checked_tokens += tokens.size();
        tokens.resize(max_selected, -1);
        require(std::equal(tokens.begin(), tokens.end(), got_selected.begin() + r * max_selected),
                "selected tokens differ");
      }
      for (const std::string stage : {"empty", "score", "select", "combined"}) {
        Launch launch(
            [&] {
              if (stage == "empty") empty_kernel<<<1, 1, 0, stream>>>();
              if (stage == "score" || stage == "combined") score();
              if (stage == "select" || stage == "combined") select();
            },
            graph, stream);
        for (bool cold : {false, true}) {
          const auto times = measure(launch, cold, eviction, iters, warmup, stream);
          require(selected.download() == got_selected && counts.download() == got_counts,
                  "selection changed after repeated launches");
          const double median = (times[(times.size() - 1) / 2] + times[times.size() / 2]) / 2;
          const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
          const size_t bytes = checked_keys * (stage == "score"      ? dim * 2
                                               : stage == "select"   ? 8
                                               : stage == "combined" ? dim * 2 + 8
                                                                     : 0);
          std::printf(
              "{\"rows\":%d,\"pools\":%d,\"capacity\":%d,\"visible_bound\":%s,\"skip_select_all\":%"
              "s,"
              "\"graph\":%s,\"cache\":\"%s\",\"stage\":\"%s\","
              "\"iterations\":%d,\"median_us\":%.3f,\"mean_us\":%.3f,\"min_us\":%.3f,\"p95_us\":%."
              "3f,"
              "\"logical_input_GB_s\":%.3f,\"oracle_keys\":%zu,\"oracle_tokens\":%zu,"
              "\"mismatches\":0}\n",
              rows, n, stride, visible_bound ? "true" : "false", skip_select_all ? "true" : "false",
              graph ? "true" : "false", cold ? "cold" : "warm", stage.c_str(), iters, median, mean,
              times.front(), times[(times.size() - 1) * 95 / 100], bytes / (median * 1000),
              checked_keys, checked_tokens);
          std::fflush(stdout);
        }
      }
    }
    cudaStreamDestroy(stream);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
