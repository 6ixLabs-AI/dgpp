// Which core a bus engine's poll thread takes, and the claim that keeps the
// next engine off it. Host-only and header-only: the choice is exercised
// without a GPU (tests/host/bus_engine_cpu_test.cpp); collective_bus.cpp
// reads the cores and pins the thread.
#pragma once

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <string>
#include <utility>
#include <vector>

namespace dgpp::net {

// Where the per-core claim files live. One fixed place every engine on the
// host resolves the same way, whoever started it and from whichever session:
// /run/user/<uid> goes away with the user's last login session (the ranks
// are started detached over ssh) and $TMPDIR differs between launchers, and
// two engines that look in different places both take the fastest core. A
// held file outlives /tmp's aging: systemd-tmpfiles skips a file that has a
// BSD lock on it. The files themselves stay behind, empty and harmless.
inline constexpr const char* kEngineCpuLockDir = "/tmp";

// (max kHz, cpu) pairs to the order the engines take cores in: fastest
// first, ties the HIGHER index first — core 0's neighbourhood carries the
// interrupt load. A GB10 gives 19..15, 9..5 (3.9 GHz), then 14..10, 4..0.
inline std::vector<int> order_engine_cpus(std::vector<std::pair<long, int>> cores) {
  std::sort(cores.begin(), cores.end(), [](const auto& a, const auto& b) {
    return a.first != b.first ? a.first > b.first : a.second > b.second;
  });
  std::vector<int> order;
  order.reserve(cores.size());
  for (const auto& core : cores) order.push_back(core.second);
  return order;
}

// One engine's core, and its exclusive lock on that core's file while it
// holds one. The engine is a busy-poll spinner: two of them on one core get
// half of it each and every cross-rank fold waits out a scheduler timeslice
// (2026-10-04, an 80B and a 35B world co-resident on two Sparks, both
// engines on cpu 19: 16.6 and 20 tok/s single stream against 129 and 150
// apart). The lock is flock's, not fcntl's: it belongs to the open file
// description, so two bus instances in one process exclude each other
// exactly as two processes do, and the kernel drops it when the holder
// closes it or dies, however it dies. Advisory: an engine built before the
// claims existed neither takes nor sees one.
class EngineCpuClaim {
 public:
  EngineCpuClaim() = default;
  EngineCpuClaim(EngineCpuClaim&& other) noexcept
      : cpu_(other.cpu_), fd_(std::exchange(other.fd_, -1)), passed_over_(other.passed_over_) {}
  EngineCpuClaim& operator=(EngineCpuClaim&& other) noexcept {
    if (this != &other) {
      release();
      cpu_ = other.cpu_;
      fd_ = std::exchange(other.fd_, -1);
      passed_over_ = other.passed_over_;
    }
    return *this;
  }
  ~EngineCpuClaim() { release(); }

  // The automatic choice: the first core of `order` whose lock can be
  // taken. When none can (every core claimed, or `dir` unusable) it is
  // order[instance % n] unclaimed, the choice from before the claims
  // existed: `instance` counts the bus instances of this process, so a
  // whole world in one process (the loopback tests) still takes successive
  // cores.
  static EngineCpuClaim first_free(const std::vector<int>& order, unsigned instance,
                                   const std::string& dir = kEngineCpuLockDir) {
    EngineCpuClaim claim;
    if (order.empty()) return claim;
    for (size_t i = 0; i < order.size(); ++i) {
      claim.fd_ = lock_core(dir, order[i]);
      if (claim.fd_ < 0) continue;
      claim.cpu_ = order[i];
      claim.passed_over_ = static_cast<int>(i);
      return claim;
    }
    claim.cpu_ = order[instance % order.size()];
    claim.passed_over_ = static_cast<int>(order.size());
    return claim;
  }

  // A core the operator named (DGPP_BUS_ENGINE_CPU): kept whoever else
  // claims it, and claimed when it is free so the automatic choices of
  // other engines pass it over. Negative: no core.
  static EngineCpuClaim fixed(int cpu, const std::string& dir = kEngineCpuLockDir) {
    EngineCpuClaim claim;
    if (cpu < 0) return claim;
    claim.cpu_ = cpu;
    claim.fd_ = lock_core(dir, cpu);
    return claim;
  }

  // The core to pin to; -1: none.
  int cpu() const { return cpu_; }
  // False: no lock is held, and another engine may be on cpu().
  bool held() const { return fd_ >= 0; }
  // How many cores ahead of cpu() in the order were not free.
  int passed_over() const { return passed_over_; }

  void release() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
  }

 private:
  // The locked descriptor, or -1 when another engine holds the core or its
  // file cannot be used. An existing file is opened as it is, without
  // O_CREAT: fs.protected_regular refuses an O_CREAT open of another user's
  // file in a sticky directory. Read access is all flock needs, and the mode
  // is set past the umask so another user's engine can open the file too.
  // The directory is shared: a link there is not followed and a FIFO does
  // not block the open.
  static int lock_core(const std::string& dir, int cpu) {
    const std::string path = dir + "/dgpp-bus-engine-cpu" + std::to_string(cpu) + ".lock";
    const int flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK;
    int fd = ::open(path.c_str(), flags);
    if (fd < 0 && errno == ENOENT) {
      fd = ::open(path.c_str(), flags | O_CREAT, 0644);
      if (fd >= 0) ::fchmod(fd, 0644);
    }
    if (fd < 0) return -1;
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
      ::close(fd);
      return -1;
    }
    return fd;
  }

  int cpu_ = -1;
  int fd_ = -1;
  int passed_over_ = 0;
};

}  // namespace dgpp::net
