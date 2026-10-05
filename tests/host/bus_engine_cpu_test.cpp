// The bus engine's core choice (net/engine_cpu.hpp), without a GPU: the
// order, the claim that sends a second engine process to the next core, and
// the unclaimed fallback. The cores are numbers only; nothing is pinned.
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/test.hpp"
#include "net/engine_cpu.hpp"

namespace {

using dgpp::net::EngineCpuClaim;

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

// A GB10's cores as the bus orders them (the layout is pinned below).
const std::vector<int> kGb10 = {19, 18, 17, 16, 15, 9, 8, 7, 6, 5,
                                14, 13, 12, 11, 10, 4, 3, 2, 1, 0};

// A private lock directory per test: the host's own engines keep theirs.
struct LockDir {
  std::string path;
  LockDir() {
    char name[] = "/tmp/dgpp-bus-engine-cpu-test-XXXXXX";
    require(::mkdtemp(name) != nullptr, "mkdtemp failed");
    path = name;
  }
  ~LockDir() {
    for (int cpu : kGb10)
      ::unlink((path + "/dgpp-bus-engine-cpu" + std::to_string(cpu) + ".lock").c_str());
    ::rmdir(path.c_str());
  }
};

// Another engine PROCESS: it makes the automatic choice as the first bus
// instance of a fresh process does (instance 0), reports the core, and keeps
// its claim until it is killed.
struct EngineProcess {
  pid_t pid = -1;
  int report = -1;  // read end: the chosen cpu and whether it holds the claim
  int go = -1;      // write end: one byte lets it choose

  explicit EngineProcess(const std::string& dir) {
    int report_pipe[2], go_pipe[2];
    require(::pipe(report_pipe) == 0 && ::pipe(go_pipe) == 0, "pipe failed");
    pid = ::fork();
    require(pid >= 0, "fork failed");
    if (pid == 0) {
      ::alarm(30);  // never outlive a failed test
      ::close(report_pipe[0]);
      ::close(go_pipe[1]);
      char byte = 0;
      if (::read(go_pipe[0], &byte, 1) != 1) ::_exit(3);
      const EngineCpuClaim claim = EngineCpuClaim::first_free(kGb10, 0, dir);
      const int out[2] = {claim.cpu(), claim.held() ? 1 : 0};
      if (::write(report_pipe[1], out, sizeof(out)) != static_cast<ssize_t>(sizeof(out)))
        ::_exit(4);
      for (;;) ::pause();
    }
    ::close(report_pipe[1]);
    ::close(go_pipe[0]);
    report = report_pipe[0];
    go = go_pipe[1];
  }
  EngineProcess(const EngineProcess&) = delete;
  EngineProcess& operator=(const EngineProcess&) = delete;
  ~EngineProcess() {
    kill();
    ::close(report);
    ::close(go);
  }

  void start() { require(::write(go, "g", 1) == 1, "could not start the engine process"); }

  // The core it took; throws unless it holds the claim.
  int cpu() {
    int out[2] = {-1, 0};
    ssize_t got;
    do {
      got = ::read(report, out, sizeof(out));
    } while (got < 0 && errno == EINTR);
    require(got == static_cast<ssize_t>(sizeof(out)), "the engine process reported nothing");
    require(out[1] == 1, "the engine process took cpu " + std::to_string(out[0]) + " unclaimed");
    return out[0];
  }

  void kill() {
    if (pid <= 0) return;
    ::kill(pid, SIGKILL);
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    pid = -1;
  }
};

std::string list(const std::vector<int>& cpus) {
  std::string text;
  for (int cpu : cpus) text += (text.empty() ? "" : ",") + std::to_string(cpu);
  return text;
}

DGPP_TEST(bus_engine_cpu_order_is_fastest_first_then_the_higher_index) {
  // cpuinfo_max_freq on a GB10: 0-4 and 10-14 are the 2.8 GHz class, 5-9
  // and 15-19 the 3.9 GHz one.
  std::vector<std::pair<long, int>> cores;
  for (int cpu = 0; cpu < 20; ++cpu)
    cores.emplace_back((cpu % 10) < 5 ? 2808000 : 3900000, cpu);
  const std::vector<int> order = dgpp::net::order_engine_cpus(cores);
  require(order == kGb10, "GB10 order is " + list(order));
  require(dgpp::net::order_engine_cpus({}).empty(), "no cores must order to none");
}

DGPP_TEST(bus_engine_cpu_instances_of_one_process_take_successive_cores) {
  // A loopback world: four bus instances alive in one process.
  LockDir dir;
  std::vector<EngineCpuClaim> world;
  for (unsigned i = 0; i < 4; ++i) {
    world.push_back(EngineCpuClaim::first_free(kGb10, i, dir.path));
    require(world[i].held(), "instance " + std::to_string(i) + " holds no claim");
    require(world[i].cpu() == kGb10[i],
            "instance " + std::to_string(i) + " took cpu " + std::to_string(world[i].cpu()));
    require(world[i].passed_over() == static_cast<int>(i), "wrong count of cores passed over");
  }
  // An engine that stops frees its core for the next one to start.
  world[1].release();
  const EngineCpuClaim next = EngineCpuClaim::first_free(kGb10, 4, dir.path);
  require(next.held() && next.cpu() == kGb10[1],
          "the freed core was not taken: cpu " + std::to_string(next.cpu()));
}

DGPP_TEST(bus_engine_cpu_second_process_takes_the_next_fastest_core) {
  // The 2026-10-04 failure: each process is its own instance 0, and both
  // took cpu 19.
  LockDir dir;
  const EngineCpuClaim first = EngineCpuClaim::first_free(kGb10, 0, dir.path);
  require(first.held() && first.cpu() == 19, "the first engine must take cpu 19");
  EngineProcess second(dir.path);
  second.start();
  const int cpu = second.cpu();
  require(cpu == 18, "the second engine process took cpu " + std::to_string(cpu) + ", not 18");
}

DGPP_TEST(bus_engine_cpu_processes_starting_together_take_distinct_cores) {
  LockDir dir;
  constexpr size_t kEngines = 6;
  std::vector<std::unique_ptr<EngineProcess>> engines;
  for (size_t i = 0; i < kEngines; ++i)
    engines.push_back(std::make_unique<EngineProcess>(dir.path));
  for (auto& engine : engines) engine->start();
  std::vector<int> taken;
  for (auto& engine : engines) taken.push_back(engine->cpu());
  std::sort(taken.begin(), taken.end());
  std::vector<int> fastest(kGb10.begin(), kGb10.begin() + kEngines);
  std::sort(fastest.begin(), fastest.end());
  require(taken == fastest, "six engine processes took cpus " + list(taken));
}

DGPP_TEST(bus_engine_cpu_claim_ends_with_its_process) {
  LockDir dir;
  EngineProcess engine(dir.path);
  engine.start();
  require(engine.cpu() == 19, "the engine process must take cpu 19");
  const EngineCpuClaim beside = EngineCpuClaim::first_free(kGb10, 0, dir.path);
  require(beside.cpu() == 18, "cpu 19 is held by a live process");
  engine.kill();  // SIGKILL: no destructor, no cleanup
  const EngineCpuClaim after = EngineCpuClaim::first_free(kGb10, 0, dir.path);
  require(after.held() && after.cpu() == 19,
          "a dead engine still holds cpu 19: took " + std::to_string(after.cpu()));
}

DGPP_TEST(bus_engine_cpu_without_a_lock_directory_keeps_the_per_process_order) {
  // Nothing can be locked under a path that is not a directory: the choice
  // is the one from before the claims, instance i on the i-th core.
  LockDir dir;
  const std::string not_a_dir = dir.path + "/dgpp-bus-engine-cpu0.lock";
  { const EngineCpuClaim make_the_file = EngineCpuClaim::fixed(0, dir.path); }
  for (unsigned i : {0u, 1u, 2u, 19u, 20u, 41u}) {
    const EngineCpuClaim claim = EngineCpuClaim::first_free(kGb10, i, not_a_dir);
    require(!claim.held(), "a claim was taken without a lock directory");
    require(claim.cpu() == kGb10[i % kGb10.size()],
            "instance " + std::to_string(i) + " took cpu " + std::to_string(claim.cpu()));
  }
}

DGPP_TEST(bus_engine_cpu_with_every_core_claimed_keeps_the_per_process_order) {
  LockDir dir;
  std::vector<EngineCpuClaim> all;
  for (unsigned i = 0; i < kGb10.size(); ++i)
    all.push_back(EngineCpuClaim::first_free(kGb10, i, dir.path));
  require(all.back().held() && all.back().cpu() == 0, "the last free core is cpu 0");
  const EngineCpuClaim extra = EngineCpuClaim::first_free(kGb10, 3, dir.path);
  require(!extra.held() && extra.cpu() == kGb10[3],
          "with no core free, instance 3 took cpu " + std::to_string(extra.cpu()));
  require(EngineCpuClaim::first_free({}, 0, dir.path).cpu() == -1, "no cores: no choice");
}

DGPP_TEST(bus_engine_cpu_override_is_kept_and_claims_its_core) {
  LockDir dir;
  const EngineCpuClaim automatic = EngineCpuClaim::first_free(kGb10, 0, dir.path);
  require(automatic.cpu() == 19, "the automatic engine must take cpu 19");
  // DGPP_BUS_ENGINE_CPU=18 on a second engine, then a third left automatic.
  const EngineCpuClaim named = EngineCpuClaim::fixed(18, dir.path);
  require(named.cpu() == 18 && named.held(), "the named core must be taken and claimed");
  EngineProcess third(dir.path);
  third.start();
  const int cpu = third.cpu();
  require(cpu == 17, "the automatic choice took cpu " + std::to_string(cpu) + " past a named 18");
  // The operator's core is kept even when another engine has it: visible
  // as an unheld claim (the bus warns), never moved.
  const EngineCpuClaim same = EngineCpuClaim::fixed(18, dir.path);
  require(same.cpu() == 18 && !same.held(), "a second engine named cpu 18");
  // A negative value leaves the engine unpinned and touches nothing.
  const EngineCpuClaim none = EngineCpuClaim::fixed(-1, dir.path);
  require(none.cpu() == -1 && !none.held(), "a negative core must be no core");
  struct stat st {};
  require(::stat((dir.path + "/dgpp-bus-engine-cpu-1.lock").c_str(), &st) != 0,
          "a lock file was made for a negative core");
}

}  // namespace

int main() { return dgpp::test::run_all(); }
