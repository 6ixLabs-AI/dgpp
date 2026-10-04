#include "loaders/hf_cache.hpp"

#include <fcntl.h>
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace dgpp {
namespace hf {
namespace {

namespace fs = std::filesystem;

// hub's on-disk naming: repo id org/name becomes models--org--name (each '/'
// is two dashes, so the mapping is unambiguous in both directions).
std::string model_dir_name(const std::string& model_id) {
  std::string name = "models--";
  for (const char c : model_id) {
    if (c == '/') name += "--";
    else name += c;
  }
  return name;
}

std::string read_first_line(const fs::path& file) {
  std::FILE* f = std::fopen(file.c_str(), "rb");
  if (!f) return {};
  std::string line;
  char buf[256];
  const size_t got = std::fread(buf, 1, sizeof(buf), f);
  std::fclose(f);
  line.assign(buf, got);
  const size_t nl = line.find('\n');
  if (nl != std::string::npos) line.resize(nl);
  const size_t last = line.find_last_not_of(" \t\r");
  if (last == std::string::npos) return {};
  line.resize(last + 1);
  return line;
}

}  // namespace

std::string default_cache_root() {
  if (const char* hub_cache = std::getenv("HF_HUB_CACHE");
      hub_cache && *hub_cache)
    return hub_cache;
  if (const char* hf_home = std::getenv("HF_HOME"); hf_home && *hf_home)
    return std::string(hf_home) + "/hub";
  const char* home = std::getenv("HOME");
  if (!home || !*home) {
    const passwd* pw = ::getpwuid(::getuid());
    home = pw ? pw->pw_dir : nullptr;
  }
  if (!home || !*home) return {};
  return std::string(home) + "/.cache/huggingface/hub";
}

std::string model_dir_in_root(const std::string& cache_root,
                              const std::string& model_id,
                              std::string* error) {
  if (model_id.empty() || model_id.front() == '/' || model_id.back() == '/') {
    *error = "malformed model id '" + model_id + "'";
    return {};
  }
  const fs::path model_root = fs::path(cache_root) / model_dir_name(model_id);
  if (!fs::is_directory(model_root)) {
    *error = "no cached model '" + model_id + "' under " + cache_root;
    return {};
  }

  // refs/main pins the snapshot the hub tooling considers current.
  fs::path snapshot = model_root / "snapshots";
  const std::string commit =
      read_first_line(model_root / "refs" / "main");
  if (!commit.empty()) {
    snapshot /= commit;
    if (!fs::is_directory(snapshot)) {
      *error = "refs/main points at commit " + commit +
               " but its snapshot directory is missing (interrupted pull?)";
      return {};
    }
  } else {
    // No ref (a fresh manual download can land here): a single snapshot is
    // unambiguous, several are not — refuse to guess.
    std::vector<fs::path> found;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(model_root / "snapshots",
                                                     ec))
      if (entry.is_directory()) found.push_back(entry.path());
    if (ec) {
      *error = "cannot list snapshots of '" + model_id + "': " + ec.message();
      return {};
    }
    if (found.empty()) {
      *error = "model '" + model_id + "' has no snapshots under " + cache_root;
      return {};
    }
    if (found.size() > 1) {
      *error = "model '" + model_id + "' has no refs/main and " +
               std::to_string(found.size()) +
               " snapshots — ambiguous, refusing to guess";
      return {};
    }
    snapshot = found[0];
  }

  // The GLM loader needs config.json in the snapshot; a broken symlink or a
  // half-finished blob fails HERE, in the cache's vocabulary, instead of
  // deep inside safetensors parsing. A Mistral-native release (2026-10-04,
  // loaders/architecture.hpp) is described by params.json instead and ships
  // no config.json: either file marks a snapshot whose description landed.
  if (!fs::is_regular_file(snapshot / "config.json") && !fs::is_regular_file(snapshot / "params.json")) {
    *error = "snapshot " + snapshot.string() +
             " has no readable config.json (incomplete download?)";
    return {};
  }
  return snapshot.string();
}

std::string model_dir(const std::string& model_id, std::string* error) {
  const std::string root = default_cache_root();
  if (root.empty()) {
    *error =
        "cannot locate the hub cache (HF_HUB_CACHE, HF_HOME, and the home "
        "directory are all unavailable)";
    return {};
  }
  return model_dir_in_root(root, model_id, error);
}

}  // namespace hf
}  // namespace dgpp
