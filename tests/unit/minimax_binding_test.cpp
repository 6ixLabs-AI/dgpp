// The MiniMax-M2 expected-tensor table: its shape on the release's config —
// pinned to the tensor counts and byte totals of the real checkpoint's
// safetensors headers (143,471 model tensors / 134,401,534,976 bytes, and
// 47,598 activation scales in a file of their own: 191,069 in all, read from
// Hugging Face 2026-10-04) — the optional activation scales, the shard list
// taken from the index, the validator's report, the per-rank weight and K/V
// bytes at world 1 and 2 and, when the checkpoint is in the hub cache, the
// full binding against every shard's header.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "minimax_config_json.hpp"
#include "models/minimax/binding.hpp"
#include "models/minimax/config.hpp"

namespace {

using dgpp::DType;
using dgpp::MinimaxExpectedTensor;
using dgpp::MinimaxTensorRole;
using dgpp::MinimaxWeightClass;
using Shape = std::vector<int64_t>;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

const MinimaxExpectedTensor& find(const std::vector<MinimaxExpectedTensor>& v,
                                  const std::string& name) {
  for (const auto& e : v)
    if (e.name == name) return e;
  throw std::runtime_error("expected tensor missing from the table: " + name);
}
bool absent(const std::vector<MinimaxExpectedTensor>& v, const std::string& name) {
  for (const auto& e : v)
    if (e.name == name) return false;
  return true;
}
bool is(const MinimaxExpectedTensor& e, DType dtype, const Shape& shape) {
  return e.dtype == dtype && e.shape == shape;
}

// A checkpoint that has every tensor of the table (activation scales included).
std::unordered_map<std::string, dgpp::MinimaxTensorDesc> present_of(
    const std::vector<MinimaxExpectedTensor>& v) {
  std::unordered_map<std::string, dgpp::MinimaxTensorDesc> present;
  for (const auto& e : v) present.emplace(e.name, dgpp::MinimaxTensorDesc{e.dtype, e.shape});
  return present;
}

// The six experts the release has no activation scales for (layer, expert).
const int kNoInputScale[6][2] = {{0, 87}, {0, 198}, {61, 24}, {61, 75}, {61, 183}, {61, 225}};

std::unordered_map<std::string, dgpp::MinimaxTensorDesc> release_headers(
    const dgpp::MinimaxTextConfig& cfg) {
  auto present = present_of(dgpp::minimax_expected_tensors(cfg));
  for (const auto& le : kNoInputScale)
    for (const char* m : {"w1", "w2", "w3"})
      present.erase("model.layers." + std::to_string(le[0]) + ".block_sparse_moe.experts." +
                    std::to_string(le[1]) + "." + m + ".input_scale");
  return present;
}

std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root =
      fs::path(home) / ".cache/huggingface/hub/models--lukealonso--MiniMax-M2.7-NVFP4/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (fs::exists(snap.path() / "config.json") &&
        fs::exists(snap.path() / "model.safetensors.index.json"))
      return snap.path();
  return {};
}

}  // namespace

DGPP_TEST(minimax_binding_table_has_the_release_shape) {
  const dgpp::MinimaxTextConfig cfg = minimax_test::release();
  // A layer: 2 norms + 6 attention + the router and its bias + 256 experts x 3 matrices x (3 + the
  // optional scale).
  const auto layer = dgpp::minimax_expected_layer_tensors(cfg, 0);
  require(layer.size() == 2 + 6 + 2 + 256 * 3 * 4, "layer " + std::to_string(layer.size()));
  const auto all = dgpp::minimax_expected_tensors(cfg);
  size_t required = 0, optional = 0, required_bytes = 0, fp4 = 0;
  for (const auto& e : all) {
    (e.optional() ? optional : required) += 1;
    if (!e.optional()) required_bytes += e.nbytes();
    fp4 += e.role == MinimaxTensorRole::Fp4Payload;
  }
  require(required == 3 + 62 * (10 + 256 * 9), "required tensors " + std::to_string(required));
  require(required == 143471, "the model shards' 143,471 tensors");
  require(optional == 62 * 256 * 3 && optional == 47616,
          "one optional activation scale per NVFP4 matrix");
  require(fp4 == 47616, "47,616 NVFP4 matrices");
  require(required_bytes == 134401534976ull,
          "the model shards' bytes: " + std::to_string(required_bytes));
  require(present_of(all).size() == all.size(), "duplicate names in the table");
  // No draft head: the config declares one, nothing ships it.
  for (const auto& e : all)
    require(e.name.find("mtp") == std::string::npos, "an MTP tensor in the table: " + e.name);
}

DGPP_TEST(minimax_binding_names_and_containers) {
  const dgpp::MinimaxTextConfig cfg = minimax_test::release();
  const auto g = dgpp::minimax_expected_global_tensors(cfg);
  require(is(find(g, "model.embed_tokens.weight"), DType::BF16, {200064, 3072}), "embedding");
  require(is(find(g, "lm_head.weight"), DType::BF16, {200064, 3072}), "head");
  require(is(find(g, "model.norm.weight"), DType::BF16, {3072}), "final norm");
  const auto l = dgpp::minimax_expected_layer_tensors(cfg, 61);
  require(is(find(l, "model.layers.61.input_layernorm.weight"), DType::BF16, {3072}), "input norm");
  require(is(find(l, "model.layers.61.post_attention_layernorm.weight"), DType::BF16, {3072}),
          "post norm");
  require(is(find(l, "model.layers.61.self_attn.q_proj.weight"), DType::BF16, {6144, 3072}),
          "q_proj: 48 x 128");
  require(is(find(l, "model.layers.61.self_attn.k_proj.weight"), DType::BF16, {1024, 3072}),
          "k_proj: 8 x 128");
  require(is(find(l, "model.layers.61.self_attn.v_proj.weight"), DType::BF16, {1024, 3072}),
          "v_proj");
  require(is(find(l, "model.layers.61.self_attn.o_proj.weight"), DType::BF16, {3072, 6144}),
          "o_proj");
  // The per-layer norms: a weight per element of the whole projection, not GLM-4.7's [128].
  require(is(find(l, "model.layers.61.self_attn.q_norm.weight"), DType::BF16, {6144}), "q_norm");
  require(is(find(l, "model.layers.61.self_attn.k_norm.weight"), DType::BF16, {1024}), "k_norm");
  require(absent(l, "model.layers.61.self_attn.q_proj.bias"), "no attention biases");
  require(is(find(l, "model.layers.61.block_sparse_moe.gate.weight"), DType::BF16, {256, 3072}),
          "router");
  require(
      is(find(l, "model.layers.61.block_sparse_moe.e_score_correction_bias"), DType::BF16, {256}),
      "router bias (BF16)");
  const MinimaxExpectedTensor w1 = find(l, "model.layers.61.block_sparse_moe.experts.255.w1.weight");
  require(is(w1, DType::U8, {1536, 1536}) && w1.role == MinimaxTensorRole::Fp4Payload &&
              w1.expert == 255 && w1.layer == 61,
          "w1 payload");
  require(is(find(l, "model.layers.61.block_sparse_moe.experts.255.w1.weight_scale"),
             DType::F8_E4M3, {1536, 192}),
          "w1 scales");
  require(
      is(find(l, "model.layers.61.block_sparse_moe.experts.255.w1.weight_scale_2"), DType::F32, {}),
      "w1 global");
  require(
      is(find(l, "model.layers.61.block_sparse_moe.experts.255.w2.weight"), DType::U8, {3072, 768}),
      "w2 payload");
  require(is(find(l, "model.layers.61.block_sparse_moe.experts.255.w2.weight_scale"),
             DType::F8_E4M3, {3072, 96}),
          "w2 scales");
  const MinimaxExpectedTensor in = find(l, "model.layers.61.block_sparse_moe.experts.255.w3.input_scale");
  require(is(in, DType::F32, {}) && in.optional() && in.unused(),
          "the activation scale is optional and unused");
  require(absent(l, "model.layers.61.block_sparse_moe.experts.256.w1.weight") &&
              absent(l, "model.layers.61.block_sparse_moe.shared_experts.w1.weight"),
          "bounds; no shared expert");
}

DGPP_TEST(minimax_binding_accepts_the_release_and_reports_by_class) {
  const dgpp::MinimaxTextConfig cfg = minimax_test::release();
  auto present = release_headers(cfg);
  require(present.size() == 191069,
          "the release's 191,069 tensors: " + std::to_string(present.size()));
  {
    const dgpp::MinimaxBindReport rep = dgpp::minimax_validate_binding(cfg, present);
    require(rep.ok() && rep.expected == 143471 && rep.matched == 143471, "the release binds");
    require(rep.optional_present == 47598 && rep.optional_absent == 18 && rep.bound() == 191069,
            "activation scales");
    require(rep.fp4_matrices == 47616, "matrices");
    require(rep.bytes == 134401725368ull && rep.loaded_bytes == 134401534976ull, "bytes");
  }
  present.erase("model.layers.7.block_sparse_moe.experts.3.w2.weight_scale_2");
  present["model.layers.5.self_attn.o_proj.weight"].dtype = DType::F8_E4M3;
  // GLM-4.7's per-head norm weight is a shape error here.
  present["model.layers.0.self_attn.q_norm.weight"].shape = {128};
  // A present activation scale must be well-formed.
  present["model.layers.1.block_sparse_moe.experts.0.w1.input_scale"].dtype = DType::BF16;
  // Nothing is skipped: a draft tensor or a shared expert is unexpected.
  present.emplace("model.mtp.layers.0.self_attn.q_proj.weight",
                  dgpp::MinimaxTensorDesc{DType::BF16, {6144, 3072}});
  const dgpp::MinimaxBindReport rep = dgpp::minimax_validate_binding(cfg, present);
  require(!rep.ok(), "a broken checkpoint is refused");
  require(
      rep.missing == 1 && rep.dtype_mismatch == 2 && rep.shape_mismatch == 1 && rep.unexpected == 1,
      "the report counts by class");
  require(rep.errors.size() == 5, "five errors");
  require(dgpp::minimax_validate_binding(cfg, {}, 5).errors.size() == 5, "errors are capped");
  const dgpp::MinimaxBindReport empty = dgpp::minimax_validate_binding(cfg, {});
  require(empty.missing == 143471 && empty.optional_absent == 47616,
          "an empty checkpoint misses everything required");
}

DGPP_TEST(minimax_shard_list_is_the_indexs) {
  // The repository's directory holds other safetensors files; only the
  // index's are the model.
  const auto index = dgpp::minijson::parse(R"({"metadata": {"total_size": 1}, "weight_map": {
      "lm_head.weight": "model-00025-of-00025.safetensors",
      "model.embed_tokens.weight": "model-00001-of-00025.safetensors",
      "model.layers.0.block_sparse_moe.experts.0.w1.input_scale": "model-inputscales.safetensors",
      "model.layers.0.block_sparse_moe.experts.0.w1.weight": "model-00001-of-00025.safetensors"}})");
  const std::vector<std::string> files = dgpp::minimax_shard_files(index.root);
  require(files.size() == 3, "three distinct shards");
  require(files[0] == "model-00001-of-00025.safetensors" &&
              files[1] == "model-00025-of-00025.safetensors" &&
              files[2] == "model-inputscales.safetensors",
          "sorted, de-duplicated");
  bool refused = false;
  try {
    (void)dgpp::minimax_shard_files(dgpp::minijson::parse(R"({"metadata": {}})").root);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  require(refused, "an index without a weight_map");
}

DGPP_TEST(minimax_weight_and_kv_bytes_per_rank) {
  const dgpp::MinimaxTextConfig cfg = minimax_test::release();
  // One rank: the whole model, 125.17 GiB — more than a Spark's 121 GiB before any K/V.
  const dgpp::MinimaxWeightBytes one = dgpp::minimax_weight_bytes(cfg, 1);
  require(one.total() == 134401534976ull, "world 1 total: " + std::to_string(one.total()));
  require(one.routed_experts == 126382768128ull + 190464ull,
          "routed experts: " + std::to_string(one.routed_experts));
  require(one.attention == 2 * 2340421632ull + 2 * 390070272ull + 761856ull + 126976ull,
          "attention: " + std::to_string(one.attention));
  require(one.embedding == 1229193216ull && one.head == 1229193216ull, "embedding and head");
  require(one.routers_and_norms == 98317312ull,
          "routers and norms: " + std::to_string(one.routers_and_norms));
  require(dgpp::minimax_kv_bytes_per_token(cfg, 1) == 253952, "K/V bytes per token at world 1");
  // Two ranks, the plan's choice (q_proj / k_proj replicated for the per-layer norm): 64.48 GiB
  // each.
  const dgpp::MinimaxWeightBytes two = dgpp::minimax_weight_bytes(cfg, 2, /*replicate_qk=*/true);
  require(two.routed_experts == 126382768128ull / 2 + 190464ull, "routed experts at world 2");
  require(two.attention == 2340421632ull + 390070272ull + 761856ull + 126976ull +
                               (2340421632ull + 390070272ull) / 2,
          "attention at world 2 (q / k replicated): " + std::to_string(two.attention));
  require(two.embedding == one.embedding && two.head * 2 == one.head,
          "the head is vocabulary-sharded, the embedding whole");
  require(two.routers_and_norms == one.routers_and_norms, "replicated");
  require(two.total() == 69230308352ull, "world 2 total per rank: " + std::to_string(two.total()));
  // The alternative (head-sharded q / k, the sum of squares over the bus): 63.20 GiB each.
  const dgpp::MinimaxWeightBytes alt = dgpp::minimax_weight_bytes(cfg, 2, /*replicate_qk=*/false);
  require(alt.total() == 67864617984ull,
          "world 2 total per rank, q / k sharded: " + std::to_string(alt.total()));
  require(dgpp::minimax_kv_bytes_per_token(cfg, 2) == 126976, "K/V bytes per token at world 2");
  for (int world : {1, 2, 4, 8})
    for (int rank = 0; rank < world; ++rank) dgpp::minimax_tp_validate_geometry(cfg, rank, world);
  for (int world : {3, 16}) {
    bool refused = false;
    try {
      dgpp::minimax_tp_validate_geometry(cfg, 0, world);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    require(refused, "world " + std::to_string(world) + " does not divide the heads");
  }
}

DGPP_TEST(minimax_binding_matches_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  const fs::path snap = landed_snapshot();
  if (snap.empty()) return;
  const dgpp::MinimaxTextConfig cfg =
      dgpp::MinimaxTextConfig::from_json_file((snap / "config.json").string());
  std::string index_text;
  {
    FILE* f = std::fopen((snap / "model.safetensors.index.json").c_str(), "rb");
    require(f != nullptr, "cannot open the shard index");
    char buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) index_text.append(buf, n);
    std::fclose(f);
  }
  const auto index = dgpp::minijson::parse(index_text);
  std::unordered_map<std::string, dgpp::MinimaxTensorDesc> present;
  size_t tensors = 0;
  for (const std::string& shard : dgpp::minimax_shard_files(index.root)) {
    const auto file = dgpp::SafetensorsFile::open((snap / shard).string());
    file->for_each([&](const dgpp::TensorInfo& t) {
      ++tensors;
      require(present.emplace(t.name, dgpp::MinimaxTensorDesc{t.dtype, t.shape}).second,
              "tensor '" + t.name + "' is in two shards");
    });
  }
  const dgpp::MinimaxBindReport rep = dgpp::minimax_validate_binding(cfg, present);
  std::string errors;
  for (const auto& e : rep.errors) errors += "\n  " + e;
  require(rep.ok(), "binding failed:" + errors);
  require(tensors == 191069 && rep.bound() == 191069 && rep.matched == 143471,
          "table and headers agree on 191,069");
  require(rep.fp4_matrices == 47616 && rep.loaded_bytes == 134401534976ull, "matrices and bytes");
}
