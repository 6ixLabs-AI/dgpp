// The Mistral-Small-4 expected-tensor table: its shape on the release's
// params.json — pinned to the tensor count and byte total of the real
// checkpoint's safetensors headers (56,313 tensors / 70,801,959,560 bytes,
// read from Hugging Face 2026-10-04) — the containers the recipe names, the
// vision tensors a text engine never loads, the validator's report, the
// per-rank weight bytes and, when the checkpoint is in the hub cache, the
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
#include "mistral4_config_json.hpp"
#include "models/mistral4/binding.hpp"
#include "models/mistral4/config.hpp"

namespace {

using dgpp::DType;
using dgpp::Mistral4ExpectedTensor;
using dgpp::Mistral4TensorRole;
using dgpp::Mistral4WeightClass;
using Shape = std::vector<int64_t>;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

const Mistral4ExpectedTensor& find(const std::vector<Mistral4ExpectedTensor>& v,
                                   const std::string& name) {
  for (const auto& e : v)
    if (e.name == name) return e;
  throw std::runtime_error("expected tensor missing from the table: " + name);
}
bool absent(const std::vector<Mistral4ExpectedTensor>& v, const std::string& name) {
  for (const auto& e : v)
    if (e.name == name) return false;
  return true;
}
bool is(const Mistral4ExpectedTensor& e, DType dtype, const Shape& shape) {
  return e.dtype == dtype && e.shape == shape;
}

size_t table_bytes(const std::vector<Mistral4ExpectedTensor>& v) {
  size_t n = 0;
  for (const auto& e : v) n += e.nbytes();
  return n;
}
size_t count_role(const std::vector<Mistral4ExpectedTensor>& v, Mistral4TensorRole role) {
  size_t n = 0;
  for (const auto& e : v) n += e.role == role;
  return n;
}

std::unordered_map<std::string, dgpp::Mistral4TensorDesc> present_of(
    const std::vector<Mistral4ExpectedTensor>& v) {
  std::unordered_map<std::string, dgpp::Mistral4TensorDesc> present;
  for (const auto& e : v) present.emplace(e.name, dgpp::Mistral4TensorDesc{e.dtype, e.shape});
  return present;
}

std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root =
      fs::path(home) /
      ".cache/huggingface/hub/models--mistralai--Mistral-Small-4-119B-2603-NVFP4/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (fs::exists(snap.path() / "params.json") &&
        fs::exists(snap.path() / "consolidated.safetensors.index.json"))
      return snap.path();
  return {};
}

}  // namespace

DGPP_TEST(mistral4_binding_table_has_the_release_shape) {
  const dgpp::Mistral4TextConfig cfg = mistral4_test::release();
  // A layer: 2 norms + 7 attention + the router + 129 experts x 3 matrices x 4 tensors.
  const auto layer = dgpp::mistral4_expected_layer_tensors(cfg, 0);
  require(layer.size() == 2 + 7 + 1 + 129 * 3 * 4, "layer " + std::to_string(layer.size()));
  require(layer.size() == 1558, "1,558 tensors per layer");
  require(dgpp::mistral4_expected_global_tensors(cfg).size() == 3, "globals");
  // The tower: 4 stem / projector tensors, 24 layers x 9, the adapter's 2.
  const auto vision = dgpp::mistral4_expected_vision_tensors(cfg);
  require(vision.size() == 4 + 24 * 9 + 2, "vision " + std::to_string(vision.size()));
  const auto all = dgpp::mistral4_expected_tensors(cfg);
  require(all.size() == 3 + 36 * 1558 + 222, "table size " + std::to_string(all.size()));
  require(all.size() == 56313, "the headers' 56,313 tensors");
  require(table_bytes(all) == 70801959560ull,
          "the headers' bytes: " + std::to_string(table_bytes(all)));
  require(count_role(all, Mistral4TensorRole::Fp4Payload) == 36 * 129 * 3, "13,932 NVFP4 matrices");
  require(count_role(all, Mistral4TensorRole::InputScale) == 13932 &&
              count_role(all, Mistral4TensorRole::Unused) == 222,
          "unused tensors");
  size_t loaded = 0, vision_bytes = 0;
  for (const auto& e : all) {
    if (!e.unused()) loaded += e.nbytes();
    if (e.cls == Mistral4WeightClass::Vision) vision_bytes += e.nbytes();
  }
  require(vision_bytes == 856944640ull, "the tower's bytes");
  require(loaded == 69944959408ull, "the text model's bytes as stored: " + std::to_string(loaded));
  // Names are unique.
  require(present_of(all).size() == all.size(), "duplicate names in the table");
}

DGPP_TEST(mistral4_binding_names_and_containers) {
  const dgpp::Mistral4TextConfig cfg = mistral4_test::release();
  const auto g = dgpp::mistral4_expected_global_tensors(cfg);
  require(is(find(g, "tok_embeddings.weight"), DType::BF16, {131072, 4096}), "embedding");
  require(is(find(g, "output.weight"), DType::BF16, {131072, 4096}), "head");
  require(is(find(g, "norm.weight"), DType::BF16, {4096}), "final norm");
  const auto l = dgpp::mistral4_expected_layer_tensors(cfg, 17);
  require(is(find(l, "layers.17.attention_norm.weight"), DType::BF16, {4096}), "attention_norm");
  require(is(find(l, "layers.17.ffn_norm.weight"), DType::BF16, {4096}), "ffn_norm");
  // MLA.
  require(is(find(l, "layers.17.attention.wq_a.weight"), DType::BF16, {1024, 4096}), "wq_a");
  require(is(find(l, "layers.17.attention.q_a_norm.weight"), DType::BF16, {1024}), "q_a_norm");
  require(is(find(l, "layers.17.attention.wq_b.weight"), DType::BF16, {4096, 1024}),
          "wq_b: 32 x (64 + 64)");
  require(is(find(l, "layers.17.attention.wkv_a_with_mqa.weight"), DType::BF16, {320, 4096}),
          "wkv_a: 256 + 64");
  require(is(find(l, "layers.17.attention.kv_a_norm.weight"), DType::BF16, {256}), "kv_a_norm");
  require(is(find(l, "layers.17.attention.wkv_b.weight"), DType::BF16, {6144, 256}),
          "wkv_b: 32 x (64 + 128)");
  require(is(find(l, "layers.17.attention.wo.weight"), DType::BF16, {4096, 4096}), "wo");
  require(is(find(l, "layers.17.gate.weight"), DType::BF16, {128, 4096}), "router");
  require(find(l, "layers.17.gate.weight").cls == Mistral4WeightClass::Router, "router class");
  // A routed expert: w1 / w3 [2048, 4096], w2 [4096, 2048], each a quadruple.
  const auto& w1 = find(l, "layers.17.experts.127.w1.weight_packed");
  require(is(w1, DType::U8, {2048, 2048}) && w1.role == Mistral4TensorRole::Fp4Payload &&
              w1.expert == 127 && w1.layer == 17 && w1.cls == Mistral4WeightClass::RoutedExpert,
          "w1 payload");
  require(is(find(l, "layers.17.experts.127.w1.weight_scale"), DType::F8_E4M3, {2048, 256}),
          "w1 scales");
  require(is(find(l, "layers.17.experts.127.w1.weight_global_scale"), DType::F32, {1}),
          "w1 global");
  require(is(find(l, "layers.17.experts.127.w2.weight_packed"), DType::U8, {4096, 1024}),
          "w2 payload");
  require(is(find(l, "layers.17.experts.127.w2.weight_scale"), DType::F8_E4M3, {4096, 128}),
          "w2 scales");
  require(is(find(l, "layers.17.experts.127.w3.weight_packed"), DType::U8, {2048, 2048}),
          "w3 payload");
  // The activation scale's dtype is the release's own: F32 on routed experts, BF16 on the shared
  // one.
  const auto& in_r = find(l, "layers.17.experts.0.w2.input_global_scale");
  require(is(in_r, DType::F32, {1}) && in_r.unused(), "routed input scale");
  const auto& in_s = find(l, "layers.17.shared_experts.w2.input_global_scale");
  require(is(in_s, DType::BF16, {1}) && in_s.unused(), "shared input scale");
  require(is(find(l, "layers.17.shared_experts.w1.weight_packed"), DType::U8, {2048, 2048}),
          "shared w1");
  require(is(find(l, "layers.17.shared_experts.w3.weight_global_scale"), DType::F32, {1}),
          "shared global");
  require(find(l, "layers.17.shared_experts.w1.weight_packed").expert == -1,
          "the shared expert has no id");
  require(absent(l, "layers.17.experts.128.w1.weight_packed") &&
              absent(l, "layers.17.attention.wq_a.bias"),
          "bounds");
  // The tower, as named in the checkpoint.
  const auto v = dgpp::mistral4_expected_vision_tensors(cfg);
  require(is(find(v, "vision_encoder.patch_conv.weight"), DType::BF16, {1024, 3, 14, 14}),
          "patch conv");
  require(is(find(v, "vision_encoder.transformer.layers.23.feed_forward.w2.weight"), DType::BF16,
             {1024, 4096}),
          "vision w2");
  require(is(find(v, "patch_merger.merging_layer.weight"), DType::BF16, {1024, 4096}),
          "patch merger");
  require(is(find(v, "vision_language_adapter.w_in.weight"), DType::BF16, {4096, 1024}),
          "adapter in");
  require(is(find(v, "vision_language_adapter.w_out.weight"), DType::BF16, {4096, 4096}),
          "adapter out");
  for (const auto& e : v)
    require(e.unused() && e.cls == Mistral4WeightClass::Vision, "vision tensors are never loaded");
}

DGPP_TEST(mistral4_binding_validator_reports_by_class) {
  const dgpp::Mistral4TextConfig cfg = mistral4_test::release();
  const auto all = dgpp::mistral4_expected_tensors(cfg);
  auto present = present_of(all);
  {
    const dgpp::Mistral4BindReport rep = dgpp::mistral4_validate_binding(cfg, present);
    require(rep.ok() && rep.expected == 56313 && rep.matched == 56313,
            "a complete checkpoint binds");
    require(rep.fp4_matrices == 13932 && rep.unused == 13932 + 222, "matrix and unused counts");
    require(rep.bytes == 70801959560ull && rep.loaded_bytes == 69944959408ull, "bytes");
  }
  present.erase("layers.7.experts.3.w2.weight_global_scale");
  present["layers.5.attention.wo.weight"].dtype = DType::F8_E4M3;
  present["layers.0.attention.wkv_b.weight"].shape = {6144, 512};
  // The shared expert's activation scale stored the way the routed ones are is a dtype error.
  present["layers.1.shared_experts.w1.input_global_scale"].dtype = DType::F32;
  // The transformers-format names are not this release's.
  present.emplace("model.layers.0.self_attn.q_a_proj.weight",
                  dgpp::Mistral4TensorDesc{DType::BF16, {1024, 4096}});
  const dgpp::Mistral4BindReport rep = dgpp::mistral4_validate_binding(cfg, present);
  require(!rep.ok(), "a broken checkpoint is refused");
  require(
      rep.missing == 1 && rep.dtype_mismatch == 2 && rep.shape_mismatch == 1 && rep.unexpected == 1,
      "the report counts by class");
  require(rep.errors.size() == 5, "five errors");
  require(dgpp::mistral4_validate_binding(cfg, {}, 5).errors.size() == 5, "errors are capped");
  require(dgpp::mistral4_validate_binding(cfg, {}).missing == all.size(),
          "an empty checkpoint misses everything");
}

DGPP_TEST(mistral4_binding_follows_the_recipe_and_the_tower) {
  const dgpp::Mistral4TextConfig base = mistral4_test::release();
  // Weight-only NVFP4: no activation scales in the file.
  const dgpp::Mistral4TextConfig a16 = mistral4_test::parse(
      mistral4_test::patched("\"input_activations\": {", "\"input_activations_off\": {"));
  const auto t = dgpp::mistral4_expected_tensors(a16);
  require(t.size() == 56313 - 13932, "table size without the activation scales");
  require(absent(t, "layers.0.experts.0.w1.input_global_scale"), "no input scale");
  const dgpp::Mistral4BindReport rep =
      dgpp::mistral4_validate_binding(a16, present_of(dgpp::mistral4_expected_tensors(base)));
  require(!rep.ok() && rep.unexpected == 13932, "recipe / checkpoint mismatch is reported");
  // A text-only checkpoint: no tower in params.json, none expected.
  const dgpp::Mistral4TextConfig text = mistral4_test::parse(
      mistral4_test::patched("\"vision_encoder\": {", "\"vision_encoder_unused\": {"));
  require(dgpp::mistral4_expected_tensors(text).size() == 56313 - 222,
          "table size without the tower");
  // NVFP4 needs K to be a multiple of 16.
  dgpp::Mistral4TextConfig odd = base;
  odd.hidden_size = 4104;
  bool refused = false;
  try {
    (void)dgpp::mistral4_expected_layer_tensors(odd, 0);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "an NVFP4 matrix whose K is not a multiple of 16");
}

DGPP_TEST(mistral4_weight_bytes_per_rank) {
  const dgpp::Mistral4TextConfig cfg = mistral4_test::release();
  // One Spark: the text model as stored, 65.14 GiB.
  const dgpp::Mistral4WeightBytes one = dgpp::mistral4_weight_bytes(cfg, 1);
  require(one.total() == 69944959408ull, "world 1 total: " + std::to_string(one.total()));
  require(one.embedding == 1073741824ull && one.head == 1073741824ull, "embedding and head");
  require(one.attention == 2019649536ull, "attention: " + std::to_string(one.attention));
  require(one.routers_and_norms == 38346752ull,
          "routers and norms: " + std::to_string(one.routers_and_norms));
  require(one.shared_experts == 509608368ull,
          "shared experts: " + std::to_string(one.shared_experts));
  require(one.routed_experts == 65229871104ull,
          "routed experts: " + std::to_string(one.routed_experts));
  // Two ranks: the sliced classes halve, the globals and the latents' projections stay.
  const dgpp::Mistral4WeightBytes two = dgpp::mistral4_weight_bytes(cfg, 2);
  require(two.embedding == one.embedding && two.head * 2 == one.head,
          "the head is vocabulary-sharded, the embedding whole");
  // wq_b + wkv_b + wo (1,623,195,648 bytes) are head-sharded; wq_a, wkv_a_with_mqa and the two
  // latent norms are not.
  require(two.attention == 396453888ull + 1623195648ull / 2,
          "attention at world 2: " + std::to_string(two.attention));
  require(two.routers_and_norms == one.routers_and_norms, "replicated");
  require(two.routed_experts == (one.routed_experts - 55296) / 2 + 55296,
          "routed experts at world 2");
  for (int world : {1, 2, 4})
    for (int rank = 0; rank < world; ++rank) dgpp::mistral4_tp_validate_geometry(cfg, rank, world);
  bool refused = false;
  try {
    dgpp::mistral4_tp_validate_geometry(cfg, 0, 3);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "world 3 does not divide 32 heads");
}

DGPP_TEST(mistral4_binding_matches_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  const fs::path snap = landed_snapshot();
  if (snap.empty()) return;
  const dgpp::Mistral4TextConfig cfg =
      dgpp::Mistral4TextConfig::from_json_file((snap / "params.json").string());
  // The shard list is the index's: nothing else in the directory is the model.
  std::string index_text;
  {
    FILE* f = std::fopen((snap / "consolidated.safetensors.index.json").c_str(), "rb");
    require(f != nullptr, "cannot open the shard index");
    char buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) index_text.append(buf, n);
    std::fclose(f);
  }
  const auto index = dgpp::minijson::parse(index_text);
  std::vector<std::string> shards;
  for (const auto& m : index.root.at("weight_map").members()) {
    const std::string file(m.value.as_string());
    bool seen = false;
    for (const auto& s : shards) seen = seen || s == file;
    if (!seen) shards.push_back(file);
  }
  std::unordered_map<std::string, dgpp::Mistral4TensorDesc> present;
  size_t tensors = 0;
  for (const std::string& shard : shards) {
    const auto file = dgpp::SafetensorsFile::open((snap / shard).string());
    file->for_each([&](const dgpp::TensorInfo& t) {
      ++tensors;
      require(present.emplace(t.name, dgpp::Mistral4TensorDesc{t.dtype, t.shape}).second,
              "tensor '" + t.name + "' is in two shards");
    });
  }
  const dgpp::Mistral4BindReport rep = dgpp::mistral4_validate_binding(cfg, present);
  std::string errors;
  for (const auto& e : rep.errors) errors += "\n  " + e;
  require(rep.ok(), "binding failed:" + errors);
  require(tensors == 56313 && rep.expected == 56313 && rep.matched == 56313,
          "table and headers agree on 56,313");
  require(rep.fp4_matrices == 13932 && rep.bytes == 70801959560ull, "matrices and bytes");
}
