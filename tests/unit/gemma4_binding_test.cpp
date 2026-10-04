// The Gemma 4 expected-tensor table: its shape on both releases' configs —
// pinned to the tensor counts and byte totals of the real checkpoints'
// safetensors headers (nvidia/Gemma-4-31B-IT-NVFP4 @ 4135a98a: 1,728 tensors
// in four shards, of which 356 belong to the vision tower and its projector,
// 31,481,767,960 bytes outside the encoders;
// bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16: 35,923 tensors in three
// shards, the same 356 for the encoders, 15,271,399,280 bytes outside them) —
// the per-class formats of the recipes, the ignore rule, the validator's
// report and, when a checkpoint is in the hub cache, the full binding
// against every shard's header.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "gemma4_config_json.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "models/gemma4/binding.hpp"
#include "models/gemma4/config.hpp"

namespace {

using dgpp::DType;
using dgpp::Gemma4ExpectedTensor;
using dgpp::Gemma4TensorRole;
using Shape = std::vector<int64_t>;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Gemma4TextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Gemma4TextConfig::parse(t.root);
}
dgpp::Gemma4TextConfig release_31b() { return parse(gemma4_test::config_json()); }

const Gemma4ExpectedTensor& find(const std::vector<Gemma4ExpectedTensor>& v, const std::string& name) {
  for (const auto& e : v)
    if (e.name == name) return e;
  throw std::runtime_error("expected tensor missing from the table: " + name);
}
bool absent(const std::vector<Gemma4ExpectedTensor>& v, const std::string& name) {
  for (const auto& e : v)
    if (e.name == name) return false;
  return true;
}
bool is(const Gemma4ExpectedTensor& e, DType dtype, const Shape& shape) { return e.dtype == dtype && e.shape == shape; }

size_t table_bytes(const std::vector<Gemma4ExpectedTensor>& v) {
  size_t n = 0;
  for (const auto& e : v) n += e.nbytes();
  return n;
}
size_t count_role(const std::vector<Gemma4ExpectedTensor>& v, Gemma4TensorRole role) {
  size_t n = 0;
  for (const auto& e : v) n += e.role == role;
  return n;
}

std::unordered_map<std::string, dgpp::Gemma4TensorDesc> present_of(const std::vector<Gemma4ExpectedTensor>& v) {
  std::unordered_map<std::string, dgpp::Gemma4TensorDesc> present;
  for (const auto& e : v) present.emplace(e.name, dgpp::Gemma4TensorDesc{e.dtype, e.shape});
  return present;
}

// The snapshot of `repo` in the hub cache with every shard of its index
// present (a download may be in flight), or empty.
std::filesystem::path landed_snapshot(const char* repo) {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root = fs::path(home) / ".cache/huggingface/hub" / repo / "snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root)) {
    const fs::path index = snap.path() / "model.safetensors.index.json";
    if (!fs::exists(snap.path() / "config.json") || !fs::exists(index)) continue;
    std::ifstream idx(index);
    const std::string text((std::istreambuf_iterator<char>(idx)), std::istreambuf_iterator<char>());
    const auto parsed = dgpp::minijson::parse(text);
    bool complete = true;
    for (const auto& m : parsed.root.at("weight_map").members())
      complete = complete && fs::exists(snap.path() / std::string(m.value.as_string()));
    if (complete) return snap.path();
  }
  return {};
}

}  // namespace

DGPP_TEST(gemma4_binding_table_has_the_31b_release_shape) {
  const dgpp::Gemma4TextConfig cfg = release_31b();
  // A sliding layer: 4 norms + layer_scalar + q_norm + k_norm + q/k/v/o (BF16) + 3 MLP sets of 4 = 23.
  const auto sliding = dgpp::gemma4_expected_layer_tensors(cfg, 0);
  require(sliding.size() == 23, "sliding layer " + std::to_string(sliding.size()));
  // A full layer has no v_proj: 22.
  const auto full = dgpp::gemma4_expected_layer_tensors(cfg, 5);
  require(full.size() == 22, "full layer " + std::to_string(full.size()));
  const auto globals = dgpp::gemma4_expected_global_tensors(cfg);
  require(globals.size() == 2, "globals");
  const auto all = dgpp::gemma4_expected_text_tensors(cfg);
  // The real headers: 1,728 tensors, 356 of them the vision tower's and its projector's.
  require(all.size() == 2 + 50 * 23 + 10 * 22, "table size");
  require(all.size() == 1372, "table size vs the checkpoint's headers: " + std::to_string(all.size()));
  require(all.size() + 356 == 1728, "table + ignored = headers");
  require(table_bytes(all) == 31481767960ull, "table bytes vs the headers: " + std::to_string(table_bytes(all)));
  require(count_role(all, Gemma4TensorRole::Fp4Payload) == 180, "nvfp4 matrices");
  require(count_role(all, Gemma4TensorRole::InputScale) == 180, "activation scales");
  // Names are unique.
  require(present_of(all).size() == all.size(), "duplicate names in the table");
}

DGPP_TEST(gemma4_binding_formats_follow_the_recipe) {
  const dgpp::Gemma4TextConfig cfg = release_31b();
  const auto all = dgpp::gemma4_expected_text_tensors(cfg);
  const std::string s = "model.language_model.layers.0.", f = "model.language_model.layers.5.";
  require(is(find(all, "model.language_model.embed_tokens.weight"), DType::BF16, {262144, 5376}), "embedding");
  require(is(find(all, "model.language_model.norm.weight"), DType::BF16, {5376}), "final norm");
  require(absent(all, "lm_head.weight") && absent(all, "model.language_model.lm_head.weight"), "no head tensor (tied)");
  for (const char* n : {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"})
    require(is(find(all, s + n + ".weight"), DType::BF16, {5376}), std::string("norm ") + n);
  require(is(find(all, s + "layer_scalar"), DType::BF16, {1}), "layer_scalar");
  // Sliding attention: 32 heads x 256, 16 KV heads, its own v_proj; all BF16 (the ignore list).
  require(is(find(all, s + "self_attn.q_norm.weight"), DType::BF16, {256}), "sliding q_norm");
  require(is(find(all, s + "self_attn.k_norm.weight"), DType::BF16, {256}), "sliding k_norm");
  require(absent(all, s + "self_attn.v_norm.weight"), "the value norm has no weight");
  require(is(find(all, s + "self_attn.q_proj.weight"), DType::BF16, {8192, 5376}), "sliding q_proj");
  require(is(find(all, s + "self_attn.k_proj.weight"), DType::BF16, {4096, 5376}), "sliding k_proj");
  require(is(find(all, s + "self_attn.v_proj.weight"), DType::BF16, {4096, 5376}), "sliding v_proj");
  require(is(find(all, s + "self_attn.o_proj.weight"), DType::BF16, {5376, 8192}), "sliding o_proj");
  require(absent(all, s + "self_attn.q_proj.weight_scale"), "BF16 attention has no NVFP4 scales");
  // Full attention: 32 heads x 512, 4 KV heads, the value read from k_proj.
  require(is(find(all, f + "self_attn.q_norm.weight"), DType::BF16, {512}), "full q_norm");
  require(is(find(all, f + "self_attn.k_norm.weight"), DType::BF16, {512}), "full k_norm");
  require(is(find(all, f + "self_attn.q_proj.weight"), DType::BF16, {16384, 5376}), "full q_proj");
  require(is(find(all, f + "self_attn.k_proj.weight"), DType::BF16, {2048, 5376}), "full k_proj");
  require(absent(all, f + "self_attn.v_proj.weight"), "a full layer has no v_proj");
  require(is(find(all, f + "self_attn.o_proj.weight"), DType::BF16, {5376, 16384}), "full o_proj");
  // The MLP: modelopt NVFP4 with the activation scale.
  for (const std::string& p : {s, f}) {
    const Gemma4ExpectedTensor gate = find(all, p + "mlp.gate_proj.weight");
    require(is(gate, DType::U8, {21504, 2688}) && gate.role == Gemma4TensorRole::Fp4Payload, "gate payload");
    require(is(find(all, p + "mlp.gate_proj.weight_scale"), DType::F8_E4M3, {21504, 336}), "gate scales");
    require(is(find(all, p + "mlp.gate_proj.weight_scale_2"), DType::F32, {}), "gate global");
    const Gemma4ExpectedTensor act = find(all, p + "mlp.gate_proj.input_scale");
    require(is(act, DType::F32, {}) && act.role == Gemma4TensorRole::InputScale && act.unused(), "gate activation scale");
    require(is(find(all, p + "mlp.up_proj.weight"), DType::U8, {21504, 2688}), "up payload");
    require(is(find(all, p + "mlp.down_proj.weight"), DType::U8, {5376, 10752}), "down payload");
    require(is(find(all, p + "mlp.down_proj.weight_scale"), DType::F8_E4M3, {5376, 1344}), "down scales");
  }
  require(find(all, s + "mlp.gate_proj.weight").layer == 0 && find(all, f + "layer_scalar").layer == 5, "layer tags");
  require(find(all, "model.language_model.norm.weight").layer == -1, "global tag");
}

DGPP_TEST(gemma4_binding_weight_only_recipe_quantizes_the_attention) {
  // The 31B geometry under the weight-only recipe (no attention on the ignore list): the projections
  // become NVFP4 sets of three, and no matrix carries an activation scale.
  const dgpp::Gemma4TextConfig cfg =
      parse(gemma4_test::config_json("", "", "\"lm_head\", \"model.embed_vision*\", \"model.vision_tower*\""));
  const auto all = dgpp::gemma4_expected_text_tensors(cfg);
  const std::string s = "model.language_model.layers.0.", f = "model.language_model.layers.5.";
  require(is(find(all, s + "self_attn.q_proj.weight"), DType::U8, {8192, 2688}), "q payload");
  require(is(find(all, s + "self_attn.q_proj.weight_scale"), DType::F8_E4M3, {8192, 336}), "q scales");
  require(is(find(all, s + "self_attn.q_proj.weight_scale_2"), DType::F32, {}), "q global");
  require(is(find(all, s + "self_attn.o_proj.weight"), DType::U8, {5376, 4096}), "o payload");
  require(is(find(all, s + "self_attn.o_proj.weight_scale"), DType::F8_E4M3, {5376, 512}), "o scales");
  require(is(find(all, f + "self_attn.o_proj.weight"), DType::U8, {5376, 8192}), "full o payload");
  require(is(find(all, f + "self_attn.k_proj.weight"), DType::U8, {2048, 2688}), "full k payload");
  require(absent(all, f + "self_attn.v_proj.weight"), "a full layer has no v_proj");
  require(is(find(all, s + "self_attn.q_norm.weight"), DType::BF16, {256}), "the norms stay BF16");
  require(count_role(all, Gemma4TensorRole::InputScale) == 0, "no activation scales");
  require(absent(all, s + "mlp.gate_proj.input_scale"), "no activation scale on the MLP");
  // sliding: 4 + 1 + 2 + 4x3 + 3x3 = 28; full: 25.
  require(dgpp::gemma4_expected_layer_tensors(cfg, 0).size() == 28, "sliding layer");
  require(dgpp::gemma4_expected_layer_tensors(cfg, 5).size() == 25, "full layer");
}

DGPP_TEST(gemma4_binding_ignores_the_encoders) {
  require(dgpp::gemma4_ignored_tensor("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"), "vision tower");
  require(dgpp::gemma4_ignored_tensor("model.vision_tower.std_bias"), "vision buffers");
  require(dgpp::gemma4_ignored_tensor("model.embed_vision.embedding_projection.weight"), "vision projector");
  require(dgpp::gemma4_ignored_tensor("model.audio_tower.conformer.0.x"), "audio tower");
  require(!dgpp::gemma4_ignored_tensor("model.language_model.embed_tokens.weight"), "the text model is not ignored");
  require(!dgpp::gemma4_ignored_tensor("lm_head.weight"), "an lm_head would be unexpected, not ignored");
  require(dgpp::gemma4_layer_prefix(17) == "model.language_model.layers.17.", "layer prefix");
}

DGPP_TEST(gemma4_binding_validator_reports) {
  const dgpp::Gemma4TextConfig cfg = release_31b();
  const auto all = dgpp::gemma4_expected_text_tensors(cfg);
  auto present = present_of(all);
  // The encoders ride along.
  present.emplace("model.vision_tower.std_bias", dgpp::Gemma4TensorDesc{DType::BF16, {1152}});
  present.emplace("model.embed_vision.embedding_projection.weight", dgpp::Gemma4TensorDesc{DType::BF16, {5376, 1152}});
  {
    const auto rep = dgpp::gemma4_validate_text_binding(cfg, present);
    require(rep.ok() && rep.expected == 1372 && rep.matched == 1372 && rep.ignored == 2, "clean binding");
    require(rep.fp4_matrices == 180, "fp4 matrices " + std::to_string(rep.fp4_matrices));
    // 1 embedding + 50 x 4 + 10 x 3 projections.
    require(rep.bf16_matrices == 231, "bf16 matrices " + std::to_string(rep.bf16_matrices));
    require(rep.bytes == 31481767960ull, "matched bytes");
  }
  {
    auto p = present;
    p.erase("model.language_model.layers.59.layer_scalar");
    p["model.language_model.layers.3.mlp.down_proj.weight"].dtype = DType::BF16;
    p["model.language_model.layers.4.self_attn.k_proj.weight"].shape = {2048, 5376};
    p.emplace("lm_head.weight", dgpp::Gemma4TensorDesc{DType::BF16, {262144, 5376}});
    p.emplace("model.language_model.layers.5.self_attn.v_proj.weight", dgpp::Gemma4TensorDesc{DType::BF16, {2048, 5376}});
    const auto rep = dgpp::gemma4_validate_text_binding(cfg, p);
    require(!rep.ok(), "must fail");
    require(rep.missing == 1 && rep.dtype_mismatch == 1 && rep.shape_mismatch == 1 && rep.unexpected == 2,
            "counts: missing " + std::to_string(rep.missing) + " dtype " + std::to_string(rep.dtype_mismatch) + " shape " +
                std::to_string(rep.shape_mismatch) + " unexpected " + std::to_string(rep.unexpected));
    require(rep.errors.size() == 5, "five errors named");
  }
  // The error list is capped; the counts are not.
  {
    const auto rep = dgpp::gemma4_validate_text_binding(cfg, {}, 4);
    require(rep.missing == 1372 && rep.errors.size() == 4, "capped errors");
  }
  // A release without the activation scales does not bind under this recipe.
  {
    auto p = present;
    p.erase("model.language_model.layers.0.mlp.gate_proj.input_scale");
    require(dgpp::gemma4_validate_text_binding(cfg, p).missing == 1, "missing activation scale");
  }
}

DGPP_TEST(gemma4_binding_matches_the_checkpoint_when_present) {
  namespace fs = std::filesystem;
  for (const char* repo : {"models--nvidia--Gemma-4-31B-IT-NVFP4", "models--bg-digitalservices--Gemma-4-26B-A4B-it-NVFP4A16"}) {
    const fs::path snap = landed_snapshot(repo);
    if (snap.empty()) continue;  // not on this box
    const dgpp::Gemma4TextConfig cfg = dgpp::Gemma4TextConfig::from_json_file((snap / "config.json").string());
    std::unordered_map<std::string, dgpp::Gemma4TensorDesc> present;
    for (const auto& entry : fs::directory_iterator(snap)) {
      if (entry.path().extension() != ".safetensors") continue;
      auto f = dgpp::SafetensorsFile::open(entry.path().string());
      f->for_each([&](const dgpp::TensorInfo& t) { present.emplace(t.name, dgpp::Gemma4TensorDesc{t.dtype, t.shape}); });
    }
    const auto rep = dgpp::gemma4_validate_text_binding(cfg, present);
    std::string errors;
    for (const auto& e : rep.errors) errors += "\n  " + e;
    require(rep.ok(), std::string(repo) + ": binding against the checkpoint failed:" + errors);
    require(rep.expected + rep.ignored == present.size(), std::string(repo) + ": table + ignored must equal the header count");
  }
}

// ---- the 26B-A4B (the MoE block, the weight-only recipe) ----------------------

DGPP_TEST(gemma4_binding_table_has_the_26b_a4b_release_shape) {
  const dgpp::Gemma4TextConfig cfg = parse(gemma4_test::config_json_26b());
  // A sliding layer: 7 norms + layer_scalar + q_norm + k_norm + q/k/v/o x 3 + 3 MLP x 3 + router 3
  // + 128 experts x 3 matrices x 3 = 1186; a full layer has no v_proj: 1183.
  require(dgpp::gemma4_expected_layer_tensors(cfg, 0).size() == 1186, "sliding layer");
  require(dgpp::gemma4_expected_layer_tensors(cfg, 5).size() == 1183, "full layer");
  const auto all = dgpp::gemma4_expected_text_tensors(cfg);
  // The real headers: 35,923 tensors, 356 of them the encoders'.
  require(all.size() == 2 + 25 * 1186 + 5 * 1183 && all.size() == 35567, "table size " + std::to_string(all.size()));
  require(all.size() + 356 == 35923, "table + ignored = headers");
  require(table_bytes(all) == 15271399280ull, "table bytes vs the headers: " + std::to_string(table_bytes(all)));
  // 25 x 4 + 5 x 3 attention, 30 x 3 MLP, 30 x 128 x 3 expert matrices; no activation scales.
  require(count_role(all, Gemma4TensorRole::Fp4Payload) == 115 + 90 + 11520, "nvfp4 matrices");
  require(count_role(all, Gemma4TensorRole::InputScale) == 0, "no activation scales");
  require(present_of(all).size() == all.size(), "duplicate names in the table");
}

DGPP_TEST(gemma4_binding_26b_a4b_formats) {
  const dgpp::Gemma4TextConfig cfg = parse(gemma4_test::config_json_26b());
  const auto all = dgpp::gemma4_expected_text_tensors(cfg);
  const std::string s = "model.language_model.layers.0.", f = "model.language_model.layers.5.";
  require(is(find(all, "model.language_model.embed_tokens.weight"), DType::BF16, {262144, 2816}), "embedding");
  // The MoE block's own norms and its BF16 router.
  for (const char* n : {"post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2",
                        "post_feedforward_layernorm", "pre_feedforward_layernorm"})
    require(is(find(all, s + n + ".weight"), DType::BF16, {2816}), std::string("norm ") + n);
  require(is(find(all, s + "router.proj.weight"), DType::BF16, {128, 2816}), "router proj");
  require(is(find(all, s + "router.scale"), DType::BF16, {2816}), "router scale");
  require(is(find(all, s + "router.per_expert_scale"), DType::BF16, {128}), "per_expert_scale");
  require(absent(all, s + "router.proj.weight_scale"), "the router is not quantized");
  // The experts: one Linear each under `moe.experts.E.`, NVFP4 sets of three.
  const Gemma4ExpectedTensor gate = find(all, s + "moe.experts.127.gate_proj.weight");
  require(is(gate, DType::U8, {704, 1408}) && gate.expert == 127 && gate.layer == 0 &&
              gate.cls == dgpp::Gemma4WeightClass::RoutedExpert,
          "expert gate payload");
  require(is(find(all, s + "moe.experts.127.gate_proj.weight_scale"), DType::F8_E4M3, {704, 176}), "expert gate scales");
  require(is(find(all, s + "moe.experts.127.gate_proj.weight_scale_2"), DType::F32, {}), "expert gate global");
  require(is(find(all, s + "moe.experts.0.up_proj.weight"), DType::U8, {704, 1408}), "expert up payload");
  require(is(find(all, s + "moe.experts.0.down_proj.weight"), DType::U8, {2816, 352}), "expert down payload");
  require(is(find(all, s + "moe.experts.0.down_proj.weight_scale"), DType::F8_E4M3, {2816, 44}), "expert down scales");
  require(absent(all, s + "moe.experts.0.gate_proj.input_scale"), "weight-only: no activation scale");
  require(absent(all, s + "moe.experts.128.gate_proj.weight"), "128 experts");
  require(absent(all, s + "experts.gate_up_proj") && absent(all, s + "experts.0.gate_proj.weight"),
          "the class's stacked form and the quantizer's pre-rename form are not this release's names");
  // The dense MLP beside them.
  require(is(find(all, s + "mlp.gate_proj.weight"), DType::U8, {2112, 1408}), "mlp gate payload");
  require(is(find(all, s + "mlp.down_proj.weight"), DType::U8, {2816, 1056}), "mlp down payload");
  // NVFP4 attention: sliding 16 heads x 256 on 8 KV heads; full 16 x 512 on 2, no v_proj.
  require(is(find(all, s + "self_attn.q_proj.weight"), DType::U8, {4096, 1408}), "sliding q payload");
  require(is(find(all, s + "self_attn.k_proj.weight"), DType::U8, {2048, 1408}), "sliding k payload");
  require(is(find(all, s + "self_attn.v_proj.weight_scale"), DType::F8_E4M3, {2048, 176}), "sliding v scales");
  require(is(find(all, s + "self_attn.o_proj.weight"), DType::U8, {2816, 2048}), "sliding o payload");
  require(is(find(all, s + "self_attn.o_proj.weight_scale"), DType::F8_E4M3, {2816, 256}), "sliding o scales");
  require(is(find(all, f + "self_attn.q_proj.weight"), DType::U8, {8192, 1408}), "full q payload");
  require(is(find(all, f + "self_attn.k_proj.weight"), DType::U8, {1024, 1408}), "full k payload");
  require(is(find(all, f + "self_attn.o_proj.weight"), DType::U8, {2816, 4096}), "full o payload");
  require(absent(all, f + "self_attn.v_proj.weight"), "a full layer has no v_proj");
  require(is(find(all, f + "self_attn.q_norm.weight"), DType::BF16, {512}), "full q_norm");
  // The validator on the table itself, and its expert count.
  const auto rep = dgpp::gemma4_validate_text_binding(cfg, present_of(all));
  require(rep.ok() && rep.fp4_matrices == 11725 && rep.expert_matrices == 11520 && rep.bf16_matrices == 31,
          "report: fp4 " + std::to_string(rep.fp4_matrices) + " experts " + std::to_string(rep.expert_matrices) + " bf16 " +
              std::to_string(rep.bf16_matrices));
  require(rep.bytes == 15271399280ull, "matched bytes");
}
