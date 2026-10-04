// The Nemotron-H expected-tensor table: its shape on both releases' configs
// — pinned to the tensor counts and byte totals of the real checkpoints'
// safetensors headers (Nano 24,147 tensors / 19,339,781,632 bytes; Super
// 165,860 / 80,297,329,824) — the per-module formats the recipes name, the
// draft block, the validator's report and, when a checkpoint is in the hub
// cache, the full binding against every shard's header.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "models/nemotron/binding.hpp"
#include "models/nemotron/config.hpp"
#include "nemotron_config_json.hpp"

namespace {

using dgpp::DType;
using dgpp::NemotronExpectedTensor;
using dgpp::NemotronTensorRole;
using dgpp::NemotronWeightClass;
using Shape = std::vector<int64_t>;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

const NemotronExpectedTensor& find(const std::vector<NemotronExpectedTensor>& v,
                                   const std::string& name) {
  for (const auto& e : v)
    if (e.name == name) return e;
  throw std::runtime_error("expected tensor missing from the table: " + name);
}
bool absent(const std::vector<NemotronExpectedTensor>& v, const std::string& name) {
  for (const auto& e : v)
    if (e.name == name) return false;
  return true;
}
bool is(const NemotronExpectedTensor& e, DType dtype, const Shape& shape) {
  return e.dtype == dtype && e.shape == shape;
}

size_t table_bytes(const std::vector<NemotronExpectedTensor>& v) {
  size_t n = 0;
  for (const auto& e : v) n += e.nbytes();
  return n;
}
size_t count_role(const std::vector<NemotronExpectedTensor>& v, NemotronTensorRole role) {
  size_t n = 0;
  for (const auto& e : v) n += e.role == role;
  return n;
}

std::unordered_map<std::string, dgpp::NemotronTensorDesc> present_of(
    const std::vector<NemotronExpectedTensor>& v) {
  std::unordered_map<std::string, dgpp::NemotronTensorDesc> present;
  for (const auto& e : v) present.emplace(e.name, dgpp::NemotronTensorDesc{e.dtype, e.shape});
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

// Every shard's header through the validator; a name in two shards is an
// error (the engine's stream refuses duplicates).
dgpp::NemotronBindReport bind_snapshot(const std::filesystem::path& snap,
                                       const dgpp::NemotronHConfig& cfg, size_t* tensors) {
  namespace fs = std::filesystem;
  std::unordered_map<std::string, dgpp::NemotronTensorDesc> present;
  *tensors = 0;
  for (const auto& entry : fs::directory_iterator(snap)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      ++*tensors;
      require(present.emplace(t.name, dgpp::NemotronTensorDesc{t.dtype, t.shape}).second,
              "tensor '" + t.name + "' is in two shards");
    });
  }
  return dgpp::nemotron_validate_binding(cfg, present);
}

void require_bound(const dgpp::NemotronBindReport& rep) {
  std::string errs;
  for (const auto& e : rep.errors) errs += "\n  " + e;
  require(rep.ok(), "binding: missing " + std::to_string(rep.missing) + ", dtype " +
                        std::to_string(rep.dtype_mismatch) + ", shape " +
                        std::to_string(rep.shape_mismatch) + ", unexpected " +
                        std::to_string(rep.unexpected) + errs);
}

}  // namespace

DGPP_TEST(nemotron_binding_table_has_the_nano_shape) {
  const dgpp::NemotronHConfig cfg = nemotron_test::nano();
  // A Mamba layer: the pre-norm, six vectors, in_proj and out_proj as NVFP4
  // sets of four.
  const auto mamba = dgpp::nemotron_expected_layer_tensors(cfg, 0);
  require(mamba.size() == 1 + 6 + 4 + 4, "mamba layer " + std::to_string(mamba.size()));
  // The Mamba layer in front of an attention layer: both projections BF16.
  const auto mamba_bf16 = dgpp::nemotron_expected_layer_tensors(cfg, 4);
  require(mamba_bf16.size() == 1 + 6 + 1 + 1,
          "excluded mamba layer " + std::to_string(mamba_bf16.size()));
  const auto attn = dgpp::nemotron_expected_layer_tensors(cfg, 5);
  require(attn.size() == 1 + 4, "attention layer " + std::to_string(attn.size()));
  // A MoE layer: the pre-norm, the router pair, 128 experts x 2 x 4, the shared expert 2 x 4.
  const auto moe = dgpp::nemotron_expected_layer_tensors(cfg, 1);
  require(moe.size() == 1 + 2 + 128 * 8 + 8, "moe layer " + std::to_string(moe.size()));
  const auto all = dgpp::nemotron_expected_tensors(cfg);
  require(all.size() ==
              3 + 17 * mamba.size() + 6 * mamba_bf16.size() + 6 * attn.size() + 23 * moe.size(),
          "table sum");
  // The real checkpoint's header: 24,147 tensors, 19,339,781,632 tensor bytes.
  require(all.size() == 24147, "table size " + std::to_string(all.size()));
  require(table_bytes(all) == 19339781632ull, "table bytes " + std::to_string(table_bytes(all)));
  require(count_role(all, NemotronTensorRole::Fp4Payload) == 5968 &&
              count_role(all, NemotronTensorRole::Fp8Payload) == 0,
          "quantized matrices");

  require(is(find(all, "backbone.embeddings.weight"), DType::BF16, {131072, 2688}), "embedding");
  require(is(find(all, "lm_head.weight"), DType::BF16, {131072, 2688}), "head");
  require(is(find(all, "backbone.norm_f.weight"), DType::BF16, {2688}), "final norm");
  require(is(find(mamba, "backbone.layers.0.norm.weight"), DType::BF16, {2688}), "pre-norm");
  // Mamba2: in_proj rows = inner 4096 + conv_dim 6144 + heads 64.
  require(is(find(mamba, "backbone.layers.0.mixer.in_proj.weight"), DType::U8, {10304, 1344}),
          "in_proj payload");
  require(
      is(find(mamba, "backbone.layers.0.mixer.in_proj.weight_scale"), DType::F8_E4M3, {10304, 168}),
      "in_proj scales");
  require(is(find(mamba, "backbone.layers.0.mixer.in_proj.weight_scale_2"), DType::F32, {}),
          "in_proj global");
  require(is(find(mamba, "backbone.layers.0.mixer.in_proj.input_scale"), DType::F32, {}),
          "in_proj input scale");
  require(is(find(mamba, "backbone.layers.0.mixer.out_proj.weight"), DType::U8, {2688, 2048}),
          "out_proj payload");
  require(
      is(find(mamba, "backbone.layers.0.mixer.out_proj.weight_scale"), DType::F8_E4M3, {2688, 256}),
      "out_proj scales");
  require(is(find(mamba, "backbone.layers.0.mixer.conv1d.weight"), DType::BF16, {6144, 1, 4}),
          "conv weight");
  require(is(find(mamba, "backbone.layers.0.mixer.conv1d.bias"), DType::BF16, {6144}), "conv bias");
  require(is(find(mamba, "backbone.layers.0.mixer.norm.weight"), DType::BF16, {4096}),
          "gated norm");
  for (const char* v : {"A_log", "D", "dt_bias"})
    require(is(find(mamba, std::string("backbone.layers.0.mixer.") + v), DType::BF16, {64}), v);
  require(
      is(find(mamba_bf16, "backbone.layers.4.mixer.in_proj.weight"), DType::BF16, {10304, 2688}),
      "BF16 in_proj");
  require(
      is(find(mamba_bf16, "backbone.layers.4.mixer.out_proj.weight"), DType::BF16, {2688, 4096}),
      "BF16 out_proj");
  require(absent(mamba_bf16, "backbone.layers.4.mixer.in_proj.weight_scale"),
          "a BF16 linear carries no scales");
  // Attention: 32 x 128 query rows, 2 x 128 key/value rows; no cache scales in this release.
  require(is(find(attn, "backbone.layers.5.mixer.q_proj.weight"), DType::BF16, {4096, 2688}),
          "q_proj");
  require(is(find(attn, "backbone.layers.5.mixer.k_proj.weight"), DType::BF16, {256, 2688}),
          "k_proj");
  require(is(find(attn, "backbone.layers.5.mixer.v_proj.weight"), DType::BF16, {256, 2688}),
          "v_proj");
  require(is(find(attn, "backbone.layers.5.mixer.o_proj.weight"), DType::BF16, {2688, 4096}),
          "o_proj");
  require(absent(attn, "backbone.layers.5.mixer.k_proj.k_scale"), "no K/V-cache scales");
  // MoE: an F32 router, two matrices per expert.
  const NemotronExpectedTensor gate = find(moe, "backbone.layers.1.mixer.gate.weight");
  require(is(gate, DType::F32, {128, 2688}) && gate.cls == NemotronWeightClass::Router,
          "router gate");
  require(is(find(moe, "backbone.layers.1.mixer.gate.e_score_correction_bias"), DType::F32, {128}),
          "router bias");
  const NemotronExpectedTensor up = find(moe, "backbone.layers.1.mixer.experts.7.up_proj.weight");
  require(is(up, DType::U8, {1856, 1344}) && up.expert == 7 &&
              up.cls == NemotronWeightClass::RoutedExpert &&
              up.role == NemotronTensorRole::Fp4Payload && up.quantized(),
          "expert up payload");
  require(is(find(moe, "backbone.layers.1.mixer.experts.7.up_proj.weight_scale"), DType::F8_E4M3,
             {1856, 168}),
          "expert up scales");
  require(
      is(find(moe, "backbone.layers.1.mixer.experts.7.down_proj.weight"), DType::U8, {2688, 928}),
      "expert down payload");
  require(is(find(moe, "backbone.layers.1.mixer.experts.7.down_proj.weight_scale"), DType::F8_E4M3,
             {2688, 116}),
          "expert down scales");
  require(is(find(moe, "backbone.layers.1.mixer.shared_experts.up_proj.weight"), DType::U8,
             {3712, 1344}),
          "shared up");
  require(is(find(moe, "backbone.layers.1.mixer.shared_experts.down_proj.weight"), DType::U8,
             {2688, 1856}),
          "shared down");
  require(absent(moe, "backbone.layers.1.mixer.experts.7.gate_proj.weight"),
          "the experts have no gate_proj");
  require(absent(moe, "backbone.layers.1.mixer.fc1_latent_proj.weight"), "no latent projections");
  require(absent(all, "mtp.layers.0.enorm.weight"), "no draft block");
}

DGPP_TEST(nemotron_binding_table_has_the_super_shape) {
  const dgpp::NemotronHConfig cfg = nemotron_test::super();
  const auto all = dgpp::nemotron_expected_tensors(cfg);
  // The real checkpoint's header: 165,860 tensors, 80,297,329,824 tensor bytes.
  require(all.size() == 165860, "table size " + std::to_string(all.size()));
  require(table_bytes(all) == 80297329824ull, "table bytes " + std::to_string(table_bytes(all)));
  require(count_role(all, NemotronTensorRole::Fp4Payload) == 40961 &&
              count_role(all, NemotronTensorRole::Fp8Payload) == 139,
          "quantized matrices");
  require(count_role(all, NemotronTensorRole::KvScale) == 16,
          "K/V-cache scales on the eight backbone attention layers");

  require(is(find(all, "backbone.embeddings.weight"), DType::BF16, {131072, 4096}), "embedding");
  // Mamba2: in_proj rows = inner 8192 + conv_dim 10240 + heads 128; per-tensor FP8 here.
  const auto mamba = dgpp::nemotron_expected_layer_tensors(cfg, 0);
  require(mamba.size() == 1 + 6 + 3 + 3, "fp8 mamba layer " + std::to_string(mamba.size()));
  const NemotronExpectedTensor in = find(mamba, "backbone.layers.0.mixer.in_proj.weight");
  require(is(in, DType::F8_E4M3, {18560, 4096}) && in.role == NemotronTensorRole::Fp8Payload,
          "fp8 in_proj payload");
  require(is(find(mamba, "backbone.layers.0.mixer.in_proj.weight_scale"), DType::F32, {}),
          "fp8 in_proj scale");
  require(is(find(mamba, "backbone.layers.0.mixer.in_proj.input_scale"), DType::F32, {}),
          "fp8 in_proj input scale");
  require(absent(mamba, "backbone.layers.0.mixer.in_proj.weight_scale_2"),
          "FP8 has no second scale");
  require(is(find(mamba, "backbone.layers.0.mixer.out_proj.weight"), DType::F8_E4M3, {4096, 8192}),
          "fp8 out_proj");
  require(is(find(mamba, "backbone.layers.0.mixer.conv1d.weight"), DType::BF16, {10240, 1, 4}),
          "conv weight");
  require(is(find(mamba, "backbone.layers.0.mixer.norm.weight"), DType::BF16, {8192}),
          "gated norm");
  // Layer 6: an FP8 in_proj beside a BF16 out_proj.
  const auto mixed = dgpp::nemotron_expected_layer_tensors(cfg, 6);
  require(is(find(mixed, "backbone.layers.6.mixer.in_proj.weight"), DType::F8_E4M3, {18560, 4096}),
          "layer 6 in_proj");
  require(is(find(mixed, "backbone.layers.6.mixer.out_proj.weight"), DType::BF16, {4096, 8192}),
          "layer 6 out_proj");
  // Attention: the K/V-cache scales beside k and v; o_proj FP8 on two layers.
  const auto attn = dgpp::nemotron_expected_layer_tensors(cfg, 69);
  require(attn.size() == 1 + 3 + 3 + 2, "attention layer 69 " + std::to_string(attn.size()));
  require(is(find(attn, "backbone.layers.69.mixer.q_proj.weight"), DType::BF16, {4096, 4096}),
          "q_proj");
  require(is(find(attn, "backbone.layers.69.mixer.k_proj.weight"), DType::BF16, {256, 4096}),
          "k_proj");
  require(is(find(attn, "backbone.layers.69.mixer.o_proj.weight"), DType::F8_E4M3, {4096, 4096}),
          "fp8 o_proj");
  const NemotronExpectedTensor ks = find(attn, "backbone.layers.69.mixer.k_proj.k_scale");
  require(is(ks, DType::F32, {}) && ks.role == NemotronTensorRole::KvScale, "k_scale");
  require(is(find(attn, "backbone.layers.69.mixer.v_proj.v_scale"), DType::F32, {}), "v_scale");
  require(is(find(all, "backbone.layers.7.mixer.o_proj.weight"), DType::BF16, {4096, 4096}),
          "BF16 o_proj");
  // MoE: a BF16 router, the experts in the 1024-wide latent, the latent projections.
  const auto moe = dgpp::nemotron_expected_layer_tensors(cfg, 1);
  require(is(find(moe, "backbone.layers.1.mixer.gate.weight"), DType::BF16, {512, 4096}),
          "router gate");
  require(is(find(moe, "backbone.layers.1.mixer.gate.e_score_correction_bias"), DType::F32, {512}),
          "router bias");
  require(
      is(find(moe, "backbone.layers.1.mixer.experts.511.up_proj.weight"), DType::U8, {2688, 512}),
      "expert up payload");
  require(is(find(moe, "backbone.layers.1.mixer.experts.511.up_proj.weight_scale"), DType::F8_E4M3,
             {2688, 64}),
          "expert up scales");
  require(is(find(moe, "backbone.layers.1.mixer.experts.511.down_proj.weight"), DType::U8,
             {1024, 1344}),
          "expert down payload");
  require(is(find(moe, "backbone.layers.1.mixer.experts.511.down_proj.weight_scale"),
             DType::F8_E4M3, {1024, 168}),
          "expert down scales");
  const NemotronExpectedTensor fc1 = find(moe, "backbone.layers.1.mixer.fc1_latent_proj.weight");
  require(is(fc1, DType::F8_E4M3, {1024, 4096}) && fc1.cls == NemotronWeightClass::LatentProj,
          "fp8 fc1");
  require(
      is(find(moe, "backbone.layers.1.mixer.fc2_latent_proj.weight"), DType::BF16, {4096, 1024}),
      "BF16 fc2");
  require(is(find(moe, "backbone.layers.1.mixer.shared_experts.up_proj.weight"), DType::F8_E4M3,
             {5376, 4096}),
          "fp8 shared up");
  // The one NVFP4 matrix outside the experts, and the shared expert's BF16 forms.
  require(is(find(moe, "backbone.layers.1.mixer.shared_experts.down_proj.weight"), DType::U8,
             {4096, 2688}),
          "nvfp4 shared down");
  require(is(find(moe, "backbone.layers.1.mixer.shared_experts.down_proj.weight_scale"),
             DType::F8_E4M3, {4096, 336}),
          "nvfp4 shared down scales");
  require(is(find(all, "backbone.layers.8.mixer.shared_experts.down_proj.weight"), DType::BF16,
             {4096, 5376}),
          "BF16 shared down");
  require(is(find(all, "backbone.layers.41.mixer.shared_experts.up_proj.weight"), DType::BF16,
             {5376, 4096}),
          "BF16 shared up");
  require(is(find(all, "backbone.layers.3.mixer.shared_experts.down_proj.weight_scale"), DType::F32,
             {}),
          "fp8 shared down scale");

  // The draft block: the fusion on its attention layer, the closing norm on
  // its MoE layer, everything BF16, no cache scales.
  const auto d0 = dgpp::nemotron_expected_layer_tensors(cfg, 88);
  require(d0.size() == 3 + 1 + 4, "draft attention layer " + std::to_string(d0.size()));
  require(d0[0].name == "mtp.layers.0.enorm.weight" && d0[0].cls == NemotronWeightClass::MtpHead &&
              d0[0].layer == 88,
          "enorm");
  require(is(find(d0, "mtp.layers.0.hnorm.weight"), DType::BF16, {4096}), "hnorm");
  require(is(find(d0, "mtp.layers.0.eh_proj.weight"), DType::BF16, {4096, 8192}), "eh_proj");
  require(is(find(d0, "mtp.layers.0.norm.weight"), DType::BF16, {4096}), "draft pre-norm");
  require(is(find(d0, "mtp.layers.0.mixer.q_proj.weight"), DType::BF16, {4096, 4096}),
          "draft q_proj");
  require(is(find(d0, "mtp.layers.0.mixer.o_proj.weight"), DType::BF16, {4096, 4096}),
          "draft o_proj");
  require(absent(d0, "mtp.layers.0.mixer.k_proj.k_scale") &&
              absent(d0, "mtp.layers.0.final_layernorm.weight"),
          "draft layer 0");
  const auto d1 = dgpp::nemotron_expected_layer_tensors(cfg, 89);
  require(d1.size() == 1 + 2 + 512 * 2 + 2 + 2 + 1, "draft moe layer " + std::to_string(d1.size()));
  require(is(find(d1, "mtp.layers.1.mixer.experts.0.up_proj.weight"), DType::BF16, {2688, 1024}),
          "draft expert up");
  require(is(find(d1, "mtp.layers.1.mixer.experts.0.down_proj.weight"), DType::BF16, {1024, 2688}),
          "draft expert down");
  require(is(find(d1, "mtp.layers.1.mixer.gate.weight"), DType::BF16, {512, 4096}), "draft router");
  require(is(find(d1, "mtp.layers.1.mixer.fc1_latent_proj.weight"), DType::BF16, {1024, 4096}),
          "draft fc1");
  require(d1.back().name == "mtp.layers.1.final_layernorm.weight" &&
              d1.back().cls == NemotronWeightClass::MtpHead,
          "final_layernorm");
  require(absent(d1, "mtp.layers.1.enorm.weight"), "the fusion is on the first draft layer only");
}

DGPP_TEST(nemotron_binding_validator_reports_by_class) {
  const dgpp::NemotronHConfig cfg = nemotron_test::nano();
  const auto all = dgpp::nemotron_expected_tensors(cfg);
  auto present = present_of(all);
  {
    const dgpp::NemotronBindReport rep = dgpp::nemotron_validate_binding(cfg, present);
    require(rep.ok() && rep.expected == all.size() && rep.matched == all.size(),
            "a complete checkpoint binds");
    require(rep.fp4_matrices == 5968 && rep.fp8_matrices == 0 && rep.bytes == 19339781632ull,
            "matrix counts and bytes");
  }
  present.erase("backbone.layers.7.mixer.out_proj.weight_scale_2");
  present["backbone.layers.5.mixer.o_proj.weight"].dtype = DType::F8_E4M3;
  present["backbone.layers.0.mixer.conv1d.weight"].shape = {6144, 4};
  // A router stored the way the other release stores it is a dtype error.
  present["backbone.layers.1.mixer.gate.weight"].dtype = DType::BF16;
  present.emplace("backbone.layers.1.mixer.experts.0.gate_proj.weight",
                  dgpp::NemotronTensorDesc{DType::BF16, {1856, 2688}});
  // Nothing is skipped: a draft block this config does not declare is unexpected.
  present.emplace("mtp.layers.0.enorm.weight", dgpp::NemotronTensorDesc{DType::BF16, {2688}});
  const dgpp::NemotronBindReport rep = dgpp::nemotron_validate_binding(cfg, present);
  require(!rep.ok(), "a broken checkpoint is refused");
  require(
      rep.missing == 1 && rep.dtype_mismatch == 2 && rep.shape_mismatch == 1 && rep.unexpected == 2,
      "the report counts by class");
  require(rep.errors.size() == 6, "six errors");
  // The error cap.
  require(dgpp::nemotron_validate_binding(cfg, {}, 5).errors.size() == 5, "errors are capped");
  require(dgpp::nemotron_validate_binding(cfg, {}).missing == all.size(),
          "an empty checkpoint misses everything");
}

DGPP_TEST(nemotron_binding_follows_the_recipe) {
  // The same config under a recipe that excludes one module more binds that
  // module in BF16: three tensors fewer (the two scales and the input scale).
  const dgpp::NemotronHConfig base = nemotron_test::nano();
  const dgpp::NemotronHConfig more = nemotron_test::parse(
      nemotron_test::kNanoConfig,
      nemotron_test::patched(nemotron_test::kNanoHfQuant, "\"lm_head\",",
                             "\"lm_head\", \"backbone.layers.0.mixer.out_proj\","));
  const auto t = dgpp::nemotron_expected_tensors(more);
  require(t.size() == dgpp::nemotron_expected_tensors(base).size() - 3,
          "table size under the wider exclude list");
  require(is(find(t, "backbone.layers.0.mixer.out_proj.weight"), DType::BF16, {2688, 4096}),
          "the excluded module is BF16");
  // A checkpoint written for the other recipe does not bind.
  const dgpp::NemotronBindReport rep =
      dgpp::nemotron_validate_binding(more, present_of(dgpp::nemotron_expected_tensors(base)));
  require(!rep.ok() && rep.dtype_mismatch == 1 && rep.unexpected == 3,
          "recipe / checkpoint mismatch is reported");
  // NVFP4 needs K to be a multiple of 16.
  dgpp::NemotronHConfig odd = base;
  odd.hidden_size = 2696;
  bool refused = false;
  try {
    (void)dgpp::nemotron_expected_layer_tensors(odd, 0);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "an NVFP4 matrix whose K is not a multiple of 16");
}

DGPP_TEST(nemotron_binding_matches_the_landed_checkpoints) {
  if (const auto snap = landed_snapshot("models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4");
      !snap.empty()) {
    const dgpp::NemotronHConfig cfg =
        dgpp::NemotronHConfig::from_json_file((snap / "config.json").string());
    size_t tensors = 0;
    const dgpp::NemotronBindReport rep = bind_snapshot(snap, cfg, &tensors);
    require_bound(rep);
    require(tensors == 24147 && rep.expected == 24147 && rep.matched == 24147,
            "nano: table and headers agree on 24,147");
    require(rep.fp4_matrices == 5968 && rep.bytes == 19339781632ull, "nano: matrices and bytes");
  }
  if (const auto snap = landed_snapshot("models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4");
      !snap.empty()) {
    const dgpp::NemotronHConfig cfg =
        dgpp::NemotronHConfig::from_json_file((snap / "config.json").string());
    size_t tensors = 0;
    const dgpp::NemotronBindReport rep = bind_snapshot(snap, cfg, &tensors);
    require_bound(rep);
    require(tensors == 165860 && rep.expected == 165860 && rep.matched == 165860,
            "super: table and headers agree on 165,860");
    require(rep.fp4_matrices == 40961 && rep.fp8_matrices == 139 && rep.bytes == 80297329824ull,
            "super: matrices and bytes");
  }
}
