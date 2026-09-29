// The Qwen3.8-Flash-Next expected-tensor table: its size
// and the n-gram shard geometry on the release's config, the TP geometry
// acceptance at the deployment worlds, and — when the checkpoint is in the
// hub cache — the full binding against every shard's header (152 089
// tensors: every expected one present with its dtype and shape, nothing
// unexpected but the vision tower).
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "loaders/safetensors.hpp"
#include "models/qwen/binding.hpp"
#include "../cuda/qwen_fixture.hpp"
#include "models/qwen/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (fs::exists(snap.path() / "config.json") &&
        fs::exists(snap.path() / "model.safetensors.index.json"))
      return snap.path();
  return {};
}

std::filesystem::path radixark_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root = fs::path(home) / ".cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (fs::exists(snap.path() / "config.json") &&
        fs::exists(snap.path() / "model.safetensors.index.json"))
      return snap.path();
  return {};
}

}  // namespace

DGPP_TEST(qwen_binding_table_has_the_release_shape) {
  const auto snap = landed_snapshot();
  if (snap.empty()) return;
  const dgpp::QwenTextConfig cfg = dgpp::QwenTextConfig::from_json_file((snap / "config.json").string());
  const auto table = dgpp::qwen_expected_text_tensors(cfg);
  // 152 089 tensors in the file less the 333 of the vision tower.
  require(table.size() == 151756, "table size " + std::to_string(table.size()));
  require(dgpp::qwen_ngram_shard_capacity(cfg) == 2500012, "shard capacity");
  int64_t rows = 0;
  for (int s = 0; s < cfg.split_ngram_parts; ++s) rows += dgpp::qwen_ngram_shard_rows(cfg, s);
  require(rows == 320001536, "shard rows cover the padded table");
  const auto layer1 = dgpp::qwen_expected_layer_tensors(cfg, 1);
  const auto layer0 = dgpp::qwen_expected_layer_tensors(cfg, 0);
  require(layer1.size() == layer0.size() + 10 + 128, "the PLE layer adds its 10 tensors and 128 shards");
  const auto draft = dgpp::qwen_expected_layer_tensors(cfg, cfg.mtp_layer());
  require(!draft.empty() && draft[0].name.rfind("mtp.layers.0.", 0) == 0, "the draft layer's prefix");
}

DGPP_TEST(qwen_tp_geometry_accepts_the_deployment_worlds) {
  const auto snap = landed_snapshot();
  if (snap.empty()) return;
  const dgpp::QwenTextConfig cfg = dgpp::QwenTextConfig::from_json_file((snap / "config.json").string());
  for (const int w : {1, 2, 4, 8})
    for (int r = 0; r < w; ++r) dgpp::qwen_tp_validate_geometry(cfg, r, w);
  bool refused = false;
  try {
    dgpp::qwen_tp_validate_geometry(cfg, 0, 3);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "world 3 refused");
}

DGPP_TEST(qwen_binding_validates_the_landed_checkpoint_when_present) {
  namespace fs = std::filesystem;
  const auto snap = landed_snapshot();
  if (snap.empty()) return;
  const dgpp::QwenTextConfig cfg = dgpp::QwenTextConfig::from_json_file((snap / "config.json").string());
  std::unordered_map<std::string, dgpp::QwenTensorDesc> present;
  std::vector<fs::path> shards;
  for (const auto& entry : fs::directory_iterator(snap))
    if (entry.path().extension() == ".safetensors") shards.push_back(entry.path());
  require(shards.size() == 131, "131 shards");
  for (const auto& path : shards) {
    auto f = dgpp::SafetensorsFile::open(path.string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      present.emplace(t.name, dgpp::QwenTensorDesc{t.dtype, t.shape});
    });
  }
  require(present.size() == 152089, "152 089 tensors in the headers");
  const dgpp::QwenBindReport rep = dgpp::qwen_validate_text_binding(cfg, present);
  std::string first = rep.errors.empty() ? "" : rep.errors[0];
  require(rep.ok(), "binding: missing " + std::to_string(rep.missing) + " dtype " +
                        std::to_string(rep.dtype_mismatch) + " shape " +
                        std::to_string(rep.shape_mismatch) + " unexpected " +
                        std::to_string(rep.unexpected) + " first: " + first);
  require(rep.vision == 333 && rep.quantized_matrices == 73728 + 1536 && rep.ngram_shards == 128,
          "the census");
}

DGPP_TEST(qwen_binding_table_has_the_autoround_hybrid_shape) {
  // The tiny hybrid: every routed expert matrix (backbone and draft) a GPTQ
  // triple, the lm_head an int8 triple, the GDN / QSA / shared projections
  // block FP8 with F32 scales; everything else BF16 as before.
  const dgpp::QwenTextConfig cfg = qwenfx::tiny_gptq_config();
  const auto table = dgpp::qwen_expected_text_tensors(cfg);
  const dgpp::QwenTextConfig fp8 = qwenfx::tiny_config();
  const auto plain = dgpp::qwen_expected_text_tensors(fp8);
  const int layers = cfg.num_hidden_layers + 1;
  const int E = cfg.num_experts;
  // Experts: 3 GPTQ tensors per matrix (vs the FP8 pair); the head: 3 (vs
  // 1); the shipped-fp8 dense classes: a scale partner each (vs BF16 alone).
  // The draft layer ships BF16 projections and shared expert: no partners there.
  const int gdn = cfg.num_gdn_layers(), qsa = cfg.num_qsa_layers();
  const size_t dense_partners = static_cast<size_t>(gdn * 3 + qsa * 4 + (layers - 1) * 3);
  require(table.size() == plain.size() + static_cast<size_t>(layers) * E * 3 + 2 + dense_partners,
          "hybrid table size " + std::to_string(table.size()) + " vs " + std::to_string(plain.size()));
  size_t codes = 0, scales = 0, zeros = 0, f32_scales = 0, head = 0;
  for (const auto& e : table) {
    if (e.role == dgpp::QwenTensorRole::GptqCodes) {
      ++codes;
      require(e.dtype == dgpp::DType::I32, "codes are I32");
      if (e.cls == dgpp::QwenWeightClass::RoutedExpert) {
        const bool down = e.name.find("down_proj") != std::string::npos;
        const int64_t N = down ? cfg.hidden_size : cfg.moe_intermediate_size;
        const int64_t K = down ? cfg.moe_intermediate_size : cfg.hidden_size;
        require(e.shape == std::vector<int64_t>{K * 4 / 32, N}, "expert qweight shape " + e.name);
      }
    } else if (e.role == dgpp::QwenTensorRole::GptqScales) {
      ++scales;
      require(e.dtype == dgpp::DType::F16, "scales are F16");
    } else if (e.role == dgpp::QwenTensorRole::GptqZeros) {
      ++zeros;
    } else if (e.role == dgpp::QwenTensorRole::Fp8Scale && e.dtype == dgpp::DType::F32) {
      ++f32_scales;
    }
    if (e.cls == dgpp::QwenWeightClass::LmHead) {
      ++head;
      if (e.role == dgpp::QwenTensorRole::GptqCodes)
        require(e.shape == std::vector<int64_t>{cfg.hidden_size * 8 / 32, cfg.vocab_size}, "head qweight shape");
      if (e.role == dgpp::QwenTensorRole::GptqZeros)
        require(e.shape == std::vector<int64_t>{cfg.hidden_size / 128, cfg.vocab_size * 8 / 32}, "head qzeros shape");
    }
  }
  require(codes == static_cast<size_t>(layers) * E * 3 + 1 && scales == codes && zeros == codes, "one triple per matrix");
  require(head == 3, "the head is one GPTQ triple");
  // Per layer: GDN qkv/z/out or QSA q/k/v/o, plus the shared expert's three.
  require(f32_scales == dense_partners, "the shipped fp8 classes carry F32 scales");
  require(dgpp::qwen_validate_text_binding(cfg, {}).missing == table.size(), "the validator walks the table");
}

DGPP_TEST(qwen_binding_table_has_the_radixark_fused_shape) {
  const auto snap = radixark_snapshot();
  if (snap.empty()) return;
  const dgpp::QwenTextConfig cfg = dgpp::QwenTextConfig::from_json_file((snap / "config.json").string());
  // The RadixArk NVFP4 checkpoint stores the MTP layer's experts as fused BF16
  // tensors, not per-expert FP8 — the loader encodes the slices to FP8 at load.
  dgpp::QwenTextConfig fused = cfg;
  fused.mtp_experts_bf16_fused = true;
  const auto table = dgpp::qwen_expected_text_tensors(fused);
  const auto unfused = dgpp::qwen_expected_text_tensors(cfg);
  // The fused table replaces 512*3*2=3072 per-expert FP8 MTP tensors with
  // 2 fused tensors, so it is smaller by 3070.
  require(table.size() == unfused.size() - 3070,
          "fused table smaller by 3070 (" + std::to_string(table.size()) + " vs " +
              std::to_string(unfused.size()) + ")");
  // The fused tensor names must be present.
  bool found_gate_up = false, found_down = false;
  for (const auto& t : table) {
    if (t.name == "mtp.layers.0.mlp.experts.gate_up_proj") found_gate_up = true;
    if (t.name == "mtp.layers.0.mlp.experts.down_proj") found_down = true;
  }
  require(found_gate_up && found_down, "fused MTP tensor names present");
  // The fused gate_up_proj is [E, 2*I, H], down_proj is [E, H, I].
  const int64_t E = cfg.num_experts, I = cfg.moe_intermediate_size, H = cfg.hidden_size;
  for (const auto& t : table) {
    if (t.name == "mtp.layers.0.mlp.experts.gate_up_proj") {
      require(t.dtype == dgpp::DType::BF16 && t.shape == std::vector<int64_t>{E, 2 * I, H},
              "gate_up_proj shape {E, 2*I, H}");
    }
    if (t.name == "mtp.layers.0.mlp.experts.down_proj") {
      require(t.dtype == dgpp::DType::BF16 && t.shape == std::vector<int64_t>{E, H, I},
              "down_proj shape {E, H, I}");
    }
  }
}
