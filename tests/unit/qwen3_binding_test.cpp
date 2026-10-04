// The plain Qwen3 family's expected-tensor table on the four real configs,
// pinned to the tensor counts and byte totals of the real checkpoints'
// safetensors headers (read from Hugging Face 2026-10-04, headers only):
//   ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4          75,090 tensors / 19,164,721,120 bytes
//   nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4  145,703 tensors / 139,202,447,840 bytes
//   Qwen/Qwen3-Reranker-0.6B                        310 tensors /  1,191,553,024 bytes
//   Qwen/Qwen3-Embedding-0.6B                       310 tensors /  1,191,553,024 bytes (bare names)
// plus each container's per-module format, the validator's report, the
// tensor-parallel geometry rule and the per-rank resident arithmetic the
// 235B's two-node plan quotes.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen3/binding.hpp"
#include "models/qwen3/config.hpp"
#include "qwen3_config_json.hpp"

namespace {

using dgpp::DType;
using dgpp::Qwen3ExpectedTensor;
using dgpp::Qwen3TensorRole;
using dgpp::Qwen3TextConfig;
using dgpp::Qwen3WeightClass;
using Shape = std::vector<int64_t>;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

Qwen3TextConfig parse(const std::string& json) {
  const auto parsed = dgpp::minijson::parse(json);
  return Qwen3TextConfig::parse(parsed.root);
}

const Qwen3ExpectedTensor& find(const std::vector<Qwen3ExpectedTensor>& v, const std::string& name) {
  for (const auto& e : v)
    if (e.name == name) return e;
  throw std::runtime_error("expected tensor missing from the table: " + name);
}
bool absent(const std::vector<Qwen3ExpectedTensor>& v, const std::string& name) {
  for (const auto& e : v)
    if (e.name == name) return false;
  return true;
}
bool is(const Qwen3ExpectedTensor& e, DType dtype, const Shape& shape) { return e.dtype == dtype && e.shape == shape; }

size_t table_bytes(const std::vector<Qwen3ExpectedTensor>& v) {
  size_t n = 0;
  for (const auto& e : v) n += e.nbytes();
  return n;
}
size_t count_role(const std::vector<Qwen3ExpectedTensor>& v, Qwen3TensorRole role) {
  size_t n = 0;
  for (const auto& e : v) n += e.role == role;
  return n;
}
size_t count_class(const std::vector<Qwen3ExpectedTensor>& v, Qwen3WeightClass cls) {
  size_t n = 0;
  for (const auto& e : v) n += e.cls == cls;
  return n;
}

dgpp::Qwen3PresentMap present_of(const std::vector<Qwen3ExpectedTensor>& v) {
  dgpp::Qwen3PresentMap present;
  for (const auto& e : v) present.emplace(e.name, dgpp::Qwen3TensorDesc{e.dtype, e.shape});
  return present;
}

}  // namespace

DGPP_TEST(qwen3_vl30b_table_is_the_checkpoints_75090_tensors) {
  const Qwen3TextConfig c = parse(qwen3_config_json::kVl30b);
  const auto all = dgpp::qwen3_expected_tensors(c);
  const auto vis = dgpp::qwen3_expected_vision_tensors(c);
  require(all.size() == 75090, "75,090 tensors, got " + std::to_string(all.size()));
  require(vis.size() == 351, "351 vision tensors, got " + std::to_string(vis.size()));
  require(dgpp::qwen3_expected_text_tensors(c).size() == 74739, "74,739 text tensors");
  require(table_bytes(all) == 19164721120ull, "19,164,721,120 bytes, got " + std::to_string(table_bytes(all)));
  require(table_bytes(vis) == 1077262816ull, "the tower's 1,077,262,816 bytes");
  // 48 x (q, k, v, o + 128 experts x 3) NVFP4 matrices, each a set of four.
  require(count_role(all, Qwen3TensorRole::Fp4Payload) == 18624, "18,624 NVFP4 matrices");
  require(count_role(all, Qwen3TensorRole::Fp4Scale) == 18624 && count_role(all, Qwen3TensorRole::Fp4Global) == 18624 &&
              count_role(all, Qwen3TensorRole::InputScale) == 18624,
          "every NVFP4 matrix is a set of four");
  require(count_role(all, Qwen3TensorRole::KvScale) == 0, "no K/V-cache scales in this container");
  require(count_class(all, Qwen3WeightClass::Vision) == 351, "the tower is the Vision class");

  const std::string L = "model.language_model.layers.47.";
  // compressed-tensors: weight_packed / weight_scale / weight_global_scale [1] / input_global_scale [1].
  require(is(find(all, L + "self_attn.q_proj.weight_packed"), DType::U8, {4096, 1024}), "q_proj codes");
  require(is(find(all, L + "self_attn.q_proj.weight_scale"), DType::F8_E4M3, {4096, 128}), "q_proj block scales");
  require(is(find(all, L + "self_attn.q_proj.weight_global_scale"), DType::F32, {1}), "q_proj global [1]");
  require(is(find(all, L + "self_attn.q_proj.input_global_scale"), DType::F32, {1}), "q_proj input global [1]");
  require(is(find(all, L + "self_attn.k_proj.weight_packed"), DType::U8, {512, 1024}), "k_proj codes");
  require(is(find(all, L + "self_attn.v_proj.weight_packed"), DType::U8, {512, 1024}), "v_proj codes");
  require(is(find(all, L + "self_attn.o_proj.weight_packed"), DType::U8, {2048, 2048}), "o_proj codes");
  require(is(find(all, L + "self_attn.o_proj.weight_scale"), DType::F8_E4M3, {2048, 256}), "o_proj block scales");
  require(is(find(all, L + "self_attn.q_norm.weight"), DType::BF16, {128}), "q_norm over head_dim");
  require(is(find(all, L + "self_attn.k_norm.weight"), DType::BF16, {128}), "k_norm over head_dim");
  require(is(find(all, L + "mlp.gate.weight"), DType::BF16, {128, 2048}), "BF16 router");
  require(is(find(all, L + "mlp.experts.127.gate_proj.weight_packed"), DType::U8, {768, 1024}), "expert gate codes");
  require(is(find(all, L + "mlp.experts.127.up_proj.weight_scale"), DType::F8_E4M3, {768, 128}), "expert up scales");
  require(is(find(all, L + "mlp.experts.127.down_proj.weight_packed"), DType::U8, {2048, 384}), "expert down codes");
  require(is(find(all, L + "mlp.experts.127.down_proj.weight_scale"), DType::F8_E4M3, {2048, 48}), "expert down scales");
  require(find(all, L + "mlp.experts.127.down_proj.weight_packed").expert == 127, "expert id");
  require(is(find(all, L + "input_layernorm.weight"), DType::BF16, {2048}), "input norm");
  require(is(find(all, "model.language_model.embed_tokens.weight"), DType::BF16, {151936, 2048}), "embedding");
  require(is(find(all, "model.language_model.norm.weight"), DType::BF16, {2048}), "final norm");
  require(is(find(all, "lm_head.weight"), DType::BF16, {151936, 2048}), "untied head");
  // No modelopt names, no shared expert, no router bias, no draft layer.
  require(absent(all, L + "self_attn.q_proj.weight") && absent(all, L + "self_attn.o_proj.weight_scale_2"), "no modelopt names");
  require(absent(all, L + "mlp.shared_expert_gate.weight") && absent(all, L + "mlp.shared_expert.gate_proj.weight_packed"),
          "no shared expert");
  require(absent(all, L + "mlp.gate.e_score_correction_bias") && absent(all, "mtp.fc.weight"), "no router bias, no draft");
  // The tower.
  require(is(find(all, "model.visual.patch_embed.proj.weight"), DType::BF16, {1152, 3, 2, 16, 16}), "Conv3d patch embed");
  require(is(find(all, "model.visual.pos_embed.weight"), DType::BF16, {2304, 1152}), "position table");
  require(is(find(all, "model.visual.blocks.26.attn.qkv.weight"), DType::BF16, {3456, 1152}), "tower qkv");
  require(is(find(all, "model.visual.blocks.26.attn.qkv.bias"), DType::BF16, {3456}), "tower qkv bias");
  require(is(find(all, "model.visual.blocks.26.mlp.linear_fc1.weight"), DType::BF16, {4304, 1152}), "tower fc1");
  require(is(find(all, "model.visual.blocks.26.mlp.linear_fc2.weight"), DType::BF16, {1152, 4304}), "tower fc2");
  require(is(find(all, "model.visual.merger.norm.weight"), DType::BF16, {1152}), "merger norm over the unmerged width");
  require(is(find(all, "model.visual.merger.linear_fc1.weight"), DType::BF16, {4608, 4608}), "merger fc1");
  require(is(find(all, "model.visual.merger.linear_fc2.weight"), DType::BF16, {2048, 4608}), "merger fc2");
  require(is(find(all, "model.visual.deepstack_merger_list.2.norm.weight"), DType::BF16, {4608}), "deepstack norm over the merged width");
  require(absent(all, "model.visual.deepstack_merger_list.3.norm.weight") && absent(all, "model.visual.blocks.27.norm1.weight"), "tower bounds");
  require(find(all, "model.visual.pos_embed.weight").unused(), "the tower is bound and never read");
}

DGPP_TEST(qwen3_235b_table_is_the_checkpoints_145703_tensors) {
  const Qwen3TextConfig c = parse(qwen3_config_json::kMoe235b);
  const auto all = dgpp::qwen3_expected_tensors(c);
  require(all.size() == 145703, "145,703 tensors, got " + std::to_string(all.size()));
  require(dgpp::qwen3_expected_vision_tensors(c).empty(), "no tower");
  require(table_bytes(all) == 139202447840ull, "139,202,447,840 bytes, got " + std::to_string(table_bytes(all)));
  // 94 x (o_proj + 128 experts x 3).
  require(count_role(all, Qwen3TensorRole::Fp4Payload) == 36190, "36,190 NVFP4 matrices");
  require(count_role(all, Qwen3TensorRole::KvScale) == 188, "94 x (k_scale, v_scale)");

  const std::string L = "model.layers.93.";
  // modelopt: q/k/v BF16 with the cache scales, o_proj NVFP4 with scalar globals.
  require(is(find(all, L + "self_attn.q_proj.weight"), DType::BF16, {8192, 4096}), "BF16 q_proj");
  require(is(find(all, L + "self_attn.k_proj.weight"), DType::BF16, {512, 4096}), "BF16 k_proj");
  require(is(find(all, L + "self_attn.v_proj.weight"), DType::BF16, {512, 4096}), "BF16 v_proj");
  require(is(find(all, L + "self_attn.k_proj.k_scale"), DType::F32, {}), "k_scale scalar");
  require(is(find(all, L + "self_attn.v_proj.v_scale"), DType::F32, {}), "v_scale scalar");
  require(find(all, L + "self_attn.k_proj.k_scale").unused(), "the cache scales are bound and unread");
  require(is(find(all, L + "self_attn.o_proj.weight"), DType::U8, {4096, 4096}), "o_proj codes");
  require(is(find(all, L + "self_attn.o_proj.weight_scale"), DType::F8_E4M3, {4096, 512}), "o_proj block scales");
  require(is(find(all, L + "self_attn.o_proj.weight_scale_2"), DType::F32, {}), "o_proj scalar global");
  require(is(find(all, L + "self_attn.o_proj.input_scale"), DType::F32, {}), "o_proj input scale");
  require(is(find(all, L + "mlp.gate.weight"), DType::BF16, {128, 4096}), "BF16 router");
  require(is(find(all, L + "mlp.experts.0.gate_proj.weight"), DType::U8, {1536, 2048}), "expert gate codes");
  require(is(find(all, L + "mlp.experts.0.gate_proj.weight_scale"), DType::F8_E4M3, {1536, 256}), "expert gate scales");
  require(is(find(all, L + "mlp.experts.0.down_proj.weight"), DType::U8, {4096, 768}), "expert down codes");
  require(is(find(all, L + "mlp.experts.0.down_proj.weight_scale"), DType::F8_E4M3, {4096, 96}), "expert down scales");
  require(is(find(all, L + "mlp.experts.0.down_proj.weight_scale_2"), DType::F32, {}), "expert scalar global");
  require(is(find(all, "model.embed_tokens.weight"), DType::BF16, {151936, 4096}), "embedding");
  require(is(find(all, "lm_head.weight"), DType::BF16, {151936, 4096}), "untied head");
  require(absent(all, L + "self_attn.q_proj.weight_packed") && absent(all, L + "self_attn.q_proj.weight_scale"), "q is not quantized");
  require(absent(all, L + "mlp.shared_expert.gate_proj.weight") && absent(all, "model.layers.94.mlp.gate.weight"), "no shared expert, 94 layers");
}

DGPP_TEST(qwen3_dense_tables_are_the_retrieval_checkpoints_310_tensors) {
  Qwen3TextConfig r = parse(qwen3_config_json::kReranker);
  const auto reranker = dgpp::qwen3_expected_tensors(r);
  require(reranker.size() == 310, "310 tensors, got " + std::to_string(reranker.size()));
  require(table_bytes(reranker) == 1191553024ull, "1,191,553,024 bytes");
  require(count_role(reranker, Qwen3TensorRole::Plain) == 310, "all BF16, nothing quantized");
  // head_dim 128 with 16 heads over hidden 1024: the projections are not square.
  require(is(find(reranker, "model.layers.27.self_attn.q_proj.weight"), DType::BF16, {2048, 1024}), "q_proj [heads * 128, H]");
  require(is(find(reranker, "model.layers.27.self_attn.k_proj.weight"), DType::BF16, {1024, 1024}), "k_proj [kv * 128, H]");
  require(is(find(reranker, "model.layers.27.self_attn.o_proj.weight"), DType::BF16, {1024, 2048}), "o_proj [H, heads * 128]");
  require(is(find(reranker, "model.layers.27.mlp.gate_proj.weight"), DType::BF16, {3072, 1024}), "dense gate");
  require(is(find(reranker, "model.layers.27.mlp.down_proj.weight"), DType::BF16, {1024, 3072}), "dense down");
  require(is(find(reranker, "model.embed_tokens.weight"), DType::BF16, {151669, 1024}), "embedding");
  require(absent(reranker, "lm_head.weight"), "tied: the head is the embedding, no lm_head tensor");
  require(absent(reranker, "model.layers.27.mlp.gate.weight"), "no router");

  // Qwen3-Embedding: the same tensors under the base model's names — the header says so.
  Qwen3TextConfig e = parse(qwen3_config_json::kEmbedding);
  dgpp::Qwen3PresentMap header;
  header.emplace("embed_tokens.weight", dgpp::Qwen3TensorDesc{DType::BF16, {151669, 1024}});
  dgpp::qwen3_apply_header_naming(e, header);
  require(e.bare_names, "a header with embed_tokens.weight and no model.embed_tokens.weight is the bare layout");
  const auto embedding = dgpp::qwen3_expected_tensors(e);
  require(embedding.size() == 310 && table_bytes(embedding) == 1191553024ull, "310 tensors / 1,191,553,024 bytes");
  require(is(find(embedding, "layers.0.self_attn.q_proj.weight"), DType::BF16, {2048, 1024}), "bare layer names");
  require(is(find(embedding, "norm.weight"), DType::BF16, {1024}) && is(find(embedding, "embed_tokens.weight"), DType::BF16, {151669, 1024}), "bare globals");
  require(absent(embedding, "model.embed_tokens.weight"), "no model. prefix");
  // The reranker's header keeps the prefix; the MoE dialects never take the bare names.
  dgpp::qwen3_apply_header_naming(r, present_of(reranker));
  require(!r.bare_names, "prefixed header");
  Qwen3TextConfig m = parse(qwen3_config_json::kMoe235b);
  dgpp::qwen3_apply_header_naming(m, header);
  require(!m.bare_names, "the MoE dialects keep their prefix");
}

DGPP_TEST(qwen3_binding_validator_reports_by_name) {
  // Two layers of the 235B's shape keep the maps small.
  Qwen3TextConfig c = parse(qwen3_config_json::kMoe235b);
  Qwen3TextConfig two = c;
  two.num_hidden_layers = 2;
  const auto table = dgpp::qwen3_expected_tensors(two);
  auto present = present_of(table);
  {
    const dgpp::Qwen3BindReport rep = dgpp::qwen3_validate_binding(two, present);
    require(rep.ok() && rep.expected == table.size() && rep.matched == table.size(), "a matching header binds");
    require(rep.bytes == table_bytes(table) && rep.fp4_matrices == 2 * 385, "bytes and NVFP4 matrices");
  }
  {
    auto p = present;
    p.erase("model.layers.1.mlp.experts.5.up_proj.weight_scale");
    const dgpp::Qwen3BindReport rep = dgpp::qwen3_validate_binding(two, p);
    require(!rep.ok() && rep.missing == 1, "one missing tensor");
    require(rep.errors.size() == 1 && rep.errors[0].find("model.layers.1.mlp.experts.5.up_proj.weight_scale") != std::string::npos, "named");
  }
  {
    auto p = present;
    p["model.layers.0.self_attn.q_proj.weight"].dtype = DType::F8_E4M3;
    p["model.layers.0.self_attn.o_proj.weight"].shape = {4096, 8192};
    const dgpp::Qwen3BindReport rep = dgpp::qwen3_validate_binding(two, p);
    require(!rep.ok() && rep.dtype_mismatch == 1 && rep.shape_mismatch == 1, "dtype and shape mismatches counted apart");
  }
  {
    auto p = present;
    // The other container's name for the same matrix, a shared expert, a draft tensor: all unexpected.
    p.emplace("model.layers.0.self_attn.o_proj.weight_packed", dgpp::Qwen3TensorDesc{DType::U8, {4096, 4096}});
    p.emplace("model.layers.0.mlp.shared_expert_gate.weight", dgpp::Qwen3TensorDesc{DType::BF16, {1, 4096}});
    p.emplace("mtp.fc.weight", dgpp::Qwen3TensorDesc{DType::BF16, {4096, 8192}});
    const dgpp::Qwen3BindReport rep = dgpp::qwen3_validate_binding(two, p);
    require(!rep.ok() && rep.unexpected == 3, "three unexpected tensors");
  }
  {
    // A truncated config (the check apps' --layers): the layers past it are out of scope.
    Qwen3TextConfig one = c;
    one.num_hidden_layers = 1;
    const dgpp::Qwen3BindReport rep = dgpp::qwen3_validate_binding(one, present);
    require(rep.ok(), "truncation is not an error");
    require(rep.out_of_scope == dgpp::qwen3_expected_layer_tensors(two, 1).size(), "layer 1 is out of scope");
  }
  {
    // Tied embeddings with a stored head: accepted as a copy, never expected.
    Qwen3TextConfig d = parse(qwen3_config_json::kReranker);
    d.num_hidden_layers = 1;
    auto p = present_of(dgpp::qwen3_expected_tensors(d));
    p.emplace("lm_head.weight", dgpp::Qwen3TensorDesc{DType::BF16, {151669, 1024}});
    const dgpp::Qwen3BindReport rep = dgpp::qwen3_validate_binding(d, p);
    require(rep.ok() && rep.tied_head_copies == 1 && rep.unexpected == 0, "a tied checkpoint's stored head is a copy");
    p["lm_head.weight"].shape = {151669, 2048};
    require(!dgpp::qwen3_validate_binding(d, p).ok(), "a head of another shape is not");
  }
  {
    // The VL tower is part of the table: a checkpoint without it does not bind.
    Qwen3TextConfig v = parse(qwen3_config_json::kVl30b);
    v.num_hidden_layers = 1;
    auto p = present_of(dgpp::qwen3_expected_tensors(v));
    require(dgpp::qwen3_validate_binding(v, p).vision == 351, "351 tower tensors matched");
    p.erase("model.visual.merger.linear_fc2.bias");
    const dgpp::Qwen3BindReport rep = dgpp::qwen3_validate_binding(v, p);
    require(!rep.ok() && rep.missing == 1 && rep.vision == 350, "a missing tower tensor is missing");
  }
}

DGPP_TEST(qwen3_tp_geometry_and_rank_bytes) {
  const Qwen3TextConfig big = parse(qwen3_config_json::kMoe235b);
  const Qwen3TextConfig vl = parse(qwen3_config_json::kVl30b);
  for (int world : {1, 2, 4, 8})
    for (int rank = 0; rank < world; ++rank) {
      dgpp::qwen3_tp_validate_geometry(big, rank, world);
      dgpp::qwen3_tp_validate_geometry(vl, rank, world);
    }
  const auto refused = [](const Qwen3TextConfig& c, int rank, int world) {
    try {
      dgpp::qwen3_tp_validate_geometry(c, rank, world);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(refused(big, 0, 3), "64 heads do not divide by 3");
  require(refused(big, 2, 2), "rank out of range");
  require(refused(vl, 0, 16), "768 / 16 = 48 is not a multiple of 32");
  require(!refused(big, 0, 16), "1536 / 16 = 96, 4 heads per rank inside one kv head");

  // The 235B at the worlds the plan quotes: what one rank holds resident.
  const dgpp::Qwen3RankBytes w1 = dgpp::qwen3_rank_resident_bytes(big, 0, 1);
  require(w1.total() == 143736344576ull, "world 1: 143,736,344,576 bytes (133.86 GiB — more than one Spark)");
  const dgpp::Qwen3RankBytes a = dgpp::qwen3_rank_resident_bytes(big, 0, 2);
  const dgpp::Qwen3RankBytes b = dgpp::qwen3_rank_resident_bytes(big, 1, 2);
  require(a.total() == 72540655616ull && b.total() == a.total(), "world 2: 72,540,655,616 bytes a rank (67.56 GiB)");
  require(a.experts == 63871005696ull, "experts: half of every gate/up row range and down column range");
  require(a.attention == 6702497792ull, "attention: 32 query heads, 2 kv heads, BF16");
  require(a.router == 98566144ull && a.embed == 1244659712ull, "router and embedding replicated");
  require(a.head == 622329856ull && a.head + b.head == w1.head, "the head's vocab halves");
  // The 30B on one Spark.
  require(dgpp::qwen3_rank_resident_bytes(vl, 0, 1).total() == 19389714432ull, "VL-30B world 1: 19,389,714,432 bytes (18.06 GiB)");
}
