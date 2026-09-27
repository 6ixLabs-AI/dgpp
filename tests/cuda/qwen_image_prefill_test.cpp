// Exercise image ownership through the real session/model/vision paths.
// The tiny vision weights live beside the existing synthetic text fixture.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "models/qwen/forward.hpp"
#include "qwen_fixture.hpp"

namespace {
using namespace dgpp;
QwenTextConfig config;
std::string fixture;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

bool same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<QwenExpectedTensor> vision_tensors(const QwenVisionConfig& c) {
  std::vector<QwenExpectedTensor> tensors;
  const auto add = [&](const std::string& name, std::vector<int64_t> shape) {
    tensors.push_back({"model.visual." + name, DType::BF16, std::move(shape)});
  };
  const auto linear = [&](const std::string& name, int out, int in) {
    add(name + ".weight", {out, in});
    add(name + ".bias", {out});
  };
  add("patch_embed.proj.weight", {c.hidden, 3, c.kTemporal, c.kPatch, c.kPatch});
  add("patch_embed.proj.bias", {c.hidden});
  add("pos_embed.weight", {c.num_position_embeddings, c.hidden});
  for (int b = 0; b < c.depth; ++b) {
    const std::string p = "blocks." + std::to_string(b) + ".";
    for (const char* norm : {"norm1", "norm2"}) {
      add(p + norm + ".weight", {c.hidden});
      add(p + norm + ".bias", {c.hidden});
    }
    linear(p + "attn.qkv", 3 * c.hidden, c.hidden);
    linear(p + "attn.proj", c.hidden, c.hidden);
    linear(p + "mlp.linear_fc1", c.intermediate, c.hidden);
    linear(p + "mlp.linear_fc2", c.hidden, c.intermediate);
  }
  add("merger.norm.weight", {c.hidden});
  add("merger.norm.bias", {c.hidden});
  linear("merger.linear_fc1", c.merged(), c.merged());
  linear("merger.linear_fc2", c.output, c.merged());
  return tensors;
}

std::vector<ImageInput> images(int color, int64_t offset = 0) {
  ImageInput im;
  im.grid = kQwenImageGrid;
  im.offset = offset;
  im.tokens = 8;
  im.width = 128;
  im.height = 64;
  im.rgb.resize(static_cast<size_t>(im.width) * im.height * 3);
  for (size_t i = 0; i < im.rgb.size(); ++i)
    im.rgb[i] = static_cast<uint8_t>((i * 13 + color * (i % 3 + 1)) % 256);
  return {im};
}

std::vector<int64_t> prompt_for(const std::vector<ImageInput>& input) {
  std::vector<int64_t> prompt(20, 42);
  for (const auto& im : input)
    std::fill(prompt.begin() + im.offset, prompt.begin() + im.offset + im.tokens,
              config.vision->tokens.pad);
  return prompt;
}

const std::vector<int64_t> cuts{4, 8, 12, 16};

void ownership(bool mtp) {
  QwenModel m(config, fixture, 64, 512, QwenResidency::Resident, nullptr, 0, 1, 2, mtp);
  const auto a = images(11), b = images(99);
  const auto prompt = prompt_for(a);
  const auto text = m.session_prefill(0, prompt, cuts);
  m.session_close(0);
  const auto solo_a = m.session_prefill_images(0, prompt, a, cuts, nullptr);
  const auto draft_a = mtp ? m.session_draft(0, {7}) : QwenModel::Outputs{};
  m.session_close(0);
  const auto solo_b = m.session_prefill_images(0, prompt, b, cuts, nullptr);
  const auto draft_b = mtp ? m.session_draft(0, {7}) : QwenModel::Outputs{};
  m.session_close(0);
  require(!same(solo_a.logits, text.logits), "cold image prefill ignored the image");
  require(!same(solo_a.logits, solo_b.logits), "different images must affect the model");

  // Both cursors begin before either advances; their windows have identical
  // positions but different pixels. The shared window must change owners.
  auto ca = m.session_prefill_begin(0, prompt, 32, 4, {}, nullptr, 0, &a);
  auto cb = m.session_prefill_begin(1, prompt, 32, 4, {}, nullptr, 0, &b);
  bool done_a = false, done_b = false;
  while (!done_a || !done_b) {
    if (!done_a) done_a = m.session_prefill_advance(ca);
    if (!done_b) done_b = m.session_prefill_advance(cb);
  }
  require(same(ca.output.logits, solo_a.logits), "interleaving changed A's image prefill");
  require(same(cb.output.logits, solo_b.logits), "interleaving changed B's image prefill");
  if (mtp) {
    require(same(m.session_draft(0, {7}).logits, draft_a.logits), "interleaving changed A's draft");
    require(same(m.session_draft(1, {7}).logits, draft_b.logits), "interleaving changed B's draft");
  }
  m.session_close(0);
  m.session_close(1);

  auto cancelled = m.session_prefill_begin(0, prompt, 32, 4, {}, nullptr, 0, &a);
  require(!m.session_prefill_advance(cancelled), "cancellation needs an unfinished prefill");
  bool rejected = false;
  try {
    (void)m.session_prefill_advance(cancelled, 3);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "invalid advance budget must be rejected");
  m.session_close(0);
  auto reused = m.session_prefill_begin(0, prompt, 32, 4);
  while (!m.session_prefill_advance(reused)) {
  }
  require(same(reused.output.logits, text.logits),
          "cancelled image prefill leaked into text slot reuse");
  (void)m.session_step(0, 7);
  m.session_close(0);
}

DGPP_TEST(qwen_image_prefill_ownership) {
  ownership(false);
}
DGPP_TEST(qwen_image_prefill_ownership_mtp) {
  ownership(true);
}

DGPP_TEST(qwen_image_prefill_attached_mtp_catchup) {
  QwenModel m(config, fixture, 64, 512, QwenResidency::Resident, nullptr, 0, 1, 2, true);
  const auto input = images(99, 4);
  const auto prompt = prompt_for(input);
  // A final-prompt snapshot leaves MTP one row behind. Its catch-up token
  // here is the first image row, so begin must expose the image too.
  (void)m.session_prefill(0, {prompt.begin(), prompt.begin() + 4});
  void* snapshot = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&snapshot, m.session_snapshot_bytes()));
  const auto meta = m.session_snapshot(0, snapshot);
  m.session_close(0);
  m.session_attach(0, snapshot, meta);
  const auto expected =
      m.session_prefill_resume_images(0, {prompt.begin() + 4, prompt.end()}, input, cuts, nullptr);
  const auto expected_draft = m.session_draft(0, {7});
  m.session_close(0);
  m.session_attach(1, snapshot, meta);
  auto cursor = m.session_prefill_begin(1, prompt, 32, 4, {}, nullptr, 4, &input);
  while (!m.session_prefill_advance(cursor)) {
  }
  require(same(cursor.output.logits, expected.logits),
          "yielding changed the attached image target");
  require(same(m.session_draft(1, {7}).logits, expected_draft.logits),
          "attached image MTP catch-up differs");
  m.session_close(1);
  m.session_release_snapshot(meta);
  DGPP_CUDA_OK(cudaFree(snapshot));
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 2;
  fixture = argv[1];
  config = qwenfx::tiny_config();
  config.vision = QwenVisionConfig{};
  auto& vision = *config.vision;
  vision.depth = 1;
  vision.hidden = 32;
  vision.heads = 1;
  vision.intermediate = 64;
  vision.output = config.hidden_size;
  vision.tokens = {3, 2, 4};
  qwenfx::write_fixture(config, fixture, qwenfx::tiny_text_json(), qwenfx::tiny_quant_json(),
                        vision_tensors(vision));
  return test::run_all();
}
