#include "models/qwen/image_stage_plan.hpp"

#include <stdexcept>

#include "common/test.hpp"

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
using dgpp::image_stage::Span;

std::vector<dgpp::image_stage::Step> run(int64_t begin_all, int64_t end_all, int64_t window,
                                         std::vector<Span> spans) {
  return dgpp::image_stage::plan(begin_all, end_all, window, spans.data(), spans.size());
}
}  // namespace

// One chunk inside the window: a single stage, exact per-image overlaps,
// images outside the chunk skipped.
DGPP_TEST(image_stage_single_step_within_window) {
  auto steps = run(0, 1500, 2049, {{100, 200}, {500, 100}, {2000, 50}});
  require(steps.size() == 1, "one step");
  require(steps[0].first == 0 && steps[0].end == 1500, "step covers chunk");
  require(steps[0].copies.size() == 2, "two overlaps");
  require(steps[0].copies[0].image == 0 && steps[0].copies[0].begin == 100 &&
              steps[0].copies[0].end == 300,
          "first overlap exact");
  require(steps[0].copies[1].image == 1 && steps[0].copies[1].begin == 500 &&
              steps[0].copies[1].end == 600,
          "second overlap exact");
}

// A chunk wider than the window splits into contiguous sub-spans, each at
// most the window wide.
DGPP_TEST(image_stage_wide_chunk_splits_at_window) {
  auto steps = run(0, 5000, 2049, {});
  require(steps.size() == 3, "three steps");
  require(steps[0].first == 0 && steps[0].end == 2049, "step 0");
  require(steps[1].first == 2049 && steps[1].end == 4098, "step 1");
  require(steps[2].first == 4098 && steps[2].end == 5000, "step 2 tail");
  for (const auto& s : steps) require(s.end - s.first <= 2049, "width bound");
}

// An image straddling a sub-span boundary is partitioned exactly: the two
// copies tile the image's rows, in order.
DGPP_TEST(image_stage_image_split_across_boundary) {
  auto steps = run(0, 5000, 2049, {{2000, 200}});
  require(steps.size() == 3, "three steps");
  require(steps[0].copies.size() == 1 && steps[0].copies[0].begin == 2000 &&
              steps[0].copies[0].end == 2049,
          "head copy");
  require(steps[1].copies.size() == 1 && steps[1].copies[0].image == 0 &&
              steps[1].copies[0].begin == 2049 && steps[1].copies[0].end == 2200,
          "tail copy");
  require(steps[2].copies.empty(), "no copy past the image");
}

// A shifted (MTP-style) range splits the same way in its own coordinates.
DGPP_TEST(image_stage_shifted_range) {
  auto steps = run(100, 2300, 2049, {{150, 50}});
  require(steps.size() == 2, "two steps");
  require(steps[0].first == 100 && steps[0].end == 2149, "shifted step 0");
  require(steps[1].first == 2149 && steps[1].end == 2300, "shifted step 1");
  require(steps[0].copies.size() == 1 && steps[0].copies[0].begin == 150 &&
              steps[0].copies[0].end == 200,
          "shifted overlap");
  require(steps[1].copies.empty(), "nothing in tail");
}

// Degenerate inputs: empty range or non-positive window plans nothing.
DGPP_TEST(image_stage_degenerate_inputs) {
  require(run(500, 500, 2049, {{0, 1000}}).empty(), "empty range");
  require(run(1000, 500, 2049, {{0, 1000}}).empty(), "inverted range");
  require(run(0, 5000, 0, {{0, 1000}}).empty(), "zero window");
}
