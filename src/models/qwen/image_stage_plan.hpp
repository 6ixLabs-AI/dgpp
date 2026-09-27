#pragma once
// Windowed image-embedding staging plan (pure integer logic behind
// QwenModel::apply_image_embeddings): split [begin_all, end_all) into
// sub-spans of at most `window` rows and list, per sub-span, each image's
// overlapping rows in image order. The consumer stages each sub-span then
// copies its overlaps, so a chunk wider than the tower's staged window is
// consumed stage-copy-repeat with every image's rows and copy order
// untouched — bitwise the single-stage chain wherever that chain fits.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dgpp {
namespace image_stage {

// One image's rows in prompt space.
struct Span {
  int64_t offset = 0;
  int64_t tokens = 0;
};

// One image's overlap with a staged sub-span.
struct Copy {
  size_t image = 0;  // index into the spans
  int64_t begin = 0;  // first row, prompt space
  int64_t end = 0;    // one past the last row
};

// One stage-then-copy step.
struct Step {
  int64_t first = 0;  // sub-span to stage
  int64_t end = 0;
  std::vector<Copy> copies;  // non-empty overlaps, image order
};

inline std::vector<Step> plan(int64_t begin_all, int64_t end_all, int64_t window,
                              const Span* spans, size_t n) {
  std::vector<Step> steps;
  if (end_all <= begin_all || window <= 0) return steps;
  for (int64_t s0 = begin_all; s0 < end_all;) {
    const int64_t s1 = std::min(s0 + window, end_all);
    Step step;
    step.first = s0;
    step.end = s1;
    for (size_t i = 0; i < n; ++i) {
      const int64_t begin = std::max(s0, spans[i].offset);
      const int64_t end = std::min(s1, spans[i].offset + spans[i].tokens);
      if (end > begin) step.copies.push_back(Copy{i, begin, end});
    }
    steps.push_back(std::move(step));
    s0 = s1;
  }
  return steps;
}

}  // namespace image_stage
}  // namespace dgpp
