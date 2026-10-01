#pragma once

#include <algorithm>
#include <cstdlib>

#include "engine/decode_outputs.hpp"

namespace dgpp {

// Serving needs one prefill output per request and every decode/verify row.
// Diagnostic constructors retain all rows. Set the override before creating
// the memory plan and model to compare against full storage and projection.
inline int compact_logits_rows(bool serving, int decode_rows, int requests) {
  const char* all = std::getenv("DGPP_PREFILL_HEAD_ALL_ROWS");
  if (!serving || (all && all[0] == '1')) return 0;
  return std::max({kDecodeRows, decode_rows, requests});
}

inline int serving_logits_capacity(int max_tokens, bool serving, int decode_rows, int requests) {
  const int compact = compact_logits_rows(serving, decode_rows, requests);
  return compact > 0 ? compact : max_tokens;
}

}  // namespace dgpp
