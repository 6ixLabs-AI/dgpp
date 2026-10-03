#pragma once

#include <stdexcept>

namespace dgpp {

// The shared K/V pool had no free block for a prefix-cache snapshot's partial
// block. Derived from runtime_error so every existing handler still sees it;
// the snapshot sites catch it specifically and skip that one snapshot (an
// optimisation) instead of failing the engine.
struct CachePoolExhausted : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace dgpp
