#pragma once

#include <algorithm>

#include "sched/scheduler.hpp"

namespace dgpp::serve {

// -1 is the serving default. Resolve it before publishing the warm record:
// every rank must run the same concrete budget, independent of HTTP timing.
inline sched::AdmissionPolicy resolve_prefill_policy(
    sched::AdmissionPolicy policy, const sched::SchedulerEngine& engine) {
  if (policy.prefill_budget_tokens != -1) return policy;
  const int64_t align = engine.prefill_chunk_alignment();
  const int64_t limit = engine.prefill_chunk_limit();
  policy.prefill_budget_tokens = 0;
  if (align > 0 && limit >= align) {
    int64_t budget = std::min(limit, std::max<int64_t>(256, align));
    if (policy.prefill_idle_budget_tokens > 0)
      budget = std::min<int64_t>(budget, policy.prefill_idle_budget_tokens);
    policy.prefill_budget_tokens = budget / align * align;
    // An engine that reads its in-flight prompts in as one walk shares a
    // tick's budget among them at no cost in walks: with nothing decoding,
    // the tick takes the whole forward (prompts that arrive together are
    // read in together, at the one-shot's rate).
    if (policy.prefill_idle_budget_tokens == 0 && engine.prefill_group_advance())
      policy.prefill_idle_budget_tokens = static_cast<int>(limit / align * align);
  }
  return policy;
}

}  // namespace dgpp::serve
