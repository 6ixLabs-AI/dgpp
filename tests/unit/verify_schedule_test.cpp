// The confidence-scheduled verify-depth policy (engine/verify_schedule.hpp): the Dinkelbach-optimal
// depth for aggregate decode throughput. Exactness is not at stake (every k
// commits the same greedy prefix); these pin the economics.
//
// The policy verifies draft i while its prefix-survival S_i exceeds the tokens
// that one row of verify time is worth elsewhere (lambda * row_ms), and stops
// at the first draft that falls below -- S_i is monotone, so kept drafts are a
// prefix. Larger lambda (time more valuable) => shallower; a larger fixed base
// lowers the reservation lambda => deeper (amortization).
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "engine/verify_schedule.hpp"

namespace {
void require(bool c, const char* w) {
  if (!c) throw std::runtime_error(w);
}
float logit(double p) { return static_cast<float>(std::log(p / (1.0 - p))); }
}  // namespace

DGPP_TEST(verify_schedule_hot_stream_verifies_whole_block) {
  // Every position ~99% likely, lambda at the plain-decode reservation rate
  // (profiled curve: base ~20 ms, ~9 ms/row): the whole block clears.
  std::vector<float> c(5, logit(0.99));
  const float lam = dgpp::verify_reservation_lambda(20.0f, 9.0f);
  require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam) == 5,
          "hot draft verifies all five");
}

DGPP_TEST(verify_schedule_cold_stream_verifies_nothing) {
  // Drafts almost certain to be rejected: verify none, decode the target token.
  std::vector<float> c(5, logit(0.02));
  const float lam = dgpp::verify_reservation_lambda(20.0f, 9.0f);
  require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam) == 0,
          "a cold stream verifies zero drafts");
}

DGPP_TEST(verify_schedule_stops_at_the_survival_cliff) {
  // Likely, then a cliff: verify the surviving prefix only.
  const std::vector<float> c = {logit(0.9), logit(0.8), logit(0.1),
                                logit(0.9), logit(0.9)};
  const float lam = dgpp::verify_reservation_lambda(20.0f, 9.0f);
  // S = .9, .72, .072, ... threshold = lam*9 = 9/29 = .310. Prefix over .310 is
  // {.9,.72}; .072 falls below and stops even though a later logit is high.
  require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam) == 2,
          "stops at the first sub-threshold survival, ignoring later spikes");
}

DGPP_TEST(verify_schedule_accept_scale_shortens_a_sampled_slot) {
  // A sampled slot with argmax drafts accepts at a fraction of the head's
  // rate per position: the scaled survival falls under the threshold sooner
  // (never later), scale 1 is the greedy rule, and the batch rule takes a
  // scale per slot.
  std::vector<float> c(5, logit(0.8));
  const float lam = dgpp::verify_reservation_lambda(20.0f, 9.0f);  // threshold .310
  // S = .8, .64, .512, .410, .328: all five clear at scale 1.
  require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam) == 5, "the greedy rule clears the block");
  require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam, 1.0) == 5, "scale 1 is the greedy rule");
  // At 0.9: p = .72 -> S = .72, .518, .373, .269: three.
  require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam, 0.9) == 3, "a scaled slot stops sooner");
  for (double s = 0.5; s < 1.0; s += 0.1)
    require(dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam, s) <=
                dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lam, s + 0.1),
            "the depth is non-decreasing in the scale");
  const float* confs[2] = {c.data(), c.data()};
  const double ones[2] = {1.0, 1.0}, mixed[2] = {1.0, 0.9};
  require(dgpp::scheduled_verify_depth_batch(confs, 2, 5, 9.0f, lam, ones) ==
              dgpp::scheduled_verify_depth_batch(confs, 2, 5, 9.0f, lam),
          "unit scales are the unscaled batch rule");
  // Mean survival of (1.0, 0.9): (.8+.72)/2 .. -> .76, .579, .442, .340, .261: four.
  require(dgpp::scheduled_verify_depth_batch(confs, 2, 5, 9.0f, lam, mixed) == 4,
          "the batch rule averages the slots' scaled survivals");
}

DGPP_TEST(verify_schedule_larger_lambda_verifies_shallower) {
  // Monotone in lambda: the more valuable decode time is, the fewer drafts.
  std::vector<float> c(5, logit(0.85));
  const int a = dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, 0.02f);
  const int b = dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, 0.05f);
  const int d = dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, 0.09f);
  require(a >= b && b >= d, "depth is non-increasing in lambda");
  require(a == 5, "cheap time verifies the whole block");
  require(d < 5, "dear time verifies less than the whole block");
}

DGPP_TEST(verify_schedule_larger_base_verifies_deeper) {
  // Amortization via the reservation rate: a larger fixed base lowers lambda,
  // which lowers the threshold, which verifies deeper.
  std::vector<float> c(5, logit(0.6));
  const float lo_base = dgpp::verify_reservation_lambda(5.0f, 9.0f);
  const float hi_base = dgpp::verify_reservation_lambda(200.0f, 9.0f);
  const int shallow = dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, lo_base);
  const int deep = dgpp::scheduled_verify_depth(c.data(), 5, 9.0f, hi_base);
  require(deep >= shallow, "a larger fixed base never verifies shallower");
}

DGPP_TEST(verify_schedule_batch_takes_the_mean_survival) {
  // Two slots at threshold 0.3 (lambda 0.1/ms x 3 ms): alone, the hot slot
  // verifies 2 (S 0.9, 0.45 > 0.3; 0.09 stops) and the cold one 0; the
  // batch's mean survival (0.5, 0.25) keeps one position: the second
  // position's row would cost two rows of time for 0.5 expected tokens.
  const std::vector<float> hot = {logit(0.9), logit(0.5), logit(0.2)};
  const std::vector<float> cold = {logit(0.1), logit(0.5), logit(0.5)};
  const float* confs[2] = {hot.data(), cold.data()};
  require(dgpp::scheduled_verify_depth(hot.data(), 3, 3.0f, 0.1f) == 2, "the hot slot alone verifies 2");
  require(dgpp::scheduled_verify_depth(cold.data(), 3, 3.0f, 0.1f) == 0, "the cold slot alone verifies 0");
  require(dgpp::scheduled_verify_depth_batch(confs, 2, 3, 3.0f, 0.1f) == 1, "the batch verifies 1");
  // One slot: the batch rule is the scalar rule.
  require(dgpp::scheduled_verify_depth_batch(confs, 1, 3, 3.0f, 0.1f) == 2, "a batch of one is the scalar rule");
  // Two hot slots: the whole block of 2 as alone (the mean is each slot's S).
  const float* hots[2] = {hot.data(), hot.data()};
  require(dgpp::scheduled_verify_depth_batch(hots, 2, 3, 3.0f, 0.1f) == 2, "twin hot slots verify as one");
  require(dgpp::scheduled_verify_depth_batch(confs, 0, 3, 3.0f, 0.1f) == 0, "no slots: nothing");
}

DGPP_TEST(verify_schedule_adaptive_lambda_converges_to_the_modeled_throughput) {
  // Six slots at depth 4 (30 rows, base 28 + 30 x 8 = 268 ms) committing 19
  // tokens a step: the EWMA converges to 19 / 268 = 0.0709 from the
  // one-stream constant 0.045; one slot at 5 rows committing 3.3 lands at
  // 3.3 / 68 = 0.0485. The update is a fixed sequence of doubles on
  // replicated inputs (the ranks agree by construction).
  double lam = 0.045;
  for (int i = 0; i < 2000; ++i) lam = dgpp::verify_lambda_update(lam, 19, 30, 28.f, 8.f, 1.0 / 64.0);
  require(std::fabs(lam - 19.0 / 268.0) < 1e-6, "six slots: the modeled six-stream rate");
  double lam1 = 0.045;
  for (int i = 0; i < 2000; ++i) lam1 = dgpp::verify_lambda_update(lam1, (i % 10 < 3) ? 4 : 3, 5, 28.f, 8.f, 1.0 / 64.0);
  require(std::fabs(lam1 - 3.3 / 68.0) < 2e-3, "one slot: the modeled one-stream rate");
  require(dgpp::verify_lambda_update(0.05, 3, 0, 0.f, 0.f, 0.5) == 0.05, "a degenerate time leaves lambda alone");
  double a = 0.045, b = 0.045;
  for (int i = 0; i < 100; ++i) { a = dgpp::verify_lambda_update(a, 7, 12, 28.f, 8.f, 1.0 / 64.0); b = dgpp::verify_lambda_update(b, 7, 12, 28.f, 8.f, 1.0 / 64.0); }
  require(a == b, "the same inputs give the same doubles");
}

DGPP_TEST(verify_schedule_degenerate_block) {
  const float lam = dgpp::verify_reservation_lambda(20.0f, 9.0f);
  require(dgpp::scheduled_verify_depth(nullptr, 0, 9.0f, lam) == 0,
          "empty block verifies nothing");
  std::vector<float> c(1, logit(0.99));
  require(dgpp::scheduled_verify_depth(c.data(), 1, 9.0f, lam) == 1,
          "a single confident draft is worth verifying");
}
