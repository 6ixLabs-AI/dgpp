#pragma once
// The on-device SAMPLING pick (DESIGN §10, the device path): the greedy
// pick's two kernels (glm_pick.hpp) generalized from top-1 to top-k per
// rank plus each slice's temperature-scaled log-sum-exp, and a verdict that
// makes sample::sample_from_prefix's decision BIT FOR BIT on the device
// — or flags the fallback for the host's exact gather. The host oracle is
// the definition; everything here mirrors its arithmetic: the same
// penalties (fused frequency step), the same fp32 temperature division, the
// same chunked normalizer order, the same deterministic exp/log
// (common/det_math.hpp), the same fp32 selector and fp64 walk, the same
// counter RNG. glm_pick_test pins the mirror against the host, value by
// value and bit by bit.
//
// Wire table: [rows][world][group] then the digest group [world][9] (the
// greedy pick's), even-padded. A rank's group is its k candidates (fp32
// logit bits as six digits, id as three — unused candidate slots carry the
// EMPTY id, an id no vocabulary reaches) followed by its slice lse (fp64
// bits as eleven digits). A greedy row writes one candidate (its canonical
// argmax) and no lse; a padding row writes nothing but zeros.
//
// Two shapes: T=1 (one row per request: accepted 1, next the decision)
// and the MTP T=2 verify (rows [next, draft] per request): row 0 accepts
// the draft with its exact probability or samples the residual
// (sample::spec_accept_from_prefix), and when the draft stands row 1
// is sampled as any step's token (sample_from_prefix) — so accepted is 2
// or 1 and next = winners[accepted-1], the greedy judge's shape. A row
// that cannot decide inside its prefix flags the fallback: row 0 by
// provisionally REJECTING (the commit then keeps only the post-row-0
// state, which is right for a reject and recoverable for an accept), row 1
// by feeding its provisional argmax; the host serves both between windows
// (GlmGraphEngineAdapter). The request's count table holds the tokens
// committed through the previous step: the local kernels read it with the
// step's fed tokens added per row (row 1 sees the draft), never writing
// it, and the verdict commits — the consumed token always, the draft only
// when it stood (sample::SampledSpeculator's context rule).
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/pick.hpp"

namespace dgpp {

// The per-request sampling spec on the device — the host writes it between
// windows (configure_sampling), the verdict advances `counter` when it
// draws. temperature <= 0 is the greedy path: the argmax, no penalties, no
// draw — bitwise the greedy pick.
struct SampleSpec {
  float temperature = 0.0f;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float repetition_penalty = 1.0f;
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;
  int32_t top_k = 0;
  int32_t logprobs = -1;  // -1: none; N >= 0: report the chosen token's
                          // logprob and its top-N alternatives. A greedy
                          // request that reports (or carries penalties)
                          // runs the full path at temperature 1 — the
                          // argmax under the raw normalizer, no draw.
  int32_t biased = 0;     // 1: the request's logit_bias row is live (the
                          // pick adds it after the penalties, in place;
                          // a biased greedy row takes the full path)
  uint64_t seed = 0;
  uint64_t counter = 0;
  // The DRAFT's temperature: the proposal is the draft head's
  // final set at THIS temperature, the request's own scaled by
  // DGPP_SPEC_PROPOSAL_TEMP (0 = the request's). Any proposal is exact;
  // the scale only moves the overlap with P, i.e. the acceptance rate.
  float draft_temperature = 0.0f;
};

// The DRAFT's own stream: a sampled draft draws from
// (seed ^ kSampleDraftSeedMix, counter) — the request's counter, which the
// verify advances every step, under a different key. So the draft's draw is
// independent of the accept test's u1/u2 (what the proposal rule needs),
// it never advances the request's counter (the host's draw accounting is
// untouched), and it carries no state of its own to keep in step across a
// fallback's push.
constexpr uint64_t kSampleDraftSeedMix = 0x9E3779B97F4A7C15ull;

constexpr int kSampleMaxTopLogprobs = 20;

// The verify rows a sampled verdict decides over: the fed token's row plus
// up to kSampleVerdictRows - 1 drafts — the families' kSpecRows (6 since
// the DeepSeek-V4.1 DSpark block of five drafts, 2026-09-14; the kernel's
// per-row tables moved to dynamic shared memory for it; 8 since the
// DFlash2 block of seven drafts on the graph worlds, 2026-10-04 — the
// dynamic tables reach ~63 KB of the GB10's 99 KB per block).
constexpr int kSampleVerdictRows = 8;

// The sampling verdict's outcome per request, beside the PickVerdict the
// device consumers (commit, token feeds) keep reading.
struct SampleOutcome {
  int32_t fallback = 0;   // a row could not decide inside its prefix
  int32_t sampled = 0;    // a stochastic decision was made (0: greedy row)
  uint64_t counter = 0;   // the spec's counter after this pick (the draws
                          // the device consumed; a fallback row's draw is
                          // reserved for the host)
  int32_t fallback_row = -1;  // which row fell back (0..rows-1), -1 none
  int32_t accepted_draft = 0; // T>=2: the first draft stood (provisional 0
                              // on a row-0 fallback)
  // Per verify row t (2026-09-06, T = 1 + drafts): the fold log-sum-exp,
  // the prefix mass under it (rows the device decided over) and the
  // outcome's log-probability. Rows the chain never reached stay zero.
  double normalizer[kSampleVerdictRows] = {};
  double covered_mass[kSampleVerdictRows] = {};
  float logprob[kSampleVerdictRows] = {};
  // The reported top logprobs per row (spec.logprobs >= 0, rows the device
  // decided): min(N, the final set) entries, as the host's Result.
  int32_t top_count[kSampleVerdictRows] = {};
  int32_t top_ids[kSampleVerdictRows][kSampleMaxTopLogprobs] = {};
  float top_logprobs[kSampleVerdictRows][kSampleMaxTopLogprobs] = {};
};

// The draft-probability confidence (2026-09-14, engine/verify_schedule.hpp)
// for a family without a confidence head: draft position c's acceptance
// logit is logit(p_c), p_c = exp(logprob[0]) of the draft pick at picker
// slot 1 + c (its outcome for `request`, `slot_stride` outcomes per slot,
// computed on the full path — the pick's spec reports logprobs — so it is
// the raw-normalizer probability of the draft's argmax, identical on every
// rank: it comes out of the pick's fold). Writes dst[c] for c < depth; a
// row the pick never decided (logprob 0 = p 1 on a padding row) is
// clamped to the same logit bound as a certain one.
void device_sample_draft_confidence(const SampleOutcome* outcomes,
                                    int slot_stride, int request, int depth,
                                    float* dst, cudaStream_t stream);

// The draft's proposal: the distribution the draft was DRAWN
// from — the draft head's own final set after the request's temperature,
// top-k, top-p and min-p — carried from the draft pick to the next step's
// verify, where the row-0 test becomes min(1, P/Q) with the (P - Q)+
// residual (sample::Proposal, sample::spec_select_from_sorted). n == 0 says
// the draft was deterministic (or its final set did not fit) and the verify
// uses the plain P(draft) rule, which is exact for any draft whatsoever —
// only the acceptance rate differs. `token` guards the pairing: a proposal
// whose token is not the draft the verify was fed is ignored (the host
// re-drafted between windows).
constexpr int kSampleProposalMax = 64;
constexpr int kSampleProposalSlots = kSampleVerdictRows - 1;  // drafts per request
struct DraftProposal {
  int32_t n = 0;
  int32_t token = -1;
  int32_t ids[kSampleProposalMax] = {};
  float mass[kSampleProposalMax] = {};
};

// The token mask of constrained decoding (M6 6g), per ROW: word 0 is the
// allowed count (0 = the row is unconstrained and the words after it are
// not read), words 1.. the bitmask over [0, vocab_size) (bit id set =
// allowed). device_sample_mask_words(vocab_size) words per row; the host
// writes a constrained row's mask before the step (the grammar state's
// next-position mask; the MTP verify's row 1 under the draft). A masked
// id is ABSENT: -inf in place, never a candidate, no mass, and the row's
// vocabulary — the completeness test's — is the allowed count.
constexpr int device_sample_mask_words(int vocab_size) {
  return 1 + (vocab_size + 31) / 32;
}

constexpr int kSampleLseDigits = 11;         // 66 bits carry the fp64 lse
constexpr int kSampleMaxCandidates = 256;    // the profiler's ceiling
constexpr int kSampleLseChunk = 256;         // == sample::kLseChunk
constexpr int kSampleMaxChunks = 1024;       // slices up to 262,144 ids
constexpr uint32_t kSampleEmptyId = (1u << (6 * kPickIdDigits)) - 1;

constexpr size_t device_sample_rank_group_slots(int candidates) {
  return static_cast<size_t>(candidates) * kPickSlotsPerRank +
         kSampleLseDigits;
}
constexpr size_t device_sample_table_elems(int rows, int world, int candidates) {
  const size_t slots =
      static_cast<size_t>(rows) * static_cast<size_t>(world) *
          device_sample_rank_group_slots(candidates) +
      static_cast<size_t>(world) * kPickSlotsPerRank;
  return slots + (slots & 1);
}
// The local pick's device scratch, in doubles: the per-chunk maxes and
// normalizer partials of `rows` rows of `vocab_count` ids (the widest
// shape is device_sample_scratch_elems(kPickMaxRows, kSampleLseChunk *
// kSampleMaxChunks), 128 KiB).
constexpr size_t device_sample_scratch_elems(int rows, int vocab_count) {
  const size_t chunks =
      (static_cast<size_t>(vocab_count) + kSampleLseChunk - 1) / kSampleLseChunk;
  return 2 * static_cast<size_t>(rows) * chunks;
}
// The widest candidate table for `rows` rows at `world` ranks that fits
// `slot_bytes` (0 when even one candidate does not).
constexpr int device_sample_candidates_that_fit(int rows, int world,
                                             size_t slot_bytes, int wanted) {
  int k = wanted;
  while (k > 0 && device_sample_table_elems(rows, world, k) * 2 > slot_bytes)
    --k;
  return k;
}

// Kernel 1 (before the fold): three launches on `stream`, every row
// independent. Row r belongs to request r / rows_per_request; `positions`
// (optional) marks a padding request by a negative first position. A
// sampled row (its spec's temperature > 0, or a greedy row that reports
// logprobs or carries penalties) applies the penalties IN PLACE on its
// logits slice (sample::apply_penalties, the context being the
// request's count table plus the step's fed tokens through this row) and
// its temperature-scaled max per 256-id chunk [one block per chunk], the
// chunk's normalizer partial in the host's order [one block per chunk],
// then [one block per row] the slice log-sum-exp folded in chunk order
// (sample::slice_logsumexp) and the exact local top-k in canonical
// order (sample::local_topk, a radix select). A greedy row writes its
// canonical argmax as the one candidate. locals[row] carries the row's
// best (and, for sampled rows, second-best) for the log. `scratch` holds
// device_sample_scratch_elems(rows, vocab_count) doubles. The count table is
// read only.
// `masks` (optional): the rows' token masks, `mask_stride` words apart
// (a constrained row takes the full path whatever its temperature).
// `bias` (optional, 2026-09-06): the requests' logit_bias rows,
// [requests][vocab_size] floats, added in place for rows whose spec says
// `biased`.
// `row_select` (optional, 2026-09-10): one candidate row per request, as
// the greedy local's — request q reads logits row
// q * source_row_stride + (row_select[q].accepted - 1). The draft pick's
// shape: its head ran on the verify layout and the pick samples the row the
// verdict accepted. `counts` may be null, and then no row is penalized (the
// draft's proposal needs no penalty: any Q is exact, see DraftProposal).
void device_sample_local(float* logits, int rows, int vocab_count, int vocab_begin, int vocab_size,
                         int rank, int world, int candidates, const SampleSpec* specs,
                         int rows_per_request, const int64_t* fed, const int64_t* positions,
                         int position_stride, const int32_t* counts, const float* bias,
                         const uint32_t* masks, int mask_stride, const uint64_t* carry_digest,
                         uint16_t* table, PickLocal* locals, double* scratch, cudaStream_t stream,
                         const PickVerdict* row_select = nullptr, int source_row_stride = 0,
                         const int32_t* request_map = nullptr);

// Kernel 2 (after the fold), one block per request plus the digest pass:
// decodes every rank's group, merges the k-way prefix in canonical order
// (sample::merge_topk), folds the lse (sample::merge_logsumexp) and
// makes sample_from_prefix's decision with the request's spec and RNG.
// Writes the PickVerdict (rows/accepted 1, next and winners[0] the
// decision — the provisional argmax on a fallback — and the digest chain
// exactly as glm_pick_verdict_batched) to the pinned mirror and the device
// copy, the SampleOutcome, and the spec's advanced counter; then
// commits the step's fed tokens into the request's count table (the
// consumed token; the draft when accepted == 2).
// `masks`/`mask_stride` as glm_sample_local's: a constrained row decides
// over its allowed count and a masked draft is rejected outright.
// `proposals_in` (optional): the drafts' proposals, [requests][slots] with
// slot t the draft fed to row t+1 — row t's accept test takes the ratio
// rule when its proposal carries that draft. `proposals_out` (optional, the
// DRAFT pick: rows_per_request == 1) receives the final set this pick drew
// from, at [q][draft_index], on the device and, when given, in a pinned
// mirror for the host's fallback. `counts` may be null (the draft pick
// commits no context).
// Sets the verdict kernel's dynamic shared-memory attribute (idempotent).
// device_sample_verdict calls it; a picker that captures graphs calls it
// from its constructor so the first launch inside a capture finds it set.
void device_sample_verdict_prepare();

void device_sample_verdict(const uint16_t* table, int rows, int world, int rank, int candidates,
                           int vocab_size, SampleSpec* specs, int requests, int rows_per_request,
                           const int64_t* fed, const int64_t* positions, int position_stride,
                           int32_t* counts, const uint32_t* masks, int mask_stride,
                           PickVerdict* verdicts, PickVerdict* device_verdicts,
                           SampleOutcome* outcomes, uint64_t* carry_digest, cudaStream_t stream,
                           const DraftProposal* proposals_in = nullptr,
                           DraftProposal* proposals_out = nullptr,
                           DraftProposal* proposals_out_host = nullptr, int draft_index = 0,
                           const int32_t* request_map = nullptr);

// counts[token] += delta (the host's correction of a request's context after
// a fallback it decided: a provisionally rejected draft joins the table
// when the host's decision accepts it after all).
void device_sample_adjust_count(int32_t* counts_row, int64_t token, int delta,
                             int vocab_size, cudaStream_t stream);

// The prefill's context: counts[ids[i]] += 1 for i < n (a request's prompt).
void device_sample_count_tokens(int32_t* counts_row, const int64_t* ids, int n,
                             int vocab_size, cudaStream_t stream);

}  // namespace dgpp
