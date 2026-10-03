# Speculative decoding with MTP

An MTP layer predicts draft tokens from the main model's hidden state and
the next input token. The main model verifies those drafts, commits the
accepted prefix and restores state after a rejection. This can produce
more than one output token per decode step.

For a GLM-5.3-Flash diagnostic run on the fabric:

```bash
scripts/fabric_run.sh -- --model unsloth/GLM-5.3-Flash-FP8 \
    --chat "Write a history of the Roman Republic." --steps 300 \
    --decode-graph --mtp
```

## Execution and correctness

At depth 1, a graph replay verifies two rows: the pending token and one
draft. Device kernels select tokens, decide acceptance, commit the accepted
rows, restore speculative state where needed, and run the draft block for
the next step. Each rank derives the same result from the gathered logits.
The host reads the verdict and updates request bookkeeping. Sampling that
cannot be resolved from the candidate table uses an exact gather fallback.

Greedy MTP must produce the same transcript as plain decode. The verify
rows preserve single-row arithmetic, and state tests cover rejection at
pool boundaries, slot reuse and scalar/batch transitions. Compare greedy
runs with `scripts/fabric_xcript.py PLAIN_DIR MTP_DIR`; the result must be
`IDENTICAL`.

Sampled MTP preserves the target distribution. GLM-5.3 uses a greedy draft;
Qwen also supports sampled drafts with an acceptance ratio and residual
sampling after rejection. A sampled transcript need not equal a plain
sampled run with the same seed. See `src/engine/speculative.hpp` and
the sampling tests for the acceptance rules.

## Serving configuration

Set `engine.decode_graph` and `engine.mtp` to true. The server requires
graph decode for MTP; the GLM diagnostic tool also has an eager speculative
path. Graph serving works on one node when the model fits, using identity
collectives.

`engine.mtp_depth` accepts 1–5 and defaults to 1 (DeepSeek-V4.1 defaults
to its DSpark depth). Each step verifies `1 + depth` rows. GLM-5.3 uses
scalar graphs beyond depth 1; Qwen and GLM-4.7 run batched draft chains at
every depth within their row limits (Qwen: sixteen slots at depth 3, the
64-row cap).
At depth 1, GLM-5.3 and Qwen can batch up to four requests. The engine
chooses among scalar and available batch graphs according to occupancy
and `graph_batch_min_live`.

Deeper drafts add verification work and state. Their benefit depends on
acceptance, prompt class, context and concurrency. Use the memory plan
before increasing capacity, and measure tokens per step as well as step
latency. [Operations](operations.md) describes the depth tradeoff, and
[benchmarks](benchmarks.md) records the results for each model.

Deeper drafts add verification work and state. Their benefit depends on
acceptance, prompt class, context and concurrency. Use the memory plan
before increasing capacity, and measure tokens per step as well as step
latency. [Operations](operations.md) describes the depth tradeoff, and
[benchmarks](benchmarks.md) records the results for each model.

## The DFlash2 block drafter (Qwen3.8-27B)

DFlash2 (`src/models/qwen/dflash2.hpp`, `src/kernels/dflash2.cu`) replaces
the MTP draft with an external block-drafter checkpoint
(`z-lab/Qwen3.8-27B-DFlash2`): five bidirectional sliding-window Qwen3 layers
over a K/V context built from the target's tapped residual streams (layers
`[5, 19, 33, 47, 61]` through the split `fc` and `hidden_norm`), two-tap
dynamic grouped convolutions around attention and the MLP, and a codebook
path selector over each mask row's top-16 candidates. The drafter shares the
target's embedding and lm head and keeps its K/V in five extra planes of the
main pool, so block tables, prefix sharing and the positional overwrite
rollback apply to it verbatim. Every target pass (prefill chunks and verify
rows alike) feeds the planes at its positions; the block forward then runs
`[bonus, mask x 7]` and proposes seven tokens.

Serve it with `engine.dflash_model` and `mtp` / `decode_graph` off
(`deploy/cluster_qwen3.8-27b_fp8_w1_dflash2.example.json`; `--no-dflash` runs
the same recipe plain). Acceptance is the ordinary greedy verify: the fed
rows (the pending token plus the drafts) run through the target, the accepted
prefix commits, the rest rolls back, and the step returns the tokens it
decided (the accepted drafts and the verify's next token), so the transcript
equals the plain world's. Sampled, grammar-constrained and penalized requests
run the plain step. The throughput line carries the drafter's per-position
acceptance in the MTP group.

### Proposal rule: the selector walk

The reference's walk (`_score_edges` + `_selector_walk_kernel` at
temperature 0): per step `l`, `scores[l][p][c] = unary[l][c] +
<pred[id(l-1, p)] * hidden[l], succ[id(l, c)]>`, walked greedily from slot 0
(`token[l] = ids[l][argmax_c scores[l][prev][c]]`), the anchor token as every
slot's predecessor at step 0. The unary term is the candidate's logit; an
earlier form of the kernel added the predecessor's, which is constant over
`c`, so the walk ignored the logits (it verified ~1.2 tokens per pass and a
per-slot top-1 stood in). Fixed, the walk beats the top-1 on every prompt
class (2026-10-03, one GB10, C1): prose 2.88 vs 2.60, code 4.92 vs 4.57,
json 6.81 vs 6.40, math 5.42 vs 5.33, chat 2.73 vs 2.51 tokens per step at
the same 164 ms/step. The block, head, top-K and walk each have host
references in `dflash2_kernels_test`.

### The batched pass and its options

Every speculating slot's verify rows ride one physical target pass
(`SessionModel::session_verify_batch`, slot-major rows with per-slot rollback
bases; `EagerEngine::step_batch`), `floor(decode_rows / 8)` slots per pass
(four at the family's 32-row ceiling). A lone slot takes the scalar path, so
a C1 transcript keeps the scalar kernel sequence. Three options, all engine
keys with the measured-best as the default:

- `engine.dflash_verify_graph` (default true): multi-slot batches replay a
  captured static verify (one 16- or 32-row graph; every slot's rows padded
  to a full 8-row block, the padding rows at position -1 which every
  state-writing kernel skips). The drafter's plane feed is a recorded node.
  A capture failure falls back to the eager batch for the server's life.
- `engine.dflash_draft_batch` (default true): one stacked block forward
  redrafts every speculating slot (row-wise GEMMs, norms, convs and RoPE
  over the stacked rows; appends, attention, head, top-K and the walk per
  slot at row offsets) — the draft weights read once per step.
- `engine.dflash_depth` (default 0 = the block): verify only the first N
  drafts per step; unverified drafts re-draft next step, so transcripts are
  exact at any value.

The drafter's stacked forwards and the taps of a wide verify run their bf16
GEMMs through the model's own streaming tensor-core form from 17 rows
(`Qwen35Model::configure_gemm_rows`): the weights read once per launch
instead of once per 4-row GEMV chunk (the 8K profile's draft phase, 260 to
92 ms/pass). Other families' GEMM instances keep the shared rule.

### Measured (2026-10-03, one GB10, greedy, exact numerics)

Against the MTP depth-2 template on the same binary (`timed_load`, 320
tokens, wall tokens/s with prefill included; the step time from the engine
counters):

| class | DFlash2 C1 | C4 | MTP d2 C1 | C4 |
|---|---:|---:|---:|---:|
| prose | 15.4 | 41.5 | 13.0 | 52.6 |
| code | 27.3 | 83.1 | 15.1 | 65.8 |
| json | 38.0 | 83.6 | 16.3 | 68.9 |
| math | 31.5 | 75.5 | 15.8 | 64.1 |
| chat | 15.1 | 47.5 | 11.8 | 56.5 |

DFlash2 steps 164 ms at C1 (2.5–6.4 tokens per step by class), 176–179 at
C2 and 200–216 at C4 against a 125 ms byte floor (24.4 GB target + 1.3 GB
head + 3.85 GB drafter); MTP depth 2 steps 151 ms at C1 (2.1–2.9 tokens).
The author's `bench_qwen35` protocol (thinking on): Q&A 22.4, Code 32.2,
JSON 35.9, Math 31.0, LongCode 25.1 tokens/s (MTP: 13.6 / 14.4 / 16.4 /
14.2 / 13.7). The remaining gap is the per-pass cost at 4–8K context (the
one-warp-per-row attention kernels, #85) and the drafter's bf16 bytes.

## Recorded GLM-5.3 result

On 2026-09-03 at TP=4, greedy depth-1 MTP accepted 88.7% of drafts on the
recorded coherent-text workload. It produced 1.89 tokens per 42.4 ms step:
22.45 ms/token, compared with 31.3 ms/token for plain decode. The draft
layer added about 7.3 GiB per rank. These figures describe that checkpoint
and workload; acceptance fell on the post-EOS text generated with
`--no-eos`.
