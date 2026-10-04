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

Serve it with `engine.dflash_model` and `mtp` off
(`deploy/cluster_qwen3.8-27b_fp8_w1.example.json`; `--no-dflash` runs
the same recipe plain on the decode graph, `--no-dflash --mtp --mtp-depth 2`
as the MTP world). Without `decode_graph` on one Spark it runs on the eager
engine; on the graph worlds (the fabric, or one Spark with the decode
graph) the block proposal is recorded inside the graph step — see "On the
graph worlds" below. A greedy request's acceptance is the ordinary greedy
verify: the fed rows (the pending token plus the drafts) run through the
target, the accepted prefix commits, the rest rolls back, and the step
returns the tokens it decided (the accepted drafts and the verify's next
token), so the transcript equals a plain world's of the same verify width:
4/4 identical to an MTP depth-4 world (both verify five or more rows, the
streaming mma class); a 3-row MTP verify or a T=1 step differs within the
family's cross-dispatch near-tie class (`mma_from_rows` 5). A sampled
request's block is verified with the sampled MTP rule (each draft stands
with its exact probability under the request's temperature, top-k / top-p
and penalties; the residual sample ends the step): the drafts are point
masses, so the target distribution is preserved as for sampled MTP.
Requests with logprobs, a logit bias or a grammar run the plain step. The throughput line carries the drafter's per-position
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

### On the graph worlds (2026-10-04, #92)

`GraphEngineAdapter` hosts the block proposal beside the MTP draft chain:
the verify is the ordinary 8-row recorded step (the sampled device pick
with no proposal behind the drafts — the point-mass rule; the sampling
verdict's row limit `kSampleVerdictRows` went 6 → 8), and after the commit
one recorded block forward off the device verdict replaces the draft
picks (`Qwen35Model::session_graph_capture_block_draft`, batched per family):
the anchor and the block's positions come off the verdict and the committed
device position (`dflash2_stage_block`), the drafter's layers run with its
heads, kv heads and MLP rows sharded across the ranks and two boundary folds
a layer, the mask rows go through the vocab-sharded head, each rank's slice
top-16 is staged as 6-bit digits and merged through one boundary fold
(`dflash2_topk_stage` / `dflash2_topk_merge`, the pick's wire form) so every
rank walks the same proposal, and the drafts land in the slot's next feed
(`dflash2_block_feed`) and a pinned per-slot mirror the engine reads when
the replay settles. The block attention is the split-key form
(`dflash2_block_attn_split`: 32 ranges and a combine; the serial walk cost
~2 ms a layer at a 2K context). The eager draft (the slot's open, the
sampled fallback's re-draft) runs the same helpers.

The exactness gate is the same world's MTP depth-5 world (six verify rows,
the same GEMM dispatch class as the 8-row verify): greedy transcripts
identical 4/4 at worlds 1, 2 and 4; the 1-row plain world differs within
the family's row-count dispatch class (as at world 1 against a 3-row MTP
verify). World 1's graph engine equals the eager engine 4/4 and runs the
same tokens faster (chat 70.9 against 95.2 ms a token).

Measured 2026-10-04 (`benchmarks/results/2026-10-04-qwen3.8-27b/raw/drafter-tp`),
C1, the arena harness (`llama-benchy --pp 2048 --tg 128`, the checkpoint's
sampling) and greedy `timed_load` by class (prose / code / json / math / chat):

| world | recipe | benchy tg128 | ms/pass | greedy C1 tok/s |
|---|---|---:|---:|---|
| 1 | drafter, graph engine | 16.4 | 176 | 17.9 / 28.6 / 40.4 / 32.5 / 17.1 |
| 2 | MTP depth 3 (the template) | 32.2 | 84 | 30.4 / 39.8 / 45.7 / 42.0 / 33.4 |
| 2 | drafter (bf12, sharded) | 31.3 | 94 | 32.1 / 50.3 / 71.0 / 57.6 / 30.3 |
| 4 | MTP depth 3 (the template) | 55.0 | 49 | 54.7 / 65.1 / 74.6 / 68.6 / 47.8 |
| 4 | drafter (bf12, sharded) | 50.1 | 56 | 52.4 / 82.5 / 115.9 / 97.1 / 50.2 |

The public cards are 58.48 (four nodes, vLLM TP=4 MTP 3) and 46.55 (two
nodes, SGLang TP=2 with a DSpark block drafter). The drafter leads on code,
JSON and math by 27–55 % at four nodes and ties prose and chat; on the
arena's book text the MTP template's 4-row verify keeps a 7 ms shorter
pass, so MTP stays the two- and four-node template and the drafter is its
mode. The nsys node trace of rank 0 at four nodes puts the target's fp8
GEMV at 35–38 ms a step for both recipes (6.9 GB a rank at 187 GB/s — the
kernel at four-node shard widths, split-K already on), the 128–140 bus
folds at 7–10 ms, the sharded drafter's bf12 GEMMs at 3.8 ms: the GEMV's
rate at these shard widths is the shared lever toward the cards.

## Recorded GLM-5.3 result

On 2026-09-03 at TP=4, greedy depth-1 MTP accepted 88.7% of drafts on the
recorded coherent-text workload. It produced 1.89 tokens per 42.4 ms step:
22.45 ms/token, compared with 31.3 ms/token for plain decode. The draft
layer added about 7.3 GiB per rank. These figures describe that checkpoint
and workload; acceptance fell on the post-EOS text generated with
`--no-eos`.
