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

`engine.mtp_depth` accepts 1–7 and defaults to 1 (DeepSeek-V4.1 defaults
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

## The DFlash2 block drafter (Qwen3.5 family)

DFlash2 (`src/models/qwen/dflash2.hpp`) replaces the MTP draft with an
external block drafter checkpoint (`z-lab/Qwen3.8-27B-DFlash2`): five
bidirectional Qwen3 layers that turn the target's tapped hidden states
(layers `[5,19,33,47,61]` through a fused `fc` + RMSNorm) and a block of
masked slots into seven drafts in one non-autoregressive pass, plus a
low-rank pairwise selector that walks top-16 candidates per slot. The
drafter shares the target's embedding and lm head and keeps its K/V in
five extra planes of the main pool, so prefix caching and rollback ride
the existing protocol.

Serve it with the drafter checkpoint in `engine.dflash_model` (or
`--dflash-model DIR_OR_ID`) and `engine.mtp` / `engine.decode_graph`
off — the drafter is the eager world-1 path (see
`deploy/cluster_qwen3.8-27b-fp8-dflash2_w1.example.json`;
`--no-dflash` runs the same recipe plain). Acceptance
is the ordinary greedy verify: the eight fed rows (pending token +
drafts) run through the target, the accepted prefix commits and the
rest rolls back, so the transcript stays exact. The throughput line
carries the drafter's per-position acceptance in the usual MTP group.
The lane's exit criterion (performance plan §7) is to beat the best
native-MTP configuration; until it does, the MTP recipe remains the
default.

## Recorded GLM-5.3 result

On 2026-09-03 at TP=4, greedy depth-1 MTP accepted 88.7% of drafts on the
recorded coherent-text workload. It produced 1.89 tokens per 42.4 ms step:
22.45 ms/token, compared with 31.3 ms/token for plain decode. The draft
layer added about 7.3 GiB per rank. These figures describe that checkpoint
and workload; acceptance fell on the post-EOS text generated with
`--no-eos`.
