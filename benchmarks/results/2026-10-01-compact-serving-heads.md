# Compact serving heads across model families, 2026-10-01

Serving now bounds FP32 vocabulary logits by the decode and MTP verification
capacity across Qwen, GLM-4.7, full GLM-5.3, GLM-5.3-Flash, MiMo and
DeepSeek-V4.1. Prefill projects each request's final row into that buffer.
Hidden states retain their original layout for cache handling and MTP.
Diagnostic constructors still allocate every row; an all-row call that
exceeds compact capacity fails before writing the head.

This extends PR #67 (`5b4ec35c`) on master `ac8e729`, reconciling the
original change with master's newer packed Qwen head. Qwen FP8 and AutoRound
packed heads preserve their
original dispatch and accumulation order. BF16 heads may switch from a
multi-row GEMM to a single-row projection. GLM Flash already selected
prefill tails, so its benefit here is smaller storage and accurate planning.

## Memory benefit

The allocation reduction is `(prefill_rows - logits_rows) * local_vocab * 4`
bytes. Logits capacity is at least eight rows and grows for the configured
decode width or request count. At prefill 2048, logits capacity 8 and TP=4:

| Family | Vocabulary | Saved per rank | Saved across four ranks |
| --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next | 248320 | 483.11 MiB | 1.887 GiB |
| GLM-4.7 | 151552 | 294.84 MiB | 1.152 GiB |
| GLM-5.3 / GLM-5.3-Flash | 154880 | 301.32 MiB | 1.177 GiB |
| DeepSeek-V4.1 | 129280 | 251.51 MiB | 0.982 GiB |

MiMo uses the same allocation rule with its configured vocabulary. Tests
check the memory-plan delta against the exact allocation change. No weight,
KV-cache, or hidden-state precision is reduced. This campaign does not claim
an end-to-end throughput improvement; its real-model timings include lazy
weight loading and are not performance measurements.

## Numerical accuracy

The new `compact_head_test` builds full and compact instances of each
non-Qwen family from the same random checkpoint. It checks complete
vocabulary logits against FP64 dots over identical BF16 weights and normalized
hidden states. Single prompts span 1–257 tokens, including dispatch and chunk
boundaries. Nonuniform groups span up to 255 total tokens. Fixed-token decode,
MTP draft outputs and restored-prefix repeats remain exact.

Final maximum relative L2 errors against FP64:

| Fixture | Full head | Compact head | Maximum token log-probability delta |
| --- | ---: | ---: | ---: |
| GLM-4.7 | 2.39e-7 | 5.93e-8 | 3.06e-6 |
| Full GLM-5.3 | 2.47e-7 | 5.89e-8 | 2.45e-6 |
| MiMo | 2.32e-7 | 5.71e-8 | 2.90e-6 |
| DeepSeek exact | 2.76e-7 | 2.76e-7 | 4.44e-16 |
| DeepSeek bounded | 2.84e-7 | 2.84e-7 | 4.44e-16 |
| GLM Flash | 6.43e-8 | 6.43e-8 | 4.44e-16 |

The GEMM-interface oracle additionally covers hidden widths
2560/4096/5120/6144/7168/8192 and original row counts 5/129/2048, with ordinary
GEMV, MMA and lossless BF12 companions: **54 shape/mode pairs passed**.
The ordinary/BF12 full head reached relative L2 1.41e-5; compact reached
1.16e-7. MMA reached 8.61e-6 in both paths. Both paths must stay below 2e-5,
and compact error may not exceed full error by more than 1e-6. The 2e-5
absolute budget accommodates the measured full-head accumulation error at
long hidden widths. Output sentinels check the compact allocation boundary.
These results support numerical soundness; they do not imply a model-quality
improvement from the smaller arithmetic error.

Qwen's FP8 and packed prefill tests retain bitwise comparisons. Its BF16
prefill comparison allows bounded numerical error while retaining exact hidden
and fixed-token continuation checks. Existing engine gates run compact
allocation through graph replay, grouped requests, slot retirement and MTP;
those semantic gates were not relaxed.

## Real GLM-4.7 comparison

`serving_head_check` ran on four GB10 nodes with
`nvidia/GLM-4.7-NVFP4`, resident weights, BF16 KV and `DGPP_BF12=both`.
Both modes use the same executable, with `--full` or `--compact`, no MTP,
one request, 1024 prefill rows and 2048 cache tokens. Each cold prefill scores
one fixed next token. The corpus concatenates the repository's quick, hard
and memorized teacher texts. There are 32 trials at each length
1/4/8/17/64/129/257/1024, totaling 256 scored tails.

| Measure | Result |
| --- | ---: |
| Full mean NLL | 4.1212312863 nats/token |
| Compact mean NLL | 4.1212344127 nats/token |
| Mean NLL change | +0.0000031263 nats/token |
| Mean absolute target log-probability change | 0.0000062578 nats |
| Maximum absolute target log-probability change | 0.0000598842 nats |
| Changed global argmax tokens | 0 / 256 |
| Correct top-1 targets | 112 / 256 in both modes |
| Matching hidden digests | 1024 / 1024 across modes; all ranks agree |

The gates were absolute mean NLL change <= 0.0001 nats and no target change
above 0.001 nats. Both passed. These measurements score selected next tokens
from excerpts, including very short contexts; their NLL is not a whole-corpus
perplexity benchmark. This is real-checkpoint evidence for GLM-4.7, plus
synthetic and kernel evidence for the other families. Full task benchmarks
were not rerun for every supported checkpoint.

Reproduce the stored comparison and hidden-state checks:

```bash
python3 benchmarks/results/2026-10-01-compact-serving-heads/analyze.py
python3 scripts/fabric_logprob.py \
  benchmarks/results/2026-10-01-compact-serving-heads/glm4-compact \
  benchmarks/results/2026-10-01-compact-serving-heads/glm4-full --prefill \
  --max-mean-nll-delta 0.0001 --big-delta 0.001 --max-big-delta-rate 0
```

To run new measurements, concatenate `benchmarks/teacher_text.txt`,
`benchmarks/teacher_text_hard.txt` and `benchmarks/teacher_text_memorized.txt`
into `/tmp/dgpp-head-teacher.txt`, then use the same deployment, binary and
settings for both modes:

```bash
DGPP_BF12=both scripts/fabric_run.sh \
  --app build-review/serving_head_check \
  --stage-file /tmp/dgpp-head-teacher.txt --fetch-logs \
  --log-dir /tmp/head-full --timeout 1200 -- \
  --model nvidia/GLM-4.7-NVFP4 \
  --teacher-file /tmp/dgpp-head-teacher.txt --full --trials 256
# Repeat with --log-dir /tmp/head-compact and --compact.
```

Select the site's four-node configuration with `DGPP_CLUSTER_CONFIG` and its
site settings with `DGPP_ENV_FILE`. The saved corpus SHA256 and score summary
are in [glm4-summary.json](2026-10-01-compact-serving-heads/glm4-summary.json).
The retained per-rank logs contain all score records, hidden digests and
completion records, excluding loading diagnostics.

## DeepSeek short-tail fix

The boundary tests exposed an existing bounded-prefill error. A 257-token
prompt with a 256-row chunk budget ends with one new token but replays a
16-row retained window. Subtracting replay rows from chunk rows gave a
negative head offset. The MTP hidden publication also indexed metadata
before the new chunk's beginning. Compute Sanitizer confirmed the invalid
writes with full storage as well as compact storage.

The replay stays within scratch storage, its final output is published at
the original prompt-tail location, and its metadata is restaged for MTP.
Both exact and bounded tests now pass, including the short final chunk,
draft and continuation checks. The final family run under Compute Sanitizer
reported **zero errors and zero leaked allocations**.

## Serving and build validation

The Release build uses `DGPP_WERROR=ON`. All targets built successfully.
All 27 host tests passed. The documentation-link test initially ran before
this report was written; its rerun passed once the linked file existed.
Focused CUDA checks passed: Qwen prefill, full-storage override and packed
prefill; Qwen, GLM-4.7, full GLM-5.3, MiMo and DeepSeek graph-engine suites;
compact family tests with and without the storage override; the 54-case FP64
oracle; and the family test under Compute Sanitizer.

A candidate GLM Flash server ran on all four nodes using the original site
configuration: four slots, MTP depth 1, CUDA decode graphs, 786432 KV tokens
and an 8-GiB prefix-cache budget. The API checks passed for streaming,
multiple choices, stop strings, logit bias and usage accounting. A repeated
prompt reused 2152 tokens. Four concurrent cold prompts of
916/1116/1316/1516 tokens each completed 32 output tokens. Metrics recorded
194 graph replays, including 18 four-slot replays, 249 drafts and 162 accepted
draft tokens over the complete smoke session, with no engine failure.

Shutdown produced the same operation-stream MD5 on every rank:
`25b6d99d34a8e31e30fed89dd02c6935`. The original `dd58d6d` deployment was
restored with the original resolved settings. Health, a completion, no engine
failure, and matching original binary SHA256 on all four nodes were verified.
Candidate identities, API output, metrics, shutdown evidence and restoration
verification are retained in the [evidence directory](2026-10-01-compact-serving-heads/).
