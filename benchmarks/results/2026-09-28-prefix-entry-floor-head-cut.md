# The prefix cache's entry floor and head cut, and arenas sized to the node

Date: 2026-09-28
Baseline: master `743f542` (the tree after PR #64).

## The field report

A user serving `nvidia/Qwen3.8-Flash-Next-NVFP4` on one Spark reported that
at the recipe's 1.5 GiB arena (13 slots of 110.3 MiB) an opencode session
re-prefilled its whole context every five to seven requests, minutes each
time, and that 55 slots made it rare. They shared a scrubbed rank-0 log and
op stream: 2026-09-27 12:20 to 09-28 03:01, 7,362 requests, 682.5 M prompt
tokens, 1.18 % computed. The files stay private; the numbers below are the
aggregates.

- The evening session was one conversation of ~400 new tokens per turn
  growing to ~180K, then compacted by the client: a compaction prompt of
  80–135K sharing 3 tokens with anything cached (cold, 54–97 s), then a
  20–38K post-compaction turn sharing its first ~18K (the system prompt and
  tool schemas) with the old conversation and finding no entry there (cold,
  16–53 s). Twenty-four cycles, 47 minutes of miss prefill in 15 hours.
- 932 of 935 misses named a changed prompt; 2 were evictions.
- Between 12:22 and 12:54 each 66–68K turn was followed by three 44–49-token
  requests answering in two tokens, each taking about 1.5 arena entries.

## The replay

The op stream replayed under an N-slot LRU arena (attach, snapshot, rolling,
hop, close, drop and retire events in their recorded order; N = 55
reproduces the run: 0 misses on attached entries, the recorded 12,203
evictions).

| Policy | 8 slots | 13 slots | 27 slots | 55 slots |
|---|---:|---:|---:|---:|
| LRU as shipped, misses on prompts ≥ 16K | 246 | 242 | 227 | 0 |
| no entries for prompts under 1,024 tokens | 149 | 2 | 0 | 0 |
| superseded-first eviction alone | 242 | 240 | 219 | 0 |
| both | 53 | 0 | 0 | 0 |

Every would-be miss at 13 and 27 slots sits in the 12:22–12:54 loop; the
evening session alone would have missed twice at 13 slots. The slot count
was never the lever. A 45-token probe's entry and a 180K conversation's
entry cost the same slot, and LRU cannot tell them apart.

## What changed

- `engine.prefix_min_tokens` (`--prefix-min-tokens`, default 1024): no
  snapshot of any kind below that position. A shorter prompt attaches to a
  matching entry but never takes a slot. Applied to the prefill cut, the
  head and body cuts, the rolling and hop snapshots and the retire-time
  entry (`Scheduler::plan_prefix`, `rolling_snapshots`, `retire`).
- `engine.prefix_head_snapshots` (`--prefix-head-snapshots`, default on):
  a cold prefill also keeps the cut at the aligned image of the prompt's
  first structural boundary past its start, on the same walk as the deepest
  cut, after it and before the body cut in slot priority, under the same
  floor, only on families whose prefill takes extra cuts. In the field log
  it would have served 24 post-compaction turns and every new conversation
  their 18K shared head.
- Both ride the settings and warm records (`pmin`, `phead`) and the config
  line's digest; a record without them decodes to what such a rank 0 ran
  (floor 0, no head cut). `/v1/metrics` reports `min_tokens`, `head_cuts`
  and `head_snapshots` under `prefix_cache`.
- Superseded-first eviction was not adopted: alone it does nothing for this
  traffic and at 55 slots it loses the changed-question fallback entries.

## Tests

- `scheduler_test`: `entryFloorKeepsShortPromptsOutOfTheArena` (no prefill
  cut, rolling or close entry under the floor; the cached run above it),
  `headCutServesTheNextConversationUnderTheSameSystemPrompt` (attach at the
  head with the cut on, cold with it off, no head under a higher floor),
  `headCutYieldsToTheDeepestCutWhenSlotsAreScarce` (one slot: the deepest
  cut, the head skipped and counted), `headCutThroughChunkedPrefill` (the
  resumable path takes and serves the head cut beside the body cut).
- `cluster_config_test`: both keys parsed, both ranges enforced.
- `fabric_serve_test`: the warm and settings records round-trip the policy;
  records without the keys decode to off; bad values are rejected.
- `scripts/serve_api_check.py`'s cached-tokens probe now runs past the floor.
- `tests/fixtures/cluster.resolved.json` follows the GLM-5.3-Flash four-node
  example's new arena budget (`site_env_test` compares the two).
- Build `cmake --build build-ci -j -- -k` clean, no warnings in the touched
  files. `ctest` host, unit, fixture and python labels: 38/38. The
  synthetic-weight GPU engine and TP tests of every family (no checkpoint):
  11/12 beside the production fabric — `dsv41_tp_test`'s group-prefill
  subtest passed its bitwise checks and then failed to start a further
  loopback bus world ("tp bus world (stream folds) failed to start"), an
  environment failure on a node whose RDMA fabric was serving GLM-5.3-Flash;
  to be re-run on an idle node with the checkpoint-labeled suites
  (`qwen_decode_test`, `glm_tp_test`) and a `serve_agentic_streams.py` pass
  before this ships.

## The recipes

Per-rank memory plans of every checked-in recipe were re-run
(`dgpp-serve --memory-plan`, build `14fb039`). The arena now takes the memory
the node has left: the plan's total plus its 4 GiB headroom against the
115.13 GiB an idle Spark reported free at the last production boot, minus a
1 GiB margin, rounded down to whole GiB.

| Recipe | Plan total before, GiB | Arena before → after, GiB | Slots before → after |
|---|---:|---|---|
| DeepSeek-V4.1-Flash w4 | 78.34 | 1.5 → 14 | 441 → 4096 (the cap) |
| GLM-4.7 w4 | 84.10 | 1.5 | 4096 (the cap) |
| GLM-5.3-Flash NVFP4 w2 | 107.01 | 1.5 → 4.5 | 21 → 65 |
| GLM-5.3-Flash NVFP4 w4 | 95.85 | 8 → 22 | 232 → 639 |
| GLM-5.3 int4/int8 w4 | 110.11 | 1.0 | 4096 (the cap) |
| Qwen FP8 w2 | 105.57 | 1.5 → 6 | 27 → 111 |
| Qwen FP8 w4 | 60.09 | 1.5 → 50 | 55 → 1846 |
| Qwen NVFP4 w1 (site 65,536 / example 262,144 pool) | 87.62 / 94.15 | 1.5 → 3 | 13 → 27 |
| Qwen NVFP4 w2 | 52.10 | 1.5 → 58 | 27 → 1074 |
| Qwen NVFP4 w2 YaRN 512K | 58.29 | 1.5 → 40 | 27 → 741 |
| Qwen NVFP4 RadixArk w1 / w2 | not planned (no checkpoint here) | 1.5 → 3 / 32 | 13 → 27 / est. 593 |
| MiMo-V2.6-Flash w2 / w4 | not planned (no checkpoint here) | 1.5 | unchanged |
| GLM-5.3-Flash FP8 w4 (site file only) | 126.87 | 8 | unchanged: the plan exceeds the node as it stands |

The one-node Qwen recipes stop at 3 GiB on purpose. In the field log fresh
text prefilled at 350–480 tok/s against 1,300–1,600 tok/s for text the server
had already processed, the rate falling inside one request exactly where the
already-seen head ended; the only content-dependent stage of the prefill is
the mmap'ed 47.7 GiB n-gram table (16 random 160-byte rows per token,
`MADV_RANDOM`, served from the page cache or the NVMe), and the reporter's
plan left about 13–17 GiB for the OS, Docker and that cache. The memory the
arena does not take is what keeps it warm. The YaRN recipe stops at 40 so its
documented 1,114,112-token variant (about 9.3 GiB more per rank) still fits.
