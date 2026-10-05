# Qwen3.6-35B-A3B

Of the models measured on this fleet with this engine, the fastest on one box and on two (the
table in [models/README.md](../README.md)). Every figure below is quoted from a record in this
repository; the record is named beside it. Registry entry: `qwen3.6-35b-a3b` in
[`sixlabs/registry/models.json`](../../sixlabs/registry/models.json).

## What it is

- Family `qwen3_5`: 40 layers (30 Gated DeltaNet + 10 full attention) and one draft layer,
  hidden 2048, a routed MoE of 256 experts, top 8 (the bind check's own lines, in the
  [bring-up record](one-box/results-2026-10-04-dgxone.md), section 2).
- A thinking model. Its vision tower is bound and not served; an image part is answered with
  HTTP 400.
- The checkpoint carries a draft head, which the templates run at depth 2.

| Checkpoint | Revision | Format | Tensors | Registry status |
|---|---|---|---|---|
| [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) (the default) | `1355db6a052410cfd62085d94b58866fd0f2c3c5` | nvfp4-modelopt | 124,135 | `working` |
| [`Qwen/Qwen3.6-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) | `95a723d08a9490559dae23d0cff1d9466213d989` | fp8-block | 63,863 | `compiled`: fails the gate, not served |
| `RedHatAI/Qwen3.6-35B-A3B-NVFP4` | — | nvfp4-compressed-tensors | — | `refused` |

## Status

In the registry's words, `working` means: "Verified on a GPU on this fleet: bound, passed the
forward-check gate, served, measured. Has a record."

- **One box, NVFP4: verified** on 2026-10-04, DGXone, engine commit `89955e0`, record
  [bring-up](one-box/results-2026-10-04-dgxone.md).
- **Two boxes: measured, not verified.** The registry's note: "no gate has been run for this
  model on the merged engine, on one box or two."
- The one-box gate has not been re-run since the upstream merge of 2026-10-04; the registry's
  `verified.engine_commit` stays at `89955e0` until it has
  ([upstream-merge record](../../sixlabs/bench/results/upstream_merge_2026-10-04.md), section 4).

## One box

Template [`deploy/cluster_qwen3.6-35b-a3b_nvfp4_w1.example.json`](../../deploy/cluster_qwen3.6-35b-a3b_nvfp4_w1.example.json):
8 seats, a 262,144-token pool, FP8 dense form, draft depth 2, and (since 2026-10-04) the tuned
prompt-reading settings, which need a build with `cecb0cd` or later.

### Load and memory

| | Figure | Conditions | Record |
|---|---|---|---|
| Load, cold (no resident image) | 47.5 s | quiet box; the launcher's own count is 48 s. An earlier cold boot beside other CPU jobs took 36.4 s | [bring-up](one-box/results-2026-10-04-dgxone.md), C.1 |
| Load, warm (resident image) | 9.0 s | the launcher's count is 10 s | the same |
| Memory planned | 36.00 GiB (30.05 device + 5.96 pinned) + 4.00 headroom | 8 seats, 262,144-token pool | the same |
| Memory measured | 39,478 MiB = 38.55 GiB | 2.55 GiB over the plan's total | the same |

"Cold" means no resident image; the page cache was warm.

### Speed

Output tok/s, all streams together. One run each.

| Benchmark | 1 stream | 2 | 4 | 8 | Conditions | Record |
|---|---|---|---|---|---|---|
| `bench_decode.py` | 97.7 prose, 120.1 code | 128.8 | 194.6 | 271.3 | build `89955e0`, automatic prompt-reading settings, first run after a cold boot; draft acceptance 74.5 % | [bring-up](one-box/results-2026-10-04-dgxone.md), C.2 |
| ShareGPT, 64 prompts a level | 96.8 | 134.3 | 180.8 | 217.9 | the same boot, 04:19 | the same, C.3 |
| ShareGPT | 96.72 | 135.37 | 182.37 | 220.11 | build `29f904b`, 11:54, thinking on, automatic settings | [against Atlas](one-box/dgxone_35b_6ixcpp_vs_atlas_2026-10-04.md) |
| ShareGPT, default settings | 96.39 | 132.53 | 171.13 | 216.27 | build `17fbf92` | [prefill](one-box/dgxone_35b_prefill_2026-10-04.md) |
| ShareGPT, tuned settings (the template now) | 93.19 | 128.47 | 176.67 | 222.74 | build `17fbf92` | the same |

The bring-up record also has DecodeBench (107.4 / 143.6 / 191.2 / 257.2) and llama-benchy
(generation 106.8 / 131.9 / 129.8 / 116.8 at depth 0).

tool-eval-bench: short **93** (13 pass, 2 partial, 0 fail), hard **89** (72 pass, 12 partial,
4 fail; 156/176), the same score on two runs the same day (bring-up C.3 and the Atlas record).

### The prompt-reading settings

From [prefill](one-box/dgxone_35b_prefill_2026-10-04.md), default against tuned, same build,
same answers (8 of 8 texts identical):

| | default | tuned |
|---|---|---|
| burst of eight 2,022-token prompts, all finished | 34.9 s | 6.73 s |
| 4 chat streams beside 2 long prompts: engine output tok/s | 13.1 | 95.4 |
| the same: a chat stream's first token, median | 41.7 s | 0.47 s |
| readers alone: prompt tokens read a second | 1,185 | 3,411 |

Long prompts under that load still wait about half a minute (31.7 s to a first token).

### Against Atlas and vLLM, measured on this fleet

Atlas ran the **same file** (same snapshot directory in both engines' logs), DGXone, the same
harness, thinking on on both sides
([against Atlas](one-box/dgxone_35b_6ixcpp_vs_atlas_2026-10-04.md)):

| ShareGPT output tok/s | 1 stream | 2 | 4 | 8 |
|---|---|---|---|---|
| 6ix.cpp | 96.72 | 135.37 | 182.37 | 220.11 |
| Atlas | 84.76 | 112.21 | 108.89 | 127.12 |
| 6ix.cpp / Atlas | 1.14× | 1.21× | 1.67× | 1.73× |

| | 6ix.cpp | Atlas |
|---|---|---|
| Load | 10 s warm; 47.5 s without a resident image | 49.1 s (it reads the checkpoint on every start) |
| Device memory | 39,478 MiB | 73,861 MiB |
| Context / seats | 262,144-token pool, 8 seats | 32,768-token context, batch 8 |
| tool-eval short | 93 | 93 |
| tool-eval hard | 89 | not run (stopped at 24 of 88 on the owner's instruction) |

How Atlas was run: its published recipe through sparkrun, with five overrides the record lists
(bind address, port, memory fraction 0.62 instead of 0.88, 32,768 context instead of 262,144,
thinking on instead of off). The harness's time-to-first-token columns are not comparable between
the two engines (the record's note 1), so they are not repeated here.

vLLM ran a **different quantization** (`Qwen/Qwen3.6-35B-A3B-FP8`, the fleet's slot
`qwen3.6-35b@dgxone`), DGXone, one run ([bring-up](one-box/results-2026-10-04-dgxone.md), E and G):

| | 6ix.cpp, NVFP4 | vLLM, FP8 |
|---|---|---|
| `bench_decode.py`, 1 stream | 97.7 prose, 120.1 code | 50.1, 50.1 |
| `bench_decode.py`, 2 / 4 / 8 streams | 128.8 / 194.6 / 271.3 | 62.6 / 105.9 / 190.0 |
| Load | 47.5 s | 404 s |
| Device memory | 38.55 GiB | 44.51 GiB |
| llama-benchy prompt processing tok/s at 1 / 2 / 4 / 8, depth 0 (6ix.cpp at automatic settings) | 1977 / 1402 / 944 / 598 | 5651 / 4802 / 4260 / 5024 |
| tool-eval short / hard | 93 / 89 (thinking on) | 90 / 85 (thinking off by default) |

vLLM's ShareGPT was not run (the endpoint is keyed and the script of the day could not send a
key). The two columns differ in weights, context, seats and thinking default.

**Published by others, not measured here:** Atlas, 116.5 tok/s at one stream with one draft token
and thinking off (its recipe's figure, quoted in the registry and the Atlas record).

### What "verified" rests on

From the [bring-up record](one-box/results-2026-10-04-dgxone.md), section 4. Read it before
quoting the word.

- The gate **as first written** (mean ≤ 0.06 nat, max ≤ 0.5) failed on 3 of 4 runs.
- The gate was restated twice that night, on instruction, after the 80B (the model known to
  work) failed it too. Under either restatement the NVFP4 checkpoint passes 4 of 4. It did not
  need the two exclusions the second restatement added.
- The evidence the record relies on is neither bound: one step per layer on the engine's own
  inputs, all 40 layers and the head agree with the numpy reference to about twice the BF16
  storage floor, where a deliberately wrong convention shows 170 to 605 floors.
- Greedy output with the draft head equals plain decode: 18 of 18 rows, and 30 of 30 rows
  (6,952 generated tokens) when reasoning is compared too.
- The raw verdicts of all eight comparisons are in
  [gate-final-8-runs.txt](one-box/gate-final-8-runs.txt).

## Two boxes

Template [`deploy/cluster_qwen3.6-35b-a3b_nvfp4_w2.example.json`](../../deploy/cluster_qwen3.6-35b-a3b_nvfp4_w2.example.json):
the one-box settings without `decode_passes_per_prefill` and `prefill_order` (single-rank only).
Record: [two-box 80B + 35B](../shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md) (DGXone +
DGXtwo, 2026-10-04, build `2ad1c4f`).

### Load and memory

| | Figure |
|---|---|
| First load | 60 s |
| Warm load | 6 s |
| Memory planned, each box | 22.76 GiB (17.28 device + 5.48 pinned) |
| Memory measured, each box | 22,950 MiB |

### Speed

ShareGPT, 64 prompts a level, all 64 successful, thinking on, one run:

| Streams | One box (morning build), tok/s | Two boxes, tok/s | Gain | Two boxes: first token median / P99, ms | per token median / P99, ms |
|---|---|---|---|---|---|
| 1 | 96.72 | 154.43 | ×1.60 | 102 / 205 | 6.01 / 9.46 |
| 2 | 135.37 | 214.11 | ×1.58 | 112 / 205 | 8.44 / 10.19 |
| 4 | 182.37 | 300.46 | ×1.65 | 129 / 257 | 11.60 / 14.76 |
| 8 | 220.11 | 367.57 | ×1.67 | 142 / 358 | 19.14 / 24.32 |

The one-box column is the morning's run on build `29f904b`, before the upstream merge changed
this family's kernels: the same checkpoint and harness, **not the same engine version**. A
same-build one-box ShareGPT has not been run.

No gate, no tool-eval and no group check on two boxes. Spot answers were right.

### Other two-box records

| Record | Result |
|---|---|
| [HEM on the same boxes](../shared/dgxone_dgxtwo_hem_sharing_2026-10-04.md) | 145.0 tok/s with no scoring; continuous scoring on DGXtwo's GPU leaves 41 % of that, on both GPUs 30 % |
| [Loadout switching](../shared/dgxone_dgxtwo_loadout_switch_2026-10-04.md) | with the 80B as one loadout: answering 29–30 s after a switch from nothing loaded |

## Did not work, and why

- **`Qwen/Qwen3.6-35B-A3B-FP8` fails the gate and was not booted or measured.** On the code text
  the engine disagrees with the reference's greedy token at row 7, where the reference's margin
  is 1.3141 nat; prose passes. Every layer passes the one-step check, and the reference makes the
  same flip against itself at BF16 arithmetic, so the record sees no wrong tensor; the gate was
  not changed again ([bring-up](one-box/results-2026-10-04-dgxone.md), 4.11 and F). This is the
  like-for-like checkpoint for the vLLM comparison, and that column is empty.
- **`RedHatAI/Qwen3.6-35B-A3B-NVFP4` is refused by name** at config parse (`group_1.weights`
  missing). The copy in DGXone's cache was an incomplete download, and was removed on 2026-10-04
  (registry).
- **Build `29f904b` refused the explicit prompt-reading settings for this family** and exited in
  60 ms ([against Atlas](one-box/dgxone_35b_6ixcpp_vs_atlas_2026-10-04.md), section 2). Accepted
  since `cecb0cd`.
- **With the automatic settings, time to a first response grew steeply with concurrency**: 1.0 /
  2.6 / 7.4 / 23.5 s at 1 / 2 / 4 / 8 in llama-benchy
  ([bring-up](one-box/results-2026-10-04-dgxone.md), C.3). The tuned settings are the answer on
  record.
- **At 16K of context with 4 and 8 concurrent requests, generation fell to 15.6 / 13.7 tok/s in
  total**: the 262,144-token pool was 97 % full and requests waited for blocks (same section).
- **Atlas's first start at a memory fraction of 0.5 failed** ("No memory left for KV cache");
  0.62 is the measured run (Atlas record, section 3).

## Open

- The one-box gate on the merged engine, and a same-build one-box ShareGPT.
- The gate, tool-eval and `group_check` on two boxes.
- The two quantizations (NVFP4 and FP8) differ in the reference by more than the port's notes
  expected (mean 0.152 and 0.221 nat on the two texts); left open in the bring-up record, 4.10.
- `sixlabs/ports/qwen3.6-35b-a3b/gpu-steps.md` still describes the gate's original bounds.

## Templates

| Use | Template |
|---|---|
| One box | [`cluster_qwen3.6-35b-a3b_nvfp4_w1.example.json`](../../deploy/cluster_qwen3.6-35b-a3b_nvfp4_w1.example.json) |
| Two boxes | [`cluster_qwen3.6-35b-a3b_nvfp4_w2.example.json`](../../deploy/cluster_qwen3.6-35b-a3b_nvfp4_w2.example.json) |
| Do not use: FP8 checkpoint, fails the gate | [`cluster_qwen3.6-35b-a3b_fp8_w1.example.json`](../../deploy/cluster_qwen3.6-35b-a3b_fp8_w1.example.json) |

## Records in this folder

- `one-box/`: [bring-up](one-box/results-2026-10-04-dgxone.md) with
  [gate-final-8-runs.txt](one-box/gate-final-8-runs.txt),
  [against Atlas](one-box/dgxone_35b_6ixcpp_vs_atlas_2026-10-04.md),
  [prefill](one-box/dgxone_35b_prefill_2026-10-04.md) with its data folder.
- Shared with the 80B, in [`models/shared/`](../shared/): the
  [two-box record](../shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md) and its logs,
  [HEM sharing](../shared/dgxone_dgxtwo_hem_sharing_2026-10-04.md),
  [loadout switching](../shared/dgxone_dgxtwo_loadout_switch_2026-10-04.md).
- Bring-up steps and gate inputs: `sixlabs/ports/qwen3.6-35b-a3b/`
  ([gpu-steps.md](../../sixlabs/ports/qwen3.6-35b-a3b/gpu-steps.md)).
