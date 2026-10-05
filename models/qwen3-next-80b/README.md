# Qwen3-Next-80B-A3B Instruct

The first model 6ixLabs ported to this engine, and the one the fleet serves as
`qwen3-next-80b-6ix`. Every figure below is quoted from a record in this repository; the record
is named beside it. Registry entry: `qwen3-next-80b` in
[`sixlabs/registry/models.json`](../../sixlabs/registry/models.json).

## What it is

- Family `qwen3_next`: Gated DeltaNet layers and full attention with a routed MoE (512 experts,
  top 10), flat config. It runs as a dialect of DGPP's `qwen3_5` stack
  ([6IXSERVE.md](../../6IXSERVE.md)).
- An Instruct checkpoint: it never reasons. Tool calls in the JSON form of its chat template.
- The checkpoint carries a draft head (`mtp.*`, 1,553 tensors), which the templates run at depth 2.

| Checkpoint | Revision | Format | Tensors | Registry status |
|---|---|---|---|---|
| [`nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`](https://huggingface.co/nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4) | `8fb2682f136cf94d932a498f18cb1e428832a912` | nvfp4-modelopt | 297,728 | `working` |

## Status

In the registry's words, `working` means: "Verified on a GPU on this fleet: bound, passed the
forward-check gate, served, measured. Has a record."

- **One box: verified** on 2026-10-03, DGXone, engine commit `f288d60`, record
  [6IXSERVE.md](../../6IXSERVE.md).
- **Two boxes: measured, not verified.** The registry's note: "the forward check has no two-rank
  mode and upstream's qwen35_tp_test has no MoE fixture."
- No gate has been run on any build that contains the upstream merge of 2026-10-04
  ([upstream-merge record](../../sixlabs/bench/results/upstream_merge_2026-10-04.md)).

## One box

Template [`deploy/cluster_qwen3-next-80b_nvfp4_w1.example.json`](../../deploy/cluster_qwen3-next-80b_nvfp4_w1.example.json):
8 seats, a 524,288-token pool, FP8 dense projections and head at load, draft depth 2, the tuned
prompt-reading settings.

### Load and memory

| | Figure | Conditions | Record |
|---|---|---|---|
| Load, cold | about 80 s | one Spark, 2026-10-03, 262,144-token pool | [6IXSERVE.md](../../6IXSERVE.md) |
| Load, warm (resident image) | 12 s | the same | [6IXSERVE.md](../../6IXSERVE.md) |
| Memory planned | 57.9 GiB | 262,144-token pool, 8 seats | [6IXSERVE.md](../../6IXSERVE.md) |
| Memory measured | 70,540 MiB | 524,288-token pool, build `2ad1c4f` | [two-box record](../shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md), section 1 |

The planned and measured figures are for different pool sizes: they are not a plan against its
own measurement.

### Speed

Output tok/s, all streams together.

| Benchmark | 1 stream | 2 | 4 | 8 | Conditions | Record |
|---|---|---|---|---|---|---|
| `bench_decode.py` | 82.2 prose, 103.6 code | 118.5 | 185.4 | 253.4 | DGXtwo alone on the GPU, 2026-10-04, build `5c20f70`, 524,288-token pool; draft acceptance 81.1 %; one run | [dgxtwo_clean](one-box/dgxtwo_clean_2026-10-04.md) |
| not named in the record | 81 prose – 102 code | 116 | 178 | 256 | one Spark, 2026-10-03, 262,144-token pool (3 streams 147, 6 streams 223) | [6IXSERVE.md](../../6IXSERVE.md) |
| ShareGPT, 64 prompts a level | 76.73 | 97.52 | 127.69 | 153.95 | build `2ad1c4f`, the two-box template's settings with `world_size` 1; one run | [two-box record](../shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md), section 4 |
| ShareGPT, production one-box template | 77.78 | 104.02 | not measured | not measured | stopped on the owner's instruction after two levels | the same |

ShareGPT latency for the first ShareGPT row, median / P99: first token 198 / 350, 205 / 506,
231 / 619, 287 / 615 ms; per token 12.36 / 17.00, 18.89 / 27.57, 29.12 / 58.83, 46.83 / 64.61 ms.

**Not results:** the ShareGPT run of 2026-10-03 on this model (85.54 / 122.77 / 164.14 / 205.88)
measured answers to empty prompts; see "What failed".

tool-eval-bench was not run on one box.

### Other one-box measurements

From [dgxtwo_clean](one-box/dgxtwo_clean_2026-10-04.md) (DGXtwo, nothing else on the GPU):

- Soak, 30 minutes: 7,973 requests, 0 failed, 0 shed; first token p50 158–167 ms, p99 323–330 ms.
- Four agentic streams of 8 turns: 28 of 28 follow-up turns attached to the cached prompt; 31 s.
- Judge sweep with ~4.5k-token prompts: 106–114 output tok/s at every level from 4 to 32
  concurrent. It does not scale, because this family read one prompt at a time in that build.

From [6IXSERVE.md](../../6IXSERVE.md) (2026-10-03): prompt reading about 2,700 tok/s to 33K
prompt tokens, about 1,400 at 134K–184K; a 249K-token prompt fits and all five planted facts in
it are retrieved.

### The scheduler settings, and where they came from

| Record | What it settled |
|---|---|
| [prefill_sweep](one-box/prefill_sweep_2026-10-04.md) | `prefill_budget_tokens` 1024, `decode_passes_per_prefill` 8, shortest prompt first: four answers beside two 17,400-token prompts keep 19.5 tok/s each instead of 2.0 |
| [prefill_group](one-box/prefill_group_2026-10-04.md) | `prefill_group: on` with `prefill_budget_per_reader: false`: a burst of eight prompts finishes in 7.36 s instead of 10.69 s |

Nothing in those records gets both sides: a 17k-token prompt arriving beside four streaming
answers still waits 25–30 s for its first token against 12 s alone.

### Against Atlas and vLLM, measured on this fleet

| Engine | Same file? | 1 stream | 2 / 4 streams | Load | GPU memory | Source |
|---|---|---|---|---|---|---|
| 6ix.cpp | — | 82.2 prose, 103.6 code | 118.5 / 185.4 | about 80 s cold, 12 s warm | see above | [dgxtwo_clean](one-box/dgxtwo_clean_2026-10-04.md) |
| Atlas | yes | 80.9 prose, 115.5 code | 108.4 / 120.1 (Atlas set to 4 seats) | 96 s, an upper bound (container start to first request) | 98.8 GiB | the registry's `baselines` entry: `bench_decode.py` on DGXtwo, 2026-10-04, before Atlas was retired; raw output on DGXtwo, not in this repository |
| vLLM | by the registry's note, yes | 35 | — | 473 s | 60.1 GiB | the registry's `baselines` entry: a fleet record of 2026-09 from the `6ixlabs` repository, **not re-measured**; draft head off, because it crashed on this checkpoint |

One run each. Atlas is ahead on code at one stream (115.5 against 103.6), level on prose (80.9
against 82.2) and behind at 2 and 4 streams (108.4 / 120.1 against 118.5 / 185.4).

Captures of the fixed prompt set from both engines on this checkpoint are in
`sixlabs/refset/results/` (`atlas_*.jsonl`, `dgpp_*.jsonl`).

**Published by others, not measured here:** Atlas, about 104 tok/s at one stream
([Coder-Next summary](../qwen3-coder-next/one-box/dgxone_coder_next_2026-10-04.md), "speeds on
record").

### What "verified" rests on

- 2026-10-03, engine `f288d60`: teacher-forced log-probabilities within 0.06 nat (mean) of the
  numpy reference on eight texts; greedy output with the draft head identical to plain decode on
  18 of 18 prompts ([6IXSERVE.md](../../6IXSERVE.md)).
- The forward-check gate that the other ports are judged by was **calibrated on this model** on
  2026-10-04 ([35B bring-up record](../qwen3.6-35b-a3b/one-box/results-2026-10-04-dgxone.md),
  sections 4.5 and 4.9): under the bounds as first written the 80B fails 4 of 4 runs; under the
  first restatement its code text in the checkpoint dense form still fails; under the final gate
  it passes 4 of 4. That record says plainly that the final gate's two exclusions were added
  because that run failed: "a calibration, not an independent test."
- One step per layer on the engine's own inputs, the check that record relies on (its section
  4.6): on the 80B a typical layer is 1.60–1.61 BF16 floors from the reference and the worst
  2.94 (median row error 1.39 % and 1.40 % of a layer's update), with 23 and 9 unexplained rows
  on the two texts.

## Two boxes

Template [`deploy/cluster_qwen3-next-80b_nvfp4_w2.example.json`](../../deploy/cluster_qwen3-next-80b_nvfp4_w2.example.json):
the one-box settings without `decode_passes_per_prefill`, `prefill_order`,
`prefill_budget_per_reader` and `prefill_group`, which the server refuses above one rank.
Record: [two-box 80B + 35B](../shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md) (DGXone +
DGXtwo, 2026-10-04, build `2ad1c4f`).

### Load and memory

| | Figure |
|---|---|
| First load | 144 s |
| Warm load | 10 s |
| Memory planned, each box | 38.12 GiB (33.18 device + 4.94 pinned) |
| Memory measured, each box | 38,373 MiB |

### Speed

ShareGPT, 64 prompts a level, all 64 successful at every level, one run each, an hour apart on
the same build:

| Streams | One box, tok/s | Two boxes, tok/s | Gain | Two boxes: first token median / P99, ms | per token median / P99, ms |
|---|---|---|---|---|---|
| 1 | 76.73 | 121.31 | ×1.58 | 128 / 273 | 7.72 / 10.21 |
| 2 | 97.52 | 161.51 | ×1.66 | 131 / 267 | 11.35 / 15.88 |
| 4 | 127.69 | 214.03 | ×1.68 | 154 / 341 | 17.49 / 31.35 |
| 8 | 153.95 | 251.45 | ×1.63 | 204 / 599 | 28.75 / 40.61 |

A second two-box run with the 35B loaded beside it and idle: 121.14 / 160.53 / 212.38 / 251.37.

tool-eval-bench on two boxes: short **93** (13 pass, 2 partial, 0 fail; 28/30), hard **80**
(66 pass, 8 partial, 14 fail; 140/176). Nine of the 14 failures are in category P (long
multi-step tasks). There is no one-box tool-eval to set it against.

### Other two-box records

| Record | Result |
|---|---|
| [YaRN ×4 on two boxes](two-box/dgxone_dgxtwo_80b_yarn_x4_2026-10-04.md) | request limit 1,048,576 tokens, 45.13 GiB planned and 45,555 MiB measured a box, 128 tok/s on one 177-token answer. **Recall at long lengths with ×4 is not tested.** |
| [HEM on the same boxes](../shared/dgxone_dgxtwo_hem_sharing_2026-10-04.md) | 123.5 tok/s with no scoring; continuous scoring on DGXtwo's GPU leaves 41 % of that, on both GPUs 26 % |
| [Loadout switching](../shared/dgxone_dgxtwo_loadout_switch_2026-10-04.md) | with the 35B as one loadout: answering 29–30 s after a switch from nothing loaded |

Not measured on two boxes: long prompts under concurrent load, and both models under load at
the same time (the two-box record, section 8). No Atlas or vLLM two-box figure for this model is
in the records.

## What failed, and why

- **Two ShareGPT runs measured empty prompts.** The chat template renders nothing for OpenAI's
  array-of-text-parts content, which `vllm bench serve` sends. The run of 2026-10-03
  (85.54 / 122.77 / 164.14 / 205.88) and the first two-box run (125.46 / 176.42 / 244.81 /
  301.28) are not results. Fixed in `2ad1c4f`, narrowed in `c29380f`
  ([two-box record](../shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md), section 6).
- **A second engine on the same boxes pinned the same CPU core.** With the 35B started beside it
  the 80B fell from 121 to 16.6 tok/s. Starting the second engine with `DGPP_BUS_ENGINE_CPU=18`
  restored both (same record, section 6).
- **vLLM's draft head crashes on this checkpoint** (fleet record quoted by the registry), which
  is why the fleet's vLLM figure is 35 tok/s.
- **The scheduler setting in use before 2026-10-04** gave long prompts everything: answers in
  progress fell to 2.0 tok/s each with pauses of 1.5–2 s
  ([prefill_sweep](one-box/prefill_sweep_2026-10-04.md)).

## Open

- The forward-check gate on two boxes, and upstream's `qwen35_tp_test` (it has no MoE fixture).
- An automatic choice of the bus thread's CPU core; until then every engine after the first on a
  box needs `DGPP_BUS_ENGINE_CPU`.
- The effect of losing the single-rank scheduler settings on two boxes under long-prompt load:
  not measured.
- YaRN: one recall point at ×2 on one box (a 302,368-token prompt, 5 of 5 planted facts, first
  token 277 s), none at ×4. `deploy/README.md` still describes the ×2 template as "unit-tested,
  not yet verified on a GPU".
- The FP8 release of the same model as a second format ([6IXSERVE.md](../../6IXSERVE.md),
  "Not done yet").

## Templates

| Use | Template |
|---|---|
| One box | [`cluster_qwen3-next-80b_nvfp4_w1.example.json`](../../deploy/cluster_qwen3-next-80b_nvfp4_w1.example.json) |
| One box, YaRN ×2 (524,288-token requests, 4 seats) | [`cluster_qwen3-next-80b_nvfp4_w1_yarn512k.example.json`](../../deploy/cluster_qwen3-next-80b_nvfp4_w1_yarn512k.example.json) |
| Two boxes | [`cluster_qwen3-next-80b_nvfp4_w2.example.json`](../../deploy/cluster_qwen3-next-80b_nvfp4_w2.example.json) |

## Records in this folder

- `one-box/`: [dgxtwo_clean](one-box/dgxtwo_clean_2026-10-04.md),
  [prefill_sweep](one-box/prefill_sweep_2026-10-04.md),
  [prefill_group](one-box/prefill_group_2026-10-04.md).
- `two-box/`: [YaRN ×4](two-box/dgxone_dgxtwo_80b_yarn_x4_2026-10-04.md).
- Shared with the 35B, in [`models/shared/`](../shared/): the
  [two-box record](../shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md) and its logs,
  [HEM sharing](../shared/dgxone_dgxtwo_hem_sharing_2026-10-04.md),
  [loadout switching](../shared/dgxone_dgxtwo_loadout_switch_2026-10-04.md).
- Elsewhere: [6IXSERVE.md](../../6IXSERVE.md) (the port and the 2026-10-03 figures).
