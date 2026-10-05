# Qwen3.5-122B-A10B

**Outcome: dropped.** It was brought up, passed the gate and was measured on one box on
2026-10-04. At 36.5 tok/s for one stream the owner's decision, the same day, was: "NO the 122b
model is too slow". No further work on it is planned, and nothing below proposes any. Every
figure is quoted from the [one-box record](one-box/results-2026-10-04-dgxone.md) or the registry (entry
`qwen3.5-122b-a10b` in [`sixlabs/registry/models.json`](../../sixlabs/registry/models.json)).

## What it is

- Family `qwen3_5`, the 35B's model class at a larger size: 48 layers (36 Gated DeltaNet + 12
  full attention) and one draft layer, hidden 3072, a routed MoE of 256 experts, top 8.
- A thinking model. Its vision tower is bound and not served.
- The checkpoint carries a draft head (785 tensors in the Sehyo release), run at depth 2.

| Checkpoint | Revision | Format | Tensors | Registry status |
|---|---|---|---|---|
| [`Sehyo/Qwen3.5-122B-A10B-NVFP4`](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4) (the default; the one that ran) | `56a6bdda` | nvfp4-compressed-tensors | 149,552 | `working` |
| [`nvidia/Qwen3.5-122B-A10B-NVFP4`](https://huggingface.co/nvidia/Qwen3.5-122B-A10B-NVFP4) | `98915d83` | nvfp4-modelopt | 148,976 | `compiled`: never loaded |
| [`Qwen/Qwen3.5-122B-A10B-FP8`](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8) | `a099dee7` | fp8-block | 76,323 | `compiled`: does not fit one box |
| `Intel/Qwen3.5-122B-A10B-int4-AutoRound` | — | int4-autoround | — | `refused` |

## Status

In the registry's words, `working` means: "Verified on a GPU on this fleet: bound, passed the
forward-check gate, served, measured. Has a record."

- **One box, Sehyo NVFP4: verified** on 2026-10-04, DGXone, engine commit `29f904b`.
- **Two boxes: not run** on this engine.
- **Dropped by the owner as too slow.**

## One box

Template [`deploy/cluster_qwen3.5-122b-a10b_nvfp4-sehyo_w1.example.json`](../../deploy/cluster_qwen3.5-122b-a10b_nvfp4-sehyo_w1.example.json):
8 seats, a 65,536-token pool (the nvidia template's is 262,144), a 1 GiB prefix cache, FP8 dense
form, draft depth 2. DGXone, 2026-10-04, one run.

### Load and memory

| | Figure | Note |
|---|---|---|
| Load, cold | 176 s | encodes the dense matrices to FP8 and writes a 67.4 GiB resident image |
| Load, warm (resident image) | 26 s | |
| Memory planned | 83.12 GiB (79.30 device + 3.82 pinned) + 4.00 headroom | 65,536-token pool, 8 seats |
| Memory measured, device | 86,941 MiB = 84.90 GiB | |
| Memory measured, the box | about 89.6 GiB of MemAvailable | 6.5 GiB more than the plan's total |

### Speed

ShareGPT, 64 prompts a level, all 64 successful at every level, thinking on:

| Streams | Output tok/s | First token median / P99, ms | Per token median / P99, ms |
|---|---|---|---|
| 1 | 36.49 | 426.56 / 1839.32 | 24.61 / 34.04 |
| 2 | 46.86 | 465.54 / 1875.50 | 37.50 / 47.04 |
| 4 | 60.04 | 666.21 / 2227.33 | 56.73 / 72.90 |
| 8 | 71.15 | 1069.18 / 2797.64 | 94.98 / 118.62 |

Draft acceptance over the run: 77.5 % (39,160 of 50,558). The 2-stream figure is the one most
likely to be slightly low: the reranker used 6–9 % of the GPU at the start and end of that level.
The first-token times are long because build `29f904b` refused the explicit prompt-reading
settings for this family and ran the automatic ones.

No tool-eval, no `bench_decode.py` (ShareGPT only, as instructed).

### The gate: PASS, 4 of 4

| Input, dense form | mean \|Δ\| | pearson | greedy rows agree | Verdict |
|---|---|---|---|---|
| code, checkpoint | 0.02206 | 0.999839 | 112/113 (miss at a 0.0463 nat margin) | PASS |
| prose, checkpoint | 0.04076 | 0.999471 | 87/87 | PASS |
| code, `fp8` (the serving form) | 0.04126 | 0.999035 | 112/113 (the same near tie) | PASS |
| prose, `fp8` (the serving form) | 0.05338 | 0.999091 | 86/87 (miss at a 0.0254 nat margin) | PASS |

Default bounds of `compare_forward_check.py` as restated on 2026-10-04 (pearson ≥ 0.995, mean ≤
0.10 nat, greedy agreement above a 0.5 nat margin). Bind check: 149,552 of 149,552 tensors.

### Against Atlas and vLLM

**Nothing was measured against this model on this fleet on another engine.**

| Engine | Figure | What it is |
|---|---|---|
| Atlas | 33.4 tok/s | **Published, not measured here.** Its own single-call decode rate at batch 1 from its recipe `qwen3.5-122b-a10b-nvfp4-single`, 16,384-token context, no speculation; not ShareGPT. Atlas was not run on the owner's instruction, and by its recipe it commits about 110 GiB before its cache, so it could not load beside DGXone's services. |
| Atlas, two Sparks | about 51 tok/s | **Published, not measured here** (registry note). |
| vLLM | 395 s load, 47.6 GiB a box | A fleet record of 2026-09 from the `6ixlabs` repository: tensor parallel over both boxes. No speed on record. |

Set beside Atlas's published figure, 6ix.cpp's 36.49 (ShareGPT throughput) and 40.6 tok/s
(decode alone, from the 24.61 ms median) are 9 % and 22 % ahead at one stream, with four times
the context. The record is explicit that this is a comparison against a claim, not a
head-to-head.

## Two boxes

Not run on this engine.

## Did not work, and why

- **Too slow to use: the outcome.** 36.49 tok/s at one stream and 71.15 at eight.
- **It does not fit one box beside the fleet's services.** With the reranker, embedder and
  Docling up, 15 GiB stayed available at 8 seats and 18.3 GiB at 4, under the fleet's 19.7 GiB
  floor. The owner waived the floor for the run; the lowest reading during ShareGPT was
  12.30 GiB, and the out-of-memory guard killed nothing.
- **The planner understated the cost**: 83.12 GiB planned, about 89.6 GiB gone.
- **`Qwen/Qwen3.5-122B-A10B-FP8` does not fit one Spark**: 117.59 GiB resident leaves no room
  for a pool (registry). It was removed from DGXone's cache on 2026-10-04.
- **`nvidia/Qwen3.5-122B-A10B-NVFP4` was never loaded**: it is not in the fleet's cache
  (registry). Its plan is 91.0 GiB with a 262,144-token pool and the prefix cache.
- **`Intel/Qwen3.5-122B-A10B-int4-AutoRound` is refused by name** at config parse (registry).
- **Atlas could not be run beside DGXone's services** (above), so its column is its published
  figure.

## Templates

| Use | Template |
|---|---|
| One box, the configuration that ran | [`cluster_qwen3.5-122b-a10b_nvfp4-sehyo_w1.example.json`](../../deploy/cluster_qwen3.5-122b-a10b_nvfp4-sehyo_w1.example.json) |
| The nvidia release (never loaded) | [`cluster_qwen3.5-122b-a10b_nvfp4_w1.example.json`](../../deploy/cluster_qwen3.5-122b-a10b_nvfp4_w1.example.json) |

## Records in this folder

- `one-box/`: [the one-box record](one-box/results-2026-10-04-dgxone.md).
- Bring-up steps and gate inputs: `sixlabs/ports/qwen3.5-122b-a10b/`
  ([gpu-steps.md](../../sixlabs/ports/qwen3.5-122b-a10b/gpu-steps.md)).
