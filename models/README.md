# Models on this fleet: what ran, how fast, and what did not work

One folder per model that has been run on 6ixLabs' two DGX Sparks (DGXone and DGXtwo) with this
engine, or beside it. Each folder has a README with the measured tables and the detailed records
they are quoted from. Models that were never loaded on a GPU here are on one page:
[candidates.md](candidates.md).

The 80B and the 35B against Atlas and vLLM are written up on one page each:
[performance-80b.md](performance-80b.md) and [performance-35b.md](performance-35b.md).

Rules these pages keep:

- Every figure comes from a record in this repository, and the record is linked beside it.
- **Verified** means the model passed the forward-check gate on this fleet. **Measured** means it
  served and was timed, and nothing more. The two are never merged.
- A figure published by someone else and not measured here says so.
- Statuses are the registry's ([`sixlabs/registry/models.json`](../sixlabs/registry/models.json));
  nothing is upgraded on these pages.

## Summary

Speeds are ShareGPT output tok/s (64 prompts a level, all streams together), the one benchmark
every measured model was run through. Each is one run. Other benchmarks are in the model folders.

| Model | Registry status, and what that covers | One box: 1 stream / 8 streams | Two boxes: 1 stream / 8 streams | Memory per box, planned / measured | Folder |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (NVFP4) | `working`. One box verified; two boxes measured, not verified | 96.72 / 220.11 | 154.43 / 367.57 | one box 36.00 GiB / 39,478 MiB; two boxes 22.76 GiB / 22,950 MiB | [qwen3.6-35b-a3b](qwen3.6-35b-a3b/README.md) |
| Qwen3-Next-80B-A3B Instruct (NVFP4) | `working`. One box verified; two boxes measured, not verified | 76.73 / 153.95 | 121.31 / 251.45 | one box 57.9 GiB planned at a 262,144-token pool, 70,540 MiB measured at a 524,288-token pool; two boxes 38.12 GiB / 38,373 MiB | [qwen3-next-80b](qwen3-next-80b/README.md) |
| Qwen3-Coder-Next (NVFP4) | `compiled`. Measured, **not verified**: served under a gate exception | 40.37 / 124.12 | not run | one box 64.51 GiB / 67,889 MiB | [qwen3-coder-next](qwen3-coder-next/README.md) |
| Qwen3.5-122B-A10B (Sehyo NVFP4) | `working`. One box verified. **Dropped as too slow** | 36.49 / 71.15 | not run | one box 83.12 GiB / 86,941 MiB (about 89.6 GiB of the box's memory) | [qwen3.5-122b-a10b](qwen3.5-122b-a10b/README.md) |
| DeepSeek-V4-Flash (0731) | `upstream`. Measured on two boxes; the gate was not run here | does not apply: two Sparks at least | 43.24 / 66.24 with thinking on (six seats, so 8 streams queue two) | two boxes 84.20 GiB / 90,406 MiB | [deepseek-v4-flash](deepseek-v4-flash/README.md) |
| Qwen3.8-Flash-Next | `upstream`. Runs in production on upstream DGPP; **not measured by 6ixLabs on this fork** | not measured here | not measured here | not measured here | [qwen3.8-flash-next](qwen3.8-flash-next/README.md) |

Notes on the table:

- The 80B's one-box column and its two-box column are the same build (`2ad1c4f`) and settings, an
  hour apart. The 35B's one-box column is the morning's run on an earlier build (`29f904b`): the
  same checkpoint and harness, not the same engine version.
- With `bench_decode.py` instead of ShareGPT the one-box figures are higher: the 80B 82.2 (prose)
  to 103.6 (code) at one stream and 253.4 at eight; the 35B 97.7 to 120.1 and 271.3.
- DeepSeek-V4-Flash with thinking off, by MiaAI-Lab's method: 65.8 tok/s at one stream, 103.3 in
  total at six.
- Fifteen other models in the registry have never been loaded on a GPU here with this engine.
  They are in [candidates.md](candidates.md), grouped by how much work stands between them and a
  run.

### Against other engines, measured on this fleet

| Model | Other engine | Same file? | Result | Record |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | Atlas | yes | 6ix.cpp ahead at every ShareGPT level: 1.14× at 1 stream, 1.21× at 2, 1.67× at 4, 1.73× at 8. Atlas held 73,861 MiB against 39,478 | [against Atlas](qwen3.6-35b-a3b/one-box/dgxone_35b_6ixcpp_vs_atlas_2026-10-04.md) |
| Qwen3.6-35B-A3B | vLLM | no (FP8 against NVFP4) | `bench_decode.py`: 97.7–120.1 against 50.1 at one stream, 271.3 against 190.0 at eight. vLLM read prompts about 3 times faster alone (6ix.cpp at its automatic settings) | [35B bring-up](qwen3.6-35b-a3b/one-box/results-2026-10-04-dgxone.md) |
| Qwen3-Next-80B | Atlas | yes | `bench_decode.py`, one stream: 82.2 prose and 103.6 code against Atlas's 80.9 and 115.5. Atlas is ahead on code. At 2 / 4 streams 118.5 / 185.4 against 108.4 / 120.1 (Atlas at 4 seats) | registry `baselines`; [80B folder](qwen3-next-80b/README.md) |
| Qwen3-Coder-Next | vLLM | no (int4 against NVFP4) | **6ix.cpp behind at every ShareGPT level**: 0.60, 0.61, 0.66, 0.64 of vLLM at 1 / 2 / 4 / 8 streams | [Coder-Next summary](qwen3-coder-next/one-box/dgxone_coder_next_2026-10-04.md) |
| DeepSeek-V4-Flash | vLLM | yes | nothing measured on the same day. Against earlier and published figures: level at one stream with thinking off, behind above it | [DeepSeek record](deepseek-v4-flash/two-box/dgxone_dgxtwo_deepseek_v4_flash_2026-10-04.md) |

## What we learned

1. **The speed came from the draft head.** The 80B and the 35B carry one and the engine runs it;
   Coder-Next has the 80B's architecture and no draft head in any checkpoint, and ran at 0.60 to
   0.66 of vLLM. On a plain pass vLLM was the faster engine for that architecture (14.3 ms
   against 24.2 ms a token, different quantizations).
   [Coder-Next summary](qwen3-coder-next/one-box/dgxone_coder_next_2026-10-04.md).
2. **Two boxes gave about 1.6× for the two routed-MoE models**, at every level from 1 to 8
   streams, and an idle second model on the same boxes cost nothing measurable.
   [Two-box record](shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md).
3. **A draft head is not enough by itself.** The 122B has one (77.5 % of drafts accepted) and
   reached 36.49 tok/s on one box; it was dropped as too slow. DeepSeek-V4-Flash has one and its
   total throughput stops growing at four streams, behind vLLM's published and earlier figures.
   [122B record](qwen3.5-122b-a10b/one-box/results-2026-10-04-dgxone.md),
   [DeepSeek record](deepseek-v4-flash/two-box/dgxone_dgxtwo_deepseek_v4_flash_2026-10-04.md).
4. **Reading prompts is this engine's weak side, and the scheduler settings decide how much it
   hurts.**
   With the automatic settings two long prompts starved four chat streams on the 35B (13.1 tok/s
   between them, 41.7 s to a first token); with the tuned settings, 95.4 tok/s and 0.47 s, with
   the same answers. The two-box templates cannot carry all of those settings yet
   (`decode_passes_per_prefill` and `prefill_order` are single-rank only).
   [35B prefill](qwen3.6-35b-a3b/one-box/dgxone_35b_prefill_2026-10-04.md),
   [80B sweep](qwen3-next-80b/one-box/prefill_sweep_2026-10-04.md).
5. **The resident image makes a restart cost seconds.** Warm starts are 6 to 26 s for every
   model measured here (two-box 35B 6 s, one-box 122B 26 s); the fleet's vLLM slots took 363 to
   473 s to load (two measured on 2026-10-04, two from fleet records). A switch from nothing to
   the 80B + 35B pair answers in 29–30 s.
   [Loadout switching](shared/dgxone_dgxtwo_loadout_switch_2026-10-04.md); the load tables in
   each folder.
6. **A single short text cannot certify a routed-MoE model at tight bounds.** The router's
   logits are rounded to BF16 before the top-k, near ties pick different experts, and the numpy
   reference fails the original gate against itself. The gate was restated twice in one night,
   and the record that did it says the second change is "a calibration, not an independent
   test". What the ports rely on is the one-step-per-layer check.
   [35B bring-up](qwen3.6-35b-a3b/one-box/results-2026-10-04-dgxone.md), section 4.
7. **Check what the benchmark actually sent.** `vllm bench serve` sends message content as an
   array of text parts. The 80B's template rendered that as nothing, so two ShareGPT runs
   measured answers to empty prompts and looked faster than the real ones (301 tok/s at eight
   streams against 251). They were thrown away and the service was fixed.
   [Two-box record](shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md), section 6.
8. **Neighbours on the same box are not free.** A second engine that pinned the same CPU core
   took the 80B from 121 to 16.6 tok/s until it was given its own core. Continuous HEM scoring
   on one GPU leaves the two chat models 41 % of their single-stream speed.
   [Two-box record](shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md), section 6;
   [HEM sharing](shared/dgxone_dgxtwo_hem_sharing_2026-10-04.md).

## Folders

| Folder | What is in it |
|---|---|
| [qwen3-next-80b/](qwen3-next-80b/README.md) | one-box records (clean benchmarks, the two scheduler sweeps), the two-box YaRN ×4 record |
| [qwen3.6-35b-a3b/](qwen3.6-35b-a3b/README.md) | the bring-up record and gate verdicts, the comparison with Atlas, the prompt-reading comparison |
| [qwen3-coder-next/](qwen3-coder-next/README.md) | the bring-up record, the comparison with vLLM, the gate verdicts |
| [qwen3.5-122b-a10b/](qwen3.5-122b-a10b/README.md) | the one-box record |
| [deepseek-v4-flash/](deepseek-v4-flash/README.md) | the two-box record and its raw data |
| [qwen3.8-flash-next/](qwen3.8-flash-next/README.md) | no record of its own: what is on record elsewhere, and upstream's figures |
| [shared/](shared/) | records that cover more than one model: the 80B + 35B two-box record and its logs, HEM sharing, loadout switching |
| [candidates.md](candidates.md) | every registry model that has not run here: which could run with little engine work, and which need kernels first |

Engine-level records stay where they were: the upstream merge
([`sixlabs/bench/results/upstream_merge_2026-10-04.md`](../sixlabs/bench/results/upstream_merge_2026-10-04.md))
and the integration check of the new families
([`sixlabs/ports/integration-2026-10-04.md`](../sixlabs/ports/integration-2026-10-04.md)). Bring-up
steps and gate inputs for every port stay in `sixlabs/ports/<model>/`.

## Acknowledgements

Almost nothing on these pages would exist without other people's work. Thank you.

### DGPP, by Stephen Hawkins

This repository is a fork of **[DGPP](https://github.com/HawkBearPig/dgpp)** (Apache-2.0,
copyright 2026 Stephen Hawkins; see [`LICENSE`](../LICENSE) and [`NOTICE`](../NOTICE), which are
his and unchanged). The whole approach comes from DGPP: a native C++/CUDA serving engine for
DGX Spark, resident images that turn a restart into seconds, tensor-parallel serving over RoCE
across Sparks, and speculative decoding through a checkpoint's own draft head. The design and
most of the code are his. Every speed on these pages is a speed of his engine with our additions
on top, and the two results that stand out here, the draft-head decode and the two-box gain,
are his mechanisms applied to more models.

The `qwen3_5` stack that all of our ports are dialects of is upstream's too: it entered DGPP as
Ahmed Samir's Qwen3.5-27B serving support (commit `cb29172`), and its tensor-parallel worlds came
with upstream's #86. Our Prometheus pull request (#81) was merged upstream on 2026-10-04 with
Stephen's own additions (peer metrics configured through the deployment JSON).

Thank you, Stephen, and thank you to DGPP's other contributors.

### How 6ixLabs extended it

Each item is in this repository's history and has a record:

- **Qwen3-Next-80B-A3B**, which DGPP did not serve, as a dialect of the `qwen3_5` stack
  (`f288d60`; [6IXSERVE.md](../6IXSERVE.md)), with bind and forward-check tools and a numpy
  reference of the model.
- **More models on the same stack**: Qwen3.6-35B-A3B (`33063e4`), Qwen3-Coder-Next (`7cad5c0`),
  Qwen3.5-122B-A10B (`3df149b`) and Qwen3.5-0.8B (`a558c9f`; never run on a GPU), and a
  **model registry** that says how far each one has been verified
  ([`sixlabs/registry/`](../sixlabs/registry/README.md)).
- **Scheduler and prompt-reading settings**: answers keep moving while long prompts are read
  (`engine.decode_passes_per_prefill`, `engine.prefill_order`), shared prompt reading for this
  family (`engine.prefill_group`, `engine.prefill_budget_per_reader`), a live request evicts
  cached prefixes before anything is shed, and generous answer limits are clamped instead of
  refused.
- **`POST /v1/score`**: the model's own log-probability of a given continuation (`5c20f70`), and
  `engine.logprobs_mode` for raw-distribution logprobs on sampled requests (`d3a3689`).
- **YaRN long context for the 80B** (`f9a333a`). DGPP already had YaRN for Qwen3.8-Flash-Next;
  this extends it to the Qwen3-Next dialect. One recall point at ×2 is on record; ×4 starts and
  answers but its recall is untested.
- **The routed-MoE models on two Sparks** (`9612a02`): upstream's #86 brought tensor parallel to
  the family's dense model; this carries it to the 80B and the 35B. Measured at about 1.6× one
  Spark; not yet verified.
- **The chat content-parts fix** (`4a4ef27`, `2ad1c4f`, `c29380f`): text-only content arrays are
  served as their joined text where a template would lose them.
- **The Prometheus exposition** at `/metrics/prometheus`, contributed upstream as pull request
  #81 and merged there on 2026-10-04.
- **The fleet benchmarks and these records**: the harness in
  [`sixlabs/bench/`](../sixlabs/README.md) and everything under `models/`.
- **Groundwork, not served**: host-side config, tensor tables and references for Nemotron-3,
  Gemma-4, Mistral-Small-4, MiniMax-M2.7 and plain Qwen3 ([candidates.md](candidates.md)).

### Atlas, by Thomas Braun of Avarok Cybersecurity

**[Atlas](https://github.com/Avarok-Cybersecurity/atlas)** is the inference engine we measured
against on the same checkpoints, and it was the first to show us these models running fast on a
Spark with a draft head. It served the 80B on this fleet, and its answers to our fixed prompt
set are kept in `sixlabs/refset/results/` beside this engine's. The same-file comparisons on these pages, for the 80B and the 35B, exist because
Atlas made them possible, and its published recipes are the only figures we have for several
models we have not run. Where Atlas is ahead we say so: on the 80B at one stream on code it is
the faster of the two (115.5 tok/s against 103.6).

Atlas is AGPL-3.0. We ran it as shipped, from its published container image and recipes with the
overrides each record lists, and read only its published recipes and documentation. **No Atlas
source is included in this repository.**

Thank you, Thomas.

### MiaAI-Lab

**[MiaAI-Lab's DeepSeek-V4-Flash on two DGX Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)**
(MIT) gave us the benchmark method for the DeepSeek record (`mia_bench_6ix.py` follows their
`scripts/benchmark-0731.py`) and the published vLLM figures that record is set against. Thank
you.

### The checkpoints

The models are their authors' work, and the quantized releases are the work of the people who
made them. The ones a record here actually loaded or measured:

- **Qwen**: the Qwen3-Next, Qwen3.5, Qwen3.6 and Qwen3.8 models themselves, and
  `Qwen/Qwen3.6-35B-A3B-FP8` (the gate, and the fleet's vLLM baseline).
- **NVIDIA**: `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` and `nvidia/Qwen3.6-35B-A3B-NVFP4`, the
  two checkpoints served and measured here on one box and on two.
- **RedHatAI**: `RedHatAI/Qwen3-Coder-Next-NVFP4`, the Coder-Next checkpoint that was served and
  measured.
- **Intel**: `Intel/Qwen3-Coder-Next-int4-AutoRound`, the checkpoint behind the fleet's vLLM
  Coder slot and so the baseline in the Coder-Next record. (This engine refuses it by name; the
  registry says why.)
- **Sehyo**: `Sehyo/Qwen3.5-122B-A10B-NVFP4`, the 122B release that was verified.
- **DeepSeek**: `deepseek-ai/DeepSeek-V4-Flash-0731`.
