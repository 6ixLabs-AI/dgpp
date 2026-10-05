# Qwen3-Next-80B and Qwen3.6-35B-A3B across both boxes — DGXone + DGXtwo, 2026-10-04

> Layout note, 2026-10-04: moved from `sixlabs/bench/results/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md` when the per-model folders under `models/` were made. Paths to other moved records were updated; nothing else was changed. Its log folder moved with it. It covers two models, so it lives in `models/shared/` and both model folders link to it.

Status: **both models serve over two Sparks and are about 1.6× faster than on one, at 1 to 8
streams.** Measured with ShareGPT and tool-eval-bench. **NOT done:** the forward-check gate at
two boxes (the tool has no two-rank mode), the 35B's one-box gate on the merged engine, and
upstream's `qwen35_tp_test`. Answers were checked by hand and by tool-eval only: treat the
two-box worlds as measured, not verified.

Every number is quoted from a log in
`models/shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04/`. Log timestamps are UTC.

## 1. Result

ShareGPT, 64 prompts a level, all 64 successful at every level, output tok/s:

| Streams | 80B one box | 80B two boxes | Gain | 35B one box (morning build) | 35B two boxes | Gain |
|---|---|---|---|---|---|---|
| 1 | 76.73 | 121.31 | ×1.58 | 96.72 | 154.43 | ×1.60 |
| 2 | 97.52 | 161.51 | ×1.66 | 135.37 | 214.11 | ×1.58 |
| 4 | 127.69 | 214.03 | ×1.68 | 182.37 | 300.46 | ×1.65 |
| 8 | 153.95 | 251.45 | ×1.63 | 220.11 | 367.57 | ×1.67 |

- The 80B pair is like for like: same build (`2ad1c4f`), same settings, same harness, one run
  each, an hour apart.
- The 35B's one-box column is this morning's run on an earlier build
  (`models/qwen3.6-35b-a3b/one-box/dgxone_35b_6ixcpp_vs_atlas_2026-10-04.md`), before the upstream merge changed this family's
  kernels. Same checkpoint and harness, not the same engine version.
- Both models were loaded on the two boxes together for the 35B run and for a second 80B run
  (121.14 / 160.53 / 212.38 / 251.37): an idle neighbour costs nothing measurable.
- Memory per box: 80B **38,373 MiB** (one box: 70,540 MiB), 35B **22,950 MiB** (one box:
  39,478 MiB). Both together leave about 43 GiB available on DGXone and 50 GiB on DGXtwo.

## 2. What changed in the engine (branch `w2/qwen35-moe`)

The upstream merge (`51408f8`) brought two- and four-Spark worlds to the `qwen3_5` family for
its dense model only and refused the routed-MoE models by name. The loader already sliced the
experts per rank and the forward already staged and folded the MLP output, so the model's part
was small (`9612a02`, `src/models/qwen/model35.cpp`): the routed chain and the memory plan are
sized from the rank's slice, the dense-MLP prefetch is skipped for the MoE models, and the two
refusals are gone.

Templates: `deploy/cluster_qwen3-next-80b_nvfp4_w2.example.json` and
`deploy/cluster_qwen3.6-35b-a3b_nvfp4_w2.example.json`. They are the one-Spark settings without
`decode_passes_per_prefill`, `prefill_order`, `prefill_budget_per_reader` and `prefill_group`,
which the server still refuses above one rank. The effect of losing those under long-prompt
load was not measured.

## 3. Setup

| | 80B | 35B |
|---|---|---|
| Checkpoint | `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` @ `8fb2682f…` | `nvidia/Qwen3.6-35B-A3B-NVFP4` @ `1355db6a…` |
| Seats / pool / draft depth | 8 / 524,288 tokens / 2 | 8 / 262,144 tokens / 2 |
| Plan, each rank | 38.12 GiB (33.18 device + 4.94 pinned) | 22.76 GiB (17.28 device + 5.48 pinned) |
| First load / warm | 144 s / 10 s | 60 s / 6 s |
| Rank 0 | DGXone, `172.17.0.1:18190` | DGXone, `172.17.0.1:18191` |
| Bus | two RoCE lanes, fabric 29990, journal 29991 | two RoCE lanes, fabric 29992, journal 29993 |

Engine `dgpp-serve 0.1.0+g2ad1c4f400aa`, built on DGXtwo (DGXone was under test), rank 0's
binary staged to rank 1 by the launcher. Rank 1 reads the checkpoints over `/mnt/dgxone-hf`;
each box has a small private hub folder of links (`~/6ixinfer-out/hfcache-qwen`), no weights
copied. Both boxes otherwise idle: HEM, embedder, reranker and Docling stopped, the embedding
jobs held.

## 4. The 80B in detail

| Streams | One box: tok/s | first token median / P99, ms | per token median / P99, ms | Two boxes: tok/s | first token | per token |
|---|---|---|---|---|---|---|
| 1 | 76.73 | 198 / 350 | 12.36 / 17.00 | 121.31 | 128 / 273 | 7.72 / 10.21 |
| 2 | 97.52 | 205 / 506 | 18.89 / 27.57 | 161.51 | 131 / 267 | 11.35 / 15.88 |
| 4 | 127.69 | 231 / 619 | 29.12 / 58.83 | 214.03 | 154 / 341 | 17.49 / 31.35 |
| 8 | 153.95 | 287 / 615 | 46.83 / 64.61 | 251.45 | 204 / 599 | 28.75 / 40.61 |

The one-box column used the two-box template's settings with `world_size` 1. With the
production one-box template (the tuned prompt-reading settings) the same test gave 77.78 at one
stream and 104.02 at two before the run was stopped on the owner's instruction
(`fleet-suite-80b-w1tuned.log`); four and eight streams were not measured.

tool-eval-bench on the two-box 80B: short **93** (13 pass, 2 partial, 0 fail; 28/30), hard
**80** (66 pass, 8 partial, 14 fail; 140/176; 195 s in scenarios). By category on the hard set:
A 5/6, B 6/6, C 6/8, D 6/6, E 5/6, F 6/6, G 5/6, H 10/10, I 17/20, J 4/6, K 21/26, L 8/8,
M 3/6, N 6/6, O 12/12, P 20/38. Nine of the 14 failures are in P (long multi-step tasks). The
35B on one box scored 93 and 89 (P 35/38): it is the better tool caller on long chains. The 80B
was not run through tool-eval on one box, so this is not a one-box against two-box comparison.

## 5. The 35B in detail (two boxes)

| Streams | tok/s | First token median / P99, ms | Per token median / P99, ms | Duration, s |
|---|---|---|---|---|
| 1 | 154.43 | 102 / 205 | 6.01 / 9.46 | 103.58 |
| 2 | 214.11 | 112 / 205 | 8.44 / 10.19 | 74.71 |
| 4 | 300.46 | 129 / 257 | 11.60 / 14.76 | 53.24 |
| 8 | 367.57 | 142 / 358 | 19.14 / 24.32 | 43.52 |

Spot answers were right ("done.", 391, Canberra, a correct Fibonacci function, and 9 for the
sheep riddle with thinking on). No gate, no tool-eval, no group check on two boxes.

## 6. Two faults found on the way

**The 80B answered empty messages under ShareGPT.** Its chat template renders nothing for
OpenAI's array-of-text-parts content: a request sent that way got a prompt of 8 tokens and the
reply "It seems your message might be incomplete". `vllm bench serve` sends that form, so the
ShareGPT run of 2026-10-03 on the 80B (85.54 / 122.77 / 164.14 / 205.88) and the first run here
(`fleet-suite-80b-w2.log`: 125.46 / 176.42 / 244.81 / 301.28) measured answers to empty prompts
and are not results. Fixed in `2ad1c4f`, narrowed in `c29380f`: the service renders one tiny
message both ways on the first chat request and, where the template loses the array (it throws,
or renders fewer tokens), serves text-only arrays as their joined text. Templates that read the
parts get the request as sent. `serve_test`: 104 tests, 0 failed. Every figure in sections 1, 4
and 5 was taken on `2ad1c4f`, which flattened always; the 35B's template reads parts itself, so
its numbers do not depend on the difference.

**Two engines on the same boxes pinned the same CPU core.** Each engine pins its bus polling
thread to the fastest core, highest index first: core 19 for both. With the 35B started beside
the 80B, the 80B fell from 121 to 16.6 tok/s and the 35B ran at 20, about 110 ms a pass, with
`/proc/pressure/cpu` at 24–28 % (avg60) on both boxes and memory pressure at zero. Starting the
second engine with `DGPP_BUS_ENGINE_CPU=18` restored both (128–130 and 149–151 tok/s on a
single answer; CPU pressure under 1 %). On a GB10 the 3.9 GHz cores are 5–9 and 15–19. An
automatic choice is a separate task; until it lands, every engine after the first on a box needs
that setting.

## 7. Start-up notes

- Start from an OpenSSH session (memlock) and stop the fleet agents and telemetry on both boxes
  until the log shows the bus up; their probes break the journal handshake.
- A second engine on the same boxes needs its own HTTP, fabric and journal ports, its own stage
  directory and `DGPP_BUS_ENGINE_CPU`.
- During the first load rank 0 logs `allreduce … STALLED` lines while rank 1 reads its half over
  NFS (33 on the 80B's first load). None were logged while serving.

## 8. Not run

- The forward-check gate at two boxes, and upstream's `qwen35_tp_test` (loopback worlds on a
  tiny dense fixture; it has no MoE fixture yet).
- The 35B's one-box gate on the merged engine, and a same-build one-box ShareGPT for it.
- Both models under load at the same time.
- Long prompts under concurrent load on two boxes (`mixed_load`, `group_check`).
- YaRN ×2 and ×4 beyond one point: at ×2 a 302,368-token prompt returned 5 of 5 planted facts
  (first token 277 s); the run was then stopped for this work.

## 9. Files

`fleet-suite-80b-w2-fixed.log` and `fleet-suite-80b-w2-with35b.log` (80B two boxes),
`fleet-suite-80b-w1cmp.log` (80B one box, same settings), `fleet-suite-80b-w1tuned.log` (80B one
box, production settings, two levels), `fleet-suite-35b-w2.log` (35B two boxes),
`fleet-suite-80b-w2-teb.log` (tool-eval), `fleet-suite-80b-w2.log` (the empty-prompt run, kept
as evidence), `up-80b-w2.log`, `up-35b-w2.log`.
