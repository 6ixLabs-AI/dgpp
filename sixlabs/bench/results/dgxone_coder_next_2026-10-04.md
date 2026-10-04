# Qwen3-Coder-Next: 6ix.cpp against vLLM, and why it is not the 80B — DGXone, 2026-10-04

Summary of the Coder-Next bring-up and comparison. The full record, with every command and its
output, is `sixlabs/ports/qwen3-coder-next/results-2026-10-04-dgxone.md`.

## Result

Qwen3-Coder-Next boots and serves on 6ix.cpp, loads four times faster than the fleet's vLLM slot,
scores the same on tool calling, and **generates 34–40 % slower than vLLM at every concurrency**.
It is not verified: it serves under a gate exception (see "Gate").

The two engines did not run the same file. 6ix.cpp refuses the checkpoint the vLLM slot serves, so
every speed and memory figure below compares engine plus quantization, not engine alone.

| | 6ix.cpp | vLLM |
|---|---|---|
| Checkpoint | `RedHatAI/Qwen3-Coder-Next-NVFP4` (4-bit float, scale per 16 weights; 44.3 GiB) | `Intel/Qwen3-Coder-Next-int4-AutoRound` (4-bit integer, scale per 128 weights; 40.5 GiB) |
| Left at 16 bits | linear-attention projections, the expert router, the shared-expert gate | the shared-expert gate |
| Served as | checkpoint dense form (`"dense_weights": "checkpoint"`), no draft head | the fleet's slot `qwen3-coder@dgxone`, run as it stands |
| Build | `29f904b`; ShareGPT on `29f904b` + the content-parts fix (`4a4ef27`) | image `coder-int4-patched` |

## Measurements

One run each, DGXone, HEM paused from 10:32:35 EDT (roughly the first 90 s of vLLM's 1-stream
ShareGPT level overlapped HEM at ~89 % GPU; everything else ran on a quiet box).

| | 6ix.cpp | vLLM |
|---|---|---|
| Load to serving, cold | 86 s | 363 s |
| Load to serving, warm restart | 12 s | not measured |
| Device memory | 67,889 MiB (planned 64.51 GiB) | 54,362 MiB |

ShareGPT, 64 prompts a level, all 64 successful at every level on both:

| Streams | Output tok/s 6ix.cpp | Output tok/s vLLM | 6ix.cpp / vLLM | First token median / p99, ms: 6ix.cpp | vLLM | Per token median / p99, ms: 6ix.cpp | vLLM |
|---|---|---|---|---|---|---|---|
| 1 | 40.37 | 67.10 | 0.60 | 153 / 339 | 141 / 353 | 24.19 / 25.37 | 14.25 / 15.11 |
| 2 | 65.57 | 107.26 | 0.61 | 155 / 333 | 156 / 309 | 28.78 / 31.29 | 17.26 / 23.07 |
| 4 | 97.58 | 148.70 | 0.66 | 177 / 384 | 170 / 452 | 38.19 / 43.19 | 24.72 / 31.50 |
| 8 | 124.12 | 192.91 | 0.64 | 207 / 535 | 217 / 438 | 58.41 / 71.85 | 38.17 / 44.24 |

Time to first token is the same on both. The gap is generation speed.

tool-eval-bench:

| Set | Engine | Score | Pass / partial / fail | Points | Time in scenarios |
|---|---|---|---|---|---|
| short (15) | 6ix.cpp | 87 | 12 / 2 / 1 | 26 / 30 | 46 s |
| short (15) | vLLM | 93 | 13 / 2 / 0 | 28 / 30 | 40 s |
| hard (88) | 6ix.cpp | 78 | 60 / 17 / 11 | 137 / 176 | 705 s |
| hard (88) | vLLM | 77 | 61 / 14 / 13 | 136 / 176 | 381 s |

Hard set by category, points (6ix.cpp, then vLLM): A 4/6, 6/6 · B 6/6, 6/6 · C 8/8, 7/8 · D 5/6,
5/6 · E 5/6, 5/6 · F 6/6, 4/6 · G 4/6, 6/6 · H 9/10, 10/10 · I 15/20, 15/20 · J 5/6, 5/6 ·
K 20/26, 19/26 · L 7/8, 5/8 · M 5/6, 4/6 · N 6/6, 5/6 · O 6/12, 10/12 · P 26/38, 24/38.

No scenario failed because the engine refused a request. The hard set took nearly twice as long on
6ix.cpp because each token costs about 24 ms against 14 ms, and 6ix.cpp ran 33 more turns.

## Why this model is slower than the 80B on the same engine

Coder-Next has the 80B's architecture and size (512 experts, top 10, hybrid linear attention and
attention; the rope base differs). A pass costs the same. The difference is the draft head.

- The 80B checkpoint carries one (`mtp.*`, 1,553 tensors in `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`)
  and its template runs it (`"mtp": true, "mtp_depth": 2`): a pass can yield more than one token.
- No Coder-Next checkpoint in the fleet's cache has one: `Qwen/Qwen3-Coder-Next-FP8`,
  `RedHatAI/Qwen3-Coder-Next-NVFP4` and `Intel/Qwen3-Coder-Next-int4-AutoRound` all have 0 draft
  tensors (safetensors headers, 2026-10-04). The bind check reports `+ 0 draft`, the plan `mtp off`.
- The figures fit: Coder is 24.2 ms a token, one token a pass, 41 tok/s. The 80B's 82 tok/s on prose
  and 104 on code is 2.0–2.5 tokens a pass at that pass cost. This is arithmetic, not a measurement:
  the 80B has not been run here with the draft off.

Two things follow, and neither is comfortable:

- On a plain pass vLLM is faster than 6ix.cpp for this architecture (14.3 ms against 24.2 ms a
  token, different quantizations). 6ix.cpp's speed on the 80B and the 35B comes from using the draft
  head, which the fleet's vLLM 80B launch cannot (it crashes on that checkpoint).
- Coder-Next cannot get that gain on any engine until a checkpoint with a draft head exists.

Not measured: how much of the 24 ms is the checkpoint dense form. The 80B serves the `fp8` form;
Coder cannot, because that form fails the gate.

## The family: which checkpoints have a draft head, and the speeds on record

Draft-head tensors, from the safetensors headers in DGXone's cache on 2026-10-04:

(Later the same day these were removed from that cache to free disk: `Qwen/Qwen3-Coder-Next-FP8`,
`Qwen/Qwen3.5-122B-A10B-FP8`, `nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4`,
`Sehyo/Qwen3.5-35B-A3B-NVFP4`, `RedHatAI/gemma-4-26B-A4B-it-NVFP4`. The counts below are what
their headers said before that.)

| Has a draft head | None |
|---|---|
| Qwen3-Next-80B (nvidia NVFP4 1,553; Qwen FP8 3,096) | Qwen3-Coder-Next (FP8, NVFP4, int4) |
| Qwen3.6-35B-A3B (Qwen FP8 1,560) | Qwen3-235B-A22B (nvidia NVFP4) |
| Qwen3.5-122B-A10B (Sehyo NVFP4 785; Qwen FP8 1,560; int4 785) | MiniMax-M2.7 (nvidia NVFP4) |
| Qwen3.5-35B-A3B (Sehyo NVFP4 785) | Nemotron-3-Nano-30B (nvidia NVFP4) |
| Qwen3.5-0.8B / 2B / 4B / 9B (15 each) | Gemma-4-26B (both NVFP4 releases) |
| Qwen3.8-27B (Qwen FP8 22) | Qwen3-VL-30B (NVFP4) |
| Qwen3.8-Flash-Next (RadixArk NVFP4 31) | |
| Nemotron-3-Super-120B (nvidia NVFP4 1,040) — no 6ix.cpp kernels yet | |
| DeepSeek-V4-Flash (4,705) | |

Single-stream tok/s on record. "measured" is this fleet with `bench_decode` (prose / code);
"published" is the engine author's own figure under their conditions; the checkpoints differ
between engines in several rows, so only the measured same-file rows are engine against engine.

| Model | 6ix.cpp | Atlas | vLLM |
|---|---|---|---|
| Qwen3-Next-80B | 82.2 / 103.6 measured | 80.9 / 115.5 measured, same file; ~104 published | ~35, fleet record, draft head off |
| Qwen3.6-35B-A3B | 97.7 / 120.1 measured; ShareGPT 96.7 | ShareGPT 84.8 measured, same file, thinking on (`dgxone_35b_6ixcpp_vs_atlas_2026-10-04.md`); 116.5 published with one draft token, thinking off | 50.1 / 50.1 measured, FP8 slot |
| Qwen3.5-122B-A10B | ShareGPT 36.5 measured, Sehyo NVFP4, 65,536-token pool (`sixlabs/ports/qwen3.5-122b-a10b/results-2026-10-04-dgxone.md`) | 33.4 on one Spark, ~51 on two, published; not runnable beside DGXone's services | no speed on record |
| Qwen3.5-35B-A3B | not set up | ~133 published | none |
| Qwen3.5-0.8B | not run | recipe exists, no speed published | no speed on record |
| Qwen3.8-27B | not run here | 23.6 published, with the draft head | none |
| Qwen3.8-Flash-Next | 42.3–50.0 one Spark, 62.6–74.6 two (upstream figures) | 16.5; 19.1 with one draft token, one Spark, published | none (SGLang over two boxes: 43.5) |
| Qwen3-Coder-Next | 40.4 measured (ShareGPT) | ~58 published, FP8, one request at a time | 67.1 measured (ShareGPT); 70.6 / 70.7 `bench_decode` |

At more than one stream the 80B on 6ix.cpp reaches 118.5 / 185.4 / 253.4 at 2 / 4 / 8; Atlas was
measured at 108.4 / 120.1 at 2 / 4 with 4 seats.

## Gate

- Prose passes in both dense forms. The original code prompt fails the greedy condition only: row
  22 in both forms, row 8 in the checkpoint form. Correlation and mean are inside the gate.
- Row 22 is reproduced by a reference that changes only the router's rounding to BF16 (the engine
  then passes against it). With residual and sublayer rounding added, the reference goes back.
- Three further code prompts pass outright in the checkpoint form (mean 0.070 / 0.070 / 0.089, no
  clear-margin miss). The `fp8` form is outside the mean bound on two of them (0.109 average over
  five code prompts) with one new clear-margin miss: it is not served.
- A control prompt (the original's first 22 tokens, then a different task) repeats row 8 and has
  its own miss at row 28, margin 0.56. Every layer passes the one-step check; no reference variant
  lands on the engine's token. Open.

The model was booted on the owner's exception for the form that passes ("Gate exception: code
prompt, position 22, checkpoint dense form"). The registry keeps it at `compiled`, not `working`.

## Defects found

- **Fixed (`4a4ef27`):** a message whose `content` is an array of text parts was refused with
  `400 the chat template rejected these messages` by templates that concatenate `message.content`
  (this model's). `vllm bench serve` sends that form, so ShareGPT hung at its first request; coding
  harnesses send it too. The service now renders text-only arrays as their texts joined when the
  template rejects the array.
- **Open:** after a tool call, the model on 6ix.cpp does not give the structured JSON answer and
  calls tools until the turn budget (tool-eval TC-65, 66, 67, 69); vLLM answers in two turns. Not
  reproduced by hand, cause not established.
- **Tooling:** `run-sharegpt-bench.sh` could not send a key to a keyed slot (fixed in the fleet
  repo, `ea1979a`; `run_fleet_suite.py` passes it since `5c9a62e`). Against a served alias the
  ShareGPT harness needs `TOK=<Hugging Face repo id>` or it dies offline on the tokenizer lookup.
- **Stale record:** the fleet's note of 80–93 tok/s with "MTP depth 2" for the vLLM Coder slot was
  not reproduced, and that checkpoint has no draft tensors.

## Not run

`bench_decode`, DecodeBench and llama-benchy on 6ix.cpp (owner's instruction: ShareGPT then tool
calling only); the `fp8` form; the same NVFP4 file on vLLM, which is the comparison that would
isolate the engine; the 80B with its draft head off.
