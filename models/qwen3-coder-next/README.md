# Qwen3-Coder-Next

It boots, serves and has been measured on one box, **and it is slower than the fleet's vLLM slot
at every concurrency.** It is not verified: it serves under a gate exception the owner accepted.
Every figure below is quoted from a record in this repository. Registry entry: `qwen3-coder-next`
in [`sixlabs/registry/models.json`](../../sixlabs/registry/models.json).

## What it is

- Family `qwen3_next`, the 80B's architecture and size: 48 layers (36 Gated DeltaNet + 12 full
  attention), hidden 2048, a routed MoE of 512 experts, top 10 (the bind check's lines, in the
  [bring-up record](one-box/results-2026-10-04-dgxone.md), B.1).
- Tool calls in the template's `<function=...>` XML form. It does not reason.
- **No draft head.** None of the three checkpoints the fleet had carries one (0 draft tensors in
  the FP8, NVFP4 and int4 releases, against 1,553 in the 80B's), so there is no speculative
  decoding for this model on any engine.

| Checkpoint | Revision | Format | Tensors | Registry status |
|---|---|---|---|---|
| [`RedHatAI/Qwen3-Coder-Next-NVFP4`](https://huggingface.co/RedHatAI/Qwen3-Coder-Next-NVFP4) (the default) | `27a8f16f463b9a13c91c332c40cf93e09717347e` | nvfp4-compressed-tensors | 296,151 | `compiled` |
| [`Intel/Qwen3-Coder-Next-int4-AutoRound`](https://huggingface.co/Intel/Qwen3-Coder-Next-int4-AutoRound) (what the fleet's vLLM slot serves) | — | int4-autoround | — | `refused` |

## Status

The registry keeps this checkpoint at `compiled`, and says why in its note: "Booted, served and
measured on DGXone 2026-10-04 in the checkpoint dense form, but NOT verified: the forward-check
gate fails on the original code prompt (greedy condition, rows 8 and 22) and it was served under
a gate exception; a control prompt's row 28 is unexplained. The fp8 dense form is outside the
gate's mean bound on code prompts and must not be served."

So: **measured, not verified.** (The registry has no status of that name; its definition of
`compiled` says "Never loaded weights or produced a token", which this checkpoint has done. The
status is left as the registry has it.)

## One box

Template [`deploy/cluster_qwen3-coder-next_nvfp4_w1.example.json`](../../deploy/cluster_qwen3-coder-next_nvfp4_w1.example.json):
8 seats, a 524,288-token pool, `"dense_weights": "checkpoint"`, no draft layer, the tuned
prompt-reading settings. Records: the [summary](one-box/dgxone_coder_next_2026-10-04.md) and the
full [bring-up record](one-box/results-2026-10-04-dgxone.md). DGXone, 2026-10-04, one run each.

The two engines did **not** run the same file: 6ix.cpp refuses the int4 checkpoint the vLLM slot
serves. Every comparison below is engine plus quantization, not engine alone.

### Load and memory

| | 6ix.cpp, NVFP4 | vLLM, int4 AutoRound |
|---|---|---|
| Load, cold | 86 s (serve log: 84.9 s) | 363 s |
| Load, warm (resident image) | 12 s (serve log: 10.9 s) | not measured |
| Memory planned | 64.51 GiB (59.30 device + 5.21 pinned) | 53.7 GiB (the launcher's footprint figure) |
| Memory measured | 67,889 MiB = 66.30 GiB | 54,362 MiB = 53.09 GiB |

### Speed

ShareGPT, 64 prompts a level, all 64 successful at every level on both, output tok/s:

| Streams | 6ix.cpp | vLLM | 6ix.cpp / vLLM | Per token median / P99, ms: 6ix.cpp | vLLM |
|---|---|---|---|---|---|
| 1 | 40.37 | 67.10 | 0.60 | 24.19 / 25.37 | 14.25 / 15.11 |
| 2 | 65.57 | 107.26 | 0.61 | 28.78 / 31.29 | 17.26 / 23.07 |
| 4 | 97.58 | 148.70 | 0.66 | 38.19 / 43.19 | 24.72 / 31.50 |
| 8 | 124.12 | 192.91 | 0.64 | 58.41 / 71.85 | 38.17 / 44.24 |

Time to first token is the same on both engines; the gap is generation speed.

`bench_decode.py`, DecodeBench and llama-benchy were not run on 6ix.cpp (the owner's
instruction: ShareGPT and tool calling only). On the vLLM slot `bench_decode.py` gave 70.6 prose
and 70.7 code at one stream, and 96.8 / 151.7 / 259.2 at 2 / 4 / 8.

tool-eval-bench:

| Set | 6ix.cpp | vLLM |
|---|---|---|
| short (15) | 87 (12 pass, 2 partial, 1 fail; 26/30) | 93 (13 pass, 2 partial, 0 fail; 28/30) |
| hard (88) | 78 (60 pass, 17 partial, 11 fail; 137/176); 705 s | 77 (61 pass, 14 partial, 13 fail; 136/176); 381 s |

No scenario failed because the engine refused a request.

**Published by others, not measured here:** Atlas, about 58 tok/s on `Qwen/Qwen3-Coder-Next-FP8`,
one request at a time (the Atlas recipes README, quoted in the registry).

### Why it is slower than the 80B on the same engine

From the [summary](one-box/dgxone_coder_next_2026-10-04.md): a pass costs the same as the 80B's.
The 80B's checkpoint has a draft head and its template runs it, so a pass can yield more than one
token; Coder-Next yields one token a pass, 24.2 ms a token, 41 tok/s. That record marks this as
arithmetic, not a measurement: the 80B has not been run here with its draft off.

On a plain pass vLLM is faster than 6ix.cpp for this architecture (14.3 ms against 24.2 ms a
token, different quantizations). How much of the 24 ms is the checkpoint dense form was not
measured.

## Two boxes

Not run. There is no two-box template for this model.

## The gate, and the exception

From the [bring-up record](one-box/results-2026-10-04-dgxone.md), section 4. The gate tool was
not edited; its printed verdict on the original code prompt is still FAIL.

| Input | Checkpoint dense form (served) | `fp8` dense form (refused) |
|---|---|---|
| prose | PASS | PASS |
| original code prompt | **FAIL**: greedy misses at row 8 (margin 0.882) and row 22 (margin 1.165) | **FAIL**: row 22 |
| three more code prompts | PASS, 3 of 3 (mean 0.070 / 0.070 / 0.089) | 1 of 3; mean 0.140 and 0.125 are outside the bound |

- Correlation and mean are inside the gate on all four original runs. The failure is the greedy
  condition only.
- Row 22 is reproduced by a reference that changes only the router's rounding to BF16; the engine
  then passes against that reference. With residual and sublayer rounding added as well, the
  reference goes back. The record states both.
- One step per layer: every layer and the head within about 1.7 BF16 floors of the reference, the
  same as the 80B, against 130–600 floors for a wrong convention. No wrong tensor was found.
- The owner accepted the checkpoint dense form on that evidence ("Gate exception: code prompt,
  position 22, checkpoint dense form"). The cost is 1.57 GiB more weights than the `fp8` form.

## Did not work, and why

- **The `fp8` dense form is not served.** Over five code prompts its mean averages 0.109 nat
  against the gate's 0.10, with one clear-margin miss the router convention does not explain.
  It was the template's default until this run; the template now says `checkpoint`.
- **`Intel/Qwen3-Coder-Next-int4-AutoRound` is refused by name.** 6,407 tensor names are
  duplicated across two shards and every dense projection is GPTQ int4 with no exact resident
  form (registry). This is the file vLLM serves, which is why there is no same-file comparison.
- **ShareGPT hung at its first request.** The chat template concatenates `message.content` as a
  string and the service handed it OpenAI's content-part array: HTTP 400. Fixed in `4a4ef27`.
- **The fleet's older note for the vLLM slot (80–93 tok/s, "MTP depth 2") was not reproduced**,
  and that checkpoint has no draft tensors.

## Open

- **The control prompt's row 28**: a greedy miss at a 0.56 nat margin that no reference variant
  reproduces. Every layer of that row passes the one-step check. The further reference variants
  were not run.
- **After a tool call the model loops instead of giving the structured JSON answer** (tool-eval
  TC-65, 66, 67, 69); vLLM answers in two turns. Not reproduced by hand, cause not established.
- The same NVFP4 file on vLLM, which is the comparison that would isolate the engine: not run.
- `sixlabs/ports/qwen3-coder-next/gpu-steps.md` still describes the old bounds in its body; its
  header carries the update.

## Templates

| Use | Template |
|---|---|
| One box, checkpoint dense form | [`cluster_qwen3-coder-next_nvfp4_w1.example.json`](../../deploy/cluster_qwen3-coder-next_nvfp4_w1.example.json) |

The ShareGPT harness needs `TOK=RedHatAI/Qwen3-Coder-Next-NVFP4` against the served alias.

## Records in this folder

- `one-box/`: [summary](one-box/dgxone_coder_next_2026-10-04.md) (also holds the table of which
  checkpoints have a draft head, across the family),
  [bring-up record](one-box/results-2026-10-04-dgxone.md),
  [gate-coder-4-runs-2026-10-04.txt](one-box/gate-coder-4-runs-2026-10-04.txt) (the four
  verdicts).
- Bring-up steps and gate inputs: `sixlabs/ports/qwen3-coder-next/`
  ([gpu-steps.md](../../sixlabs/ports/qwen3-coder-next/gpu-steps.md)).
