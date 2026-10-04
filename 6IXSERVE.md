# 6ix.cpp

Named 6ixServe, then 6ixInfer and Infer, until 2026-10-04; the server binary is still `6ix-Serve` and
this file keeps its name.

6ixLabs' inference serving engine for NVIDIA DGX Spark (GB10): one box first, the same build across
two or more over RoCE.

**6ix.cpp is built on [DGPP](https://github.com/HawkBearPig/dgpp)** (Apache-2.0, © its authors — see
`LICENSE` and `NOTICE`, which stay as they are). `main` here is DGPP's `master` plus 6ixLabs' work;
the engine keeps DGPP's names (`dgpp-serve`, `scripts/dgpp-cluster`, the `DGPP_*` settings) so
upstream changes keep merging cleanly. Everything in `README.md` and `docs/` is DGPP's own
documentation and applies unchanged.

## What 6ixLabs added

**Qwen3-Next-80B-A3B** (`nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`), which DGPP did not serve
(upstream issue HawkBearPig/dgpp#77). It runs as a dialect of DGPP's `qwen3_5` family — the same
walk with the routed MoE in the dense MLP's place — on one Spark today:

- flat `qwen3_next` config, the checkpoint's tensor table (297,728 tensors), a loader for its fused
  GDN projections and modelopt NVFP4 matrices, 8-bit dense projections at load
  (`engine.dense_weights: "fp8"`), speculative decoding through the checkpoint's MTP head;
- the JSON tool-call form this model's chat template uses, in the parser and the forced-call grammar;
- `apps/qwen35_bind_check`, `apps/qwen35_forward_check`, and `tools/qwen3next_reference.py`, a
  pure-numpy reference of the model reading the same checkpoint.

**In the shared engine** (every model family gets these):

- the server process is `6ix-Serve` (`dgpp-serve` stays beside it as a link);
- the Prometheus exposition at `/metrics/prometheus` (upstream PR #81, merged upstream on 2026-10-04
  with peer metrics configured through the deployment JSON: `"ports": {"metrics": N}`);
- interleaved prefill for the `qwen3_5` family (`engine.prefill_budget_tokens`);
- two scheduler fixes for a K/V pool full of cached prefixes: a live request evicts idle cache
  entries before anything is shed, and the block arithmetic counts the draft rows a reserve takes
  (`sixlabs/bench/pool_pressure.py` reproduces the condition);
- `engine.logprobs_mode` — sampled requests report logprobs under the model's raw distribution
  (OpenAI's and vLLM's meaning) instead of the truncated one the draw was made from; `raw` is the
  default on one node. Detectors that read top-N logprobs (CIFF, HIFF) need it.

## Measured (one DGX Spark, 2026-10-03)

8-bit dense and head, MTP depth 2, eight request slots, a 262,144-token pool, 57.9 GiB planned.

| Concurrent streams | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| tok/s, all streams together | 81 (prose) – 102 (code) | 116 | 147 | 178 | 223 | 256 |

- Prefill: about 2,700 tok/s to 33K prompt tokens, about 1,400 at 134K–184K; a 249K-token prompt
  fits and all five planted facts in it are retrieved.
- Boot: about 80 s cold, 12 s from the resident image.
- Correctness: teacher-forced log-probabilities within 0.06 nat (mean) of the numpy reference on
  eight texts; greedy output with MTP identical to plain decode on 18 of 18 prompts.

## Not done yet

Two or more boxes for this model: upstream's `qwen3_5` family runs on two and four Sparks since
2026-10-04 (#86), but only its dense dialect (Qwen3.8-27B); the routed-MoE dialects (this model, the
35B, the 122B) are refused above one box (`Qwen35Model`: "run at world 1 only"); the FP8 release
of the same model as a second format; a deployment template in `deploy/`; a model-level fixture test.

## Running it

As DGPP's `docs/getting-started.md` describes, with a world-1 deployment JSON naming
`nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`. Build the server with
`cmake --build --preset release --target dgpp_serve_app`.
