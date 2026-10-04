# Qwen3.5-122B-A10B NVFP4 on one Spark — 6ix.cpp (measured) against Atlas (published figure) (DGXone, 2026-10-04)

Status: **DONE.** 6ix.cpp: gate passed 4 of 4, booted, ShareGPT 1/2/4/8 measured, engine down. Atlas: NOT run
for this model, on the owner's instruction ("skip atlas for the 122B use the published number"); its column is
the figure from its own recipe. No tool-eval (not wanted). Every number is quoted from a log in this folder.

Checkpoint: `Sehyo/Qwen3.5-122B-A10B-NVFP4`, snapshot `56a6bdda33285ba2d5688e4f71f6c714649497b4` (75.9 GiB, 3
safetensors files) — "checkpoint B" of `sixlabs/ports/qwen3.5-122b-a10b/gpu-steps.md` and the file Atlas's
single-node recipe names. Nothing was downloaded.

Engine: `dgpp-serve 0.1.0+g29f904b8eaa7.dirty (git 29f904b8eaa7, cuda 13.0)`; the 122B port (3df149b) is an
ancestor of that commit; `qwen35_bind_check` and `qwen35_forward_check` were already built (09:07 today), nothing
was built or edited.

## 1. The table

| | 6ix.cpp — measured | Atlas — **published, not measured here** |
|---|---|---|
| source | `fleet-suite-122b-6ixcpp.log` (13:01:39–13:24:00, `DONE`, err=0) | recipe `qwen3.5/qwen3.5-122b-a10b-nvfp4-single.yaml`, its description |
| context | 65,536 tokens (`kv_capacity` 65536, BF16 pool) | 16,384 (`max_model_len`) |
| seats | 8 | `max_batch_size: 1` (`max_num_seqs: 4`): one request at a time |
| speculation | draft head on, depth 2; accepted 39,160 of 50,558 drafted = **77.5 %** (position 1 85.9 %, position 2 69.1 %) | none (no `speculative` / `num_drafts` in the recipe) |
| thinking | on (engine default; the smoke reply was 8 reasoning tokens) | not set in the recipe (model default) |
| memory | **86,941 MiB** device (84.90 GiB); about 89.6 GiB of MemAvailable in all (see section 4) | `gpu_memory_utilization: 0.92` = 112.0 GiB budget of the 121.7 GiB box; "~1.5–2 GB headroom remains for KV cache" |
| cold load | **176 s** (`rank 0 serving: ok (176s)`, `boot 174.7s`; encodes the dense matrices to FP8 and writes a 67.4 GiB resident image) | not published |
| warm load | **26 s** (`boot 24.6s`, from the resident image) | no equivalent |
| **ShareGPT 1 stream** — output tok/s | **36.49** | **33.4 tok/s** "single-call decode … at batch=1" (its own measurement, not ShareGPT) |
| successful / TTFT median, P99 / TPOT median, P99 | 64 / 426.56, 1839.32 ms / 24.61, 34.04 ms | — |
| **ShareGPT 2 streams** — output tok/s | **46.86** | one request at a time |
| successful / TTFT median, P99 / TPOT median, P99 | 64 / 465.54, 1875.50 ms / 37.50, 47.04 ms | — |
| **ShareGPT 4 streams** — output tok/s | **60.04** | one request at a time |
| successful / TTFT median, P99 / TPOT median, P99 | 64 / 666.21, 2227.33 ms / 56.73, 72.90 ms | — |
| **ShareGPT 8 streams** — output tok/s | **71.15** | one request at a time |
| successful / TTFT median, P99 / TPOT median, P99 | 64 / 1069.18, 2797.64 ms / 94.98, 118.62 ms | — |
| benchmark duration at 1 / 2 / 4 / 8 | 438.31 / 341.33 / 266.41 / 224.82 s | — |

How to read it:

- **Only the 1-stream row has an Atlas figure, and it is not the same measurement.** Atlas's 33.4 tok/s is its
  published single-call decode rate; 6ix.cpp's 36.49 is ShareGPT output throughput, which includes prompt
  reading. 6ix.cpp's decode alone at 1 stream is 24.61 ms a token (median TPOT) = 40.6 tok/s. On those figures
  6ix.cpp is ahead by 9 % (ShareGPT throughput) to 22 % (decode rate) at one stream, with 4× the context and with
  the draft head; I did not measure Atlas, so this is a comparison against its claim, not a head-to-head.
- **Why Atlas was not run:** by its own recipe it commits about 110 GiB before KV (0.92 × 121.7 = 112.0 GiB
  budget, "~1.5–2 GB headroom remains for KV cache"); 102.9 GiB was available with the three residents up, so it
  cannot load beside them at all. Its batch size of 1 would also have queued the 2/4/8-stream levels.
- 6ix.cpp's TTFT here is a real first-token time (it holds its role chunk until a token is ready). It is long:
  0.43 s median at 1 stream and 1.07 s at 8, P99 1.8–2.8 s. The engine's prefill line explains it: `prefill budget
  256 tokens/tick (0 = full prompt), 256 with nothing decoding; 1 decode pass(es) per chunk while both are in
  flight; equal shares` — the build's default for the qwen3_5 family (it refuses the explicit prefill settings).
- Same total output at every level (tok/s × duration = 15,994–15,996 tokens), 64 of 64 successful everywhere,
  0 ERROR lines in the serve log. One run; no run-to-run spread.

## 2. Bind check (headers only) — OK

```
config: qwen3_5, 48 layers (36 GDN + 12 full attention) + 1 draft, hidden 3072, vocab 248320
mlp: routed MoE, 256 experts top-8, intermediate 1024, shared expert 1024
checkpoint: 3 shards, 149885 tensors in headers
binding: expected 149552 | matched 149552 (missing 0, dtype 0, shape 0) | unexpected 0 | out of scope 0 | vision skipped 333
quantized matrices: 37056
binding OK: every tensor of the table is present with its dtype and shape
```

These are the numbers `gpu-steps.md` predicted for checkpoint B.

## 3. The numerical gate — PASS, 4 of 4 (before any speed number)

`gate-122b.log` (job `run_gate_122b.py`, 12:41:10–12:46:16, `DONE`, err=0). Numpy reference
(`tools/qwen3next_reference.py score`, full depth, CPU): code 113 tokens `mean_logprob=-1.32105 seconds=118.54
peak_rss_gb=22.35`; prose 87 tokens `mean_logprob=-1.64789 seconds=88.79 peak_rss_gb=22.44`.
`qwen35_forward_check` (streaming) compared by `sixlabs/bench/compare_forward_check.py`, default bounds
(pearson ≥ 0.995, mean ≤ 0.10 nat, greedy agreement above a 0.5 nat margin). Verdict lines, all four:
`PASS (gate: pearson >= 0.995, mean <= 0.1, greedy agreement above a 0.5 nat margin; reference logprob >= -15; greedy rows from 2)`

| input, dense form | mean \|Δ\| | median | max (position) | pearson | greedy rows agree | verdict | original 0.06 / 0.5 gate (for reference) |
|---|---|---|---|---|---|---|---|
| code, checkpoint | 0.02206 | 0.00085 | 0.23456 (50) | 0.999839 | 112/113 (miss at a 0.0463 nat margin) | **PASS** | would pass |
| prose, checkpoint | 0.04076 | 0.00997 | 0.26082 (28) | 0.999471 | 87/87 | **PASS** | would pass |
| code, `fp8` (the serving form) | 0.04126 | 0.00132 | 0.97546 (96) | 0.999035 | 112/113 (same near tie) | **PASS** | would fail (max) |
| prose, `fp8` (the serving form) | 0.05338 | 0.02167 | 0.53060 (60) | 0.999091 | 86/87 (miss at a 0.0254 nat margin) | **PASS** | would fail (max) |

No far-tail positions were excluded in any run (0 of 112 / 0 of 86). Not run: the 4-layer staged state
comparison, the reference's `--variant` collapse checks, the A-against-B reference comparison (checkpoint A is not
in the cache), and MTP-against-plain-decode transcripts — the gate that was asked for is the table above.

## 4. Memory — the planner understated the real cost by about 9 GiB

Planner (`6ix-Serve --memory-plan`, loads nothing), 65,536 context, prefix cache 1 GiB, fp8, draft depth 2, 8 seats:
`model weights (resident) 70.33 GiB; loader staging … 2.88 GiB (2.88 GiB pinned); kv pool 1.63 GiB; layer scratch
0.66 GiB; activations 0.09 GiB; moe scratch … 1.03 GiB (0.01 GiB pinned); dense fp8 prefill bridge … 0.09 GiB;
blockwise fp8 lm head … 0.71 GiB; mtp scratch 0.19 GiB; gdn states 1.14 GiB; spec snapshot rows 3.43 GiB; prefix
cache 0.86 GiB; engine 0.08 GiB` → `memory plan total 83.12 GiB (79.30 GiB device + 3.82 GiB pinned) + 4.00 GiB
headroom … the memory plan fits`. With the 2.88 GiB of staging freed after load that predicts about 80.2 GiB
resident and about 22 GiB left.

Measured:

| config | load | device (nvidia-smi) | MemAvailable, engine down → up (settled) | real cost | plan (minus staging) |
|---|---|---|---|---|---|
| 8 seats (cold boot, 12:46:52) | 176 s | 86,941 MiB = 84.90 GiB | 104.8 (read right after the down; 102.6 by the engine before the boot, with the checkpoint in the page cache) → 15.0–15.3 GiB | **≈ 89.6 GiB** (87.5 against the pre-boot reading) | 80.24 GiB |
| 4 seats (warm, 12:51:44) | 32 s | 84,541 MiB = 82.56 GiB | 104.9 → 18.3–18.4 GiB | ≈ 86.5 GiB | 77.94 GiB |
| 8 seats (warm, 13:00:19 — the measured boot) | 26 s | 86,941 MiB | 105.3 → 15.60 GiB | ≈ 89.7 GiB | 80.24 GiB |

**The planner's 83.12 GiB understated the real cost: about 89.6 GiB of MemAvailable at 8 seats, 9.4 GiB more
than the plan's resident figure and 6.5 GiB more than its total.** (The 35B was 2.55 GiB over its plan and
Coder-Next about 1.8 GiB.) Device memory alone is 84.90 GiB against a planned 79.30 GiB device.

The fleet's 19.7 GiB floor was therefore not met at 8 seats (15 GiB) or at 4 seats (18.3 GiB). I took the engine
down and reported; Mark waived the floor for this run (8 seats as specified, no resident stopped). During
ShareGPT, **lowest MemAvailable sampled: 12,900,432 kB = 12.30 GiB** (13:09:35, at the start of the 2-stream
level; the dips coincide with the harness container starting each level). earlyoom's own once-a-minute log has a
minimum of 12,772 MiB in the run window. It never fell below the 11.5 GiB alert line, and **earlyoom killed
nothing** (0 kill lines since 13:00).

What would have met the floor, by the planner relative to the measured 4-seat point (estimates, not booted):
8 seats with the draft head off ≈ 20–21 GiB left; 4 seats, draft off ≈ 21.6; 4 seats + 32,768 context + no prefix
cache ≈ 20.0 (marginal). With the draft head on, 4 seats at depth 1 (≈ 18.9) and 2 seats (≈ 19.2) do not.
The full-context plan (262,144 tokens, prefix cache 4 GiB: total 91.00 GiB) was not booted.

## 5. Settings

Site config `~/6ixinfer-ports/deploy/cluster_qwen3.5-122b-a10b_nvfp4_w1.json` (copy in this folder) = the
template `cluster_qwen3.5-122b-a10b_nvfp4_w1.example.json` with three changes: `model`
`Sehyo/Qwen3.5-122B-A10B-NVFP4` (template: the nvidia repo), `kv_capacity` 65536 (template 262144),
`prefix_cache_gib` 1 (template 4). HTTP 172.17.0.1:18190.

```json
"engine": {
  "max_concurrency": 8, "kv_capacity": 65536, "kv_dtype": "bf16", "dense_weights": "fp8",
  "default_max_tokens": 32768, "queue_limit": 64, "max_connections": 256, "decode_graph": true,
  "mtp": true, "mtp_depth": 2, "prefix_cache_gib": 1, "admission": "grow", "admission_window": 256,
  "model_alias": "Qwen3.5-122B-A10B", "sampling_candidates": 128, "stats_interval_s": 10,
  "max_tokens_overflow": "clamp"
}
```

No `prefill_*` / `decode_passes_per_prefill` keys (this build refuses them for the qwen3_5 family). The engine's
config line for the measured boot: `conc=8 kv=65536 kvdt=bf16 … dw=fp8 … mtp=1 mtpd=2 … pcgib=1 … pfbudget=-1`.
Suite command: `TOK=Sehyo/Qwen3.5-122B-A10B-NVFP4 python3 sixlabs/bench/run_fleet_suite.py --host 172.17.0.1
--port 18190 --model Qwen3.5-122B-A10B --maxc 8 --only sharegpt`.

Atlas recipe as published (for the record; not launched): `model: Sehyo/Qwen3.5-122B-A10B-NVFP4`, `container:
azeezish/atlas-gb10:latest`, `max_model_len 16384`, `kv_cache_dtype fp8`, `kv_high_precision_layers auto`,
`gpu_memory_utilization 0.92`, `scheduling_policy slai`, `max_batch_size 1`, `max_num_seqs 4`, `oom_guard_mb
1024`, `ssm_cache_slots 0`, `tool_call_parser qwen3_coder`; description: "Verified single-call decode 33.4 tok/s
at batch=1; 4-way concurrent requests serve cleanly per Atlas QUICKSTART."

## 6. What contaminated, failed, or was not done

- **Other GPU traffic during the run** (three `nvidia-smi pmon` samples per poll, about once a minute):
  `BGErerankerOne` at 9 % at 13:09:4x and 6 % at 13:10:46 (start of the 2-stream level) and 9 % at 13:14:53 (end
  of the 2-stream level), with `6ix-Serve` at 86–88 % in those samples instead of 95 %; `DGXembedOne` at 1 % at
  13:18:00 and 13:19:02 (4-stream level). The 1-stream and 8-stream levels had no other GPU user in my samples.
  The 2-stream figure (46.86) is the one most likely to be slightly low.
- **Memory floor waived** for this run (section 4); the run sat at 12.3–15.6 GiB MemAvailable.
- **Three boots**: 8 seats cold (took it down: under the floor), 4 seats warm (still under; taken down), 8 seats
  warm (measured). No failed launch.
- **Atlas not run** (owner's instruction); no 2/4/8 Atlas figures exist.
- No tool-eval, no DecodeBench/llama-benchy (ShareGPT only, as instructed). No cold-load repeat.
- **Disk:** the 122B resident image is 72,422,764,800 bytes (`~/6ixinfer-out/resident-cache/d608ed4ab0ffae99.img`);
  `/` went from 196 GiB to 129 GiB free (97 % used).
- **Left on the box:** no engine, no Atlas, ports 18190/18290 closed, 105 GiB available. Running containers = the
  11:48 baseline minus `HEMnliOne` and `qwen3.5-0.8b` (both taken down on Mark's instruction by the coordinator);
  reranker, embedder, docling up. Repository `~/6ixinfer-ports` untouched (same `git status` as at the start; the
  two site JSON files are ignored by git).

## 7. Files in this folder

`fleet-suite-122b-6ixcpp.log` (the numbers); `gate-122b.log`, `run_gate_122b.py`; `up-122b-6ixcpp-cold.log`,
`serve-122b-6ixcpp-cold-8seats.log`; `up-122b-6ixcpp-warm-4seats.log`, `serve-122b-6ixcpp-warm-4seats.log`;
`up-122b-6ixcpp-warm-8seats.log`, `serve-122b-6ixcpp-bench-1300.log` (the measured boot);
`cluster_qwen3.5-122b-a10b_nvfp4_w1.json`; `containers-end.txt`. On DGXone: gate outputs in
`~/6ixinfer-out/q122/`, ShareGPT result `~/bench-results/sharegpt/sharegpt-bench-20261004-130139.txt`.
