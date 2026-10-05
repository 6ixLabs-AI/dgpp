# DeepSeek-V4-Flash on 6ix.cpp across both boxes — DGXone + DGXtwo, 2026-10-04

> Layout note, 2026-10-04: moved from `sixlabs/bench/results/dgxone_dgxtwo_deepseek_v4_flash_2026-10-04.md` when the per-model folders under `models/` were made. Paths to other moved records were updated; nothing else was changed. Its data folder moved with it.

Status: **DONE** for ShareGPT and for MiaAI-Lab's benchmark method on 6ix.cpp (one lane and two
lanes with thinking off; two lanes with thinking on). **NOT RUN:** the vLLM six-seat baseline on
the same day (stopped before launch on the owner's instruction, 18:08 EDT) and the forward-check
gate on this fleet, so the registry keeps the checkpoint at `upstream`. The engine is down.

Every 6ix.cpp number below is quoted from a file in
`models/deepseek-v4-flash/two-box/dgxone_dgxtwo_deepseek_v4_flash_2026-10-04/`. Log timestamps are UTC
(20:52 UTC = 16:52 EDT).

## 1. Result

- It boots and serves over both boxes with a 262,144-token pool and six seats: 196 s on the first
  load, 24 s from the resident images, 90,406 MiB on each GPU.
- One stream decodes at 65.8–67.1 tok/s with thinking off and 52.5–57.0 with thinking on
  (MiaAI-Lab's method). ShareGPT, where the model thinks by default: 43.2 output tok/s.
- Total throughput stops growing at four streams, at about 105 tok/s (thinking off, short
  prompts). Per-stream speed falls 67 → 47 → 28 → 18.5 at 1 / 2 / 4 / 6 streams.
- A second RoCE lane changed nothing that can be measured in decode, and about 2 % in prompt
  reading.
- Against MiaAI-Lab's published vLLM figures for the same checkpoint, 6ix.cpp is behind at every
  concurrency, and far behind at six streams (71–105 tok/s total against 191). Section 6.

## 2. Setup

| | |
|---|---|
| Engine | `dgpp-serve 0.1.0+g17fbf9220a85 (git 17fbf9220a85, cuda 13.0)`; tree at `7916644` (a bench-script commit after the binary's) |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`, 155.4 GiB, one copy on each box |
| Config | `deploy/cluster_deepseek-v4-flash_mxfp4-fp8_w2.example.json` with three changes: `max_concurrency` 4 → 6, `kv_capacity` 1,048,576 → 262,144, `prefix_cache_gib` 8 → 1 (`engine_config.json` in the folder) |
| Draft head | depth 5 with the scheduled verify depth (`mtp_schedule_row_ms` 7.2, `mtp_schedule_base_ms` 31.2), as the template ships |
| K/V | BF16, 262,144 tokens shared by the six seats and the cached prefixes |
| Ranks | rank 0 DGXone `10.77.0.1` (HTTP `172.17.0.1:18190`), rank 1 DGXtwo `10.77.0.2` |
| One lane | `rocep1s0f0`, GID index 3 — `bus: rank 1 up — 1 peer(s) x 1 lane(s) … bulk pace 170.0 Gb/s per QP` |
| Two lanes | adds `roceP2p1s0f0` (10.78.0.1 / 10.78.0.2), GID index 3 — `bus: rank 0 up — 1 peer(s) x 2 lane(s) … bulk pace 85.0 Gb/s per QP` |
| Box state | DGXone: HEM, the 0.8B slot, reranker, embedder and Docling down. DGXtwo: the 80B down; the HEM container up and idle (727 MiB on the GPU) |

The checkpoint is served from a private hub folder on each box (`~/6ixinfer-out/hfcache-deepseek`,
`refs/main` → `9e165c30…`, the snapshot a symlink into the main cache): the main cache's
`refs/main` for this repo points at a stub snapshot and was left as it is. `DGPP_ENV_FILE` selects
the `.env` (`.env.w2-deepseek`, `.env.w2-deepseek-2lane`).

## 3. Load and memory

| | |
|---|---|
| First load | **196 s** (`rank 0 serving: ok (196s)`); rank 1 `model constructed in 181.0s`, 46 layers captured into its resident image |
| Warm load | **24 s** (`rank 0 serving: ok (24s)`, `boot 22.9s`); `46/46 layers present`, `model constructed in 16.1s` |
| Resident image | 77.4 GiB on each box |
| Plan, each rank | **84.20 GiB** (83.12 device + 1.09 pinned) + 4.00 headroom |
| Device memory | **90,406 MiB** (88.29 GiB) on each box — 5.2 GiB over the device plan |
| MemAvailable while serving | DGXone 16–17 GiB (under the fleet's 19.7 GiB floor), DGXtwo about 20 GiB |

Plan items (identical on both ranks): model weights 78.84; attention cache pool 1.71; attention
scratch 0.84; activations 0.54; draft head screen 0.25; MoE scratch 0.90; prefix cache 0.99;
engine 0.07; gemm workspace 0.06.

## 4. ShareGPT (one lane)

64 prompts a level, all 64 successful at every level. The harness sends no thinking switch and
this model thinks unless told not to, so this is a thinking-on run. Six seats: the 8-stream level
queues two requests, which is what its first-token times show.

| Streams | Output tok/s | First token median / P99, ms | Per token median / P99, ms | Duration, s |
|---|---|---|---|---|
| 1 | 43.24 | 370 / 912 | 20.12 / 27.23 | 356.38 |
| 2 | 53.74 | 408 / 1,783 | 32.55 / 48.42 | 286.74 |
| 4 | 63.19 | 501 / 1,987 | 56.61 / 78.32 | 243.89 |
| 8 | 66.24 | 4,196 / 25,950 | 80.30 / 113.94 | 232.65 |

## 5. MiaAI-Lab's method

`mia_bench_6ix.py` follows `scripts/benchmark-0731.py` from
github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark (MIT): the same prompt text and
instruction, temperature 0.6, top_p 0.95, a unique first line per request so nothing is reused,
decode rate = output tokens / (finish − first token). Two differences: the prompt is sized from
the engine's own token count (the original uses vLLM's `/tokenize`), and thinking is set per
request. `max_tokens` 2,048 (the original's default is 4,096). Prompts measured 300 / 2,092 /
8,236 tokens for the 256 / 2,048 / 8,192 targets.

### 5.1 Thinking off: one lane against two (three runs a case, the median run shown)

| Prompt | Streams | First token, s: 1 lane / 2 lanes | Prompt reading tok/s: 1 / 2 | Per stream tok/s: 1 / 2 | Total tok/s: 1 / 2 | Per-stream runs, 2 lanes |
|---|---|---|---|---|---|---|
| 256 | 1 | 0.42 / 0.42 | 713 / 712 | 67.1 / 65.8 | 63.6 / 62.5 | 65.0, 68.7, 65.8 |
| 256 | 2 | 0.66 / 0.65 | 456 / 463 | 47.5 / 46.8 | 89.2 / 87.8 | 46.6, 46.8, 47.2 |
| 256 | 4 | 1.14 / 1.15 | 264 / 261 | 28.0 / 28.0 | 104.5 / 104.4 | 28.0, 29.6, 27.7 |
| 256 | 6 | 1.66 / 1.61 | 181 / 187 | 18.7 / 18.5 | 105.2 / 103.3 | 18.4, 19.2, 18.5 |
| 2,048 | 1 | 2.00 / 1.97 | 1,048 / 1,059 | 66.5 / 66.7 | 52.9 / 53.1 | 65.0, 66.7, 67.5 |
| 2,048 | 2 | 3.76 / 3.72 | 556 / 562 | 45.8 / 45.6 | 67.9 / 68.2 | 45.4, 46.0, 45.6 |
| 2,048 | 4 | 7.28 / 7.21 | 287 / 290 | 28.0 / 27.2 | 79.8 / 78.4 | 28.0, 27.2, 26.9 |
| 2,048 | 6 | 11.05 / 10.73 | 189 / 195 | 18.5 / 18.1 | 78.9 / 77.4 | 17.3, 18.1, 18.6 |
| 8,192 | 1 | 7.66 / 7.52 | 1,075 / 1,095 | 64.8 / 66.7 | 32.9 / 33.8 | 65.4, 66.7, 68.2 |
| 8,192 | 2 | 15.20 / 14.88 | 542 / 553 | 45.8 / 45.8 | 38.8 / 39.1 | 45.8, 45.4, 46.2 |
| 8,192 | 4 | 33.21 / 32.71 | 248 / 252 | 27.3 / 27.9 | 39.4 / 40.1 | 28.4, 27.9, 15.2 |
| 8,192 | 6 | 30.99 / 30.65 | 274 / 277 | 10.4 / 10.5 | 35.5 / 35.7 | 10.5, 10.5, 10.5 |

No request was cut at the token cap; every answer was 512–517 tokens.

**The second lane:** per-stream decode moved between −2.9 % and +2.9 %, inside the run-to-run
spread of a single configuration. Prompt reading at 8,192 tokens rose 1.9 % (one stream) and
2.0 % (two). With two lanes the engine paces each queue pair at 85 Gb/s instead of 170: the same
total.

**Two things in this table are not explained:** at 8,192 tokens and six streams, decode falls to
10.4–10.5 tok/s a stream on both lane settings and in all three runs; and one run in three at
8,192 × 4 decodes at 15.2–15.5 instead of 27–28, also on both. Decode there is measured from each
stream's first token, while other streams may still be reading their prompts. Not investigated.

### 5.2 Thinking on (two lanes, one run a case)

| Prompt | Streams | First token, s | Per stream tok/s | Total tok/s | Output tokens (median) | Of which reasoning | Cut at 2,048 |
|---|---|---|---|---|---|---|---|
| 256 | 1 | 0.43 | 52.5 | 51.2 | 922 | 417 | 0 of 1 |
| 256 | 2 | 0.66 | 42.1 | 69.2 | 906 | 458 | 0 of 2 |
| 256 | 4 | 1.12 | 23.5 | 74.2 | 1,736 | 1,288 | 1 of 4 |
| 256 | 6 | 2.95 | 14.2 | 71.0 | 2,003 | 1,811 | 3 of 6 |
| 2,048 | 1 | 1.97 | 57.0 | 54.0 | 2,048 | 2,048 | 1 of 1 |
| 2,048 | 2 | 3.64 | 39.5 | 71.7 | 2,027 | 1,771 | 1 of 2 |
| 2,048 | 4 | 6.80 | 21.6 | 60.2 | 1,521 | 1,270 | 2 of 4 |
| 2,048 | 6 | 9.07 | 15.3 | 61.5 | 838 | 378 | 2 of 6 |

`thinking: true` with no `reasoning_effort` — the engine's "low": the think block is opened and no
effort instruction is added. There is no thinking cap; several requests reasoned to the token
cap. One run a case, so no spread.

Engine counters over the two-lane boot (both sweeps above): 10,362 decode steps, 31,457 rows,
66,915 of 72,526 drafted tokens accepted (92.3 %).

## 6. Against vLLM — published and earlier figures, none measured today

**MiaAI-Lab, published** (their repository; same checkpoint, their two Sparks, their DSpark vLLM
image, six seats, 1M context, their script with its defaults):

| Prompt | Streams | Per stream tok/s | Total tok/s | 6ix.cpp per stream / total, thinking on | 6ix.cpp, thinking off |
|---|---|---|---|---|---|
| 256 | 1 | 75.4 | 69.1 | 52.5 / 51.2 | 65.8 / 62.5 |
| 256 | 2 | 58.3 | 104.9 | 42.1 / 69.2 | 46.8 / 87.8 |
| 256 | 4 | 46.8 | 164.5 | 23.5 / 74.2 | 28.0 / 104.4 |
| 256 | 6 | 36.9 | 191.2 | 14.2 / 71.0 | 18.5 / 103.3 |
| 2,048 | 1 | 68.8 | 62.0 | 57.0 / 54.0 | 66.7 / 53.1 |
| 2,048 | 2 | 57.0 | 97.6 | 39.5 / 71.7 | 45.6 / 68.2 |
| 2,048 | 4 | 44.0 | 154.7 | 21.6 / 60.2 | 27.2 / 78.4 |
| 2,048 | 6 | 34.7 | 143.7 | 15.3 / 61.5 | 18.1 / 77.4 |
| 8,192 | 1 / 2 / 4 / 6 | 73.9 / 49.8 / 37.4 / 23.6 | — | not run | 66.7 / 45.8 / 27.9 / 10.5 per stream |

Their script sends no thinking switch, so their figures were taken with thinking on at their
server's default (their compose file defaults to effort `low`, their example settings file to
`max`; no thinking cap; `max_tokens` 4,096). The thinking-on column is the like-for-like one.
Their own later baseline (a different checkpoint build, their PR #195) was 56 tok/s at one
stream, and that PR says decode was 15–25 % under their 14 August figures.

**This fleet's vLLM, earlier** (`6ixlabs` repo, `cluster/recipes/deepseek-v4-flash/README.md`,
2026-09-02; same checkpoint and boxes, 1M context, six seats, 0.835 of memory): with
MiaAI-Lab's short benchmark (thinking off, 128 output tokens) 69.8 tok/s at one stream, 58.6 at
two, 41.5 a stream and about 175 total at six. With `benchmark-0731.py`, total tok/s for the
stock and the fleet's tuned settings: 109.2 / 130.3 at 256 × 6, 86.2 / 98.9 at 256 × 4,
100.3 / 114.1 at 2,048 × 6, 78.8 / 76.4 at 8,192 × 4, 81.2 / 81.8 at 8,192 × 6.

**Reading:** at one stream 6ix.cpp is level with vLLM on these boxes when thinking is off (66–67
against 69.8). Above one stream it is behind: vLLM's total keeps growing to six streams, 6ix.cpp's
stops at four. At 8,192-token prompts and four or six streams 6ix.cpp delivers about half of
vLLM's total (36–40 against 76–82). These comparisons span different days, context sizes and
thinking settings: they show the shape, not a ratio to quote.

## 7. Start-up notes

- **The first start failed.** The fleet agent's probe (it requests `/metrics` from every listening
  port) reached rank 0's journal port during the peer handshake and rank 0 aborted
  (`journal: bad hello … GET /metrics … curl/8`; that log was overwritten by the next boot). Used
  since: stop `serveraihub-agent`, `serveraihub-agent-docker` and `saih-telemetry` on DGXone and
  `serveraihub-agent`, `serveraihub-agent-fabric` and `saih-telemetry` on DGXtwo before `up`, start
  them again once the log shows the bus up. A fix on the agent's side is in progress.
- **Start from an OpenSSH session** (`ssh dgxone 'ssh -o BatchMode=yes mark@10.77.0.1 "…"'`): a
  Tailscale SSH session has an 8 MiB memlock limit and RDMA registration fails.
- **The second lane** is NetworkManager connection `cx7-cluster-2` on each box
  (`enP2p1s0f0np0`, 10.78.0.1 and 10.78.0.2), made permanent by the owner.
- `down` ends with `op streams: identical across 2 ranks`.

## 8. Not run

- vLLM with six seats on the same day. For whoever runs it: MiaAI-Lab's profile as shipped (1M
  context, 0.835 of memory, about 101.6 GiB a box) does not fit beside DGXone's services above
  the out-of-memory guard's line; the fleet's launch tree is `~/dspark-2x`, where `ENV_FILE`
  selects the settings; six seats at 262,144 tokens and 0.75 of memory matches this record's
  context.
- The forward-check gate on this fleet.
- tool-eval-bench (owner: not needed).
- The 32K and 128K prompt lengths of MiaAI-Lab's default sweep; thinking on at 8,192 tokens;
  repeats of the thinking-on cases.
- Eight seats.

## 9. Files

| File | What |
|---|---|
| `sharegpt.log` | the ShareGPT run (section 4) |
| `mia-bench-6ixcpp-w2-1lane-thinking-off.json`, `mia-bench-6ixcpp-w2-1lane.log` | one lane, every request (section 5.1) |
| `mia-bench-6ixcpp-w2-2lane-thinking-off.json`, `mia-bench-6ixcpp-w2-2lane.log` | two lanes (section 5.1) |
| `mia-bench-6ixcpp-w2-2lane-thinking-on.json`, `mia-bench-6ixcpp-w2-2lane-thinking-on.log` | thinking on (section 5.2) |
| `mia_bench_6ix.py` | the benchmark script |
| `engine_config.json` | the deployment config as run |
