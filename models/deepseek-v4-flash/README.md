# DeepSeek-V4-Flash (0731)

Served and measured over both boxes on 2026-10-04. At one stream it is level with vLLM on these
boxes when thinking is off; above one stream it is behind. The model family is upstream DGPP's
work, not a 6ixLabs port. Every 6ix.cpp figure below is quoted from the
[two-box record](two-box/dgxone_dgxtwo_deepseek_v4_flash_2026-10-04.md). Registry entry:
`deepseek-v4-flash` in [`sixlabs/registry/models.json`](../../sixlabs/registry/models.json).

## What it is

- Family `deepseek_v4`, served by upstream DGPP. A thinking model; it thinks unless a request
  says not to.
- The checkpoint carries a draft head (4,705 draft tensors), run at depth 5 with the scheduled
  verify depth, as the template ships.

| Checkpoint | Revision | Format | Size | Registry status |
|---|---|---|---|---|
| [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` | mxfp4-fp8 | 155.4 GiB | `upstream` |

## Status

In the registry's words, `upstream` means: "Served by upstream DGPP with a template and published
measurements; not verified on this fleet by 6ixLabs."

**Measured on this fleet, not verified.** The registry's note: "Served and measured on this
fleet; the forward-check gate was not run here, so the status stays upstream."

## One box

Not applicable. The registry and `deploy/` have this model on two and four Sparks only.

## Two boxes

Template [`deploy/cluster_deepseek-v4-flash_mxfp4-fp8_w2.example.json`](../../deploy/cluster_deepseek-v4-flash_mxfp4-fp8_w2.example.json)
with three changes for this run: 6 seats instead of 4, a 262,144-token pool instead of 1,048,576,
a 1 GiB prefix cache instead of 8 (the config as run is in the data folder). DGXone + DGXtwo,
engine build `17fbf92`.

### Load and memory

| | Figure |
|---|---|
| First load | 196 s |
| Warm load (resident images) | 24 s |
| Resident image | 77.4 GiB on each box |
| Memory planned, each box | 84.20 GiB (83.12 device + 1.09 pinned) + 4.00 headroom |
| Memory measured, each box | 90,406 MiB = 88.29 GiB, 5.2 GiB over the device plan |
| MemAvailable while serving | DGXone 16–17 GiB (under the fleet's 19.7 GiB floor), DGXtwo about 20 GiB |

### Speed: ShareGPT

64 prompts a level, all 64 successful, one RoCE lane. The harness sends no thinking switch, so
this is a **thinking-on** run. Six seats: the 8-stream level queues two requests.

| Streams | Output tok/s | First token median / P99, ms | Per token median / P99, ms |
|---|---|---|---|
| 1 | 43.24 | 370 / 912 | 20.12 / 27.23 |
| 2 | 53.74 | 408 / 1,783 | 32.55 / 48.42 |
| 4 | 63.19 | 501 / 1,987 | 56.61 / 78.32 |
| 8 | 66.24 | 4,196 / 25,950 | 80.30 / 113.94 |

### Speed: MiaAI-Lab's method

`mia_bench_6ix.py` follows `scripts/benchmark-0731.py` from
[MiaAI-Lab's repository](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
(MIT). 256-token prompt target, two lanes; per-stream decode tok/s and total tok/s:

| Streams | Thinking off: per stream | total | Thinking on: per stream | total |
|---|---|---|---|---|
| 1 | 65.8 | 62.5 | 52.5 | 51.2 |
| 2 | 46.8 | 87.8 | 42.1 | 69.2 |
| 4 | 28.0 | 104.4 | 23.5 | 74.2 |
| 6 | 18.5 | 103.3 | 14.2 | 71.0 |

Thinking off is the median of three runs; thinking on is one run a case. Total throughput stops
growing at four streams. The record has the 2,048- and 8,192-token prompts as well.

A second RoCE lane changed nothing measurable in decode (−2.9 % to +2.9 %, inside the
run-to-run spread) and about 2 % in prompt reading at 8,192 tokens.

No tool-eval (the owner: not needed).

### Against vLLM

**No vLLM figure was measured on the same day.** The six-seat vLLM baseline was stopped before
launch on the owner's instruction.

| Source | Figures, tok/s | What it is |
|---|---|---|
| 6ix.cpp, thinking on | 52.5 at one stream; totals 69.2 / 74.2 / 71.0 at 2 / 4 / 6 | measured here, 2026-10-04 |
| 6ix.cpp, thinking off | 65.8 at one stream; totals 87.8 / 104.4 / 103.3 at 2 / 4 / 6 | measured here, 2026-10-04 |
| vLLM, MiaAI-Lab | 75.4 at one stream; totals 104.9 / 164.5 / 191.2 at 2 / 4 / 6 | **published by MiaAI-Lab, not measured here**: their two Sparks, their DSpark vLLM image, six seats, 1M context, thinking on at their server's default |
| vLLM, this fleet, earlier | 69.8 at one stream, 58.6 at two, 41.5 a stream and about 175 total at six | a fleet record of 2026-09-02 from the `6ixlabs` repository: MiaAI-Lab's short benchmark, thinking off, 128 output tokens, 1M context. A different day, context size and method |
| Upstream DGPP, the template as shipped | 40.4–72.8 at one stream; 67.5–101.1 at four requests | **published by upstream** (`docs/benchmarks.md`), not measured here |

The record's reading: level with vLLM on these boxes at one stream with thinking off (66–67
against 69.8); behind above one stream, because vLLM's total keeps growing to six streams and
6ix.cpp's stops at four. "These comparisons span different days, context sizes and thinking
settings: they show the shape, not a ratio to quote."

## Did not work, and why

- **The first start failed.** The fleet agent's probe requested `/metrics` from rank 0's journal
  port during the peer handshake and rank 0 aborted. Since then the fleet agents and telemetry
  are stopped on both boxes until the log shows the bus up.
- **A Tailscale SSH session cannot start it**: its 8 MiB memlock limit fails RDMA registration.
  Start from an OpenSSH session.
- **DGXone ran under the fleet's memory floor** (16–17 GiB available against 19.7).
- **Two things in the benchmark table are not explained**: at 8,192-token prompts and six streams
  decode falls to 10.4–10.5 tok/s a stream in all three runs, and one run in three at 8,192 × 4
  decodes at 15.2–15.5 instead of 27–28. Not investigated.

## Open

- The forward-check gate on this fleet.
- vLLM with six seats on the same day and at the same context.
- Eight seats; the 32K and 128K prompt lengths of MiaAI-Lab's sweep; thinking on at 8,192 tokens.

## Templates

| Use | Template |
|---|---|
| Two boxes, as shipped (4 seats, 1M context) | [`cluster_deepseek-v4-flash_mxfp4-fp8_w2.example.json`](../../deploy/cluster_deepseek-v4-flash_mxfp4-fp8_w2.example.json) |
| Two boxes, as run here (6 seats, 262,144-token pool) | [`engine_config.json`](two-box/dgxone_dgxtwo_deepseek_v4_flash_2026-10-04/engine_config.json) |
| Four Sparks (the fleet has two) | [`cluster_deepseek-v4-flash_mxfp4-fp8_w4.example.json`](../../deploy/cluster_deepseek-v4-flash_mxfp4-fp8_w4.example.json) |

## Records in this folder

- `two-box/`: [the two-box record](two-box/dgxone_dgxtwo_deepseek_v4_flash_2026-10-04.md) and its
  data folder (the ShareGPT log, every request of the three benchmark sweeps, the script and the
  config as run).
