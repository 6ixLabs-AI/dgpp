# Qwen3.6-35B-A3B NVFP4 — 6ix.cpp against Atlas, same checkpoint (DGXone, 2026-10-04)

> Layout note, 2026-10-04: moved from `sixlabs/bench/results/dgxone_35b_6ixcpp_vs_atlas_2026-10-04.md` when the per-model folders under `models/` were made. Paths to other moved records were updated; nothing else was changed. The logs and launch scripts it lists as "in this folder" (section 4) are not in this repository; it says the full logs are on DGXone under `~/6ixinfer-logs/`.

Status: **DONE** for ShareGPT on both engines. Tool-eval: 6ix.cpp complete; Atlas short complete, hard stopped
at 24 of 88 on the owner's instruction ("we do not need the tool calling numbers", 12:35). Both engines are down.
Every number below is quoted from a log in this folder.

Checkpoint on both sides: `nvidia/Qwen3.6-35B-A3B-NVFP4`, snapshot
`…/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5` (21.8 GiB) —
both engines' logs name this same directory. Thinking ON on both sides. One run each; no run-to-run spread.

## 1. The table

| | 6ix.cpp | Atlas |
|---|---|---|
| engine | `dgpp-serve 0.1.0+g29f904b8eaa7.dirty (git 29f904b8eaa7, cuda 13.0)` | image `azeezish/atlas-gb10:latest`, id `sha256:faa6e5820c42dd86d0ae9e12cbaf2a2e9f32d0f7a9ab348a75ee54d4253929a6` (created 2026-08-15) |
| cold load | not re-measured today (the morning's quiet-box figure: 47.5 s, no resident image) | **49.1 s** from `docker run` (12:20:17.48) to `Server live and ready` (12:21:06.54); it reads the checkpoint on every start. First request answered at +59.8 s (I sent it 10 s after ready) |
| warm load | **10 s** (`rank 0 serving: ok (10s)`, `boot 9.2s`, from the resident image) | no equivalent |
| device memory (nvidia-smi) | **39,478 MiB** (38.55 GiB) | **73,861 MiB** (72.13 GiB) after load, 73,901 MiB at the end |
| MemAvailable while serving | 53–57 GiB | 22.3–24.4 GiB |
| context / seats | 262,144-token K/V pool (BF16), 8 seats | `--max-seq-len 32768`, FP8 KV with 4 of 10 attention layers BF16, 810,992 KV tokens, max_batch 8 |
| speculation | draft head on, depth 2; accepted 87,762 of 108,934 drafted = **80.6 %** (position 1 87.7 %, position 2 73.4 %) over the whole boot | enabled, 1 draft (`Speculative decoding: ENABLED (1 drafts/step)`), gate `auto`; `/metrics`: verify accept 3,229 / reject 832 = 79.5 % of 4,061 verify steps, against 77,153 generated tokens — speculative steps covered a small minority of the decode |
| thinking | on (engine default, no cap) | on (`thinking_default=true`), capped: `max_thinking_budget=768` |
| **ShareGPT 1 stream** — output tok/s | **96.72** | 84.76 |
| successful requests | 64 | 64 |
| TTFT median / P99 (harness) | 150.14 / 507.84 ms | 2.19 / 4.15 ms — NOT a first-token time, see note 1 |
| TPOT median / P99 (harness) | 9.32 / 11.98 ms | 11.78 / 74.16 ms — includes the wait for the first token, see note 1 |
| **ShareGPT 2 streams** — output tok/s | **135.37** | 112.21 |
| successful requests | 64 | 64 |
| TTFT median / P99 (harness) | 154.35 / 767.64 ms | 3.50 / 5.51 ms (note 1) |
| TPOT median / P99 (harness) | 13.03 / 15.91 ms | 17.90 / 80.07 ms |
| **ShareGPT 4 streams** — output tok/s | **182.37** | 108.89 |
| successful requests | 64 | 64 |
| TTFT median / P99 (harness) | 181.20 / 920.50 ms | 3.25 / 7.07 ms (note 1) |
| TPOT median / P99 (harness) | 19.14 / 26.62 ms | 37.75 / 102.27 ms |
| **ShareGPT 8 streams** — output tok/s | **220.11** | 127.12 |
| successful requests | 64 | 64 |
| TTFT median / P99 (harness) | 331.70 / 1048.14 ms | 3.50 / 7.39 ms (note 1) |
| TPOT median / P99 (harness) | 31.32 / 46.60 ms | 62.13 / 128.43 ms |
| benchmark duration at 1 / 2 / 4 / 8 | 165.38 / 118.16 / 87.71 / 72.67 s | 188.71 / 142.55 / 146.89 / 125.83 s |
| tool-eval short | **93** — 15 scenarios, 13 pass, 2 partial, 0 fail; 28/30 points; 61 s; A:6/6 B:6/6 C:6/6 D:5/6 E:5/6 | **93** — 13 pass, 2 partial, 0 fail; 28/30; 58 s; A:6/6 B:6/6 C:6/6 D:5/6 E:5/6 |
| tool-eval hard | **89** — 88 scenarios, 72 pass, 12 partial, 4 fail; 156/176 points; 758 s; A:6/6 B:6/6 C:8/8 D:5/6 E:5/6 F:6/6 G:5/6 H:10/10 I:18/20 J:6/6 K:22/26 L:7/8 M:3/6 N:6/6 O:8/12 P:35/38 | **not run** — stopped at 24 of 88 on the owner's instruction (21 pass, 3 partial at the stop; not a score) |

**Which is faster (ShareGPT output tok/s):** 6ix.cpp at every level — 1.14× at 1 stream, 1.21× at 2, 1.67× at 4,
1.73× at 8. Both engines produced the same total output at every level (tok/s × duration = 15,995–15,996 tokens
on both sides), so the durations compare directly. **Load:** 6ix.cpp 10 s warm against Atlas 49.1 s; like for
like (no resident image) 6ix.cpp was 47.5 s this morning, i.e. level. **Memory:** Atlas held 1.87× what 6ix.cpp
held (73,861 against 39,478 MiB), with one eighth of the context. **Tool calling:** level on the short set (93 / 93).

### Note 1 — Atlas's 2–7 ms "TTFT" is its role chunk, not a token

One streamed request to Atlas at 12:35:40 (same request shape the harness sends), first SSE data lines:

```
+   21.29 ms  #1  … "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null,"logprobs":null}]}
+  186.06 ms  #2  … "choices":[{"index":0,"delta":{"reasoning_content":"Here's "},"finish_reason":null,"logprobs":null}]}
```

Atlas sends a role-only chunk as soon as it accepts the request; the first token arrives later (its own log for that
request: `Done: 64 tokens (length) 88.9 tok/s, TTFT=117.5ms`). The harness stamps the first chunk. 6ix.cpp holds its
role chunk until a token is ready (`generation_service.cpp`, `write_stream_event`: "Delay the preamble until an
actual completion chunk is ready"), so its harness TTFT is a real first-token time. Consequences:

- The harness TTFT columns are not comparable between the two engines.
- The harness TPOT is (request time − TTFT) / (tokens − 1); with a ~3 ms TTFT, Atlas's TPOT absorbs its
  first-token wait, which is why its P99 TPOT is 74–128 ms. Output tok/s and benchmark duration are unaffected.
- Atlas's own first-token time, from its `Done:` log lines grouped by level (its clock; level boundaries taken
  from the suite's elapsed times, so a request or two may sit in the neighbouring level):
  median / P99 = 190.3 / 431.9 ms (1 stream), 190.6 / 414.7 (2), 191.6 / 414.4 (4), 189.4 / 416.4 (8).
  That figure does not grow with concurrency, so it probably excludes time spent waiting to be scheduled; I could
  not check what it includes without reading Atlas source, which is not permitted.

### Note 2 — did Atlas really serve 8 at once? Yes.

`Scheduler started (batched mode, max_batch=8, mtp=true, ngram=false, num_drafts=1, policy=slai, chunked_prefill=true, max_prefill_tokens=8192)`
and, at the start of the 8-stream level, `Captured CUDA graph for batch size 8 (n=8, slots=Some([0, 1, 2, 3, 4, 5, 6, 7]))`
(16:30:39 UTC; batch-4 graphs at 16:28:03, batch-2 at 16:25:30). So requests were not queued above a cap. It
batched and still did not scale: its own per-request rates (median of the `Done:` lines) were 92.7 / 63.2 / 29.6 /
17.4 tok/s at 1 / 2 / 4 / 8 streams. Its boot line lists `ssm_batched_recurrent=false` (help text: "Batched
multi-sequence GDN recurrent decode kernel (default: off). One strided launch across the batch instead of one per
sequence"); the published recipe leaves that default. I did not test whether turning it on changes the scaling.

## 2. Settings

### 6ix.cpp

Site config `~/6ixinfer-ports/deploy/cluster_qwen3.6-35b-a3b_nvfp4_w1.json` — identical to the template
`cluster_qwen3.6-35b-a3b_nvfp4_w1.example.json` (the morning's settings); HTTP 172.17.0.1:18190:

```json
"engine": {
  "max_concurrency": 8, "kv_capacity": 262144, "kv_dtype": "bf16", "dense_weights": "fp8",
  "default_max_tokens": 32768, "queue_limit": 64, "max_connections": 256, "decode_graph": true,
  "mtp": true, "mtp_depth": 2, "prefix_cache_gib": 4, "admission": "grow", "admission_window": 256,
  "model_alias": "Qwen3.6-35B-A3B", "sampling_candidates": 128, "stats_interval_s": 10,
  "max_tokens_overflow": "clamp"
}
```

Prefill = the build's default, same as the morning's measurement. Engine line:
`rank 0: prefill budget 256 tokens/tick (0 = full prompt), 256 with nothing decoding; 1 decode pass(es) per chunk while both are in flight; equal shares`

**The registry's qwen3_5 prefill defaults are refused by build 29f904b.** With `prefill_budget_tokens` 1024,
`prefill_idle_budget_tokens` 4096, `decode_passes_per_prefill` 8, `prefill_order` shortest in the engine block
the engine exited in 60 ms: `ERROR --prefill-budget-tokens requires a Qwen or GLM-5.3-Flash graph engine`
(`serve-35b-6ixcpp-attempt1-refused.log`). An explicit budget is accepted only for the families `qwen4_exp`,
`glm5`, `qwen3_next` (`apps/dgpp_serve.cpp:2464`); this checkpoint is family `qwen3_5`. All four keys were dropped
(coordinator's ruling: run the verified template, not an untested three-key mix).

Memory plan: `36.00 GiB (30.05 GiB device + 5.96 GiB pinned) + 4.00 GiB headroom`. Defaults reported by
`/v1/models`: temperature 1, top_p 0.95, top_k 20, min_p 0, repetition_penalty 1. 0 ERROR lines in the serve log.

### Atlas

Recipe: `~/.cache/sparkrun/registries/atlas/recipes/qwen3.6/qwen3.6-35b-a3b-nvfp4.yaml` (`model:
nvidia/Qwen3.6-35B-A3B-NVFP4`, `container: azeezish/atlas-gb10:latest`). The serve command is sparkrun 0.3.6's own
rendering of that recipe (`sparkrun run … --dry-run`), launched with `docker run` and the fleet's compose
mechanics (network host, ipc host, HF cache mount, `HF_HUB_OFFLINE=1`), restart policy none
(`launch_atlas_35b.sh` in this folder):

```
spark serve nvidia/Qwen3.6-35B-A3B-NVFP4 --port 18290 --bind 172.17.0.1
  --gpu-memory-utilization 0.62 --max-seq-len 32768
  --kv-cache-dtype fp8 --kv-high-precision-layers auto
  --tool-call-parser qwen3_coder --disable-tool-grammar true
  --fp8-kv-calibration-tokens 256 --scheduling-policy slai
  --ssm-cache-slots 128 --ssm-checkpoint-interval 32
  --mtp-quantization bf16 --num-drafts 1 --enable-prefix-caching --speculative
```

As published (unchanged): `kv_cache_dtype fp8`, `fp8_kv_calibration_tokens 256`, `kv_high_precision_layers auto`,
`scheduling_policy slai`, `speculative true`, `num_drafts 1`, `mtp_quantization bf16`, `enable_prefix_caching true`,
`ssm_cache_slots 128`, `ssm_checkpoint_interval 32`, `tool_call_parser qwen3_coder`, `disable_tool_grammar true`.
The recipe sets no batch size; Atlas's default is 8.

Overrides, each with its reason:

| recipe | used | why |
|---|---|---|
| `host: 0.0.0.0` | `--bind 172.17.0.1` | fleet rule: nothing on 0.0.0.0. (sparkrun renders `--host`; this image's help documents `--bind`.) |
| `port: 8888` | `--port 18290` | free, not in `PORT-MANIFEST.txt`, not listening |
| `gpu_memory_utilization: 0.88` | `0.62` | must leave ≥ 19.7 GiB MemAvailable beside the residents. 0.5 was tried first and failed (below). At 0.62: `121.7 GB total × 62% util = 75.4 GB budget; 50.4 GB pre-KV + 14.2 GB reserve → 10.8 GB for KV → 50687 blocks × 16 tok/block = 810992 max KV tokens` |
| `max_model_len: 262144` | `--max-seq-len 32768` | enough for these benches; sized to the workload |
| `disable_thinking: true` | flag removed (thinking on) | coordinator's ruling: 6ix.cpp ran with thinking on, so the one Atlas pass matches it. The recipe gives no reason for the setting (a bare `disable_thinking: true` line, no comment; its description does not mention thinking) |

Checks before the suite, thinking on: "17 * 23" → `finish stop`, content `'391'`, reasoning_content 568 chars /
252 reasoning tokens; a tool request → `finish tool_calls`, `get_weather {"city":"Paris"}`, content null.

Atlas behaviours stated by its own log that differ from 6ix.cpp:
`Model behavior: max_thinking_budget=768, thinking_default=true` (reasoning capped at 768 tokens, or 90 % of
`max_tokens` when that is smaller) and `Default sampling: temperature=1, top_k=20, top_p=0.95, top_n_sigma=1, min_p=0.08`.

## 3. What contaminated, failed, or was not done

- **6ix.cpp start, attempt 1 (11:51:17):** refused the prefill keys (above). The launcher does not notice an
  exited engine and waited; interrupted at 11:52:36. The measured boot (11:53:31) was launched by the coordinator
  with the template config.
- **6ix.cpp load:** warm only. No cold load and no extra down/up today (the first `up` found the morning's resident
  image; removing it was not approved).
- **HEM** went from paused to stopped at 11:56:12 (coordinator, on Mark's instruction; a sub-second unpause before
  the stop), during 6ix.cpp's 1-stream level. GPU stayed at 93–96 % on `6ix-Serve` in every sample.
- **Atlas start, attempt 1 (12:18:13, fraction 0.5):** `Error: Failed to build model … No memory left for KV cache:
  total GPU = 121.7 GB, --gpu-memory-utilization 50% → budget 60.8 GB, but 50.2 GB already consumed + 14.2 GB
  inference reserve = 64.5 GB committed.` (`atlas-35b-serve-attempt1.clean-trimmed.log`). That container also carried
  `--disable-thinking`; it never served. Attempt 2 (0.62, thinking on) is the measured one.
- **Atlas ran with 22.3–24.4 GiB MemAvailable** (floor 19.7), 6ix.cpp with 53–57 GiB.
- **Other GPU traffic seen in my samples** (one `nvidia-smi pmon` sample per poll, about once a minute):
  during 6ix.cpp's suite none; after it finished, `DGXembedOne` 6 % at 12:16:51. During Atlas's suite:
  `BGErerankerOne` 1 % at 12:24:50 (1-stream level), `DGXembedOne` 1 % at 12:26:54 (2-stream level) and at 12:33:00
  (tool-eval short), `BGErerankerOne` 8 % at 12:34:02 (tool-eval hard). Small, and only Atlas's side saw any.
- **CPU:** a `~/6ixinfer-check` build (`build_dgpp.py -j 6`) ran from 11:47 and had ended by 11:53:54, before the
  first measurement; none was running at any later poll. `griff-ui` was restarted by someone else at about 11:53.
- **Atlas tool-eval hard:** stopped at 12:35:19 on the owner's instruction (suite runner pid 1214132 killed after
  reading its cmdline, container `tool-eval-bench-123343` stopped). 6ix.cpp's hard score has no Atlas counterpart.
- **Left on the box:** nothing of mine. Atlas container stopped and removed 12:36:29; no `6ix-Serve`; ports 18190
  and 18290 closed. Running containers = the 11:48 baseline minus `HEMnliOne` (stopped on Mark's instruction) and,
  from 12:36, minus `qwen3.5-0.8b` (taken down by the coordinator on Mark's instruction). `benches-done` was NOT
  touched (the hold job no longer exists). The 35B site config is back to the template; the morning's copy is also at
  `~/6ixinfer-out/q36/cluster_qwen3.6-35b-a3b_nvfp4_w1.json.morning-0356`.

## 4. Files in this folder

`fleet-suite-35b-6ixcpp.log`, `fleet-suite-35b-atlas.log` (the suite logs the numbers come from);
`up-35b-6ixcpp-cold.log`, `serve-35b-6ixcpp-bench-1153.log` (the measured 6ix.cpp boot);
`up-35b-6ixcpp-attempt1.log`, `serve-35b-6ixcpp-attempt1-refused.log`;
`atlas-35b-serve.clean-trimmed.log`, `atlas-35b-serve-attempt1.clean-trimmed.log` (colour codes, per-layer load
lines and rendered prompts removed; full logs on DGXone under `~/6ixinfer-logs/`);
`launch_atlas_35b.sh`, `launch_atlas_35b.attempt1.sh`; `baseline-containers.txt`, `containers-now.txt`.
ShareGPT result files on DGXone: `~/bench-results/sharegpt/sharegpt-bench-20261004-115436.txt` (6ix.cpp),
`…-122201.txt` (Atlas). Tool-eval reports: `~/tool-eval-bench/report-20261004-120234.json`, `-120335.json`
(6ix.cpp), `-123244.json` (Atlas short).
