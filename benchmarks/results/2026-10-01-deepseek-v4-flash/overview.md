# Benchmarks

Measured 2026-10-02 (UTC), using source revision `904f76c8519a` and one release binary across the cluster. All results on this page come from this campaign. [Raw results, exact configurations and commands](../benchmarks/results/2026-10-01-deepseek-v4-flash/README.md).

[Performance](#serving-performance) · [Configuration](#hardware-and-configuration) · [Method](#workload-and-timing) · [Decode modes](#decode-modes) · [Quality](#quality-and-correctness) · [Long context](#long-context) · [Reproduce](#reproduce)

## Serving performance

Rates are tokens per second. Each range spans the five prompt classes' medians, with three repetitions per class. **C** is concurrent requests. The single-request column measures engine decode; the loaded column measures completed output over full request wall time, including admission and prefill. The two columns have different timing scopes.

Rows are grouped by model family and node count, with configuration options next to each other throughout this page. **Nodes** is the tensor-parallel world size. **KV** is the shared key/value-cache token pool; **slots** is the configured concurrent-request limit. In configuration labels, K = 1,024 tokens.

| Model / weights | Nodes | Options | C1 engine tok/s | Loaded wall tok/s | Cold prefill seconds: ~2K / ~8K / ~32K |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 65.4–125.9 | C6: 134.0–186.0 | 1.359 / 5.471 / 28.022 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 40.4–72.8 | C4: 67.5–101.1 | 2.029 / 8.015 / 38.026 |

Rows cover the checked-in deployment templates, the GLM Flash FP8 checkpoint, and the three configuration variants described below. For Qwen NVFP4, the Options column identifies the dense projection format; the expert weights remain NVFP4 in both cases. All Qwen NVFP4 rows map the n-gram table from NVMe. YaRN 512K denotes the extended-context configuration.

The two-node GLM-5.3-Flash NVFP4/FP8 rows use the same checkpoint: one has a 163,840-token FP8 KV pool and four request slots; the other has a 262,144-token FP8 KV pool and two slots. NVFP4/FP8 identifies the [mixed-weight checkpoint](model_cards/GLM-5.3-Flash-NVFP4-FP8.md), which combines NVFP4 main-stack routed experts with the remaining tensors from the FP8 release, retaining their original formats.

## Hardware and configuration

The cluster consists of four GB10 systems: MSI EdgeXpert (rank 0), ASUS GX10 (rank 1), NVIDIA DGX Spark (rank 2), and AI TOP ATOM (rank 3). Each has about 121 GiB of OS-visible unified memory and uses the 200 Gb/s RoCE fabric. Measurements use one, two or four nodes, CUDA 13.0, driver 580.173.02 and the CMake `release` preset.

Only one benchmark uses the cluster at a time; no profiler is attached. Models and decode graphs are loaded before timed requests. Rank 0 and the local microbenchmarks set `CUDA_DEVICE_MAX_CONNECTIONS=32`; peer processes inherit their login environment. The nodes differ in firmware and some OS packages; the active kernel, GPU driver, CUDA and RDMA versions match. [Full environment comparison](../benchmarks/results/2026-10-01-deepseek-v4-flash/environment.md).

Four-node launches after Node 3's second recovery also collect host memory and pressure approximately every second, GPU state approximately every five seconds, and Node 3's live kernel journal. Node 3's telemetry uses its separate Wi-Fi management link. These probes run during timing; earlier launches have no continuous telemetry.

MTP is speculative multi-token prediction. A pass can commit several output tokens. DeepSeek uses its configured adaptive depth schedule. KV capacity below is the deployment's shared token pool; it is not a promise that every concurrent request can use that full context.

| Model / weights | Nodes | Options | Request slots | KV pool tokens | KV format | Default MTP depth |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 6 | 1,048,576 | model default | 5 (adaptive) |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 4 | 1,048,576 | model default | 5 (adaptive) |

GLM-5.3-Flash FP8 uses the four-node Flash template with the FP8 model ID and `kv_capacity=393216`. The alternative Qwen BF16 row selects `dense_weights=checkpoint` and `fp8_head=gemv`; full GLM's FP8 KV row selects `kv_dtype=fp8`, `kv_capacity=212992`, and a 1.5 GiB prefix cache; the two-node GLM-5.3-Flash NVFP4/FP8 256K KV row selects two slots, `kv_capacity=262144`, and a 2 GiB prefix cache (the 160K KV row uses 1.5 GiB). Every effective configuration is saved in the [configuration matrix](../benchmarks/results/2026-10-01-deepseek-v4-flash/matrix.json).

## Workload and timing

- **Decode:** the repository's prose, code, JSON, math and chat corpus; 256 maximum output tokens, three repetitions. Greedy decoding chooses the highest-scoring next token (temperature zero). Sampled decoding draws from the token distribution (temperature one here); other sampling parameters retain each model's server defaults, recorded in its log. Thinking is disabled where the model exposes that setting; GLM-5.3 uses its default reasoning mode.
- **Concurrency:** C1/C2/C4, plus the template's maximum if it exceeds four; the two-slot templates use C1/C2. Requests within a phase have distinct prompts, and each concurrency uses a different subset of its class. Repetitions reuse prompts with the configured prefix cache enabled. Loaded wall rates therefore describe this repeated-request workload.
- **Cold prefill:** three different prompts per target length, constructed from the prepared GSM8K text with a distinct prefix. Each sample must report zero cached tokens. Targets are approximate; actual tokenizer counts, engine prefill time and client time to first token are retained.
- **Engine decode rate:** `1000 × (tokens_generated − prompts_prefilled) / step_ms`. It excludes prefill and the first token emitted by each prefill. `ms/pass = step_ms / decode_steps`; at C1, `tokens/pass = decode_tokens / decode_steps`. Scheduler counters are reconciled against completed requests.
- **Client wall rate:** total completion tokens divided by the interval from launching a request group to its last completion. An SSE update may contain multiple tokens; update counts are never used as token counts.

The [detailed tables](../benchmarks/results/2026-10-01-deepseek-v4-flash/README.md) include every measured concurrency, sampled results, per-class engine pass times, time to first token, and the minimum/median/maximum across greedy repetitions. Ranges in the overview are variation across prompt classes, not confidence intervals.

## Single-request decode by class

Engine tokens/s, greedy; median of three repetitions.

| Model / weights | Nodes | Options | Prose | Code | JSON | Math | Chat |
|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 65.4 | 113.3 | 125.9 | 94.2 | 70.3 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 40.4 | 61.7 | 72.8 | 58.6 | 43.7 |

## Decode modes

Single-request engine tokens/s, shown as the range of the five class medians. Plain decoding has MTP disabled. The deeper run uses depth two, or fixed depth five for DeepSeek. Full GLM's depth-two and DeepSeek's depth-five runs use two request slots. Other settings are saved with each run.

| Model / weights | Nodes | Options | Plain | Template default | Deeper MTP |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 43.5–43.7 | 65.4–125.9 | 56.7–128.3 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 29.9–30.0 | 40.4–72.8 | 32.9–75.2 |

A dash means that deeper MTP was not part of that deployment's mode sweep. Prompt-dependent acceptance is included in these rates; pass times and committed tokens per pass are recorded separately.

## Quality and correctness

All deployments use all 164 HumanEval problems, the same first 300 GSM8K test problems and 100 deterministic schema-extraction records. Evaluation concurrency equals each deployment's request-slot count. Temperature is zero; reasoning effort is low with thinking enabled and none with thinking disabled. The generation cap is 2,048 tokens (512 for extraction), and thinking uses the mode described above.

HumanEval executes generated code in a container without network access or host-file mounts. Scores are specific to the repository's chat prompts, execution wrapper and token budgets. The HumanEval wrapper executes the returned code block independently; omitted context helpers can cause a failure, as observed for `encode_cyclic` in HumanEval/38. Raw responses and execution errors are retained.

| Model / weights | Nodes | Options | HumanEval | GSM8K | Schema extraction | Responses at token cap |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 139/164 | 296/300 | 100/100 | 0 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 141/164 | 295/300 | 100/100 | 0 |

Complete matching operation streams were collected for 12/12 recorded launches. The detailed record reports greedy plain/MTP transcript comparisons and solo/batched text checks for each deployment, including any failures.

| Model / weights | Nodes | Options | Rank checks passed | Plain/default greedy match | Solo/batched greedy text |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 6/6 | 5/5 classes | identical |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 6/6 | 5/5 classes | identical |

A class counts as a plain/default match only when all three repetitions in both modes produce the same text and token count. A differing transcript fails the exact-text check. Current dense kernels can use different floating-point reduction orders for different batch shapes, so identical greedy text across batch sizes is not guaranteed. A text mismatch alone does not establish request-state contamination. Rank operation-stream agreement checks execution order, not numerical equality or request isolation; single-node launches have only one stream to record.

For MiMo-V2.6-Flash every deployment's solo and batched decode transcripts matched (the isolation column above). The check does not measure logit error or prove isolation for all workloads. Performance tables use the default kernel setting.

## Long context

Each deterministic parcel document is generated once cold and twice more with the prefix cache enabled. Decode columns are medians over those three 256-token requests; cold prefill is one sample. These requests use the model's default reasoning mode. These are timing measurements; retrieval accuracy is measured separately.

| Model / weights | Nodes | Options | Prompt tokens | Cold prefill (s) | Decode ms/pass | Tokens/pass | Engine tok/s |
|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 3,844 | 2.297 | 37.83 | 3.00 | 78.7 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 32,438 | 26.277 | 39.14 | 2.97 | 75.8 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 121,443 | 199.030 | 45.28 | 3.40 | 75.1 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 3,844 | 3.348 | 60.14 | 2.90 | 47.6 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 32,438 | 35.443 | 64.40 | 3.49 | 54.3 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 121,443 | 234.959 | 66.06 | 3.04 | 45.8 |


## Selection microbenchmarks

The Qwen QSA and DeepSeek CSA2 benchmarks run on an idle GB10, using CUDA graphs, five warmups and thirty timed iterations per shape. QSA checks scores and selections against a host oracle; CSA2 checks selection against a host sort of the production score keys, not an independent score-arithmetic oracle. Warm inputs are repeated; cold samples evict L2 before timing. These kernel timings are separate from end-to-end service throughput.

[Complete current selection timings and correctness results](../benchmarks/results/2026-10-01-deepseek-v4-flash/README.md#selection-microbenchmarks).

## Reproduce

Use the same revision, prepared checkpoints and cluster site settings. Stop any serving workload before the campaign; only one world can own the benchmark fabric at a time.

The following runs one deployment. Use a clean checkout of the recorded source revision for an exact code match.

```bash
cmake --preset release
cmake --build --preset release -j 4
python3 scripts/prepare_data.py download
export CUDA_DEVICE_MAX_CONNECTIONS=32
CONFIG=deploy/cluster_qwen-3.8-flash-next_nvfp4_w2.example.json
python3 scripts/dgpp-cluster up --config "$CONFIG"
python3 scripts/timed_load.py 127.0.0.1 18080 \
  --concurrency 1,2,4 --classes all --repeat 3 --max-tokens 256 \
  --json-out /tmp/decode.json
python3 scripts/serve_prefill_probe.py 127.0.0.1 18080 2048 8192 32768 \
  --repeat 3 --no-think --seed 7 --tag current-20260922 \
  --json-out /tmp/prefill.json
python3 scripts/dgpp-cluster down --config "$CONFIG"
```

Use the configured HTTP host/port if they differ. Omit `--no-think` for GLM-5.3. To repeat a cold-prefill run on the same server, change the tag so it cannot reuse a previous prefix. Sampled sweeps add `--temperature 1`. Mode sweeps use the saved configurations: plain decoding sets `mtp=false`, `mtp_depth=1` and `mtp_schedule=false`; deeper runs use the recorded depth, scheduling and slot settings.

The [campaign runner](../benchmarks/results/2026-10-01-deepseek-v4-flash/run.py), [isolated evaluator](../benchmarks/results/2026-10-01-deepseek-v4-flash/isolated_eval.py) and [long-context client](../benchmarks/results/2026-10-01-deepseek-v4-flash/long_context.py) preserve the full procedure. Quality evaluation requires Docker and the Python image pinned by both registry digest and local image ID in `manifest.json`. Every invocation, including the microbenchmark shapes, is recorded in `raw/**/*.command.json`. The runner is resumable and checks this campaign's binary hash; do not reuse its completed result directory for a new measurement.
