# Benchmarks

Measured 2026-09-22 (UTC), using source revision `a3ed8994a34c` and one release binary across the cluster. All results on this page come from this campaign, except the MiMo-V2.6-Flash rows, which come from [a second campaign (2026-09-22/23)](../benchmarks/results/2026-09-22-mimo-v26-flash-opt/README.md) on the tree that adds the family and its decode / prefill kernels (revision `a895db96bb17` plus the changes its manifest records) with the same runner, workload and timing scope, and the DeepSeek-V4-Flash rows, which come from [a third campaign (2026-10-02)](../benchmarks/results/2026-10-01-deepseek-v4-flash/README.md) on the tree that adds that family (revision `904f76c8519a` plus the changes its manifest records), again with the same runner, workload and timing scope. [Raw results, exact configurations and commands](../benchmarks/results/2026-09-22-current/README.md).

[Performance](#serving-performance) · [Configuration](#hardware-and-configuration) · [Method](#workload-and-timing) · [Decode modes](#decode-modes) · [Quality](#quality-and-correctness) · [Long context](#long-context) · [Reproduce](#reproduce)

## Serving performance

Rates are tokens per second. Each range spans the five prompt classes' medians, with three repetitions per class. **C** is concurrent requests. The single-request column measures engine decode; the loaded column measures completed output over full request wall time, including admission and prefill. The two columns have different timing scopes.

Rows are grouped by model family and node count, with configuration options next to each other throughout this page. **Nodes** is the tensor-parallel world size. **KV** is the shared key/value-cache token pool; **slots** is the configured concurrent-request limit. In configuration labels, K = 1,024 tokens.

| Model / weights | Nodes | Options | C1 engine tok/s | Loaded wall tok/s | Cold prefill seconds: ~2K / ~8K / ~32K |
|---|---|---|---|---|---|
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 31.3–36.0 | C4: 56.3–61.8 | 2.380 / 9.204 / 39.950 |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 31.2–35.8 | C2: 43.5–46.6 | 2.367 / 9.180 / 39.596 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 54.0–61.8 | C4: 102.8–114.1 | 1.497 / 5.847 / 26.686 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 45.4–52.1 | C4: 73.1–78.4 | 1.714 / 6.508 / 29.140 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 42.3–50.0 | C4: 83.4–95.4 | 1.839 / 7.485 / 30.266 |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 31.9–38.4 | C4: 72.2–83.6 | 1.707 / 6.939 / 28.424 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 62.6–74.6 | C4: 124.5–143.8 | 1.311 / 5.188 / 20.871 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 64.5–75.8 | C2: 96.1–111.1 | 1.278 / 5.117 / 20.975 |
| Qwen3.8-Flash-Next FP8 | 2 | Default | 44.7–54.8 | C4: 89.3–101.7 | 1.311 / 5.178 / 21.110 |
| Qwen3.8-Flash-Next FP8 | 4 | Default | 65.3–83.3 | C4: 143.3–164.9 | 1.113 / 4.440 / 18.315 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | 44.7–71.5 | C4: 77.3–118.6 | 1.294 / 4.417 / 17.817 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3, prefill levers on (the template) | 45.9–71.5 | C4: 78.0–116.3 | 1.210 / 4.226 / 17.096 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 26.8–30.6 | C8: 51.1–54.2 | 4.785 / 24.209 / 169.138 |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 27.8–30.5 | C8: 51.4–54.8 | 4.747 / 24.735 / 186.045 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | 39.8–75.5 | C6: 85.3–123.4 | 1.602 / 5.866 / 27.087 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 40.4–72.8 | C4: 67.5–101.1 | 2.029 / 8.015 / 38.026 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 65.4–125.9 | C6: 134.0–186.0 | 1.359 / 5.471 / 28.022 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 42.3–49.2 | C4: 70.1–78.5 | 3.503 / 12.972 / 57.213 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 39.9–48.2 | C4: 72.1–78.5 | 3.486 / 13.201 / 58.513 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 75.8–85.8 | C4: 128.7–142.1 | 1.771 / 6.547 / 28.428 |

Rows cover the checked-in deployment templates, the GLM Flash FP8 checkpoint, and the three configuration variants described below. The AutoRound int4/int8 hybrid has two rows: the exact default chain, and the same deployment with the two prefill levers its template turns on (`engine.prefill_bf16_partials`, `engine.prefill_fp8_gemm`; not bitwise the default chain — see the README's accuracy and correctness section); the levers touch prefill-shaped launches only, and the levers-on decode cells were re-measured under them. The Qwen3.8-Flash-Next AutoRound int4/int8 row is the single-Spark hybrid (`Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN`, `deploy/cluster_qwen-3.8-flash-next_autoround-int4_w1.json`), measured 2026-09-29 on the `qwen-autoround-int4` branch with the head in bit planes and MTP depth 3 ([record](../benchmarks/results/2026-09-28-qwen-autoround-int4/README.md), `serve_docrows_r18/` for the decode cells and the long-context parcels, session U (round 18) for the prefill probe — after the wide packed-int expert GEMM, the prefill's combine inside the next norm pass, the batched page advice, the compact tile list over the routed segments and the wide kernel's L2 prefetch — round 15 for the task-level quality, `serve_docrows/` for the plain and deeper decode modes and the consistency checks); its optional draft vocabulary slice (`engine.draft_vocab`) is not in the row. Its decode step at context, measured through the endpoint after round 16's attention gather: 52.6 ms at under 200 prompt tokens, 56.3–56.5 ms from 12K to 47K (the sparse attention's selection budget is fixed, so its cost is), with prefill at 0.63–0.65 ms per token to 47K before round 17 (0.55 at 8K–32K after rounds 17–18). The MiMo-V2.6-Flash rows cover its two templates (four nodes with a 128K BF16 pool; two nodes with a 256K FP8 pool) and the two-node template's BF16-cache variant at 128K. The DeepSeek-V4-Flash rows are its two templates (the 0731 release as shipped, the model's full 1M-token context, the DSpark draft under the confidence-scheduled verify depth); its cached attention rows are BF16 rows holding the release's own FP8-quantized values, with no `kv_dtype` choice. The same four-node deployment under llama-benchy (pp2048 / tg128, the checkpoint's default sampling at temperature 1, thinking on) generates 64.3–65.1 tokens/s over three 40-run legs and processes the prompt at about 1,380 tokens/s ([record](../benchmarks/results/2026-10-01-deepseek-v4-flash/notes.md)). For Qwen NVFP4, the Options column identifies the dense projection format; the expert weights remain NVFP4 in both cases. All Qwen NVFP4 rows map the n-gram table from NVMe. YaRN 512K denotes the extended-context configuration.

The two-node GLM-5.3-Flash NVFP4/FP8 rows use the same checkpoint: one has a 163,840-token FP8 KV pool and four request slots; the other has a 262,144-token FP8 KV pool and two slots. NVFP4/FP8 identifies the [mixed-weight checkpoint](model_cards/GLM-5.3-Flash-NVFP4-FP8.md), which combines NVFP4 main-stack routed experts with the remaining tensors from the FP8 release, retaining their original formats.

## Hardware and configuration

The cluster consists of four GB10 systems: MSI EdgeXpert (rank 0), ASUS GX10 (rank 1), NVIDIA DGX Spark (rank 2), and AI TOP ATOM (rank 3). Each has about 121 GiB of OS-visible unified memory and uses the 200 Gb/s RoCE fabric. Measurements use one, two or four nodes, CUDA 13.0, driver 580.173.02 and the CMake `release` preset.

Only one benchmark uses the cluster at a time; no profiler is attached. Models and decode graphs are loaded before timed requests. Rank 0 and the local microbenchmarks set `CUDA_DEVICE_MAX_CONNECTIONS=32`; peer processes inherit their login environment. The nodes differ in firmware and some OS packages; the active kernel, GPU driver, CUDA and RDMA versions match. [Full environment comparison](../benchmarks/results/2026-09-22-current/environment.md).

Four-node launches after Node 3's second recovery also collect host memory and pressure approximately every second, GPU state approximately every five seconds, and Node 3's live kernel journal. Node 3's telemetry uses its separate Wi-Fi management link. These probes run during timing; earlier launches have no continuous telemetry.

MTP is speculative multi-token prediction. A pass can commit several output tokens. Both DeepSeek families use their configured adaptive depth schedule (DeepSeek-V4-Flash for sampled requests too; six concurrent requests there batch at up to four of the block's five drafts, the family's 32-row decode batch). KV capacity below is the deployment's shared token pool; it is not a promise that every concurrent request can use that full context.

| Model / weights | Nodes | Options | Request slots | KV pool tokens | KV format | Default MTP depth |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 4 | 163,840 | fp8 | 1 |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 2 | 262,144 | fp8 | 1 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 4 | 786,432 | bf16 | 1 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 4 | 393,216 | bf16 | 1 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 4 | 65,536 | bf16 | 1 |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 4 | 65,536 | bf16 | 1 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 4 | 262,144 | bf16 | 1 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 2 | 532,480 | bf16 | 1 |
| Qwen3.8-Flash-Next FP8 | 2 | Default | 4 | 262,144 | bf16 | 1 |
| Qwen3.8-Flash-Next FP8 | 4 | Default | 4 | 262,144 | bf16 | 1 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | 4 | 262,144 | bf16 | 3 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 8 | 122,880 | bf16 | 1 |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 8 | 212,992 | fp8 | 1 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | 6 | 131,072 | model default | 4 (adaptive) |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 4 | 1,048,576 | bf16 | 5 (adaptive) |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 6 | 1,048,576 | bf16 | 5 (adaptive) |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 4 | 131,072 | bf16 | 1 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 4 | 262,144 | fp8 | 1 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 4 | 131,072 | bf16 | 1 |

MiMo-V2.6-Flash's two-node BF16 KV row selects `kv_dtype=bf16` and `kv_capacity=131072` on its two-node template (whose shipped form keeps the K/V cache in the fp8 row form, 57 KiB per token per rank, for a 256K pool). GLM-5.3-Flash FP8 uses the four-node Flash template with the FP8 model ID and `kv_capacity=393216`. The alternative Qwen BF16 row selects `dense_weights=checkpoint` and `fp8_head=gemv`; full GLM's FP8 KV row selects `kv_dtype=fp8`, `kv_capacity=212992`, and a 1.5 GiB prefix cache; the two-node GLM-5.3-Flash NVFP4/FP8 256K KV row selects two slots, `kv_capacity=262144`, and a 2 GiB prefix cache (the 160K KV row uses 1.5 GiB). Every effective configuration is saved in the [configuration matrix](../benchmarks/results/2026-09-22-current/matrix.json).

## Workload and timing

- **Decode:** the repository's prose, code, JSON, math and chat corpus; 256 maximum output tokens, three repetitions. Greedy decoding chooses the highest-scoring next token (temperature zero). Sampled decoding draws from the token distribution (temperature one here); other sampling parameters retain each model's server defaults, recorded in its log. Thinking is disabled where the model exposes that setting; GLM-5.3 uses its default reasoning mode.
- **Concurrency:** C1/C2/C4, plus the template's maximum if it exceeds four; the two-slot templates use C1/C2. Requests within a phase have distinct prompts, and each concurrency uses a different subset of its class. Repetitions reuse prompts with the configured prefix cache enabled. Loaded wall rates therefore describe this repeated-request workload.
- **Cold prefill:** three different prompts per target length, constructed from the prepared GSM8K text with a distinct prefix. Each sample must report zero cached tokens. Targets are approximate; actual tokenizer counts, engine prefill time and client time to first token are retained.
- **Engine decode rate:** `1000 × (tokens_generated − prompts_prefilled) / step_ms`. It excludes prefill and the first token emitted by each prefill. `ms/pass = step_ms / decode_steps`; at C1, `tokens/pass = decode_tokens / decode_steps`. Scheduler counters are reconciled against completed requests.
- **Client wall rate:** total completion tokens divided by the interval from launching a request group to its last completion. An SSE update may contain multiple tokens; update counts are never used as token counts.

The [detailed tables](../benchmarks/results/2026-09-22-current/README.md) include every measured concurrency, sampled results, per-class engine pass times, time to first token, and the minimum/median/maximum across greedy repetitions. Ranges in the overview are variation across prompt classes, not confidence intervals.

## Single-request decode by class

Engine tokens/s, greedy; median of three repetitions.

| Model / weights | Nodes | Options | Prose | Code | JSON | Math | Chat |
|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 35.2 | 35.7 | 36.0 | 33.9 | 31.3 |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 35.0 | 35.6 | 35.8 | 33.8 | 31.2 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 59.1 | 60.3 | 61.8 | 59.6 | 54.0 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 50.2 | 51.0 | 52.1 | 48.8 | 45.4 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 45.5 | 49.0 | 50.0 | 47.5 | 42.3 |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 33.1 | 37.9 | 38.4 | 36.8 | 31.9 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 69.8 | 72.7 | 74.6 | 72.4 | 62.6 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 66.6 | 73.0 | 75.8 | 73.5 | 64.5 |
| Qwen3.8-Flash-Next FP8 | 2 | Default | 48.4 | 53.4 | 54.8 | 51.1 | 44.7 |
| Qwen3.8-Flash-Next FP8 | 4 | Default | 72.6 | 81.4 | 83.3 | 78.9 | 65.3 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | 51.0 | 59.5 | 71.5 | 63.1 | 44.7 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 29.8 | 30.6 | 30.4 | 30.0 | 26.8 |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 30.2 | 30.5 | 30.1 | 29.3 | 27.8 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | 39.8 | 62.2 | 75.5 | 63.0 | 45.8 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 40.4 | 61.7 | 72.8 | 58.6 | 43.7 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 65.4 | 113.3 | 125.9 | 94.2 | 70.3 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 45.0 | 46.0 | 49.2 | 46.1 | 42.3 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 43.2 | 44.6 | 48.2 | 46.8 | 39.9 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 76.8 | 81.3 | 85.8 | 83.9 | 75.8 |

## Decode modes

Single-request engine tokens/s, shown as the range of the five class medians. Plain decoding has MTP disabled. The deeper run uses depth two, fixed depth five for DeepSeek, or depth four for the Qwen AutoRound hybrid (its template default is three). Full GLM's depth-two and DeepSeek-V4.1's depth-five runs use two request slots; DeepSeek-V4-Flash's fixed depth five keeps its template's slots, and its fixed depth three (38.6–62.5 on two nodes, 59.0–104.3 on four) is in the campaign record. Other settings are saved with each run.

| Model / weights | Nodes | Options | Plain | Template default | Deeper MTP |
|---|---|---|---|---|---|
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 24.0–24.1 | 31.3–36.0 | — |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 24.1 | 31.2–35.8 | — |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 39.7–39.8 | 54.0–61.8 | — |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 34.3–34.4 | 45.4–52.1 | — |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 32.7–32.8 | 42.3–50.0 | 44.0–60.7 |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 24.3–24.4 | 31.9–38.4 | 35.0–48.0 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 47.0–47.6 | 62.6–74.6 | 68.1–92.3 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 47.2–47.4 | 64.5–75.8 | 65.9–91.5 |
| Qwen3.8-Flash-Next FP8 | 2 | Default | 35.7–35.9 | 44.7–54.8 | 48.4–68.1 |
| Qwen3.8-Flash-Next FP8 | 4 | Default | 50.6–50.8 | 65.3–83.3 | 71.7–106.6 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | 31.6–31.7 | 44.7–71.5 | 35.9–65.9 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 19.8–20.1 | 26.8–30.6 | 28.0–35.0 |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 19.8–20.1 | 27.8–30.5 | — |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | 32.2–32.3 | 39.8–75.5 | 34.2–82.7 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 29.9–30.0 | 40.4–72.8 | 32.9–75.2 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 43.5–43.7 | 65.4–125.9 | 56.7–128.3 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 33.8–33.9 | 42.3–49.2 | — |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 33.1–33.4 | 39.9–48.2 | — |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 57.4–57.9 | 75.8–85.8 | — |

A dash means that deeper MTP was not part of that deployment's mode sweep. Prompt-dependent acceptance is included in these rates; pass times and committed tokens per pass are recorded separately.

## Quality and correctness

All deployments use all 164 HumanEval problems, the same first 300 GSM8K test problems and 100 deterministic schema-extraction records. Evaluation concurrency equals each deployment's request-slot count. Temperature is zero; reasoning effort is low with thinking enabled and none with thinking disabled. The generation cap is 2,048 tokens (512 for extraction), and thinking uses the mode described above.

HumanEval executes generated code in a container without network access or host-file mounts. Scores are specific to the repository's chat prompts, execution wrapper and token budgets. The HumanEval wrapper executes the returned code block independently; omitted context helpers can cause a failure, as observed for `encode_cyclic` in HumanEval/38. Raw responses and execution errors are retained.

| Model / weights | Nodes | Options | HumanEval | GSM8K | Schema extraction | Responses at token cap |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 157/164 | 292/300 | 100/100 | 0 |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 158/164 | 292/300 | 100/100 | 0 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 157/164 | 292/300 | 100/100 | 0 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 154/164 | 295/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 160/164 | 294/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 158/164 | 291/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 159/164 | 293/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 161/164 | 293/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next FP8 | 2 | Default | 160/164 | 292/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next FP8 | 4 | Default | 158/164 | 291/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | 159/164 | 292/300 | 100/100 | 1 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3, prefill levers on (the template) | 159/164 | 291/300 | 100/100 | 2 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 159/164 | 292/300 | 100/100 | 0 |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 160/164 | 293/300 | 100/100 | 0 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | 160/164 | 296/300 | 100/100 | 0 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 141/164 | 295/300 | 100/100 | 0 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 139/164 | 296/300 | 100/100 | 0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 154/164 | 295/300 | 100/100 | 0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 153/164 | 294/300 | 100/100 | 0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 154/164 | 294/300 | 100/100 | 0 |

With thinking off, DeepSeek-V4-Flash ends 14 (two nodes) and 15 (four nodes) of its HumanEval answers right after the last line of code, without the closing code fence the extractor requires; 12 and 13 of those pass when the open block is executed (153/164 and 152/164). Plain decode gives the same answers, and the release's own inference code ranks end-of-sequence first at that position on the real checkpoint ([record](../benchmarks/results/2026-10-01-deepseek-v4-flash/notes.md)). The table keeps the strict counts.

Complete matching operation streams were collected for 54/54 recorded launches of the current-code campaign, 11/11 of the MiMo campaign and 12/12 of the DeepSeek-V4-Flash campaign. The detailed record reports greedy plain/MTP transcript comparisons and solo/batched text checks for each deployment, including any failures.

| Model / weights | Nodes | Options | Rank checks passed | Plain/default greedy match | Solo/batched greedy text |
|---|---|---|---|---|---|
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 3/3 | 5/5 classes | different |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 3/3 | 5/5 classes | identical |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 7/7 | 5/5 classes | different |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 4/4 | 5/5 classes | identical |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 5/5 | 5/5 classes | identical |
| Qwen3.8-Flash-Next FP8 | 2 | Default | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next FP8 | 4 | Default | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | 5/5 | 5/5 classes | different |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 4/4 | 5/5 classes | identical |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 3/3 | 5/5 classes | identical |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | 5/5 | 5/5 classes | identical |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 6/6 | 5/5 classes | identical |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 6/6 | 5/5 classes | identical |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 3/3 | 5/5 classes | identical |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 4/4 | 5/5 classes | identical |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 4/4 | 5/5 classes | identical |

A class counts as a plain/default match only when all three repetitions in both modes produce the same text and token count. A differing transcript fails the exact-text check. Current dense kernels can use different floating-point reduction orders for different batch shapes, so identical greedy text across batch sizes is not guaranteed. A text mismatch alone does not establish request-state contamination. Rank operation-stream agreement checks execution order, not numerical equality or request isolation; single-node launches have only one stream to record.

For GLM-5.3-Flash NVFP4/FP8 on four nodes, repeated solo runs and C2 matched; C3/C4 reproducibly diverged. With `DGPP_DENSE_GEMV_ROWS=8` or `256`, every tested C1–C4 response matched its configuration's solo response. This supports kernel dispatch as the explanation for that mismatch. The diagnostic does not measure logit error or prove isolation for all workloads. [Procedure and full responses](../benchmarks/results/2026-09-22-current/isolation.md). Tracked in [issue #32](https://github.com/HawkBearPig/dgpp/issues/32). Performance tables use the default kernel setting.

GLM-4.7 attempts were interrupted when a node became unreachable during cold prefill. GLM-4.7 is excluded from this benchmark matrix at the user's request. Its failed attempts are retained separately from the reported results. The available logs do not establish the outages' cause. [Interruption and recovery record](../benchmarks/results/2026-09-22-current/interruptions.md).

## Long context

Each deterministic parcel document is generated once cold and twice more with the prefix cache enabled. Decode columns are medians over those three 256-token requests; cold prefill is one sample. These requests use the model's default reasoning mode. These are timing measurements; retrieval accuracy is measured separately.

| Model / weights | Nodes | Options | Prompt tokens | Cold prefill (s) | Decode ms/pass | Tokens/pass | Engine tok/s |
|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 1,845 | 1.245 | 32.56 | 1.92 | 58.9 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 7,921 | 5.389 | 32.96 | 1.98 | 60.0 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 32,682 | 26.287 | 33.60 | 1.96 | 58.4 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 1,845 | 1.418 | 39.05 | 1.92 | 49.1 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 7,921 | 5.971 | 39.28 | 1.95 | 49.6 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 32,682 | 27.876 | 39.96 | 1.98 | 49.5 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 3,855 | 2.491 | 28.33 | 1.77 | 62.5 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 32,299 | 20.799 | 28.30 | 1.78 | 63.0 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 129,826 | 93.769 | 29.69 | 1.83 | 61.8 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 259,964 | 216.397 | 31.45 | 1.83 | 58.3 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 520,053 | 658.938 | 37.24 | 1.89 | 50.7 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | 3,855 | 2.107 | 57.00 | 2.97 | 52.0 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | 32,299 | 17.523 | 57.10 | 3.19 | 55.8 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP depth 3 | 129,826 | 82.015 | 58.57 | 3.11 | 53.1 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | 3,836 | 2.576 | 53.94 | 2.93 | 54.3 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | 32,365 | 25.446 | 56.39 | 3.07 | 54.5 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | Default | 121,107 | 172.642 | 59.31 | 2.97 | 50.0 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 3,844 | 3.348 | 60.14 | 2.90 | 47.6 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 32,438 | 35.443 | 64.40 | 3.49 | 54.3 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 121,443 | 234.959 | 66.06 | 3.04 | 45.8 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 3,844 | 2.297 | 37.83 | 3.00 | 78.7 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 32,438 | 26.277 | 39.14 | 2.97 | 75.8 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 121,443 | 199.030 | 45.28 | 3.40 | 75.1 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 3,864 | 6.071 | 42.58 | 1.89 | 44.4 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 32,382 | 56.763 | 47.38 | 1.99 | 42.0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 130,288 | 338.640 | 63.72 | 1.99 | 31.3 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 238,855 | 838.780 | 82.15 | 1.99 | 24.2 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 3,864 | 3.071 | 23.72 | 1.92 | 80.8 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 32,382 | 27.606 | 25.80 | 1.99 | 77.2 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 121,266 | 142.864 | 31.60 | 1.99 | 63.1 |

YaRN retrieval uses five planted numeric codes at 5%, 25%, 50%, 75% and 95% of each document, with a 768-token response budget including reasoning. A hit means the expected code occurs in the response, including reasoning, after removing non-digit characters. Prefix reuse repeats the final probe. The two-stream test checks completion; admission may queue a request when the shared KV pool is full.

| Target tokens | Actual prompt tokens | Retrieval hits | Repeated prefix cached | Concurrent requests completed |
|---|---|---|---|---|
| 262,144 | 261,388 | 5/5 | 99.997% | 2/2 |
| 524,288 | 522,619 | 5/5 | 99.999% | 2/2 |

The ~4K control retrieved 5/5 codes. [Full retrieval and concurrency record](../benchmarks/results/2026-09-22-current/raw/qwen-yarn-w2/long/default/retrieval.json).

## Selection microbenchmarks

The Qwen QSA and DeepSeek CSA2 benchmarks run on an idle GB10, using CUDA graphs, five warmups and thirty timed iterations per shape. QSA checks scores and selections against a host oracle; CSA2 checks selection against a host sort of the production score keys, not an independent score-arithmetic oracle. Warm inputs are repeated; cold samples evict L2 before timing. These kernel timings are separate from end-to-end service throughput.

[Complete current selection timings and correctness results](../benchmarks/results/2026-09-22-current/README.md#selection-microbenchmarks).

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

The [campaign runner](../benchmarks/results/2026-09-22-current/run.py), [isolated evaluator](../benchmarks/results/2026-09-22-current/isolated_eval.py) and [long-context client](../benchmarks/results/2026-09-22-current/long_context.py) preserve the full procedure. Quality evaluation requires Docker and the Python image pinned by both registry digest and local image ID in `manifest.json`. Every invocation, including the microbenchmark shapes, is recorded in `raw/**/*.command.json`. The runner is resumable and checks this campaign's binary hash; do not reuse its completed result directory for a new measurement.
