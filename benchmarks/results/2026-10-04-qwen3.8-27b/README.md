# Current benchmark results

Source: `d97649b2a41d579e88b81f8a24c54afa56d2a66e`.

All values below are calculated by `summarize.py` from this directory's raw results. The saved `*.command.json` files identify commands, start/end times and exit codes. `manifest.json` identifies the binary, hardware records, checkpoints and datasets.

[Reference harness, HumanEval fences and the reference cross-check](notes.md)

Deployments follow the [overview](../../../docs/benchmarks.md) order: model family, node count, then configuration options. KV labels describe the shared key/value-cache token pool (K = 1,024 tokens); slots are the configured concurrent-request limit.

## Qwen3.8-27B FP8 · 1 node · DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots

Configuration: [qwen27b-fp8-w1](configs/qwen27b-fp8-w1.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 17.8 [17.8–17.9] | 17.7 [17.7–17.7] | 164.22 | 2.93 | 177 |
| prose | 2 | 26.8 [26.1–26.8] | 26.4 [25.7–26.4] | 179.49 | 2.50 | 344 |
| prose | 4 | 42.2 [42.0–43.3] | 41.4 [41.1–42.3] | 219.56 | 2.54 | 693 |
| prose | 8 | 46.4 [46.0–46.4] | 45.4 [44.9–45.4] | 225.26 | 2.66 | 1449 |
| code | 1 | 28.6 [28.6–28.6] | 28.1 [28.1–28.2] | 165.01 | 4.72 | 204 |
| code | 2 | 47.7 [47.7–47.7] | 46.3 [46.3–46.3] | 178.27 | 4.68 | 362 |
| code | 4 | 77.6 [77.4–78.0] | 74.2 [74.0–74.6] | 216.15 | 4.72 | 708 |
| code | 8 | 82.4 [82.2–82.7] | 78.8 [78.6–79.0] | 217.62 | 4.98 | 1475 |
| json | 1 | 40.8 [40.7–40.8] | 39.6 [39.6–39.7] | 164.66 | 6.71 | 179 |
| json | 2 | 68.9 [68.9–68.9] | 65.8 [65.8–65.9] | 176.34 | 6.89 | 368 |
| json | 4 | 90.4 [87.9–91.8] | 85.7 [83.5–87.0] | 205.12 | 6.14 | 716 |
| json | 8 | 102.1 [100.6–102.9] | 96.3 [95.0–97.0] | 219.65 | 5.96 | 1476 |
| math | 1 | 32.9 [32.9–33.0] | 32.0 [32.0–32.1] | 164.84 | 5.43 | 253 |
| math | 2 | 63.1 [62.9–63.1] | 60.7 [60.6–60.8] | 179.75 | 5.86 | 356 |
| math | 4 | 84.8 [84.6–86.1] | 80.6 [80.4–81.8] | 215.18 | 5.48 | 724 |
| math | 8 | 89.3 [87.8–89.4] | 84.8 [83.4–84.9] | 221.74 | 5.18 | 1509 |
| chat | 1 | 17.0 [17.0–17.0] | 16.8 [16.8–16.8] | 165.28 | 2.80 | 179 |
| chat | 2 | 29.3 [29.3–29.3] | 28.9 [28.8–28.9] | 181.10 | 2.68 | 359 |
| chat | 4 | 49.6 [49.6–49.7] | 48.3 [48.2–48.4] | 220.95 | 2.95 | 696 |
| chat | 8 | 49.8 [49.5–50.0] | 48.5 [48.3–48.7] | 222.83 | 2.91 | 1456 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 14.7 | 14.6 |
| prose | 8 | 34.5 | 33.9 |
| code | 1 | 24.4 | 24.1 |
| code | 8 | 61.9 | 59.7 |
| json | 1 | 33.7 | 33.0 |
| json | 8 | 72.6 | 69.5 |
| math | 1 | 27.6 | 27.0 |
| math | 8 | 65.9 | 63.4 |
| chat | 1 | 12.8 | 12.7 |
| chat | 8 | 37.3 | 36.6 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2054, 2043, 2029 | 2.145 | 2.141–2.147 | 1.048 | 2.159 |
| 8192 | 8029, 8078, 8245 | 6.997 | 6.876–7.230 | 0.866 | 7.018 |
| 32768 | 32421, 32460, 32404 | 30.480 | 30.460–30.623 | 0.941 | 30.522 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| mtp2 | prose | 17.8 | 132.62 | 2.36 | yes |
| mtp2 | code | 20.1 | 133.75 | 2.68 | yes |
| mtp2 | json | 22.0 | 131.93 | 2.90 | yes |
| mtp2 | math | 21.0 | 132.11 | 2.77 | no |
| mtp2 | chat | 16.1 | 132.73 | 2.14 | no |
| plain | prose | 8.8 | 113.14 | 1.00 | yes |
| plain | code | 8.8 | 113.08 | 1.00 | yes |
| plain | json | 8.8 | 113.08 | 1.00 | yes |
| plain | math | 8.8 | 113.09 | 1.00 | no |
| plain | chat | 8.8 | 113.07 | 1.00 | no |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 7 others (256 tokens): DIFFERENT

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 8. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 157/164 | 0 | 254 |
| gsm8k | 293/300 | 4 | 400 |
| extract | 100/100 | 0 | 63 |

Raw results: [raw/qwen27b-fp8-w1/](raw/qwen27b-fp8-w1/).

## Qwen3.8-27B FP8 · 2 nodes · MTP depth 3, FP8 head, 256K BF16 KV, 8 slots

Configuration: [qwen27b-fp8-w2](configs/qwen27b-fp8-w2.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 34.0 [33.9–34.0] | 33.3 [33.3–33.4] | 79.84 | 2.71 | 152 |
| prose | 2 | 52.1 [52.1–52.2] | 51.0 [51.0–51.1] | 88.16 | 2.38 | 227 |
| prose | 4 | 95.0 [94.9–95.6] | 90.8 [89.4–91.6] | 101.40 | 2.45 | 459 |
| prose | 8 | 143.8 [143.7–144.2] | 136.4 [136.3–137.1] | 120.27 | 2.39 | 663 |
| code | 1 | 40.5 [40.4–40.5] | 39.5 [39.4–39.6] | 79.78 | 3.23 | 153 |
| code | 2 | 73.7 [73.5–74.4] | 71.2 [71.1–72.0] | 87.83 | 3.33 | 228 |
| code | 4 | 124.3 [120.7–125.9] | 117.2 [113.8–118.6] | 100.01 | 3.22 | 482 |
| code | 8 | 199.3 [199.1–202.8] | 184.8 [184.6–188.3] | 120.40 | 3.20 | 689 |
| json | 1 | 46.4 [46.4–46.4] | 45.1 [45.1–45.2] | 79.64 | 3.70 | 177 |
| json | 2 | 85.7 [85.7–85.7] | 82.6 [82.5–82.6] | 87.53 | 3.83 | 228 |
| json | 4 | 135.6 [135.5–135.6] | 127.1 [126.8–127.2] | 97.70 | 3.64 | 482 |
| json | 8 | 236.4 [236.4–237.4] | 217.4 [217.1–217.4] | 119.85 | 3.71 | 709 |
| math | 1 | 42.6 [42.5–42.7] | 41.5 [41.3–41.5] | 79.80 | 3.40 | 178 |
| math | 2 | 81.3 [80.4–81.4] | 78.4 [77.6–78.6] | 88.27 | 3.59 | 229 |
| math | 4 | 119.7 [119.6–121.0] | 112.7 [112.5–114.0] | 99.16 | 3.21 | 484 |
| math | 8 | 195.9 [195.7–196.7] | 181.8 [181.6–181.9] | 115.73 | 3.31 | 739 |
| chat | 1 | 30.9 [30.8–30.9] | 30.4 [30.3–30.4] | 80.16 | 2.48 | 152 |
| chat | 2 | 51.9 [51.9–51.9] | 50.8 [50.8–50.8] | 88.60 | 2.34 | 228 |
| chat | 4 | 95.1 [94.9–95.2] | 90.8 [90.8–91.0] | 99.26 | 2.61 | 459 |
| chat | 8 | 163.6 [163.3–163.8] | 154.3 [154.2–154.8] | 122.28 | 2.63 | 682 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 29.7 | 29.2 |
| prose | 8 | 153.2 | 144.9 |
| code | 1 | 39.6 | 38.6 |
| code | 8 | 197.4 | 183.1 |
| json | 1 | 45.3 | 44.0 |
| json | 8 | 221.0 | 203.6 |
| math | 1 | 41.7 | 40.6 |
| math | 8 | 201.1 | 185.9 |
| chat | 1 | 29.8 | 29.3 |
| chat | 8 | 157.8 | 148.4 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2054, 2043, 2029 | 1.477 | 1.474–1.480 | 0.724 | 1.500 |
| 8192 | 8029, 8078, 8245 | 4.916 | 4.864–5.129 | 0.609 | 4.933 |
| 32768 | 32421, 32460, 32404 | 21.155 | 21.128–21.211 | 0.653 | 21.211 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth2 | prose | 32.7 | 72.24 | 2.36 | yes |
| depth2 | code | 37.3 | 72.02 | 2.68 | yes |
| depth2 | json | 40.2 | 72.02 | 2.90 | yes |
| depth2 | math | 38.9 | 71.97 | 2.80 | yes |
| depth2 | chat | 29.7 | 72.24 | 2.14 | yes |
| dflash2 | prose | 33.5 | 88.58 | 2.97 | yes |
| dflash2 | code | 53.6 | 88.13 | 4.72 | yes |
| dflash2 | json | 76.4 | 87.88 | 6.71 | yes |
| dflash2 | math | 61.6 | 88.07 | 5.43 | yes |
| dflash2 | chat | 31.6 | 88.58 | 2.80 | no |
| plain | prose | 16.5 | 60.50 | 1.00 | yes |
| plain | code | 16.5 | 60.49 | 1.00 | yes |
| plain | json | 16.5 | 60.54 | 1.00 | yes |
| plain | math | 16.5 | 60.52 | 1.00 | yes |
| plain | chat | 16.5 | 60.51 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 7 others (256 tokens): IDENTICAL

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 8. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 158/164 | 0 | 247 |
| gsm8k | 291/300 | 6 | 409 |
| extract | 100/100 | 0 | 63 |

Raw results: [raw/qwen27b-fp8-w2/](raw/qwen27b-fp8-w2/).

## Qwen3.8-27B FP8 · 4 nodes · MTP depth 3, FP8 head, 256K BF16 KV, 8 slots

Configuration: [qwen27b-fp8-w4](configs/qwen27b-fp8-w4.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 57.9 [57.9–58.1] | 56.4 [56.0–56.4] | 46.32 | 2.68 | 127 |
| prose | 2 | 90.1 [87.6–90.2] | 87.6 [85.1–87.8] | 50.97 | 2.38 | 153 |
| prose | 4 | 155.6 [155.2–156.9] | 148.7 [145.4–150.2] | 61.90 | 2.45 | 305 |
| prose | 8 | 213.3 [213.0–213.8] | 203.0 [202.8–203.4] | 85.40 | 2.43 | 435 |
| code | 1 | 69.9 [69.4–69.9] | 67.0 [66.8–67.1] | 46.18 | 3.23 | 153 |
| code | 2 | 125.4 [125.2–125.6] | 120.4 [120.4–120.7] | 50.84 | 3.29 | 178 |
| code | 4 | 196.7 [196.6–201.8] | 185.1 [184.9–189.5] | 61.05 | 3.17 | 306 |
| code | 8 | 290.0 [288.7–291.0] | 270.0 [268.8–270.1] | 83.47 | 3.23 | 483 |
| json | 1 | 79.9 [79.7–80.1] | 76.3 [76.3–76.4] | 46.27 | 3.70 | 154 |
| json | 2 | 147.4 [145.9–148.8] | 140.5 [138.9–141.6] | 50.67 | 3.86 | 179 |
| json | 4 | 220.0 [216.2–220.6] | 205.2 [202.4–205.8] | 60.04 | 3.64 | 308 |
| json | 8 | 334.8 [334.4–335.1] | 308.1 [306.9–308.4] | 83.46 | 3.69 | 489 |
| math | 1 | 73.3 [73.2–73.6] | 70.8 [70.7–71.1] | 46.41 | 3.40 | 128 |
| math | 2 | 139.4 [138.8–140.4] | 132.8 [131.9–134.4] | 51.04 | 3.57 | 179 |
| math | 4 | 194.7 [194.2–195.4] | 182.4 [181.8–183.2] | 60.91 | 3.20 | 334 |
| math | 8 | 304.1 [302.9–304.3] | 281.3 [279.4–281.4] | 82.83 | 3.38 | 512 |
| chat | 1 | 50.4 [50.3–50.4] | 49.1 [49.0–49.2] | 46.45 | 2.34 | 128 |
| chat | 2 | 92.8 [92.0–92.9] | 90.1 [89.3–90.3] | 51.30 | 2.39 | 154 |
| chat | 4 | 157.4 [156.8–157.7] | 150.1 [149.6–150.2] | 60.01 | 2.62 | 306 |
| chat | 8 | 231.8 [231.3–232.2] | 219.7 [219.2–219.8] | 85.46 | 2.60 | 458 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 53.1 | 51.5 |
| prose | 8 | 213.4 | 202.7 |
| code | 1 | 73.2 | 70.3 |
| code | 8 | 300.8 | 279.0 |
| json | 1 | 79.1 | 75.8 |
| json | 8 | 317.8 | 292.9 |
| math | 1 | 74.7 | 71.5 |
| math | 8 | 287.1 | 265.3 |
| chat | 1 | 52.9 | 51.4 |
| chat | 8 | 224.0 | 211.9 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2054, 2043, 2029 | 1.054 | 1.051–1.058 | 0.518 | 1.077 |
| 8192 | 8029, 8078, 8245 | 3.726 | 3.692–3.909 | 0.461 | 3.753 |
| 32768 | 32421, 32460, 32404 | 15.627 | 15.548–15.721 | 0.481 | 15.679 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth2 | prose | 58.5 | 40.34 | 2.36 | yes |
| depth2 | code | 66.8 | 40.21 | 2.68 | yes |
| depth2 | json | 71.9 | 40.29 | 2.90 | yes |
| depth2 | math | 69.5 | 40.30 | 2.80 | yes |
| depth2 | chat | 53.0 | 40.44 | 2.14 | yes |
| dflash2 | prose | 56.4 | 51.98 | 2.93 | yes |
| dflash2 | code | 90.9 | 51.94 | 4.72 | yes |
| dflash2 | json | 129.5 | 51.81 | 6.71 | yes |
| dflash2 | math | 107.0 | 51.79 | 5.54 | yes |
| dflash2 | chat | 53.9 | 51.96 | 2.80 | no |
| plain | prose | 31.2 | 32.08 | 1.00 | yes |
| plain | code | 31.1 | 32.15 | 1.00 | yes |
| plain | json | 31.1 | 32.14 | 1.00 | yes |
| plain | math | 31.1 | 32.16 | 1.00 | yes |
| plain | chat | 31.1 | 32.13 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 7 others (256 tokens): DIFFERENT

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 8. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 156/164 | 1 | 251 |
| gsm8k | 293/300 | 5 | 406 |
| extract | 100/100 | 0 | 63 |

Raw results: [raw/qwen27b-fp8-w4/](raw/qwen27b-fp8-w4/).
