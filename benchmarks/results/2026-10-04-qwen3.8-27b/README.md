# Current benchmark results

Source: `e6e928a10dec8ed7303c9d6446ef68c4d62fb1bf`.

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
| prose | 1 | 33.4 [33.3–33.5] | 32.7 [32.7–32.9] | 81.30 | 2.71 | 152 |
| prose | 2 | 51.9 [51.9–52.0] | 50.9 [50.9–50.9] | 90.90 | 2.42 | 204 |
| prose | 4 | 93.9 [92.4–94.0] | 88.6 [88.4–90.1] | 103.39 | 2.45 | 481 |
| prose | 8 | 143.8 [142.8–143.8] | 136.3 [135.6–136.4] | 120.26 | 2.39 | 683 |
| code | 1 | 39.7 [39.7–39.8] | 38.8 [38.7–38.8] | 81.31 | 3.23 | 177 |
| code | 2 | 71.3 [71.1–71.5] | 69.0 [68.8–69.3] | 90.56 | 3.31 | 228 |
| code | 4 | 118.7 [117.7–120.2] | 112.4 [111.5–113.7] | 102.27 | 3.19 | 480 |
| code | 8 | 200.9 [200.4–201.9] | 186.9 [186.3–187.5] | 120.90 | 3.22 | 689 |
| json | 1 | 45.6 [45.5–45.6] | 44.4 [44.3–44.4] | 81.12 | 3.70 | 152 |
| json | 2 | 83.3 [83.2–83.4] | 80.2 [80.2–80.2] | 90.08 | 3.86 | 227 |
| json | 4 | 131.6 [131.6–131.9] | 123.4 [123.3–123.7] | 100.63 | 3.66 | 484 |
| json | 8 | 234.6 [234.4–236.6] | 216.3 [215.0–217.1] | 119.73 | 3.71 | 709 |
| math | 1 | 41.9 [41.8–42.0] | 40.8 [40.6–40.8] | 81.18 | 3.40 | 177 |
| math | 2 | 78.0 [77.9–78.1] | 75.2 [74.9–75.2] | 90.78 | 3.57 | 230 |
| math | 4 | 116.9 [116.6–118.4] | 110.3 [110.2–111.7] | 101.47 | 3.21 | 481 |
| math | 8 | 190.4 [189.1–195.6] | 177.1 [175.9–181.2] | 115.21 | 3.29 | 739 |
| chat | 1 | 30.4 [30.3–30.4] | 29.9 [29.7–29.9] | 81.47 | 2.48 | 153 |
| chat | 2 | 50.2 [48.4–50.3] | 49.2 [47.4–49.3] | 91.26 | 2.34 | 230 |
| chat | 4 | 93.1 [93.1–94.1] | 89.2 [89.0–90.2] | 101.42 | 2.61 | 459 |
| chat | 8 | 163.5 [161.9–163.6] | 154.4 [151.8–154.6] | 122.33 | 2.63 | 684 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 28.0 | 27.6 |
| prose | 8 | 147.7 | 140.0 |
| code | 1 | 39.6 | 38.5 |
| code | 8 | 201.4 | 186.4 |
| json | 1 | 45.3 | 44.2 |
| json | 8 | 215.5 | 198.7 |
| math | 1 | 41.1 | 40.0 |
| math | 8 | 206.3 | 190.2 |
| chat | 1 | 30.4 | 29.9 |
| chat | 8 | 153.3 | 144.6 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2054, 2043, 2029 | 1.480 | 1.477–1.484 | 0.724 | 1.501 |
| 8192 | 8029, 8078, 8245 | 4.923 | 4.879–5.102 | 0.609 | 4.937 |
| 32768 | 32421, 32460, 32404 | 21.009 | 20.997–21.086 | 0.648 | 21.055 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth2 | prose | 31.6 | 74.68 | 2.36 | yes |
| depth2 | code | 36.1 | 74.42 | 2.68 | yes |
| depth2 | json | 38.9 | 74.44 | 2.90 | yes |
| depth2 | math | 37.7 | 74.36 | 2.80 | yes |
| depth2 | chat | 28.6 | 74.80 | 2.14 | yes |
| dflash2 | prose | 32.6 | 90.91 | 2.97 | yes |
| dflash2 | code | 52.0 | 90.76 | 4.72 | yes |
| dflash2 | json | 74.1 | 90.55 | 6.71 | yes |
| dflash2 | math | 59.9 | 90.56 | 5.43 | yes |
| dflash2 | chat | 30.8 | 90.98 | 2.80 | no |
| plain | prose | 15.8 | 63.25 | 1.00 | yes |
| plain | code | 15.9 | 63.07 | 1.00 | yes |
| plain | json | 15.8 | 63.18 | 1.00 | yes |
| plain | math | 15.9 | 62.94 | 1.00 | yes |
| plain | chat | 15.9 | 62.99 | 1.00 | yes |

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
| prose | 1 | 55.4 [55.2–55.8] | 53.8 [53.5–54.2] | 48.48 | 2.68 | 129 |
| prose | 2 | 87.4 [86.6–87.7] | 85.1 [84.2–85.2] | 54.05 | 2.39 | 154 |
| prose | 4 | 144.9 [144.2–145.3] | 138.4 [136.0–138.8] | 65.19 | 2.45 | 307 |
| prose | 8 | 195.8 [195.6–196.2] | 187.2 [186.8–187.5] | 84.03 | 2.40 | 432 |
| code | 1 | 67.3 [67.3–67.5] | 64.9 [64.8–65.0] | 47.93 | 3.23 | 154 |
| code | 2 | 122.3 [120.8–122.3] | 117.4 [116.0–117.5] | 53.45 | 3.33 | 179 |
| code | 4 | 187.5 [186.9–188.0] | 176.5 [176.1–177.6] | 64.22 | 3.16 | 307 |
| code | 8 | 285.8 [284.1–290.4] | 266.7 [265.1–270.8] | 85.66 | 3.27 | 483 |
| json | 1 | 77.4 [77.3–77.5] | 74.2 [74.1–74.2] | 47.74 | 3.70 | 154 |
| json | 2 | 141.0 [140.9–141.2] | 134.3 [134.0–134.8] | 53.19 | 3.86 | 180 |
| json | 4 | 209.4 [209.3–210.8] | 196.2 [195.8–197.0] | 62.48 | 3.63 | 308 |
| json | 8 | 329.8 [329.4–330.1] | 304.9 [304.6–305.0] | 85.91 | 3.70 | 486 |
| math | 1 | 71.2 [70.9–71.3] | 68.8 [68.3–68.9] | 47.78 | 3.40 | 129 |
| math | 2 | 132.2 [132.1–132.2] | 126.6 [126.3–126.7] | 53.60 | 3.57 | 180 |
| math | 4 | 185.6 [185.2–188.8] | 174.8 [174.5–177.2] | 63.90 | 3.21 | 329 |
| math | 8 | 296.6 [295.9–296.7] | 274.2 [274.0–274.4] | 84.91 | 3.37 | 512 |
| chat | 1 | 48.8 [48.8–48.9] | 47.7 [47.6–47.8] | 47.95 | 2.34 | 128 |
| chat | 2 | 86.4 [86.1–86.7] | 83.9 [83.8–84.4] | 53.68 | 2.35 | 155 |
| chat | 4 | 151.2 [151.1–154.9] | 144.6 [144.3–147.8] | 63.68 | 2.64 | 307 |
| chat | 8 | 218.7 [218.6–219.8] | 207.6 [207.6–208.9] | 84.80 | 2.56 | 435 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 49.2 | 48.1 |
| prose | 8 | 211.1 | 200.4 |
| code | 1 | 66.3 | 64.0 |
| code | 8 | 273.9 | 255.1 |
| json | 1 | 76.6 | 73.4 |
| json | 8 | 315.2 | 288.7 |
| math | 1 | 69.6 | 67.1 |
| math | 8 | 278.5 | 258.6 |
| chat | 1 | 52.2 | 50.9 |
| chat | 8 | 218.7 | 206.6 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2054, 2043, 2029 | 1.047 | 1.042–1.057 | 0.514 | 1.070 |
| 8192 | 8029, 8078, 8245 | 3.719 | 3.687–3.870 | 0.460 | 3.747 |
| 32768 | 32421, 32460, 32404 | 15.521 | 15.487–15.626 | 0.479 | 15.576 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth2 | prose | 55.0 | 42.94 | 2.36 | yes |
| depth2 | code | 62.5 | 42.97 | 2.68 | yes |
| depth2 | json | 67.5 | 42.93 | 2.90 | yes |
| depth2 | math | 65.3 | 42.94 | 2.80 | yes |
| depth2 | chat | 49.7 | 43.08 | 2.14 | yes |
| dflash2 | prose | 53.9 | 54.41 | 2.93 | yes |
| dflash2 | code | 86.9 | 54.35 | 4.72 | yes |
| dflash2 | json | 122.9 | 54.60 | 6.71 | yes |
| dflash2 | math | 102.4 | 54.13 | 5.54 | yes |
| dflash2 | chat | 51.4 | 54.54 | 2.80 | no |
| plain | prose | 28.6 | 34.93 | 1.00 | yes |
| plain | code | 28.6 | 34.93 | 1.00 | yes |
| plain | json | 28.6 | 34.99 | 1.00 | yes |
| plain | math | 28.6 | 34.92 | 1.00 | yes |
| plain | chat | 28.7 | 34.88 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 7 others (256 tokens): DIFFERENT

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 8. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 156/164 | 1 | 251 |
| gsm8k | 293/300 | 5 | 406 |
| extract | 100/100 | 0 | 63 |

Raw results: [raw/qwen27b-fp8-w4/](raw/qwen27b-fp8-w4/).
