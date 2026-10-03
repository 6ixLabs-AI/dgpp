# Current benchmark results

Source: `904f76c8519ad6b1e55dc1eafffbb05e71c49f41`.

All values below are calculated by `summarize.py` from this directory's raw results. The saved `*.command.json` files identify commands, start/end times and exit codes. `manifest.json` identifies the binary, hardware records, checkpoints and datasets.

[Reference harness, HumanEval fences and the reference cross-check](notes.md)

Deployments follow the [overview](../../../docs/benchmarks.md) order: model family, node count, then configuration options. KV labels describe the shared key/value-cache token pool (K = 1,024 tokens); slots are the configured concurrent-request limit.

## DeepSeek-V4-Flash MXFP4/FP8 · 4 nodes · 1M BF16 KV, 6 slots

Configuration: [dsv4-w4](configs/dsv4-w4.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 65.4 [65.2–65.9] | 63.0 [62.6–63.4] | 33.03 | 2.16 | 151 |
| prose | 2 | 93.1 [92.9–93.2] | 89.0 [89.0–89.4] | 41.18 | 1.92 | 203 |
| prose | 4 | 126.5 [125.3–126.9] | 120.8 [119.3–121.1] | 56.59 | 1.86 | 303 |
| prose | 6 | 140.4 [139.6–140.6] | 134.0 [133.1–134.5] | 71.13 | 1.72 | 380 |
| code | 1 | 113.3 [111.9–113.7] | 104.6 [103.5–105.5] | 39.34 | 4.47 | 177 |
| code | 2 | 128.0 [127.6–129.9] | 120.1 [119.4–121.4] | 53.83 | 3.59 | 228 |
| code | 4 | 161.3 [160.7–161.8] | 151.9 [151.5–152.0] | 79.03 | 3.42 | 305 |
| code | 6 | 173.2 [169.4–174.5] | 163.6 [159.9–164.1] | 96.35 | 2.99 | 406 |
| json | 1 | 125.9 [125.4–126.6] | 115.7 [114.4–115.9] | 42.36 | 5.31 | 153 |
| json | 2 | 169.7 [169.2–171.2] | 155.7 [154.9–157.4] | 63.95 | 5.54 | 230 |
| json | 4 | 183.6 [182.9–184.1] | 170.6 [170.4–171.4] | 88.20 | 4.88 | 307 |
| json | 6 | 199.7 [198.7–199.9] | 186.0 [186.0–186.6] | 119.68 | 4.21 | 408 |
| math | 1 | 94.2 [89.7–95.5] | 87.7 [83.9–89.1] | 37.61 | 3.59 | 179 |
| math | 2 | 150.3 [148.8–151.9] | 138.9 [137.5–139.9] | 58.91 | 4.51 | 255 |
| math | 4 | 159.0 [154.5–160.6] | 149.5 [144.3–150.7] | 78.23 | 3.45 | 330 |
| math | 6 | 178.0 [177.6–179.5] | 166.8 [166.2–168.3] | 105.25 | 3.21 | 435 |
| chat | 1 | 70.3 [69.0–72.0] | 67.1 [66.0–68.7] | 32.80 | 2.36 | 154 |
| chat | 2 | 96.9 [96.9–97.5] | 92.8 [92.6–92.9] | 44.61 | 2.24 | 206 |
| chat | 4 | 128.7 [128.0–130.2] | 122.2 [122.0–124.0] | 61.43 | 2.07 | 307 |
| chat | 6 | 148.9 [148.7–149.2] | 141.8 [141.7–141.8] | 76.21 | 1.99 | 383 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 63.6 | 61.2 |
| prose | 6 | 135.8 | 129.7 |
| code | 1 | 95.3 | 88.9 |
| code | 6 | 164.1 | 154.4 |
| json | 1 | 120.9 | 110.7 |
| json | 6 | 191.1 | 178.2 |
| math | 1 | 91.9 | 85.5 |
| math | 6 | 165.3 | 155.0 |
| chat | 1 | 66.0 | 63.1 |
| chat | 6 | 142.1 | 134.9 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2085, 2114, 2094 | 1.359 | 1.339–1.360 | 0.643 | 1.374 |
| 8192 | 8375, 8416, 8528 | 5.471 | 5.444–5.550 | 0.650 | 5.502 |
| 32768 | 33720, 33778, 33700 | 28.022 | 27.948–28.108 | 0.832 | 28.065 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth3 | prose | 62.6 | 35.42 | 2.22 | yes |
| depth3 | code | 97.7 | 35.74 | 3.49 | yes |
| depth3 | json | 106.1 | 35.86 | 3.81 | yes |
| depth3 | math | 88.1 | 35.74 | 3.15 | yes |
| depth3 | chat | 71.0 | 35.58 | 2.52 | yes |
| depth5 | prose | 56.7 | 42.46 | 2.41 | yes |
| depth5 | code | 111.9 | 42.98 | 4.81 | yes |
| depth5 | json | 128.3 | 43.21 | 5.54 | yes |
| depth5 | math | 85.3 | 42.70 | 3.64 | yes |
| depth5 | chat | 66.8 | 42.92 | 2.87 | yes |
| plain | prose | 43.7 | 22.90 | 1.00 | yes |
| plain | code | 43.6 | 22.94 | 1.00 | yes |
| plain | json | 43.6 | 22.96 | 1.00 | yes |
| plain | math | 43.5 | 23.00 | 1.00 | yes |
| plain | chat | 43.6 | 22.94 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 5 others (256 tokens): IDENTICAL

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 6. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 139/164 | 0 | 153 |
| gsm8k | 296/300 | 0 | 134 |
| extract | 100/100 | 0 | 53 |

Raw results: [raw/dsv4-w4/](raw/dsv4-w4/).

## DeepSeek-V4-Flash MXFP4/FP8 · 2 nodes · 1M BF16 KV, 4 slots

Configuration: [dsv4-w2](configs/dsv4-w2.json).

### Greedy serving

Rates are tokens/s. Brackets show minimum–maximum across three repetitions; the leading value is the median. TTFT is the median of per-phase mean request TTFTs.

| Class | C | Engine tok/s | Wall tok/s | ms/pass | Tokens/pass/request | TTFT ms |
|---|---|---|---|---|---|---|
| prose | 1 | 40.4 [40.2–40.4] | 39.1 [39.0–39.1] | 51.80 | 2.09 | 205 |
| prose | 2 | 55.3 [55.2–55.3] | 53.5 [53.4–53.5] | 67.83 | 1.90 | 280 |
| prose | 4 | 69.9 [69.8–69.9] | 67.5 [67.5–67.6] | 96.79 | 1.73 | 407 |
| code | 1 | 61.7 [60.5–62.6] | 58.5 [57.2–59.2] | 63.54 | 3.92 | 229 |
| code | 2 | 77.2 [76.1–77.8] | 73.2 [72.3–74.0] | 88.05 | 3.47 | 306 |
| code | 4 | 86.6 [86.2–87.0] | 83.0 [82.5–83.3] | 127.49 | 2.94 | 433 |
| json | 1 | 72.8 [72.7–73.0] | 68.0 [67.9–68.4] | 70.09 | 5.10 | 229 |
| json | 2 | 98.3 [98.2–98.5] | 92.2 [92.1–92.3] | 110.34 | 5.60 | 306 |
| json | 4 | 106.8 [105.9–107.1] | 101.1 [100.3–101.3] | 151.53 | 4.77 | 433 |
| math | 1 | 58.6 [55.3–59.0] | 55.4 [52.3–55.4] | 60.86 | 3.59 | 253 |
| math | 2 | 88.6 [88.1–89.0] | 83.1 [82.5–83.5] | 99.24 | 4.51 | 357 |
| math | 4 | 90.7 [90.2–91.1] | 86.6 [85.9–86.7] | 126.99 | 3.33 | 458 |
| chat | 1 | 43.7 [42.1–43.8] | 42.1 [40.7–42.1] | 51.63 | 2.26 | 205 |
| chat | 2 | 57.5 [56.9–57.6] | 55.4 [54.9–55.6] | 70.39 | 2.14 | 306 |
| chat | 4 | 73.0 [71.8–73.8] | 70.5 [69.3–71.1] | 104.40 | 1.92 | 408 |

### Sampled serving

Temperature one; medians of three repetitions. Other sampling parameters use each model's server defaults.

| Class | C | Engine tok/s | Wall tok/s |
|---|---|---|---|
| prose | 1 | 41.5 | 40.2 |
| prose | 4 | 68.0 | 65.7 |
| code | 1 | 57.3 | 54.2 |
| code | 4 | 84.6 | 81.0 |
| json | 1 | 74.9 | 69.9 |
| json | 4 | 108.2 | 102.1 |
| math | 1 | 50.7 | 48.0 |
| math | 4 | 93.0 | 88.1 |
| chat | 1 | 40.6 | 39.2 |
| chat | 4 | 70.3 | 67.8 |

### Cold prefill

| Target tokens | Actual prompt tokens (all samples) | Prefill median s | Prefill min–max s | ms/token | TTFT median s |
|---|---|---|---|---|---|
| 2048 | 2085, 2114, 2094 | 2.029 | 1.996–2.057 | 0.969 | 2.063 |
| 8192 | 8375, 8416, 8528 | 8.015 | 7.919–8.102 | 0.950 | 8.041 |
| 32768 | 33720, 33778, 33700 | 38.026 | 37.667–38.076 | 1.126 | 38.057 |

### Decode mode checks

| Mode | Class | Engine tok/s | ms/pass | Tokens/pass | All greedy repetitions match default |
|---|---|---|---|---|---|
| depth3 | prose | 39.1 | 57.70 | 2.26 | yes |
| depth3 | code | 56.2 | 58.14 | 3.27 | yes |
| depth3 | json | 63.8 | 58.80 | 3.75 | yes |
| depth3 | math | 56.0 | 58.39 | 3.27 | yes |
| depth3 | chat | 41.5 | 57.90 | 2.41 | yes |
| depth5 | prose | 32.9 | 70.47 | 2.32 | yes |
| depth5 | code | 61.5 | 71.49 | 4.40 | yes |
| depth5 | json | 75.2 | 72.19 | 5.43 | yes |
| depth5 | math | 58.5 | 71.51 | 4.18 | yes |
| depth5 | chat | 36.5 | 71.38 | 2.60 | yes |
| plain | prose | 30.0 | 33.35 | 1.00 | yes |
| plain | code | 29.9 | 33.41 | 1.00 | yes |
| plain | json | 29.9 | 33.39 | 1.00 | yes |
| plain | math | 29.9 | 33.42 | 1.00 | yes |
| plain | chat | 29.9 | 33.42 | 1.00 | yes |

Solo/batched exact-text check: == isolation: prompt 0 alone (256 tokens) vs beside 3 others (256 tokens): IDENTICAL

### Quality

One greedy response per problem, using the repository's chat prompts and concurrency 4. Token caps include reasoning. HumanEval runs generated Python in the pinned container recorded in `manifest.json`.

| Task | Passed / evaluated | At token cap | Mean completion tokens |
|---|---|---|---|
| humaneval | 141/164 | 0 | 154 |
| gsm8k | 295/300 | 0 | 136 |
| extract | 100/100 | 0 | 53 |

Raw results: [raw/dsv4-w2/](raw/dsv4-w2/).
