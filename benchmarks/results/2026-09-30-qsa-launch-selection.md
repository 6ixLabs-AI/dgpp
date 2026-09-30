# QSA launch bounds and select-all follow-up — 2026-09-30

PR #66 bounds prefill scoring by the last position in the chunk, including
cached history, without changing the workspace stride. Its original commit is
`6655cd2`; the review baseline `0ee6041` merges that change with master
`a23d8b1` so the follow-up comparison includes the current request-bounded
workspace, AutoRound support and attention dispatch on both sides.

The follow-up passes the selection budget to scoring and skips rows whose
visible pool count fits it. Selection emits tokens `[0, position]` directly
for those rows, including the incomplete tail and normal `-1` padding.
Compression and ring maintenance still run. With the release's four-token
pools and 512-pool budget, scoring first becomes necessary at 2052 tokens.
Decode/verify retain the capacity-sized grid and make the skip decision from
device positions on every graph replay. The scoring API defaults to writing
every visible key for diagnostic callers.

This is shared Qwen behavior: FP8, NVFP4 and AutoRound int4 experts all feed
the BF16 QSA indexer, including when dense weights are FP8. GLM DSA and
DeepSeek CSA already bound prefill scoring by visible history; their cache
and selection layouts differ, so this change does not introduce a common
cross-family selector.

## Decode-grid experiment

A fixed grid of `min(ceil(capacity / 256), max(1, 96 / rows))` stripes per
row partitioned each row's visible pools on the device, rounding stripe
widths to eight pools. It preserved every scoring key and selected token,
including graph replay. It was rejected because wider batches regressed.

GB10, CI build, CUDA graphs, 40 samples with L2 eviction outside the CUDA
event interval; capacity 131072 pools. Cold scoring medians in microseconds:

| Rows | Visible pools | Full grid | Bounded grid |
| ---: | ---: | ---: | ---: |
| 1 | 513 | 28.384 | 7.904 |
| 8 | 16384 | 92.896 | 136.832 |
| 16 | 16384 | 145.120 | 261.856 |
| 64 | 16384 | 455.648 | 1466.080 |
| 64 | 131072 | 8572.160 | 11722.096 |

The production decode grid remains unchanged. The microbenchmark now accepts
`--capacity`, `--visible-bound` and `--skip-select-all` to separate workspace
capacity, prefill launch bounds and score elision while retaining its host
oracle for every required score and selected token.

## Regression and build validation

The candidate runtime is `eb00fa7`, built with the CI preset, GCC 13,
CUDA 13.0 and `sm_121a`, with warnings treated as errors. The full build
passed. Across the complete CTest inventory, 179 cases passed and two were
skipped (`mimo_tokenizer_test` and `mimo_chat_template_test`, whose optional
checkpoint files were unavailable).

The QSA suite has 12 cases. The new select-all case covers one, two and four
index heads; budgets of 8, 512 and 1024 pools; shuffled physical pages for
two requests; inactive rows; complete and incomplete tails; stripe edges;
invalid budgets; untouched score storage and guards. A graph captured below
the budget replays at the budget, above it, and back below it. Both graph
and host-bounded launches match the independent host score/token oracle.
Compute Sanitizer memcheck reports zero errors for this case and the
original PR's visible-bound case.

Model prefill coverage uses lengths 127, 128, 129, 256, 513, 1024, 2047,
2048, 2049, 2051, 2052 and 2053 with FP8 experts and both NVFP4 dense modes.
Each mode checks repeatable forward logits, exact cold-forward/session
agreement and four subsequent decode steps. Warp-versus-partial comparison
checks every row's relative L2 and winner margin. The FP8 fixture retains
the existing requirement to demonstrate a numerical difference between the
two dispatches; NVFP4 may agree exactly after expert rounding. Its numerical
bounds are unchanged.

The NVFP4 engine cases exercise eager/graph parity, slot reuse, continuation
and MTP over two-rank loopback worlds with both dense formats. The existing
AutoRound forward, decode and engine cases also pass. Wider Qwen tests cover
32/64-row MTP graphs and compaction/fallback.

Two test setup corrections were needed: the prefill fixture capacity is
2056, so its aligned session chunk holds all 2053 rows in the same execution
shape as the diagnostic forward; the isolated checkout uses the existing
NumPy 2.5.3 environment for DeepSeek/MiMo reference generation. The affected
cases were rebuilt or reconfigured and rerun successfully.

## Select-all microbenchmark

Cold combined score/selection medians, CUDA graphs, 30 samples and five
warmups, with L2 eviction outside the event interval. Capacity is the pool
count rounded up to 16; the prefill-shaped runs have 2048 consecutive rows.
Both binaries check selected tokens against the host oracle; scoring checks
exclude only the keys that the candidate deliberately does not produce.

| Rows | Pools at final row | Baseline µs | Candidate µs |
| ---: | ---: | ---: | ---: |
| 1 | 512 | 53.376 | 3.776 |
| 8 | 512 | 54.976 | 4.000 |
| 64 | 512 | 61.168 | 7.904 |
| 2048 | 512 | 540.320 | 94.480 |
| 2048 | 1024 | 1228.512 | 1226.528 |

The last row models an already populated history, where almost all rows
still need selection. These are component timings, not full-model speedups.
Reproduce the candidate's prefill-shaped pair with:

```bash
build-ci/qsa_index_bench --rows 2048 --pools 512,1024 --graph \
  --iters 30 --warmup 5 --visible-bound --skip-select-all
```

The baseline uses the same command without the last two flags. The wider
decode matrix covers rows 1, 2, 8, 16 and 64 and pool counts 256, 512, 513,
2048, 16384, 65536 and 131072. Register counts, static shared memory and
spills are unchanged for both scoring and selection (48 registers, no spills).

The initial 64-row/131072-pool combined timing varied between runs. An
alternating baseline/candidate/candidate/baseline repeat gave cold medians
of 10323.680/10225.504/10183.632/10491.024 µs, with every oracle check passing.
The repeat does not show the initial apparent long-context regression; it
also does not establish a long-context speedup.

## Real-model fabric comparison

The same baseline and candidate ran serially on the four-node GB10 RoCE
cluster in three configurations:

- `Qwen/Qwen3.8-Flash-Next-FP8`, four ranks, revision
  `236dfdf285828023ca3bcd3f37366c58a3469b13`.
- `nvidia/Qwen3.8-Flash-Next-NVFP4`, two ranks, checkpoint/BF16 dense weights,
  revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`.
- The same NVFP4 checkpoint on two ranks with FP8 dense weights and MMA head.

All configurations used four request slots, 32768 KV tokens in BF16,
`bf12+bf16` weight storage, a 4 GiB prefix cache, decode graphs, MTP depth 1,
and prefill budgets of 256 busy / 2048 idle tokens. NVFP4 mapped the n-gram
table. An initial test configuration incorrectly combined the MMA FP8 head
with checkpoint/BF16 dense weights; startup rejected it, the setting was
corrected, and the valid baseline was restarted. Peer logs append across
launches, so log validation checks the final process banner and its commit.

For each configuration, all 19 sequential requests matched the baseline
exactly: output choices, top-five log probabilities and usage. Raw prompt
lengths were 3, 4, 1023, 1024, 1025, 2047, 2048, 2049, 2051, 2052, 2053,
4096 and 8193. Chat requests covered cold prompts, identical repeats and
appended conversations, reusing 2068 or 8212 cached tokens. Four 64-token
greedy transcripts also matched exactly in each configuration.

Each arm completed four simultaneous 64-token requests with four-slot MTP
graph replays and passed `serve_api_check.py`. Concurrent transcript equality
was not a gate. Every successful world stopped cleanly, with no engine error
or operation-stream divergence in that process's logs. Operation streams
matched across every participating rank within each world:

| Configuration | Baseline MD5 | Candidate MD5 |
| --- | --- | --- |
| FP8, four ranks | `b698fdfdad92b36a81ed593d6e3e92d2` | `c0454ac91703d3dd5ee4fc9e7354ed5f` |
| NVFP4, BF16 dense | `b2fec0e437f060603d0b2796e2760aef` | `7fd0eff29e30b60405891179d7b931f2` |
| NVFP4, FP8 dense | `39cb72dda436915b50ea16ce9c7efe71` | `690b59a2d8863d7a71e06ed5e03668e3` |

Cold prefill probes used three identical prompts per requested length in
each arm, with no cached tokens. Medians in milliseconds:

| Configuration | Arm | ~2K | ~4K | ~8K |
| --- | --- | ---: | ---: | ---: |
| FP8 | Baseline | 942.0 | 1754.3 | 3498.0 |
| FP8 | Candidate | 940.1 | 1746.6 | 3494.6 |
| NVFP4 / BF16 dense | Baseline | 991.8 | 1792.6 | 3506.3 |
| NVFP4 / BF16 dense | Candidate | 958.2 | 1776.2 | 3526.0 |
| NVFP4 / FP8 dense | Baseline | 976.3 | 1804.2 | 3594.8 |
| NVFP4 / FP8 dense | Candidate | 966.6 | 1805.9 | 3596.2 |

These are a small serving sanity comparison, not a new published throughput
result. Host compilation overlapped part of the baseline serving run; GPU
tests and deployments were serialized. The standard clients were
`serve_greedy_transcript.py --max-tokens 64`, `serve_api_check.py`, and
`serve_prefill_probe.py` with lengths 2048/4096/8192, three repeats,
`--no-think`, seed 7 and tag `pr66-followup`.

The original four-rank GLM-5.3-Flash deployment was restored using its
original configuration and binary. All four recorded processes were alive
and `/health` returned `ok` before publishing the follow-up commits.
