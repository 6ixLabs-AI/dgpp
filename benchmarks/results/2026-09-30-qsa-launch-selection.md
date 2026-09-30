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
