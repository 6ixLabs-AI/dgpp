# Row-batched decode absorb / value-out projections — shipped (2026-09-26)

The DSA layer's decode `absorb_q` and `vout` kernels (`src/kernels/dsa.cu`) launched one block per
(row, head); from two decode rows up they now take row-batched forms in which a block reads its head's
`kv_b` slice once for a tile of rows, every row keeping the one-row kernel's FMA chain — so every row's
bits are the one-row kernel's (`dsa_absorb_and_vout_row_batches_match_the_rows_alone`, 1..16 rows,
64 and 16 heads, with and without a rope tail). One row keeps the one-row kernels. Commit c272090.

**Verdict: shipped.** Full GLM-5.3 (eight slots, MTP depth 1): C1/C2 unchanged, C4 −1.1 %, C8 −1.7 %
(ms/step, against two bracketing baselines); every C1/C2 greedy transcript identical across the three
legs. GLM-5.3-Flash (four slots): no regression, transcripts identical (regression check only, below).

## Why

The context microbench (`kvb_bench.cu`, real rank-0 `kv_b` slices of six full GLM-5.3 layers, the
production kernels from `libdgpp_kernels.a` against twins, bitwise-gated, behind the layer's real
16 MiB prefetch window with register-spin gaps taken from the decode profile) found the one-row
kernels' cost at batched rows to be their per-(row, head) re-reads of `kv_b`: 94 µs per layer at
sixteen rows against 38 at one, L2 hits though the re-reads are. Row-batched twins, us per layer
(both kernels, in context, gaps 25/25 µs): m=1 +0.6, m=2 −0.2, m=4 −1.8, m=8 −16.5, m=16 −34.6.
The bench was built to test `kv_b` as native int8 first; that form is 1–2 % at one or two rows and
a loss at 4–16 rows (its per-row re-decode), so the format stayed and the structure changed.

Two vout designs lost on the way: unrolling four rows' activation loads beside the weights doubled
the register footprint (ncu: 44 % occupancy against production's 75–85 %, slower at every row count),
and one block walking every row for a (d-slab, head) serialized the rows behind a scalar-staged c;
the shipped form holds the warp's weight row in registers, stages four c rows with float4 loads and
runs them one after another, grid z = the row tiles.

## Fabric (four nodes, release binaries, `scripts/timed_load.py --classes all --repeat 2`)

Engine ms/step = `step_ms / decode_steps` per phase; per class the median of the two repeats, then
the mean over the five classes. base1 → cand1 → base2 (closing baseline), production down throughout.
Binaries: base `0.1.0+g29bd5b275b3b` (master), cand `0.1.0+gc27209023e01`.

### `deploy/cluster_glm-5.3_int4-int8_w4.json` — HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64, 8 slots, MTP depth 1

| | base1 | cand1 | base2 | cand vs bases | A/A spread |
|---|---:|---:|---:|---:|---:|
| C1 (2 rows) | 64.13 | 63.93 | 63.91 | −0.14 % | 0.34 % |
| C2 (4 rows) | 91.94 | 91.64 | 91.92 | −0.31 % | 0.02 % |
| C4 (8 rows) | 149.99 | 148.51 | 150.43 | **−1.13 %** | 0.29 % |
| C8 (16 rows) | 249.61 | 244.78 | 248.31 | **−1.68 %** | 0.52 % |

Per class (base1 / cand1 / base2), C8: prose 242.2 / 238.6 / 241.3, code 252.0 / 245.8 / 252.2,
json 242.0 / 235.9 / 238.4, math 258.6 / 254.8 / 256.0, chat 253.2 / 248.7 / 253.6; C4: prose
147.6 / 145.2 / 145.8, code 151.9 / 150.8 / 152.1, json 150.4 / 149.2 / 150.9, math 148.8 / 147.2 /
150.0, chat 151.4 / 150.1 / 153.3. C1 and C2 greedy transcripts identical across the three legs
(30/30 each). The bench's projection was −1.2..−1.5 ms/step at eight rows and −2.5..−2.8 at sixteen
(×75 layers); the fabric read −1.5 and −4.8 ms/step against base1, −1.9 and −3.5 against base2.

### `deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json` — HawkBearPig/GLM-5.3-Flash-NVFP4-FP8, 4 slots, MTP depth 1 (regression check only)

The user asked for a regression check, not a benchmark, on the production model: one full baseline
leg (`flash_base1`, C1–C4, `--repeat 2`) and one short candidate pass (`flash_cand_short`: C1 and C4,
every class once, the same prompts). Its eleven DSA layers take the new kernels from two rows up,
so every MTP step runs them.

| | base1 | cand (short) | |
|---|---:|---:|---|
| C1 (2 rows) | 31.73 | 31.70 | −0.1 % |
| C4 (8 rows) | 63.68 | 63.45 | −0.4 % |

C1 greedy transcripts identical, 5/5 classes; engine decode 57.7–62.5 tok/s at C1 and 110–111 at
C4, the benchmark table's range. No regression; the Flash cells of `docs/benchmarks.md` are not
re-measured for this change (the user's call).

## Suite

`ctest --test-dir build-ci` on the changed tree (`raw/ctest-full.txt`): 163 passed, 2 skipped (the
MiMo tokenizer / chat-template tests, no checkpoint on this build), 2 failed — `site_env_test`
(`tests/fixtures/cluster.resolved.json` still says `default_max_tokens` 256 after master's 29bd5b2
raised every template to 32768) and `qwen_chat_template_test` (`qwen_tool_grammar_accepts_the_golden_
turns_over_the_real_tokenizer`, id 271) — both on master before this change, neither near it.

## Files

`raw/glm_{base1,cand1,base2}.*`, `raw/flash_base1.*`, `raw/flash_cand_short.{json,load.txt}` (`.{json,load.txt,metrics.json,up.txt,down.txt,version.txt}`),
`raw/ctest-full.txt`, `kvb_bench.cu` (the context microbench; build against build-ci's
`libdgpp_kernels.a` as in `~/claude-scratch/2026-09-21-bf10-context-bench/build_bench.sh`),
`leg.sh` / `summarize.py` (the legs and their reading).

## Integration into master (2026-09-27)

Merged `kvb-decode-rows` at `8a0e7cc` into master at `14fb039`. The merge
was conflict-free. Neither `src/kernels/dsa.cu` nor `tests/cuda/dsa_test.cu`
had changed on master since the branch's `29bd5b2` base; the integrated
implementation and kernel regression are identical to the measured branch.

Fresh integration checks:

```bash
cmake --build build-ci --target dsa_test dgpp_serve_app unit_tests scheduler_test serve_test -j 2
ctest --test-dir build-ci -R '^(unit_tests|scheduler_test|serve_test)$' --output-on-failure -j1
git diff --cached --check
```

The CI build passed with warnings treated as errors, including the CUDA
kernel, its regression-test executable, and the server. All three host suites
passed: 276 unit cases, 88 service cases, and 73 scheduler cases (437 total).
The whitespace check passed. GPU execution and fabric benchmarks were not
repeated during integration because production serving occupied the GPU.
The GPU and performance evidence above remains the September 26 validation
of the unchanged kernel implementation; it is not a new measurement of
the combined master tree.
