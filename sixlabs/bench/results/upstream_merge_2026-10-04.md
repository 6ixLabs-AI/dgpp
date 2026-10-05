# Upstream DGPP 769dc5e merged into 6ix.cpp — build and tests, DGXone, 2026-10-04

Branch `merge/upstream-2026-10-04`: origin/main `997071d` + upstream master `769dc5e`
(HawkBearPig/dgpp), merge commit `d7af6f5`, built and tested at `01e0c26` (the merge plus a
`6IXSERVE.md` note; the engine source is the merge's).

**State: builds clean, every host test passes; on main since 51408f8 (Mark, the same evening).
The checks of section 4 were not run by this work: the 35B's one-box gate on the merged engine,
upstream's GPU kernel tests and the Qwen3.8-27B bring-up are still owed.**

Later the same evening (not part of this record's build): 9612a02 lifted the one-box refusal of
section 2 and the 80B and the 35B were served and measured over both boxes on the merged engine
(`dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md`: 35B 154.4 / 214.1 / 300.5 / 367.6 tok/s at
1/2/4/8 streams). Those figures are measured, not verified: the gate below has not been run on
any build that contains this merge.

## 1. What came in

Thirteen upstream commits, all dated 2026-10-03/04 (two more in the range, 565d097 and e65796e,
are the Prometheus commits main already had):

| commit | what | reaches which model here |
|---|---|---|
| 6c75c40 (#86) | `qwen3_5` on two and four Sparks: tensor parallel for the dense dialect | Qwen3.8-27B on two boxes |
| 7944975 (#89) | the scale GEMM's large-row FP8 products run on the fp8-weight GEMM (63–66 TF against 26–30; bitwise the same result) | prompt reading, every family that routes large FP8 products there |
| e6e928a (#92) | the DFlash2 block drafter on the graph worlds (the fabric, or one Spark with the decode graph) | Qwen3.8-27B |
| d97649b (#93) | boundary prefetch windows (world 2 and up only) and the dense MLP's split-K workspace | Qwen3.8-27B; the prefetch part needs two boxes |
| ecf0e33 | one Qwen3.8-27B template per world size; `_w1` is now the DFlash2 recipe, `_w1_dflash2` is gone | Qwen3.8-27B |
| 24330a7, dfaa140, 2c1b5b5 | the Qwen3.8-27B campaign, its benchmark rows and notes | documentation |
| c3c0aa1, 1b21b5f, 3358326, 9c52927, 769dc5e | the Prometheus exposition as upstream merged it (PR #81): peer metrics from the deployment JSON, bind hostnames resolved, the last inter-token gap recorded before a record retires | every family |

For the 35B and the 80B on one Spark the decode path is unchanged by these commits: the routed-MoE
dialects take none of #86, #92 or #93, and #89 is on the prompt-reading side. Expect their
ShareGPT figures to move little; the measurement in section 4 is what will say.

## 2. Conflicts and what was chosen

Three files. `LICENSE` and `NOTICE` are upstream's, unchanged.

`src/models/qwen/model35.cpp` (five places):

| place | upstream | ours | merged |
|---|---|---|---|
| session parameters | this rank's vocabulary slice; position ceiling from the checkpoint | whole vocabulary; ceiling from `context_limit()` (the YaRN ramp lifts it) | upstream's slice, our ceiling |
| `dense_mlp` width | the resident FP8 gate rows (a rank's slice) | the config's width; a BF16 branch for Qwen3.5-0.8B | FP8 form as upstream; the BF16 branch reads the loader's local width (it has no FP8 image). One Spark gets the number it had |
| the draft layer's MLP | staged buffer + boundary fold around the dense MLP | the routed MoE or the dense MLP | the stage and fold around whichever runs |
| `run_rows`' MLP | the same, plus the prefetch window | the same choice | the same, the prefetch kept |
| end of `run_rows` | join the prefetch stream | `finish_run` with the packed span heads (`/v1/score`, shared walk) | both, in that order |

One addition that is neither side's (and stood for two hours: 9612a02 replaced it with the
sharded chain): **the routed-MoE dialects (Qwen3-Next 80B, Qwen3.6-35B-A3B,
Qwen3.5-122B-A10B) are refused above one box**, in the constructor and in `plan_memory`. Upstream
removed the family's "world must be 1"; the MoE chain here is sized for one Spark and has never
run sharded, and under the new folds its output would be summed once per rank. The refusal names
the dialects.

`src/models/qwen/model35.hpp`: both sides' declarations.

`deploy/README.md`: upstream's three Qwen3.8-27B rows replace our two (ours were the merge base's,
untouched); our thirteen rows for other models stay. The paragraph on templates without MTP keeps
upstream's wording for the drafter and our sentence about a checkpoint with no draft layer.

Merged without conflict but looked at: `apps/dgpp_serve.cpp` (the family table, the prefill-budget
refusal that passes `qwen3_5`, the single-rank refusal of `decode_passes_per_prefill` /
`prefill_order` / `prefill_budget_per_reader`), `src/engine/graph_engine.hpp` (scoring stays
world-1), `src/kernels/sample_pick.hpp` (`kSampleVerdictRows` 6 → 8 for the drafter's block of
seven; the raw-logprobs kernel has its own shared memory).

`sixlabs/registry/models.json`: Qwen3.8-27B gains worlds 2 and 4 (status `upstream`, upstream's
published rows as `measured`) and loses the `dflash2` variant that named the removed template.

## 3. Build and tests

DGXone (aarch64, GCC 13.3.0, nvcc 13.0), tree `~/6ixinfer-check` at `01e0c26`, incremental in the
directories built from main that morning, nice 10, `-j 6`. Nothing else on the box but the fleet's
reranker, embedder and Docling (96 GiB available); no engine up.

| | main (as recorded at fb3ad6a / 1b2d21d) | merged |
|---|---|---|
| `release` build (`dgpp_serve_app`, then the test and check targets) | clean, no warnings | clean, no warnings (77 s + 83 s) |
| `ci` build (everything, warnings are errors) | clean, no warnings | clean, no warnings (208 s) |
| `unit_tests`, both presets | 516 of 516 | **517 of 517** |
| `serve_test`, both presets | 102 of 102 | **103 of 103** |
| `scheduler_test`, both presets | 86 of 86 | **86 of 86** |
| `fabric_serve_test`, both presets | not recorded | 4 of 4 |
| `http_server_test`, both presets | not recorded | 18 of 18 |
| `registry.py check` | OK | OK (21 models, 14 families) |
| `test_registry.py` | 18 of 18 | 18 of 18 |
| `portability_test.py` | 19 of 19 since e0236ea | 19 of 19 |

The two new cases are upstream's: `dflash2_sampled_verify_accepts_by_probability` (unit, #92) and
`serve_interTokenMetrics_surviveRetirementBeforePassPublication` (serve, 9c52927). No engine was
on the GPU, so the three `arena_*` unit cases ran.

Text layer, the same battery and the same stand-in cache (`~/6ixinfer-logs/integration-hfcache`)
as the integration batches of that morning, compared log by log with `integration-cmp.sh` against
`combo2-release` / `combo2-ci` (the tree main took as 1b2d21d). In both presets **34 logs are
identical** after timestamps and timings are removed. Three differ, none from the merge:

- `portability_py`: the baseline batch predates e0236ea and has its one error; now 19 of 19.
- `qwen3moe_tool_calls`: skipped. Its checkpoint (`nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4`)
  left DGXone's cache that afternoon (f619190); the baseline's 4 of 4 was against the real files.
- `registry_unittest`: the run time in its last line.

| text test | verdict (both presets) | same as main's batch |
|---|---|---|
| tokenizer, Qwen3.8-Flash-Next corpus | 2 of 2 | yes |
| chat template, Qwen3.8-Flash-Next corpus | 10 of 10 | yes |
| tokenizer, DeepSeek-V4-Flash-0731 | 2 of 2 | yes |
| prompt, DeepSeek-V4-Flash-0731 | 3 of 3 | yes |
| tool calls: Qwen3-Next-80B, Qwen3-VL, Qwen3-Coder-Next | 4 of 4 each | yes |
| tool calls, MiniMax on NVIDIA's tokenizer | 4 of 4 | yes |
| Gemma-4 26B: tokenizer, chat | 2 of 2, 5 of 5 | yes |
| MiniMax tokenizer on NVIDIA's repository | 2 of 2 | yes |
| probe, Qwen3.8-27B: tokenizer, template | 2 of 2, 10 of 10 | yes |
| probe, Qwen3.6-35B: tokenizer, template | 1 of 2, 8 of 10 | yes (the same failures as on main: the corpus is another checkpoint's) |
| GLM-4.7 stand-in: tokenizer, template | 1 of 2, 9 of 10 | yes (a stand-in tokenizer; the same failures as on main) |
| GLM-5.3, GLM-5.3 full, DeepSeek-V4.1, MiMo, Gemma-31B, Mistral, MiniMax (lukealonso) | not runnable: no such checkpoint on the box | yes, skipped there too |

Not re-run: the Gemma / MiniMax / Mistral / plain-Qwen3 bind checks and `glm_vision_frontend`
(upstream touched none of their sources).

Not run, because they need the GPU and Mark's go: upstream's kernel tests `qwen35_tp_test`
(loopback worlds of 2 and 4 against the world-1 forward on a tiny fixture — the only test of the
tensor-parallel path, which was hand-merged in `model35.cpp`), `fp8w_gemm_test` and
`dflash2_kernels_test`. They are built in `build-ci`.

Also checked on the Mac before the box was free: every `.cpp` of the merged tree through
`clang++ -fsyntax-only` against stand-in CUDA headers gives no error that clean main or clean
upstream does not give for the same file (318 files; what remains are Linux-only calls).

Logs on DGXone: `~/6ixinfer-logs/build-merge-upstream-{release,release-tests,ci}.log`,
`tests-merge-upstream-{release,ci}.log`, `integration-tests-merge-{release,ci}/`. The runner is
`~/6ixinfer-out/jobs/merge_host_tests.py`. The tree's earlier uncommitted state (the combination
that became 1b2d21d) is kept as `~/6ixinfer-logs/check-tree-before-reset-2026-10-04.patch`.

## 4. Not done yet — needs the GPU

As of 20:30 EDT both boxes serve the 80B and the 35B over two Sparks, so none of this has a
free GPU; items 1 and 2 are correctness and are owed whatever is decided about one-box speed.

1. The three kernel tests above.
2. Qwen3.6-35B-A3B: `qwen35_forward_check` on the port's prose and code inputs against the numpy
   reference (`compare_forward_check.py`, default bounds), then ShareGPT at 1/2/4/8 and
   `group_check.py --thinking-off --seed 4242`. Before: 93.2 / 128.5 / 176.7 / 222.7 output tok/s,
   p99 first token 366 / 327 / 409 / 510 ms on build 17fbf92
   (`dgxone_35b_prefill_2026-10-04.md`). Until the gate has passed on this build, the registry's
   `verified.engine_commit` for the 35B stays where it is.
3. Qwen3.8-27B from upstream's one-box template with the DFlash2 drafter: answers first
   (`group_check`), then ShareGPT. Upstream's own one-box rows (`docs/benchmarks.md`): 17.0–40.8
   tok/s single stream by prompt class, 45.4–96.3 at eight streams.
