# Integration check of the three overnight ports on DGXone (2026-10-04)

Written by the integration agent; saved by the main session from its final report (the agent's own
file write was refused). Engine: 6ix.cpp (binary `6ix-Serve`). Nothing was loaded on a GPU, no
engine was booted, no weights were read; bind checks read safetensors headers only.

## Brief

- **All three families and the combination build under `release` and `ci` and pass every unit test
  on DGXone** (GCC 13.3.0, nvcc 13.0.88). The combination was built at 5859c3f: 516 of 516 unit
  tests in both presets.
- **The four patches are rebased on 48cafb8 but not built there.** The rebase changed no added,
  removed or context line of any patch, only hunk offsets. GCC has not seen the 48cafb8 tree.
- **No text-layer regression on the checkpoints this box has** (Qwen3.8-Flash-Next,
  DeepSeek-V4-Flash-0731, Qwen3-Next-80B, Qwen3-Coder-Next, plus partial coverage on GLM-4.7-Flash,
  Qwen3.8-27B, Qwen3.6-35B). GLM-5.3, GLM-4.7 goldens, DeepSeek-V4.1 and MiMo could not be tested
  here.
- **One real break found and fixed:** the Gemma patch made main's `qwen35_bf16_template_is_sequence`
  fail. jinja2 3.1.6 confirms Gemma's behaviour is right, so the test's expectation was corrected.
- **GCC-only failures fixed:** 11 `-Wdangling-reference` warnings (errors under `ci`), 6 in Gemma
  and 5 in Mistral/MiniMax. Plain Qwen3 needed no compiler fix.
- **Combination-only problems fixed:** Gemma and Mistral disagreed on macro-argument rules
  (Mistral's matches Jinja and is kept), and a Mistral test used `map` as its "unsupported filter"
  example while Gemma adds `map`.
- **Drafts:** the plain-Qwen3 draft compiles and links entirely, server included. The Mistral and
  MiniMax kernels compile but their CUDA tests do not; the Gemma draft does not compile. All three
  failures have one cause. Nothing was run; options stay OFF.
- **Still to do when the box is free:** build the combination at 48cafb8, then Gemma and
  Mistral/MiniMax alone there, then a fresh baseline.

Files beside this report: `all-new-families.patch`, `gemma.patch`, `mistral-minimax.patch`,
`qwen3-plain.patch`.

## 1. Summary

| | builds `release` | builds `ci` | `unit_tests` | text layer vs baseline | built at |
|---|---|---|---|---|---|
| clean main | yes | yes | 391 of 391 | — | b577b34 |
| Gemma-4 | yes (6 warnings as delivered, none after fixes) | yes, after fixes | 443 of 443 after fix (442 before) | identical | b577b34 |
| Mistral-Small-4 + MiniMax-M2.7 | yes (5 warnings as delivered, none after fix) | yes, after fix | 444 of 444 | identical | b577b34 |
| plain Qwen3 | yes, as delivered | yes, as delivered | 411 of 411 | identical | 5859c3f |
| all three together | yes | yes | 516 of 516 | identical | 5859c3f |

- Gemma and Mistral/MiniMax were built on their own at b577b34 only. At 5859c3f they were built as
  part of the combination.
- `ci` was not run on the as-delivered Gemma or Mistral trees; the warnings seen in `release` are
  errors under `-Werror`.
- Mistral's 444 ran on the as-delivered release build and on the fixed `ci` build. The fixed release
  build was rebuilt clean but its `unit_tests` was not re-run.

## 2. How it was run

- Every build went through `build_dgpp.py` at nice 10, `-j 6`, logged as
  `~/6ixinfer-logs/integration-<name>.log`.
- Before each build and each test binary: no measurement process, no fresh `*.wait` file, at least
  25 GiB available.
- main moved three times: b577b34 → 5859c3f → 48cafb8.
  - b577b34 → 5859c3f changed `apps/dgpp_serve.cpp`, the graph engine, `model35`, the scheduler and
    `cluster_config`. Nothing in `src/text`, `tests/unit` or the text host tests.
  - 5859c3f → 48cafb8 changed `src/serve/generation_service.cpp`, new `chat_content.hpp`,
    `serve_test.cpp`, registry JSON and records.
- Not run on purpose: `gemma4_host_check` and `qwen3_bind_check --host-forward` on real weights,
  every `tests/cuda/*` binary, every draft binary.
- Three `arena_*` unit cases allocate 1 MiB on the device. When an engine was on the GPU (base-ci,
  qwen3-release, combo-ci) they were run with `CUDA_VISIBLE_DEVICES` empty and skipped.

## 3. Baseline

Clean main at b577b34: `release --fresh` 561 s, `ci` 463 s, no warnings. `unit_tests` 391 of 391,
`serve_test` 98 of 98.

**As the box's HF cache stands, four existing text tests fail on clean main.** The repositories
their corpora name are stub snapshots holding only `config.json`:

- `qwen_tokenizer_test` (1 of 2 failed)
- `qwen_chat_template_test` (7 of 10 failed)
- `dsv4_tokenizer_test` (1 of 2 failed)
- `dsv4_prompt_test` (2 of 3 failed)

For a useful baseline `HF_HUB_CACHE` was pointed at a directory of symlinks,
`~/6ixinfer-logs/integration-hfcache`:

| corpus names | stand-in | accepted by the corpus hash check |
|---|---|---|
| `Qwen/Qwen3.8-Flash-Next-FP8` | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | yes |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | that repo's complete snapshot `9e165c30` | yes |
| `nvidia/GLM-4.7-NVFP4` | `cyankiwi/GLM-4.7-Flash-AWQ-4bit` | no: different tokenizer, so the golden differential refuses and the other cases run |

Not runnable on this box at all: GLM-5.3 (no `tokenizer.json` anywhere), DeepSeek-V4.1, MiMo-V2.6,
`glm_vision_frontend_test`.

Two probes were added using the Qwen3.8 corpus's binaries: `Qwen/Qwen3.8-27B-FP8` and
`nvidia/Qwen3.6-35B-A3B-NVFP4`. Their baseline logs come from main's `ci` build.

## 4. Per family

### 4.1 Gemma-4, alone at b577b34

- `gemma4_bind_check` on `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16`: `binding OK`, 35567
  expected and matched, 356 ignored, 35923 in headers. This matches the gpu-steps text line for line.
- `gemma4_chat_test` on the 26B corpus: 5 of 5, 30 renders byte-exact.
- Tokenizer goldens on the 26B's `tokenizer.json`: 198 of 198.
- The 31B checkpoint is not on the box, so its corpus and bind check did not run there.
- Existing text tests: 23 of 23 logs identical to baseline in both presets.

Fixes:

1. `src/models/gemma4/reference.cpp:203-205`: three `-Wdangling-reference` warnings on
   `const TensorInfo&` bound to `expect(...)` with temporary arguments. Nothing actually dangles;
   changed to copies.
2. `tests/unit/gemma4_binding_test.cpp:145,149,287`: same warning; copies, as main's Nemotron test
   does.
3. `tests/unit/qwen35_bf16_test.cpp` (main's test): it expected `{{ u is sequence }}` to be `False`
   for undefined `u`. Gemma's interpreter change makes it `True`, and jinja2 3.1.6's sandboxed
   environment renders the test's line as `True|True|True|False|False|True|True`. That one expected
   value was changed and the evidence noted in a comment.

### 4.2 Mistral-Small-4 and MiniMax-M2.7, alone at b577b34

- Three-way apply had 4 conflicts with main (`deploy/README.md`, `architecture.{hpp,cpp}`,
  `portability_test.py`); both sides kept.
- Neither `mistralai/Mistral-Small-4-119B-2603-NVFP4` nor `lukealonso/MiniMax-M2.7-NVFP4` is on the
  box, so no Mistral checkpoint test ran there.
- `nvidia/MiniMax-M2.7-NVFP4` has the same tokenizer and template hashes: tokenizer 472 of 472,
  `minimax_tool_calls_test` 4 of 4 (24 renders byte-exact).
- `minimax_bind_check` on that NVIDIA release is **refused**: `kv_cache_scheme: a KV cache scheme is
  not implemented`. The port accepts lukealonso's container only.

Fix: `tests/unit/mistral4_binding_test.cpp:139,155,157` and
`tests/unit/minimax_binding_test.cpp:142,158`, the same `-Wdangling-reference`; copies. The two plan
headings were changed from "6ixInfer" to "6ix.cpp".

### 4.3 Plain Qwen3, alone at 5859c3f

- `qwen3_bind_check` on `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4`: `binding OK`, 75090 == 75090 tensors,
  9.33 GiB per rank at world 2.
- `qwen3_bind_check` on `nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4`: `binding OK`, 145703 == 145703,
  67.56 GiB per rank.
- Both match the gpu-steps text exactly.
- `qwen3vl_tool_calls_test` and `qwen3moe_tool_calls_test`: 4 of 4 each on real files.
- The 80B's own corpus through the modified test file is unchanged.
- Reranker and Embedding 0.6B are not in the cache; not run.

One change, not a compiler fix: `apps/qwen3_bind_check.cpp` printed
`table count 310 == header count 311` for a checkpoint with a second stored tied head. It now says
`310 + 1 tied head copy == 311`. The pass/fail rule is untouched.

## 5. Shared text layer against real checkpoints

"=" means the log is identical to clean main's after removing timestamps and timings.

| test | clean main | Gemma | Mistral/MiniMax | Qwen3 | all three |
|---|---|---|---|---|---|
| tokenizer, Qwen3.8-Flash-Next | 2/2, 94 cases | = | = (+2 lines) | = | = (+2 lines) |
| chat template, Qwen3.8-Flash-Next | 10/10 | = | = | = | = |
| tokenizer, DeepSeek-V4-Flash-0731 | 2/2, 144 cases | = | = (+2 lines) | = | = (+2 lines) |
| prompt, DeepSeek-V4-Flash-0731 | 3/3, 64 renders | = | = | = | = |
| tool calls, Qwen3-Next-80B | 4/4 | = | = | = | = |
| tool calls, Qwen3-Coder-Next | 4/4 | = | = | = | = |
| tokenizer, GLM-4.7-Flash stand-in | 1/2 | = | = (+2 lines) | = | = (+2 lines) |
| chat template, GLM-4.7-Flash stand-in | 9/10 | = | = | = | = |
| probe, Qwen3.8-27B | 2/2 and 10/10 | = | = (+2 lines) | = | = (+2 lines) |
| probe, Qwen3.6-35B tokenizer | 1/2 | = | = (+2 lines) | = | = (+2 lines) |
| probe, Qwen3.6-35B template | 8/10 | = | one line differs in `release` | one line differs in `release` | = |
| `serve_test` | 98/98 | 98/98 | 98/98 | 98/98 | 98/98 |

No test that passes on main fails with any patch or with all three.

- "+2 lines": the Mistral patch adds two synthetic tokenizer loads to `tokenizer_test`, each
  printing one log line.
- The one differing line is in a case that fails on main too.
  `qwen_reasoning_effort_renders_each_level` dereferences an empty `std::optional` on the 35B, whose
  template reads no `reasoning_effort`. Its message varies between binaries; the Qwen3 patch, which
  touches no text file, shows the same variation.

**Not covered:** the Gemma and Mistral patches change code that GLM-5.3, GLM-4.7, DeepSeek-V4.1 and
MiMo use, and those golden differentials ran nowhere today.

## 6. The combination

`all-new-families.patch` is 162 files, exactly the union of the three patches' files.

Conflicts: Gemma against Mistral/MiniMax, 14 files and 36 hunks; Qwen3 on top, 6 files and 12
hunks. Registry files keep both sides. The text layer was merged by hand.

Three places where the patches disagreed:

1. **Macro arguments.** Gemma refuses a missing parameter and words keyword errors its own way.
   Mistral follows Jinja: errors read `macro 'm' takes no keyword argument '…'` and a missing
   parameter is undefined, which Mistral's template needs. jinja2 3.1.6 agrees with Mistral. The
   combination takes Mistral's code, and three expectations in
   `chat_template_gemma4_forms_test.cpp` were changed to match. `gemma.patch` alone keeps Gemma's
   rule.
2. **`dict.get` and the `list` filter** were added by both. Mistral's implementations are kept as
   the superset, with Gemma's argument check on `list`.
3. **`tests/host/chat_template_test.cpp`** used `map('upper')` as its unsupported-filter example;
   Gemma adds that filter. The example is now `reverse`. Without this the Qwen3.8 chat-template
   test fails in the combined tree only.

Results at 5859c3f: both presets build with no warning, `unit_tests` 516 of 516
(391 + 52 + 53 + 20), `portability_test.py` OK, `registry.py check` OK.

On the Mac with Apple clang and the earlier agents' header shim, final tree at 48cafb8:

- 128 of 128 text-layer unit tests.
- Mistral-Small-4: tokenizer 376 of 376, tool-calls test 5 of 5 (31 renders byte-exact).
- MiniMax on lukealonso's files: 472 of 472 and 4 of 4.
- Gemma 31B corpus: 5 of 5.
- `generation_service.cpp` passes a syntax check.

This is not the real toolchain, but it is the only place the Mistral and Gemma-31B files exist.

main's newer content-parts fallback (4a4ef27) retries when a template throws on array content, and
both patches make the interpreter throw less. On the Mac, main's text layer was compared with the
combination's on five served templates and five request shapes. They render the same bytes wherever
main renders and throw wherever main throws. GLM, DeepSeek and MiMo templates were not checked.

## 7. Draft builds

Combined tree at 5859c3f, all four options ON in `build-drafts`, warnings not errors, `make -k`.
Nothing built here was executed.

| family | target | result | cause |
|---|---|---|---|
| plain Qwen3 | loader, layers, `qwen3_loader_test`, `qwen3_forward_test`, `qwen3_forward_check`, `dgpp_serve_app` with the family | all compile and link, no warning | — |
| Mistral-Small-4 | `src/kernels/mistral4_attn.cu` | compiles | — |
| Mistral-Small-4 | `tests/cuda/mistral4_attn_test.cu` | 6 errors, one cause | reaches `loaders/minijson.hpp` via `mla_reference.hpp` → `config.hpp`; nvcc rejects that header |
| MiniMax-M2.7 | `src/kernels/minimax_attn.cu` | compiles | — |
| MiniMax-M2.7 | `tests/cuda/minimax_attn_test.cu` | 6 errors, same cause | via `attn_reference.hpp` → `config.hpp` |
| Gemma-4 | `src/models/gemma4/forward_draft.cu` | 6 errors, same cause | includes `config.hpp`, `binding.hpp`, `safetensors.hpp` |
| Gemma-4 | `gemma4_forward_check`, `gemma4_forward_draft_test` | not built | depend on the above |

Main already notes "minijson + nvcc don't mix" in three `.cu` tests. None of it was fixed: the agent
was stopped before a fix could be compiled.

- Mistral and MiniMax look small: forward-declare `minijson::Value` in the two `config.hpp` files
  and move the include to the `.cpp`.
- Gemma is structural: the checkpoint loading must move out of the `.cu`. nvcc never got past the
  includes, so none of the author's listed kernel risks has been tested.

## 8. Not done

1. Build and test the combination at 48cafb8, both presets.
2. Gemma and Mistral/MiniMax each alone at 48cafb8.
3. A clean-main baseline at 48cafb8.
4. The Mistral/MiniMax draft fix, then the draft build again.
5. The GLM-5.3, GLM-4.7, DeepSeek-V4.1 and MiMo checkpoint tests, on a machine that has them.

## 9. Before landing

- `sixlabs/registry/models.json` still says `on_main: false` and `branch: ports/…` for the four
  families. It was not touched; the check passes either way.
- `x is iterable` is still false for undefined `x` in the interpreter (Jinja says true). Neither
  patch changes it.
- `RedHatAI/gemma-4-26B-A4B-it-NVFP4`, also in this cache, is refused by the Gemma port (`only
  modelopt is implemented`).

## 10. Where things are

- **Mac copy:** `/private/tmp/claude-501/-Users-markgriffith/1c3e1a32-2dc7-44f5-9c50-ef42f168b6cd/scratchpad/integration`,
  branch `work/integration` at 48cafb8, the combination as uncommitted changes. No commits, no
  pushes.
- **Box:** `~/6ixinfer-check` is on `check` at 5859c3f with the combined tree uncommitted.
  `build-drafts` and a git-ignored `CMakeUserPresets.json` are new and can be deleted.
- **Logs:** `~/6ixinfer-logs/integration-*.log`, `integration-tests-<name>/`,
  `integration-hfcache/` (symlinks only), `integration-cmp.sh`.
- One log was overwritten: the first `integration-mistral-ci` attempt failed at configure because
  the agent's own sync script had removed the family's new files. The script was fixed and re-run;
  no result here comes from that attempt.

## Addendum — the combination at fb3ad6a, the commit it landed on (13:30–13:50 EDT)

Sections 1 and 8 above describe 5859c3f / 48cafb8. After the benchmarks freed the box the
combination was built at fb3ad6a, incrementally in the directories built from scratch that morning:

| | clean main (fb3ad6a) | combination |
|---|---|---|
| `release` build | clean, no warnings | clean, no warnings |
| `ci` build | clean, no warnings | clean, no warnings |
| `unit_tests` (both presets) | 391 of 391 | 516 of 516 |
| `serve_test` (both presets) | 102 of 102 | 102 of 102 |
| `registry.py check`, `test_registry.py` | OK, 18 of 18 | OK, 18 of 18 |
| `portability_test.py` | 18 of 19 | 18 of 19, same error; 19 of 19 with the fix that follows this commit |

- No engine was on the GPU, so the three `arena_*` unit cases ran rather than skipped.
- Text layer: every existing test gives the same verdict as the fb3ad6a baseline. The Qwen3.6-35B
  probe's one varying line flipped on clean main itself this time: it is that test's undefined
  behaviour on that checkpoint, not a patch effect.
- Family tests in the combination, both presets, as at 5859c3f: Gemma 26B bind OK (35567 of 35567),
  chat 5 of 5 (30 renders byte-exact), tokenizer 198 of 198; MiniMax on NVIDIA's repository
  tokenizer 472 of 472, tool calls 4 of 4; Qwen3 bind OK on VL-30B (75090) and 235B (145703),
  tool-call corpora 4 of 4 each.
- The four patch files are byte-identical to the ones rebased on 48cafb8; fb3ad6a changed no file
  they touch. Only the combination was built at fb3ad6a. The draft fix (section 7) is not done.
- The `portability_test.py` failure came from fb3ad6a itself: the new Sehyo 122B template had no
  entry in the test's table and no row in `deploy/README.md`.
