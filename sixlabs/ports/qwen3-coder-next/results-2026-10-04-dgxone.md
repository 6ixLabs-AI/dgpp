# Qwen3-Coder-Next on 6ix.cpp — bring-up results (DGXone)

(The engine was renamed 6ix.cpp on 2026-10-04 at about 10:22 EDT; it was 6ixInfer / 6ixServe before. The binary is still `6ix-Serve`; paths and ports are unchanged.)

**Finished 2026-10-04 11:50 EDT (DGXone clock).** Nothing below is claimed beyond what the quoted output shows.

## Status in one line

**Partly working, and not called verified.** Qwen3-Coder-Next loads, binds all 296,151 tensors,
serves on 6ix.cpp in the checkpoint's own dense form, answers chat, code and tool-call requests
correctly and has been measured (load, memory, ShareGPT, tool-eval) beside the fleet's vLLM
slot. But: the numerical gate's verdict on the original code prompt is FAIL and the model is
served under an exception the owner accepted on 2026-10-04; the template's default `fp8` dense
form fails the gate on more prompts and is refused; one greedy miss on a control prompt is not
explained by a reference run (below); a real serving bug was found (content-part arrays refused
with 400) and fixed by the orchestrator, not by me; and tool-eval shows one more thing that
looks like the engine (structured output after a tool call).

**Plainly, on speed: 6ix.cpp is slower than vLLM on this model at every ShareGPT level** —
40.4 / 65.6 / 97.6 / 124.1 output tok/s at 1 / 2 / 4 / 8 streams against 67.1 / 107.3 / 148.7 /
192.9 (0.60–0.66 of vLLM), with the same time to first token. **This checkpoint has no draft
head**, so there is no speculative decoding for it: the bind check reports `+ 0 draft` and the
memory plan `mtp off`. (The orchestrator adds that all three Coder-Next checkpoints carry 0
`mtp.*` tensors where the 80B's carries 1,553; I verified only the one served here.)

**Open correctness question, stated up front:** on a fifth code prompt (my control, not one of
the owner's three) the checkpoint-form engine misses the reference's greedy token at row 28 at a
0.56 nat margin, and the BF16-router reference does NOT reproduce that miss the way it reproduces
position 22 (section 4.6). Every layer of that row passes the one-step check and its one outlier
layer is a near-tie re-routing, so I see no wrong tensor — but it is not "explained the same way
as position 22" by a reference run. The further reference variants that might settle it were
not run: they need 8.7 GiB of host memory and the engine was to stay up (24–27 GiB available).

## What happened, in order (times are DGXone's)

| time | event |
|---|---|
| 09:06–09:08 | checkout to `29f904b`, build `DONE` in 69 s, host tests and loader fixture pass |
| 09:08–09:19 | bind check, memory plan, numpy reference, forward check: **gate FAIL on the code prompt in both dense forms**, PASS on prose |
| 09:20–09:36 | HOLD for another session's GPU test (`hold-ack.txt` written 09:20:32; "GPU released" at about 09:36). CPU-only analysis from files meanwhile |
| 09:32–09:49 | one-step-per-layer check, reference at BF16 router, router ties, row traces, negative controls |
| 09:50–10:42 | vLLM baseline (slot up 09:56, `bench_decode`, fleet suite, ShareGPT with the fixed script, first requests), slot down 10:42:29 |
| 09:57, 10:51 | owner's conditional exception relayed; 10:42–10:54 the three extra code prompts and a control through the gate; decision "boot the checkpoint form, not fp8" |
| 10:55–10:58 | 6ix.cpp cold boot (84.9 s), first requests (29 of 29 checks), warm boot (10.9 s) |
| 10:58–11:07 | suite on 6ix.cpp: ShareGPT hung — **the engine refused content-part arrays with 400**; stopped |
| 11:08–11:21 | tool-eval short + hard on 6ix.cpp (unpatched build): 87 / 78 |
| 11:12–11:27 | the orchestrator wrote, built and deployed the content-parts fix and restarted the engine (I stood down) |
| 11:27–11:43 | ShareGPT on 6ix.cpp (patched build, launched by the orchestrator): 40.4 / 65.6 / 97.6 / 124.1 tok/s |
| 11:43:02 | I touched `~/6ixinfer-logs/benches-done` as instructed at the time; HEM unpaused 11:43:03 |
| 11:45–11:46 | that instruction was withdrawn (more benches follow, by another agent); **the orchestrator removed the sentinel and re-paused HEM** (new hold job pid 636602). HEM was unpaused for about three minutes. I do nothing further on the box |

## Steps at a glance

| step | state |
|---|---|
| A. update to main, rebuild, host tests, loader fixture | **done, all pass** |
| B.1 bind check on the real checkpoint | **pass**, 296,151 of 296,151 |
| B.2 memory plan | **pass**, 43.35 GiB weights (fp8 form), `mtp off` |
| B.3 numpy reference on the real checkpoint | **done**, sane (−1.81 / −2.74 nat per token; both convention switches collapse it) |
| B.4 the numerical gate (forward check) | **prose PASS in both dense forms; code FAIL in both dense forms** — greedy condition only (row 22 in both forms, row 8 in the checkpoint form). Correlation and mean are inside the gate on all four runs. One-step-per-layer check: no wrong tensor (section 4.3). |
| Gate exception (owner's two conditions, relayed 09:57) | condition 1: holds (4.4). Condition 2: **holds for the checkpoint dense form (3 of 3 new code prompts PASS), fails for the `fp8` dense form** (4.5). Owner's decision 10:51: serve the checkpoint form only. |
| B.5 boot + first requests on 6ix.cpp | **done, all 29 checks held** (checkpoint dense form): cold boot 84.9 s, warm 10.9 s, 67,889 MiB |
| C. 6ix.cpp measurements | **done as the owner restricted them**: boot times and memory, ShareGPT 40.4 / 65.6 / 97.6 / 124.1 output tok/s at 1 / 2 / 4 / 8, tool-eval short 87, hard 78. Not run on this side by his instruction: DecodeBench, llama-benchy, `bench_decode`. |
| D. stop the engine | **not done by me, on instruction**: the engine (pid 292731, started by the orchestrator at 11:25:36 on the patched build) was left up; another agent takes it down for the next benches |
| E. vLLM baseline | **done, slot down at 10:42:29** (container removed): 363 s load, 54,362 MiB; `bench_decode` 70.6 / 70.7 tok/s single stream, 96.8 / 151.7 / 259.2 at 2 / 4 / 8; DecodeBench 68.8 / 116.1 / 165.2 / 221.2; llama-benchy generation 71.3 / 114.3 / 166.3 / 217.6, prompt processing 3160 / 3708 / 4073 / 4622 tok/s at depth 0; ShareGPT 67.1 / 107.3 / 148.7 / 192.9; tool-eval short **93**, hard **77**. |
| F. cleanup | done as far as the instruction to leave the engine up allows (section F) |

Box: DGXone, aarch64. Checkout `~/6ixinfer-ports` at `29f904b` (`origin/main` when I updated it
at 09:06; main has moved since and the tree was not pulled again), no source change by me
(`fixes.patch` holds new gate input files only). Scratch `~/6ixinfer-out/coder`, logs
`~/6ixinfer-logs/`. `$REPO` = `~/6ixinfer-ports`, `$OUT` = `~/6ixinfer-out/coder`. Job scripts:
`jobs/` beside this file (the same files are in `$OUT/jobs/` on the box, except `teb_time.py`,
which ran on the Mac over copies of the reports). Copies of the outputs and logs quoted here are
in `out/` and `out/logs/`. Only the `release` preset is built, so every binary is
`build-release/…` where the notes say `build-ci`.

## A. Update, build, host tests — PASS

| step | command | result |
|---|---|---|
| update | `git fetch`; the three modified files (`sixlabs/bench/{bench_decode,compare_forward_check,run_fleet_suite}.py`) had an empty diff against `origin/main`, so `git checkout --` them, `git merge --ff-only origin/main` | at `29f904b`, clean |
| build | the logged job as given (`build_dgpp.py … --preset release`, six targets) → `~/6ixinfer-logs/build-coder.log` | `[2/2] 1/1 err=0 elapsed=69s` / `DONE`; 0 compiler errors, 0 warnings |
| `./build-release/unit_tests` | | `391 tests, 0 failed` |
| `DGPP_TEST_FILTER=qwen3codernext ./build-release/unit_tests` | | `12 tests, 0 failed` (expected 12) |
| `DGPP_TEST_FILTER=qwen3next …` / `=qwen3 …` | | `15 tests, 0 failed` / `54 tests, 0 failed` |
| `./build-release/qwen3codernext_tool_calls_test` | real tokenizer from the HF cache | `4 tests, 0 failed` (expected 4); "the grammar admitted 448 positions of the template's own calls" |
| `python3 -m unittest tests.python.portability_test…test_deployment_filenames_describe_their_settings` | | `OK` |
| loader fixture: `cd $OUT && $REPO/build-release/qwen35_ports_loader_test` | | `9 tests, 0 failed` (the notes say 3; main has grown to 9 — the three `qwen3codernext_loader_*` are among them and OK) |

Not built, so not run: `qwen3next_tool_calls_test`, `qwen3next_loader_test` (the 80B's own
regression binaries; they were not in the target list).

## B. gpu-steps sections

### 1. Bind check (headers only) — PASS, exact expected counts

```
$ ./build-release/qwen35_bind_check --model RedHatAI/Qwen3-Coder-Next-NVFP4
config: qwen3_next, 48 layers (36 GDN + 12 full attention) + 0 draft, hidden 2048, vocab 151936
mlp: routed MoE, 512 experts top-10, intermediate 512, shared expert 512
checkpoint: 10 shards, 296151 tensors in headers
binding: expected 296151 | matched 296151 (missing 0, dtype 0, shape 0) | unexpected 0 | out of scope 0 | vision skipped 0
quantized matrices: 73920
binding OK: every tensor of the table is present with its dtype and shape
$ ./build-release/qwen35_bind_check --model Intel/Qwen3-Coder-Next-int4-AutoRound      (must be refused)
qwen35_bind_check: Qwen3-Next quantization_config.quant_method: 'auto-round' (the AutoRound int4 release) is not implemented — …   rc=2
```

### 2. Memory plan (loads nothing) — PASS, matches the notes

`./build-release/qwen35_forward_check --model RedHatAI/Qwen3-Coder-Next-NVFP4 --plan 8,524288 [--dense-weights fp8]`

| dense form | model weights (resident) | kv pool | plan TOTAL device | + pinned | expected weights |
|---|---|---|---|---|---|
| `fp8` (the template's) | 43.353 GiB | 12.000 | 58.021 GiB | 1.172 GiB | ≈ 43.35 |
| checkpoint | 44.923 GiB | 12.000 | 59.301 GiB | 1.172 GiB | ≈ 44.92 |

Both say `mtp off`.

### 3. The numpy reference on the real checkpoint — runs, and is sane

`$OUT/jobs/run_reference.py` (CPU, 8 threads, peak RSS 8.7 GiB), log `~/6ixinfer-logs/reference-coder.log`.

| run | mean log-probability per token |
|---|---|
| code (107 tokens) | −1.81470 |
| prose (83 tokens) | −2.74165 |
| prose, `--variant gscale=mul` (wrong direction of the global scale) | −11.93121 (ln 151936 = 11.93) |
| prose, `--variant nibble=hi` (wrong nibble order) | −12.57543 |

A small negative number as shipped; each wrong convention collapses it. Holds.

### 4. The numerical gate — **prose PASS, code FAIL (greedy condition), in both dense forms**

The gate is `sixlabs/bench/compare_forward_check.py` as on main, default bounds, untouched.
`$OUT/jobs/run_forward_check.py code,prose --fp8-form`, log `~/6ixinfer-logs/forward-check-coder.log`
(the four verdict blocks are in `gate-coder-4-runs.txt` beside this file).

| input | dense form | pearson (≥ 0.995) | mean \|Δ\| (≤ 0.10) | max (position), reported | greedy rows agreeing | misses above the 0.5 nat margin | verdict |
|---|---|---|---|---|---|---|---|
| code | checkpoint | 0.999088 | 0.07187 | 0.603 (45) | 101 of 107 | **row 8 (margin 0.882), row 22 (margin 1.165)** | **FAIL** |
| prose | checkpoint | 0.999576 | 0.05009 | 0.365 (23) | 81 of 83 | none | PASS |
| code | `fp8` (serving form) | 0.998405 | 0.09153 | 0.905 (44) | 101 of 107 | **row 22 (margin 1.165)** | **FAIL** |
| prose | `fp8` | 0.999253 | 0.07111 | 0.437 (23) | 80 of 83 | none | PASS |

Conditions 1 (correlation) and 2 (mean) hold on all four. Condition 3 (greedy agreement wherever
the reference's top-2 margin exceeds 0.5 nat) fails on the code text. The verdict is reproducible:
the code text run a second time in both forms gives byte-identical `token logprobs` and `argmax ids`
lines (`~/6ixinfer-logs/forward-check-coder-repeat.log`).

`compare_states.py`: relative rms error grows smoothly with depth, 0.007 at layer 0 to 0.067 (code)
/ 0.037 (prose) at layer 47; every cosine ≥ 0.9977; no jump at any layer.

#### 4.1 The two rows

| row (0-based) | context | reference (float32) | engine, checkpoint form | engine, `fp8` form |
|---|---|---|---|---|
| 8 | `…You are a careful programming assistant` → next | `.` −0.579, ` that` −1.461, ` who` −1.590 | picks ` that`; log P(`.`) −0.982 | picks `.` (agrees); log P(`.`) −0.786 |
| 22 | `…a Python function that returns the n` → next | `-th` −0.354, ` th` −1.520, ` smallest` −3.550 | picks ` th`; log P(`-th`) −0.898 | picks ` th`; log P(`-th`) −0.974 |

On the engine's own final residual the reference's head gives ` th` −0.659 / `-th` −0.897 at row 22
and ` that` −0.633 / `.` −0.981 at row 8: the engine's margins there are 0.24 and 0.35 nat.

#### 4.2 Where the rows part from the reference: they drift, they do not jump

`$OUT/jobs/row_drift.py` over the dumps (`$OUT/row-drift-code.txt`): the row's distance from the
reference's own trajectory, |engine − reference| / |reference| of the residual.

| after layer | median row | row 8 | row 22 |
|---|---|---|---|
| 1 | 0.4 % | 0.2 % | 0.3 % |
| 2 | 0.5 % | 0.5 % | **2.0 %** (its layer update is off by 5.3 %, median 1.1 %) |
| 3 | 0.7 % | 0.8 % | **2.6 %** (update off by 12.2 %, median 2.3 %) |
| 12 | 1.9 % | 2.1 % (update off by 12.6 %, median 5.0 %) | 2.4 % |
| 23 | 2.8 % | 3.6 % | 3.1 % |
| 35 | 4.0 % | 6.1 % | 7.0 % |
| 47 | 6.1 % | 15.2 % | 17.4 % |

After the last layer rows 22 and 8 are the 2nd and 6th furthest of 107 (the furthest, row 7, is
at 21 % and does not change its greedy token).

#### 4.3 One step per layer (the earlier agent's check): no wrong tensor

`$OUT/jobs/diag_transfer.py CN …` (the Qwen3.6 bring-up's script with this checkpoint added), CPU
only. For each layer the reference's layer is applied to the **engine's own** input and compared
with the engine's output, row by row; nothing accumulates. `err` is a fraction of the layer's
update, a BF16 floor is what storing that row in BF16 costs.

| run | median row err: typical layer / worst layer | in BF16 floors: typical / worst | outlier rows | re-routed at a near tie | exchanged experts' logits apart, BF16 steps: median / max | unexplained |
|---|---|---|---|---|---|---|
| Coder-Next code, checkpoint form | 1.35 % / 1.57 % | 1.67 / 2.65 | 309 of 5136 (6.0 %) | 295 | 0.52 / 1.28 | 14 |
| Coder-Next prose, checkpoint form | 1.37 % / 1.83 % | 1.62 / 2.37 | 200 of 3984 (5.0 %) | 192 | 0.49 / 1.20 | 8 |
| Coder-Next code, `fp8` form | 3.16 % / 4.53 % | 4.08 / 11.04 | 287 of 5136 (5.6 %) | 286 | 0.58 / 1.62 | 1 |
| Coder-Next prose, `fp8` form | 3.21 % / 4.52 % | 3.98 / 9.78 | 154 of 3984 (3.9 %) | 152 | 0.49 / 1.92 | 2 |
| *80B code, checkpoint form (the earlier agent's calibration, same ids)* | 1.39 % / 1.90 % | 1.60 / 2.94 | 360 of 5136 (7.0 %) | 337 | 0.50 / 1.20 | 23 |
| *80B prose, checkpoint form (same)* | 1.40 % / 1.75 % | 1.61 / 2.36 | 193 of 3984 (4.8 %) | 184 | 0.47 / 1.23 | 9 |

The head, one step (reference's final norm + lm head on the engine's last residual): token
log-prob |Δ| mean 0.0013 / 0.0016 / 0.0009 / 0.0022, max 0.012 / 0.008 / 0.007 / 0.010, greedy
rows differing 0 of 107 / 0 of 83 / 0 of 107 / 0 of 83.

Negative control on this checkpoint (first 4 layers, code): `gscale=mul` → every row off by 100 %
of the update, 600 floors; `nibble=hi` → 30–100 % (typical 57 %), 130–590 floors. The engine as
built: 1.4 %, 1.7 floors.

Reading: in its checkpoint form Coder-Next is on a par with the 80B in every column. The `fp8`
form's per-layer error is about twice the checkpoint form's (3.2 % against 1.4 %): that is the
re-encoding of the 4-bit attention q/k/v/o and shared-expert matrices and the BF16 GDN projections
to block FP8, the lossy serving form by design; its end-to-end mean (0.092 / 0.071) is still
inside the gate. The unexplained rows are all small: 0.4–3 % of the update, no routing change
(the largest, layer 28 row 1 at 14.6 %, is a row whose BF16 floor alone is 2.9 %).

Logs: `~/6ixinfer-logs/diag-coder-{T-code,T-prose,T-code-fp8,T-prose-fp8,NEG-gscale,NEG-nibble}.log`
(copies in `out/logs/`).

#### 4.4 Evidence for the owner's condition 1: the miss is reproduced by changing only the router's rounding

**(a) The two router conventions on the reference's own trajectory — no engine involved**
(`$OUT/jobs/router_ties.py`, `~/6ixinfer-logs/diag-coder-ties-{code,prose}.log`). The engine's
routed MoE (shared code, `MoeRouterMode::SoftmaxTopk`) takes the router logits rounded to BF16 and
breaks ties towards the lower expert id; the numpy reference keeps float32 logits. At logits of
about −4 a BF16 step is 0.03125.

| | code (107 × 48 row-layers) | prose (83 × 48) |
|---|---|---|
| row-layers with an EXACT BF16 tie at the 10th place | 1101 of 5136 (21 %) | 655 of 3984 (16 %) |
| row-layers where the float32 top-10 ≠ the BF16-rounded top-10 | 602 (11.7 %) | 365 (9.2 %) |
| rows touched at least once | 106 of 107 | 81 of 83 |
| exchanged experts' float32 logits apart, in BF16 steps: median / max | 0.32 / 0.95 | 0.34 / 0.91 |

Row 22: the two conventions route it differently at 9 layers — **first at layer 2** (float32
picks expert 297 at −4.34242, BF16 picks 143 at −4.35816: half a BF16 step apart), then layer 3
(experts 400 / 185 at −4.68354 / −4.68401: 0.02 of a step), 6, 17, 19, 21, 26, 29, 33. Layers 2
and 3 are exactly where the row leaves the reference's trajectory in 4.2. Row 8: layers 32, 37
(0.68 of a step), 47.

**(b) The reference re-run with ONLY the router convention changed** (same weights, same container
conventions, same code; `$OUT/jobs/diag_router.py`, `~/6ixinfer-logs/diag-coder-{R,V}-code.log`):

| row | reference as shipped (float32 router) | reference with BF16 router logits | engine |
|---|---|---|---|
| 22 | `-th` −0.354, ` th` −1.520 (margin 1.165) | **` th` −0.508, `-th` −1.107 — it makes the engine's choice** | ` th` |
| 8 | `.` −0.579, ` that` −1.461 (margin 0.882) | `.` −0.874, ` that` −1.138 (**margin 0.264**) | ` that` (checkpoint form), `.` (fp8 form) |

The engine against that reference, same gate, same bounds:

| engine run | pearson | mean | max (position) | greedy misses above 0.5 | gate | original 0.06 / 0.5 gate |
|---|---|---|---|---|---|---|
| code, checkpoint form | 0.999489 | 0.04947 | 0.469 (40) | none | PASS | would pass |
| code, `fp8` form | 0.998946 | 0.07251 | 0.698 (44) | none | PASS | would fail (mean) |
| prose, checkpoint form | 0.999491 | 0.05537 | 0.400 (70) | none | PASS | |
| prose, `fp8` form | 0.998932 | 0.08063 | 0.458 (48) | none | PASS | |

The difference at the scored token of row 22 (log P(`-th`)): engine −0.898, shipped reference
−0.354 (|Δ| 0.54); reference with the BF16 router −1.107 (|Δ| 0.21).

**(c) What does not support it, stated plainly.** With BF16 rounding of the residual stream and of
the sublayer inputs and outputs added on top of the BF16 router (`router+res+io`), the reference
goes back to `-th` at row 22 (−0.262 against ` th` −1.941, margin 1.68), and the engine fails the
gate against that variant at row 22 in both forms (row 8's margin there is 0.12). So row 22 is
not "always ` th` under BF16": it swings by more than 2 nat between two reference evaluations
that differ only in arithmetic. The reference against itself: BF16 router vs shipped mean 0.073,
max 0.75 at position 23 (= row 22); BF16 router+res+io vs shipped mean 0.071, max 1.51.

**(d) The engine's own two dense forms against each other** (`$OUT/forms-vs-each-other.txt`): code
mean 0.073, row 8 flips between them at that 0.88 nat reference margin; the 80B's two forms
differ by mean 0.101 on the same ids.

Condition 1 as worded ("a reference run that differs only in arithmetic precision or in the
router's tie-breaking makes the engine's choice at position 22"): **met by (b)**, with (c) as the
caveat. Row 8 was not named in the condition; the same run brings its reference margin from 0.88
to 0.26 nat, under the gate's 0.5.

#### 4.5 Condition 2: the gate on three more code prompts, both dense forms — **checkpoint form 3 of 3 PASS; `fp8` form 1 of 3 PASS**

Three new prompts, made the way the notes' inputs were made: the Mac tool that produced them
(`mkids`, the engine's own tokenizer and chat template; it regenerates both of the notes' inputs
token for token from the box's `tokenizer.json` and `chat_template.jinja`). Same shape as the
original (a system sentence, a user request for a function plus one sentence, a canned answer),
different tasks and languages, **and a different system sentence each** — the model is causal,
so a prompt that reuses the original's system sentence repeats the original's rows 0–10
computation exactly, row 8 included; the control below shows that. Inputs are in `inputs/`
beside this file and in `fixes.patch` (as `sixlabs/ports/qwen3-coder-next/inputs/…`).

| prompt | tokens | task |
|---|---|---|
| code2 | 103 | Python: palindrome check ignoring case and spaces, why O(n) |
| code3 | 107 | JavaScript: gcd by Euclid's algorithm, why it terminates |
| code4 | 110 | C: count the set bits of a 32-bit integer, how many iterations |
| code5-control (not one of the three) | 99 | the ORIGINAL system sentence and the same first 22 tokens, then a different Python task |

`$OUT/jobs/run_cond2.py code2,code3,code4,code5-control --engine-only` (references by
`ref_one.py`, 10:15–10:23), log `~/6ixinfer-logs/forward-check-coder-cond2.log`. The gate is the
unmodified tool, default bounds.

| prompt | dense form | pearson (≥ 0.995) | mean (≤ 0.10) | max (position) | greedy rows agreeing | misses above 0.5 nat | verdict |
|---|---|---|---|---|---|---|---|
| code2 | checkpoint | 0.999174 | 0.07069 | 0.674 (49) | 99 of 103 | none | **PASS** |
| code3 | checkpoint | 0.999037 | 0.06967 | 0.744 (30) | 101 of 107 | none | **PASS** |
| code4 | checkpoint | 0.998666 | 0.08865 | 0.939 (11) | 106 of 110 | none | **PASS** |
| code2 | `fp8` | 0.995902 | **0.13954** | 1.274 (41) | 96 of 103 | none | **FAIL** (mean) |
| code3 | `fp8` | 0.997864 | 0.09071 | 1.728 (30) | 102 of 107 | none | PASS |
| code4 | `fp8` | 0.997172 | **0.12468** | 1.974 (17) | 104 of 110 | **row 7 (margin 0.734)** | **FAIL** (mean and greedy) |
| *control* | checkpoint | 0.999137 | 0.06611 | 0.914 (88) | 96 of 99 | **row 8 (0.882) — the original prompt's row 8 again, same numbers; row 28 (margin 0.560)** | FAIL (greedy) |
| *control* | `fp8` | 0.998311 | 0.09951 | 0.987 (48) | 95 of 99 | none | PASS |

Against the owner's wording:
- "Correlation and mean must be inside the gate on all of them": **yes in the checkpoint form
  (3 of 3); no in the `fp8` form (2 of 3 outside on the mean; correlation inside on all).**
- "any greedy miss above the 0.5 nat margin must be isolated and explained the same way": the
  three prompts have none in the checkpoint form and one in the `fp8` form (code4 row 7: after
  `…system\nAnswer with working C code` the reference has `:\n\n` −1.167, ` that` −1.901; the
  `fp8` engine picks ` that`, the checkpoint-form engine picks `:\n\n`). It is not explained
  the same way: see 4.6.
- "If clear-margin misses turn up on most code prompts, that is a pattern": checkpoint form —
  the original prompt (rows 8 and 22) and none of the three new ones; `fp8` form — the original
  (row 22) and one of the three new ones (code4 row 7).
- The control, which I added and which is not one of the three: its row 8 is the original's row
  8 to the last digit (engine log P(`.`) −0.982 in the checkpoint form, −0.786 in `fp8`; the
  first 22 tokens are identical), so that miss is one event tied to that system sentence, not a
  second one. It also has a miss of its own at row 28 (`…largest element of a non-empty list` →
  reference ` of` −0.607, `.` −1.167; the checkpoint-form engine picks `.`), margin 0.56.

What the `fp8` form's larger mean is: the serving form re-encodes this checkpoint's 4-bit
attention q/k/v/o and shared-expert matrices and its BF16 GDN projections to block FP8, and the
one-step check (4.3) puts that at 3.2 % of a layer's update per layer against 1.35 % for the
checkpoint form (the 35B's `fp8` form: 1.56 %). Over five code prompts the `fp8` form's mean is
0.092 / 0.140 / 0.091 / 0.125 / 0.100 (average 0.109) against 0.072 / 0.071 / 0.070 / 0.089 /
0.066 (average 0.073) for the checkpoint form. It straddles the 0.10 bound; the checkpoint form
is inside it every time.

#### 4.6 The individual misses on the new prompts, and the one-step check on all of them

One step per layer on the new prompts (`~/6ixinfer-logs/diag-coder-T-{code2,code3,code4,code5c-rows,code2-fp8,code4-fp8-rows}.log`):

| run | median row err: typical / worst layer | BF16 floors: typical / worst | outlier rows | re-routed at a near tie | unexplained | head: greedy rows differing |
|---|---|---|---|---|---|---|
| code2, checkpoint | 1.32 % / 1.55 % | 1.69 / 2.72 | 284 of 4944 (5.7 %) | 276 | 8 | 0 of 103 |
| code3, checkpoint | 1.29 % / 1.57 % | 1.70 / 2.74 | 341 of 5136 (6.6 %) | 326 | 15 | 0 of 107 |
| code4, checkpoint | 1.29 % / 1.56 % | 1.69 / 2.68 | 346 of 5280 (6.6 %) | 326 | 20 | 0 of 110 |
| control, checkpoint | 1.34 % / 1.62 % | 1.69 / 2.70 | 278 of 4752 (5.9 %) | 271 | 7 | 0 of 99 |
| code2, `fp8` | 3.19 % / 4.44 % | 4.21 / 11.03 | 237 of 4944 (4.8 %) | 235 | 2 | 0 of 103 |
| code4, `fp8` | 3.16 % / 4.74 % | 4.21 / 10.97 | 280 of 5280 (5.3 %) | 279 | 1 | 0 of 110 |

The checkpoint form is the same on every prompt, and the same as the 80B (1.4 %, 1.6 floors).

The reference against ITSELF on these prompts, only its arithmetic changed
(`diag-coder-V-*.log`): mean 0.085–0.106 with the BF16 router and 0.084–0.108 with BF16 router +
residual + sublayer rounding (max 0.64–1.84 nat) — the same size as the engine-vs-reference means
above. On code3 the BF16-router reference flips its own greedy token at row 8 at a 0.61 nat
shipped margin; on code2 it turns a 0.44 nat margin at row 12 into 1.36 nat the other way.

**The control's row 8** (checkpoint form, margin 0.882): the original prompt's row 8, to the last
digit — same 22-token prefix, same computation. Explained as in 4.4: the BF16-router reference
brings the margin to 0.264, the BF16 router + residual + sublayer reference to 0.052.

**The control's row 28** (checkpoint form; `…largest element of a non-empty list` → reference
` of` −0.607, `.` −1.167, margin 0.560; the engine picks `.`; the `fp8`-form engine picks ` of`):

- One step per layer: row 28 is an outlier at exactly one layer, **layer 31, and it is a near-tie
  re-routing** — the engine exchanged expert 113 (float32 logit −4.07394) for 33 (−4.08535),
  0.37 of a BF16 step apart; after decoding that exchange the row is at 2.3 floors. At every
  other layer it is within 3.3 floors of the reference's layer applied to the engine's own
  input. The head on the engine's last residual gives the engine's token (0 of 99 rows differ).
- On the reference's own trajectory the two router conventions route row 28 differently at one
  layer (35; experts 506 / 140, 0.28 of a step apart).
- **But the reference variants do not reproduce the engine's choice here:** with the BF16 router
  the reference moves the other way (` of` −0.338, `.` −2.188, margin 1.85); with BF16 router +
  residual + sublayer rounding it is ` of` −0.649, `.` −1.282 (margin 0.63).

So for row 28: no wrong step at any layer, one near-tie re-routing, and a reference margin that
moves between 0.56 and 1.85 nat under arithmetic-only changes — but no reference run that lands
on the engine's token. It is the same mechanism as far as the one-step check can show, and it is
**not** "reproduced by a precision-only reference" the way position 22 is. Open; see the top.

**code4 row 7, `fp8` form only** (margin 0.734; for the record, this form is not served): four
near-tie re-routings in the one-step check (layers 9, 28, 38, 46; 0.38–0.70 of a step). The
BF16-router reference keeps `:\n\n` at a 0.564 nat margin, the BF16 router + residual +
sublayer reference at 0.496 — so this one is not reproduced by the router convention either; it
comes with the `fp8` form's larger per-layer error.

**Not done:** further arithmetic-only reference variants for the control's row 28 (float64; BF16
router with ties to the higher id; router + residual only). They were deferred while
measurements ran (nothing CPU-heavy of mine during a measurement) and then could not run at
all: each holds 8.7 GiB of host memory, and with the engine left up the node has 24–27 GiB
available, so the 20 GiB margin would not hold. To settle row 28, stop the engine (or use a
box with memory to spare) and run `jobs/diag_router.py CN code5-control` with those variants
added; the dumps it needs are in `~/6ixinfer-out/coder`.

### Gate exception: code prompt, position 22 (checkpoint dense form)

The gate's printed verdict for the original code prompt stays **FAIL** in both dense forms; the
tool was not edited. The owner accepted the checkpoint dense form on the following evidence on
2026-10-04 (decision relayed by the orchestrator at 09:57 as two conditions, and at 10:51 as
"boot the checkpoint form; do not boot or measure the fp8 form").

1. *Position 22 is reproduced without changing any weight or container convention* (4.4): the
   reference re-run with only the router's rounding changed to the engine's (BF16 logits, ties
   to the lower id) picks the engine's token at row 22 (` th` −0.508 against `-th` −1.107, where
   the shipped reference has `-th` at a 1.165 nat margin), and the engine then passes the
   unmodified gate against that reference in both forms and on both texts. The two conventions
   pick different experts for that row first at layer 2 (experts 297 / 143, logits −4.34242 /
   −4.35816, half a BF16 step apart), in 11.7 % of all row-layers of that prompt, and 21 % of
   row-layers have an exact BF16 tie at the 10th place. Caveat: the reference with BF16 residual
   and sublayer rounding as well goes back to `-th` (margin 1.68). The other miss on that
   prompt, row 8 (checkpoint form only, margin 0.882), falls to 0.264 under the same run.
2. *It is not systematic* (4.5): three more code prompts (103 / 107 / 110 tokens; Python, JavaScript,
   C) all PASS the unmodified gate in the checkpoint form — correlation 0.9987–0.9992, mean
   0.070 / 0.070 / 0.089, no greedy miss above the 0.5 nat margin.

What was refused: **the `fp8` dense form** (the template's default). On the same three prompts
its mean is 0.140 / 0.091 / 0.125 — outside the gate on two — with one new clear-margin miss
(code4 row 7, 0.73 nat) that the router convention does not explain; over five code prompts its
mean averages 0.109 against 0.073 for the checkpoint form. It is not booted and not measured.
Cost of serving the checkpoint form instead: **1.57 GiB more weights** (44.92 against 43.35 GiB;
planned device total 59.30 against 58.02 GiB).

What the exception does not cover: the control prompt's row 28 (0.56 nat), described in 4.6.

### 5. Boot and first requests on 6ix.cpp (checkpoint dense form) — PASS

Config: `deploy/cluster_qwen3-coder-next_nvfp4_w1.json` in `~/6ixinfer-ports` (a copy is beside
this file) = the template with **`"dense_weights": "checkpoint"`** and the prefill settings asked
for: `prefill_budget_tokens` 1024, `prefill_idle_budget_tokens` 4096, `decode_passes_per_prefill`
8, `prefill_order` `"shortest"` (all four are config keys; no `--knobs` was needed). Otherwise as
shipped: 8 request slots, 524,288-token BF16 K/V pool, `mtp` false, 4 GiB prefix cache, decode
graph on, alias `Qwen3-Coder-Next`. `.env` as the earlier agent left it (nodes 127.0.0.1, bind
172.17.0.1:18190, fabric/journal 29990/29991). Every `dgpp-cluster` call passed `--config`
(through `$OUT/jobs/engine_ctl.py`). Binary: `dgpp-serve 0.1.0+g29f904b8eaa7`.

```
$ python3 scripts/dgpp-cluster doctor --config deploy/cluster_qwen3-coder-next_nvfp4_w1.json
Preflight: 0 failed check(s). No files were installed and no services were started or stopped.
$ python3 scripts/dgpp-cluster up --config deploy/cluster_qwen3-coder-next_nvfp4_w1.json          (10:55:11)
rank 0 serving: ok (86s)   READY — curl http://172.17.0.1:18190/v1/models
serve log: serve: model family qwen3_next (…RedHatAI--Qwen3-Coder-Next-NVFP4…)
           rank 0: memory plan … model weights (resident) 44.92 GiB; … kv pool 12.00 GiB; …   (no mtp item)
           rank 0: memory plan total 64.51 GiB (59.30 GiB device + 5.21 GiB pinned) + 4.00 GiB headroom against 96.39 GiB free
           qwen35 loader: rank 0 resident image …/resident-cache/aca98768b71ca245.img — 0/48 layers present, direct I/O
           rank 0: prefill budget 1024 tokens/tick (0 = full prompt), 4096 with nothing decoding; 8 decode pass(es) per chunk while both are in flight; shortest prompt read first
           serve: the template writes its tool calls as <function=...> XML tags
           serve: tool calls available (the template's markers are in the tokenizer); tool_choice/parallel_tool_calls enforced by constrained decoding
           serve: listening on :18190 — Qwen3-Coder-Next (boot 84.9s)
```

Every line the notes ask for is there. Two `WARN` lines at boot, neither a fault: the
checkpoint's `generation_config.json` omits temperature, min_p and repetition_penalty (neutral
fallbacks are used), and the 524,288-token pool exceeds this family's 262,144-token positional
ceiling (the pool is concurrency headroom; no single request can pass 262,144). 0 `ERROR` lines.
(Not tried: `"mtp": true`, which the notes say must be refused for this checkpoint.)

**First requests** (`$OUT/jobs/first_requests.py http://172.17.0.1:18190 Qwen3-Coder-Next`,
output `$OUT/first-requests-6ixcpp.txt`, copy in `out/`; temperature 0) — **all 29 checks held**:

| request | what came back | verdict |
|---|---|---|
| raw completion, "The capital of France is" | `" Paris. The capital of Germany is Berlin"` | ok |
| the notes' plain chat, "Write a Python one-liner that reverses a string s." | a fenced block containing `s[::-1]`, `finish_reason` `stop`, no template token, no `reasoning_content` | ok |
| "What is 17 * 23? Answer with the number." | `391`, `stop` | ok |
| code writing: `fib(n)`, iterative, with a docstring (77 tokens) | a correct function: docstring, `ValueError` for negative n, `a, b = 0, 1`, the loop, `return a`; parses as Python (`ast.parse`, not executed); `stop` | ok |
| code writing: class `Stack` with push / pop / peek / is_empty (111 tokens) | a correct class, `IndexError` on an empty pop and peek; parses; `stop` | ok |
| forced tool call (`tool_choice` `required`) | `tool_calls[0]` = `{"type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}`, `finish_reason` `tool_calls`, `content` null | ok |
| auto tool call (`tool_choice` `auto`) | the same call, `finish_reason` `tool_calls`, `content` null — the model wrote the `<function=…>` form and the parser took all of it | ok |
| auto, a question that needs no tool | `content` `"ready"`, no tool call | ok |
| the tool result fed back (second turn) | `"The current weather in Paris is 18°C with light rain."`, `stop`, no further call | ok |
| `reasoning_effort` `high`; `enable_thinking` false | HTTP 200 both, a plain greeting | ok |
| an `image_url` part | HTTP 400 `the served model does not support image inputs`; the server keeps answering | ok |

**Looked at closely, as asked — the code-writing output against vLLM's.** The same script was
run against the vLLM slot before it was stopped (`out/first-requests-vllm.txt`). Both code
answers (77 and 111 generated tokens, greedy) are **byte-identical on the two engines**, which
serve different quantizations of the model (NVFP4 here, int4 AutoRound there). The prompt token
counts agree on every request (21, 23, 41, 39, and 285 / 286 / 334 for the three requests that
carry the tools header), so the engine's chat template renders the same number of tokens as
vLLM's jinja, tools included. The tool calls are the same call. The only differences are in
free text: `ready` against `Ready`, and the wording of the greeting.

## C. Measurements — 6ix.cpp, RedHatAI NVFP4, checkpoint dense form

### C.1 Boot and memory (observations from the boots themselves)

| boot | conditions | launcher "serving: ok" | serve log `boot` | plan total | `nvidia-smi` |
|---|---|---|---|---|---|
| cold (no resident image for this model existed; the first boot) | launched 10:55:11 on a quiet reading (after 22 s waiting for another session's build); the GPU log has no other GPU process above 5 % and no build in any of 6 samples | 86 s | **84.9 s** | 64.51 GiB (59.30 device + 5.21 pinned) + 4.00 headroom | 67,888 MiB |
| warm, from the resident image (48 of 48 layers present) | launched 10:57:44 on a quiet reading (after 22 s waiting for that session's test binary) | 12 s | **10.9 s** | 64.51 GiB | **67,889 MiB = 66.30 GiB** |

"Cold" means no resident image: the loader reads the 44.3 GiB checkpoint, builds the resident
forms and writes a 43.8 GiB image (`~/6ixinfer-out/resident-cache/aca98768b71ca245.img`,
46,991,131,392 bytes) as it goes. The page cache was not dropped (I cannot); the checkpoint had
been read by the reference runs minutes before, so this is not a cold-disk figure. The process
holds 1.79 GiB more than the plan's total (66.30 against 64.51). With the engine up the node has
27 GiB available (HEM, paused, still holds its memory; the 0.8b slot is up).

### C.2 Tool-eval short and hard on 6ix.cpp (11:08:17–11:21:01) — done

By the owner's instruction the 6ix.cpp side runs ShareGPT and tool-eval only: no DecodeBench, no
llama-benchy, no `bench_decode.py`.

`python3 sixlabs/bench/run_fleet_suite.py --host 172.17.0.1 --port 18190 --model Qwen3-Coder-Next --maxc 8 --only teb-short,teb-hard`,
launched through the quiet gate on the warm boot (binary `29f904b`, before any fix), log
`~/6ixinfer-logs/fleet-suite-coder-6ixcpp-teb.log`; `DONE`, 755 s. Over the run: no other GPU
process above 5 % and no build or test of the other session in any of 50 samples; HEM paused.

| | 6ix.cpp, NVFP4 (checkpoint dense form) | vLLM, int4 AutoRound |
|---|---|---|
| tool-eval short (15) | **87** — 12 pass, 2 partial (TC-11, TC-14), 1 fail (TC-03) | **93** — 13 pass, 2 partial (TC-11, TC-14), 0 fail |
| tool-eval hard (88) | **78** — 60 pass, 17 partial, 11 fail; 137 of 176 points; safety gate not passed (3 warnings: TC-48, TC-60, TC-74) | **77** — 61 pass, 14 partial, 13 fail; 136 of 176 points; safety gate not passed (5 warnings: TC-18, TC-48, TC-60, TC-74, TC-84; rated "safety-capped") |
| report files | `~/tool-eval-bench/report-20261004-110817.json`, `…-110904.json` | `…-100758.json`, `…-100839.json` |
| time for the 88 hard scenarios / median time to first token | 705 s / 834 ms | 381 s / 370 ms |

**Why the hard set took almost twice as long on 6ix.cpp (705 s against 381 s)** — from the two
report files alone (`jobs/teb_time.py`; nothing was run for this):

| | 6ix.cpp | vLLM |
|---|---|---|
| turns over the 88 scenarios | 290 | 257 |
| completion tokens / prompt tokens | 24,225 / 663,536 | 21,459 / 558,007 |
| least-squares fit over the 88 scenarios, time = a·prompt + b·completion + c·turns | b = **25.8 ms per generated token (38.8 tok/s)**, a = 54 µs per prompt token, c = 0.17 s per turn; mean abs residual 0.26 s | b = **14.1 ms per generated token (70.8 tok/s)**, a = 13 µs, c = 0.27 s; residual 0.13 s |
| by that fit: generation / prompt / per-turn | 624 s / 36 s / 48 s | 303 s / 7 s / 68 s |
| the 66 scenarios with the same number of turns on both | 406 s, 13,995 completion tokens | 256 s, 14,291 completion tokens (ratio 1.59) |
| the 17 scenarios where 6ix.cpp took more turns (99 against 56) | 279 s | 86 s |
| time to first token of the first turn, median / p90 | 834 / 1309 ms | 370 / 399 ms |

Two things, in that order. **(1) Generation is slower per token:** about 26 ms against 14 ms.
vLLM's fitted 70.8 tok/s is its `bench_decode` figure (70.6) to the digit, so the fit can be
trusted to that extent; 6ix.cpp's 38.8 tok/s is the same kind of estimate (40.0 from the short
set) — **an inference from tool-eval's timings, not a decode benchmark**, for this model with no
draft layer and the BF16 dense form. On scenarios that ran the same number of turns and
produced the same number of tokens, 6ix.cpp took 1.59× as long. **(2) It ran more turns:** 33
more, 2,766 more generated tokens, almost all in 17 scenarios — the four structured-output
scenarios below alone are 111 s against 12 s (TC-66: 61.6 s and 2,260 tokens in 8 turns against
2.4 s and 109 tokens in 2). Prompt reading is a small part of either total by the fit, though
the first token of a fresh 4,000-token prompt takes 0.83 s against 0.37 s.

**Did any scenario fail because the engine refused a request?** No. `$OUT/jobs/teb_refusals.py`
searches every scenario's raw log, summary and note for an HTTP error, a 4xx/5xx, "rejected", a
timeout or a traceback: none in either 6ix.cpp report (and none in vLLM's). The harness
recorded tool calls on every turn where the model made one. TC-03 (short and hard) is the model
sending the e-mail to a guessed address without calling `get_contacts` first
(`failure_kind` `wrong_args`), where vLLM's int4 weights look the contact up.

The two engines give the same verdict on 69 of the 88 hard scenarios. Of the 19 that differ, 10
are better on 6ix.cpp (TC-18, 38, 41, 43, 53, 54, 61, 73, 80, 84) and 9 are better on vLLM
(TC-03, 21, 22, 33, 65, 66, 67, 69, 76) — mostly the scatter two quantizations of one model give.
**One group is not scatter and the owner should look at it: the four "tool, then structured
output" scenarios TC-65, 66, 67, 69.** On vLLM each passes in 2 turns: the tool call, then the
JSON answer as content. On 6ix.cpp, after the tool result the model does not write the JSON as
content; it keeps making tool calls that try to carry it — `calculator("round(28)")`,
`run_code` with `print(json.dumps({...}))` — until the 8-turn budget (7 or 8 turns each, all four
`partial`, three of them `budget_exceeded`). Four scenarios of one category behaving the same
way point at the engine rather than the weights — my guess, not verified, is how a
`response_format` / JSON constraint combines with `tool_choice` `auto` for this model's
`<function=…>` form (prose or a fenced block is masked, and the model prefers a tool call to a
bare `{`). I have not reproduced it by hand: I was told to send nothing to the box while the
engine is rebuilt. It costs 4 of the 176 points.

### C.3 ShareGPT on 6ix.cpp — first attempt refused by the engine (a bug); run on the fixed build, 11:27:36–11:42:50

**The bug.** The first suite run (10:58:12, `--only sharegpt,teb-short,teb-hard`, log kept as
`~/6ixinfer-logs/fleet-suite-coder-6ixcpp-run1-stopped.log`) started ShareGPT with the script's
external harness (`dgxone:5000/nvidia/vllm:25.12-py3`, `vllm bench serve`; there is no harness
inside this engine — on the vLLM slot it ran from the slot's own container). The harness sat at
`Waiting for endpoint to become up in 600 seconds` with the GPU idle. Cause, found by the
orchestrator and reproduced by me at 11:07:37:

```
$ curl …/v1/chat/completions -d '{"model":"Qwen3-Coder-Next","temperature":0,"max_completion_tokens":8,
      "messages":[{"role":"user","content":[{"type":"text","text":"Say hi"}]}]}'
{"error":{"message":"the chat template rejected these messages: chat-template: line 98: '+' needs two numbers (or, for '+', two strings)","type":"invalid_request_error","param":"messages","code":null}}      HTTP 400
$ the same with "content":"Say hi"
… "content":"Hi there! 😊 How can I" … "finish_reason":"length" … "completion_tokens":8      HTTP 200
```

Coder-Next's chat template concatenates `message.content` as a string, and the service handed it
the OpenAI content-part array as it was (`src/serve/generation_service.cpp`, the render after
`parse_chat`). vLLM flattens text parts for such templates, which is why its run worked. A real
compatibility bug of the port, not a benchmark quirk: any client that sends content parts was
refused. (`max_completion_tokens` is honoured: 8 tokens, `length`.)

**The fix is the orchestrator's, not mine** (branch `fix/content-parts`, commits `4a4ef27` +
`66aa940`, applied and built by it from 11:21: a new `src/serve/chat_content.hpp`, a retry at the render site, four tests in
`tests/host/serve_test.cpp`). By its account: built in `~/6ixinfer-ports/build-release`
(`build-content-parts.log`), `serve_test` 102 tests 0 failed (`serve-test-content-parts.log`),
engine down 11:25:34 / up 11:25:36 with the same site config, READY in 12 s (serve log: boot
11.2 s), and the request above now returns 200 with 8 completion tokens and a usage chunk. I did
not run those checks myself; what I saw is the staged files in the checkout, the new engine
process (pid 292731, 67,889 MiB) and the harness getting through. I had started the same fix on
the Mac and stood down at 11:12 before anything reached the box
(`draft-content-parts-NOT-APPLIED.patch`, for reference only).

**The run** (launched by the orchestrator, 11:27:36, on the patched build; its first relaunch at
11:26:10 died in 8 s because `TOK` was not set and the alias is not a tokenizer repository):
`TOK=RedHatAI/Qwen3-Coder-Next-NVFP4 python3 sixlabs/bench/run_fleet_suite.py --host 172.17.0.1 --port 18190 --model Qwen3-Coder-Next --maxc 8 --only sharegpt`,
log `~/6ixinfer-logs/fleet-suite-coder-6ixcpp-sharegpt.log`, `DONE` after 914 s; result file
`~/bench-results/sharegpt/sharegpt-bench-20261004-112736.txt` (+ `.raw`). HEM was paused
throughout (10:32:35–11:43:03). My 15-second GPU log has no other GPU process above 5 % in any of
61 samples; another session's build in `~/6ixinfer-check` shows in the 11:27:36 sample — the
moment of launch, while the harness was still starting — and in none after.

| ShareGPT, 64 prompts per level | 1 stream | 2 | 4 | 8 |
|---|---|---|---|---|
| **6ix.cpp** output tok/s | **40.4** | **65.6** | **97.6** | **124.1** |
| vLLM output tok/s | 67.1 | 107.3 | 148.7 | 192.9 |
| 6ix.cpp / vLLM | 0.60 | 0.61 | 0.66 | 0.64 |
| 6ix.cpp time to first token, mean (P99), ms | 162 (339) | 175 (333) | 188 (384) | 247 (535) |
| vLLM time to first token, mean (P99), ms | 162 (353) | 163 (309) | 188 (452) | 235 (438) |
| 6ix.cpp mean time per output token, ms (P99) | 24.27 (25.37) | 28.93 (31.29) | 38.47 (43.19) | 58.60 (71.85) |
| vLLM mean time per output token, ms (P99) | 14.13 (15.11) | 17.57 (23.07) | 25.13 (31.50) | 37.78 (44.24) |
| 6ix.cpp inter-token latency P99, ms | 39.7 | 51.0 | 106.2 | 226.4 |
| vLLM inter-token latency P99, ms | 26.8 | 44.7 | 76.5 | 118.7 |
| successful requests | 64 of 64 | 64 of 64 | 64 of 64 | 64 of 64 |

At one stream 6ix.cpp generates a token every 24.3 ms (41 tok/s) where vLLM takes 14.1 ms
(71 tok/s); the tool-eval timings gave the same figure independently (25.8 ms, C.2). Time to
first token is the same on both. This is the checkpoint dense form of a model with no draft
layer; the brief expected a lower single-stream speed than the 80B for that reason. What I
cannot say from one run each: how much of the gap is the missing draft layer, how much the BF16
dense form (the faster `fp8` form is the one refused), and how much vLLM's int4 kernels.

Two aborted files from the earlier attempts are in the fleet's result directory and are **not
results**: `~/bench-results/sharegpt/sharegpt-bench-20261004-105812.txt` (the hung run) and
`…-112610.txt` (the orchestrator's 8-second attempt), each with its `.raw`.

## D. Stopping the engine — not done by me, on instruction

The brief's step D (stop the engine, confirm the memory is back) was replaced first by "leave the
engine up" and then, at 11:45, by "do not launch, stop or restart anything more": a separate
agent takes the Coder engine down with my site config and uses the GPU for the owner's next
benches. My own engine process (pid 3996662, the warm boot of 10:57:44) was stopped by the
orchestrator at 11:25:34 to deploy its fix; the process serving at my last look (11:46:29) was
pid 292731, started by the orchestrator at 11:25:36, 67,889 MiB. My own stop at 10:57:12
(between the cold and the warm boot) did return the memory: `serve: stopped cleanly`, no
`6ix-Serve` in `nvidia-smi --query-compute-apps`, 96 GiB available.

## E. vLLM baseline — `qwen3-coder@dgxone` (vLLM, `Intel/Qwen3-Coder-Next-int4-AutoRound`) — done

Started only through `~/6ixlabs/cluster/bin/delta-launch`, compose file untouched, no `DELTA_*`
override. Pre-checks at 09:49:49: nothing of mine running, 99 GiB available, no build in
`~/6ixinfer-check`, the GLM download process gone (stopped at 09:41 by the orchestrator),
`nvidia-smi pmon -c 3 -s u` showed every GPU process at 0 % or idle in all three samples.

```
$ ~/6ixlabs/cluster/bin/delta-launch qwen3-coder@dgxone up          (launched 09:50:11)
[admit] qwen3-coder@dgxone needs 53.7 GiB; 99.0 GiB available, 0 GiB still to be claimed by loading slots; must leave 19.7 GiB
[admit] fits, nothing to evict
[delta-launch] waiting up to 900s for :8231 to serve (expected ~332s)...
  serving after 363s.
[delta-launch] done.                                                 (rc 0 at 09:56:51, wall 400 s with the reconcile)
```

- **Load time: 363 s** (`~/logs/delta-launch-timings.log`: `served_in=363s min_avail_during_load=25GiB avail_now=43.1GiB`).
- **GPU memory: 54,362 MiB = 53.09 GiB** (pid 2782544 `VLLM::EngineCore`), 42 GiB left available.
- Serving `qwen3-coder`, root `Intel/Qwen3-Coder-Next-int4-AutoRound`, `max_model_len` 262144;
  401 without the key, 200 with it. Recipe: 8 seats, FP8 KV cache of 12 GiB, prefix caching,
  chunked prefill, `--tool-call-parser qwen3_coder`.
### E.1 `bench_decode.py` (09:58:02, first run after the load)

Quiet check at 09:57:55: no build in `~/6ixinfer-check`, every other GPU process at 0 % or idle
in three `pmon` samples (another session's `unit_tests` had just exited). During the run the
15-second GPU log (`~/6ixinfer-logs/gpu-watch-coder.log`) shows only the slot itself busy.

```
$ with_fleet_key.sh python3 sixlabs/bench/bench_decode.py --base http://172.17.0.1:8231 --model qwen3-coder --streams 1,2,4,8 --api-key-env FLEET_KEY
single stream:   70.6 tok/s  (prose)
single stream:   70.7 tok/s  (code)
2 streams:       96.8 tok/s aggregate
4 streams:      151.7 tok/s aggregate
8 streams:      259.2 tok/s aggregate
metrics: HTTP Error 404: Not Found          (vLLM has no /v1/metrics; expected)
```

### E.2 Fleet suite (09:59:29–10:15:04, 935 s; `DONE with errors` — the one "error" is the skipped ShareGPT step)

`with_fleet_key.sh python3 sixlabs/bench/run_fleet_suite.py --host 172.17.0.1 --port 8231 --model qwen3-coder --maxc 8 --api-key-env FLEET_KEY`,
log `~/6ixinfer-logs/fleet-suite-coder-vllm.log`. Quiet check at 09:59:25: nothing else above 0 %.
Over the whole suite the 15-second GPU log (`~/6ixinfer-logs/gpu-watch-coder.log`) has no other
GPU process above 5 % in any of 62 samples (HEM, the reranker/embedder and the 0.8b slot all idle).

| step | result | where the result file is |
|---|---|---|
| 1 DecodeBench | aggregate tok/s at 1 / 2 / 4 / 8 streams: **68.8 / 116.1 / 165.2 / 221.2**; per-stream median 70.6 / 59.5 / 42.3 / 28.1; median TTFT 0.18 / 0.19 / 0.29 / 0.29 s; 0 failed | run_id `20261004-095927-237cde` in `~/.local/state/saih-decodebench/history.json` |
| 2 llama-benchy (prompt 2048, gen 128, 3 runs) | depth 0 — generation tok/s at c1 / c2 / c4 / c8: **71.3 / 114.3 / 166.3 / 217.6**, prompt processing 3160 / 3708 / 4073 / 4622 tok/s, time to first response 0.62 / 0.94 / 1.71 / 3.00 s; depth 4096 — 68.9 / 112.9 / 123.5 / 104.1, pp 4458 / 4426 / 4628 / 4707; depth 16384 — 65.5 / 93.2 / 46.4 / 36.0, pp 4246 / 4297 / 4358 / 4348 | `~/llama-benchy-20261004-095927.json` |
| 3 ShareGPT (64 prompts per level) — run separately, 10:31:06–10:41:17, see E.3 | output tok/s at c1 / c2 / c4 / c8: **67.1 / 107.3 / 148.7 / 192.9**; TTFT mean 162 / 163 / 188 / 235 ms, P99 353 / 309 / 452 / 438 ms; mean per-token time 14.13 / 17.57 / 25.13 / 37.78 ms; ITL P99 26.8 / 44.7 / 76.5 / 118.7 ms; 64 of 64 successful at every level | `~/bench-results/sharegpt/sharegpt-bench-20261004-103106.txt` (+ `.raw`) |
| 4 tool-eval short (15 scenarios) | **final score 93** — 13 pass, 2 partial (TC-11, TC-14), 0 fail | `~/tool-eval-bench/report-20261004-100758.json` |
| 5 tool-eval hard (88 scenarios) | **final score 77** — 61 pass, 14 partial, 13 fail (TC-18, 38, 41, 43, 48, 60, 68, 71, 74, 80, 84, 87, 88) | `~/tool-eval-bench/report-20261004-100839.json` |

The key: 0 occurrences in the suite log, the llama-benchy result file, the `bench_decode` output
and the launcher log (counted with the key read into the environment of the counting command).

During the load itself (09:50:11–09:56:51) another vLLM engine core on the box (pid 1375679, the
embedder/reranker side) was above 5 % of the GPU in 3 of 26 samples (peak 58 %); HEM was idle.
The 363 s is therefore "one load with a short burst of someone else's GPU use in it".

### E.3 ShareGPT against the keyed slot (the step the first suite run skipped)

The orchestrator replaced `~/run-sharegpt-bench.sh` (sha256 `41062248f47b24f7…`, reads
`OPENAI_API_KEY`) and `sixlabs/bench/run_fleet_suite.py` in `~/6ixinfer-ports` (the version of
main `5c9a62e`, staged with `git checkout origin/main -- …`; the rest of the tree stays at
`29f904b`) at about 10:18. I read both diffs before running them with the key.

`with_fleet_key.sh python3 sixlabs/bench/run_fleet_suite.py --host 172.17.0.1 --port 8231 --model qwen3-coder --maxc 8 --only sharegpt --api-key-env FLEET_KEY`,
log `~/6ixinfer-logs/fleet-suite-coder-vllm-sharegpt.log`.

- **A first start at 10:24:04 was my mistake and was stopped after about 40 s.** I launched it in
  the same command as the quiet check, and the check had just printed a build running in
  `~/6ixinfer-check` (started 10:23:26). I stopped my own run: the suite's process group by its
  recorded PID (3399172), then the `vllm bench serve` worker it had started inside the slot's
  container (pid 3002 there, 37 s old). The slot stayed healthy. Left behind in the fleet's
  result directory: `~/bench-results/sharegpt/sharegpt-bench-20261004-102404.txt` (262 bytes)
  and `.raw` — **an aborted run, not a result**; I did not delete or rename them.
- The real run went through a gate (`$OUT/jobs/when_quiet.py`): it waited 264 s — for that
  session's build and test binary, then for HEM (93–94 % of the GPU from about 10:28 to 10:31) —
  and launched at **10:31:06** on a quiet reading.
- The orchestrator reports HEM at about 89 % for roughly the first 90 s of the c=1 level (HEM was
  then paused at 10:32:35 for the rest of the benches). My 15-second GPU log does not show it:
  HEM's last busy sample is 10:30:43 (71 %), and it reads 0 % or idle in every sample from
  10:31:13 on. Either way the c=1 level shows no slow stretch: mean per-token time 14.13 ms,
  median 14.25, **P99 15.11 ms** — 70.8 tok/s, the same as `bench_decode`'s 70.6. (Its
  "output tok/s" of 67.1 is lower because it counts each request's time to first token.) I did
  not rerun the level.
- Over the run: another vLLM engine core on the box (pid 1375679) above 5 % in 2 of 42 samples
  (peak 16 %); no build or test of the other session in any sample.

### E.4 The same first requests against vLLM, then stopped

`first_requests.py http://172.17.0.1:8231 qwen3-coder` with the key (`$OUT/first-requests-vllm.txt`):
all 29 checks held — plain chat, the two code-writing requests, forced and auto tool calls, the
tool result fed back, the refusals. Kept for comparison with 6ix.cpp's answers to the same requests.

```
$ ~/6ixlabs/cluster/bin/delta-launch qwen3-coder@dgxone down        (10:42:17)
 Container qwen3-coder Stopped … Removed
[delta-launch] qwen3-coder down; marked stopped for delta-guard.     (rc 0, 10:42:29)
```

Container gone (`docker ps -a` has no `qwen3-coder`), port 8231 closed, compute apps back to the
five that were there before, 95 GiB available.

## F. Clean up, and the state of the box when I stopped (last look 11:46:29)

- **`benches-done` and HEM — an instruction crossed with my action.** I ran
  `touch ~/6ixinfer-logs/benches-done` at 11:43:02, after the last measurement, as instructed
  then; `hold-hem.log` showed `unpaused (sentinel present); paused now: false` / `DONE` at
  11:43:03. A message withdrawing that step (HEM must stay paused for benches another agent runs
  next) reached me afterwards. By 11:46:29 the orchestrator had put it right itself: the
  sentinel file no longer exists, a new hold job is running (`hold_hem_until.py 15:30 …`,
  pid 636602, about 11:45:55) and `HEMnliOne` is `Up … (Paused)` again. **HEM ran unpaused for
  about three minutes (11:43:03 to about 11:45:55).** I did not pause, unpause or otherwise touch
  HEM or the sentinel after 11:43:02.
- **No process of mine is running on DGXone.** The last one, the read-only GPU logger
  (`gpu_watch.py`, pid 3445646), was stopped at 11:43:29, before the "stop nothing more"
  message. vLLM slot `qwen3-coder`: down since 10:42:29, container removed, 8231 closed, marked
  stopped for delta-guard as before.
- **The 6ix.cpp engine for Coder-Next was still up at my last look** — pid 292731, started by
  the orchestrator 11:25:36, `172.17.0.1:18190`, **67,889 MiB**; 24 GiB of host memory available
  beside it. Another agent takes it down
  (`cd ~/6ixinfer-ports && python3 scripts/dgpp-cluster down --config deploy/cluster_qwen3-coder-next_nvfp4_w1.json`).
- `~/6ixinfer-ports` is `29f904b` plus, staged by the orchestrator and not by me,
  `sixlabs/bench/run_fleet_suite.py` (main `5c9a62e`) and the content-parts fix
  (`src/serve/chat_content.hpp`, `src/serve/generation_service.cpp`, `tests/host/serve_test.cpp`);
  plus my eight untracked gate inputs under `sixlabs/ports/qwen3-coder-next/inputs/`, my site
  config `deploy/cluster_qwen3-coder-next_nvfp4_w1.json` (git-ignored) and `log/`. The tree is
  behind `origin/main`; it was not pulled. `build-release/6ix-Serve` is the patched build.
- Left on disk, mine: `~/6ixinfer-out/resident-cache/aca98768b71ca245.img` (43.8 GiB, this
  model's resident image in the checkpoint dense form — **I did not remove it**: the engine was
  up, and then I was told to touch nothing; delete it once Coder-Next is no longer wanted on
  6ix.cpp), `~/6ixinfer-out/coder` (690 MB: dumps, references, job scripts), logs under
  `~/6ixinfer-logs/` (`*coder*`). Not mine, still there: the 35B's 20.4 GiB image
  `ed9be330d5e51f08.img`. Disk: 198 GiB free at 11:43.
- Not results, in the fleet's result directories: three aborted ShareGPT files
  (`~/bench-results/sharegpt/sharegpt-bench-20261004-{102404,105812,112610}.txt` + `.raw`) and
  no tool-eval report from the 45-second run I stopped at 11:07:53.

## G. The table

One DGX Spark (DGXone), 2026-10-04. **The two columns are different quantizations of the same
model** — 6ix.cpp serves `RedHatAI/Qwen3-Coder-Next-NVFP4` (4-bit floating point, in the
checkpoint's own dense form), vLLM serves `Intel/Qwen3-Coder-Next-int4-AutoRound` (4-bit integer)
— **and every figure is a single run.** Settings: 6ix.cpp — 8 request slots, 524,288-token BF16
K/V pool (12 GiB), `dense_weights` `checkpoint`, no draft layer (this checkpoint has none, so no
speculative decoding), 4 GiB prefix cache, `prefill_budget_tokens` 1024,
`prefill_idle_budget_tokens` 4096, `decode_passes_per_prefill` 8, `prefill_order` `shortest`.
vLLM — the fleet's recipe as written: 8 seats, 262,144 context, FP8 K/V cache of 12 GiB, prefix
caching, chunked prefill.

| | 6ix.cpp, NVFP4 (`RedHatAI/…-NVFP4`) | vLLM, int4 (`Intel/…-int4-AutoRound`) |
|---|---|---|
| numerical gate | checkpoint dense form: prose PASS, original code prompt FAIL (rows 8 and 22, greedy condition) — **served under the owner's exception** (three more code prompts PASS); `fp8` dense form refused | not applicable |
| load time, to serving | **86 s** cold by the launcher (serve log: boot 84.9 s; no resident image); **12 s** warm from the resident image (serve log: 10.9 s) | **363 s** |
| planned memory | 64.51 GiB (59.30 device + 5.21 pinned) | 53.7 GiB (the launcher's footprint figure) |
| measured GPU memory | **67,889 MiB = 66.30 GiB** | **54,362 MiB = 53.09 GiB** |
| ShareGPT, output tok/s at 1 / 2 / 4 / 8 streams | **40.4 / 65.6 / 97.6 / 124.1** | **67.1 / 107.3 / 148.7 / 192.9** |
| ShareGPT, time to first token, mean (P99), ms | 162 (339) / 175 (333) / 188 (384) / 247 (535) | 162 (353) / 163 (309) / 188 (452) / 235 (438) |
| ShareGPT, mean time per output token, ms | 24.27 / 28.93 / 38.47 / 58.60 | 14.13 / 17.57 / 25.13 / 37.78 |
| tool-eval short (15) | **87** (12 pass, 2 partial, 1 fail) | **93** (13 pass, 2 partial) |
| tool-eval hard (88) | **78** (60 pass, 17 partial, 11 fail) | **77** (61 pass, 14 partial, 13 fail) |
| tok/s single stream (prose, code) — `bench_decode` | **not run** (owner's instruction). ShareGPT's per-token time gives 41 tok/s at one stream; tool-eval's timings give 39. | 70.6, 70.7 |
| tok/s at 2 / 4 / 8 streams — `bench_decode`, aggregate | **not run** (ShareGPT above is the only multi-stream figure) | 96.8 / 151.7 / 259.2 |
| DecodeBench, aggregate tok/s at 1 / 2 / 4 / 8 | not run | 68.8 / 116.1 / 165.2 / 221.2 |
| llama-benchy, generation tok/s at c1 / c2 / c4 / c8, depth 0 | not run | 71.3 / 114.3 / 166.3 / 217.6 |
| llama-benchy, prompt processing tok/s, same | not run | 3160 / 3708 / 4073 / 4622 |
| llama-benchy, generation tok/s at depth 16384 | not run | 65.5 / 93.2 / 46.4 / 36.0 |

Which build each 6ix.cpp figure came from: load times, memory, first requests and tool-eval —
`29f904b` unpatched; ShareGPT — `29f904b` plus the orchestrator's content-parts fix (the fix
only changes requests that were refused before).

How to read it, and what it does not show:

- **Speed: 6ix.cpp is slower than vLLM on this model at every ShareGPT level** — 0.60, 0.61,
  0.66 and 0.64 of vLLM's output throughput at 1, 2, 4 and 8 streams (40 against 67 tok/s at
  one stream, 124 against 193 at eight; 24.3 against 14.1 ms per token at one stream), with the
  same time to first token. It is a model with no draft head, served in its BF16 dense form
  because the faster `fp8` form failed the numerical gate; the brief said to expect a lower single-stream speed than the 80B and not
  to treat it as a fault. I have no measurement that splits the gap between those causes, and no
  `bench_decode`, DecodeBench or llama-benchy figure for 6ix.cpp at all — so nothing here about
  its prompt processing under load or its behaviour at long context.
- **Load: 84.9 s against 363 s cold, 10.9 s from the resident image. Memory: 66.30 against
  53.09 GiB** — 6ix.cpp holds 13 GiB more. Its weights are 44.92 GiB in the checkpoint dense
  form (43.35 in the refused `fp8` form) plus a 12 GiB K/V pool; vLLM's int4 weights are smaller.
  With Coder-Next up on 6ix.cpp beside the box's standing services, 24–27 GiB stay available.
- **Tool-eval: 87 / 78 against 93 / 77** — the same model in two quantizations, the same verdict
  on 69 of 88 hard scenarios, no request refused by either engine. Four of the scenarios 6ix.cpp
  loses (TC-65, 66, 67, 69: a tool call followed by a structured JSON answer) look like an
  engine matter rather than the weights; see C.2. The hard set took 705 s against 381 s.
- **Correctness.** The gate's verdict on the original code prompt is still FAIL; the model is
  served under the owner's exception; the control prompt's row 28 is open (top of this file,
  4.6). What supports the port: 296,151 of 296,151 tensors bound; every layer and the head
  within about 1.7 BF16 floors of the reference on six prompts, the same as the 80B, against
  130–600 for a wrong convention; 29 of 29 first-request checks; two code answers byte-identical
  to vLLM's; tool calls in the model's own `<function=…>` form parsed on both the forced and the
  free path.

## H. What did not run, and why

| not run | why |
|---|---|
| 6ix.cpp in the `fp8` dense form (boot, any measurement) | fails the gate on code prompts (mean above 0.10 on two of three, 0.109 on average over five; one unexplained clear-margin miss); owner's decision |
| `bench_decode.py`, DecodeBench, llama-benchy on 6ix.cpp | owner's instruction, relayed at about 10:46: ShareGPT and tool-eval only on this side |
| step D (stop the engine, confirm the memory is back) at the end | orchestrator's instruction: leave the engine up, then touch nothing more — another agent takes it down |
| further reference variants for the control's row 28 | 8.7 GiB each; not possible beside the engine within the 20 GiB margin (4.6) |
| a hand reproduction of the structured-output behaviour (TC-65/66/67/69) | told to send nothing to the box while the engine was rebuilt, and afterwards to launch nothing and report |
| the content-parts fix: my own build and test of it | the orchestrator took it over at 11:12; its build and `serve_test` results are its own report (C.3) |
| `"mtp": true` must be refused for this checkpoint (notes, section 6) | not tried |
| the `ci` preset (warnings as errors); `qwen3next_tool_calls_test`, `qwen3next_loader_test` | only the `release` preset and the six listed targets were built |
| the refset replay against both engines (notes, end of section 6) | not asked for in the brief's steps; the first-requests comparison (5) is what there is |
| the W4A4 question | the checkpoint's recipe quantizes activations to 4 bits as well (`input_activations`, dynamic, group 16); the engine and the numpy reference both run it with unquantized activations, as the notes' section 7 already lists as open. The gate compares the engine with that reference, so it says nothing about agreement with a W4A4 execution. |

## I. For the owner: things on the box and in the engine worth knowing

1. **Content-part arrays were refused with 400 by this model** (string-content chat template).
   Fixed by the orchestrator on `fix/content-parts`; the running engine has the fix; main does
   not yet (as far as I know).
2. **Structured output after a tool call** (tool-eval TC-65/66/67/69): on 6ix.cpp the model does
   not emit the JSON answer as content and loops on tool calls. Not reproduced by hand, cause
   not established (C.2).
3. **The default template (`dense_weights` `fp8`) should not be used for this model as it
   stands**: that form fails the gate on code prompts. The site config used here differs from
   `deploy/cluster_qwen3-coder-next_nvfp4_w1.example.json` in `dense_weights` (`checkpoint`) and
   in the four prefill settings.
4. **`run_fleet_suite.py` needs `TOK=RedHatAI/Qwen3-Coder-Next-NVFP4` for ShareGPT** against the
   alias `Qwen3-Coder-Next`; without it the harness dies in 8 s on the tokenizer lookup.
5. **The gate's greedy condition and this router.** On the reference's own trajectory 21 % of
   row-layers have an exact BF16 tie at the 10th expert, and the float32 and BF16 conventions
   pick different experts in 9–12 % of row-layers. The reference flips its own greedy token at
   margins up to 1.17 nat when only that convention changes. A correct engine will keep tripping
   condition 3 on some single prompts.
6. **Memory with the engine up:** 24–27 GiB available on DGXone (earlyoom line 9.7).
   And: HEM ran unpaused from 11:43:03 to about 11:45:55 because of my `touch` (F).
7. Housekeeping: the three aborted ShareGPT files and the two resident images listed in F; the
   GLM download was stopped at 09:41 by the orchestrator at 40.6 of 195 GB (not by me).
8. `gpu-steps.md` for this port still describes the old 0.06 / 0.5 bounds, `build-ci` paths,
   "3 tests" for the loader fixture (it is 9 now) and `fp8` as the expected serving form. I did
   not edit it.
