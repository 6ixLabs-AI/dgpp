# Qwen3.6-35B-A3B on 6ix.cpp — bring-up results (DGXone)

> Layout note, 2026-10-04: moved from `sixlabs/ports/qwen3.6-35b-a3b/results-2026-10-04-dgxone.md` when the per-model folders under `models/` were made. Paths to other moved records were updated; nothing else was changed. `gate-final-8-runs.txt` moved with it; `gpu-steps.md` and `inputs/` stayed in `sixlabs/ports/qwen3.6-35b-a3b/`. The `jobs/` folder, `fixes.patch`, `build-status.md` and `unloaded.txt` it names are not in this repository.

**Finished 2026-10-04 05:20 EDT.** Nothing below is claimed beyond what the quoted output shows.

Status in one line: **partly.** Checkpoint A (NVFP4) builds, boots, answers, and is measured, and the vLLM
baseline is measured; checkpoint B (FP8) was not booted or measured because it fails the final
numerical gate on one text; the numerical gate itself was changed twice during the night, on
instruction (section 4). The table is section G.

Box: DGXone, aarch64, GCC 13.3.0, CUDA 13.0.88. Checkout `~/6ixinfer-ports` at `89955e0`; no
engine source was changed (`fixes.patch` holds three Python bench scripts only, see
`build-status.md`). Scratch `~/6ixinfer-out/q36`, logs `~/6ixinfer-logs/`.
`$REPO` = `~/6ixinfer-ports`, `$OUT` = `~/6ixinfer-out/q36`.

Note on the notes: `gpu-steps.md` runs the host tests from `build-ci`; only the `release` preset
was built here (the job that was already running), so every binary below is `build-release/…`.
The unit tests use their own check macros, not `assert`, so `-DNDEBUG` does not hollow them out.

## A. Build

| step | command | result |
|---|---|---|
| release build | `python3 sixlabs/bench/build_dgpp.py /home/mark/6ixinfer-ports -j 8 --preset release --target dgpp_serve_app --target qwen35_bind_check --target qwen35_forward_check --target unit_tests --target qwen35_ports_loader_test` | **PASS** — `[2/2] 1/1 err=0 elapsed=305s` / `DONE`, no compiler errors or warnings, no source change needed |

## B. gpu-steps sections

### 1. Host tests and the loader fixture — PASS

```
DGPP_TEST_FILTER=qwen36 ./build-release/unit_tests   ->  9 tests, 0 failed   (expected 9)
DGPP_TEST_FILTER=qwen35 ./build-release/unit_tests   ->  9 tests, 0 failed
DGPP_TEST_FILTER=qwen3  ./build-release/unit_tests   -> 44 tests, 0 failed
cd $OUT && $REPO/build-release/qwen35_ports_loader_test -> 5 tests, 0 failed (expected 5)
   [ OK ] qwen3codernext_loader_resident_values_are_the_checkpoints
   [ OK ] qwen3codernext_loader_rank_slices_tile_world_one
   [ OK ] qwen3codernext_loader_fp8_form_is_the_encode_of_the_bf16_form
   [ OK ] qwen36_nvfp4_mixed_loader_resident_values_are_the_checkpoints
   [ OK ] qwen36_fp8_loader_resident_values_are_the_checkpoints
```

### 2. Bind checks (headers only) — PASS, both checkpoints, exact expected counts

```
$ ./build-release/qwen35_bind_check --model nvidia/Qwen3.6-35B-A3B-NVFP4
config: qwen3_5, 40 layers (30 GDN + 10 full attention) + 1 draft, hidden 2048, vocab 248320
mlp: routed MoE, 256 experts top-8, intermediate 512, shared expert 512
checkpoint: 3 shards, 124468 tensors in headers
binding: expected 124135 | matched 124135 (missing 0, dtype 0, shape 0) | unexpected 0 | out of scope 0 | vision skipped 333
quantized matrices: 30971
binding OK: every tensor of the table is present with its dtype and shape

$ ./build-release/qwen35_bind_check --model Qwen/Qwen3.6-35B-A3B-FP8
checkpoint: 42 shards, 64196 tensors in headers
binding: expected 63863 | matched 63863 (missing 0, dtype 0, shape 0) | unexpected 0 | out of scope 0 | vision skipped 333
quantized matrices: 31745
binding OK: every tensor of the table is present with its dtype and shape
```

### 3. Memory plan (loads nothing) — PASS, matches the notes

`./build-release/qwen35_forward_check --model … --plan 8,262144 [--dense-weights fp8]`

| checkpoint | dense form | model weights (resident) | plan TOTAL device | + pinned | expected weights |
|---|---|---|---|---|---|
| A NVFP4 | `fp8` | 20.930 GiB | 29.087 GiB | 1.917 GiB | ≈ 20.93 |
| A NVFP4 | checkpoint | 21.075 GiB | 28.758 GiB | 1.917 GiB | 21.08 |
| B FP8 | `fp8` | 34.055 GiB | 42.211 GiB | 1.917 GiB | ≈ 34.06 |
| B FP8 | checkpoint | 34.055 GiB | 41.738 GiB | 1.917 GiB | ≈ 34.06 |

All four say `mtp on`; KV pool 5.500 GiB, `mtp scratch` 0.125 GiB.

### 4. Numerical gate: forward check against the numpy reference — **checkpoint A: FAILED as originally written (3 of 4 runs), PASSES the gate as restated twice on 2026-10-04 (4 of 4 under either restatement); checkpoint B: FAILS the final gate on the code text (4.11); both verified layer by layer**

Short version, stated plainly:

- **The gate as the notes define it did not pass.** With the original `compare_forward_check.py`
  and its bounds (mean ≤ 0.06 nat, max ≤ 0.5), checkpoint A passes on one of four runs (4.2).
- **The bound was then changed — on instruction, not on my own initiative.** After seeing the
  80B calibration (4.5) the session that owns the 80B port restated the gate (relayed by the
  orchestrator): correlation ≥ 0.995, mean ≤ 0.10, max reported but not bounded, greedy
  agreement wherever the reference's margin exceeds 0.5 nat. I implemented exactly that in
  `sixlabs/bench/compare_forward_check.py` (in `fixes.patch`; the old verdict is still printed).
  Under it checkpoint A passes all four runs (4.9). The 80B's own code/checkpoint-form
  calibration run failed that restated gate too (mean 0.105, correlation 0.9942, one greedy
  disagreement at a 0.73 nat margin). When I reported that, the same session added two
  exclusions (far-tail tokens; greedy misses on the first two rows), which I also implemented as
  instructed; with them all eight runs pass (4.9, last table). **So the gate was changed twice
  tonight, the second time in direct response to the one run of the known-good model that still
  failed. The 35B did not need either exclusion — it passes with and without them.**
- **Independently of either bound, neither the engine nor the reference computes the wrong
  function.** Checked one layer at a time on the engine's own inputs, all 40 layers and the head
  agree with the reference to about twice the BF16 storage floor; a deliberately wrong convention
  shows 70–300 times that (4.6). This, not the restated bound, is what I rely on.
- **The original bound is below this model's own BF16 noise.** The reference moves from *itself*
  by the same amount (mean 0.054–0.071, max 0.35–1.74) when only its arithmetic is held to BF16
  (4.4). The working 80B fails the original per-text bounds with the same tools on 4 of 4 runs.

#### 4.1 The reference on the real checkpoint

`tools/qwen3next_reference.py score` / `dump`, CPU, log `~/6ixinfer-logs/reference-A.log`. First
run on real weights, no errors. `mean_logprob` −1.966 (code, 113 tokens), −2.023 (prose, 87).
The template tokens and the canned answer are predicted at ≈ 0 (the code answer's tokens at
−0.0002 … −0.02); the mean is carried by the unpredictable user-prompt tokens. Nothing near −12.4.

#### 4.2 The gate as written (checkpoint A)

`qwen35_forward_check` (streaming) → **unmodified** `compare_forward_check.py`, default bounds.
Logs `~/6ixinfer-logs/forward-check-A.log`, outputs `$OUT/q36A-*.txt`.

| input | dense form | mean \|Δ\| | median | max (position) | pearson | greedy rows agree | verdict |
|---|---|---|---|---|---|---|---|
| code | checkpoint (exact) | 0.07846 | 0.00426 | 0.63291 (36) | 0.999296 | 110/113 | **FAIL** |
| prose | checkpoint (exact) | 0.05843 | 0.02468 | 0.37438 (62) | 0.999335 | 86/87 | PASS |
| code | `--dense-weights fp8` | 0.06098 | 0.00727 | 0.86169 (7) | 0.999157 | 110/113 | **FAIL** |
| prose | `--dense-weights fp8` (not in the notes; added) | 0.06906 | 0.02814 | 0.49421 (29) | 0.998726 | 87/87 | **FAIL** |

Every greedy disagreement sits at a reference margin under 0.3 nat (0.0075, 0.052, 0.096, 0.096,
0.30). `compare_states.py`: cosines ≥ 0.9977 at every layer, relative rms error 0.014 at layer 0
rising to 0.068 (code) / 0.066 (prose).

#### 4.3 Where they diverge: layer 0, the MoE router's expert ids

At layer 0 the engine's residual is within 0.40 % of the reference in the median row, but 12 of
113 rows are off by 2–8 %. Those are the rows whose 8th and 9th router logits nearly tie
(reference margins 0.001–0.017). Recomputing the reference's layer-0 output with the 8th expert
swapped for the 9th reproduces the engine's row: row 26 4.73 % → 0.37 %, row 67 3.84 % → 0.29 %,
row 70 4.35 % → 0.38 %, row 92 1.98 % → 0.32 %, row 95 3.62 % → 0.45 %, row 107 3.88 % → 0.32 %
(`~/6ixinfer-logs/diag-bf16-A-code.log`).

Why: the engine's routed MoE — `MoeRouterMode::SoftmaxTopk` in `src/models/glm/moe.hpp` /
`src/kernels/glm_moe.cu`, **shared code that the 80B and Flash-Next already run, not something
this port added** — rounds the router logits to BF16, takes an fp32 softmax over them, picks the
top-k on the logits with ties to the lower expert id, and rounds the picked weights to BF16 (the
kernel's comments describe this as following "the reference's bf16 Linear output"). The numpy
reference keeps float32 logits. At logits of 4–16 a BF16 step is 0.03–0.06, wider than those
margins. A token routed to a different eighth expert differs by a few percent in that layer and
the two runs never re-converge: 40 layers of this is the end-to-end gap.

#### 4.4 The reference against itself at BF16 precision

Same weights, same code, only the arithmetic changed (`$OUT/jobs/diag_router.py`, logs
`~/6ixinfer-logs/diag-router-A-{code,prose}.log`): the reference with BF16 router logits, and
with BF16 residual stream / sublayer inputs and outputs as well.

| compared (log-probabilities of the same ids) | code: mean / max | prose: mean / max |
|---|---|---|
| reference[BF16 router] vs reference as shipped | 0.07109 / 1.330 | 0.05573 / 0.347 |
| reference[BF16 router + activations] vs reference as shipped | 0.06730 / 1.741 | 0.05440 / 0.533 |
| reference[BF16 router + activations] vs reference[BF16 router] | 0.06105 / 1.169 | 0.05770 / 0.566 |
| **engine vs reference as shipped** (the gate) | 0.07846 / 0.633 | 0.05843 / 0.374 |
| engine (fp8 form) vs reference as shipped | 0.06098 / 0.862 | 0.06906 / 0.494 |

The engine is as far from the reference as the reference is from itself under a BF16-level
change. Putting the engine's router convention into the reference does not bring the two
together either (engine vs reference[BF16 router]: code 0.0687 / 1.72, prose 0.0583 / 0.375):
any two evaluations that differ in the last bit of an activation route some rows differently.
On these texts the bounds 0.06 / 0.5 sit inside that noise, so PASS or FAIL here is close to a
coin flip and says nothing about whether a tensor is wrong.

#### 4.5 The same gate on the 80B, the port that is known to work

Same binaries, same reference tool, same default bounds (`$OUT/jobs/calib_80b.py`, log
`~/6ixinfer-logs/calib-80b.log`). Inputs: the Coder-Next port's two id files (the 80B's own
tokenizer family; 107 and 83 tokens).

| 80B input | dense form | mean \|Δ\| | median | max (position) | pearson | verdict |
|---|---|---|---|---|---|---|
| code | checkpoint | 0.10475 | 0.00030 | 5.10672 (106) | 0.994186 | FAIL |
| code | `fp8` | 0.06356 | 0.00040 | 1.86588 (106) | 0.999040 | FAIL |
| prose | checkpoint | 0.05435 | 0.01510 | 1.18045 (15) | 0.999268 | FAIL (max) |
| prose | `fp8` | 0.08572 | 0.02546 | 0.75394 (23) | 0.999261 | FAIL |

The known-good model fails all four. Where the 0.06 comes from: `6IXSERVE.md` records "within
0.06 nat (mean) … on eight texts", and `sixlabs/bench/compare_logprobs.py` prints that as the
`overall` line — the average of eight texts' means, with the worst position reported but not
bounded. `compare_forward_check.py` (written for these ports, never run on a GPU before tonight)
applies that eight-text average as a ceiling on each single text and adds a 0.5 per-position
bound of its own. A ceiling set at a population's average is failed by about half of the
population, working or not. So the question is not "why does the 35B miss a bound the 80B meets":
on these inputs, with this tooling, the 80B does not meet it, and the 35B's numbers
(0.058–0.078 mean, 0.37–0.86 max) are no worse than the 80B's (0.054–0.105 mean, 0.75–5.1 max).

Caveat on this calibration: the ids were rendered with Coder-Next's chat template, not the 80B
Instruct's own. Both tools score any id sequence, so the comparison is valid, but these are not
the eight texts the 0.06 was measured on, and I did not re-run those.

#### 4.6 A test that can tell a wrong tensor from rounding: one step per layer

`$OUT/jobs/diag_transfer.py` (CPU only, over the dumps the forward check wrote). For each layer
L the reference's layer L is applied to the **engine's own** residual entering L and compared
with the engine's residual leaving L, row by row. Nothing accumulates, so routing chaos cannot
build up. `err` is the row's error as a fraction of the layer's update; `floor` is what storing
that row in BF16 costs in the same units; `u = err / floor`. Rows above 3× the layer's median `u`
are examined: the engine's expert selection is decoded by least squares over the 16 experts with
the highest router logits, and the reference's MoE is recomputed with that selection.

| run | median row err (of the layer's update): typical layer / worst layer | in BF16 floors: typical / worst layer | outlier rows | of which re-routed at a near tie | spread of the exchanged experts' logits (BF16 steps): median / max | unexplained |
|---|---|---|---|---|---|---|
| A code, exact | 1.01 % / 1.32 % | 2.04 / 2.38 | 249 of 4520 (5.5 %) | 246 | 0.48 / 0.99 | 3 |
| A prose, exact | 1.01 % / 1.23 % | 2.04 / 2.38 | 159 of 3480 (4.6 %) | 157 | 0.49 / 1.16 | 2 |
| A code, `fp8` form | 1.56 % / 1.86 % | 3.08 / 7.68 | 260 of 4520 (5.8 %) | 259 | 0.50 / 1.05 | 1 |
| A prose, `fp8` form | 1.56 % / 1.81 % | 3.11 / 7.14 | 163 of 3480 (4.7 %) | 161 | 0.51 / 1.08 | 2 |
| *80B code, exact (calibration)* | 1.39 % / 1.90 % | 1.60 / 2.94 | 360 of 5136 (7.0 %) | 337 | 0.50 / 1.20 | 23 |
| *80B prose, exact (calibration)* | 1.40 % / 1.75 % | 1.61 / 2.36 | 193 of 3984 (4.8 %) | 184 | 0.47 / 1.23 | 9 |

The two 80B rows are the known-good port under the same check (48 layers, top-10 of 512): the
35B is on a par with it (a smaller error as a fraction of the update, a slightly larger one in
floor units, fewer unexplained rows).

GDN and attention layers are alike (2.01 vs 2.07 floors in the exact form); no layer stands out.
Every routing difference is between experts whose router logits are within about one BF16 step.
The `fp8` form's extra ≈ 0.55 % per layer is the shared expert re-encoded to block FP8 at load —
the lossy serving form, by design; its "worst layer" in floor units is layer 39, where the floor
itself is smallest (err there is 1.69 %, like every other layer).

The head, one step (the reference's final norm + lm head on the engine's last residual, against
the engine's own log-probabilities):

| run | final-norm row err, max | token log-prob \|Δ\|: mean / max | greedy rows differing |
|---|---|---|---|
| A code, exact | 0.19 % | 0.00148 / 0.01070 | 0 of 113 |
| A prose, exact | 0.19 % | 0.00218 / 0.01591 | 0 of 87 |
| A code, `fp8` form | 0.19 % | 0.00159 / 0.01325 | 0 of 113 |
| A prose, `fp8` form | 0.20 % | 0.00190 / 0.01122 | 0 of 87 |
| *80B code, exact (calibration)* | 0.18 % | 0.00121 / 0.01379 | 0 of 107 |
| *80B prose, exact (calibration)* | 0.21 % | 0.00198 / 0.01200 | 1 of 83 (reference margin 0.0004) |

Negative control — the same check with the reference deliberately given a wrong convention
(first 4 layers of A code), to show what a wrong tensor looks like on this scale:

| wrong convention | median row err, typical layer | in BF16 floors |
|---|---|---|
| none (the engine as built) | 1.0 % | 2.0 |
| `nibble=hi` (NVFP4 nibble order) | 71 % | 170 |
| `fp8scale=div` (FP8 scale direction) | 100 % | 605 |
| `ws2=div` (NVFP4 multiplier direction) | 100 % | 605 |

Logs: `~/6ixinfer-logs/diag-transfer-{A-code,A-prose,A-code-fp8,A-prose-fp8,NEG-nibble,NEG-fp8scale,NEG-ws2,N80-code,N80-prose}.log`.

The unexplained rows, all of them:
- layer 31, row 1 (all four runs): err 0.36–0.48 % of the update — *below* the layer's typical
  1 %. It counts as an outlier only in floor units: row 1 carries a residual 40–130× larger than
  its update through layers 12–29 (measured), and at layer 31 its output floor is 0.02 % of the
  update — consistent with that layer cancelling the large component, so the output is small
  while the intermediates were rounded at the large scale.
- layer 3, rows 42 and 73 (A code, exact): err 2.0–2.1 %, twice the layer's typical 1.0 %, no
  routing change. An attention layer. Small, and not attributed to anything.
- layer 34 row 1 (A prose exact, 1.2 %) and layer 6 row 1 (A prose fp8, 1.1 %): same kind as the
  first item, at the typical error level.

#### 4.7 What I conclude

1. Which side is off: **neither.** The engine and the reference implement the same function;
   they differ in arithmetic (BF16 buffers and BF16 router logits in the engine, float32 in the
   reference), and a top-8-of-256 router turns that into different experts for ≈ 5 % of rows per
   layer, each within one BF16 step of a tie.
2. The written gate cannot certify this (or any) routed-MoE model on a single ~100-token text at
   0.06 / 0.5: the reference fails it against itself, and the 80B fails it.
3. Checkpoint A is numerically right on the prefill path the forward check exercises, in both
   dense forms, on the evidence of 4.6: every layer and the head within ≈ 2 BF16 floors (3 in the
   fp8 form), against 170–600 for a wrong convention.

#### 4.8 Still open

- **The gate as originally written is still a FAIL on 3 of 4 runs for A.** It was restated
  twice afterwards, on instruction (4.9); `gpu-steps.md` of both ports still describes the
  original bounds and I did not edit it. Whether the gate should instead be the eight-text
  average it was derived from, or the one-step check of 4.6, is the owner's decision.
- `diag_transfer.py` and the other check scripts live in `~/6ixinfer-out/q36/jobs/` on the box
  (copies in `jobs/` beside this file), not in the repository.
- The forward check covers the prefill path with streaming residency. The serving decode kernels
  and the resident loader are exercised only by section 5 (requests, MTP against plain decode).
- Checkpoint B (FP8): fails the final gate on one text and was not measured — see 4.11.
- The 80B's own eight texts were not re-run, so the 0.06 average itself is not re-verified here.

#### 4.9 The gate as restated on 2026-10-04, and the 35B beside the 80B

`sixlabs/bench/compare_forward_check.py` after the change (three conditions, all required:
pearson ≥ 0.995, mean ≤ 0.10, greedy token equal wherever the reference's top-2 margin > 0.5;
max reported). Run over the outputs already on disk — same engine runs, same reference files as
4.2 and 4.5. The two models' inputs are the same two prompts (same system/user/assistant text),
each rendered with its own template: 113 / 87 tokens for the 35B (its template adds an empty
`<think>` block), 107 / 83 for the 80B.

| prompt, dense form | 35B-A3B NVFP4: mean / max (pos) / pearson / greedy | verdict | 80B NVFP4: mean / max (pos) / pearson / greedy | verdict |
|---|---|---|---|---|
| code, checkpoint | 0.07846 / 0.633 (36) / 0.999296 / 110 of 113, all 3 misses near ties (≤ 0.097) | PASS | 0.10475 / 5.107 (106) / 0.994186 / 105 of 107, one miss at margin 0.729 | **FAIL** (all three conditions) |
| prose, checkpoint | 0.05843 / 0.374 (62) / 0.999335 / 86 of 87, miss at 0.299 | PASS | 0.05435 / 1.180 (15) / 0.999268 / 81 of 83, misses at 0.063, 0.051 | PASS |
| code, `fp8` | 0.06098 / 0.862 (7) / 0.999157 / 110 of 113, misses ≤ 0.096 | PASS | 0.06356 / 1.866 (106) / 0.999040 / 106 of 107, miss at 0.073 | PASS |
| prose, `fp8` | 0.06906 / 0.494 (29) / 0.998726 / 87 of 87 | PASS | 0.08572 / 0.754 (23) / 0.999261 / 80 of 83, misses ≤ 0.051 | PASS |

Old bounds, for reference: the 35B would pass 1 of 4, the 80B 0 of 4.

On "not worse than the 80B by more than a small factor": the 35B's mean is 0.75×, 1.08×, 0.96×
and 0.81× the 80B's on the four rows, its worst position is smaller on every row, and its
correlation is within 0.0006 of the 80B's or better. It is not worse.

The 80B's code/checkpoint-form run was the odd one out: a greedy disagreement at row 0 at a
0.73 nat reference margin and a 5.1 nat difference at position 106. The 80B's owner explained
both (relayed by the orchestrator), and the saved files bear the explanation out:
- position 106 is a newline straight after `<|im_end|>`, a token both sides call impossible —
  reference −24.27, engine −19.17 in the checkpoint form and −22.41 in `fp8` (read from the
  saved outputs). It is 5.1 of that text's 11.1 nat of total difference.
- row 0's context is the single token `<|im_start|>`. From the engines' own state dumps, the
  80B's row 0 carries a residual up to 131× its layer update (a typical row: 5.6×); on the 35B
  it is row 1, up to 141× (typical 3.0×). Those rows have the largest BF16 floor in the sequence.

**The gate as finally set (second change, same day, same instruction path):** the three
conditions above, with (a) positions whose scored token the *reference* puts below −15
excluded from the mean, the correlation and the reported max, and (b) a greedy disagreement on
the first two rows listed but not counted. All eight comparisons re-run from the saved files
(on the Mac; full output in `gate-final-8-runs.txt` beside this file):

| run | original gate (0.06 / 0.5) | first restatement | **final gate** | far-tail positions excluded (reference; engine) | gated mean / pearson / max (pos) | greedy |
|---|---|---|---|---|---|---|
| 35B code, checkpoint | fail | pass | **PASS** | 1: pos 26 (−16.75; −16.59) | 0.07768 / 0.999269 / 0.633 (36) | 110 of 113, misses all near ties |
| 35B prose, checkpoint | pass | pass | **PASS** | 0 | 0.05843 / 0.999335 / 0.374 (62) | 86 of 87, near tie |
| 35B code, `fp8` | fail | pass | **PASS** | 1: pos 26 (−16.75; −16.54) | 0.05959 / 0.999038 / 0.862 (7) | 110 of 113, near ties |
| 35B prose, `fp8` | fail | pass | **PASS** | 0 | 0.06906 / 0.998726 / 0.494 (29) | 87 of 87 |
| 80B code, checkpoint | fail | **fail** | **PASS** | 3: pos 50 (−21.81; −21.77), 90 (−21.63; −21.99), 106 (−24.27; −19.17) | 0.05439 / 0.998015 / 1.519 (82) | 105 of 107; the row-0 miss at margin 0.729 listed, not counted |
| 80B prose, checkpoint | fail | pass | **PASS** | 2: pos 18 (−21.05; −21.05), 25 (−15.22; −15.15) | 0.05479 / 0.998780 / 1.180 (15) | 81 of 83, near ties |
| 80B code, `fp8` | fail | pass | **PASS** | 3: pos 50, 90, 106 (−24.27; −22.41) | 0.04669 / 0.999359 / 0.456 (10) | 106 of 107, near tie |
| 80B prose, `fp8` | fail | pass | **PASS** | 2: pos 18 (−21.05; −20.71), 25 (−15.22; −14.88) | 0.07940 / 0.998914 / 0.754 (23) | 80 of 83, near ties |

Nothing fails under the final gate. What should be kept in mind about it, plainly:
- The only verdict the two exclusions change is the 80B's code/checkpoint run. They were added
  after that run failed and because it failed. The reasons given are physically sensible and
  the measurements above support them, but a gate adjusted until the known-good model passes is
  a calibration, not an independent test.
- Exclusion (b) drops rows 0–1 from the greedy condition whatever the margin. Exclusion (a)
  drops tokens the reference considers nearly impossible. An error confined to those places
  would no longer be seen by this script. A wrong tensor or convention would still fail it at
  every other position.
- The evidence I rely on for "the 35B is numerically right" is 4.6, which uses neither.

#### 4.10 Is the reference itself right? (the notes' three checks, CPU only)

`$OUT/jobs/ref_checks.py`, log `~/6ixinfer-logs/ref-checks.log`.

1. `mean_logprob` is a small negative number: A code −1.966, A prose −2.023, B code −2.000,
   B prose −1.945. **Holds.**
2. A and B (one model, two quantizations) on the same ids: code mean |A − B| 0.152, median 0.019,
   max 1.44, pearson 0.9967, same top-1 in 111 of 113 rows; prose mean 0.221, median 0.114, max
   1.27, pearson 0.9900, same top-1 in 79 of 87 rows. **The notes expected "a few hundredths of a
   nat" — that is not met.** It is not a container error (that collapses to −12.4, see 3), and it
   is 2–4× the BF16 self-noise of 4.4; a 4-bit and an 8-bit quantization of the experts are
   different weights, so some gap is expected, but I have no independent number for how large it
   should be. Left open.
3. Each convention switch collapses the result (code ids): A `fp8scale=div` −13.35, A `ws2=div`
   −12.42, A `nibble=hi` −13.64, B `fp8scale=div` −12.42, against −1.97 / −2.00 as shipped.
   **Holds.**

#### 4.11 Checkpoint B (FP8) — **FAILS the final gate on the code text (greedy condition, row 7); passes on prose; every layer passes the one-step check. NOT measured.**

Bind check and memory plan pass (sections 2, 3); the reference is computed (4.10). Forward check
at 04:41, log `~/6ixinfer-logs/forward-check-B.log`, final gate (`compare_forward_check.py` as in
`fixes.patch`):

| input | form | mean / pearson / max (pos) over gated positions | greedy | original gate | first restatement | **final gate** |
|---|---|---|---|---|---|---|
| code | checkpoint | 0.06847 / 0.997960 / 1.822 (9) | 112 of 113; **row 7: engine 248046 (`<\|im_end\|>`), reference 17313, reference margin 1.3141 nat** | fail | fail | **FAIL** |
| prose | checkpoint | 0.06059 / 0.998608 / 0.624 (29) | 85 of 87, both misses near ties | fail | pass | PASS |
| code | `--dense-weights fp8` | identical to the checkpoint form | | fail | fail | **FAIL** |
| prose | `--dense-weights fp8` | identical to the checkpoint form | | fail | pass | PASS |

(For this checkpoint `--dense-weights fp8` changes nothing the forward check exercises: the two
forms' state dumps are byte-identical. Only the lm head's form differs, and the forward check's
head path evidently reads the BF16 head either way — so B's block-FP8 head is not covered by it.)

As instructed for a failure of this gate, B was **not booted and not measured**, and here is
where it diverges:

- **Not at a layer.** One step per layer on the engine's own inputs
  (`~/6ixinfer-logs/diag-transfer-B-{code,prose}.log`): median row error 1.02 % / 1.03 % of the
  layer's update (worst layer 1.35 % / 1.27 %), 2.04 / 2.03 BF16 floors; outlier rows 271 of 4520
  and 175 of 3480, of which 269 and 174 are re-routings between experts within 1.18 / 1.43 BF16
  steps of each other; unexplained 2 and 1 (row 1 at layers 31 and 34, 0.4–1.2 % of the update).
  Head, one step: token log-prob |Δ| mean 0.0017 / 0.0019, max 0.019 / 0.009, greedy rows
  differing 0 of 113 and 0 of 87. This is A's profile exactly.
- **Row 7 drifts, it does not jump** (`$OUT/row-trace-B-code-7.txt`). Its distance from the
  reference's trajectory grows layer by layer — 0.4 % after layer 0, 2.1 % after 7, 3.8 % after
  11, 6.4 % after 18, 10 % after 24, 21 % after 31, 27 % after 39 — against 4–6 % for the median
  row at the end. The row is the token ` programming` in "You are a careful programming
  assistant."; the model is choosing between ` assistant` and ending the turn. On the engine's
  final residual the reference's own head gives `<|im_end|>` −0.65, ` assistant` −1.03; on its own
  residual, ` assistant` −0.31, `<|im_end|>` −1.62.
- **The reference does the same thing to itself.** Re-evaluated with BF16 router logits and BF16
  activations (same weights, same code, `~/6ixinfer-logs/diag-router-B-code.log`), the reference
  picks 248046 at row 7 where its float32 run picks 17313 at that 1.31 nat margin; with only the
  router convention changed the margin shrinks from 1.31 to 0.38 nat. On checkpoint A the same
  experiment flips row 13 at a 1.14 nat margin. And the engine **passes** this gate — and the
  original 0.06 / 0.5 gate — against the reference evaluated with the engine's router convention:
  mean 0.0505, pearson 0.99961, max 0.474, 112 of 113 greedy with no miss above the margin.

What this means, plainly: I see no wrong tensor in checkpoint B, and the same evidence that
supports A supports B. But condition 3 of the gate (greedy agreement wherever the reference's
margin exceeds 0.5 nat) is one the reference does not meet against itself at BF16 arithmetic —
it flips at 1.14 and 1.31 nat — so a correct BF16 engine can fail it, and B does. I have not
changed the gate again and I have not measured B. Whether B is measured is the owner's call.

#### 4.12 For the 80B's owner: does this say anything about the shape dependence?

The question was whether the batching-dependent log-probabilities on the 80B come from the MoE
path — per-tile activation scaling, or a routed expert flipping at a near tie. From the one-step
check (4.6), measured on the 35B and on the 80B itself:

- **Router near-tie flips are sufficient to explain it, and they are the only large per-row
  effect I see.** In every layer about 5–7 % of rows are outliers (the 80B: 7.0 % of 5136
  row-layers on code, 4.8 % of 3984 on prose), and nearly all of them are rows the engine and
  the reference route differently — 337 of 360 and 184 of 193 on the 80B decode to a different
  expert selection, the exchanged experts' router logits at most 1.23 BF16 steps apart (median
  0.5). Such a row is off by roughly 6–70 BF16 floors in that layer (a few percent to tens of
  percent of the layer's update); an unflipped row is off by about 2.
- **So any last-bit change upstream is enough.** The router's logits are rounded to BF16 before
  the top-k (`MoeRouterMode::SoftmaxTopk`), ties go to the lower expert id, and at logits of 4–16
  a step is 0.03–0.06: two logits that round to the same BF16 value are an exact tie, and a
  one-ulp difference in a BF16 activation feeding the router dot product — which a different
  chunking can produce through a different GEMM lowering or accumulation order (`docs/testing.md`
  describes row-count-dependent dense lowering as "tolerance-equal, not bitwise") — moves one of
  them across a rounding boundary and changes the expert. Over 40–48 layers almost every token
  passes through several such rows, which is why the two chunkings never re-converge and why the
  per-token differences look like the engine-vs-reference ones (correlation 0.999, mean a few
  hundredths, single tokens off by half a nat or more).
- **Positions with the largest end-to-end differences are not cleanly "the flipped ones"**, and
  I would not expect them to be: after 40+ layers nearly every position has been through at
  least one flip. The clean statement is per layer, not per position.
- **Activation scaling per tile: no evidence for it in these runs.** The W4A4 activation
  workspace is 0.000 GiB in the 35B's plans (I did not print the 80B's), and the unflipped rows sit at a uniform ≈ 2 BF16
  floors in every layer and both layer kinds; a per-tile scale effect would show as a
  row-block-dependent error and I see none. I did not test two chunkings of the engine against
  each other, so this is "no sign of it", not "ruled out".
- A direct test that would settle it on the 80B: dump the routed expert ids of the same tokens
  under the two chunkings (the prefill path's `MoeTraceStaging`, which `tests/python/
  route_trace_test.py` already reads) and diff them; the first layer and row where the ids
  differ is where the two runs part. `$OUT/jobs/diag_transfer.py` does the equivalent offline
  from two `--dump-states` files plus the reference.

### 5. First boot, the draft layer, first requests (checkpoint A) — PASS, with two expectations of the notes not met as written

Config: `deploy/cluster_qwen3.6-35b-a3b_nvfp4_w1.json` (an unedited copy of the example) and
`.env` in `~/6ixinfer-ports` (nodes 127.0.0.1, bind 172.17.0.1:18190, fabric/journal 29990/29991,
log dir `~/6ixinfer-ports/log`, stage dir `~/6ixinfer-out/stage`, resident cache
`~/6ixinfer-out/resident-cache`). Every `dgpp-cluster` call passed `--config`.

```
$ python3 scripts/dgpp-cluster doctor --config deploy/cluster_qwen3.6-35b-a3b_nvfp4_w1.json
Preflight: 0 failed check(s). No files were installed and no services were started or stopped.
$ python3 scripts/dgpp-cluster up --config deploy/cluster_qwen3.6-35b-a3b_nvfp4_w1.json
rank 0 serving: ok (38s)   READY — curl http://172.17.0.1:18190/v1/models
serve log: serve: model family qwen3_5
           rank 0: memory plan … model weights (resident) 20.93 GiB; … mtp scratch 0.12 GiB; …
           rank 0: memory plan total 36.00 GiB (30.05 GiB device + 5.96 GiB pinned) + 4.00 GiB headroom
           serve: listening on :18190 — Qwen3.6-35B-A3B (boot 36.4s)
```

(Boot times are in section C; this first boot ran beside other CPU jobs.) Serve logs are kept as
`~/6ixinfer-logs/serve-A-boot1-mtp.log` and `serve-A-boot2-nomtp.log`.

**First requests** (`$OUT/jobs/first_requests.py http://172.17.0.1:18190 Qwen3.6-35B-A3B`, output
`$OUT/first-requests-A.txt`; temperature 0):

| request | what came back | verdict |
|---|---|---|
| raw completion, "The capital of France is" | `" Paris.\nA. True\nB"` | ok |
| plain chat, thinking on, `max_tokens` 256 (the notes' request) | `reasoning_content` non-empty and correct (works 17×20 + 17×3 = 391), `content` empty, `finish_reason` `length` — the reasoning used all 256 tokens | **expectation not met as written** (the notes wanted content 391 and `stop`) |
| the same with `max_tokens` 2048 | `finish_reason` `stop`, `content` `"\n\n391"`, 312 reasoning tokens of 317 | ok — the model needs 317 tokens; the notes' 256 was too small |
| plain chat, `enable_thinking` false | `content` `"391"`, no `reasoning_content`, `stop` | ok |
| forced tool call (`tool_choice` `required`) | `tool_calls[0].function` = `get_weather` / `{"city": "Paris"}`, `finish_reason` `tool_calls` | ok |
| auto tool call (`tool_choice` `auto`) | the same call, `finish_reason` `tool_calls` | ok |
| auto, a question that needs no tool | `content` `"ready"`, no tool call | ok |
| an `image_url` part | HTTP 400 `the served model does not support image inputs`; the server keeps answering | ok |

Two things seen here that are not this port's doing but the owner may want to know: with
thinking on, `content` starts with the `"\n\n"` that follows `</think>` (`"\n\n391"`), and an
auto tool call comes back with `content` `"\n\n"` beside the `tool_calls` (the forced call has
`content` null).

**The draft layer (MTP):**

1. Greedy output with MTP equals plain decode. Restarted with `--knobs=--no-mtp` and replayed the
   fixed prompt set (`sixlabs/refset/gen_prompts.py`, 18 non-echo rows):
   - `sixlabs/bench/compare_transcripts.py q36A_short_mtp.jsonl q36A_short_nomtp.jsonl` →
     `identical: 18/18 non-echo rows` (as on the 80B).
   - That tool compares `content` only, which on this thinking model is empty for the 8 rows
     whose reasoning runs to `max_tokens`. So the same prompts were also captured in full
     (`$OUT/jobs/capture_full.py`: reasoning kept, thinking on with `max_tokens` 1024, and
     thinking off) and compared field by field (`compare_full.py`): `identical: 30/30 rows
     (6952 generated tokens compared)`.
   - The 8 `echo` rows are refused by this server for every model (`'echo' is not supported in
     this server version`, and there is no `/tokenize`): the teacher-forced comparison is
     section 4, not this.
   - The notes' command `--knobs "--no-mtp"` is rejected by the launcher's argument parser
     (`argument --knobs: expected one argument`); it has to be written `--knobs=--no-mtp`.
2. Acceptance, from `/v1/metrics` → `scheduler.spec_decode`, depth 2:
   after the first requests 541 of 616 drafts accepted (position 1: 286/308 = 93 %, position 2:
   255/308 = 83 %); after the prompt-set captures 6611 of 7776 (91 % / 79 %). Well above half:
   the draft layer is bound correctly, stacked experts included.

The reference's `mtp` mode (the CPU-side cross-check of the gate/up order with
`--variant stacked=up_gate`) was not run: the engine-side acceptance above already answers the
question it exists for.

## C. Measurements — 6ix.cpp, checkpoint A (NVFP4) — done

Config as shipped in the template: 8 request slots, 262,144-token K/V pool (BF16), `dense_weights`
`fp8`, MTP depth 2, 4 GiB prefix cache, decode graph on. Served alias `Qwen3.6-35B-A3B`.

### C.1 Boot

| boot | conditions | launcher "serving: ok" | serve log `boot` | plan total | `nvidia-smi` |
|---|---|---|---|---|---|
| 1st, cold (no resident image) | other CPU jobs running (a reference run of mine, another session's compile) | 38 s | 36.4 s | 36.00 GiB | 39,478 MiB |
| `--no-mtp`, from the resident image | quiet | 8 s | 7.8 s | 33.64 GiB | 37,023 MiB |
| **cold again, image removed first** | **quiet box (the timed one)** | **48 s** | **47.5 s** | **36.00 GiB (30.05 device + 5.96 pinned) + 4.00 headroom** | **39,478 MiB = 38.55 GiB** |
| from the resident image, MTP on | quiet box, 04:47 | 10 s | 9.0 s | 36.00 GiB | 39,478 MiB |

"Cold" here means no resident image: the loader reads the checkpoint, builds the resident forms
and writes a 20.4 GB image to `~/6ixinfer-out/resident-cache/` as it goes. The page cache was
warm (the checkpoint had been read by the reference runs), and I could not drop it. The two cold
boots differ by 11 s and the quiet one was the slower; I do not know why (the image write to NVMe
is the obvious suspect) and did not chase it.

The process holds 2.55 GiB more than the plan's total says (38.55 against 36.00).

Pre-measurement checks, taken at 03:55 before the first number: no `~/6ixinfer-check` build
running; `nvidia-smi pmon -c 3 -s u` showed every other GPU process (the two vLLM engine cores,
the three python services) at 0 % or idle in all three samples; load average 1.7.

### C.2 `bench_decode.py` (03:55:45, first run after the cold boot)

```
$ python3 sixlabs/bench/bench_decode.py --base http://172.17.0.1:18190 --model Qwen3.6-35B-A3B --streams 1,2,4,8
single stream:   97.7 tok/s  (prose)
single stream:  120.1 tok/s  (code)
2 streams:      128.8 tok/s aggregate
4 streams:      194.6 tok/s aggregate
8 streams:      271.3 tok/s aggregate
mtp depth 2: accepted 1913/2568 = 74.5%  per position [1084, 829]/[1284, 1284]
```

One run, as asked. Output kept in `$OUT/bench-decode-A.txt`.

### C.3 Fleet suite

`TOK=nvidia/Qwen3.6-35B-A3B-NVFP4 python3 sixlabs/bench/run_fleet_suite.py --host 172.17.0.1 --port 18190 --model Qwen3.6-35B-A3B --maxc 8`
from `~/6ixinfer-ports`, detached, log `~/6ixinfer-logs/fleet-suite-A.log`.

`TOK` is the one thing added to the command as given. The ShareGPT script resolves a tokenizer
repo from the served id; for `Qwen3.6-35B-A3B` its lookup finds nothing usable (the matching
cache entry `Qwen/Qwen3.6-35B-A3B` holds only a README) and would hand the harness a name that is
not a repository. `TOK` is that script's own documented override; it changes no measurement.

Results — all five steps ran, `err=0`, `DONE` after 2661 s (03:56–04:41):

| step | result | where the result file is |
|---|---|---|
| 1 DecodeBench (max_tokens 512, mixed prompts, `ignore_eos`) | aggregate tok/s at 1 / 2 / 4 / 8 streams: **107.4 / 143.6 / 191.2 / 257.2**; per-stream median 109.9 / 78.2 / 55.0 / 35.8; median TTFT 0.10 / 0.15 / 0.35 / 0.44 s; MTP accept rate 0.79 / 0.73 / 0.69 / 0.69; 0 failed | run_id `20261004-035636-258364` in `~/.local/state/saih-decodebench/history.json` (the app's recent-runs list) |
| 2 llama-benchy (prompt 2048, gen 128, 3 runs) | depth 0 — generation tok/s at c1 / c2 / c4 / c8: **106.8 / 131.9 / 129.8 / 116.8**, prompt processing 1977 / 1402 / 944 / 598 tok/s; depth 4096 — 105.4 / 132.2 / 150.7 / 136.9, pp 1899 / 1413 / 939 / 589; depth 16384 — 98.1 / 129.0 / **15.6 / 13.7**, pp 1879 / 1335 / 1665 / 1634 | `~/llama-benchy-20261004-035635.json` |
| 3 ShareGPT (64 prompts per level) | output tok/s at c1 / c2 / c4 / c8: **96.8 / 134.3 / 180.8 / 217.9**; TTFT mean 186 / 214 / 271 / 341 ms, P99 502 / 593 / 884 / 1012 ms; ITL P99 25.8 / 51.0 / 151 / 202 ms; 64 of 64 successful at every level | `~/bench-results/sharegpt/sharegpt-bench-20261004-041906.txt` (+ `.raw`) |
| 4 tool-eval short (15 scenarios) | **final score 93** — 13 pass, 2 partial (TC-11 Simple Math, TC-14 Malformed Response), 0 fail | `~/tool-eval-bench/report-20261004-042707.json` |
| 5 tool-eval hard (88 scenarios) | **final score 89** — 72 pass, 12 partial, 4 fail (TC-31, TC-60, TC-68, TC-88) | `~/tool-eval-bench/report-20261004-042809.json` |

Two things in those numbers that are the engine's, not the benchmark's, and that the owner will
want to see:
- **llama-benchy at depth 16384, 4 and 8 concurrent: generation falls to 15.6 / 13.7 tok/s in
  total.** The serve log at the time shows the K/V pool at 97 % (3,991 of 4,096 blocks) with
  2 requests live and 2 queued: eight 18K-token prompts plus the prefix-cache entries that share
  the 262,144-token pool do not fit, so requests wait for blocks.
- **Time to first response grows steeply with concurrency** even at depth 0: 1.0 s at c1,
  2.6 s at c2, 7.4 s at c4, 23.5 s at c8 (74 s at depth 4096, c8). The log line `prefill budget
  256 tokens/tick` is the cause: prompts are prefilled 256 tokens at a time between decode
  passes (735 tok/s in total while 4 requests decode, against 1,977 tok/s alone). The notes'
  section 6 already lists interleaved-prefill tuning as open for this family name.

Over the whole measured boot (176,393 tokens generated, 818 prompts): MTP accepted 108,254 of
135,848 drafted tokens = 79.7 % (position 1: 87.2 %, position 2: 72.1 %). 0 `ERROR` lines in the
serve log (`~/6ixinfer-logs/serve-A-boot3-cold-measured.log`).

## D. Engine stopped

`python3 scripts/dgpp-cluster down --config …/cluster_qwen3.6-35b-a3b_nvfp4_w1.json` at 04:41
(after the suite) and again at 04:47 (after the resident-image boot): `serve: stopped cleanly`,
no `6ix-Serve` process left, none in `nvidia-smi --query-compute-apps`, 97 GiB available.

## E. vLLM baseline — `qwen3.6-35b@dgxone` (vLLM, `Qwen/Qwen3.6-35B-A3B-FP8`) — done

Started and stopped only through `~/6ixlabs/cluster/bin/delta-launch`, compose file untouched,
no `DELTA_*` override. Pre-checks at 04:48: my engine down and off the GPU; 97.8 GiB available;
the only fleet slot running on DGXone was the protected `qwen3.5-0.8b`, so the launcher's
eviction path had nothing it could stop.

```
$ ~/6ixlabs/cluster/bin/delta-launch qwen3.6-35b@dgxone up        (launched 04:48:55)
[admit] qwen3.6-35b@dgxone needs 45 GiB; 97.8 GiB available, 0 GiB still to be claimed by loading slots; must leave 19.7 GiB
[admit] fits, nothing to evict
[delta-launch] waiting up to 900s for :8221 to serve (expected ~420s)...
  serving after 404s.
[delta-launch] done.                                              (exit code 0 at 04:56:34)
```

- **Load time: 404 s** from `up` to serving (the launcher's own timer; `~/logs/delta-launch-timings.log`
  has `served_in=404s min_avail_during_load=44GiB`).
- **GPU memory: 45,577 MiB = 44.51 GiB** (`nvidia-smi --query-compute-apps`, pid 2042750
  `VLLM::EngineCore`), 44.5 GiB left available on the node. Not the 64 GiB expected: the fleet's
  own footprint file already says "measured 44.5 GiB beside the 0.8b … was 64" for this slot
  since its KV cache was capped at 8 GiB.
- Serving `qwen3.6-35b`, root `Qwen/Qwen3.6-35B-A3B-FP8`, `max_model_len` 131072; 401 without the
  key, 200 with it. Recipe facts that matter when reading the comparison: 24 seats
  (`--max-num-seqs 24`), FP8 KV cache, 8 GiB KV, thinking off by default at the server.
- The launcher's last step (`delta-ensure`) printed three lines about LiteLLM aliases of DGXtwo
  slots that are not answering (`… dgxtwo:8282 not responding -> alias REMOVED …`). That is the
  fleet's reconcile, run by the launcher; I did not touch LiteLLM or DGXtwo.

Pre-measurement checks at 04:56: no `~/6ixinfer-check` build, launcher finished, `pmon` showed no
GPU process above 0 %.

### E.1 `bench_decode.py` (04:56:50, first run after the load)

```
$ with_fleet_key.sh python3 sixlabs/bench/bench_decode.py --base http://172.17.0.1:8221 --model qwen3.6-35b --streams 1,2,4,8 --api-key-env FLEET_KEY
single stream:   50.1 tok/s  (prose)
single stream:   50.1 tok/s  (code)
2 streams:       62.6 tok/s aggregate
4 streams:      105.9 tok/s aggregate
8 streams:      190.0 tok/s aggregate
metrics: HTTP Error 404: Not Found          (vLLM has no /v1/metrics; expected)
```

`with_fleet_key.sh` (`$OUT/jobs/`) reads `FLEET_KEY` from `~/.config/6ixlabs/secrets.env` into the
command's environment; the key was never printed, logged or put on a command line by anything I
ran (checked: 0 lines of the logs contain it; no process of mine has it in `argv`).

### E.2 Fleet suite (04:57–05:17, 1188 s; `DONE with errors` — the one "error" is the skipped ShareGPT step)

`with_fleet_key.sh python3 sixlabs/bench/run_fleet_suite.py --host 172.17.0.1 --port 8221 --model qwen3.6-35b --maxc 8 --api-key-env FLEET_KEY`,
log `~/6ixinfer-logs/fleet-suite-vllm.log`.

| step | result | where the result file is |
|---|---|---|
| 1 DecodeBench | aggregate tok/s at 1 / 2 / 4 / 8 streams: **49.3 / 76.3 / 107.4 / 141.5**; per-stream median 50.0 / 39.0 / 27.3 / 17.9; median TTFT 0.14 / 0.28 / 0.29 / 0.36 s; 0 failed | run_id `20261004-045731-79d2bc` in `~/.local/state/saih-decodebench/history.json` |
| 2 llama-benchy | depth 0 — generation tok/s at c1 / c2 / c4 / c8: **49.6 / 74.4 / 99.8 / 114.0**, prompt processing 5651 / 4802 / 4260 / 5024 tok/s, time to first response 0.40 / 0.67 / 1.55 / 2.42 s; depth 4096 — 49.0 / 63.8 / 74.1 / 79.8, pp 5071 / 4639 / 5459 / 5390; depth 16384 — 46.2 / 55.1 / 43.4 / 39.2, pp 5509 / 5500 / 5524 / 5607 | `~/llama-benchy-20261004-045730.json` |
| 3 ShareGPT | **not run** — the endpoint requires the fleet key and `~/run-sharegpt-bench.sh` (the fleet's script) has no way to send one: its own `/v1/models` probe and its harness call are unauthenticated. A reason outside this port; I did not modify that script. | — |
| 4 tool-eval short | **final score 90** — 13 pass, 1 partial (TC-14), 1 fail (TC-03) | `~/tool-eval-bench/report-20261004-050639.json` |
| 5 tool-eval hard | **final score 85** — 71 pass, 7 partial, 10 fail | `~/tool-eval-bench/report-20261004-050742.json` |

The key: 0 occurrences in the suite log and in the llama-benchy result file; none of the
processes I started had it on a command line (checked during the llama-benchy and tool-eval
steps).

### E.3 Stopped

```
$ ~/6ixlabs/cluster/bin/delta-launch qwen3.6-35b@dgxone down        (05:17:50)
 Container qwen3.6-35b Stopped … Removed
[delta-launch] qwen3.6-35b down; marked stopped for delta-guard.     (rc 0)
```

Container gone (`docker ps -a` has no `qwen3.6-35b`), port 8221 closed, `nvidia-smi` compute apps
back to the five that were there before, 97.4 GiB available, the slot marked stopped again as it
was before I started it. `unloaded.txt` written at 05:18.

## F. Checkpoint B (FP8) on 6ix.cpp — NOT booted, NOT measured

It fails the final numerical gate on the code text (section 4.11: one greedy disagreement at a
1.31 nat reference margin; the reference shows the same flip against itself at BF16
arithmetic; every layer passes the one-step check). The instruction for a failure of that gate
was to stop and report where it diverges, so the FP8 column below is empty. What exists for B:
unit and loader-fixture tests (section 1), bind check (2), memory plan — 34.06 GiB of weights,
42.2 GiB device + 1.9 pinned planned at 8 slots / 262,144 tokens (3) — the reference (4.10), the
forward check and the one-step check (4.11).

## G. The table

One DGX Spark (DGXone). 6ix.cpp: the template's settings (8 slots, 262,144-token BF16 K/V pool,
MTP depth 2, `dense_weights` `fp8`, 4 GiB prefix cache). vLLM: the fleet's recipe as written
(24 seats, 131,072 context, FP8 KV capped at 8 GiB).

| | 6ix.cpp, NVFP4 (`nvidia/…-NVFP4`) | 6ix.cpp, FP8 (`Qwen/…-FP8`) | vLLM, FP8 (`Qwen/…-FP8`) |
|---|---|---|---|
| numerical gate | passes the final gate 4 of 4 (failed the original 3 of 4) | **fails the final gate on 1 of 2 texts** | not applicable |
| load time, to serving | **47.5 s** cold on a quiet box (36.4 s on the first, busier boot); 9.0 s from the resident image | not measured | **404 s** |
| planned memory | 36.00 GiB (30.05 device + 5.96 pinned) | 42.2 GiB device + 1.9 pinned by `--plan` (not booted) | 45 GiB (the fleet's footprint file) |
| measured GPU memory | **39,478 MiB = 38.55 GiB** | not measured | **45,577 MiB = 44.51 GiB** |
| `bench_decode` single stream, prose | **97.7 tok/s** | — | **50.1 tok/s** |
| `bench_decode` single stream, code | **120.1 tok/s** | — | **50.1 tok/s** |
| `bench_decode` 2 streams (aggregate) | 128.8 | — | 62.6 |
| `bench_decode` 4 streams | 194.6 | — | 105.9 |
| `bench_decode` 8 streams | 271.3 | — | 190.0 |
| DecodeBench, aggregate tok/s at 1 / 2 / 4 / 8 | 107.4 / 143.6 / 191.2 / 257.2 | — | 49.3 / 76.3 / 107.4 / 141.5 |
| llama-benchy, generation tok/s at c1 / c2 / c4 / c8, depth 0 | 106.8 / 131.9 / 129.8 / 116.8 | — | 49.6 / 74.4 / 99.8 / 114.0 |
| llama-benchy, prompt processing tok/s, same | 1977 / 1402 / 944 / 598 | — | 5651 / 4802 / 4260 / 5024 |
| llama-benchy, time to first response, same | 1.0 / 2.6 / 7.4 / 23.5 s | — | 0.40 / 0.67 / 1.55 / 2.42 s |
| llama-benchy, generation tok/s at depth 16384 | 98.1 / 129.0 / 15.6 / 13.7 | — | 46.2 / 55.1 / 43.4 / 39.2 |
| ShareGPT, output tok/s at c1 / c2 / c4 / c8 | 96.8 / 134.3 / 180.8 / 217.9 (TTFT P99 0.50 / 0.59 / 0.88 / 1.01 s) | — | not run (keyed endpoint, see E.2) |
| tool-eval short (15) | 93 | — | 90 |
| tool-eval hard (88) | 89 | — | 85 |

How to read it, and what it does not show:

- **The two measured columns are different weights.** 6ix.cpp ran the NVFP4 release (4-bit
  experts), vLLM the FP8 release. The like-for-like column — 6ix.cpp on the FP8 weights — is
  the one that is empty. The FP8 checkpoint is 13 GiB larger in memory on 6ix.cpp (34.06 GiB of
  weights against 20.93) and I have no speed figure for it.
- **Decode: 6ix.cpp is about 2× vLLM at one stream** (97.7–120.1 against 50.1 by `bench_decode`;
  107.4 against 49.3 by DecodeBench; 106.8 against 49.6 by llama-benchy) and 1.4–1.8× at eight
  streams by `bench_decode` and DecodeBench (271 / 257 against 190 / 142). By llama-benchy at
  eight streams the two are level (116.8 against 114.0): there the prompts are 2,048 tokens and
  6ix.cpp's prefill is the bottleneck.
- **Prefill: vLLM is about 3× 6ix.cpp alone and 8× at eight concurrent** (5,651 against 1,977
  tok/s; 5,024 against 598), and its time to first response stays under 2.5 s where 6ix.cpp's
  reaches 23 s. At 16K of context with 4–8 concurrent requests 6ix.cpp's generation collapses
  (15.6 / 13.7 tok/s in total) because the requests queue for K/V pool blocks; vLLM's does not.
- **Load: 47.5 s against 404 s; memory: 38.55 against 44.51 GiB** — with different context and
  seat settings on the two sides (6ix.cpp: 262,144-token pool, 8 seats; vLLM: 131,072, 24 seats).
- **The tool-eval scores are not like for like either.** The vLLM slot serves with thinking off
  by default (`--default-chat-template-kwargs '{"enable_thinking": false}'`); 6ix.cpp serves
  with thinking on. Same model family, different weights, different default mode: 93 / 89
  against 90 / 85 says both work, not that one is better.
- Every figure is one run. `bench_decode` was run once per engine, as asked, right after boot;
  I have no run-to-run spread for it.
- On the earlier record for this hardware: vLLM was noted at 63.6 GiB and a 430 s load; tonight
  it measured 44.51 GiB and 404 s (its KV cache is capped at 8 GiB in the current recipe).
  Atlas's published 116.5 tok/s single stream sits inside 6ix.cpp's 97.7 (prose) – 120.1
  (code) range; I did not run Atlas.
