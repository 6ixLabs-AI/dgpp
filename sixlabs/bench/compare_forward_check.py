#!/usr/bin/env python3
"""compare_forward_check.py — one qwen35_forward_check run against the numpy host reference's
`score` of the same token ids (tools/qwen3next_reference.py score --out REF.json).

  python3 compare_forward_check.py ENGINE.txt REF.json [--mean 0.10] [--pearson 0.995]
                                   [--margin 0.5] [--tail -15] [--skip-rows 2]

ENGINE.txt is the forward check's stdout. Compared: the teacher-forced log-probability of every
given token (the engine's `token logprobs:` line against the reference's `logprobs`), and the
greedy next token at every position (the engine's `argmax ids:` against the reference's top-1).

The gate (exit 0 only when all three hold):
  1. the correlation of the engine's and the reference's log-probabilities is at least --pearson;
  2. the mean |engine - reference| is at most --mean nat;
  3. wherever the reference's margin between its two best tokens is above --margin nat, the
     engine's greedy token is the reference's. A disagreement at a smaller margin is a near tie:
     it is listed with the margin, and it does not fail the gate.
The worst position's |difference| is REPORTED with its position; it is not bounded.
Two exclusions, each listed in the output whenever it applies:
  a. conditions 1 and 2 and the reported max are taken over the positions whose scored token the
     REFERENCE gives a log-probability of at least --tail (default -15);
  b. a greedy disagreement on the first --skip-rows rows (default 2) is listed, not counted.

HISTORY OF THIS GATE — it was changed twice on 2026-10-04, both times on the instruction of the
session that owns the 80B port, relayed by the Qwen3.6-35B-A3B bring-up's orchestrator, and both
times after measurements on DGXone (the engine at 89955e0, this reference). Every statistic of
the earlier gates is still printed, so each verdict can be read off any run:

  ORIGINAL: mean <= 0.06 and max <= 0.5, over all positions of one text.
    It sat on a correct engine's own noise floor. The 0.06 was the 80B's AVERAGE over eight
    short texts (6IXSERVE.md; compare_logprobs.py's `overall` line), applied as a ceiling on
    every single text; the 0.5 per-position bound was never measured on anything.
      The 80B, the port known to work, one ~100-token text at a time
      (ids: sixlabs/ports/qwen3-coder-next/inputs):
          text   dense form   mean |d|   max |d| (pos)   pearson
          code   checkpoint   0.10475    5.107 (106)     0.994186
          code   fp8          0.06356    1.866 (106)     0.999040
          prose  checkpoint   0.05435    1.180 (15)      0.999268
          prose  fp8          0.08572    0.754 (23)      0.999261
      The numpy reference against ITSELF on the Qwen3.6-35B-A3B NVFP4 checkpoint, same weights
      and code, only its arithmetic held to BF16 (router logits, or router logits + activations):
          code   mean 0.067-0.071, max 1.33-1.74        prose  mean 0.054-0.056, max 0.35-0.53
      Reported by the 80B port's session (not measured in this bring-up): the engine's log-
      probabilities of the same tokens through two different chunkings agree to pearson 0.9993,
      mean 0.036, worst token 0.44; the same request twice is bit-identical.
    A routed MoE picks its top-k experts from logits that nearly tie in several percent of rows
    per layer; two correct evaluations that differ in the last bit of a BF16 activation route
    some rows to different experts and do not re-converge. A single position can be off by a nat
    or more in a correct engine, which is why the max is reported and not bounded.

  FIRST RESTATEMENT: conditions 1-3 above, over all positions and all rows.
    The 80B's code / checkpoint-form run was still outside it (mean 0.10475, pearson 0.994186,
    one greedy miss at a 0.73 nat margin). The two exclusions below are that session's answer to
    exactly that run; they were added after it failed, not before.

  EXCLUSION a (--tail): a token the reference itself puts below e^-15 (3e-7) is in the far tail
    of the distribution, where both sides' values are differences of large logits and say
    nothing about any greedy or sampled output. The 80B's position 106 is a newline straight
    after <|im_end|>: reference -24.27, engine -19.17 (checkpoint form) and -22.41 (fp8). That
    one token was 5.1 of the text's 11.1 nat of total difference. A mean in nats over a text is
    dominated by such tokens whenever one occurs.

  EXCLUSION b (--skip-rows): the first rows have one or two tokens of context and carry a
    residual far larger than the layer's update, so their BF16 storage floor is the largest in
    the sequence. Measured from the engines' own state dumps on the code text: the 80B's row 0
    reaches 131x its update (a typical row: 5.6x), the 35B's row 1 reaches 141x (typical: 3.0x).
    The 80B's row 0 — context: the single token <|im_start|> — is where its checkpoint form
    picked another token at a 0.73 nat reference margin; its fp8 form agrees with the reference
    there.

What these exclusions cost in sensitivity: a wrong tensor or convention is far outside any of
the three gates — the log-probabilities collapse towards -ln(vocab) at every position and the
correlation with them — so neither exclusion can hide one. They do stop the gate from seeing an
error confined to the far tail or to the first two rows.
"""
import argparse
import json
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("engine")
ap.add_argument("ref")
ap.add_argument("--mean", type=float, default=0.10, help="ceiling on the mean |engine - reference| (nat)")
ap.add_argument("--pearson", type=float, default=0.995, help="floor on the correlation of the two log-probability series")
ap.add_argument("--margin", type=float, default=0.5,
                help="a greedy disagreement fails only where the reference's top-2 margin exceeds this (nat)")
ap.add_argument("--tail", type=float, default=-15.0,
                help="positions whose scored token the reference puts below this log-probability are excluded "
                     "from the mean, the correlation and the reported max")
ap.add_argument("--skip-rows", type=int, default=2,
                help="a greedy disagreement on the first N rows is listed but not counted")
a = ap.parse_args()
OLD_MEAN, OLD_MAX = 0.06, 0.5          # the original gate: printed for reference, not applied

lp_line = am_line = None
for line in open(a.engine):
    if line.startswith("token logprobs:"):
        lp_line = line.split(":", 1)[1]
    elif line.startswith("argmax ids:"):
        am_line = line.split(":", 1)[1]
if lp_line is None:
    sys.exit(f"{a.engine}: no 'token logprobs:' line (did the forward check fail?)")
eng = np.array([float(x) for x in lp_line.split(",")])
ref_doc = json.load(open(a.ref))
ref = np.array(ref_doc["logprobs"][1:], dtype=float)          # entry i is log P(ids[i] | ids[<i])
if eng.size != ref.size:
    sys.exit(f"token counts differ: engine {eng.size + 1}, reference {ref.size + 1} — not the same ids")
d = np.abs(eng - ref)
pearson_all = float(np.corrcoef(eng, ref)[0, 1])
print(f"tokens {eng.size + 1}: mean logprob engine {eng.mean():.5f} reference {ref.mean():.5f}")
print(f"|engine - reference|: mean {d.mean():.5f}  median {np.median(d):.5f}  max {d.max():.5f} "
      f"(position {int(d.argmax()) + 1})  pearson {pearson_all:.6f}")
keep = ref >= a.tail
tail = np.nonzero(~keep)[0]
print(f"far-tail positions excluded (reference logprob < {a.tail:g}): {tail.size} of {ref.size}")
for i in tail:
    print(f"  position {int(i) + 1}: reference {ref[i]:.2f}, engine {eng[i]:.2f}, |difference| {d[i]:.3f}")
if keep.sum() < 2:
    sys.exit("fewer than two positions left after the far-tail exclusion — nothing to gate on")
dk, pos = d[keep], np.nonzero(keep)[0]
mean, pearson = float(dk.mean()), float(np.corrcoef(eng[keep], ref[keep])[0, 1])
worst = int(pos[dk.argmax()]) + 1
if tail.size:
    print(f"|engine - reference| over the {int(keep.sum())} gated positions: mean {mean:.5f}  median {np.median(dk):.5f}  "
          f"max {dk.max():.5f} (position {worst})  pearson {pearson:.6f}")
flips = decisive = early = 0
greedy_ok = False
if am_line is not None:
    eng_top = [int(x) for x in am_line.split(",")]
    tops = ref_doc["top"][1:] + [ref_doc["next_top"]]         # row t predicts position t + 1
    n = min(len(eng_top), len(tops))
    for t in range(n):
        if eng_top[t] != tops[t][0][0]:
            flips += 1
            margin = tops[t][0][1] - tops[t][1][1]
            over = margin > a.margin
            if over and t < a.skip_rows:
                early += 1
                note = f" — above the margin, but one of the first {a.skip_rows} rows: listed, not counted"
            elif over:
                decisive += 1
                note = " — ABOVE the margin: fails"
            else:
                note = ", a near tie"
            print(f"  argmax differs at row {t}: engine {eng_top[t]}, reference {tops[t][0][0]} "
                  f"(reference margin {margin:.4f} nat{note})")
    print(f"greedy next token: {n - flips}/{n} rows agree; disagreements where the reference's margin "
          f"exceeds {a.margin}: {decisive} counted, {early} on the first {a.skip_rows} rows not counted")
    greedy_ok = decisive == 0
else:
    print("greedy next token: no 'argmax ids:' line in the engine output — the greedy check cannot be made (fails)")
checks = [(f"pearson {pearson:.6f} >= {a.pearson}", pearson >= a.pearson),
          (f"mean {mean:.5f} <= {a.mean}", mean <= a.mean),
          (f"greedy token agrees wherever the reference margin > {a.margin} (rows from {a.skip_rows} on)", greedy_ok)]
for text, held in checks:
    print(f"  [{'ok' if held else 'FAILED'}] {text}")
print(f"  [reported, not bounded] max {dk.max():.5f} at position {worst}")
old_ok = d.mean() <= OLD_MEAN and d.max() <= OLD_MAX
first_ok = pearson_all >= a.pearson and d.mean() <= a.mean and am_line is not None and decisive + early == 0
print(f"original gate (mean <= {OLD_MEAN}, max <= {OLD_MAX}, all positions; for reference only): "
      f"{'would pass' if old_ok else 'would fail'}")
print(f"first restatement (the three conditions over all positions and rows; for reference only): "
      f"{'would pass' if first_ok else 'would fail'}")
ok = all(held for _, held in checks)
print("PASS" if ok else "FAIL", f"(gate: pearson >= {a.pearson}, mean <= {a.mean}, greedy agreement above a {a.margin} nat margin; "
      f"reference logprob >= {a.tail:g}; greedy rows from {a.skip_rows})")
sys.exit(0 if ok else 1)
