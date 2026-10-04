#!/usr/bin/env python3
"""compare_forward_check.py — one qwen35_forward_check run against the numpy host reference's
`score` of the same token ids (tools/qwen3next_reference.py score --out REF.json).

  python3 compare_forward_check.py ENGINE.txt REF.json [--mean 0.06] [--max 0.5]

ENGINE.txt is the forward check's stdout. Compared: the teacher-forced log-probability of every
given token (the engine's `token logprobs:` line against the reference's `logprobs`), and the
greedy next token at every position (the engine's `argmax ids:` against the reference's top-1).
Exit 0 when the mean |difference| is within --mean nat and the worst position within --max nat
(defaults: the 80B's measured agreement, 0.06 mean, and a loose per-position bound); the argmax
agreement is reported, and a disagreement is listed with the reference's own margin, since a
near tie may legitimately flip.
"""
import argparse
import json
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("engine")
ap.add_argument("ref")
ap.add_argument("--mean", type=float, default=0.06)
ap.add_argument("--max", type=float, default=0.5)
a = ap.parse_args()

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
print(f"tokens {eng.size + 1}: mean logprob engine {eng.mean():.5f} reference {ref.mean():.5f}")
print(f"|engine - reference|: mean {d.mean():.5f}  median {np.median(d):.5f}  max {d.max():.5f} "
      f"(position {int(d.argmax()) + 1})  pearson {np.corrcoef(eng, ref)[0, 1]:.6f}")
flips = 0
if am_line is not None:
    eng_top = [int(x) for x in am_line.split(",")]
    tops = ref_doc["top"][1:] + [ref_doc["next_top"]]         # row t predicts position t + 1
    n = min(len(eng_top), len(tops))
    for t in range(n):
        if eng_top[t] != tops[t][0][0]:
            flips += 1
            margin = tops[t][0][1] - tops[t][1][1]
            print(f"  argmax differs at row {t}: engine {eng_top[t]}, reference {tops[t][0][0]} "
                  f"(reference margin {margin:.4f} nat)")
    print(f"greedy next token: {n - flips}/{n} rows agree")
ok = d.mean() <= a.mean and d.max() <= a.max
print("PASS" if ok else "FAIL", f"(bounds: mean <= {a.mean}, max <= {a.max})")
sys.exit(0 if ok else 1)
