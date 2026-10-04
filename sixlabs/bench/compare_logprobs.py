#!/usr/bin/env python3
"""compare_logprobs.py — DGPP's teacher-forced token log-probabilities against the numpy host
reference and against Atlas, for the echo rows of the reference set.

  python3 compare_logprobs.py DGPP_OUT_DIR   (files <row>.txt = qwen35_forward_check output)
"""
import json, os, sys
import numpy as np

out_dir = sys.argv[1]
atlas = {}
for l in open("atlas_short.jsonl"):
    r = json.loads(l)
    if r["kind"] == "echo":
        atlas[r["id"]] = r
print(f"{'row':16s} {'tok':>4s} | {'mean lp dgpp':>12s} {'ref':>8s} {'atlas':>8s} | "
      f"{'dgpp-ref mean|d|':>16s} {'max|d|':>7s} {'pearson':>8s} | {'dgpp-atlas mean|d|':>18s} {'max|d|':>7s} {'pearson':>8s}")
tot = []
for rid in sorted(atlas):
    p = os.path.join(out_dir, rid + ".txt")
    if not os.path.exists(p):
        continue
    line = [l for l in open(p) if l.startswith("token logprobs:")]
    if not line:
        print(f"{rid:16s} no logprobs in {p}")
        continue
    d = np.array([float(x) for x in line[0].split(":", 1)[1].split(",")])
    ref = np.array(json.load(open(f"ref_accept/{rid}.json"))["ref"]["logprobs"][1:], dtype=float)
    a = np.array([np.nan if v is None else v for v in atlas[rid]["token_logprobs"][1:len(d) + 1]], dtype=float)
    n = min(len(d), len(ref), len(a))
    d, ref, a = d[:n], ref[:n], a[:n]
    ok = ~np.isnan(a)
    dr, da = np.abs(d - ref), np.abs(d[ok] - a[ok])
    print(f"{rid:16s} {n + 1:4d} | {d.mean():12.4f} {ref.mean():8.4f} {a[ok].mean():8.4f} | "
          f"{dr.mean():16.4f} {dr.max():7.3f} {np.corrcoef(d, ref)[0, 1]:8.5f} | "
          f"{da.mean():18.4f} {da.max():7.3f} {np.corrcoef(d[ok], a[ok])[0, 1]:8.5f}")
    tot.append((dr.mean(), dr.max(), da.mean()))
if tot:
    t = np.array(tot)
    print(f"overall: dgpp-ref mean|d| {t[:, 0].mean():.4f} (worst position {t[:, 1].max():.3f}); dgpp-atlas mean|d| {t[:, 2].mean():.4f}")
