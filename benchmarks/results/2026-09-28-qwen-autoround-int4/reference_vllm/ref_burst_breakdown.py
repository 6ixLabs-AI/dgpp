#!/usr/bin/env python3
"""Kernel bursts of an nsys report (bursts separated by idle gaps over --gap-ms, default 100)
and the per-kernel table of one burst (--burst N, negative from the end; default -1).
Usage: ref_burst_breakdown.py REPORT.nsys-rep|REPORT.sqlite [--burst N] [--top N] [--gap-ms X] [--list]"""
import os, sqlite3, subprocess, sys
from collections import defaultdict

rep = sys.argv[1]
burst_sel, top, gap_ms, list_only = -1, 40, 100.0, False
i = 2
while i < len(sys.argv):
    if sys.argv[i] == "--burst": burst_sel = int(sys.argv[i + 1]); i += 2
    elif sys.argv[i] == "--top": top = int(sys.argv[i + 1]); i += 2
    elif sys.argv[i] == "--gap-ms": gap_ms = float(sys.argv[i + 1]); i += 2
    elif sys.argv[i] == "--list": list_only = True; i += 1
    else: raise SystemExit(f"unknown argument {sys.argv[i]}")
db = rep if rep.endswith(".sqlite") else os.path.splitext(rep)[0] + ".sqlite"
if not os.path.exists(db):
    subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", db, rep], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
c = sqlite3.connect(db)
rows = c.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start").fetchall()
if not rows: raise SystemExit("no kernels")
bursts = []
cur = [rows[0]]
for r in rows[1:]:
    if r[0] - cur[-1][1] > gap_ms * 1e6:
        bursts.append(cur); cur = [r]
    else:
        cur.append(r)
bursts.append(cur)
t0 = rows[0][0]
def busy(b):
    # union of kernel intervals
    total = 0; end = -1
    for st, en, _ in b:
        if st > end: total += en - st; end = en
        elif en > end: total += en - end; end = en
    return total
print(f"{len(bursts)} bursts (gap > {gap_ms:.0f} ms):")
for idx, b in enumerate(bursts):
    wall = b[-1][1] - b[0][0]
    print(f"  [{idx:3d}] t={(b[0][0]-t0)/1e9:8.2f}s  wall {wall/1e6:9.1f} ms  busy {busy(b)/1e6:9.1f} ms  kernels {len(b):6d}")
if list_only: raise SystemExit(0)
b = bursts[burst_sel]
wall = b[-1][1] - b[0][0]
print(f"\nburst {burst_sel}: {len(b)} kernels over {wall/1e6:.1f} ms wall, GPU busy {busy(b)/1e6:.1f} ms")
agg = defaultdict(lambda: [0.0, 0])
for st, en, name in b:
    short = name.split("(")[0]
    short = short.replace("void ", "")
    agg[short][0] += (en - st) / 1e6; agg[short][1] += 1
tot = sum(v[0] for v in agg.values())
print(f"{'kernel':90s} {'ms':>9s} {'n':>6s} {'us/inst':>8s} {'share':>6s}")
for name, (ms, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"{name[:90]:90s} {ms:9.2f} {n:6d} {1000*ms/n:8.1f} {100*ms/tot:5.1f}%")
