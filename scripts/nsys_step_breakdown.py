#!/usr/bin/env python3
"""The decode step's kernel breakdown from an nsys report of rank 0.

Exports the report to sqlite (once; reused when present), takes the kernels
after the last graph capture (the steady decode: from the first instance of
the per-step marker kernel, default 'spec_commit_kernel', to its last), groups
them by name and prints per-kernel GPU time per step, instances per step and
the share — plus the step count, the GPU busy time per step, and the wall
time per step across the window (the difference is host/launch/collective
idle). --marker NAME picks another once-per-step kernel; --top N rows.

--burst: no marker — the window is the last burst of kernels (bursts
separated by idle gaps over 100 ms), i.e. the last request's prefill (plus
its one decode step): totals per kernel over the burst, no per-step scaling.

Usage: nsys_step_breakdown.py REPORT.nsys-rep [--marker NAME] [--top N] [--skip S] [--burst] [--max-window-ms X]
"""
import os
import sqlite3
import subprocess
import sys

rep = sys.argv[1]
marker, top, skip, burst = "spec_commit_kernel", 40, 20, False
max_window_ms = 0.0  # --max-window-ms: drop marker windows longer than this (a prefill between steps)
i = 2
while i < len(sys.argv):
    if sys.argv[i] == "--burst":
        burst = True; i += 1
    elif sys.argv[i] == "--marker":
        marker = sys.argv[i + 1]; i += 2
    elif sys.argv[i] == "--top":
        top = int(sys.argv[i + 1]); i += 2
    elif sys.argv[i] == "--skip":
        skip = int(sys.argv[i + 1]); i += 2
    elif sys.argv[i] == "--max-window-ms":
        max_window_ms = float(sys.argv[i + 1]); i += 2
    else:
        raise SystemExit(f"unknown argument {sys.argv[i]}")
db = os.path.splitext(rep)[0] + ".sqlite"
if not os.path.exists(db) or os.path.getmtime(db) < os.path.getmtime(rep):
    subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                    "-o", db, rep], check=True, stdout=subprocess.DEVNULL)
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id = k.demangledName ORDER BY k.start")
rows = cur.fetchall()
if not rows:
    raise SystemExit("no kernels in the report")


def short(name):
    # The name with its template arguments (the GEMV instantiations differ
    # only there), without the parameter list: cut at the first '(' outside
    # any <...>.
    n = name.replace("void ", "")
    for p in ("dgpp::", "(anonymous namespace)::", "<unnamed>::", "net::"):
        n = n.replace(p, "")
    depth, cut = 0, len(n)
    for j, ch in enumerate(n):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            cut = j
            break
    n = n[:cut].replace("GemvEpilogue)", "").replace("(", "")
    return n[:88]


if burst:
    # Bursts: kernels separated by an idle gap over 100 ms; the last one.
    starts = [0]
    for j in range(1, len(rows)):
        if rows[j][0] - rows[j - 1][1] > 100_000_000:
            starts.append(j)
    win = rows[starts[-1]:]
    t0, t1 = win[0][0], max(r[1] for r in win)
    wall = t1 - t0
    by = {}
    busy = 0
    for s, e, name in win:
        d = e - s
        busy += d
        n = short(name)
        t, c = by.get(n, (0, 0))
        by[n] = (t + d, c + 1)
    print(f"last burst: {len(win)} kernels over {wall / 1e6:.1f} ms wall, GPU busy "
          f"{busy / 1e6:.1f} ms ({100.0 * busy / wall:.1f} %); {len(starts)} bursts in the report")
    print(f"{'kernel':<88} {'ms':>8} {'n':>6} {'us/inst':>8} {'share':>6}")
    for n, (t, c) in sorted(by.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"{n:<88} {t / 1e6:8.2f} {c:6d} {t / c / 1e3:8.1f} {100.0 * t / busy:5.1f}%")
    raise SystemExit(0)

marks = [r[0] for r in rows if short(r[2]).startswith(marker) or marker in r[2]]
if len(marks) < skip + 3:
    raise SystemExit(f"only {len(marks)} instances of the marker '{marker}' "
                     f"(kernels: {sorted(set(short(r[2]) for r in rows))[:30]})")
# The window: from the (skip)-th marker to the last marker — whole steps.
# With --max-window-ms only the marker-to-marker windows at most that long
# count (a decode step; a prefill between two requests' steps is dropped).
t0, t1 = marks[skip], marks[-1]
if max_window_ms > 0:
    spans = [(marks[j], marks[j + 1]) for j in range(skip, len(marks) - 1)
             if marks[j + 1] - marks[j] <= max_window_ms * 1e6]
    if not spans:
        raise SystemExit("no marker window within --max-window-ms")
    steps = len(spans)
    win = []
    k = 0
    for a, b in spans:
        while k < len(rows) and rows[k][0] < a:
            k += 1
        while k < len(rows) and rows[k][0] < b:
            win.append(rows[k]); k += 1
    wall = sum(b - a for a, b in spans)
else:
    steps = len(marks) - 1 - skip
    win = [r for r in rows if t0 <= r[0] < t1]
    wall = t1 - t0
by = {}
busy = 0
for s, e, name in win:
    d = e - s
    busy += d
    n = short(name)
    t, c = by.get(n, (0, 0))
    by[n] = (t + d, c + 1)
print(f"window: {steps} steps between markers '{marker}' (skipping the first {skip}); "
      f"wall {wall / steps / 1e6:.3f} ms/step, GPU busy {busy / steps / 1e6:.3f} ms/step "
      f"({100.0 * busy / wall:.1f} %), {len(win) / steps:.1f} kernels/step")
print(f"{'kernel':<88} {'ms/step':>8} {'n/step':>7} {'us/inst':>8} {'share':>6}")
for n, (t, c) in sorted(by.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"{n:<88} {t / steps / 1e6:8.3f} {c / steps:7.1f} {t / c / 1e3:8.1f} "
          f"{100.0 * t / busy:5.1f}%")
