#!/usr/bin/env python3
"""build_dgpp.py — configure + build a DGPP checkout, with the fleet's progress line.

  python3 build_dgpp.py <checkout> [-j N] [--target T ...] [--fresh]

Prints `[s/2] ... elapsed=Ns` heartbeats so the build shows in the Job view like any other job.
"""
import argparse, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("checkout")
ap.add_argument("-j", type=int, default=8)
ap.add_argument("--target", action="append", default=[])
ap.add_argument("--fresh", action="store_true")
ap.add_argument("--preset", default="release")
a = ap.parse_args()
t0 = time.time()


def step(i, cmd):
    print(f"[{i}/2] 0/1 err=0 elapsed={int(time.time() - t0)}s :: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=a.checkout)
    print(f"[{i}/2] 1/1 err={int(rc != 0)} elapsed={int(time.time() - t0)}s", flush=True)
    if rc:
        print("FAILED", flush=True)
        sys.exit(rc)


step(1, ["cmake"] + (["--fresh"] if a.fresh else []) + ["--preset", a.preset])
cmd = ["cmake", "--build", "--preset", a.preset, "-j", str(a.j)]
for t in a.target:
    cmd += ["--target", t]
step(2, cmd)
print("DONE", flush=True)
