#!/usr/bin/env python3
"""yarn_compare.py — is the engine's YaRN ramp the reference's YaRN ramp?

  python3 yarn_compare.py DIR

DIR holds, per text T:
  ref_yarn2_T.json, ref_plain_T.json   tools/qwen3next_reference.py score [--variant yarn_factor=2] --out
  eng_yarn2_T.txt [, eng_plain_T.txt]  qwen35_forward_check [--rope-scaling 2] output

Teacher-forced token log-probabilities, mean absolute difference per text. The ramp is right when
the engine with YaRN sits as close to the reference with YaRN as the two sit without it, and
clearly further from the reference WITHOUT it (that distance is the ramp's own effect).
"""
import glob, json, os, sys

d = sys.argv[1]


def ref(path):
    return [x for x in json.load(open(path))["logprobs"][1:]]


def eng(path):
    line = [l for l in open(path) if l.startswith("token logprobs:")]
    return [float(x) for x in line[0].split(":", 1)[1].split(",")] if line else None


def mad(a, b):
    n = min(len(a), len(b))
    return sum(abs(a[i] - b[i]) for i in range(n)) / n if n else float("nan")


texts = sorted(os.path.basename(p)[len("ref_yarn2_"):-len(".json")] for p in glob.glob(os.path.join(d, "ref_yarn2_*.json")))
print(f"{'text':16s} {'tok':>4s} | {'ramp effect':>11s} | {'eng yarn - ref yarn':>19s} {'eng yarn - ref plain':>20s} | {'eng plain - ref plain':>21s} | verdict")
bad = 0
for t in texts:
    ry = ref(os.path.join(d, f"ref_yarn2_{t}.json"))
    pp = os.path.join(d, f"ref_plain_{t}.json")
    rp = ref(pp) if os.path.exists(pp) else None
    ey_p, ep_p = os.path.join(d, f"eng_yarn2_{t}.txt"), os.path.join(d, f"eng_plain_{t}.txt")
    ey = eng(ey_p) if os.path.exists(ey_p) else None
    ep = eng(ep_p) if os.path.exists(ep_p) else None
    effect = mad(ry, rp) if rp else float("nan")
    a = mad(ey, ry) if ey else float("nan")
    b = mad(ey, rp) if ey and rp else float("nan")
    c = mad(ep, rp) if ep and rp else float("nan")
    if ey is None:
        verdict = "no engine output yet"
    elif rp is None:
        verdict = "no plain reference"
    else:
        ok = a < b
        bad += not ok
        verdict = "closer to the YaRN reference" if ok else "NOT closer to the YaRN reference"
    print(f"{t:16s} {len(ry) + 1:4d} | {effect:11.4f} | {a:19.4f} {b:20.4f} | {c:21.4f} | {verdict}")
print("RESULT:", "FAIL" if bad else "ok")
sys.exit(1 if bad else 0)
