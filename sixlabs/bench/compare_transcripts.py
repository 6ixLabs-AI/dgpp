#!/usr/bin/env python3
"""compare_transcripts.py — two captures of the same prompt set (capture_refset.py), row by row.

  python3 compare_transcripts.py atlas_short.jsonl dgpp_short.jsonl
"""
import json, sys


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip()}


def text_of(r):
    if r["kind"] == "completion":
        return r.get("text") or ""
    if r.get("tool_calls"):
        return json.dumps([(c["function"]["name"], json.loads(c["function"]["arguments"]))
                           for c in r["tool_calls"]], sort_keys=True)
    return r.get("content") or ""


A, D = load(sys.argv[1]), load(sys.argv[2])
same = n = 0
for k, a in A.items():
    d = D.get(k)
    if a["kind"] == "echo" or d is None:
        continue
    n += 1
    if "error" in d:
        print(f"{k:12s} ERROR {d['error'][:160]}")
        continue
    ta, td = text_of(a), text_of(d)
    i = 0
    while i < min(len(ta), len(td)) and ta[i] == td[i]:
        i += 1
    eq = ta == td
    same += eq
    u = d.get("usage") or {}
    tag = "SAME     " if eq else "DIFF@%-4d" % i
    print(f"{k:12s} {tag} finish {str(d.get('finish_reason')):10s} out_tok {str(u.get('completion_tokens')):4s} "
          f"{d.get('seconds')}s  B: {td[:64]!r}")
    if not eq:
        print(f"{'':22s}A: ...{ta[max(0, i - 24):i + 48]!r}")
        print(f"{'':22s}B: ...{td[max(0, i - 24):i + 48]!r}")
print(f"identical: {same}/{n} non-echo rows")
echo = [d for d in D.values() if d["kind"] == "echo"]
if echo:
    print("echo rows in B:", echo[0].get("error", "ok")[:160])
