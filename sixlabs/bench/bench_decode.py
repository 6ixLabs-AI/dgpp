#!/usr/bin/env python3
"""bench_decode.py — single-stream and concurrent decode speed of an OpenAI-compatible server.

  python3 bench_decode.py --base http://127.0.0.1:18090 --model Qwen3-Next-80B [--streams 1,2,4]
                          [--api-key-env NAME]

Decode rate is (tokens of a long run - tokens of a short run) / (time difference), so prompt
prefill and connection overhead cancel. Greedy, raw completions, two prompts (prose and code).

--api-key-env NAME: the endpoint wants `Authorization: Bearer <key>` (a fleet vLLM slot); the key
is read from the environment variable NAME. It is never taken on the command line (argv shows in
`ps` and in job logs) and never printed.
"""
import argparse, json, os, sys, time, urllib.request
import concurrent.futures as cf

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--streams", default="1,2")
ap.add_argument("--api-key-env", default="", metavar="NAME",
                help="environment variable holding the endpoint's bearer key (default: no auth)")
a = ap.parse_args()
PROMPTS = ["Once upon a time, in a village at the edge of a great forest, there lived",
           "def quicksort(arr):\n"]
HEADERS = {"content-type": "application/json"}
if a.api_key_env:
    if not os.environ.get(a.api_key_env):
        sys.exit(f"--api-key-env {a.api_key_env}: that variable is unset or empty")
    HEADERS["Authorization"] = "Bearer " + os.environ[a.api_key_env]


def gen(n, p):
    b = {"model": a.model, "prompt": p, "temperature": 0, "max_tokens": n}
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(a.base + "/v1/completions", data=json.dumps(b).encode(),
                               headers=HEADERS), timeout=600)
    d = json.loads(r.read())
    return time.time() - t, d["usage"]["completion_tokens"]


gen(8, PROMPTS[0])
for p in PROMPTS:
    t1, n1 = gen(32, p)
    t2, n2 = gen(288, p)
    print(f"single stream: {(n2 - n1) / (t2 - t1):6.1f} tok/s  ({'prose' if p is PROMPTS[0] else 'code'})")
for s in [int(x) for x in a.streams.split(",") if int(x) > 1]:
    t = time.time()
    with cf.ThreadPoolExecutor(s) as ex:
        rs = list(ex.map(lambda i: gen(256, PROMPTS[i % 2]), range(s)))
    dt = time.time() - t
    print(f"{s} streams:     {sum(r[1] for r in rs) / dt:6.1f} tok/s aggregate")
try:
    m = json.loads(urllib.request.urlopen(urllib.request.Request(a.base + "/v1/metrics", headers=HEADERS),
                                          timeout=10).read())
    sd = (m.get("scheduler") or {}).get("spec_decode") or {}
    if sd.get("num_draft_tokens_total"):
        print(f"mtp depth {sd.get('depth')}: accepted {sd['num_accepted_tokens_total']}/{sd['num_draft_tokens_total']} "
              f"= {100.0 * sd['num_accepted_tokens_total'] / sd['num_draft_tokens_total']:.1f}%  per position "
              f"{sd.get('num_accepted_tokens_per_pos_total')}/{sd.get('num_draft_tokens_per_pos_total')}")
except Exception as e:
    print("metrics:", e)
