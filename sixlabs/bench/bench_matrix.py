#!/usr/bin/env python3
"""bench_matrix.py — context x concurrency measurements of a DGPP deployment (2026-10-03).

  python3 bench_matrix.py --base http://172.17.0.1:18090 --model Qwen3-Next-80B \
      --log ~/dgpp-next/log/deployments/<id>/serve_r0.log --out bench_matrix.jsonl

Each cell sends N concurrent chat requests whose prompts are DIFFERENT filler documents of about
the stated size (so the prefix cache cannot help), greedy, 256 output tokens. The timings are the
server's own, read from its log line for each request id ("retired ...: prefill P tok in X ms;
decode D tok / passes in S s: R tok/s"), plus the wall clock of the whole cell. One JSONL row per
cell, flushed as it completes.
"""
import argparse, json, os, random, re, sys, time, urllib.request
import concurrent.futures as cf

SUBJ = ["The surveyor", "A night clerk", "The ferry captain", "An apprentice", "The archivist",
        "A beekeeper", "The signalman", "A cartographer", "The quartermaster", "A glassblower"]
VERB = ["recorded", "repaired", "counted", "delivered", "measured", "catalogued", "inspected", "sealed"]
OBJ = ["the brass fittings", "nine crates of lamp oil", "the tide tables", "a ledger of arrivals",
       "the eastern stairwell", "forty lengths of rope", "the harbour bell", "a box of wax seals"]
WHEN = ["before dawn", "at midday", "during the storm", "on the third day", "after the market closed",
        "while the fog held", "at the turn of the tide", "by lantern light"]


def prompt(tokens, seed):
    rng = random.Random(seed)
    n = max(1, int(tokens / 18.7))                     # ~18.7 tokens a line (measured)
    lines = [f"Entry {i}: {rng.choice(SUBJ)} {rng.choice(VERB)} {rng.choice(OBJ)} {rng.choice(WHEN)}."
             for i in range(n)]
    return (f"Log {seed}. Read it, then write a detailed summary of what the entries describe.\n\n" +
            "\n".join(lines) + "\n\nNow write the summary.")


def call(a, text, max_tokens):
    b = {"model": a.model, "messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": max_tokens}
    t = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(b).encode(),
                                   headers={"content-type": "application/json"}), timeout=3600)
        d = json.loads(r.read())
        return {"id": d["id"], "wall": time.time() - t, "prompt_tokens": d["usage"]["prompt_tokens"],
                "completion_tokens": d["usage"]["completion_tokens"]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:200], "wall": time.time() - t}


RET = re.compile(r"request '([^']+)' retired.*?prefill (\d+) tok \((\d+) cached\) in (\d+) ms; decode (\d+) tok / (\d+) pass(?:es)? in "
                 r"([\d.]+) s: ([\d.]+) tok/s, [\d.]+ ms/tok, (\d+) ms/pass, ([\d.]+) tok/pass")


def server_stats(log, ids):
    out = {}
    for line in open(os.path.expanduser(log), errors="replace"):
        m = RET.search(line)
        if m and m.group(1) in ids:
            out[m.group(1)] = {"prefill_tok": int(m.group(2)), "cached": int(m.group(3)), "prefill_ms": int(m.group(4)),
                               "decode_tok": int(m.group(5)), "passes": int(m.group(6)), "decode_s": float(m.group(7)),
                               "tok_s": float(m.group(8)), "ms_pass": int(m.group(9)), "tok_pass": float(m.group(10))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", default="1000x1,1000x2,1000x4,1000x8,8000x1,8000x2,8000x4,8000x8,"
                                       "32000x1,32000x2,32000x4,64000x1,128000x1")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warm", action="store_true",
                    help="send each prompt once first (4 output tokens), so the measured round hits the prefix "
                         "cache and times DECODE under concurrency without the prompts' prefills in the way")
    a = ap.parse_args()
    cells = [(int(c.split("x")[0]), int(c.split("x")[1])) for c in a.cells.split(",")]
    t0, err, seed = time.time(), 0, 1000
    print(f"[1/1] 0/{len(cells)} err=0 elapsed=0s", flush=True)
    with open(a.out, "a") as out:
        for i, (ctx, n) in enumerate(cells, 1):
            prompts = [prompt(ctx, seed + j) for j in range(n)]
            seed += n
            if a.warm:
                for p in prompts:
                    call(a, p, 4)
            t = time.time()
            with cf.ThreadPoolExecutor(n) as ex:
                rs = list(ex.map(lambda p: call(a, p, a.max_tokens), prompts))
            wall = time.time() - t
            time.sleep(1.0)                                    # let the server write its lines
            st = server_stats(a.log, {r["id"] for r in rs if "id" in r})
            ok = [r for r in rs if "id" in r and r["id"] in st]
            row = {"context": ctx, "streams": n, "warm": a.warm, "wall_s": round(wall, 2),
                   "errors": [r["error"] for r in rs if "error" in r]}
            if ok:
                s = [st[r["id"]] for r in ok]
                dec_window = max(x["decode_s"] for x in s)
                row.update(prompt_tokens=ok[0]["prompt_tokens"], cached_tokens=[x["cached"] for x in s],
                           prefill_ms=[x["prefill_ms"] for x in s],
                           prefill_tok_s=round(sum(x["prefill_tok"] for x in s) / (sum(x["prefill_ms"] for x in s) / 1000.0), 0),
                           per_stream_tok_s=[x["tok_s"] for x in s],
                           aggregate_tok_s=round(sum(x["decode_tok"] for x in s) / dec_window, 1),
                           ms_pass=round(sum(x["ms_pass"] for x in s) / len(s), 1),
                           tok_pass=round(sum(x["tok_pass"] for x in s) / len(s), 2),
                           time_to_last_first_token_s=round(max(r["wall"] - st[r["id"]]["decode_s"] for r in ok), 2))
            err += bool(row["errors"])
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[1/1] {i}/{len(cells)} err={err} elapsed={int(time.time() - t0)}s :: {json.dumps(row)[:230]}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
