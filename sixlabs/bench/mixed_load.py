#!/usr/bin/env python3
"""mixed_load.py — what do answers in progress cost when long prompts keep arriving?

  python3 mixed_load.py --base http://172.17.0.1:18090 --model Qwen3-Next-80B --label budget4096 \
      --out mixed_load.jsonl

Two kinds of client run together for --seconds:
  writers   stream long answers from a short prompt (the people chatting)
  readers   send a long prompt nobody has sent before and take a few tokens (retrieval / judging)
and the same writers then run alone, as the baseline.

The failure this measures (2026-10-04): with `engine.prefill_budget_tokens` at 4096, a tick that
has a prompt to read spends about 1.5 s reading before the writers get their next step. Under a
retrieval benchmark the whole engine put out 24 tokens/s and the gap between tokens was 2.2 s at
p95, against 136 tokens/s at four writers with nothing to read.

One JSONL row per phase, appended to --out; the row carries the engine's own admission settings so
runs at different budgets can sit in one file.
"""
import argparse, json, statistics, sys, threading, time, urllib.request


def filler(seed, lines):
    # about 18.7 tokens a line; the first line differs per seed, so no two prompts share a prefix
    return "\n".join(f"Ledger {seed:05d} record {i:06d}: account {(i * 7919 + seed) % 100000:05d} posted {(i * 31 + seed) % 9973} units."
                     for i in range(lines))


def stream(base, body, deadline, timeout=900):
    """One streamed request. Returns (ttft_s, [gap_s between content chunks], usage, error)."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, last, ttft, gaps, usage = time.time(), None, None, [], {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                if ch and (ch[0].get("delta") or {}).get("content"):
                    now = time.time()
                    if ttft is None:
                        ttft = now - t0
                    else:
                        gaps.append(now - last)
                    last = now
                    if deadline and now > deadline:
                        break       # closing the socket cancels the request
    except Exception as e:
        return ttft, gaps, usage, repr(e)[:200]
    return ttft, gaps, usage, None


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def admission(m):
    """The engine's admission settings, wherever /metrics carries them."""
    for v in m.values():
        if isinstance(v, dict) and isinstance(v.get("admission"), dict):
            return v["admission"]
    return m.get("admission") or {}


def phase(a, name, writers, readers):
    stop = time.time() + a.seconds
    lock = threading.Lock()
    W = {"chunks": 0, "gaps": [], "ttft": [], "err": 0, "tokens": 0}
    R = {"n": 0, "prompt_tokens": 0, "ttft": [], "err": 0}
    seed = [int(time.time()) % 50000 * 10]

    def writer(i):
        k = 0
        while time.time() < stop:
            k += 1
            body = {"model": a.model, "max_tokens": a.answer_tokens, "temperature": 0.7,
                    "messages": [{"role": "user", "content": f"Write a long, detailed technical essay (writer {i}, part {k}) on the history "
                                  "of database storage engines. Keep writing section after section."}]}
            ttft, gaps, usage, e = stream(a.base, body, stop)
            with lock:
                W["err"] += e is not None
                W["gaps"] += gaps
                W["chunks"] += len(gaps) + (ttft is not None)
                W["tokens"] += (usage or {}).get("completion_tokens") or 0
                if ttft is not None:
                    W["ttft"].append(ttft)

    def reader(i):
        while time.time() < stop:
            with lock:
                seed[0] += 1
                s = seed[0]
            # the answer is discarded (this client only measures the read-in), so its limit is small on purpose
            body = {"model": a.model, "max_tokens": 8, "temperature": 0,
                    "messages": [{"role": "user", "content": filler(s, a.lines) + "\n\nReply with the single word: done."}]}
            ttft, _, usage, e = stream(a.base, body, None)
            with lock:
                R["err"] += e is not None
                if usage:
                    R["n"] += 1
                    R["prompt_tokens"] += usage.get("prompt_tokens") or 0
                if ttft is not None:
                    R["ttft"].append(ttft)

    m0 = json.load(urllib.request.urlopen(a.base + "/metrics", timeout=10))
    t0 = time.time()
    ts = [threading.Thread(target=writer, args=(i,)) for i in range(writers)] + \
         [threading.Thread(target=reader, args=(i,)) for i in range(readers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    m1 = json.load(urllib.request.urlopen(a.base + "/metrics", timeout=10))
    out_tokens = m1["service"]["tokens_out"] - m0["service"]["tokens_out"]
    row = {"label": a.label, "phase": name, "writers": writers, "readers": readers, "wall_s": round(wall, 1),
           "engine_out_tok_s": round(out_tokens / wall, 1),
           "per_writer_tok_s": round(out_tokens / wall / max(writers, 1), 1),
           "gap_ms_p50": pct(W["gaps"], 0.5) and round(pct(W["gaps"], 0.5) * 1000),
           "gap_ms_p95": pct(W["gaps"], 0.95) and round(pct(W["gaps"], 0.95) * 1000),
           "gap_ms_max": W["gaps"] and round(max(W["gaps"]) * 1000),
           "writer_ttft_ms_p50": W["ttft"] and round(statistics.median(W["ttft"]) * 1000),
           "reader_requests": R["n"], "reader_prompt_tok_s": round(R["prompt_tokens"] / wall),
           "reader_prompt_tokens_each": R["n"] and R["prompt_tokens"] // R["n"],
           "reader_ttft_s_p50": R["ttft"] and round(statistics.median(R["ttft"]), 2),
           "errors": W["err"] + R["err"],
           "prefill_budget_tokens": admission(m1).get("prefill_budget_tokens"),
           "prefill_idle_budget_tokens": admission(m1).get("prefill_idle_budget_tokens")}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--writers", type=int, default=4)
    ap.add_argument("--readers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=640, help="lines per long prompt (about 19 tokens each; 640 is about 12k tokens)")
    ap.add_argument("--answer-tokens", type=int, default=65536, help="a runaway stop only: the phase's deadline ends the stream")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--phases", default="writers_alone,readers_alone,mixed",
                    help="which phases to run; writers_alone under someone else's load measures what a person chatting sees")
    a = ap.parse_args()
    every = {"writers_alone": (a.writers, 0), "readers_alone": (0, a.readers), "mixed": (a.writers, a.readers)}
    phases = [(n, *every[n]) for n in a.phases.split(",")]
    t0, err = time.time(), 0
    print(f"[1/1] 0/{len(phases)} err=0 elapsed=0s", flush=True)
    with open(a.out, "a") as out:
        for i, (name, w, r) in enumerate(phases, 1):
            row = phase(a, name, w, r)
            err += row["errors"]
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[1/1] {i}/{len(phases)} err={err} elapsed={int(time.time() - t0)}s :: {json.dumps(row)}", flush=True)
            time.sleep(3)
    print("DONE", flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
