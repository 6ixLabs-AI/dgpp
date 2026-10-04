#!/usr/bin/env python3
"""pool_pressure.py — fill the KV pool with cached prompts, then make a live request grow through it.

  python3 pool_pressure.py --base http://172.17.0.1:18090 --model Qwen3-Next-80B

The failure this reproduces (2026-10-03): the prefix cache's snapshots share the KV pool, and after a
run of long prompts 44 idle entries pinned 4075 of 4096 blocks. A live request then had nowhere to
grow: one answer was shed at 604 tokens, and a later one hit the engine's own reserve on an exact
block boundary and took the server down. Passing means: every long answer finishes at its full
length (finish_reason length at max_tokens, or stop), and the server still answers afterwards.
"""
import argparse, concurrent.futures, json, sys, time, urllib.request


def post(base, body, timeout=900):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def pool(base):
    m = json.load(urllib.request.urlopen(base + "/metrics", timeout=10))["scheduler"]
    return m["pool_blocks_in_use"], m["pool_blocks_total"]


def filler(seed, lines):
    # ~18.7 tokens a line on this tokenizer; the first line differs per seed so no two prompts share a prefix.
    return "\n".join(f"Ledger {seed:03d} record {i:06d}: account {(i * 7919 + seed) % 100000:05d} posted {(i * 31 + seed) % 9973} units."
                     for i in range(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fill", type=int, default=12, help="long prompts sent to fill the pool")
    ap.add_argument("--lines", type=int, default=1700, help="lines per long prompt (about 19 tokens each)")
    ap.add_argument("--long-answers", type=int, default=6)
    ap.add_argument("--answer-tokens", type=int, default=3000)
    ap.add_argument("--block-tokens", type=int, default=64)
    a = ap.parse_args()
    t0, err = time.time(), 0
    total = a.fill + a.long_answers + 1
    done = 0
    for i in range(a.fill):
        r = post(a.base, {"model": a.model, "max_tokens": 8, "temperature": 0,
                          "messages": [{"role": "user", "content": filler(i, a.lines) + "\n\nReply with the single word: done."}]})
        done += 1
        print(f"[1/3] {done}/{total} err={err} elapsed={int(time.time() - t0)}s :: fill {i} prompt_tokens={r['usage']['prompt_tokens']}", flush=True)
    # The long answers run TOGETHER and must need more than the pool has free, or nothing is
    # forced: the first run of this script left 74 blocks free and each lone answer fitted in them.
    in_use, blocks = pool(a.base)
    need = a.long_answers * a.answer_tokens
    free_tokens = (blocks - in_use) * a.block_tokens
    print(f"[2/3] pool {in_use}/{blocks} blocks before the long answers: {free_tokens} tokens free, "
          f"{need} wanted -> eviction {'REQUIRED' if need > free_tokens else 'not required (raise --long-answers)'}", flush=True)

    def long_answer(i):
        return post(a.base, {"model": a.model, "max_tokens": a.answer_tokens, "temperature": 0.7,
                             "messages": [{"role": "user", "content": f"Write a very long, detailed technical essay (number {i}) on the history of "
                                           "database storage engines. Do not stop early; keep writing section after section."}]})

    with concurrent.futures.ThreadPoolExecutor(a.long_answers) as ex:
        futs = {ex.submit(long_answer, i): i for i in range(a.long_answers)}
        for f in concurrent.futures.as_completed(futs):
            i = futs[f]
            try:
                r = f.result()
                c, u = r["choices"][0], r["usage"]
                ok = c["finish_reason"] == "stop" or u["completion_tokens"] >= a.answer_tokens
                note = f"{u['completion_tokens']} tokens, finish_reason {c['finish_reason']} -> {'OK' if ok else 'CUT SHORT'}"
            except Exception as e:
                ok, note = False, "request failed: " + repr(e)[:160]
            err += not ok
            done += 1
            print(f"[2/3] {done}/{total} err={err} elapsed={int(time.time() - t0)}s :: long answer {i}: {note}", flush=True)
    in_use, blocks = pool(a.base)
    print(f"[2/3] pool {in_use}/{blocks} blocks after the long answers", flush=True)
    try:
        r = post(a.base, {"model": a.model, "max_tokens": 8, "temperature": 0, "messages": [{"role": "user", "content": "Say OK."}]}, timeout=60)
        alive = bool(r["choices"])
    except Exception as e:
        alive = False
        print("server did not answer afterwards:", repr(e)[:200], flush=True)
    err += not alive
    done += 1
    print(f"[3/3] {done}/{total} err={err} elapsed={int(time.time() - t0)}s :: server alive afterwards: {alive}", flush=True)
    print("PASS" if not err else "FAIL", flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
