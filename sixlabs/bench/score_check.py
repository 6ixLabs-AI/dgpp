#!/usr/bin/env python3
"""score_check.py — is POST /v1/score the model's teacher-forced log-probability?

  python3 score_check.py --base http://172.17.0.1:18090 --model Qwen3-Next-80B \
      --ids-json ids/echo_prose_en.json --expected eng_fp8_echo_prose_en.txt

Four checks, each a PASS/FAIL line:
  1  the tail's rows are the right rows: a continuation scored in one call against each of its
     tokens scored alone (prompt grown by one token, one-token continuation — the ordinary
     last-row path every request's first token takes). A row off by one would be off by whole
     nats and uncorrelated. They are NOT bit-equal: this engine's numbers move a little with
     the shape of the chunk a row was read in (measured 2026-10-04: about 0.05 nat a token,
     0.4 on the odd one), so the bound is correlation and mean difference. The same request
     sent twice must be identical.
  2  against the forward check: `--expected` is qwen35_forward_check output for the same ids
     (run it with the serving dense format, --dense-weights fp8). Its "token logprobs:" line is
     log P(ids[t+1] | ids[..t]) for every t; /v1/score over (ids[:P], ids[P:P+C]) is entries
     P-1 .. P+C-2. A different code path (one-shot walk, no session), so the bound is the
     engine's path-to-path noise: correlation and mean error, with the worst token reported.
  3  against the engine's own generation: a greedy completion with logprobs, then its text
     scored after the same prompt. Same tokens; log-probabilities within decode-vs-read-in noise.
  4  across a chunk boundary: one long sequence scored with two different prompt/continuation
     splits, so the read-in is chunked differently (one split's tail would straddle the
     4096-token grid). The positions both calls score must agree.
The engine must be otherwise idle.
"""
import argparse, json, sys, urllib.request


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:400].decode(errors='replace')}")


def score(a, prompt_ids, cont_ids, top=0):
    r = post(a.base, "/v1/score", {"model": a.model, "prompt_ids": prompt_ids, "continuation_ids": cont_ids, "top_logprobs": top})
    assert r["prompt_tokens"] == len(prompt_ids) and r["continuation_tokens"] == len(cont_ids), r
    assert [t["id"] for t in r["tokens"]] == cont_ids, "the scored ids are not the ids sent"
    return [t["logprob"] for t in r["tokens"]], r


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    c = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx, vy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    return c / (vx * vy) ** 0.5 if vx > 0 and vy > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids-json", required=True)
    ap.add_argument("--filler-json", required=True, help="a second text's ids, the long prompt's filler")
    ap.add_argument("--expected", required=True)
    a = ap.parse_args()
    load = lambda p: (lambda d: d["ids"] if isinstance(d, dict) else d)(json.load(open(p)))
    ids, filler = load(a.ids_json), load(a.filler_json)
    line = [l for l in open(a.expected) if l.startswith("token logprobs:")][0]
    exp = [float(x) for x in line.split(":", 1)[1].split(",")]      # exp[t] = log P(ids[t+1] | ids[..t])
    fails = 0

    print("[1/4] the tail's rows against the same tokens scored one at a time")
    for P, C in [(40, 12), (20, 32)]:
        tail, _ = score(a, ids[:P], ids[P:P + C])
        single = [score(a, ids[:P + i], [ids[P + i]])[0][0] for i in range(C)]
        diffs = [abs(x - y) for x, y in zip(tail, single)]
        r, mean_abs, worst = pearson(tail, single), sum(diffs) / C, max(diffs)
        ok = r >= 0.995 and mean_abs <= 0.1
        fails += not ok
        print(f"   prompt {P} + continuation {C}: sum {sum(tail):.4f} in one call vs {sum(single):.4f} one at a time, "
              f"correlation {r:.4f}, mean |difference| {mean_abs:.4f}, worst {worst:.4f} -> {'PASS' if ok else 'FAIL'}")
        top = sorted(range(C), key=lambda i: -diffs[i])[:3]
        print("      largest three: " + "; ".join(f"token {i}: {tail[i]:.3f} vs {single[i]:.3f}" for i in top))
    again, _ = score(a, ids[:20], ids[20:52])
    same = again == tail
    fails += not same
    print(f"   the same request twice: {'identical' if same else 'DIFFERENT, worst %.4f' % max(abs(x - y) for x, y in zip(again, tail))}"
          f" -> {'PASS' if same else 'FAIL'}")

    print(f"[2/4] against the forward check ({len(ids)} tokens; a different code path)")
    for P, C in [(40, 12), (20, 32), (60, 15), (1, 8)]:
        C = min(C, len(ids) - P)
        got, _ = score(a, ids[:P], ids[P:P + C])
        want = exp[P - 1:P - 1 + C]
        diffs = [g - w for g, w in zip(got, want)]
        r, mean_abs, worst = pearson(got, want), sum(map(abs, diffs)) / C, max(map(abs, diffs))
        ok = r >= 0.99 and mean_abs <= 0.25
        fails += not ok
        print(f"   prompt {P:3d} + continuation {C:2d}: sum {sum(got):9.4f} vs {sum(want):9.4f}, correlation {r:.4f}, "
              f"mean |error| {mean_abs:.4f}, worst {worst:.4f} -> {'PASS' if ok else 'FAIL'}")
        if P == 1:
            print("      per token (scored vs forward check): " + "; ".join(f"{g:.2f} vs {w:.2f}" for g, w in zip(got, want)))
    got, r = score(a, ids[:40], ids[40:44], top=3)
    alts = [len(t["top_logprobs"]) for t in r["tokens"]]
    best_first = all(t["top_logprobs"][0]["logprob"] >= t["logprob"] - 1e-6 for t in r["tokens"])
    ok = alts == [3, 3, 3, 3] and best_first
    fails += not ok
    print(f"   alternatives: {alts} per token, the first never below the scored token: {best_first} -> {'PASS' if ok else 'FAIL'}")

    print("[3/4] against the engine's own greedy generation (decode path against read-in path)")
    for prompt in ["The three primary colours are red, yellow and", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"]:
        g = post(a.base, "/v1/completions", {"model": a.model, "prompt": prompt, "max_tokens": 16, "temperature": 0, "logprobs": 1})
        ch = g["choices"][0]
        gen_lp = ch["logprobs"]["token_logprobs"]
        s = post(a.base, "/v1/score", {"model": a.model, "prompt": prompt, "continuation": ch["text"]})
        sc_lp = [t["logprob"] for t in s["tokens"]]
        same_tokens = [t["token"] for t in s["tokens"]] == ch["logprobs"]["tokens"]
        worst = max(abs(x - y) for x, y in zip(gen_lp, sc_lp)) if len(gen_lp) == len(sc_lp) else float("inf")
        ok = same_tokens and worst <= 0.2
        fails += not ok
        print(f"   generated {ch['text'][:50]!r}: same tokens {same_tokens}, sum {sum(gen_lp):.4f} generated vs "
              f"{sum(sc_lp):.4f} scored, worst token off by {worst:.4f} -> {'PASS' if ok else 'FAIL'}")

    print("[4/4] across a chunk boundary")
    # 4071 tokens of another text, then this text for the first time: positions 4091-4102 are its
    # tokens 21-32, which nothing earlier in the prompt gives away.
    long = [ids[0]] + (filler[1:] * 60)[:4070] + ids[1:61]
    a_lp, _ = score(a, long[:4091], long[4091:4103])      # a 12-row tail the 4096 grid line would cut
    b_lp, _ = score(a, long[:4097], long[4097:4103])      # the same last six positions, read after the line
    worst = max(abs(x - y) for x, y in zip(a_lp[6:], b_lp))
    ok = worst <= 0.2 and min(b_lp) < -0.05
    fails += not ok
    print(f"   positions 4097-4102, in a 12-row tail across the 4096 line vs a 6-row tail after it: "
          f"{[round(x, 3) for x in a_lp[6:]]} vs {[round(x, 3) for x in b_lp]}, worst off by {worst:.4f} -> {'PASS' if ok else 'FAIL'}")
    print("RESULT:", "FAIL" if fails else "ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
