#!/usr/bin/env python3
"""group_check.py — several prompts read in one walk: are the answers still right, and what does it buy?

  python3 group_check.py --base http://172.17.0.1:18090 --model Qwen3-Next-80B --label auto --out group.jsonl

Every prompt is a ledger nobody has sent before plus a lookup question whose answer is one of
its lines, so a prompt read wrongly (another request's rows, a state carried from the wrong
slot) answers wrongly. Greedy, with log-probabilities.

  1  one at a time          the reference answers and their log-probabilities
  2  four at once           equal lengths: the same answers, log-probabilities within the engine's
                            batching noise
  3  four lengths at once   1k / 3k / 6k / 9k tokens: the spans finish in different walks
  4  beside a long answer   three prompts arriving while another request is streaming
  5  a burst of eight       2k-token prompts, 64-token answers: first token of each and the wall
                            time for all of them (the number to compare between engine settings)
Rows 1-4 are PASS/FAIL; row 5 is a measurement. One JSONL row per run in --out.
"""
import argparse, concurrent.futures as cf, json, sys, threading, time, urllib.request


def ledger(seed, lines):
    # about 33 tokens a line on this tokenizer; the first line differs per seed (no shared prefix)
    return "\n".join(f"Ledger {seed:05d} record {i:06d}: account {(i * 7919 + seed) % 100000:05d} posted {(i * 31 + seed) % 9973} units."
                     for i in range(lines))


def question(seed, lines):
    i = (seed * 37) % lines
    want = f"{(i * 7919 + seed) % 100000:05d}"
    text = (ledger(seed, lines) + f"\n\nWhich account does record {i:06d} name? Reply with the five digits only.")
    return text, want


def ask(a, prompt, max_tokens=12, stream_stats=False):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "logprobs": True, "top_logprobs": 1, "stream": True, "stream_options": {"include_usage": True}}
    if a.thinking_off:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, ttft, text, lps, usage = time.time(), None, [], [], {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for ch in d.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    ttft = ttft or time.time() - t0
                    text.append(piece)
                for e in ((ch.get("logprobs") or {}).get("content") or []):
                    lps.append(e["logprob"])
    return {"text": "".join(text), "lps": lps, "ttft": ttft, "wall": time.time() - t0, "prompt_tokens": usage.get("prompt_tokens")}


def together(a, prompts, max_tokens=12):
    with cf.ThreadPoolExecutor(len(prompts)) as ex:
        return list(ex.map(lambda p: ask(a, p, max_tokens), prompts))


def compare(name, got, want_answers, ref=None):
    right = sum(w in g["text"] for g, w in zip(got, want_answers))
    line = f"{right}/{len(got)} answers right"
    ok = right == len(got)
    if ref is not None:
        same = sum(g["text"] == r["text"] for g, r in zip(got, ref))
        diffs = [abs(x - y) for g, r in zip(got, ref) if len(g["lps"]) == len(r["lps"]) for x, y in zip(g["lps"], r["lps"])]
        mean = sum(diffs) / len(diffs) if diffs else float("nan")
        line += f", {same}/{len(got)} identical to the one-at-a-time text, log-probabilities off by {mean:.4f} on average (worst {max(diffs) if diffs else float('nan'):.4f})"
        ok = ok and (not diffs or mean <= 0.15)
    print(f"   {name}: {line} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking-off", action="store_true",
                    help="send chat_template_kwargs.enable_thinking false: a model that reasons by default (Qwen3.5/3.6) "
                         "spends these short answers' token budgets on reasoning otherwise, and every row fails")
    a = ap.parse_args()
    s0 = int(time.time()) % 40000 + 100
    fails, row = 0, {"label": a.label}

    print("[1/5] one at a time", flush=True)
    qs = [question(s0 + k, 90) for k in range(4)]                 # about 3k tokens each
    ref = [ask(a, q) for q, _ in qs]
    fails += not compare("four 3k-token prompts, one at a time", ref, [w for _, w in qs])
    row["prompt_tokens_3k"] = ref[0]["prompt_tokens"]

    print("[2/5] four at once", flush=True)
    got = together(a, [q for q, _ in qs])
    # (the same prompts again: they are no longer fresh, but a 3k prompt is below what the engine caches from)
    fails += not compare("the same four, sent together", got, [w for _, w in qs], ref)

    print("[3/5] four lengths at once", flush=True)
    mixed = [question(s0 + 10 + k, n) for k, n in enumerate((30, 90, 180, 270))]
    alone = [ask(a, q) for q, _ in mixed]
    fails += not compare("1k / 3k / 6k / 9k, one at a time", alone, [w for _, w in mixed])
    mixed2 = [question(s0 + 20 + k, n) for k, n in enumerate((30, 90, 180, 270))]
    got = together(a, [q for q, _ in mixed2])
    fails += not compare("1k / 3k / 6k / 9k fresh prompts, sent together", got, [w for _, w in mixed2])

    print("[4/5] beside a long answer", flush=True)
    done = threading.Event()
    writer = threading.Thread(target=lambda: (ask(a, "Write a long, detailed essay on the history of database storage engines.", 600), done.set()))
    writer.start()
    time.sleep(1.5)
    busy = [question(s0 + 30 + k, 90) for k in range(3)]
    got = together(a, [q for q, _ in busy])
    fails += not compare("three fresh 3k prompts while another request streams", got, [w for _, w in busy])
    row["busy_ttft_s"] = [round(g["ttft"], 2) for g in got]
    writer.join()

    print("[5/5] a burst of eight", flush=True)
    burst = [question(s0 + 40 + k, 60) for k in range(8)]         # about 2k tokens each
    t0 = time.time()
    got = together(a, [q + " Then explain in two sentences what a ledger is." for q, _ in burst], max_tokens=64)
    wall = time.time() - t0
    ttfts = sorted(g["ttft"] for g in got)
    right = sum(w in g["text"] for g, (_, w) in zip(got, burst))
    row.update({"burst_prompt_tokens": got[0]["prompt_tokens"], "burst_ttft_s": [round(t, 2) for t in ttfts],
                "burst_ttft_mean_s": round(sum(ttfts) / 8, 2), "burst_wall_s": round(wall, 2), "burst_right": right})
    print(f"   eight {got[0]['prompt_tokens']}-token prompts: first tokens at {row['burst_ttft_s']} s (mean {row['burst_ttft_mean_s']}), "
          f"all eight finished in {row['burst_wall_s']} s, {right}/8 answers right", flush=True)
    fails += right != 8
    row["fails"] = fails
    open(a.out, "a").write(json.dumps(row) + "\n")
    print("RESULT:", "FAIL" if fails else "ok", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
