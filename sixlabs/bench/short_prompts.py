#!/usr/bin/env python3
"""short_prompts.py — a handful of ordinary prompts, greedy, with the speed of each.

  python3 short_prompts.py --base http://172.17.0.1:18090 --model Qwen3-Next-80B --out short.jsonl

A regression check for a setting that changes the model's arithmetic at every position (YaRN):
run it with the setting on and off and read the two files side by side. Every prompt has a
checkable answer; `ok` is that check, not a judgement of style.
"""
import argparse, json, sys, time, urllib.request

PROMPTS = [
    ("arith", "What is 17 * 23? Reply with the number only.", lambda s: "391" in s),
    ("capital", "What is the capital of Australia? One word.", lambda s: "canberra" in s.lower()),
    ("code", "Write a Python function is_prime(n) that returns True for primes. Code only.", lambda s: "def is_prime" in s),
    ("json", 'Return exactly this JSON and nothing else: {"a": 1, "b": [2, 3]}', lambda s: '"b"' in s and "[2, 3]" in s.replace("[2,3]", "[2, 3]")),
    ("reason", "Tom is taller than Ann. Ann is taller than Raj. Who is the shortest? One word.", lambda s: "raj" in s.lower()),
    ("essay", "Explain in about 150 words how a B-tree differs from an LSM tree.", lambda s: len(s.split()) > 80),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    err, t00 = 0, time.time()
    with open(a.out, "w") as out:
        for i, (pid, prompt, check) in enumerate(PROMPTS, 1):
            body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16384,
                    "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
            req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            t0, ttft, text, usage = time.time(), None, [], {}
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
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
                s, wall = "".join(text), time.time() - t0
                ct = usage.get("completion_tokens") or 0
                row = {"id": pid, "ok": bool(check(s)), "completion_tokens": ct, "ttft_ms": ttft and round(ttft * 1000),
                       "tok_s": round((ct - 1) / (wall - ttft), 1) if ttft and ct > 8 and wall > ttft else None, "answer": s}
            except Exception as e:
                row = {"id": pid, "ok": False, "error": repr(e)[:300]}
            err += not row["ok"]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[1/1] {i}/{len(PROMPTS)} err={err} elapsed={int(time.time() - t00)}s :: {pid}: ok={row['ok']} "
                  f"{row.get('tok_s')} tok/s :: {(row.get('answer') or row.get('error') or '')[:90]!r}", flush=True)
    print("DONE", flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
