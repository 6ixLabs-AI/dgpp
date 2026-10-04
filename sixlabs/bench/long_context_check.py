#!/usr/bin/env python3
"""long_context_check.py — planted-fact retrieval at chosen prompt lengths, against one engine.

  python3 long_context_check.py --base http://172.17.0.1:18090 --model Qwen3-Next-80B \
      --tokens 300000,400000,500000 --out long_yarn.jsonl

Builds the reference set's log document (sixlabs/refset/gen_prompts.py: filler entries with five
"vault code" lines at 5 / 27 / 50 / 73 / 95 % of the text), asks for the five codes, and counts
how many come back. One JSONL row per length, written as it lands; lengths already in --out are
skipped. --tokens is the PROMPT size to aim for (the generator's lines run about 6 % over its own
estimate on this tokenizer, which is allowed for here).
"""
import argparse, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "refset"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_prompts import long_doc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", required=True, help="comma list of prompt sizes to aim for")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=7200)
    a = ap.parse_args()
    sizes = [int(x) for x in a.tokens.split(",")]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
                if "error" not in r:
                    done.add(r["target"])
            except Exception:
                pass
    t00, err, n = time.time(), 0, len(done)
    print(f"[1/1] {n}/{len(sizes)} err=0 elapsed=0s", flush=True)
    with open(a.out, "a") as out:
        for target in sizes:
            if target in done:
                continue
            doc, facts = long_doc(int(target / 1.06), seed=target)
            q = ("Below is a log. Read it, then answer the question after it.\n\n" + doc +
                 "\n\nQuestion: list the vault code for each of these rooms, one per line, as "
                 "'room: code': " + ", ".join(nm for nm, _ in facts) + ".")
            body = {"model": a.model, "messages": [{"role": "user", "content": q}], "max_tokens": 16384,
                    "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
            row = {"target": target, "chars": len(q), "expect": [c for _, c in facts]}
            t0 = time.time()
            try:
                req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                text, ttft, usage = [], None, {}
                with urllib.request.urlopen(req, timeout=a.timeout) as r:
                    for raw in r:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        d = json.loads(line[5:])
                        if d.get("usage"):
                            usage = d["usage"]
                        for ch in d.get("choices") or []:
                            piece = (ch.get("delta") or {}).get("content")
                            if piece:
                                if ttft is None:
                                    ttft = time.time() - t0
                                text.append(piece)
                answer = "".join(text)
                wall = time.time() - t0
                found = [c for _, c in facts if c in answer]
                pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
                row.update({"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": ttft and round(ttft, 1),
                            "wall_s": round(wall, 1), "found": len(found), "of": len(facts),
                            "found_by_depth": [c in answer for _, c in facts],
                            "prefill_tok_s": pt and ttft and round(pt / ttft),
                            "decode_tok_s": ct and ttft and wall > ttft and round((ct - 1) / (wall - ttft), 1),
                            "answer": answer[:600]})
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"[:400]
                if hasattr(e, "read"):
                    try:
                        row["error"] += " | " + e.read()[:300].decode("utf-8", "replace")
                    except Exception:
                        pass
                err += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n += 1
            brief = row.get("error") or (f"{row['prompt_tokens']} prompt tokens, {row['found']}/{row['of']} facts "
                                         f"{row['found_by_depth']}, first token {row['ttft_s']} s, "
                                         f"prefill {row['prefill_tok_s']} tok/s, decode {row['decode_tok_s']} tok/s")
            print(f"[1/1] {n}/{len(sizes)} err={err} elapsed={int(time.time() - t00)}s :: {target}: {brief}", flush=True)
    print("DONE", flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
