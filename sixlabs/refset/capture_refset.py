#!/usr/bin/env python3
"""capture_refset.py — record what a serving engine says for a fixed prompt set (2026-10-03).

Written to pin down Qwen3-Next-80B's behaviour on Atlas before the DGPP port exists, and to be run
again, unchanged, against the DGPP build. One JSONL row per prompt, written and flushed as soon as
the engine answers, so a killed run resumes where it stopped (rows already present are skipped).

Row kinds (the `kind` field of each prompt):
  echo        /v1/completions with echo+logprobs: the engine's log-probability for every token of
              a text it is GIVEN (teacher forcing). The cleanest comparison between two engines:
              no sampling, no stop rules, no chat template.
  completion  /v1/completions, greedy continuation of a raw prompt.
  chat        /v1/chat/completions, greedy, with the checkpoint's own chat template.
  tools       chat with a tool list; records the tool calls the engine parsed.

Every row also stores the prompt's token ids from the engine's /tokenize (when it has one), so a
host reference can replay exactly the same ids.

  python3 capture_refset.py --base http://10.77.0.2:8262 --model qwen3-next-80b-atlas \
      --prompts prompts_short.jsonl --out atlas_short.jsonl
"""
import argparse, json, os, sys, time, urllib.request


def post(base, path, body, timeout):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(base, model, text, timeout):
    try:
        d = post(base, "/tokenize", {"model": model, "prompt": text}, timeout)
        return d.get("tokens")
    except Exception:
        return None          # an engine without /tokenize: the row simply has no ids


def run_one(a, p):
    kind, row = p["kind"], {"id": p["id"], "kind": p["kind"]}
    t0 = time.time()
    if kind == "echo":
        row["token_ids"] = tokenize(a.base, a.model, p["text"], a.timeout)
        d = post(a.base, "/v1/completions", {"model": a.model, "prompt": p["text"], "echo": True,
                 "logprobs": a.top, "max_tokens": 1, "temperature": 0}, a.timeout)
        lp = d["choices"][0].get("logprobs") or {}
        row.update(tokens=lp.get("tokens"), token_logprobs=lp.get("token_logprobs"),
                   top_logprobs=lp.get("top_logprobs"))
    elif kind == "completion":
        row["token_ids"] = tokenize(a.base, a.model, p["prompt"], a.timeout)
        d = post(a.base, "/v1/completions", {"model": a.model, "prompt": p["prompt"],
                 "logprobs": a.top, "max_tokens": p.get("max_tokens", 128), "temperature": 0}, a.timeout)
        c = d["choices"][0]
        lp = c.get("logprobs") or {}
        row.update(text=c.get("text"), finish_reason=c.get("finish_reason"), tokens=lp.get("tokens"),
                   token_logprobs=lp.get("token_logprobs"), top_logprobs=lp.get("top_logprobs"))
    elif kind in ("chat", "tools"):
        body = {"model": a.model, "messages": p["messages"], "temperature": 0,
                "max_tokens": p.get("max_tokens", 256), "logprobs": True, "top_logprobs": a.top}
        if kind == "tools":
            body["tools"] = p["tools"]
        d = post(a.base, "/v1/chat/completions", body, a.timeout)
        c = d["choices"][0]
        m = c.get("message") or {}
        row.update(content=m.get("content"), tool_calls=m.get("tool_calls"),
                   finish_reason=c.get("finish_reason"), logprobs=(c.get("logprobs") or {}).get("content"))
    else:
        raise ValueError("unknown kind " + kind)
    row["usage"] = d.get("usage")
    row["seconds"] = round(time.time() - t0, 3)
    if "expect" in p:                     # a planted fact the answer should contain (long-context rows)
        row["expect"] = p["expect"]
        row["found"] = [e for e in p["expect"] if e in (row.get("content") or row.get("text") or "")]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()

    prompts = [json.loads(l) for l in open(a.prompts) if l.strip()]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
                if "error" not in r:
                    done.add(r["id"])
            except Exception:
                pass
    t0, err, n = time.time(), 0, len(done)
    print(f"[1/1] {n}/{len(prompts)} err=0 elapsed=0s (resuming)" if n else
          f"[1/1] 0/{len(prompts)} err=0 elapsed=0s", flush=True)
    with open(a.out, "a") as out:
        for p in prompts:
            if p["id"] in done:
                continue
            try:
                row = run_one(a, p)
            except Exception as e:
                err += 1
                row = {"id": p["id"], "kind": p["kind"], "error": f"{type(e).__name__}: {e}"[:400]}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            n += 1
            print(f"[1/1] {n}/{len(prompts)} err={err} elapsed={int(time.time() - t0)}s", flush=True)
    print("DONE", flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
