#!/usr/bin/env python3
"""deltaui_ab.py — run real questions through the ΔUI app's own pipeline, mode against mode.

  python3 deltaui_ab.py --modes sixserve,expert --out deltaui_ab.jsonl

It calls the app's local Ensemble API (127.0.0.1:8110, bearer token from the app's config
directory), so every run is the pipeline the visible app runs — keywords, corpus retrieval, web and
research, the answer, GRIFF/HIFF scoring — in the app's hidden API webview, without touching the
visible thread. One JSONL row per (prompt, mode), flushed as it lands; rows already present are
skipped on a re-run.

The app must be running with its API switched on (Connections page). A mode whose local model is
not loaded answers 409 and the row records that.
"""
import argparse, json, os, sys, time, urllib.request

PROMPTS = [
    ("rag_specdec", "What is speculative decoding and when does it actually improve LLM serving throughput?"),
    ("rag_orderbook", "What is order book slope and how is it used in high-frequency trading?"),
    ("rag_hawkes", "Explain Hawkes processes and how they are used in high-frequency finance."),
    ("fact_quant", "What is the difference between FP8 and NVFP4 weight quantization, and what does each cost in accuracy?"),
    ("code_rust", "Write a Rust function merge_intervals that merges overlapping closed intervals given as Vec<(i64, i64)>, "
                  "and three unit tests for it."),
    ("code_python", "Write a Python asyncio token-bucket rate limiter class with type hints and a short usage example."),
    ("long_compare", "Compare vLLM, SGLang and TensorRT-LLM for serving mixture-of-experts models: batching, KV cache "
                     "management, quantization support and multi-GPU scaling."),
]
TOKEN_PATHS = ["~/Library/Containers/com.griff.delta/Data/Library/Application Support/com.griff.delta/api_token",
               "~/.config/delta/api_token"]


def token():
    for p in TOKEN_PATHS:
        p = os.path.expanduser(p)
        if os.path.exists(p):
            return open(p).read().strip()
    sys.exit("no api_token file found (is the app's API on?)")


def run(tok, prompt, mode, timeout):
    req = urllib.request.Request("http://127.0.0.1:8110/v1/ensemble",
                                 data=json.dumps({"prompt": prompt, "mode": mode}).encode(),
                                 headers={"authorization": "Bearer " + tok, "content-type": "application/json"})
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}", "wall_s": round(time.time() - t, 1)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:300], "wall_s": round(time.time() - t, 1)}
    a = (r.get("answers") or [{}])[0]
    perf = a.get("perf") or {}
    text = a.get("text") or ""
    return {"wall_s": round(time.time() - t, 1), "model": a.get("model"), "api_base": perf.get("api_base"),
            "cloud_fallback": perf.get("cloud_fallback"), "tok_s": perf.get("tok_s"), "ttft_ms": perf.get("ttft_ms"),
            "griff": a.get("griff"), "hiff": a.get("hiff"), "p_hallu": a.get("p_hallu"), "risk_band": a.get("risk_band"),
            "timings": r.get("timings"), "cites": len(r.get("cites") or []), "words": len(text.split()), "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="sixserve,expert")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    modes = a.modes.split(",")
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
                if "error" not in r:
                    done.add((r["id"], r["mode"]))
            except Exception:
                pass
    tok = token()
    total = len(PROMPTS) * len(modes)
    n, err, t0 = len(done), 0, time.time()
    print(f"[1/1] {n}/{total} err=0 elapsed=0s", flush=True)
    with open(a.out, "a") as out:
        for pid, prompt in PROMPTS:
            for mode in modes:
                if (pid, mode) in done:
                    continue
                row = {"id": pid, "mode": mode, "prompt": prompt}
                row.update(run(tok, prompt, mode, a.timeout))
                err += "error" in row
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                n += 1
                brief = row.get("error") or f"{row['model']} ttft {row['ttft_ms']}ms {row['tok_s'] and round(row['tok_s'], 1)} tok/s"
                print(f"[1/1] {n}/{total} err={err} elapsed={int(time.time() - t0)}s :: {pid} {mode}: {brief}", flush=True)
    print("DONE", flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
