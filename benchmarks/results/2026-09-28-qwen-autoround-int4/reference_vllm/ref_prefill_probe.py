#!/usr/bin/env python3
"""Cold prefill through the reference server: the same prompts as scripts/serve_prefill_probe.py (GSM8K prose,
seeded, a distinct nonce per request, thinking off), max_tokens 1, temperature 0, streamed. Reports the visible
TTFT (prompt tokens / TTFT = the smoke-test's definition) and the server's per-step counters around the request.
Usage: ref_prefill_probe.py HOST PORT LEN... [--repeat N] [--seed S] [--tag T] [--json-out F]"""
import argparse, hashlib, http.client, json, random, statistics, sys, time
from pathlib import Path
sys.path.insert(0, '/home/stephen/workspace/dgpp-autoround/scripts')
from data_paths import data_dir, require_file

def metrics(host, port):
    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.request("GET", "/metrics"); r = conn.getresponse(); body = r.read().decode(); conn.close()
    out = {}
    for line in body.splitlines():
        if line.startswith("vllm:prompt_tokens_total") or line.startswith("vllm:scheduled_ctx_tokens_total") \
           or line.startswith("vllm:scheduled_iterations_total") or line.startswith("vllm:num_requests_running"):
            k, v = line.rsplit(" ", 1); out[k.split("{")[0]] = float(v)
    return out

def prompt_of(words, n_words, nonce, rng):
    start = rng.randrange(0, max(1, len(words) - n_words - 1))
    return f"Reference {nonce}. Summarize the following in one sentence.\n\n" + " ".join(words[start:start + n_words])

def ask(host, port, model, prompt):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
               "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    conn = http.client.HTTPConnection(host, port, timeout=1800)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(payload), {"Content-Type": "application/json"})
    r = conn.getresponse()
    if r.status != 200: raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
    ttft = None; usage = None
    while True:
        line = r.readline()
        if not line: break
        line = line.decode().strip()
        if not line.startswith("data:"): continue
        data = line[5:].strip()
        if data == "[DONE]": break
        obj = json.loads(data)
        if ttft is None and obj.get("choices") and (obj["choices"][0].get("delta") or {}).get("content") is not None:
            ttft = time.perf_counter() - t0
        if ttft is None and obj.get("choices") and obj["choices"][0].get("finish_reason"):
            ttft = time.perf_counter() - t0
        if obj.get("usage"): usage = obj["usage"]
    wall = time.perf_counter() - t0; conn.close()
    return {"ttft_s": ttft if ttft is not None else wall, "wall_s": wall, "usage": usage}

def main():
    p = argparse.ArgumentParser(); p.add_argument("host"); p.add_argument("port", type=int)
    p.add_argument("lengths", type=int, nargs="*", default=[2048, 8192, 32768]); p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--seed", type=int, default=7); p.add_argument("--tag", default=""); p.add_argument("--json-out")
    p.add_argument("--model", default="qwen"); a = p.parse_args()
    raw = require_file(data_dir() / "gsm8k_test.jsonl").read_bytes()
    words = [w for line in raw.splitlines() for w in json.loads(line)["question"].split()]
    rng = random.Random(a.seed)
    nonce = lambda v: f"{a.tag}-{v}" if a.tag else v
    cal = ask(a.host, a.port, a.model, prompt_of(words, 400, nonce("cal-0"), rng))
    per_word = (cal["usage"]["prompt_tokens"] - 30) / 400
    print(f"[calibrate] 400 words -> {cal['usage']['prompt_tokens']} prompt tokens ({per_word:.3f} tok/word)", flush=True)
    report = {"model": a.model, "seed": a.seed, "tag": a.tag, "data_sha256": hashlib.sha256(raw).hexdigest(), "samples": []}
    for length in a.lengths:
        n_words = max(8, int((length - 30) / per_word)); samples = []
        for rep in range(a.repeat):
            prompt = prompt_of(words, n_words, nonce(f"n{length}-r{rep}-{rng.randrange(1 << 30)}"), rng)
            m0 = metrics(a.host, a.port); resp = ask(a.host, a.port, a.model, prompt); time.sleep(0.5); m1 = metrics(a.host, a.port)
            pt = resp["usage"]["prompt_tokens"]; cached = (resp["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            steps = m1.get("vllm:scheduled_iterations_total", 0) - m0.get("vllm:scheduled_iterations_total", 0)
            ctx = m1.get("vllm:scheduled_ctx_tokens_total", 0) - m0.get("vllm:scheduled_ctx_tokens_total", 0)
            s = {"requested_tokens": length, "repeat": rep, "prompt_tokens": pt, "cached_tokens": cached,
                 "ttft_s": resp["ttft_s"], "wall_s": resp["wall_s"], "tok_per_s_ttft": pt / resp["ttft_s"],
                 "ms_per_token_ttft": 1000 * resp["ttft_s"] / pt, "steps": steps, "ctx_tokens_scheduled": ctx}
            samples.append(s); report["samples"].append(s)
            if a.json_out: Path(a.json_out).write_text(json.dumps(report, indent=2) + "\n")
            print(f"[n{length} r{rep}] prompt {pt} tok (cached {cached}) ttft {resp['ttft_s']*1000:.0f} ms = {s['ms_per_token_ttft']:.2f} ms/token "
                  f"({s['tok_per_s_ttft']:.0f} tok/s); wall {resp['wall_s']*1000:.0f} ms; {steps:.0f} steps, {ctx:.0f} ctx tokens", flush=True)
        print(f"[n{length}] median: {statistics.median(x['ms_per_token_ttft'] for x in samples):.2f} ms/token; "
              f"ttft {statistics.median(x['ttft_s'] for x in samples)*1000:.0f} ms", flush=True)
main()
