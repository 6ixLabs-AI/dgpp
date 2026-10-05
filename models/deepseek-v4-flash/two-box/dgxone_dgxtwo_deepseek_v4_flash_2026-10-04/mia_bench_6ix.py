#!/usr/bin/env python3
"""mia_bench_6ix.py — MiaAI-Lab's DeepSeek two-Spark sweep, run against an engine with no /tokenize.

The method is scripts/benchmark-0731.py from github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
(MIT): a prompt of "benchmark context datum " repeated to a target length behind a unique first
line, the instruction "Return exactly 128 numbered lowercase English words, then stop.",
temperature 0.6, top_p 0.95, streamed; per request the time to the first token and
output tokens / (finish - first token); per case the medians and the aggregate.

Two differences, both forced by the engine under test:
  - prompt sizing: Mia's script counts tokens with vLLM's /tokenize. Here the tokens per repeated
    unit are measured once from the engine's own usage (two 1-token requests), and the repeat count
    is computed from that. Reported prompt_tokens are the engine's, as in the original.
  - thinking: --thinking off|on sends chat_template_kwargs.thinking (the original relies on the
    server's default and an optional thinking_token_budget).

  setsid python3 /home/mark/6ixinfer-out/mia_bench_6ix.py --base-url http://172.17.0.1:18190/v1 \
      --model deepseek-ai/DeepSeek-V4-Flash-0731 --thinking off,on --out-dir /home/mark/6ixinfer-out/mia-bench \
      --tag 6ixcpp-w2 > /home/mark/6ixinfer-logs/mia-bench-6ixcpp-w2.log 2>&1 </dev/null &
"""
import argparse, asyncio, json, math, os, statistics, sys, time, urllib.error, urllib.request

UNIT = "benchmark context datum "
INSTRUCTION = "\nReturn exactly 128 numbered lowercase English words, then stop."


def post(url, body, timeout=3600):
    for attempt in range(4):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}") from None
        except urllib.error.URLError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def usage_tokens(a, text):
    """The engine's prompt-token count for this user text (one generated token, thinking off)."""
    r = post(f"{a.base_url}/chat/completions", {"model": a.model, "messages": [{"role": "user", "content": text}], "max_tokens": 1,
                                                 "temperature": 0, "chat_template_kwargs": {"thinking": False}})
    return r["usage"]["prompt_tokens"]


def calibrate(a):
    lo = usage_tokens(a, "unique request cal-a " + UNIT * 50)
    hi = usage_tokens(a, "unique request cal-b " + UNIT * 150)
    per_unit = (hi - lo) / 100
    wrapper = lo - 50 * per_unit          # the chat template plus the unique first words
    return per_unit, wrapper


def build_prompt(target, nonce, per_unit):
    # the original grows the text until /tokenize says count >= target; the first words are a few tokens
    return f"unique request {nonce} " + UNIT * max(1, math.ceil(target / per_unit))


def stream_one(a, prompt, thinking):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt + INSTRUCTION}], "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0.6, "top_p": 0.95, "max_tokens": a.max_tokens,
            "chat_template_kwargs": {"thinking": thinking}}
    req = urllib.request.Request(f"{a.base_url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started, first, usage, finish, reasoning_chars, content_chars = time.perf_counter(), None, None, None, 0, 0
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if choices and choices[0].get("finish_reason"):
                finish = choices[0]["finish_reason"]
            r = delta.get("reasoning") or delta.get("reasoning_content") or ""
            c = delta.get("content") or ""
            if first is None and (r or c):
                first = time.perf_counter()
            reasoning_chars += len(r)
            content_chars += len(c)
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    usage = usage or {}
    out = usage.get("completion_tokens", 0)
    ttft = (first or finished) - started
    return {"ttft_s": ttft, "elapsed_s": finished - started, "prompt_tokens": usage.get("prompt_tokens", 0),
            "prefill_tok_s": usage.get("prompt_tokens", 0) / max(0.001, ttft), "output_tokens": out,
            "output_tok_s": out / max(0.001, finished - (first or finished)), "finish_reason": finish,
            "truncated": finish == "length", "reasoning_chars": reasoning_chars, "content_chars": content_chars,
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}


async def run_case(a, target, concurrency, thinking, per_unit, tag):
    prompts = [build_prompt(target, f"{tag}-p{target}-c{concurrency}-r{i}-{int(time.time())}", per_unit) for i in range(concurrency)]
    started = time.perf_counter()
    results = await asyncio.gather(*[asyncio.to_thread(stream_one, a, p, thinking) for p in prompts])
    elapsed = time.perf_counter() - started
    total = sum(r["output_tokens"] for r in results)
    med = lambda k: statistics.median(r[k] for r in results)
    return {"prompt_target": target, "concurrency": concurrency, "elapsed_s": elapsed, "aggregate_tok_s": total / max(0.001, elapsed),
            "median_ttft_s": med("ttft_s"), "median_prefill_tok_s": med("prefill_tok_s"), "median_output_tok_s": med("output_tok_s"),
            "median_output_tokens": med("output_tokens"), "median_prompt_tokens": med("prompt_tokens"),
            "truncated": sum(r["truncated"] for r in results), "median_reasoning_tokens": statistics.median((r["reasoning_tokens"] or 0) for r in results),
            "requests": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-lengths", default="256,2048,8192")
    ap.add_argument("--concurrency", default="1,2,4,6")
    ap.add_argument("--thinking", default="off,on", help="comma list of off / on")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=1, help="runs per case; the row reports the median run by decode rate")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    lengths = [int(x) for x in a.prompt_lengths.split(",")]
    concs = [int(x) for x in a.concurrency.split(",")]
    modes = a.thinking.split(",")
    per_unit, wrapper = calibrate(a)
    print(f"calibration: {per_unit:.3f} tokens per unit, {wrapper:.1f} tokens of template and first words", flush=True)
    total, done, err, t0 = len(modes) * len(lengths) * len(concs), 0, 0, time.time()
    for mode in modes:
        thinking = mode == "on"
        report = {"model": a.model, "base_url": a.base_url, "max_tokens": a.max_tokens, "thinking": mode, "tag": a.tag,
                  "tokens_per_unit": per_unit, "cases": []}
        print(f"=== thinking {mode}", flush=True)
        print("prompt  c   ttft_s  prefill_tok/s  decode_tok/s  aggregate_tok/s  out_tokens  reasoning_tokens  truncated", flush=True)
        for length in lengths:
            for c in concs:
                try:
                    runs = [asyncio.run(run_case(a, length, c, thinking, per_unit, f"{a.tag}-{mode}-{k}")) for k in range(a.repeats)]
                    case = sorted(runs, key=lambda r: r["median_output_tok_s"])[len(runs) // 2]
                    case["all_decode_tok_s"] = [round(r["median_output_tok_s"], 1) for r in runs]
                    report["cases"].append(case)
                    print(f"{length:6d} {c:2d} {case['median_ttft_s']:8.2f} {case['median_prefill_tok_s']:14.0f} {case['median_output_tok_s']:13.1f} "
                          f"{case['aggregate_tok_s']:16.1f} {case['median_output_tokens']:11.0f} {case['median_reasoning_tokens']:17.0f} {case['truncated']:10d}"
                          + (f"   runs {case['all_decode_tok_s']}" if a.repeats > 1 else ""), flush=True)
                except Exception as e:
                    err += 1
                    print(f"{length:6d} {c:2d}  FAILED: {e!r}"[:300], flush=True)
                done += 1
                print(f"[{min(done + 1, total)}/{total}] {done}/{total} err={err} elapsed={int(time.time() - t0)}s", flush=True)
        json.dump(report, open(os.path.join(a.out_dir, f"mia-bench-{a.tag}-thinking-{mode}.json"), "w"), indent=1)
    print("DONE" if not err else "DONE with errors", flush=True)


if __name__ == "__main__":
    main()
