#!/usr/bin/env python3
"""Measure cold prefill and repeated decode on deterministic parcel documents."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from qwen_yarn_release_check import Lane, filler
from timed_load import COUNTERS, reconciled_metrics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--lengths", type=int, nargs="+", required=True)
parser.add_argument("--json-out", type=Path, required=True)
args = parser.parse_args()
lane = Lane("http://127.0.0.1:18080")
before = reconciled_metrics(lane.host, lane.port)
calibration = lane.ask("Current benchmark calibration.\n" + "".join(filler(20260922, 256)), 1)
before = reconciled_metrics(lane.host, lane.port, before, [calibration])
tokens_per_record = (calibration["usage"]["prompt_tokens"] - 30) / 256
report = {"model": lane.model, "seed": 20260922, "max_tokens": 256,
          "thinking": "server_default", "calibration": calibration, "samples": []}
for length in args.lengths:
    count = max(16, int((length * 0.995 - 256) / tokens_per_record))
    prompt = (f"Current benchmark document {length}.\n" + "".join(filler(20260922, count)) +
              "\nWrite a numbered list describing the first 200 parcels and their cities. "
              "Continue until you have listed all 200.")
    for repeat in range(3):
        answer = lane.ask(prompt, 256)
        after = reconciled_metrics(lane.host, lane.port, before, [answer])
        delta = {name: after[name] - before[name] for name in COUNTERS}
        usage = answer["usage"]
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        if (delta["prompts_prefilled"] != 1 or delta["prompt_tokens"] != usage["prompt_tokens"] or
                delta["prompt_tokens_computed"] != usage["prompt_tokens"] - cached or
                delta["decode_rows"] != delta["decode_steps"]):
            raise RuntimeError("scheduler counters do not describe the completed request")
        if repeat == 0 and cached:
            raise RuntimeError("cold long-context request unexpectedly used the prefix cache")
        generated = delta["tokens_generated"] - 1
        sample = {"requested_tokens": length, "repeat": repeat, "engine": delta,
                  "engine_tokens_per_s": generated * 1000 / delta["step_ms"],
                  "ms_per_pass": delta["step_ms"] / delta["decode_steps"],
                  "tokens_per_pass": generated / delta["decode_steps"],
                  "answer": answer}
        report["samples"].append(sample)
        args.json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"{length} r{repeat}: {usage['prompt_tokens']} prompt tokens, {cached} cached, "
              f"prefill {delta['prefill_ms'] / 1000:.3f}s, "
              f"decode {sample['engine_tokens_per_s']:.2f} tok/s, "
              f"{sample['ms_per_pass']:.2f} ms/pass", flush=True)
        before = after
