#!/usr/bin/env python3
"""run_35b_prefill_compare.py — the 35B on 6ix.cpp with the default prefill settings, then with the
tuned ones, same build, same checks: group_check, mixed_load, ShareGPT.

  cd ~/6ixinfer-ports && setsid python3 /home/mark/6ixinfer-out/run_35b_prefill_compare.py \
      > /home/mark/6ixinfer-logs/prefill35b-compare.log 2>&1 </dev/null &

Stops before measuring the tuned settings' speed if their correctness check fails. Takes the
engine down at the end. Results: ~/6ixinfer-out/prefill35b/{group,mixed}.jsonl and this log.
"""
import glob, json, os, subprocess, sys, time, urllib.request

REPO = os.path.expanduser("~/6ixinfer-ports")
OUT = os.path.expanduser("~/6ixinfer-out/prefill35b")
BASE, MODEL, TOK = "http://172.17.0.1:18190", "Qwen3.6-35B-A3B", "nvidia/Qwen3.6-35B-A3B-NVFP4"
DEFAULT = os.path.join(REPO, "deploy/cluster_qwen3.6-35b-a3b_nvfp4_w1.json")
TUNED = os.path.join(REPO, "deploy/cluster_qwen3.6-35b-a3b_nvfp4_w1_tunedprefill.json")
KEYS = {"prefill_budget_tokens": 1024, "prefill_idle_budget_tokens": 4096, "decode_passes_per_prefill": 8,
        "prefill_order": "shortest"}
STEPS = ["up default", "smoke default", "group_check default", "mixed_load default", "sharegpt default", "down default",
         "up tuned", "smoke tuned", "group_check tuned", "mixed_load tuned", "sharegpt tuned", "down tuned"]
t0, err, done = time.time(), 0, 0


def beat(msg):
    print(f"[{done + 1 if done < len(STEPS) else done}/{len(STEPS)}] {done}/{len(STEPS)} err={err} elapsed={int(time.time() - t0)}s :: {msg}",
          flush=True)


def run(cmd, env=None, cwd=REPO):
    print("    $ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=cwd, env={**os.environ, **(env or {})}, stdout=sys.stdout, stderr=subprocess.STDOUT)


def serve_log():
    logs = sorted(glob.glob(os.path.join(REPO, "log/deployments/*/serve_r0.log")), key=os.path.getmtime)
    return logs[-1] if logs else None


def up(cfg):
    rc = run(["python3", "scripts/dgpp-cluster", "up", "--config", cfg])
    log = serve_log()
    if log:
        for line in open(log, errors="replace"):
            if "prefill budget" in line or "ERROR" in line or "listening on" in line or "config: model=" in line:
                print("    engine: " + line.strip()[24:300], flush=True)
    mem = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory,name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    print("    device memory: " + "; ".join(l for l in mem.splitlines() if "6ix-Serve" in l), flush=True)
    return rc


def smoke():
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Reply with the single word: done."}], "max_tokens": 12,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=120))
        msg = r["choices"][0]["message"]
        text, usage = msg.get("content") or "", r.get("usage", {})
        print(f"    thinking-off reply: {text!r} reasoning: {bool(msg.get('reasoning_content'))} usage: {usage}", flush=True)
        return 0 if text.strip() else 1
    except Exception as e:                                                    # an HTTP 400 lands here
        detail = e.read().decode()[:300] if hasattr(e, "read") else repr(e)
        print("    smoke failed: " + detail, flush=True)
        return 1


def metrics(tag):
    try:
        m = json.load(urllib.request.urlopen(BASE + "/metrics", timeout=10))
        s = m.get("scheduler", {})
        keep = {k: s[k] for k in s if any(w in k for w in ("mtp", "draft", "accept", "tokens_generated", "prompt_tokens"))}
        print(f"    metrics after {tag}: {json.dumps(keep)[:500]}", flush=True)
    except Exception as e:
        print(f"    metrics unavailable: {e!r}", flush=True)


def phase(label, cfg):
    global err, done
    results = {}
    for name in ("up", "smoke", "group_check", "mixed_load", "sharegpt", "down"):
        step = f"{name} {label}"
        beat(step + " starting")
        if name == "up":
            rc = up(cfg)
        elif name == "smoke":
            rc = smoke()
        elif name == "group_check":
            rc = run(["python3", "sixlabs/bench/group_check.py", "--base", BASE, "--model", MODEL, "--label", label,
                      "--out", os.path.join(OUT, "group.jsonl"), "--thinking-off", "--seed", "4242"])
            rc = 0      # its FAIL counts lookups the model misses by itself; judged below against the other phase
        elif name == "mixed_load":
            rc = run(["python3", "sixlabs/bench/mixed_load.py", "--base", BASE, "--model", MODEL, "--label", label,
                      "--out", os.path.join(OUT, "mixed.jsonl"), "--thinking-off"])
        elif name == "sharegpt":
            rc = run(["python3", "sixlabs/bench/run_fleet_suite.py", "--host", "172.17.0.1", "--port", "18190", "--model", MODEL,
                      "--maxc", "8", "--only", "sharegpt"], env={"TOK": TOK})
            metrics("sharegpt " + label)
        else:
            rc = run(["python3", "scripts/dgpp-cluster", "down", "--config", cfg])
        results[name] = rc
        err += rc != 0
        done += 1
        beat(f"{step} {'done' if rc == 0 else 'FAILED rc=%d' % rc}")
        if rc != 0 and name in ("up", "smoke"):
            # no numbers from an engine that did not come up or cannot turn thinking off
            print(f"    stopping the {label} phase: {name} failed", flush=True)
            if name != "up":
                run(["python3", "scripts/dgpp-cluster", "down", "--config", cfg])
            done = STEPS.index("down " + label) + 1
            return False
    return True


def compare_phases():
    """The same prompts one at a time under the two settings: same text, how far apart the log-probabilities."""
    rows = {}
    try:
        for line in open(os.path.join(OUT, "group.jsonl")):
            r = json.loads(line)
            rows[r["label"]] = r
        d, t = rows["default"], rows["tuned"]
    except Exception as e:
        print(f"    no cross-phase comparison: {e!r}", flush=True)
        return
    for key in ("ref", "alone"):
        same = sum(x["text"] == y["text"] for x, y in zip(d[key], t[key]))
        right_d = sum(x["want"] in x["text"] for x in d[key]); right_t = sum(y["want"] in y["text"] for y in t[key])
        diffs = [abs(a - b) for x, y in zip(d[key], t[key]) if len(x["lps"]) == len(y["lps"]) for a, b in zip(x["lps"], y["lps"])]
        mean = sum(diffs) / len(diffs) if diffs else float("nan")
        print(f"    COMPARE {key}: {same}/{len(d[key])} texts identical between default and tuned; right {right_d} -> {right_t}; "
              f"log-probabilities apart by {mean:.4f} on average (worst {max(diffs) if diffs else float('nan'):.4f}, {len(diffs)} tokens)",
              flush=True)
    for k in ("burst_ttft_s", "burst_ttft_mean_s", "burst_wall_s", "burst_right", "busy_ttft_s", "fails"):
        print(f"    COMPARE {k}: default {d.get(k)}  tuned {t.get(k)}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in ("group.jsonl", "mixed.jsonl"):                      # one row per label per run: start clean
        if os.path.exists(os.path.join(OUT, f)):
            os.replace(os.path.join(OUT, f), os.path.join(OUT, f + ".prev"))
    cfg = json.load(open(DEFAULT))
    assert not any(k in cfg["engine"] for k in KEYS), "the default config already carries prefill keys"
    cfg["engine"].update(KEYS)
    json.dump(cfg, open(TUNED, "w"), indent=2)
    print("tuned config: " + TUNED + " adds " + json.dumps(KEYS), flush=True)
    print(subprocess.run([os.path.join(REPO, "build-release/6ix-Serve"), "--version"], capture_output=True, text=True).stdout.strip(),
          flush=True)
    ok = phase("default", DEFAULT)
    if not ok:
        print("the default phase failed: the tuned phase is not run (there would be nothing valid to compare it with)", flush=True)
    else:
        phase("tuned", TUNED)
        compare_phases()
    beat("finished")
    print("DONE" if err == 0 else "DONE with errors", flush=True)


if __name__ == "__main__":
    main()
