#!/usr/bin/env python3
"""run_fleet_suite.py — the fleet's whole benchmark set against one engine, one after another.

  python3 run_fleet_suite.py --host 172.17.0.1 --port 18090 --model Qwen3-Next-80B --maxc 8

Runs on the box, in the order below, each exactly as the Hub's Bench buttons launch it (same
scripts, same image, same result locations), so the results show in the Hub beside every other
engine's:

  1 decodebench        saih-telemetry /decodebench/run (decode tok/s per concurrency, acceptance)
  2 llama-benchy       ~/llama-benchy-venv (prompt 2048, gen 128, depth 0 / 4096 / 16384)
  3 sharegpt           ~/run-sharegpt-bench.sh (real chat turns, per concurrency)
  4 tool-eval (short)  saih/tool-eval-bench --short
  5 tool-eval (hard)   saih/tool-eval-bench --hardmode

One at a time on purpose: each saturates the engine, and two together measure each other.
"""
import argparse, json, os, subprocess, sys, time, urllib.request

HOME = os.path.expanduser("~")
TEB_IMAGE = "saih/tool-eval-bench:2.6.1-6be685f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--maxc", type=int, default=8)
    ap.add_argument("--only", default="", help="comma list of steps to run (decodebench,llama-benchy,sharegpt,teb-short,teb-hard)")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"
    levels = [c for c in (1, 2, 4, 8, 16) if c <= a.maxc]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    t0, err = time.time(), 0
    steps = ["decodebench", "llama-benchy", "sharegpt", "teb-short", "teb-hard"]
    only = [s for s in a.only.split(",") if s]

    def line(i, done, note=""):
        print(f"[{i}/{len(steps)}] {done}/1 err={err} elapsed={int(time.time() - t0)}s {note}", flush=True)

    def gov_wait(max_s=1500):
        """Hold until the thermal governor is not actively throttling (DecodeBench refuses to run
        under it, and the others would quietly measure the clock cap). Says what it saw."""
        try:
            sys.path.insert(0, os.path.join(HOME, "6ixlabs", "agent"))
            import asyncio
            from routers import baseline
        except Exception as e:
            print(f"    governor: cannot read ({e!r})"[:200], flush=True)
            return
        t = time.time()
        while True:
            st = asyncio.run(baseline._gov_state())
            if not st.get("throttling") or time.time() - t > max_s:
                print(f"    governor: throttling={st.get('throttling')} ceiling={st.get('ceiling_mhz')} MHz "
                      f"after waiting {int(time.time() - t)}s ({st.get('reason')})", flush=True)
                return
            time.sleep(20)

    def sh(cmd, env=None):
        print("    $ " + " ".join(cmd), flush=True)
        return subprocess.call(cmd, env={**os.environ, **(env or {})}, stdout=sys.stdout, stderr=subprocess.STDOUT)

    for i, step in enumerate(steps, 1):
        if only and step not in only:
            continue
        line(i, 0, f":: {step} starting")
        gov_wait()
        rc = 0
        try:
            if step == "decodebench":
                sys.path.insert(0, os.path.join(HOME, "6ixlabs", "agent"))
                import internal_sign
                body = {"base_url": base, "model": a.model, "concurrencies": levels, "max_tokens": 512, "wait": True}
                r = None
                for tele in ("http://172.18.0.1:7742", "http://127.0.0.1:7742"):
                    try:
                        p, h, data = internal_sign.signed("POST", "/decodebench/run", body)
                        r = json.load(urllib.request.urlopen(urllib.request.Request(tele + p, data, h), timeout=3600))
                        break
                    except OSError as e:
                        print(f"    {tele}: {e!r}"[:200], flush=True)
                if not r or not r.get("ok", True) or r.get("error"):
                    rc = 1
                    print("    decodebench:", json.dumps(r)[:600], flush=True)
                else:
                    print("    run_id", r.get("run_id"), flush=True)
                    for w in r.get("waves") or r.get("results") or []:
                        print("    " + json.dumps({k: w.get(k) for k in ("concurrency", "agg_tok_s", "per_stream_tok_s", "accept_rate",
                                                                         "accept_len", "ttft_p50_ms", "errors") if k in w}), flush=True)
            elif step == "llama-benchy":
                out = os.path.join(HOME, f"llama-benchy-{stamp}.json")
                rc = sh([os.path.join(HOME, "llama-benchy-venv", "bin", "llama-benchy"), "--base-url", base + "/v1", "--model", a.model,
                         "--pp", "2048", "--tg", "128", "--depth", "0", "4096", "16384",
                         "--concurrency", *[str(c) for c in levels if c <= 8], "--runs", "3",
                         "--latency-mode", "generation", "--format", "json", "--save-result", out])
                print("    saved", out, flush=True)
            elif step == "sharegpt":
                rc = sh([os.path.join(HOME, "run-sharegpt-bench.sh")],
                        {"PORT": str(a.port), "HOST": a.host, "MODEL": a.model, "MAXC": str(a.maxc),
                         "CONCS": " ".join(str(c) for c in levels)})
            else:
                work = os.path.join(HOME, "tool-eval-bench")
                os.makedirs(work, exist_ok=True)
                rc = sh(["docker", "run", "--rm", "--network", "host", "--user", f"{os.getuid()}:{os.getgid()}",
                         "--name", f"tool-eval-bench-{time.strftime('%H%M%S')}", "-v", f"{work}:/work", "-e", "HOME=/work",
                         "-e", "TZ=America/New_York", "-v", "/etc/localtime:/etc/localtime:ro", TEB_IMAGE,
                         "run", "--base-url", base, "--seed", "42", "--label", f"{os.uname().nodename}-{a.host}-port{a.port}",
                         "--json-file", f"/work/report-{time.strftime('%Y%m%d-%H%M%S')}.json", "--output-dir", "/work", "--no-live",
                         "--model", a.model, "--short" if step == "teb-short" else "--hardmode"])
        except Exception as e:
            rc = 1
            print(f"    {step} raised {e!r}"[:400], flush=True)
        err += rc != 0
        line(i, 1, f":: {step} {'done' if rc == 0 else 'FAILED rc=%s' % rc}")
    print("DONE" if not err else "DONE with errors", flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
