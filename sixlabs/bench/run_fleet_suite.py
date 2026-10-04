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

--api-key-env NAME: the engine is a keyed fleet slot (vLLM --api-key). The key is read from the
environment variable NAME and reaches each step without ever being on a command line (argv shows
in `ps`, in the Jobs view and in this log) or in this log's text: DecodeBench takes it in its
signed request body, llama-benchy gets it appended in-process, tool-eval-bench reads
TOOL_EVAL_API_KEY. ~/run-sharegpt-bench.sh has no way to send a key (its own /v1/models probe is
unauthenticated), so against a keyed slot that step is reported as skipped, not run.
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
    ap.add_argument("--api-key-env", default="", metavar="NAME",
                    help="environment variable holding the endpoint's bearer key (default: no auth)")
    a = ap.parse_args()
    key = os.environ.get(a.api_key_env, "") if a.api_key_env else ""
    if a.api_key_env and not key:
        sys.exit(f"--api-key-env {a.api_key_env}: that variable is unset or empty")

    def scrub(text):
        return text.replace(key, "<key>") if key else text
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
        print("    $ " + scrub(" ".join(cmd)), flush=True)
        if not key:
            return subprocess.call(cmd, env={**os.environ, **(env or {})}, stdout=sys.stdout, stderr=subprocess.STDOUT)
        # A keyed run: the child's output passes through here so a tool that echoes its
        # configuration cannot put the key in this log.
        p = subprocess.Popen(cmd, env={**os.environ, **(env or {})}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace")
        for out_line in p.stdout:
            sys.stdout.write(scrub(out_line))
            sys.stdout.flush()
        return p.wait()

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
                if key:
                    body["api_key"] = key
                r = None
                for tele in ("http://172.18.0.1:7742", "http://127.0.0.1:7742"):
                    try:
                        p, h, data = internal_sign.signed("POST", "/decodebench/run", body)
                        r = json.load(urllib.request.urlopen(urllib.request.Request(tele + p, data, h), timeout=3600))
                        break
                    except OSError as e:
                        print(scrub(f"    {tele}: {e!r}")[:200], flush=True)
                if not r or not r.get("ok", True) or r.get("error"):
                    rc = 1
                    print("    decodebench:", scrub(json.dumps(r))[:600], flush=True)
                else:
                    print("    run_id", r.get("run_id"), flush=True)
                    for w in r.get("waves") or r.get("results") or []:
                        # the wave's own field names (agent/decodebench.py); the first version of this
                        # line asked for names the service never returns and printed no rate at all
                        print("    " + json.dumps({k: w.get(k) for k in ("concurrency", "aggregate_tok_s", "per_stream_median_tok_s",
                                                                         "ttft_median_s", "accept_rate", "accept_length", "ok", "failed",
                                                                         "errors") if k in w}), flush=True)
            elif step == "llama-benchy":
                out = os.path.join(HOME, f"llama-benchy-{stamp}.json")
                args = ["--base-url", base + "/v1", "--model", a.model,
                        "--pp", "2048", "--tg", "128", "--depth", "0", "4096", "16384",
                        "--concurrency", *[str(c) for c in levels if c <= 8], "--runs", "3",
                        "--latency-mode", "generation", "--format", "json", "--save-result", out]
                venv = os.path.join(HOME, "llama-benchy-venv", "bin")
                if key:
                    # llama-benchy takes its key only as --api-key: append it inside the process
                    # (the same entry point, the same arguments) so it is never in this argv.
                    launch = ("import os, sys; from llama_benchy.__main__ import main; "
                              "sys.argv = ['llama-benchy'] + sys.argv[1:] + ['--api-key', os.environ['BENCH_API_KEY']]; "
                              "sys.exit(main())")
                    rc = sh([os.path.join(venv, "python"), "-c", launch, *args], {"BENCH_API_KEY": key})
                else:
                    rc = sh([os.path.join(venv, "llama-benchy"), *args])
                print("    saved", out, flush=True)
            elif step == "sharegpt":
                if key:
                    rc = 1
                    print("    sharegpt: SKIPPED — ~/run-sharegpt-bench.sh cannot send a key (its /v1/models probe "
                          "and its harness call are unauthenticated), and this endpoint requires one", flush=True)
                else:
                    rc = sh([os.path.join(HOME, "run-sharegpt-bench.sh")],
                            {"PORT": str(a.port), "HOST": a.host, "MODEL": a.model, "MAXC": str(a.maxc),
                             "CONCS": " ".join(str(c) for c in levels)})
            else:
                work = os.path.join(HOME, "tool-eval-bench")
                os.makedirs(work, exist_ok=True)
                rc = sh(["docker", "run", "--rm", "--network", "host", "--user", f"{os.getuid()}:{os.getgid()}",
                         "--name", f"tool-eval-bench-{time.strftime('%H%M%S')}", "-v", f"{work}:/work", "-e", "HOME=/work",
                         "-e", "TZ=America/New_York", "-v", "/etc/localtime:/etc/localtime:ro",
                         *(["-e", "TOOL_EVAL_API_KEY"] if key else []), TEB_IMAGE,
                         "run", "--base-url", base, "--seed", "42", "--label", f"{os.uname().nodename}-{a.host}-port{a.port}",
                         "--json-file", f"/work/report-{time.strftime('%Y%m%d-%H%M%S')}.json", "--output-dir", "/work", "--no-live",
                         "--model", a.model, "--short" if step == "teb-short" else "--hardmode"],
                        {"TOOL_EVAL_API_KEY": key} if key else None)
        except Exception as e:
            rc = 1
            print(scrub(f"    {step} raised {e!r}")[:400], flush=True)
        err += rc != 0
        line(i, 1, f":: {step} {'done' if rc == 0 else 'FAILED rc=%s' % rc}")
    print("DONE" if not err else "DONE with errors", flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
