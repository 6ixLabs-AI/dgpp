#!/usr/bin/env python3
"""Run the current-code campaign sequentially; retain commands and every result.

Run on rank zero after building the release server. The selected cluster must
be idle. Completed commands are resumable; failed commands remain in the log.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PYTHON = sys.executable
sys.path.insert(0, str(ROOT / "scripts"))
from site_env import resolve_config

TELEMETRY = {}


def start_telemetry(directory, config):
    cfg = resolve_config(str(config))
    if len(cfg['nodes']) != 4:
        return
    management = json.loads((HERE / 'manifest.json').read_text()).get('telemetry', {}).get('node3_management_ssh')

    def ssh_prefix(rank):
        host = cfg['nodes'][rank]
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8']
        if rank == 2 and management:
            host = management['host']
            command += ['-o', 'StrictHostKeyChecking=yes',
                        '-o', 'HostKeyAlias=' + management['host_key_alias']]
        return command + [f"{cfg['ssh_user']}@{host}"]

    running = []
    TELEMETRY[str(directory)] = running
    for rank, host in enumerate(cfg['nodes']):
        output = (directory / f'node{rank}-telemetry.jsonl').open('w')
        if rank == 0:
            command = [PYTHON, str(HERE / 'node_telemetry.py')]
        else:
            command = ssh_prefix(rank) + ['python3', '-']
        proc = subprocess.Popen(command, stdin=subprocess.PIPE if rank else subprocess.DEVNULL,
                                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        running.append((proc, output))
        if rank:
            try:
                proc.stdin.write((HERE / 'node_telemetry.py').read_bytes())
                proc.stdin.close()
            except OSError:
                pass
    output = (directory / 'node2-live-kernel.log').open('w')
    proc = subprocess.Popen(ssh_prefix(2) + ['sudo', '-n', 'journalctl', '-k', '-f',
                                           '-n', '0', '-o', 'short-iso'],
                            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                            start_new_session=True)
    running.append((proc, output))


def stop_telemetry(directory):
    for proc, output in TELEMETRY.pop(str(directory), []):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=3)
            except ProcessLookupError:
                pass
        output.close()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(command, output, timeout=7200, resume=True):
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = output.with_suffix(output.suffix + ".command.json")
    if resume and receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get("returncode") == 0:
            print(f"{now()} SKIP {output.relative_to(HERE)}", flush=True)
            return True
        # Preserve failed/interrupted attempts when a command is retried.
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for path in (output, receipt, output.with_suffix(".json")):
            if path.exists():
                shutil.copy2(path, path.with_name(path.name + ".attempt-" + stamp))
    record = {"command": list(map(str, command)), "cwd": str(ROOT), "started_at": now()}
    save(receipt, record)
    print(f"{now()} RUN {shlex.join(record['command'])}", flush=True)
    with output.open("w") as log:
        try:
            proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout)
            record["returncode"] = proc.returncode
        except subprocess.TimeoutExpired:
            record["returncode"] = 124
    record["finished_at"] = now()
    save(receipt, record)
    print(f"{now()} EXIT {record['returncode']} {output.relative_to(HERE)}", flush=True)
    return record["returncode"] == 0


def client(script, *args):
    return [PYTHON, str(ROOT / "scripts" / script), "127.0.0.1", "18080", *map(str, args)]


def thinking_args(entry):
    # GLM-5.3 templates always reason and reject enable_thinking=false.
    return [] if "GLM-5.3" in entry["model"] else ["--no-think"]


def launch(entry, stage, mode="default", overrides=None):
    directory = HERE / "raw" / entry["id"] / stage / mode
    directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HERE / entry["config"]).read_text())
    if overrides:
        cfg["engine"].update(overrides)
    config = directory / "config.json"
    save(config, cfg)
    server = directory / "server"
    command = [PYTHON, str(ROOT / "scripts/dgpp-cluster"), "up", "--config", str(config),
               "--bin", str(ROOT / "build-release/dgpp-serve"), "--log-dir", str(server)]
    expected = json.loads((HERE / "manifest.json").read_text())["binary_sha256"]
    if hashlib.sha256((ROOT / "build-release/dgpp-serve").read_bytes()).hexdigest() != expected:
        raise RuntimeError("the release binary changed during the campaign")
    start_telemetry(directory, config)
    if not run(command, directory / "up.log", timeout=900, resume=False):
        stop(directory, config, server)
        raise RuntimeError(f"startup failed: {directory}")
    try:
        with urllib.request.urlopen("http://127.0.0.1:18080/v1/models", timeout=30) as response:
            models = json.load(response)
        if models["data"][0]["id"] != entry["model"]:
            raise RuntimeError("unexpected served model")
    except Exception:
        stop(directory, config, server)
        raise
    save(directory / "models.json", models)
    return directory, config, server


def stop(directory, config, server):
    try:
        ok = run([PYTHON, str(ROOT / "scripts/dgpp-cluster"), "down", "--config", str(config),
                  "--log-dir", str(server)], directory / "down.log", timeout=420, resume=False)
    finally:
        stop_telemetry(directory)
    digests = {p.name: hashlib.md5(p.read_bytes()).hexdigest()
               for p in server.glob("serve_rank*.ops")}
    expected = len(resolve_config(str(config))["nodes"])
    save(directory / "rank-identity.json", {"expected_ranks": expected, "md5": digests,
         "passed": len(digests) == expected and len(set(digests.values())) == 1})
    if not ok:
        raise RuntimeError("cluster did not stop cleanly")


def performance(entry):
    marker = HERE / "raw" / entry["id"] / "performance" / "complete.json"
    if marker.exists():
        return
    directory, config, server = launch(entry, "performance")
    passed = {}
    try:
        cs = [1, 2, 4] if entry["slots"] >= 4 else [1, 2]
        if entry["slots"] > 4:
            cs.append(entry["slots"])
        passed["greedy"] = run(client("timed_load.py", "--concurrency", ",".join(map(str, cs)),
            "--classes", "all", "--repeat", 3, "--max-tokens", 256,
            "--json-out", directory / "greedy.json"), directory / "greedy.log")
        passed["prefill"] = run(client("serve_prefill_probe.py", 2048, 8192, 32768,
            "--repeat", 3, *thinking_args(entry), "--seed", 7, "--tag", "dsv4-20261001",
            "--json-out", directory / "prefill.json"), directory / "prefill.log")
        passed["sampled"] = run(client("timed_load.py", "--concurrency", f"1,{entry['slots']}",
            "--classes", "all", "--repeat", 3, "--max-tokens", 256, "--temperature", 1,
            "--json-out", directory / "sampled.json"), directory / "sampled.log")
        passed["isolation"] = run(client("serve_load.py", "--concurrency", 1,
            "--max-tokens", 256, "--isolation", entry["slots"], "--classes", "chat",
            "--json-out", directory / "isolation.json"), directory / "isolation.log")
    finally:
        stop(directory, config, server)
    save(marker, {"finished_at": now(), "checks": passed})


def modes(entry):
    variants = [("plain", {"mtp": False, "mtp_depth": 1, "mtp_schedule": False})]
    if entry["id"].startswith("dsv4"):
        # The template schedules the block's depth: the fixed depths beside it.
        variants.append(("depth5", {"mtp_depth": 5, "mtp_schedule": False}))
        variants.append(("depth3", {"mtp_depth": 3, "mtp_schedule": False}))
    for mode, overrides in variants:
        marker = HERE / "raw" / entry["id"] / "modes" / mode / "complete.json"
        if marker.exists():
            continue
        directory, config, server = launch(entry, "modes", mode, overrides)
        try:
            ok = run(client("timed_load.py", "--concurrency", 1, "--classes", "all",
                "--repeat", 3, "--max-tokens", 256, "--json-out", directory / "greedy.json"),
                directory / "greedy.log")
        finally:
            stop(directory, config, server)
        save(marker, {"finished_at": now(), "passed": ok})


def quality(entry):
    marker = HERE / "raw" / entry["id"] / "quality" / "complete.json"
    if marker.exists():
        return
    directory, config, server = launch(entry, "quality")
    try:
        ok = run([PYTHON, str(HERE / "isolated_eval.py"), "127.0.0.1", "18080",
            "--out", str(directory / "eval"), "--tasks", "humaneval,gsm8k,extract",
            "--limit", "300", "--concurrency", str(entry["slots"]), "--max-tokens", "2048",
            *thinking_args(entry), "--allow-code-execution"], directory / "eval.log", timeout=21600)
    finally:
        stop(directory, config, server)
    save(marker, {"finished_at": now(), "passed": ok})


def prefill(entry):
    """A fresh-server cold-prefill run, independently repeatable."""
    marker = HERE / "raw" / entry["id"] / "prefill" / "complete.json"
    if marker.exists():
        return
    directory, config, server = launch(entry, "prefill")
    try:
        ok = run(client("serve_prefill_probe.py", 2048, 8192, 32768,
            "--repeat", 3, *thinking_args(entry), "--seed", 7, "--tag", "dsv4-20261001",
            "--json-out", directory / "prefill.json"), directory / "prefill.log")
    finally:
        stop(directory, config, server)
    save(marker, {"finished_at": now(), "passed": ok})


def micro():
    out = HERE / "raw/micro"
    for rows in (2, 16):
        run([str(ROOT / "build-release/qsa_index_bench"), "--graph", "--pools",
             "16384,65322,131072", "--rows", str(rows), "--iters", "30", "--warmup", "5"],
            out / f"qsa-rows{rows}.jsonl")
    for rows in (1, 5, 30):
        for context in (4096, 16384, 131072):
            run([str(ROOT / "build-release/csa2_select_bench"), "--ctx", str(context),
                 "--rows", str(rows), "--iters", "30", "--warmup", "5"],
                out / f"csa2-ctx{context}-rows{rows}.jsonl")


def long(entry):
    lengths = {"dsv4-w4": [4096, 32768, 122000],
               "dsv4-w2": [4096, 32768, 122000]}.get(entry["id"])
    if not lengths:
        return
    marker = HERE / "raw" / entry["id"] / "long" / "complete.json"
    if marker.exists():
        return
    directory, config, server = launch(entry, "long")
    passed = {}
    try:
        passed["timing"] = run([PYTHON, str(HERE / "long_context.py"), "--lengths",
            *map(str, lengths), "--json-out", str(directory / "long.json")], directory / "long.log")
    finally:
        stop(directory, config, server)
    save(marker, {"finished_at": now(), "checks": passed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["all", "performance", "modes", "quality", "prefill", "micro", "long"])
    parser.add_argument("--models", help="comma-separated IDs from matrix.json")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
    if args.stage == "micro":
        micro()
        return
    entries = json.loads((HERE / "matrix.json").read_text())
    for entry in entries:
        if args.models and entry["id"] not in args.models.split(","):
            continue
        if args.stage == "all":
            performance(entry)
            if not (HERE / "raw" / entry["id"] / "performance/default/prefill.json").exists():
                prefill(entry)
            modes(entry)
            long(entry)
            quality(entry)
        else:
            globals()[args.stage](entry)


if __name__ == "__main__":
    main()
