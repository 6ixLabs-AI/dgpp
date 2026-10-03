#!/usr/bin/env python3
"""Use the repository evaluator, running generated Python in an isolated container."""
from pathlib import Path
import json
import runpy
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
# The current API rejects a nonzero reasoning effort alongside an explicit
# thinking-off switch. Match the requested no-thinking evaluation mode.
if "--no-think" in sys.argv and "--reasoning-effort" not in sys.argv:
    sys.argv.extend(["--reasoning-effort", "none"])
original_run = subprocess.run
expected_image = json.loads(Path(__file__).with_name("manifest.json").read_text())["eval_image"]
image = original_run(["docker", "image", "inspect", expected_image, "--format", "{{.Id}}"],
                     capture_output=True, text=True, check=True).stdout.strip()
if image != expected_image:
    raise RuntimeError("evaluation container does not match the campaign manifest")


def isolated_run(command, **kwargs):
    if len(command) != 2 or command[0] != sys.executable:
        raise RuntimeError(f"unexpected evaluation subprocess: {command!r}")
    name = "dgpp-eval-" + uuid.uuid4().hex
    isolated = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
                "--memory", "512m", "--pids-limit", "64", "--cpus", "1", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
                "--user", "65534:65534", "-i", image, "python3", "-"]
    try:
        return original_run(isolated, input=Path(command[1]).read_bytes(), **kwargs)
    except subprocess.TimeoutExpired:
        original_run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
        raise


print(f"HumanEval container image: {image}", flush=True)
subprocess.run = isolated_run
runpy.run_path(str(ROOT / "scripts/serve_eval.py"), run_name="__main__")
