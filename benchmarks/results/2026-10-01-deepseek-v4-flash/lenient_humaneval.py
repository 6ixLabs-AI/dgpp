#!/usr/bin/env python3
"""HumanEval answers that failed only because the model ended without the closing code fence: run the open
block as the code (the same pinned, isolated container as isolated_eval.py) and report the count that passes.
lenient_humaneval.py [DEPLOYMENT ...]   ->   raw/<deployment>/quality/default/eval/humaneval_lenient.json"""
import json, os, subprocess, sys, uuid
from pathlib import Path
HERE = Path(__file__).resolve().parent
image = json.loads((HERE / "manifest.json").read_text())["eval_image"]
data = Path(os.environ.get("DGPP_DATA_DIR", HERE.parents[2] / "build-ci/eval_data")) / "HumanEval.jsonl"
items = {j["task_id"]: j for j in map(json.loads, open(data))}

def run(program):
    name = "dgpp-eval-" + uuid.uuid4().hex
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only", "--memory", "512m",
           "--pids-limit", "64", "--cpus", "1", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "--user", "65534:65534", "-i", image, "python3", "-"]
    try:
        return subprocess.run(cmd, input=program.encode(), capture_output=True, timeout=60).returncode == 0
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
        return False

for dep in sys.argv[1:] or [e["id"] for e in json.loads((HERE / "matrix.json").read_text())]:
    base = HERE / "raw" / dep / "quality/default/eval"
    rows = [json.loads(l) for l in open(base / "humaneval.jsonl")]
    strict = sum(r["ok"] for r in rows)
    opened = [r for r in rows if not r["ok"] and (r["content"] or "").count("```") % 2 == 1]
    passed = []
    for r in opened:
        code = r["content"].split("```", 1)[1]
        code = code.split("\n", 1)[1] if "\n" in code else ""  # drop the language tag line
        item = items[r["id"]]
        if run(code + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"):
            passed.append(r["id"])
    out = {"strict_passed": strict, "n": len(rows), "unclosed_fence_failures": [r["id"] for r in opened],
           "pass_when_the_open_block_is_executed": passed, "lenient_passed": strict + len(passed)}
    (base / "humaneval_lenient.json").write_text(json.dumps(out, indent=2) + "\n")
    print(dep, f"strict {strict}/{len(rows)}; {len(opened)} unclosed-fence failures, {len(passed)} pass when executed; lenient {strict + len(passed)}/{len(rows)}")
