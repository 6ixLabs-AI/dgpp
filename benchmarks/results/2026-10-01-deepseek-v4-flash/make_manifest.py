#!/usr/bin/env python3
"""Write manifest.json for this campaign: the revision, the release binary's
hash, the site settings, the deployment templates, the checkpoint revision
and the evaluation image (the same fields the current-code campaign's
manifest carries and its runner / evaluator / summarizer read)."""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from site_env import settings  # noqa: E402

CURRENT = HERE.parent / "2026-09-22-current" / "manifest.json"


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main():
    current = json.loads(CURRENT.read_text())
    binary = ROOT / "build-release/dgpp-serve"
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip().splitlines()[0]
    values = settings()
    snapshot = sorted((Path.home() / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots").iterdir())
    manifest = {
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "tracked_diff": git("diff", "--stat", "HEAD"),
        "site_settings": {k: values[k] for k in current["site_settings"] if k in values},
        "templates": json.loads((HERE / "matrix.json").read_text()),
        "datasets": current.get("datasets"),
        "eval_image": current["eval_image"],
        "eval_image_repo_digests": current.get("eval_image_repo_digests"),
        "eval_image_platform": current.get("eval_image_platform"),
        "checkpoint_revisions": {"models--deepseek-ai--DeepSeek-V4-Flash-0731": [p.name for p in snapshot]},
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "version": version,
        "benchmark_environment": {"CUDA_DEVICE_MAX_CONNECTIONS": "32"},
        "benchmark_environment_scope": current.get("benchmark_environment_scope"),
        "evaluation_thinking_controls": current.get("evaluation_thinking_controls"),
        "telemetry": current.get("telemetry"),
        "note": "The DeepSeek-V4-Flash-0731 campaign: the family's first measurement, on the tree that adds it "
                "(tracked_diff lists the uncommitted files); the other deployments' numbers are the "
                "earlier campaigns'.",
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("commit", "binary_sha256", "version")}, indent=2))


if __name__ == "__main__":
    main()
