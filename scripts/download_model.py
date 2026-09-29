"""Download once on rank 0, then sync the snapshot to the configured peers.

Use --config FILE for a deployment or --model ORG/NAME for a local download.
The default cache is ~/.cache/huggingface/hub; HF_HUB_CACHE and HF_HOME override
it. Only rank 0 contacts Hugging Face. Peers receive the selected snapshot and
its referenced blobs over rsync/SSH, not credentials. Stop deployments using
this checkpoint before downloading or syncing it.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from cache_sync import activate, peer_cache, sync_snapshot
from cluster_doctor import cache_root, cached_snapshot, checkpoint_size, require_head, shard_bytes
from site_env import cache_environment, config_argument, resolve_config, settings


def download(model, revision, root):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ValueError("install requirements-download.txt in a venv before downloading") from error
    return Path(snapshot_download(repo_id=model, revision=revision, cache_dir=str(root), max_workers=4))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=config_argument, help="deployment JSON; download locally and sync its peers")
    source.add_argument("--model", help="Hugging Face repository for a local-only download")
    parser.add_argument("--cache-dir", type=Path, help="local-only override; use .env for deployment cache paths")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--activate", action="store_true", help="explicitly select a non-main downloaded revision")
    parser.add_argument("--local-only", action="store_true", help="skip peers for this invocation")
    parser.add_argument("--ngram-table-model", help="local-only: the repository holding the Qwen n-gram table's shards")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sync-only", action="store_true", help="sync the active local snapshot without contacting Hugging Face")
    mode.add_argument("--verify-only", action="store_true", help="check active snapshots locally and over SSH, without downloads or writes")
    args = parser.parse_args(argv)
    if args.config and args.cache_dir:
        parser.error("use HF_HUB_CACHE or DGPP_NODE_OVERRIDES in .env with --config")
    if (args.sync_only or args.verify_only) and (args.activate or args.revision != "main"):
        parser.error("sync-only and verify-only use the active snapshot; omit --revision and --activate")
    if args.sync_only and (not args.config or args.local_only):
        parser.error("--sync-only needs --config and cannot be combined with --local-only")
    values = settings()
    cfg = resolve_config(args.config, values) if args.config else None
    if cfg:
        require_head(cfg["nodes"][0])
    model = cfg["model"] if cfg else args.model
    # The Qwen n-gram table from another repository (engine.ngram_table_model,
    # the AutoRound hybrid): downloaded and synced after the model, verified
    # by shard headers alone (it is not a checkpoint of its own).
    table_model = (cfg.get("engine", {}) or {}).get("ngram_table_model") if cfg else args.ngram_table_model
    env = {**os.environ, **(cfg["node_env"][0] if cfg else cache_environment(values))}
    root = args.cache_dir.expanduser() if args.cache_dir else cache_root(env)
    peers = list(enumerate(cfg["nodes"]))[1:] if cfg and not args.local_only else []
    if peers:
        for tool in (["ssh"] if args.verify_only else ["ssh", "rsync"]):
            if not shutil.which(tool):
                parser.error(f"{tool} is required for peer synchronization")
    # This is the only Hub call. No peer command invokes this downloader.
    snapshot = (cached_snapshot(model, root) if args.sync_only or args.verify_only
                else download(model, args.revision, root))
    size = checkpoint_size(snapshot)
    if args.activate:
        activate(snapshot)
    if cached_snapshot(model, root).resolve() != snapshot.resolve():
        raise ValueError("another revision is active; use --activate to select the downloaded revision before syncing")
    print(f"Verified on rank 0: {model}, revision {snapshot.name}, {size / 2**30:.1f} GiB of indexed weights", flush=True)
    for rank, host in peers:
        target = f"{cfg['ssh_user']}@{host}"
        if args.verify_only:
            peer_cache(target, model, cfg["node_env"][rank], verify=True, revision=snapshot.name)
            print(f"Verified rank {rank} ({host}): revision {snapshot.name}", flush=True)
        else:
            sync_snapshot(snapshot, target, cfg["node_env"][rank])
    if table_model:
        table = (cached_snapshot(table_model, root) if args.sync_only or args.verify_only
                 else download(table_model, "main", root))
        tsize = shard_bytes(table)
        if cached_snapshot(table_model, root).resolve() != table.resolve():
            raise ValueError("another revision of the table repository is active; activate it before syncing")
        print(f"Verified on rank 0: n-gram table {table_model}, revision {table.name}, {tsize / 2**30:.1f} GiB of shards", flush=True)
        for rank, host in peers:
            target = f"{cfg['ssh_user']}@{host}"
            if args.verify_only:
                peer_cache(target, table_model, cfg["node_env"][rank], verify=True, revision=table.name)
                print(f"Verified rank {rank} ({host}): table revision {table.name}", flush=True)
            else:
                sync_snapshot(table, target, cfg["node_env"][rank])
    print("Snapshot validation checks metadata and shard lengths; transfers use rsync content checksums.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"checkpoint setup: {error}", file=sys.stderr)
        raise SystemExit(2)
