#!/usr/bin/env python3
"""hf_adopt_local_dir.py — move a `hf download --local-dir` copy of a model into the HF hub cache.

  python3 hf_adopt_local_dir.py --src /home/mark/models/qwen3-next-80b-fp8 \
      --repo Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 [--hub ~/.cache/huggingface/hub] [--dry-run]

A local-dir download keeps, per file, `.cache/huggingface/download/<file>.metadata` = the commit it
came from and the file's etag (the LFS sha256, or the git blob sha1 for small files). That is all
the hub cache needs: blobs/<etag>, snapshots/<commit>/<file> -> ../../blobs/<etag>, refs/main.

Order, so nothing is moved unless everything checks out:
  1. every file is hashed and compared with its etag (a mismatch stops the run, nothing moved);
  2. an existing cache entry for the repo that is not ours to write (a root-owned abandoned
     download) is renamed aside inside the hub directory — renamed, never deleted;
  3. the files are renamed into blobs/ (same filesystem: no copy) and the snapshot is linked.
Leftovers in the source directory (the `.cache` folder with its metadata and any stale
`.incomplete` chunks) are left where they are.
"""
import argparse, hashlib, os, sys, time
from concurrent.futures import ThreadPoolExecutor

t0 = time.time()


def log(step, done, total, err, msg=""):
    print(f"[{step}/3] {done}/{total} err={err} elapsed={int(time.time() - t0)}s {msg}".rstrip(), flush=True)


def digest(path, etag):
    """sha256 of the bytes (LFS files) or git's blob sha1 (regular files), by the etag's length."""
    size = os.path.getsize(path)
    if len(etag) == 64:
        h = hashlib.sha256()
    elif len(etag) == 40:
        h = hashlib.sha1()
        h.update(b"blob %d\0" % size)
    else:
        raise ValueError(f"{path}: unrecognised etag {etag!r}")
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--hub", default=os.path.expanduser("~/.cache/huggingface/hub"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    meta_dir = os.path.join(a.src, ".cache", "huggingface", "download")
    files, commits = [], set()
    for name in sorted(os.listdir(a.src)):
        path = os.path.join(a.src, name)
        if name == ".cache" or not os.path.isfile(path) or os.path.islink(path):
            continue
        mp = os.path.join(meta_dir, name + ".metadata")
        if not os.path.exists(mp):
            sys.exit(f"no download metadata for {name}: cannot name its blob; nothing moved")
        lines = open(mp).read().split("\n")
        commits.add(lines[0].strip())
        files.append((name, lines[1].strip().strip('"')))
    if len(commits) != 1:
        sys.exit(f"files come from more than one commit {sorted(commits)}; nothing moved")
    commit = commits.pop()
    print(f"{a.repo} @ {commit}: {len(files)} files, "
          f"{sum(os.path.getsize(os.path.join(a.src, n)) for n, _ in files) / 2**30:.2f} GiB", flush=True)

    # 1. verify every file against its etag
    bad, done = [], 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {n: pool.submit(digest, os.path.join(a.src, n), e) for n, e in files}
        for n, e in files:
            got = futs[n].result()
            done += 1
            if got != e:
                bad.append(n)
            log(1, done, len(files), len(bad), f"{n} {'OK' if got == e else 'MISMATCH got ' + got}")
    if bad:
        sys.exit(f"hash mismatch in {bad}; nothing moved")

    repo_dir = os.path.join(a.hub, "models--" + a.repo.replace("/", "--"))
    if a.dry_run:
        print(f"dry run: would adopt into {repo_dir}", flush=True)
        return 0

    # 2. an existing entry we cannot write into is renamed aside (never deleted)
    if os.path.lexists(repo_dir) and not os.access(os.path.join(repo_dir, "blobs"), os.W_OK):
        aside = os.path.join(a.hub, ".stale-unwritable." + os.path.basename(repo_dir) + time.strftime(".%Y%m%d-%H%M%S"))
        os.rename(repo_dir, aside)
        log(2, 1, 1, 0, f"existing entry was not writable; renamed aside to {aside}")
    else:
        log(2, 1, 1, 0, "no unwritable entry in the way")

    # 3. blobs, snapshot links, ref
    blobs = os.path.join(repo_dir, "blobs")
    snap = os.path.join(repo_dir, "snapshots", commit)
    os.makedirs(blobs, exist_ok=True)
    os.makedirs(snap, exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "refs"), exist_ok=True)
    for i, (name, etag) in enumerate(files, 1):
        blob = os.path.join(blobs, etag)
        src = os.path.join(a.src, name)
        if os.path.exists(blob):
            if os.path.getsize(blob) != os.path.getsize(src):
                sys.exit(f"{blob} already exists with a different size; stopped after {i - 1} files")
            os.remove(src)       # the same content is already a blob (identical etag and size)
        else:
            os.rename(src, blob)
        link = os.path.join(snap, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.join("..", "..", "blobs", etag), link)
        log(3, i, len(files), 0, name)
    with open(os.path.join(repo_dir, "refs", "main"), "w") as f:
        f.write(commit)
    print(f"DONE snapshot {snap}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
