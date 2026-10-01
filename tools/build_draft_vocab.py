#!/usr/bin/env python3
"""Build a draft vocabulary set for engine.draft_vocab (the opt-in draft head slice).

The set is the tokenizer's byte-level alphabet, every added/special token, and the
most frequent merged tokens by BPE merge rank (an earlier merge is a more frequent
pair in the tokenizer's training corpus), sorted ascending, written as a
one-dimensional int32 .npy. Tokens outside the set are never proposed by the
draft; the target verifies every proposal, so outputs are unchanged and only the
draft acceptance can move (about 1 % of tokens fall outside a 65,536-token set on
English and code with Qwen3.8-Flash-Next).

usage: build_draft_vocab.py TOKENIZER.json OUT.npy [--size 65536]
"""
import argparse
import json
import struct
import sys


def write_npy_int32(path, values):
    header = "{'descr': '<i4', 'fortran_order': False, 'shape': (%d,), }" % len(values)
    pad = 64 - ((10 + len(header) + 1) % 64)
    header = header + " " * pad + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00")
        f.write(struct.pack("<H", len(header)))
        f.write(header.encode("ascii"))
        f.write(struct.pack("<%di" % len(values), *values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=65536)
    args = ap.parse_args()
    tok = json.load(open(args.tokenizer))
    model = tok["model"]
    if model.get("type") != "BPE":
        sys.exit("the tokenizer must be BPE")
    vocab = model["vocab"]  # token -> id
    merges = model["merges"]
    added = tok.get("added_tokens", [])
    chosen = []
    seen = set()

    def take(i):
        if i not in seen:
            seen.add(i)
            chosen.append(i)

    for a in added:
        take(int(a["id"]))
    # The alphabet: every vocab token no merge produces.
    produced = set()
    for m in merges:
        pair = m if isinstance(m, list) else m.split(" ", 1)
        produced.add("".join(pair))
    for t, i in vocab.items():
        if t not in produced:
            take(int(i))
    n_base = len(chosen)
    for m in merges:
        if len(chosen) >= args.size:
            break
        pair = m if isinstance(m, list) else m.split(" ", 1)
        t = "".join(pair)
        i = vocab.get(t)
        if i is not None:
            take(int(i))
    chosen.sort()
    write_npy_int32(args.out, chosen)
    print(f"{len(chosen)} ids -> {args.out} (added/special + alphabet {n_base}, then merges by rank; vocab {len(vocab)})")


if __name__ == "__main__":
    main()
