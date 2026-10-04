#!/usr/bin/env python3
"""compare_states.py — DGPP's --dump-states (raw bf16) against the numpy reference's h_NN.npy.

  python3 compare_states.py DGPP.bf16 REF_DIR T [HIDDEN]

Block l of the DGPP dump is the residual after layer l = the reference's h_{l+1}; the last block is
the final normed hidden. Prints, per layer: relative rms error, cosine, worst row's cosine.
"""
import sys, os, glob
import numpy as np

path, ref, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
H = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
raw = np.fromfile(path, dtype=np.uint16)
blocks = raw.size // (T * H)
x = (raw.astype(np.uint32) << 16).view(np.float32).reshape(blocks, T, H)
print(f"{blocks - 1} layers + final, T={T}")
worst = 0.0
for l in range(blocks - 1):
    f = os.path.join(ref, f"h_{l + 1:02d}.npy")
    if not os.path.exists(f):
        print(f"layer {l:2d}: no reference {f}")
        continue
    r = np.load(f).astype(np.float64)
    d = x[l].astype(np.float64)
    rel = np.sqrt(((d - r) ** 2).mean() / (r ** 2).mean())
    cos = (d * r).sum() / np.sqrt((d * d).sum() * (r * r).sum())
    rc = (d * r).sum(1) / np.sqrt((d * d).sum(1) * (r * r).sum(1))
    worst = max(worst, rel)
    print(f"layer {l:2d}: rel rms err {rel:.5f}  cosine {cos:.6f}  worst row cosine {rc.min():.6f} (row {int(rc.argmin())})")
print(f"worst relative error {worst:.5f}")
