#!/usr/bin/env python3
"""Per-kernel comparison of two nsys sqlite exports' last bursts (the prefill), by kernel-name substring:
total ms, launches, the median of the chunk-sized launches (those over half the longest), and the GPU busy share.
Usage: compare_profiles.py A.sqlite B.sqlite"""
import sqlite3, statistics, sys
from collections import defaultdict

def burst(db):
    c = sqlite3.connect(db)
    rows = c.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start").fetchall()
    bursts, cur = [], [rows[0]]
    for r in rows[1:]:
        if r[0] - cur[-1][1] > 100e6: bursts.append(cur); cur = [r]
        else: cur.append(r)
    bursts.append(cur)
    b = bursts[-1]
    wall = (b[-1][1] - b[0][0]) / 1e6
    busy, end = 0, -1
    for st, en, _ in b:
        if st > end: busy += en - st; end = en
        elif en > end: busy += en - end; end = en
    return b, wall, busy / 1e6

NAMES = ["packq_gemm_wide_kernel<unsigned short", "packq_gemm_wide_kernel<float", "nvjet", "cutlass", "combine_norm_kernel", "mix_finish_kernel",
         "combine_dots_kernel", "moe_accum_ordered_kernel", "moe_router_dots_tiled_kernel", "moe_router_dots", "gdn_chunk_state_kernel", "gdn_chunk_prep_kernel",
         "stage_wait_kernel", "qsa_attn_prefill_warp_kernel", "index_score_kernel", "fp8_dequant_blocks_kernel", "moe_swiglu_clamp_kernel",
         "gated_rmsnorm_kernel", "kda_conv_tiled_kernel", "moe_accum_kernel", "moe_round_bf16_kernel", "hash_ids_rows_kernel", "publish_stage_kernel"]

def table(b):
    per = defaultdict(list)
    for st, en, name in b:
        for n in NAMES:
            if n in name: per[n].append((en - st) / 1e3); break
    out = {}
    for n, v in per.items():
        v = sorted(v, reverse=True)
        big = [d for d in v if d > 0.5 * v[0]]
        out[n] = (sum(v) / 1e3, len(v), statistics.median(big), len(big))
    return out

a, b = sys.argv[1], sys.argv[2]
ba, wa, busya = burst(a); bb, wb, busyb = burst(b)
print(f"A: {a}\n   wall {wa:.0f} ms busy {busya:.0f} ms kernels {len(ba)}\nB: {b}\n   wall {wb:.0f} ms busy {busyb:.0f} ms kernels {len(bb)}")
ta, tb = table(ba), table(bb)
print(f"\n{'kernel':44s} {'A ms':>8s} {'B ms':>8s} {'delta':>8s} | {'A n':>5s} {'B n':>5s} | {'A med us':>9s} {'B med us':>9s} (chunk-sized launches)")
for n in NAMES:
    if n not in ta and n not in tb: continue
    A = ta.get(n, (0, 0, 0, 0)); B = tb.get(n, (0, 0, 0, 0))
    print(f"{n:44s} {A[0]:8.0f} {B[0]:8.0f} {B[0]-A[0]:+8.0f} | {A[1]:5d} {B[1]:5d} | {A[2]:9.0f} {B[2]:9.0f}")
