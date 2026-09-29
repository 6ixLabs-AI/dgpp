#!/usr/bin/env python3
"""One MTP pass from the nsys sqlite export, in time order (graph-node stream ids
are synthetic, so 'main' = everything but the l2_prefetch kernels): phases split at
the head launches (packq_gemv<8,...>) and the commit; per phase the span, the union
busy time and the idle gaps."""
import sqlite3, sys
db = sys.argv[1]; which = int(sys.argv[2]) if len(sys.argv) > 2 else 100
verbose = "-v" in sys.argv
con = sqlite3.connect(db); cur = con.cursor()
cur.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName ORDER BY k.start")
rows = cur.fetchall()
def short(n):
    n = n.replace("void ", "")
    for p in ("dgpp::", "(anonymous namespace)::", "<unnamed>::", "net::"): n = n.replace(p, "")
    d = 0; cut = len(n)
    for j, ch in enumerate(n):
        if ch == "<": d += 1
        elif ch == ">": d -= 1
        elif ch == "(" and d == 0: cut = j; break
    return n[:cut].replace("(int)", "").replace("(bool)", "")
marks = [i for i, r in enumerate(rows) if "spec_commit_kernel" in r[2]]
a, b = marks[which], marks[which + 1]
step = [(s, e, short(n)) for s, e, n in rows[a:b + 1]]
t0 = step[0][0]
pf = [(s, e) for s, e, n in step if n.startswith("l2_prefetch")]
main = [(s, e, n) for s, e, n in step if not n.startswith("l2_prefetch")]
def union(iv):
    tot = 0; cs = ce = None
    for s, e in sorted(iv):
        if cs is None: cs, ce = s, e
        elif s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    if cs is not None: tot += ce - cs
    return tot
print(f"step {which}: {len(step)} kernels, {(step[-1][1]-t0)/1e6:.3f} ms commit→commit; prefetch kernels {len(pf)} busy {union(pf)/1e3:.0f} us")
# phase boundaries in time: commit end, each head launch start/end, the sample kernels
bounds = [("commit", step[0][0], step[0][1])]
for s, e, n in main:
    if n.startswith("packq_gemv_kernel<8"): bounds.append(("head", s, e))
bounds.append(("end", step[-1][0], step[-1][1]))
names = ["draft1", "draft2", "draft3", "verify"]
phases = []
for i in range(len(bounds) - 1):
    ps, pe = bounds[i][2], bounds[i + 1][1]
    iv = [(max(s, ps), min(e, pe)) for s, e, n in main if e > ps and s < pe and not n.startswith("packq_gemv_kernel<8") and not n.startswith("spec_commit")]
    kinds = {}
    for s, e, n in main:
        if s >= ps and e <= pe: kinds[n] = kinds.get(n, 0) + (e - s)
    top = sorted(kinds.items(), key=lambda kv: -kv[1])[:6]
    phases.append((names[i] if i < 4 else f"p{i}", ps, pe, union(iv), len(iv), bounds[i + 1][2] - bounds[i + 1][1] if bounds[i + 1][0] == "head" else 0, top))
print(f"{'phase':10s} {'start us':>9s} {'span us':>9s} {'busy us':>9s} {'idle us':>8s} {'n':>5s} {'head us':>8s}")
for name, ps, pe, bz, k, hd, top in phases:
    print(f"{name:10s} {(ps-t0)/1e3:9.1f} {(pe-ps)/1e3:9.1f} {bz/1e3:9.1f} {(pe-ps-bz)/1e3:8.1f} {k:5d} {hd/1e3:8.1f}")
    if verbose:
        for n, t in top: print(f"             {t/1e3:8.1f} us  {n}")
print(f"commit kernel {(step[0][1]-step[0][0])/1e3:.1f} us; total main union {union([(s,e) for s,e,n in main])/1e3:.1f} us of {(step[-1][1]-t0)/1e3:.1f}")
if "-k" in sys.argv:
    # the kernel list of one phase (index after -k)
    pi = int(sys.argv[sys.argv.index("-k") + 1]); ps, pe = phases[pi][1], phases[pi][2]
    last = ps
    for s, e, n in main:
        if s >= ps - 1 and e <= pe + 1:
            print(f"  +{(s-t0)/1e3:9.1f} gap {(s-last)/1e3:6.1f} dur {(e-s)/1e3:7.1f} {n}"); last = max(last, e)
