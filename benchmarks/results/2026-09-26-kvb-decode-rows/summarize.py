#!/usr/bin/env python3
"""ms/step per concurrency for each leg: per class the median over repeats of step_ms/decode_steps,
then the mean over classes (the 2026-09-20 A/B's reading); plus C1/C2 transcript identity across legs."""
import json, sys, statistics, collections, hashlib, glob, os
raw = os.path.expanduser('~/claude-scratch/2026-09-26-kvb-rows-ab/raw')
args = sys.argv[1:]
if args and args[0] == '--raw': raw = args[1]; args = args[2:]
legs = args
def load(name):
    d = json.load(open(f'{raw}/{name}.json'))
    ph = d['phases'] if isinstance(d, dict) and 'phases' in d else d
    per = collections.defaultdict(lambda: collections.defaultdict(list))  # conc -> class -> [ms/step]
    texts = collections.defaultdict(dict)
    seen = collections.Counter()
    for p in ph:
        e = p.get('engine') or {}
        c = p.get('concurrency'); k = p.get('class') or p.get('classes') or p.get('name')
        rep = seen[(c, k)]; seen[(c, k)] += 1
        if e and e.get('decode_steps'): per[c][k].append(e['step_ms'] / e['decode_steps'])
        if c in (1, 2):
            for i, r in enumerate(p.get('requests') or []):
                t = r.get('text') if isinstance(r, dict) else None
                if t is not None: texts[(c, k, rep, i)] = hashlib.md5(t.encode()).hexdigest()
    return per, texts
tables = {n: load(n) for n in legs}
concs = sorted({c for per, _ in tables.values() for c in per})
print('ms/step (mean over classes of the per-class median)')
print('conc ' + ' '.join(f'{n:>12s}' for n in legs))
for c in concs:
    row = []
    for n in legs:
        per = tables[n][0]
        vals = [statistics.median(v) for v in per[c].values()] if c in per else []
        row.append(statistics.mean(vals) if vals else float('nan'))
    print(f'C{c:<3} ' + ' '.join(f'{x:12.2f}' for x in row))
    if len(row) >= 3:
        base = (row[0] + row[-1]) / 2
        print(f'      cand vs bases {100 * (row[1] / base - 1):+.2f} %   (A/A spread {100 * abs(row[0] - row[-1]) / base:.2f} %)')
if all(t for _, t in tables.values()):
    a = tables[legs[0]][1]
    for n in legs[1:]:
        b = tables[n][1]
        same = sum(1 for k in a if k in b and a[k] == b[k]); tot = sum(1 for k in a if k in b)
        print(f'transcripts identical {legs[0]} vs {n}: {same}/{tot}')
