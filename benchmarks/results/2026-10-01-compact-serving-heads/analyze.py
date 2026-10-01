#!/usr/bin/env python3
"""Reproduce the GLM-4.7 compact-head comparison from the retained records."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'scripts'))
from fabric_logprob import load_run, logprobs

runs = {}
hidden = {}
for mode in ('full', 'compact'):
    directory = HERE / ('glm4-' + mode)
    steps, ranks = load_run(str(directory), prefill=True)
    runs[mode] = (steps, ranks)
    hidden[mode] = []
    for rank in ranks:
        records = re.findall(
            r'\[head_hidden\] rank (\d+) step (\d+): ([0-9a-f]+) length (\d+)',
            (directory / f'r{rank}.log').read_text())
        assert len(records) == len(steps), 'missing or duplicate hidden records'
        assert {int(r) for r, _, _, _ in records} == {rank}, 'wrong hidden rank'
        assert [int(s) for _, s, _, _ in records] == sorted(steps), 'wrong hidden positions'
        hidden[mode].append([(int(s), h, int(n)) for _, s, h, n in records])
    assert all(rows == hidden[mode][0] for rows in hidden[mode]), 'hidden differs across ranks'
assert runs['full'][1] == runs['compact'][1]
assert set(runs['full'][0]) == set(runs['compact'][0])
assert hidden['full'] == hidden['compact'], 'head changed hidden state'
full = logprobs(*runs['full'])
compact = logprobs(*runs['compact'])
assert full and set(full) == set(compact)
assert all(runs['full'][0][i]['target'] == runs['compact'][0][i]['target'] for i in full)
deltas = [compact[i][0] - full[i][0] for i in full]
summary = {
    'positions': len(full),
    'ranks': len(runs['full'][1]),
    'hidden_matches': len(full) * len(runs['full'][1]),
    'full_mean_nll': -sum(x[0] for x in full.values()) / len(full),
    'compact_mean_nll': -sum(x[0] for x in compact.values()) / len(compact),
    'mean_nll_delta': -sum(deltas) / len(deltas),
    'mean_abs_logprob_delta': sum(map(abs, deltas)) / len(deltas),
    'max_abs_logprob_delta': max(map(abs, deltas)),
    'argmax_changes': sum(runs['full'][0][i]['argmax'] != runs['compact'][0][i]['argmax'] for i in full),
    'top1_hits_full': sum(x[1] for x in full.values()),
    'top1_hits_compact': sum(x[1] for x in compact.values()),
}
assert abs(summary['mean_nll_delta']) <= 1e-4, 'mean NLL gate'
assert summary['max_abs_logprob_delta'] <= 1e-3, 'per-target logprob gate'
print(json.dumps(summary, indent=2))
