#!/usr/bin/env python3
"""Summarize sampled telemetry for completed four-node launches."""
import csv
import datetime
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def summarize(path):
    # A node whose telemetry channel never connected (this campaign: rank 2's
    # management link was down, so its stream is one ssh error line) is
    # recorded as unavailable; the launch itself ran over the fabric.
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not any(r.get('kind') == 'memory' for r in records):
        return {'path': str(path.relative_to(HERE)), 'unavailable': True,
                'detail': path.read_text().strip().splitlines()[:1]}
    memory = [r for r in records if r['kind'] == 'memory']
    gpu = [r for r in records if r['kind'] == 'gpu']
    temperatures = []
    thermal = 0
    query_errors = 0
    for record in gpu:
        if record.get('returncode') != 0:
            query_errors += 1
            continue
        fields = next(csv.reader([record['data']], skipinitialspace=True))
        temperatures.append(float(fields[0]))
        thermal += any(flag == 'Active' for flag in fields[3:5])
    pressure = {'some': [], 'full': []}
    for record in memory:
        for line in record['pressure'].splitlines():
            kind, *fields = line.split()
            values = dict(field.split('=', 1) for field in fields)
            pressure[kind].append(float(values['avg10']))
    counters = memory[0]['vmstat']
    counter_deltas = {key: memory[-1]['vmstat'][key] - counters[key]
                      for key in counters}
    assert all(value >= 0 for value in counter_deltas.values()), 'counter reset: ' + str(path)
    return {
        'path': str(path.relative_to(HERE)),
        'first_at': memory[0]['at'], 'last_at': memory[-1]['at'],
        'memory_samples': len(memory), 'gpu_samples': len(gpu),
        'minimum_available_GiB': min(r['MemAvailable_KiB'] for r in memory) / 1048576,
        'maximum_swap_used_MiB': max(r['SwapTotal_KiB'] - r['SwapFree_KiB'] for r in memory) / 1024,
        'maximum_temperature_C': max(temperatures) if temperatures else None,
        'active_thermal_samples': thermal, 'gpu_query_errors': query_errors,
        'maximum_memory_psi_avg10_percent': {kind: max(values) for kind, values in pressure.items()},
        'vmstat_deltas': counter_deltas,
    }


def main():
    launches = []
    for receipt in sorted((HERE / 'raw').glob('**/down.log.command.json')):
        data = json.loads(receipt.read_text())
        if data.get('returncode') != 0:
            continue
        paths = sorted(receipt.parent.glob('node*-telemetry.jsonl'))
        if not paths:
            continue
        assert len(paths) == 4, 'incomplete telemetry: ' + str(receipt.parent)
        launches.append({'directory': str(receipt.parent.relative_to(HERE)),
                         'nodes': [summarize(path) for path in paths]})
    result = {'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'Completed launches only; sampled extrema include startup and shutdown, '
                       'and do not establish transient peaks between observations.',
              'launches': launches}
    (HERE / 'raw/telemetry-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Summarized {len(launches)} completed launches.')


if __name__ == '__main__':
    main()
