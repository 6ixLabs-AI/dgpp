#!/usr/bin/env python3
"""Stream host memory each second and GPU state every five seconds."""
import datetime
import json
from pathlib import Path
import subprocess
import time


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def emit(value):
    print(json.dumps(value), flush=True)


tick = 0
try:
    while True:
        memory = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemFree', 'MemAvailable', 'Cached', 'SwapTotal',
                       'SwapFree', 'SwapCached', 'Mlocked', 'Unevictable', 'Slab'):
                memory[key + '_KiB'] = int(value.split()[0])
        vm = {}
        for line in Path('/proc/vmstat').read_text().splitlines():
            key, value = line.split()
            if key.startswith(('pswp', 'pgscan', 'pgsteal', 'allocstall')) or key == 'oom_kill':
                vm[key] = int(value)
        emit({'at': stamp(), 'kind': 'memory', **memory, 'vmstat': vm,
              'pressure': Path('/proc/pressure/memory').read_text().strip()})
        if tick % 5 == 0:
            try:
                p = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown',
                                    '--format=csv,noheader'], capture_output=True, text=True, timeout=2)
                emit({'at': stamp(), 'kind': 'gpu', 'returncode': p.returncode,
                      'data': p.stdout.strip(), 'error': p.stderr.strip()})
            except (OSError, subprocess.TimeoutExpired) as error:
                emit({'at': stamp(), 'kind': 'gpu', 'error': str(error)})
        tick += 1
        time.sleep(1)
except BrokenPipeError:
    pass
