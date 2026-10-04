# Reading several prompts in one walk (`engine.prefill_group`), Qwen3-Next-80B (2026-10-04)

`sixlabs/bench/group_two.sh` on DGXtwo, 09:33–09:43 EDT, nothing else on the GPU in either run.
Build d06a1cd; 8 slots; prefill 1024 a tick while decoding / 4096 idle; 8 decode passes a chunk;
shortest first. `group_check.py`: every prompt is a ledger nobody has sent before plus a lookup
whose answer is one of its lines.

Correctness, both settings: answers right one at a time (4/4), four together (4/4, identical
text), four lengths together 1k/3k/6k/9k (4/4), three beside a streaming answer (3/3), a burst
of eight (8/8).

| | `off` (one prompt a walk) | `on` (shared walk) |
|---|---|---|
| burst of eight 1,958-token prompts, 64-token answers: first tokens | 0.66, 1.29, 4.17, 4.81, 7.68, 7.71, 9.44, 9.46 s (mean 5.65) | all eight at 5.24 s |
| the same burst: all eight finished | 10.69 s | 7.31 s |
| three 3k prompts arriving beside a streaming answer: first tokens | 8.38, 4.23, 11.38 s | all three at 3.55 s |
| 2 fresh 17k prompts alone: read rate (first token) | 2,650 tok/s (13.2 s) | 3,349 tok/s (10.4 s) |
| 4 answers beside those 2 prompts: answers | 20.3 tok/s each, pause p95 0.19 s / max 0.49 s | 12.7 tok/s each, pause p95 0.67 s / max 0.96 s |
| the same: prompts read (first token) | 1,251 tok/s (27.3 s) | 1,809 tok/s (22.1 s) |

Reading it:
- Prompts that arrive together are read together and finish together: the burst's total time
  falls by a third and its slowest first token from 9.5 s to 5.2 s. The price is the head start
  the first arrivals had (0.7 s becomes 5.2 s).
- Two prompts in one walk are read 26 % faster than in alternating walks.
- Beside answers in progress the original rule gives each reading prompt its own quantum, so
  with two prompts a chunk is 2,048 rows: prompts gain 45 %, answers lose a third and pause for
  up to a second. `engine.prefill_budget_per_reader: false` keeps the walk at the budget in
  total; its numbers are not measured yet (the engine was taken by a benchmark before the run).

On DGXone the same check passed for correctness, but HEM held about 91 % of that GPU through
the `on` run, so its timings are not used.
