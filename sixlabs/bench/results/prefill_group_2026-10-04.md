# Reading several prompts in one walk (`engine.prefill_group`), Qwen3-Next-80B (2026-10-04)

`sixlabs/bench/group_two.sh` on DGXtwo, 09:33–09:43 EDT, nothing else on the GPU in either run.
Build d06a1cd; 8 slots; prefill 1024 a tick while decoding / 4096 idle; 8 decode passes a chunk;
shortest first. `group_check.py`: every prompt is a ledger nobody has sent before plus a lookup
whose answer is one of its lines.

Correctness, both settings: answers right one at a time (4/4), four together (4/4, identical
text), four lengths together 1k/3k/6k/9k (4/4), three beside a streaming answer (3/3), a burst
of eight (8/8).

| | `off` (one prompt a walk) | `on`, budget per reading prompt (the original rule) | `on`, budget the walk's total (`prefill_budget_per_reader: false`) — **chosen** |
|---|---|---|---|
| burst of eight 1,958-token prompts, 64-token answers: first tokens | 0.66 to 9.46 s (mean 5.65) | all eight at 5.24 s | all eight at 5.22 s |
| the same burst: all eight finished | 10.69 s | 7.31 s | 7.36 s |
| three 3k prompts arriving beside a streaming answer: first tokens | 8.38, 4.23, 11.38 s | all three at 3.55 s | all three at 5.86 s |
| 2 fresh 17k prompts alone: read rate (first token) | 2,650 tok/s (13.2 s) | 3,349 tok/s (10.4 s) | 3,339 tok/s (10.4 s) |
| 4 answers beside those 2 prompts: answers | 20.3 tok/s each, pause p95 0.19 s / max 0.49 s | 12.7 tok/s each, pause p95 0.67 s / max 0.96 s | 16.7 tok/s each, pause p95 0.45 s / max 0.53 s |
| the same: prompts read (first token) | 1,251 tok/s (27.3 s) | 1,809 tok/s (22.1 s) | 1,265 tok/s (27.5 s) |

The third column was measured at 09:59-10:04 EDT on the next build (07a32ba), same box, same
settings otherwise, nothing else on the GPU; every correctness check passed there too.

Reading it:
- Prompts that arrive together are read together and finish together: the burst's total time
  falls by a third and its slowest first token from 9.5 s to 5.2 s. The price is the head start
  the first arrivals had (0.7 s becomes 5.2 s).
- Two prompts in one walk are read 26 % faster than in alternating walks.
- Beside answers in progress the original rule gives each reading prompt its own quantum, so
  with two prompts a chunk is 2,048 rows: prompts gain 45 %, answers lose a third and pause for
  up to a second. With the budget as the walk's total the answers keep 16.7 tokens/s and their
  longest pause stays at half a second, as without sharing; the prompts are read at the
  unshared rate. Against no sharing at all that setting still costs the answers 18 % (a walk of
  two 512-row spans takes longer than one 1,024-row span), which is what buys the burst and
  idle-read gains.
- Chosen for the 80B: `prefill_group: on`, `prefill_budget_per_reader: false`.

On DGXone the same check passed for correctness, but HEM held about 91 % of that GPU through
the `on` run, so its timings are not used.
