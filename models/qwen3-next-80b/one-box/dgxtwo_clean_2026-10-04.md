# Qwen3-Next-80B on DGXtwo, alone on the GPU (2026-10-04, 07:14–07:57 EDT)

> Layout note, 2026-10-04: moved from `sixlabs/bench/results/dgxtwo_clean_2026-10-04.md` when the per-model folders under `models/` were made. Paths to other moved records were updated; nothing else was changed.

`sixlabs/bench/after_papers.sh`, unattended, after the RAG papers benchmark. Build 5c20f70's
binary; 8 slots; 524,288-token pool; fp8 dense; MTP depth 2; prefill 1024 tokens a tick while
decoding / 4096 idle; 8 decode passes per chunk; shortest prompt first. Before each benchmark:
no other process on the GPU, no request in flight. One run each. Raw output: DGXone
`~/dgpp-refset/dgxtwo_benches_20261004/`.

| benchmark | result |
|---|---|
| `bench_decode.py`, single stream | 82.2 tok/s prose, 103.6 tok/s code |
| `bench_decode.py`, 2 / 4 / 8 streams | 118.5 / 185.4 / 253.4 tok/s in total; MTP acceptance 81.1 % |
| `mixed_load.py`, 4 answers alone | 36.4 tok/s each (145.7 total), pause p95 76 ms |
| `mixed_load.py`, 2 fresh 17k-token prompts alone | read at 2,914 tok/s, first token 12.0 s |
| `mixed_load.py`, both together | answers 17.3 tok/s each, pause p95 172 ms / max 507 ms; prompts read at 1,221 tok/s, first token 27.4 s |
| soak, 30 min (`scripts/serve_soak.py`) | 7,973 requests, 0 failed, 0 shed, 84 of 84 burst requests served; short requests' first token p50 158–167 ms, p99 323–330 ms, the same in all three 10-minute windows (ratio 1.02); 28 ms a token at p50; pool steady at 3,849–3,861 blocks |
| agentic streams, 4 x 8 turns | 28 of 28 follow-up turns attached to the cached prompt (80 % of prompt tokens); 31 s wall; first token median 3.9 s |
| judge sweep (`conc_sweep.py`, 24 rows a level, ~4.5k-token prompts) | 0.31–0.33 rows/s and 106–114 output tok/s at every level from 4 to 32 concurrent: 1.03x from 4 to 32; 100 % ok |

Against the same benchmarks run on DGXone earlier that night under HEM and another session's
load: the soak served 3.5 times the requests (7,973 against 2,271) with a first-token p50 of
0.16 s where it had drifted from 0.26 to 1.6 s; the agentic run took 31 s against 76 s.

What does not scale: anything whose requests each carry a prompt of thousands of tokens. The
judge sweep is flat from 4 to 32 concurrent because this family reads one prompt at a time.
Decode-bound work scales (253 tok/s at 8 streams against 82–104 alone).
