# Qwen3.6-35B-A3B: the default prefill settings against the tuned ones — DGXone, 2026-10-04

> Layout note, 2026-10-04: moved from `sixlabs/bench/results/dgxone_35b_prefill_2026-10-04.md` when the per-model folders under `models/` were made. Paths to other moved records were updated; nothing else was changed. Its data folder moved with it.

One build (`17fbf92`, the first with `cecb0cd`, which lets the `qwen3_5` family take an explicit
prefill budget), one checkpoint (`nvidia/Qwen3.6-35B-A3B-NVFP4`), one box, the same three checks
run twice: the template as it was, then with

    "prefill_budget_tokens": 1024, "prefill_idle_budget_tokens": 4096,
    "decode_passes_per_prefill": 8, "prefill_order": "shortest"

The engine's own line for each:

    default  prefill budget 256 tokens/tick (0 = full prompt), 256 with nothing decoding; 1 decode pass(es) per chunk while both are in flight; equal shares
    tuned    prefill budget 1024 tokens/tick (0 = full prompt), 4096 with nothing decoding; 8 decode pass(es) per chunk while both are in flight; shortest prompt read first

Nothing else was on the GPU (HEM and the 0.8B slot down; reranker, embedder and Docling idle).
The engine was restarted between the two halves, so nothing was answered from a prefix cache.
One run each. Driver, rows and the suite's lines are in `dgxone_35b_prefill_2026-10-04/`.

## Result

The tuned settings give the same answers and remove the prompt-reading stall. They are now the
35B template's settings.

| | default | tuned |
|---|---|---|
| `group_check.py --thinking-off --seed 4242`, rows 1–4 | PASS | PASS |
| the same eight prompts one at a time, default against tuned | — | 8 of 8 texts identical; log-probabilities apart by 0.0001 and 0.0004 on average (worst 0.0028) |
| burst of eight 2,022-token prompts: first tokens (s) | 27.3 – 33.8, mean 32.25 | 0.49 – 6.31, mean 3.54 |
| the burst: all eight finished | 34.9 s | 6.73 s |
| three 3k prompts beside a streaming answer: first tokens (s) | 10.5 / 11.0 / 11.1 | 9.0 / 3.6 / 7.2 |

`mixed_load.py --thinking-off` (4 writers streaming, 2 readers sending 21,707-token prompts nobody
has sent before, 60 s a phase):

| phase | | default | tuned |
|---|---|---|---|
| writers alone | engine output tok/s | 186.1 | 187.4 |
| readers alone | prompt tokens read a second | 1,185 | 3,411 |
| readers alone | reader's first token, median | 36.5 s | 12.9 s |
| mixed | engine output tok/s | **13.1** | **95.4** |
| mixed | a writer's first token, median | **41.7 s** | **0.47 s** |
| mixed | gap between a writer's tokens, p95 / max | 276 / 699 ms | 145 / 404 ms |
| mixed | prompt tokens read a second | 1,041 | 1,322 |
| mixed | reader's first token, median | 41.3 s | 31.7 s |

ShareGPT (64 prompts a level, all 64 successful everywhere, thinking on as served):

| streams | output tok/s default | tuned | first token median / p99, ms: default | tuned | per token median / p99, ms: default | tuned |
|---|---|---|---|---|---|---|
| 1 | 96.39 | 93.19 | 153 / 503 | 153 / 366 | 9.43 / 12.33 | 9.56 / 16.03 |
| 2 | 132.53 | 128.47 | 154 / 605 | 155 / 327 | 13.23 / 19.79 | 13.91 / 25.09 |
| 4 | 171.13 | 176.67 | 229 / 845 | 180 / 409 | 20.02 / 31.03 | 19.51 / 26.66 |
| 8 | 216.27 | 222.74 | 242 / 1013 | 207 / 510 | 31.78 / 45.52 | 30.98 / 40.79 |

## What it says

- Reading is 2.9 times faster with nothing decoding (the idle budget of 4096), and a burst of
  short prompts is answered shortest first: the first of eight answers at 0.5 s instead of 27 s.
- With the default settings four people chatting were starved by two long prompts: 13 tokens a
  second between them and 42 s to a first token. With the tuned settings they keep half their
  throughput (95 of 187) and their first token comes in half a second. Long prompts under that
  load still wait about half a minute.
- ShareGPT's prompts are short, so its throughput moves by 3 % either way (down at 1 and 2 streams,
  up at 4 and 8), which one run cannot tell from noise. Its worst-case first token halves at every
  level above one stream.
- The answers do not change: the log-probabilities of the same prompts under the two settings
  agree to the third decimal.

## Not shown here

- The 122B (`Sehyo/…`, the same MoE path) has not been run with these settings; its template is
  unchanged.
- The family's dense checkpoints (Qwen3.8-27B, Qwen3.5-0.8B) compute one FP8 activation scale over
  a chunk's rows, so a larger chunk changes their numbers; their answers have to be checked before
  they take these settings. The registry's `qwen3_5` family defaults therefore stay empty.
- `group_check.py`'s pass rule counts lookups answered right. With a clock seed the 35B missed one
  of four with identical text alone and together (a model miss, not an engine one); with
  `--seed 4242` it answers all of them. The row that judges the settings is the default-against-tuned
  comparison on the same seed.
