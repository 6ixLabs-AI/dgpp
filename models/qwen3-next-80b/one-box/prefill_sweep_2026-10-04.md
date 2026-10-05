# Answers beside long prompts: scheduler settings, Qwen3-Next-80B (2026-10-04)

> Layout note, 2026-10-04: moved from `sixlabs/bench/results/prefill_sweep_2026-10-04.md` when the per-model folders under `models/` were made. Paths to other moved records were updated; nothing else was changed.

`sixlabs/bench/prefill_sweep.sh` on DGXone, 05:23–05:46 EDT, build 5c20f70, 8 slots, 524,288-token
pool, fp8 dense, MTP depth 2. `mixed_load.py`: 4 writers streaming long answers, 2 readers sending
fresh 17,400-token prompts, 50 s a phase. Every run clean: no HEM scoring call during it and no
other process above 5 % of the GPU. One run per setting.

| `prefill_budget_tokens` / `decode_passes_per_prefill` / order | 4 answers alone, tok/s each | 2 prompts alone, tok/s read (first token) | together: answers, tok/s each | pause p95 / max, ms | together: prompts, tok/s read (first token) |
|---|---|---|---|---|---|
| 4096 / 1 / fair (the setting before 2026-10-04) | 30.5 (first run after boot) | 2,811 (12.0 s) | 2.0 | 1,477 / 2,078 | 2,662 (11.5 s) |
| 256 / 4 / shortest | 36.3 | 2,768 (12.3 s) | 18.3 | 227 / 302 | 878 (30.2 s) |
| 512 / 4 / shortest | 35.3 | 2,587 (12.8 s) | 12.5 | 351 / 498 | 1,118 (27.7 s) |
| 1024 / 4 / shortest | 35.7 | 2,839 (11.7 s) | 11.8 | 453 / 579 | 1,444 (24.5 s) |
| 256 / 2 / shortest | 35.3 | 2,824 (11.9 s) | 12.6 | 248 / 277 | 1,143 (27.3 s) |
| **1024 / 8 / shortest (chosen)** | 35.4 | 2,912 (11.6 s) | **19.5** | 224 / 521 | **1,204 (27.4 s)** |
| 2048 / 8 / shortest | 35.9 | 2,849 (11.5 s) | 14.3 | 176 / 850 | 1,435 (23.5 s) |

`prefill_idle_budget_tokens` is 4096 in every row but the first (where one budget serves both).

Reading it:
- The old setting gives the prompts everything: answers in progress fall from 30–36 to 2 tokens/s
  each, with pauses of 1.5–2 s.
- 1024 / 8 is better than 256 / 4 on both sides — answers 19.5 against 18.3 tokens/s, prompts read
  37 % faster — because a 1024-token chunk carries less fixed cost per token than a 256-token
  one. Its longest pause is half a second against 0.3 s.
- Nothing here gets both. With answers at about half their solo speed, prompts are read at about
  40 % of theirs, and a 17k-token prompt arriving beside four streaming answers waits 25–30 s for
  its first token against 12 s alone. The lever that would change that is reading several
  prompts in one walk (`kPrefillGroupAdvance`), which this family does not have.
- Not tried: more than 8 passes; a time slice instead of a token count.
