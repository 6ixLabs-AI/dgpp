# HEM sharing the boxes with the two-box 80B and 35B — DGXone + DGXtwo, 2026-10-04

Status: **measured, short.** One 20-second window per model per case. The figures below were
reported by the session that ran the test on the evening of 2026-10-04 and are written down here
as reported; **no log of this test is in the repository.**

HEM is the fleet's hallucination scorer, a small model. The question was what it costs the two
chat models when it scores on the same boxes, and whether moving it to the CPU is a way out.

## Setup

- Qwen3-Next-80B and Qwen3.6-35B-A3B, both served over two boxes (the two-box worlds of
  [dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md](dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md)).
- The models were measured at one stream.
- Scoring load: two scoring requests in flight per scorer, each 16 sentences against an
  8,000-character document, continuously.
- One 20 s window per model per case.

## Result

Single-stream speed with no scoring running: **80B 123.5 tok/s, 35B 145.0 tok/s.**

| Where HEM scores | 80B, share of its no-scoring speed | 35B, share of its no-scoring speed | Scores a minute | Seconds a score |
|---|---|---|---|---|
| nowhere (baseline) | 100 % (123.5 tok/s) | 100 % (145.0 tok/s) | — | — |
| DGXtwo's GPU only | 41 % | 41 % | 12.1 | 9.6 |
| the GPU of both boxes | 26 % | 30 % | 23.0 | 10.1 |
| DGXtwo's CPU only (`HEM_DEVICE=cpu`) | 99 % | 98 % | 1.6 | 60.6 |
| the CPU of both boxes | 56 % | 57 % | 2.3 | 80.6 |

One score on DGXtwo's GPU with both models idle: **2.9 s** (four runs).

## How far to trust it

- One 20 s window per model per case: no run-to-run spread.
- The CPU cases completed only 2 and 5 scores in their windows, so their rates and times rest on
  very few samples.
- The scoring load is continuous and two requests deep per scorer: scoring never stops while
  the model is timed.
- Only the five cases in the table were reported. Nothing here covers more than one stream on the
  models, or DGXone's GPU alone.

## Reading the table

- Continuous scoring on one GPU leaves each model 41 % of its speed; on both GPUs, 26 % and 30 %.
- Scoring on one box's CPU leaves the models at 98–99 %, and a score takes 60.6 s instead of
  9.6 s. Scoring on both boxes' CPUs leaves the models at 56–57 % and a score takes 80.6 s.
- With both models idle a score on DGXtwo's GPU takes 2.9 s.

## Conclusion (the owner's)

- For single-user chat, keep HEM on DGXtwo's GPU: the models are idle when scoring runs.
- Keep background scoring off the GPUs while the models are busy.

## Linked from

[Qwen3-Next-80B](../qwen3-next-80b/README.md), [Qwen3.6-35B-A3B](../qwen3.6-35b-a3b/README.md).
