# Qwen3-Next-80B over two boxes with YaRN ×4 — DGXone + DGXtwo, 2026-10-04

Status: **it starts, plans, fits and answers. Recall at long lengths with ×4 has NOT been
tested.** One short answer was timed. The figures below were reported by the session that ran the
test on the evening of 2026-10-04 and are written down here as reported; **no log of this test is
in the repository.**

## Config

The two-box template (`deploy/cluster_qwen3-next-80b_nvfp4_w2.example.json`) plus:

```json
"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144},
"kv_capacity": 1089536,
"max_concurrency": 8
```

That is YaRN ×4 over the checkpoint's 262,144 positions, a 1,089,536-token pool and 8 seats
(the keys are written here as the one-box YaRN template writes them).
There is no template for this in `deploy/`; the one YaRN template for this model is the one-box
×2 one (`deploy/cluster_qwen3-next-80b_nvfp4_w1_yarn512k.example.json`).

## Result

| | |
|---|---|
| Request context limit | 1,048,576 tokens |
| Memory plan, each box | 45.13 GiB (40.19 device + 4.94 pinned) |
| Memory measured, each box | 45,555 MiB |
| Warm start | 12 s |
| Speed | 128 tok/s on a 177-token answer |

For comparison, the same model on two boxes without YaRN (524,288-token pool) plans 38.12 GiB a
box and measured 38,373 MiB
([two-box record](../../shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md)).

## What this does not show

- **Recall at long lengths with ×4 has not been tested.** Nothing here says the model answers
  correctly from a prompt of several hundred thousand tokens at this setting.
- The only YaRN recall point from that night is ×2 on one box: a 302,368-token prompt, 5 of 5
  planted facts returned, first token after 277 s.
- The speed is one answer of 177 tokens, not a benchmark. No ShareGPT, concurrency or
  long-prompt figure was reported for this setting, and no cold start.
- The two-box world is measured, not verified: the forward check has no two-rank mode (see the
  two-box record).

## Linked from

[Qwen3-Next-80B](../README.md).
