# Loadout switch times with the 80B and the 35B as one loadout — DGXone + DGXtwo, 2026-10-04

Status: **measured, two runs each.** The figures below were reported by the session that ran the
test on the evening of 2026-10-04 and are written down here as reported. Raw data is on DGXone in
`/home/mark/switch-bench/`; **it is not in this repository.**

"Pair" is one loadout holding both models, Qwen3-Next-80B and Qwen3.6-35B-A3B on 6ix.cpp, on
DGXone + DGXtwo (their two-box record is
[dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md](dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md)).
"Qwen3.8 on DGPP" is the fleet's Qwen3.8 loadout, which the production DGPP checkout serves
(`sixlabs/README.md`).

## Result

Seconds, the range over two runs:

| Switch | Until the models answer | Until the switch job completes |
|---|---|---|
| nothing loaded → pair | 29–30 | 45–48 |
| pair → Qwen3.8 on DGPP | 50–69 | 68–86 |
| Qwen3.8 on DGPP → pair | 41–51 | 92–104 |

No-op switches: 0.1–0.5 s.

## How it was driven

Through the fleet agent's route handlers, not by signed requests from the app. The app's own
path (signing, the front door) is therefore not in these times.

## How far to trust it

- Two runs each; the table gives the range, not a median.
- The report does not say whether a resident image was present for each start. For reference,
  the two-box record has each model's start alone: first load 144 s and warm 10 s for the 80B,
  60 s and 6 s for the 35B.
- Nothing was reported for a switch while a request is in flight, or for the app's signed path.

## Linked from

[Qwen3-Next-80B](../qwen3-next-80b/README.md), [Qwen3.6-35B-A3B](../qwen3.6-35b-a3b/README.md),
[Qwen3.8-Flash-Next](../qwen3.8-flash-next/README.md).
