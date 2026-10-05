# Qwen3.6-35B-A3B on DGX Spark: 97 tok/s and 10 s load on one Spark, 154 tok/s and 6 s load on two

Thanks to Stephen Hawkins (@stephen.douglas.hawkins), who wrote DGPP, the engine these results
come from, and to Thomas Braun (@tbraun96) of Avarok Cybersecurity, whose Atlas is the engine
they are compared with.

The model is `nvidia/Qwen3.6-35B-A3B-NVFP4` on [a fork of DGPP](https://github.com/6ixLabs-AI/dgpp) that adds it,
with 8 concurrent seats, thinking on and the checkpoint's draft head at depth 2. Measured on 2026-10-04, one run
each. The title figures are one stream and a warm start.

## Speed

Output tok/s, all streams together. The last two columns are a
different benchmark (`bench_decode.py`), so compare them only with each other:

| Streams | Atlas, one Spark | This fork, one Spark | This fork, two Sparks | vLLM FP8, one Spark (`bench_decode.py`) | This fork, one Spark (`bench_decode.py`) |
|---|---|---|---|---|---|
| 1 | 84.76 | 96.72 | 154.43 | 50.1 | 97.7 |
| 2 | 112.21 | 135.37 | 214.11 | 62.6 | 128.8 |
| 4 | 108.89 | 182.37 | 300.46 | 105.9 | 194.6 |
| 8 | 127.12 | 220.11 | 367.57 | 190.0 | 271.3 |

- Two Sparks give about 1.6 times one Spark at every level.
- **Against Atlas**, on the same checkpoint, this fork is 1.14 times faster at one stream and
  1.73 times at eight.
- **Against vLLM**, this fork is 1.95 times faster at one stream and 1.43 times at eight.
- First token on two Sparks: 102 ms median at one stream, 142 ms at eight.

## Start time and memory

| | This fork, one Spark | This fork, two Sparks | Atlas, one Spark | vLLM, one Spark |
|---|---|---|---|---|
| Cold start | 47.5 s | 60 s | 49 s | 404 s |
| Warm start | 10 s | 6 s | no figure | no figure |
| GPU memory | 38.6 GiB | 22.4 GiB on each | 72.1 GiB | 44.5 GiB |

A warm start reuses the resident image DGPP keeps from the first load.

## What I changed in DGPP

The code is at https://github.com/6ixLabs-AI/dgpp.

The 35B went in as a second routed-expert model on DGPP's `qwen3_5` family (256 experts, 8 used
per token), after the Qwen3-Next-80B port: about 1,800 lines in 23 files including its tests.
