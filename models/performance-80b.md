# Qwen3-Next-80B on DGX Spark: 77 tok/s and 12 s load on one Spark, 121 tok/s and 10 s load on two

Thanks to Stephen Hawkins (@stephen.douglas.hawkins), who wrote DGPP, the engine these results
come from, and to Thomas Braun (@tbraun96) of Avarok Cybersecurity, whose Atlas is the engine
they are compared with and the first to show this model running fast on a Spark.

The model is `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` on [a fork of DGPP](https://github.com/6ixLabs-AI/dgpp) that adds it,
with 8 concurrent seats and the checkpoint's draft head at depth 2. Measured on 2026-10-04, one run
each. The title figures are one stream and a warm start.

## Speed

Output tok/s, all streams together. The last two columns are a different benchmark
(`bench_decode.py`), so compare them only with each other:

| Streams | vLLM FP8, one Spark | This fork, one Spark | This fork, two Sparks | Atlas, one Spark (`bench_decode.py`) | This fork, one Spark (`bench_decode.py`) |
|---|---|---|---|---|---|
| 1 | 43.16 | 76.73 | 121.31 | 80.9 | 82.2 |
| 2 | no figure | 97.52 | 161.51 | 108.4 | 118.5 |
| 4 | 87.65 | 127.69 | 214.03 | 120.1 | 185.4 |
| 8 | 110.14 | 153.95 | 251.45 | not run | 253.4 |

- Two Sparks give about 1.6 times one Spark at every level.
- **Against vLLM**, this fork is 1.8 times faster at one stream and 1.4 times at eight on one
  Spark, and 2.8 and 2.3 times on two. There is no vLLM figure on two Sparks.
- **Against Atlas**, on the same checkpoint, this fork is level at one stream and 1.5 times
  faster at four. The one-stream row is prose; on code Atlas is ahead, 115.5 against 103.6.
  Atlas was set to 4 seats, so it has no eight-stream figure.
- On two Sparks Atlas gave 78 tok/s, one request at a time, in a different test on 2026-10-02.
- First token on two Sparks: 128 ms median at one stream, 204 ms at eight.

## Start time and memory

| | This fork, one Spark | This fork, two Sparks | Atlas, one Spark |
|---|---|---|---|
| Cold start | about 80 s | 144 s | 96 s at most |
| Warm start | 12 s | 10 s | no figure |
| GPU memory | 68.9 GiB | 37.5 GiB on each | 98.8 GiB |

A warm start reuses the resident image DGPP keeps from the first load.

## What I changed in DGPP

The code is at https://github.com/6ixLabs-AI/dgpp.

DGPP did not serve this model. It already had a `qwen3_5` model family and, for
Qwen3.8-Flash-Next, a routed mixture-of-experts layer. The 80B is close to both, so I added it
as a new dialect of `qwen3_5` with the routed experts in the dense MLP's place, not as a new
family. The first working version was one change of about 7,400 lines in 32 files: about 1,800
of engine code, the rest tests, checks and a reference model.

- **Config and tensor table** (`config35`, `binding35`): the flat `qwen3_next` config and the
  checkpoint's 297,728 tensors.
- **Loader** (`loader35`): the fused linear-attention projections, NVIDIA's NVFP4 layout, the
  experts packed as NVFP4, and the dense projections re-encoded to 8 bits at load.
- **Model** (`model35`): the routed expert layer in the forward pass, the draft-head pass that
  gives the speed, and the memory plan.
- **Tool calls** (`tool_parser`, `tool_grammar`): the JSON form this model's chat template uses.
- **Checks**: a pure-numpy reference of the model that reads the same checkpoint
  (`tools/qwen3next_reference.py`), a bind check, and a forward check that compares the
  engine's log-probabilities with the reference.

Changes after that:

- **Two Sparks.** Upstream DGPP added two-Spark serving for this family's dense model. I
  extended it to the routed-expert models, about 40 lines in `model35.cpp`.
- **Scheduler.** Answers keep streaming while long prompts are read, several prompts can be read
  in one pass, and a live request evicts cached prefixes before anything is dropped.
- **Long context.** YaRN for this model, up to 1,048,576 tokens in one request, off by default.
- **Chat content.** Message content sent as an array of text parts is joined before the chat
  template sees it. The template rendered it as an empty prompt.
