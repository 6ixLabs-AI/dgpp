# DeepSeek-V4-Flash on DGPP

This documents how DGPP, a C++/CUDA inference engine for DGX Spark, serves
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
It is **not** a republished checkpoint: the engine loads the upstream
release as shipped, with no re-quantization pass. The community int4 and
EXL3 repositories of this model re-encode experts that are already 4-bit
and are not used.

## The checkpoint, as loaded

156 GB across 48 shards. The routed experts — 256 per layer, in the 43
layers and the three draft stages — are MXFP4 (e2m1 nibbles with an e8m0
scale per 32 along K). The attention projections, the indexer's query
projection, the shared expert and the draft's input projection are FP8
e4m3 with e8m0 scales on 128×128 blocks. The compressors, the router, the
embedding and the LM head are BF16; the hyper-connection coefficient
matrices are fp32 and are rounded to BF16 once at load. The engine decodes
every code exactly into BF16 and runs activations in BF16. The cached
attention rows hold the release's own quantize-dequantized values (FP8 on
448 dimensions, the rotated 64 in BF16), stored as BF16 rows: 1 KiB per
row, about 6 KiB per context token per rank. There is no `kv_dtype` knob
for this family.

## Architecture

43 layers at hidden size 4096 over four hyper-connection residual streams,
collapsed and re-expanded around every sublayer by a two-pass mHC. Every
layer attends with 64 heads over one shared 512-wide latent (K is V) in a
128-token sliding window with a learned per-head sink. From layer 2 on the
layers alternate two compressed-context paths: ratio 4 (overlapping groups
of eight tokens pooled into an entry every four, a 64-head indexer that
selects 512 entries per query) and ratio 128 (every entry attended). The
feed-forward is a 256-expert MoE, top 6, with one shared expert; layers
0–2 route by a token-id table instead of the learned router. The DSpark
draft appends three window-only stages that predict a block of five tokens
per decode pass, with a rank-256 Markov head chaining the block's picks
and a confidence head. The context is 1,048,576 tokens.

## Serving

Four DGX Spark nodes hold the weights at 40.6 GiB per rank (tensor
parallel, world 4: the head-sharded query and output projections, the
experts sliced on their intermediate dimension, the vocabulary-sharded LM
head); the full 1M-token pool, six request slots and a 14 GiB prefix cache
plan at 63.2 GiB per rank. Two nodes (world 2) hold 78.8 GiB of weights per
rank and plan at 95.8 GiB with the same 1M-token pool, four slots and an
8 GiB prefix cache.

Decode runs a CUDA-graph step with the DSpark draft: the ratio-128
compressors project a whole group once, when it completes (the open
group's layer inputs wait in a ring), and the draft block's logits come
off a block-FP8 copy of the LM head with every candidate of a pick
recomputed from the BF16 rows, so the drafts are the BF16 head's. The templates schedule
the verify depth from the draft's confidence head (`engine.mtp_schedule`
over the block's five drafts): a pass verifies the leading drafts whose
survival pays for another row, at any depth from one to five — mostly two
or three on prose, the whole block on code and JSON. Six concurrent
requests at the full block would be 36 rows against the family's 32-row
decode batch; they batch at up to four drafts each instead. A sampled request's drafts are the draft head's argmax,
accepted with the target's probability of that token (`engine.mtp_draft`,
`greedy` for this family) — exact for the output distribution, and
measured ahead of drawing the drafts — and follow the same schedule at 0.93
of the head's acceptance (`engine.mtp_schedule_sampled_scale`). A greedy
request's transcript is plain decode's at every depth.

Prefill runs in 4,096-row forwards and is bitwise the same for any
chunking of a prompt, so prefix snapshots and grouped admissions reuse it
exactly. On a multi-node world a forward of 512 rows or more runs each
layer in two row blocks so a block's all-reduce runs under the other
block's compute. Prompts are read in through resumable cursors: the
in-flight prompts' next chunks ride one forward per scheduler tick
(prompts that arrive together are read in together and start decoding
together; a later arrival interleaves with the running decodes at a
256-token quantum per reading prompt, the whole forward when nothing
decodes — `engine.prefill_budget_tokens`, `engine.admission_gather_ms`). The tokenizer is the release's byte-level BPE; the prompt renderer
is a port of the checkpoint's own `encoding/encoding_dsv4.py` (there is no
Jinja template), including its three reasoning-effort preambles (the
service's `low` is the model's default); tool calls are emitted and
constrained in the model's DSML tag format.

## Measured

Four nodes: 65.4 / 113.3 / 125.9 / 94.2 / 70.3 engine tokens/s for one
greedy request on prose / code / JSON / math / chat (43.6 with the draft
off), 134–186 tokens/s of completed output across six requests, cold
prefill 1.36 / 5.47 / 28.0 s at 2K / 8K / 34K tokens, and 75 tokens/s at a
121K-token context. Two nodes: 40.4 / 61.7 / 72.8 / 58.6 / 43.7 for one
request (29.9 plain), 68–101 across four, prefill 2.03 / 8.02 / 38.0 s.
Under llama-benchy (pp2048 / tg128, the checkpoint's default sampling) the
four-node deployment generates 64.2 tokens/s for one request (40 runs)
and, with each request's prompt processing time subtracted
(`--latency-mode generation`), 90.6 / 121.1 / 66.4 tokens/s in total at
two, five and ten concurrent requests (ten queue behind the six slots).
See
[the benchmark tables](../benchmarks.md) for the current numbers and
[the campaign record](../../benchmarks/results/2026-10-01-deepseek-v4-flash/README.md)
for the evidence. The forward pass is cross-checked layer by layer against
the release's own `inference/model.py` on the real checkpoint
(`tools/dsv4_torch_reference.py`).

## License and attribution

The model, its weights, tokenizer and prompt encoder are DeepSeek's; see
the upstream model card for its license and terms. This file describes
only the engine's support and adds no weights.
