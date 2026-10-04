# Qwen3.8-27B-FP8 on DGPP

This documents how DGPP, a C++/CUDA inference engine for DGX Spark, serves
[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) — the
dense 27B of the Qwen3.8 line (`Qwen3_5ForConditionalGeneration`, text
only here) — and, on one Spark, the
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
block drafter beside it. Neither is a republished checkpoint: the engine
loads both releases as shipped.

## The checkpoint, as loaded

28.8 GiB across 66 shards. Every projection — the Gated-DeltaNet layers'
qkv / z / out, the full-attention layers' q / k / v / o, the dense MLPs'
gate / up / down — is FP8 e4m3 with BF16 scales on 128 × 128 blocks. The
embedding, the lm head, the norms, the DeltaNet's `A_log` / `dt_bias` /
conv taps / `in_proj_a` / `in_proj_b`, and the MTP draft's `fc` are BF16.
The engine decodes every FP8 code exactly into BF16 (`bf16(code × scale)`,
one rounding) and runs activations in BF16 with fp32 accumulation. The
default chain is exact; the two lossy levers are explicit keys
(`engine.dense_weights: fp8` requantizes the lm head to block FP8 — the
templates ship it, half the head's bytes a decode row at the same
HumanEval / GSM8K; `engine.prefill_fp8_per_tensor` runs the prefill GEMMs
on cuBLASLt's per-tensor e4m3 kernels — off, it changes long-context
transcripts). The checkpoint's own BF16 K/V rows are cached as BF16: 2 KiB
a token a kv head, four kv heads of 256 at world 1.

## Architecture

64 layers at hidden size 5120: 48 Gated-DeltaNet layers (16 key heads and
48 value heads of 128, a 4-tap causal conv, the swish output gate, an fp32
recurrent state) and 16 full-attention layers (24 query heads and 4 kv
heads of 256, a `[q | gate]` projection with a sigmoid output gate,
rotate-half RoPE over 64 of the 256 dims, θ = 10⁷), each followed by a
SwiGLU MLP of 17,408; a vocabulary of 248,320; one MTP draft layer (a full-
attention layer fed `fc([norm(embed(next)) | norm(hidden)])`). The context
is 262,144 tokens. The DFlash2 drafter is a five-layer Qwen3 backbone (32
heads of 128, 8 kv heads, MLP 17,408) fed the target's residual at five
taps through a split `fc`, two-tap dynamic grouped convolutions around its
attention and MLP, and a rank-256 top-16 selector walk that proposes a
block of seven drafts from one anchor.

## Serving

**One Spark** (`cluster_qwen3.8-27b_fp8_w1`): the drafter recipe. 27.5 GiB
of weights, a 256K-token BF16 KV pool (21.0 GiB), the 3.6 GiB drafter and
its five K/V planes, eight request slots and a 1.4 GiB prefix cache plan at
62 GiB; the eager engine runs one block proposal a step (verify rows up to
eight, the multi-slot verify as a captured graph, the redrafts stacked
across slots), the drafter's bf16 matrices packed lossless 12-bit. The MTP
depth-2 world with the decode graph is the template's mode
(`--no-dflash --mtp --mtp-depth 2`), plain decode `--no-dflash`, the
drafter on the graph engine `--decode-graph` (the same transcripts, faster).

**Two and four Sparks** (`cluster_qwen3.8-27b_fp8_w2`, `_w4`): tensor
parallel — the DeltaNet and attention heads, the MLP and the lm head sliced
across the ranks, two boundary folds a layer (the attention / DeltaNet
output and the MLP output, bf16 on the wire) — MTP depth 3 with the decode
graph; 15.0 GiB of weights and a 13 GiB KV pool a rank at 256K on two
Sparks, eight slots. The DFlash2 drafter is each template's mode since
2026-10-04 (`--no-mtp --dflash-model z-lab/Qwen3.8-27B-DFlash2
--bf16-weights bf12`): the block proposal recorded inside the graph step on
every rank, the drafter's heads and MLP rows sharded across the ranks, the
ranks' top-16 lists merged through one fold ([#92](https://github.com/HawkBearPig/dgpp/issues/92)).
On the arena harness (llama-benchy pp2048 tg128 at the checkpoint's
sampling) the MTP templates read 32.2 tok/s on two nodes and 55.0 on four
(the public cards: 46.55 and 58.48), the drafter 31.3 and 50.1; by class the
drafter leads code, JSON and math by 27–55 % on four nodes and ties prose
and chat ([docs/mtp.md](../mtp.md#on-the-graph-worlds-2026-10-04-92)).

Both worlds decode through the family's own kernels: the DeltaNet chunked
and recurrent forms, the query-tiled prefill attention and split-KV decode
walk over the 256-wide heads, the fp8 GEMVs with the k = 17,408 down
projection on the streaming tensor-core form; the lossless 12-bit form of
the bf16 decode weights (`engine.bf16_weights: bf12`) packs the drafter's
layers, its fc taps and the MTP fc (−3 % a step at C1, bitwise).

## Numerics and gates

`qwen35_forward_test`: a tiny synthetic block-FP8 release from the binding
table against a numpy double reference of the whole stack and the draft
block (teacher-forced strict: per-layer residual l2 0.17–0.44 %, the final
hidden bitwise, the top-8 logits exact to 2.4e-7). `qwen35_tp_test`: worlds
2 and 4 in one process against the world-1 forward (cross-rank bitwise,
merged top-1 exact on every row). T=1 and MTP greedy transcripts are
identical within a world; the drafter world's equal an MTP depth-4 world's
(the same verify width) at world 1 and an MTP depth-5 world's at worlds 2
and 4 (4/4 each). Across worlds the bf16 wire fold moves near ties
(one of four 256-token prompts identical between worlds 1 and 2); HumanEval
39/40 and GSM8K 39/40 at both.

## Measurements

The serving numbers — single-request engine rates by prompt class,
concurrency sweeps, cold prefill at 2K / 8K / 32K, the decode modes and the
full HumanEval / GSM8K / schema evaluation — are the
[benchmarks page](../benchmarks.md)'s Qwen3.8-27B rows, from the
[2026-10-04 campaign](../../benchmarks/results/2026-10-04-qwen3.8-27b/README.md).
Physics on one GB10 (248 GB/s): a T=1 step streams 27 GB — 105 ms — and
measures 114; at world 2 a rank streams 12.2 GB (49 ms) plus the head's
reads and measures 63 T=1 / 74.5 an MTP pass, GPU-busy 98.9 % with the 135
boundary folds a pass at 4.2 ms
([#90](https://github.com/HawkBearPig/dgpp/issues/90)).
