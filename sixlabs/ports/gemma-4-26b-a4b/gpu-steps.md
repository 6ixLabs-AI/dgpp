# Gemma-4-26B-A4B (`bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16`): what to run once a box is free

Read [the 31B's steps](../gemma-4-31b/gpu-steps.md) first: this is the same family, the same code
and the same state — **host side written and tested on a Mac, nothing compiled with nvcc or GCC, no
weight loaded, no serving engine; `6ix-Serve` refuses the checkpoint by name.** This file lists
only what differs.

| | |
|---|---|
| checkpoint | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16` @ `main` (no revision pinned: record the snapshot's sha when it is downloaded), 3 shards, 16.4 GB on disk, not gated. A **community** quantization of `google/gemma-4-26B-A4B-it` (modelopt 0.43, weight-only NVFP4), not an NVIDIA or Google release |
| what the model is | NOT "26B with about 4B active behind a routed MoE in place of the MLP". Thirty layers of width 2816; every layer has a dense GeGLU MLP (2112 wide) **and**, beside it, a routed MoE of 128 experts (704 wide, top 8), each branch under its own norm, summed. The model card's figures are 25.2B total, 3.8B active |
| container | attention projections, dense MLP and all 11,520 expert matrices NVFP4 with **no** `input_scale`; routers (`proj`, `scale`, `per_expert_scale`), norms and the embedding BF16; the experts stored one `Linear` each under `layers.L.moe.experts.E.{gate,up,down}_proj` (the class's stacked `experts.gate_up_proj` split by the release's quantizer: gate = rows 0..703) |
| template | this release ships an **earlier chat template** than the 31B's (hash `230e3de90b279789` against `c9fa6ee62cd50023`): a role `tool` message is a turn of its own (`<|turn>tool`), not a `<|tool_response>` folded into the assistant's turn; reasoning is never rendered back; a system message whose content is a list of parts prints as that list's Python repr. The engine reproduces it byte for byte — including the last — because that is what transformers does with this file. If the 31B's behaviour is wanted, that is a decision to replace the release's template, not a bug to fix here |
| tokenizer | identical to the 31B's (same file, same hash) |

`$IN` = `$REPO/sixlabs/ports/gemma-4-31b/inputs` — the two templates render the two shared prompts
to the same 84 and 114 ids, so the 31B's input files serve both.

## 1. Host tests

As the 31B's step 1; the seven tests this port adds are in the same filter
(`gemma4_config_parses_the_26b_a4b_release`, `gemma4_config_moe_block_is_checked`,
`gemma4_binding_table_has_the_26b_a4b_release_shape`, `gemma4_binding_26b_a4b_formats`,
`gemma4_ref_route_…`, `gemma4_ref_moe_checkpoint_is_the_python_one`,
`gemma4_ref_moe_forward_matches_the_numpy_reference`).

## 2. Checkpoint, tokenizer and template

```bash
hf download bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16        # 16.4 GB
$REPO/build-ci/gemma4_bind_check --model bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16
ctest --test-dir build-ci -R 'gemma4_26b_chat_test' --output-on-failure
```

Expected from the bind check (produced on the Mac by the app itself, 2026-10-04, from the real
config.json and header-only stand-ins of the three shards: the real header bytes, no weights):

```
config: gemma4, 30 layers (25 sliding + 5 full), hidden 2816, vocab 262144, recipe nvfp4-weight-only
attention: 16 heads; sliding 8 KV heads x 256, window 1024, 128 rotary pairs (theta 10000); full 2 KV heads x 512, 64 rotary pairs (theta 1e+06), value from k_proj
mlp: dense GeGLU, intermediate 2112; embedding scale 53; logit soft-cap 30
moe: 128 experts top-8 beside the dense MLP, intermediate 704, router input scale 0.018798828125
checkpoint: 3 shards, 35923 tensors in headers
binding: expected 35567 | matched 35567 (missing 0, dtype 0, shape 0) | unexpected 0 | ignored (encoders) 356
matrices: 11725 NVFP4 (11520 of them experts), 31 BF16; matched bytes 15271399280
table 35567 + ignored 356 = 35923; headers 35923
binding OK: every tensor of the table is present with its dtype and shape
```

If the repository has moved since, the tensor count or a shape will say so here.

Expected from the template test (ran on the Mac against this release's files):

```
gemma4_chat_test: 30 renders byte-exact, 30 id sequences equal (model bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16, template 230e3de90b279789)
gemma4_chat_test: 10 tool-call turns (13 calls) round-trip render -> encode -> parse; 1 undecidable block flushed as text
```

## 3–5. Reference, host check, draft forward

The 31B's steps 3, 4 and 5 with `--ckpt $SNAP` / `--model bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16`:

```bash
python3 $REPO/tools/gemma4_reference.py score --ckpt $SNAP --ids-json $IN/gemma4-ids-prose.json --out $OUT/ref26-prose.json
$REPO/build-ci/gemma4_host_check --model bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16 \
      --ids-json $IN/gemma4-ids-prose.json --layers 4 > $OUT/host26-prose-l4.txt
python3 $REPO/tools/gemma4_reference.py score --ckpt $SNAP --ids-json $IN/gemma4-ids-prose.json --layers 4 --out $OUT/ref26-prose-l4.json
python3 $REPO/sixlabs/bench/compare_forward_check.py $OUT/host26-prose-l4.txt $OUT/ref26-prose-l4.json --mean 0.01 --max 0.05
$REPO/build-gemma/gemma4_forward_check --model bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16 \
      --ids-json $IN/gemma4-ids-prose.json > $OUT/fc26-prose.txt
python3 $REPO/sixlabs/bench/compare_forward_check.py $OUT/fc26-prose.txt $OUT/ref26-prose.json --mean 0.01 --max 0.05
```

Sizes: the reference dequantizes a layer's attention and MLP (about 0.2 GB of fp32) and only the
experts a prompt's tokens pick; the host check loads **every** expert of the layers it runs (about
3.3 GB of fp32 a layer, 3 GB of it experts: use `--layers`, and `--layers 4` is about 16 GB with the
embedding); the draft forward uploads the checkpoint's own 14.2 GiB.

What is particular to this model when a number disagrees:

- **A different pick is not automatically a bug.** The router's top 8 of 128 can sit on a near tie;
  two fp32 implementations may then pick differently at one token and the outputs part by more than
  rounding. `tools/gemma4_reference.py dump --detail-layers L` writes `LNN_router_logits` and
  `LNN_router_ids`: look at the gap between the 8th and 9th logit at the token where the layers'
  `h_NN` lines first differ before concluding anything. (On the synthetic fixture the smallest such
  gap is 4e-4 and all three implementations pick alike.)
- **The two bf16 constants.** The reference multiplies the embedding by bf16(sqrt(2816)) = 53.0 and
  the router's input by bf16(2816^-0.5) = 0.018798828125, because that is what both the transformers
  class and the vLLM model file this release ships do on a bf16 checkpoint (verified with torch for
  the scalars; the exact values would be 53.066 and 0.0188445). `--variant embscale=exact` and
  `--variant routerscale=exact` show how much each moves.
- **The renormalization of the top-8 weights cannot be seen in the output**: it multiplies a token's
  expert sum by one factor, and `post_feedforward_layernorm_2` divides it straight back out. An
  engine may skip it; the reference keeps it because the class does.

## What the serving engine could reuse here (not done)

The routed part has the shape of the engine's existing MoE (`models/glm/moe_layer.hpp` with
`MoeRouterMode::SoftmaxTopk`, the NVFP4 expert matrices of `models/qwen/moe_layer.hpp`), after two
foldings at load — `router.scale * H^-0.5` into the router matrix's columns, with the weightless
norm in front; `per_expert_scale[e]` into expert e's `down_proj` scale — and with one thing those
kernels do not have: the expert activation is GeGLU (`gelu_tanh`), where every fused gate/up kernel
there is SiLU. There is no shared expert; the dense MLP beside the experts is a separate branch
under its own norm, not a shared expert's weighted sum. None of this is written or tested.

## Not verified (beyond the 31B's list)

1. Anything on this release's weights. Its header layout was read; its tensor bodies never were.
2. That the unfused experts are split gate-first. This comes from the release's own
   `quantize_gemma4_moe.py` (`gate_proj <- gate_up_proj[:, :I]`), and the reference's cross-check
   against the transformers class stacks them back the same way — on synthetic weights. If the
   reference scores this checkpoint near −12.5 per token while the 31B scores well, swap the two
   names first.
3. The release itself: it is a community quantization whose card says it was validated under vLLM
   with a patched model file. Whether its numbers are good is not something this port can know.
