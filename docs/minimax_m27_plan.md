# MiniMax-M2.7 (lukealonso/MiniMax-M2.7-NVFP4) on 6ix.cpp — architecture facts, host references and the two-rank sharding plan (2026-10-04)

Status: **host side only.** The config parser, the expected-tensor table and the host
references are built in the default tree and unit-tested on a Mac; the binding table
equals the real checkpoint's safetensors headers (read from Hugging Face, no weight
bodies). The numpy reference agrees with transformers' `MiniMaxM2ForCausalLM` on a
synthetic model of the same class. Nothing here has been compiled by nvcc, loaded a
weight or produced a token. There is no model assembly and no serve family: the
server refuses the family by name. §4 is the plan a multi-rank assembly should
follow; §3's draft kernel (`-DDGPP_BUILD_MINIMAX_DRAFT=ON`) is unverified.

Sources: the checkpoint's `config.json`, `hf_quant_config.json`,
`model.safetensors.index.json`, the headers of its 26 indexed shards, and the
`modeling_minimax_m2.py` / `configuration_minimax_m2.py` it ships (identical to
`MiniMaxAI/MiniMax-M2.7`'s); transformers 5.8.1's `models/minimax_m2`; the base
repository's deployment guides.

## 0. Summary

MiniMax-M2.7 (`MiniMaxM2ForCausalLM`, `model_type minimax_m2`) is a 62-layer
dense-attention MoE with no dense layers: every layer is grouped-query attention
(48 query heads over 8 K/V heads of 128, no biases, half-split partial RoPE on the
first 64 dims, θ = 5e6) and a routed MoE of 256 experts (top-8, sigmoid scores plus
`e_score_correction_bias` for the selection, the picked scores renormalized, no
scaling factor, **no shared expert**). Hidden 3072, vocab 200 064, 196 608 positions,
RMSNorm eps 1e-6, an untied head. About 229 B parameters, 10 B active.

It is GLM-4.7's shape (docs/glm47_plan.md) with four differences:

1. **The q/k norm is per layer** (`qk_norm_type: per_layer`): one RMSNorm over the
   whole q projection (6144 = 48 × 128) and one over the whole k projection (1024),
   a weight per element, applied before the split into heads. GLM-4.7 norms each
   head alone with one shared `[128]` weight. This is the family's one new operator
   and the one thing tensor parallelism has to work around (§4.2).
2. No attention biases, no shared expert, no dense layers, `routed_scaling_factor` 1.
3. The router bias is BF16 in the file (GLM-4.7's is F32).
4. **There is no MTP head.** `config.json` says `use_mtp: true`, `num_mtp_modules: 3`,
   `mtp_transformer_layers: 1`; neither this release nor `MiniMaxAI/MiniMax-M2.7`
   ships a draft tensor and the published modeling code defines no such module.

The release quantizes only the routed experts (modelopt 0.39 NVFP4); everything else
is BF16. The base release is FP8 throughout (attention included); this one was made
by converting that to BF16 and requantizing the experts.

**It needs two Sparks.** The weights are 125.17 GiB as stored; a Spark has 121.6 GiB.

## 1. The model

### 1.1 Configuration facts

| field | value |
|---|---|
| `architectures` / `model_type` | `MiniMaxM2ForCausalLM` / `minimax_m2` (flat, no `text_config`) |
| layers | 62, all attention + MoE (`attn_type_list`: 62 × 1; `sliding_window` null) |
| hidden / vocab / eps | 3072 / 200 064 / 1e-6; `hidden_act` silu; `tie_word_embeddings` false; 196 608 positions |
| attention | 48 heads, 8 kv heads, `head_dim` 128, no biases, `use_qk_norm` true, `qk_norm_type` per_layer, `rotary_dim` 64 = `partial_rotary_factor` 0.5 × 128, `rope_theta` 5e6, rope type default |
| MoE | `num_local_experts` 256, `num_experts_per_tok` 8, `scoring_func` sigmoid, `use_routing_bias` true, `intermediate_size` 1536 (an expert's width), `shared_intermediate_size` 0 |
| declared, absent | `use_mtp` true, `num_mtp_modules` 3, `mtp_transformer_layers` 1 — no tensors, no code |
| tokens | `config.json`: bos 1, eos 2 — **byte tokens, not control tokens**. `generation_config.json`: bos 200019 (`]~b]`), eos 200020 (`[e~[`). `tokenizer_config.json`: bos `]~!b[` (200034), eos `[e~[`. Serving must take its stop id from the generation config / tokenizer, never from `config.json` |
| generation defaults | temperature 1.0, top_p 0.95, top_k 40 |
| quantization | modelopt NVFP4 (`quant_algo` NVFP4, group 16, weights and input activations 4-bit float); `ignore`: `lm_head`, every `model.layers.L.self_attn*`, every `model.layers.L.block_sparse_moe.gate` |

### 1.2 One layer (the checkpoint's `MiniMaxM2DecoderLayer`)

```
h   = rms(x, input_layernorm)
q   = rms(q_proj h, q_norm)   # over all 6144 outputs — every head under one RMS
k   = rms(k_proj h, k_norm)   # over all 1024 outputs
v   = v_proj h
q, k: RoPE on dims 0..63 of each head, pairs (i, i + 32), angle = pos · θ^(−i/32)
a   = softmax(q kᵀ / √128, causal) v      # query head h reads K/V head h // 6
x  += o_proj(concat_h a)
h   = rms(x, post_attention_layernorm)
s   = sigmoid(gate h);  top-8 of s + e_score_correction_bias (ties: lower id)
w   = s[top] / Σ s[top]
x  += Σ_e w_e · w2_e(silu(w1_e h) · w3_e h)   # w1 gate, w3 up, w2 down
```

`logits = lm_head(rms(x, model.norm))`.

### 1.3 Things the third-party description had wrong or left out

- "an MTP draft head": none exists (above).
- "attention": the per-layer q/k norm is not GLM-4.7's and is not optional.
- The repository holds **safetensors files that are not the model** (checked from
  their headers): ten `model-*inputscales*.safetensors` variants that repeat the
  47 598 `input_scale` names (other calibrations; bodies not read), eight `amax*.safetensors` holding
  calibration state under other names (`input_quantizer`, `weight_quantizer`,
  `k_bmm_quantizer`, …), and three `*.bak` copies (10.9 GB). Only the 26 files
  `model.safetensors.index.json` names are the checkpoint. A loader that globs
  `*.safetensors` (as the other families' do) binds duplicates. `minimax_shard_files()`
  reads the index; `minimax_bind_check` reports how many other files it ignored.
- Download only what is needed (134.4 GB instead of 145.5 GB):
  `hf download lukealonso/MiniMax-M2.7-NVFP4 --include "model-000*-of-00025.safetensors" "model-inputscales.safetensors" "*.json" "*.jinja" "*.txt"`.

## 2. Tensor census and the binding table

From the headers at `db821d7a` (HTTP range requests, 2026-10-04):

| tensors | count | bytes |
|---|---|---|
| 25 model shards | 143 471 | 134 401 534 976 |
| `model-inputscales.safetensors` (activation scales) | 47 598 | 190 392 |
| **total = `model.safetensors.index.json`** | **191 069** | 134 401 725 368 |

Per layer (2314 required tensors): two layer norms, `self_attn.{q,k,v,o}_proj`,
`self_attn.{q,k}_norm` (`[6144]`, `[1024]`), `block_sparse_moe.gate`,
`block_sparse_moe.e_score_correction_bias` (BF16 `[256]`), and per expert `w1`,
`w2`, `w3` each as `weight` U8 `[N, K/2]`, `weight_scale` F8_E4M3 `[N, K/16]`,
`weight_scale_2` F32 `[]`. Globals: `model.embed_tokens.weight`, `model.norm.weight`,
`lm_head.weight`.

`input_scale` F32 `[]` is **optional** in the table: six experts (layer 0: 87, 198;
layer 61: 24, 75, 183, 225) have none — the calibration never routed to them — so
18 of the 47 616 possible are absent. A present one must be an F32 scalar.

`minimax_validate_binding` on the real headers: required 143 471, matched 143 471,
activation scales 47 598 present / 18 absent, nothing unexpected, bound 191 069.

| class | bytes as stored | GiB |
|---|---|---|
| routed experts (NVFP4 payload + block scales + scalars) | 126 382 958 592 | 117.70 |
| attention (BF16): q_proj 2.18, o_proj 2.18, k_proj 0.36, v_proj 0.36, norms | 5 461 872 640 | 5.09 |
| embedding | 1 229 193 216 | 1.14 |
| head | 1 229 193 216 | 1.14 |
| routers, biases, norms | 98 317 312 | 0.09 |
| **the model** | **134 401 534 976** | **125.17** |

## 3. Numerics, reuse and what is new

The references the engine's numerics are pinned to: `tools/minimax_m2_reference.py`
(full precision; agrees with transformers 5.8.1's `MiniMaxM2ForCausalLM` to fp32
rounding, 1e-6 on logits spanning 4.5, on a synthetic model — `tools/minimax_m2_synth.py
hf-check`) and `models/minimax/attn_reference.hpp` (the device's rounding points;
within 0.6 % of the numpy reference at every stage — `tests/unit/minimax_attn_test.cpp`).

| step | what serves it | state |
|---|---|---|
| q/k/v projections, o_proj, head | the bf16 GEMV / GEMM interface, as GLM-4.7 | existing |
| **q/k finish: per-layer norm + RoPE + K/V append** | `glm4_qkv_finish` norms per head — a different function (0.22 relative error on the test vectors). New: `minimax_qk_rstd` + `minimax_qkv_finish` (`kernels/minimax_attn.hpp`) | **draft, unverified** |
| paged split-KV GQA attention, combine | `glm4_attn_partial`, `glm4_attn_combine` unchanged (48 / 8 heads, six per K/V head) | existing |
| K/V pool | GLM-4.7's (`models/glm4/kv_pool`), no draft layer | existing |
| router | `MoeRouterMode::SigmoidBias`; the bias converted BF16 → F32 at load; picks and weights equal the numpy reference's on the test vectors | existing |
| routed experts | the NVFP4 slot kernels with `n_shared_experts` 0 and `routed_scaling_factor` 1; the global is `1 / weight_scale_2` (the engine's divisor, GLM-4.7's loader convention). The host oracle `glm_moe_ref_forward` reproduces the numpy MoE to 0.7 % once it honours a chain without a shared expert (one-line change in `models/glm/moe_reference.cpp`) | existing kernels; the combination NVFP4 + no shared expert + sigmoid router has not been run |
| activation quantization (W4A4) | not applied: `input_scale` unused, as for GLM-4.7; the model card itself launches with `B12X_MOE_FORCE_A16=1` | — |
| MTP | none | — |

## 4. Sharding plan for two ranks

The rules are DESIGN §5.1–5.2's, as GLM-4.7 applies them: hidden rows replicated at
every block boundary; one all-reduce after the attention output projection and one
after the MoE; routers and norms replicated; every expert on every rank, sliced on
its intermediate dimension; the head vocabulary-sharded; the embedding whole on
every rank.

### 4.1 Which tensor splits how (world 2, rank r ∈ {0, 1})

| tensor (per layer unless global) | file shape | split | rank r holds | bytes per rank |
|---|---|---|---|---|
| `input_layernorm`, `post_attention_layernorm` | `[3072]` ×2 | replicated | whole | 12 288 |
| `self_attn.q_proj` | `[6144, 3072]` BF16 | **replicated** (§4.2 option A) | whole | 37 748 736 |
| `self_attn.q_norm` | `[6144]` | replicated | whole | 12 288 |
| `self_attn.k_proj` | `[1024, 3072]` BF16 | **replicated** (option A) | whole | 6 291 456 |
| `self_attn.k_norm` | `[1024]` | replicated | whole | 2 048 |
| `self_attn.v_proj` | `[1024, 3072]` BF16 | rows by K/V head | rows `[512r, 512r + 512)` → `[512, 3072]` | 3 145 728 |
| `self_attn.o_proj` | `[3072, 6144]` BF16 | columns by query head | columns `[3072r, 3072r + 3072)` → `[3072, 3072]` | 18 874 368 |
| `block_sparse_moe.gate` | `[256, 3072]` BF16 | replicated | whole | 1 572 864 |
| `block_sparse_moe.e_score_correction_bias` | `[256]` BF16 | replicated (F32 at load) | whole | 512 |
| `experts.E.w1` (gate), `w3` (up) | `[1536, 3072]` NVFP4 | rows of the intermediate dim | rows `[768r, 768r + 768)`: `weight` `[768, 1536]`, `weight_scale` `[768, 192]` | 1 327 104 each |
| `experts.E.w2` (down) | `[3072, 1536]` NVFP4 | columns of the intermediate dim | columns `[768r, 768r + 768)`: packed byte columns `[384r, 384r + 384)`, scale columns `[48r, 48r + 48)` | 1 327 104 |
| `experts.E.*.weight_scale_2` | F32 `[]` ×3 | replicated | whole | 12 |
| `experts.E.*.input_scale` | F32 `[]` | never loaded | — | 0 |
| `model.embed_tokens` (global) | `[200064, 3072]` BF16 | replicated (a row gather) | whole | 1 229 193 216 |
| `model.norm` (global) | `[3072]` | replicated | whole | 6 144 |
| `lm_head` (global) | `[200064, 3072]` BF16 | rows by vocabulary | rows `[100032r, 100032r + 100032)` | 614 596 608 |

Head assignment: rank r owns query heads `[24r, 24r + 24)` and K/V heads
`[4r, 4r + 4)`; query head h reads K/V head h // 6, so a rank's query heads are
exactly its K/V heads' groups (`minimax_tp_validate_geometry`). The expert slice
starts at element 768r: a multiple of the 16-element NVFP4 block and of the GEMV
core's 32, and an even packed-byte boundary.

**Per rank, world 2** (`minimax_weight_bytes(cfg, 2)`, pinned in
`tests/unit/minimax_binding_test.cpp`):

| class | bytes | GiB |
|---|---|---|
| routed experts: 62 × 256 × (3 × 1 327 104 + 12) | 63 191 574 528 | 58.85 |
| attention: 62 × (37 748 736 + 12 288 + 6 291 456 + 2 048 + 3 145 728 + 18 874 368) | 4 096 626 688 | 3.82 |
| embedding | 1 229 193 216 | 1.14 |
| head slice | 614 596 608 | 0.57 |
| routers, biases, norms | 98 317 312 | 0.09 |
| **weights per rank** | **69 230 308 352** | **64.48** |
| K/V pool: bf16 K and V of 4 K/V heads × 128 × 62 layers = 126 976 B per token; 196 608 tokens | 24 964 497 408 | 23.25 |
| prefix arena (the template's) | 4 294 967 296 | 4.00 |
| **planned, before scratch** | | **91.7** |

That leaves about 30 GiB of a 121.6 GiB node for the engine's scratch, the decode
graph and the OS; the pool could grow to roughly 320 K tokens. With q_proj / k_proj
head-sharded instead (option B) the weights are 63.20 GiB per rank.
(World 4 also divides — 12 query heads, 2 K/V heads, 384-wide expert slices, 34.1 GiB
of weights per rank; world 8 is the last that does.)

### 4.2 The per-layer norm across ranks

The RMS of q spans all 48 heads and the RMS of k all 8, but a rank owns half of
each. A rank cannot norm its slice from its slice. Two ways:

**Option A — replicate `q_proj` and `k_proj` (the plan's choice).** Every rank
projects and norms the whole row, then keeps its heads. The rstd is the same bits on
both ranks by construction (same kernel, same replicated input); each rank's K/V
rows are bit for bit the single-rank cache's slice, which
`minimax_two_rank_shards_sum_to_the_single_rank_layer` shows on the host reference.
No new bus primitive, no new collective. Cost: 1.27 GiB more resident per rank, and
per decode step per rank the other half of q_proj and k_proj is read: 22.0 MB per
layer, 1.37 GB per step.

**Option B — head-shard them and all-reduce the sums of squares.** Each rank sums
the squares of its own heads' rounded projections; one extra collective per layer
carries `[rows, 2]` fp32 (q and k together); rstd = rsqrt((s₀ + s₁) / n + eps),
identical on both ranks because the two-term sum commutes (at world 4 the fold must
be in rank order). Cost: 62 more latency-pool collectives per step, and a fold of
fp32 payloads the bus does not have today (`BoundaryReducer::reduce` folds bf16
hidden rows; a sum of squares rounded to bf16 would move the norm by a bf16 ulp).

Which is cheaper is a fabric measurement, as the GR placement was for Qwen
(docs/qwen38_flash_next_plan.md, Q0). With that document's constants — an extra
in-graph collective at two nodes ≈ 36 µs, weight bandwidth ≈ 230–240 GB/s:

| | extra per decode step, per rank | needs |
|---|---|---|
| A, BF16 as shipped | 1.37 GB ≈ 5.7–5.9 ms | nothing new |
| A, q_proj / k_proj block-FP8 at load | 0.68 GB ≈ 2.9 ms | an FP8-at-load option for the dense projections (the Qwen `dense_weights: "fp8"` path; the base release's attention is FP8, so this re-encodes back toward the source) |
| B | 62 × 36 µs ≈ 2.2 ms | an fp32 fold on the bus, a rank-order rule past world 2 |

These are estimates from constants measured on other models, not measurements of
this one. Start with A: it is exact by construction and needs no bus work, so the
first multi-rank parity run tests the model and not a new collective. Measure B
after, with the fold probe (`BoundaryReducer::probe`) standing in for it first.

### 4.3 One layer on rank r

1. `h = rms(x, input_layernorm)` — replicated input, replicated weight: the same bits on both ranks.
2. `q_dot = q_proj h` (6144, whole), `k_dot = k_proj h` (1024, whole), `v_dot = v_proj[rows of r] h` (512).
3. `minimax_qk_rstd` over the whole q row and the whole k row.
4. `minimax_qkv_finish` on the rank's heads: q heads `[24r, 24r + 24)` to the q buffer, k and v heads `[4r, 4r + 4)` appended to the rank's paged K/V cache (width 4 × 128).
5. `glm4_attn_partial` / `glm4_attn_combine`: 24 query heads over 4 K/V heads.
6. `o_proj[:, columns of r]` × the heads' output → a partial hidden row → **fold 1** (`BoundaryReducer::reduce`, rank-order sum) → replicated.
7. `h = rms(x, post_attention_layernorm)`; the router on the replicated row: the same eight experts and weights on both ranks.
8. For each picked expert: gate / up rows `[768r, …)`, the activation, down columns `[768r, …)`; the fp32 chain, one bf16 rounding → **fold 2** → replicated.

After the last layer: `rms(x, model.norm)`, the rank's 100 032 head rows, the
distributed pick. 124 folds per token (62 × 2) and the pick — GLM-4.7's count per
layer; nothing else crosses the bus under option A.

### 4.4 Decode traffic per token per rank (world 2, an estimate)

Attention BF16: q 37.7 + k 6.3 (whole, option A) + v 3.1 + o 18.9 = 66.1 MB per
layer; experts: 8 × 3 × 1.33 = 31.9 MB; router 1.6 MB → 99.5 MB × 62 = 6.17 GB, plus
the head slice 0.61 GB: **≈ 6.8 GB per step, a 28 ms floor at 240 GB/s**, and
124 folds ≈ 4.5 ms at 36 µs. Two thirds of the bytes are the BF16 attention
projections — the same lever docs/glm47_plan.md records for GLM-4.7: quantizing them
at load (block FP8) would bring the floor to about 18 ms.

### 4.5 Boot and validation, in order

1. `minimax_bind_check --world 2` on each node: the binding, the per-rank bytes above.
2. Shard parity (the `glm_shard_parity` pattern): each rank's resident slices equal the full load's views — q_proj / k_proj whole and equal across ranks (they join the replicated-tensor digest), v_proj rows, o_proj columns, the expert slices, the head rows.
3. Loopback world 2 on one node over a truncated stack (`--layers N`): the folded residual after every layer bitwise the world-1 run of the same layers.
4. The fabric: matching operation streams on both ranks, greedy transcripts identical to world 1 on the truncated stack, then the full stack against `tools/minimax_m2_reference.py score` (teacher-forced log-probabilities; the reference needs ~130 GiB of page cache to run the full model, so it runs on a node with the engine down, or on a truncated stack with `--layers`).

## 5. Text stack

**Tokenizer.** `tokenizer.json`: BPE under NFC, 200 000 base ids and 54 added
tokens from 200 000 (`<think>` 200050, `</think>` 200051, `<minimax:tool_call>`
200052, `</minimax:tool_call>` 200053 are *not* special; the role and control
markers are). Before this port the engine could not load it: the Split regex is
OpenAI's o200k pattern (cased letter runs with an optional contraction, numbers in
runs of three) and the file spells the split `Removed` + `invert: true`. The
tokenizer now has a scanner for it (pattern "o200k") and accepts that spelling for
that regex only; the regex leaves no gaps (probed over every codepoint), so the
pieces are the same as an `Isolated` split's. Checked on the Mac against HF
tokenizers 0.22.2 and 0.23.2: 472 golden cases byte-exact, 0 id mismatches on
1.16 M random and real strings plus 8.1 M enumerated ones; the only differences are
on codepoints newer than the repo's Unicode 15.0 tables.

**Chat template.** Renders byte-exactly through the engine's interpreter (24 golden
renders against jinja2, ids included; 2 template refusals). BOS `]~!b[` is a
literal in the template; turns are `]~b]role\n … [e~[\n`; the generation prompt is
`]~b]ai\n<think>\n` (the reply starts inside reasoning). A system turn is always
emitted (a model-identity line when the request has none); tools are listed as
`<tool>{json}</tool>` lines in it. Reasoning of earlier turns is kept only after the
last user message.

**Tool calls and reasoning.** One `<minimax:tool_call>` block holding one
`<invoke name="…">` per call with `<parameter name="…">value</parameter>` children;
values are text, typed from the request's schema (`ToolFormat::kMinimaxXml`; the
typing order is vLLM's parser's). 12 assistant turns / 9 calls round-trip render →
encode → parse over the real tokenizer. **The forced-call grammar does not cover
this form**: `tool_choice` required / named, `parallel_tool_calls: false` and
`response_format` are refused by name; `auto` runs unconstrained. The reference
servers' "append think" reasoning mode leaves the reasoning in content; the engine's
parser splits it at `</think>`.

The stop id is 200020 (`[e~[`) from `generation_config.json`; `config.json`'s
`eos_token_id` 2 is a byte token.

## 6. What is not done

- The resident loader (shard list from the index; the BF16 bias to F32; `1 / weight_scale_2`; the two-rank slices above).
- The model assembly on the session core, the K/V pool wiring, the decode graph, the prefix snapshot, the memory plan.
- The serve family (stop ids from `generation_config.json`, the tool-call form, the reasoning markers).
- Any run of the draft kernels.
