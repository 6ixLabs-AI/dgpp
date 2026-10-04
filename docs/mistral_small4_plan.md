# Mistral-Small-4 (mistralai/Mistral-Small-4-119B-2603-NVFP4) on 6ixInfer — architecture facts, host references and what remains (2026-10-04)

Status: **host side only.** The `params.json` parser, the expected-tensor table and
the host references are built in the default tree and unit-tested on a Mac; the
binding table equals the real checkpoint's safetensors headers (read from Hugging
Face, no weight bodies). The numpy reference agrees with transformers'
`Mistral4ForCausalLM` on a synthetic model of the same class. Nothing here has been
compiled by nvcc, loaded a weight or produced a token. There is no model assembly
and no serve family: the server refuses the family by name. §4's draft kernel and
its test (`-DDGPP_BUILD_MISTRAL_DRAFT=ON`) are unverified.

Sources: the checkpoint's `params.json`, `consolidated.safetensors.index.json` and
the headers of its 13 shards; transformers' `models/mistral4` (`modeling_mistral4.py`,
`convert_mistral4_weight_to_hf.py` — the meaning of each `params.json` key) at main
and at 5.8.1; vLLM's `transformers_utils/configs/mistral.py`,
`models/mistral_large_3.py` and `models/deepseek_v2.py` (the runtime the model card
names); the base repository's `config.json`.

## 0. Summary

The release is in **Mistral's own format**: `params.json` and
`consolidated-*.safetensors` with Mistral's tensor names, no `config.json`. The
architecture registry recognizes such a directory by the shape of `params.json`
(`detect_architecture_params`), and the hub-cache resolver accepts a snapshot that
has `params.json` in place of `config.json`.

The model is **not** "standard attention plus a routed MoE". It is DeepSeek-V3's
shape — vLLM serves it with its DeepSeek-V3 class — which the engine already knows
from the full GLM-5.3:

- **Multi-head latent attention.** 36 layers, hidden 4096, 32 heads. The query goes
  through a 1024-wide latent (`wq_a`, `q_a_norm`, `wq_b`) to `[64 nope | 64 rope]`
  per head; keys and values come from one 256-wide latent per token plus a 64-wide
  rope key shared by every head (`wkv_a_with_mqa` → `[256 | 64]`, `kv_a_norm`,
  `wkv_b` → `[64 k_nope | 128 v]` per head). The cache holds the latent and the
  rope key: 640 bytes per token per layer in bf16, where a GQA model of this size
  would hold several KB.
- **Interleaved rope with YaRN's frequency blend**: θ 1e4, factor 128 over an
  8192-token original context (lanes 0–12 keep their frequency, 25–31 are divided by
  128, 13–24 ramp), 1 048 576 positions.
- **`yarn.apply_scale` is false**: no YaRN magnitude correction. See §1.3 — the two
  published implementations disagree here.
- **Llama-4 query scaling**: the query at position p is multiplied by
  1 + 0.1·ln(1 + ⌊p / 8192⌋).
- **MoE in every layer**: 128 experts, top-4, softmax over all then the picked
  probabilities renormalized, `routed_scale` 1, plus one always-on shared expert of
  the same width (2048).
- Vocabulary 131 072, RMSNorm eps 1e-6, an untied head. 119 B parameters, about
  6.5 B active. A Pixtral vision tower is in the checkpoint; this port is text only.

The release quantizes only the routed and shared experts (compressed-tensors 0.13
`nvfp4-pack-quantized`); attention, routers, norms, embedding and head are BF16.

**It fits one Spark**: 65.14 GiB of text weights as stored, plus 5.63 GiB for a
262 144-token latent pool.

## 1. The model

### 1.1 `params.json` facts

| key | value | meaning (transformers' `convert_config`) |
|---|---|---|
| `dim` / `n_layers` / `vocab_size` | 4096 / 36 / 131 072 | hidden_size / num_hidden_layers |
| `norm_eps` | 1e-6 | every RMSNorm, the latents' included |
| `n_heads`, `n_kv_heads`, `head_dim` | 32, 32, 128 | 32 heads; `head_dim` = nope + rope |
| `q_lora_rank`, `kv_lora_rank` | 1024, 256 | the query and key/value latents |
| `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim` | 64, 64, 128 | per-head dims |
| `rope_theta` | 10 000 | |
| `yarn` | factor 128, original 8192, `beta` 32 (fast), `alpha` 1 (slow), `apply_scale` false | YaRN frequency blend, no magnitude |
| `llama_4_scaling` | beta 0.1, original 8192 | the query scale |
| `max_position_embeddings` | 1 048 576 | (the model card serves 262 144) |
| `moe` | 128 experts, top-4, 1 shared, `expert_hidden_dim` 2048, `routed_scale` 1.0, `first_k_dense_replace` 0, `route_every_n` 1, groups 1 | softmax router, `norm_topk_prob` true (the conversion pins it) |
| `hidden_dim` | 12 288 | the dense MLP's width — unused: no layer is dense |
| `tied_embeddings` | false | |
| `vision_encoder` | Pixtral: hidden 1024, 24 layers, patch 14, merge 2 | present, never loaded |
| `quantization_config` | compressed-tensors, `nvfp4-pack-quantized`, group 16, target `Linear`; `input_activations` 4-bit `dynamic: "local"`; ten `ignore` rules | experts NVFP4, the rest BF16 |

`params.json` carries no token ids: bos `<s>` 1, eos `</s>` 2, pad `<pad>` 11 are the
tokenizer's (the transformers conversion pins the same three).

### 1.2 One layer (transformers' `Mistral4DecoderLayer`; names are the checkpoint's)

```
h    = rms(x, attention_norm)
q    = wq_b · rms(wq_a h, q_a_norm)            → per head [q_nope 64 | q_rope 64]
ckv  = wkv_a_with_mqa h                         → [latent 256 | rope key 64]
c    = rms(ckv[:256], kv_a_norm)                the cached latent
kr   = rope(ckv[256:], pos)                     the cached rope key, shared by the heads
q_rope ← rope(q_rope, pos);   q ← q · (1 + 0.1·ln(1 + ⌊pos/8192⌋))
wkv_b c                                         → per head [k_nope 64 | v 128]
score[h, t, s] = (q_nope·k_nope + q_rope·kr) / √128,  s ≤ t
x   += wo(concat_h softmax(score) v)
h    = rms(x, ffn_norm)
p    = softmax(gate h);  top-4;  w = p[top] / Σ p[top]
x   += Σ_e w_e · w2_e(silu(w1_e h) · w3_e h)  +  w2(silu(w1 h) · w3 h)   # shared
```

`logits = output(rms(x, norm))`. The rope pairs are interleaved `(2i, 2i+1)`.
transformers writes the rotated pairs de-interleaved; q and k are permuted alike so
the scores are identical, and the engine keeps the pairs in place (the GLM-5.3
table and kernels).

### 1.3 The YaRN scale: two published implementations disagree

`params.json` says `"apply_scale": false`.

- **vLLM** (the model card's runtime) maps that to rope `attention_factor = 1`, and
  `deepseek_v2.py` then takes the rope type `deepseek_llama_scaling`, for which the
  attention keeps `scaling = qk_head_dim ** -0.5`.
- **transformers main** (since PR #47435, 2026-07-20, "Fix yarn `mscale_all_dim` for
  DeepSeek v2 and Mistral 4") multiplies the scaling by `yarn_get_mscale(factor)²` =
  (0.1·ln 128 + 1)² = 2.2058 whenever `mscale_all_dim` is truthy — and its
  conversion script writes `mscale_all_dim: 1.0` unconditionally, as the base
  repository's `config.json` has it. transformers up to 5.8.1 did not.

The engine follows the checkpoint's own flag and vLLM: **1/√128, no mscale**
(`Mistral4TextConfig::attention_scale`); `apply_scale: true` is refused at load.
This has not been settled on real weights. `tools/mistral4_reference.py --variant
yarn_mscale=on` computes the other reading; the wrong one costs teacher-forced
likelihood on any prompt, so the first forward check on a Spark decides it (the
gpu-steps file has the command). The unit tests show the two readings are 17 % apart
on a layer's output — far outside what a correct reading leaves.

### 1.4 Activation quantization

The recipe is W4A4: besides the 4-bit weights it quantizes each expert matrix's
*input* to 4 bits with dynamic per-16 scales under a stored global one
(`input_global_scale`). vLLM applies that at run time. The engine does not — it
feeds the experts full-precision activations, as it does for every other NVFP4
release — and neither do the references here. Expect the engine to be *closer to
the unquantized model* than vLLM is, and therefore not bit-comparable to a vLLM
capture; the forward check is against the numpy reference, not against vLLM.

## 2. Tensor census and the binding table

From the headers at `45331841` (HTTP range requests, 2026-10-04): **56 313 tensors,
70 801 959 560 bytes**, in 13 shards (the last holds every `input_global_scale`).

Per layer (1558 tensors): `attention_norm`, `ffn_norm`, `attention.{wq_a, q_a_norm,
wq_b, wkv_a_with_mqa, kv_a_norm, wkv_b, wo}`, `gate`, 128 × `experts.E.{w1, w2, w3}`
and `shared_experts.{w1, w2, w3}` (w1 gate, w3 up, w2 down), each expert matrix as
`weight_packed` U8 `[N, K/2]`, `weight_scale` F8_E4M3 `[N, K/16]`,
`weight_global_scale` F32 `[1]` (a **divisor**) and `input_global_scale` `[1]` —
F32 on routed experts, **BF16 on the shared one** (the release's own
inconsistency; a dtype is part of the binding). Globals: `tok_embeddings.weight`,
`norm.weight`, `output.weight`. Vision: 222 BF16 tensors.

`mistral4_validate_binding` on the real headers: expected 56 313, matched 56 313,
nothing missing, mismatched or unexpected; 13 932 NVFP4 matrices; 14 154 tensors
never loaded (13 932 activation scales, 222 vision).

| class | bytes as stored | GiB |
|---|---|---|
| routed experts (128 × 36) | 65 229 871 104 | 60.75 |
| shared experts | 509 608 368 | 0.47 |
| attention (BF16): wo 1.13, wq_a 0.28, wq_b 0.28, wkv_b 0.11, wkv_a 0.09 | 2 019 649 536 | 1.88 |
| embedding | 1 073 741 824 | 1.00 |
| head | 1 073 741 824 | 1.00 |
| routers, norms | 38 346 752 | 0.04 |
| **the text model** | **69 944 959 408** | **65.14** |
| vision (not loaded) | 856 944 640 | 0.80 |

## 3. Placement

**One Spark (the target).** 65.14 GiB of weights as stored. The latent cache is
(256 + 64) × 2 = 640 bytes per token per layer, 23 040 per token: a 262 144-token
pool is 5.63 GiB, the model's full 1 048 576 positions 22.5 GiB. With a 4 GiB prefix
arena the plan is about 75 GiB before scratch on a 121.6 GiB node.

**Two ranks (not needed, recorded for the geometry check).** DESIGN §5's rules as
the full GLM-5.3 applies them: `wq_a`, `wkv_a_with_mqa` and the two latent norms
replicated (every rank caches the same latent, so the latent cache is replicated
too); `wq_b` rows, `wkv_b` rows and `wo` columns by head (16 each); experts sliced
on the intermediate dimension (1024); the head by vocabulary; the embedding whole.
33.27 GiB per rank (`mistral4_weight_bytes(cfg, 2)`).

Decode traffic per token (one rank, an estimate): attention BF16 56.1 MB per layer,
experts 5 × 3 × 4.72 = 70.8 MB, router 1.0 MB → 127.9 MB × 36 = 4.6 GB, plus the
head 1.07 GB: **≈ 5.7 GB per step, a 24 ms floor at 240 GB/s** — the experts are
45 % of it, the BF16 attention 36 %, the head 19 %.

## 4. Numerics, reuse and what is new

The references the engine's numerics are pinned to: `tools/mistral4_reference.py`
(full precision; agrees with transformers 5.8.1's `Mistral4ForCausalLM` to fp32
rounding, 2e-6 on logits spanning 3.8, at three position offsets, on a synthetic
model — `tools/mistral4_synth.py hf-check`) and `models/mistral4/mla_reference.hpp`
(the device's rounding points; within 0.4–1.0 % of the numpy reference at every
stage, in float and double — `tests/unit/mistral4_mla_test.cpp`).

| step | what serves it | state |
|---|---|---|
| fused `[wq_a | wkv_a]` projection, `wq_b`, `wo`, head | the bf16 GEMV / GEMM interface, as GLM-5.3's `DsaLayer` | existing |
| the two latent norms | `dsa_fused_qkv_rmsnorm` (eps from the config) | existing |
| interleaved rope from a table | `dsa_rope_interleave`; the **table** is new — YaRN-blended inverse frequencies (`Mistral4TextConfig::rope_inv_freq`, the engine's existing `yarn_rope_inv_freq_host`) in the table recipe of `dsa_rope_table_host` (`mistral4_ref::rope_table`) | existing kernel, new host table (tested) |
| **Llama-4 query scale** | `mistral4_llama4_scale` (`kernels/mistral4_attn.hpp`): the multiplier from a host table, so the device's result is the reference's bits | **draft, unverified** |
| latent append with the rope tail, bf16 | `dsa_latent_append` | existing |
| absorbed query, combine, value projection | `dsa_absorb_q`, `dsa_attn_combine`, `dsa_vout_gemm` | existing |
| **dense causal attention over the latent cache** | `dsa_attn_dense` is compiled for kv_lora 512 / 256 without a rope tail and 512 with one — **not 256 + 64**, this model's. `dsa_attn_partial` (the split kernel over an explicit token list) is generic and is what the draft test uses | the missing instantiation is the one real kernel gap |
| the layer object | `DsaLayer` assumes an indexer (or a previous indexed layer's selection) on every layer; this model has none. It needs an always-dense mode or a small layer of its own | **not written** |
| router | `MoeRouterMode::SoftmaxTopk` (Qwen's); picks equal the numpy reference's on the test vectors | existing |
| routed + shared experts | the NVFP4 slot kernels with the NVFP4 shared expert in the chain (GLM-4.7's arrangement); `weight_global_scale` is the engine's divisor as stored. The host oracle `glm_moe_ref_forward` reproduces the numpy MoE block to 0.2 % | existing kernels; softmax router + NVFP4 shared expert has not been run together |
| vision | none | not planned here |

`tests/cuda/mistral4_attn_test.cu` (draft) is the kernel sequence above at the
release's attention geometry against the host reference; it also prints whether
`dsa_attn_dense` launches for 256 + 64.

## 5. Text stack

**Tokenizer.** The repository ships Mistral's own `tekken.json` (v15) *and* a
`tokenizer.json` (BPE, `ignore_merges`, 131 072 ids of which 0–999 are control
tokens). The engine loads `tokenizer.json`; before this port it could not — the
Split regex (cased letter runs, one number per pretoken, punctuation runs that
swallow `/`) was not one of the pinned patterns. `src/text/tokenizer.cpp` now has a
scanner for it (pattern "tekken"), with two new Unicode tables (Ll, Lu+Lt).
Checked on the Mac against HF tokenizers 0.22.2 and 0.23.2: 376 golden cases
byte-exact, and 0 id mismatches on 1.16 M random and real strings plus 3.26 M
enumerated ones; the only differences found are on the 5 055 codepoints Unicode 16
assigns and the repo's Unicode 15.0 tables do not know. `tekken.json` and
`tokenizer.json` are the same tokenizer on ordinary text (all 130 072 BPE tokens at
id = rank + 1000, 0 differences on 4.4 M strings through tiktoken); they differ by
design on text that *spells* a control token (`"<s>"` is three ordinary tokens to
tekken, id 1 to `tokenizer.json`, and to the engine).

**Chat template.** `chat_template.jinja` renders byte-exactly through the engine's
interpreter (31 golden renders against jinja2 as transformers configures it, ids
included; 13 template refusals by message) after five additions to the interpreter:
dict literals, list `+`, `dict.get`, the `join` / `list` filters and macro keyword
arguments. Things to know:
- the template writes `<s>` and `</s>` through `bos_token` / `eos_token`: the serve
  family must put both in the render globals or the prompt silently loses them;
- with no leading system message it injects a default system prompt containing the
  literal placeholders `{today}` and `{yesterday}` (the model card fills them
  client-side from `SYSTEM_PROMPT.txt`; mistral-common, which vLLM uses, injects no
  default prompt at all);
- tools and `[MODEL_SETTINGS]{"reasoning_effort": …}` go before the first user
  message; only `none` and `high` are accepted, the default is `none`.

**Tool calls and reasoning.** `[TOOL_CALLS]name[ARGS]{json}` per call, the turn
ended by `</s>`; reasoning between `[THINK]` and `[/THINK]`. All four markers are
single control ids (`[TOOL_CALLS]` 9, `[ARGS]` 32, `[THINK]` 34, `[/THINK]` 35) and
*special*, so a skip-special decode drops them: the parser keys on ids
(`ToolFormat::kMistral`). 22 assistant turns / 12 calls round-trip render → encode →
parse over the real tokenizer; the forced-call grammar (`tool_choice` required /
named) is implemented and walked on the host, never against logits.

mistral-common was read, not run (it is not installed here): it agrees with the
jinja file on BOS, tool and settings placement, call spelling, tool results and
think chunks, and differs in normalisations (it re-dumps tool schemas, strips
trailing spaces of assistant text, reorders tool results to call order, and
encodes each content piece separately so user text can never become a control
token).

## 6. What is not done

- The resident loader (shard list from the index; the vision tensors and activation
  scales skipped; BF16 attention; NVFP4 expert views with the stored divisor).
- The attention layer object (§4) and the `dsa_attn_dense` instantiation for
  256 + 64; the model assembly on the session core; the latent pool wiring (no index
  caches); the decode graph, the prefix snapshot, the memory plan.
- The serve family (bos / eos from the tokenizer, the tool-call form, the reasoning
  markers, `reasoning_effort`).
- Settling §1.3 on real weights.
- Any run of the draft kernel.
