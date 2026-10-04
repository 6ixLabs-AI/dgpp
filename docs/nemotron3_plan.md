# Nemotron-3 (Nano 30B-A3B, Super 120B-A12B): architecture and port

Status, 2026-10-04: the host-side description only. The config parser, the
expected-tensor table, a numpy reference of the whole forward and a host
reference of the Mamba2 mixer exist and are tested on a host; **no kernel, no
loader and no model assembly exists, and nothing here has run on a GPU or
against the real weights.** `detect_architecture` recognizes the family so
the parser and the table can be reached; no app serves it.

| piece | file | checked how |
|---|---|---|
| config parser | `src/models/nemotron/config.{hpp,cpp}` | unit tests; both releases' real `config.json` / `hf_quant_config.json` |
| tensor table | `src/models/nemotron/binding.{hpp,cpp}` | both releases' real safetensors headers: 24,147 / 165,860 tensors, byte totals equal |
| numpy reference | `tools/nemotron_reference.py` | synthetic checkpoints only (`tools/nemotron_synth.py selftest`) |
| Mamba2 host reference | `src/models/nemotron/mamba2_reference.{hpp,cpp}` | against the numpy reference's vectors, 1e-12 in double |

The reference implementation is the release's own `modeling_nemotron_h.py`
(remote code in the snapshot directory; Nano's and Super's differ in form,
not in the served arithmetic, with the one exception in §3.3).

## 1. The model

`NemotronHForCausalLM`, `model_type` `nemotron_h`. A pre-norm residual stack
with **one mixer per layer**:

    x = embeddings[ids]
    per layer L:  x += mixer_L(rms(x, layers.L.norm))
    logits = lm_head @ rms(x, norm_f)

`rms(x, w) = x * rsqrt(mean(x^2) + eps) * w` with a plain weight and
`eps = layer_norm_epsilon` (1e-5). The mixer's kind is the L-th character of
`hybrid_override_pattern`: `M` Mamba2, `E` routed MoE, `*` attention.

| | Nano | Super |
|---|---|---|
| layers (M / E / *) | 52 (23 / 23 / 6) | 88 (40 / 40 / 8) |
| attention layers | 5, 12, 19, 26, 33, 42 | 7, 16, 25, 36, 47, 58, 69, 78 |
| hidden | 2688 | 4096 |
| Mamba heads x head dim, groups, state | 64 x 64, 8, 128 | 128 x 64, 8, 128 |
| attention heads / kv heads / head dim | 32 / 2 / 128 | 32 / 2 / 128 |
| experts, top-k, scale | 128, 6, 2.5 | 512, 22, 5.0 |
| expert inner / shared inner | 1856 / 3712 | 2688 / 5376 |
| latent | none | 1024 |
| draft block | none | `mtp.layers.{0,1}` (`*E`) |
| vocab, positions | 131072, 262144 | 131072, 262144 |

**Mamba2.** `in_proj` gives `[gate z (inner) | x B C (inner + 2 * groups *
state) | dt (heads)]` with `inner = heads * head_dim` (`expand` is in the
config and unused: Nano's inner width is 4096, not 2 x 2688). `[x B C]` goes
through a depthwise causal convolution over time (kernel 4, **with a bias**)
and SiLU. Per head `h`, with group `g = h / (heads / groups)`:

    dt   = softplus(dt_raw[h] + dt_bias[h])
    S[h] = S[h] * exp(-exp(A_log[h]) * dt) + outer(dt * x[h], B[g])     S[h] is [head_dim, state], fp32
    y[h] = S[h] C[g] + D[h] * x[h]

then `u = y * silu(z)`, an RMS norm over each of the `groups` channel groups
of `inner / groups` (`u * rsqrt(mean(u^2) + eps) * norm.weight`), and
`out_proj`. The reference computes a whole sequence with the chunked SSD
algorithm (`chunk_size` 128) and a decode step with the recurrence; they are
the same function (`tools/nemotron_synth.py selftest` holds a transcription
of the chunked algorithm equal to the recurrence at 1e-15).

**Attention.** GQA: `q_proj`, `k_proj`, `v_proj`, causal softmax at
`1 / sqrt(head_dim)`, query head `h` reading KV head `h / 16`, `o_proj`.
**There is no positional encoding.** `rope_theta` and
`partial_rotary_factor` are in `config.json`; neither release's modeling
code reads them (the attention forward projects and calls the softmax).
Position reaches the model through the Mamba layers.

**MoE.** The GLM / DeepSeek router rule (`MoeRouterMode::SigmoidBias`):
`s = sigmoid(gate.weight @ h)`, top-k of `s + e_score_correction_bias`,
weights `s[picked] / (sum + 1e-20) * routed_scaling_factor`. The experts are
**two matrices**, `down(relu(up @ u)^2)` (`mlp_hidden_act` `relu2`): no
`gate_proj`, no SwiGLU. One shared expert of the same form reads `h` and is
added with weight 1. With `moe_latent_size` (Super) the routed experts run in
the latent: `u = fc1_latent_proj @ h`, the weighted expert sum is taken in
the latent, and `fc2_latent_proj` maps it back; the router and the shared
expert read the full hidden.

**Draft block (Super).** `mtp.layers.0` is an attention layer carrying
`enorm`, `hnorm` and `eh_proj [H, 2H]`; `mtp.layers.1` is a MoE layer
carrying `final_layernorm`; all BF16. The release's modeling code does not
run it (`_keys_to_ignore_on_load_unexpected = [r"mtp.*"]`), so its forward
is not specified by the checkpoint's own code. The names are DeepSeek-V3's
(`x = eh_proj @ [enorm(embed(next)) | hnorm(hidden)]`, the blocks, the final
norm, the shared head); the model card says the heads share weights (one
block applied recursively, three drafts in the vendor's vLLM recipe).
`tools/nemotron_reference.py mtp` measures the open conventions.

## 2. The checkpoints

Tensor names: `backbone.embeddings.weight`, `backbone.layers.L.norm.weight`,
`backbone.layers.L.mixer.*`, `backbone.norm_f.weight`, `lm_head.weight`,
`mtp.layers.M.*`. `src/models/nemotron/binding.hpp` lists the classes.

Formats are modelopt's, per `nn.Linear`:

- **NVFP4**: `X.weight` U8 `[N, K/2]`, `X.weight_scale` e4m3 `[N, K/16]`,
  `X.weight_scale_2` F32 scalar, `X.input_scale` F32 scalar — the container
  `models/qwen/loader35.cpp` already reads for Qwen3-Next.
- **FP8** (Super only): `X.weight` e4m3 `[N, K]`, `X.weight_scale` F32
  scalar (one per tensor), `X.input_scale` F32 scalar.
- **BF16** otherwise.

The recipe comes in two forms and the form identifies the release:

- **Nano** (`modelopt 0.29.0`): no `quantization_config` in `config.json`;
  `hf_quant_config.json` says NVFP4 / group 16 and lists `exclude_modules`
  (the head, every attention projection, the Mamba layer in front of each
  attention layer, every convolution). Every other backbone linear is NVFP4.
  The router gate is an **F32** parameter; there are no K/V-cache scales.
- **Super** (`modelopt 0.43.0.dev`): `config.json` carries
  `quantization_config.quantized_layers`, 41,100 modules each with its own
  format — every routed expert NVFP4, and **per-layer** choices elsewhere
  (FP8 on 28 of 40 `in_proj`, 29 of 40 `out_proj`, 39 shared `up_proj`, 37
  shared `down_proj` plus one NVFP4, 3 `fc1_latent_proj`, 1
  `fc2_latent_proj`, 2 of 8 `o_proj`; BF16 for the rest). The router gate is
  **BF16**, and the eight backbone attention layers carry `k_proj.k_scale` /
  `v_proj.v_scale`. `hf_quant_config.json` repeats the map; the parser holds
  the two equal.

Neither checkpoint has a tensor name in two shards.

## 3. The references

### 3.1 `tools/nemotron_reference.py`

`score`, `generate`, `dump`, `mtp`; the CLI and structure of
`tools/qwen3next_reference.py`. `--variant key=value` flips one convention
at a time (`nibble`, `ws2`, `fp8scale`, `conv`, `grpmap`, `dtclamp`,
`gatenorm`, `kvmap`, `pos`, `rbias`, `renorm`, `act`, `mtpcat`): on the real
checkpoint the right value is the one that leaves natural text likely.

### 3.2 `tools/nemotron_synth.py`

Writes toy checkpoints with the releases' names, containers and recipes
(`ckpt --preset nano|super`), runs the reference's self-test, and generates
`tests/unit/nemotron_mamba2_vectors.hpp`.

### 3.3 Conventions not settled by the files in the snapshot

1. **The gated norm** is `mamba_ssm`'s `rmsnorm_fn(..., group_size,
   norm_before_gate=False)`, which is not in the snapshot: gate first, then
   the grouped norm, eps inside the root. Taken from that library as
   remembered and from vLLM's `Mixer2RMSNormGated`.
2. **The time-step clamp.** The CUDA path (`mamba_ssm`, and vLLM) passes
   `dt_limit = (0, inf)`: no clamp. Super's CPU fallback (`torch_forward`)
   clamps `dt` from below at `time_step_min` (0.001); Nano's does not. The
   references follow the CUDA path (`--variant dtclamp=min` is the other).
3. **FP8 scale direction**: `W = e4m3(weight) * weight_scale`.
4. **NVFP4**: low nibble = even column, `weight_scale_2` multiplies — as
   confirmed for Qwen3-Next's modelopt container, not re-confirmed here.
5. **The draft block**, entirely (§1).

## 4. What the engine still needs

Nothing below is written. In dependency order:

1. **Loader.** NVFP4 sets through the Qwen3-Next path (`Fp4Set`,
   `load_fp4_slice`, `qwen3next_fp4_dequant_bf16`). Per-tensor FP8 is new:
   either dequantized to BF16 at load (a rounding: an e4m3 code times an F32
   scalar does not fit BF16's significand) or kept as codes under a scale
   grid filled with the one scalar, which the block-scale FP8 kernels then
   read exactly. Nano's F32 router gate against Super's BF16 one.
2. **Mamba2 kernels**, to `mamba2_ref` (§3): the convolution needs a bias
   (`kda_causal_conv_silu_bf16` has none) at 6144 / 10240 channels; the scan
   is a new recurrence (state fp32 `[heads, head_dim, state]`: 2 MiB a layer
   on Nano, 4 MiB on Super — 46 / 160 MiB a slot), with the post-row
   snapshot and replay forms speculative decode needs; the gated norm is
   grouped (`qwen_norm`'s gated forms are per head).
3. **Attention** without positions, 128-wide heads, 2 KV heads: the paged
   GQA path with the rotation off.
4. **MoE chain** for two-matrix `relu^2` experts with a shared expert of a
   different inner width, the latent projections around the routed sum, and
   **top-22** routing: `GlmMoeConfig::validate_config` caps `top_k` at 16.
5. **State and snapshots**: a per-slot arena of the Mamba states and
   convolution tails (the `KdaStatePool` / `Qwen35Model::gdn_rec` pattern),
   prefix-cache snapshots of it, the K/V pool at 1,024 bytes a token per
   attention layer.
6. **Draft block**, once §3.3 item 5 is measured.
7. **Text**: the tokenizer is a BPE over a Unicode-class split regex with
   `ignore_merges`; the template is ChatML with `<think>` and the
   `<tool_call><function=...><parameter=...>` call form.
