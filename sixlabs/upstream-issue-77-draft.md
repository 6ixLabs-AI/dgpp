**Title:** Feature request: Qwen3-Next-80B-A3B (`qwen3_next`) on the Qwen path

Hi Stephen, a feature request this time (following #73 and #74).

We'd like to run `Qwen3-Next-80B-A3B-Instruct` on DGPP. It's still a popular model on GB10, and on our 2× Spark pair the alternatives are slow to load and switch: vLLM can't use MTP on the NVFP4 checkpoint, and Atlas gets about 90–104 tok/s decode with MTP, using EP=2 across both boxes. DGPP's load time and its MTP path are exactly what this model needs.

It looks like a close cousin of Qwen3.8-Flash-Next, so most of `src/models/qwen` should carry over. Below is what we found when we compared the two configs. We haven't written any code for it; this is only to save you some reading.

### Checkpoint

`nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` (ModelOpt NVFP4, group size 16, about 48 GB, `Qwen3NextForCausalLM`, `model_type: qwen3_next`).

- Expert and shared-expert MLPs and `o_proj` are NVFP4 (`weight_scale`, `weight_scale_2`, `input_scale`).
- These stay BF16: `q_proj`, `k_proj`, `v_proj` (with FP8 `k_scale`/`v_scale` for the KV cache), the linear-attention `in_proj_qkvz`, `in_proj_ba` and `conv1d`, the router `gate`, `shared_expert_gate` and `lm_head`. That's 242 ignored modules in total.
- MTP: one layer under `mtp.*`, entirely BF16. It has **per-expert** tensors (`mtp.layers.0.mlp.experts.N.{gate,up,down}_proj`), not RadixArk's `bf16_fused` layout, plus `mtp.fc`, `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden` and `mtp.norm`.

### What already matches Qwen3.8-Flash-Next

| | Qwen3.8-Flash-Next | Qwen3-Next-80B |
|---|---|---|
| layers / pattern | 48, `full_attention_interval` 4 | 48, `full_attention_interval` 4 |
| experts | 512, top-10, shared expert | 512, top-10, shared expert (sigmoid `shared_expert_gate`) |
| full attention | 24 Q / 2 KV heads, `head_dim` 256 | 16 Q / 2 KV heads, `head_dim` 256 |
| GDN | 16 K / 48 V heads, 128-wide, conv 4 | 16 K / 32 V heads, 128-wide, conv 4 |
| `partial_rotary_factor` | 0.25 | 0.25 (`rope_theta` 1e7) |
| hidden / `moe_intermediate_size` | 2560 / 640 | 2048 / 512 |
| vocab / tied | 248320 / no | 151936 / no |
| max positions | 262144 | 262144 |

As far as we can tell, the FP4 and pack-quantized GEMV kernels already accept K = 2048 and 512.

### What the Qwen path currently requires that Qwen3-Next doesn't have

From `src/models/qwen/config.cpp`:

1. **`model_type` must be `qwen4_exp_text`**, inside a `text_config` object. Qwen3-Next is a flat config.
2. **`hc_count` is required and must be 4** (the 4-branch gated residual). Qwen3-Next uses a plain residual stream.
3. **The QSA indexer is required** (`indexer_n_heads`, `indexer_kv_heads`, `indexer_head_dim`, `indexer_budget`, `indexer_compress_ratio`). Qwen3-Next's full-attention layers are dense GQA with a gated output: `q_proj` emits query plus gate (`2 × 16 × 256`), and the output is multiplied by `sigmoid(gate)` before `o_proj`.
4. **Per-layer embeddings / n-gram vocab**: these look optional already (`ple_layer_ids`), so probably fine.

There are also two Qwen3-Next conventions that may differ from Qwen3.8 (we haven't checked how Qwen3.8 handles them):
- `Qwen3NextRMSNorm` is zero-centred (`x̂ · (1 + w)`), including `q_norm`/`k_norm`. That can be folded into the weights at load time.
- The GDN's gated output norm (`norm` plus a SiLU gate from `z`, out of `in_proj_qkvz`) and the interleaved `qkvz`/`ba` split, as in HF `Qwen3NextGatedDeltaNet.fix_query_key_value_ordering`.

### What we'd hope for

- A `w2` (TP=2) template like `cluster_qwen-3.8-flash-next_nvfp4-radixark_w2`, with MTP on. A `w1` template too if it fits on one Spark (48 GB of weights, so it may).
- Happy to test on 2× Spark (GB10, CUDA 13.0, one 200G RoCE lane) and report decode, prefill, MTP acceptance and Polyglot results against Atlas and vLLM on the same prompts. We can also run `tools/qwen_reference_dump.py`-style layer parity checks if you add a reference path for it.

If you'd rather take a PR, say so and we'll have a go at items 2 and 3 on our side. We'd want to know first whether you'd prefer a separate `src/models/qwen3next` or flags on the existing Qwen path.
