# Port 5 — Nemotron-3 (Nano, Super): checkpoint facts

Written 2026-10-04 (overnight). Everything here was read from the two snapshots' own files —
`config.json`, `hf_quant_config.json`, `modeling_nemotron_h.py`, `configuration_nemotron_h.py`,
`chat_template.jinja`, `tokenizer_config.json`, the summary fields of `tokenizer.json`, and the
safetensors **headers** of every shard. No tensor data was read anywhere. Where a statement rests
on something else (memory of `mamba_ssm` / vLLM), it says so.

How the files were obtained: read-only `ssh dgxone` between about 02:10 and 02:45 (ls, `gzip -c` of
the small text files, two header-only python scripts at `nice 19`, one `json.load` of
`tokenizer.json` per snapshot). That was inside the brief I was launched with; the rule changed to
"no ssh at all" at about 02:45 and nothing was run on either box after I saw it. DGXtwo was never
touched.

| | Nano | Super |
|---|---|---|
| repo | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` |
| snapshot | `6efb4a2a1c1fa277ce7b3df7a1416255011b1c99` (`refs/main`) | `4f0cf9daaeb7a4d5e23f80a00e7ed15f0e03caf6` |
| shards | 5 | 17 |
| tensors (headers) | **24,147** | **165,860** |
| tensors (engine table) | **24,147** | **165,860** |
| duplicate names across shards | 0 | 0 |
| tensor bytes | 19,339,781,632 (18.01 GiB) | 80,297,329,824 (74.78 GiB) |
| file bytes (headers included) | 19,342,796,520 | 80,317,948,856 |
| bytes by dtype | U8 15.25 GB, BF16 2.16 GB, F8_E4M3 1.91 GB, F32 31.7 MB | U8 56.38 GB, BF16 12.04 GB, F8_E4M3 11.87 GB, F32 0.4 MB |
| producer | modelopt 0.29.0 | modelopt 0.43.0.dev63+g449e700f9 |
| transformers | 4.53.2 | 4.57.6 |

DGXone also holds an older Nano snapshot, `ce1b118a…`; it differs from `6efb4a2a…` in `README.md`
only (config, both `.py` files, the recipe and the template compare equal).

Every shard's size equals `8 + header length + last data offset`: no trailing bytes.

## 1. Architecture (from `modeling_nemotron_h.py`)

`NemotronHForCausalLM`, `model_type` `nemotron_h`. One mixer per layer, pre-norm, residual:
`x += mixer(rms(x))`; kinds from `hybrid_override_pattern` (`M` Mamba2, `E` MoE, `*` attention).

| | Nano | Super |
|---|---|---|
| pattern | `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` | `MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME` |
| layers M / E / * | 23 / 23 / 6 | 40 / 40 / 8 |
| attention at | 5, 12, 19, 26, 33, 42 | 7, 16, 25, 36, 47, 58, 69, 78 |
| hidden, vocab | 2688, 131072 | 4096, 131072 |
| `layer_norm_epsilon` | 1e-5 | 1e-5 |
| Mamba heads x head_dim (inner) | 64 x 64 (4096) | 128 x 64 (8192) |
| `ssm_state_size`, `n_groups`, `conv_kernel` | 128, 8, 4 | 128, 8, 4 |
| conv channels (inner + 2 x 8 x 128) | 6144 | 10240 |
| `in_proj` rows (inner + conv + heads) | 10304 | 18560 |
| attention heads / kv / head_dim | 32 / 2 / 128 | 32 / 2 / 128 |
| experts, top-k, `routed_scaling_factor` | 128, 6, 2.5 | 512, 22, 5.0 |
| expert inner, shared inner | 1856, 3712 | 2688, 5376 |
| `moe_latent_size` | — | 1024 |
| MTP | — | `num_nextn_predict_layers` 1, `mtp_hybrid_override_pattern` `*E` |
| `max_position_embeddings` | 262144 | 262144 |

### Things that differ from the brief, or are easy to get wrong

1. **The attention has no positional encoding at all.** The brief said "full rotary". `config.json`
   does carry `rope_theta: 10000` and `partial_rotary_factor: 1.0`, but neither modeling file reads
   them: the attention forward is q/k/v projection, `repeat_kv`, softmax at `head_dim^-0.5`, `o_proj`
   (Nano's forward even has `# position_embeddings: ... #TODO` commented out). The reference has a
   `--variant pos=rope` so the real checkpoint can show that adding a rotation makes it worse.
2. **The experts are two matrices**, `down_proj(relu(up_proj(x))^2)` — `mlp_hidden_act: relu2`, no
   `gate_proj`, no SwiGLU. The shared expert (`shared_experts`, plural in the name, one module) has
   the same form, reads the full hidden, and is added with weight 1 (no sigmoid gate as in
   Qwen3-Next).
3. **The router is the GLM / DeepSeek rule**, not Qwen's softmax: `sigmoid` scores, the top-k taken
   on `score + e_score_correction_bias`, the picked *uncorrected* scores divided by
   `(sum + 1e-20)` and multiplied by `routed_scaling_factor`. `n_group = topk_group = 1`, so the
   group masking in `get_topk_indices` is a no-op.
4. **Super's top-k is 22.** `GlmMoeConfig::validate_config` refuses `top_k > 16` ("kernel register
   selection"). The parser here does not refuse it; the MoE kernels will.
5. **Mamba2**: the inner width is `mamba_num_heads * mamba_head_dim`; `expand: 2` is unused (Nano:
   4096, not 5376). `in_proj` has no MLP slice (`d_mlp = 0`): its rows are exactly
   `[gate | xBC | dt]`. The convolution has a **bias**. `A_log`, `D`, `dt_bias` are per head
   (`[64]` / `[128]`), BF16 in the checkpoint. `mixer.norm.weight` is `[inner]` and the norm is
   grouped: 8 groups of 512 (Nano) / 1024 (Super).
6. **`mamba_proj_bias`** is in the config and read by nothing; `in_proj` / `out_proj` take their
   bias flag from `use_bias` (false).
7. **`norm_eps`** is in the config beside `layer_norm_epsilon` and read by nothing (both 1e-5).
8. **Super's modeling file is a rewrite** (transformers 4.57 style): the module is `self.model`
   and `backbone.` is renamed on load. The checkpoint's names are `backbone.*` in both releases.
9. **Super's CPU fallback differs from its CUDA path** in one place: `torch_forward` clamps the
   time step from below at `time_step_min` (0.001); the `mamba_ssm` path passes
   `dt_limit = (0, inf)`. Nano's fallback clamps to `(0, inf)` (nothing). The served path is the
   CUDA one, so the references do not clamp.
10. **Super's `config.json` is 7.4 MB** (the layer map, twice: `quantized_layers` and the two
    `config_groups` target lists). `minijson::Value::find` is a linear scan, so the parser walks
    `members()` once instead of looking names up.

## 2. Tensor census

Class names with `L` for the layer index and `E` for the expert index; `n` is the tensor count.

### Nano (24,147)

| class | n | dtype [shape] |
|---|---|---|
| `backbone.embeddings.weight` | 1 | BF16 [131072, 2688] |
| `lm_head.weight` | 1 | BF16 [131072, 2688] |
| `backbone.norm_f.weight` | 1 | BF16 [2688] |
| `backbone.layers.L.norm.weight` | 52 | BF16 [2688] |
| `…mixer.A_log`, `.D`, `.dt_bias` | 23 each | BF16 [64] |
| `…mixer.conv1d.weight` / `.bias` | 23 / 23 | BF16 [6144, 1, 4] / [6144] |
| `…mixer.norm.weight` | 23 | BF16 [4096] |
| `…mixer.in_proj.weight` | 23 | 17 x U8 [10304, 1344]; 6 x BF16 [10304, 2688] |
| `…mixer.in_proj.{weight_scale, weight_scale_2, input_scale}` | 17 each | F8_E4M3 [10304, 168]; F32 []; F32 [] |
| `…mixer.out_proj.weight` | 23 | 17 x U8 [2688, 2048]; 6 x BF16 [2688, 4096] |
| `…mixer.out_proj.{weight_scale, weight_scale_2, input_scale}` | 17 each | F8_E4M3 [2688, 256]; F32 []; F32 [] |
| `…mixer.{q,o}_proj.weight` | 6 each | BF16 [4096, 2688] / [2688, 4096] |
| `…mixer.{k,v}_proj.weight` | 6 each | BF16 [256, 2688] |
| `…mixer.gate.weight` | 23 | **F32** [128, 2688] |
| `…mixer.gate.e_score_correction_bias` | 23 | F32 [128] |
| `…mixer.experts.E.up_proj.{weight, weight_scale, weight_scale_2, input_scale}` | 2,944 each | U8 [1856, 1344]; F8_E4M3 [1856, 168]; F32 []; F32 [] |
| `…mixer.experts.E.down_proj.{…}` | 2,944 each | U8 [2688, 928]; F8_E4M3 [2688, 116]; F32 []; F32 [] |
| `…mixer.shared_experts.up_proj.{…}` | 23 each | U8 [3712, 1344]; F8_E4M3 [3712, 168]; F32 []; F32 [] |
| `…mixer.shared_experts.down_proj.{…}` | 23 each | U8 [2688, 1856]; F8_E4M3 [2688, 232]; F32 []; F32 [] |

Count: 3 globals + 17 x 15 + 6 x 9 (Mamba) + 6 x 5 (attention) + 23 x 1,035 (MoE) = 24,147.
NVFP4 matrices: 5,968. The six BF16 Mamba layers are 4, 11, 18, 25, 32, 41 — each the layer in
front of an attention layer.

### Super (165,860)

| class | n | dtype [shape] |
|---|---|---|
| `backbone.embeddings.weight`, `lm_head.weight` | 1, 1 | BF16 [131072, 4096] |
| `backbone.norm_f.weight`; `backbone.layers.L.norm.weight` | 1; 88 | BF16 [4096] |
| `…mixer.A_log`, `.D`, `.dt_bias` | 40 each | BF16 [128] |
| `…mixer.conv1d.weight` / `.bias` | 40 / 40 | BF16 [10240, 1, 4] / [10240] |
| `…mixer.norm.weight` | 40 | BF16 [8192] |
| `…mixer.in_proj.weight` | 40 | 28 x **F8_E4M3** [18560, 4096]; 12 x BF16 |
| `…mixer.out_proj.weight` | 40 | 29 x F8_E4M3 [4096, 8192]; 11 x BF16 |
| `…mixer.q_proj.weight` | 8 | BF16 [4096, 4096] |
| `…mixer.{k,v}_proj.weight` | 8 each | BF16 [256, 4096] |
| `…mixer.k_proj.k_scale`, `…v_proj.v_scale` | 8 each | F32 [] |
| `…mixer.o_proj.weight` | 8 | 2 x F8_E4M3 [4096, 4096] (layers 69, 78); 6 x BF16 |
| `…mixer.gate.weight` | 40 | **BF16** [512, 4096] |
| `…mixer.gate.e_score_correction_bias` | 40 | F32 [512] |
| `…mixer.experts.E.up_proj.{weight, weight_scale, weight_scale_2, input_scale}` | 20,480 each | U8 [2688, 512]; F8_E4M3 [2688, 64]; F32 []; F32 [] |
| `…mixer.experts.E.down_proj.{…}` | 20,480 each | U8 [1024, 1344]; F8_E4M3 [1024, 168]; F32 []; F32 [] |
| `…mixer.shared_experts.up_proj.weight` | 40 | 39 x F8_E4M3 [5376, 4096]; 1 x BF16 (layer 41) |
| `…mixer.shared_experts.down_proj.weight` | 40 | 37 x F8_E4M3 [4096, 5376]; 1 x U8 [4096, 2688] (layer 1, NVFP4); 2 x BF16 (layers 8, 76) |
| `…mixer.fc1_latent_proj.weight` | 40 | 3 x F8_E4M3 [1024, 4096] (layers 1, 3, 5); 37 x BF16 |
| `…mixer.fc2_latent_proj.weight` | 40 | 1 x F8_E4M3 [4096, 1024] (layer 3); 39 x BF16 |
| FP8 companions `X.weight_scale`, `X.input_scale` | 139 each | F32 [] |
| `mtp.layers.0.{enorm, hnorm, norm}.weight` | 3 | BF16 [4096] |
| `mtp.layers.0.eh_proj.weight` | 1 | BF16 [4096, 8192] |
| `mtp.layers.0.mixer.{q,o}_proj.weight`; `{k,v}_proj.weight` | 2; 2 | BF16 [4096, 4096]; [256, 4096] |
| `mtp.layers.1.{norm, final_layernorm}.weight` | 2 | BF16 [4096] |
| `mtp.layers.1.mixer.gate.weight`; `.gate.e_score_correction_bias` | 1; 1 | BF16 [512, 4096]; F32 [512] |
| `mtp.layers.1.mixer.experts.E.{up,down}_proj.weight` | 512 each | BF16 [2688, 1024] / [1024, 2688] |
| `mtp.layers.1.mixer.shared_experts.{up,down}_proj.weight` | 1 each | BF16 [5376, 4096] / [4096, 5376] |
| `mtp.layers.1.mixer.fc{1,2}_latent_proj.weight` | 1 each | BF16 [1024, 4096] / [4096, 1024] |

Quantized matrices: 40,961 NVFP4 (40,960 expert matrices + layer 1's shared `down_proj`), 139 FP8.
The draft block is 1,040 tensors and 5.48 GiB, all BF16 (its 512 experts alone are 5.25 GiB); the
backbone is 69.3 GiB.

FP8 layer lists (the rest of each class is BF16):

- `in_proj`: 0, 2, 4, 6, 9, 11, 13, 15, 18, 20, 27, 29, 31, 40, 49, 51, 53, 55, 57, 60, 62, 66, 68, 71, 73, 82, 84, 86
- `out_proj`: 0, 2, 4, 9, 11, 13, 18, 38, 40, 42, 44, 49, 51, 53, 55, 57, 60, 62, 64, 66, 68, 71, 73, 75, 77, 80, 82, 84, 86
- `shared_experts.up_proj`: every MoE layer but 41
- `shared_experts.down_proj`: every MoE layer but 1 (NVFP4), 8 and 76 (BF16)
- `fc1_latent_proj`: 1, 3, 5 — `fc2_latent_proj`: 3 — `o_proj`: 69, 78

These lists regenerate `quantization_config.quantized_layers` exactly (41,100 entries; checked in
Python against the real file, and they are what `tests/unit/nemotron_config_json.hpp` builds).

## 3. Quantization containers

Both are NVIDIA modelopt exports; per `nn.Linear` module `X`:

| format | tensors | dequantization the references use |
|---|---|---|
| NVFP4 | `X.weight` U8 [N, K/2]; `X.weight_scale` F8_E4M3 [N, K/16]; `X.weight_scale_2` F32 []; `X.input_scale` F32 [] | `W[n,k] = e2m1(code) * e4m3(weight_scale[n, k/16]) * weight_scale_2`, low nibble = even column |
| FP8 | `X.weight` F8_E4M3 [N, K]; `X.weight_scale` F32 []; `X.input_scale` F32 [] | `W = e4m3(weight) * weight_scale` |
| BF16 | `X.weight` | — |

`input_scale` is the activation-quantization scale of the recipe (W4A4 / W8A8); the references run
full-precision activations and ignore it. `k_scale` / `v_scale` are the FP8 K/V-cache scales
(Super only) and are likewise ignored.

Where the recipe lives:

- **Nano**: `config.json` has **no** `quantization_config`. `hf_quant_config.json`:
  `quant_algo: NVFP4`, `kv_cache_quant_algo: FP8`, `group_size: 16`, `exclude_modules` with 60
  exact names (no wildcards): `lm_head`; `in_proj` / `out_proj` of layers 4, 11, 18, 25, 32, 41;
  `q/k/v/o_proj` of the six attention layers; `conv1d` of all 23 Mamba layers. Rule verified
  against the headers: every `nn.Linear` not on the list is NVFP4, everything on it is BF16. Despite
  `kv_cache_quant_algo: FP8` there are **no** `k_scale` / `v_scale` tensors.
- **Super**: `config.json` has `quantization_config` (`quant_method: modelopt`,
  `quant_algo: MIXED_PRECISION`, `kv_cache_scheme: {dynamic: false, num_bits: 8, type: float}`,
  `ignore: []`, `config_groups.group_0` = 8-bit float with 139 targets, `group_1` = 4-bit float
  per 16 with 40,961 targets, `quantized_layers` = 41,100 entries
  `{quant_algo: FP8}` / `{quant_algo: NVFP4, group_size: 16}`). `hf_quant_config.json` repeats
  `quantized_layers` (identical). Verified against the headers: a module is quantized exactly when
  the map names it, in the format the map gives; 0 mismatches over 42,482 modules.

The router gate is `nn.Parameter` (not a Linear) and is never quantized: F32 in Nano (its
modeling code creates it with `dtype=torch.float32`), BF16 in Super. The config carries nothing
that says which, so the parser ties it to the recipe form.

## 4. The MTP block (Super)

`mtp.layers.0`: `enorm.weight`, `hnorm.weight`, `eh_proj.weight [4096, 8192]`, `norm.weight`,
`mixer.{q,k,v,o}_proj.weight` — an attention layer carrying the fusion.
`mtp.layers.1`: `norm.weight`, `mixer.*` of a full latent MoE layer (router, 512 experts, shared
expert, `fc1/fc2_latent_proj`), `final_layernorm.weight`.
All BF16, no K/V-cache scales, not in the quantization map.

The release's `modeling_nemotron_h.py` **does not implement it**:
`_keys_to_ignore_on_load_unexpected = [r"mtp.*"]`, and nothing else in the file mentions `mtp`.
What is known from the files: the tensor names (DeepSeek-V3's), the pattern `*E`, and the model
card: "MTP layers using a shared-weight design across prediction heads" and a vLLM recipe with
`--speculative_config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'`. So one
block, applied recursively for up to three drafts. What is **not** known from the files: the
concat order into `eh_proj`, whether `hidden` is taken before or after `norm_f`, and how the draft
attention's K/V history is kept. `tools/nemotron_reference.py mtp` measures all of them on a text
(draft vs the main model's own next token); vLLM's `nemotron_h_mtp` implementation is the code to
read when a network is available (not read tonight).

## 5. Tokenizer, template, tool calls

- `tokenizer.json`: `model.type` BPE, vocab 131,072, 269,443 merges, `ignore_merges: true`,
  `byte_fallback: false`; no normalizer; pre-tokenizer a `Sequence` starting with a `Split` on a
  Unicode-class regex (`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}…]+|…`, the
  Mistral "Tekken" style); `ByteLevel` decoder and post-processor. Nano's and Super's
  `tokenizer.json` differ by one byte in length (17,077,485 / 17,077,484) and in hash; the summary
  fields above are identical. I did not diff the two files.
- Ids 0–999 are added tokens. `<unk>` 0, `<s>` 1, `</s>` 2, `[INST]` 3 … `[TOOL_CALLS]` 9 (special,
  unused by the template), `<|im_start|>` 10, `<|im_end|>` 11 (special), and — **not** special —
  `<think>` 12, `</think>` 13, `<tool_call>` 14, `</tool_call>` 15, `<tool_response>` 16,
  `</tool_response>` 17; the rest are `<SPECIAL_n>`.
- `config.json` `eos_token_id` 2, `bos_token_id` 1, `pad_token_id` 0. `generation_config.json`
  `eos_token_id` **[2, 11]** (both), `do_sample: true`, temperature 1.0, `top_p` 1.0 (Nano) /
  0.95 (Super). The tokenizer's `eos_token` is `<|im_end|>`; `add_bos_token: false`.
- Template (ChatML; Nano's and Super's differ only by Super's `low_effort` switch, which appends
  `\n\n{reasoning effort: low}` to the last user message): a `<|im_start|>system\n…<|im_end|>\n`
  block is **always** emitted (empty when there is no system message and no tools); no `<s>`.
  The generation prompt is `<|im_start|>assistant\n<think>\n` with `enable_thinking` (the default)
  and `<|im_start|>assistant\n<think></think>` without. History assistant turns before the last
  user message have their reasoning cut to `<think></think>` (`truncate_history_thinking`, default
  on); an assistant turn with no think tags gets `<think></think>` prepended.
- Tools: declared in the system block as XML (`<tools><function><name>…</name><description>…
  <parameters><parameter><name>…<type>…`), followed by an instruction block. A call is
  `<tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n</function>\n</tool_call>` —
  the form vLLM parses with `--tool-call-parser qwen3_coder` (the model card's recipe), i.e. the
  same call form as Qwen3-Coder-Next. Tool results come back inside a `user` turn as
  `<tool_response>\n…\n</tool_response>`.
- Reasoning: the model card serves it with a `deepseek_r1`-derived parser
  (`nano_v3_reasoning_parser.py` / `super_v3_reasoning_parser.py`, both in the snapshots): text up
  to `</think>` is reasoning; with `enable_thinking=false` (or Super's `force_nonempty_content`)
  and no content, the text is returned as content instead.

## 6. Numerics facts for the engine

- State per sequence: Mamba SSM state fp32 `[heads, head_dim, 128]` = 2 MiB a layer (Nano) /
  4 MiB (Super) → 46 MiB / 160 MiB; conv tails 3 rows of 6144 / 10240 BF16 a layer; K/V
  `2 heads x 128 x (k + v)` BF16 = 1,024 bytes a token per attention layer → 6,144 / 8,192 bytes a
  token (+1,024 for the draft attention) — 1.6 / 2.1 GiB at 262,144 tokens.
- The reference keeps the SSM state in `mamba_ssm_cache_dtype: float32`; the model card's vLLM
  recipe for Super overrides it to `float16`.
- Weights on one Spark: Nano 18.0 GiB; Super 74.8 GiB as shipped (69.3 GiB without the draft block).
