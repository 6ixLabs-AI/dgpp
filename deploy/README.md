# Deployment templates

One template per model, quantization and world size:
`cluster_<model>_<quant>_w<n>.example.json`. An optional trailing
`_<variant>` names a template that deviates from that shape in one documented
engine setting rather than in its size — today
`cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json`, the same two-Spark
NVFP4 deployment with the YaRN rope ramp switched on. Base shapes stay unique;
a variant may repeat a base only with that suffix. Local copies use the same name
without `.example` and stay Git-ignored; the launcher fills the nodes, the SSH
user and the HTTP/fabric/journal port defaults from the site's `.env`
(`scripts/site_env.py`). Set `"ports": {"metrics": 29972}` in a deployment
to enable each peer's Prometheus listener; an omitted port or `0` disables
it. Rank 0 serves metrics on its HTTP endpoint. See the
[per-rank metrics](../docs/openai-compatibility.md#per-rank-families).

- Model names are `glm-5.3-flash`, `glm-5.3` (the full model), `glm-4.7`,
  `qwen-3.8-flash-next`, `qwen3.8-27b`, `qwen3-next-80b`, `qwen3-coder-next`,
  `qwen3.6-35b-a3b`, `qwen3.5-122b-a10b`, `qwen3.5-0.8b`, `qwen3-vl-30b-a3b`, `deepseek-v4.1-flash`, `mimo-v2.6-flash`,
  `mistral-small-4` and `minimax-m2.7`,
  and `gemma-4-31b` and `gemma-4-26b-a4b` (templates ahead of their engine:
  see their rows).
- The quant names the checkpoint representation: `fp8`, `nvfp4`, `bf16` (an
  unquantized release),
  `nvfp4-fp8` for the custom GLM-5.3-Flash hybrid, `int4-int8` for the full
  GLM-5.3's pack-quantized release (int4 group-64 routed experts, int8
  attention and shared experts), `mxfp4-fp8` for DeepSeek-V4.1-Flash as it
  ships (MXFP4 experts, FP8 dense and attention, FP8 Engram tables mapped
  from the NVMe) and for MiMo-V2.6-Flash as it ships (MXFP4 experts, FP8
  fused attention projections and dense MLPs, BF16 o_proj / head). The
  JSON's `model` field gives the exact Hugging Face repository.
- `w<n>` gives the participating node count. Nothing restricts that number to
  the counts in use today; a world is refused by the engine's geometry check or
  a rank's memory plan, not by a list of allowed sizes.

Every template enables MTP (the block draft on DeepSeek) at the depth the
family measured best, with the decode graph, at the request-slot count and
cache budget that measured at or above every other shape tried — except a
`_dflash2` variant, which names `engine.dflash_model` instead of MTP: the
external block drafter replacing the MTP draft (the eager engine on one
Spark without the decode graph; the block proposal recorded inside the
graph step on the fabric) — and a checkpoint that carries no draft layer
(Qwen3-Coder-Next: no `mtp.*` tensors), whose template runs the plain
decode graph. The shapes
a template does not name are knobs appended at boot:
`scripts/dgpp-cluster up --config FILE --knobs "FLAGS"`.

| Template | Deployment | The shapes it replaced, as knobs |
|---|---|---|
| [cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json](cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json) | GLM-5.3-Flash hybrid on four nodes: MTP depth 1, bf16 latent cache, 768K context, a 22 GiB prefix arena (639 snapshots), four request slots | the 8K-context template: `--kv-capacity 8192 --prefix-cache-gib 1.5`; T=1: `--no-mtp` |
| [cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json](cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json) | the same hybrid on two nodes: MTP depth 1, FP8 latent cache, 160K context, four request slots; the BF16 decode weights resident in their 12-bit form alone (`"bf12"`: 110.0 GiB per rank with its 4.5 GiB prefix arena of 65 snapshots, 1.8 GiB of the node left at boot) | both forms resident (no prefill cost, 132K context): `--bf16-weights bf12+bf16 --kv-capacity 135168`; the 256K-context two-slot shape: `--max-concurrency 2 --kv-capacity 262144 --prefix-cache-gib 2` |
| [cluster_qwen-3.8-flash-next_fp8_w4.example.json](cluster_qwen-3.8-flash-next_fp8_w4.example.json) | Qwen3.8-Flash-Next FP8 on four nodes: MTP depth 1, 256K context, four request slots, a 50 GiB prefix arena (1846 snapshots) | T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_fp8_w2.example.json](cluster_qwen-3.8-flash-next_fp8_w2.example.json) | the same on two nodes | T=1: `--no-mtp` |
| [cluster_qwen-3.8-flash-next_nvfp4_w1.example.json](cluster_qwen-3.8-flash-next_nvfp4_w1.example.json) | Qwen3.8-Flash-Next NVFP4 on one Spark: MTP depth 1, the dense projections FP8 at load (`dense_weights: "fp8"`: 31 ms/step T=1 and 21–26 ms/token against the BF16 stack's 38 and 31–38), the n-gram table mapped, a 262144-token shared KV pool and 4096-token busy/idle prefill budgets | the BF16 dense stack: `--dense-weights checkpoint --fp8-head gemv`; T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_nvfp4_w2.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2.example.json) | Qwen3.8-Flash-Next NVFP4 on two Sparks: MTP depth 1, FP8 dense projections, the n-gram table mapped, 262K context, four request slots and a 58 GiB prefix arena (1074 snapshots) | the resident n-gram table: `--ngram-table resident`; T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_nvfp4-radixark_w1.example.json](cluster_qwen-3.8-flash-next_nvfp4-radixark_w1.example.json) | RadixArk's Qwen3.8-Flash-Next NVFP4 on one Spark: tuned engine configuration (4K prefill chunks with matching 4096-token busy/idle budgets, MTP depth 2, grow admission), dense projections FP8 at load, n-gram table mapped, 256K context | the BF16 dense stack: `--dense-weights checkpoint`; T=1: `--no-mtp`; depth 1: `--mtp-depth 1` |
| [cluster_qwen-3.8-flash-next_nvfp4-radixark_w2.example.json](cluster_qwen-3.8-flash-next_nvfp4-radixark_w2.example.json) | RadixArk's Qwen3.8-Flash-Next NVFP4 on two Sparks: identical engine configuration to the NVIDIA release, MTP depth 1, FP8 dense projections, n-gram table mapped, 262K context and four request slots | the resident n-gram table: `--ngram-table resident`; T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json) | the two-Spark NVFP4 deployment with the opt-in YaRN ramp: `engine.rope_scaling` yarn ×2 over the checkpoint's 262 144 positions, so one request reaches 524 288 tokens; two request slots, a 565 248-token pool, the n-gram table mapped, FP8 dense projections, MTP depth 1 | two full-length streams at once: `--kv-capacity 1114112` (see [the YaRN notes](#the-512k-yarn-template-enginerope_scaling)); the plain 262K template: [cluster_qwen-3.8-flash-next_nvfp4_w2.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2.example.json) |
| [cluster_qwen-3.8-flash-next_autoround-int4_w1.example.json](cluster_qwen-3.8-flash-next_autoround-int4_w1.example.json) | the AutoRound int4/int8 hybrid (`Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN`) on one Spark in its packed form: MTP depth 3, the dense projections FP8 at load, a 256K BF16 KV pool, four request slots, a 3 GiB prefix cache; `engine.draft_vocab` (a `.npy` id set for the draft head) is opt-in and off | T=1: `--no-mtp` |
| [cluster_glm-4.7_nvfp4_w4.example.json](cluster_glm-4.7_nvfp4_w4.example.json) | GLM-4.7 NVFP4 on four nodes: MTP depth 1, 256K context, four request slots | T=1: `--no-mtp`; depth 2: `--mtp-depth 2` (single-stream +4–13 %, measured behind depth 1 under concurrency before the 2026-09-14 lowering) |
| [cluster_glm-5.3_int4-int8_w4.example.json](cluster_glm-5.3_int4-int8_w4.example.json) | the full GLM-5.3 (int4/int8 RTN) on four nodes: MTP depth 1, eight request slots (sixteen decode rows; c=4 the four-slot shape's 41–42 tok/s, c=8 48–50 aggregate), 120K bf16 context, the embedding vocab-sharded; the BF16 decode weights resident in their 12-bit form alone (`"bf12"`: 110.1 GiB per rank under the 4 GiB headroom — 0.3 GiB under the BF16 plan, where both forms resident stopped at 100K) | both forms resident (100K context): `--bf16-weights bf12+bf16 --kv-capacity 102400`; T=1: `--no-mtp` (144K context with `--kv-capacity 147456`); the fp8 latent cache at 208K: `--kv-dtype fp8 --kv-capacity 212992 --prefix-cache-gib 1.5`; depth 2 at two slots: `--mtp-depth 2 --max-concurrency 2` |
| [cluster_mimo-v2.6-flash_mxfp4-fp8_w4.example.json](cluster_mimo-v2.6-flash_mxfp4-fp8_w4.example.json) | MiMo-V2.6-Flash as shipped on four nodes: MTP depth 1 (the first of the release's three draft layers), the BF16 o_proj / head / eh_proj resident in their 12-bit form alone (`"bf12"`), 128K context (every layer's K/V paged, the sliding-window layers read theirs through the window), four request slots; text prompts only (the vision and audio encoders are not served) | T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_mimo-v2.6-flash_mxfp4-fp8_w2.example.json](cluster_mimo-v2.6-flash_mxfp4-fp8_w2.example.json) | the same on two nodes: MTP depth 1, the K/V cache in the fp8 row form (`"kv_dtype": "fp8"`: 58 KiB per token per rank against 109 in bf16), 256K context (97.5 GiB of the 121.6 GiB per rank), four request slots | T=1: `--no-mtp` |
| [cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json](cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json) | DeepSeek-V4.1-Flash as shipped on four nodes: six request slots at DSpark depth 4 (30 decode rows in one batched replay, the family's 32-row cap) with the confidence-scheduled verify depth (λ 0.045), the bounded prefill, 128K context — the six-stream shape the vLLM recipe reports its aggregate at; single and dual streams measure the same as the two-slot shapes did | the two-slot depth-5 shape: `--max-concurrency 2 --mtp-depth 5`; the two-slot depth-4 shape: `--max-concurrency 2`; T=1 at four slots: `--no-mtp --max-concurrency 4` |
| [cluster_deepseek-v4-flash_mxfp4-fp8_w4.example.json](cluster_deepseek-v4-flash_mxfp4-fp8_w4.example.json) | DeepSeek-V4-Flash (the 0731 release) as shipped on four nodes: MXFP4 routed experts and FP8 dense projections consumed as they are, the DSpark block draft (five drafts) under the confidence-scheduled verify depth (`"mtp_schedule": true` with this world's measured row and base costs: a pass verifies as many drafts as their survival pays for, any depth from one to five — mostly two or three on prose, the whole block on code and JSON), sampled requests scheduled too with the draft's most likely tokens (`engine.mtp_draft` greedy for this family, `engine.mtp_schedule_sampled_scale` 0.93), six request slots, the model's full 1M-token context (63.2 GiB per rank with the 14 GiB prefix cache; the compressed caches take 6.2 GiB per rank at 1M) | T=1: `--no-mtp`; a fixed depth instead of the schedule: set `"mtp_schedule": false` and `"mtp_depth"` (3 measured best fixed on prose, 5 on code and JSON); sampled requests drafting by draw: `--mtp-draft sampled` |
| [cluster_deepseek-v4-flash_mxfp4-fp8_w2.example.json](cluster_deepseek-v4-flash_mxfp4-fp8_w2.example.json) | the same on two nodes: the scheduled DSpark depth with the two-node costs, four request slots, the full 1M-token context (95.8 GiB of the 121.6 GiB per rank with an 8 GiB prefix cache) | T=1: `--no-mtp` |
| [cluster_qwen3.8-27b_fp8_w1.example.json](cluster_qwen3.8-27b_fp8_w1.example.json) | Qwen3.8-27B-FP8 on one Spark: the dense 27B (48 Gated-DeltaNet + 16 full-attention layers) from its native block-FP8 checkpoint with the DFlash2 block drafter (`engine.dflash_model: z-lab/Qwen3.8-27B-DFlash2`, five bidirectional draft layers off five target taps, a block of eight; the eager world-1 path, no MTP), 256K context, eight request slots, the lm head requantized to block FP8 (`dense_weights: "fp8"`), the drafter's bf16 matrices packed lossless 12-bit (`bf16_weights: "bf12"`, −3 % a step), the rest exact. The drafter is the faster single-node recipe on every prompt class (C1 18–40 tok/s greedy against MTP depth 2's 16–22; the same model, an exact verify), so the MTP world ships as this template's mode, not a second file | plain decode: `--no-dflash` (the resident plain world on the decode graph); MTP depth 2: `--no-dflash --mtp --mtp-depth 2`; the drafter on the graph engine: `--decode-graph` (transcripts identical to the eager engine's, chat 71 vs 95 ms/token); the checkpoint's BF16 head: `--dense-weights checkpoint`; the opt-in prefill recipe `--prefill-fp8-per-tensor`; `--dflash-depth N` verifies the first N drafts |
| [cluster_qwen3.8-27b_fp8_w2.example.json](cluster_qwen3.8-27b_fp8_w2.example.json) | Qwen3.8-27B-FP8 on two Sparks (tensor parallel: the GDN and attention heads, the MLP and the lm head split across the ranks, two boundary folds a layer), MTP depth 3 with the decode graph (depth 3 leads depth 2 by 6–18 % across the classes), 256K context, eight request slots, the FP8 lm head. Greedy C1 34 / 41 / 46 / 43 / 31 tok/s (prose / code / JSON / math / chat) at 80 ms a pass; the DFlash2 drafter mode 32 / 50 / 71 / 58 / 30 at 94 ms, ahead on code / JSON / math and level on prose and chat | T=1: `--no-mtp`; depth 2: `--mtp-depth 2`; the DFlash2 block drafter (sharded across the ranks, the block proposal in the graph step): `--no-mtp --dflash-model z-lab/Qwen3.8-27B-DFlash2 --bf16-weights bf12`; the checkpoint's BF16 head: `--dense-weights checkpoint` |
| [cluster_qwen3.8-27b_fp8_w4.example.json](cluster_qwen3.8-27b_fp8_w4.example.json) | Qwen3.8-27B-FP8 on four Sparks (tensor parallel: the GDN and attention heads — one kv head a rank — the MLP and the lm head split across the ranks, two boundary folds a layer), MTP depth 3 (the public TP=4 reference's three speculative tokens) with the decode graph, 256K context, eight request slots, the FP8 lm head. Greedy C1 58 / 70 / 80 / 73 / 50 tok/s (prose / code / JSON / math / chat) at 46 ms a pass; the DFlash2 drafter mode 55 / 86 / 121 / 101 / 53 at 52 ms, ahead on code / JSON / math by 29–58 % and level on prose and chat | T=1: `--no-mtp`; depth 2: `--mtp-depth 2`; the DFlash2 block drafter: `--no-mtp --dflash-model z-lab/Qwen3.8-27B-DFlash2 --bf16-weights bf12`; the checkpoint's BF16 head: `--dense-weights checkpoint` |
| [cluster_qwen3-next-80b_nvfp4_w1.example.json](cluster_qwen3-next-80b_nvfp4_w1.example.json) | Qwen3-Next-80B-A3B-Instruct NVFP4 on one Spark (the Qwen3Next dialect of the `qwen3_5` stack, [6IXSERVE.md](../6IXSERVE.md)): MTP depth 2, the dense projections and the head FP8 at load, eight request slots, a 524288-token pool, interleaved prefill (256-token busy / 4096-token idle budgets, four decode passes a chunk), answer limits clamped | the BF16 dense stack: `--dense-weights checkpoint`; T=1: `--no-mtp` |
| [cluster_qwen3-next-80b_nvfp4_w1_yarn512k.example.json](cluster_qwen3-next-80b_nvfp4_w1_yarn512k.example.json) | the same with the opt-in YaRN ramp (`engine.rope_scaling` yarn ×2 over the checkpoint's 262 144 positions: one request reaches 524 288 tokens), four request slots, a 565 248-token pool; unit-tested, not yet verified on a GPU | the plain 262K template: [cluster_qwen3-next-80b_nvfp4_w1.example.json](cluster_qwen3-next-80b_nvfp4_w1.example.json) |
| [cluster_qwen3-next-80b_nvfp4_w2.example.json](cluster_qwen3-next-80b_nvfp4_w2.example.json) | Qwen3-Next-80B-A3B-Instruct NVFP4 over two Sparks (tensor parallel: each rank holds half of every expert's rows, of the shared expert, of the attention and Gated-DeltaNet heads and of the vocabulary; the router is replicated): the one-Spark template's slots, pool and draft depth, without the scheduler settings that are single-rank only for now (`decode_passes_per_prefill`, `prefill_order`, `prefill_budget_per_reader`, `prefill_group`). Measured on DGXone + DGXtwo 2026-10-04: about 1.6× one Spark at 1 to 8 streams, 38.4 GiB a box; the two-rank forward check has not been run ([record](../models/shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md)). A second engine on the same boxes needs `DGPP_BUS_ENGINE_CPU` set to another fast core | one Spark: [cluster_qwen3-next-80b_nvfp4_w1.example.json](cluster_qwen3-next-80b_nvfp4_w1.example.json) |
| [cluster_qwen3-coder-next_nvfp4_w1.example.json](cluster_qwen3-coder-next_nvfp4_w1.example.json) | Qwen3-Coder-Next on one Spark from `RedHatAI/Qwen3-Coder-Next-NVFP4` (the same model class in its compressed-tensors NVFP4 container: attention q/k/v/o, shared expert and routed experts NVFP4, the Gated DeltaNet and the router BF16): no draft layer in the checkpoint, so no MTP — the plain decode graph; the 80B template's slots, pool and prefill budgets; tool calls in the template's `<function=...>` XML form. Written 2026-10-04 from the checkpoint's headers; NOT yet run on a GPU | the BF16 dense stack (the checkpoint's own values, no FP8 re-encode): `--dense-weights checkpoint` |
| [cluster_qwen3.6-35b-a3b_nvfp4_w1.example.json](cluster_qwen3.6-35b-a3b_nvfp4_w1.example.json) | Qwen3.6-35B-A3B on one Spark from `nvidia/Qwen3.6-35B-A3B-NVFP4` (the Qwen3.5 dialect of the `qwen3_5` stack with the routed MoE: 256 experts top-8; modelopt mixed precision — NVFP4 experts, shared expert and head, per-tensor FP8 Gated-DeltaNet and attention projections, a BF16 draft layer): MTP depth 2 with the decode graph, eight request slots, a 262144-token pool, about 21 GiB of weights; text only (the vision tower is not served). Written 2026-10-04 from the checkpoint's headers; NOT yet run on a GPU | the shared expert and draft layer as shipped: `--dense-weights checkpoint`; T=1: `--no-mtp` |
| [cluster_qwen3.6-35b-a3b_nvfp4_w2.example.json](cluster_qwen3.6-35b-a3b_nvfp4_w2.example.json) | Qwen3.6-35B-A3B NVFP4 over two Sparks (tensor parallel, as the 80B's two-Spark template): the one-Spark template's slots, pool and draft depth, without `decode_passes_per_prefill` and `prefill_order` (single-rank only for now). Measured on DGXone + DGXtwo 2026-10-04: 154 tok/s at one stream, 368 at eight, 22.9 GiB a box; no gate run on the merged engine yet ([record](../models/shared/dgxone_dgxtwo_two_box_80b_35b_2026-10-04.md)) | one Spark: [cluster_qwen3.6-35b-a3b_nvfp4_w1.example.json](cluster_qwen3.6-35b-a3b_nvfp4_w1.example.json) |
| [cluster_qwen3.6-35b-a3b_fp8_w1.example.json](cluster_qwen3.6-35b-a3b_fp8_w1.example.json) | the same model from `Qwen/Qwen3.6-35B-A3B-FP8` (block FP8 throughout, the routed experts and the draft layer included; about 34 GiB of weights): the same shape; `dense_weights: "fp8"` requantizes the BF16 head alone. NOT yet run on a GPU | the checkpoint's BF16 head: `--dense-weights checkpoint`; T=1: `--no-mtp` |
| [cluster_qwen3.5-122b-a10b_nvfp4_w1.example.json](cluster_qwen3.5-122b-a10b_nvfp4_w1.example.json) | Qwen3.5-122B-A10B on one Spark from `nvidia/Qwen3.5-122B-A10B-NVFP4` (the 35B's model class at hidden 3072, 48 layers, 256 experts top-8; modelopt NVFP4 on the routed experts alone — the Gated DeltaNet, the attention, the shared expert, the head and the draft layer ship BF16): `dense_weights: "fp8"` encodes those BF16 matrices to block FP8 at load, about 70 GiB of weights (75 GiB as shipped), MTP depth 2 with the decode graph, eight request slots, a 262144-token pool (6 GiB); text only. `Sehyo/Qwen3.5-122B-A10B-NVFP4` (compressed-tensors NVFP4 on the attention, the shared expert and the experts) binds under the same template; `Qwen/Qwen3.5-122B-A10B-FP8` binds too, but its 118 GiB of weights leave no room for a pool on a 121 GiB Spark. Written 2026-10-04 from the checkpoints' headers; NOT yet run on a GPU | the BF16 matrices as shipped: `--dense-weights checkpoint`; T=1: `--no-mtp` |
| [cluster_qwen3.5-122b-a10b_nvfp4-sehyo_w1.example.json](cluster_qwen3.5-122b-a10b_nvfp4-sehyo_w1.example.json) | Qwen3.5-122B-A10B on one Spark from `Sehyo/Qwen3.5-122B-A10B-NVFP4` (compressed-tensors NVFP4 on the attention, the shared expert and the experts), the deployment verified on DGXone on 2026-10-04 ([record](../models/qwen3.5-122b-a10b/one-box/results-2026-10-04-dgxone.md)): `dense_weights: "fp8"`, MTP depth 2 with the decode graph, eight request slots, a 65536-token pool, a 1 GiB prefix arena; text only | the BF16 matrices as shipped: `--dense-weights checkpoint`; T=1: `--no-mtp` |
| [cluster_qwen3.5-0.8b_bf16_w1.example.json](cluster_qwen3.5-0.8b_bf16_w1.example.json) | Qwen3.5-0.8B on one Spark from its unquantized checkpoint (`Qwen/Qwen3.5-0.8B`: the dense Qwen3.5 dialect, 24 layers, every matrix BF16, tied embeddings — the head reads the embedding): the BF16 matrices bound as they ship (`dense_weights: "checkpoint"`, the dense MLP through the BF16 GEMM), MTP depth 2 with the decode graph, eight request slots, a 262144-token pool, about 1.4 GiB of weights; thinking is off unless a request sets `enable_thinking`; text only. Written 2026-10-04 from the checkpoint's headers; NOT yet run on a GPU | the 27B's FP8 paths (every dense matrix and the head encoded to block FP8 at load): `--dense-weights fp8`; T=1: `--no-mtp` |
| [cluster_gemma-4-31b_nvfp4_w1.example.json](cluster_gemma-4-31b_nvfp4_w1.example.json) | **NOT SERVABLE YET.** Gemma-4-31B on one Spark from `nvidia/Gemma-4-31B-IT-NVFP4` — the shape the deployment is meant to take once the family has a serving engine: text only, no MTP (the checkpoint has no draft head), four request slots, a 262144-token pool. Today `6ix-Serve` refuses a Gemma 4 checkpoint by name; the family's host side (config, tensor table, host reference, tokenizer, template, call notation) is in the build and a draft GPU forward check sits behind `-DDGPP_BUILD_GEMMA_DRAFT=ON` ([gpu-steps](../sixlabs/ports/gemma-4-31b/gpu-steps.md)). The numbers here are arithmetic, not measurements: 29.3 GiB of weights as shipped, 80 KiB of BF16 K/V per token for the ten full-attention layers (20 GiB for this pool), and — if the fifty sliding layers keep a 1024-row ring per slot rather than paging every token (800 KiB a token) — 0.78 GiB per slot for them | none until it runs |
| [cluster_gemma-4-26b-a4b_nvfp4_w1.example.json](cluster_gemma-4-26b-a4b_nvfp4_w1.example.json) | **NOT SERVABLE YET.** Gemma-4-26B-A4B on one Spark from `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16` (a community weight-only NVFP4 release: attention, dense MLP and the 128 experts of every layer in NVFP4, the routers BF16) — the same family and the same state as the row above: refused by `6ix-Serve` today, host side tested, a draft forward check behind `-DDGPP_BUILD_GEMMA_DRAFT=ON` ([gpu-steps](../sixlabs/ports/gemma-4-26b-a4b/gpu-steps.md)). Arithmetic, not measurements: 14.2 GiB of weights as shipped, 20 KiB of BF16 K/V per token for the five full-attention layers (5 GiB for this pool), 0.2 GiB per slot for the twenty-five sliding layers' 1024-row rings (200 KiB a token if paged instead) | none until it runs |
| [cluster_mistral-small-4_nvfp4_w1.example.json](cluster_mistral-small-4_nvfp4_w1.example.json) | **NOT SERVABLE YET (2026-10-04).** Mistral-Small-4 on one Spark from `mistralai/Mistral-Small-4-119B-2603-NVFP4` (Mistral-native `params.json`; latent attention over a softmax-routed MoE, compressed-tensors NVFP4 experts): the shape the plan sizes — 65.14 GiB of text weights as stored, a 262144-token latent pool of 5.6 GiB, four request slots, the plain decode graph (the checkpoint has no draft layer). The host side (config, binding table, references) is built and tested; the server refuses the family by name until its device assembly exists ([docs/mistral_small4_plan.md](../docs/mistral_small4_plan.md)). Text only | none measured |
| [cluster_minimax-m2.7_nvfp4_w2.example.json](cluster_minimax-m2.7_nvfp4_w2.example.json) | **NOT SERVABLE YET (2026-10-04).** MiniMax-M2.7 on two Sparks from `lukealonso/MiniMax-M2.7-NVFP4` (GQA with a per-layer q/k norm over a sigmoid-routed MoE, modelopt NVFP4 experts; 125.2 GiB of weights, more than one Spark holds): the shape the sharding plan sizes — 64.5 GiB of weights and a 196608-token K/V pool of 23.2 GiB per rank, four request slots, the plain decode graph (config.json declares an MTP head no checkpoint carries). The host side is built and tested; the server refuses the family by name ([docs/minimax_m27_plan.md](../docs/minimax_m27_plan.md)) | none measured |
| [cluster_qwen3-vl-30b-a3b_nvfp4_w1.example.json](cluster_qwen3-vl-30b-a3b_nvfp4_w1.example.json) | Qwen3-VL-30B-A3B-Instruct on one Spark from `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4`, TEXT PATH ONLY (the plain Qwen3 family, `models/qwen3`: full attention in every layer, a routed MoE with no shared expert, compressed-tensors NVFP4 — attention q/k/v/o and experts NVFP4, router and head BF16): the checkpoint's vision tower is bound and not served, image inputs are refused by name; no draft layer, so no MTP — the plain decode graph; eight request slots, a 262144-token pool (about 24 GiB of K/V beside about 18 GiB of weights); tool calls in the template's JSON form. Written 2026-10-04 from the checkpoint's headers. **Needs a build configured with `-DDGPP_BUILD_QWEN3_PLAIN_DRAFT=ON`** (the family's GPU walk is an unverified draft, off by default: the server otherwise refuses the checkpoint by name); NOT yet compiled or run on a GPU — [gpu-steps](../sixlabs/ports/qwen3-vl-30b-a3b/gpu-steps.md) | none measured |

The Qwen NVFP4 templates use `engine.fp8_head: "mma"` after matched
[real-checkpoint numerical checks](../benchmarks/results/2026-09-21-qwen-fp8-head-numerics.md)
on one and two Sparks, including the YaRN recipe. Streaming MMA applies above
the dense GEMV threshold and up to the configured decode-row capacity; short
prefills in that interval use it too. Pass `--knobs "--fp8-head gemv"` to
restore the previous head path. For BF16 dense weights, use
`--knobs "--dense-weights checkpoint --fp8-head gemv"`. The FP8-checkpoint
templates retain their BF16 dense/head stack and do not select this setting.

`engine.bf16_weights` (`"checkpoint"` by default; every template sets `"bf12"`
or `"bf12+bf16"`) keeps a lossless 12-bit form of each BF16 matrix the decode
GEMV streams — the sign+mantissa byte and a 4-bit exponent code, exact side
tables for the outliers — so a decode launch of up to eight rows reads 0.75 of
those bytes and returns the same bits (the transcripts do not move; it is a
storage format, not a quantization). Measured single-stream decode:
GLM-5.3-Flash +6–7 % (+4–6 % at two to four live requests), GLM-4.7 +11–12 %
(+5–7 % at four), the full GLM-5.3 +5–6.5 %. The two values differ in what stays
resident, and the memory plan carries either:

| value | resident | memory against `checkpoint` | prefill |
|---|---|---|---|
| `"bf12+bf16"` | both forms: decode streams the 12-bit one, prefill reads the BF16 bytes in place | GLM-5.3-Flash +2.0 GiB per rank on four nodes (+3.8 on two), GLM-4.7 +4.9, the full GLM-5.3 +1.3 | unchanged |
| `"bf12"` | the 12-bit form ALONE: each matrix's BF16 bytes go back to the node as its layer loads, and a prefill GEMM expands the rows it reads into a small scratch (the same bits, so the same results) | GLM-5.3-Flash −0.5 GiB per rank on four nodes (−1.0 on two), GLM-4.7 −1.4, the full GLM-5.3 −0.3 | about 10 ms per prefill chunk on four-node GLM-5.3-Flash (20 on two nodes): +1–2 % on 8K–32K prompts, +3–4 % at 2K, +25–40 ms to a short prompt's first token; within 1 % on the full GLM-5.3 |

The templates with memory to spare (four-node GLM-5.3-Flash, GLM-4.7) take
`"bf12+bf16"`; the two sized to their nodes' ceiling (two-node GLM-5.3-Flash,
the full GLM-5.3) take `"bf12"`, which is what returned their contexts to 160K
and 120K. Decode is the same either way — identical transcripts, measured on
every family. Under `"bf12"` a decode batch past eight rows (the full
GLM-5.3's sixteen-row shape) runs eight-row packed launches, level with the
BF16 algorithm it replaces. On Qwen3.8-Flash-Next-FP8 the key packs the GDN and
QSA projections, the draft block's and the head — 1.65 → 1.24 GiB per rank on
four nodes, 3.20 → 2.41 on two — for +6.4 / +3.8 / +2.4 / +1.3 % at one to
four live requests on four nodes and +9.0 / +6.7 / +6.6 / +4.6 % on two,
transcripts identical; that family keeps both forms resident under either
value (every Qwen recipe has the room), its GR sites, routers and shared
experts stay BF16 (they are read warm behind the prefetcher, where the packed
form loses), and under `dense_weights: "fp8"` (the NVFP4 templates) the
projections and the head are already FP8, so nothing is packed. DeepSeek
accepts the key and packs nothing yet (its BF16 sites ride the tensor-core
kernels). `--bf16-weights checkpoint` restores the BF16-only form.

`engine.embed_sharding` (`"replicated"` by default, `"vocab"` in the full
GLM-5.3 template) decides whether every rank holds the whole embedding table
or its lm-head slice of the rows: `vocab` frees 1.33 GiB per rank at world 4
for one small fold per token lookup and changes no number; the other families
ignore it. `kv_dtype` affects only the GLM-5.3 latent caches (bf16, fp8 or
fp4); Qwen's and GLM-4.7's K/V caches stay BF16.

## Single-Spark NVIDIA Qwen NVFP4 defaults

The NVIDIA NVFP4 `w1` example uses a 262144-token shared KV pool and sets
both `prefill_budget_tokens` and `prefill_idle_budget_tokens` to 4096.
These budgets select the engine's supported 4K prefill chunks both when
idle and when another request is generating. The profile retains MTP depth 1,
full admission, four request slots and its existing precision settings.

The pool is shared across those slots. It permits a single request to approach
the checkpoint's 262144-token total context, including its generated output;
it does not reserve 256K for each of four simultaneous requests. Full admission
accounts for the prompt plus requested output, and cached prefixes also occupy
pool blocks. Increase KV capacity separately when concurrent contexts need more
space, using the engine's startup memory plan.

Larger chunks favor cold-prompt processing and can interrupt existing output
streams for seconds. For frequent interactive arrivals, a smaller busy budget
can keep decode opportunities closer together while retaining 4096 when idle.
For example, append this override on the next normal start:

```bash
python3 scripts/dgpp-cluster up \
  --config deploy/cluster_qwen-3.8-flash-next_nvfp4_w1.json \
  --knobs "--prefill-budget-tokens 256 --prefill-idle-budget-tokens 4096"
```

Existing local deployment files are not overwritten by an example update.
Copy the three fields into the local config and use the normal restart process
to adopt them.

Since 2026-09-28 the templates give the prefix arena (`prefix_cache_gib`) the
memory the node has left under the memory plan's 4 GiB headroom, less a 1 GiB
margin, because a snapshot slot is only ever spent on a prompt above
`engine.prefix_min_tokens` (1024 by default) and an idle GiB buys nothing. The
one-node Qwen templates are the exception: their n-gram table is mmap'ed, and
the memory they leave free is the page cache that keeps fresh-text prefill at
full speed, so they stay at 3 GiB (27 snapshots). The
[sizing guide](../docs/prefix-cache.md) has every template's slot count and the
entry policy's keys. See the [single-GB10 observations](../benchmarks/results/2026-09-27-qwen-nvfp4-w1-prefill.md)
for measured prefill performance, concurrent-stream effects and setup limits.

## The 512K YaRN template (`engine.rope_scaling`)

[cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json)
is ready to run on the two-Spark kit: copy it without `.example`, fill the site
fields through `.env` as usual, and start the world with
`scripts/dgpp-cluster up --config deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.json`.
The one setting it adds over the plain two-Spark template is the rope ramp:

```json
"rope_scaling": {"rope_type": "yarn", "factor": 2.0,
                 "original_max_position_embeddings": 262144,
                 "beta_fast": 32, "beta_slow": 1, "attn_factor": 1.0,
                 "mrope_cache_factor": 4.0}
```

**YaRN raises the positional ceiling. It does not enlarge the K/V pool.** The
ramp re-derives the rope table's inverse frequencies so a token can be *placed*
at a position up to `original_max_position_embeddings × factor` = 524 288; the
pool the tokens are *stored* in is `engine.kv_capacity`, and the ramp moves no
byte of it (`docs/qwen38_flash_next_plan.md` §1.9.1; the plan check asserts the
total is identical with the knob on and off). Two numbers follow from that, and
`dgpp-serve` prints the smaller of them at startup and on `/v1/models` as the
request context limit:

- the **ceiling**: 524 288 positions with this template (262 144 without it),
- the **pool**: `kv_capacity` rounded up to the 64-token block, ~13 KB per
  token per rank at world 2 (`docs/qwen38_checkpoint_budget.md`).

The template sets `kv_capacity` 565 248 — the ceiling plus the 32 768-token
`default_max_tokens` the template carries plus 8 192 tokens of room, because a
request reserves its prompt *and* its answer from the pool: a 524 288-token
prompt from a client that omits `max_tokens` needs 557 056, and a pool cut at
the ceiling would refuse the longest request it exists to serve. With
`max_concurrency` 2 that pool seats one full-length stream, or two streams of up
to ~282 000 tokens each (~249 000 each when both omit `max_tokens`), or any mix
that fits; a third request waits at the door (`queue_limit` 8) rather than
sharing bytes. For two concurrent 524 288-token streams set `--kv-capacity
1114112` (1 048 576 when both clients send their own, smaller `max_tokens`).

Memory headroom: at TP=2 the resident weights are 44.1 GiB per rank, the pool is
7.45 GiB at 565 248 tokens (14.68 GiB at 1 114 112) and the plan's activations,
the prefix arena and the engine's tables come on top — 96.8 GiB per rank for
this shape with its 40 GiB arena (56.8 GiB without it, as the plan of
2026-09-28 measures it), plus the 4 GiB headroom the pre-flight check adds
(`--memory-plan` prints the itemised total and refuses the configuration before
the first allocation if the node cannot cover it). The 1 114 112-token variant
adds about 9.3 GiB per rank, which is why the arena stops at 40 GiB rather than
filling the node. Nothing about the rope ramp changes those numbers: they move
with `kv_capacity`, `max_concurrency` and `prefix_cache_gib`.

The 40 GiB prefix arena holds 741 snapshots per rank at TP=2.
Long prompts also retain an earlier chunk snapshot so changed questions can
share the document's KV blocks. Snapshot size does not grow with context;
KV-pool capacity limits how many independent long documents remain cached.
See [prefix-cache sizing](../docs/prefix-cache.md) for all recipes and retention
estimates before increasing the arena budget.

Two things to check on the boot log, both added with this knob:

- `serve: request context limit 524288 tokens — the lesser of the 524288-token
  positional ceiling (engine.rope_scaling yarn x2 over 262144 positions) and the
  565248-token K/V pool …` — if the ceiling it names is 262144 the knob did not
  reach the process,
- `engine.rope_scaling is set but the loaded model is the <family> family` is a
  **startup error**, not a note: the ramp is built for the Qwen3.8-Flash-Next
  QSA rope and no other family's, and the world refuses to start with an
  operator's setting left unapplied.

The checkpoint must be a plain one (`rope_parameters.rope_type: "default"`),
which the NVFP4 and FP8 releases are; a checkpoint that already bakes YaRN into
its own `rope_parameters` is rejected at load, for the reason
`docs/qwen38_flash_next_plan.md` §1.9.1 gives.
`scripts/qwen_yarn_release_check.py` is the opt-in, long-context acceptance run
for this template ([docs/qwen_yarn_release_check.md](../docs/qwen_yarn_release_check.md)).

## The consolidation (2026-09-14)

The earlier consolidation reduced twenty-five templates to eight. The new
two-Spark NVFP4 recipe brings the tracked set to nine. Every `_plain` variant
(MTP measured faster per token in every family), every `_mtp2` / `_mtp5`
variant (a depth is
a knob), the `_c6` / `_c8` slot variants (the wider shape measured at or above
the narrower one at every concurrency, so it is the template), the
`_large-cache` variants (the four-node GLM-5.3-Flash template took the large
cache; the two-node one and the full GLM-5.3's fp8 cache are knobs) and the
Qwen single-Spark BF16-dense variants (the FP8 dense stack measured faster).
The retired names map to the table's third column; historical changelog
entries and benchmark records keep the old names.

| Retired template | Now |
|---|---|
| `cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1` (8K context), `…_w4_mtp1_large-cache` | `cluster_glm-5.3-flash_nvfp4-fp8_w4` (the large cache) |
| `cluster_glm-5.3-flash_nvfp4-fp8_w2_mtp1`, `…_w2_mtp1_large-cache` | `cluster_glm-5.3-flash_nvfp4-fp8_w2` (four slots, 160K; 132K since the 2026-09-19 companions) |
| `cluster_qwen-3.8-flash-next_fp8_w4_{mtp1,plain}`, `…_w2_{mtp1,plain}` | `cluster_qwen-3.8-flash-next_fp8_w4`, `…_w2` |
| `cluster_qwen-3.8-flash-next_nvfp4_w1_{mtp1,plain,mtp1_dense-fp8,plain_dense-fp8,mtp2_dense-fp8}` | `cluster_qwen-3.8-flash-next_nvfp4_w1` (FP8 dense) |
| `cluster_glm-4.7_nvfp4_w4_{mtp1,plain,mtp2}` | `cluster_glm-4.7_nvfp4_w4` |
| `cluster_glm-5.3_int4-int8_w4_{plain,mtp1,mtp1_large-cache,mtp1_c8,mtp2}` | `cluster_glm-5.3_int4-int8_w4` (eight slots) |
| `cluster_deepseek-v4.1-flash_mxfp4-fp8_w4_{plain,mtp4,mtp4_c6,mtp5}` | `cluster_deepseek-v4.1-flash_mxfp4-fp8_w4` (six slots, depth 4) |

## Existing deployments and logs

Stop a running deployment using its old config path **before** renaming the
file. Log and staging namespaces are derived from the absolute config path;
the new filename gets a new namespace. Existing logs are not moved or deleted,
and a new filename does not take ownership of a process started with the old
one. Update saved commands and automation to use the new name.

The generic runtime files `cluster.resolved.json` and the peer's staged
`cluster.json` are unchanged: they are generated by the launcher, not templates.
