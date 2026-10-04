# Qwen3-235B-A22B-Instruct-2507 (NVFP4) on two Sparks — the sharding plan

Written 2026-10-04 from the checkpoint's `config.json` and safetensors headers (no weight bytes
read) and from the engine's own placement rules (DESIGN.md §5.1–5.2, §6; `src/models/glm4`,
`src/models/mimo`). **A plan: nothing in it has run.** The arithmetic is the binding table's —
`qwen3_rank_resident_bytes` in `src/models/qwen3/binding.cpp`, pinned by the unit test
`qwen3_tp_geometry_and_rank_bytes` and printed by `qwen3_bind_check --world 2`.

| | |
|---|---|
| checkpoint | `nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4` @ `1c4ec35802abf3a20672be7e5c51a098796f71d6`: 28 shards, **145,703 tensors, 139,202,447,840 bytes (129.6 GiB)** |
| model | `Qwen3MoeForCausalLM`: 94 layers, hidden 4096, 64 query / 4 KV heads of 128 (16 query heads per KV head), full rotary theta 5e6, 128 experts top-8 of intermediate 1536, no shared expert, vocab 151936, untied head, 262,144 positions |
| container | modelopt NVFP4: the routed experts and `o_proj` NVFP4 (`weight` U8 [N, K/2], `weight_scale` e4m3 [N, K/16], `weight_scale_2` F32 scalar — a multiplier); `q/k/v_proj`, the router, the head, the norms BF16; `k_scale` / `v_scale` and `input_scale` bound and unread |

## Why two nodes

Resident at world 1 the model is **143,736,344,576 bytes (133.9 GiB)**: more than the file (the
NVFP4 `o_proj` is dequantized to BF16 for the dense GEMM) and more than one Spark holds. At world 2
each rank holds **72,540,655,616 bytes (67.56 GiB)**; at world 4, 36,942,811,136 (34.41 GiB).

## The split, tensor by tensor (rank r of W = 2)

The engine's tensor parallelism replicates the hidden state at every block boundary and shards
*inside* a block (DESIGN.md §5.1). For this model that is GLM-4.7's attention placement and MiMo's
expert placement, unchanged:

| tensor (per layer L of 94) | checkpoint form | split | rank r holds | resident form | bytes per rank per layer |
|---|---|---|---|---|---|
| `input_layernorm.weight`, `post_attention_layernorm.weight` [4096] | BF16 | replicated | all | BF16 | 16,384 |
| `self_attn.q_proj.weight` [8192, 4096] | BF16 | rows, by query head | rows [4096 r, 4096 (r+1)) — heads 32 r … 32 r + 31 | BF16 [4096, 4096] | 33,554,432 |
| `self_attn.k_proj.weight`, `v_proj.weight` [512, 4096] | BF16 | rows, by KV head | rows [256 r, 256 (r+1)) — KV heads 2 r, 2 r + 1 | BF16 [256, 4096] each | 4,194,304 |
| `self_attn.o_proj` [4096, 8192] | NVFP4 | **columns**, by query head (4096 r is a 16-block boundary) | columns [4096 r, 4096 (r+1)) | dequantized to BF16 [4096, 4096] at load | 33,554,432 |
| `self_attn.q_norm.weight`, `k_norm.weight` [128] | BF16 | replicated | all | BF16 | 512 |
| `mlp.gate.weight` [128, 4096] (the router) | BF16 | replicated — every rank must pick the same eight experts | all | BF16 | 1,048,576 |
| `mlp.experts.E.gate_proj`, `up_proj` [1536, 4096], E = 0…127 | NVFP4 | rows of the intermediate dim | rows [768 r, 768 (r+1)) | NVFP4 as shipped: codes [768, 2048], scales [768, 256], one F32 divisor (1 / `weight_scale_2`) | 2 × 1,769,476 per expert |
| `mlp.experts.E.down_proj` [4096, 1536] | NVFP4 | **columns** of the intermediate dim (768 r is a 16-block boundary) | columns [768 r, 768 (r+1)) | NVFP4: codes [4096, 384], scales [4096, 48], one divisor | 1,769,476 per expert |
| `k_proj.k_scale`, `v_proj.v_scale`, every `input_scale` | F32 | — | never read | — | 0 |

Globals: `model.embed_tokens.weight` [151936, 4096] replicated (a row gather; 1,244,659,712 bytes);
`model.norm.weight` replicated; `lm_head.weight` vocab-sharded, rows [75968 r, 75968 (r+1))
(622,329,856 bytes).

Every query head a rank holds reads a KV head the same rank holds (head h reads KV head h / 16;
rank r's heads 32 r … 32 r + 31 read KV heads 2 r, 2 r + 1), so attention needs no cross-rank K/V.

Per rank, summed:

| class | bytes per rank (W = 2) | GiB |
|---|---|---|
| routed experts (94 × 128 × 3 slices) | 63,871,005,696 | 59.48 |
| attention q/k/v/o (BF16) | 6,702,497,792 | 6.24 |
| embedding (replicated) | 1,244,659,712 | 1.16 |
| head (half the vocabulary) | 622,329,856 | 0.58 |
| routers (replicated) | 98,566,144 | 0.09 |
| norms (replicated) | 1,596,416 | 0.00 |
| **total** | **72,540,655,616** | **67.56** |

(The loader's 256-byte grant alignment adds a few megabytes; the memory plan's `model weights
(resident)` line is the number to read on the box.)

The checkpoint itself must sit on **both** boxes' NVMe (139.2 GB each): every rank reads its own
slices from a local copy. The byte reconcile the stream runs at boot must hold:
`source_bytes(rank 0) + source_bytes(rank 1) == world-1 source bytes + verbatim bytes` (the
replicated set is read once per rank), and the two ranks' replicated digests must be equal.

## State: the K/V pool

`Glm4KvPool`, 64-token blocks, BF16, per rank only its own KV heads: 94 layers × 2 KV heads × 128
× 2 (K and V) × 2 bytes = **96,256 bytes per token per rank**. A 262,144-token pool (the model's
positional ceiling) is 25.2 GB = **23.5 GiB per rank**; 131,072 tokens is 11.75 GiB. No other
per-request state (no recurrent layers): a prefix snapshot is a block list.

Per-rank memory at the proposed shape (4 slots, 262,144-token pool): weights 67.6 + K/V 23.5 =
91.1 GiB, plus the session core, the MoE and attention scratch and the loader's pinned staging
(about one layer, 0.7 GiB, until the stack is resident) — the engine's own memory plan
(`qwen3_forward_check --plan 4,262144,2`) is the authority, and the check at boot refuses a shape
that does not fit. For scale: GLM-4.7 serves at 77.4 GiB per rank on these boxes. If it is tight,
halve the pool.

## Collectives

Per layer, two all-reduce sums of the block output `[rows, 4096]` in BF16 — after `o_proj` (each
rank's partial over its own heads) and after the MoE (each rank's partial over its half of every
selected expert's intermediate dim) — exactly the two folds `Qwen3Model::enqueue_layer` stages.
94 layers make **188 collectives per decode step** (GLM-4.7's 93 layers record 186; the bus
recorder's budget is 256). A decode row is 8,192 bytes — one latency slot; the family's
`lat_slot_bytes` is rows × 8,192. The head is vocab-sharded and the pick merges the two halves
(the existing sampler collective). A 2048-token prefill chunk folds 16.8 MB per collective over the
bulk pool.

The MoE rounds its fp32 chain to BF16 once per rank *before* the fold (`launch_moe_round_bf16`), as
MiMo's routed-only chain does; the sum of two BF16 partials is then rounded by the reducer. World 2
is therefore tolerance-equal to world 1, not bitwise (GLM-4.7's world 2 showed one near-tie flip on
one fixture row).

## What a decode step reads (the speed floor, an estimate)

Per token per rank at W = 2, T = 1: attention 6.70 GB (BF16, every layer's q/k/v/o), the eight
selected experts' slices 94 × 8 × 3 × 1,769,472 = 3.99 GB, the head 0.62 GB, routers 0.10 GB —
**about 11.4 GB**. At the 240 GB/s the engine's other plans use for the weight stream that is a
48 ms floor, plus 188 collectives (47 µs each measured on GLM-4.7: about 9 ms): **of the order of
55–60 ms per token single-stream, unmeasured.** GLM-4.7 at world 4 reads 9.7 GB and measured 49 ms.

The levers, none built:

1. The BF16 attention projections are 59 % of the bytes. `o_proj` ships NVFP4 and is *expanded* 3.6×
   by the load-time dequant; serving it through the NVFP4 GEMV core (`launch_fp4_gemv_bf16`, the
   path GLM-4.7's dense MLP takes) would cut 3.15 GB per rank per token to 0.89. It needs an
   attention layer whose output projection takes a `GlmFp4Matrix`.
2. Block FP8 at load for q/k/v (the qwen3_5 stack's `engine.dense_weights = fp8`): halves the rest.
3. Four nodes: 34.4 GiB and about 5.8 GB per token per rank, 1 KV head and 16 query heads per rank
   (the geometry check accepts W = 4, 8 and 16; W = 3 is refused — 64 heads).

## What exists tonight, and what does not

Exists (all unverified on a GPU):
- the binding table at any world and its per-rank arithmetic (host, unit-tested);
- the draft loader's rank slices (`Qwen3LocalGeometry`, `Qwen3LoaderFamily::Builder`) with a fixture
  gate that loads both ranks of a world of 2 and compares every slice
  (`tests/cuda/qwen3_loader_test.cpp`, `qwen3_loader_resident_values_*`). It has run on the Mac
  against a CUDA runtime shim (host memory standing in for the device): `3 tests, 0 failed`. That
  checks the slice arithmetic and the byte formula, not a device;
- the walk's two folds (`Qwen3Model::stage` / `fold`, GLM-4.7's code path), inert at world 1.

Does not exist: any multi-rank run, test or tool for this family. The serving family **refuses
`world_size > 1` by name** (`Qwen3Family::refuse_world` in `apps/dgpp_serve.cpp`) so that a config
cannot reach an unverified fabric path. Lifting it is the last step of the ladder below.

## The ladder to world 2

Each step gates the next.

1. **World 1 first, on the 30B** (`sixlabs/ports/qwen3-vl-30b-a3b/gpu-steps.md` §3–7): the walk has
   to be right at all before it is right in halves.
2. **The 235B's own single-rank gate** (`gpu-steps.md` in this folder, §4): the full 94-layer
   diagnostic forward on ONE Spark in streaming residency against `tools/qwen3_reference.py`.
3. **Loader slices**: `qwen3_loader_test` — `qwen3_loader_resident_values_modelopt` covers both
   ranks of world 2 on the fixture. Then on the real checkpoint, a `--rank` / `--world` option on
   `qwen3_forward_check --plan` and a load-only pass per rank (model on `apps/glm4_load_check.cpp`)
   to see the byte reconcile and the digests on real shards.
4. **A loopback world of 2 on the fixture**, modelled line for line on `tests/cuda/glm4_tp_test.cpp`:
   two `Qwen3Model`s on one GPU joined by the loopback reducer; gates: the ranks' residuals are
   bitwise equal at every boundary, world 2 agrees with world 1 under the near-tie rule, prefill
   chunking and decode steps agree.
5. **The engines over the loopback world**, on `tests/cuda/glm4_engine_test.cpp`: the graph engine's
   scalar and batched replays against the eager engine, world 2 against world 1 on three prompts.
6. **Two Sparks**: remove `refuse_world`; boot from the JSON below; the op-stream digests equal on
   both ranks; `apps/serving_head_check.cpp` gains this architecture (it scores teacher-forced
   rows through the real bus) and its log-probabilities are compared with the reference's
   (`sixlabs/bench/compare_forward_check.py`) and with step 2's world-1 numbers.
7. First requests, then greedy transcripts at 1 and 4 streams against each other.

Proposed deployment (to become `deploy/cluster_qwen3-235b-a22b_nvfp4_w2.example.json` once step 6
passes — it is deliberately **not** in `deploy/` tonight, where every template is expected to boot):

```json
{
  "model": "nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4",
  "world_size": 2,
  "engine": {
    "max_concurrency": 4,
    "kv_capacity": 262144,
    "kv_dtype": "bf16",
    "default_max_tokens": 32768,
    "queue_limit": 8,
    "max_connections": 64,
    "decode_graph": true,
    "mtp": false,
    "prefix_cache_gib": 2,
    "admission": "full",
    "admission_window": 256,
    "model_alias": "Qwen3-235B-A22B",
    "sampling_candidates": 128,
    "stats_interval_s": 10,
    "max_tokens_overflow": "clamp"
  },
  "paths": { "resident_cache": "" }
}
```

(`tests/python/portability_test.py` will then need the model in its `models` and `no_draft` sets,
and `deploy/README.md` a catalogue row.)

## What I expect to be wrong

- The memory plan at W = 2 with a 262K pool may not leave the headroom the boot check wants; the
  91 GiB above excludes scratch I could not size without running `plan_memory`.
- The decode-rows ceiling and the attention split count (32, GLM-4.7's default for 12 query heads
  per KV head) are inherited, not tuned for 16 heads per KV head and 32 heads per rank.
- The first resident load reads 67.6 GiB per rank through the page cache; GLM-4.7's first boot took
  about three minutes to capture its image at 53 GiB.
