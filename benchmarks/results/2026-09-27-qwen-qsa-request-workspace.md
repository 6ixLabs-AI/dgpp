# Qwen QSA workspace bounded by request context

Tested source `9297c381262dc887d494636b557dd2a034bf883c`, based directly on
official `652cd40331c344c2a51f153768d7d3e23601f465`, on one GB10 with CUDA 13.0.
The only production changes are the memory-plan calculation and corresponding
`QwenQsaLayer` allocation in `src/models/qwen/forward.cpp`.

Scoring indexes logical keys belonging to one request. A shared KV pool larger
than the request's positional ceiling therefore need not enlarge every scoring
row. Use the existing effective request ceiling, rounded up by the indexer
compression ratio, for both planning and allocation. Physical KV capacity is
unchanged. The existing score launcher uses the smaller stride for its grid;
valid score arithmetic and the kernel API remain unchanged.

## Memory

For `nvidia/Qwen3.8-Flash-Next-NVFP4`, 4096 prefill rows, an 850048-token shared
pool, a 262144-token request ceiling and compression ratio four:

| Item | Previous | Bounded |
| --- | ---: | ---: |
| Scoring keys per row | 212512 | 65536 |
| Scoring-key workspace | 6.4854 GiB | 2.0000 GiB |
| Total startup memory plan | 113.68 GiB | 109.19 GiB |

The exact reduction is 4816109568 bytes (4.4853515625 GiB). Both server memory
plans used identical configuration: four slots, MTP-1, BF16 KV, FP8 dense/MMA,
`bf12+bf16`, mapped n-gram table, 1.5 GiB prefix snapshots, full admission and
4096/4096 prefill budgets. No precision, scheduling or pool setting changed.
There is no general throughput claim from this allocation comparison.

## Validation

- Fresh Release build and all 27 host checks passed. One host executable was
  initially omitted from the build list; after building it, its rerun passed.
- Twelve selected native Qwen groups passed, including memory-plan, forward,
  YaRN/context-boundary, QSA reference/selection, prefill and decode checks.
- The new scoring test compares wide and bounded strides bitwise for keys and
  selected tokens, with guard regions, inactive rows, shuffled physical KV
  blocks and graph replay. Cases cover 4096 rows with short/tail positions and
  eight rows at 262144/262145 context ceilings.
- The new world-of-one graph test compares a shared-pool-sized workspace with
  a 193-token per-request bound under identical four-slot MTP-1 schedules,
  including retirement and reuse while peers generate. BF16 and FP8/MMA
  transcripts match exactly.
- Existing session checks also pass with a 193-token fixture ceiling below
  shared capacity, including cached resume, snapshots, graph verification and MTP.
- Compute Sanitizer memcheck reports zero errors and zero leaked bytes for the
  new scoring test and the model's cross-context-boundary check.
- Five serial actual-checkpoint requests agree in all 435 generated token IDs,
  complete messages, finish reasons and cache counts. Inputs range from 67 to
  12401 tokens; cached cases reuse 12264 or 12376 tokens. Both arms use
  temperature 0, seed 42, xhigh and a 256-token diagnostic limit; every request
  completes naturally with `stop`.

Two registered world-of-two fabric groups fail initialization because the
machine exposes no RDMA devices (`1..2 lane devices required`). Their kernel-only
subcases do not substitute for fabric validation. The single-node replacement
above exercises the deployed topology; no multi-node coverage is claimed.

No full model benchmark or real-checkpoint 256K stress run was performed for
this change. The bounded checks establish the tested numerical and memory
properties, not a general quality score or invariance under arbitrary concurrent
request timing. The dynamic visible-context launch cap, workspace tiling and
compact-logit changes remain outside this patch.
