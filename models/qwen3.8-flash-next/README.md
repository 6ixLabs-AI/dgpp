# Qwen3.8-Flash-Next

One of the models upstream DGPP serves, and the one the fleet runs on upstream DGPP itself.
**6ixLabs has no speed measurement of it on this fork in this repository.** What is on record
here is one set of loadout switch times, upstream's published figures, and the fleet's earlier
SGLang figure. Registry entry: `qwen3.8-flash-next` in
[`sixlabs/registry/models.json`](../../sixlabs/registry/models.json).

## What it is

- Family `qwen4_exp`, served by upstream DGPP. A thinking model with vision served.
- Its checkpoint carries a draft head (31 draft tensors in the RadixArk NVFP4 release, by the
  [Coder-Next summary](../qwen3-coder-next/one-box/dgxone_coder_next_2026-10-04.md)'s table).
- Not a 6ixLabs port: the family, the templates and the measurements below are upstream's.

| Checkpoint | Format | Worlds with a template | Registry status |
|---|---|---|---|
| [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (the default) | nvfp4-modelopt | 1, 2 | `upstream` |
| [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | nvfp4-modelopt | 1, 2 | `upstream` |
| [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | fp8-block | 2, 4 | `upstream` |
| [`Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN`](https://huggingface.co/Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN) | int4-autoround | 1 | `upstream` |

The registry pins no revision for these.

## Status

In the registry's words, `upstream` means: "Served by upstream DGPP with a template and published
measurements; not verified on this fleet by 6ixLabs."

## What has run on this fleet

- **In production, on upstream DGPP.** `sixlabs/README.md` (2026-10-03): the production DGPP
  checkout `~/dgpp` on DGXone "serves Qwen3.8 and is never built in". That is upstream's build,
  not this fork's. The records write "Qwen3.8"; the registry's only other Qwen3.8 model, the 27B,
  is recorded as not run here, so this folder takes the fleet's Qwen3.8 loadout to be this model.
- **Loadout switching, 2026-10-04**
  ([record](../shared/dgxone_dgxtwo_loadout_switch_2026-10-04.md)): from the 80B + 35B pair to
  Qwen3.8 on DGPP the models answer after 50–69 s and the switch job completes after 68–86 s;
  back to the pair, 41–51 s and 92–104 s. Two runs each.
- **Text layer only, on this fork**: the tokenizer and chat-template tests pass on the
  Qwen3.8-Flash-Next corpus (2 of 2 and 10 of 10), with the RadixArk release standing in for the
  FP8 release the corpus names
  ([integration record](../../sixlabs/ports/integration-2026-10-04.md), section 3;
  [upstream-merge record](../../sixlabs/bench/results/upstream_merge_2026-10-04.md), section 3).
  No GPU was used.

## One box

Not measured by 6ixLabs.

## Two boxes

Not measured by 6ixLabs on this engine. The switch times above are the only two-box figures from
this fleet in the repository.

## Figures published by others, not measured here

| Source | Configuration | 1 stream, tok/s | 4 requests, tok/s |
|---|---|---|---|
| Upstream DGPP (`docs/benchmarks.md`) | NVFP4, one Spark, FP8 dense | 42.3–50.0 | 83.4–95.4 |
| Upstream DGPP (`docs/benchmarks.md`) | NVFP4, two Sparks, FP8 dense | 62.6–74.6 | 124.5–143.8 |
| Upstream DGPP (`docs/benchmarks.md`) | AutoRound int4, one Spark | 44.7–71.5 | 77.3–118.6 |
| Atlas, published | one Spark | 16.5; 19.1 with one draft token | — |

The Atlas row is as quoted in the Coder-Next summary's "speeds on record" table.

**The fleet's earlier figure on another engine:** SGLang, tensor parallel over both boxes,
2026-08-28: 43.5 tok/s at one stream and 122.3 at four. It is a fleet record in the `6ixlabs`
repository (quoted by the registry), not re-measured and not in this tree.

## Did not work, and why

Nothing is on record as having failed for this model on this fork: it has not been brought up on
it.

## Templates

All upstream's, in [`deploy/`](../../deploy/README.md):

| Use | Template |
|---|---|
| One box, NVFP4 | [`cluster_qwen-3.8-flash-next_nvfp4_w1.example.json`](../../deploy/cluster_qwen-3.8-flash-next_nvfp4_w1.example.json) |
| Two boxes, NVFP4 | [`cluster_qwen-3.8-flash-next_nvfp4_w2.example.json`](../../deploy/cluster_qwen-3.8-flash-next_nvfp4_w2.example.json) |
| Two boxes, NVFP4, YaRN 512K | [`cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json`](../../deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json) |
| One or two boxes, RadixArk NVFP4 | [`…radixark_w1`](../../deploy/cluster_qwen-3.8-flash-next_nvfp4-radixark_w1.example.json), [`…radixark_w2`](../../deploy/cluster_qwen-3.8-flash-next_nvfp4-radixark_w2.example.json) |
| One box, AutoRound int4 | [`cluster_qwen-3.8-flash-next_autoround-int4_w1.example.json`](../../deploy/cluster_qwen-3.8-flash-next_autoround-int4_w1.example.json) |
| Two boxes, FP8 | [`cluster_qwen-3.8-flash-next_fp8_w2.example.json`](../../deploy/cluster_qwen-3.8-flash-next_fp8_w2.example.json) |
| Four Sparks, FP8 (the fleet has two) | [`cluster_qwen-3.8-flash-next_fp8_w4.example.json`](../../deploy/cluster_qwen-3.8-flash-next_fp8_w4.example.json) |

## Records

- Shared, in [`models/shared/`](../shared/):
  [loadout switching](../shared/dgxone_dgxtwo_loadout_switch_2026-10-04.md).
- Upstream's own documentation: [docs/benchmarks.md](../../docs/benchmarks.md),
  [docs/qwen38_single_spark.md](../../docs/qwen38_single_spark.md).
