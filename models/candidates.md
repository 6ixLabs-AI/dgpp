# Candidates: models that have not run here

Which models could benefit from this engine's approach with little work, and which need new
kernels or other engine work first. Every model in the registry that has never been loaded on a
GPU on this fleet with this engine is on this page. Models that have run are in their own
folders ([models/README.md](README.md)).

**The lesson the records support.** On this engine the speed advantage came from two things. The
first is the checkpoint's draft head: the 80B and the 35B have one, and with it the 35B is ahead
of Atlas on the same file at every level and the 80B is ahead of the fleet's vLLM record (taken
with the draft head off, because vLLM crashed with it on), while
Coder-Next has no draft head in any checkpoint and ran at 0.60 to 0.66 of vLLM. The second is
two-box serving: about 1.6× one box for the two routed-MoE models, measured. Two cautions from
the same records: a draft head did not make the 122B fast enough to keep, and DeepSeek-V4-Flash,
which has one, is behind vLLM above one stream.

How to read the entries:

- Facts come from the registry ([`sixlabs/registry/models.json`](../sixlabs/registry/models.json)),
  the port notes in `sixlabs/ports/`, the plan documents in `docs/`, `deploy/README.md`, and the
  draft-head table in the
  [Coder-Next summary](qwen3-coder-next/one-box/dgxone_coder_next_2026-10-04.md)
  ("The family: which checkpoints have a draft head").
- **Every speed on this page is someone else's published figure. None is ours.**
- A line marked **Projection** is an expectation, not a measurement.
- **The fleet has two Sparks.** A model that needs four cannot run here.

## 1. Likely to benefit, with little or no new engine work

The family already serves on this engine and the checkpoint has a draft head. What is missing is
a run, a gate, a download, or hardware.

### Runnable on this fleet

| Model | Why it is a candidate | Draft head | Fits | What is missing | Speed on record (not ours) |
|---|---|---|---|---|---|
| **Qwen3.8-27B** (`Qwen/Qwen3.8-27B-FP8`), status `upstream` | Family `qwen3_5`, the family of the working 35B and 122B, in its dense form. Upstream serves it and ships a template per world size | Yes: 22 draft tensors in the FP8 release; the one-box template uses the DFlash2 block drafter instead, the two-box template MTP depth 3 | one Spark; also two | Never run here. The upstream-merge record lists its bring-up as owed: answers first (`group_check`), then ShareGPT. Its prompt-reading settings need their own check: the dense checkpoints compute one FP8 activation scale over a chunk's rows, so a larger chunk changes their numbers | Upstream, one Spark: 17.8 prose and 28.6 code at one stream, 45.4–96.3 at eight. Upstream, two Sparks: 34.0 and 40.5, 136.4–217.4 at eight. Atlas, published: 23.6 with the draft head |
| **Qwen3.5-0.8B** (`Qwen/Qwen3.5-0.8B`), status `compiled` | Family `qwen3_5`, dense form, from an unquantized BF16 checkpoint. The code is on main and builds | Yes: 15 draft tensors | one Spark (about 1.4 GiB of weights) | Never loaded. The BF16 dense-MLP branch has never executed; `dense_weights: "fp8"` is the fallback that puts it on the 27B's kernels. Then the gate. Steps: [gpu-steps](../sixlabs/ports/qwen3.5-0.8b/gpu-steps.md) | None. The fleet's vLLM slot has a load time (131 s) and a footprint (5.0 GiB) on record, no speed; Atlas has a recipe and no published speed |
| **MiMo-V2.6-Flash** (`XiaomiMiMo/MiMo-V2.6-Flash-RL`), status `upstream` | Family `mimo_v2`, served by upstream; text only | Upstream's template runs MTP depth 1, the first of the release's three draft layers (`deploy/README.md`); no tensor count on record | two Sparks (97.5 GiB of 121.6 GiB a rank at 256K context) | Not on DGXone: its text tests could not run there. A download, then a run and the gate | Upstream, two Sparks: 42.3–49.2 at one stream, 70.1–78.5 at four requests |
| **GLM-5.3-Flash** (`HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`), status `upstream` | Family `glm5`, served by upstream, vision served | Upstream's template runs MTP depth 1 (`deploy/README.md`); no tensor count on record | two Sparks, tightly: 110.0 GiB a rank, 1.8 GiB of the node left at boot | A 195.1 GB download: it is not in the fleet's cache, and a GLM download was stopped at 40.6 of 195 GB on 2026-10-04 (Coder-Next bring-up record, section I). By the 110.0 GiB figure it would leave no room for the fleet's other services on either box | Upstream, two Sparks: 31.3–36.0 at one stream, 56.3–61.8 at four requests |

**Projection** for the two models on the `qwen3_5` family: they are the closest to a result,
because the family, the draft-head decode and (for the 27B) the two-box world already exist and
have run here on sibling models. Whether they are *faster than the alternatives* is not known:
upstream's own one-box figures for the 27B (17.8–28.6 tok/s) are well under the 35B's, and the
0.8B is small enough that load time and footprint may matter more than tok/s.

**Projection** for MiMo and GLM-5.3-Flash: these are upstream's models running upstream's
templates, so there is no 6ixLabs engine work in front of them, only a download, a run and a
gate. GLM-5.3-Flash would need both boxes to itself.

### Needs four Sparks: cannot run on this fleet

| Model | Family | Template | Draft head |
|---|---|---|---|
| **GLM-5.3 (full)** (`HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64`), status `upstream` | `glm_moe_dsa`, served | four Sparks only | upstream's template runs MTP depth 1 |
| **GLM-4.7** (`nvidia/GLM-4.7-NVFP4`), status `upstream` | `glm4_moe`, served | four Sparks only | upstream's template runs MTP depth 1 |
| **DeepSeek-V4.1-Flash** (`deepseek-ai/DeepSeek-V4.1-Flash`), status `upstream` | `deepseek_v41`, served | four Sparks only | upstream's template runs the DSpark draft at depth 4 |

Nothing is missing in the engine for these. What is missing is two more Sparks. The registry
carries no speed for them; upstream's figures are in [docs/benchmarks.md](../docs/benchmarks.md).
The four-Spark worlds of Qwen3.8-27B, Qwen3.8-Flash-Next, GLM-5.3-Flash, MiMo and
DeepSeek-V4-Flash are out of reach for the same reason.

### Named in the draft-head table, not in the registry

The Coder-Next summary's table lists these as having a draft head. None has a registry entry, a
template or a bind check on record, so nothing says they bind on this engine.

| Model | Draft head | On record |
|---|---|---|
| Qwen3.5-35B-A3B (Sehyo NVFP4) | 785 tensors | "not set up" here; Atlas publishes about 133 tok/s. Removed from DGXone's cache on 2026-10-04 |
| Qwen3.5-2B / 4B / 9B | 15 tensors each | nothing else |

## 2. Needs new kernels or other engine work first

The `groundwork` models. In the registry's words: "Config, tensor table and references exist;
there is no kernel, loader or assembly. Cannot be served." Ordered roughly from the least missing to
the most.

| Model | What is missing, as the notes state it | Draft head | Fits | Speed on record (not ours) |
|---|---|---|---|---|
| **Qwen3-VL-30B-A3B, text path** (`ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4`) | Family `qwen3`, not served. **No new kernel**: it reuses GLM-4.7's attention layer and K/V pool, the shared routed-MoE layer and the `qwen3_5` loader's NVFP4 dequant. The loader, the model walk and the forward check compile and link behind `-DDGPP_BUILD_QWEN3_PLAIN_DRAFT=ON` and have never run on a GPU. Vision: the tower is bound and not served. Steps: [gpu-steps](../sixlabs/ports/qwen3-vl-30b-a3b/gpu-steps.md) | **None** | one Spark (18.06 GiB resident) | Atlas, published: 97 |
| **Qwen3-235B-A22B** (`nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4`) | The same family and draft code as the 30B, which has to be verified first. Then two ranks: the family refuses `world_size` above 1 until the [sharding plan](../sixlabs/ports/qwen3-235b-a22b/sharding-plan.md) has been carried out. A 139.2 GB download (removed from DGXone's cache on 2026-10-04) | **None** | two Sparks (133.86 GiB resident; 67.56 GiB a rank by the plan) | none |
| **Mistral-Small-4 119B** (`mistralai/Mistral-Small-4-119B-2603-NVFP4`) | Family `mistral4`, not served. Config, binding, tokenizer and tool calls are on main. Missing, from [the plan](../docs/mistral_small4_plan.md): the resident loader; the attention layer object and its latent-attention kernel instantiation; the model assembly, latent pool wiring, decode graph, prefix snapshot and memory plan; the serve family; settling which YaRN scale is right on real weights; and any run of the draft kernel (it compiles, its CUDA test does not yet). Vision: a Pixtral tower is in the checkpoint; text only. Not on either box (65.9 GiB) | **None** (`deploy/README.md`: "the checkpoint has no draft layer") | one Spark (65.14 GiB of text weights; about 75 GiB planned with a 262,144-token cache) | none |
| **MiniMax-M2.7** (`lukealonso/MiniMax-M2.7-NVFP4`) | Family `minimax`, not served. Config, binding, tokenizer and tool calls are on main. Missing, from [the plan](../docs/minimax_m27_plan.md): the resident loader with its two-rank slices; the model assembly, K/V pool wiring, decode graph, prefix snapshot and memory plan; the serve family; and any run of the draft kernels. One new operator: a per-layer q/k norm, which is also what tensor parallelism has to work around. Only lukealonso's container is accepted; NVIDIA's release is refused (its KV cache scheme is not implemented). A 134.4 GB download, needed on both ranks or a shared mount | **None**: the config declares one, no checkpoint ships it | two Sparks (125.17 GiB of weights against a 121.6 GiB node; about 64 GiB a rank by the plan) | none |
| **Gemma-4 31B** (`nvidia/Gemma-4-31B-IT-NVFP4`) | Family `gemma4`, not served. Config, binding, the SentencePiece tokenizer path, chat template and host reference are on main. The GPU side is one draft forward behind `-DDGPP_BUILD_GEMMA_DRAFT=ON` that **does not compile under nvcc yet**; the integration record calls the fix structural (the checkpoint loading has to move out of the `.cu` file). There is no serving engine for the family. Vision: the tower is skipped. Not in any box's cache as far as the port notes know (32.6 GB). Steps: [gpu-steps](../sixlabs/ports/gemma-4-31b/gpu-steps.md) | **None** (`deploy/README.md`: "the checkpoint has no draft head") | one Spark (29.3 GiB of weights as shipped, by arithmetic) | none |
| **Gemma-4 26B-A4B** (`bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16`) | The same family and the same state as the 31B. A community quantization. `RedHatAI/gemma-4-26B-A4B-it-NVFP4` is refused by the port (only modelopt is implemented). Steps: [gpu-steps](../sixlabs/ports/gemma-4-26b-a4b/gpu-steps.md) | **None** in either NVFP4 release | one Spark (14.2 GiB of weights as shipped, by arithmetic) | Atlas, published: 67 |
| **Nemotron-3 Nano 30B-A3B** (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`) | Family `nemotron_h`, recognised and refused by name. **No kernel, no loader, no assembly, no serve wiring, no template.** From [the plan](../docs/nemotron3_plan.md), in dependency order: a loader (per-tensor FP8 is new); Mamba2 kernels (a convolution with a bias, a new scan recurrence, a grouped gated norm); attention without positions; an MoE chain for two-matrix experts with top-22 routing (the shared MoE config caps `top_k` at 16); per-slot Mamba state and snapshots; and the text side (tokenizer and template). The numpy reference has never run on the real weights. Steps: [gpu-steps](../sixlabs/ports/nemotron-3/gpu-steps.md) | **None** | one Spark (18.0 GiB) | Atlas, published: 88 |
| **Nemotron-3 Super 120B-A12B** (`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`) | The same groundwork and the same list as Nano, plus the draft block | **Yes**: 1,040 draft tensors (5.48 GiB, all BF16) | one Spark (74.8 GiB) | Atlas, published: 24 |

Also in `sixlabs/ports/`, not in the registry: **Qwen3-Reranker-0.6B and Qwen3-Embedding-0.6B**
([notes](../sixlabs/ports/qwen3-retrieval/gpu-steps.md)). Config parse and binding table only;
there is no GPU path for the dense Qwen3 dialect and no rerank or embedding endpoint. They are
not generation models, so the draft-head question does not arise.

### Would the approach pay after the work? Projections

These are expectations drawn from the records above, not measurements.

- **No draft head: Qwen3-VL-30B, Qwen3-235B, Mistral-Small-4, MiniMax-M2.7, both Gemma-4 models,
  Nemotron-3 Nano.** The draft-head gain cannot apply. The one measured case of this engine on a
  plain pass is Coder-Next, which was slower than vLLM; it is one model, one architecture and
  two different quantizations, so it is a warning and not a prediction. What the engine would
  still bring is the resident image (restarts in seconds) and, for the two models that need two
  Sparks, a way to serve them at all on this fleet. For those two the second box is the price of
  fitting, not a speed-up over a one-box run that cannot exist.
- **Draft head, and the most work: Nemotron-3 Super.** It is the one groundwork model with a
  draft head, so it is the one where the approach has the mechanism to pay. It also needs
  everything on Nano's list first, and the plan puts the draft block after a convention that is
  still to be measured. Atlas's published 24 tok/s is the only speed on record.
- **Least work: Qwen3-VL-30B.** No new kernel and a draft that already compiles and links. It is
  the cheapest to find out about, and by the first point the least likely to be fast.

## Not on this page

- Checkpoints of models that have run, which failed the gate or were refused: the FP8 release of
  the 35B, and the other 122B releases. They are in the "Did not work, and why" sections of
  [qwen3.6-35b-a3b](qwen3.6-35b-a3b/README.md) and
  [qwen3.5-122b-a10b](qwen3.5-122b-a10b/README.md). The 122B was dropped as too slow and is not a
  candidate.
- Qwen3.8-Flash-Next, which runs on this fleet on upstream DGPP and has
  [its own folder](qwen3.8-flash-next/README.md).
