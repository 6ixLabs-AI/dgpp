# Qwen3.8-27B campaign notes

## The public references and the harness

The two spark-arena cards for this checkpoint are the figures the family is
measured against: four nodes, vLLM TP=4 with MTP at three speculative tokens
and an FP8 KV cache (58.48 tok/s at C1); two nodes, SGLang TP=2 with the
RadixArk DSpark block drafter (block 7, 1.86B parameters, five
full-attention layers and a rank-256 Markov head — the same shape as the
DFlash2 drafter this family ships on one node) and an FP8 KV cache
(46.55 tok/s at C1). Both run their harness at the checkpoint's own
sampling (`generation_config.json`: `do_sample`, temperature 1.0, top-k 20,
top-p 0.95), so the like-for-like figure here is the harness's `tg128` at
C1 under that sampling, not the page's greedy per-class engine rate. The
`llama-benchy` legs (`raw/llama-benchy/`: `--pp 2048 --tg 128 --runs 40
--concurrency 1 2 4` against each world, `uvx llama-benchy`) are that
measurement; the page's rows keep the repository's workload.

## Sampled acceptance equals greedy acceptance on this model

The two-node MTP depth-2 deployment commits 2.15–2.91 tokens a pass greedy
and 2.19–2.94 sampled (prose / code / json / math / chat, the performance
stage's `greedy.json` and `sampled.json`): the sampled verify's acceptance
under the checkpoint's sampling is the greedy one within noise, so the
harness's sampled rate reads the engine's greedy rate for the MTP worlds.

## The drafter and sampled requests

Before 2026-10-04 the DFlash2 drafter verified greedy slots only; a sampled
slot ran the plain step (8.4 tok/s at C1 on one node — the single-node
`sampled.json` of this campaign's performance stage, measured on the
campaign binary). The sampled verify of the block (each draft stands with
its exact probability, the residual sample ends the step) landed after the
campaign binary was fixed; its numbers are a follow-up record, not this
campaign's.

## The plain-mode incident

The single-node template's first "plain" mode run (`dflash_model` cleared,
the template's `decode_graph: false` kept) landed on the eager engine in
streaming residency — 1,540 ms a step, 0.65 tok/s — and was stopped after
one class (`raw/qwen27b-fp8-w1/modes/plain` of that run was discarded). The
plain world of a drafter template is the resident plain world on the
decode graph (`decode_graph: true`, no MTP), which is what the redo
measures and what the `--no-dflash` knob now selects. The campaign binary
predates that knob change; the redo uses the config override, not the knob.

## One recipe per world size

The single-node template is the drafter's: on the campaign's greedy
corpus it leads MTP depth 2 on every class (the modes table carries the
MTP depth-2 world as this template's `mtp2` mode). The two-node and
four-node templates are MTP at depth 3; the drafter runs on those worlds
since 2026-10-04 ([#92](https://github.com/HawkBearPig/dgpp/issues/92),
`raw/drafter-tp/`) and is their mode: on the arena harness the MTP
template keeps a shorter pass (two nodes 32.2 vs 31.3 tok/s, four nodes
55.0 vs 50.1), while the drafter leads code / JSON / math by 27–55 % on four
nodes and ties prose and chat (the `dflash2` mode in the modes table).

## Telemetry gaps

Node 2's telemetry collector could not reach its management address from
the head node during this campaign (`node2-telemetry.jsonl`: the ssh
probe's "No route to host"), so the per-node utilization record covers
nodes 0, 1 and 3 for the two- and four-node launches. The measurements
themselves run over the fabric and are complete.

## The isolation check across worlds

The performance stage's isolation check (one prompt alone against the same
prompt beside seven others, 256 greedy tokens) reads DIFFERENT on every
world of this family — late on one Spark (character 966) and early on four
(character 26). It is the row-count dispatch class, not a batching fault:
a lone MTP depth-3 slot verifies 4 rows on the `scale_gemv` family while
eight slots verify 32 rows on `mma_gemv`'s two-tile form, and the two
kernels round their fp8 products differently, so near ties move (the same
class as the C2 transcripts since the QSA workspace change and the
drafter-vs-plain comparisons above). Within one dispatch class the
transcripts are bitwise: the drafter's 8-row verify equals the MTP depth-5
world's 4/4 at worlds 1, 2 and 4.

## The drafter-at-TP legs (`raw/drafter-tp/`)

`w1-eager`, `w1-graph`, `w2-dflash2`, `w4-dflash2` (the first build:
replicated drafter, serial block attention), `w2-plain` / `w4-plain` (the
1-row references: differ by the row-count dispatch class), `w2-mtp5` /
`w4-mtp5` (the same-dispatch references: identical 4/4), `w2-mtp3` /
`w4-mtp3` (the templates under the arena harness), `w*-dflash2-bf12` (the
bf12 A/B, +4.5 %), `w*-dflash2-v2` (sharded drafter + split attention), the
nsys node traces `w4-dflash2-nsys`, `w4-mtp3-nsys`, `w4-dflash2-v2-nsys`
(per-kernel GPU time: the scratch `nsys_kernels.py` sums them per step).
Every leg boots a site recipe with `build-ci/dgpp-serve` (not the campaign's
release binary) and runs `serve_greedy_transcript.py`, `timed_load.py` C1
greedy and sampled, and llama-benchy at the arena's settings.
