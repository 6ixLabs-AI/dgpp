# Qwen3.8-Flash-Next AutoRound int4 hybrid — the record (2026-09-28, in progress)

Checkpoint: `Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN`
(revision 19f9710c); the n-gram table from `Qwen/Qwen3.8-Flash-Next-FP8`
(236dfdf2). Plan: docs/qwen38_autoround_int4_plan.md. Branch
`qwen-autoround-int4`.

## S1 — the packed core at group 128 / f16 scales

`packq_gemv_test`, `packq_gemm_test`, `glm_moe_test` (the packq cases):
every group-128 / f16 case matches the FP64 oracle with zero mismatches at
K = 128 … 2560 (both widths, m = 1 … 65), the decode slot path is bitwise
the host chain, the sliced ranks fold to the unsliced oracle, and the
format-0 cases are unchanged.

`moe_slot_bench` at the Qwen geometry (E 512, H 2560, I 640, top-10, no
shared expert, softmax router), rank 0 of the fabric beside an idle
production server (GPU at 0 %), 200 iterations after 20 warm-ups, min of
the per-call times:

| rows | NVFP4 experts | packed int4 g128/f16 | change |
|---|---|---|---|
| 1 | 132.7 us (208 GB/s at 28 MB) | 117.4 us (216 GB/s at 25 MB) | −11.5 % |
| 2 | 264.6 us (209 GB/s at 55 MB) | 247.2 us (205 GB/s at 51 MB) | −6.6 % |
| 4 | 484.8 us (228 GB/s at 111 MB) | 441.7 us (230 GB/s at 101 MB) | −8.9 % |

Per 48-layer pass: the routed-expert class 6.4 → 5.6 ms at one row, 12.7
→ 11.9 at two, 23.3 → 21.2 at four. The bytes shrink by 7.6 % (4.156 vs
4.5 bits per weight); the packed core runs the same or a higher fraction
of line rate than the fp4 core at every row count, so the S1 gate (at
least the fp4 rate) holds with margin.

## S2 — loader and fixture gates

- `qwen_loader_repacks_the_autoround_hybrid_exactly`: every expert matrix's
  words are the fixture's GPTQ words transposed, its scales the f16 scales
  transposed, `packq_decode` equals GPTQ's own dequant; the shipped fp8
  classes land as codes + F32 scales byte for byte; the head is the int8
  triple transposed; a world whose down-projection K slice is not a whole
  group is refused by name, so is the bf16 dense mode, so is a zero word
  off the symmetric constant. Green.
- `qwen_config_parses_the_autoround_hybrid`, `qwen_binding_table_has_the_autoround_hybrid_shape`: green.
- The fixture-level gates on the hybrid's tiny twin (experts and shared
  expert of 256, expert scales at 0.0034 so the tiny model's routing has no
  near-tie against the pure-Python reference):
  - `qwen_gptq_forward_test` (parity against `tools/qwen_reference_dump.py`,
    which dequantizes the GPTQ triples exactly): every layer within 5 ulps,
    l2 ≤ 0.0033, 0 of 576 routing slots differ, draft logits top-1 exact. Green.
  - `qwen_gptq_smoke`, `qwen_gptq_prefill_head` (the packed head's last row
    bitwise the full product through 257 rows, both kernel families): green.
  - `qwen_engine_gptq` (loopback world 2: the MTP graph == plain decode at
    depths 1 and 2, the mmap'ed table == resident, sixteen wide slots, graph
    == eager): green.
  - `qwen_gptq_decode_test` (prefill == forward bitwise, the chunked and
    group prefills, the incremental decode against the re-forward, two
    interleaved slots, the resized prefill): green once the fixture's expert
    scale base moved to 0.00265 — the 0.0034 base had a single decode step
    hit a routing near-tie between the decode chain and the re-forward (one
    row at 3.6 % relative l2, top-1 equal); the factor sweep over both gates
    found 0.00265 flip-free with flip-free neighbours.
- The regression run on the full build (`ctest -R "packq|glm_moe|glm_dsa|qwen|unit_tests|cluster_config"`):
  71 of 71 passed — the packed cores and tile at both scale formats, the
  shared MoE layer, full GLM-5.3's loader on its own packed format, every
  Qwen suite on the FP8 and NVFP4 fixtures (the engines, the TP loopback,
  the fp8 head variants, bf12), the hybrid's gates, the unit tests.

## S5 — the real checkpoint (streaming, beside the production server)

`qwen_load_check --checkpoint-dir <hybrid snapshot 19f9710c> --world 1
--streaming --mtp --layers 3 --ngram-table mmap --ngram-table-dir <FP8
snapshot 236dfdf2> --dense-weights fp8 --image-dir off` on rank 0, the
production GLM world resident (103 GiB) and idle:

- binding: every one of the 227,526 tensors matched the expectation table
  (the draft layer's q/k/v/o and shared expert are BF16 in this checkpoint,
  the 48 backbone layers' are fp8 — found by this check, then encoded in the
  binding); the n-gram table admitted from the FP8 snapshot's shards (48.67
  GiB mapped, per-tensor scale 0.000199), nothing else from them;
- the zeros of every loaded expert matrix and of the head verified as the
  symmetric constant; three layers + the draft layer + globals in 10.8 s
  (3.94 GiB resident for three layers → 1.31 GiB per layer, globals 1.83
  GiB: about 66.6 GiB for the whole model, the plan's 67);
- the source-byte plan reconciled with the bytes read (5.81 GiB);
- the mmap'ed table's gather: a 2-row decode pass p50 360 us, p90 425, p99
  655 (the NVFP4 template's 359 / 430 / 858), a 2048-row chunk 114–156 ms.

**The shipped fp8 dense classes are our recipe, bitwise.** Eight backbone
matrices of the hybrid (layer 0's GDN qkv / z / out, layer 3's QSA q / k /
o, layer 0's shared gate / down: 108 M codes, 6,680 block scales) compared
against `loaders/fp8_quant.hpp`'s recipe applied on the CPU to the same
tensors of `nvidia/Qwen3.8-Flash-Next-NVFP4` (BF16 originals): every code
and every F32 scale equal. The hybrid's side layers are therefore what
`engine.dense_weights = fp8` already computes at load from the NVFP4
release, and every evaluation of that template's dense stack carries over.
(`dense_fp8_compare.py` beside this file; reproduce with the two
snapshots and the tensor names.)

## S5/S6 — the hybrid served on one Spark (2026-09-29, 00:05–00:30)

`deploy/cluster_qwen-3.8-flash-next_autoround-int4_w1.json` (the example
verbatim; `model_alias: qwen`), the release binary of this branch,
production down, rank 0 alone. Cold boot 66 s including the GPTQ
transpose and the image capture; 26 s from the image after that. Memory
plan 89.32 GiB (84.47 device + 4.85 pinned; weights 65.31 resident) + 4
GiB headroom against 115 GiB free. Graph variants 1,572 kernel nodes + 1
host node (the mmap gather), the 2 / 3 / 4-slot batch families. The API
check passes (usage counts reasoning tokens inside completion_tokens, the
prefix cache hits).

**Correctness.** Greedy transcripts (chat / code / math / json, 256
tokens, thinking on): depth 2 and depth 3 identical to depth 1, 4 of 4
each. Cold prefill (GSM8K text, `serve_prefill_probe.py`): 1.16 / 0.75 /
0.67 ms per token at 512 / 2K / 8K (the NVFP4 template: 1.15 / 1.14 /
1.11).

**Single stream, greedy, thinking on** (`serve_mtp_classes.py`, 300
tokens; ms per pass and tokens per pass from the retire lines):

| class | depth 1 | depth 2 | depth 3 |
|---|---|---|---|
| chat | 40 ms/pass, 1.57 tok/pass, 27.5 ms/tok | 49, 1.83, 28.6 | 57, 1.90, 31.9 |
| code | 40, 1.76, 24.7 | 49, 2.20, 24.1 | 57, 2.51, 24.5 |
| prose | 40, 1.85, 23.2 | 48, 2.37, 22.0 | 56, 2.69, 22.4 |
| json | 40, 1.85, 23.4 | 48, 2.53, 20.8 | 56, 2.96, 20.9 |
| math | 40, 1.86, 23.4 | 48, 2.60, 20.4 | 56, 3.15, 19.8 |

The sampled pace at depth 1 (`serve_bench.py`, 200 tokens): 26.1 ms per
token, TTFT 278 ms.

**The author's script, verbatim** (`bench_qwen35.sh` pointed at :18080,
`model: qwen`, temperature 0, thinking on, completion tokens over the whole
request's wall time; two runs each, the mean):

| cell | to beat | depth 1 | depth 2 | depth 3 |
|---|---|---|---|---|
| Q&A (65 → 62 tok) | 49.1 | 38.1 | 41.0 | 43.2 |
| Code (72 → 512 tok, our cap) | 59.1 | 41.8 | 46.9 | 47.7 |
| JSON (90 → 828 tok) | 62.6 | 46.4 | 56.5 | 61.9 |
| Math (71 → 64 tok, cap) | 55.5 | 39.2 | 45.5 | 51.2 |
| LongCode (79 → 2048 tok, cap) | 60.3 | 41.5 | 45.5 | 45.2 |

Read against the plan's floor table (§6.3): the depth-3 pass sits at 56–57
ms, the extrapolation, against the plan's byte floor (stated as 39 ms then; corrected to 48 ms by the profile below) — the chain-row latency
(three serial draft steps at ~5 ms each over their bytes), the dense-stack
kernels under line rate and the launch seam are the 17 ms; and the
chained acceptance on code (2.51 of a possible 4 at depth 3, p1 70 %) is
below the author's 2.65 of 3 accepted — the two defects §7 lists first.

**Where the tokens go (per-position acceptance from the retire lines, depth 3).**
The author's LongCode prompt with thinking on: p1 77 %, p2 58 %, p3 43 %,
2.77 tokens per pass at 60–61 ms — the completion is reasoning text (on
the Code prompt 456 of our 512 tokens are reasoning; the author's run
finished at 276 tokens, a different greedy trajectory of the same prompt).
The same prompt with thinking off (`enable_thinking: false`): p1 95 %, p2
89 %, p3 81 %, 3.64 tokens per pass at 60 ms — the author's 2.65 of 3
exactly. So in code text the draft accepts as theirs does; the thinking-on
cells measure how long each stack's greedy trajectory reasons.

**The author's script with thinking off** (their "5–10 % faster than the
table" regime; two runs, the mean):

| cell | hybrid, depth 3 | NVFP4 template, depth 3 (same engine) |
|---|---|---|
| Q&A (25 → 182 tok) | 55.4 | 53.3 |
| Code (32 → 188 tok) | 59.1 | 59.6 |
| JSON (50 → 849 tok) | 63.1 | 64.3 |
| Math (31 → 9 tok) | 22.2 (TTFT-bound) | 23.6 |
| LongCode (39 → 2048 tok) | 59.6 | 61.6 |

**The NVFP4 template on the same prompts, thinking on** (the A/B for the
draft weights): Q&A 44.3, Code 50.6, JSON 61.7, Math 49.4, LongCode 50.8;
its classes at depth 3: 55–57 ms per pass, 1.87–3.22 tokens per pass
(the hybrid: 56–57, 1.90–3.15). The int4 draft experts cost no acceptance
and the int4 backbone experts buy no pass time against NVFP4 here: the
1.5 ms they save is what the hybrid's BF16 hyper-connections, PLE and
indexer projections (fp8 at load on the NVFP4 template) give back.

**What is left, against the author's thinking-off numbers:** 6–10 %,
made of their draft-vocabulary subset (+3–5 %, not used here by decision)
and the pass — 56 ms at short context, 60 ms at 2K, against the 48 ms
byte floor of the plan's §6.3 (corrected 2026-09-29, see the profile below). The shipped depth is 3 (every cell, both
regimes).

**Task-level quality (serve_eval.py, depth 3, greedy, thinking off,
max_tokens 2048, seed 20260908; `serve/eval_depth3.json/`):**

| task | hybrid int4, depth 3 | FP8 template (2026-09-09 baseline) |
|---|---|---|
| HumanEval (164) | 159 = 97.0 % | 94.5 % |
| GSM8K (300) | 292 = 97.3 % (1 truncated) | 97.7 % |
| schema extraction (100) | 100 % | 100 % |

Every cell within the run-to-run band of the FP8 baseline; the accuracy
goal holds at the shipped depth.

## S6 — the depth-3 pass under nsys and the plain-world gate (2026-09-29, 01:05–01:14)

`scripts/fabric_qwen_profile.sh deploy/cluster_qwen-3.8-flash-next_autoround-int4_w1.json
profile_depth3 --bin build-release/dgpp-serve --knobs "--mtp-depth 3"`: the
world under `nsys -t cuda --cuda-graph-trace=node`, one warm request, one
400-token request (thinking on), 196 steps in the window. Outputs in
`profile_depth3/` (`breakdown.txt`, `r0.nsys-rep`, `r0.sqlite`, the world's
logs). Wall 63.0 ms/step under the tracer (the unprofiled pass is 56–60);
kernel time 76.8 ms/step of which 15.8 is the L2 prefetch stream; 1,687
kernels per step.

**One pass in time order** (step 100, 62.3 ms; the graph-node stream ids
are synthetic, so "busy" is the union of every non-prefetch kernel):

| phase | span ms | busy | idle | kernels | what |
|---|---|---|---|---|---|
| commit | 1.05 | 1.05 | 0 | 1 | the retract copy of the 36 GDN states (113 MB) on steps where a draft row failed |
| draft 1 | 4.02 | 4.0 | 0.03 | 49 + head | layer 1.27 (q/k/v/o BF16 at 4 rows 0.60, experts 0.22, fc cutlass tile 0.09, GR 0.08) + head 2.75 |
| draft 2 | 4.05 | 4.0 | 0.02 | 46 + head | layer 1.28 (q/k/v/o at 1 row 0.49, GR pair 0.20, pick 0.12, experts 0.15) + head 2.77 |
| draft 3 | 4.02 | 4.0 | 0.02 | 45 + head | same |
| verify | 47.7 | 46.1 | 1.57 | 1,311 + head | 48 layers 44.9 + head 2.79 |
| sample + verdict | 0.43 | 0.42 | 0 | 10 | partials 0.19, local 0.11, verdict 0.06 |
| total | 62.3 | 60.6 | 1.7 | 1,467 | |

**Per kernel in the verify** (48 layers, 4 rows; floor = bytes at 233.6 GB/s):

| kernel | per launch us | floor us | launches | ms/pass | gap ms |
|---|---|---|---|---|---|
| `scale_gemv_multi<4>` (GDN qkv+z 41.9 MB / QSA q+k+v 34.1 MB fp8) | 178.8 | 171 | 48 | 8.6 | 0.4 |
| `moe_slot_gate_up_swiglu_packq<4,2560,1>` + `moe_slot_down_packq<4,640,1>` | 224.6 + 101.6 | at rate (~30 distinct experts x 2.54 MB) | 48 | 15.7 | 0 |
| `scale_gemv<4>` (out/o proj 15.7 MB fp8, N 2560 over K 6144) | 93.3 | 67 | 48 | 4.5 | 1.25 |
| `gr_down_inject<4>` (6.5 MB BF16) | 36.2 | 28 | 96 | 3.5 | 0.8 |
| `bf16_gemv<4>` (GR up 6.5 MB and the small BF16 classes) | 32.4 | 28 | 122 | 3.96 | 0.4 |
| `kda_recurrent<128,1,0,1>` (3.15 MB fp32 state read + written) | 48.2 | 27 | 36 | 1.73 | 0.76 |
| `shared_gate_up_swiglu<4,1>` + `shared_down_accum_round<4,1>` (3.3 + 1.6 MB) | 21.5 + 12.0 | 14 + 7 | 48 | 1.6 | 0.6 |
| `moe_router_dots<1>` (2.6 MB BF16) | 18.9 | 11 | 51 | 0.96 | 0.4 |
| ~600 launches ≤ 6 us (group_rmsnorm 103, mix_finish 107, combine_apply 103, gate_act 100, norm_rope 45, kda_conv 36, gated_rmsnorm 36, slot order/accum 100, index/kv/attn ~50) | 1.3–5.4 | ~0 | ~600 | 1.7 | 1.4 |
| idle between nodes | | 0 | | 1.57 | 1.57 |
| dense stack + gaps, total | | 18.7 + 0 | | 26.4 + 1.6 | 7.7 + 1.6 |

The head: `packq_gemv<8,2560,1|4,1,float>` 2,775–2,801 us per launch against
2,720 (0.636 GB int8 at line rate), four launches per pass (verify at 4
rows, draft 1 at 4 rows, drafts 2 and 3 at one row), 11.2 ms of the pass at
97 % of line rate. The experts: at line rate. So the pass is 48.0 ms of
bytes at line rate plus 10–12 ms of listed defects: the dense-stack kernels
under rate (7.7), the node gaps (1.6), the draft steps' kernels (1.7), the
commit copy (0.9), the sample (0.4). The plan's §6.3 now carries the
corrected table (its first total, 38.5 ms, was an addition error over rows
that summed to 46.5; the draft layer's BF16 q/k/v/o, 100 MB per step, were
also missing).

**The plain-world gate** (`--knobs=--no-mtp`, same binary and recipe):
`serve_greedy_transcript.py --compare serve/transcripts_depth3.json` —
identical transcripts 4 of 4 (chat / code / math / json, 256 tokens,
thinking on); so depth 3 == depth 2 == depth 1 == plain, the MTP == T=1
gate holds on the real checkpoint. T=1: 32.6–33.0 ms/step (stats lines;
`serve/transcripts_plain.log`) against a 26.6 ms floor (4.36 GB dense +
1.22 GB experts at one row + 0.64 GB head): the same dense-stack defects.
The NVFP4 template's T=1 is 30.3 because its recipe encodes the 1.26 GB of
BF16 hyper-connections to fp8 at load; the hybrid reads them as shipped.

Production (the GLM-5.3-Flash world on four nodes) restored at 01:14 from its
recorded config and binary; answers.

## S6 round 1 — replay, pick, act_up (2026-09-29, 02:46–02:53)

Three changes, one release binary (`serve_round1/`):

1. **The GDN recurrent state's checkpoint-and-replay form** (`KdaReplay`,
   kernels/kda.hpp; plan §7 item 3a): no post-row snapshot stores, no
   commit copy of the state; the next pass replays the accepted rows from
   their saved inputs (bitwise the snapshot it replaces).
2. **The greedy pick, chunked**: sample_prepare leaves each 256-id chunk's
   smallest composite key in `maxes`, the row block reduces 970 keys
   instead of walking 248K logits in one block.
3. **The gate activation folded into the batched rows' up GEMV**
   (`gr_act_up<4>`): one launch and one graph node fewer per GR site.

**Gates.** `serve_greedy_transcript.py --compare serve/transcripts_depth3.json`:
identical transcripts 4 of 4. 58/58 Qwen + pick + KDA tests green
(the prefix-cache hot == cold and the MTP rollback rounds included).

**The pass (nsys, the same 400-token sampled request as the baseline;
`serve_round1/profile/`):** 57.94 ms at step 100 against 62.29 (−4.4 ms,
−7.0 %); 1,576 kernels per step against 1,676; the verify's idle between
nodes 1.20 ms against 1.57.

| kernel | baseline us | round 1 us | note |
|---|---|---|---|
| spec_commit (per pass) | 1,047 | 129 | the 113 MB state copy gone; what remains is the 12 MB copy of the saved rows (row-major since, 3 MB at depth 3: next binary) |
| kda_recurrent<4 rows> | 46.6 | 29.9 | one state read + one write (floor 25 us) — no snapshot stores |
| scale_gemv<4> out/o proj | 93.4 | 78.3 | the snapshot write-back no longer competes (68.5 cold in isolation) |
| gr_act_up<4> (was gate_act 1.0 + bf16_gemv 32.4) | 33.4 | 31.0 | one node instead of two, 96 sites |
| sample_local (this profile's request is SAMPLED: the greedy path is not exercised here) | 116.6 | 116.9 | the bench's greedy requests take the chunked path |

**Classes (300 tokens, greedy, thinking on; wall ms/token):** chat 31.9 →
30.7, code 24.5 → 23.7, prose 22.4 → 21.7, json 20.9 → 20.3, math 19.8 →
19.4 (−2 to −4 %); acceptance unchanged (the transcripts are identical).

**The retire line's ms/pass is biased low on short requests**: it divides
(wall since admission − prefill ms) by the pass count, and the prefill
figure absorbs the first pass(es) — a 4-pass request prints 16 ms/pass, a
17-pass one 42. Read pass times from the stats lines (ms/step over a 10 s
window) or nsys; requests over ~100 passes are within 1 %.

**Sustained line rate confirmed:** `qwen_dense_bench --sustain 30` beside the
idle production world holds 247 GB/s for 30 s at 2.47–2.51 GHz SM clock and
84 W (`sustain_bench.txt`, `sustain_clocks.txt`); the SoC's software power cap
does not lower the delivered DRAM rate. The floor is 248 GB/s, sustained.

**The author's protocol, round 1** (`bench_qwen35_dgpp.sh`, the second run;
`serve_round1/bench_qwen35*.log`):

| cell | author | baseline think on | round 1 think on | baseline think off | round 1 think off |
|---|---|---|---|---|---|
| Q&A | 49.1 | 43.6 | 44.3 | 55.8 | 57.1 |
| Code | 59.1 | 47.8 | 49.2 | 59.4 | 60.7 |
| JSON | 62.6 | 62.1 | **63.0** | 63.5 | 64.8 |
| Math | 55.5 | 50.7 | 51.1 | 21.9 (9 tok, TTFT-bound) | 23.0 |
| LongCode | 60.3 | 45.3 | 46.6 | 59.9 | 60.9 |

+2–3 % per cell; JSON with thinking on now clears the author's number. The
retire lines (thinking on): Code 56 ms/pass at 2.90 tok/pass, JSON 56 at
3.71, LongCode 59 at 2.77 (the last 2K tokens at ~2K context).

**ncu on the serving process does not work** (`ncu_pass/`): the graph-level
run failed at the warm-up ("Failed to profile kda_recurrent_kernel"), ncu
killed the server, and the launcher waited for READY for 46 minutes with
production down. The memory-side byte question (does the prefetch
double-read?) is answered instead by an nsys pass with the prefetch off
against on: the consumers that shorten with it on are the ones its bytes
reach.

## S6 round 2 — the combine in the norm launch, the replay rows row-major, the gather timed (2026-09-29, 03:59–04:07)

`serve_round2/`. Transcripts identical 4 of 4. The author's protocol (run 2),
thinking on: Q&A 44.4, Code 49.5, JSON 63.6, Math 51.3, LongCode 46.7; off:
56.9 / 60.7 / 64.9 / 21.9 / 61.3 — round 1 within noise (the 96 combine
launches removed per pass are worth ~0.2 ms; the pass is DRAM-bound and a
launch that overlapped the prefetch stream's reads cost no DRAM time).
1,493 kernels per step (1,576 in round 1, 1,676 at the baseline).

**The prefetch double-reads.** The same request profiled with the prefetch on
and off (`profile/`, `profile_nopf/`; `pf_compare.py` in the job's scratch):
wall 60.79 against 60.61 ms/step — no difference — while the side stream
reads ~3.9 GB per pass (15.6 ms at line rate) of which the consumers find
96 MB in L2 (their total shortening 387 us/step: gr_down_inject 44.3 → 34.9
us, gr_act_up 36.1 → 30.9, the multi GEMV 183 → 177, the router 21.9 →
18.6, the shared gate/up 24.8 → 21.4). The expert stream between a window
and its consumer evicts the rest. That the wall is unchanged with 3.8 GB of
extra reads means the DRAM has that much slack: with the prefetch off the
pass moves ~11.5 GB in 60.6 ms = 190 GB/s, 77 % of the line rate, and the
consumers themselves leave it idle. (The 2026-09-29 02:15 sweep saw off =
+2 ms on another request; either way the prefetch is not a lever until the
consumers saturate the DRAM, and then its 3.8 GB become the defect.)

**In-step kernels run ~8–10 % below their isolated rate even with the
prefetch off**: the int8 head 2,750–2,800 us in the step against 2,526
(1 row) / 2,617 (4 rows) cold in isolation (256 / 247 GB/s — the packq
GEMV reaches the line rate alone); the multi GEMV 183 against 167; the
experts 236 GB/s. A purely streaming kernel with 31,040 blocks is not
latency-bound, so the in-step penalty is a rate difference: the next
session samples the SM clock during a decode run (the sustained bench held
2.47–2.51 GHz at 84 W; a busier step under the SoC's power cap may not).

**The n-gram host gather** (`DGPP_QWEN_PLE_STAGE_TIMING=1`): mean 886 us,
max 4.3 ms per call (64 rows of the 48 GB mmap) — the layer-2 gap of
~770 us is the callback's own duration, not hidden behind layers 0–1
(1.9 ms) as the fork/join design intends, so the host node also starts
late. Next: the advise/copy split (instrumented), a persistent gather
thread woken by an event recorded after the hash kernel and a device
spin-wait before the gather kernel (the engine's stage handshake), and
parallel faults.

**The in-step rate penalty is TLB reach (2026-09-29 04:15).** The isolated
bench over an 8 GiB ring instead of 512 MiB (the model's working set is
65 GB per pass): the multi GEMV 167 → 185.6 us (in the step 177–183), the
out_proj GEMV 68.5 → 87.6 (in the step 78–93), the head 2,617 → 2,689 at
four rows (in the step 2,750–2,800). The clock is not it: a 1,024-token
greedy decode ran at 2,515–2,541 MHz and 60 W (the saturating stream
draws 84 W at the same clock) — the DRAM is under-used, not throttled.
Every SM walks every 2 MB page of a matrix it streams (blocks interleave
across SMs), so a kernel pays ~pages x walk latency per SM: the head's
318 pages cost ~70 us, the 21-page multi ~16, and the out_proj's 8 pages
~20 (fewer bytes in flight to hide them). Two fixes: larger pages for the
resident images (the VMM API's recommended granularity, queried below), or
persistent blocks that stream contiguous row ranges so each SM touches
1/48 of the pages. Worth ~2.8 ms per pass across the dense stack and heads.

**The gather (this session):** advise 249 us (64 madvise calls) + copy 191
us (the faults) = 440 us mean once the page cache is warm (886 cold);
the layer-2 wait is longer than the call because the host node starts
late. `process_madvise` (one syscall for 64 ranges), copies across a small
pool and an event-woken gather thread with a device spin-wait are the fix.

**Huge pages (`tlb_pages/`, 04:16):** with production down and 109 GB
free, the kernel reserved zero 1 GB pages and six 32 MB ones (physical
fragmentation: 1 GB pages need boot-time reservation); raw host memory
through ATS streams at 65–80 % of the device rate (host2m: the multi 261
us, the head 3,455), so the weights stay in device memory. The 8 GiB
device ring reproduces the step exactly: multi 191 us, out_proj 78.5,
gr_down 36.6, head 2,790 — the in-step numbers to the microsecond. The
lever left is the kernels' page footprint per SM: persistent blocks that
stream contiguous row ranges (one staging of the activations per block
instead of per eight rows, which also removes the out_proj's 48 KB-per-48
KB staging overhead).

**Row spans per block do not recover the TLB penalty** (04:30; the fp8 core
with `DGPP_GEMV_BLOCKS` 96 / 192 / 384 / 768 blocks streaming contiguous
row ranges, 8 GiB ring): out_proj 74–80 us against 76.9 with one group per
block, the multi 183–193 against 188.9, the QSA q/k/v 147–152 against 146
— within ±4 %. The misses are served by a walker every SM shares; which SM
issues them does not matter. With 1 GB pages unreservable at runtime and
host memory through ATS at 65–80 %, the in-step streaming rate for a 65 GB
working set is ~225–230 GB/s on this box, and the pass floor at that rate
is ~50 ms (11.5 GB). The form stays behind the knob (default 0: the
original grid).

## S6 round 3 — the n-gram gather off the host node (2026-09-29, 04:26–04:35)

`serve_round3/`. The PLE layer's host node (cudaLaunchHostFunc on a side
stream) is replaced by a publish kernel after the hash (`qwen_ple_publish_stage`:
the rows and a sequence to pinned words), a gather thread polling that
word, and a device spin-wait before the gather kernel (`glm_stage_wait`);
the mmap gather issues its page advice in one `process_madvise` call.
Transcripts identical 4 of 4; 58/58 tests. In the pass: `stage_wait_kernel`
1.7 us, then `gather_staged` — the 772 us layer-2 gap is gone; the pass
at step 100 59.7 ms/step (60.8 in round 2; run-to-run noise ±1 ms).
The gather itself: advise 433 us + copy 297 us, all of it now behind the
1.9 ms of layers 0 and 1. The author's cells: 44.8 / 49.5 / 63.7 / 51.4 /
46.8 thinking on, 57.0 / 60.7 / 65.0 / 21.9 / 61.3 off.

**The breakdown's ms/step includes the prompt's prefill.** `nsys_step_breakdown`
skips 20 markers — exactly the warm request's passes — so its window
opens on the profiled request's prefill (300 ms for the default prompt,
9.8 s for a 10K one) and its per-step average carries that. The
commit-to-commit intervals (`step_timeline.py`) are the clean pass; every
per-step figure in this record from a timeline is clean, the breakdown
tables' wall figures are ~1.2 ms high at short context.

**The pass at ~10K context** (`profile_ctx2k/`, the sweep's 6.8K-word
prompt = ~10K tokens): 60–65 ms per step against 57–59 at short context.
The growth is the QSA attention over the indexer's 2,048 selected keys:
`attn_partial_kernel<256>` 180 us per layer (12 verify layers + 3 draft
steps at ~115 us) for ~4–8 MB of K/V — ten times its byte floor, a
latency-bound kernel; the same cost stands at 2K context, where the
budget covers the whole context, so it is the LongCode cell's +3 ms too.

**The prefetch as a translation warmer** (`sweep_touch/`, 04:37–04:42; the
1024-token sampled request, stats ms/step; ±2 ms run to run): the byte
prefetch (baseline) 57.5–58.2; `DGPP_L2_PREFETCH_FORM=touch` with one
128-byte line per 2 MB page 58.3–58.8; per 64 KB 61.0; per 64 KB with a
32 MB window 60.8; off 59.1. A touch per page holds the pass while
reading 1/16,384 of the bytes: what the byte prefetch delivers is the page
walks ahead of the consumer, and it costs 3.9 GB of DRAM reads per pass
to do it. Not adopted as the default on this evidence alone (the pass is
unchanged at one stream; the freed bandwidth matters where the DRAM is
the limiter — the row batches at C2–C4 — which is the measurement that
would decide it). The form and its knobs stay in kernels/l2_prefetch.cu.

## S6 round 4 — the attention gather, the shared expert beside the experts (2026-09-29, 05:10)

`serve_round4/`. Two changes, both bitwise by construction:

1. **The QSA attention's gather chased three dependent loads per vector.**
   `attn_partial_kernel<256>` resolved topk → block table → physical row
   inside the gather loop, eleven iterations per thread in series, ~25 us
   a tile; at 2K–10K context the kernel ran 180 us per verify layer and
   ~115 per draft step for 4–16 MB of K/V. The tile's physical rows are
   now resolved once into shared memory by 32 threads and every row load
   of the tile issues together. The prefill kernels (qsa_prefill.cu,
   qsa_warp.cu) chase the same way — a prefill item.
2. **The shared expert's gate/up half on a side stream** beside the
   routed experts (forked at the MoE's start, joined before the down
   half): its 21 us launch — a 3.3 MB matrix at 55–61 % of the line rate,
   ramp-bound — ran on the chain after the routed stream. Same kernels
   (`qwen_moe_shared_gate_up_decode`, `qwen_moe_shared_down_decode`), so
   bitwise the one-call tail. `DGPP_QWEN_SHARED_SIDE=off` keeps the chain.

62/62 tests (the Qwen, QSA, MoE, pick and KDA suites).

**Round 4 measured** (05:10–05:18): transcripts identical 4 of 4; the cells
44.4 / 49.7 / 63.8 / 51.5 / 47.0 thinking on, 57.6 / 61.0 / 65.2 / 21.9 /
61.6 off. At ~10K context `attn_partial` 182 → 153 us per launch (the
offsets in shared memory removed two of the three dependent loads per
vector; the load-store loop still ran its eleven iterations per thread as
serial round trips — the next binary stages a round's loads in registers
first). The shared gate/up on the side stream: 35 us there against 21 on
the chain and the clean step unchanged (58.2 → 58.3 short, 62.4 → 62.1 at
10K): the experts are DRAM-bound, the shared bytes cost the same DRAM
time wherever they run — a null result, the side stream is opt-in now
(`DGPP_QWEN_SHARED_SIDE=on`). At 10K context the selection pipeline is
153 + 122 (`index_score`) + 49 (`select_from_keys`) us per QSA layer.

**Round 5** (05:24–05:32, `serve_round5/`): the gather's loads staged in
registers per round — no change (attn_partial 155 us; the clean step 58.3
short / 62.5 at 10K; transcripts identical; LongCode 47.1 / 61.7). The
gather was not the kernel's bound after all: per tile the scores loop ran
a five-stage shuffle tree per token, 32 tokens in series (~4 us of a
tile's ~7), so the kernel is latency-bound in its own reductions. Round 6
computes every lane's 32 partials first and runs the trees over all 32 at
once (the same tree per token: bitwise).

**The draft layer's one-row GR path** (`sweep_gr/`): `DGPP_QWEN_GR_FUSED=off`
57.6 ms/step against 57.1–58.4 for the fused one-row kernel — no
difference; the fused kernel's 118 us in the step is the platform's rate
for those bytes there, not the fusion.

**Rounds 6–7, the attention kernel at 2K+ context** (`serve_round6/`,
`serve_round7/`; per-launch times from decode-only steps, 15 launches per
pass at ~10K context):

| form | attn_partial us | clean step ms |
|---|---|---|
| baseline (round 3) | 181 | 61.9 |
| offsets in shared memory (round 4) | 158 | 61.9 |
| loads staged in registers per round (round 5) | 154 | 62.0 |
| scores' shuffle trees batched over the tile (round 6) | 137 | 61.8 |
| next tile fetched behind the compute (round 7, 125 registers) | 170 | 61.9 |

Round 6's form stays (bitwise; the pipeline is off, one register-lean
revision away). The kernel went 181 → 137 us and the clean step did not
move: the 0.7 ms per pass it gave back at 10K context is not visible in
the step, and the LongCode cell (2K context) reads 61.4–61.8 across rounds
5–7. `index_score` (27 us) and `select_from_keys` (23 us) are small at
every context; the breakdown table's per-launch figures for them (122–132
us in some runs) are the prefill's launches of the same kernels inside
the window, not the decode's.

**The touch-form prefetch at concurrency** (`c4_touch/`, 05:50–05:59;
timed_load, greedy, 320 tokens, prose / code tok/s, two runs averaged):

| | C1 | C2 | C4 |
|---|---|---|---|
| byte prefetch | 44.9 / 55.8 | 53.0 / 76.2 | 74.4 / 105.4 |
| touch per 2 MB page | 43.9 / 54.6 | 51.2 / 73.8 | 73.6 / 104.4 |
| byte prefetch, closing | 44.7 / 55.8 | 53.1 / 76.4 | 74.5 / 105.7 |

The touch form is 1–3 % slower at every concurrency: the byte prefetch's
L2 hits (small as they are) and whatever DRAM locality its streaming
gives the consumers are worth more than the bandwidth it spends. Decided:
the byte prefetch stays the default; the touch form remains a knob.

## S6 round 8 — the draft layer's bf12, the head dump, the greedy profile (2026-09-29, 06:46–06:58)

`serve_round8/`. Three sessions on one binary: the head-input dump
(`head_dump.bin`, 5,571 rows: 3,351 draft + 2,220 verify head inputs from
the five bench prompts thinking on and off plus two longer ones), the
plain gate + benches, and a greedy short-context profile (`profile_t0/`,
400 tokens, T=0: the pick kernels' true greedy cost, and the pass without
prefill kernels in the window).

- **bf12 for the draft layer under fp8 dense weights**: the packer had
  returned early for every layer under `dense_weights: fp8`, including the
  MTP layer whose q/k/v/index/o and the two fc matrices are bf16 in this
  checkpoint. Now packed (7 matrices, 0.12 → 0.09 GiB); 4/4 transcripts
  identical; the draft's q_proj runs 195 us for 47 MB (line rate).
  Bench nothink run 2: Code 61.0 / JSON 65.8 / LongCode 62.1 (round 7:
  60.8 / 65.1 / 61.4). Thinking on run 2: 44.7 / 50.0 / 64.3 / 52.3 / 47.5.
- **The greedy profile** (122 passes, 56.66 ms/pass, 1,482 kernels/pass,
  true GPU idle 0.38 ms): head 4 × 2.79 ms; moe gate_up 11.4, down 5.2;
  GDN in-proj multi 8.5 (177 us for 42 MB, 95 % of line rate); out-proj
  3.7 (78 us for 15.7 MB, 81 %); hyper-connection sites 97 × (35 + 31 + 8
  + 2.5 us) = 7.9 ms against a 5.1 ms byte floor; the pick 30 us per row
  set (the 116/54 us figures of the 01:05 profile were a sampled request's
  radix select and fp64 exp); spec_commit 40 us (the 875 us of the 01:05
  profile was the pre-replay state copy). The pass by phase: draft step 1
  (accepted rows, 4-row kernels) 1.26 + 2.76 ms, steps 2–3 0.90 + 2.75
  each, verify 42.0 + 2.79.
- **The head bit-plane study** (`~/.claude/jobs/566ee5c5/tmp/headplane/study.py`
  over 300 dumped rows against the real int8 head, signed scales handled):
  6-bit high plane → bound ±3.7 logits at the top, candidates 4 / 840 / 40k
  (median / mean / p99), exact argmax always inside; 5 bits ±9.2 with a
  fat tail; 4 bits useless. Basis for §6.6 of the plan and issue #69.

## S6 round 9 — the fused mix A/B and the prefetch size cap (2026-09-29, 07:07–07:20)

`serve_round9/`. ms/step from the server stats over 1024-token requests
(the pass), the LongCode/JSON/Code nothink cells where run.

| setting | ms/step | note |
|---|---|---|
| act_up + mix_finish (`DGPP_QWEN_GR_MIX_FUSED=off`) | 57.8 / 58.1 | 4/4 transcripts identical; LongCode 62.2 |
| act_up_mix, first cut (rows consumed one after another) | 58.8 / 59.2 | bitwise, LongCode 60.4: four load latencies per warp |
| prefetch cap 3 MB (`DGPP_QWEN_PREFETCH_MAX_MB=3`) | 60.2 / 60.3 | |
| prefetch cap 8 MB | 58.7 / 59.2 | |
| prefetch cap 32 MB | 58.9 | |
| prefetch off | 60.4 / 61.1 | |
| act_up_mix first cut, closing | 58.7 / 58.9 | |

Decided: the byte prefetch stays exactly as it is (its value is on the big
streams too — capping loses or is neutral, off costs 1.5–2 ms); the fused
mix was rewritten with every row's chunk batch issued before any is
consumed (the fp8 pair kernel's discipline), re-measured in round 10.

## S6 round 10 — the plane head's first cut, the hoisted mix, the slice's first boot (2026-09-29, 07:39–08:01)

`serve_round10/`. Greedy 1024-token requests (the stats' ms/step is the
pass), the transcript gate on every setting, `profile_t0/` a greedy
400-token profile of the plane binary.

| setting | ms/step | 691-token greedy request | gate |
|---|---|---|---|
| planes on, act_up_mix (hoisted loads) | 56.8 / 57.3 | 18.53 s | 4/4 identical |
| planes off (`DGPP_QWEN_HEAD_PLANES=off`) | 56.7 / 56.9 | 18.45 s | 4/4 |
| planes on, `DGPP_QWEN_GR_MIX_FUSED=off` | 56.6 / 56.7 | 18.40 s | 4/4 |
| planes on, closing | 57.0 / 57.1 | 18.53 s | 4/4 |
| `--draft-vocab` (the slice) | — | weight bump OOM: the slice was not in the globals estimate | |

Bench cells with the planes: nothink 57.1 / 60.9 / 65.1 / 21.8 / 61.5,
thinking on 44.7 / 49.6 / 63.9 / 51.2 / 46.9 — flat against round 8.

The profile says why (122 passes, 57.13 ms, 1,384 kernels):
`head_pass1` 2,552 us per call for 477 MB (187 GB/s: op-bound — the
per-code integer decode and int-to-float conversions, against the int8
chain's 2,780 us at line rate for 636 MB); `head_pass2` 336 us per call
(every one of the 31k blocks staged the activations before deciding
candidacy — the head's own bytes again, out of L2); `gr_act_up_mix` 44.7 us
per site against act_up 30.8 + mix_finish 2.5 (32 uint4 of batch registers
per thread at four rows: occupancy). The exactness held everywhere (the
argmax form ran on every greedy call; sampled requests took the full form).
Fixes in round 11: a two-bit-plane bit order that makes the six-bit decode
the int4 core's fp16 pair trick (v = 16 (1024 + nb) + 4 (1024 + md) −
20606.5, exact), candidacy before staging, the fused mix's batch registers
sized to the row (two chunks at lowrank 320), the slice in the bump.

## S6 round 11 — the slice lands; the plane head's second cut (2026-09-29, 08:10–08:27)

`serve_round11/`. The same protocol as round 10.

| setting | ms/step | 691-token greedy | gate |
|---|---|---|---|
| planes (per-group layout, fp16 decode, candidacy first), act_up_mix (sized batch) | 56.6 / 56.8 | 18.5 s | 4/4 |
| planes off | 56.5 / 56.7 | | 4/4 |
| planes, `DGPP_QWEN_GR_MIX_FUSED=off` | 56.5 / 56.7 | | 4/4 |
| planes, closing | 56.5 / 56.7 | | 4/4 |
| **`--draft-vocab` (65,536 ids, `tools/build_draft_vocab.py`)** | **50.2 / 51.0** | **16.5 s** | **4/4 identical** |

The slice's cells (opt-in; never a headline): thinking off Code 66.3–66.6 /
JSON 72.3–72.9 / LongCode 68.8–68.9; thinking on 55.4–55.5 / 70.1–70.3 /
52.3–52.4 (the default binary 60.9 / 65.1 / 61.5 and 49.6 / 63.9 / 46.9).
Against the author's default config (their slice on): thinking off +7 /
+10 / +10 %, thinking on −2 / +13 / +8 %.

The profile: `head_pass1` still 2,555 us — the fp16 decode changed
nothing, so the six-bit pass was never op-bound: the per-group interleave
leaves the skipped 32-byte sector in the same 64-byte DRAM burst as the mid
plane, and the memory moves every byte. `head_pass2` 265 us (the per-row
bound reads and `-inf` scatter). `gr_act_up_mix` 31.2 us per site (act_up
30.8 + mix_finish 2.5 before: parity, one launch fewer). The folded routed
accumulation made `shared_down_accum_round` 18.2 us (12.1 + the 1.7 us
accumulate before): lane 0 issued ten serial loads. Round 12: per-row
planes (each plane a contiguous 640/1280-byte run), the candidates
compacted into a list for the exact pass, the fold's loads across lanes.

## S6 round 12 — the plane head in per-row layout, the slice on it, the depth sweep (2026-09-29, 08:34–)

`serve_round12/`. The same protocol (greedy 691-token requests, the
stats' ms/step, the transcript gate, the author's bench).

| setting | ms/step | 691-token greedy | nothink Code / JSON / LongCode | thinking on | gate |
|---|---|---|---|---|---|
| **planes (per-row layout), fused mix, folded accumulation** | **53.9 / 54.1** | 17.5 s | **64.1 / 68.9 / 65.0** | **52.1 / 67.3 / 49.6** | 4/4 identical |
| planes off (`DGPP_QWEN_HEAD_PLANES=off`) | 56.3 / 56.7 | 18.35 s | | | 4/4 |
| planes + `--draft-vocab` (opt-in) | 49.1 / 49.3 | 16.1 s | 68.7 / 74.8 / 70.5 | 56.8 / 72.2 / 53.6 | 4/4 |
| `--mtp-depth 4` | | | 62.0–62.5 / 67.5–67.7 / 63.0–63.2 | | 0/4: the five-row verify takes other kernel families (chunked GEMVs, cuBLAS for the GR down past the staging budget) — not bitwise depth 3, and slower |

The per-row plane layout is the version that pays: −2.6 ms per pass at
identical transcripts, the default binary's cells +5–6 % across the board.
Against the author's default configuration (their 65k draft slice on):
thinking off 64.1 / 68.9 / 65.0 vs 62.0 / 66.0 / 62.4 without any slice;
with ours, 68.7 / 74.8 / 70.5. Thinking on with the slice 56.8 / 72.2 /
53.6 vs their 56.5 / 62.1 / 48.5.
| `--mtp-depth 2` | 47.0 / 47.1 | 16.5 s | | | 4/4 |
| `--mtp-depth 4` + the slice | 60.5 / 60.6 | 18.4 s (660 tokens) | 68.2 / 74.4 / 70.0 | 48.7 / 64.4 / 48.2 | 0/4 (as depth 4) |
| planes, closing | 54.0 / 54.2 | 17.57 s | | | 4/4 |

Depth 2 accepts fewer tokens per pass (1.9–2.0 against 2.1–2.3) but its
three-row verify is 7 ms cheaper, and on the reasoning-heavy greedy request
it finishes 6 % sooner than depth 3; on the author's code and JSON cells
depth 3 has stayed ahead in every round. Workload-dependent — the adaptive
verify depth (engine.mtp_schedule) is where that decision belongs.

The round-12 profile (`serve_round12/profile_t0/`, greedy 400 tokens, 122
passes, 54.14 ms per pass, 1,337 kernels, true idle 0.35 ms):
`head_pass1` 2,099 us per call for 477 MB (227 GB/s — line rate, the
kernel is bandwidth-bound again), `head_candidates` 16.7 us,
`head_exact_list` 8.0 us: 2.13 ms per head read against 2.78 for the
row layout, four reads a pass. `gr_act_up_mix` 31.1 us per site (act_up +
mix_finish 33.3 before, one launch fewer), `gr_down_inject` 31.6,
`shared_down_accum_round` 16.5 with the folded routed accumulation (lane 0's
ten serial loads; the lane-parallel form is round 13's), the pass's
non-prefetch kernel time 53.45 ms of the 54.14 wall.

## S6 round 13 — the hyper-connection down companion at world 1, the fold's loads (2026-09-29, 09:06–09:22)

`serve_round13/`. The round-12 binary plus two folds: the hyper-connection
down matrices' 12-bit companions (packed at world 1), and the routed
accumulation folded into the shared down with lane-parallel loads.

| setting | ms/step | 691-token greedy | nothink LongCode | gate |
|---|---|---|---|---|
| companion on (default of this binary) | 56.2 / 56.4 | 18.27 s | 62.5 | 4/4 |
| `DGPP_QWEN_GR_BF12=off` | 53.9 / 54.2 | 17.56 s | | 4/4 |
| companion on, closing | 56.2 / 56.5 | 18.27 s | | 4/4 |
| `--mtp-depth 2` (companion on) | 48.7 / 48.9 | 17.08 s | 57.0 (thinking on 47.1) | 4/4 |

The profile (`profile_t0/`): `gr_down_inject_bf12` 46.5 us per site
against the bf16 kernel's 31.6 — the packed decode on 41 blocks of 320
long rows loses more in ops than 25 % of the bytes saves, at world 1 as
the 2026-09-26 bench found for the fabric worlds; +2 ms a pass. REJECTED:
default off (`DGPP_QWEN_GR_BF12=on` re-enables), the bitwise gate kept.
`shared_down_accum_round` with the fold 16.8 us against 12.1 + 1.7
unfused: a warp's ten slot loads land in ten sectors after its dot; the
next cut stages the block's rows' contributions cooperatively before the
dot. Depth 2 loses the code cells (57.0 vs 62.5) and holds the
reasoning-heavy request; depth 3 stays.

## S6 round 14 — the benchmarks-document campaign (2026-09-29, 09:23–09:34)

`serve_round14/`: the docs/benchmarks.md protocol on the round-13 binary
with the hyper-connection companion off by default (`timed_load.py
--concurrency 1,2,4 --classes all --repeat 3 --max-tokens 256`,
`CUDA_DEVICE_MAX_CONNECTIONS=32`, greedy; then sampled at T=1).

| | prose | code | json | math | chat |
|---|---|---|---|---|---|
| C1 engine tok/s (median of 3) | 50.6 | 59.2 | 71.2 | 62.7 | 44.4 |
| C1 ms/pass, tok/pass | 52.5, 2.66 | 53.2, 3.15 | 53.4, 3.81 | 53.5, 3.36 | 53.2, 2.36 |
| C2 loaded wall tok/s | 56.1 | 79.0 | 88.5 | 80.9 | 60.2 |
| C4 loaded wall tok/s | 75.8 | 105.6 | 120.1 | 104.6 | 85.9 |
| sampled T=1, C1 engine | 44.4 | 57.3 | 66.7 | 59.5 | 41.6 |
| sampled T=1, C4 wall | 71.7 | 78.8 | 117.6 | 101.5 | 72.4 |

Rows for the document: C1 engine 44.4–71.2, C4 loaded 75.8–120.1
(against the NVFP4 single-Spark template's 42.3–50.0 and 83.4–95.4). The
cold prefill probe found no `data/gsm8k_test.jsonl` in the worktree (the
main tree's is linked now); it runs in round 15.

## S6 round 15 — the final state (2026-09-29, 09:36–09:50)

`serve_round15/`: the shipped defaults (per-row plane head, act_up_mix,
the routed accumulation folded with cooperative staging, the
hyper-connection companion off), 4/4 transcripts identical on every
setting.

| | ms/step | nothink Code / JSON / LongCode | thinking on |
|---|---|---|---|
| default | 53.8 / 54.1 | 64.0 / 68.7 / 64.9 | 52.1 / 67.1 / 49.7 |
| `--draft-vocab` (opt-in) | 49.1 / 49.2 | 68.3 / 74.9 / 70.6 | 56.7 / 72.3 / 53.7 |

Cold prefill (`serve_prefill_probe.py`, no thinking, three repeats): 1.525
/ 5.152 / 20.005 s at 2K / 8K / 32K (0.73 / 0.63 / 0.61 ms per token).
The profile (`profile_t0/`): 54.12 ms per pass, 1,337 kernels, true idle
0.35 ms; `shared_down_accum_round` with the staged fold 15.3 us (the
unfused chain's 12.1 + 1.7 and a launch gap: neutral, one launch fewer);
the head reads 2,115 + 17 + 8 us each.

Where the campaign ends against the author's published cells (their
default config, the 65k draft slice on): thinking off 64.0 / 68.7 / 64.9
vs 62.0 / 66.0 / 62.4 with no slice, 68.3 / 74.9 / 70.6 with ours;
thinking on 52.1 / 67.1 / 49.7 vs 56.5 / 62.1 / 48.5 without the slice
(the Code cell is a different trajectory: theirs stops at 276 tokens, ours
reasons to the 512 cap), 56.7 / 72.3 / 53.7 with it.

## S6 round 16 — the per-launch intercept, measured; the prologue prefetch (2026-09-29, 15:00–)

The question the campaign ended on: how much of the pass's ~4 ms above
the in-step floor is per-launch ramp and tail that only a fused or
persistent form could recover. `qwen_dense_bench --sweep` (new) answers
it two ways.

**The bytes-per-launch sweep** (`sweep_before.log`; stream launches, cold
ring, least squares us = a + bytes/rate per family, `a` the intercept):

| family (rows 1 / 4) | intercept us | asymptotic GB/s | x launches per pass |
|---|---|---|---|
| out_proj fp8 scale_gemv (k 6144) | 10.7–13.9 / 10.3–12.6 | 245–253 / 242–248 | 0.54–0.71 / 0.53–0.64 ms |
| in_proj fp8 multi (k 2560) | 8.7–10.8 / 5.4–7.1 | 249–253 / 227–229 | 0.45–0.55 / 0.27–0.36 ms |
| hyper-connection down+inject bf16 (k 10240) | 5.5–7.3 / 7.2–9.7 | 236–241 / 229–236 | 0.56–0.75 / 0.73–0.99 ms |
| hyper-connection up bf16 (k 320) | 5.4–6.5 / 5.1–7.1 | 246–249 / 226–231 | 0.55–0.66 / 0.52–0.73 ms |
| slot gate_up packq int4 g128 (k 2560) | 7.0–7.1 / 8.4–8.7 | 246 / 242 | 0.36 / 0.43–0.45 ms |
| slot down packq int4 g128 (k 640) | 6.3–6.7 / 6.1–7.4 | 245–248 / 221–223 | 0.32–0.34 / 0.31–0.38 ms |

An empty kernel back to back on the stream costs 2.06 us per launch, so
the intercepts above are ~2 us of stream launch gap plus 3–12 us inside
the kernel; the asymptotic rates sit above the 233.6 GB/s ring floor —
the floor rate is conservative by ~5 % for launches past ~10 MB.

**The graph-form chain** (`--only pdl`): the hyper-connection pair as a
dependent chain, down [256 x 10240] → up [10240 x 256], 51 cold pairs per
chain, 10.5 MB per pair against a 44.9 us floor, median of five. Forms of
one naive kernel (a warp per row, the production pass structure) beside
the production pair itself:

| us per pair, graph launch | rows 1 | rows 4 |
|---|---|---|
| naive, weights issued before the staging | 46.1 | 49.8 |
| PDL, wait at the top (the earlier null probe's form) | 45.2 | 48.5 |
| PDL, first pass issued before the wait | 44.4 | 47.6 |
| naive, the production order (stage, barrier, then loads) | 46.6 | 50.1 |
| the production order + L2 prefetch of the first pass before the staging | 45.1 | 47.9 |
| the production pair (`qwen_gr_down_inject_bf16` + `launch_bf16_gemv`) | 47.3 | 48.6 |

On the stream the same forms sit 4–5 us per pair higher (the launch gap
the graph removes; PDL hides it there, which is all PDL does). So: in
the graph, the production pair already runs at 92–95 % of the ring floor;
the whole reachable prologue is ~1 us per launch at 1 row and ~0.5 us at
4 rows, and PDL's early issue and a plain L2 prefetch ahead of the
staging reach the same point. The in-situ "7.5 vs 5.1 ms" for the
hyper-connection sites was per-site accounting under the prefetch
stream, not a per-launch defect; the k-split / bf12 / cooperative
restructure of that site is off the plan (≤ 0.4 ms per pass reachable).
The megakernel's payoff is bounded by the ~1,000 small latency-chain
launches, not by the GEMV ramps.

**Shipped from it: the prologue prefetch** (`gemv::prefetch_row_head`,
gemv_common.cuh): every decode GEMV warp issues `prefetch.global.L2` for
its first weight pass (≤ 4 KB, the row range it will read) before the
block stages its activations — fp8 rows and multi, bf16 single / dual /
multi, hyper-connection down_inject and act_up_mix, the packq slot
gate_up and down, the shared expert's gate_up and down. Nothing reaches
a register: bitwise by construction.

**On the fabric it loses** (`serve_prologue/`, two binaries interleaved
A B A B at world 1, greedy 1024-token request, the 10-second stats
window; the nothink bench on the closing pair):

| leg | ms/step | greedy 691 tokens | nothink Code / JSON / LongCode |
|---|---|---|---|
| A1 before | 53.8 | 17.45–17.50 s | |
| B1 prologue prefetch | 54.6 | 17.73–17.75 s | |
| A2 before | 53.7 | 17.47–17.48 s | 63.9–64.4 / 68.6–68.9 / 65.0–65.2 |
| B2 prologue prefetch | 54.3 | 17.68–17.72 s | 62.9–63.0 / 68.0–68.1 / 64.3–64.4 |

+1.1–1.5 % per step, transcripts 4/4 identical. In the step the dense
matrices are already in L2 when their kernel starts (the side stream's
209 prefetch launches run ahead), so the first pass's DRAM round trip the
bench hides does not exist there, and the added prefetch instructions —
and, in the slot kernels, the expert view lookup moved ahead of the
staging — are pure cost. REVERTED (every call site back to the production
order, the helper removed); the sweep and the chain harness stay in the
bench. The lesson is the escape-split one again: a cold-ring bench
measures a regime the step never runs; only the two-binary A/B decides.
The corollary matters more than the lever: with the GEMVs L2-fed, the
pass's ~4 ms over the byte floor is the small latency-chain launches
(≈ 2 ms), the prefetch stream's own DRAM contention and the head's line
rate — not GEMV ramps.

**The decode step at context** (`serve_ctx/ctx_decode.log`; one greedy
no-think request per length, the engine's step_ms / decode_steps deltas
after the first token; the campaign's other decode numbers are all at
≤ 80-token prompts):

| prompt tokens | ms/step | tok/step | prefill (TTFT) |
|---|---|---|---|
| 131 / 188 | 52.6 / 52.6 | 2.6 / 2.5 | 0.38 / 0.40 s |
| 3,195 / 3,162 | 55.7 / 56.0 | 2.4 / 3.0 | 2.12 / 2.05 s |
| 11,685 / 11,909 | 56.9 / 55.6 | 3.0 / 2.6 | 7.29 / 7.39 s |
| 47,738 / 46,799 | 55.8 / 56.0 | 2.6 / 2.2 | 30.2 / 29.9 s |

The step is +3.2 ms (6 %) from the short-prompt cells at 3K and flat from
there to 47K — the sparse attention's selection budget is fixed, so its
cost is; only the index scoring grows and it is small. Prefill holds
0.63–0.65 ms per token to 47K. That 3.2 ms is the attention chain at
context (select, the listed attention over the selected keys, the index
score): larger than every short-context fold left on the ledger together,
so the register-lean attention pipeline (the pipelined tile loop the
round-7 profile measured at 170 us against the serial 137 for its 125
registers) is the next item; its floor is set by the profile at context
(`serve_ctx_profile/`).

**The step at 12K context, profiled** (`serve_ctx_profile2/profile_ctx/breakdown.txt`;
two 1000-token completions at 11.6K / 12.7K prompt tokens, 667 decode
steps, the prefill windows dropped with the breakdown tool's new
`--max-window-ms`): 57.3 ms/step under nsys, 1,353 kernels. Against the
short-context profile of round 15 the change is:

| kernel | short context | 12K | delta |
|---|---|---|---|
| attn_partial<256> (15/step) | 0.419 ms (27.8 us) | 2.050 ms (136.6 us) | +1.63 |
| kda_recurrent (36 → 52/step) | 1.052 | 1.571 | +0.52 (the prefix cache's state snapshots at context) |
| index_score (15/step) | 0.129 (8.5 us) | 0.399 (26.6 us) | +0.27 |
| select_from_keys (15/step) | 0.292 | 0.280 | 0 |
| everything else | | | +0.8 (a fraction of a percent on each streaming kernel) |

The listed attention at 12K reads 4 rows × 2,048 selected keys × K and V
× 2 kv heads × 512 B = 16.8 MB a layer: 72 us at the line rate against
the 136.6 measured, so ~1 ms/step of the 2.05 is the gather's latency,
not its bytes (the tile loop resolves, gathers and computes in series;
the register-pipelined form of round 7 lost to its 125 registers). The
register-lean form built here: the tile's K and V rows as cp.async
groups into the same shared buffers, the next tile's K copy issued once
this tile's scores have read kt (it runs under the PV phase), the next V
copy once PV has read vt (under the next tile's scores) — the arithmetic
the serial loop's in the same order, bitwise (`qsa_test` gates the two
forms against each other); `DGPP_QSA_ASYNC=0` keeps the serial gather.

**The async gather on the fabric** (`serve_qsa_async/`; one binary,
`DGPP_QSA_ASYNC=0` (serial) against the default, legs S A S A; each leg
one 600-token completion at 12.0K, 11.9K and 47.6K prompt tokens, then
the short-context greedy request; every leg produced the same steps and
tokens per request — the decode is bitwise, and the closing leg's four
transcripts are identical to the depth-3 record):

| leg | 12.0K ms/step | 11.9K | 47.6K | short-context greedy | TTFT 12K / 47K |
|---|---|---|---|---|---|
| S1 serial | 56.7 | 56.8 | 56.9 | 53.8 | 7.41 / 30.26 s |
| A1 async | 56.4 | 56.3 | 56.4 | 53.8 | 7.26 / 29.75 s |
| S2 serial | 56.8 | 56.8 | 56.9 | 54.0 | 7.29 / 29.97 s |
| A2 async | 56.5 | 56.4 | 56.4 | 54.0 | 7.25 / 29.70 s |

−0.4 to −0.5 ms a step at every context from 12K up (the attention chain
2.05 → ~1.6 ms), nothing at short context, prefill's chunk tails (the
decode kernel serves rows under 128) −1.5 %. SHIPPED as the default. What
the listed attention still holds at context: ~0.5 ms/step over its byte
floor (the two gathers' remaining latency, the 3× re-read of each kv
head across the head groups from L2, the serial `expf` loop on lane 0),
and the index scoring 0.4 ms; both are the next items at context.

**The short-context bundle** (`serve_bundle/`; legs old / N0 / N1 twice:
old = the round-15 binary, N0 = the new binary with the batched norm fold
off (the GDN a/b projections as bf16 problems of the fp8 in-projection
launch, one launch fewer per GDN layer), N1 = the same with the norm fold
on; greedy 1024-token request twice, the 10-second stats window):

| leg | ms/step | greedy 691 tokens | nothink Code / JSON / LongCode |
|---|---|---|---|
| old a / b | 53.6 / 53.8 | 17.48–17.55 s | 64.0–64.2 / 68.4–68.8 / 64.8–64.9 |
| N0 a / b (a/b in the multi) | 53.5 / 53.6 | 17.41–17.50 s | |
| N1 a / b (+ the norm fold) | 54.4 / 54.3 | 17.68–17.73 s | 63.1–63.3 / 67.9–68.0 / 64.4 |

The a/b projections in the multi launch: −0.1 to −0.2 ms a step in both
pairs (the noise floor, never slower), bitwise (`scale_gemm_test`): kept.
The batched group norm staged into the down + inject GEMV with one block
reduction for every (row, group): bitwise (`qwen_gr_test`, bf16 and fp8),
and **+0.8 ms a step** — the 41 down blocks each recompute the sixteen
norms of the four rows, and that redundant work costs more than the 8 us
launch it removes, as the sequential form did on 2026-09-09. Kept behind
`DGPP_QWEN_GR_NORM_FOLD=on`, default off. Transcripts 4/4 identical on
the closing leg.

**Closing state of round 16** (17:15): the release binary carries the
async attention gather (on), the a/b projections inside the in-projection
multi launch, and the norm fold off. The full suite on the fully rebuilt
CI tree: 172 of 176 pass; three were branch-level test debts from earlier
rounds, fixed here — `qwen_loader_test` expected the checkpoint's row
layout for the head (it now permutes its expectation into the planes
when the resident head is in them), `portability_test` had no
filename slug or catalogue row for the hybrid template (both added), and
`setup_tools_test` needed the doctor's "no shards" case to raise the
OSError the activation contract expects — and `dsv41_tp_test` ("tp bus
world failed to start") fails the same way on master's build beside the
production world, an environment failure, not the branch's.

**The context attention forms** (`serve_ctx_forms/`; the r16 shipped binary
against the new binary's geometries by environment, one 600-token
completion at 12.0K, 11.9K and 47.6K prompt tokens per leg, then the
short-context request; every leg the same steps and tokens per request,
the closing leg's four transcripts identical):

| leg | 12.0K | 11.9K | 47.6K ms/step | short | TTFT 47K |
|---|---|---|---|---|---|
| old 1 / 2 (shipped: 6 heads per block, one per warp; the serial index loop) | 56.2 / 56.3 | 56.3 / 56.3 | 56.4 / 56.4 | 53.6 / 53.7 | 30.2 / 29.8 s |
| A: the same geometry, the denominator's expf hoisted into the PV loop, index batch 1 | 56.4 | 56.3 | 56.6 | 53.8 | 31.2 s |
| B 1 / 2: 12 heads per block (a kv head's rows gathered once), one per warp, index batch 4 | 56.5 / 56.5 | 56.5 / 56.5 | 56.7 / 56.7 | 53.8 / 53.9 | 31.1 / 31.3 s |
| C 1 / 2: 12 heads per block, two per warp, index batch 4 | 56.4 / 56.5 | 56.4 / 56.6 | 56.6 / 56.6 | 53.8 / 53.8 | 31.2 / 31.1 s |

Nothing beats the shipped form: +0.1 to +0.3 ms a step at every
context for every geometry, and the prefill's 47K TTFT +1 s (+3.5 %) on
every new-binary leg — the batched index score (four pools' loads in
flight per warp, the same chain per pool) is no faster than the serial
loop at decode and slower where the prefill scores every pool for its
chunk tails; the kv-head-per-block gather saves L2 traffic the kernel
was not bound by (the second read of a tile hits L2 already), and costs
residency (384 threads at 125 registers: one block per SM). Reverted to
the shipped kernels (the geometry stays pinnable for the test, which
keeps the forms bitwise). What the listed attention and the index score
still hold at context is latency that these shapes do not reach.

**The benchmark document's remaining rows** (`serve_docrows/`, the campaign
runner's commands by hand on the shipped binary; 19:00): decode modes at
C1 (`timed_load`, five classes, three repetitions, the range of the class
medians) — plain (`--no-mtp --mtp-depth 1`) 31.6–31.7 tok/s, the template's
depth 3 44.4–71.2 (round 14), depth 4 35.9–65.9 (prose 67.0 ms a pass at
2.41 tokens against depth 3's 52.5 at 2.66: the five-row verify leaves the
four-row kernel families and prose accepts the fourth draft almost never);
the plain and default runs' greedy texts and token counts identical in
5/5 classes; the isolation probe (`isolation_probe.py`) identical on
repeated solo runs, different beside three others (the batch-shape
reduction-order class the other deployments show); op streams identical
on every row-producing launch (5/5); long-context parcels
(`long_context.py`, 4K / 32K / 128K): 3,855 tokens 2.337 s cold prefill,
57.05 ms a pass, 2.97 tokens a pass, 52.0 tok/s; 32,299 tokens 19.540 s,
57.06 ms, 3.19, 55.9; 129,826 tokens 89.831 s, 58.46 ms, 3.11, 53.2.

## S6 round 17 — the prefill (2026-09-29, 19:03–)

Against the author's ~2,100–2,200 tok/s (vLLM, 8192-token chunks) ours
runs 1,340 tok/s at 2K and ~1,600 from 8K to 47K (0.63 ms a token). The
profile (`serve_prefill_profile/`, nsys, the prefill probe's one-token
request at 32K and 8K, the last burst of kernels): 22.0 s of GPU time for
32,186 tokens (99.8 % busy).

| class | 32K ms | share | 8K share |
|---|---|---|---|
| packed-int expert GEMMs (gate/up bf16 out + down f32 out) | 7,517 | 34 % | 38 % |
| cuBLAS / cutlass dense GEMMs (fp8 dequantized, hyper-connections) | 3,276 | 15 % | 14 % |
| hyper-connection elementwise passes (combine_apply, group norm, mix, dots, gate act) | 2,697 | 12 % | 12 % |
| GDN chunked recurrence (state + prep) | 1,757 | 8 % | 8 % |
| expert accumulation (ordered f32, accum, round) | 1,258 | 6 % | 5 % |
| n-gram stage wait (the host gather) | 1,069 | 5 % | |
| listed attention (prefill warp kernel) | 1,030 | 5 % | 4 % |
| index score | 773 | 3.5 % | |
| router dots (tiled) | 596 | 2.7 % | 2.6 % |
| fp8 dequant to bf16 (per chunk per matrix) | 451 | 2 % | |

The dense stack's cuBLAS GEMMs run near 190 TFLOPS. The packed-int
expert GEMM (`packq_gemm_kernel`: 32 x 64 tiles, four warps, the int4 →
bf16 decode per fragment on the MMA path) runs 26 TFLOPS on the gate/up
shape (`packq_prefill_bench --m 4096 --n 640 --k 2560 --experts 512
--top-k 10`: 5.0–5.3 ms for 134 GFLOP), a seventh of what the tensor
cores do on the dense stack. That kernel is the prefill's gap to the
author, and the first item; the elementwise passes, the stage wait and
the chunk budget follow.

**The peak, calibrated** (`micro_gemm_peak`): cuBLASLt BF16 at m=2048,
n=24576, k=4096 runs 89.3 TFLOPS on this GPU — the dense tensor-core rate
to measure the expert GEMM against, not the 190 the profile's nvjet
kernel suggested. On `packq_prefill_bench --m 4096 --n 640 --k 2560 --sf 1
--experts 512 --top-k 10` the narrow kernel's 4.6–4.7 ms are 29 TFLOPS of
useful work; the bench's segments average 80 rows, so 64-row tiles carry
38 % padding and the rate on the work done is ~47 TFLOPS: half the peak.

**The wide kernel** (`packq_gemm_wide_kernel`, `--variant 1`, the default
for int4 rows; `DGPP_PACKQ_GEMM=narrow` keeps the old one): 64 x 128 x 64
tiles, eight warps, a two-stage cp.async pipeline, the k-step's codes
decoded once into a bf16 smem tile (nibble | 0x4300 is 128 + code in bf16
exactly, less 136 in bf16 is code − 8 exactly) so the MMA loop is
ldmatrix and mma, the row map and the scales hoisted off the issue path;
per element the same four m16n8k16 steps per 64-code group into a fresh
partial then one fma with the group's scale — bitwise the narrow kernel
(`bitwise_vs_narrow=yes` on both shapes, `packq_gemm_test` 3/3). Three
forms measured: three stages at one block per SM (66 KB): +5 % gate/up,
+20 % down — ncu: 16.7 % occupancy, 8.9 cycles per issued instruction,
latency-bound; two stages at two blocks per SM (46 KB, ≤ 128 registers):
−8 % gate/up (4.29 ms), −12 % down (6.38 ms), tensor pipe 44 % active;
the hoists: neutral. Left in the kernel: the ldmatrix → mma chains ("wait"
2.5 stall cycles per issue) a register-pipelined k-loop would cover if the
registers allowed, and the tile padding, which the 8192-token chunk
halves.

**Session Q was not the wide kernel** (`serve_prefill_ab/`): every
release build since the kernel landed had failed on a test file's
function-pointer ternary that the launcher's new parameter broke, so the
"new" legs ran the 17:56 binary (the narrow kernel): 0.70 / 0.61 / 0.61
ms per token against the shipped 0.71–0.72 / 0.60–0.62 / 0.61 — the same
binary's noise, and the 17,166-token greedy completion's sha identical
(927ffd8a) as it must be. The 8192-token chunk cannot come from a knob
("prefill budget ... no larger than the engine's"): the budget sizes the
prefill workspaces at load, so it takes a recipe copy. Both are session
R's, on a binary whose build is verified by its timestamp.

**Session R — the prefill bundle on the verified binary** (`serve_prefill_bundle/`;
release 19:50:28, tests 30/30; the prefill probe, three repeats per
length, engine prefill ms per token; the 17,166-token greedy completion's
sha per leg). The bundle: the wide packed-int expert GEMM, the prefill's
combine riding the next site's norm pass (`defer_combine` returns the
pending combine for prefill rows too, so `combine_norm` replaces
`combine_apply` + `group_rmsnorm`), and the prefill chunk's page advice
batched (the `total <= 4096` guard had every chunk take the per-row
`madvise` loop).

| leg | 2K | 8K | 32K | sha |
|---|---|---|---|---|
| shipped a / b | 0.73 / 0.72 | 0.63 / 0.60 | 0.61 / 0.61 (19.91 / 19.70 s) | 927ffd8a |
| bundle, 4096-token chunks | 0.67 | 0.56 | 0.57 (18.44 s) | 927ffd8a |

−7 % at 2K, −9 % at 8K, −7 % at 32K, bitwise: 32K prefill 1,590 → 1,745
tok/s. SHIPPED. The 8192-token chunk is refused at boot even from a
recipe copy ("a budget no larger than the engine's prefill capacity"): a
second limit sizes the engine's prefill rows; the profile leg (the recipe
copy under nsys) failed for the same reason and is rerun on the 4096
recipe.

**Session S — the 8192-token chunk** (`serve_chunk8192/`; the serve app
now raises the model's forward rows to a configured prefill budget above
the family's 4096, and the memory plan sizes the workspaces for it: 98.6
GiB against 88.7 at 4096; the bundle binary, release 20:32:35):

| leg | 2K | 8K | 32K | sha |
|---|---|---|---|---|
| bundle, 4096-token chunks | 0.67 | 0.56 (4.52 s) | 0.57 (18.47 s) | 927ffd8a |
| bundle, 8192-token chunks a / b | 0.67 / 0.70 | 0.57 / 0.56 (4.60 / 4.52 s) | 0.57 / 0.57 (18.63 / 18.39 s) | 927ffd8a |

A wash: the expert tiles' padding that halves at 8192 buys nothing the
larger chunk's other passes do not give back (the accumulation, the
elementwise passes and the dense GEMMs all scale with the chunk). The
recipe stays at 4096 (10 GiB less); the forward-rows change stays, since
it makes the recipe key mean what it says.

**The segment-aware tile** (`wide_tile<M16>`, 2026-09-29 20:50): the real
routing leaves most experts under 64 rows a chunk, and a 64-row block
spends its MMA and ldmatrix work on the padding. Each block now reads its
segment's rows and picks a warp layout: up to 16 rows, one m16 tile with
the eight warps over two n8 tiles each; up to 32, two m16 tiles with four
warps each over four n8 tiles; else the 2 x 4 layout. The pipeline, the
decode and every element's chain are unchanged, so the modes are bitwise
each other and the narrow kernel (`packq_gemm_test` 3/3, the bench's
byte compare). The bench's new `--distribution zipf` (expert e with
probability ~ (1 + e)^-0.8: 8 segments ≤ 16 rows, 169 ≤ 32, 200 ≤ 64 of
512, one hot expert at 2,290) against the narrow kernel: gate/up 5.3–5.5
→ 4.33–4.43 ms (−16..−20 %), down 7.4–7.6 → 6.6–6.8 ms (−9..−11 %);
uniform routing unchanged (−8 / −12 %). Fabric A/B against the round-17
bundle binary: session T.

**Session T — the segment-aware tile on the fabric** (`serve_segaware/`; the
round-17 bundle binary against it, legs base / new / base / new, the
prefill probe and the long-prompt sha):

| leg | 2K | 8K | 32K | sha |
|---|---|---|---|---|
| bundle a / b | 0.72 / 0.69 | 0.60 / 0.57 | 0.57 / 0.57 (18.36 / 18.51 s) | 927ffd8a |
| segment-aware a / b | 0.69 / 0.69 | 0.57 / 0.57 | 0.57 / 0.57 (18.41 / 18.42 s) | 927ffd8a |

A wash, bitwise. The Zipf bench's −16..−20 % does not exist in the step:
the real chunk's segments are not where the modes bite, or the launches
are bound elsewhere — the profile's per-launch grids settle it (below).

**The chunk profile's expert launches** (`serve_chunk8192/profile_4096/`,
the bundle at 4,096-token chunks; the sqlite's `packq_gemm_wide` launches
by grid): the 8K probe's two chunks run 96 layer triples with m-tiles
10–64 per launch — in most layers one routed expert takes 3,000–4,096 of
the chunk's 4,096 tokens (a hot expert in nearly every token's top-10) —
at gate 4.0–4.4, up 4.0–5.3, down 5.4–7.0 ms a launch, ~14 ms a layer,
670 ms a chunk (30 % of the prefill). The 556-token calibration prompt's
launches (m-tiles 5–8) take 2.0–2.5 ms for the same weight stream. The
grid then is n-tiles × 62 m-tiles × 512 segments = 160K blocks of which
~97 % read their segment and exit. Nothing else runs on the GPU during a
launch (no overlapping kernels or copies in the trace).

**The compact tile list** (`launch_moe_tile_list`, `MoeTile`, the layer's
`tile_list_for`; `DGPP_MOE_TILE_LIST=0` keeps the max_rows grid): one
block scans the segments' tile counts and writes (segment, first row) per
real 64-row tile; the wide kernel runs a 1-D grid of n-tiles × capacity
blocks (capacity = segments + rows / 64, bounded on the host without a
sync) and block b takes tile b / n-tiles. The same blocks do the same
tiles: bitwise (`packq_gemm_test` 3/3 with the list checked against the
host's, the bench's byte compare). The `routed_seg_max_rows` host sync per
layer is gone on that path. In the bench the empty blocks were not the
cost: gate/up unchanged, down −6..−16 % (its 20 n-tiles quadruple the
empty blocks).

**Where the launch's time goes** (the bench at m=4,096 × top-10 over 512
experts, gate shape n=640 k=2,560, the rows gathered through a row map as
the layer reads them — `--gather 1`, new — and the buffers in cudaMalloc
memory — `--device-copy 1`, new: the managed-memory buffers the bench
had used cost ~1 ms a launch of page mapping and hid every lever below):

| form | gate, Zipf | gate, uniform | note |
|---|---|---|---|
| wide kernel, tile list | 4.6–5.0 | 4.4–4.6 | the in-situ numbers |
| the weights L2-resident (`--same-weights 1`) | 2.65–2.85 | 2.65–2.85 | the compute-side floor with the partial tiles' padding |
| 10 hot experts × 4,096 rows | 2.1–2.4 | | no padding: 134 GFLOP at 58–65 TFLOPS |
| hot experts, the rows scattered (`--shuffle-rows 1`) | 2.4–2.5 | | the gather costs ~0.1 ms |
| 3 stages at one block per SM | 4.6–4.9 | | deeper lookahead alone does nothing |
| 4–5 stages at one block per SM | 4.5–4.8 | | nor more bytes in flight |
| the codes read as a tile-major layout (fake addresses) | 4.4–4.5 | | nor the DRAM pattern |
| **+ L2 prefetch of step g+3 (`DGPP_PACKQ_PREFETCH`, default 3)** | **3.46–3.61** | **3.07–3.23** | −25 %, bitwise |

ncu on the launch (before the prefetch): issue every 4.9 cycles a
scheduler, 0.30 eligible of 3.96 warps, long_scoreboard 31 % + barrier
26 % of the warp cycles, the tensor pipe ~45 % busy, 2 blocks per SM
(124 registers, 46 KB smem). The weight stream (419 MB a launch, 1.8 ms
at the line rate) was not overlapping the 2.7 ms of compute; the
two-stage pipeline's one step of lookahead is short of the loaded DRAM
latency, and `prefetch.global.L2` of the step three ahead (each
activation row's line every step, each code row's line every fourth)
closes it: gate 3.5 ms = the weights at the line rate + 1.7 ms. The
down shape (n=2,560 k=640, 10 k-steps, fp32 out = 419 MB written) goes
6.8–7.1 → 5.7–5.9 ms (Zipf) and 6.5–6.7 → 5.6–5.8 (uniform); a
556-token launch 3.6–3.7 → 2.0–2.2. What remains: the down's fp32
output (its 1.8 ms of writes; a bf16 intermediate is not bitwise the
ordered fp32 accumulation — the user's call), the partial tiles' B-side
work (the same code decode and staging for a 16-row tile as for 64),
and the register-decode three-stage form (no `bd` buffer: 41 KB keeps
two blocks per SM) if the prefetch leaves latency on the table in situ.
Fabric A/B: session U.

**Session U — the tile list and the prefetch on the fabric**
(`serve_tiles_prefetch/`, 22:55–23:03; the round-17 bundle binary
against the new one, legs base / new / base / new at 4,096-token chunks,
the prefill probe 2K/8K/32K × 3 and the long-prompt greedy sha per leg):

| leg | 2K | 8K | 32K | sha |
|---|---|---|---|---|
| bundle a / b | 1.398 / 1.417 s (0.69 / 0.70) | 4.605 / 4.609 s (0.57) | 18.457 / 18.481 s (0.57) | 927ffd8a |
| tile list + prefetch a / b | 1.294 / 1.294 s (0.64) | 4.413 / 4.420 s (0.55) | 17.801 / 17.832 s (0.55) | 927ffd8a |

−7.4 % / −4.2 % / −3.5 %, bitwise (the 17,166-token greedy completion's
sha on every leg). SHIPPED: `DGPP_MOE_TILE_LIST` and `DGPP_PACKQ_PREFETCH`
(default 3) are the knobs. The step is smaller than the bench's −25 % on
the gate/up launch because the expert GEMMs are ~30 % of the prefill and
the down launch gains less (its fp32 output). The benchmark-document
cells: 1.294 / 4.417 / 17.817 s.

**Session V — the benchmark-document rows on the round-18 binary**
(`serve_docrows_r18/`, 23:34–23:46; `timed_load` C1/C2/C4 × all classes
× 3 at 256 greedy tokens, then the long-context parcels 4K / 32K / 128K,
one cold and two cached each): C1 engine prose / code / json / math /
chat 51.0 / 59.5 / 71.5 / 63.1 / 44.7 (round 14: 50.6 / 59.2 / 71.2 /
62.7 / 44.4), C2 58.3–94.8, C4 engine 79.6–124.6, C4 wall with the
prefill 77.3–118.6 (75.8–120.1); the parcels: 3,855 tokens cold prefill
2.107 s (2.337), 32,299 17.523 s (19.540), 129,826 82.015 s (89.831),
decode 57.0 / 57.1 / 58.6 ms a pass at 2.97 / 3.19 / 3.11 tokens a pass —
the decode rows within noise of round 14, the prefill −10 / −10 / −9 %.
These are the document's cells now.

**Session W — the shipped binary's prefill, profiled at 32K and 8K**
(`serve_prefill_profile_r18/`, 2026-09-30 04:19–04:21; nsys, the prefill
probe's one-token request at 4,096-token chunks; 32K: 19.54 s of GPU time
under the tracer for 32,186 tokens (18.5 s unprofiled, 17.8 s in the
document's measurement); 8K: 5.39 s for 8,192 (4.43 unprofiled)):

| bucket | 32K s | 32K share | 8K share | floor at 32K |
|---|---|---|---|---|
| elementwise glue (combine_norm 1.22, accumulation 1.24, mix_finish 0.68, combine_dots 0.44, fp8 dequant 0.45, swiglu 0.27, gated norm 0.22, rounding 0.11, rope/norm 0.08) | 4.7 | 24 % | 24 % | ≤ 0.8 (the four-stream state read and written once a site) |
| expert GEMMs (wide gate/up 2.78 at 2.58 ms a launch, down 2.22 at 4.11) | 5.0 | 26 % | 29 % | 2.4 compute at the cuBLAS peak; 2.1 weights at 4K chunks, 1.0 at 8K |
| dense GEMMs (nvjet 2.56, cutlass 0.73) | 3.3 | 17 % | 15 % | near the bf16 cuBLAS peak |
| GDN chunk recurrence (state 1.09, prep 0.69) | 1.8 | 9 % | 9 % | unmeasured |
| QSA prefill attention 1.05 + indexer 0.90 | 2.0 | 10 % | 6 % | ~0.5 |
| n-gram table wait (stage_wait) | 1.1 | 5.5 % | < 1 % | 0 (overlappable) |
| router dots | 0.6 | 3 % | 3 % | 0.06 |

Against the author's README figure (~2,100–2,200 tok/s: a live per-step
rate at 8,192-token chunks, SEQS 16, the n-gram table in another
machine's RAM over RDMA; the model card itself publishes decode rates
only) ours is 1,840 tok/s at 8K–32K and 1,580 at 2K. The compute floor
for ~6B active parameters at 32K is 4.5–7 s on this GPU; both stacks sit
2.5–3.5x above it. The reference's stack: NVIDIA's model package in the
`vllm/vllm-openai:qwen38-flash-next` image, GPTQ-Marlin experts (~75 % of
the tensor peak at large M against our ~45 %), the flash-linear-attention
Triton chunk kernels, compiled elementwise math, 8,192-token chunks. The
ledger's order: the glue fused to one pass a site (−2..−3 s), the expert
GEMM at Marlin-class efficiency with the register-side decode and three
stages (−2 s, then 8K chunks pay), the n-gram gather a chunk ahead (−1 s),
the router as a GEMM and the QSA prefill forms (−0.5..−1 s): ~−6 s bitwise,
~2,800 tok/s. Not bitwise, the user's call: the down projection's bf16
intermediate (−0.9 s), the fp8 dense stack on the fp8 tensor cores
(−1.5 s). Recommended first: the author's stack profiled on this box.

**Session X — the author's stack on this box** (2026-09-30 04:47–05:22;
`~/claude-scratch/2026-09-30-reference-vllm/`: the author's repository at
5e8ae8a built on the pinned `vllm/vllm-openai:qwen38-flash-next` base
(vLLM 0.1.dev20073, torch 2.13/cu130, NVIDIA's `qwen3_8_flash_next/nvidia`
model package), served with their `serve.sh` settings — MTP 3, 8 seqs,
8,192-token chunks, prefix cache, DET_TOPK, DRAFT_VOCAB, the n-gram table
mmap'ed from the local FP8 snapshot, NOT over RDMA — and probed with our
prompts (the same GSM8K prose, seed 7, thinking off, max_tokens 1, the
visible TTFT). Two launcher lessons: their draft-vocabulary patch reads
the checkpoint index at the fixed `/model` path, and their
`splitting_ops` list must be copied verbatim (the PLE mmap op
`vllm::ple_mmap_lookup` copies device ids to the host and cannot run
under CUDA graph capture).

| prompt | reference TTFT, two repeats | reference best | ours (session U) |
|---|---|---|---|
| 2K (2,058 tok) | 1.487 / 1.122 s | 1.12 s (1,827 tok/s) | 1.294 s |
| 8K | 5.395 (7 scheduler steps) / 4.419 s | 4.42 s (1,824 tok/s) | 4.417 s |
| 32K | 15.754 / 15.416 s | 15.42 s (2,114 tok/s) | 17.817 s |

The published ~2,100–2,200 tok/s is reproduced at 32K on an NVMe-backed
table; at 2K–8K the reference runs ~1,825 tok/s. Its first request at a
new length is slow (shape warm-up); best-of-two is its steady state. The
smoke test's `word` × 8,000 prompt gives 2,371 tok/s (degenerate routing
keeps the expert weights in L2) and its repeat hits the prefix cache
(10,825). Against the best-of-two we are level at 8K and 13 % behind at
2K and 32K. The first nsys report lost the probe phase (the engine
process's CUDA buffers at the SIGINT shutdown); session X4 flushes them
every 5 s.

**Session X4 — the reference's kernels** (`run_nsys_0930_0523/`, nsys
with 5 s CUDA buffer flushes; the trace still loses its last ~40 s at the
SIGINT shutdown, but holds three complete 8,192-token chunks of the 32K
prefill: 3.21 / 3.43 s of GPU time each, and 0.35–0.43 s of GPU idle
between chunks — its n-gram gather runs synchronously at each chunk's
start, ~130K table rows from NVMe). Per 8,192 tokens against our r18
profile (two 4,096-token chunks, 4.9 s under the tracer):

| kernel class | reference, 8,192 tokens | ours, 8,192 tokens | note |
|---|---|---|---|
| routed experts | Marlin 677 ms (102 launches, 6.6 ms: w13 and w2 per layer) | wide packed 1,250 ms (gate 2.58 + up 2.58 + down 4.11 ms per 4,096) | theirs 1.8x faster; bf16 activations and partials, atomic-add split-k |
| dense projections | cutlass fp8 blockwise GEMM 580 ms | nvjet/cutlass bf16 822 ms + fp8 dequant 114 ms | fp8 tensor cores with the block scales in the epilogue (not bitwise our bf16 chain) |
| QSA prefill attention + indexer + top-k | 416 + 66 + 34 = 516 ms | 263 + 191 + 34 = 488 ms | level |
| hyper-connection glue | combine_norm 297 + gate_mix 166 = 463 ms | combine_norm 304 + mix_finish 170 + combine_dots 109 = 583 ms | level once combine_norm reaches the line rate |
| MoE partial sum | moe_sum (bf16) 97 ms | accumulation (fp32 ordered) 310 ms | −0.2 s per 8K with bf16 partials (not bitwise) |
| GDN recurrence | 462 ms (FLA chunk kernels + conv + norm) | 540 ms | −0.08 s per 8K |
| swiglu | 66 ms | 67 ms | level |
| n-gram gather | ~400 ms idle per chunk | 56 ms per 4,096 chunk (112 per 8K), 0 with the chunk-ahead staging | ours ahead |

The reference's 32K TTFT of 15.4 s is ~4 x 3.4 s of GPU work plus ~1.5 s
of gather gaps; ours 17.8 s is 8 x 2.2 s. The gap is the expert GEMM
(−0.55 s per 8K), the fp8 dense stack (−0.35 s) and the bf16 partials
(−0.2 s); everything else is level or ours. Bitwise levers left on our
side: the expert GEMM's efficiency (their Marlin at ~65 % of the tensor
peak against our ~45 %: the register-decode three-stage form, and w13
as one launch halving the activation re-reads), the GDN chunk kernels,
combine_norm to the line rate, the router dots, the chunk-ahead gather.
The user's call: fp8 dense GEMMs on the tensor cores with block scales
(−1.4 s at 32K), bf16 down partials (−0.9 s at 32K).

**Session Y — the round-19 levers on the fabric** (`serve_r19_levers/`,
05:39–05:51; the r18 binary against the r19 one — the router tile at
four tokens a warp, the combine-norm's batched loads, the chunk-ahead
n-gram staging — and the r19 binary's `DGPP_PACKQ_GEMM=wide3`
(register-decode three-stage expert GEMM) and `DGPP_MOE_PACKQ_DOWN_BF16=1`
(bf16 down partials, not bitwise) legs; prefill probe 2K/8K/32K × 3, the
long-prompt sha):

| leg | 2K | 8K | 32K | sha |
|---|---|---|---|---|
| r18 base a / b | 1.324 / 1.282 s | 4.576 / 4.386 s | 17.723 / 17.694 s | 927ffd8a |
| r19 default a / b | 1.296 / 1.301 | 4.406 / 4.414 | 17.902 / 17.910 | **5d614d0e** |
| r19 wide3 | 1.312 | 4.474 | 18.174 | 5d614d0e |
| r19 bf16 down partials | 1.264 | 4.233 | 17.350 | 5d614d0e |

Base leg a ran on a page cache the reference runs had displaced. The
default was level at 2K/8K, +1 % at 32K, and NOT bitwise. The
register-decode kernel adds nothing in situ (as on the idle-GPU bench:
gate Zipf 3.5–3.6 vs 3.5–3.6 ms, uniform 3.2 vs 3.2, down 5.9–6.1 vs
5.9–6.2, L2-resident 2.1–2.3 vs 2.1–2.4). The bf16 partials −3..−4 %
across the lengths; on this one prompt the 40-token greedy completion is
identical to the r19 default's (the sha change is not theirs).

**Session Z — the sha change isolated** (`serve_r19_prestage/`, 05:52–05:56):
`DGPP_QWEN_PLE_PRESTAGE=0` on the r19 binary gives 1.302 / 4.457 /
17.952 s and the base's sha 927ffd8a — the router and the combine-norm are
bitwise and buy nothing measurable; the prestage is the change. Its
self-check (`DGPP_QWEN_PLE_PRESTAGE_CHECK=1`: both channels gathered,
compared at the PLE layer) showed, on every interior 4,096-row chunk,
exactly as many differing rows from row 0 as the FOLLOWING chunk has
(4,096 → 3,796 → 9 → 0): the prestage buffer is single, and the next
chunk's prestage — issued at the chunk's start — overwrote it while this
chunk's rows still sat there. Fixed by issuing the prestage after the PLE
layer's gather consumed the staging (the wait and the conversion precede
the new publish in stream order), and by keying a chunk's claim on the
request id and a first/last-token check as well as the ids pointer, rows
and position. Session Z2 verifies.

**Session Z2 — the prestage fixed, verified** (`serve_r19_prestage2/`,
05:59–06:07): the self-check leg reports 0 differing rows and 0 differing
hash ids on every matched chunk (the 4,096-row chunks included) and the
base's sha 927ffd8a; the default twice and the r18 base once, all on a
warm page cache: r19 1.294 / 4.410 / 17.794 and 1.302 / 4.413 / 17.860 s
against r18 1.296 / 4.410 / 17.791 — level, and the sha 927ffd8a on every
leg.

**Session W2 — the r19 binary profiled at 32K** (`serve_prefill_profile_r19/`;
`compare_profiles.py` in the job tmp against the r18 profile, the same
31,955 kernels): stage_wait 1,078 → 21 ms (the prestage does remove the
gather wait, −1.06 s of GPU time at 32K under the tracer), the router's
four-token tile 607 → 859 ms (+253: 2.09 vs 1.48 ms a launch — slower in
situ, reverted), the combine-norm's batched loads 1,216 → 1,208 ms
(neutral, reverted); everything else within ±40 ms. Under the tracer the
r19 burst is 18.6 s of GPU time against r18's 19.4; the probes on a warm
cache are level because the r18 wait is short there (its gathers hit the
page cache): the prestage's value is the cold-table case (the first
requests after a boot, a table displaced by other traffic), measured in
session Z3 with the caches dropped before every boot.

**Session Z3 — the prestage with a cold table** (`serve_r19_prestage_cold/`,
06:12–06:18; the r18 binary against r19b — the chunk-ahead staging alone,
the router and combine-norm forms as shipped, the register-decode kernel
and the bf16 partials opt-in — the page caches dropped before every boot,
the 32K probe twice per leg (the first cold, the second on rows the first
did not touch), the long-prompt sha):

| leg | 32K r0 / r1 | median | sha |
|---|---|---|---|
| r18 base a | 18.505 / 18.341 s | 18.423 | 927ffd8a |
| r19b a | 17.710 / 17.652 s | 17.681 | 927ffd8a |
| r18 base b | 18.640 / 18.410 s | 18.525 | 927ffd8a |
| r19b b | 17.873 / 17.904 s | 17.889 | 927ffd8a |

−0.65..−0.75 s at 32K when the table is cold (−4 %), level when warm
(session Z2), bitwise. SHIPPED: `DGPP_QWEN_PLE_PRESTAGE=0` keeps the
one-channel staging, `DGPP_QWEN_PLE_PRESTAGE_CHECK=1` runs both and
compares. The round's other forms: the four-token router tile (bitwise,
+0.25 s at 32K in situ) and the combine-norm's batched loads (bitwise,
neutral) reverted; the register-decode three-stage expert GEMM kept as
`DGPP_PACKQ_GEMM=wide3` (bitwise, level); the bf16 down partials kept as
`DGPP_MOE_PACKQ_DOWN_BF16=1` (−3..−4 % at every length, not bitwise,
default off). Against the reference on this box (session X) the standing
is now: 2K 1.29 vs 1.12 s, 8K 4.41 vs 4.42, 32K 17.7 (cold 17.7–17.9)
vs 15.42 (its cold-table number, 15.4–15.8).

**Session P — gate and up as one launch** (`serve_r19_pair/`, 06:36–06:45;
the wide kernel's n-tiles past the gate's width take the up projection's
view and output, each token row gathered once for both — bitwise by
construction and in `packq_gemm_test`; `DGPP_MOE_PACKQ_PAIR=1`): the bench
said −17 % (Zipf) / −9 % (uniform) on gate+up, but its second projection
re-reads the first's weights from L2; on the fabric, two launches
1.329 / 4.434 / 17.601 and 1.282 / 4.348 / 17.556 s against one launch
1.267 / 4.379 / 17.659 and 1.294 / 4.382 / 17.677 — level to +1 %, sha
927ffd8a on every leg. Opt-in, default off: the activation re-gather is
not what bounds the kernel either.

**ncu on the shipped wide kernel, device memory** (the Zipf gate launch,
3.70 ms under the profiler): tensor pipe 40 % of peak sustained, memory
throughput 60 % (L2), issue 0.29 a scheduler-cycle with 0.48 eligible of
3.96 warps; the warp cycles: barrier 24 %, wait 15 %, long scoreboard
15 %, math pipe throttle 12 %, mio throttle 11 %, selected 7 %. The
kernel is issue-side bound — two barriers a k-step and the
ldmatrix/mma dependency chain of a 32 x 32 warp tile — not DRAM-bound;
the reference's Marlin reaches ~65 % on the same layer. The next form is
a larger warp tile (more MMAs per ldmatrix and per barrier: 64 x 64 per
warp, or a 128-wide block) with the same per-element chain; the two
accuracy trades (the scale folded into the bf16 fragment, bf16 partials)
remain the user's decision. The reference launcher, probe and breakdown
scripts are copied to `reference_vllm/` beside this file.

## Round 20 — the accuracy trades as deployment keys (2026-09-30, 15:19–)

The user's decision: the non-bitwise prefill levers exist in dgpp, off by
default, on only through a deployment's config (`engine.prefill_bf16_partials`,
`engine.prefill_fold_scales`, `engine.prefill_fp8_gemm`; plan §6.15). New
this round: the folded-scale wide kernel (`packq_gemm` variant 3), the fp8
prefill GEMM pair (`kernels/fp8_gemm`: the per-token 1 x 128 e4m3 quantizer
and a 128 x 128 x 64 mma.sync e4m3 GEMM with per-group fp32 promotion and a
grouped tile order), the keys through the config parser, the worker record
and the serve app, the gates (`packq_gemm_test`: the fold within 2^-7
relative RMS of the exact chain; `fp8_gemm_test`: 1e-7 of the exact sum of
its quantized inputs, 2.5–2.8 % of the dequantized chain), and
`fp8_gemm_bench` (the fp8 GEMM at cuBLASLt's bf16 rate, 86–98 TF: −2..−4 %
at the large dense shapes with the dequant gone, +57 % at [320 x 10240]
where the quantizer's pass over the activation costs more than the GEMM —
the hook skips n < 1,024; cuBLASLt 13.0's block-scale modes return no
kernel on this GPU under any pairing).

**Session R20** (`serve_r20/`, 15:19–15:31; one binary, five configs, the
prefill probe 2K/8K/32K x3 and the 17K-prompt greedy sha per leg; the
first leg cold):

| leg | 2K | 8K | 32K | vs base2 | greedy sha |
|---|---|---|---|---|---|
| base1 (cold) | 1.338 | 4.444 | 17.695 | | 927ffd8a |
| bf16 partials | 1.204 | 4.198 | 17.118 | −6.6 / −3.8 / −2.8 % | 927ffd8a |
| fold scales | 1.276 | 4.450 | 18.235 | −1.0 / +2.0 / +3.6 % | 927ffd8a |
| fp8 GEMM | 1.249 | 4.352 | 17.579 | −3.1 / −0.3 / −0.2 % | 927ffd8a |
| all three | 1.208 | 4.327 | 17.827 | −6.3 / −0.8 / +1.3 % | 927ffd8a |
| base2 | 1.289 | 4.363 | 17.606 | | 927ffd8a |

Each leg's server log names its levers (`prefill levers on ...`). The bf16
partials pay at every length; the fp8 GEMM pays little (its kernel runs at
the bf16 rate, so it saves the dequant launches alone); the fold loses at
32K (its per-code multiply in the decode phase costs more issue slots than
the per-group fma it removes, on a kernel that is issue-bound), and cancels
the other two when combined. Every single-lever leg kept the long prompt's
greedy transcript — the argmax survived the changed arithmetic on this
prompt; that is not a bitwise claim.

**Session R20b** (`serve_r20b/`, 15:31–): the pair that pays — bf16
partials + fp8 GEMM, no fold — around a base leg, with the task evals on
the pair at the baseline's settings (depth 3, greedy, thinking off,
max_tokens 2048, seed 20260908):

| leg | 2K | 8K | 32K | greedy sha |
|---|---|---|---|---|
| pair1 | 1.209 | 4.236 | 17.116 | 857f15c6 (43 tokens) |
| base3 | 1.294 | 4.373 | 17.671 | 927ffd8a |
| pair2 | 1.210 | 4.226 | 17.096 | 857f15c6 |

The pair against the two base legs around it: −6.3 / −3.2 / −3.1 % at
2K / 8K / 32K (1.21 / 4.23 / 17.10 s against 1.29 / 4.37 / 17.64), both
pair legs within 0.1 % of each other.

The pair diverges the long prompt's greedy transcript (the first leg to:
the combined perturbation crossed an argmax margin each lever alone did
not). Evals on the pair: HumanEval 159/164 (366 s), GSM8K 291/300 (990 s),
extraction 100/100 (70 s), two responses at the token cap — against the
default chain's 159 / 292 / 100 (one at cap) and the templates' band
(HumanEval 153–161, GSM8K 291–296): within noise. The AutoRound example
template turns the pair on from this round; the fold stays off everywhere.

**Session R20c** (`serve_r20c/`): the docs/benchmarks.md decode protocol
on the pair config (timed_load C1/C2/C4 x all classes x 3, 256 tokens,
greedy) for the levers-on row's decode cells — the levers touch
prefill-shaped launches only, so the cells are expected level with round
14's — and are (16:01–16:08):

| | prose | code | json | math | chat |
|---|---|---|---|---|---|
| C1 engine tok/s (median of 3) | 48.0 | 59.5 | 71.5 | 64.0 | 45.9 |
| C1 ms/pass, tok/pass | 52.1, 2.50 | 52.9, 3.15 | 53.2, 3.81 | 53.1, 3.40 | 52.9, 2.43 |
| C2 loaded wall tok/s | 55.3 | 80.6 | 89.0 | 82.9 | 59.0 |
| C4 loaded wall tok/s | 78.0 | 107.1 | 116.3 | 104.6 | 82.3 |

The benchmarks-document row for the levers-on template: C1 engine
45.9–71.5, C4 loaded 78.0–116.3, cold prefill 1.210 / 4.226 / 17.096
(the exact chain's row stays 44.7–71.5, 77.3–118.6, 1.294 / 4.417 /
17.817). The full ctest of the tree passed 176/177 with `dsv41_tp_test`
failing on a port collision — its four loopback worlds bound 29968–29971,
the deployment defaults' fabric and journal ports, so it could not start
beside a live deployment; moved to 29980+ and passing alone (3/3).
