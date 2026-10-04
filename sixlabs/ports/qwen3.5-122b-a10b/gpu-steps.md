# Port 4 — Qwen3.5-122B-A10B: what to run once a GPU is free

**Update 2026-10-04 (DGXone): checkpoint B (`Sehyo/Qwen3.5-122B-A10B-NVFP4`) was run through these
steps and is verified. The record is `results-2026-10-04-dgxone.md` beside this file.** What the
run showed that the text below does not say:

- Bind check `expected 149552 | matched 149552`; the gate passed on prose and code in both dense
  forms (recalibrated bounds: pearson ≥ 0.995, mean ≤ 0.10 nat, greedy agreement above 0.5 nat).
- It served with `"kv_capacity": 65536, "prefix_cache_gib": 1`, 8 seats, fp8 dense form, draft head
  on (template `deploy/cluster_qwen3.5-122b-a10b_nvfp4-sehyo_w1.example.json`). Cold load 176 s
  (writes a 67.4 GiB resident image), warm 26 s, 86,941 MiB.
- The planner understates this model: 83.12 GiB planned, about 89.6 GiB of MemAvailable gone at 8
  seats. Beside the reranker, embedder and Docling that left 15 GiB, under the fleet's 19.7 GiB
  floor; 4 seats left 18.3 GiB. The 262,144-token pool plans at 91.0 GiB and was not booted.
- Do not add `prefill_*` keys: the serve binary refuses an explicit prefill budget for `qwen3_5`.
- ShareGPT: 36.5 / 46.9 / 60.0 / 71.2 output tok/s at 1 / 2 / 4 / 8 streams. Checkpoint A
  (`nvidia/…`) is not in the fleet's cache and has not been loaded.

The rest of this file is as it was written before the run.

Status when this was written (2026-10-04, overnight): **code complete on the host side for three
checkpoints, nothing has run on a GPU, and no weight byte of this model has been read** — the
configs and safetensors headers were read from Hugging Face from the Mac (HTTP range requests);
none of these checkpoints is in a DGX cache that I know of. "Expected" below means derived from
those headers and from Mac host runs (Apple clang, a CUDA header shim).

Same model class as port 2 (`Qwen3_5MoeForConditionalGeneration`, `qwen3_5_moe`): hidden 3072, 48
layers (36 GDN + 12 full attention), 256 experts top-8 (intermediate 1024, shared expert 1024), 32
query heads on 2 KV heads, 16 GDN key heads x 64 value heads, one draft layer, a 333-tensor vision
tower (bound past, not served).

| | checkpoint @ revision | headers | container | binds | resident weights (loader formulas, draft layer included) |
|---|---|---|---|---|---|
| A (start here) | `nvidia/Qwen3.5-122B-A10B-NVFP4` @ `98915d83` | 9 shards, 149,309 tensors, 77.7 GiB | modelopt `NVFP4`, one group + an ignore list: the routed experts alone are NVFP4; GDN, attention, shared expert, router, head BF16; BF16 draft layer with per-expert matrices | table 148,976 == 149,309 − 333 vision | **70.33 GiB** with `dense_weights: "fp8"` (the template), 74.66 GiB as shipped |
| B | `Sehyo/Qwen3.5-122B-A10B-NVFP4` @ `56a6bdda` | 3 shards, 149,885 tensors, 75.9 GiB | compressed-tensors `nvfp4-pack-quantized`: attention q/k/v/o, shared expert and routed experts NVFP4; GDN, router, head BF16; the same BF16 draft layer | table 149,552 == 149,885 − 333 | the same to within kilobytes (its NVFP4 attention and shared expert are dequantized to BF16 on the host at load) |
| C | `Qwen/Qwen3.5-122B-A10B-FP8` @ `a099dee7` | 39 shards, 76,656 tensors, 118.4 GiB | block FP8 throughout (experts and draft layer included); A_log and the GDN norm weight stored F32 | table 76,323 == 76,656 − 333 | **117.59 GiB — does not fit a 121 GiB Spark with a K/V pool.** Bind-checkable only; needs world 2 |
| — | `Intel/Qwen3.5-122B-A10B-int4-AutoRound` | 15 shards, 120,567 tensors | AutoRound GPTQ int4 on the experts and on every GDN and attention projection | refused by name at config parse | — |

**Does A fit in 121 GiB with a KV pool?** By the loader's own byte formulas: 70.33 GiB of weights.
The K/V pool of the template (262,144 tokens, BF16): 12 attention layers x 2 KV heads x 256 x (K+V)
x 2 bytes = 24,576 bytes a token = 6.0 GiB. GDN recurrent state: 36 layers x 64 heads x 128 x 128
fp32 = 151 MB a request slot, 1.2 GiB for eight. With the 4 GiB prefix cache that is about 82 GiB
before scratch — it fits, with room. The plan line of step 3 is the authority; I could not run it.

- Patch: `/Users/markgriffith/6ix-ports-overnight/4-qwen3.5-122b-a10b.patch` — port 4 only,
  against `main` at `a558c9f` (ports 1–3 are there).
- Inputs: `/Users/markgriffith/6ix-ports-overnight/inputs/qwen35-122b-ids-code.json` (113 tokens),
  `qwen35-122b-ids-prose.json` (87 tokens) — made on the Mac with the engine's tokenizer and this
  model's own chat template; byte-identical to port 2's `qwen36-ids-*.json` (same tokenizer, and
  the two templates render these messages alike). Copy to the box as `$IN/`.

What the patch adds beyond the 122B (the family-coverage fixes; table in
`5-qwen35-family-coverage.md`): the GDN's `A_log` / `norm.weight` bind in BF16 or F32; a tied
config whose checkpoint also stores `lm_head.weight` binds; a config naming a draft layer the
checkpoint lacks gets one clear line; AutoRound is refused by name on this dialect.

`$REPO` = a build copy with the patch applied, `$OUT` = scratch, `$SNAP` = the snapshot directory.

## 0. Get the checkpoint, apply, build

```bash
huggingface-cli download nvidia/Qwen3.5-122B-A10B-NVFP4      # 83.5 GB on disk: check free space first
cd $REPO && git apply --check /path/to/4-qwen3.5-122b-a10b.patch && git apply /path/to/4-qwen3.5-122b-a10b.patch
cmake --preset ci
cmake --build build-ci -j 8 --target unit_tests qwen35_ports_loader_test qwen3next_loader_test qwen35_bind_check
cmake --preset release && cmake --build --preset release -j 8
cmake --build build-release -j 8 --target qwen35_forward_check qwen35_bind_check
```

Touched translation units: `src/models/qwen/{config35,binding35,loader35}.cpp` (no kernel and no
`model35.cpp` change in this port). Likeliest build failure: a GCC `-Werror` diagnostic Apple clang
did not raise.

## 1. Host tests and the loader fixture

```bash
cd $REPO
DGPP_TEST_FILTER=qwen35_122b ./build-ci/unit_tests  # expect 6 tests, 0 failed
DGPP_TEST_FILTER=qwen3 ./build-ci/unit_tests        # everything on this stack: 0 failed (54 on the Mac)
cd $OUT && $REPO/build-ci/qwen35_ports_loader_test  # expect 9 tests, 0 failed; new: qwen35_nvfp4_experts_*, qwen35_nvfp4_packed_*, qwen35_fp8_loader_reads_f32_gdn_vectors
cd $OUT && $REPO/build-ci/qwen3next_loader_test     # 3 tests, 0 failed (regression: the shared builders changed)
```

## 2. Bind check (headers only — safe to run before the download finishes verifying)

```bash
$REPO/build-ci/qwen35_bind_check --model nvidia/Qwen3.5-122B-A10B-NVFP4
```

Expected:

```
config: qwen3_5, 48 layers (36 GDN + 12 full attention) + 1 draft, hidden 3072, vocab 248320
mlp: routed MoE, 256 experts top-8, intermediate 1024, shared expert 1024
checkpoint: 9 shards, 149309 tensors in headers
binding: expected 148976 | matched 148976 (missing 0, dtype 0, shape 0) | unexpected 0 | out of scope 0 | vision skipped 333
quantized matrices: 36864
binding OK: every tensor of the table is present with its dtype and shape
```

B: `checkpoint: 3 shards, 149885 tensors in headers`, `expected 149552 | matched 149552`,
`quantized matrices: 37056`. C: `39 shards, 76656 tensors`, `expected 76323 | matched 76323`,
`quantized matrices: 37939`. (All three produced on the Mac by the same validator over the real
`config.json` and the real headers.)

## 3. Memory plan (loads nothing)

```bash
$REPO/build-release/qwen35_forward_check --model nvidia/Qwen3.5-122B-A10B-NVFP4 --plan 8,262144 --dense-weights fp8
$REPO/build-release/qwen35_forward_check --model nvidia/Qwen3.5-122B-A10B-NVFP4 --plan 8,262144
```

Expected "model weights (resident)": ≈ **70.33 GiB** (fp8), ≈ 74.66 GiB (as shipped). **Read the
whole plan before loading anything**: this is the first model on this stack whose weights are more
than half the box. If the plan's total is not comfortably under the box's free memory, stop here.

## 4. The numerical gate: forward check against the numpy reference

`tools/qwen3next_reference.py` reads both NVFP4 containers on this dialect (on synthetic
checkpoints its weights agree bit for bit with what the C++ loader holds). **Memory: it converts
weights lazily and caches them — on a 122B model a full-depth run can hold tens of GB. Do not run
it on the box that is serving, and use `--layers` to stage it:**

```bash
cd $REPO
python3 tools/qwen3next_reference.py score --ckpt $SNAP --ids-json $IN/qwen35-122b-ids-code.json --layers 4 --out $OUT/q122-ref-code-L4.json --threads 8
python3 tools/qwen3next_reference.py dump  --ckpt $SNAP --ids-json $IN/qwen35-122b-ids-code.json --layers 4 --out $OUT/q122-refdump-code-L4 --detail-layers 0,3 --threads 8
$REPO/build-release/qwen35_forward_check --model nvidia/Qwen3.5-122B-A10B-NVFP4 --layers 4 \
    --ids-json $IN/qwen35-122b-ids-code.json --dump-states $OUT/q122-code-L4.bf16 > $OUT/q122-code-L4.txt
python3 sixlabs/bench/compare_states.py $OUT/q122-code-L4.bf16 $OUT/q122-refdump-code-L4 113 3072
```

Four layers cover one of each kind (three GDN, one attention) and every new shape; the states
must agree to a small relative error with cosines ≈ 1. Then full depth, when a box has the memory
for the reference (watch `peak_rss_gb` in its last line):

```bash
for k in code prose; do
  python3 tools/qwen3next_reference.py score --ckpt $SNAP --ids-json $IN/qwen35-122b-ids-$k.json --out $OUT/q122-ref-$k.json --threads 8
  $REPO/build-release/qwen35_forward_check --model nvidia/Qwen3.5-122B-A10B-NVFP4 \
      --ids-json $IN/qwen35-122b-ids-$k.json > $OUT/q122-$k.txt
  python3 sixlabs/bench/compare_forward_check.py $OUT/q122-$k.txt $OUT/q122-ref-$k.json
done
$REPO/build-release/qwen35_forward_check --model nvidia/Qwen3.5-122B-A10B-NVFP4 --dense-weights fp8 \
    --ids-json $IN/qwen35-122b-ids-code.json > $OUT/q122-code-fp8.txt
python3 sixlabs/bench/compare_forward_check.py $OUT/q122-code-fp8.txt $OUT/q122-ref-code.json
```

Pass: `PASS` (mean ≤ 0.06 nat, worst position ≤ 0.5). Is the reference itself right? Its
`mean_logprob` must be a small negative number (about −12.4 = ln 248320 means a container
convention is wrong), and `--variant ws2=div` / `--variant nibble=hi` must each collapse it.

Reading a failure, most likely first:
- **layer 3 (the first attention layer), prefill only**: 32 query heads on 2 KV heads is 16 query
  heads a KV head — the attention kernel's limit, and above its tile kernel's group of 8, so
  `full_attn_prefill` takes the row form (`src/kernels/full_attn.cu`, `group > kMaxTileGroup`).
  No model served so far runs that branch. It is the decode kernel, so correctness is likely, but
  a long prompt will prefill slowly through these 12 layers; measure before judging throughput.
- layer 0: the GDN at 64 value heads on 16 key heads (ratio 4; the 80B and the 35B run 2, the 27B
  3) and H = 3072; under `--dense-weights fp8` its BF16 projections are encoded to block FP8 at
  load (q/k/v gather rows are 128-aligned: 16 x 128, 16 x 128, 64 x 128).
- the MoE: experts as the 80B's modelopt set (the same kernels), I = 1024 (the 80B and 35B run
  512), shared expert BF16 through the dense path.
- states fine, log-probabilities off: the BF16 head (`--dense-weights fp8` re-encodes it).

## 5. First boot, the draft layer, first requests

```bash
cd $REPO
cp deploy/cluster_qwen3.5-122b-a10b_nvfp4_w1.example.json deploy/cluster_qwen3.5-122b-a10b_nvfp4_w1.json
python3 scripts/dgpp-cluster doctor --config deploy/cluster_qwen3.5-122b-a10b_nvfp4_w1.json
python3 scripts/dgpp-cluster up     --config deploy/cluster_qwen3.5-122b-a10b_nvfp4_w1.json
```

Log: `serve: model family qwen3_5`, a plan whose weights are ≈ 70 GiB, an `mtp scratch` item,
`READY`. The first load encodes about 14 GiB of BF16 matrices to FP8 (the GDN, the attention, the
shared experts, the draft layer's experts) and writes the resident image:
expect minutes, not seconds. Alias `Qwen3.5-122B-A10B`; thinking is **on** unless the request sends
`"enable_thinking": false` (the template opens `<think>`; the tool-call form is the XML
`<function=…>` one, probed on the Mac over the real tokenizer and template):

```bash
URL=http://127.0.0.1:8000/v1/chat/completions
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3.5-122B-A10B","temperature":0,"max_tokens":256,
  "messages":[{"role":"user","content":"What is 17 * 23? Answer with the number."}]}'
# expect: message.reasoning_content non-empty, message.content containing 391, finish_reason "stop"
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3.5-122B-A10B","temperature":0,"max_tokens":64,
  "chat_template_kwargs":{"enable_thinking":false},
  "messages":[{"role":"user","content":"What is 17 * 23? Answer with the number."}]}'
# expect: no reasoning_content, content 391
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3.5-122B-A10B","temperature":0,"max_tokens":256,
  "messages":[{"role":"user","content":"What is the weather in Paris?"}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice":"required"}'
# expect: tool_calls[0].function == {"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}, finish_reason "tool_calls"
```

The draft layer (a wrong draft is never visible in the output, only in acceptance):
1. Greedy transcripts with MTP equal plain decode (`--knobs "--no-mtp"`, `sixlabs/refset/capture_refset.py`,
   `sixlabs/bench/compare_transcripts.py`).
2. `curl -s http://127.0.0.1:8000/v1/metrics | python3 -m json.tool` → `scheduler.spec_decode`
   accepted / attempted well above half on code. The draft layer here is BF16 with **per-expert**
   matrices (`mtp.layers.0.mlp.experts.E.{gate,up,down}_proj.weight`, the 80B's layout — not the
   35B nvidia release's two stacked tensors), each encoded to block FP8 at load.
   `python3 tools/qwen3next_reference.py mtp --ckpt $SNAP --ids-json $IN/qwen35-122b-ids-code.json`
   prints the rate to expect (the engine implements `post_norm/history`).

Checkpoint B serves under the same template with `"model": "Sehyo/Qwen3.5-122B-A10B-NVFP4"`; run
§2–§4 for it first (its attention and shared expert go through the compressed-tensors dequant,
`--variant gscale=mul` must collapse the reference). A and B are the same model in two
quantizations: their reference log-probabilities for the same ids should agree to a few hundredths
of a nat on average, which checks both containers against each other without the engine.

## 6. Still open after a pass

- World > 1 (the only way checkpoint C can be served); interleaved prefill (gated to the
  `qwen3_next` family name in `apps/dgpp_serve.cpp`); `engine.rope_scaling` (refused for this
  dialect); `engine.prefill_fp8_per_tensor` (refused for routed-MoE checkpoints).
- `kv_dtype: "fp8"`: nvidia's recipe names an FP8 K/V-cache scheme but stores no `k_scale` /
  `v_scale` tensors (its attention is on the ignore list), so there are no static cache scales
  to read; the template keeps the BF16 pool.
- The source-byte accounting of the two GDN float vectors is at the table's BF16 width; checkpoint
  C stores them F32, so its "bytes read" figure is low by 384 bytes a GDN layer (documented at
  `load_float_vector_f32` in `src/models/qwen/loader35.cpp`).
- The vision tower; a jinja2 golden corpus for this template; measured throughput.
