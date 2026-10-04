# Port 2 — Qwen3.6-35B-A3B: what to run once a GPU is free

Status when this was written (2026-10-04, overnight): **code complete on the host side for two
checkpoints, nothing has run on a GPU.** "Expected" below means derived from the checkpoints'
headers and from Mac host runs (Apple clang, a CUDA header shim), not from a run on a Spark.

| | checkpoint | snapshot (DGXone `/home/mark/.cache/huggingface/hub/...`) | container |
|---|---|---|---|
| A (start here) | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5` (`refs/main` points here; the other snapshot dir `491c2f1e…` has no weights) | modelopt `MIXED_PRECISION`: NVFP4 experts / shared expert / lm_head, per-tensor FP8 GDN + attention projections, BF16 draft layer with stacked experts |
| B | `Qwen/Qwen3.6-35B-A3B-FP8` | `models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989` | block FP8 (128x128, BF16 `weight_scale_inv`) on everything incl. experts and the draft layer; BF16 head |

Not ported: `RedHatAI/Qwen3.6-35B-A3B-NVFP4` (compressed-tensors; the cache holds only
`model_mtp.safetensors` and `model_visual.safetensors`, 352 tensors — an incomplete download) and
`unsloth/…-NVFP4` (config only). The engine refuses the RedHat config by name
(`group_1.weights missing`).

- Patch: `/Users/markgriffith/6ix-ports-overnight/2-qwen3.6-35b-a3b.patch` (cumulative: contains
  port 1; against `main` at `07f4f32`).
- Inputs (made on the Mac with the engine's tokenizer + this model's template):
  `/Users/markgriffith/6ix-ports-overnight/inputs/qwen36-ids-code.json` (113 tokens),
  `qwen36-ids-prose.json` (87 tokens). Copy to the box as `$IN/`.

`$REPO` = a build copy with the patch applied, `$OUT` = scratch, `$SNAP_A` / `$SNAP_B` = the two
snapshot directories.

## 0. Apply and build

```bash
cd $REPO && git apply --check /path/to/2-qwen3.6-35b-a3b.patch && git apply /path/to/2-qwen3.6-35b-a3b.patch
cmake --preset ci
cmake --build build-ci -j 8 --target unit_tests qwen35_ports_loader_test qwen3next_loader_test \
      qwen35_bind_check qwen35_forward_check chat_template_test
cmake --preset release && cmake --build --preset release -j 8
cmake --build build-release -j 8 --target qwen35_forward_check qwen35_bind_check
```

## 1. Host tests and the loader fixture

```bash
cd $REPO
DGPP_TEST_FILTER=qwen36 ./build-ci/unit_tests       # expect 9 tests, 0 failed
DGPP_TEST_FILTER=qwen35 ./build-ci/unit_tests       # the 27B's config/binding tests: all OK (regression)
DGPP_TEST_FILTER=qwen3 ./build-ci/unit_tests        # everything on this stack: 0 failed
cd $OUT && $REPO/build-ci/qwen35_ports_loader_test  # expect 5 tests, 0 failed (3 Coder-Next + qwen36_nvfp4_mixed_* + qwen36_fp8_*)
```

## 2. Bind checks (headers only)

```bash
$REPO/build-ci/qwen35_bind_check --model nvidia/Qwen3.6-35B-A3B-NVFP4
$REPO/build-ci/qwen35_bind_check --model Qwen/Qwen3.6-35B-A3B-FP8
```

Expected for A:

```
config: qwen3_5, 40 layers (30 GDN + 10 full attention) + 1 draft, hidden 2048, vocab 248320
mlp: routed MoE, 256 experts top-8, intermediate 512, shared expert 512
checkpoint: 3 shards, 124468 tensors in headers
binding: expected 124135 | matched 124135 (missing 0, dtype 0, shape 0) | unexpected 0 | out of scope 0 | vision skipped 333
quantized matrices: 30971
binding OK: every tensor of the table is present with its dtype and shape
```

Expected for B: `checkpoint: 42 shards, 64196 tensors in headers`, `binding: expected 63863 |
matched 63863 … vision skipped 333`, `quantized matrices: 31745`, `binding OK`.

(Both sets of numbers were produced on the Mac by the same validator over the real `config.json`
and a dump of the real headers.)

## 3. Memory plan (loads nothing)

```bash
$REPO/build-release/qwen35_forward_check --model nvidia/Qwen3.6-35B-A3B-NVFP4 --plan 8,262144 --dense-weights fp8
$REPO/build-release/qwen35_forward_check --model Qwen/Qwen3.6-35B-A3B-FP8     --plan 8,262144 --dense-weights fp8
```

Expected "model weights (resident)" (the loader's byte formulas, draft layer included): A ≈ **20.93
GiB** (21.08 without `--dense-weights fp8`), B ≈ **34.06 GiB**. Both say `mtp on`.

## 4. The numerical gate: forward check against the numpy reference

The reference (`tools/qwen3next_reference.py`, extended to this dialect; it has run only on
synthetic checkpoints, where its weights agree bit for bit with what the C++ loader holds). CPU
only, mmaps the shards, keeps a few GB of converted weights (the head alone is 2 GB in float32):
**not on a box near its memory limit.**

```bash
cd $REPO
for S in A B; do eval SNAP=\$SNAP_$S
  for k in code prose; do
    python3 tools/qwen3next_reference.py score --ckpt $SNAP --ids-json $IN/qwen36-ids-$k.json --out $OUT/q36$S-ref-$k.json --threads 8
    python3 tools/qwen3next_reference.py dump  --ckpt $SNAP --ids-json $IN/qwen36-ids-$k.json --out $OUT/q36$S-refdump-$k --detail-layers 0,3 --threads 8
  done
done
```

Is the reference itself right? Three checks that need no engine:
1. `mean_logprob` on these natural texts is a small negative number (a working model predicts the
   canned answers well); about −12.4 (ln 248320) means a container convention is wrong.
2. A and B are the same model in two quantizations: their reference log-probabilities for the same
   ids must agree closely (compare the two JSONs' `logprobs`: mean |difference| of a few
   hundredths of a nat, same top-1 almost everywhere). A large disagreement localizes the bug to
   one container.
3. The convention switches must each collapse the result:
   `--variant fp8scale=div` (A and B: FP8 scales multiply), `--variant ws2=div` (A: NVFP4
   multiplier), `--variant nibble=hi`.

Then the engine (streaming residency, exact form first):

```bash
for M in nvidia/Qwen3.6-35B-A3B-NVFP4:A Qwen/Qwen3.6-35B-A3B-FP8:B; do S=${M##*:}; MODEL=${M%%:*}
  for k in code prose; do
    $REPO/build-release/qwen35_forward_check --model $MODEL --ids-json $IN/qwen36-ids-$k.json \
        --dump-states $OUT/q36$S-$k.bf16 > $OUT/q36$S-$k.txt
    python3 sixlabs/bench/compare_forward_check.py $OUT/q36$S-$k.txt $OUT/q36$S-ref-$k.json
    python3 sixlabs/bench/compare_states.py $OUT/q36$S-$k.bf16 $OUT/q36$S-refdump-$k \
        $(python3 -c "import json;print(len(json.load(open('$IN/qwen36-ids-$k.json'))))") 2048
  done
done
# and the serving form of A (shared expert, draft layer and head re-encoded to block FP8):
$REPO/build-release/qwen35_forward_check --model nvidia/Qwen3.6-35B-A3B-NVFP4 --dense-weights fp8 \
    --ids-json $IN/qwen36-ids-code.json > $OUT/q36A-code-fp8.txt
python3 sixlabs/bench/compare_forward_check.py $OUT/q36A-code-fp8.txt $OUT/q36A-ref-code.json
```

Pass: `compare_forward_check.py` prints `PASS` (mean ≤ 0.06 nat, worst position ≤ 0.5, greedy
token equal except at near ties); `compare_states.py` shows a smooth, slowly growing relative
error and cosines ≈ 1. Reading a failure:
- a jump at layer 0 (a GDN layer): the per-tensor FP8 projections (A) or the MoE; bisect with
  `--layers 1` on both tools and the reference's `L00_*` sublayer dumps (`mixer_out` = the GDN,
  `moe_out`, `router_ids`);
- fine through layer 2, a jump at layer 3: the attention projections;
- states fine, logprobs off: the head (A: the NVFP4 head is dequantized on the host at load; with
  `--dense-weights fp8` it is then re-encoded to block FP8 — if only that form fails, serve with
  `"dense_weights": "checkpoint"`).

The engine's FP8 kernels use each weight as bf16(e4m3 code × scale); the reference keeps the fp32
product. That is the stack's existing convention (the 27B's) and part of the tolerance.

## 5. First boot, the draft layer, first requests

```bash
cd $REPO
cp deploy/cluster_qwen3.6-35b-a3b_nvfp4_w1.example.json deploy/cluster_qwen3.6-35b-a3b_nvfp4_w1.json
python3 scripts/dgpp-cluster doctor --config deploy/cluster_qwen3.6-35b-a3b_nvfp4_w1.json
python3 scripts/dgpp-cluster up     --config deploy/cluster_qwen3.6-35b-a3b_nvfp4_w1.json
```

Log: `serve: model family qwen3_5`, a plan whose weights are ≈ 21 GiB, an `mtp scratch` item,
`READY`. Then (alias `Qwen3.6-35B-A3B`; this is a thinking model: `reasoning_content` carries the
`<think>` block unless `"enable_thinking": false` is sent):

```bash
URL=http://127.0.0.1:8000/v1/chat/completions
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3.6-35B-A3B","temperature":0,"max_tokens":256,
  "messages":[{"role":"user","content":"What is 17 * 23? Answer with the number."}]}'
# expect: message.reasoning_content non-empty, message.content containing 391, finish_reason "stop"
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3.6-35B-A3B","temperature":0,"max_tokens":64,
  "chat_template_kwargs":{"enable_thinking":false},
  "messages":[{"role":"user","content":"What is 17 * 23? Answer with the number."}]}'
# expect: no reasoning_content, content 391
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3.6-35B-A3B","temperature":0,"max_tokens":256,
  "messages":[{"role":"user","content":"What is the weather in Paris?"}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice":"required"}'
# expect: tool_calls[0].function == {"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}, finish_reason "tool_calls"
# an image_url part must be refused (text only), not crash.
```

The draft layer (MTP) needs its own check, because a wrong draft never changes the output — the
verify pass is exact — it only stops being accepted:
1. Greedy transcripts with MTP must equal plain decode: restart with `--knobs "--no-mtp"`, replay
   the same greedy prompts (`sixlabs/refset/capture_refset.py`), compare with
   `sixlabs/bench/compare_transcripts.py` — identical, as on the 80B (18/18).
2. Acceptance: `curl -s http://127.0.0.1:8000/v1/metrics | python3 -m json.tool` (the server's own port) →
   `scheduler.spec_decode`: accepted / attempted per position well above half on code. Near zero
   means the draft layer is mis-bound. For checkpoint A the suspects are the stacked draft experts
   (`mtp.layers.0.mlp.experts.gate_up_proj`: the loader takes an expert's first I rows as gate,
   the next I as up — transformers' `chunk(2)` order — and encodes each slice to block FP8) and
   the BF16 draft attention. The reference measures the same thing on the CPU:
   `python3 tools/qwen3next_reference.py mtp --ckpt $SNAP_A --ids-json $IN/qwen36-ids-code.json`
   prints how often the draft equals the main model's next token under the four conventions; the
   engine implements `post_norm/history`. `--variant stacked=up_gate` must make that rate
   collapse; if it *raises* it, the gate/up order assumed here is wrong (fix `build_moe35` in
   `src/models/qwen/loader35.cpp` and the binding comment).
3. Checkpoint B's draft layer is FP8 per expert like its backbone: same two checks, no stacking
   question.

Then repeat §5 for B with `deploy/cluster_qwen3.6-35b-a3b_fp8_w1.example.json`.

## 6. Still open after a pass

- World > 1; interleaved prefill (`engine.prefill_budget_tokens` is gated to the `qwen3_next`
  family name in `apps/dgpp_serve.cpp`; this model serves under `qwen3_5`, so the templates leave
  it out — lifting the gate is one condition, and needs the resumable-prefill checks first).
- `engine.rope_scaling` (YaRN) is refused for this dialect; `engine.prefill_fp8_per_tensor` is
  refused for any routed-MoE checkpoint.
- The vision tower (not served), a jinja2 golden corpus for this template (it loads and renders in
  the engine's interpreter; the parser and the forced-call grammar were probed on the Mac over the
  real tokenizer with a hand-written reasoning + call turn), measured throughput.
