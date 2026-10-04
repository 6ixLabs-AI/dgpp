# Port 3 — Qwen3.5-0.8B: what to run once a GPU is free

Status when this was written (2026-10-04, overnight): **code complete on the host side, nothing has
run on a GPU.** "Expected" below means derived from the checkpoint's headers and from Mac host runs
(Apple clang, a CUDA header shim), not from a run on a Spark.

| checkpoint | snapshot | container |
|---|---|---|
| `Qwen/Qwen3.5-0.8B` | `models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17` (one shard, `model.safetensors-00001-of-00001.safetensors`, 488 tensors) | no `quantization_config`: every matrix BF16; `tie_word_embeddings: true` (no stored `lm_head.weight`); the GDN's `A_log` and `norm.weight` are F32; a one-layer dense draft (`mtp.*`, BF16); a 153-tensor vision tower (bound past, not served) |

What is new in the engine for it (everything else is the 27B's dense walk):

- `Qwen35QuantKind::Bf16` (a config with no `quantization_config`), tied embeddings on the Qwen3.5
  dialect (the head pointer is the resident embedding), F32 `A_log` / F32 norm weight read as such;
- the dense MLP bound in BF16 runs through the BF16 GEMM interface (`Qwen35Model::dense_mlp`, the
  branch on `m.gate != nullptr`) — **new GPU-side code, never executed**;
- `engine.dense_weights: "fp8"` instead encodes every dense matrix and the head to block FP8 at
  load, which puts the model on exactly the 27B's kernels (the fallback if the BF16 branch fails);
- the chat template's `is sequence` test in the engine's template interpreter.

- Patch: `/Users/markgriffith/6ix-ports-overnight/3-qwen3.5-0.8b.patch` — port 3 only, against
  `main` at `89955e0` (ports 1 and 2 are already there).
- Inputs (made on the Mac with the engine's tokenizer + this model's template):
  `/Users/markgriffith/6ix-ports-overnight/inputs/qwen35-0.8b-ids-code.json` (113 tokens),
  `qwen35-0.8b-ids-prose.json` (87 tokens). Copy to the box as `$IN/`.

`$REPO` = a build copy with the patch applied, `$OUT` = scratch, `$SNAP` = the snapshot directory.
The checkpoint is 1.7 GB on disk; fetch it with the site's usual downloader if it is not in the
cache (`huggingface-cli download Qwen/Qwen3.5-0.8B`).

## 0. Apply and build

```bash
cd $REPO && git apply --check /path/to/3-qwen3.5-0.8b.patch && git apply /path/to/3-qwen3.5-0.8b.patch
cmake --preset ci
cmake --build build-ci -j 8 --target unit_tests qwen35_ports_loader_test qwen3next_loader_test \
      qwen35_bind_check chat_template_test
cmake --preset release && cmake --build --preset release -j 8
cmake --build build-release -j 8 --target qwen35_forward_check qwen35_bind_check
```

Most likely first build failure: a GCC `-Werror` diagnostic Apple clang did not raise (the Mac
checks used `-Wall -Wextra -Wpedantic`, not GCC's set). The touched translation units are
`src/models/qwen/{config35,binding35,loader35,model35}.cpp` and `src/text/chat_template.cpp`.

## 1. Host tests and the loader fixture

```bash
cd $REPO
DGPP_TEST_FILTER=qwen35_bf16 ./build-ci/unit_tests  # expect 4 tests, 0 failed
DGPP_TEST_FILTER=qwen3 ./build-ci/unit_tests        # everything on this stack: 0 failed (48 on the Mac)
./build-ci/chat_template_test                       # the template interpreter's own suite: 0 failed
cd $OUT && $REPO/build-ci/qwen35_ports_loader_test  # expect 6 tests, 0 failed (the new one: qwen35_bf16_loader_resident_values_are_the_checkpoints)
cd $OUT && $REPO/build-ci/qwen3next_loader_test     # 3 tests, 0 failed (regression)
```

The loader fixture test is the first real exercise of the copy pass with CUDA's allocator; on the
Mac it ran against a malloc-backed stand-in.

## 2. Bind check (headers only)

```bash
$REPO/build-ci/qwen35_bind_check --model Qwen/Qwen3.5-0.8B
```

Expected:

```
config: qwen3_5, 24 layers (18 GDN + 6 full attention) + 1 draft, hidden 1024, vocab 248320
mlp: dense SwiGLU, intermediate 3584
checkpoint: 1 shards, 488 tensors in headers
binding: expected 335 | matched 335 (missing 0, dtype 0, shape 0) | unexpected 0 | out of scope 0 | vision skipped 153
quantized matrices: 0
binding OK: every tensor of the table is present with its dtype and shape
```

(The numbers were produced on the Mac by the same validator over the real `config.json` and a dump
of the real header.)

## 3. Memory plan (loads nothing)

```bash
$REPO/build-release/qwen35_forward_check --model Qwen/Qwen3.5-0.8B --plan 8,262144
$REPO/build-release/qwen35_forward_check --model Qwen/Qwen3.5-0.8B --plan 8,262144 --dense-weights fp8
```

Expected "model weights (resident)", draft layer included: ≈ **1.44 GiB** as shipped, ≈ **0.96
GiB** under `--dense-weights fp8` (0.48 GiB of either is the embedding, which is also the head).

## 4. The numerical gate: forward check against the numpy reference

`tools/qwen3next_reference.py` reads this dialect and container (tied head, F32 vectors, dense
MLP). It has run only on synthetic checkpoints, where its weights agree bit for bit with what the
C++ loader holds. The model is small: the reference takes seconds per prompt on any CPU.

```bash
cd $REPO
for k in code prose; do
  python3 tools/qwen3next_reference.py score --ckpt $SNAP --ids-json $IN/qwen35-0.8b-ids-$k.json --out $OUT/q08-ref-$k.json --threads 8
  python3 tools/qwen3next_reference.py dump  --ckpt $SNAP --ids-json $IN/qwen35-0.8b-ids-$k.json --out $OUT/q08-refdump-$k --detail-layers 0,3 --threads 8
done
```

Is the reference itself right? `mean_logprob` on these texts must be a modest negative number (a
0.8B model predicts the canned answers less well than the big ones, but far better than chance);
about −12.4 (ln 248320) means a layout assumption is wrong. Because this checkpoint is plain BF16
it is also the one model here that `transformers` itself can score on a CPU in a minute, given a
build that has the `Qwen3_5ForConditionalGeneration` class (the config was written by
`4.57.0.dev0`; I have not checked which release carries it): its log-probabilities for the same
ids are the independent oracle the quantized ports do not have, and worth the ten minutes — they
would validate the reference's Qwen3.5 dialect for port 2 as well.

Then the engine, the shipped form first (the new BF16 MLP branch), then the FP8 form:

```bash
for k in code prose; do
  $REPO/build-release/qwen35_forward_check --model Qwen/Qwen3.5-0.8B --ids-json $IN/qwen35-0.8b-ids-$k.json \
      --dump-states $OUT/q08-$k.bf16 > $OUT/q08-$k.txt
  python3 sixlabs/bench/compare_forward_check.py $OUT/q08-$k.txt $OUT/q08-ref-$k.json
  python3 sixlabs/bench/compare_states.py $OUT/q08-$k.bf16 $OUT/q08-refdump-$k \
      $(python3 -c "import json;print(len(json.load(open('$IN/qwen35-0.8b-ids-$k.json'))))") 1024
done
$REPO/build-release/qwen35_forward_check --model Qwen/Qwen3.5-0.8B --dense-weights fp8 \
    --ids-json $IN/qwen35-0.8b-ids-code.json > $OUT/q08-code-fp8.txt
python3 sixlabs/bench/compare_forward_check.py $OUT/q08-code-fp8.txt $OUT/q08-ref-code.json
```

Pass: `compare_forward_check.py` prints `PASS` (mean ≤ 0.06 nat, worst position ≤ 0.5);
`compare_states.py` shows a small, slowly growing relative error and cosines ≈ 1. The shipped form
has no quantization anywhere, so its error should be the smallest of the three ports'; the FP8
form adds the 27B's quantization error (a 0.8B model is more sensitive to it than a 27B — if the
FP8 form's mean is above tolerance while the shipped form passes, that is a property of
re-encoding a small model, not a bug: serve the shipped form, as the template does).

Reading a failure:
- the shipped form fails at layer 0 and the FP8 form passes: the BF16 MLP branch in
  `Qwen35Model::dense_mlp` (argument order of `gw_.gemm->matmul`, scratch sizes at I = 3584);
- both fail at layer 0: the GDN at this shape — 16 key heads and 16 value heads (ratio 1; the 27B
  and the MoE models run ratio 2 or 3), F32 `A_log` (the loader keeps it fp32; transformers holds
  the parameter in the module's dtype — a rounding-level difference only), F32 norm weight rounded
  to BF16 at load. Bisect with `--layers 1` and the reference's `L00_*` dumps;
- fine through layer 2, off at layer 3: attention at 8 heads / 2 KV heads / head_dim 256,
  H = 1024;
- states fine, log-probabilities off: the tied head (`build_globals` in
  `src/models/qwen/loader35.cpp` points the head at the resident embedding; under
  `--dense-weights fp8` the head is an FP8 encode of the embedding).

## 5. First boot, the draft layer, first requests

```bash
cd $REPO
cp deploy/cluster_qwen3.5-0.8b_bf16_w1.example.json deploy/cluster_qwen3.5-0.8b_bf16_w1.json
python3 scripts/dgpp-cluster doctor --config deploy/cluster_qwen3.5-0.8b_bf16_w1.json
python3 scripts/dgpp-cluster up     --config deploy/cluster_qwen3.5-0.8b_bf16_w1.json
```

Log: `serve: model family qwen3_5`, a plan whose weights are ≈ 1.4 GiB, an `mtp scratch` item,
`READY`. Alias `Qwen3.5-0.8B`. This template's thinking is **off** unless the request turns it on
(the opposite default from the 35B's):

```bash
URL=http://127.0.0.1:8000/v1/chat/completions
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3.5-0.8B","temperature":0,"max_tokens":64,
  "messages":[{"role":"user","content":"What is the capital of France? One word."}]}'
# expect: no reasoning_content, content "Paris", finish_reason "stop"
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3.5-0.8B","temperature":0,"max_tokens":512,
  "chat_template_kwargs":{"enable_thinking":true},
  "messages":[{"role":"user","content":"What is 17 * 23? Answer with the number."}]}'
# expect: message.reasoning_content non-empty; content with a number (a 0.8B model may get 391 wrong — that is not an engine fault)
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3.5-0.8B","temperature":0,"max_tokens":256,
  "messages":[{"role":"user","content":"What is the weather in Paris?"}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice":"required"}'
# expect: tool_calls[0].function.name == "get_weather" with JSON arguments naming Paris, finish_reason "tool_calls"
# an image_url part must be refused (text only), not crash.
```

The draft layer (a wrong draft never changes the output, it only stops being accepted):
1. Greedy transcripts with MTP equal plain decode: restart with `--knobs "--no-mtp"`, replay the
   same greedy prompts (`sixlabs/refset/capture_refset.py`), compare with
   `sixlabs/bench/compare_transcripts.py` — identical.
2. Acceptance: `curl -s http://127.0.0.1:8000/v1/metrics | python3 -m json.tool` →
   `scheduler.spec_decode` accepted / attempted. The reference prints what to expect on the CPU:
   `python3 tools/qwen3next_reference.py mtp --ckpt $SNAP --ids-json $IN/qwen35-0.8b-ids-code.json`
   (the engine implements `post_norm/history`). A rate near zero with a correct main model means
   the draft layer is mis-bound; here it is the 27B's dense draft path with BF16 matrices (through
   `build_full_next(p, true)` and the BF16 MLP), so suspect those two.

## 6. Still open after a pass

- World > 1; interleaved prefill (`engine.prefill_budget_tokens` is gated to the `qwen3_next`
  family name in `apps/dgpp_serve.cpp`; this model serves under `qwen3_5`).
- `engine.prefill_fp8_per_tensor` is refused with the shipped BF16 form (it requantizes block-FP8
  matrices); it is accepted with `dense_weights: "fp8"` and untested there.
- `engine.rope_scaling` (YaRN) is refused for this dialect.
- The vision tower (not served); a jinja2 golden corpus for this template (it loads and renders in
  the engine's interpreter; the parser and the forced-call grammar were probed on the Mac over the
  real tokenizer); measured throughput — at this size the per-token cost is dominated by the head
  (248320 x 1024) and fixed overheads, so tokens/s will say little about the kernels.
