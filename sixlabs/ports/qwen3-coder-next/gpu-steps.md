# Port 1 — Qwen3-Coder-Next: what to run once a GPU is free

Status when this was written (2026-10-04, overnight): **code complete on the host side, nothing
has run on a GPU.** Everything below marked "expected" is what the code should print, derived from
the checkpoint's headers and from Mac host runs — not from a run on a Spark.

- Checkpoint: `RedHatAI/Qwen3-Coder-Next-NVFP4` @ `27a8f16f463b9a13c91c332c40cf93e09717347e`
  (DGXone: `/home/mark/.cache/huggingface/hub/models--RedHatAI--Qwen3-Coder-Next-NVFP4/snapshots/27a8f16f463b9a13c91c332c40cf93e09717347e`;
  DGXtwo sees it under `/mnt/dgxone-hf/hub/...`).
- Patch: `/Users/markgriffith/6ix-ports-overnight/1-qwen3-coder-next.patch` (cumulative, against
  `main` at `07f4f32`; later ports' patches contain it).
- Inputs made on the Mac with the engine's own tokenizer and template (copy them to the box):
  `/Users/markgriffith/6ix-ports-overnight/inputs/coder-next-ids-code.json` (107 tokens) and
  `coder-next-ids-prose.json` (83 tokens).

Below, `$REPO` is a build copy of the repository with the patch applied (the usual one is DGXone
`~/dgpp-next`; do not build in the production `~/dgpp`), `$SNAP` is the snapshot directory above,
`$IN` is where you copied the two id files, and `$OUT` is a scratch directory.

## 0. Apply and build

```bash
cd $REPO && git status --short            # expect a clean tree at 07f4f32 (or later main)
git apply --check /path/to/1-qwen3-coder-next.patch && git apply /path/to/1-qwen3-coder-next.patch

# Test build (warnings are errors in this preset — the Mac checks used clang, GCC may find more):
cmake --preset ci
cmake --build build-ci -j 8 --target unit_tests qwen3codernext_tool_calls_test qwen3next_tool_calls_test \
      qwen35_ports_loader_test qwen3next_loader_test qwen35_bind_check qwen35_forward_check
# Server:
cmake --preset release && cmake --build --preset release -j 8        # builds dgpp_serve_app
cmake --build build-release -j 8 --target qwen35_forward_check qwen35_bind_check
```

If the `ci` build stops on a warning in a file this patch touches, that is the first thing to fix;
the files are listed in the report (section "likely to break first").

## 1. Host tests (no GPU, no weights)

```bash
cd $REPO
./build-ci/unit_tests | tail -3                       # expect "N tests, 0 failed" (three arena tests fail when
                                                      # the GPU is full: unrelated, see commit f9a333a)
DGPP_TEST_FILTER=qwen3codernext ./build-ci/unit_tests  # expect 12 tests, 0 failed
DGPP_TEST_FILTER=qwen3next ./build-ci/unit_tests       # the 80B's: still all OK (regression)
./build-ci/qwen3codernext_tool_calls_test              # needs the checkpoint in the HF cache; expect 4 tests, 0 failed
                                                      # (exit code 2 = checkpoint not found = skipped, not passed)
./build-ci/qwen3next_tool_calls_test tests/data/qwen3next_chat_template_goldens.jsonl   # the 80B's JSON form: 4 OK
python3 -m unittest tests.python.portability_test.PortabilityTest.test_deployment_filenames_describe_their_settings
```

All of these passed on the Mac (against a CUDA header shim, Apple clang) — on the box they run
against the real toolchain for the first time.

## 2. The loader on its synthetic fixture (needs a little device memory, not the model)

```bash
cd $OUT && $REPO/build-ci/qwen35_ports_loader_test     # expect 3 tests, 0 failed (qwen3codernext_loader_*)
$REPO/build-ci/qwen3next_loader_test                   # the 80B fixture: 3 OK (regression)
```

## 3. Bind check against the real checkpoint (headers only, no weights loaded)

```bash
$REPO/build-ci/qwen35_bind_check --model RedHatAI/Qwen3-Coder-Next-NVFP4
```

Expected, exactly:

```
config: qwen3_next, 48 layers (36 GDN + 12 full attention) + 0 draft, hidden 2048, vocab 151936
mlp: routed MoE, 512 experts top-10, intermediate 512, shared expert 512
checkpoint: 10 shards, 296151 tensors in headers
binding: expected 296151 | matched 296151 (missing 0, dtype 0, shape 0) | unexpected 0 | out of scope 0 | vision skipped 0
quantized matrices: 73920
binding OK: every tensor of the table is present with its dtype and shape
```

(The same validator, fed the real `config.json` and a dump of the real headers, printed these
numbers on the Mac.) Also confirm the Intel release is refused by name:

```bash
$REPO/build-ci/qwen35_bind_check --model Intel/Qwen3-Coder-Next-int4-AutoRound
# expect: "... quant_method: 'auto-round' (the AutoRound int4 release) is not implemented ..." and exit 2
```

## 4. Memory plan (loads nothing)

```bash
$REPO/build-release/qwen35_forward_check --model RedHatAI/Qwen3-Coder-Next-NVFP4 --plan 8,524288 --dense-weights fp8
$REPO/build-release/qwen35_forward_check --model RedHatAI/Qwen3-Coder-Next-NVFP4 --plan 8,524288
```

Expected "model weights (resident)": about **43.35 GiB** with `--dense-weights fp8` and **44.92 GiB**
without (the loader's byte formulas, computed on the Mac from the real config; the 80B is 44.90 /
46.50 with its draft layer). The line must say `mtp off` (this checkpoint has no draft layer).

## 5. The numerical gate: forward check against the numpy reference

Run the reference first (CPU only; it mmaps the shards and keeps about 8 GB of converted weights —
**do not run it on a box that is near its memory limit**; `--threads 8`):

```bash
cd $REPO
for k in code prose; do
  python3 tools/qwen3next_reference.py score --ckpt $SNAP --ids-json $IN/coder-next-ids-$k.json \
          --out $OUT/cn-ref-$k.json --threads 8
  python3 tools/qwen3next_reference.py dump  --ckpt $SNAP --ids-json $IN/coder-next-ids-$k.json \
          --out $OUT/cn-refdump-$k --detail-layers 0,3 --threads 8
done
```

Sanity of the reference itself (it has only ever run on a synthetic checkpoint): the `score`
output's `mean_logprob` on these natural texts should be a small negative number (a working model
predicts the canned answer well: roughly -1 to -3 nat/token); around -12 (≈ ln 151936) means the
reference reads the container wrong. The convention switches exist to prove each assumption —
each of these must make the mean logprob collapse:

```bash
python3 tools/qwen3next_reference.py score --ckpt $SNAP --ids-json $IN/coder-next-ids-prose.json --variant gscale=mul
python3 tools/qwen3next_reference.py score --ckpt $SNAP --ids-json $IN/coder-next-ids-prose.json --variant nibble=hi
```

Then the engine, exact form first (BF16 dense stack), streaming residency (one layer on the device
at a time, so it fits beside other work — it still needs a free GPU):

```bash
for k in code prose; do
  $REPO/build-release/qwen35_forward_check --model RedHatAI/Qwen3-Coder-Next-NVFP4 \
      --ids-json $IN/coder-next-ids-$k.json --dump-states $OUT/cn-$k.bf16 > $OUT/cn-$k.txt
  python3 sixlabs/bench/compare_forward_check.py $OUT/cn-$k.txt $OUT/cn-ref-$k.json
  python3 sixlabs/bench/compare_states.py $OUT/cn-$k.bf16 $OUT/cn-refdump-$k $(python3 -c "import json;print(len(json.load(open('$IN/coder-next-ids-$k.json'))))") 2048
done
```

Pass:
- `compare_forward_check.py` prints `PASS`: mean |engine − reference| ≤ 0.06 nat (the 80B's
  measured agreement with its 8-bit dense stack; the BF16 form here should be tighter), worst
  position ≤ 0.5 nat, and the greedy next token agrees on every row or differs only where the
  reference's own margin is small (a few hundredths of a nat).
- `compare_states.py`: the relative error grows slowly and smoothly with depth and every cosine
  stays ≈ 1. A wrong layout shows as a jump to ~1.0 at one layer: bisect with `--layers N` on both
  tools (`qwen35_forward_check --layers N`, `qwen3next_reference.py ... --layers N`). Layer 0 is a
  GDN layer (BF16 projections, the gather), layer 3 the first attention layer (NVFP4 q/k/v/o
  dequantized at load) — a jump at layer 3 and not at 0–2 points at the attention dequant, a jump
  at layer 0 at the MoE (experts' global scale) or the GDN.

Then the serving form (what the template uses):

```bash
$REPO/build-release/qwen35_forward_check --model RedHatAI/Qwen3-Coder-Next-NVFP4 --dense-weights fp8 \
    --ids-json $IN/coder-next-ids-code.json > $OUT/cn-code-fp8.txt
python3 sixlabs/bench/compare_forward_check.py $OUT/cn-code-fp8.txt $OUT/cn-ref-code.json
```

Expected: still `PASS` at the 0.06 bound. This form re-encodes the already-4-bit attention and
shared-expert matrices into block FP8 (as the 80B does for o_proj and out_proj only); if it fails
the bound while the BF16 form passes, serve with `"dense_weights": "checkpoint"` and record the
measured difference.

## 6. First boot and first requests

```bash
cd $REPO
cp deploy/cluster_qwen3-coder-next_nvfp4_w1.example.json deploy/cluster_qwen3-coder-next_nvfp4_w1.json
python3 scripts/dgpp-cluster doctor --config deploy/cluster_qwen3-coder-next_nvfp4_w1.json
python3 scripts/dgpp-cluster up     --config deploy/cluster_qwen3-coder-next_nvfp4_w1.json
python3 scripts/dgpp-cluster paths  --config deploy/cluster_qwen3-coder-next_nvfp4_w1.json   # where the log is
```

In the rank-0 log, expect:
- `serve: model family qwen3_next (...Qwen3-Coder-Next-NVFP4...)`
- `serve: the template writes its tool calls as <function=...> XML tags`  (the 80B prints
  `one JSON object (the Hermes form)` — if Coder-Next prints that, the template sniff is wrong)
- a memory plan whose weights line is ≈ 43.4 GiB and no "mtp scratch" item
- `READY`

If the config says `"mtp": true` the server must refuse to start with `engine.mtp is on, but this
checkpoint carries no draft layer`.

Requests (replace host/port with the deployment's; the alias is `Qwen3-Coder-Next`):

```bash
URL=http://127.0.0.1:8000/v1/chat/completions
# 1. plain chat, greedy
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3-Coder-Next","temperature":0,"max_tokens":64,
  "messages":[{"role":"user","content":"Write a Python one-liner that reverses a string s."}]}'
# expect: a sensible answer (e.g. s[::-1]), finish_reason "stop", no "<|im_start|>" in the text

# 2. a forced tool call (the grammar) — expect choices[0].message.tool_calls[0] ==
#    {"type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}} (arguments a JSON string),
#    finish_reason "tool_calls", and no <tool_call> text left in content
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3-Coder-Next","temperature":0,"max_tokens":128,
  "messages":[{"role":"user","content":"What is the weather in Paris?"}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice":"required"}'

# 3. the same with "tool_choice":"auto": the model should still call the tool by itself, in the
#    <function=...> form its template taught it; a JSON-looking block in content means the model
#    and the parser disagree about the form.

# 4. reasoning_effort / enable_thinking are accepted and ignored (no <think> in this model):
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3-Coder-Next","max_tokens":16,"reasoning_effort":"high",
  "messages":[{"role":"user","content":"Say hi."}]}'      # expect 200, not 400
```

Then the fleet comparison: replay `sixlabs/refset` (short prompts and the tool-call rows) against
this server and against the vLLM deployment of `Intel/Qwen3-Coder-Next-int4-AutoRound`
(`sixlabs/refset/capture_refset.py`, `sixlabs/bench/compare_transcripts.py`). They are different
quantizations of the same weights, so expect the same answers in substance, not token-identical
transcripts.

Finally `python3 scripts/dgpp-cluster down --config deploy/cluster_qwen3-coder-next_nvfp4_w1.json`.

## 7. Not covered by any step above (still open after a pass)

- The chat template has **no jinja2 golden corpus**: its renders were compared with hand-derived
  strings for two plain prompts and by structure for the tools header. To close it, add a
  Coder-Next section to `tools/gen_chat_template_goldens.py` (as the Qwen3-Next one) in the venv
  that pins jinja2 3.1.2, write `tests/data/qwen3codernext_chat_template_goldens.jsonl`, and compare
  in `tests/host/qwen3codernext_tool_calls_test.cpp` as the 80B's test does.
- World > 1 (the `qwen3_5` stack is single-node), the W4A4 prefill opt-in with this container's
  `input_global_scale` (the loader keeps no static activation scale for it; `DGPP_MOE_W4A4=1`
  would quantize activations dynamically), a model-level fixture gate, measured throughput.
- The Intel int4 AutoRound checkpoint itself: see the report for exactly what it would need.
