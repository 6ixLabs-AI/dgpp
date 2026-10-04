# Qwen3-VL-30B-A3B-Instruct (NVFP4), text path — what to run once a GPU is free

Status when this was written (2026-10-04, overnight, on a Mac): **the host side has run; nothing has
been compiled with nvcc or GCC, loaded on a GPU, or produced a token.** "Expected" below means
derived from the checkpoint's `config.json` and safetensors *headers* (read from Hugging Face; no
weight bytes were downloaded) and from Mac host runs (Apple clang against a CUDA header shim), not
from a run on a Spark.

| | |
|---|---|
| checkpoint | `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4` @ `3c6162d5513d26f008628eebe9b4355559b4a305` (public, 4 shards, 19.2 GB). **Not known to be on either box** — step 0 downloads it. |
| model class | `Qwen3VLMoeForConditionalGeneration` / `qwen3_vl_moe`: 48 layers, hidden 2048, 32 query / 4 KV heads of 128, full rotary (theta 5e6), q/k head norms, a routed MoE of 128 experts top-8 with **no shared expert**, vocab 151936 (the 80B's tokenizer, byte for byte), untied head. |
| container | compressed-tensors `nvfp4-pack-quantized`: attention q/k/v/o and every expert NVFP4; router, head, norms and the whole vision tower BF16. |
| scope | **TEXT ONLY.** The tower's 351 tensors are bound (the checkpoint validates whole) and never loaded; an image part is refused by name. |
| engine family | `qwen3_vl_moe` in `src/models/qwen3/` (new files). No new kernel: GLM-4.7's attention layer and K/V pool, the shared routed MoE layer, the qwen3_5 loader's NVFP4 dequant. |

What is in which build:

- **Always built** (compile-checked and unit-tested on the Mac): `dgpp_models_qwen3` — config,
  binding table, host reference; `qwen3_bind_check`; the three `qwen3_*` unit-test files; the
  chat-template / tool-call corpora. In this build the server **refuses** the checkpoint by name.
- **Only with `-DDGPP_BUILD_QWEN3_PLAIN_DRAFT=ON`** (an unverified draft):
  `src/models/qwen3/loader.{hpp,cpp}`, `model.{hpp,cpp}`, `apps/qwen3_forward_check.cpp`,
  `tests/cuda/qwen3_loader_test.cpp`, `tests/cuda/qwen3_forward_test.cpp` (helpers in
  `tests/cuda/qwen3_gpu_fixture.hpp`), and the family in the server's registry. All of it is
  syntax-checked with Apple clang against a CUDA header shim. One part has also *run*, on the Mac,
  against a CUDA runtime shim whose device memory is `malloc` and whose copies are `memcpy`: the
  loader and its test (`3 tests, 0 failed`). That exercises the loader's host logic — names, roles,
  dequant, slice arithmetic, the byte plan — and nothing a device does.

`$REPO` = a build copy with the patch applied, `$OUT` = scratch, `$IN` =
`$REPO/sixlabs/ports/qwen3-vl-30b-a3b/inputs`, `$SNAP` = the snapshot directory. Run nothing here on
a box that is benchmarking.

## 0. The checkpoint

```bash
hf download ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4 --revision 3c6162d5513d26f008628eebe9b4355559b4a305
SNAP=~/.cache/huggingface/hub/models--ig1--Qwen3-VL-30B-A3B-Instruct-NVFP4/snapshots/3c6162d5513d26f008628eebe9b4355559b4a305
```

## 1. Default build: host tests and the bind check (no GPU work)

```bash
cd $REPO
cmake --preset ci
cmake --build build-ci -j 8 --target unit_tests qwen3_bind_check qwen3next_tool_calls_test
DGPP_TEST_FILTER=qwen3_ ./build-ci/unit_tests        # expect: 21 tests, 0 failed
./build-ci/unit_tests | tail -1                      # the whole suite: 0 failed
./build-ci/qwen3_bind_check --checkpoint-dir $SNAP --world 2
ctest --test-dir build-ci -R 'qwen3vl_tool_calls_test' --output-on-failure   # needs the checkpoint in the HF cache
```

The filter matches this port's twenty cases — `qwen3_vl30b_*` (2), `qwen3_235b_*` (2), `qwen3_dense_*`
(2), `qwen3_architecture_*`, `qwen3_config_*` (3), `qwen3_binding_*`, `qwen3_tp_*`, `qwen3_reference_*`
(6), `qwen3_fp4_*`, `qwen3_moe_oracle_*` — and one neighbour whose name contains it
(`qwen36_architecture_is_the_qwen3_5_stack`). On the Mac those 21 passed, and so did a 219-case
slice built together with this port's changes on main at `7cc561a`: the config / binding /
tool-parser / grammar test files of every family, the Nemotron-3 and Qwen3.5-0.8B / 122B ones
included. The rest of `unit_tests` (the files that need CUDA or the serving stack) has not been
built with these changes.

The `ci` preset is `-Werror` with GCC; the Mac build was clang `-Wall -Wextra -Wpedantic` clean. A
GCC-only warning here is the first thing that can go wrong, and it would be in a file of this port
(`src/models/qwen3/{config,binding,reference,moe_reference}.cpp`, `apps/qwen3_bind_check.cpp`,
`tests/unit/qwen3_*`).

Expected from the bind check (every number below was produced on the Mac by the same validator over
the real `config.json` and a dump of the real headers):

```
config: qwen3_vl_moe, 48 layers, hidden 2048, 32 query / 4 kv heads x 128, rope theta 5000000, vocab 151936, max positions 262144
mlp: routed MoE, 128 experts top-8, intermediate 768, no shared expert
weights: NVFP4, compressed-tensors nvfp4-pack-quantized (q/k/v/o and the experts)
vision: a 27-block tower (hidden 1152) is BOUND and NOT SERVED — text path only, image inputs are refused
checkpoint: 4 shards, 75090 tensors in headers, 19164721120 tensor bytes
table: 75090 tensors (74739 text + 351 vision)
binding: expected 75090 | matched 75090 (missing 0, dtype 0, shape 0) | unexpected 0 | out of scope 0 | vision bound 351 | tied head copies 0 | matched bytes 19164721120
nvfp4 matrices: 18624
table count 75090 == header count 75090; table bytes 19164721120 == header bytes 19164721120
binding OK: every tensor of the table is present with its dtype and shape
resident, world 2 rank 0: experts 8153800704, attention 905969664, dense mlp 0, router 25165824, norms 421888, embedding 622329856, head 311164928 | total 10018852864 bytes (9.33 GiB)
```

(World 1 is 19,389,714,432 bytes, 18.06 GiB: the experts as shipped, the attention projections
dequantized to BF16.)

The tool-call test (`qwen3vl_tool_calls_test`, the 80B's binary on this checkpoint's corpus) passed
on the Mac against the real `tokenizer.json` and `chat_template.jinja`: 8 renders byte-exact with 8
id lists equal to HF tokenizers, 5 calls round-tripped, the forced-call grammar admits them.

## 2. The reference on the real checkpoint (CPU, no engine)

`tools/qwen3_reference.py` is the ground truth. On synthetic checkpoints it reproduces transformers'
`Qwen3VLMoeForConditionalGeneration` (text only) to 4e-15 once the reference's three float32 islands
are taken in float64, and to 4e-7 as transformers ships (`tools/qwen3_synth.py torch-check`). What a
synthetic checkpoint cannot prove is the *container* reading, so check that first:

```bash
cd $REPO
for k in prose code; do
  python3 tools/qwen3_reference.py score --ckpt $SNAP --ids-json $IN/qwen3vl-ids-$k.json --out $OUT/vl-ref-$k.json --threads 8
done
```

1. The last line's `mean_logprob` on these natural texts must be a small negative number (roughly
   −1 to −3). About −11.9 (ln 151936) means a container convention is wrong.
2. Each of these must **collapse** it (run on the 83-token prose ids; `--layers 4` is enough to see it):
   `--variant nibble=hi`, `--variant gscale=mul`, `--variant gateup=up`, `--variant norm=1p`,
   `--variant rope=adjacent`, `--variant qknorm=off`. `gateup` is the one this port could not take
   from an earlier port: llm-compressor 0.8.1 unpacks transformers' fused `gate_up_proj` into
   `gate_proj = gate_up[:, :I].T`, `up_proj = gate_up[:, I:].T` (read from its source), and the
   reference applies SiLU to `gate_proj`. If `gateup=up` scores *better*, the release was written
   with another version and `tools/qwen3_reference.py`, `src/models/qwen3/reference.cpp` and the
   binding comment must swap.
3. The C++ reading of the same bytes, without a GPU (about a minute for two layers; plain loops):

```bash
./build-ci/qwen3_bind_check --checkpoint-dir $SNAP --layers 2 --host-forward $IN/qwen3vl-ids-prose.json > $OUT/vl-host-L2.txt
python3 tools/qwen3_reference.py score --ckpt $SNAP --layers 2 --dtype float64 --variant ropeangle=f64 \
    --ids-json $IN/qwen3vl-ids-prose.json --out $OUT/vl-ref-L2.json
python3 sixlabs/bench/compare_forward_check.py $OUT/vl-host-L2.txt $OUT/vl-ref-L2.json --mean 1e-4 --max 1e-3
```

   Expect `PASS` with differences near 1e-6 (the reference rounds its dequantized weights to float32
   the same way; on the synthetic checkpoints the two agree to 1e-13).

## 3. Turn the draft on and build it

```bash
cd $REPO
cmake --preset ci -DDGPP_BUILD_QWEN3_PLAIN_DRAFT=ON
cmake --build build-ci -j 8 --target qwen3_loader_test qwen3_forward_test qwen3_forward_check
cmake --preset release -DDGPP_BUILD_QWEN3_PLAIN_DRAFT=ON
cmake --build build-release -j 8 --target qwen3_forward_check dgpp_serve_app
```

(`-DDGPP_BUILD_QWEN3_PLAIN_DRAFT=OFF`, or a fresh build directory, turns it back off.) What I expect
to break on the first real build, most likely first:

1. A GCC or nvcc diagnostic the clang syntax check did not raise, in `src/models/qwen3/loader.cpp`,
   `model.cpp`, `apps/qwen3_forward_check.cpp` or the two `tests/cuda/qwen3_*_test.cpp`. The `ci`
   preset is `-Werror`.
2. Link order / a missing library in the three `if(DGPP_BUILD_QWEN3_PLAIN_DRAFT)` blocks of
   `CMakeLists.txt` (never configured: CMake needs nvcc). `dgpp_models_qwen3_loader` must see
   `qwen3next_fp4_dequant_bf16` / `qwen35_fp4_packed_dequant_bf16` from `dgpp_models_qwen_loader`.
3. `ServeGraphEngineOf<Qwen3Model>` reaching a draft-layer path the model does not have. The class
   sets `kDraftChain` and `kBatchedDraftChain` false and throws from `mtp_run_rows`; the family
   refuses `engine.mtp`. Syntax-checked, never instantiated by a real compiler.

## 4. The fixture gates (GPU, seconds)

```bash
cd $OUT && $REPO/build-ci/qwen3_loader_test      # expect: 3 tests, 0 failed
cd $OUT && $REPO/build-ci/qwen3_forward_test     # expect: 4 tests, 0 failed
```

Synthetic checkpoints of both NVFP4 containers (3 layers, hidden 256, 4 query / 2 KV heads of 128,
8 experts top-3 of 128, vocab 512 — the unit presets widened to the kernels' tile sizes); both
binaries write them under the working directory.

- `qwen3_loader_test` — the loader alone, run first: every resident tensor at world 1 against the
  file (and, for the projections, against the host dequant), each rank's slices at world 2, the
  NVFP4 divisor's direction per container, the byte formula against the plan, and two refusals
  (world 3; the dense dialect). **This one passed on the Mac's runtime shim**; on a Spark it adds
  the real allocator and the real copies.
- `qwen3_forward_test` — the walk, never run anywhere: the cold forward against the host reference
  with the engine's rounding points, per layer and at the logits (both containers); session prefill
  + T=1 steps against the cold forward; the refusal of a vision placeholder.

The numeric bounds are first guesses (residual rel l2 < 0.02 per layer, log-probabilities within 0.1
max / 0.03 mean of the host walk, routes equal on 90 %): on the Mac the host walk with the engine's
rounding points sits 0.04 max / 0.02 mean nat from the exact one on these fixtures, and the GPU
should be much closer to the former than that. A failure that names a layer or `q_proj rows` /
`payload columns` is a real defect; a bound missed by a small factor is a bound to re-set from the
printed numbers.

Reading a failure:
- `… w1 layer 0: q_proj rows` — the loader's NVFP4-to-BF16 dequant or its slice arithmetic;
- `… divisor` — the global scale's direction (modelopt's is stored as its reciprocal);
- loader cases pass, `layer 0 residual` fails — the walk: norm, attention or MoE (`host compare` in
  step 6 localizes it on the real checkpoint; here bisect by making `num_hidden_layers` 1);
- `qwen3_loader_test` fails on a Spark although it passed on the Mac — the device side of the
  loader: the grant allocator, the pinned staging copies, a use of the resident image;
- everything passes at world 1, `w2 rank 1` fails — the rank slices only (nothing at world 2 is served).

## 5. Memory plan (loads nothing)

```bash
$REPO/build-release/qwen3_forward_check --checkpoint-dir $SNAP --plan 8,262144
```

Expected: `model weights (resident; the vision tower stays on disk)` ≈ **18.1 GiB** (18.06 GiB plus
the 256-byte alignment of every grant); `kv cache pool` = 262144 × 48
layers × 4 KV heads × 128 × 2 (K and V) × 2 bytes = **24.0 GiB**; a total well under one Spark.

## 6. The numerical gate: forward check against the reference

```bash
cd $REPO
for k in prose code; do
  python3 tools/qwen3_reference.py dump --ckpt $SNAP --ids-json $IN/qwen3vl-ids-$k.json --out $OUT/vl-refdump-$k --detail-layers 0,1 --threads 8
  ./build-release/qwen3_forward_check --checkpoint-dir $SNAP --ids-json $IN/qwen3vl-ids-$k.json \
      --dump-states $OUT/vl-$k.bf16 > $OUT/vl-$k.txt
  python3 sixlabs/bench/compare_forward_check.py $OUT/vl-$k.txt $OUT/vl-ref-$k.json
  python3 sixlabs/bench/compare_states.py $OUT/vl-$k.bf16 $OUT/vl-refdump-$k \
      $(python3 -c "import json;print(len(json.load(open('$IN/qwen3vl-ids-$k.json'))))") 2048
done
# and, where a layer parts from the reference, the same walk on the host with the engine's rounding points:
./build-release/qwen3_forward_check --checkpoint-dir $SNAP --ids-json $IN/qwen3vl-ids-prose.json --layers 4 --host-compare
```

Pass: `compare_forward_check.py` prints `PASS` at its defaults (mean ≤ 0.06 nat, worst position ≤
0.5 — the 80B's measured agreement; this stack has not been measured), the greedy next token equal
except at near ties; `compare_states.py` shows a smooth, slowly growing relative error with cosines
≈ 1; `host compare layer N: rel l2` stays small (the host walk has the engine's rounding points, so
it should be the tighter of the two).

Reading a failure:
- a jump at layer 0 in both comparisons: the attention (the reference's `L00_attn_q` / `attn_k` /
  `attn_ctx` dumps against the kernel's contract) or the MoE (`L00_router_ids`, `mlp_out`);
- fine against the host walk, off against numpy: the engine's rounding points are the gap — look at
  the bf16 dequant of q/k/v/o (the experts are exact in both);
- states fine, log-probabilities off: the head (BF16, a plain GEMM).

The inputs are 83 tokens (prose) and 107 (code), rendered on the Mac through the engine's own
template interpreter and tokenizer from `inputs/messages-*.json`, and equal to HF tokenizers' ids.

## 7. First boot and first requests

```bash
cd $REPO
cp deploy/cluster_qwen3-vl-30b-a3b_nvfp4_w1.example.json deploy/cluster_qwen3-vl-30b-a3b_nvfp4_w1.json
python3 scripts/dgpp-cluster doctor --config deploy/cluster_qwen3-vl-30b-a3b_nvfp4_w1.json
python3 scripts/dgpp-cluster up     --config deploy/cluster_qwen3-vl-30b-a3b_nvfp4_w1.json
```

Log: `serve: model family qwen3_vl_moe`, a WARN that the checkpoint's vision tower is bound and not
served, `the template writes its tool calls as one JSON object (the Hermes form)`, a plan with ≈ 18
GiB of weights, `READY`. (A build without the option stops here with: `the checkpoint is a
qwen3_vl_moe model (the plain Qwen3 family). This build binds and validates it … but does not serve
it`.) Then (alias `Qwen3-VL-30B-A3B`):

```bash
URL=http://127.0.0.1:8000/v1/chat/completions
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3-VL-30B-A3B","temperature":0,"max_tokens":64,
  "messages":[{"role":"user","content":"What is 17 * 23? Answer with the number."}]}'
# expect: content containing 391, no reasoning_content, finish_reason "stop"
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3-VL-30B-A3B","temperature":0,"max_tokens":256,
  "messages":[{"role":"user","content":"What is the weather in Paris?"}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.",
    "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice":"required"}'
# expect: tool_calls[0].function.name == "get_weather", .arguments a JSON string for {"city": "Paris"}, finish_reason "tool_calls"
curl -s $URL -H 'content-type: application/json' -d '{"model":"Qwen3-VL-30B-A3B","max_tokens":16,
  "messages":[{"role":"user","content":[{"type":"text","text":"What is this?"},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]}]}'
# expect: HTTP 400, code "unsupported_content_type", message:
#   "this checkpoint is a Qwen3-VL model, but its vision tower is not served: the engine runs the text
#    path only (send text content parts; image and video inputs are refused)"
```

`"mtp": true` in the config must refuse the start with `engine.mtp is on, but the plain Qwen3
checkpoints carry no draft layer`. Greedy transcripts must be stable across slots: replay the
refset's greedy prompts at 1 and 8 streams (`sixlabs/refset/capture_refset.py`) and compare with
`sixlabs/bench/compare_transcripts.py`.

## 8. Still open after a pass

- Vision: not served. The tower is `Qwen3VLVisionModel` with three DeepStack mergers that add
  features to the residual after layers 0–2, and MROPE positions for the placeholder rows; the engine
  has a Qwen3-VL-style tower for Qwen3.8 (`src/models/qwen/vision.*`), not wired here.
- A literal `<|image_pad|>` / `<|video_pad|>` typed as *text* tokenizes to the placeholder id and
  would run through the text path as an ordinary embedding row (the reference implementation raises:
  no features to scatter). `qwen3_forward_check` refuses such ids; the serving path does not look.
- Interleaved prefill (`engine.prefill_budget_tokens` is gated to three family names in
  `apps/dgpp_serve.cpp`; the template leaves it out), `engine.rope_scaling` (refused: the family has
  no ramp), `engine.bf16_weights` (packs nothing here), the FP8 dense form (`engine.dense_weights`:
  not wired — the BF16 attention projections are 1.8 GiB), W4A4 expert prefill
  (`DGPP_MOE_W4A4`; the compressed-tensors `input_global_scale` is not read).
- World > 1: refused by name (`the plain Qwen3 family is single-node for now`). The loader's rank
  slices are covered by `qwen3_loader_test`; the walk's two folds are GLM-4.7's and untested here — see
  `sixlabs/ports/qwen3-235b-a22b/sharding-plan.md`.
- `POST /v1/score` (landed on main the same night): this family answers 501 `score_unsupported`.
  `Qwen3Model` does not declare `kScoreTail`; honouring `RowRun::tail_rows` is the two lines
  `Qwen35Model::run_rows` gained, to add once the walk itself has passed.
- Measured throughput; a decode-rows ceiling (32 is GLM-4.7's for the same attention layer).
- Formatting: the new files are hand-formatted in the neighbours' style and are not
  `clang-format` clean (Apple clang-format 17 reports differences in them, as it does in the
  neighbouring families' files on main). `format-check` is its own target; no build depends on it.
