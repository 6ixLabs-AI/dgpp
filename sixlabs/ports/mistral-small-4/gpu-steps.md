# Mistral-Small-4: what to run once a Spark is free

Status when this was written (2026-10-04, overnight): **the host side is written and tested on a
Mac; nothing has been compiled by nvcc or GCC, no weight has been loaded, no token produced.**
There is no model assembly and no serve family, so there is no "first request" yet — step 7 says
what the server prints instead. Everything marked "expected" is derived from the checkpoint's
headers and from Mac host runs, not from a run on a Spark.

- Checkpoint: `mistralai/Mistral-Small-4-119B-2603-NVFP4` @ `45331841b631f4e281df8e959ea3cc9beb84298a`
  (not gated; 13 shards, 65.9 GiB on disk). Mistral-native: `params.json`, no `config.json`.
  `hf download mistralai/Mistral-Small-4-119B-2603-NVFP4` — it is not on either box yet.
- Patch: `/Users/markgriffith/6ix-ports-overnight/mistral-minimax/1-mistral-small-4.patch`
  (against `main` at `89955e0`; `2-minimax-m2.7.patch` is cumulative and contains it).
- Plan and facts: `docs/mistral_small4_plan.md`.
- Inputs (token ids from the checkpoint's own template through jinja2 and HF tokenizers, made on
  the Mac): `sixlabs/ports/mistral-small-4/inputs/mistral4-ids-code.json` (111 tokens, an explicit
  system message) and `mistral4-ids-prose.json` (615 tokens: the template's default system prompt).

Below `$REPO` is a build copy with the patch applied (not the production checkout), `$SNAP` the
snapshot directory, `$IN` = `$REPO/sixlabs/ports/mistral-small-4/inputs`, `$OUT` a scratch directory.

## 0. Apply and build

```bash
cd $REPO && git status --short                      # a clean tree at 89955e0 or later main
git apply --check /path/to/1-mistral-small-4.patch && git apply /path/to/1-mistral-small-4.patch

# The default build: host libraries, unit tests, the bind check. The drafts are OFF.
cmake --preset ci        # on a FRESH clone add -DDGPP_BUILD_BENCHMARKS=OFF: main's CMakeLists names
                         # benchmarks/micro/qwen_dense_bench.cu, which is not in git at 89955e0
cmake --build build-ci -j 8 --target unit_tests mistral4_bind_check mistral4_tool_calls_test tokenizer_test
```

This is the first time GCC sees these files (the Mac used Apple clang and a CUDA header shim), and
the `ci` preset treats warnings as errors: if a GCC-only warning stops it, reconfigure with
`cmake --preset ci -DDGPP_WERROR=OFF` to get on, and fix the warning afterwards. The files are
host-only C++20 and listed in the report under "expected to break first"; `dgpp_models_mistral4`
links nothing but `dgpp_loaders` and `dgpp_rope_scaling`. The production build
(`cmake --build --preset release`, target `dgpp_serve_app`) compiles none of the new model files —
of this patch it sees only the text stack (`src/text`), `loaders/architecture.cpp`,
`loaders/hf_cache.cpp`, `models/glm/moe_reference.cpp` and the two refusals in `apps/dgpp_serve.cpp`.

**The draft** (one new kernel and its CUDA test; never compiled by nvcc) is opt-in:

```bash
cmake --preset ci -DDGPP_BUILD_MISTRAL_DRAFT=ON
cmake --build build-ci -j 8 --target mistral4_attn_test
# back to the default afterwards: cmake --preset ci -DDGPP_BUILD_MISTRAL_DRAFT=OFF
```

The option adds `dgpp_kernels_mistral4_draft` and `mistral4_attn_test` only. It does not add a
serve family; the server refuses this model by name either way.

## 1. Host tests (no GPU, no weights)

```bash
cd $REPO
DGPP_TEST_FILTER=mistral4 ./build-ci/unit_tests       # expect "19 tests, 0 failed"
./build-ci/unit_tests | tail -1                        # the whole suite
```

The 19: the `params.json` parser on the real file and its refusals, the architecture registry on a
`params.json` directory, the YaRN table and Llama-4 scale against frozen reference values, the
binding table pinned to 56,313 tensors / 70,801,959,560 bytes, the per-rank bytes, and the host
attention / router / MoE references against the numpy reference's vectors (the run prints the
measured errors; the stage errors were 0.004–0.010 on the Mac).

With the checkpoint in the hub cache, `mistral4_binding_matches_the_landed_checkpoint` also binds
every shard's header (it returns silently when the snapshot is absent — step 2 is the explicit run).

## 2. Bind check against the real checkpoint (headers only)

```bash
$REPO/build-ci/mistral4_bind_check --model mistralai/Mistral-Small-4-119B-2603-NVFP4
```

Expected, exactly (after the `model:` line):

```
config: mistral4, 36 layers, hidden 4096, vocab 131072
attention: latent (MLA), 32 heads, q latent 1024, kv latent 256, nope 64 + rope 64, v 128; yarn factor 128 over 8192, llama-4 beta 0.1 over 8192
mlp: routed MoE (softmax), 128 experts top-4, intermediate 2048, 1 shared expert; vision tower present (never loaded)
checkpoint: 13 shards, 56313 tensors in headers
binding: expected 56313 | matched 56313 (missing 0, dtype 0, shape 0) | unexpected 0
nvfp4 matrices: 13932; never loaded (activation scales, vision): 14154 tensors; bytes in headers 70801959560, of which the text model 69944959408
binding OK: every tensor of the table is present with its dtype and shape
weights per rank at world 1, as stored: 69944959408 bytes = 65.141 GiB (routed experts 60.750, shared 0.475, attention 1.881, embedding 1.000, head 1.000, routers and norms 0.036)
latent cache (replicated on every rank): 23040 bytes per token; a 262144-token pool is 5.625 GiB
```

The same validator printed the binding lines on the Mac from the real `params.json` and a dump of
the real headers. Exit code 0 is a pass; 1 is a binding failure with the first errors listed.

## 3. The numpy reference on the real weights (CPU only; the first time it sees them)

It mmaps the shards and keeps about 10 GB of converted weights. **Do not run it on a box that is
benchmarking or near its memory limit.** About half an hour for the 111-token input.

```bash
cd $REPO
python3 tools/mistral4_reference.py score --ckpt $SNAP --ids-json $IN/mistral4-ids-code.json \
        --out $OUT/ms4-ref-code.json --threads 8
```

Pass: the last line's `mean_logprob` is a small negative number (a working model predicts the
canned answer well: roughly −1 to −3 nat per token). Around −11.8 (≈ ln 131072) means the reference
reads the container or the model wrong. Each of these must make it collapse (they exist to prove a
convention, one at a time):

```bash
for v in gscale=mul nibble=hi rope=half split=rope_nope; do
  python3 tools/mistral4_reference.py score --ckpt $SNAP --ids-json $IN/mistral4-ids-code.json --variant $v --threads 8 | tail -1
done
```

**The open question this step settles** (docs/mistral_small4_plan.md §1.3): the attention's softmax
scale. Run the other reading and compare the two `mean_logprob` values:

```bash
python3 tools/mistral4_reference.py score --ckpt $SNAP --ids-json $IN/mistral4-ids-code.json \
        --variant yarn_mscale=on --threads 8 | tail -1
```

Expected: the default (`yarn_mscale=off`, the checkpoint's `apply_scale: false`, vLLM's reading) is
clearly better. If `on` is clearly better instead, transformers main is right and
`Mistral4TextConfig::attention_scale()` and `mistral4_ref::Geometry::scale` must take the factor
(0.1·ln 128 + 1)² = 2.2058 — change both, regenerate nothing (the vectors are of the synthetic
config), and rerun step 1. Record both numbers either way.

The Llama-4 scale only acts past position 8192; these inputs do not reach it. `--pos0 8192` scores
the same ids as if they started there (rope is relative, so only the scale changes): the mean
log-probability should stay close to the `--pos0 0` value, and `--variant llama4=off --pos0 8192`
shows what the scale is worth.

## 4. The draft kernel test (needs a GPU, a few hundred MB; build with the option ON)

```bash
$REPO/build-ci/mistral4_attn_test
```

Expected: `2 tests, 0 failed`, and three informative lines — the latent norm within 1 bf16 ulp of
the host reference, the rotated rope key bitwise and the query within 2 ulps, the heads' output
within the split-kernel tolerance class — then

```
[ .. ] dsa_attn_dense at kv_lora 256 + rope 64: ...
```

which says whether the list-free dense attention kernel is compiled for this model's geometry.
`kernels/dsa.hpp` documents 512 / 256 without a rope tail and 512 with one, so expect "NOT
compiled": that instantiation is the one real kernel gap (the plan, §4). This test has never been
compiled; a compile error in it is a draft bug, not an engine bug.

## 5. What does not exist yet

There is no loader, no layer object, no model and no forward check for this family, so there is no
memory plan to print and no engine-versus-reference comparison to run. The order to build them, and
the gate for each, is docs/mistral_small4_plan.md §6. The first numerical gate, once a forward
check exists, is `sixlabs/bench/compare_forward_check.py` against `$OUT/ms4-ref-code.json` at the
80B's bound (mean ≤ 0.06 nat) — the reference's output already has the schema that script reads.

## 6. The text stack (host; needs the checkpoint's tokenizer.json and chat_template.jinja in the hub cache)

```bash
cd $REPO
./build-ci/tokenizer_test tests/data/mistral4_tokenizer_goldens.jsonl
./build-ci/mistral4_tool_calls_test tests/data/mistral4_chat_template_goldens.jsonl
DGPP_TEST_FILTER=tool_ ./build-ci/unit_tests
```

Expected: the tokenizer gate prints 2 tests, 0 failed (376 cases byte-exact, encode and both
decodes); `mistral4_tool_calls_test` prints
5 tests, 0 failed (31 renders byte-exact with matching ids, 13 template refusals by message, 22
turns / 12 calls round-tripped through the parser, 15 grammar walks). Exit code 2 from either means
"checkpoint not found", not a pass. All of this passed on the Mac against the real files.

## 7. The server

```bash
cp deploy/cluster_mistral-small-4_nvfp4_w1.example.json deploy/cluster_mistral-small-4_nvfp4_w1.json
python3 scripts/dgpp-cluster up --config deploy/cluster_mistral-small-4_nvfp4_w1.json
```

Expected today: rank 0 exits during startup with

```
serve: <snapshot> is a Mistral-Small-4 checkpoint (mistral4): recognized, bound and referenced on the host, but its device assembly is an unverified draft with no serve family yet — see docs/mistral_small4_plan.md and sixlabs/ports/mistral-small-4/gpu-steps.md
```

That refusal is the designed behaviour (before this patch the directory had no `config.json` and
the server stopped at "cannot open config"; it must never fall through to the GLM family). When a
family exists, the first requests are the usual three — plain chat at temperature 0, a forced tool
call (`tool_choice: "required"`: expect `tool_calls[0].function.arguments` as a JSON string and no
`[TOOL_CALLS]` text in content), and `reasoning_effort: "high"` (expect reasoning between `[THINK]`
and `[/THINK]` separated from content) — and the things the family must state are in the report's
text-stack section.

## 8. Still open after every step above passes

- §1.3's scale on real weights (step 3 decides it for the reference; the engine follows).
- Activation quantization: vLLM quantizes the experts' inputs to 4 bits (the recipe is W4A4); this
  engine and its references do not. A vLLM capture is therefore not a bit-level oracle.
- The vision tower (not served), `reasoning_effort` values other than none / high, the template's
  default system prompt with its literal `{today}` placeholders.
- Everything in §5.
