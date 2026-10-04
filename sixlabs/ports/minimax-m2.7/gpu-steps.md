# MiniMax-M2.7: what to run once the Sparks are free

Status when this was written (2026-10-04, overnight): **the host side is written and tested on a
Mac; nothing has been compiled by nvcc or GCC, no weight has been loaded, no token produced.**
The model needs two ranks (125.17 GiB of weights against a 121.6 GiB node). There is no model
assembly and no serve family: what exists beyond the host side is a sharding plan
(docs/minimax_m27_plan.md §4) and one unverified draft kernel. Everything marked "expected" is
derived from the checkpoint's headers and from Mac host runs.

- Checkpoint: `lukealonso/MiniMax-M2.7-NVFP4` @ `db821d7a3ce29ee96d80a1cae88d878d8586b54e` (not
  gated). **Download only the model's files** — the repository also holds 18 safetensors files
  that are not the model and three `.bak` copies (10.9 GB):

  ```bash
  hf download lukealonso/MiniMax-M2.7-NVFP4 --include "model-000*-of-00025.safetensors" \
      "model-inputscales.safetensors" "*.json" "*.jinja" "*.txt"
  ```

  134.4 GB. Both ranks need it (or the shared mount).
- Patch: `/Users/markgriffith/6ix-ports-overnight/mistral-minimax/2-minimax-m2.7.patch`
  (cumulative, against `main` at `89955e0`; it contains the Mistral-Small-4 port).
- Plan and facts: `docs/minimax_m27_plan.md`.
- Inputs (token ids from the checkpoint's own template through jinja2 and HF tokenizers, made on
  the Mac): `sixlabs/ports/minimax-m2.7/inputs/minimax-ids-prose.json` (111 tokens) and
  `minimax-ids-code.json` (107 tokens).

Below `$REPO` is a build copy with the patch applied (not the production checkout), `$SNAP` the
snapshot directory, `$IN` = `$REPO/sixlabs/ports/minimax-m2.7/inputs`, `$OUT` a scratch directory.

## 0. Apply and build

```bash
cd $REPO && git status --short                      # a clean tree at 89955e0 or later main
git apply --check /path/to/2-minimax-m2.7.patch && git apply /path/to/2-minimax-m2.7.patch

# The default build: host libraries, unit tests, the bind check. The drafts are OFF.
cmake --preset ci        # on a FRESH clone add -DDGPP_BUILD_BENCHMARKS=OFF: main's CMakeLists names
                         # benchmarks/micro/qwen_dense_bench.cu, which is not in git at 89955e0
cmake --build build-ci -j 8 --target unit_tests minimax_bind_check minimax_tool_calls_test tokenizer_test
```

This is the first time GCC sees these files (the Mac used Apple clang and a CUDA header shim), and
the `ci` preset treats warnings as errors: if a GCC-only warning stops it, reconfigure with
`cmake --preset ci -DDGPP_WERROR=OFF` to get on, and fix the warning afterwards. The production
build (`cmake --build --preset release`, target `dgpp_serve_app`) compiles none of the new model
files.

**The draft** (two new kernels and their CUDA test; never compiled by nvcc) is opt-in:

```bash
cmake --preset ci -DDGPP_BUILD_MINIMAX_DRAFT=ON
cmake --build build-ci -j 8 --target minimax_attn_test
# back to the default afterwards: cmake --preset ci -DDGPP_BUILD_MINIMAX_DRAFT=OFF
```

The option adds `dgpp_kernels_minimax_draft` and `minimax_attn_test` only. It does not add a serve
family; the server refuses this model by name either way.

## 1. Host tests (no GPU, no weights)

```bash
cd $REPO
DGPP_TEST_FILTER=minimax ./build-ci/unit_tests        # expect "17 tests, 0 failed"
./build-ci/unit_tests | tail -1                        # the whole suite
```

The 17: the config parser on the real file and its refusals, the registry, the binding table
pinned to 143,471 required tensors / 134,401,534,976 bytes with the release's 47,598 activation
scales (18 absent), the shard list taken from the index, the per-rank weight and K/V bytes at
world 1 and 2, and the host attention reference against the numpy reference's vectors — including
`minimax_two_rank_shards_sum_to_the_single_rank_layer`, the host-level statement of the sharding
plan (each rank's K/V rows are the single rank's slice bit for bit; the two o_proj partial sums add
up to the single-rank output), and `minimax_per_head_finish_is_not_this_models_norm` (GLM-4.7's
existing finish kernel computes a different function: 0.22 relative error on the vectors).

## 2. Bind check against the real checkpoint (headers only), on each node

```bash
$REPO/build-ci/minimax_bind_check --model lukealonso/MiniMax-M2.7-NVFP4          # --world 2 is the default
```

Expected, exactly (after the `model:` line; the count of ignored files is 0 with the download
filter above, 18 when the whole repository was fetched):

```
config: minimax_m2, 62 layers, hidden 3072, vocab 200064
attention: GQA, 48 query heads over 8 kv heads x 128, per-layer q/k norm, rotary 64 (theta 5e+06)
mlp: routed MoE (sigmoid + bias), 256 experts top-8, intermediate 1536, no shared expert
note: config.json declares an MTP head (3 modules); no published checkpoint carries one and the table expects none
checkpoint: 26 shards named by the index, 191069 tensors in their headers; 0 other safetensors files in the directory ignored
binding: required 143471 | matched 143471 (missing 0, dtype 0, shape 0) | unexpected 0
nvfp4 matrices: 47616; activation scales (never loaded): 47598 present, 18 absent; bound 191069 of 191069 tensors; bytes in headers 134401725368, of which the model 134401534976
binding OK: every required tensor is present with its dtype and shape
weights per rank at world 2, as stored (q_proj / k_proj replicated): 69230308352 bytes = 64.476 GiB (routed experts 58.852, attention 3.815, embedding 1.145, head 0.572, routers and norms 0.092)
weights per rank at world 2, as stored (q_proj / k_proj head-sharded): 67864617984 bytes = 63.204 GiB (routed experts 58.852, attention 2.543, embedding 1.145, head 0.572, routers and norms 0.092)
K/V cache per rank: 126976 bytes per token; a 196608-token pool is 23.250 GiB
```

The same validator printed the binding numbers on the Mac from the real `config.json` and a dump of
the real headers. `--world 1` prints 134401534976 bytes = 125.171 GiB: more than one node holds.

## 3. The numpy reference on the real weights (CPU only; the first time it sees them)

It mmaps all 26 shards (125 GiB) and keeps the BF16 weights it has converted (about 12 GB for the
full stack). It cannot run on a node that is serving or benchmarking. Two ways:

```bash
cd $REPO
# (a) a truncated stack — a few layers' pages only; this is the form to compare an engine's
#     truncated forward with, and the one to bisect with:
python3 tools/minimax_m2_reference.py score --ckpt $SNAP --ids-json $IN/minimax-ids-code.json \
        --layers 4 --out $OUT/mm-ref-code-l4.json --threads 8
# (b) the full model, on an idle node (the page cache will hold what fits):
python3 tools/minimax_m2_reference.py score --ckpt $SNAP --ids-json $IN/minimax-ids-code.json \
        --out $OUT/mm-ref-code.json --threads 8
```

Pass for (b): the last line's `mean_logprob` is a small negative number (roughly −1 to −3 nat per
token on this canned answer). Around −12.2 (≈ ln 200064) means the reference reads the container or
the model wrong. (A truncated stack's log-probabilities mean nothing by themselves.) Each of these
must make (b) collapse — they exist to prove one convention at a time:

```bash
for v in ws2=div nibble=hi rope=interleave qknorm=head bias=off w13=up_gate; do
  python3 tools/minimax_m2_reference.py score --ckpt $SNAP --ids-json $IN/minimax-ids-code.json --variant $v --threads 8 | tail -1
done
```

`qknorm=head` is the important one: it is GLM-4.7's norm with this checkpoint's weights, and it
must be clearly worse than the default (the per-layer norm).

## 4. The draft kernel test (needs a GPU, about 1 GB; build with the option ON)

```bash
$REPO/build-ci/minimax_attn_test
```

Expected: `3 tests, 0 failed` — the per-layer q/k finish within 2 bf16 ulps of the host reference
for every head and for each rank's slice of a two-rank world (v bitwise), and the attention step at
48 / 8 heads on GLM-4.7's unchanged split-KV kernels. This test has never been compiled; a compile
error in it is a draft bug, not an engine bug.

## 5. What does not exist yet

There is no loader, no model and no forward check for this family, single-rank or multi-rank; there
is no memory plan to print. The build order and the gate for each step are in
docs/minimax_m27_plan.md §4.5 and §6: shard parity first (each rank's slices against the full
load's views), then a loopback world 2 on one node over a truncated stack with the folded residual
bitwise the world-1 run, then the fabric. The first numerical gate, once a forward check exists, is
`sixlabs/bench/compare_forward_check.py` against the reference's `score` output of the same ids and
the same `--layers` (the reference's JSON already has the schema that script reads).

## 6. The text stack (host; needs the checkpoint's tokenizer.json and chat_template.jinja in the hub cache)

```bash
cd $REPO
./build-ci/tokenizer_test tests/data/minimax_tokenizer_goldens.jsonl
./build-ci/minimax_tool_calls_test tests/data/minimax_chat_template_goldens.jsonl
DGPP_TEST_FILTER=tool_ ./build-ci/unit_tests
```

Expected: the tokenizer gate prints 2 tests, 0 failed (472 cases byte-exact, encode and both
decodes); `minimax_tool_calls_test` prints
4 tests, 0 failed (24 renders byte-exact with matching ids, 2 template refusals, 12 turns / 9 calls
round-tripped through the parser). Exit code 2 from either means "checkpoint not found", not a
pass. All of this passed on the Mac against the real files.

The forced-call grammar does **not** cover this model's tool-call form: `tool_choice: "required"` or
a named tool, `parallel_tool_calls: false` and `response_format` would be refused by the service;
`tool_choice: "auto"` runs unconstrained.

## 7. The server

```bash
cp deploy/cluster_minimax-m2.7_nvfp4_w2.example.json deploy/cluster_minimax-m2.7_nvfp4_w2.json
python3 scripts/dgpp-cluster up --config deploy/cluster_minimax-m2.7_nvfp4_w2.json
```

Expected today: rank 0 exits during startup with

```
serve: <snapshot> is a MiniMax-M2 checkpoint (minimax_m2): recognized, bound and referenced on the host, but it needs two ranks and its device assembly is a plan, not code — see docs/minimax_m27_plan.md and sixlabs/ports/minimax-m2.7/gpu-steps.md
```

That refusal is the designed behaviour: before this patch the registry did not know the class and
the checkpoint was refused as an unsupported architecture; it must never fall through to the GLM
family. When a family exists, the first requests are plain chat at temperature 0 (the stop id is
200020 `[e~[` from `generation_config.json`, **not** `config.json`'s eos 2), a tool call under
`tool_choice: "auto"` (expect `tool_calls[0].function.arguments` as a JSON string and no
`<minimax:tool_call>` text in content), and a reply whose reasoning (`<think>` … `</think>`) is
separated from content.

## 8. Still open after every step above passes

- Option A against option B for the per-layer norm across ranks (the plan, §4.2): a fabric
  measurement; the plan's numbers are estimates from other models' constants.
- Activation quantization (`input_scale`): unused here, as the model card's own launch line
  (`B12X_MOE_FORCE_A16=1`) has it.
- The MTP head `config.json` declares: no checkpoint carries one.
- The forced-call grammar for the `<invoke>` form.
- Everything in §5.
