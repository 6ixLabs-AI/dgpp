# Gemma-4-31B (`nvidia/Gemma-4-31B-IT-NVFP4`): what to run once a box is free

Status when this was written (2026-10-04, overnight, on a Mac): **the family's host side is
written and tested; nothing has been compiled with nvcc or GCC, no weight has been loaded, no token
produced. There is no serving engine for this family yet** — `6ix-Serve` refuses a Gemma 4
checkpoint by name. What exists on the GPU side is a draft one-sequence forward and its check,
behind a CMake option that is OFF by default.

"Expected" below means derived from the checkpoint's headers (read from Hugging Face by range
request) and from Mac host runs (Apple clang, the CUDA header shim), never from a run on a Spark.

| | |
|---|---|
| checkpoint | `nvidia/Gemma-4-31B-IT-NVFP4` @ `4135a98a9b728a548947683219633b25682223ac`, 4 shards, 32.6 GB on disk (not gated; **not in any box's cache as far as this port knows — download it first**) |
| class | `Gemma4ForConditionalGeneration` (`model_type` gemma4); text only here, the vision tower is skipped |
| container | modelopt 0.37.0 NVFP4: the 180 MLP matrices NVFP4 (with an `input_scale` each, unused), all attention projections, norms and the embedding BF16; no `lm_head` tensor (tied) |
| reference | `tools/gemma4_reference.py` (numpy), checked against transformers 5.8.1's own class on a synthetic checkpoint |

`$REPO` = a build copy with the patch applied, `$OUT` = scratch, `$SNAP` = the snapshot directory,
`$IN` = `$REPO/sixlabs/ports/gemma-4-31b/inputs` (84 and 114 token ids, made with HF tokenizers and
jinja2 from the two shared prompts; the engine's tokenizer and template reproduce both).

## 0. Build

The default build carries everything that was tested on the Mac:

```bash
cd $REPO
cmake --preset ci && cmake --build build-ci -j 8 --target unit_tests tokenizer_test gemma4_chat_test \
      gemma4_bind_check gemma4_host_check
```

The draft GPU side is opt-in. Turn it on in a build directory of its own, so that a failure there
cannot touch the build the benchmarks use:

```bash
cmake --preset ci -B build-gemma -DDGPP_BUILD_GEMMA_DRAFT=ON
cmake --build build-gemma -j 8 --target gemma4_forward_draft_test gemma4_forward_check
```

`DGPP_BUILD_GEMMA_DRAFT=ON` adds three targets and nothing else: `dgpp_models_gemma4_draft`
(`src/models/gemma4/forward_draft.cu`), `gemma4_forward_check`, `gemma4_forward_draft_test`.
It does not change `6ix-Serve`.

## 1. Host tests (no checkpoint, no GPU)

```bash
cd $REPO
DGPP_TEST_FILTER=gemma4 ./build-ci/unit_tests          # expect: 42 tests, 0 failed (35 with the first patch alone:
                                                       # the 26B-A4B port adds seven)
DGPP_TEST_FILTER=chat_template_ ./build-ci/unit_tests  # expect: 10 tests, 0 failed
./build-ci/unit_tests                                  # everything: 0 failed (regression: the tokenizer,
                                                       # the template interpreter and the tool parser are shared)
```

On the Mac these 52 passed, together with the 90 existing tool-parser / tool-grammar / JSON-grammar
/ minijson / utf8 unit tests built beside them. A failure here on Linux is a GCC-versus-clang
difference in host code, not a model problem.

## 2. Checkpoint, tokenizer and template (headers and text files only)

```bash
hf download nvidia/Gemma-4-31B-IT-NVFP4        # 32.6 GB
$REPO/build-ci/gemma4_bind_check --model nvidia/Gemma-4-31B-IT-NVFP4
ctest --test-dir build-ci -R 'gemma4_tokenizer_test|gemma4_chat_test' --output-on-failure
```

Expected from the bind check (this exact text was produced on the Mac by the app itself, from the
real config.json and header-only stand-ins of the four shards — each shard's real 8-byte length and
header JSON followed by a hole of the real size; no weight was downloaded):

```
config: gemma4, 60 layers (50 sliding + 10 full), hidden 5376, vocab 262144, recipe nvfp4-mlp
attention: 32 heads; sliding 16 KV heads x 256, window 1024, 128 rotary pairs (theta 10000); full 4 KV heads x 512, 64 rotary pairs (theta 1e+06), value from k_proj
mlp: dense GeGLU, intermediate 21504; embedding scale 73.5; logit soft-cap 30
checkpoint: 4 shards, 1728 tensors in headers
binding: expected 1372 | matched 1372 (missing 0, dtype 0, shape 0) | unexpected 0 | ignored (encoders) 356
matrices: 180 NVFP4 (0 of them experts), 231 BF16; matched bytes 31481767960
table 1372 + ignored 356 = 1728; headers 1728
binding OK: every tensor of the table is present with its dtype and shape
```

Expected from the two tests (both ran on the Mac against the release's tokenizer.json and
chat_template.jinja):

```
glm_tokenizer differential: 198/198 cases byte-exact (revision 0xf1a1b0db90a1fd04)
gemma4_chat_test: 30 renders byte-exact, 30 id sequences equal (model nvidia/Gemma-4-31B-IT-NVFP4, template c9fa6ee62cd50023)
gemma4_chat_test: 10 tool-call turns (13 calls) round-trip render -> encode -> parse; 1 undecidable block flushed as text
```

A `revision` or `template` hash that differs means the repository moved: regenerate the goldens
(`tools/gen_tokenizer_goldens.py`, `tools/gen_chat_template_goldens.py`, both with
`--model nvidia/Gemma-4-31B-IT-NVFP4`) and read the diff before trusting anything else here.

## 3. The numpy reference on the real weights (CPU, about 10 GB of RAM)

```bash
python3 $REPO/tools/gemma4_reference.py score --ckpt $SNAP --ids-json $IN/gemma4-ids-prose.json \
        --out $OUT/ref-prose.json --threads 8
python3 $REPO/tools/gemma4_reference.py score --ckpt $SNAP --ids-json $IN/gemma4-ids-code.json \
        --out $OUT/ref-code.json --threads 8
```

It converts one layer at a time (about 2 GB of fp32 at a time, 60 layers, the head in row chunks).
Not timed on a Spark; an estimate is a few minutes per prompt.

**What a pass looks like.** There is no second implementation on the box to compare with, so this
step is a sanity gate, not a parity gate: the prompts are ordinary fluent text, and a model that
reads them correctly finds each next token likely. A model that knows nothing scores ln(1/262144) =
−12.5 per token; a wrong convention anywhere in the stack lands near that. Expect a mean far above
it and most `rank` values 0 or single digits (no number was measured for this model: record the
two means when this first runs). If the mean is near −12, do not go on to step 4: flip one
convention at a time (`--variant norm=1p`, `embscale=exact`,
`keqv=normed`, `globalrope=full`, `window=plus1`, `nibble=hi`, `ws2=div`, `qkscale=sqrt`) and see
which one brings it down. Every one of those conventions was checked against the transformers
class, but only on synthetic weights; this run is the first time the reference meets the release.

## 4. The C++ host reference on the real weights (CPU, no GPU)

```bash
$REPO/build-ci/gemma4_host_check --model nvidia/Gemma-4-31B-IT-NVFP4 --ids-json $IN/gemma4-ids-prose.json \
      --layers 4 > $OUT/host-prose-l4.txt
python3 $REPO/tools/gemma4_reference.py score --ckpt $SNAP --ids-json $IN/gemma4-ids-prose.json \
        --layers 4 --out $OUT/ref-prose-l4.json
python3 $REPO/sixlabs/bench/compare_forward_check.py $OUT/host-prose-l4.txt $OUT/ref-prose-l4.json --mean 0.01 --max 0.05
```

`--layers 4` keeps it to about 14 GB of fp32 (2 GB a layer, 5.6 GB for the embedding) and, at
roughly half a second per token per layer on one core, some minutes. Both sides are fp32 with no
rounding to bf16, so the expectation is agreement far inside the bounds (on the synthetic checkpoint
the two agree to 2e-5 on every residual value); `PASS` and `greedy next token: 84/84 rows agree` is
the pass. This gates the tensor table, the NVFP4 decode and the C++ operators on real weights with
no kernel involved.

## 5. The draft GPU forward

First the synthetic checkpoint — it needs no weights and is the test the draft already passes when
its `.cu` is compiled as plain C++:

```bash
cd $OUT && $REPO/build-gemma/gemma4_forward_draft_test      # expect: 4 tests, 0 failed (3 with the first patch alone)
```

Then the release:

```bash
$REPO/build-gemma/gemma4_forward_check --model nvidia/Gemma-4-31B-IT-NVFP4 \
      --ids-json $IN/gemma4-ids-prose.json > $OUT/fc-prose.txt
python3 $REPO/sixlabs/bench/compare_forward_check.py $OUT/fc-prose.txt $OUT/ref-prose.json --mean 0.01 --max 0.05
$REPO/build-gemma/gemma4_forward_check --model nvidia/Gemma-4-31B-IT-NVFP4 \
      --ids-json $IN/gemma4-ids-code.json > $OUT/fc-code.txt
python3 $REPO/sixlabs/bench/compare_forward_check.py $OUT/fc-code.txt $OUT/ref-code.json --mean 0.01 --max 0.05
```

The check uploads the checkpoint's own bytes (about 29.3 GiB) plus an fp32 K/V cache for the
sequence (about 1.8 MiB a token), runs one thread per output element — slow by design — and prints
the same `h_NN: residual rms … max …` line per layer as `gemma4_host_check`, then `argmax ids:` and
`token logprobs:`. Pass: `PASS` from the comparison on both prompts. If it fails, run the host
check and the forward check with the same `--layers N` and `diff` their `h_NN` lines: the first
layer whose line differs is where to look (layer 5 is the first full-attention layer).

Do not run it beside a benchmark: it takes the GPU and about 31 GiB of the box's memory.

## 6. First request

There is none to make. `6ix-Serve` answers a Gemma 4 checkpoint with

```
serve: gemma4: this engine cannot serve the Gemma 4 family yet — …
```

and that is the designed behaviour until the family has an engine. What the engine needs that is
not written: resident weights in the kernels' formats, a paged K/V pool (the ten full layers page;
the fifty sliding layers want a 1024-row ring per slot — paging them would cost 800 KiB a token),
an attention kernel for 256- and 512-wide heads with the q/k/v norms and both rotary kinds, the
graph decode, sampling (the model card's defaults are temperature 1.0, top-k 64, top-p 0.95), a
grammar for the call notation (without it `tool_choice` required / named and JSON-schema outputs
are refused; `auto` works through the parser), and a `Gemma4Family` in `apps/dgpp_serve.cpp` with a
`TextFrontend` given `bos_token` `"<bos>"` and the stop ids 1, 106 and 50 (`<eos>`, `<turn|>`,
`<|tool_response>`: generation_config.json's list). `deploy/cluster_gemma-4-31b_nvfp4_w1.example.json`
is the shape that deployment is meant to take; nothing reads it today.

## What was and was not verified

Ran on the Mac (2026-10-04):

- the config parser and the tensor table on the real config.json and the real headers (table 1372 +
  356 ignored = 1728 headers; byte totals equal);
- the numpy reference against transformers 5.8.1's `Gemma4ForCausalLM` on a synthetic checkpoint in
  the release's container: max |logit difference| 9e-6 whole-sequence and token by token, and every
  one of 18 wrong conventions moves the logits by 2e-3 to 4.7;
- the C++ host reference and the draft forward (as plain C++) against the numpy reference on that
  checkpoint;
- the tokenizer against HF tokenizers 0.22.2 on the release's tokenizer.json: the 198-case corpus,
  and 6,002 random texts (452,110 ids) outside the tree;
- the template interpreter against jinja2 3.1.6 on the release's chat_template.jinja: 30 cases;
- the call notation: 13 calls rendered by the template and parsed back.

Not verified, in order of how likely each is to bite:

1. `forward_draft.cu` under nvcc: the `G4_LAUNCH` macro's launch line, `__global__` kernels in an
   anonymous namespace, `cos` / `sin` / `powf` / `expf` / `tanhf` and the `DGPP_HD` decoders in
   device code, a 1-D grid of up to T x 262144 / 256 blocks for the head.
2. The host code under GCC (it was compiled with Apple clang, `-Wall -Wextra -Wpedantic`, no
   warnings): `std::format` uses, the lambda in `std::stable_sort`, `<unistd.h>` in two tests.
3. Any number on the real weights: the reference has never read the release's tensors. Steps 3–5
   are the first time.
4. The embedding scale. The reference casts sqrt(5376) to bf16 (73.5), as transformers does when the
   table is bf16; an implementation that keeps 73.32 is 0.25 % off on the embedding's share of the
   residual. `--variant embscale=exact` shows the size of it.
5. Speed and memory of everything in steps 3–5: estimates only.
