# Port 5 — Nemotron-3 (Nano, Super): what to run once a box is free

Status when this was written (2026-10-04, overnight): **host side only, and only (a)–(c).** The
config parser, the tensor table, a numpy reference and a host reference of the Mamba2 mixer exist
and pass their tests on the Mac. **There is no CUDA kernel, no loader, no model assembly, no serve
wiring and no deployment template — (d) was not started.** Nothing has run on a GPU, nothing has
been built with the real toolchain, and the numpy reference has **never run on the real weights**
(only on synthetic checkpoints). Every "expect" below is what the code should print, derived from
the checkpoints' headers and from Mac runs.

- Checkpoints (DGXone paths; DGXtwo sees them under `/mnt/dgxone-hf/hub/...`):
  - `NANO=/home/mark/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/snapshots/6efb4a2a1c1fa277ce7b3df7a1416255011b1c99`
    (5 shards, 18.0 GiB)
  - `SUPER=/home/mark/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/snapshots/4f0cf9daaeb7a4d5e23f80a00e7ed15f0e03caf6`
    (17 shards, 74.8 GiB)
- Patch: `/Users/markgriffith/6ix-ports-overnight/5-nemotron-3.patch` — Nemotron only (18 files,
  listed at the end), against `main` at `a558c9f`. It applies with or without port 4's patch
  (`4-qwen3.5-122b-a10b.patch`), in either order (checked with `git apply --check` both ways).
  Facts about the checkpoints: `5-nemotron-notes.md` beside this file. Architecture:
  `docs/nemotron3_plan.md`.

**What the patch puts in the default build, and what it does not.**
- In the default build: one static library of three host sources (`dgpp_models_nemotron`:
  `config.cpp`, `binding.cpp`, `mamba2_reference.cpp` — no CUDA, linked only by `unit_tests`) and
  three unit-test files added to `unit_tests`. All of it was compiled (Apple clang 17, the
  repository's `-Wall -Wextra -Wpedantic`, zero warnings) and its 19 tests pass on the Mac.
- `6ix-Serve` gains no family: `loaders/architecture` recognises `NemotronHForCausalLM` /
  `nemotron_h`, and `make_family` in `apps/dgpp_serve.cpp` **refuses it by name** ("NemotronH
  (Nemotron-3) checkpoints bind but are not served yet…") instead of letting it fall through to
  the GLM family. The serve binary's registry is otherwise untouched.
- **There is no draft kernel in this patch and therefore no CMake option.** Part (d) — the Mamba2
  CUDA kernels, the loader and the model assembly — was not written: it cannot be compiled or
  checked here, and §6 below is the specification for it. When it is written it must not be able
  to break the default build: put its sources behind
  `option(DGPP_BUILD_NEMOTRON_DRAFT "Nemotron-H draft kernels, not yet compiled on a Spark" OFF)`,
  add the family to `make_family` only under `#ifdef DGPP_BUILD_NEMOTRON_DRAFT` (a
  `target_compile_definitions` on the serve target inside the `if(DGPP_BUILD_NEMOTRON_DRAFT)`
  block), and turn it on with `cmake --preset ci -DDGPP_BUILD_NEMOTRON_DRAFT=ON` (or
  `--preset release -D…`) in a build directory of its own.
- `$REPO` is a build copy with the patch applied (not the production tree), `$OUT` a scratch
  directory, `$PY` a python3 with numpy (and, for step 3, `tokenizers` from
  `requirements-tools.txt`).

Steps 1–2 need no GPU and no weights. Steps 3–6 need **no GPU either** — they are CPU and memory
only — but they read a whole checkpoint through the page cache, so run them when no engine is
resident on that box (18 GiB for Nano, 75 GiB for Super, on top of the process's own memory).

## 0. Apply, format, build

```bash
cd $REPO && git status --short                      # expect a clean tree
git apply --check /path/to/5-nemotron-3.patch && git apply /path/to/5-nemotron-3.patch

# The new files are formatted with the repository's .clang-format by Xcode's clang-format 17; if
# the box's version disagrees, run it again:
clang-format -i src/models/nemotron/*.{hpp,cpp} tests/unit/nemotron_{config,binding,mamba2}_test.cpp \
                tests/unit/nemotron_config_json.hpp      # NOT nemotron_mamba2_vectors.hpp (generated)

cmake --preset ci                                   # warnings are errors in this preset
cmake --build build-ci -j 8 --target dgpp_models_nemotron unit_tests
```

`dgpp_models_nemotron` is three host sources (`config.cpp`, `binding.cpp`,
`mamba2_reference.cpp`); it links nothing CUDA. If the `ci` build stops on a warning in one of the
files listed at the end, that is the first thing to fix: they have only been compiled by Apple
clang 17 (`-Wall -Wextra -Wpedantic`, zero warnings), never by GCC.

## 1. Host tests

```bash
cd $REPO
DGPP_TEST_FILTER=nemotron ./build-ci/unit_tests     # expect 19 tests, 0 failed
./build-ci/unit_tests | tail -3                     # the whole suite, as before
```

Two of the 19 read the real checkpoints when they are in `~/.cache/huggingface/hub` (they return
silently when not):

- `nemotron_config_reads_the_landed_checkpoints` — parses the real `config.json` (+
  `hf_quant_config.json`) and requires the parsed recipe to equal the transcribed one.
- `nemotron_binding_matches_the_landed_checkpoints` — opens every shard (mmap, header pages only),
  refuses a name in two shards, and requires table == headers: **24,147** tensors /
  19,339,781,632 bytes / 5,968 NVFP4 matrices for Nano; **165,860** / 80,297,329,824 / 40,961
  NVFP4 + 139 FP8 for Super.

On the Mac these two were run against a header-only copy of the real files (the real headers
with sparse, never-written bodies) and passed; a renamed tensor in that copy failed the test.

Pass = `19 tests, 0 failed` on a box that has both snapshots.

## 2. The numpy reference on its synthetic checkpoints (seconds, ~30 MB)

```bash
cd $REPO
$PY tools/nemotron_synth.py ckpt --preset nano  --seed 0 --out $OUT/synth-nano
$PY tools/nemotron_synth.py ckpt --preset super --seed 0 --out $OUT/synth-super
$PY tools/nemotron_synth.py selftest --ckpt $OUT/synth-nano      # expect 6 x [PASS], "SELFTEST OK"
$PY tools/nemotron_synth.py selftest --ckpt $OUT/synth-super     # expect 7 x [PASS], "SELFTEST OK"

# The Mamba2 test vectors regenerate byte for byte (they did on the Mac, numpy 2.0.2):
$PY tools/nemotron_synth.py mamba-vectors --ckpt $OUT/synth-nano --layer 0 --tokens 11 --seed 7 \
    --out $OUT/vectors.hpp
cmp $OUT/vectors.hpp tests/unit/nemotron_mamba2_vectors.hpp && echo same
```

If `cmp` differs on the box, look at the size of the difference before anything else: the inputs
are drawn with `numpy.random.RandomState` (a frozen stream) and printed at 9 digits, the expected
values at 17, so another libm can move last digits of the expected values without meaning
anything. The unit test's own tolerance (1e-12) is the judge.

What the self-test shows: the readers invert the writers (78 NVFP4 matrices on `nano`; 65 NVFP4 +
12 FP8 on `super`, max |diff| 0); the Mamba recurrence equals an independent transcription of the
reference's chunked SSD algorithm on every Mamba layer at chunk 4 / 5 / 64 (Mac: 4e-16); whole,
chunked and token-by-token feeding agree (Mac: 7e-15); greedy generation reproduces teacher
forcing; float32 stays within 4e-6 of float64 on the logits. What it does **not** show: that the container
conventions are the real ones (the writers and readers share one reading), or anything about real
weights.

On the Mac, numpy 2.0.2 over Accelerate prints spurious `RuntimeWarning: divide by zero / overflow
/ invalid value encountered in matmul` on finite inputs (the products equal `einsum`'s exactly);
the Mac runs used `PYTHONWARNINGS=ignore::RuntimeWarning`. On the box (OpenBLAS) such a warning
would be real — do not suppress it there.

## 3. Token ids for the real checkpoints (untested: written without the tokenizer at hand)

The reference takes a JSON list of ids. Make two inputs with the checkpoint's own tokenizer —
plain prose, and a chat-formatted turn (the template always emits a system block and no `<s>`):

```bash
$PY - <<'EOF'
import json, os
from tokenizers import Tokenizer
snap = os.environ["NANO"]                      # the two releases' vocabularies are the same size;
tok = Tokenizer.from_file(snap + "/tokenizer.json")   # make the Super inputs from $SUPER to be safe
prose = ("The Amazon rainforest covers much of northwestern Brazil and extends into Colombia, Peru "
         "and other South American countries. It is the largest tropical rainforest in the world "
         "and is famed for its biodiversity. It is crisscrossed by thousands of rivers, including "
         "the powerful Amazon.")
chat = ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>The capital of France is Paris.<|im_end|>\n")
out = os.environ["OUT"]
for name, text in (("prose", prose), ("chat", chat)):
    ids = tok.encode(text, add_special_tokens=False).ids
    json.dump(ids, open(f"{out}/nemotron-{name}.json", "w"))
    print(name, len(ids), ids[:12])
EOF
```

Check before using them: the chat ids must start with `10` (`<|im_start|>`) and contain `11`
(`<|im_end|>`) and `12, 13` (`<think>`, `</think>`) as single tokens. If the special strings come
out split into pieces, build the list by hand around those ids.

## 4. The reference on the real Nano (CPU only)

```bash
cd $REPO
$PY tools/nemotron_reference.py score --ckpt $NANO --ids-json $OUT/nemotron-prose.json \
    --threads 8 --out $OUT/nano-prose.json | tail -5
$PY tools/nemotron_reference.py score --ckpt $NANO --ids-json $OUT/nemotron-chat.json \
    --threads 8 --out $OUT/nano-chat.json | tail -5
```

Memory and time (estimates, not measurements): about 7–8 GB of float32 weights kept in the
process plus the 18 GiB checkpoint passing through the page cache; a 60-token text should take a
few minutes (the routed experts are dequantized on demand: up to `min(128, 6 x tokens)` experts a
MoE layer, 23 layers). `--layers N` runs a prefix of the stack for a quick first look; `--no-keep`
trades time for memory.

**Pass** = the last line reads like a language model, not like noise:

- `mean_logprob` on the prose well above the uniform floor of `-ln(131072) = -11.78` — a working
  LM should be somewhere around -1.5 to -3.5 nats a token on plain English, with `rank 0` on most
  positions after the first few;
- on the chat input, the answer tokens (`The capital of France is Paris.`) at rank 0 or 1 with
  logprobs near 0 once the sentence is under way.

If it reads like noise (`mean_logprob` near -10 or worse), a convention is wrong. Sweep them — each
run flips one; the **right** convention is the one that is likely:

```bash
for v in nibble=hi ws2=div conv=reversed grpmap=mod gatenorm=norm_first kvmap=mod \
         pos=rope dtclamp=min rbias=off renorm=off act=silu; do
  echo "$v: $($PY tools/nemotron_reference.py score --ckpt $NANO --ids-json $OUT/nemotron-prose.json \
              --threads 8 --variant $v | tail -1)"
done
```

Expected, if the defaults are right: every variant is clearly worse than the default except
`dtclamp=min`, which should be equal or within noise (it only acts on time steps under 0.001).
Two results would overturn something written down as fact and should stop the port until
understood: **`pos=rope` better than the default** (then the attention does use rotary after all),
or **`dtclamp=min` clearly better** (then the CPU fallback's clamp is the trained behaviour).

Then the state plumbing on real weights and a first generation:

```bash
$PY tools/nemotron_reference.py score --ckpt $NANO --ids-json $OUT/nemotron-prose.json --threads 8 \
    --chunk 1 | tail -1          # expect the same mean_logprob as the whole-sequence run (to ~1e-4)
$PY tools/nemotron_reference.py generate --ckpt $NANO --ids-json $OUT/nemotron-chat.json --threads 8 \
    --max-new 16                 # prints ids; decode them with the tokenizer and read them
$PY tools/nemotron_reference.py dump --ckpt $NANO --ids-json $OUT/nemotron-prose.json --threads 8 \
    --out $OUT/nano-dump --detail-layers 0,1,5      # the tensors an engine port is compared to
```

This still leaves the reference unverified against **another implementation**. The strongest
check available is the one used for Qwen3-Next: teacher-forced logprobs from a server that runs the
model's own code (vLLM with `--trust-remote-code`, or transformers + `mamba_ssm`) on the same ids,
compared position by position. Until that exists, "the reference is right" means only "it is a
plausible LM and every wrong convention is worse".

## 5. The reference on the real Super (CPU only)

As step 4 with `--ckpt $SUPER`, plus `--no-keep` unless the box has ~35 GB to spare for the
process (kept float32 weights are about 29 GB: 40 x `in_proj` 304 MB, 40 x `out_proj` 134 MB,
40 shared experts 176 MB, the head 2.1 GB). With `--no-keep` the process stays at a few GB and
chunked feeding re-dequantizes every layer per chunk, so score whole sequences. The checkpoint is
75 GiB through the page cache. A 60-token text: perhaps 10–20 minutes (estimate; up to 512 experts
a MoE layer, 40 layers).

Add `fp8scale=div` to the sweep (Super has 139 FP8 matrices; Nano none).

The draft block — the one part with no reference code at all:

```bash
for cat in eh he; do
  $PY tools/nemotron_reference.py mtp --ckpt $SUPER --ids-json $OUT/nemotron-prose.json --threads 8 \
      --no-keep --variant mtpcat=$cat --out $OUT/super-mtp-$cat.json | tail -3
done
```

It prints, for each of `pre_norm` / `post_norm` (the hidden fed to `hnorm` taken before or after
`norm_f`) x `history` / `no_history` (the draft attention reading the earlier pairs' K/V or only
its own), how often the draft's top-1 equals the main model's own next token. **Pass** = one
combination of (`mtpcat`, hidden, history) clearly ahead, at a rate a draft head should reach
(Qwen3-Next's was 60–80% on prose); that combination is then the convention. If all eight are near
zero the block's form is wrong (not just a convention) — read vLLM's Nemotron-H MTP code.

## 6. What (d) still needs

Nothing below exists. Signatures are proposals in the style of `src/kernels/kda.hpp`.

### 6.1 Loader (`src/models/nemotron/loader.{hpp,cpp}`)

- NVFP4 sets: reuse `Fp4Set`, `load_fp4_slice`, `qwen3next_fp4_dequant_bf16`
  (`src/models/qwen/loader35.cpp`) — same modelopt container. Routed experts stay 4-bit for
  `fp4_gemv`; Nano's NVFP4 `in_proj` / `out_proj` and shared expert either stay 4-bit or go to BF16
  / FP8 at load as Qwen3-Next's dense projections do (`engine.dense_weights`).
- Per-tensor FP8 (Super: 139 matrices) is **new**. `e4m3(code) x F32 scalar` does not fit BF16
  exactly, so "dequantize to BF16" rounds. Exact alternative: keep the codes and hand the block-
  scale FP8 kernels a scale grid filled with the one scalar.
- The format of each matrix comes from `NemotronHConfig::quant_of(module)`; Super differs layer by
  layer, so the loader cannot assume a class has one format.
- Router gate: F32 on Nano, BF16 on Super (`cfg.router_dtype`).
- `k_proj.k_scale` / `v_proj.v_scale` (Super): read only for an fp8 K/V pool.

### 6.2 Mamba2 kernels (`src/kernels/mamba2.{hpp,cu}`), oracle `mamba2_ref`

```cpp
// Depthwise causal conv WITH BIAS + SiLU over [x | B | C]; state layout as KDA's conv state.
// kda_causal_conv_silu_bf16 has no bias argument; channels are 6144 (Nano) / 10240 (Super).
void mamba2_causal_conv_silu_bf16(const void* src, int64_t src_row_stride, const void* weight,
                                  const void* bias, void* conv_state, int state_width, void* dst,
                                  int tokens, int channels, int conv_width, cudaStream_t stream,
                                  const KdaConvSnapshots& snap = {});
// + _batched(..., const KdaRequestRows& requests, ...)

// The recurrence. xbc: bf16 [tokens, inner + 2*groups*N] (post-conv); dt_raw: bf16 [tokens, H]
// with a row stride (in_proj's last H columns); a_log, d, dt_bias: fp32 [H];
// state: fp32 [H, head_dim, N] in/out, N contiguous; out: bf16 [tokens, H * head_dim].
void mamba2_scan_fwd(const void* xbc, const void* dt_raw, int64_t dt_row_stride, const float* a_log,
                     const float* d, const float* dt_bias, float* state, void* out, int tokens,
                     int heads, int head_dim, int state_dim, int groups, cudaStream_t stream,
                     const KdaStateSnapshots& snap = {}, const KdaReplay& replay = {});
// + _batched(..., float* states, int64_t request_state_stride, ..., const KdaRequestRows&, ...)

// y = rms_over_groups(x * silu(gate)) * w; gate rows have a stride (in_proj's first columns).
void mamba2_gated_group_rmsnorm_bf16(const void* x, const void* gate, int64_t gate_row_stride,
                                     const void* weight, void* y, int64_t rows, int inner,
                                     int groups, float eps, cudaStream_t stream);
```

- Decode step: one row per request, `heads x head_dim` independent rows of `N = 128` FMAs — small.
- Prefill: the same recurrence sequential in time and parallel over `heads x head_dim` is correct
  at any length; the tensor-core form is the chunked SSD algorithm (the analogue of
  `gdn_chunk.cu`), a later lever. `mamba2_ref::mixer` is the oracle for both.
- Rounding points to hold the kernels to: `Rounding::Bf16` in `mamba2_reference.hpp` (conv output,
  scan output, norm output in BF16; state fp32). They were read off the reference's dtype flow and
  have **not** been compared with a device.

### 6.3 State layout and snapshots

Per request slot, per Mamba layer (ordinal `m` of 23 / 40), the `KdaStatePool` layout
`[slot][layer][recurrent fp32][conv bf16]`:

| | Nano | Super |
|---|---|---|
| SSM state fp32 `[heads, head_dim, 128]` | 524,288 floats = 2 MiB | 1,048,576 floats = 4 MiB |
| conv tail bf16 `[channels, 3 + spec]` | 6144 x (3 + spec) | 10240 x (3 + spec) |
| per slot, all Mamba layers | 46 MiB + 0.9 MB | 160 MiB + 2.5 MB |

- Prefix cache: a snapshot is the whole slot (`Qwen35Model::session_snapshot_bytes` pattern) —
  46 / 160 MiB an entry plus the K/V blocks. That is large next to the K/V it sits beside
  (1,024 bytes a token per attention layer: 6 KB / 8 KB a token).
- Speculative decode (Super's MTP): post-row state snapshots (`KdaStateSnapshots`) cost 4 MiB a
  layer a row — 160 MiB a verify row. Use the checkpoint-and-replay form (`KdaReplay`): keep the
  pass's rows as their inputs (the post-conv row, 10,240 BF16, and `dt`, 128) and re-run them.
- `model.{hpp,cpp}`: `mamba_rec(slot, ord)` / `mamba_conv(slot, ord)` accessors as
  `Qwen35Model::gdn_rec` / `gdn_conv`; `write_state_snapshot` / `read_state_snapshot`.

### 6.4 Attention

GQA, 32 query heads, 2 KV heads, head_dim 128, **no rotation, no q/k norm, no output gate**.
`QwenFullAttnLayer` (`src/models/qwen/layers.hpp`) implements 256-wide heads with all three; the
GLM-4.7 path (`src/kernels/glm4_attn.*`) is paged GQA with partial RoPE. Needed: one of them with
the rotation off at 128-wide heads — check whether `rotary_dim = 0` is accepted before writing a
new kernel. K/V pool: the paged pool, 2 x 128 BF16 per token for k and for v.

### 6.5 MoE

- Router: `MoeRouterMode::SigmoidBias` as is (`src/models/glm/moe.hpp`) — the same rule, bias and
  1e-20 included. Gate rows are F32 on Nano (the GLM router takes a BF16 gate).
- **`top_k` 22** (Super) against `GlmMoeConfig::validate_config`'s cap of 16.
- Experts: two matrices and `relu(x)^2`; the engine's chains are three-matrix SwiGLU
  (`src/kernels/glm_moe.cu`, `qwen_moe.cu`). A new activation / slot form, NVFP4 weights
  (`fp4_gemv`).
- Shared expert: weight 1 (GLM's rule), but its inner width differs from the experts' (3712 vs
  1856; 5376 vs 2688) and on Super its two matrices are FP8, NVFP4 or BF16 depending on the layer.
- Latent (Super): `fc1_latent_proj` (4096 -> 1024) before the routed chain and `fc2_latent_proj`
  (1024 -> 4096) after the weighted sum; expert K is 1024, N is 2688.

### 6.6 Draft block, text, serving

- MTP once step 5 has fixed its conventions; one block applied recursively (the model card's
  recipe drafts three tokens).
- Tokenizer: BPE with a Unicode-class `Split` regex and `ignore_merges`; needs goldens before it is
  trusted. Template: ChatML with the always-present system block, `<think>` handling and
  history truncation. Tool calls: the `<function=…><parameter=…>` form (Qwen3-Coder-Next's parser).
  Stop ids 2 and 11.
- `apps/dgpp_serve.cpp`: a `NemotronFamily`; **today `make_family` has no branch for
  `ModelArchitecture::NemotronH` and falls through to `GlmFamily`**, so serving a Nemotron
  checkpoint now fails with "missing text_config object" instead of the old "unsupported
  architecture" refusal. Add an explicit refusal there until the family exists.
- `deploy/cluster_nemotron3_*_w1.example.json`, a bind-check app (the scratch tool
  `bind_from_tsv.cpp` is the model), TP geometry (`world` must divide the Mamba heads and keep
  whole B/C groups per rank).

## 7. Likely to break first on a real build

1. GCC `-Werror` in the new files (Apple clang only so far).
2. `clang-format --dry-run --Werror` (the `format-check` target) on the new files if the box's
   clang-format lays something out differently from Xcode's 17 (the tree as a whole does not pass
   that target today, so nothing gates on it).
3. Nothing else in the build should notice the port: no existing target gained a source file
   except `unit_tests` (three test files, one more library on its link line), and `6ix-Serve`
   gained one refusal branch.
4. Another port's patch touching the same lines: every new family adds an enum value in
   `loaders/architecture.hpp`, a detection branch in `architecture.cpp` and a branch in
   `make_family` — adjacent additions that `git apply` may reject and a three-way merge resolves
   by keeping both.

## Files

New: `src/models/nemotron/{config,binding,mamba2_reference}.{hpp,cpp}`,
`tools/nemotron_reference.py`, `tools/nemotron_synth.py`,
`tests/unit/nemotron_{config,binding,mamba2}_test.cpp`, `tests/unit/nemotron_config_json.hpp`,
`tests/unit/nemotron_mamba2_vectors.hpp` (generated), `docs/nemotron3_plan.md`.

Edited (small, additive): `src/loaders/architecture.hpp` (enum value `NemotronH`, its name),
`src/loaders/architecture.cpp` (detection branch), `apps/dgpp_serve.cpp` (the refusal by name in
`make_family`), `CMakeLists.txt`
(the `dgpp_models_nemotron` library after `dgpp_models_mimo`; a `target_sources` /
`target_link_libraries` pair after the `unit_tests` target).
