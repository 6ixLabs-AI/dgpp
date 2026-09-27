# Qwen image prefill: PR #62 review follow-up

PR #62 (`e7d81fd`, based on `652cd40`) splits image staging into windows of
at most 2,049 rows and enables scheduler-visible image prefill chunks. This
removes the vision workspace limit from the language model's 4,096-token
prefill chunks.

Review reproduced two lifecycle regressions with the CUDA text fixture and
a stubbed encoder:

- Cold image prefill registered images before opening the slot, whose reset
  immediately removed that registration. The PR made no encoder calls and
  returned text-only logits; the base commit staged the image correctly.
- Two live image cursors with MTP disabled could reuse the shared staging
  window solely because their token positions matched. Alternating their
  chunks made the second request consume the first request's image rows.

The follow-up keeps the borrowed image vector on each cursor. A scoped
registration exposes it to the model only while synchronous prefill,
begin, or advance is executing. Entry and exit invalidate the shared staging
window, including exception exits. Slot resets no longer erase an executing
synchronous prefill's image borrow. Begin also exposes the images while an
attached prefix's MTP state catches up by one row.

The new `qwen_image_prefill_test` writes a small synthetic BF16 vision tower
alongside the existing text fixture and executes the actual vision encoder
and CUDA model. It checks cold image inputs against text-only input,
different pixels, interleaved cursors with and without MTP, target and draft
logits, cancellation, invalid budgets, text slot reuse, and an attached
prefix whose MTP catch-up consumes the first image row. Synchronous and
resumable comparisons use identical chunk boundaries.

The stock scheduler still advances one unfinished prefill at a time. The
interleaving regression exercises the model API directly; this change does
not add simultaneous image-prefill admission.

## Validation

- Full CI-preset build passed with warnings treated as errors.
- `unit_tests`, `serve_test`, and `scheduler_test` passed, including the
  PR's five staging-plan tests (276 unit cases).
- All 27 host suites excluding checkpoints passed, including 14 Python
  suites. Eight of nine checkpoint-backed host suites passed. The unchanged
  Qwen tool-grammar test rejects newline token 271 in `tool_call_and_response`.
  A separate build of the exact base commit `652cd40` reproduces the same
  failure; this PR changes none of that test's grammar or template sources.
- All three new CUDA cases pass with the real synthetic vision encoder.
  Linking those same tests against the original PR reproduces both cold
  image failures and the missing image row during attached MTP catch-up.
- The remaining native/fixture run passed 129 tests with no failures in
  562.6 seconds; two optional MiMo tokenizer/template cases were skipped.
  This includes Qwen forward, decode, TP, graph/MTP/compaction variants,
  vision kernels, the other model families, and the bus suites.
  Across all 168 CTest entries: 165 passed, two skipped, and the one
  independently reproduced baseline grammar failure described above.
- The existing GLM deployment stopped cleanly before GPU/RDMA execution;
  its four operation streams matched. Four-node Qwen preflight passed.

## Four-node serving validation

The test world used `deploy/cluster_qwen-3.8-flash-next_fp8_w4.json`, with
`prefill_budget_tokens=256` and `prefill_idle_budget_tokens=4096`, four
concurrent slots, MTP and decode graphs enabled, and a 1.5 GiB prefix cache.
The checkpoint revision was `236dfdf285828023ca3bcd3f37366c58a3469b13`.
The CI server binary's SHA-256 was
`87e3e8dcd927fb3fb661d083d8285684456dfe6c6c090536294a90adc36f271a`.

The following checks ran serially against that world:

```bash
python3 scripts/vision_api_check.py --url http://127.0.0.1:18080
python3 scripts/vision_prefix_cache_check.py --url http://127.0.0.1:18080
python3 scripts/prefill_fairness_check.py --url http://127.0.0.1:18080 --images --image-count 12 --decoders 2
python3 scripts/serve_api_check.py 127.0.0.1 18080
```

The vision API and prefix-cache checks passed: cold images, streaming,
multiple choices, different pixels, simultaneous requests, images crossing
staging/chunk boundaries, generated-continuation attachment, and twelve-image
histories. A 6,151-token image prompt reused 6,144 tokens; the twelve-image
12,603-token history reused 12,596 tokens. Cache-disabled requests also
returned the expected colors. Idle image requests used 4,096-token chunks.

Two existing probes needed model-independent input/budget corrections.
The history check now uses 1,024-pixel squares, reaching 1,024 visual tokens
on both families; its previous 896-pixel squares produced only 784 tokens
on Qwen. The fairness probe now allows 256 completion tokens so reasoning
can finish before the required `ready` answer. Its initial 32-token run
observed decode progress but exhausted its output budget without visible
content. The answer and scheduling assertions remain intact.

The twelve-image fairness check passed with a 30,509-token cold prompt in
slot 0 and two active streams in slots 1 and 2. Each stream emitted 244
events during the observed 36.888-second prefill; the largest event gap,
including admission, was 1.248 seconds. The long request returned `ready`
after 112 completion tokens, including 109 reasoning tokens. Both active
streams completed their 1,024-token budgets. This checks continued progress;
it does not establish a latency target or a throughput improvement.

The general request-contract probe passed all stop, multi-choice streaming,
logit-bias and usage checks. No rank reported an engine failure. After a
clean shutdown, all four operation streams had MD5
`eb491668d064b248723240e4a8ee85cc`.

The original four-node GLM service was restored using its unchanged release
binary and `deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json`. All four ranks
were alive, `/health` returned `ok`, `/v1/models` identified the original
GLM checkpoint, and the engine reported no failure.
