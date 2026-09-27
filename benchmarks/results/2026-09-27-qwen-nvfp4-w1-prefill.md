# Single-GB10 NVIDIA Qwen NVFP4 prefill budgets

Observed on 2026-09-26 UTC / 2026-09-27 CEST. These are results from an
operator's completed runs, not a new controlled benchmark campaign.

## Configuration

- Official DGPP source `652cd40331c344c2a51f153768d7d3e23601f465`, fresh Release build.
- One GB10, `nvidia/Qwen3.8-Flash-Next-NVFP4`.
- Four slots, MTP depth 1, BF16 KV, FP8 dense weights/MMA head,
  `bf16_weights: bf12+bf16`, mapped n-gram table, 1.5 GiB prefix snapshots.
- Site-specific shared KV capacity 850048, not the base example's 65536.
  Both measured budget settings used that same pool. The proposed example
  instead uses 262144; these measurements do not directly benchmark that new
  default pool size.
- The initial automatic busy budget resolved to 256; idle reused that budget.
  The later configuration explicitly set both budgets to 4096.

The code already supports 4096-token Qwen prefills through merged PR #58.
The measurements use the same official engine revision and MTP depth.

## Observed throughput

`tool-eval-bench 2.7.1.dev7+gbd35ba91b` with `llama-benchy 0.4.0`,
`pp2048 tg128 @ d16384`, concurrency one. Actual cold inputs were about
18.5K tokens including depth and formatting, not just the 2048-token label.

| Busy / idle budget | PP tok/s | TG tok/s | TTFT |
| --- | ---: | ---: | ---: |
| 256 / reuse busy | 593 | 43.3 | 31.364 s |
| 4096 / 4096 | 1346 | 43.9 | 13.813 s |

The 256 row is from a three-repetition sweep; the 4096 row is a single
measurement with different generated prompt content. These observations
support evaluating larger chunks, but are not a matched repeated A/B or a
performance guarantee for either the old 64K-pool template, the proposed
256K-pool default or other machines.

At 4096/4096, a concurrency-two cold-prompt example reported 1322 PP tok/s
and 14.2 aggregate TG tok/s. Server logs showed about 14 seconds of the
second request's prefill inside the first-to-last-output measurement window,
with only 3.454 seconds of overlapping decode afterward. A full chunk took
roughly three seconds. Larger chunks therefore do not eliminate the
interruption of ongoing streams by incoming long prompts. The concurrent
TG metric includes that interference; it is not sustained decode throughput.

## Proposed default and verification scope

Update only three fields in the existing NVIDIA w1 example: shared KV capacity
65536 to 262144, and explicit 4096-token busy/idle prefill budgets. Keep MTP
at depth 1, full admission, four slots, the checkpoint and precision settings.
The new pool aligns the effective single-request context ceiling with the
checkpoint's 262144-token limit, while concurrent requests share its capacity.
It is substantially smaller than the working 850048-token site pool.

The proposed config has been parsed and compared with the upstream template;
only those three fields differ. The patch passes applicability and whitespace
checks against official `652cd40`. The 4096 budgets are running on one GB10
with the larger site pool; a fresh startup/throughput check of the exact
262144-token proposed template has not been run during preparation. Engine
memory-plan validation still applies when deploying the new default.

No engine code, extra template or published headline benchmark number changes.
The global automatic budget and other NVIDIA world sizes retain their existing
settings. No GPU workload, restart, rebuild or evaluation was started while
preparing this proposal.
