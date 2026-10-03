# DeepSeek-V4-Flash campaign notes

## Reference harness: llama-benchy tg128

llama-benchy's `tg128` (generate 128 tokens after a ~2,048-token prompt) is
the single-request figure DGX Spark owners report for this model. Against
the four-node template, at the harness's defaults — the checkpoint's own
sampling (`generation_config.json`: temperature 1.0, top_p 1.0), thinking
on, the model's `low` reasoning effort:

```
llama-benchy --base-url http://HEAD:18080/v1 --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --tokenizer SNAPSHOT_DIR --pp 2048 --tg 128 --runs 40 --concurrency 1 \
  --latency-mode none --skip-coherence
```

| Leg (40 runs each) | Binary | tg128 tokens/s (mean ± std) | Peak tokens/s | pp2048 tokens/s |
|---|---|---|---|---|
| [1](raw/dsv4-w4/llama-benchy/dsv4_benchy_r6-all093.txt) | two fixes before the campaign's | 65.07 ± 7.46 | 65.50 | 1,380 |
| [2](raw/dsv4-w4/llama-benchy/dsv4_benchy_r7-final.txt) | one fix before the campaign's | 64.83 ± 6.16 | 65.40 | 1,377 |
| [3](raw/dsv4-w4/llama-benchy/dsv4_benchy_r8-final.txt) | the fourth run's | 64.27 ± 5.35 | 64.85 | 1,376 |
| [4](raw/dsv4-w4/llama-benchy/dsv4_benchy_r9-campaign5.txt) | the campaign's (`manifest.json`, the fifth run) | 64.19 ± 5.65 | 64.70 | 1,490 |

Legs 1–3 differ in the multi-request batching and sampling-fallback fixes
only, which a single-request run does not reach; leg 4 is the binary after
the prefill fold overlap and the grouped read-in, which a single sampled
tg128 run does not reach either (the prompt processing it reports moved,
1,376 → 1,490). The four means span 64.2–65.1 (64.6 over the 160 runs). The harness's own run-to-run standard
deviation is 5–7 tokens/s — each run is a different passage of the source
text, sampled — so one 40-run mean carries about ±1 token/s. The generated
text is sampled prose, the class where the draft is accepted least (the
by-class tables in the record: 63 tokens/s on prose, 120 on JSON).

### Concurrency

The same harness at concurrency 1, 2, 5 and 10 with `--latency-mode
generation` (each request's prompt processing subtracted from its
generation time), 10 runs each, on the campaign's binary — the fifth
run's legs; the earlier legs of 2026-10-02 15:35 UTC (the binary before
the grouped read-in, when concurrent prompts were admitted one whole
prompt at a time) beside them:

| Nodes | Concurrency | tg128 tokens/s, total (before → now) | pp2048 tokens/s, total (before → now) |
|---|---|---|---|
| 4 | 1 | 66.3 → 66.0 | 1,486 → 1,608 |
| 4 | 2 | 58.8 → 90.6 | 1,371 → 1,630 |
| 4 | 5 | 56.5 → 121.1 | 1,348 → 1,664 |
| 4 | 10 | 51.9 → 66.4 | 997 → 987 |
| 2 | 1 | 39.7 → 40.5 | 1,018 → 1,063 |
| 2 | 2 | 36.3 → 56.1 | 948 → 1,116 |
| 2 | 5 | 32.4 → 44.7 | 576 → 566 |

Raw: [four nodes, now](raw/dsv4-w4/llama-benchy/concurrency-1-2-5-10-campaign5.txt),
[before](raw/dsv4-w4/llama-benchy/concurrency-1-2-5-10.txt);
[two nodes, now](raw/dsv4-w2/llama-benchy/concurrency-1-2-5-campaign5.txt),
[before](raw/dsv4-w2/llama-benchy/concurrency-1-2-5.txt). Ten concurrent
requests on four nodes queue behind the six slots (the block draft takes
five rows per slot of the family's 32-row decode batch), and five on two
nodes behind four: those rows are two waves.

## HumanEval: answers that end without the closing fence

With thinking off the model ends some answers right after the last line of
code, without the closing code fence. The evaluation's extractor requires a
closed block, so those answers fail whatever the code does.
`lenient_humaneval.py` runs the open block as the code in the same pinned
container:

| Deployment | Strict | Unclosed-fence failures | Pass when the open block is executed | Lenient |
|---|---|---|---|---|
| Four nodes | 138/164 | 15 | 13 | 151/164 |
| Two nodes | 140/164 | 14 | 12 | 152/164 |

This is the model's behaviour, not the engine's: plain decode produces the
same 15 answers as the speculative decode, and the release's own
`inference/model.py`, chained over all 43 layers on the real checkpoint for
one of these prompts, ranks end-of-sequence above the newline that would
open the closing fence (33.13 against 32.48; the engine 33.22 against
32.09). The published tables keep the strict counts.

## Cross-check against the release's code

`tools/dsv4_torch_reference.py` runs the release's `inference/model.py`
layer code on the real checkpoint against the engine's per-layer dump:
isolated per-layer relative l2 0.0003–0.004 with routes and selections
equal; chained over 43 layers the head's argmax is equal (relative l2
0.003–0.06, expert routes flipping on BF16 rounding noise). The DSpark
draft over five decode steps: the block's base argmax equal on every row of
every step, the Markov-biased first row's argmax equal, base logits within
0.066 relative l2.
