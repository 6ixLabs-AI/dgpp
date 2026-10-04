# sixlabs/ — everything 6ixLabs keeps beside the engine

DGPP's own tree is untouched outside the files the port changes; our tooling lives here.

| Path | What |
|---|---|
| `refset/gen_prompts.py`, `refset/capture_refset.py` | The fixed prompt set (short prompts, tool calls, long-context retrieval at 8K–235K) and the capture that replays it against any OpenAI-compatible server. |
| `refset/results/` | Captures kept as evidence: Atlas serving the same checkpoint (`atlas_*.jsonl`), and this engine (`dgpp_*.jsonl`). |
| `refset/hf_adopt_local_dir.py` | Moves a `hf download --local-dir` copy of a model into the HF hub cache, hash-verified first. |
| `bench/restart_test.sh key=value ...` | Restart the TEST deployment on DGXone with engine settings changed (it edits the one test config; the launcher keys a deployment on its config path). |
| `bench/bench_decode.py` | Single-stream and N-stream decode speed. |
| `bench/bench_matrix.py` | Context x concurrency cells, timings read from the server's own log. |
| `bench/compare_states.py`, `compare_logprobs.py`, `compare_transcripts.py` | The engine against the numpy host reference (`tools/qwen3next_reference.py`) and against another capture. |
| `bench/compare_forward_check.py` | One `qwen35_forward_check` run against the reference's `score` of the same ids (no capture needed): the first numerical gate of a new checkpoint. |
| `bench/build_dgpp.py` | Configure + build a checkout as a logged job. |
| `upstream-issue-77-draft.md` | The feature request as first drafted (its checkpoint section was corrected in the issue thread). |
| `registry/` | The model registry: `models.json` says which family serves each model, which checkpoints it takes, how it is launched and how far it has been verified (working / upstream / compiled / groundwork / refused); `registry.py` checks it against the tree, prints the status and speed table, and resolves an entry into a deployment config. |

## Where things run (2026-10-03)

- Build and test copy: DGXone `~/dgpp-next` (an rsync of this repository; the production DGPP checkout
  `~/dgpp` there serves Qwen3.8 and is never built in).
- Test deployment: DGXone `172.17.0.1:18090`, model `Qwen3-Next-80B`; config
  `deploy/cluster_qwen3-next-80b_nvfp4_w1.json` and `.env` in that copy (both git-ignored: site settings).
- Gateway name: LiteLLM `qwen3-next-80b-6ix`.
- Weights: the HF cache on DGXone (`nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`; the FP8 release is there too).
- Working files of the captures and benchmarks: DGXone `~/dgpp-refset/`.
