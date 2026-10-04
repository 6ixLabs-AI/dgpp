# Qwen3-Reranker-0.6B / Qwen3-Embedding-0.6B — what exists, what to run, what a serving path needs

Status (2026-10-04, overnight, on a Mac): **config parse and binding table only, as asked — both
verified against the real headers. There is no GPU path for the dense dialect and no rerank or
embedding endpoint; none was built.** The numpy reference and the C++ host reference do cover the
dense model (they are the same code as the MoE dialects'), so there is something to check a future
path against.

| | `Qwen/Qwen3-Reranker-0.6B` @ `e61197ed` | `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614` |
|---|---|---|
| class | `Qwen3ForCausalLM` / `qwen3` | the same class in `config.json`, but the file is the **base model's** state dict |
| shape | 28 layers, hidden 1024, 16 query / 8 KV heads of 128 (so `q_proj` is [2048, 1024]), dense SwiGLU of 3072, theta 1e6, vocab 151669, tied embeddings | identical |
| differs | `max_position_embeddings` 40960, `eos_token_id` 151645 | 32768, 151643 |
| tensors | **310**, 1,191,553,024 bytes, BF16, names `model.layers.L.*`, no `lm_head.weight` (tied) | **310**, 1,191,553,024 bytes, BF16, names `layers.L.*`, `embed_tokens.weight`, `norm.weight` — **no `model.` prefix**, no head |
| table vs headers | 310 == 310, bytes equal | 310 == 310, bytes equal (the header says which naming: `qwen3_apply_header_naming`) |
| tokenizer.json | byte-identical to the 80B's / 30B's / 235B's (sha256 `aeb13307…`) | same vocabulary, merges and added tokens; **a different `post_processor`**: a `TemplateProcessing` that appends `<|endoftext|>` (151643) to every sequence |

Corrections to what the task statement assumed: nothing in the shape; two things beside it — the
Embedding release's bare tensor names, and its tokenizer's appended EOS (below).

## What to run (host only; no GPU step exists)

```bash
cd $REPO
cmake --preset ci && cmake --build build-ci -j 8 --target unit_tests qwen3_bind_check
DGPP_TEST_FILTER=qwen3_dense ./build-ci/unit_tests     # expect: 2 tests, 0 failed
./build-ci/qwen3_bind_check --model Qwen/Qwen3-Reranker-0.6B
./build-ci/qwen3_bind_check --model Qwen/Qwen3-Embedding-0.6B
```

Expected for both: `config: qwen3, 28 layers, hidden 1024, 16 query / 8 kv heads x 128, rope theta
1000000, vocab 151669, max positions 40960` (32768 for the Embedding), `mlp: dense SwiGLU,
intermediate 3072, tied embeddings`, `weights: BF16` (the Embedding adds `, base-model names (no
model. prefix)`), `checkpoint: 1 shards, 310 tensors in headers, 1191553024 tensor bytes`, `binding:
expected 310 | matched 310 …`, `table count 310 == header count 310; table bytes 1191553024 ==
header bytes 1191553024`, `binding OK`.

The server refuses both checkpoints by name in every build (`the checkpoint is a dense Qwen3 model …
there is no serving path`), and `qwen3_forward_check` (a draft-only target) refuses the dense dialect.

The reference's two read-outs of one prefill (CPU, seconds on a 0.6B model; `ids.json` = the token
ids of a formatted pair or text, made with any Qwen3 tokenizer):

```bash
python3 tools/qwen3_reference.py retrieve --ckpt $RERANKER --ids-json ids.json --yes 9693 --no 2152   # yes / no logits, p_yes
python3 tools/qwen3_reference.py retrieve --ckpt $EMBEDDING --ids-json ids.json                       # the L2-normalized last-token hidden
./build-ci/qwen3_bind_check --checkpoint-dir $RERANKER --host-forward ids.json                        # the C++ host walk, same ids
```

`retrieve` has run on the synthetic dense checkpoints only (where the model's forward matches
transformers' `Qwen3ForCausalLM` to 4e-15); it has not been compared with the model cards' example
scores on the real weights. That comparison is the first thing to do with the real checkpoints:
the Reranker card's two example pairs and the Embedding card's 2×2 similarity matrix.

## What a prefill-only "score" path and an embedding path would need from the serving layer

Both are one primitive — **prefill a prompt, read the final-norm hidden state at its last position,
generate nothing** — with two read-outs:

- score: `p_yes = sigmoid(logit[yes] − logit[no])`, the two logits being the last hidden state's
  dot products with head rows 9693 (`yes`) and 2152 (`no`) (both single tokens, checked with the
  engine's tokenizer and HF's; the repo's `1_LogitScore/config.json` names the same ids). The head is
  tied: those are two rows of the embedding, so the full 151,669-row head never has to run;
- embedding: that hidden row, L2-normalized (`1_Pooling`: last-token; `2_Normalize`), optionally
  truncated to a requested dimension and renormalized.

**What main already has (it landed on 2026-10-04 while this note was being written, commit
`5c20f70`): `POST /v1/score`.** It states the model's log-probability of a continuation you supply
after a prompt you supply (`prompt` or `prompt_ids`, `continuation` or `continuation_ids`,
`top_logprobs` 0–20), computed in the read-in pass: the scheduler's `SchedulerRequest::score_tokens`,
the graph engine's `configure_score` / `take_score`, the session core's `session_set_score_tail`,
`RowRun::tail_rows` and `Outputs::tail_logits`. It serves a family whose walk declares
`kScoreTail` — today the qwen3_5 family alone — at world 1, 1 to 256 tokens, never cached, never
grouped; anything else answers 501 `score_unsupported`. That is most of a score path, so the list
below is what is missing *beyond* it. Read on the Mac, in the source; this family has not been
behind that route on any machine.

For a reranker pair that route would be used as: `prompt_ids` = the formatted pair,
`continuation_ids` = `[9693]`, `top_logprobs` = 20. The answer carries log p(yes) exactly, and
log p(no) whenever `no` is among the 20 most likely tokens (usual for this model, not guaranteed —
otherwise a second call with `[2152]`). `p_yes = sigmoid(logp_yes − logp_no)`: a difference of two
log-softmax values is the difference of the two logits.

What is missing, bottom to top:

1. **A GPU walk for the dense dialect.** The family's draft model (`Qwen3Model`) is MoE-only and
   does not declare `kScoreTail`. Dense needs a BF16 SwiGLU MLP block (three GEMMs through the
   existing GEMM interface) and a tied head (the head reading the resident embedding). Both now
   have a pattern on main in the qwen3_5 family — the Qwen3.5-0.8B port, commit `a558c9f`:
   `Qwen35Model::dense_mlp`'s BF16 branch and its tied head; by that commit's own statement they
   have not executed on a GPU either. The attention layer is the one already used (16 / 8 heads of
   128). The whole model is 1.11 GiB resident. A one-token continuation needs only the prefill's
   usual last row, so `kScoreTail` costs nothing here beyond declaring it and honouring
   `RowRun::tail_rows` (two lines in `Qwen35Model::run_rows`).
2. **Rerank over that route: batching and the shared prefix.** `/v1/score` takes one prompt and is
   never cached and never grouped, so N documents are N cold prefills that each re-read the system
   turn, the instruction and the query. A rerank request wants the documents as spans of one walk
   (`prefill_group` exists for cold prompts in one forward, each span's last row) or the
   token-prefix cache allowed for this request kind (point 6). It also wants the two logits read
   from one prefill rather than a top-20 list: rows 9693 and 2152 of the tied embedding against
   the last hidden row, with no need for the 151,669-row head at all.
3. **An engine operation that hands back a hidden row** — the embedding path; `/v1/score` does not
   give it. The scoring request still generates (and discards) one token and returns
   log-probabilities only. Needed: a prefill that skips the head and the pick, returns the last row
   of `final_hidden_bits` (the session core already keeps the final hidden rows), reserves no
   blocks for generation, and closes. In the scheduler and the journal that is a request kind with
   no sampler, no stop logic, no token stream and no `max_tokens`, admitted by prompt tokens alone.
   At world > 1 (where `/v1/score` is refused today) the hidden state is replicated at the block
   boundary and the embedding is replicated, so rank 0 could read both results with no extra
   collective; a vocab-sharded *head* would put 9693 and 2152 both in rank 0's half, which stops
   mattering once the two rows are read from the embedding.
4. **Endpoints.** Beside `/v1/chat/completions`, `/v1/completions`, `/v1/score`, `/v1/models`,
   `/v1/files` and metrics, wanted: `/v1/embeddings` (OpenAI's shape: `input` string or array,
   `encoding_format` float | base64, `dimensions`) and a rerank endpoint (the common shape:
   `query`, `documents[]`, `top_n` → `results[{index, relevance_score}]`) that builds the prompts
   (point 5) and returns `p_yes` per document.
5. **Prompt construction in code, not through the checkpoint's template.** The Reranker's
   `chat_template.jinja` uses `selectattr` / `map` / `first` / `default`; the engine's template
   interpreter refuses it (`unsupported filter 'selectattr'`, tried on the Mac). The format is fixed
   anyway: the system turn "Judge whether the Document meets the requirements based on the Query and
   the Instruct provided. Note that the answer can only be \"yes\" or \"no\".", a user turn
   `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}`, then the suffix
   `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` (ids 151645, 198, 151644, 77091, 198,
   151667, 271, 151668, 271); truncation applies to the middle (the card uses 8192 tokens).
   The Embedding has no template: a query is `Instruct: {task}\nQuery:{query}`, a document is the
   bare text, and **`<|endoftext|>` is appended** — its hidden state is the embedding. The engine's
   tokenizer refuses this `tokenizer.json` today (`post_processor is neither null, ByteLevel nor a
   Sequence of ByteLevel … special-token injection is not implemented at encode`, tried on the
   Mac), so the path either teaches the tokenizer that post-processor or loads the tokens and
   appends 151643 itself. Left padding, which the cards insist on, exists only to put the last
   token at a fixed index in a padded batch; rows of one walk need none.
6. **The prefix cache, on for rerank and off for embeddings.** Every document of one rerank request
   shares the system turn, the instruction and the query: the engine's token-prefix cache gives that
   reuse for free if the prefill-only request is allowed to hit and leave snapshots; an embedding
   input shares nothing and should leave none.
7. **A numerics decision for the embedding.** The walk's final norm writes BF16 (8 significant
   bits a component). For a logit difference that is noise; for a cosine it is an error to measure
   against `qwen3_reference.py retrieve` before deciding whether the last row's norm should be
   handed back in fp32.
8. **Deployment.** One model per server process: a retrieval model is its own deployment (its own
   port and gateway name) beside a generation model, with a pool sized for 8K–32K prompts and no
   decode graph.
