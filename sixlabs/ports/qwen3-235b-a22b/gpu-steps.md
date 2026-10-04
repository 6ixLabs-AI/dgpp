# Qwen3-235B-A22B-Instruct-2507 (NVFP4) — what to run once a GPU is free

Status when this was written (2026-10-04, overnight, on a Mac): **config, binding table, numpy
reference and host reference have run (on the real headers and on synthetic checkpoints); nothing
has been compiled with nvcc or GCC, loaded on a GPU, or produced a token; nothing multi-rank exists
beyond a plan.** "Expected" below means derived from the checkpoint's `config.json` and safetensors
headers (read from Hugging Face; no weight bytes were downloaded) and from Mac host runs.

| | |
|---|---|
| checkpoint | `nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4` @ `1c4ec35802abf3a20672be7e5c51a098796f71d6` (public, 28 shards, 139.2 GB = 129.6 GiB). **Not known to be on either box.** |
| model class | `Qwen3MoeForCausalLM` / `qwen3_moe`: 94 layers, hidden 4096, 64 query / 4 KV heads of 128, full rotary (theta 5e6), q/k head norms, 128 experts top-8 of 1536, no shared expert, vocab 151936 (the 80B's tokenizer), untied head. |
| container | modelopt NVFP4: experts and `o_proj` NVFP4; `q/k/v_proj`, router, head BF16; `k_scale` / `v_scale` and `input_scale` bound, unread. |
| what is served | **Nothing yet.** Resident it is 133.9 GiB at world 1 — it needs two Sparks — and the family refuses `world_size > 1` by name until [sharding-plan.md](sharding-plan.md)'s ladder has been climbed. What can run on ONE Spark is everything below: the bind check, the reference, and the full 94-layer diagnostic forward in *streaming* residency (one layer resident at a time). |

The code is the same family as the 30B (`src/models/qwen3/`, the `Moe` dialect and the modelopt
container); do `sixlabs/ports/qwen3-vl-30b-a3b/gpu-steps.md` first — it is the smaller model, it
fits one box, and every defect it finds is a defect here.

`$REPO` = a build copy with the patch applied, `$OUT` = scratch, `$IN` =
`$REPO/sixlabs/ports/qwen3-235b-a22b/inputs` (the same 83- and 107-token prompts as the 30B's: the
two templates render these conversations identically), `$SNAP` = the snapshot directory.

## 0. The checkpoint

```bash
hf download nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4 --revision 1c4ec35802abf3a20672be7e5c51a098796f71d6
SNAP=~/.cache/huggingface/hub/models--nvidia--Qwen3-235B-A22B-Instruct-2507-NVFP4/snapshots/1c4ec35802abf3a20672be7e5c51a098796f71d6
```

139 GB on the box's NVMe. One box is enough for every step here; world 2 needs it on both.

## 1. Bind check and the per-rank arithmetic (default build, headers only)

```bash
cd $REPO
cmake --preset ci && cmake --build build-ci -j 8 --target unit_tests qwen3_bind_check qwen3next_tool_calls_test
DGPP_TEST_FILTER=qwen3_ ./build-ci/unit_tests        # expect: 21 tests, 0 failed
./build-ci/qwen3_bind_check --checkpoint-dir $SNAP --world 2
ctest --test-dir build-ci -R 'qwen3moe_tool_calls_test' --output-on-failure
```

Expected (produced on the Mac by the same validator over the real `config.json` and a dump of the
real headers):

```
config: qwen3_moe, 94 layers, hidden 4096, 64 query / 4 kv heads x 128, rope theta 5000000, vocab 151936, max positions 262144
mlp: routed MoE, 128 experts top-8, intermediate 1536, no shared expert
weights: NVFP4, modelopt (o_proj and the experts; q/k/v BF16), FP8 K/V-cache scales bound (unread)
checkpoint: 28 shards, 145703 tensors in headers, 139202447840 tensor bytes
table: 145703 tensors (145703 text + 0 vision)
binding: expected 145703 | matched 145703 (missing 0, dtype 0, shape 0) | unexpected 0 | out of scope 0 | vision bound 0 | tied head copies 0 | matched bytes 139202447840
nvfp4 matrices: 36190
table count 145703 == header count 145703; table bytes 139202447840 == header bytes 139202447840
binding OK: every tensor of the table is present with its dtype and shape
resident, world 2 rank 0: experts 63871005696, attention 6702497792, dense mlp 0, router 98566144, norms 1596416, embedding 1244659712, head 622329856 | total 72540655616 bytes (67.56 GiB)
resident, world 2 rank 1: … total 72540655616 bytes (67.56 GiB)
```

`qwen3moe_tool_calls_test` (the 80B's binary on this checkpoint's corpus) passed on the Mac against
the real tokenizer and template: 9 renders byte-exact with ids equal to HF tokenizers', 4 calls
round-tripped, the forced-call grammar admits them. Note what the template does that the 80B's does
not: an assistant turn after the last user query that carries `reasoning_content` is re-rendered
with a `<think>` block, and a *trailing* assistant turn gets an empty one. The template never reads
`enable_thinking` (checked through the engine's interpreter), so the engine treats the family as
instruct-only (`reasoning_effort` accepted and ignored). Whether the model itself ever opens a
`<think>` block is a thing to look at in the first transcripts, not something read here.

## 2. The reference on the real checkpoint (CPU, no engine)

```bash
cd $REPO
for k in prose code; do
  python3 tools/qwen3_reference.py score --ckpt $SNAP --ids-json $IN/qwen3moe-ids-$k.json --out $OUT/235-ref-$k.json --threads 8
done
```

This mmaps all 28 shards and walks them once; a scoring pass keeps no layer's matrices (peak RSS is
the head, 2.5 GB, plus one layer), but 130 GiB go through the page cache: **not on a box that is
serving or benchmarking.** Expect minutes, not seconds (each layer dequantizes up to 128 experts of
18.9 M weights).

1. `mean_logprob` must be a small negative number on these natural texts; about −11.9 (ln 151936)
   means a container convention is wrong.
2. These must each **collapse** it (`--layers 4` is enough): `--variant ws2=div` (modelopt's
   per-tensor scale multiplies), `--variant nibble=hi`, `--variant gateup=up`, `--variant norm=1p`,
   `--variant rope=adjacent`, `--variant qknorm=off`.
3. The C++ reading of the same bytes (two layers; plain loops, a few minutes at this width):

```bash
./build-ci/qwen3_bind_check --checkpoint-dir $SNAP --layers 2 --host-forward $IN/qwen3moe-ids-prose.json > $OUT/235-host-L2.txt
python3 tools/qwen3_reference.py score --ckpt $SNAP --layers 2 --dtype float64 --variant ropeangle=f64 \
    --ids-json $IN/qwen3moe-ids-prose.json --out $OUT/235-ref-L2.json
python3 sixlabs/bench/compare_forward_check.py $OUT/235-host-L2.txt $OUT/235-ref-L2.json --mean 1e-4 --max 1e-3
```

## 3. Turn the draft on and build it

As the 30B's §3: the draft is off unless the build is configured with
`-DDGPP_BUILD_QWEN3_PLAIN_DRAFT=ON`.

```bash
cd $REPO
cmake --preset ci -DDGPP_BUILD_QWEN3_PLAIN_DRAFT=ON
cmake --build build-ci -j 8 --target qwen3_loader_test qwen3_forward_test
cmake --preset release -DDGPP_BUILD_QWEN3_PLAIN_DRAFT=ON
cmake --build build-release -j 8 --target qwen3_forward_check
cd $OUT && $REPO/build-ci/qwen3_loader_test && $REPO/build-ci/qwen3_forward_test   # 3 tests, then 4 tests, 0 failed
```

The fixture gates cover this container: `qwen3_loader_resident_values_modelopt` (world 1 and both
ranks of a world of 2; passed on the Mac's CUDA runtime shim, never on a device) and
`qwen3_forward_matches_the_host_walk_modelopt` (never run). The modelopt case is the one that
matters most here: this container names its NVFP4 *codes* `.weight`, the same name a BF16 matrix
has, and the loader tells them apart by the binding table's role, not by the name.

## 4. The single-rank gate: the full model, one Spark, streaming

```bash
cd $REPO
./build-release/qwen3_forward_check --checkpoint-dir $SNAP --plan 4,262144,2     # loads nothing
for k in prose code; do
  python3 tools/qwen3_reference.py dump --ckpt $SNAP --ids-json $IN/qwen3moe-ids-$k.json --out $OUT/235-refdump-$k --detail-layers 0,1 --threads 8
  ./build-release/qwen3_forward_check --checkpoint-dir $SNAP --ids-json $IN/qwen3moe-ids-$k.json \
      --dump-states $OUT/235-$k.bf16 > $OUT/235-$k.txt
  python3 sixlabs/bench/compare_forward_check.py $OUT/235-$k.txt $OUT/235-ref-$k.json
  python3 sixlabs/bench/compare_states.py $OUT/235-$k.bf16 $OUT/235-refdump-$k \
      $(python3 -c "import json;print(len(json.load(open('$IN/qwen3moe-ids-$k.json'))))") 4096
done
./build-release/qwen3_forward_check --checkpoint-dir $SNAP --ids-json $IN/qwen3moe-ids-prose.json --layers 4 --host-compare
```

The plan line should read about **67.6 GiB** of weights for rank 0 of 2 and **23.5 GiB** of K/V
(262,144 tokens × 96,256 bytes). The forward runs without `--resident`: the stream holds one layer
(about 1.4 GiB at world 1: 1.27 GiB of experts, 0.13 of BF16 attention) and the globals (2.3 GiB),
and reads all 130 GiB off the NVMe once per forward — minutes per prompt, and a page-cache load the
box should not carry beside anything else.

Pass: as the 30B's §6 (`PASS` at mean ≤ 0.06 nat / worst ≤ 0.5; smooth per-layer errors; the host
walk close). This is the number that says the walk is right on this model at world 1; world 2 is
then a question of the halves and the folds only.

## 5. World 2

Not built, not served: [sharding-plan.md](sharding-plan.md) has the split tensor by tensor, the
per-rank bytes, the collectives, the proposed deployment JSON and the ladder (a loopback TP test on
the fixture, the engines over it, then two Sparks). With the patch as it is, a `world_size: 2`
config stops at boot with `the plain Qwen3 family is single-node for now: world_size 2 is planned …
not verified`.

## 6. Still open

- Everything multi-rank (the plan's ladder), and with it any served token from this model.
- The three byte levers in the plan (an NVFP4 `o_proj` on the fp4 GEMV core; block FP8 for q/k/v at
  load; four nodes).
- `k_scale` / `v_scale`: the recipe's FP8 K/V-cache scales are bound and unread; the pool is BF16.
- W4A4 expert prefill (`DGPP_MOE_W4A4`): modelopt's `input_scale` is not read by this loader, so the
  layer would use dynamic activation scales; untested.
- The model card gives the context as 262,144 natively and "extendable up to 1,010,000"; this
  family has no rope ramp (`engine.rope_scaling` is refused), so one request stops at 262,144.
