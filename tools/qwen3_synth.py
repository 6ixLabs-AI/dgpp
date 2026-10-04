#!/usr/bin/env python3
"""qwen3_synth.py — tiny synthetic checkpoints of the plain Qwen3 family and the checks that run
on them (2026-10-04). No GPU. numpy only, except `torch-check`, which needs torch + transformers.

The synthetic checkpoints carry the real releases' tensor names, dtypes, containers and recipes at
toy sizes, so tools/qwen3_reference.py, the engine's config parser / binding table and its host
reference (src/models/qwen3/reference.hpp) can be exercised end to end on a laptop.

  ckpt        --preset vl|moe|dense|dense-bare --out DIR [--seed N]
        writes config.json (+ hf_quant_config.json for moe), ids.json and model.safetensors.
          vl          Qwen3VLMoeForConditionalGeneration, compressed-tensors nvfp4-pack-quantized:
                      q/k/v/o and the experts as weight_packed / weight_scale / weight_global_scale /
                      input_global_scale, a BF16 router and head, a BF16 vision tower (never run).
          moe         Qwen3MoeForCausalLM, modelopt NVFP4: experts and o_proj as weight /
                      weight_scale / weight_scale_2 / input_scale, BF16 q/k/v with k_scale / v_scale.
          dense       Qwen3ForCausalLM, BF16, tied embeddings, `model.` names (Qwen3-Reranker's layout).
          dense-bare  the same under the base model's names, no `model.` prefix (Qwen3-Embedding's).
  selftest    --ckpt DIR
        through qwen3_reference.Model: (1) the NVFP4 reader against an independent scalar decode;
        (2) feeding a sequence whole, in chunks and token by token gives the same logits; (3) greedy
        generation reproduces teacher-forced scoring of its own output; (4) every layout variant of
        the reference changes the logits (the checkpoint can tell the conventions apart).
  torch-check --ckpt DIR
        transformers' OWN classes (Qwen3ForCausalLM, Qwen3MoeForCausalLM, Qwen3VLMoeTextModel and
        the whole Qwen3VLMoeForConditionalGeneration fed text only) in float64 with this checkpoint's
        dequantized weights, against qwen3_reference in float64: logits and every hidden state.
  vectors     --out tests/unit/qwen3_reference_vectors.hpp [--seed N]
        the test vectors of tests/unit/qwen3_reference_test.cpp: per preset the config.json text,
        the content digest of the checkpoint (the C++ fixture regenerates the same tensors and must
        reach the same digest), token ids, and what qwen3_reference computes from them in float64.

How the tensors are made: every tensor is filled from an integer stream seeded by its NAME
(splitmix64 over FNV-1a), in its STORED form — BF16 values that are exact in 8 significant bits,
NVFP4 code bytes, e4m3 block scales and F32 scales taken straight from the stream. No float
encoder is involved, so tests/unit/qwen3_fixture.hpp regenerates the same bytes in C++ from the
same rules, in any order.

What a pass here does and does not show: the numpy reference computes what transformers' classes
compute (torch-check), and the C++ host reference reads the same containers the same way
(vectors). The NVFP4 container conventions themselves (nibble order, which way each global scale
goes, llm-compressor's gate/up unpacking) are the 80B and Coder-Next ports' reading of the real
checkpoints; a synthetic checkpoint written by the same reading cannot prove them — the first
forward check on a real checkpoint does (sixlabs/ports/qwen3-vl-30b-a3b/gpu-steps.md).
"""
import argparse
import json
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_reference as ref  # noqa: E402

M64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


# ----------------------------------------------------------------------------------------------
# The integer stream
# ----------------------------------------------------------------------------------------------
def fnv1a64(data, h=0xCBF29CE484222325):
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & M64
    return h


def stream(name, seed, n):
    """n splitmix64 outputs for the tensor `name`: output i is mix(fnv1a(name) ^ seed' + (i + 1) * GAMMA)."""
    base = (fnv1a64(name.encode()) ^ ((seed * 0xD6E8FEB86659FD93) & M64)) & M64
    with np.errstate(over="ignore"):
        z = np.uint64(base) + (np.arange(1, n + 1, dtype=np.uint64) * np.uint64(GAMMA))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def to_bf16(w):
    """float32 -> BF16 bits, round to nearest even (finite inputs)."""
    u = np.ascontiguousarray(w, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def fill(name, dtype, shape, role, seed, packed):
    """The stored bits of one tensor (the rules tests/unit/qwen3_fixture.hpp repeats)."""
    n = int(np.prod(shape, dtype=np.int64)) if len(shape) else 1
    r = stream(name, seed, n)
    if role == "payload":                                 # U8: two e2m1 codes a byte
        a = ((r >> np.uint64(24)) & np.uint64(0xFF)).astype(np.uint8)
    elif role == "scale":                                 # e4m3 in [0.5, 1.875]: exponent 6 or 7, 3 mantissa bits
        e = np.uint64(6) + ((r >> np.uint64(16)) & np.uint64(1))
        a = ((e << np.uint64(3)) | ((r >> np.uint64(20)) & np.uint64(7))).astype(np.uint8)
    elif role == "global":                                # F32: modelopt's multiplier, compressed-tensors' divisor
        k = ((r >> np.uint64(11)) % np.uint64(64 if packed else 16)).astype(np.float32)
        a = (np.float32(64.0) + k) if packed else ((np.float32(16.0) + k) / np.float32(2048.0))
        a = a.astype(np.float32)
    elif role == "one":                                   # F32: the unused activation / cache scales
        a = np.ones(n, np.float32)
    elif role == "norm":                                  # BF16 in [0.75, 1.25], exact
        k = ((r >> np.uint64(11)) % np.uint64(65)).astype(np.float32) - np.float32(32.0)
        a = to_bf16(np.float32(1.0) + k / np.float32(128.0))
    else:                                                 # BF16 in [-0.248, 0.248], exact
        k = ((r >> np.uint64(11)) % np.uint64(255)).astype(np.float32) - np.float32(127.0)
        a = to_bf16(k / np.float32(512.0))
    return a.reshape(shape)


# ----------------------------------------------------------------------------------------------
# The synthetic configs (toy sizes; head_dim stays 128, the engine's attention kernels' shape)
# ----------------------------------------------------------------------------------------------
_TEXT = {
    "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 1, "eos_token_id": 2,
    "head_dim": 128, "hidden_act": "silu", "hidden_size": 64, "initializer_range": 0.02,
    "intermediate_size": 96, "max_position_embeddings": 4096, "num_attention_heads": 4,
    "num_hidden_layers": 3, "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "use_cache": True,
    "vocab_size": 256,
}
_MOE = {"decoder_sparse_step": 1, "mlp_only_layers": [], "moe_intermediate_size": 32, "norm_topk_prob": True,
        "num_experts": 8, "num_experts_per_tok": 3, "router_aux_loss_coef": 0.001}
_FP4_GROUP = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}


def make_config(preset):
    """-> (config.json dict, hf_quant_config.json dict or None)."""
    L = _TEXT["num_hidden_layers"]
    if preset in ("dense", "dense-bare"):
        c = dict(_TEXT)
        c.update({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "max_window_layers": L,
                  "rope_scaling": None, "rope_theta": 1000000, "sliding_window": None, "tie_word_embeddings": True,
                  "torch_dtype": "bfloat16", "transformers_version": "4.51.3", "use_sliding_window": False})
        return c, None
    if preset == "moe":
        c = dict(_TEXT)
        c.update(_MOE)
        ignore = ["lm_head"]
        for l in range(L):
            ignore += [f"model.layers.{l}.mlp.gate", f"model.layers.{l}.self_attn.k_proj",
                       f"model.layers.{l}.self_attn.q_proj", f"model.layers.{l}.self_attn.v_proj"]
        producer = {"name": "modelopt", "version": "0.0.1.synthetic"}
        c.update({"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "dtype": "bfloat16",
                  "max_window_layers": L, "output_router_logits": False, "rope_scaling": None, "rope_theta": 5000000,
                  "sliding_window": None, "tie_word_embeddings": False, "transformers_version": "4.56.0",
                  "use_sliding_window": False,
                  "quantization_config": {
                      "config_groups": {"group_0": {"input_activations": dict(_FP4_GROUP), "weights": dict(_FP4_GROUP),
                                                    "targets": ["Linear"]}},
                      "ignore": ignore, "quant_algo": "NVFP4",
                      "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
                      "producer": producer, "quant_method": "modelopt"}})
        hf = {"producer": producer, "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8",
                                                     "group_size": 16, "exclude_modules": ignore}}
        return c, hf
    if preset == "vl":
        t = dict(_TEXT)
        t.update(_MOE)
        t.update({"dtype": "bfloat16", "model_type": "qwen3_vl_moe_text", "rope_theta": 5000000,
                  "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"}})
        v = {"deepstack_visual_indexes": [0, 1], "depth": 2, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 32,
             "in_channels": 3, "initializer_range": 0.02, "intermediate_size": 48, "model_type": "qwen3_vl_moe",
             "num_heads": 4, "num_position_embeddings": 16, "out_hidden_size": t["hidden_size"], "patch_size": 4,
             "spatial_merge_size": 2, "temporal_patch_size": 2}
        ignore = []
        for b in range(v["depth"]):
            ignore += [f"model.visual.blocks.{b}.attn.qkv", f"model.visual.blocks.{b}.attn.proj",
                       f"model.visual.blocks.{b}.mlp.linear_fc1", f"model.visual.blocks.{b}.mlp.linear_fc2"]
        ignore += ["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"]
        for i in range(len(v["deepstack_visual_indexes"])):
            ignore += [f"model.visual.deepstack_merger_list.{i}.linear_fc1",
                       f"model.visual.deepstack_merger_list.{i}.linear_fc2"]
        ignore += [f"model.language_model.layers.{l}.mlp.gate" for l in range(L)] + ["lm_head"]
        wq = {"actorder": None, "block_structure": None, "dynamic": False, "group_size": 16, "num_bits": 4,
              "observer": "static_minmax", "observer_kwargs": {}, "strategy": "tensor_group", "symmetric": True,
              "type": "float"}
        aq = dict(wq)
        aq["dynamic"] = "local"
        c = {"architectures": ["Qwen3VLMoeForConditionalGeneration"], "dtype": "bfloat16", "image_token_id": 250,
             "model_type": "qwen3_vl_moe",
             "quantization_config": {
                 "config_groups": {"group_0": {"format": "nvfp4-pack-quantized", "input_activations": aq,
                                               "output_activations": None, "targets": ["Linear"], "weights": wq}},
                 "format": "nvfp4-pack-quantized", "global_compression_ratio": None, "ignore": ignore,
                 "kv_cache_scheme": None, "quant_method": "compressed-tensors", "quantization_status": "compressed",
                 "sparsity_config": {}, "transform_config": {}, "version": "0.12.3.synthetic"},
             "text_config": t, "tie_word_embeddings": False, "transformers_version": "4.57.1", "video_token_id": 251,
             "vision_config": v, "vision_end_token_id": 249, "vision_start_token_id": 248}
        return c, None
    raise SystemExit(f"unknown preset {preset!r}")


def table(cfg, bare=False):
    """[(name, safetensors dtype, shape, fill role)] of the whole checkpoint — written from the releases'
    headers, independently of src/models/qwen3/binding.cpp (the C++ fixture must reach the same digest)."""
    mt = cfg["model_type"]
    t = cfg["text_config"] if mt == "qwen3_vl_moe" else cfg
    P = "model.language_model." if mt == "qwen3_vl_moe" else ("" if bare else "model.")
    packed = mt == "qwen3_vl_moe"
    modelopt = mt == "qwen3_moe"
    H, V, d = t["hidden_size"], t["vocab_size"], t["head_dim"]
    Q, KV = t["num_attention_heads"] * d, t["num_key_value_heads"] * d
    out = []

    def bf16(name, shape, role="w"):
        out.append((name, "BF16", tuple(shape), role))

    def fp4(base, n, k):
        if packed:
            out.extend([(base + ".weight_packed", "U8", (n, k // 2), "payload"),
                        (base + ".weight_scale", "F8_E4M3", (n, k // 16), "scale"),
                        (base + ".weight_global_scale", "F32", (1,), "global"),
                        (base + ".input_global_scale", "F32", (1,), "one")])
        else:
            out.extend([(base + ".weight", "U8", (n, k // 2), "payload"),
                        (base + ".weight_scale", "F8_E4M3", (n, k // 16), "scale"),
                        (base + ".weight_scale_2", "F32", (), "global"),
                        (base + ".input_scale", "F32", (), "one")])

    bf16(P + "embed_tokens.weight", (V, H))
    bf16(P + "norm.weight", (H,), "norm")
    if not cfg.get("tie_word_embeddings", False):
        bf16("lm_head.weight", (V, H))
    for l in range(t["num_hidden_layers"]):
        p = f"{P}layers.{l}."
        bf16(p + "input_layernorm.weight", (H,), "norm")
        bf16(p + "post_attention_layernorm.weight", (H,), "norm")
        a = p + "self_attn."
        for nm, n, k in (("q_proj", Q, H), ("k_proj", KV, H), ("v_proj", KV, H)):
            fp4(a + nm, n, k) if packed else bf16(a + nm + ".weight", (n, k))
        if modelopt:
            out.append((a + "k_proj.k_scale", "F32", (), "one"))
            out.append((a + "v_proj.v_scale", "F32", (), "one"))
        fp4(a + "o_proj", H, Q) if (packed or modelopt) else bf16(a + "o_proj.weight", (H, Q))
        bf16(a + "q_norm.weight", (d,), "norm")
        bf16(a + "k_norm.weight", (d,), "norm")
        m = p + "mlp."
        if mt == "qwen3":
            I = t["intermediate_size"]
            bf16(m + "gate_proj.weight", (I, H))
            bf16(m + "up_proj.weight", (I, H))
            bf16(m + "down_proj.weight", (H, I))
        else:
            I = t["moe_intermediate_size"]
            bf16(m + "gate.weight", (t["num_experts"], H))
            for e in range(t["num_experts"]):
                fp4(f"{m}experts.{e}.gate_proj", I, H)
                fp4(f"{m}experts.{e}.up_proj", I, H)
                fp4(f"{m}experts.{e}.down_proj", H, I)
    if mt == "qwen3_vl_moe":
        v = cfg["vision_config"]
        D, VI = v["hidden_size"], v["intermediate_size"]
        Mw = D * v["spatial_merge_size"] ** 2
        vp = "model.visual."

        def lin(base, n, k):
            bf16(base + ".weight", (n, k))
            bf16(base + ".bias", (n,))

        def ln(base, w):
            bf16(base + ".weight", (w,), "norm")
            bf16(base + ".bias", (w,))

        bf16(vp + "patch_embed.proj.weight", (D, v["in_channels"], v["temporal_patch_size"], v["patch_size"], v["patch_size"]))
        bf16(vp + "patch_embed.proj.bias", (D,))
        bf16(vp + "pos_embed.weight", (v["num_position_embeddings"], D))
        for b in range(v["depth"]):
            bp = f"{vp}blocks.{b}."
            ln(bp + "norm1", D)
            ln(bp + "norm2", D)
            lin(bp + "attn.qkv", 3 * D, D)
            lin(bp + "attn.proj", D, D)
            lin(bp + "mlp.linear_fc1", VI, D)
            lin(bp + "mlp.linear_fc2", D, VI)
        mergers = [("merger", D)] + [(f"deepstack_merger_list.{i}", Mw) for i in range(len(v["deepstack_visual_indexes"]))]
        for nm, norm_w in mergers:
            ln(vp + nm + ".norm", norm_w)
            lin(vp + nm + ".linear_fc1", Mw, Mw)
            lin(vp + nm + ".linear_fc2", v["out_hidden_size"], Mw)
    return out, packed


def build(preset, seed):
    """-> (config dict, hf_quant_config dict or None, {name: (dtype, stored bits)})."""
    cfg, hf = make_config(preset)
    rows, packed = table(cfg, bare=(preset == "dense-bare"))
    tensors = {}
    for name, dt, shape, role in rows:
        tensors[name] = (dt, fill(name, dt, shape, role, seed, packed))
    return cfg, hf, tensors


def digest(tensors):
    """Order-independent content digest: the sum (mod 2^64) over tensors of FNV-1a over
    "name|dtype|d0,d1,..|" and the tensor's bytes; and the tensor count."""
    total = 0
    for name, (dt, arr) in tensors.items():
        head = f"{name}|{dt}|{','.join(str(int(x)) for x in arr.shape)}|".encode()
        total = (total + fnv1a64(np.ascontiguousarray(arr).tobytes(), fnv1a64(head))) & M64
    return total, len(tensors)


def write_safetensors(path, tensors):
    """tensors: ordered {name: (dtype string, numpy array of the stored bits)}."""
    header, blobs, off = {}, [], 0
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dt, "shape": [int(x) for x in arr.shape], "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    header["__metadata__"] = {"format": "pt"}
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * (-len(hj) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


IDS = [1, 17, 133, 42, 201, 7, 99, 64, 3, 180, 25, 11]   # no id is a vision placeholder (248..251)


def write_ckpt(preset, out, seed):
    cfg, hf, tensors = build(preset, seed)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    if hf is not None:
        with open(os.path.join(out, "hf_quant_config.json"), "w") as f:
            json.dump(hf, f, indent=4)
    with open(os.path.join(out, "ids.json"), "w") as f:
        json.dump(IDS, f)
    write_safetensors(os.path.join(out, "model.safetensors"), tensors)
    return cfg, tensors


def cmd_ckpt(a):
    cfg, tensors = write_ckpt(a.preset, a.out, a.seed)
    d, n = digest(tensors)
    nbytes = sum(arr.nbytes for _, arr in tensors.values())
    print(f"wrote {a.out}: preset {a.preset}, {n} tensors, {nbytes} bytes, content digest {d:#018x}")


# ----------------------------------------------------------------------------------------------
# selftest
# ----------------------------------------------------------------------------------------------
def _scalar_fp4(ck, prefix):
    """An independent, element-at-a-time decode of one NVFP4 matrix (float64 from the stored bits)."""
    packed = prefix + ".weight_packed" in ck
    codes = ck.raw(prefix + (".weight_packed" if packed else ".weight"))
    scale = ck.raw(prefix + ".weight_scale")
    g = float(ck.raw(prefix + (".weight_global_scale" if packed else ".weight_scale_2")).reshape(()))
    mag = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    n, half = codes.shape
    out = np.empty((n, 2 * half), np.float32)
    for r in range(n):
        for c in range(2 * half):
            byte = int(codes[r, c // 2])
            nib = (byte >> 4) if (c & 1) else (byte & 0xF)
            v = mag[nib & 7] * (-1.0 if nib & 8 else 1.0)
            s8 = int(scale[r, c // 16])
            e, m = (s8 >> 3) & 0xF, s8 & 7
            s = (m / 8.0) * 2.0 ** -6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
            # the reference's float32 order: the block scale times (or over) the global first
            sg = np.float32(s) / np.float32(g) if packed else np.float32(s) * np.float32(g)
            out[r, c] = np.float32(v) * sg
    return out


def cmd_selftest(a):
    with open(os.path.join(a.ckpt, "ids.json")) as f:
        ids = json.load(f)
    m = ref.Model(a.ckpt, dtype="float64", threads=2)
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("[ OK ] " if ok else "[FAIL] ") + what, flush=True)
        fails += 0 if ok else 1

    # (1) the NVFP4 reader
    fp4 = sorted(n[:-len(".weight_scale")] for n in m.ck.index if n.endswith(".weight_scale"))
    if fp4:
        picks = [fp4[0], fp4[len(fp4) // 2], fp4[-1]]
        worst = max(float(np.abs(ref.Model(a.ckpt, dtype="float32", threads=2)._fp4(p) - _scalar_fp4(m.ck, p)).max())
                    for p in picks)
        check(worst == 0.0, f"NVFP4 reader == an element-at-a-time decode on {len(picks)} of {len(fp4)} matrices (max |d| {worst})")
    else:
        check(True, "no NVFP4 matrices in this checkpoint (BF16 dense)")

    # (2) chunk invariance
    whole, _ = ref.run_forward(m, ids)
    lw = m.logits(whole)
    for chunk in (5, 1):
        part, _ = ref.run_forward(m, ids, chunk=chunk)
        d = float(np.abs(m.logits(part) - lw).max())
        check(d < 1e-9, f"feeding {chunk} token(s) at a time == the whole sequence (max |d logit| {d:.3e})")

    # (3) greedy == scoring its own output
    new, steps = ref.greedy(m, ids[:6], 6, stop_eos=False)
    res = ref.score_ids(m, ids[:6] + new)
    d = max(abs(res["logprobs"][6 + j] - steps[j]["top"][0][1]) for j in range(len(new)))
    top1 = all(res["top"][6 + j][0][0] == new[j] for j in range(len(new)))
    check(d < 1e-9 and top1, f"greedy generation reproduces teacher-forced scoring ({len(new)} tokens, max |d logprob| {d:.3e})")

    # (4) every layout variant must move the logits
    flips = {"rope": ["adjacent", "none"], "kvmap": ["mod"], "norm": ["1p"], "qknorm": ["off"], "gateup": ["up"]}
    if m.n_exp:
        flips["renorm"] = ["off"]
    if fp4:
        flips["nibble"] = ["hi"]
        flips["gscale" if m.model_type == "qwen3_vl_moe" else "ws2"] = ["mul" if m.model_type == "qwen3_vl_moe" else "div"]
    for key, values in flips.items():
        for val in values:
            mv = ref.Model(a.ckpt, dtype="float64", threads=2, variants={key: val})
            xv, _ = ref.run_forward(mv, ids)
            d = float(np.abs(mv.logits(xv) - lw).max())
            check(d > 1e-3, f"variant {key}={val} changes the logits (max |d| {d:.3e})")
    print(f"{'selftest OK' if fails == 0 else 'SELFTEST FAILED'} ({a.ckpt})")
    return 1 if fails else 0


# ----------------------------------------------------------------------------------------------
# torch-check: transformers' own classes
# ----------------------------------------------------------------------------------------------
def cmd_torch_check(a):
    import torch
    import transformers
    from transformers import AutoConfig

    with open(os.path.join(a.ckpt, "ids.json")) as f:
        ids = json.load(f)
    # Two readings of the same weights. transformers as it ships keeps three float32 islands in a
    # float64 model — the RMSNorm interior (`hidden_states.to(torch.float32)`), the router's softmax
    # (`dtype=torch.float`) and the rotary angles — so its numbers carry float32 noise (about 1e-6
    # here) and are compared at 1e-4 with the reference's default. With those three taken in float64
    # (precision only: the same formulas, the modules' own code otherwise) everything is float64 and
    # the two must agree to rounding noise — the layout proof.
    refs = {}
    for angle in ("f32", "f64"):
        m = ref.Model(a.ckpt, dtype="float64", threads=2, variants={"ropeangle": angle})
        states = []
        x, _ = ref.run_forward(m, ids, tap=lambda n, arr: states.append(np.array(arr)) if n.startswith("h_") else None)
        # transformers records the embedding, every layer's output but the last, and the NORMED last
        # hidden state (hidden_states[-1] is post model.norm).
        refs[angle] = (m.logits(x), states[:-1] + [m.final_norm(x)])

    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    cfg.pop("quantization_config", None)
    mt = cfg["model_type"]
    tok = torch.tensor([ids])
    P = m.prefix                # (m: the last reference built — the weights do not depend on the variant)

    def W(prefix):
        return torch.from_numpy(np.array(m.linear(prefix), np.float64))

    def D(name):
        return torch.from_numpy(np.array(m.dense(name), np.float64))

    fails = 0

    def compare(what, angle, logits, hidden):
        nonlocal fails
        ref_logits, states = refs[angle]
        tol = 1e-9 if angle == "f64" else 1e-4
        dl = float(np.abs(logits - ref_logits).max())
        dh = max(float(np.abs(h - s).max()) for h, s in zip(hidden, states)) if hidden is not None else 0.0
        ok = dl < tol and dh < tol and (hidden is None or len(hidden) == len(states))
        argmax = bool((logits.argmax(-1) == ref_logits.argmax(-1)).all())
        print(f"{'[ OK ]' if ok and argmax else '[FAIL]'} {what} [{'as shipped (float32 norm interior, router softmax, rotary angles)' if angle == 'f32' else 'those three in float64'}]: "
              f"max |d logit| {dl:.3e}, max |d hidden state| {dh:.3e} over {0 if hidden is None else len(hidden)} states "
              f"(bound {tol:g}), argmax equal {argmax}", flush=True)
        fails += 0 if (ok and argmax) else 1

    def rotary64(rot, theta, dim):
        """The module's own rotary with the ANGLES in float64: the same inv_freq formula, the same
        position axes and (the VL text model) the module's own apply_interleaved_mrope."""
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))

        def fwd(x, position_ids):
            if hasattr(rot, "apply_interleaved_mrope"):
                if position_ids.ndim == 2:
                    position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
                freqs = position_ids[..., None].double() * inv                      # [3, bs, seq, dim / 2]
                freqs = rot.apply_interleaved_mrope(freqs.clone(), rot.mrope_section)
            else:
                freqs = position_ids[..., None].double() * inv                      # [bs, seq, dim / 2]
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos(), emb.sin()

        rot.forward = fwd

    def float64_islands(*norm_classes):
        """RMSNorm without its float32 interior and softmax without a forced dtype (process-wide:
        this process does nothing else afterwards)."""
        def norm_forward(self, hidden_states):
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))

        for cls in norm_classes:
            cls.forward = norm_forward
        orig = torch.nn.functional.softmax
        torch.nn.functional.softmax = lambda input, dim=None, _stacklevel=3, dtype=None: orig(input, dim=dim)

    def layer_sd(sd, dst_prefix, L, fused):
        s = f"{P}layers.{L}."
        d = f"{dst_prefix}layers.{L}."
        sd[d + "input_layernorm.weight"] = D(s + "input_layernorm.weight")
        sd[d + "post_attention_layernorm.weight"] = D(s + "post_attention_layernorm.weight")
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[d + f"self_attn.{nm}.weight"] = W(s + "self_attn." + nm)
        sd[d + "self_attn.q_norm.weight"] = D(s + "self_attn.q_norm.weight")
        sd[d + "self_attn.k_norm.weight"] = D(s + "self_attn.k_norm.weight")
        if not m.n_exp:
            for nm in ("gate_proj", "up_proj", "down_proj"):
                sd[d + f"mlp.{nm}.weight"] = W(s + "mlp." + nm)
            return
        sd[d + "mlp.gate.weight"] = D(s + "mlp.gate.weight")
        if fused:
            # the inverse of llm-compressor's unpacking: gate_up[e] = [gate_proj.T | up_proj.T], down[e] = down_proj.T
            gu, dn = [], []
            for e in range(m.n_exp):
                g_, u_, d_ = (W(f"{s}mlp.experts.{e}.{nm}") for nm in ("gate_proj", "up_proj", "down_proj"))
                gu.append(torch.cat([g_.T, u_.T], dim=-1))
                dn.append(d_.T)
            sd[d + "mlp.experts.gate_up_proj"] = torch.stack(gu)
            sd[d + "mlp.experts.down_proj"] = torch.stack(dn)
        else:
            for e in range(m.n_exp):
                for nm in ("gate_proj", "up_proj", "down_proj"):
                    sd[d + f"mlp.experts.{e}.{nm}.weight"] = W(f"{s}mlp.experts.{e}.{nm}")

    print(f"# transformers {transformers.__version__}, torch {torch.__version__}, model_type {mt}")
    with torch.no_grad():
        if mt in ("qwen3", "qwen3_moe"):
            from transformers import Qwen3ForCausalLM, Qwen3MoeForCausalLM
            hc = AutoConfig.for_model(**cfg)
            model = (Qwen3ForCausalLM if mt == "qwen3" else Qwen3MoeForCausalLM)(hc).double().eval()
            sd = {"model.embed_tokens.weight": D(P + "embed_tokens.weight"), "model.norm.weight": D(P + "norm.weight"),
                  "lm_head.weight": D(m.head_name)}
            for L in range(m.total_layers):
                layer_sd(sd, "model.", L, fused=False)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            assert not missing and not unexpected, (missing, unexpected)
            for angle in ("f32", "f64"):
                if angle == "f64":
                    rotary64(model.model.rotary_emb, m.theta, m.head_dim)
                    float64_islands(type(model.model.norm))
                out = model(input_ids=tok, output_hidden_states=True)
                compare(type(model).__name__, angle, out.logits[0].numpy(), [h[0].numpy() for h in out.hidden_states])
        else:
            from transformers import Qwen3VLMoeForConditionalGeneration
            from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextModel
            hc = AutoConfig.for_model(**cfg)
            fused = hasattr(Qwen3VLMoeTextModel(hc.text_config).layers[0].mlp.experts, "gate_up_proj")
            # (i) the text model alone
            text = Qwen3VLMoeTextModel(hc.text_config).double().eval()
            sd = {"embed_tokens.weight": D(P + "embed_tokens.weight"), "norm.weight": D(P + "norm.weight")}
            for L in range(m.total_layers):
                layer_sd(sd, "", L, fused)
            missing, unexpected = text.load_state_dict(sd, strict=False)
            assert not missing and not unexpected, (missing, unexpected)
            # (ii) the whole model, fed text only: get_rope_index's three equal axes, then the MROPE rotary
            full = Qwen3VLMoeForConditionalGeneration(hc).double().eval()
            fsd = {"model.language_model." + k: v for k, v in sd.items()}
            fsd["lm_head.weight"] = D(m.head_name)
            missing, unexpected = full.load_state_dict(fsd, strict=False)
            assert not unexpected and all(k.startswith("model.visual.") for k in missing), (missing, unexpected)
            head = np.array(m.dense(m.head_name), np.float64)
            for angle in ("f32", "f64"):
                if angle == "f64":
                    rotary64(text.rotary_emb, m.theta, m.head_dim)
                    rotary64(full.model.language_model.rotary_emb, m.theta, m.head_dim)
                    float64_islands(type(text.norm))
                out = text(input_ids=tok)
                compare(f"Qwen3VLMoeTextModel (experts {'fused' if fused else 'per expert'})", angle,
                        out.last_hidden_state[0].numpy() @ head.T, None)
                out = full(input_ids=tok)
                compare("Qwen3VLMoeForConditionalGeneration, text only (the tower at its init, unused)", angle,
                        out.logits[0].numpy(), None)
    print(f"{'torch-check OK' if fails == 0 else 'TORCH-CHECK FAILED'} ({a.ckpt})")
    return 1 if fails else 0


# ----------------------------------------------------------------------------------------------
# vectors
# ----------------------------------------------------------------------------------------------
def _carray(name, values, ctype="double", per_line=4, fmt="{:.17g}"):
    vals = [fmt.format(v) for v in values]
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return f"inline constexpr {ctype} {name}[{max(1, len(vals))}] = {{\n    " + ",\n    ".join(lines or ["0"]) + "};\n"


def cmd_vectors(a):
    presets = ["vl", "moe", "dense", "dense-bare"]
    out = ["// GENERATED by tools/qwen3_synth.py vectors --seed %d — do not edit.\n" % a.seed,
           "// Per preset: the synthetic checkpoint's config.json, the content digest of its tensors (the\n"
           "// fixture in tests/unit/qwen3_fixture.hpp regenerates them and must reach it), the token ids, and\n"
           "// what tools/qwen3_reference.py computes from them in float64.\n",
           "#pragma once\n#include <cstdint>\n\nnamespace qwen3_reference_vectors {\n\n",
           f"inline constexpr uint64_t kSeed = {a.seed}ull;\n",
           f"inline constexpr int kTokens = {len(IDS)};\n",
           _carray("kIds", IDS, "int64_t", 16, "{}"), "\n"]
    for preset in presets:
        ns = preset.replace("-", "_")
        with tempfile.TemporaryDirectory() as d:
            cfg, tensors = write_ckpt(preset, d, a.seed)
            dg, n = digest(tensors)
            nbytes = sum(arr.nbytes for _, arr in tensors.values())
            m = ref.Model(d, dtype="float64", threads=2, variants={"ropeangle": "f64"})
            taps = {}
            x, _ = ref.run_forward(m, IDS, tap=lambda nm, arr: taps.__setitem__(nm, np.array(arr)),
                                   detail_layers=range(m.total_layers))
            logits = m.logits(x)
            res = ref.score_ids(m, IDS)
        out.append(f"namespace {ns} {{\n")
        out.append("inline constexpr char kConfigJson[] = R\"QW3(" + json.dumps(cfg, indent=1) + ")QW3\";\n")
        out.append(f"inline constexpr bool kBareNames = {'true' if preset == 'dense-bare' else 'false'};\n")
        out.append(f"inline constexpr uint64_t kDigest = {dg:#018x}ull;\n")
        out.append(f"inline constexpr int kTensors = {n};\n")
        out.append(f"inline constexpr uint64_t kBytes = {nbytes}ull;\n")
        out.append(f"inline constexpr int kVocab = {m.vocab};\n")
        out.append(f"inline constexpr int kLayers = {m.total_layers};\n")
        out.append(f"inline constexpr int kTopK = {m.top_k if m.n_exp else 0};\n")
        # the residual stream after the embedding and after every layer: sum and sum of |x| over [T, H]
        hs = [taps[f"h_{L:02d}"] for L in range(m.total_layers + 1)]
        out.append(_carray("kStateSum", [float(h.sum()) for h in hs]))
        out.append(_carray("kStateAbsSum", [float(np.abs(h).sum()) for h in hs]))
        out.append(_carray("kLogitsLast", logits[-1].tolist()))
        out.append(_carray("kLogitsFirst", logits[0].tolist()))
        out.append(_carray("kLogprobs", res["logprobs"][1:]))
        if m.n_exp:
            rid = np.concatenate([np.sort(taps[f"L{L:02d}_router_ids"], axis=-1).reshape(-1) for L in range(m.total_layers)])
            out.append(_carray("kRouterIds", rid.tolist(), "int32_t", 24, "{}"))
        else:
            out.append(_carray("kRouterIds", [], "int32_t", 24, "{}"))
        out.append(f"}}  // namespace {ns}\n\n")
        print(f"{preset}: {n} tensors, {nbytes} bytes, digest {dg:#018x}, mean logprob {np.mean(res['logprobs'][1:]):.6f}")
    out.append("}  // namespace qwen3_reference_vectors\n")
    with open(a.out, "w") as f:
        f.write("".join(out))
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description="synthetic plain-Qwen3 checkpoints and the checks on them")
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("ckpt")
    p.add_argument("--preset", required=True, choices=["vl", "moe", "dense", "dense-bare"])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_ckpt)
    p = sub.add_parser("selftest")
    p.add_argument("--ckpt", required=True)
    p.set_defaults(fn=cmd_selftest)
    p = sub.add_parser("torch-check")
    p.add_argument("--ckpt", required=True)
    p.set_defaults(fn=cmd_torch_check)
    p = sub.add_parser("vectors")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_vectors)
    a = ap.parse_args()
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
