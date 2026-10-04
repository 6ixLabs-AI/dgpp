#!/usr/bin/env python3
"""gemma4_synth.py — a tiny synthetic Gemma 4 checkpoint in the releases' container (2026-10-04).

Writes config.json + model.safetensors with the real tensor names, dtypes and formats (BF16
norms / embedding / attention, modelopt NVFP4 sets for the MLP) for a model small enough to run
anywhere: the fixture the numpy reference (tools/gemma4_reference.py), the transformers class
(tools/gemma4_torch_check.py) and the engine's host reference (src/models/gemma4/reference.cpp,
tests/unit/gemma4_reference_test.cpp) are all run on.

  make     --out DIR [--recipe mlp|weight-only] [--seed N]   write the checkpoint
                         mlp:         the 31B release's shape — BF16 attention, NVFP4 MLP with input_scale
                         weight-only: the 26B-A4B release's — NVFP4 attention / MLP / experts, no
                                      input_scale, the MoE block (8 experts, top 3) with BF16 routers
  vectors  --out FILE.hpp [--seed N]                          the unit test's expected values: the
                                                              numpy reference on the checkpoint

THE CONTENT IS A PURE FUNCTION OF (tensor name, seed) — "Gemma4Synth v1" — so the C++ fixture
(tests/unit/gemma4_synth.hpp) regenerates the same bytes without reading a file:

  stream(name, seed)[k] = mix(fnv1a64(name) ^ seed + (k + 1) * 0x9E3779B97F4A7C15)     (uint64, wrapping)
  mix(z): z = (z ^ z >> 30) * 0xBF58476D1CE4E5B9; z = (z ^ z >> 27) * 0x94D049BB133111EB; z ^ z >> 31
  u[k] = float32(stream[k] >> 40) / 2^23 - 1                                           in [-1, 1), exact

  BF16 matrix / embedding   bf16(u * amp)             amp 0.25 (projections), 0.5 (embedding),
                                                      4.0 (router.proj: logits wide enough to route by)
  BF16 norm weight          bf16(1 + 0.25 u)
  BF16 layer_scalar         bf16(0.875 + 0.125 u)
  NVFP4 `.weight`           byte k = stream[k] >> 56                      (two random e2m1 codes)
  NVFP4 `.weight_scale`     0x38 + (stream[k] >> 60)                      (e4m3 1.0 .. 3.75)
  NVFP4 `.weight_scale_2`   (1 + (stream[0] >> 60)) / 256                 (F32, a multiple of 2^-8)
  NVFP4 `.input_scale`      1.0                                           (present under the mlp recipe; unused)

No encoder is involved: the NVFP4 codes ARE the random draw, so both languages write identical
bytes and what is under test is the decode.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)


def fnv1a64(name):
    h = 1469598103934665603
    for b in name.encode():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def stream(name, seed, n):
    with np.errstate(over="ignore"):
        z = np.uint64(fnv1a64(name) ^ seed) + np.arange(1, n + 1, dtype=np.uint64) * GOLDEN
        z = (z ^ (z >> np.uint64(30))) * M1
        z = (z ^ (z >> np.uint64(27))) * M2
        return z ^ (z >> np.uint64(31))


def uniform(name, seed, n):
    """float32 in [-1, 1), exact (24 bits)."""
    return (stream(name, seed, n) >> np.uint64(40)).astype(np.float32) / np.float32(8388608.0) - np.float32(1.0)


def to_bf16(x):
    """float32 -> BF16 bits, round to nearest even."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


# The synthetic geometry: both attention kinds, a window shorter than the sequence, a soft-cap
# small enough to bite. NVFP4 needs every K a multiple of 16.
TEXT_CONFIG = {
    "attention_bias": False, "attention_dropout": 0.0, "attention_k_eq_v": True, "bos_token_id": 2,
    "dtype": "bfloat16", "enable_moe_block": False, "eos_token_id": 1, "final_logit_softcapping": 1.5,
    "global_head_dim": 16, "head_dim": 8, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 32,
    "hidden_size_per_layer_input": 0, "initializer_range": 0.02, "intermediate_size": 64,
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "sliding_attention", "full_attention"],
    "max_position_embeddings": 4096, "model_type": "gemma4_text", "num_attention_heads": 4,
    "num_global_key_value_heads": 1, "num_hidden_layers": 6, "num_key_value_heads": 2,
    "num_kv_shared_layers": 0, "pad_token_id": 0, "rms_norm_eps": 1e-06,
    "rope_parameters": {
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
    "sliding_window": 4, "tie_word_embeddings": True, "use_bidirectional_attention": "vision",
    "use_cache": True, "use_double_wide_mlp": False, "vocab_size": 96, "vocab_size_per_layer_input": 96,
}

# The token ids every consumer scores (12 tokens: three windows' worth for the sliding layers).
SYNTH_IDS = [2, 17, 45, 9, 80, 33, 5, 61, 17, 94, 28, 70]


MOE_CONFIG = {"enable_moe_block": True, "num_experts": 8, "top_k_experts": 3, "moe_intermediate_size": 16}


def text_config(recipe):
    t = dict(TEXT_CONFIG)
    if recipe == "weight-only":
        t.update(MOE_CONFIG)
    return t


def config_json(recipe):
    t = text_config(recipe)
    layers = t["num_hidden_layers"]
    ignore = ["lm_head", "model.embed_vision*"]
    if recipe == "mlp":
        ignore += [f"model.language_model.layers.{l}.self_attn*" for l in range(layers)]
    else:
        ignore += [f"model.language_model.layers.{l}.router*" for l in range(layers)]
    ignore.append("model.vision_tower*")
    q = {"config_groups": {"group_0": {
            "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "targets": ["Linear"]}},
         "ignore": ignore, "quant_algo": "NVFP4",
         "producer": {"name": "modelopt", "version": "synthetic"}, "quant_method": "modelopt"}
    return {"architectures": ["Gemma4ForConditionalGeneration"], "audio_config": None, "audio_token_id": 93,
            "dtype": "bfloat16", "eos_token_id": [1, 6], "image_token_id": 92, "model_type": "gemma4",
            "text_config": t, "tie_word_embeddings": True, "video_token_id": 94,
            "vision_config": {"model_type": "gemma4_vision"}, "quantization_config": q}


def tensors(recipe, seed):
    """[(name, safetensors dtype, shape, bytes)] in the table's order (globals, then each layer)."""
    t = text_config(recipe)
    H, I, V = t["hidden_size"], t["intermediate_size"], t["vocab_size"]
    heads = t["num_attention_heads"]
    out = []

    def bf16(name, shape, kind):
        n = int(np.prod(shape))
        u = uniform(name, seed, n)
        if kind == "norm":
            x = np.float32(1.0) + np.float32(0.25) * u
        elif kind == "scalar":
            x = np.float32(0.875) + np.float32(0.125) * u
        elif kind == "embed":
            x = u * np.float32(0.5)
        elif kind == "router":
            x = u * np.float32(4.0)
        else:
            x = u * np.float32(0.25)
        out.append((name, "BF16", list(shape), to_bf16(x).tobytes()))

    def fp4(base, rows, cols, act_scale):
        s = stream(base + ".weight", seed, rows * (cols // 2))
        out.append((base + ".weight", "U8", [rows, cols // 2], (s >> np.uint64(56)).astype(np.uint8).tobytes()))
        s = stream(base + ".weight_scale", seed, rows * (cols // 16))
        out.append((base + ".weight_scale", "F8_E4M3", [rows, cols // 16],
                    (np.uint64(0x38) + (s >> np.uint64(60))).astype(np.uint8).tobytes()))
        s0 = int(stream(base + ".weight_scale_2", seed, 1)[0] >> np.uint64(60))
        out.append((base + ".weight_scale_2", "F32", [], np.float32((1 + s0) / 256.0).tobytes()))
        if act_scale:
            out.append((base + ".input_scale", "F32", [], np.float32(1.0).tobytes()))

    def matrix(base, rows, cols, nvfp4):
        if nvfp4:
            fp4(base, rows, cols, False)
        else:
            bf16(base + ".weight", [rows, cols], "matrix")

    mp = "model.language_model."
    bf16(mp + "embed_tokens.weight", [V, H], "embed")
    bf16(mp + "norm.weight", [H], "norm")
    for l, kind in enumerate(t["layer_types"]):
        p = f"{mp}layers.{l}."
        sliding = kind == "sliding_attention"
        hd = t["head_dim"] if sliding else t["global_head_dim"]
        kv = t["num_key_value_heads"] if sliding else t["num_global_key_value_heads"]
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            bf16(p + n + ".weight", [H], "norm")
        bf16(p + "layer_scalar", [1], "scalar")
        q4 = recipe == "weight-only"
        bf16(p + "self_attn.q_norm.weight", [hd], "norm")
        bf16(p + "self_attn.k_norm.weight", [hd], "norm")
        matrix(p + "self_attn.q_proj", heads * hd, H, q4)
        matrix(p + "self_attn.k_proj", kv * hd, H, q4)
        if sliding:
            matrix(p + "self_attn.v_proj", kv * hd, H, q4)
        matrix(p + "self_attn.o_proj", H, heads * hd, q4)
        act = recipe == "mlp"
        fp4(p + "mlp.gate_proj", I, H, act)
        fp4(p + "mlp.up_proj", I, H, act)
        fp4(p + "mlp.down_proj", H, I, act)
        if t.get("enable_moe_block"):
            E, Im = t["num_experts"], t["moe_intermediate_size"]
            for n in ("post_feedforward_layernorm_1", "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2"):
                bf16(p + n + ".weight", [H], "norm")
            bf16(p + "router.proj.weight", [E, H], "router")
            bf16(p + "router.scale", [H], "norm")
            bf16(p + "router.per_expert_scale", [E], "norm")
            for e in range(E):
                fp4(f"{p}moe.experts.{e}.gate_proj", Im, H, act)
                fp4(f"{p}moe.experts.{e}.up_proj", Im, H, act)
                fp4(f"{p}moe.experts.{e}.down_proj", H, Im, act)
    # One encoder tensor, as the releases carry a vision tower the engine ignores.
    bf16("model.vision_tower.std_bias", [8], "norm")
    return out


def content_hash(items):
    """FNV-1a-64 over every tensor's bytes, tensors in name order: the fixture's identity."""
    h = 1469598103934665603
    for _, _, _, data in sorted(items):
        for b in data:
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def write_safetensors(path, items):
    header, off = {}, 0
    for name, dtype, shape, data in items:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    header["__metadata__"] = {"format": "pt"}
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * (-len(hj) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for _, _, _, data in items:
            f.write(data)


def make(out, recipe, seed):
    os.makedirs(out, exist_ok=True)
    items = tensors(recipe, seed)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config_json(recipe), f, indent=1)
    write_safetensors(os.path.join(out, "model.safetensors"), items)
    with open(os.path.join(out, "ids.json"), "w") as f:
        json.dump(SYNTH_IDS, f)
    return items


def cmd_make(a):
    items = make(a.out, a.recipe, a.seed)
    print(f"wrote {a.out}: {len(items)} tensors, recipe {a.recipe}, seed {a.seed}, "
          f"content hash {content_hash(items):#018x}")


def _carray(name, arr, per_line=6):
    flat = np.asarray(arr, np.float32).reshape(-1)
    rows = []
    for i in range(0, flat.size, per_line):
        rows.append("    " + ", ".join(f"{float(v):.9g}f" for v in flat[i:i + per_line]) + ",")
    return f"inline constexpr float {name}[{flat.size}] = {{\n" + "\n".join(rows) + "\n};\n"


def cmd_vectors(a):
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gemma4_reference as ref

    parts = ["// GENERATED by tools/gemma4_synth.py vectors — do not edit. The numpy reference\n"
             "// (tools/gemma4_reference.py, float32) on the synthetic checkpoint \"Gemma4Synth v1\".\n"
             "#pragma once\n#include <cstdint>\n\nnamespace gemma4_vectors {\n\n"
             f"inline constexpr uint64_t kSeed = {a.seed}ull;\n"
             f"inline constexpr int kTokens = {len(SYNTH_IDS)};\n"
             f"inline constexpr int64_t kIds[{len(SYNTH_IDS)}] = {{{', '.join(str(i) for i in SYNTH_IDS)}}};\n\n"]
    for recipe, tag in (("mlp", "Mlp"), ("weight-only", "Moe")):
        with tempfile.TemporaryDirectory() as d:
            items = make(d, recipe, a.seed)
            model = ref.Model(d)
            taps = {}
            st = model.new_state()
            layers = list(range(model.n_layers))
            x = model.forward(SYNTH_IDS, st, tap=lambda n, v: taps.__setitem__(n, np.array(v)), detail_layers=layers)
            logits = model.logits(x)
            lp = model.logprobs(x)
            # The same sequence fed one token at a time must give the same stream (the KV cache path).
            st2 = model.new_state()
            x2 = np.concatenate([model.forward([t], st2) for t in SYNTH_IDS])
            assert np.abs(x2 - x).max() < 1e-4, np.abs(x2 - x).max()
            parts.append(f"// --- recipe {recipe} ---------------------------------------------------\n")
            parts.append(f"inline constexpr uint64_t kContentHash{tag} = {content_hash(items):#018x}ull;\n")
            parts.append(f"inline constexpr int kTensors{tag} = {len(items)};\n")
            with open(os.path.join(d, "config.json")) as f:
                parts.append(f"inline constexpr const char* kConfigJson{tag} = R\"JSON({f.read()})JSON\";\n")
            last = len(SYNTH_IDS) - 1
            if recipe == "mlp":
                parts.append(_carray(f"kEmbedLast{tag}", taps["h_00"][last]))
                for L in (0, 2):
                    parts.append(_carray(f"kL{L}AttnQLast{tag}", taps[f"L{L:02d}_attn_q"][last]))
                    parts.append(_carray(f"kL{L}AttnKLast{tag}", taps[f"L{L:02d}_attn_k"][last]))
                    parts.append(_carray(f"kL{L}AttnVLast{tag}", taps[f"L{L:02d}_attn_v"][last]))
                    parts.append(_carray(f"kL{L}AttnOut{tag}", taps[f"L{L:02d}_attn_out"]))
                    parts.append(_carray(f"kL{L}MlpOut{tag}", taps[f"L{L:02d}_mlp_out"]))
            else:
                # The MoE block: layer 0's routing (ids best first, weights after the renormalization
                # and per_expert_scale) and the experts' sum; a full layer's NVFP4 attention output.
                ids0 = taps["L00_router_ids"]
                k = ids0.shape[1]
                parts.append(f"inline constexpr int kTopK{tag} = {k};\n")
                parts.append(f"inline constexpr int kL0RouterIds{tag}[{ids0.size}] = {{"
                             + ", ".join(str(int(i)) for i in ids0.reshape(-1)) + "};\n")
                parts.append(_carray(f"kL0RouterW{tag}", taps["L00_router_w"]))
                parts.append(_carray(f"kL0MoeOut{tag}", taps["L00_moe_out"]))
                parts.append(_carray(f"kL2AttnOut{tag}", taps["L02_attn_out"]))
                # How far apart the k-th and (k+1)-th router probabilities are at their closest: the
                # margin a consumer's own arithmetic has before a selection can flip.
                gaps = []
                for L in range(model.n_layers):
                    lg = np.sort(taps[f"L{L:02d}_router_logits"].astype(np.float64), axis=-1)[:, ::-1]
                    gaps.append(float((lg[:, k - 1] - lg[:, k]).min()))
                parts.append(f"// smallest gap between the top-k boundary's router logits, per layer: "
                             + ", ".join(f"{g:.4f}" for g in gaps) + "\n")
            for L in range(model.n_layers + 1):
                parts.append(_carray(f"kHidden{L}{tag}", taps[f"h_{L:02d}"]))
            parts.append(_carray(f"kFinalNorm{tag}", model.final_norm(x)))
            parts.append(_carray(f"kLogits{tag}", logits))
            parts.append(_carray(f"kLogprobsLast{tag}", lp[last]))
    parts.append("\n}  // namespace gemma4_vectors\n")
    with open(a.out, "w") as f:
        f.write("".join(parts))
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make")
    p.add_argument("--out", required=True)
    p.add_argument("--recipe", choices=["mlp", "weight-only"], default="mlp")
    p.add_argument("--seed", type=int, default=20261004)
    p.set_defaults(fn=cmd_make)
    p = sub.add_parser("vectors")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261004)
    p.set_defaults(fn=cmd_vectors)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
