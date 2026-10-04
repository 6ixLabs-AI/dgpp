#!/usr/bin/env python3
"""minimax_m2_synth.py — tiny synthetic MiniMax-M2 checkpoints and the checks that run on them
(2026-10-04). No GPU. `ckpt`, `selftest` and `vectors` need numpy only; `hf-check` additionally
needs torch and a transformers that ships MiniMaxM2ForCausalLM.

The synthetic checkpoint carries the real release's tensor names, containers and quirks
(lukealonso/MiniMax-M2.7-NVFP4: modelopt NVFP4 routed experts, BF16 everything else, the
activation scales in a file of their own with a few experts' missing, a config.json that declares
an MTP head the file does not have, and a stray safetensors file that repeats a tensor name and is
not in the index) at toy sizes, so tools/minimax_m2_reference.py and the engine's config parser /
binding table can be exercised end to end on a laptop.

  ckpt     --out DIR [--seed N]
        writes config.json, generation_config.json, the model-*.safetensors shards,
        model-inputscales.safetensors, model.safetensors.index.json, the stray file, ids.json and
        expected_dequant.npz (the float value every quantized matrix must dequantize to).
  selftest --ckpt DIR
        through minimax_m2_reference.Model: (1) every quantized matrix dequantizes to
        expected_dequant.npz exactly, and the stray file is not read; (2) feeding a sequence
        whole, in chunks and token by token gives the same logits; (3) greedy generation
        reproduces teacher-forced scoring of its own output; (4) each debug variant changes the
        logits (a variant that changes nothing is a convention the test cannot see).
  hf-check --ckpt DIR
        the same weights loaded into transformers' MiniMaxM2ForCausalLM (float64, eager attention
        and experts), logits compared with the reference's. This is the check against the model's
        published implementation.
  vectors  --ckpt DIR --out tests/unit/minimax_attn_vectors.hpp
        the test vectors of tests/unit/minimax_attn_test.cpp: one layer's attention weights as
        BF16 bits, a deterministic BF16 input, the layer's NVFP4 expert containers, and what the
        reference makes of them in float64.

What a pass here does and does not show: the reference is self-consistent and agrees with
transformers on a random model of the same class. The writers follow the same reading of the
modelopt container as the readers (nibble order, the multiplier), so those conventions are only
proven on the real checkpoint.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minimax_m2_reference as ref  # noqa: E402

_E2M1_MAG = ref._E2M1[:8].astype(np.float64)
_E4M3_MAG = ref._E4M3[:127].astype(np.float64)


def _nearest(x, grid):
    """Index of the nearest value of an ascending non-negative grid, saturating, ties to the even index."""
    x = np.clip(x, grid[0], grid[-1])
    hi = np.clip(np.searchsorted(grid, x, side="left"), 1, grid.size - 1)
    lo = hi - 1
    dlo, dhi = x - grid[lo], grid[hi] - x
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (hi % 2 == 0))
    return np.where(pick_hi, hi, lo)


def to_bf16(w):
    """float32 -> BF16 bits, round to nearest even (finite inputs)."""
    u = np.ascontiguousarray(w, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def encode_nvfp4(w):
    """float [N, K] -> (codes U8 [N, K/2], block scales e4m3 U8 [N, K/16], weight_scale_2 F32,
    the float32 matrix those decode to). modelopt's recipe: weight_scale_2 = amax / (6 * 448) (a
    MULTIPLIER at decode), block scale = e4m3(amax(block) / 6 / weight_scale_2),
    code = e2m1(w / (scale * weight_scale_2))."""
    n, k = w.shape
    assert k % 16 == 0
    amax = float(np.abs(w).max())
    ws2 = np.float32(amax / (6.0 * 448.0)) if amax > 0 else np.float32(1.0)
    blk = np.asarray(w, np.float64).reshape(n, k // 16, 16)
    sc = _nearest(np.abs(blk).max(axis=-1) / 6.0 / float(ws2), _E4M3_MAG).astype(np.uint8)
    scale = (ref._E4M3[sc] * ws2).astype(np.float32)
    inv = np.where(scale > 0, 1.0 / np.where(scale > 0, scale, 1), 0.0)
    q = blk * inv[:, :, None]
    code = (_nearest(np.abs(q), _E2M1_MAG) | ((q < 0) << 3)).astype(np.uint8).reshape(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)                     # low nibble = even column
    deq = (ref._E2M1[code].reshape(n, k // 16, 16) * scale[:, :, None]).reshape(n, k).astype(np.float32)
    return packed, sc, ws2, deq


def write_safetensors(path, tensors):
    """tensors: ordered {name: (dtype string, numpy array of the stored bits)}."""
    header, blobs, off = {}, [], 0
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
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
    return off


# ----------------------------------------------------------------------------------------------
# The synthetic checkpoint
# ----------------------------------------------------------------------------------------------
LAYERS = 3
# The experts whose activation scale the calibration never produced (the release: 6 of 15,872).
NO_INPUT_SCALE = {(0, 3), (2, 5)}


def make_config():
    ignore = ["lm_head"]
    for l in sorted(range(LAYERS), key=str):                       # the release sorts the names as strings
        ignore += [f"model.layers.{l}.block_sparse_moe.gate", f"model.layers.{l}.self_attn*"]
    q = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
    return {
        "vocab_size": 128, "max_position_embeddings": 4096, "hidden_size": 64, "intermediate_size": 32,
        "num_hidden_layers": LAYERS, "num_attention_heads": 6, "sliding_window": None, "num_key_value_heads": 2,
        "hidden_act": "silu", "initializer_range": 0.02, "rms_norm_eps": 1e-06, "use_cache": True,
        "rope_theta": 100, "attention_dropout": 0.0, "head_dim": 16, "num_experts_per_tok": 2,
        "num_local_experts": 8, "output_router_logits": False, "router_aux_loss_coef": 0.001,
        "router_jitter_noise": 0.0, "use_qk_norm": True, "rotary_dim": 8, "partial_rotary_factor": 0.5,
        "transformers_version": "5.5.3", "architectures": ["MiniMaxM2ForCausalLM"], "dtype": "bfloat16",
        "pad_token_id": None, "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": False,
        "attn_type_list": [1] * LAYERS,
        "auto_map": {"AutoConfig": "configuration_minimax_m2.MiniMaxM2Config",
                     "AutoModelForCausalLM": "modeling_minimax_m2.MiniMaxM2ForCausalLM"},
        "model_type": "minimax_m2", "mtp_transformer_layers": 1, "num_mtp_modules": 3,
        "qk_norm_type": "per_layer", "scoring_func": "sigmoid", "shared_intermediate_size": 0, "use_mtp": True,
        "use_routing_bias": True,
        "rope_parameters": {"rope_type": "default", "rope_theta": 100, "partial_rotary_factor": 0.5},
        "quantization_config": {
            "config_groups": {"group_0": {"input_activations": dict(q), "weights": dict(q), "targets": ["Linear"]}},
            "ignore": ignore, "quant_algo": "NVFP4",
            "producer": {"name": "modelopt", "version": "0.39.0.dev290+gf9d9a71de.d20260407"},
            "quant_method": "modelopt"},
    }


def build(seed):
    rng = np.random.default_rng(seed)
    c = make_config()
    H, V, L = c["hidden_size"], c["vocab_size"], c["num_hidden_layers"]
    nh, nk, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    I, E = c["intermediate_size"], c["num_local_experts"]
    weights, scales, expect = {}, {}, {}

    def bf16(name, shape, std=None, gain=False):
        if gain:
            a = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            a = rng.standard_normal(shape) * (std if std is not None else 1.0 / np.sqrt(shape[-1]))
        weights[name] = ("BF16", to_bf16(a.astype(np.float32)))

    weights_tail = {}
    bf16("lm_head.weight", (V, H))
    bf16("model.embed_tokens.weight", (V, H), std=1.0)
    for l in range(L):
        p = f"model.layers.{l}."
        bf16(p + "block_sparse_moe.e_score_correction_bias", (E,), std=0.3)
        for e in range(E):
            for m, (n, k) in (("w1", (I, H)), ("w2", (H, I)), ("w3", (I, H))):
                w = rng.standard_normal((n, k)) / np.sqrt(k)
                w[rng.integers(0, n)] *= 6.0                   # one loud row: block scales far apart
                packed, sc, ws2, deq = encode_nvfp4(w)
                base = p + f"block_sparse_moe.experts.{e}.{m}"
                weights[base + ".weight"] = ("U8", packed)
                weights[base + ".weight_scale"] = ("F8_E4M3", sc)
                weights[base + ".weight_scale_2"] = ("F32", np.array(ws2, np.float32))
                if (l, e) not in NO_INPUT_SCALE:
                    scales[base + ".input_scale"] = ("F32", np.array(rng.uniform(0.01, 0.2), np.float32))
                expect[base] = deq
        bf16(p + "block_sparse_moe.gate.weight", (E, H), std=2.0 / np.sqrt(H))
        bf16(p + "input_layernorm.weight", (H,), gain=True)
        bf16(p + "post_attention_layernorm.weight", (H,), gain=True)
        bf16(p + "self_attn.k_norm.weight", (nk * d,), gain=True)
        bf16(p + "self_attn.k_proj.weight", (nk * d, H))
        bf16(p + "self_attn.o_proj.weight", (H, nh * d))
        bf16(p + "self_attn.q_norm.weight", (nh * d,), gain=True)
        bf16(p + "self_attn.q_proj.weight", (nh * d, H))
        bf16(p + "self_attn.v_proj.weight", (nk * d, H))
    weights_tail["model.norm.weight"] = ("BF16", to_bf16((1.0 + 0.1 * rng.standard_normal(H)).astype(np.float32)))
    weights.update(weights_tail)
    return c, weights, scales, expect


def cmd_ckpt(a):
    os.makedirs(a.out, exist_ok=True)
    c, weights, scales, expect = build(a.seed)
    names = list(weights)
    half = len(names) // 2
    shards = [("model-00001-of-00002.safetensors", {n: weights[n] for n in names[:half]}),
              ("model-00002-of-00002.safetensors", {n: weights[n] for n in names[half:]}),
              ("model-inputscales.safetensors", scales)]
    total, wmap = 0, {}
    for fname, tensors in shards:
        total += write_safetensors(os.path.join(a.out, fname), tensors)
        for n in tensors:
            wmap[n] = fname
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wmap.items()))}, f, indent=2)
    # A file that is NOT the model: it repeats tensor names (here with other values). The release has
    # eighteen of these; a reader that globs the directory binds them.
    first_scale = next(iter(scales))
    write_safetensors(os.path.join(a.out, "amax_checkpoint.safetensors"),
                      {first_scale: ("F32", np.array(123.0, np.float32)),
                       "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_2": ("F32", np.array(9.0, np.float32))})
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(c, f, indent=4)
    with open(os.path.join(a.out, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 100, "do_sample": True, "eos_token_id": 101, "temperature": 1.0, "top_p": 0.95,
                   "top_k": 40}, f, indent=2)
    rng = np.random.default_rng(a.seed + 1)
    ids = [100] + [int(t) for t in rng.integers(3, 100, 39)]
    with open(os.path.join(a.out, "ids.json"), "w") as f:
        json.dump({"ids": ids}, f)
    np.savez(os.path.join(a.out, "expected_dequant.npz"), **expect)
    print(f"wrote {len(wmap)} tensors ({total} bytes) in {len(shards)} shards + 1 stray file, "
          f"{len(expect)} NVFP4 matrices ({len(scales)} activation scales), {len(ids)} ids to {a.out}")


# ----------------------------------------------------------------------------------------------
# selftest
# ----------------------------------------------------------------------------------------------
def _logits(model, ids, pos0=0, chunk=None):
    st = model.new_state(pos0)
    if chunk is None:
        return model.logits(model.forward(ids, st)).astype(np.float64)
    out = [model.logits(model.forward(ids[b:b + chunk], st)) for b in range(0, len(ids), chunk)]
    return np.concatenate(out).astype(np.float64)


def cmd_selftest(a):
    ids = ref.load_ids(os.path.join(a.ckpt, "ids.json"))
    m = ref.Model(a.ckpt, dtype="float64")
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + name + (" — " + detail if detail else ""))

    exp = np.load(os.path.join(a.ckpt, "expected_dequant.npz"))
    worst = 0.0
    for prefix in exp.files:
        worst = max(worst, float(np.abs(m._fp4(prefix) - exp[prefix].astype(np.float64)).max()))
    check(f"dequant: {len(exp.files)} NVFP4 matrices equal expected_dequant.npz", worst == 0.0, f"max |diff| {worst:g}")
    check("shards: only the index's files are read", "amax_checkpoint.safetensors" not in m.ck.shards,
          f"{m.ck.shards}")
    base = _logits(m, ids)
    off = _logits(m, ids, pos0=1000)
    # Not exactly nothing: the angles are fp32 products and the cos / sin fp32 values, so a shifted
    # sequence rounds differently (about 1e-5 on these logits); a position bug moves them by O(1).
    check("positions: rope is relative (an offset start changes the logits by fp32 rounding only)",
          np.abs(base - off).max() < 1e-3, f"max |diff| {np.abs(base - off).max():.3g}")
    for chunk in (7, 1):
        d = np.abs(_logits(m, ids, chunk=chunk) - base).max()
        check(f"chunking: chunk {chunk} equals the whole sequence", d < 1e-9, f"max |diff| {d:.3g}")
    new, _ = ref.greedy(m, ids[:8], 12, stop_eos=False)
    res = ref.score_ids(m, ids[:8] + new, chunk=5)
    tops = [res["top"][i][0][0] for i in range(8, 8 + len(new))]
    check("generate: greedy ids are the scorer's top-1", tops == new, f"{new}")
    for k, v in (("nibble", "hi"), ("ws2", "div"), ("rope", "interleave"), ("qknorm", "head"), ("bias", "off"),
                 ("w13", "up_gate")):
        mv = ref.Model(a.ckpt, dtype="float64", variants={k: v})
        d = np.abs(_logits(mv, ids) - base).max()
        check(f"variant {k}={v} changes the logits", np.isfinite(d) and d > 1e-3, f"max |diff| {d:.4f}")
    f32 = ref.Model(a.ckpt, dtype="float32")
    d = np.abs(_logits(f32, ids) - base).max()
    check("float32 agrees with float64", d < 5e-3, f"max |diff| {d:.3g}")
    print("SELFTEST " + ("OK" if fails == 0 else f"FAILED ({fails})"))
    sys.exit(1 if fails else 0)


# ----------------------------------------------------------------------------------------------
# hf-check: transformers' MiniMaxM2ForCausalLM on the same weights
# ----------------------------------------------------------------------------------------------
def cmd_hf_check(a):
    import torch
    import transformers
    from transformers import MiniMaxM2Config, MiniMaxM2ForCausalLM

    m = ref.Model(a.ckpt, dtype="float64")
    c = m.ck.cfg
    cfg = MiniMaxM2Config(
        vocab_size=c["vocab_size"], hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
        num_hidden_layers=c["num_hidden_layers"], num_attention_heads=c["num_attention_heads"],
        num_key_value_heads=c["num_key_value_heads"], head_dim=c["head_dim"],
        max_position_embeddings=c["max_position_embeddings"], rms_norm_eps=c["rms_norm_eps"],
        num_experts_per_tok=c["num_experts_per_tok"], num_local_experts=c["num_local_experts"],
        rope_parameters=dict(c["rope_parameters"]), bos_token_id=None, eos_token_id=None)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    model = MiniMaxM2ForCausalLM(cfg).double().eval()
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))  # noqa: E731
    sd = {"model.embed_tokens.weight": t(ref.bf16_to_f32(m.embed_bits)), "model.norm.weight": t(m.final_norm),
          "lm_head.weight": t(m._bf16("lm_head.weight"))}
    for l in range(c["num_hidden_layers"]):
        w = m.layer_weights(l)
        b = f"model.layers.{l}."
        sd.update({
            b + "input_layernorm.weight": t(w["in_norm"]), b + "post_attention_layernorm.weight": t(w["post_norm"]),
            b + "self_attn.q_proj.weight": t(w["q"]), b + "self_attn.k_proj.weight": t(w["k"]),
            b + "self_attn.v_proj.weight": t(w["v"]), b + "self_attn.o_proj.weight": t(w["o"]),
            b + "self_attn.q_norm.weight": t(w["q_norm"]), b + "self_attn.k_norm.weight": t(w["k_norm"]),
            b + "mlp.gate.weight": t(w["gate"]), b + "mlp.e_score_correction_bias": t(w["bias"])})
        ex = [m.expert(l, e) for e in range(m.n_experts)]
        sd[b + "mlp.experts.gate_up_proj"] = t(np.stack([np.concatenate([w1, w3], axis=0) for w1, _, w3 in ex]))
        sd[b + "mlp.experts.down_proj"] = t(np.stack([w2 for _, w2, _ in ex]))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "inv_freq" not in k]
    if missing or unexpected:
        print(f"FAIL state dict: missing {missing[:6]} unexpected {list(unexpected)[:6]}")
        sys.exit(1)
    print(f"transformers {transformers.__version__}: MiniMaxM2ForCausalLM, {len(sd)} tensors loaded")
    ids = ref.load_ids(os.path.join(a.ckpt, "ids.json"))
    fails = 0
    for pos0 in (0, 777):
        with torch.no_grad():
            pos = torch.arange(pos0, pos0 + len(ids))[None]
            hf = model(input_ids=torch.tensor([ids]), position_ids=pos).logits[0].numpy()
        d = float(np.abs(_logits(m, ids, pos0=pos0) - hf).max())
        worst_variant = min(float(np.abs(_logits(ref.Model(a.ckpt, dtype="float64", variants={k: v}), ids, pos0=pos0) - hf).max())
                            for k, v in (("qknorm", "head"), ("rope", "interleave"), ("bias", "off"), ("w13", "up_gate")))
        # Not 1e-12: transformers keeps fp32 islands in a float64 model (the RMSNorms cast to
        # float32, the eager softmax is dtype=float32, the rope's cos / sin are fp32).
        ok = d < 1e-5
        fails += not ok
        print(("PASS " if ok else "FAIL ") + f"pos0 {pos0}: max |reference - transformers| = {d:.3e} "
              f"(logits span {np.abs(hf).max():.2f}; the nearest wrong variant is off by {worst_variant:.3f})")
    print("HF-CHECK " + ("OK" if fails == 0 else f"FAILED ({fails})"))
    sys.exit(1 if fails else 0)


# ----------------------------------------------------------------------------------------------
# vectors: the C++ host reference's test data
# ----------------------------------------------------------------------------------------------
def _c_array(name, ctype, arr, fmt, per_line=12):
    flat = np.asarray(arr).reshape(-1)
    out = [f"inline constexpr {ctype} {name}[{flat.size}] = {{"]
    for i in range(0, flat.size, per_line):
        out.append("    " + ", ".join(fmt(v) for v in flat[i:i + per_line]) + ",")
    out.append("};")
    return "\n".join(out)


def cmd_vectors(a):
    m = ref.Model(a.ckpt, dtype="float64")
    layer, T = a.layer, a.tokens
    rng = np.random.default_rng(a.seed)
    h_bits = to_bf16(rng.standard_normal((T, m.H)).astype(np.float32))
    h = ref.bf16_to_f32(h_bits).astype(np.float64)
    pos0 = a.pos0
    p = f"model.layers.{layer}."
    w = {n: m.ck.raw(p + f"self_attn.{n}.weight") for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")}
    u16 = lambda v: f"0x{int(v):04x}"                                    # noqa: E731
    f32 = lambda v: f"{float(np.float32(v))!r}f"                          # noqa: E731
    f64 = lambda v: f"{float(v)!r}"                                       # noqa: E731
    u8 = lambda v: str(int(v))                                           # noqa: E731
    lines = [
        "// GENERATED by tools/minimax_m2_synth.py vectors — do not edit.",
        f"// One layer (layer {layer}) of the synthetic MiniMax-M2 checkpoint, a BF16 input of T tokens at",
        "// positions kPos0.., and tools/minimax_m2_reference.py's float64 results: the normed and",
        "// rotated q / k heads, v, the attention output, the router's picks and weights, and the",
        "// MoE block's output with the layer's NVFP4 expert containers.",
        "#pragma once",
        "#include <cstdint>",
        "",
        "namespace minimax_vectors {",
        f"inline constexpr int kHidden = {m.H}, kHeads = {m.heads}, kKvHeads = {m.kv_heads}, kHeadDim = {m.d};",
        f"inline constexpr int kRotaryDim = {m.rot}, kTokens = {T}, kExperts = {m.n_experts}, kTopK = {m.top_k};",
        f"inline constexpr int kInter = {m.inter};",
        f"inline constexpr double kTheta = {m.theta!r};",
        f"inline constexpr float kEps = {m.eps!r}f;",
        f"inline constexpr long long kPos0 = {pos0};",
    ]
    for n, arr in w.items():
        lines.append(_c_array("k_" + n, "uint16_t", arr, u16))
    lines.append(_c_array("k_gate", "uint16_t", m.ck.raw(p + "block_sparse_moe.gate.weight"), u16))
    lines.append(_c_array("k_router_bias", "uint16_t", m.ck.raw(p + "block_sparse_moe.e_score_correction_bias"), u16))
    lines.append(_c_array("k_hidden", "uint16_t", h_bits, u16))
    lines.append(_c_array("k_inv_freq", "float", m.inv_freq, f32, 6))
    taps = {}
    st = m.new_state(pos0)
    pos = pos0 + np.arange(T, dtype=np.int64)
    out = m.attention(layer, h, pos, st, tap=lambda k, v: taps.__setitem__(k.split(".", 1)[1], np.array(v)))
    for n in ("q", "k", "v", "attn_heads"):
        lines.append(_c_array("k_ref_" + n, "double", taps[n], f64, 6))
    lines.append(_c_array("k_ref_out", "double", out, f64, 6))
    # The per-head reading of the same weights (GLM-4.7's norm): what the engine's existing
    # qkv finish would compute. The test shows it is a different function.
    mh = ref.Model(a.ckpt, dtype="float64", variants={"qknorm": "head"})
    taps_h = {}
    mh.attention(layer, h, pos, mh.new_state(pos0), tap=lambda k, v: taps_h.__setitem__(k.split(".", 1)[1], np.array(v)))
    lines.append(_c_array("k_ref_q_per_head", "double", taps_h["q"], f64, 6))
    rid, rw = m.route(layer, h)
    order = np.argsort(rid, axis=-1, kind="stable")
    lines.append(_c_array("k_route_ids", "int", np.take_along_axis(rid, order, axis=-1), lambda v: str(int(v))))
    lines.append(_c_array("k_route_w", "double", np.take_along_axis(rw, order, axis=-1), f64, 6))
    lines.append(_c_array("k_moe_out", "double", m.moe(layer, h), f64, 6))
    # The NVFP4 containers in the engine's matrix order — expert e's gate (w1), up (w3), down (w2)
    # at index e * 3 + m — with the engine's global: a DIVISOR, 1 / weight_scale_2 in fp32.
    prefixes = [p + f"block_sparse_moe.experts.{e}.{mat}" for e in range(m.n_experts) for mat in ("w1", "w3", "w2")]
    lines.append(_c_array("k_fp4_payloads", "uint8_t", np.concatenate(
        [m.ck.raw(q + ".weight").reshape(-1) for q in prefixes]), u8, 24))
    lines.append(_c_array("k_fp4_scales", "uint8_t", np.concatenate(
        [m.ck.raw(q + ".weight_scale").reshape(-1) for q in prefixes]), u8, 24))
    lines.append(_c_array("k_fp4_divisors", "float", np.array(
        [np.float32(1.0) / np.float32(m.ck.raw(q + ".weight_scale_2").reshape(())) for q in prefixes], np.float32), f32, 6))
    lines += ["}  // namespace minimax_vectors", ""]
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {a.out}: layer {layer}, {T} tokens from position {pos0}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ckpt"); p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=20261004)
    p.set_defaults(fn=cmd_ckpt)
    p = sub.add_parser("selftest"); p.add_argument("--ckpt", required=True); p.set_defaults(fn=cmd_selftest)
    p = sub.add_parser("hf-check"); p.add_argument("--ckpt", required=True); p.set_defaults(fn=cmd_hf_check)
    p = sub.add_parser("vectors"); p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True)
    p.add_argument("--layer", type=int, default=1); p.add_argument("--tokens", type=int, default=12)
    p.add_argument("--pos0", type=int, default=37); p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_vectors)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
