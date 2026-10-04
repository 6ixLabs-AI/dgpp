#!/usr/bin/env python3
"""mistral4_synth.py — tiny synthetic Mistral-Small-4 checkpoints and the checks that run on them
(2026-10-04). No GPU. `ckpt`, `selftest` and `vectors` need numpy only; `hf-check` additionally
needs torch and a transformers that ships Mistral4ForCausalLM.

The synthetic checkpoint carries the real release's tensor names, containers and recipe
(mistralai/Mistral-Small-4-119B-2603-NVFP4: Mistral-native names, compressed-tensors NVFP4 routed
and shared experts, BF16 everything else, a Pixtral tower that is never read, every
`input_global_scale` in the last shard) at toy sizes, so tools/mistral4_reference.py and the
engine's config parser / binding table can be exercised end to end on a laptop.

  ckpt     --out DIR [--seed N]
        writes params.json, the consolidated-*.safetensors shards with their index, ids.json and
        expected_dequant.npz (the float value every quantized matrix must dequantize to).
  selftest --ckpt DIR
        through mistral4_reference.Model: (1) every quantized matrix dequantizes to
        expected_dequant.npz exactly; (2) the absorbed MLA equals the expanded per-head K/V form;
        (3) feeding a sequence whole, in chunks and token by token gives the same logits, from
        position 0 and from a position past the Llama-4 / YaRN boundary; (4) greedy generation
        reproduces teacher-forced scoring of its own output; (5) each debug variant changes the
        logits (a variant that changes nothing is a convention the test cannot see).
  hf-check --ckpt DIR
        the same weights loaded into transformers' Mistral4ForCausalLM (float64, eager attention)
        through the inverse of convert_mistral4_weight_to_hf.py's renaming, logits compared with
        the reference's at positions 0.. and at an offset. This is the check against the model's
        own published implementation; it also reports what the installed transformers does about
        the YaRN scale (see mistral4_reference.py, "THE YARN SCALE").
  vectors  --ckpt DIR --out tests/unit/mistral4_mla_vectors.hpp
        the test vectors of tests/unit/mistral4_mla_test.cpp: one layer's attention weights as BF16
        bits, a deterministic BF16 input, and what the reference makes of them in float64.

What a pass here does and does not show: the reference is self-consistent and agrees with
transformers on a random model of the same class. The writers follow the same reading of the
compressed-tensors container as the readers (nibble order, the divisor), so those conventions are
only proven on the real checkpoint.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mistral4_reference as ref  # noqa: E402

# ----------------------------------------------------------------------------------------------
# Encoders (the inverse of mistral4_reference's decoders)
# ----------------------------------------------------------------------------------------------
_E2M1_MAG = ref._E2M1[:8].astype(np.float64)           # 0 .. 6, ascending, index = code
_E4M3_MAG = ref._E4M3[:127].astype(np.float64)         # 0 .. 448, ascending, index = code


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


def bf16_round(w):
    return ref.bf16_to_f32(to_bf16(w))


def encode_nvfp4_ct(w):
    """float [N, K] -> (codes U8 [N, K/2], block scales e4m3 U8 [N, K/16], weight_global_scale F32,
    the float32 matrix those decode to). compressed-tensors' recipe: weight_global_scale =
    448 * 6 / amax (a DIVISOR at decode), block scale = e4m3(amax(block) / 6 * weight_global_scale),
    code = e2m1(w / (scale / weight_global_scale))."""
    n, k = w.shape
    assert k % 16 == 0
    amax = float(np.abs(w).max())
    g = np.float32(448.0 * 6.0 / amax) if amax > 0 else np.float32(1.0)
    blk = np.asarray(w, np.float64).reshape(n, k // 16, 16)
    sc = _nearest(np.abs(blk).max(axis=-1) / 6.0 * float(g), _E4M3_MAG).astype(np.uint8)        # [N, K/16]
    scale = (ref._E4M3[sc] / g).astype(np.float32)
    inv = np.where(scale > 0, 1.0 / np.where(scale > 0, scale, 1), 0.0)
    q = blk * inv[:, :, None]
    code = (_nearest(np.abs(q), _E2M1_MAG) | ((q < 0) << 3)).astype(np.uint8).reshape(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)                            # low nibble = even column
    deq = (ref._E2M1[code].reshape(n, k // 16, 16) * scale[:, :, None]).reshape(n, k).astype(np.float32)
    return packed, sc, g, deq


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
QUANT_CONFIG = {
    "config_groups": {"NVFP4A16": {
        "format": "nvfp4-pack-quantized",
        "input_activations": {"actorder": None, "block_structure": None, "dynamic": "local", "group_size": 16,
                              "num_bits": 4, "observer": "static_minmax", "observer_kwargs": {},
                              "strategy": "tensor_group", "symmetric": True, "type": "float"},
        "output_activations": None,
        "targets": ["Linear"],
        "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": 16, "num_bits": 4,
                    "observer": "static_minmax", "observer_kwargs": {}, "scale_dtype": "torch.float8_e4m3fn",
                    "strategy": "tensor_group", "symmetric": True, "type": "float", "zp_dtype": None}}},
    "format": "nvfp4-pack-quantized",
    "global_compression_ratio": None,
    "ignore": ["model.embed_tokens", "re:patch_merger.*", "re:vision_encoder.*", "re:vision_language_adapter.*",
               "re:.*kv_a_proj_with_mqa$", "re:.*q_a_proj$", "re:.*gate$", "re:.*self_attn.*", "re:.*attention.*",
               "lm_head"],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "sparsity_config": {},
    "transform_config": {},
    "version": "0.13.0",
}

# The toy shape. The YaRN band (theta 100, rope 16, original 64, beta 4 / alpha 1) spans lanes 1..5
# of 8, so kept, blended and fully interpolated frequencies all occur; a 40-token sequence from
# position 40 crosses the Llama-4 step at 64 and several of its floors with --pos0.
PARAMS = {
    "dim": 64, "head_dim": 24, "hidden_dim": 96, "kv_lora_rank": 16,
    "llama_4_scaling": {"beta": 0.1, "original_max_position_embeddings": 64},
    "max_position_embeddings": 8192,
    "moe": {"expert_hidden_dim": 32, "expert_model_parallel": 1, "expert_parallel": 1, "first_k_dense_replace": 0,
            "num_expert_groups": 1, "num_expert_groups_per_tok": 1, "num_experts": 8, "num_experts_per_tok": 2,
            "num_shared_experts": 1, "route_every_n": 1, "routed_scale": 1.0},
    "n_heads": 4, "n_kv_heads": 4, "n_layers": 3, "norm_eps": 1e-06, "q_lora_rank": 32,
    "qk_nope_head_dim": 8, "qk_rope_head_dim": 16,
    "quantization_config": QUANT_CONFIG,
    "rope_theta": 100.0, "tied_embeddings": False, "v_head_dim": 16,
    "vision_encoder": {"adapter_bias": False, "add_pre_mm_projector_layer_norm": True, "hidden_size": 8,
                       "image_break_token_id": 12, "image_end_token_id": 13, "image_size": 8, "image_token_id": 10,
                       "intermediate_size": 16, "max_image_size": 8, "mm_projector_id": "patch_merge",
                       "num_attention_heads": 2, "num_channels": 3, "num_hidden_layers": 1, "patch_size": 2,
                       "rope_theta": 10000.0, "spatial_merge_size": 2},
    "vocab_size": 128,
    "yarn": {"alpha": 1, "apply_scale": False, "beta": 4, "factor": 8, "original_max_position_embeddings": 64},
}


def build(seed):
    """-> (weights {name: (dtype, bits)} in the release's shard order, scales {name: (dtype, bits)},
    expected dequantized matrices {prefix: float32})."""
    rng = np.random.default_rng(seed)
    p = PARAMS
    H, V, L = p["dim"], p["vocab_size"], p["n_layers"]
    nh, ql, kl = p["n_heads"], p["q_lora_rank"], p["kv_lora_rank"]
    nope, rope, vd = p["qk_nope_head_dim"], p["qk_rope_head_dim"], p["v_head_dim"]
    I, E = p["moe"]["expert_hidden_dim"], p["moe"]["num_experts"]
    weights, scales, expect = {}, {}, {}

    def bf16(name, shape, std=None, gain=False):
        if gain:
            a = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            a = rng.standard_normal(shape) * (std if std is not None else 1.0 / np.sqrt(shape[-1]))
        weights[name] = ("BF16", to_bf16(a.astype(np.float32)))

    def fp4(prefix, n, k, shared):
        w = rng.standard_normal((n, k)) / np.sqrt(k)
        w[rng.integers(0, n)] *= 6.0                       # one loud row: block scales far apart
        packed, sc, g, deq = encode_nvfp4_ct(w)
        weights[prefix + ".weight_packed"] = ("U8", packed)
        weights[prefix + ".weight_scale"] = ("F8_E4M3", sc)
        weights[prefix + ".weight_global_scale"] = ("F32", np.array([g], np.float32))
        act = np.array([rng.uniform(40.0, 400.0)], np.float32)
        # Checkpoint truth: the routed experts' activation scale is F32 [1], the shared expert's BF16 [1].
        scales[prefix + ".input_global_scale"] = ("BF16", to_bf16(act)) if shared else ("F32", act)
        expect[prefix] = deq

    for l in range(L):
        a = f"layers.{l}."
        bf16(a + "attention.kv_a_norm.weight", (kl,), gain=True)
        bf16(a + "attention.q_a_norm.weight", (ql,), gain=True)
        bf16(a + "attention.wkv_a_with_mqa.weight", (kl + rope, H))
        bf16(a + "attention.wkv_b.weight", (nh * (nope + vd), kl))
        bf16(a + "attention.wo.weight", (H, nh * vd))
        bf16(a + "attention.wq_a.weight", (ql, H))
        bf16(a + "attention.wq_b.weight", (nh * (nope + rope), ql), std=2.0 / np.sqrt(ql))
        bf16(a + "attention_norm.weight", (H,), gain=True)
        for e in range(E):
            for m, (n, k) in (("w1", (I, H)), ("w2", (H, I)), ("w3", (I, H))):
                fp4(a + f"experts.{e}.{m}", n, k, shared=False)
        bf16(a + "ffn_norm.weight", (H,), gain=True)
        bf16(a + "gate.weight", (E, H), std=2.0 / np.sqrt(H))
        for m, (n, k) in (("w1", (I, H)), ("w2", (H, I)), ("w3", (I, H))):
            fp4(a + f"shared_experts.{m}", n, k, shared=True)
    bf16("norm.weight", (H,), gain=True)
    bf16("output.weight", (V, H))
    v = p["vision_encoder"]
    vh, vi, ps = v["hidden_size"], v["intermediate_size"], v["patch_size"]
    bf16("patch_merger.merging_layer.weight", (vh, vh * v["spatial_merge_size"] ** 2))
    bf16("pre_mm_projector_norm.weight", (vh,), gain=True)
    bf16("tok_embeddings.weight", (V, H), std=1.0)
    bf16("vision_encoder.ln_pre.weight", (vh,), gain=True)
    bf16("vision_encoder.patch_conv.weight", (vh, v["num_channels"], ps, ps), std=0.1)
    for l in range(v["num_hidden_layers"]):
        t = f"vision_encoder.transformer.layers.{l}."
        for m in ("wk", "wo", "wq", "wv"):
            bf16(t + f"attention.{m}.weight", (vh, vh))
        bf16(t + "attention_norm.weight", (vh,), gain=True)
        bf16(t + "feed_forward.w1.weight", (vi, vh))
        bf16(t + "feed_forward.w2.weight", (vh, vi))
        bf16(t + "feed_forward.w3.weight", (vi, vh))
        bf16(t + "ffn_norm.weight", (vh,), gain=True)
    bf16("vision_language_adapter.w_in.weight", (H, vh))
    bf16("vision_language_adapter.w_out.weight", (H, H))
    return weights, scales, expect


def cmd_ckpt(a):
    os.makedirs(a.out, exist_ok=True)
    weights, scales, expect = build(a.seed)
    names = list(weights)
    half = len(names) // 2
    shards = [("consolidated-00001-of-00003.safetensors", {n: weights[n] for n in names[:half]}),
              ("consolidated-00002-of-00003.safetensors", {n: weights[n] for n in names[half:]}),
              ("consolidated-00003-of-00003.safetensors", scales)]      # the release keeps every input scale last
    total, wmap = 0, {}
    for fname, tensors in shards:
        total += write_safetensors(os.path.join(a.out, fname), tensors)
        for n in tensors:
            wmap[n] = fname
    with open(os.path.join(a.out, "consolidated.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wmap.items()))}, f, indent=2)
    with open(os.path.join(a.out, "params.json"), "w") as f:
        json.dump(PARAMS, f, indent=2, sort_keys=True)
    rng = np.random.default_rng(a.seed + 1)
    ids = [1] + [int(t) for t in rng.integers(3, PARAMS["vocab_size"], 39)]
    with open(os.path.join(a.out, "ids.json"), "w") as f:
        json.dump({"ids": ids}, f)
    np.savez(os.path.join(a.out, "expected_dequant.npz"), **expect)
    print(f"wrote {len(wmap)} tensors ({total} bytes) in {len(shards)} shards, {len(expect)} NVFP4 matrices, "
          f"{len(ids)} ids to {a.out}")


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

    # (1) the containers
    exp = np.load(os.path.join(a.ckpt, "expected_dequant.npz"))
    worst = 0.0
    for prefix in exp.files:
        got = m._fp4(prefix)
        worst = max(worst, float(np.abs(got - exp[prefix].astype(np.float64)).max()))
    check(f"dequant: {len(exp.files)} NVFP4 matrices equal expected_dequant.npz", worst == 0.0, f"max |diff| {worst:g}")
    band = m.yarn_band
    kept = int((m.inv_freq == ref.yarn_inv_freq(m.rope, m.theta, 1.0, m.l4_orig, 4, 1)[0]).sum())
    check("yarn: the band has kept, blended and interpolated lanes",
          0 < band[0] < band[1] < m.rope // 2 - 1, f"band {band}, lanes {m.rope // 2}, unchanged {kept}")

    base0 = _logits(m, ids)
    off = m.l4_orig - 8                               # crosses the Llama-4 step inside the sequence
    base1 = _logits(m, ids, pos0=off)
    check("positions: an offset start changes the logits", np.abs(base0 - base1).max() > 1e-3,
          f"max |diff| {np.abs(base0 - base1).max():.4f}")

    # (2) absorbed vs expanded
    me = ref.Model(a.ckpt, dtype="float64", variants={"mla": "expand"})
    d = max(np.abs(_logits(me, ids) - base0).max(), np.abs(_logits(me, ids, pos0=off) - base1).max())
    check("mla: absorbed scores equal the expanded per-head K/V form", d < 1e-9, f"max |diff| {d:.3g}")

    # (3) chunking
    for pos0, base in ((0, base0), (off, base1)):
        for chunk in (7, 1):
            d = np.abs(_logits(m, ids, pos0=pos0, chunk=chunk) - base).max()
            check(f"chunking: pos0 {pos0} chunk {chunk} equals the whole sequence", d < 1e-9, f"max |diff| {d:.3g}")

    # (4) generation vs scoring
    new, _ = ref.greedy(m, ids[:8], 12, stop_eos=False)
    res = ref.score_ids(m, ids[:8] + new, chunk=5)
    tops = [res["top"][i][0][0] for i in range(8, 8 + len(new))]
    check("generate: greedy ids are the scorer's top-1", tops == new, f"{new}")

    # (5) every variant is visible
    for k, v in (("nibble", "hi"), ("gscale", "mul"), ("rope", "half"), ("yarn_mscale", "on"), ("llama4", "off"),
                 ("split", "rope_nope")):
        mv = ref.Model(a.ckpt, dtype="float64", variants={k: v})
        d = np.abs(_logits(mv, ids, pos0=off) - base1).max()
        check(f"variant {k}={v} changes the logits", np.isfinite(d) and d > 1e-3, f"max |diff| {d:.4f}")
    f32 = ref.Model(a.ckpt, dtype="float32")
    d = np.abs(_logits(f32, ids) - base0).max()
    check("float32 agrees with float64", d < 5e-3, f"max |diff| {d:.3g}")
    print("SELFTEST " + ("OK" if fails == 0 else f"FAILED ({fails})"))
    sys.exit(1 if fails else 0)


# ----------------------------------------------------------------------------------------------
# hf-check: transformers' Mistral4ForCausalLM on the same weights
# ----------------------------------------------------------------------------------------------
def cmd_hf_check(a):
    import torch
    import transformers
    from transformers import Mistral4Config, Mistral4ForCausalLM
    from transformers.models.mistral4 import modeling_mistral4 as mm

    m = ref.Model(a.ckpt, dtype="float64")
    p = m.ck.params
    moe, y, l4 = p["moe"], p["yarn"], p["llama_4_scaling"]
    # convert_mistral4_weight_to_hf.py's convert_config, field for field.
    cfg = Mistral4Config(
        hidden_size=p["dim"], num_hidden_layers=p["n_layers"], intermediate_size=p["hidden_dim"],
        num_attention_heads=p["n_heads"], num_key_value_heads=p["n_kv_heads"], rms_norm_eps=p["norm_eps"],
        vocab_size=p["vocab_size"], tie_word_embeddings=p["tied_embeddings"],
        max_position_embeddings=p["max_position_embeddings"], q_lora_rank=p["q_lora_rank"],
        qk_rope_head_dim=p["qk_rope_head_dim"], qk_nope_head_dim=p["qk_nope_head_dim"],
        kv_lora_rank=p["kv_lora_rank"], v_head_dim=p["v_head_dim"], n_routed_experts=moe["num_experts"],
        num_experts_per_tok=moe["num_experts_per_tok"], first_k_dense_replace=moe["first_k_dense_replace"],
        n_shared_experts=moe["num_shared_experts"], moe_intermediate_size=moe["expert_hidden_dim"],
        routed_scaling_factor=moe["routed_scale"], n_group=moe["num_expert_groups"],
        topk_group=moe["num_expert_groups_per_tok"], norm_topk_prob=True,
        rope_parameters={"rope_type": "yarn", "rope_theta": p["rope_theta"], "factor": float(y["factor"]),
                         "original_max_position_embeddings": y["original_max_position_embeddings"],
                         "beta_fast": float(y["beta"]), "beta_slow": float(y["alpha"]), "mscale_all_dim": 1.0,
                         "mscale": 1.0, "llama_4_scaling_beta": l4["beta"],
                         "partial_rotary_factor": p["qk_rope_head_dim"] / (p["qk_nope_head_dim"] + p["qk_rope_head_dim"])})
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"       # the class's own loop (grouped_mm has no float64 path)
    model = Mistral4ForCausalLM(cfg).double().eval()
    L, E = p["n_layers"], moe["num_experts"]
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))  # noqa: E731
    sd = {"model.embed_tokens.weight": t(ref.bf16_to_f32(m.embed_bits)), "model.norm.weight": t(m.final_norm),
          "lm_head.weight": t(m._bf16("output.weight"))}
    for l in range(L):
        w = m.layer_weights(l)
        b = f"model.layers.{l}."
        sd.update({
            b + "input_layernorm.weight": t(w["attn_norm"]), b + "post_attention_layernorm.weight": t(w["ffn_norm"]),
            b + "self_attn.q_a_proj.weight": t(w["wq_a"]), b + "self_attn.q_a_layernorm.weight": t(w["q_a_norm"]),
            b + "self_attn.q_b_proj.weight": t(w["wq_b"]), b + "self_attn.kv_a_proj_with_mqa.weight": t(w["wkv_a"]),
            b + "self_attn.kv_a_layernorm.weight": t(w["kv_a_norm"]), b + "self_attn.kv_b_proj.weight": t(w["wkv_b"]),
            b + "self_attn.o_proj.weight": t(w["wo"]), b + "mlp.gate.weight": t(w["gate"]),
            b + "mlp.shared_experts.gate_proj.weight": t(w["shared"][0]),
            b + "mlp.shared_experts.down_proj.weight": t(w["shared"][1]),
            b + "mlp.shared_experts.up_proj.weight": t(w["shared"][2])})
        ex = [m.expert(l, e) for e in range(E)]
        sd[b + "mlp.experts.gate_up_proj"] = t(np.stack([np.concatenate([w1, w3], axis=0) for w1, _, w3 in ex]))
        sd[b + "mlp.experts.down_proj"] = t(np.stack([w2 for _, w2, _ in ex]))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "inv_freq" not in k]
    if missing or unexpected:
        print(f"FAIL state dict: missing {missing[:6]} unexpected {list(unexpected)[:6]}")
        sys.exit(1)
    hf_scales = hasattr(mm, "yarn_apply_mscale")
    attn = model.model.layers[0].self_attn
    plain = (p["qk_nope_head_dim"] + p["qk_rope_head_dim"]) ** -0.5
    print(f"transformers {transformers.__version__}: Mistral4Attention.scaling = {attn.scaling:.6f} "
          f"(1/sqrt(qk_head_dim) = {plain:.6f}); yarn_apply_mscale {'present' if hf_scales else 'absent'}; "
          f"rope_interleave {cfg.rope_interleave}")
    ids = ref.load_ids(os.path.join(a.ckpt, "ids.json"))
    fails = 0
    for pos0 in (0, m.l4_orig - 8, 5 * m.l4_orig + 3):
        with torch.no_grad():
            pos = torch.arange(pos0, pos0 + len(ids))[None]
            hf = model(input_ids=torch.tensor([ids]), position_ids=pos).logits[0].numpy()
        best = None
        for variant in ("off", "on"):
            mv = m if variant == "off" else ref.Model(a.ckpt, dtype="float64", variants={"yarn_mscale": "on"})
            d = float(np.abs(_logits(mv, ids, pos0=pos0) - hf).max())
            print(f"  pos0 {pos0:4d}: max |reference(yarn_mscale={variant}) - transformers| = {d:.3e}")
            best = d if best is None else min(best, d)
        want = "on" if abs(attn.scaling - plain) > 1e-9 else "off"
        mv = m if want == "off" else ref.Model(a.ckpt, dtype="float64", variants={"yarn_mscale": "on"})
        d = float(np.abs(_logits(mv, ids, pos0=pos0) - hf).max())
        # Not 1e-12: transformers keeps three fp32 islands in a float64 model — Mistral4RMSNorm
        # casts to float32, eager_attention_forward's softmax is dtype=float32, the rope's cos / sin
        # are fp32 — so the two agree to fp32 rounding (about 2e-6 on these logits). A wrong
        # convention moves the logits by O(1): see the yarn_mscale=on line above.
        ok = d < 1e-5
        fails += not ok
        print(("PASS " if ok else "FAIL ") + f"pos0 {pos0}: reference(yarn_mscale={want}) equals transformers "
              f"({d:.3e}); logits span {np.abs(hf).max():.2f}")
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
    p = m.ck.params
    layer, T = a.layer, a.tokens
    rng = np.random.default_rng(a.seed)
    h_bits = to_bf16(rng.standard_normal((T, m.H)).astype(np.float32))
    h = ref.bf16_to_f32(h_bits).astype(np.float64)
    # Two position runs: from 0 (the Llama-4 scale is exactly 1), and across the fifth Llama-4
    # step (the scale goes from 1 + 0.1 ln 5 to 1 + 0.1 ln 6), deep in the YaRN-interpolated range.
    runs = (0, 5 * m.l4_orig - T // 2)
    w = {n: m.ck.raw(f"layers.{layer}.attention.{n}.weight") for n in
         ("wq_a", "q_a_norm", "wq_b", "wkv_a_with_mqa", "kv_a_norm", "wkv_b", "wo")}
    gate_bits = m.ck.raw(f"layers.{layer}.gate.weight")
    lines = [
        "// GENERATED by tools/mistral4_synth.py vectors — do not edit.",
        f"// One attention layer (layer {layer}) of the synthetic Mistral-Small-4 checkpoint (seed in the",
        "// generator), a BF16 input of T tokens, and tools/mistral4_reference.py's float64 results:",
        "// the YaRN inverse frequencies, the Llama-4 query scales, the rotated rope keys, the cached",
        "// latents, the attention output per position run, the router's picks and weights, and the",
        "// layer's NVFP4 expert containers with the MoE block's output.",
        "#pragma once",
        "#include <cstdint>",
        "",
        "namespace mistral4_vectors {",
        f"inline constexpr int kHidden = {m.H}, kHeads = {m.heads}, kQLora = {m.q_lora}, kKvLora = {m.kv_lora};",
        f"inline constexpr int kNope = {m.nope}, kRope = {m.rope}, kV = {m.vdim}, kTokens = {T};",
        f"inline constexpr int kExperts = {m.n_experts}, kTopK = {m.top_k};",
        f"inline constexpr double kTheta = {m.theta!r}, kYarnFactor = {m.yarn_factor!r};",
        f"inline constexpr int kYarnOriginal = {int(p['yarn']['original_max_position_embeddings'])};",
        f"inline constexpr double kYarnBetaFast = {float(p['yarn']['beta'])!r}, kYarnBetaSlow = {float(p['yarn']['alpha'])!r};",
        f"inline constexpr double kLlama4Beta = {m.l4_beta!r};",
        f"inline constexpr int kLlama4Original = {m.l4_orig};",
        f"inline constexpr float kEps = {m.eps!r}f;",
        f"inline constexpr int kRuns = {len(runs)};",
        f"inline constexpr long long kRunPos0[{len(runs)}] = {{{', '.join(str(r) for r in runs)}}};",
    ]
    u16 = lambda v: f"0x{int(v):04x}"                                    # noqa: E731
    f32 = lambda v: f"{float(np.float32(v))!r}f"                          # noqa: E731
    f64 = lambda v: f"{float(v)!r}"                                       # noqa: E731
    for n, arr in w.items():
        lines.append(_c_array("k_" + n, "uint16_t", arr, u16))
    lines.append(_c_array("k_gate", "uint16_t", gate_bits, u16))
    lines.append(_c_array("k_hidden", "uint16_t", h_bits, u16))
    lines.append(_c_array("k_inv_freq", "float", m.inv_freq, f32, 6))
    for r, pos0 in enumerate(runs):
        taps = {}
        st = m.new_state(pos0)
        pos = pos0 + np.arange(T, dtype=np.int64)
        out = m.attention(layer, h, pos, st, tap=lambda k, v: taps.__setitem__(k.split(".", 1)[1], np.array(v)))
        lines.append(_c_array(f"k_run{r}_llama4", "double", m.llama4(pos).astype(np.float64), f64, 6))
        lines.append(_c_array(f"k_run{r}_latent", "double", taps["latent"], f64, 6))
        lines.append(_c_array(f"k_run{r}_k_rope", "double", taps["k_rope"], f64, 6))
        lines.append(_c_array(f"k_run{r}_q", "double", taps["q"], f64, 6))
        lines.append(_c_array(f"k_run{r}_attn_heads", "double", taps["attn_heads"], f64, 6))
        lines.append(_c_array(f"k_run{r}_out", "double", out, f64, 6))
    # The MoE of the same layer on the same rows: the router's picks (ascending expert id, as the
    # engine's oracle reports them) and the block's output, with the NVFP4 containers verbatim in
    # the engine's matrix order — expert e's gate (w1), up (w3), down (w2) at index e * 3 + m, the
    # shared triple last; the global is the stored weight_global_scale (the engine's divisor).
    rid, rw = m.route(layer, h)
    order = np.argsort(rid, axis=-1, kind="stable")
    lines.append(_c_array("k_route_ids", "int", np.take_along_axis(rid, order, axis=-1), lambda v: str(int(v))))
    lines.append(_c_array("k_route_w", "double", np.take_along_axis(rw, order, axis=-1), f64, 6))
    lines.append(_c_array("k_moe_out", "double", m.moe(layer, h), f64, 6))
    u8 = lambda v: str(int(v))                                           # noqa: E731
    prefixes = [f"layers.{layer}.experts.{e}.{mat}" for e in range(m.n_experts) for mat in ("w1", "w3", "w2")]
    prefixes += [f"layers.{layer}.shared_experts.{mat}" for mat in ("w1", "w3", "w2")]
    lines.append(f"inline constexpr int kInter = {m.inter};")
    lines.append(_c_array("k_fp4_payloads", "uint8_t", np.concatenate(
        [m.ck.raw(q + ".weight_packed").reshape(-1) for q in prefixes]), u8, 24))
    lines.append(_c_array("k_fp4_scales", "uint8_t", np.concatenate(
        [m.ck.raw(q + ".weight_scale").reshape(-1) for q in prefixes]), u8, 24))
    lines.append(_c_array("k_fp4_globals", "float", np.concatenate(
        [np.asarray(m.ck.raw(q + ".weight_global_scale"), np.float32).reshape(-1) for q in prefixes]), f32, 6))
    lines += ["}  // namespace mistral4_vectors", ""]
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {a.out}: layer {layer}, {T} tokens, runs at positions {runs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ckpt"); p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=20261004)
    p.set_defaults(fn=cmd_ckpt)
    p = sub.add_parser("selftest"); p.add_argument("--ckpt", required=True); p.set_defaults(fn=cmd_selftest)
    p = sub.add_parser("hf-check"); p.add_argument("--ckpt", required=True); p.set_defaults(fn=cmd_hf_check)
    p = sub.add_parser("vectors"); p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True)
    p.add_argument("--layer", type=int, default=1); p.add_argument("--tokens", type=int, default=12)
    p.add_argument("--seed", type=int, default=7); p.set_defaults(fn=cmd_vectors)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
