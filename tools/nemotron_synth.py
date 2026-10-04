#!/usr/bin/env python3
"""nemotron_synth.py — tiny synthetic Nemotron-H checkpoints and the checks that run on them
(2026-10-04). No GPU, no torch: numpy only. The synthetic checkpoints carry the real releases'
tensor names, containers and recipes at toy sizes, so tools/nemotron_reference.py and the engine's
config parser / binding table can be exercised end to end on a laptop.

  ckpt          --preset nano|super --out DIR [--seed N]
        writes config.json, hf_quant_config.json, generation_config.json, ids.json and the
        safetensors shards (super: two shards), plus expected_dequant.npz — the float value every
        quantized matrix must dequantize to.
          nano   the Nvfp4Exclude recipe (hf_quant_config.json alone): NVFP4 on every linear but
                 the excluded ones, an F32 router gate, no K/V scales, no draft block.
          super  the MixedPrecision recipe (config.json's quantization_config.quantized_layers):
                 NVFP4 experts, per-tensor FP8 and BF16 mixed layer by layer, a latent MoE, a BF16
                 router gate, k_scale / v_scale, the `*E` draft block under mtp.
  selftest      --ckpt DIR
        through nemotron_reference.Model: (1) every quantized matrix dequantizes to
        expected_dequant.npz exactly; (2) the Mamba recurrence equals an independent transcription
        of the reference's chunked SSD algorithm (modeling_nemotron_h.py torch_forward), output and
        final state; (3) feeding a sequence whole, in chunks and token by token gives the same
        logits; (4) greedy generation reproduces teacher-forced scoring of its own output.
  mamba-vectors --ckpt DIR --out tests/unit/nemotron_mamba2_vectors.hpp [--layer L] [--tokens T]
        the Mamba2 test vectors of tests/unit/nemotron_mamba2_test.cpp: one Mamba layer's weights
        and a deterministic in_proj output as float32, and what nemotron_reference.Model.mamba_core
        makes of them in float64 (conv, time step, scan, gated norm, final states).

What a pass here does and does not show: the reference is self-consistent, its recurrence is the
reference algorithm, and its readers invert this file's writers. The writers follow the same
reading of the modelopt containers as the readers, so the container conventions themselves (nibble
order, scale direction) are only proven on a real checkpoint.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nemotron_reference as ref  # noqa: E402

# ----------------------------------------------------------------------------------------------
# Encoders (the inverse of nemotron_reference's decoders)
# ----------------------------------------------------------------------------------------------
_E2M1_MAG = ref._E2M1[:8].astype(np.float64)          # 0 .. 6, ascending, index = code
_E4M3_MAG = ref._E4M3[:127].astype(np.float64)        # 0 .. 448, ascending, index = code


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


def encode_nvfp4(w):
    """float [N, K] -> (codes U8 [N, K/2], block scales e4m3 U8 [N, K/16], weight_scale_2 F32,
    the float32 matrix those decode to). modelopt's recipe: weight_scale_2 = amax / (6 * 448),
    block scale = e4m3(amax(block) / 6 / weight_scale_2), code = e2m1(w / (scale * weight_scale_2))."""
    n, k = w.shape
    assert k % 16 == 0
    amax = float(np.abs(w).max())
    ws2 = np.float32(amax / (6.0 * 448.0)) if amax > 0 else np.float32(1.0)
    blk = np.asarray(w, np.float64).reshape(n, k // 16, 16)
    sc = _nearest(np.abs(blk).max(axis=-1) / 6.0 / float(ws2), _E4M3_MAG).astype(np.uint8)   # [N, K/16]
    scale = ref._E4M3[sc] * ws2                                                             # float32
    inv = np.where(scale > 0, 1.0 / np.where(scale > 0, scale, 1), 0.0)
    q = blk * inv[:, :, None]
    code = (_nearest(np.abs(q), _E2M1_MAG) | ((q < 0) << 3)).astype(np.uint8).reshape(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)                        # low nibble = even column
    deq = (ref._E2M1[code].reshape(n, k // 16, 16) * scale[:, :, None]).reshape(n, k).astype(np.float32)
    return packed, sc, ws2, deq


def encode_fp8(w):
    """float [N, K] -> (codes e4m3 U8 [N, K], weight_scale F32, the float32 matrix those decode to):
    one scale per tensor, amax / 448."""
    amax = float(np.abs(w).max())
    s = np.float32(amax / 448.0) if amax > 0 else np.float32(1.0)
    q = np.asarray(w, np.float64) / float(s)
    code = (_nearest(np.abs(q), _E4M3_MAG) | ((q < 0) << 7)).astype(np.uint8)
    return code, s, (ref._E4M3[code] * s).astype(np.float32)


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


# ----------------------------------------------------------------------------------------------
# The synthetic checkpoints
# ----------------------------------------------------------------------------------------------
BASE_CONFIG = {
    "architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h",
    "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
    "chunk_size": 4, "conv_kernel": 4, "expand": 2, "head_dim": 8, "hidden_dropout": 0.0, "hidden_size": 32,
    "hybrid_override_pattern": "MEMEM*EME", "initializer_range": 0.02, "intermediate_size": 64,
    "layer_norm_epsilon": 1e-05, "mamba_head_dim": 8, "mamba_hidden_act": "silu", "mamba_num_heads": 4,
    "mamba_proj_bias": False, "mamba_ssm_cache_dtype": "float32", "max_position_embeddings": 4096,
    "mlp_bias": False, "mlp_hidden_act": "relu2", "moe_intermediate_size": 32,
    "moe_shared_expert_intermediate_size": 48, "n_group": 1, "n_groups": 2, "n_routed_experts": 8,
    "n_shared_experts": 1, "norm_eps": 1e-05, "norm_topk_prob": True, "num_attention_heads": 4,
    "num_experts_per_tok": 3, "num_hidden_layers": 9, "num_key_value_heads": 2, "num_logits_to_keep": 1,
    "partial_rotary_factor": 1.0, "rescale_prenorm_residual": True, "residual_in_fp32": False,
    "rope_theta": 10000, "routed_scaling_factor": 2.5, "sliding_window": None, "ssm_state_size": 16,
    "tie_word_embeddings": False, "time_step_floor": 0.0001, "time_step_max": 0.1, "time_step_min": 0.001,
    "topk_group": 1, "use_bias": False, "use_cache": True, "use_conv_bias": True, "use_mamba_kernels": True,
    "vocab_size": 96,
}
_KIND = {"M": "mamba", "E": "moe", "*": "attention"}


def layer_linears(cfg, prefix, kind):
    """(module name, rows, cols) of every nn.Linear of one layer, in the reference's order."""
    H = cfg["hidden_size"]
    p = prefix + ".mixer."
    if kind == "mamba":
        inner = cfg["mamba_num_heads"] * cfg["mamba_head_dim"]
        conv_dim = inner + 2 * cfg["n_groups"] * cfg["ssm_state_size"]
        return [(p + "in_proj", inner + conv_dim + cfg["mamba_num_heads"], H), (p + "out_proj", H, inner)]
    if kind == "attention":
        q, kv = cfg["num_attention_heads"] * cfg["head_dim"], cfg["num_key_value_heads"] * cfg["head_dim"]
        return [(p + "q_proj", q, H), (p + "k_proj", kv, H), (p + "v_proj", kv, H), (p + "o_proj", H, q)]
    X = cfg.get("moe_latent_size") or H
    I, S = cfg["moe_intermediate_size"], cfg["moe_shared_expert_intermediate_size"]
    out = []
    for e in range(cfg["n_routed_experts"]):
        out += [(f"{p}experts.{e}.up_proj", I, X), (f"{p}experts.{e}.down_proj", X, I)]
    out += [(p + "shared_experts.up_proj", S, H), (p + "shared_experts.down_proj", H, S)]
    if cfg.get("moe_latent_size"):
        out += [(p + "fc1_latent_proj", X, H), (p + "fc2_latent_proj", H, X)]
    return out


def build(preset, seed):
    """-> (config dict, hf_quant_config dict, {tensor name: (dtype, bits)}, {module: dequantized float32})."""
    rng = np.random.RandomState(seed)                     # the legacy generator: a frozen stream
    cfg = dict(BASE_CONFIG)
    kinds = [_KIND[ch] for ch in cfg["hybrid_override_pattern"]]
    H, V = cfg["hidden_size"], cfg["vocab_size"]
    quant = {}                                            # module -> "NVFP4" | "FP8" (absent: BF16)
    if preset == "super":
        cfg.update({"moe_latent_size": 16, "num_nextn_predict_layers": 1, "mtp_hybrid_override_pattern": "*E",
                    "moe_shared_expert_overlap": False, "routed_scaling_factor": 5.0, "dtype": "bfloat16"})
        # A layer-by-layer mix like the release's: every expert NVFP4; in_proj / out_proj FP8 on some
        # Mamba layers and BF16 on others; the shared expert FP8, NVFP4 or BF16; one latent projection
        # of each side FP8; the attention o_proj FP8.
        for L, kind in enumerate(kinds):
            for name, _, _ in layer_linears(cfg, f"backbone.layers.{L}", kind):
                leaf = name.split(".")[-1]
                if ".experts." in name:
                    quant[name] = "NVFP4"
                elif leaf == "in_proj" and L in (0, 4):
                    quant[name] = "FP8"
                elif leaf == "out_proj" and L in (0, 2):
                    quant[name] = "FP8"
                elif "shared_experts" in name and leaf == "up_proj" and L != 8:
                    quant[name] = "FP8"
                elif "shared_experts" in name and leaf == "down_proj" and L != 1:
                    quant[name] = "NVFP4" if L == 3 else "FP8"
                elif leaf == "fc1_latent_proj" and L == 1:
                    quant[name] = "FP8"
                elif leaf == "fc2_latent_proj" and L == 3:
                    quant[name] = "FP8"
                elif leaf == "o_proj":
                    quant[name] = "FP8"
        layers = {n: ({"quant_algo": "NVFP4", "group_size": 16} if a == "NVFP4" else {"quant_algo": "FP8"})
                  for n, a in quant.items()}
        groups = {
            "group_0": {"input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                        "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                        "targets": sorted(n for n, a in quant.items() if a == "FP8")},
            "group_1": {"input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                        "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                        "targets": sorted(n for n, a in quant.items() if a == "NVFP4")}}
        cfg["quantization_config"] = {
            "config_groups": groups, "quantized_layers": layers, "ignore": [], "quant_algo": "MIXED_PRECISION",
            "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
            "producer": {"name": "modelopt", "version": "synthetic"}, "quant_method": "modelopt"}
        hf = {"producer": {"name": "modelopt", "version": "synthetic"},
              "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": "FP8", "quantized_layers": layers}}
    else:
        cfg["torch_dtype"] = "bfloat16"
        # The release's exclude list: the head, the attention projections, the Mamba layer feeding
        # each attention layer, every convolution.
        exclude = ["lm_head"]
        for L, kind in enumerate(kinds):
            p = f"backbone.layers.{L}.mixer."
            if kind == "mamba" and L + 1 < len(kinds) and kinds[L + 1] == "attention":
                exclude += [p + "in_proj", p + "out_proj"]
            if kind == "attention":
                exclude += [p + "q_proj", p + "k_proj", p + "v_proj", p + "o_proj"]
        exclude += [f"backbone.layers.{L}.mixer.conv1d" for L, kind in enumerate(kinds) if kind == "mamba"]
        for L, kind in enumerate(kinds):
            for name, _, _ in layer_linears(cfg, f"backbone.layers.{L}", kind):
                if name not in exclude:
                    quant[name] = "NVFP4"
        hf = {"producer": {"name": "modelopt", "version": "synthetic"},
              "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", "group_size": 16,
                               "exclude_modules": exclude}}

    T, deq = {}, {}

    def bf16(name, w):
        T[name] = ("BF16", to_bf16(w))

    def scalar(name, v):
        T[name] = ("F32", np.array(v, np.float32).reshape(()))

    def linear(name, rows, cols):
        w = (rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)
        algo = quant.get(name)
        if algo == "NVFP4":
            packed, sc, ws2, d = encode_nvfp4(w)
            T[name + ".weight"] = ("U8", packed)
            T[name + ".weight_scale"] = ("F8_E4M3", sc)
            scalar(name + ".weight_scale_2", ws2)
            scalar(name + ".input_scale", 0.01 + rng.rand())
            deq[name] = d
        elif algo == "FP8":
            code, s, d = encode_fp8(w)
            T[name + ".weight"] = ("F8_E4M3", code)
            scalar(name + ".weight_scale", s)
            scalar(name + ".input_scale", 0.01 + rng.rand())
            deq[name] = d
        else:
            bf16(name + ".weight", w)

    def norm(name, n):
        bf16(name, 1.0 + 0.1 * rng.standard_normal(n).astype(np.float32))

    def block(prefix, kind, backbone):
        norm(prefix + ".norm.weight", H)
        p = prefix + ".mixer."
        if kind == "mamba":
            nh = cfg["mamba_num_heads"]
            inner = nh * cfg["mamba_head_dim"]
            conv_dim = inner + 2 * cfg["n_groups"] * cfg["ssm_state_size"]
            bf16(p + "A_log", np.log(rng.uniform(1.0, 16.0, nh)).astype(np.float32))
            bf16(p + "D", (1.0 + 0.2 * rng.standard_normal(nh)).astype(np.float32))
            step = np.exp(rng.uniform(np.log(0.001), np.log(0.1), nh))
            bf16(p + "dt_bias", (step + np.log(-np.expm1(-step))).astype(np.float32))   # softplus^-1
            bf16(p + "conv1d.weight", (0.4 * rng.standard_normal((conv_dim, 1, cfg["conv_kernel"]))).astype(np.float32))
            bf16(p + "conv1d.bias", (0.1 * rng.standard_normal(conv_dim)).astype(np.float32))
            norm(p + "norm.weight", inner)
        elif kind == "moe":
            E = cfg["n_routed_experts"]
            g = (rng.standard_normal((E, H)) / np.sqrt(H)).astype(np.float32)
            if preset == "super":
                bf16(p + "gate.weight", g)
            else:
                T[p + "gate.weight"] = ("F32", g)
            T[p + "gate.e_score_correction_bias"] = ("F32", (0.05 * rng.standard_normal(E)).astype(np.float32))
        for name, rows, cols in layer_linears(cfg, prefix, kind):
            linear(name, rows, cols)
        if kind == "attention" and backbone and preset == "super":
            scalar(p + "k_proj.k_scale", 0.01 + rng.rand())
            scalar(p + "v_proj.v_scale", 0.01 + rng.rand())

    bf16("backbone.embeddings.weight", rng.standard_normal((V, H)).astype(np.float32))
    for L, kind in enumerate(kinds):
        block(f"backbone.layers.{L}", kind, True)
    norm("backbone.norm_f.weight", H)
    bf16("lm_head.weight", (rng.standard_normal((V, H)) / np.sqrt(H)).astype(np.float32))
    if preset == "super":
        norm("mtp.layers.0.enorm.weight", H)
        norm("mtp.layers.0.hnorm.weight", H)
        bf16("mtp.layers.0.eh_proj.weight", (rng.standard_normal((H, 2 * H)) / np.sqrt(2 * H)).astype(np.float32))
        block("mtp.layers.0", "attention", False)
        block("mtp.layers.1", "moe", False)
        norm("mtp.layers.1.final_layernorm.weight", H)
    return cfg, hf, T, deq


def cmd_ckpt(a):
    cfg, hf, T, deq = build(a.preset, a.seed)
    os.makedirs(a.out, exist_ok=True)
    names = list(T)
    shards = [names] if a.preset == "nano" else [names[:len(names) // 2], names[len(names) // 2:]]
    for i, part in enumerate(shards, 1):
        fn = "model.safetensors" if len(shards) == 1 else f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        write_safetensors(os.path.join(a.out, fn), {n: T[n] for n in part})
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(a.out, "hf_quant_config.json"), "w") as f:
        json.dump(hf, f, indent=1)
    with open(os.path.join(a.out, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 1, "eos_token_id": [2, 11], "pad_token_id": 0}, f)
    ids = [1] + [int(t) for t in np.random.RandomState(a.seed + 1).randint(3, cfg["vocab_size"], 23)]
    with open(os.path.join(a.out, "ids.json"), "w") as f:
        json.dump(ids, f)
    np.savez(os.path.join(a.out, "expected_dequant.npz"), **deq)
    nbytes = sum(arr.nbytes for _, arr in T.values())
    print(f"# {a.preset}: {len(T)} tensors, {nbytes} bytes, {len(deq)} quantized matrices, "
          f"{len(shards)} shard(s) -> {a.out}")


# ----------------------------------------------------------------------------------------------
# The reference's chunked SSD algorithm, transcribed on its own (modeling_nemotron_h.py,
# NemotronHMamba2Mixer.torch_forward, the "ssd naive implementation" branch) — batch dim dropped.
# ----------------------------------------------------------------------------------------------
def _segment_sum(a):
    """[..., T] -> [..., T, T]: entry (i, j) = sum_{k = j+1 .. i} a[k] for i >= j, -inf above the diagonal."""
    T = a.shape[-1]
    x = np.repeat(a[..., None], T, axis=-1)                     # x[..., i, j] = a[i]
    x = np.where(np.tril(np.ones((T, T), bool), -1), x, 0.0)
    s = np.cumsum(x, axis=-2)
    return np.where(np.tril(np.ones((T, T), bool), 0), s, -np.inf)


def ssd_chunked(x, dt, A, B, C, D, chunk):
    """x [T, heads, head_dim] (after the conv), dt [T, heads] (after softplus), A [heads] (negative),
    B, C [T, heads, N] (already repeated per head), D [heads]. Returns (y [T, heads, head_dim],
    final state [heads, head_dim, N])."""
    T, nh, hd = x.shape
    pad = (chunk - T % chunk) % chunk

    def padded(t):
        return np.concatenate([t, np.zeros((pad,) + t.shape[1:], t.dtype)]) if pad else t

    D_res = D[:, None] * padded(x)
    xd = padded(x * dt[..., None]).reshape(-1, chunk, nh, hd)           # [c, l, h, p]
    Ad = padded(A[None, :] * dt).reshape(-1, chunk, nh)                 # [c, l, h]
    Bc = padded(B).reshape(-1, chunk, nh, B.shape[-1])                  # [c, l, h, n]
    Cc = padded(C).reshape(-1, chunk, nh, C.shape[-1])
    Ad = Ad.transpose(2, 0, 1)                                          # [h, c, l]
    A_cumsum = np.cumsum(Ad, axis=-1)
    # 1. intra-chunk (diagonal blocks)
    L = np.exp(_segment_sum(Ad))                                        # [h, c, l, s]
    G = np.einsum("clhn,cshn->clsh", Cc, Bc)
    M = G * L.transpose(1, 2, 3, 0)
    Y_diag = np.einsum("clsh,cshp->clhp", M, xd)
    # 2. the state each chunk leaves (B terms)
    decay_states = np.exp(A_cumsum[:, :, -1:] - A_cumsum)               # [h, c, l]
    states = np.einsum("clhn,hcl,clhp->chpn", Bc, decay_states, xd)     # [c, h, p, n]
    # 3. inter-chunk recurrence (A terms)
    states = np.concatenate([np.zeros_like(states[:1]), states])        # [c + 1, h, p, n]
    last = np.concatenate([np.zeros_like(A_cumsum[:, :1, -1]), A_cumsum[:, :, -1]], axis=1)   # [h, c + 1]
    decay_chunk = np.exp(_segment_sum(last))                            # [h, c + 1, c + 1]
    new_states = np.einsum("hzc,chpn->zhpn", decay_chunk, states)
    states, final = new_states[:-1], new_states[-1]
    # 4. state -> output per chunk (C terms)
    Y_off = np.einsum("clhn,chpn,hcl->clhp", Cc, states, np.exp(A_cumsum))
    y = (Y_diag + Y_off).reshape(-1, nh, hd) + D_res
    return y[:T], final


def cmd_selftest(a):
    fails = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    m = ref.Model(a.ckpt, dtype="float64", threads=2)
    ids = ref.load_ids(os.path.join(a.ckpt, "ids.json"))

    # (1) every quantized matrix dequantizes to what the writer intended
    exp = np.load(os.path.join(a.ckpt, "expected_dequant.npz"))
    worst, kinds = 0.0, {"nvfp4": 0, "fp8": 0}
    for name in exp.files:
        kinds[m.fmt(name)] += 1
        worst = max(worst, float(np.abs(m._linear_nocache(name).astype(np.float32) - exp[name]).max()))
    check("dequant equals the written values", worst == 0.0 and sum(kinds.values()) == len(exp.files),
          f"{kinds['nvfp4']} NVFP4 + {kinds['fp8']} FP8 matrices, max |diff| {worst:g}")

    # (2) the recurrence vs the reference's chunked SSD, on every Mamba layer of a real forward
    taps = {}
    ref.run_forward(m, ids, tap=lambda n, arr: taps.__setitem__(n, np.array(arr)),
                    detail_layers=range(m.n_layers))
    worst_y = worst_s = 0.0
    n_mamba = 0
    for L, kind in enumerate(m.kinds):
        if kind != "mamba":
            continue
        n_mamba += 1
        p = f"backbone.layers.{L}.mixer"
        c = taps[f"L{L:02d}_mamba_conv"].astype(np.float64)
        inner, G, N, nh, hd = m.m_inner, m.m_groups, m.m_state, m.m_heads, m.m_hd
        x = c[:, :inner].reshape(-1, nh, hd)
        B = np.repeat(c[:, inner:inner + G * N].reshape(-1, G, N), nh // G, axis=1)
        C = np.repeat(c[:, inner + G * N:].reshape(-1, G, N), nh // G, axis=1)
        A = -np.exp(m.dense(p + ".A_log").astype(np.float64))
        for chunk in (4, 5, 64):                       # padded, ragged and single-chunk
            y, final = ssd_chunked(x, taps[f"L{L:02d}_mamba_dt"].astype(np.float64), A, B, C,
                                   m.dense(p + ".D").astype(np.float64), chunk)
            worst_y = max(worst_y, float(np.abs(y - taps[f"L{L:02d}_mamba_scan"]).max()))
            worst_s = max(worst_s, float(np.abs(final - taps[f"L{L:02d}_mamba_state"]).max()))
    check("Mamba recurrence equals the reference's chunked SSD", n_mamba > 0 and worst_y < 1e-9 and worst_s < 1e-9,
          f"{n_mamba} layers x chunk 4/5/64: max |dy| {worst_y:.3g}, max |dstate| {worst_s:.3g}")

    # (3) whole vs chunked vs token-by-token feeding
    whole = m.logits(ref.run_forward(m, ids)[0])
    check("the logits are finite", bool(np.isfinite(whole).all()), f"[{whole.shape[0]}, {whole.shape[1]}]")
    worst = 0.0
    for chunk in (1, 5, 7):
        worst = max(worst, float(np.abs(m.logits(ref.run_forward(m, ids, chunk=chunk)[0]) - whole).max()))
    check("chunked feeding equals whole-sequence feeding", worst < 1e-9, f"chunks 1/5/7: max |dlogit| {worst:.3g}")

    # (4) greedy generation vs teacher forcing of its own output
    new, steps = ref.greedy(m, ids[:8], 6, stop_eos=False)
    res = ref.score_ids(m, ids[:8] + new)
    worst = max(abs(res["logprobs"][8 + i] - steps[i]["top"][0][1]) for i in range(len(new)))
    top1 = all(res["top"][8 + i][0][0] == new[i] for i in range(len(new)))
    check("greedy generation equals teacher forcing of its output", top1 and worst < 1e-9,
          f"{len(new)} tokens: max |dlogprob| {worst:.3g}")

    # (5) the float32 run stays near the float64 one (a noise-floor figure, not a gate on its size)
    m32 = ref.Model(a.ckpt, dtype="float32", threads=2)
    d = float(np.abs(m32.logits(ref.run_forward(m32, ids)[0]) - whole).max())
    check("float32 run within 1e-2 of the float64 run", d < 1e-2, f"max |dlogit| {d:.3g}")

    # (6) the draft block runs (Super preset); its conventions are not checked by anything here
    if m.mtp_kinds:
        x, _ = ref.run_forward(m, ids)
        lp = m.mtp_forward(ids[1:], x[:-1], m.mtp_new_state())
        ok = lp.shape == (len(ids) - 1, m.vocab) and bool(np.isfinite(lp).all()) and \
            bool(np.allclose(np.exp(lp).sum(axis=-1), 1.0))
        check("the MTP draft block runs and normalizes (conventions unverified)", ok)

    print("SELFTEST " + ("OK" if not fails else "FAILED: " + ", ".join(fails)))
    return 1 if fails else 0


# ----------------------------------------------------------------------------------------------
# Mamba2 test vectors for the C++ host reference
# ----------------------------------------------------------------------------------------------
def cmd_mamba_vectors(a):
    m = ref.Model(a.ckpt, dtype="float64", threads=2)
    if m.kinds[a.layer] != "mamba":
        raise SystemExit(f"layer {a.layer} is {m.kinds[a.layer]}, not a Mamba layer")
    p = f"backbone.layers.{a.layer}.mixer"
    nh, hd, N, G, K = m.m_heads, m.m_hd, m.m_state, m.m_groups, m.m_conv
    inner, cd = m.m_inner, m.m_conv_dim
    rows = inner + cd + nh
    # A deterministic in_proj output, float32 so both sides start from identical bits.
    proj = (0.8 * np.random.RandomState(a.seed).standard_normal((a.tokens, rows))).astype(np.float32)
    taps, st = {}, {}
    out = m.mamba_core(p, proj.astype(np.float64), st, tap=lambda n, arr: taps.__setitem__(n, np.array(arr, np.float64)))

    def f32(name):
        return m.dense(name).astype(np.float32)

    arrays = [
        ("float", "kConvW", f32(p + ".conv1d.weight").reshape(cd, K), "conv1d.weight [conv_dim, K]"),
        ("float", "kConvB", f32(p + ".conv1d.bias"), "conv1d.bias [conv_dim]"),
        ("float", "kALog", f32(p + ".A_log"), "A_log [heads]"),
        ("float", "kD", f32(p + ".D"), "D [heads]"),
        ("float", "kDtBias", f32(p + ".dt_bias"), "dt_bias [heads]"),
        ("float", "kNormW", f32(p + ".norm.weight"), "norm.weight [inner]"),
        ("float", "kProj", proj, "in_proj output [tokens, inner + conv_dim + heads]"),
        ("double", "kWantConv", taps["mamba_conv"], "[x | B | C] after conv + SiLU [tokens, conv_dim]"),
        ("double", "kWantDt", taps["mamba_dt"], "softplus(dt + dt_bias) [tokens, heads]"),
        ("double", "kWantScan", taps["mamba_scan"], "S C + D x [tokens, heads, head_dim]"),
        ("double", "kWantOut", out, "gated-norm output [tokens, inner]"),
        ("double", "kWantSsm", st["ssm"], "final SSM state [heads, head_dim, N]"),
        ("double", "kWantConvTail", st["conv"], "final conv tail: the last K - 1 pre-conv rows [K - 1, conv_dim]"),
    ]
    lines = [
        "// GENERATED by tools/nemotron_synth.py mamba-vectors — do not edit.",
        "// Regenerate (python3 with numpy; the synthetic checkpoint is deterministic):",
        "//   python3 tools/nemotron_synth.py ckpt --preset nano --seed 0 --out <DIR>",
        f"//   python3 tools/nemotron_synth.py mamba-vectors --ckpt <DIR> --layer {a.layer} --tokens {a.tokens} "
        f"--seed {a.seed} --out tests/unit/nemotron_mamba2_vectors.hpp",
        "// One Mamba2 layer of the synthetic checkpoint: its weights and a deterministic in_proj output",
        "// as float32, and what tools/nemotron_reference.py (Model.mamba_core, --dtype float64) computes",
        "// from them, as float64.",
        "#pragma once",
        "",
        "namespace nemotron_mamba2_vectors {",
        "",
        f"inline constexpr int kHeads = {nh}, kHeadDim = {hd}, kState = {N}, kGroups = {G}, kConvKernel = {K};",
        f"inline constexpr int kTokens = {a.tokens};",
        f"inline constexpr double kEps = {m.eps!r};",
        "",
    ]
    for ctype, name, arr, what in arrays:
        flat = np.asarray(arr).reshape(-1)
        fmt = (lambda v: f"{float(v):.9g}f") if ctype == "float" else (lambda v: f"{float(v):.17g}")
        lines.append(f"// {what}")
        lines.append(f"inline constexpr {ctype} {name}[{flat.size}] = {{")
        for i in range(0, flat.size, 6):
            lines.append("    " + ", ".join(fmt(v) for v in flat[i:i + 6]) + ",")
        lines.append("};")
        lines.append("")
    lines.append("}  // namespace nemotron_mamba2_vectors")
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"# wrote {a.out}: layer {a.layer}, {a.tokens} tokens, heads {nh} x {hd}, state {N}, groups {G}, kernel {K}")


def main():
    ap = argparse.ArgumentParser(description="synthetic Nemotron-H checkpoints, the reference's self-test, Mamba2 test vectors")
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("ckpt", help="write a tiny synthetic checkpoint")
    p.add_argument("--preset", choices=["nano", "super"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_ckpt)
    p = sub.add_parser("selftest", help="consistency checks of nemotron_reference.py on a synthetic checkpoint")
    p.add_argument("--ckpt", required=True)
    p.set_defaults(fn=cmd_selftest)
    p = sub.add_parser("mamba-vectors", help="write the C++ Mamba2 test vectors")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--tokens", type=int, default=11)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_mamba_vectors)
    a = ap.parse_args()
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
