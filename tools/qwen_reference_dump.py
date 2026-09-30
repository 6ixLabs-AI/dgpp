#!/usr/bin/env python3
"""Qwen3.8-Flash-Next full-model reference dump (Q4, 2026-09-09).

Computes the text-model forward — embedding on every hyper branch, the
gated-residual-wired decoder layers (the PLE injection at its layer, Gated
DeltaNet or Qwen Sparse Attention, the softmax-routed MoE with its gated
shared expert), the final hyper mixer, the lm head — in pure python doubles
with bf16 rounding at the engine's boundaries, from the SAME checkpoint the
engine runs, and writes the comparable outputs (tokens, every layer's hyper
state, the final read, per-token top-k logits, the routing decisions) as a
DGPPQWND dump read by tests/cuda/qwen_forward_test.cpp.

The wiring is pinned to transformers' modular_qwen4_exp.py
(Qwen4ExpTextDecoderLayer, Qwen4ExpTextPLELayer, Qwen4ExpTextQSAIndexer,
Qwen3_5Attention, Qwen3_5GatedDeltaNet, Qwen3NextSparseMoeBlock,
Qwen4ExpTextGatedResidual); the per-module rounding points are the ones the
C++ oracles under src/models/qwen/*_reference.cpp spell out.

Usage:
  qwen_reference_dump.py gen-pure --checkpoint-dir DIR --out FILE [--tokens T] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from operator import mul

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kda_reference_dump as krd  # noqa: E402
from dsa_reference_dump import fp8_from_bits  # noqa: E402
from glm_reference_dump import read_safetensors_index, load_tensor  # noqa: E402

MAGIC = b"DGPPQWND"
VERSION = 1

bf16_round = krd.bf16_round
bf16_bits = krd.bf16_bits
make_rng = krd.make_rng


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def silu(x: float) -> float:
    return x * sigmoid(x)


def softplus(x: float) -> float:
    return x if x > 20.0 else math.log1p(math.exp(x))


def dot(a, b) -> float:
    return sum(map(mul, a, b))


def gemv(x, rows):
    """[bf16(x . row) for row in rows] — a bf16 Linear."""
    return [bf16_round(dot(x, r)) for r in rows]


def as_rows(flat, n, k):
    return [flat[i * k:(i + 1) * k] for i in range(n)]


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

def load_i64(entries, name):
    path, offset, nbytes, dtype_str, shape = entries[name]
    if dtype_str != "I64":
        raise ValueError("expected I64 for %s" % name)
    with open(path, "rb") as f:
        f.seek(offset)
        return list(struct.unpack("<%dq" % (nbytes // 8), f.read(nbytes)))


def load_bf16(entries, name):
    dtype, shape, values = load_tensor(entries, name)
    if dtype not in ("BF16", "F32"):
        raise ValueError("expected BF16/F32 for %s" % name)
    return shape, values


def fp16_from_bits(h: int) -> float:
    """IEEE binary16 bits -> float, exact (subnormals included)."""
    sign = -1.0 if h & 0x8000 else 1.0
    exp = (h >> 10) & 0x1F
    man = h & 0x3FF
    if exp == 0:
        return sign * man * 2.0 ** -24
    if exp == 31:
        return sign * float("inf") if man == 0 else float("nan")
    return sign * (1.0 + man / 1024.0) * 2.0 ** (exp - 15)


def load_words(entries, name, fmt):
    path, offset, nbytes, dtype_str, shape = entries[name]
    size = {"I": 4, "H": 2}[fmt]
    with open(path, "rb") as f:
        f.seek(offset)
        buf = f.read(nbytes)
    return shape, struct.unpack("<%d%s" % (nbytes // size, fmt), buf)


def dequant_gptq(entries, base, bits):
    """The AutoRound hybrid's GPTQ triple (qweight I32 [K*bits/32, N] with codes
    packed along K low first, scales F16 [K/g, N]) -> EXACT (code - 2^(bits-1))
    x f16(scale) weights as [N, K] (the packed core's exact policy: no
    per-weight rounding; docs/qwen38_autoround_int4_plan.md D1)."""
    (wr, N), words = load_words(entries, base + ".qweight", "I")
    (groups, _), scales = load_words(entries, base + ".scales", "H")
    per = 32 // bits
    K = wr * per
    g = K // groups
    mask = (1 << bits) - 1
    off = 1 << (bits - 1)
    sc = [fp16_from_bits(h) for h in scales]
    out = [0.0] * (N * K)
    for i in range(wr):
        row = i * N
        for n in range(N):
            w = words[row + n]
            base_k = i * per
            for j in range(per):
                k = base_k + j
                out[n * K + k] = (((w >> (bits * j)) & mask) - off) * sc[(k // g) * N + n]
    return N, K, out


def load_head(entries):
    """The lm_head as shipped: BF16, or the hybrid's int8 g128 GPTQ triple."""
    if "lm_head.qweight" in entries:
        return dequant_gptq(entries, "lm_head", 8)[2]
    return load_bf16(entries, "lm_head.weight")[1]


def load_dense(entries, name):
    """A dense projection as shipped: BF16, or (the hybrid) block FP8 with
    F32 scales through the engine's fp8 weight policy."""
    if entries[name][3] == "F8_E4M3":
        rows, cols, values = dequant_blocks(entries, name)
        return (rows, cols), values
    return load_bf16(entries, name)


def dequant_blocks(entries, name, block=128):
    """E4M3 payload x BF16 block scales (weight_scale_inv) -> bf16-rounded
    weights, one rounding of decode x scale (the engine's policy)."""
    _, (rows, cols), payload = load_tensor(entries, name)
    (sr, sc), scales = load_bf16(entries, name + "_scale_inv")
    out = [0.0] * (rows * cols)
    for r in range(rows):
        srow = (r // block) * sc
        base = r * cols
        for c in range(cols):
            out[base + c] = bf16_round(fp8_from_bits(payload[base + c]) * scales[srow + c // block])
    return rows, cols, out


def text_config(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    rope = tc.get("rope_parameters", {})
    eos = tc["eos_token_id"]
    return {
        "hidden": tc["hidden_size"], "vocab": tc["vocab_size"],
        "num_layers": tc["num_hidden_layers"], "layer_types": tc["layer_types"],
        "eps": tc.get("rms_norm_eps", 1e-6), "eos": eos[0] if isinstance(eos, list) else eos,
        "hc": tc["hc_count"], "lowrank": tc["hc_lowrank"],
        "gdn_kh": tc["linear_num_key_heads"], "gdn_vh": tc["linear_num_value_heads"],
        "gdn_kd": tc["linear_key_head_dim"], "gdn_vd": tc["linear_value_head_dim"],
        "gdn_conv": tc["linear_conv_kernel_dim"],
        "heads": tc["num_attention_heads"], "kv_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"],
        "rotary": int(tc["head_dim"] * rope.get("partial_rotary_factor", 1.0)),
        "theta": float(rope.get("rope_theta", 1e7)),
        "idx_heads": tc["indexer_n_heads"], "idx_dim": tc["indexer_head_dim"],
        "budget": tc["indexer_budget"], "kpool": tc["indexer_compress_ratio"],
        "experts": tc["num_experts"], "top_k": tc["num_experts_per_tok"],
        "inter": tc["moe_intermediate_size"], "shared_inter": tc["shared_expert_intermediate_size"],
        "norm_topk": tc.get("norm_topk_prob", True),
        "ple_layers": [i - 1 for i in tc.get("ple_layer_ids", [])],
        "ple_embed": tc.get("ple_embed_dim", 0), "ple_conv": tc.get("ple_conv_kernel_size", 4),
        "ngram": tc.get("ngram_size", 3), "heads_per_ngram": tc.get("heads_per_ngram", 8),
        "ngram_divisor": tc.get("make_ngram_vocab_size_divisible_by", 128),
        "ngram_parts": tc.get("split_ngram_parts", 1),
    }


LAYER = "model.language_model.layers.%d."


def gr_site(entries, prefix, with_inject):
    hc_norm = load_bf16(entries, prefix + "hc_norm.weight")[1]
    (r, w), down = load_bf16(entries, prefix + "input_mix_weight_down.weight")
    (w2, r2), up = load_bf16(entries, prefix + "input_mix_weight_up.weight")
    site = {"norm": hc_norm, "down": as_rows(down, r, w), "up": as_rows(up, w2, r2), "rank": r}
    if with_inject:
        (n, w3), inj = load_bf16(entries, prefix + "block_inject_weight.weight")
        site["inject"] = as_rows(inj, n, w3)
    return site


def layer_weights(cfg, entries, layer, prefix=None):
    """A main-stack layer, or (prefix given) the draft layer: one QSA layer
    under "mtp.layers.0." with its own GR sites and MoE, no PLE."""
    p = prefix if prefix is not None else LAYER % layer
    H = cfg["hidden"]
    lw = {"kind": "full_attention" if prefix is not None else cfg["layer_types"][layer]}
    lw["attn_gr"] = gr_site(entries, p + "attn_hyper_connection.", True)
    lw["mlp_gr"] = gr_site(entries, p + "mlp_hyper_connection.", True)
    if lw["kind"] == "linear_attention":
        q = p + "linear_attn."
        (c, _), qkv = load_dense(entries, q + "in_proj_qkv.weight")
        (c2, _, cw), conv = load_bf16(entries, q + "conv1d.weight")
        (zw, _), z = load_dense(entries, q + "in_proj_z.weight")
        (vh, _), a = load_bf16(entries, q + "in_proj_a.weight")
        _, b = load_bf16(entries, q + "in_proj_b.weight")
        lw["gdn"] = {
            "qkv": as_rows(qkv, c, H), "conv": as_rows(conv, c2, cw), "z": as_rows(z, zw, H),
            "a": as_rows(a, vh, H), "b": as_rows(b, vh, H),
            "a_log": load_bf16(entries, q + "A_log")[1], "dt_bias": load_bf16(entries, q + "dt_bias")[1],
            "norm": load_bf16(entries, q + "norm.weight")[1],
            "out": as_rows(load_dense(entries, q + "out_proj.weight")[1], H, zw),
        }
    else:
        q = p + "self_attn."
        (qw, _), qp = load_dense(entries, q + "q_proj.weight")
        (kw, _), kp = load_dense(entries, q + "k_proj.weight")
        _, vp = load_dense(entries, q + "v_proj.weight")
        (_, ow), op = load_dense(entries, q + "o_proj.weight")
        (iw, _), ip = load_bf16(entries, q + "indexer.index_qk_proj.weight")
        lw["qsa"] = {
            "q": as_rows(qp, qw, H), "k": as_rows(kp, kw, H), "v": as_rows(vp, kw, H),
            "o": as_rows(op, H, ow), "q_norm": load_bf16(entries, q + "q_norm.weight")[1],
            "k_norm": load_bf16(entries, q + "k_norm.weight")[1],
            "idx": as_rows(ip, iw, H), "idx_q_norm": load_bf16(entries, q + "indexer.q_layernorm.weight")[1],
            "idx_k_norm": load_bf16(entries, q + "indexer.k_layernorm.weight")[1],
        }
    m = p + "mlp."
    (E, _), gate = load_bf16(entries, m + "gate.weight")
    experts = []
    for e in range(E):
        ep = m + "experts.%d." % e
        if (ep + "gate_proj.qweight") in entries:
            _, _, g = dequant_gptq(entries, ep + "gate_proj", 4)
            _, _, u = dequant_gptq(entries, ep + "up_proj", 4)
            _, _, d = dequant_gptq(entries, ep + "down_proj", 4)
        else:
            _, _, g = dequant_blocks(entries, ep + "gate_proj.weight")
            _, _, u = dequant_blocks(entries, ep + "up_proj.weight")
            _, _, d = dequant_blocks(entries, ep + "down_proj.weight")
        I = cfg["inter"]
        experts.append((as_rows(g, I, H), as_rows(u, I, H), as_rows(d, H, I)))
    S = cfg["shared_inter"]
    lw["moe"] = {
        "gate": as_rows(gate, E, H), "experts": experts,
        "s_gate": as_rows(load_dense(entries, m + "shared_expert.gate_proj.weight")[1], S, H),
        "s_up": as_rows(load_dense(entries, m + "shared_expert.up_proj.weight")[1], S, H),
        "s_down": as_rows(load_dense(entries, m + "shared_expert.down_proj.weight")[1], H, S),
        "sg": load_bf16(entries, m + "shared_expert_gate.weight")[1],
    }
    if prefix is None and layer in cfg["ple_layers"]:
        pp = p + "ple."
        W = cfg["hc"] * H
        E_ = cfg["ple_embed"]
        heads = (cfg["ngram"] - 1) * cfg["heads_per_ngram"]
        hd = E_ // heads
        ep = pp + "ple_embedding."
        vocab_sizes = load_i64(entries, ep + "ngram_heads_vocab_sizes")
        offsets = load_i64(entries, ep + "ngram_heads_offsets")
        total = sum(vocab_sizes)
        padded = (total + cfg["ngram_divisor"] - 1) // cfg["ngram_divisor"] * cfg["ngram_divisor"]
        codes = []
        for s in range(cfg["ngram_parts"]):
            _, (rows, hd2), payload = load_tensor(entries, ep + "ngram_embedding.shard_%d.weight" % s)
            if hd2 != hd:
                raise ValueError("table head_dim disagrees")
            codes.extend(payload)
        if len(codes) != padded * hd:
            raise ValueError("table rows %d != padded %d" % (len(codes) // hd, padded))
        conv_shape, conv = load_bf16(entries, pp + "conv1d.weight")
        cwd = conv_shape[-1]
        lw["ple"] = {
            "mult": load_i64(entries, ep + "layer_multipliers"), "vocab": vocab_sizes, "offset": offsets,
            "heads": heads, "hd": hd, "heads_per_ngram": cfg["heads_per_ngram"], "codes": codes,
            "scale": load_bf16(entries, ep + "ngram_embedding.weight_scale")[1][0],
            "key": as_rows(load_bf16(entries, pp + "key_proj.weight")[1], W, E_),
            "value": as_rows(load_bf16(entries, pp + "value_proj.weight")[1], H, E_),
            "norm_key": load_bf16(entries, pp + "norm_key.weight")[1],
            "norm_query": load_bf16(entries, pp + "norm_query.weight")[1],
            "norm_conv": load_bf16(entries, pp + "norm_conv.weight")[1],
            "conv": as_rows(conv, W, cwd),
        }
    return lw


# ---------------------------------------------------------------------------
# modules (double interior, bf16 at the engine's boundaries)
# ---------------------------------------------------------------------------

def rmsnorm1(x, w, eps):
    """The (1 + w) RMSNorm with one rounding."""
    ss = sum(v * v for v in x)
    rstd = 1.0 / math.sqrt(ss / len(x) + eps)
    return [bf16_round(v * rstd * (1.0 + wi)) for v, wi in zip(x, w)]


def group_norm(x, w, hc, H, eps):
    out = []
    for i in range(hc):
        out.extend(rmsnorm1(x[i * H:(i + 1) * H], w[i * H:(i + 1) * H], eps))
    return out


def gr_mix(R, site, hc, H, eps):
    """Returns (x [H], Rn [hc*H])."""
    Rn = group_norm(R, site["norm"], hc, H, eps)
    t = gemv(Rn, site["down"])
    t = [bf16_round(silu(v / hc)) for v in t]
    logits = gemv(t, site["up"])
    G = [bf16_round(sigmoid(v)) for v in logits]
    x = []
    for j in range(H):
        acc = 0.0
        for i in range(hc):
            acc += bf16_round(G[i * H + j] * Rn[i * H + j])
        x.append(bf16_round(acc / hc))
    return x, Rn


def gr_combine(R, Rn, y, site, hc, H):
    dots = gemv(Rn, site["inject"])
    for i in range(hc):
        s = 2.0 * bf16_round(sigmoid(dots[i] / hc))
        base = i * H
        for j in range(H):
            R[base + j] = bf16_round(R[base + j] + bf16_round(y[j] * s))


def gdn_forward(x_rows, w, cfg, T):
    """Returns out rows [T][H]."""
    kh, vh, K, V = cfg["gdn_kh"], cfg["gdn_vh"], cfg["gdn_kd"], cfg["gdn_vd"]
    CW = cfg["gdn_conv"]
    C = 2 * kh * K + vh * V
    kv_ratio = vh // kh
    scale = f32(K ** -0.5)
    qkv = [gemv(x, w["qkv"]) for x in x_rows]
    # The causal depthwise conv (state zero) + silu; the KDA kernel's order:
    # w[CW-1]*x first, then the history oldest-first with fma.
    hist = [[0.0] * (CW - 1) for _ in range(C)]
    conv = []
    for t in range(T):
        row = []
        for c in range(C):
            wc = w["conv"][c]
            xv = qkv[t][c]
            acc = wc[CW - 1] * xv
            for j in range(CW - 1):
                acc += wc[j] * hist[c][j]
            row.append(bf16_round(silu(acc)))
            hist[c] = hist[c][1:] + [xv]
        conv.append(row)
    z = [gemv(x, w["z"]) for x in x_rows]
    a = [gemv(x, w["a"]) for x in x_rows]
    b = [gemv(x, w["b"]) for x in x_rows]
    core = [[0.0] * (vh * V) for _ in range(T)]
    for h in range(vh):
        hk = h // kv_ratio
        A = math.exp(w["a_log"][h])
        S = [[0.0] * K for _ in range(V)]  # [V][K]
        for t in range(T):
            q = conv[t][hk * K:(hk + 1) * K]
            k = conv[t][kh * K + hk * K:kh * K + (hk + 1) * K]
            v = conv[t][2 * kh * K + h * V:2 * kh * K + (h + 1) * V]
            qn = 1.0 / math.sqrt(sum(u * u for u in q) + 1e-6)
            kn = 1.0 / math.sqrt(sum(u * u for u in k) + 1e-6)
            kq = [u * kn for u in k]
            qq = [u * qn * scale for u in q]
            decay = math.exp(-(A * softplus(a[t][h] + w["dt_bias"][h])))
            beta = sigmoid(b[t][h])
            for vv in range(V):
                srow = S[vv]
                d = dot(srow, kq) * decay
                u = (v[vv] - d) * beta
                srow = [s * decay + u * kk for s, kk in zip(srow, kq)]
                S[vv] = srow
                core[t][h * V + vv] = bf16_round(dot(srow, qq))
    out = []
    for t in range(T):
        normed = []
        for h in range(vh):
            xs = core[t][h * V:(h + 1) * V]
            zs = z[t][h * V:(h + 1) * V]
            ss = sum(u * u for u in xs)
            rstd = 1.0 / math.sqrt(ss / V + cfg["eps"])
            for i in range(V):
                uu = bf16_round(xs[i] * rstd)
                p = bf16_round(uu * w["norm"][i])
                normed.append(bf16_round(p * sigmoid(zs[i])))
        out.append(gemv(normed, w["out"]))
    return out


def rope_tables(cfg):
    R = cfg["rotary"]
    inv = [f32(1.0 / (f32(cfg["theta"]) ** f32(2 * i / R))) for i in range(R // 2)]
    return inv


def norm_rope(x, w, pos, inv, R, eps):
    xn = rmsnorm1(x, w, eps)
    half = R // 2
    out = list(xn)
    for d in range(R):
        i = d if d < half else d - half
        ang = f32(f32(float(pos)) * inv[i])
        c = bf16_round(f32(math.cos(ang)))
        s = bf16_round(f32(math.sin(ang)))
        rot = -xn[d + half] if d < half else xn[d - half]
        out[d] = bf16_round(bf16_round(xn[d] * c) + bf16_round(rot * s))
    return out


def qsa_forward(x_rows, w, cfg, T):
    H, D, R = cfg["hidden"], cfg["head_dim"], cfg["rotary"]
    lh, lkv = cfg["heads"], cfg["kv_heads"]
    nH, Di, kpool = cfg["idx_heads"], cfg["idx_dim"], cfg["kpool"]
    select_k = cfg["budget"] // kpool
    inv = rope_tables(cfg)
    eps = cfg["eps"]
    scale = f32(D ** -0.5)
    hpk = lh // lkv
    q = [gemv(x, w["q"]) for x in x_rows]
    k = [gemv(x, w["k"]) for x in x_rows]
    v = [gemv(x, w["v"]) for x in x_rows]
    idx = [gemv(x, w["idx"]) for x in x_rows]
    qn = [[norm_rope(q[t][h * 2 * D:h * 2 * D + D], w["q_norm"], t, inv, R, eps) for h in range(lh)] for t in range(T)]
    kn = [[norm_rope(k[t][h * D:(h + 1) * D], w["k_norm"], t, inv, R, eps) for h in range(lkv)] for t in range(T)]
    qi = [[norm_rope(idx[t][h * Di:(h + 1) * Di], w["idx_q_norm"], t, inv, R, eps) for h in range(nH)] for t in range(T)]
    raw = [idx[t][nH * Di:(nH + 1) * Di] for t in range(T)]
    # Compressed keys of the complete blocks.
    comp = []
    for b in range(T // kpool):
        mean = []
        for d in range(Di):
            acc = 0.0
            for s in range(kpool):
                acc += raw[b * kpool + s][d]
            mean.append(bf16_round(acc / kpool))
        comp.append(norm_rope(mean, w["idx_k_norm"], b * kpool, inv, R, eps))
    out = []
    for t in range(T):
        visible = (t + 1) // kpool
        scores = []
        for b in range(visible):
            s = 0.0
            for h in range(nH):
                s += max(dot(qi[t][h], comp[b]), 0.0)
            scores.append(s / math.sqrt(Di))
        order = sorted(range(visible), key=lambda i: (-scores[i], i))[:select_k]
        sel = sorted(order)
        toks = [b * kpool + s for b in sel for s in range(kpool)]
        toks.extend(range((t + 1) // kpool * kpool, t + 1))
        o = []
        for h in range(lh):
            kvh = h // hpk
            qh = qn[t][h]
            s = [dot(qh, kn[j][kvh]) * scale for j in toks]
            m = max(s)
            e = [math.exp(v_ - m) for v_ in s]
            l = sum(e)
            c = [0.0] * D
            for j, pj in zip(toks, e):
                p = bf16_round(pj)
                vrow = v[j][kvh * D:(kvh + 1) * D]
                for d in range(D):
                    c[d] += p * vrow[d]
            gate = q[t][h * 2 * D + D:(h + 1) * 2 * D]
            for d in range(D):
                oo = bf16_round(c[d] / l)
                o.append(bf16_round(oo * bf16_round(sigmoid(gate[d]))))
        out.append(gemv(o, w["o"]))
    return out


def moe_forward(x_rows, w, cfg):
    H, E, K = cfg["hidden"], cfg["experts"], cfg["top_k"]
    out, routes = [], []
    sg = w["sg"]
    for x in x_rows:
        logits = gemv(x, w["gate"])
        m = max(logits)
        ex = [math.exp(v - m) for v in logits]
        z = sum(ex)
        p = [v / z for v in ex]
        sel = sorted(sorted(range(E), key=lambda i: (-logits[i], i))[:K])
        denom = sum(p[i] for i in sel) if cfg["norm_topk"] else 1.0
        weights = [bf16_round(p[i] / denom) for i in sel]
        acc = [0.0] * H
        for e, we in zip(sel, weights):
            g_, u_, d_ = w["experts"][e]
            g = gemv(x, g_)
            u = gemv(x, u_)
            act = [bf16_round(bf16_round(silu(gi)) * ui) for gi, ui in zip(g, u)]
            for j in range(H):
                acc[j] += we * dot(act, d_[j])
        g = gemv(x, w["s_gate"])
        u = gemv(x, w["s_up"])
        act = [bf16_round(bf16_round(silu(gi)) * ui) for gi, ui in zip(g, u)]
        ws = bf16_round(sigmoid(bf16_round(dot(x, sg))))
        for j in range(H):
            acc[j] += ws * dot(act, w["s_down"][j])
        out.append([bf16_round(v) for v in acc])
        routes.append(sel)
    return out, routes


def ple_forward(R_rows, tokens, w, cfg, T):
    """R_rows [T][hc*H] updated in place."""
    H, hc = cfg["hidden"], cfg["hc"]
    W = hc * H
    eos = cfg["eos"]
    heads, hd, hpn = w["heads"], w["hd"], w["heads_per_ngram"]
    E = heads * hd
    mult = w["mult"]
    hist = [[0.0] * 9 for _ in range(W)]
    for t in range(T):
        y0 = tokens[t]
        prev1 = tokens[t - 1] if t >= 1 else eos
        prev2 = tokens[t - 2] if t >= 2 else eos
        y1 = prev1
        y2 = eos if prev1 == eos else prev2
        m0, m1, m2 = y0 * mult[0], y1 * mult[1], y2 * mult[2]
        e = []
        for h in range(heads):
            mix = m0 ^ m1
            if h >= hpn:
                mix ^= m2
            row = mix % w["vocab"][h] + w["offset"][h]
            e.extend(bf16_round(fp8_from_bits(c) * w["scale"]) for c in w["codes"][row * hd:(row + 1) * hd])
        key = gemv(e, w["key"])
        kn = group_norm(key, w["norm_key"], hc, H, cfg["eps"])
        val = gemv(e, w["value"])
        R = R_rows[t]
        qn = group_norm(R, w["norm_query"], hc, H, cfg["eps"])
        gv = []
        for i in range(hc):
            acc = 0.0
            for d in range(H):
                acc += bf16_round(kn[i * H + d] * qn[i * H + d])
            g = bf16_round(acc)
            g = bf16_round(g / f32(math.sqrt(H)))
            sign = -1.0 if g < 0 else (1.0 if g > 0 else 0.0)
            g = bf16_round(math.sqrt(max(abs(g), 1e-6))) * sign
            s = bf16_round(sigmoid(g))
            gv.extend(bf16_round(s * vd) for vd in val)
        un = group_norm(gv, w["norm_conv"], hc, H, cfg["eps"])
        for c in range(W):
            wc = w["conv"][c]
            u = un[c]
            acc = 0.0
            for kk in range(3):
                acc += wc[kk] * hist[c][kk * 3]
            acc += wc[3] * u
            cv = bf16_round(acc)
            act = bf16_round(silu(cv))
            ple = bf16_round(gv[c] + act)
            R[c] = bf16_round(R[c] + ple)
            hist[c] = hist[c][1:] + [u]


def reference_forward(cfg, entries, tokens, progress=False):
    H, hc = cfg["hidden"], cfg["hc"]
    T = len(tokens)
    _, embed = load_bf16(entries, "model.language_model.embed_tokens.weight")
    R = [list(embed[tok * H:(tok + 1) * H]) * hc for tok in tokens]
    layer_states, routes = [], []
    for layer in range(cfg["num_layers"]):
        w = layer_weights(cfg, entries, layer)
        if progress:
            print("layer %d (%s)" % (layer, w["kind"]), file=sys.stderr, flush=True)
        if "ple" in w:
            ple_forward(R, tokens, w["ple"], cfg, T)
        mixed = [gr_mix(R[t], w["attn_gr"], hc, H, cfg["eps"]) for t in range(T)]
        x_rows = [m[0] for m in mixed]
        if w["kind"] == "linear_attention":
            y = gdn_forward(x_rows, w["gdn"], cfg, T)
        else:
            y = qsa_forward(x_rows, w["qsa"], cfg, T)
        for t in range(T):
            gr_combine(R[t], mixed[t][1], y[t], w["attn_gr"], hc, H)
        mixed = [gr_mix(R[t], w["mlp_gr"], hc, H, cfg["eps"]) for t in range(T)]
        x_rows = [m[0] for m in mixed]
        y, route = moe_forward(x_rows, w["moe"], cfg)
        for t in range(T):
            gr_combine(R[t], mixed[t][1], y[t], w["mlp_gr"], hc, H)
        layer_states.append([v for row in R for v in row])
        routes.append(route)
    mixer = gr_site(entries, "model.language_model.hyper_connection_mixer.", False)
    h_rows = [gr_mix(R[t], mixer, hc, H, cfg["eps"])[0] for t in range(T)]
    lm = load_head(entries)
    lm_rows = as_rows(lm, cfg["vocab"], H)
    logits = [[f32(dot(h, r)) for r in lm_rows] for h in h_rows]
    return layer_states, h_rows, logits, routes


def mtp_forward(cfg, entries, tokens, R_last, progress=False):
    """The draft block over the prompt (SGLang's Qwen4ExpForCausalLMMTP): row
    q embeds tokens[q + 1] and takes the main stack's hyper state after its
    last layer at position q — R_mtp[i] = bf16(fc_e(norm_H(e)) + fc_h(norm_W(R)[i]))
    per branch, the whole hyper state normalized at once (GemmaRMSNorm over
    hc*H) — then the draft layer, its mixer and the shared lm head. Returns
    (h_rows, logits) for the T-1 rows."""
    H, hc = cfg["hidden"], cfg["hc"]
    W = hc * H
    eps = cfg["eps"]
    T = len(tokens) - 1
    _, embed = load_bf16(entries, "model.language_model.embed_tokens.weight")
    w_e = load_bf16(entries, "mtp.pre_fc_norm_embedding.weight")[1]
    w_h = load_bf16(entries, "mtp.pre_fc_norm_hidden.weight")[1]
    fc_e = as_rows(load_bf16(entries, "mtp.fc_embedding.weight")[1], H, H)
    fc_h = as_rows(load_bf16(entries, "mtp.fc_hidden.weight")[1], H, H)
    R = []
    for q in range(T):
        tok = tokens[q + 1]
        ein = gemv(rmsnorm1(embed[tok * H:(tok + 1) * H], w_e, eps), fc_e)
        hn = rmsnorm1(R_last[q * W:(q + 1) * W], w_h, eps)
        row = []
        for i in range(hc):
            enc = gemv(hn[i * H:(i + 1) * H], fc_h)
            row.extend(bf16_round(a + b) for a, b in zip(ein, enc))
        R.append(row)
    w = layer_weights(cfg, entries, 0, prefix="mtp.layers.0.")
    if progress:
        print("mtp layer (%s)" % w["kind"], file=sys.stderr, flush=True)
    mixed = [gr_mix(R[t], w["attn_gr"], hc, H, eps) for t in range(T)]
    y = qsa_forward([m[0] for m in mixed], w["qsa"], cfg, T)
    for t in range(T):
        gr_combine(R[t], mixed[t][1], y[t], w["attn_gr"], hc, H)
    mixed = [gr_mix(R[t], w["mlp_gr"], hc, H, eps) for t in range(T)]
    y, _ = moe_forward([m[0] for m in mixed], w["moe"], cfg)
    for t in range(T):
        gr_combine(R[t], mixed[t][1], y[t], w["mlp_gr"], hc, H)
    mixer = gr_site(entries, "mtp.hyper_connection_mixer.", False)
    h_rows = [gr_mix(R[t], mixer, hc, H, eps)[0] for t in range(T)]
    lm = load_head(entries)
    lm_rows = as_rows(lm, cfg["vocab"], H)
    logits = [[f32(dot(h, r)) for r in lm_rows] for h in h_rows]
    return h_rows, logits


def topk_row(values, k):
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]
    return order, [values[i] for i in order]


def write_dump(path, cfg_summary, tensors):
    header = {"format": "dgpp-qwen-reference-dump", "version": VERSION, "backend": "pure",
              "config": cfg_summary, "tensors": {}}
    payload = bytearray()
    for name, (dtype, shape, blob) in tensors.items():
        header["tensors"][name] = {"dtype": dtype, "shape": list(shape), "offset": len(payload), "nbytes": len(blob)}
        payload.extend(blob)
    hb = json.dumps(header, indent=1, sort_keys=True).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(hb)))
        f.write(hb)
        f.write(bytes(payload))


def gen_pure(args):
    cfg = text_config(args.checkpoint_dir)
    entries = read_safetensors_index(args.checkpoint_dir)
    rng = make_rng(args.seed)
    tokens = [int(rng() * cfg["vocab"]) % cfg["vocab"] for _ in range(args.tokens)]
    if args.tokens > 9:
        tokens[9] = cfg["eos"]  # an EOS inside the prompt: the n-gram context resets
    layer_states, h_rows, logits, routes = reference_forward(cfg, entries, tokens, progress=True)
    topk = 8
    top_ids, top_vals = [], []
    for row in logits:
        ids, vals = topk_row(row, topk)
        top_ids.extend(ids)
        top_vals.extend(vals)
    T, H, W = args.tokens, cfg["hidden"], cfg["hc"] * cfg["hidden"]
    L, K = cfg["num_layers"], cfg["top_k"]
    flat_routes = [e for layer in routes for row in layer for e in row]
    tensors = {
        "tokens": ("I64", [T], struct.pack("<%dq" % T, *tokens)),
        "final_hidden": ("BF16", [T, H], krd.bf16_to_bytes([v for row in h_rows for v in row])),
        "topk_ids": ("I32", [T, topk], struct.pack("<%di" % (T * topk), *top_ids)),
        "topk_logits": ("F32", [T, topk], struct.pack("<%df" % (T * topk), *top_vals)),
        "layer_states": ("BF16", [L, T, W], krd.bf16_to_bytes([v for st in layer_states for v in st])),
        "route_ids": ("I32", [L, T, K], struct.pack("<%di" % len(flat_routes), *flat_routes)),
    }
    cfg_summary = {"hidden": H, "vocab": cfg["vocab"], "num_layers": L, "tokens": T, "top_k": topk, "hc": cfg["hc"]}
    if args.mtp and T >= 2:
        mh_rows, mlogits = mtp_forward(cfg, entries, tokens, layer_states[-1], progress=True)
        m_ids, m_vals = [], []
        for row in mlogits:
            ids, vals = topk_row(row, topk)
            m_ids.extend(ids)
            m_vals.extend(vals)
        tensors["mtp_final_hidden"] = ("BF16", [T - 1, H], krd.bf16_to_bytes([v for row in mh_rows for v in row]))
        tensors["mtp_topk_ids"] = ("I32", [T - 1, topk], struct.pack("<%di" % ((T - 1) * topk), *m_ids))
        tensors["mtp_topk_logits"] = ("F32", [T - 1, topk], struct.pack("<%df" % ((T - 1) * topk), *m_vals))
        cfg_summary["mtp_rows"] = T - 1
    write_dump(args.out, cfg_summary, tensors)
    print("wrote %s (%d tokens, %d layers)" % (args.out, T, L))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen-pure")
    g.add_argument("--checkpoint-dir", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--tokens", type=int, default=72)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--mtp", action=argparse.BooleanOptionalAction, default=True,
                   help="also dump the draft block's rows over the prompt (default on)")
    g.set_defaults(func=gen_pure)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
