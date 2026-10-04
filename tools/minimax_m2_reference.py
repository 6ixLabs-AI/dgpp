#!/usr/bin/env python3
"""minimax_m2_reference.py — host reference forward pass for MiniMax-M2.7 (2026-10-04), reading
the modelopt NVFP4 release lukealonso/MiniMax-M2.7-NVFP4 (config.json, the shards named by
model.safetensors.index.json).

Pure Python + numpy: safetensors headers parsed by hand, shards mmap'd, weights dequantized to
float, activations in full precision. It is the ground truth the engine port (src/models/minimax)
is compared against, so it is written for clarity and checkability, not speed. Single process:
the whole model at once (about 125 GiB of mapped shards; the routed experts are dequantized on
demand, so the resident set is the BF16 weights plus the page cache the kernel chooses to keep).

  score    --ids-json F [--out F.json]   teacher-forced log P(ids[i] | ids[<i]) + top-5 per position
  generate --ids-json F --max-new N      greedy continuation, prints the new token ids
  dump     --ids-json F --out DIR        .npy of the residual stream after every layer (+ sublayers)
  all modes: --ckpt DIR, --layers N (first N layers only), --threads N, --dtype float32|float64,
             --pos0 P, --chunk N, --variant key=value

THE MODEL (what this file computes; names are checkpoint tensor names)

  x = model.embed_tokens[ids]                                 no scaling
  per layer L:  x += attention(rms(x, input_layernorm), pos)
                x += moe(rms(x, post_attention_layernorm))
  logits = lm_head @ rms(x, model.norm)                       lm_head is untied

  rms(x, w) = x * rsqrt(mean(x^2) + eps) * w                  eps = rms_norm_eps = 1e-6, plain weight

  Attention — grouped-query, 48 query heads over 8 K/V heads, head_dim 128, no biases:
    q = rms(q_proj @ h, q_norm)      over ALL 6144 outputs at once ("qk_norm_type": "per_layer":
    k = rms(k_proj @ h, k_norm)      one RMS over every head's elements, a weight per element —
                                     not GLM-4.7's per-head norm), then split into heads
    v = v_proj @ h                   -> [8, 128]
    RoPE on dims 0..63 of each q and k head (rotary_dim 64 = head_dim * partial_rotary_factor):
        pairs (i, i + 32), angle = pos * theta^(-i/32), theta = rope_theta = 5e6
    causal softmax attention, scale 1/sqrt(128); query head h reads K/V head h // 6
    out = o_proj @ concat_h(...)     [48 * 128] -> 3072

  MoE (256 experts, top-8, no shared expert — "shared_intermediate_size": 0):
    s = sigmoid(block_sparse_moe.gate @ h)                      over all 256
    the 8 largest of s + e_score_correction_bias (ties: the lower expert id)
    w = s[top] / sum(s[top])                                    the UNBIASED scores, renormalized
    y = sum_e w_e * w2_e @ (silu(w1_e @ h) * (w3_e @ h))
    (w1 = gate projection, w3 = up projection, w2 = down projection)

  NVFP4 (modelopt; every w1 / w2 / w3 of the routed experts): `.weight` U8 [N, K/2], two E2M1
    codes per byte, LOW nibble = EVEN column; `.weight_scale` F8_E4M3 [N, K/16]; `.weight_scale_2`
    F32 scalar, a MULTIPLIER:
        W[n, k] = e2m1(code[n, k]) * e4m3(weight_scale[n, k // 16]) * weight_scale_2
    `.input_scale` (F32 scalars, in model-inputscales.safetensors, absent for six experts the
    calibration never routed to) is the recipe's static W4A4 activation scale. This file — like
    the engine, and like the model card's own launch line (B12X_MOE_FORCE_A16=1) — runs the expert
    matrices on full-precision activations and never reads those tensors.

WHAT THE CHECKPOINT DOES NOT HAVE
  config.json says `use_mtp: true`, `num_mtp_modules: 3`, `mtp_transformer_layers: 1`. Neither
  this release nor MiniMaxAI/MiniMax-M2.7 ships an MTP tensor, and the published modeling code
  (modeling_minimax_m2.py, transformers' minimax_m2) defines no such module. There is no draft
  head to compute. Likewise config.json's bos_token_id 1 / eos_token_id 2 are not the tokenizer's
  control ids (generation_config.json: bos 200019, eos 200020).

  The repository also holds 18 safetensors files that are NOT the model (calibration by-products:
  ten model-*inputscales*.safetensors variants that repeat the input_scale names, eight
  amax*.safetensors under other names) and three *.bak copies. Only the files
  model.safetensors.index.json names are read; a directory glob would bind duplicates.

Arithmetic is float32 by default (the softmax and the final log-softmax in float64);
`--dtype float64` runs everything in float64 (a noise-floor check).
"""
import os
import sys


def _early_threads(default=8):
    n = default
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
        elif a.startswith("--threads="):
            n = int(a.split("=", 1)[1])
    return max(1, n)


_THREADS = _early_threads()
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(_THREADS))

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import mmap  # noqa: E402
import resource  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

DEFAULT_CKPT = ("/home/mark/.cache/huggingface/hub/models--lukealonso--MiniMax-M2.7-NVFP4/"
                "snapshots/db821d7a3ce29ee96d80a1cae88d878d8586b54e")

assert sys.byteorder == "little", "the BF16 / NVFP4 decoders assume a little-endian host"

# ----------------------------------------------------------------------------------------------
# Debug variants: every convention that could be read two ways, flippable in isolation with
# `--variant key=value` (repeatable). The defaults are this file's reading; a variant exists to
# PROVE a convention on the real checkpoint (the wrong value must cost likelihood), not to be used.
# ----------------------------------------------------------------------------------------------
VARIANT_DEFAULTS = {
    "nibble": "lo",       # lo: low nibble = even column | hi: high nibble = even column
    "ws2": "mul",         # mul: W = e2m1 * e4m3 * weight_scale_2 | div: ... / weight_scale_2
    "rope": "half",       # half: pairs (i, i + 32) | interleave: pairs (2i, 2i + 1)
    "qknorm": "layer",    # layer: one RMS over all heads | head: an RMS per head (GLM-4.7's form)
    "bias": "on",         # on: e_score_correction_bias joins the selection key | off
    "w13": "gate_up",     # gate_up: silu(w1 h) * (w3 h) | up_gate: silu(w3 h) * (w1 h)
}

_ST_DTYPES = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "F64": "<f8", "U8": "u1", "I8": "i1",
              "F8_E4M3": "u1", "I32": "<i4", "I64": "<i8", "BOOL": "u1"}


def _e2m1_table():
    mag = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
    return np.concatenate([mag, -mag]).astype(np.float32)


def _e4m3_table():
    t = np.zeros(256, np.float64)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 0x7
        if e == 0:
            v = (m / 8.0) * 2.0 ** -6
        elif e == 15 and m == 7:
            v = float("nan")
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = s * v
    return t.astype(np.float32)


_E2M1 = _e2m1_table()
_E4M3 = _e4m3_table()


def bf16_to_f32(bits):
    return (np.ascontiguousarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def dequant_nvfp4(packed, scale, ws2, nibble="lo", mode="mul"):
    """packed U8 [N, K/2], scale e4m3 bits U8 [N, K/16], ws2 float -> float32 [N, K]."""
    n, half = packed.shape
    k = half * 2
    codes = np.empty((n, k), np.uint8)
    lo, hi = packed & 0xF, packed >> 4
    if nibble == "lo":
        codes[:, 0::2], codes[:, 1::2] = lo, hi
    else:
        codes[:, 0::2], codes[:, 1::2] = hi, lo
    s = _E4M3[scale]
    s = (s * np.float32(ws2)) if mode == "mul" else (s / np.float32(ws2))
    w = _E2M1[codes].reshape(n, k // 16, 16) * s[:, :, None]
    return w.reshape(n, k).astype(np.float32)


class Checkpoint:
    """name -> raw tensor view. A safetensors file is: u64 little-endian header length N, N bytes
    of JSON, then the data; offsets are relative to byte 8 + N. The shard list is
    model.safetensors.index.json's — the release's directory holds other safetensors files that
    repeat tensor names (see the module docstring); without an index, every *.safetensors."""

    def __init__(self, ckpt_dir):
        self.dir = ckpt_dir
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.maps, self.index = [], {}
        idx = os.path.join(ckpt_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                names = sorted(set(json.load(f)["weight_map"].values()))
            shards = [os.path.join(ckpt_dir, n) for n in names]
        else:
            shards = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"no .safetensors under {ckpt_dir}")
        self.shards = [os.path.basename(s) for s in shards]
        for path in shards:
            with open(path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n))
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            base = 8 + n
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                if name in self.index:
                    raise ValueError(f"tensor {name} appears in two shards")
                b, e = meta["data_offsets"]
                self.index[name] = (len(self.maps), meta["dtype"], tuple(meta["shape"]), base + b, base + e)
            self.maps.append(mm)

    def __contains__(self, name):
        return name in self.index

    def raw(self, name):
        shard, dt, shape, b, e = self.index[name]
        npdt = np.dtype(_ST_DTYPES[dt])
        a = np.frombuffer(self.maps[shard], dtype=npdt, count=(e - b) // npdt.itemsize, offset=b)
        return a.reshape(shape)

    def dtype(self, name):
        return self.index[name][1]

    def release_pages(self):
        for mm in self.maps:
            try:
                mm.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError, ValueError):
                pass


class State:
    """Per-sequence K/V cache, so a sequence can be fed in chunks (or token by token)."""

    def __init__(self, n_layers, pos0=0):
        self.pos = pos0
        self.layers = [None] * n_layers   # {"k": [n, kv_heads, d], "v": [n, kv_heads, d]}


def silu(x):
    with np.errstate(over="ignore"):
        return x / (1.0 + np.exp(-x))


class Model:
    def __init__(self, ckpt_dir, layers=None, dtype="float32", variants=None):
        self.ck = Checkpoint(ckpt_dir)
        c = self.ck.cfg
        self.dt = np.dtype(dtype).type
        self.var = dict(VARIANT_DEFAULTS)
        self.var.update(variants or {})
        for k in self.var:
            if k not in VARIANT_DEFAULTS:
                raise ValueError(f"unknown variant {k}")
        self.H = c["hidden_size"]
        self.vocab = c["vocab_size"]
        self.eps = float(c["rms_norm_eps"])
        self.total_layers = c["num_hidden_layers"]
        self.n_layers = self.total_layers if layers is None else min(layers, self.total_layers)
        self.heads, self.kv_heads = c["num_attention_heads"], c["num_key_value_heads"]
        self.d = c.get("head_dim") or self.H // self.heads
        rp = c.get("rope_parameters") or {}
        if rp.get("rope_type", "default") != "default":
            raise ValueError("only the default rope is implemented")
        self.theta = float(rp.get("rope_theta", c.get("rope_theta")))
        factor = float(rp.get("partial_rotary_factor", c.get("partial_rotary_factor", 1.0)))
        self.rot = int(self.d * factor)
        if c.get("rotary_dim", self.rot) != self.rot:
            raise ValueError("rotary_dim disagrees with head_dim * partial_rotary_factor")
        if not c.get("use_qk_norm", False) or c.get("qk_norm_type", "per_layer") != "per_layer":
            raise ValueError("only the per-layer q/k norm is implemented")
        if c.get("scoring_func", "sigmoid") != "sigmoid" or not c.get("use_routing_bias", True):
            raise ValueError("only the sigmoid router with its correction bias is implemented")
        if c.get("shared_intermediate_size", 0) != 0:
            raise ValueError("a shared expert is not implemented (the release has none)")
        if c.get("sliding_window") is not None or any(t != 1 for t in c.get("attn_type_list", [])):
            raise ValueError("only full attention on every layer is implemented")
        if c.get("tie_word_embeddings", False):
            raise ValueError("tied embeddings are not implemented")
        self.n_experts, self.top_k = c["num_local_experts"], c["num_experts_per_tok"]
        self.inter = c["intermediate_size"]
        # torch: base ** (arange(0, rot, 2, float32) / rot), then 1 / it, in fp32.
        e = np.arange(0, self.rot, 2, dtype=np.float32) / np.float32(self.rot)
        self.inv_freq = (np.float32(1.0) / (np.float32(self.theta) ** e).astype(np.float32)).astype(np.float32)
        self.scale = self.d ** -0.5
        gen = os.path.join(ckpt_dir, "generation_config.json")
        self.eos = None
        if os.path.exists(gen):
            with open(gen) as f:
                self.eos = json.load(f).get("eos_token_id")
        self._cache = {}
        self.embed_bits = self.ck.raw("model.embed_tokens.weight")
        self.final_norm = self._bf16("model.norm.weight")
        self.head = None

    # ---- weights ---------------------------------------------------------------------------
    def _bf16(self, name):
        if self.ck.dtype(name) != "BF16":
            raise ValueError(f"{name}: expected BF16, the checkpoint stores {self.ck.dtype(name)}")
        return bf16_to_f32(self.ck.raw(name)).astype(self.dt)

    def _fp4(self, prefix):
        ws2 = float(np.asarray(self.ck.raw(prefix + ".weight_scale_2"), np.float32).reshape(()))
        w = dequant_nvfp4(self.ck.raw(prefix + ".weight"), self.ck.raw(prefix + ".weight_scale"), ws2,
                          nibble=self.var["nibble"], mode=self.var["ws2"])
        return w.astype(self.dt)

    def layer_weights(self, l):
        if l in self._cache:
            return self._cache[l]
        p = f"model.layers.{l}."
        w = {k: self._bf16(p + n) for k, n in (
            ("in_norm", "input_layernorm.weight"), ("post_norm", "post_attention_layernorm.weight"),
            ("q", "self_attn.q_proj.weight"), ("k", "self_attn.k_proj.weight"),
            ("v", "self_attn.v_proj.weight"), ("o", "self_attn.o_proj.weight"),
            ("q_norm", "self_attn.q_norm.weight"), ("k_norm", "self_attn.k_norm.weight"),
            ("gate", "block_sparse_moe.gate.weight"), ("bias", "block_sparse_moe.e_score_correction_bias"))}
        self._cache[l] = w
        return w

    def expert(self, l, e):
        p = f"model.layers.{l}.block_sparse_moe.experts.{e}."
        return tuple(self._fp4(p + m) for m in ("w1", "w2", "w3"))

    # ---- blocks ----------------------------------------------------------------------------
    def rms(self, x, w):
        x64 = x.astype(np.float64)
        r = 1.0 / np.sqrt((x64 * x64).mean(axis=-1, keepdims=True) + self.eps)
        return (x64 * r).astype(self.dt) * w

    def qk_norm(self, x, w, heads):
        """x [T, heads * d]: the per-layer norm (one RMS over the row) or, as a variant, per head."""
        if self.var["qknorm"] == "layer":
            return self.rms(x, w)
        T = x.shape[0]
        return self.rms(x.reshape(T, heads, self.d), w.reshape(heads, self.d)).reshape(T, heads * self.d)

    def rope(self, x, pos):
        """x [T, heads, d]: rotates the first `rot` dims of every head."""
        ang = pos.astype(np.float32)[:, None] * self.inv_freq[None, :]          # fp32 product, as torch
        c = np.cos(ang.astype(np.float64)).astype(np.float32).astype(self.dt)[:, None, :]
        s = np.sin(ang.astype(np.float64)).astype(np.float32).astype(self.dt)[:, None, :]
        out = x.copy()
        h = self.rot // 2
        if self.var["rope"] == "half":
            x0, x1 = x[..., :h], x[..., h:self.rot]
            out[..., :h] = x0 * c - x1 * s
            out[..., h:self.rot] = x1 * c + x0 * s
        else:
            x0, x1 = x[..., 0:self.rot:2], x[..., 1:self.rot:2]
            out[..., 0:self.rot:2] = x0 * c - x1 * s
            out[..., 1:self.rot:2] = x1 * c + x0 * s
        return out

    def attention(self, l, h, pos, st, tap=None):
        w = self.layer_weights(l)
        T, nh, nk, d = h.shape[0], self.heads, self.kv_heads, self.d
        q = self.qk_norm(h @ w["q"].T, w["q_norm"], nh).reshape(T, nh, d)
        k = self.qk_norm(h @ w["k"].T, w["k_norm"], nk).reshape(T, nk, d)
        v = (h @ w["v"].T).reshape(T, nk, d)
        q, k = self.rope(q, pos), self.rope(k, pos)
        cache = st.layers[l]
        if cache is None:
            k_all, v_all = k, v
        else:
            k_all = np.concatenate([cache["k"], k]); v_all = np.concatenate([cache["v"], v])
        st.layers[l] = {"k": k_all, "v": v_all}
        n = k_all.shape[0]
        rep = nh // nk
        qg = q.reshape(T, nk, rep, d)
        scores = np.einsum("tgrd,sgd->grts", qg, k_all).astype(np.float64) * self.scale
        first = n - T
        mask = np.arange(n)[None, :] > (first + np.arange(T))[:, None]
        scores = np.where(mask[None, None], -np.inf, scores)
        scores -= scores.max(axis=-1, keepdims=True)
        pr = np.exp(scores); pr /= pr.sum(axis=-1, keepdims=True)
        o = np.einsum("grts,sgd->tgrd", pr.astype(self.dt), v_all).reshape(T, nh, d)
        out = o.reshape(T, nh * d) @ w["o"].T
        if tap is not None:
            tap(f"layer{l}.q", q); tap(f"layer{l}.k", k); tap(f"layer{l}.v", v); tap(f"layer{l}.attn_heads", o)
        return out

    def route(self, l, h):
        """-> (top ids [T, k] by rank, weights [T, k]) — sigmoid scores, biased selection."""
        w = self.layer_weights(l)
        logits = (h @ w["gate"].T).astype(np.float64)
        with np.errstate(over="ignore"):
            s = 1.0 / (1.0 + np.exp(-logits))
        key = s + (w["bias"].astype(np.float64) if self.var["bias"] == "on" else 0.0)
        order = np.argsort(-key, axis=-1, kind="stable")[:, :self.top_k]             # ties: lower id first
        wt = np.take_along_axis(s, order, axis=-1)
        wt = wt / wt.sum(axis=-1, keepdims=True)
        return order, wt.astype(self.dt)

    def mlp(self, ws, x):
        w1, w2, w3 = ws
        if self.var["w13"] == "up_gate":
            w1, w3 = w3, w1
        return (silu(x @ w1.T) * (x @ w3.T)) @ w2.T

    def moe(self, l, h, tap=None):
        ids, wt = self.route(l, h)
        y = np.zeros_like(h)
        for e in np.unique(ids):
            rows, slot = np.nonzero(ids == e)
            y[rows] += self.mlp(self.expert(l, int(e)), h[rows]) * wt[rows, slot][:, None]
        if tap is not None:
            tap(f"layer{l}.route_ids", ids); tap(f"layer{l}.route_w", wt)
        return y

    # ---- the walk --------------------------------------------------------------------------
    def new_state(self, pos0=0):
        return State(self.n_layers, pos0)

    def forward(self, ids, st, on_layer=None, tap=None, keep_weights=True):
        """ids -> the final-normed hidden rows [T, H] (pass them to logprobs)."""
        ids = np.asarray(ids, np.int64)
        pos = st.pos + np.arange(ids.size, dtype=np.int64)
        st.pos += ids.size
        x = bf16_to_f32(self.embed_bits[ids]).astype(self.dt)
        if tap is not None:
            tap("embed", x)
        for l in range(self.n_layers):
            w = self.layer_weights(l)
            a = self.attention(l, self.rms(x, w["in_norm"]), pos, st, tap)
            x = x + a
            if tap is not None:
                tap(f"layer{l}.attn", a); tap(f"layer{l}.mid", x)
            m = self.moe(l, self.rms(x, w["post_norm"]), tap)
            x = x + m
            if tap is not None:
                tap(f"layer{l}.moe", m); tap(f"layer{l}.out", x)
            if not keep_weights:
                self._cache.pop(l, None)
            self.ck.release_pages()
            if on_layer is not None:
                on_layer(l + 1)
        return self.rms(x, self.final_norm)

    def logits(self, xn):
        if self.head is None:
            self.head = self._bf16("lm_head.weight")
        return xn @ self.head.T

    def logprobs(self, xn):
        z = self.logits(xn).astype(np.float64)
        z -= z.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


# ----------------------------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------------------------
def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 30) if sys.platform == "darwin" else r / (1 << 20)


def load_ids(path):
    with open(path) as f:
        d = json.load(f)
    ids = d["ids"] if isinstance(d, dict) else d
    if not ids:
        raise ValueError("empty id list")
    return [int(i) for i in ids]


def top_desc(lp, k):
    idx = np.argpartition(-lp, min(k, lp.size - 1))[:k]
    return idx[np.lexsort((idx, -lp[idx]))]


def score_ids(model, ids, chunk=256, pos0=0, on_layer=None, top=5):
    """Teacher-forced: entry i is log P(ids[i] | ids[<i]) (entry 0 is null), `top` the row's top-k
    [(id, logprob)], `next_top` the prediction after the last token — the schema
    sixlabs/bench/compare_forward_check.py reads."""
    t0 = time.time()
    st = model.new_state(pos0)
    lps, ranks, tops = [None], [None], [None]
    last = None
    for b in range(0, len(ids), chunk):
        part = ids[b:b + chunk]
        lp = model.logprobs(model.forward(part, st, on_layer=on_layer if b == 0 else None))
        for j in range(len(part)):
            i = b + j + 1
            row = lp[j]
            if i < len(ids):
                lps.append(float(row[ids[i]]))
                ranks.append(int((row > row[ids[i]]).sum()))
                tops.append([[int(t), float(row[t])] for t in top_desc(row, top)])
            else:
                last = [[int(t), float(row[t])] for t in top_desc(row, top)]
    return {"ids": ids, "logprobs": lps, "rank": ranks, "top": tops, "next_top": last,
            "layers": model.n_layers, "pos0": pos0, "variants": model.var,
            "seconds": round(time.time() - t0, 2), "peak_rss_gb": round(peak_rss_gb(), 2)}


def greedy(model, ids, max_new, pos0=0, stop_eos=True, top=5):
    st = model.new_state(pos0)
    x = model.forward(ids, st)[-1:]
    new, steps = [], []
    for step in range(max_new):
        lp = model.logprobs(x)[0]
        t = top_desc(lp, top)
        tok = int(t[0])
        new.append(tok)
        steps.append({"id": tok, "top": [[int(i), float(lp[i])] for i in t], "margin": float(lp[t[0]] - lp[t[1]])})
        if stop_eos and model.eos is not None and tok == model.eos:
            break
        if step + 1 < max_new:
            x = model.forward([tok], st)
    return new, steps


def make_model(a):
    variants = {}
    for v in a.variant or []:
        k, _, val = v.partition("=")
        variants[k] = val
    return Model(a.ckpt, layers=a.layers, dtype=a.dtype, variants=variants)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)
        f.write("\n")


def cmd_score(a):
    ids = load_ids(a.ids_json)
    model = make_model(a)
    res = score_ids(model, ids, chunk=a.chunk, pos0=a.pos0,
                    on_layer=lambda d: print(f"# layer {d}/{model.n_layers}", file=sys.stderr, flush=True))
    if a.out:
        write_json(a.out, res)
    for i in range(1, len(ids)):
        t = " ".join(f"{t}:{lp:.4f}" for t, lp in res["top"][i])
        print(f"pos {i:4d} id {ids[i]:6d} logprob {res['logprobs'][i]:10.5f} rank {res['rank'][i]:5d} | {t}")
    print("next | " + " ".join(f"{t}:{lp:.4f}" for t, lp in res["next_top"]))
    lps = np.array(res["logprobs"][1:], np.float64)
    if lps.size:
        print(f"# tokens={len(ids)} layers={model.n_layers} mean_logprob={lps.mean():.5f} "
              f"seconds={res['seconds']} peak_rss_gb={res['peak_rss_gb']}", flush=True)


def cmd_generate(a):
    ids = load_ids(a.ids_json)
    model = make_model(a)
    t0 = time.time()
    new, steps = greedy(model, ids, a.max_new, pos0=a.pos0, stop_eos=not a.ignore_eos)
    if a.out:
        write_json(a.out, {"prompt_ids": ids, "new_ids": new, "steps": steps, "layers": model.n_layers,
                           "seconds": round(time.time() - t0, 2), "peak_rss_gb": round(peak_rss_gb(), 2)})
    print(json.dumps(new))


def cmd_dump(a):
    ids = load_ids(a.ids_json)
    model = make_model(a)
    os.makedirs(a.out, exist_ok=True)
    files = {}

    def tap(name, arr):
        np.save(os.path.join(a.out, name + ".npy"), np.asarray(arr))
        files[name] = list(np.asarray(arr).shape)

    st = model.new_state(a.pos0)
    xn = model.forward(ids, st, tap=tap)
    tap("final_norm", xn)
    tap("logprobs_last", model.logprobs(xn[-1:])[0])
    write_json(os.path.join(a.out, "manifest.json"), {"ids": ids, "pos0": a.pos0, "layers": model.n_layers,
                                                       "variants": model.var, "files": files})
    print(f"wrote {len(files)} arrays to {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--ckpt", default=DEFAULT_CKPT)
        p.add_argument("--ids-json", required=True)
        p.add_argument("--layers", type=int, default=None)
        p.add_argument("--threads", type=int, default=_THREADS)
        p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
        p.add_argument("--pos0", type=int, default=0)
        p.add_argument("--chunk", type=int, default=256)
        p.add_argument("--variant", action="append")
        p.add_argument("--out", default=None)

    p = sub.add_parser("score"); common(p); p.set_defaults(fn=cmd_score)
    p = sub.add_parser("generate"); common(p); p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--ignore-eos", action="store_true"); p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("dump"); common(p); p.set_defaults(fn=cmd_dump)
    a = ap.parse_args()
    if a.cmd == "dump" and not a.out:
        ap.error("dump needs --out DIR")
    a.fn(a)


if __name__ == "__main__":
    main()
