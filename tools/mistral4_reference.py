#!/usr/bin/env python3
"""mistral4_reference.py — host reference forward pass for Mistral-Small-4 (2026-10-04), reading
the Mistral-native NVFP4 release mistralai/Mistral-Small-4-119B-2603-NVFP4 (`params.json` and the
`consolidated-*.safetensors` shards; there is no config.json in that repository).

Pure Python + numpy: safetensors headers parsed by hand, shards mmap'd, weights dequantized to
float, activations in full precision. It is the ground truth the engine port (src/models/mistral4)
is compared against, so it is written for clarity and checkability, not speed. Text only: the
Pixtral vision tower and its projector (`vision_encoder.*`, `vision_language_adapter.*`,
`patch_merger.*`, `pre_mm_projector_norm.*`) are present in the checkpoint and never read.

  score    --ids-json F [--out F.json]   teacher-forced log P(ids[i] | ids[<i]) + top-5 per position
  generate --ids-json F --max-new N      greedy continuation, prints the new token ids
  dump     --ids-json F --out DIR        .npy of the residual stream after every layer (+ sublayers)
  all modes: --ckpt DIR, --layers N (first N layers only), --threads N, --dtype float32|float64,
             --pos0 P (the first token's position; default 0), --chunk N, --variant key=value

THE MODEL (what this file computes; names are checkpoint tensor names, [..] params.json keys)

  x = tok_embeddings[ids]                                     no scaling
  per layer L:  x += attention(rms(x, attention_norm), pos)
                x += moe(rms(x, ffn_norm))
  logits = output @ rms(x, norm)                              `output` is untied [tied_embeddings false]

  rms(x, w) = x * rsqrt(mean(x^2) + eps) * w                  eps = norm_eps = 1e-6, plain weight

  Attention — multi-head latent attention (DeepSeek-V2's MLA), 32 heads [n_heads]:
    q    = wq_b @ rms(wq_a @ h, q_a_norm)        -> [32, 128] per head = [nope 64 | rope 64]
    ckv  = wkv_a_with_mqa @ h                    -> [320] = [latent 256 | rope key 64]
    c    = rms(ckv[:256], kv_a_norm)             the cached latent (one per token, all heads)
    kr   = rope(ckv[256:], pos)                  the cached rope key (one per token, all heads)
    wkv_b @ c                                    -> [32, 192] per head = [k_nope 64 | v 128]
    score[h, t, s] = (q_nope[t,h] . k_nope[s,h] + rope(q_rope[t,h], pos_t) . kr[s])
                     * l4(pos_t) / sqrt(128)                           s <= t (causal)
    out  = wo @ concat_h(softmax_s(score) @ v[s, h])                   [32 * 128] -> 4096
    rope: INTERLEAVED pairs (x[2i], x[2i+1]) rotated by angle pos * inv_freq[i]   (i < 32)
          inv_freq is YaRN's blend [yarn]: base 1/theta^(2i/64), theta 10000; lanes below the
          beta_fast (32) correction dim keep their frequency, those above the beta_slow (1) dim are
          divided by factor (128), a linear ramp between; the band is computed from
          original_max_position_embeddings (8192).
    l4(pos) = 1 + beta * ln(1 + floor(pos / 8192))             [llama_4_scaling] beta 0.1; multiplies
          the whole query (nope and rope lanes) of the token AT pos
    softmax scale: 1/sqrt(nope + rope) and nothing else — see "THE YARN SCALE" below.

  MoE (128 experts [moe.num_experts], top-4 [moe.num_experts_per_tok], one shared expert):
    p = softmax(gate @ h) over all 128; the 4 largest (ties: the lower expert id);
    w = p[top] / (sum(p[top]) + 1e-20) * routed_scale (1.0)
    y = sum_e w_e * w2_e @ (silu(w1_e @ h) * (w3_e @ h))  +  shared: w2 @ (silu(w1 @ h) * (w3 @ h))
    (w1 = gate projection, w3 = up projection, w2 = down projection)

  NVFP4 (compressed-tensors `nvfp4-pack-quantized`; every w1/w2/w3 of the routed and shared experts):
    `.weight_packed` U8 [N, K/2], two E2M1 codes per byte, LOW nibble = EVEN column;
    `.weight_scale` F8_E4M3 [N, K/16]; `.weight_global_scale` F32 [1], a DIVISOR:
        W[n, k] = e2m1(code[n, k]) * (e4m3(weight_scale[n, k // 16]) / weight_global_scale)
    E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 (bit 3 = sign).
    (compressed-tensors' own reading, from its source: compressors/nvfp4/helpers.py
    unpack_fp4_from_uint8 — `low = a & 0x0F` first — and quantization/lifecycle/forward_helpers.py
    _dequantize — `scale = scale / global_scale`.)
    `.input_global_scale` (F32 [1] on routed experts, BF16 [1] on shared ones) is the recipe's
    activation scale: the release was calibrated as W4A4 (4-bit activations into each expert
    matrix, dynamic per-16 local scales under a static global one). This file — like the engine —
    runs the expert matrices on full-precision activations and never reads those tensors.

THE YARN SCALE (a disagreement between the two published implementations, 2026-10-04)
  params.json says `"yarn": {..., "apply_scale": false}`. vLLM (the model card's runtime) maps that
  to rope `attention_factor = 1` and, in deepseek_v2.py, to rope type `deepseek_llama_scaling`,
  for which the attention keeps scaling = qk_head_dim**-0.5. transformers' Mistral4Attention
  (main, since PR #47435 of 2026-07-20) multiplies the scaling by yarn_get_mscale(factor)**2 =
  (0.1 ln 128 + 1)^2 = 2.2058 whenever `mscale_all_dim` is truthy, and its conversion script
  writes `mscale_all_dim: 1.0` unconditionally; transformers up to 5.8.1 did not.
  This file follows the checkpoint's own flag and vLLM: no mscale. `--variant yarn_mscale=on`
  computes the transformers-main reading instead, so the two can be told apart on real text
  (the wrong one costs teacher-forced likelihood on any prompt).

Arithmetic is float32 by default (the softmax and the final log-softmax in float64);
`--dtype float64` runs everything in float64 (a noise-floor check).

Memory: non-expert weights are converted once and kept (about 7 GB float32 for all 36 layers);
routed experts are dequantized on demand, per layer, only for the experts the chunk routes to.
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
import math  # noqa: E402
import mmap  # noqa: E402
import resource  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

DEFAULT_CKPT = ("/home/mark/.cache/huggingface/hub/models--mistralai--Mistral-Small-4-119B-2603-NVFP4/"
                "snapshots/45331841b631f4e281df8e959ea3cc9beb84298a")

assert sys.byteorder == "little", "the BF16 / NVFP4 decoders assume a little-endian host"

# ----------------------------------------------------------------------------------------------
# Debug variants. Every convention that could be read two ways can be flipped in isolation with
# `--variant key=value` (repeatable). The defaults are this file's reading; a variant exists to
# PROVE a convention on the real checkpoint (the wrong value must cost likelihood), not to be used.
# ----------------------------------------------------------------------------------------------
VARIANT_DEFAULTS = {
    "nibble": "lo",          # lo: low nibble = even column | hi: high nibble = even column
    "gscale": "div",         # div: W = e2m1 * (e4m3 / weight_global_scale) | mul: ... * weight_global_scale
    "rope": "interleave",    # interleave: pairs (2i, 2i+1) | half: pairs (i, i + 32)
    "yarn_mscale": "off",    # off: scaling = 1/sqrt(128) | on: * (0.1 ln(factor) + 1)^2 (transformers main)
    "llama4": "on",          # on: the query is scaled by 1 + beta ln(1 + floor(pos / 8192)) | off
    "mla": "absorbed",       # absorbed: scores against the cached latent | expand: K/V expanded per head
    "split": "nope_rope",    # nope_rope: a head's q is [nope | rope] | rope_nope: [rope | nope]
}

# ----------------------------------------------------------------------------------------------
# Number formats
# ----------------------------------------------------------------------------------------------
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


def dequant_nvfp4(packed, scale, gscale, nibble="lo", mode="div"):
    """packed U8 [N, K/2], scale e4m3 bits U8 [N, K/16], gscale float -> float32 [N, K]."""
    n, half = packed.shape
    k = half * 2
    codes = np.empty((n, k), np.uint8)
    lo, hi = packed & 0xF, packed >> 4
    if nibble == "lo":
        codes[:, 0::2], codes[:, 1::2] = lo, hi
    else:
        codes[:, 0::2], codes[:, 1::2] = hi, lo
    s = _E4M3[scale]                                                   # [N, K/16] float32
    s = (s / np.float32(gscale)) if mode == "div" else (s * np.float32(gscale))
    w = _E2M1[codes].reshape(n, k // 16, 16) * s[:, :, None]
    return w.reshape(n, k).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Checkpoint access
# ----------------------------------------------------------------------------------------------
class Checkpoint:
    """name -> raw tensor view. A safetensors file is: u64 little-endian header length N, N bytes
    of JSON ({name: {dtype, shape, data_offsets: [begin, end]}}), then the data; offsets are
    relative to byte 8 + N. The shard list is `consolidated.safetensors.index.json`'s when that
    file is present (the release), every *.safetensors of the directory otherwise."""

    def __init__(self, ckpt_dir):
        self.dir = ckpt_dir
        with open(os.path.join(ckpt_dir, "params.json")) as f:
            self.params = json.load(f)
        self.maps, self.index = [], {}
        idx = os.path.join(ckpt_dir, "consolidated.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                names = sorted(set(json.load(f)["weight_map"].values()))
            shards = [os.path.join(ckpt_dir, n) for n in names]
        else:
            shards = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"no .safetensors under {ckpt_dir}")
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
        """The tensor's stored bits as a read-only numpy view on the mmap (BF16 -> uint16,
        F8_E4M3 / U8 -> uint8, F32 -> float32)."""
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


# ----------------------------------------------------------------------------------------------
# YaRN inverse frequencies (vLLM DeepseekScalingRotaryEmbedding._compute_inv_freq, transformers
# _compute_yarn_parameters: the same arithmetic, float32 as torch runs it)
# ----------------------------------------------------------------------------------------------
def yarn_inv_freq(rope_dim, theta, factor, original_max, beta_fast, beta_slow):
    f32 = np.float32
    pos_freqs = (f32(theta) ** (np.arange(0, rope_dim, 2, dtype=np.float32) / f32(rope_dim))).astype(np.float32)
    extrap = (f32(1.0) / pos_freqs).astype(np.float32)
    interp = (f32(1.0) / (f32(factor) * pos_freqs)).astype(np.float32)

    def corr_dim(rot):
        return rope_dim * math.log(original_max / (rot * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(corr_dim(beta_fast)), 0)
    high = min(math.ceil(corr_dim(beta_slow)), rope_dim - 1)
    hi = high + 0.001 if low == high else high
    ramp = np.clip((np.arange(rope_dim // 2, dtype=np.float32) - f32(low)) / f32(hi - low), 0, 1).astype(np.float32)
    mask = (f32(1.0) - ramp).astype(np.float32)                       # 1: keep the frequency
    return (interp * (f32(1.0) - mask) + extrap * mask).astype(np.float32), (low, high)


def yarn_get_mscale(scale, mscale=1.0):
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


# ----------------------------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------------------------
class State:
    """Per-sequence cache, so a sequence can be fed in chunks (or token by token)."""

    def __init__(self, n_layers, pos0=0):
        self.pos = pos0                   # the next token's position
        self.layers = [None] * n_layers   # {"c": latent [n, kv_lora], "kr": rope key [n, rope]}


def silu(x):
    with np.errstate(over="ignore"):           # exp(-x) -> inf for a very negative x: x / inf = -0
        return x / (1.0 + np.exp(-x))


class Model:
    def __init__(self, ckpt_dir, layers=None, dtype="float32", variants=None):
        self.ck = Checkpoint(ckpt_dir)
        p = self.ck.params
        self.dt = np.dtype(dtype).type
        self.var = dict(VARIANT_DEFAULTS)
        self.var.update(variants or {})
        for k, v in self.var.items():
            if k not in VARIANT_DEFAULTS:
                raise ValueError(f"unknown variant {k}")
        self.H = p["dim"]
        self.vocab = p["vocab_size"]
        self.eps = float(p["norm_eps"])
        self.total_layers = p["n_layers"]
        self.n_layers = self.total_layers if layers is None else min(layers, self.total_layers)
        self.heads = p["n_heads"]
        self.q_lora, self.kv_lora = p["q_lora_rank"], p["kv_lora_rank"]
        self.nope, self.rope, self.vdim = p["qk_nope_head_dim"], p["qk_rope_head_dim"], p["v_head_dim"]
        moe = p["moe"]
        self.n_experts, self.top_k = moe["num_experts"], moe["num_experts_per_tok"]
        self.inter = moe["expert_hidden_dim"]
        self.n_shared = moe.get("num_shared_experts", 0)
        self.routed_scale = float(moe.get("routed_scale", 1.0))
        if moe.get("first_k_dense_replace", 0) != 0 or moe.get("route_every_n", 1) != 1:
            raise ValueError("dense layers (first_k_dense_replace / route_every_n) are not implemented")
        if moe.get("num_expert_groups", 1) != 1 or moe.get("num_expert_groups_per_tok", 1) != 1:
            raise ValueError("grouped routing is not implemented")
        if self.n_shared not in (0, 1):
            raise ValueError("more than one shared expert is not implemented")
        if p.get("tied_embeddings", False):
            raise ValueError("tied embeddings are not implemented")
        y = p["yarn"]
        if y.get("apply_scale", True):
            raise ValueError("yarn.apply_scale true is not implemented (the release ships false)")
        self.theta = float(p.get("rope_theta", 10000.0))
        self.yarn_factor = float(y["factor"])
        self.inv_freq, self.yarn_band = yarn_inv_freq(self.rope, self.theta, self.yarn_factor,
                                                      int(y["original_max_position_embeddings"]),
                                                      float(y["beta"]), float(y["alpha"]))
        l4 = p.get("llama_4_scaling") or {}
        self.l4_beta = float(l4.get("beta", 0.0)) if self.var["llama4"] == "on" else 0.0
        self.l4_orig = int(l4.get("original_max_position_embeddings", 1))
        self.scale = (self.nope + self.rope) ** -0.5
        if self.var["yarn_mscale"] == "on":
            self.scale *= yarn_get_mscale(self.yarn_factor) ** 2
        self.bos, self.eos = 1, 2        # <s>, </s> of the tokenizer (params.json carries no ids)
        self._cache = {}
        self.embed_bits = self.ck.raw("tok_embeddings.weight")                 # BF16 bits, rows on demand
        self.final_norm = self._bf16("norm.weight")
        self.head = None                                                       # converted on first use

    # ---- weights ---------------------------------------------------------------------------
    def _bf16(self, name):
        if self.ck.dtype(name) != "BF16":
            raise ValueError(f"{name}: expected BF16, the checkpoint stores {self.ck.dtype(name)}")
        return bf16_to_f32(self.ck.raw(name)).astype(self.dt)

    def _fp4(self, prefix):
        g = float(np.asarray(self.ck.raw(prefix + ".weight_global_scale"), np.float32).reshape(()))
        w = dequant_nvfp4(self.ck.raw(prefix + ".weight_packed"), self.ck.raw(prefix + ".weight_scale"), g,
                          nibble=self.var["nibble"], mode=self.var["gscale"])
        return w.astype(self.dt)

    def layer_weights(self, l):
        if l in self._cache:
            return self._cache[l]
        p = f"layers.{l}."
        w = {k: self._bf16(p + n) for k, n in (
            ("attn_norm", "attention_norm.weight"), ("ffn_norm", "ffn_norm.weight"),
            ("wq_a", "attention.wq_a.weight"), ("q_a_norm", "attention.q_a_norm.weight"),
            ("wq_b", "attention.wq_b.weight"), ("wkv_a", "attention.wkv_a_with_mqa.weight"),
            ("kv_a_norm", "attention.kv_a_norm.weight"), ("wkv_b", "attention.wkv_b.weight"),
            ("wo", "attention.wo.weight"), ("gate", "gate.weight"))}
        if self.n_shared:
            w["shared"] = tuple(self._fp4(p + f"shared_experts.{m}") for m in ("w1", "w2", "w3"))
        self._cache[l] = w
        return w

    def expert(self, l, e):
        p = f"layers.{l}.experts.{e}."
        return tuple(self._fp4(p + m) for m in ("w1", "w2", "w3"))

    # ---- blocks ----------------------------------------------------------------------------
    def rms(self, x, w):
        x64 = x.astype(np.float64)
        r = 1.0 / np.sqrt((x64 * x64).mean(axis=-1, keepdims=True) + self.eps)
        return (x64 * r).astype(self.dt) * w

    def rope_rotate(self, x, pos):
        """x [..., T, rope] (T the axis before last), pos [T] -> rotated, same layout."""
        # The angle is the fp32 product and the cos / sin are fp32 values, as transformers and vLLM
        # build them (and as the engine's table does before its BF16 rounding): the trig is taken
        # in double and rounded to fp32, whatever the working dtype.
        ang = pos.astype(np.float32)[:, None] * self.inv_freq[None, :]
        c = np.cos(ang.astype(np.float64)).astype(np.float32).astype(self.dt)
        s = np.sin(ang.astype(np.float64)).astype(np.float32).astype(self.dt)
        out = np.empty_like(x)
        if self.var["rope"] == "interleave":
            x0, x1 = x[..., 0::2], x[..., 1::2]
            out[..., 0::2] = x0 * c - x1 * s
            out[..., 1::2] = x1 * c + x0 * s
        else:
            h = self.rope // 2
            x0, x1 = x[..., :h], x[..., h:]
            out[..., :h] = x0 * c - x1 * s
            out[..., h:] = x1 * c + x0 * s
        return out

    def llama4(self, pos):
        return (1.0 + self.l4_beta * np.log(1.0 + np.floor(pos.astype(np.float64) / self.l4_orig))).astype(self.dt)

    def attention(self, l, h, pos, st, tap=None):
        w = self.layer_weights(l)
        T, nh, nope, rope, vd = h.shape[0], self.heads, self.nope, self.rope, self.vdim
        q = self.rms(h @ w["wq_a"].T, w["q_a_norm"]) @ w["wq_b"].T                  # [T, nh * (nope + rope)]
        q = q.reshape(T, nh, nope + rope)
        if self.var["split"] == "nope_rope":
            q_nope, q_rope = q[..., :nope], q[..., nope:]
        else:
            q_rope, q_nope = q[..., :rope], q[..., rope:]
        ckv = h @ w["wkv_a"].T                                                     # [T, kv_lora + rope]
        c_new = self.rms(ckv[:, :self.kv_lora], w["kv_a_norm"])
        kr_new = self.rope_rotate(ckv[:, self.kv_lora:], pos)
        q_rot = self.rope_rotate(q_rope.transpose(1, 0, 2), pos).transpose(1, 0, 2)  # [T, nh, rope]
        l4 = self.llama4(pos)[:, None, None]
        q_nope, q_rot = q_nope * l4, q_rot * l4
        cache = st.layers[l]
        if cache is None:
            c_all, kr_all = c_new, kr_new
        else:
            c_all = np.concatenate([cache["c"], c_new]); kr_all = np.concatenate([cache["kr"], kr_new])
        st.layers[l] = {"c": c_all, "kr": kr_all}
        n = c_all.shape[0]
        kvb = w["wkv_b"].reshape(nh, nope + vd, self.kv_lora)                      # head-major rows
        w_uk, w_uv = kvb[:, :nope, :], kvb[:, nope:, :]
        if self.var["mla"] == "absorbed":
            q_abs = np.einsum("thd,hdc->thc", q_nope, w_uk)                         # [T, nh, kv_lora]
            scores = np.einsum("thc,sc->hts", q_abs, c_all)
        else:
            k_nope = np.einsum("sc,hdc->shd", c_all, w_uk)                          # [n, nh, nope]
            scores = np.einsum("thd,shd->hts", q_nope, k_nope)
        scores = scores + np.einsum("thr,sr->hts", q_rot, kr_all)
        scores = scores.astype(np.float64) * self.scale
        first = n - T                                                              # cache index of row 0
        mask = np.arange(n)[None, :] > (first + np.arange(T))[:, None]             # s > t: masked
        scores = np.where(mask[None], -np.inf, scores)
        scores -= scores.max(axis=-1, keepdims=True)
        pr = np.exp(scores); pr /= pr.sum(axis=-1, keepdims=True)
        pr = pr.astype(self.dt)
        if self.var["mla"] == "absorbed":
            ctx = np.einsum("hts,sc->thc", pr, c_all)                               # [T, nh, kv_lora]
            o = np.einsum("thc,hdc->thd", ctx, w_uv)                                # [T, nh, v]
        else:
            v = np.einsum("sc,hdc->shd", c_all, w_uv)
            o = np.einsum("hts,shd->thd", pr, v)
        out = o.reshape(T, nh * vd) @ w["wo"].T
        if tap is not None:
            tap(f"layer{l}.q", np.concatenate([q_nope, q_rot], axis=-1))
            tap(f"layer{l}.latent", c_new); tap(f"layer{l}.k_rope", kr_new); tap(f"layer{l}.attn_heads", o)
        return out

    def route(self, l, h):
        """-> (top ids [T, k] ascending-by-rank, weights [T, k]) — softmax, top-k, renormalized."""
        w = self.layer_weights(l)
        logits = (h @ w["gate"].T).astype(np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        p = np.exp(logits); p /= p.sum(axis=-1, keepdims=True)
        order = np.argsort(-p, axis=-1, kind="stable")[:, :self.top_k]             # ties: lower id first
        wt = np.take_along_axis(p, order, axis=-1)
        wt = wt / (wt.sum(axis=-1, keepdims=True) + 1e-20) * self.routed_scale
        return order, wt.astype(self.dt)

    def mlp(self, ws, x):
        w1, w2, w3 = ws
        return (silu(x @ w1.T) * (x @ w3.T)) @ w2.T

    def moe(self, l, h, tap=None):
        ids, wt = self.route(l, h)
        y = np.zeros_like(h)
        for e in np.unique(ids):
            rows, slot = np.nonzero(ids == e)
            y[rows] += self.mlp(self.expert(l, int(e)), h[rows]) * wt[rows, slot][:, None]
        if self.n_shared:
            y = y + self.mlp(self.layer_weights(l)["shared"], h)
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
            a = self.attention(l, self.rms(x, w["attn_norm"]), pos, st, tap)
            x = x + a
            if tap is not None:
                tap(f"layer{l}.attn", a); tap(f"layer{l}.mid", x)
            m = self.moe(l, self.rms(x, w["ffn_norm"]), tap)
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
            self.head = self._bf16("output.weight")
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
        if stop_eos and tok == model.eos:
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
