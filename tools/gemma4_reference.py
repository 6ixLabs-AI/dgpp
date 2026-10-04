#!/usr/bin/env python3
"""gemma4_reference.py — host reference forward pass for Gemma 4 (2026-10-04): the text model of
nvidia/Gemma-4-31B-IT-NVFP4 and, with the MoE block, of bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16
(pass its snapshot as --ckpt). Gemma4ForConditionalGeneration; the vision tower is not run.

Pure Python + numpy, reading the NVFP4 checkpoint directly (safetensors headers parsed by hand,
shards mmap'd). It is the ground truth the engine port is compared against, so it is written for
clarity and checkability, not speed: one straightforward implementation of every block, weights
dequantized to float, activations in full precision (the checkpoint's `input_scale`
activation-quant scales are ignored). The model it follows is the transformers class
(models/gemma4/modeling_gemma4.py); tools/gemma4_torch_check.py runs that class and this file on
the same synthetic checkpoint (tools/gemma4_synth.py) and compares them.

  score    --ids-json F [--out F.json]   teacher-forced log P(ids[i] | ids[<i]) + top-5 per position
  generate --ids-json F --max-new N      greedy continuation, prints the new token ids
  dump     --ids-json F --out DIR        .npy of the residual stream after every layer (+ sublayers)
  all modes: --ckpt DIR, --layers N (first N layers only), --threads N, --dtype float32|float64,
             --cache-gb G (dequantized layer weights kept between calls; 0 keeps none)

THE MODEL (what this file computes; names are checkpoint tensor names under `model.language_model.`)

  x = embed_tokens[ids] * bf16(sqrt(H))                   73.5 for H = 5376 (the reference casts the
                                                          scale to the table's dtype)
  per layer L:  x = x + rms(attn(rms(x, input_layernorm)), post_attention_layernorm)
                x = x + rms(mlp(rms(x, pre_feedforward_layernorm)), post_feedforward_layernorm)
                x = x * layer_scalar
  logits = embed_tokens @ rms(x, norm)                    the head is the embedding (no lm_head tensor)
  logits = tanh(logits / 30) * 30                         final_logit_softcapping

  rms(x, w) = x * (mean(x^2) + eps)^-0.5 * w              PLAIN weight (not 1 + w), eps = 1e-6
  mlp(h)    = down_proj @ (gelu_tanh(gate_proj @ h) * (up_proj @ h))
  gelu_tanh(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))

  The MoE block (text_config.enable_moe_block — the 26B-A4B: 128 experts, top 8, on EVERY layer,
  beside the dense MLP, not instead of it). The second sublayer becomes
      m = rms(mlp(rms(x, pre_feedforward_layernorm)), post_feedforward_layernorm_1)
        + rms(moe(x), post_feedforward_layernorm_2)
      x = x + rms(m, post_feedforward_layernorm)
    moe(x): the router reads the residual stream itself, the experts their own norm of it:
      r      = rms(x) * router.scale * bf16(H^-0.5)       no weight in that norm; 0.018798828125 for 2816
      p      = softmax(router.proj @ r)                    over all experts
      top-8 of p; w_e = p_e / sum(top-8 p) * router.per_expert_scale[e]
      h2     = rms(x, pre_feedforward_layernorm_2)
      moe(x) = sum_e w_e * down_e @ (gelu_tanh(gate_e @ h2) * (up_e @ h2))
    The release stores every expert as three Linears (`layers.L.moe.experts.E.gate_proj` / `up_proj`
    / `down_proj`): its quantizer split the class's stacked `experts.gate_up_proj` [E, 2I, H] into the
    gate half (rows 0..I) and the up half.

  Attention — two kinds on config.layer_types (the 31B: 5 sliding, 1 full, repeated):
    sliding_attention: 32 query heads, 16 KV heads, head_dim 256, k_proj AND v_proj;
                       a row at position p reads keys p - 1023 .. p (kv > p - sliding_window);
                       RoPE over the whole head: pairs (i, i + 128), angle = p * 1e4^(-2i/256)
    full_attention:    32 query heads, 4 KV heads, head_dim 512, NO v_proj: the value is the
                       k_proj output (attention_k_eq_v); causal over everything;
                       RoPE "proportional": pairs (i, i + 256) for i < 64 only,
                       angle = p * 1e6^(-2i/512); the other 192 pairs are not rotated
    both:  q = rope(rms(q_proj @ h, q_norm))              per head over the head's dims
           k = rope(rms(k_proj @ h, k_norm))
           v = rms(v_proj @ h | k_proj @ h)               no weight; the full layers' value is the
                                                          k_proj output BEFORE k_norm and RoPE
           scores = q . k                                 NO 1/sqrt(d) (the reference's scaling is 1)
           query head h reads KV head h // (32 / kv_heads); softmax; o_proj on the flattened heads
    rope(x)[d]         = x[d] cos(a_d) - x[d + half] sin(a_d)
    rope(x)[d + half]  = x[d + half] cos(a_d) + x[d] sin(a_d)

  NVFP4 (modelopt): `.weight` U8 [N, K/2], two E2M1 codes per byte, LOW nibble = EVEN column;
    `.weight_scale` F8_E4M3 [N, K/16]; `.weight_scale_2` F32 scalar:
        W[n, k] = e2m1(code[n, k]) * e4m3(weight_scale[n, k // 16]) * weight_scale_2
    E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 (bit 3 = sign).
    The 31B release quantizes the MLP only; its attention projections, norms and embedding are BF16.
    The 26B-A4B release quantizes the attention projections, the MLP and the experts; its routers,
    norms and embedding are BF16. Either way a matrix is read in whatever form the file holds it.

Arithmetic is float32 by default; the RoPE angles and the final log-softmax are float64.
`--dtype float64` runs everything in float64 (a noise-floor check).

Memory: nothing but the norms is kept by default — a layer's matrices are converted when the
layer runs and dropped (the 31B's float32 weights would be about 120 GB). `score` therefore reads
every layer once; `generate` reads every layer once PER TOKEN unless --cache-gb is large enough.
The head is applied in row chunks straight from the BF16 table.
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


# The thread budget (a shared box: 8 unless told otherwise), set before numpy loads its BLAS.
_THREADS = _early_threads()
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(_THREADS)

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import mmap  # noqa: E402
import resource  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

DEFAULT_CKPT = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/"
                                  "snapshots/4135a98a9b728a548947683219633b25682223ac")

assert sys.byteorder == "little", "the BF16 / NVFP4 decoders assume a little-endian host"

# ----------------------------------------------------------------------------------------------
# Debug variants. Every layout assumption can be flipped in isolation with `--variant key=value`
# (repeatable); the defaults are the conventions of the reference implementation. They exist to
# PROVE a convention (the wrong value must give garbage), not to be used.
# ----------------------------------------------------------------------------------------------
VARIANT_DEFAULTS = {
    "nibble": "lo",        # lo: low nibble = even column | hi: high nibble = even column
    "ws2": "mul",          # mul: W = e2m1 * e4m3 * weight_scale_2 | div: ... / weight_scale_2
    "norm": "plain",       # plain: w | 1p: (1 + w)                 (Gemma 2 / 3 wrote 1 + w)
    "embscale": "bf16",    # bf16: bf16(sqrt(H)) | exact: sqrt(H) | none
    "qkscale": "none",     # none: scores = q . k | sqrt: q . k / sqrt(head_dim)
    "rope": "half",        # half: pairs (i, i + head_dim/2) | adjacent: pairs (2i, 2i+1) | none
    "globalrope": "proportional",  # proportional: 64 pairs, exponent over the whole head | full: every pair
    "kvmap": "div",        # div: query head h -> KV head h // groups | mod: h % kv_heads
    "keqv": "raw",         # raw: full-layer value = k_proj output | normed: after k_norm | roped: after RoPE
    "vnorm": "on",         # on: values RMS-normalized (no weight) | off
    "window": "incl",      # incl: keys (p - W, p] | plus1: [p - W, p] | none: full causal
    "scalar": "on",        # on: x *= layer_scalar | off
    "softcap": "on",       # on: tanh(logits / cap) * cap | off
    "act": "gelu_tanh",    # gelu_tanh | gelu (erf) | silu
    # The MoE block.
    "routerscale": "bf16",  # bf16: bf16(H^-0.5) | exact: H^-0.5 | none
    "routerin": "resid",    # resid: the router reads x | normed: rms(x, pre_feedforward_layernorm_2)
    "routernorm": "on",     # on: weightless rms before router.scale | off
    "expertin": "norm2",    # norm2: experts read rms(x, pre_feedforward_layernorm_2) | mlp: the dense MLP's input
    "renorm": "on",         # on: top-k probabilities renormalized to sum 1 | off (NO visible effect: the
                            # norm after the experts divides a token's common factor back out)
    "pes": "on",            # on: weights times router.per_expert_scale | off
    "moecombine": "sum",    # sum: norm_1(mlp) + norm_2(moe) | moeonly | mlponly
}

# ----------------------------------------------------------------------------------------------
# Number formats
# ----------------------------------------------------------------------------------------------
_ST_DTYPES = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "F64": np.float64,
              "U8": np.uint8, "I8": np.int8, "F8_E4M3": np.uint8, "F8_E5M2": np.uint8,
              "I16": np.int16, "I32": np.int32, "I64": np.int64, "U16": np.uint16,
              "U32": np.uint32, "U64": np.uint64, "BOOL": np.bool_}


def _e2m1_table():
    """4-bit E2M1: bit 3 sign, bits 2..1 exponent, bit 0 mantissa (exponent 0 is subnormal)."""
    mag = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
    return np.concatenate([mag, -mag])


def _e4m3_table():
    """8-bit OCP E4M3: sign, 4 exponent bits (bias 7), 3 mantissa bits; no infinities, max 448."""
    t = np.zeros(256, np.float64)
    for c in range(256):
        sign = -1.0 if c & 0x80 else 1.0
        e, m = (c >> 3) & 0xF, c & 0x7
        if e == 0:
            v = (m / 8.0) * 2.0 ** -6
        elif e == 15 and m == 7:
            v = float("nan")
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[c] = sign * v
    return t.astype(np.float32)


_E2M1 = _e2m1_table()
_E4M3 = _e4m3_table()


def _pair_table(low_is_even):
    """byte -> the two float32 it holds, packed in one uint64 so that a single 1-D take expands a
    [N, K/2] byte matrix into a [N, K] float32 matrix (little-endian: the first float is the even column)."""
    b = np.arange(256)
    pair = np.empty((256, 2), np.float32)
    lo, hi = _E2M1[b & 0xF], _E2M1[b >> 4]
    pair[:, 0], pair[:, 1] = (lo, hi) if low_is_even else (hi, lo)
    return pair.view(np.uint64).reshape(256).copy()


def bf16_to_f32(u16):
    """BF16 bits -> float32: the 16 bits are the high half of the float32 pattern (exact)."""
    u = u16.astype(np.uint32)
    u <<= 16
    return u.view(np.float32)


def f32_to_bf16_value(x):
    """A float rounded to the nearest BF16 (ties to even), returned as float32."""
    u = np.array([x], np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return bf16_to_f32(u.astype(np.uint16))[0]


# ----------------------------------------------------------------------------------------------
# Elementwise maths (dtype-preserving)
# ----------------------------------------------------------------------------------------------
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def gelu_tanh(x):
    """torch.nn.functional.gelu(x, approximate="tanh")."""
    dt = x.dtype.type
    return dt(0.5) * x * (dt(1.0) + np.tanh(dt(_SQRT_2_OVER_PI) * (x + dt(0.044715) * x * x * x)))


def gelu_erf(x):
    from math import erf
    return (0.5 * x * (1.0 + np.vectorize(erf)(x / math.sqrt(2.0)))).astype(x.dtype)


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax64(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def top_desc(row, k):
    """Indices of the k largest entries, largest first (ties: lower index first)."""
    k = min(k, row.shape[-1])
    idx = np.argpartition(-row, k - 1)[:k]
    return idx[np.lexsort((idx, -row[idx]))]


# ----------------------------------------------------------------------------------------------
# Checkpoint: safetensors shards, parsed by hand and mmap'd
# ----------------------------------------------------------------------------------------------
class Checkpoint:
    """name -> raw tensor view. A safetensors file is: u64 little-endian header length N, N bytes
    of JSON ({name: {dtype, shape, data_offsets: [begin, end]}}), then the data; offsets are
    relative to byte 8 + N."""

    def __init__(self, ckpt_dir):
        self.dir = ckpt_dir
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.maps, self.index = [], {}
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

    def release_pages(self):
        """Drops the mapped pages from this process's resident set (they stay in the page cache)."""
        for mm in self.maps:
            try:
                mm.madvise(mmap.MADV_DONTNEED)
            except (AttributeError, OSError, ValueError):
                pass


# ----------------------------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------------------------
class State:
    """Per-sequence state, so a sequence can be fed in chunks (or token by token): the keys and
    values every attention layer has produced so far."""

    def __init__(self, n_layers):
        self.pos = 0                      # tokens consumed so far
        self.layers = [None] * n_layers   # {"k": [S, kv, hd], "v": [S, kv, hd]}


class Model:
    PREFIX = "model.language_model."

    def __init__(self, ckpt_dir, layers=None, dtype="float32", variants=None, cache_gb=0.0):
        self.ck = Checkpoint(ckpt_dir)
        root = self.ck.cfg
        if root.get("model_type") != "gemma4" or "text_config" not in root:
            raise SystemExit(f"{ckpt_dir}: not a Gemma4ForConditionalGeneration checkpoint "
                             f"(model_type {root.get('model_type')!r})")
        c = root["text_config"]
        self.dt = np.dtype(dtype).type
        self.var = dict(VARIANT_DEFAULTS)
        self.var.update(variants or {})
        self.mp = self.PREFIX
        self.H = c["hidden_size"]
        self.vocab = c["vocab_size"]
        self.eps = float(c.get("rms_norm_eps", 1e-6))
        self.total_layers = c["num_hidden_layers"]
        self.n_layers = self.total_layers if layers is None else max(0, min(int(layers), self.total_layers))
        types = c.get("layer_types") or ["sliding_attention" if (i + 1) % 6 else "full_attention"
                                         for i in range(self.total_layers)]
        self.sliding = [t == "sliding_attention" for t in types]
        self.heads = c["num_attention_heads"]
        self.kv_sliding = c["num_key_value_heads"]
        self.k_eq_v = bool(c.get("attention_k_eq_v", False))
        self.kv_global = (c.get("num_global_key_value_heads") or self.kv_sliding) if self.k_eq_v else self.kv_sliding
        self.hd_sliding = c.get("head_dim", 256)
        self.hd_global = c.get("global_head_dim", 512)
        self.window = c["sliding_window"]
        rp = c.get("rope_parameters") or {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0}}
        self.theta_sliding = float(rp["sliding_attention"]["rope_theta"])
        self.theta_global = float(rp["full_attention"]["rope_theta"])
        # transformers _compute_proportional_rope_parameters: int(factor * head_dim // 2) rotated pairs.
        self.pairs_global = int(rp["full_attention"].get("partial_rotary_factor", 1.0) * self.hd_global // 2)
        self.cap = c.get("final_logit_softcapping")
        eos = root.get("eos_token_id", c.get("eos_token_id"))
        self.eos = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        for key, want in (("hidden_size_per_layer_input", 0), ("num_kv_shared_layers", 0)):
            if c.get(key, 0) != want:
                raise SystemExit(f"text_config.{key} = {c.get(key)!r}: not implemented by this reference")
        self.moe_on = bool(c.get("enable_moe_block"))
        if self.moe_on:
            self.n_experts = int(c["num_experts"])
            self.top_k = int(c["top_k_experts"])
            if self.var["routerscale"] == "bf16":
                self.router_scale = self.dt(f32_to_bf16_value(np.float32(self.H ** -0.5)))
            elif self.var["routerscale"] == "exact":
                self.router_scale = self.dt(self.H ** -0.5)
            else:
                self.router_scale = self.dt(1.0)
        if c.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise SystemExit(f"text_config.hidden_activation = {c.get('hidden_activation')!r}: not implemented")
        self._pair = _pair_table(self.var["nibble"] == "lo")
        self._cache, self._cache_bytes, self._cache_budget = {}, 0, int(float(cache_gb) * (1 << 30))
        self._small = {}
        if self.var["embscale"] == "bf16":
            self.embed_scale = self.dt(f32_to_bf16_value(np.float32(math.sqrt(self.H))))
        elif self.var["embscale"] == "exact":
            self.embed_scale = self.dt(math.sqrt(self.H))
        else:
            self.embed_scale = self.dt(1.0)

    # ---------------- weights ----------------
    def vec(self, name):
        """A small BF16 / F32 tensor (norm weights, scalars) as float, kept."""
        w = self._small.get(name)
        if w is None:
            w = self._dense(name)
            self._small[name] = w
        return w

    def _dense(self, name):
        raw = self.ck.raw(name)
        st_dtype = self.ck.index[name][1]
        if st_dtype == "BF16":
            w = bf16_to_f32(raw)
        elif st_dtype == "F32":
            w = np.array(raw, np.float32)
        else:
            raise TypeError(f"{name}: expected BF16 or F32, found {st_dtype}")
        return w if self.dt == np.float32 else w.astype(self.dt)

    def _fp4(self, prefix):
        """A modelopt NVFP4 linear dequantized to a dense float [N, K] matrix."""
        codes = self.ck.raw(prefix + ".weight")              # U8      [N, K/2]
        scale = self.ck.raw(prefix + ".weight_scale")        # E4M3    [N, K/16]
        ws2 = np.float32(self.ck.raw(prefix + ".weight_scale_2").reshape(()))
        n, half = codes.shape
        w = self._pair[codes].view(np.float32)               # [N, K]: byte j -> columns 2j, 2j+1
        s = _E4M3[scale]                                     # [N, K/16]
        s = s * ws2 if self.var["ws2"] == "mul" else s / ws2
        w = w.reshape(n, half // 8, 16)
        w *= s[:, :, None]
        w = w.reshape(n, 2 * half)
        return w if self.dt == np.float32 else w.astype(self.dt)

    def linear(self, prefix):
        """Weight of a linear layer as a dense float matrix, whichever way the checkpoint stores it
        (NVFP4 or BF16). Kept while the cache budget lasts, else converted per call."""
        w = self._cache.get(prefix)
        if w is not None:
            return w
        st_dtype = self.ck.index[prefix + ".weight"][1]
        w = self._fp4(prefix) if st_dtype == "U8" else self._dense(prefix + ".weight")
        if self._cache_bytes + w.nbytes <= self._cache_budget:
            self._cache[prefix] = w
            self._cache_bytes += w.nbytes
        return w

    def layer_prefix(self, L):
        return f"{self.mp}layers.{L}"

    # ---------------- blocks ----------------
    def rms(self, x, w=None):
        """Gemma4RMSNorm: x * (mean(x^2) + eps)^-0.5 [* w]."""
        ms = np.mean(np.square(x), axis=-1, keepdims=True) + self.dt(self.eps)
        y = x * np.power(ms, self.dt(-0.5))
        if w is None:
            return y
        return y * ((1.0 + w) if self.var["norm"] == "1p" else w)

    def act(self, x):
        if self.var["act"] == "gelu":
            return gelu_erf(x)
        if self.var["act"] == "silu":
            return silu(x)
        return gelu_tanh(x)

    def embed(self, ids):
        raw = self.ck.raw(self.mp + "embed_tokens.weight")
        rows = np.ascontiguousarray(raw[np.asarray(ids, np.int64)])
        x = bf16_to_f32(rows) if self.ck.index[self.mp + "embed_tokens.weight"][1] == "BF16" else np.array(rows, np.float32)
        return x.astype(self.dt) * self.embed_scale

    def geometry(self, L):
        """(head_dim, kv_heads, rotated pairs, theta, window or None) of layer L."""
        if self.sliding[L]:
            return self.hd_sliding, self.kv_sliding, self.hd_sliding // 2, self.theta_sliding, self.window
        pairs = self.pairs_global if self.var["globalrope"] == "proportional" else self.hd_global // 2
        return self.hd_global, self.kv_global, pairs, self.theta_global, None

    def rope(self, x, pos, pairs, theta):
        """x [T, heads, hd] at absolute positions pos [T]: the first `pairs` pairs rotate, pair i
        with inverse frequency theta^(-2i / hd) — the exponent's divisor is the whole head."""
        if self.var["rope"] == "none":
            return x
        hd = x.shape[-1]
        half = hd // 2
        inv = np.power(float(theta), -2.0 * np.arange(pairs, dtype=np.float64) / hd)
        ang = np.asarray(pos, np.float64)[:, None] * inv[None, :]          # [T, pairs]
        cos, sin = np.cos(ang).astype(self.dt)[:, None, :], np.sin(ang).astype(self.dt)[:, None, :]
        out = x.copy()
        if self.var["rope"] == "adjacent":
            a, b = x[..., 0:2 * pairs:2], x[..., 1:2 * pairs:2]
            out[..., 0:2 * pairs:2] = a * cos - b * sin
            out[..., 1:2 * pairs:2] = b * cos + a * sin
            return out
        a, b = x[..., :pairs], x[..., half:half + pairs]
        out[..., :pairs] = a * cos - b * sin
        out[..., half:half + pairs] = b * cos + a * sin
        return out

    def attention(self, L, h, st_layer, pos0, tap=None):
        """h [T, H] (already normalized) -> o_proj output [T, H]; appends this chunk's keys and
        values to the layer's cache."""
        p = self.layer_prefix(L) + ".self_attn"
        T = h.shape[0]
        hd, kvh, pairs, theta, window = self.geometry(L)
        nh = self.heads
        pos = np.arange(pos0, pos0 + T)
        q = (h @ self.linear(p + ".q_proj").T).reshape(T, nh, hd)
        q = self.rope(self.rms(q, self.vec(p + ".q_norm.weight")), pos, pairs, theta)
        kraw = (h @ self.linear(p + ".k_proj").T).reshape(T, kvh, hd)
        knorm = self.rms(kraw, self.vec(p + ".k_norm.weight"))
        k = self.rope(knorm, pos, pairs, theta)
        if p + ".v_proj.weight" in self.ck:
            vraw = (h @ self.linear(p + ".v_proj").T).reshape(T, kvh, hd)
        else:                                   # attention_k_eq_v: the k_proj output is the value
            vraw = {"raw": kraw, "normed": knorm, "roped": k}[self.var["keqv"]]
        v = self.rms(vraw) if self.var["vnorm"] == "on" else vraw
        if tap is not None:
            tap(f"L{L:02d}_attn_q", q)
            tap(f"L{L:02d}_attn_k", k)
            tap(f"L{L:02d}_attn_v", v)
        if st_layer.get("k") is None:
            K, V = k, v
        else:
            K, V = np.concatenate([st_layer["k"], k]), np.concatenate([st_layer["v"], v])
        st_layer["k"], st_layer["v"] = K, V
        S = K.shape[0]
        groups = nh // kvh
        kv_of = (np.arange(nh) // groups) if self.var["kvmap"] == "div" else (np.arange(nh) % kvh)
        # scores[t, h, s] = q[t, h] . K[s, kv_of[h]]
        scores = np.einsum("thd,shd->ths", q, K[:, kv_of, :], optimize=True)
        if self.var["qkscale"] == "sqrt":
            scores = scores * self.dt(1.0 / math.sqrt(hd))
        s_idx = np.arange(S)[None, :]
        visible = s_idx <= pos[:, None]                                   # causal
        if window is not None and self.var["window"] != "none":
            w = window + (1 if self.var["window"] == "plus1" else 0)
            visible &= s_idx > (pos[:, None] - w)                         # kv_idx > q_idx - sliding_window
        scores = np.where(visible[:, None, :], scores, self.dt(-np.inf))
        probs = softmax(scores)
        ctx = np.einsum("ths,shd->thd", probs, V[:, kv_of, :], optimize=True)   # [T, heads, hd]
        if tap is not None:
            tap(f"L{L:02d}_attn_ctx", ctx)
        return ctx.reshape(T, nh * hd) @ self.linear(p + ".o_proj").T

    def mlp(self, L, h):
        p = self.layer_prefix(L) + ".mlp"
        g = self.act(h @ self.linear(p + ".gate_proj").T)
        g *= h @ self.linear(p + ".up_proj").T
        return g @ self.linear(p + ".down_proj").T

    def route(self, L, x):
        """x [T, H] (the residual stream) -> (expert ids [T, k], weights [T, k], best first)."""
        p = self.layer_prefix(L)
        r = x
        if self.var["routerin"] == "normed":
            r = self.rms(x, self.vec(p + ".pre_feedforward_layernorm_2.weight"))
        if self.var["routernorm"] == "on":
            r = self.rms(r)
        r = r * self.vec(p + ".router.scale") * self.router_scale
        logits = r @ self.linear(p + ".router.proj").T
        probs = softmax(logits)
        ids = np.stack([top_desc(row, self.top_k) for row in probs])
        w = np.take_along_axis(probs, ids, axis=-1)
        if self.var["renorm"] == "on":
            w = w / w.sum(axis=-1, keepdims=True)
        if self.var["pes"] == "on":
            w = w * self.vec(p + ".router.per_expert_scale")[ids]
        return ids, w.astype(self.dt), logits

    def moe(self, L, x, mlp_in, tap=None):
        """The routed experts' output [T, H] for the residual stream x."""
        p = self.layer_prefix(L)
        ids, w, logits = self.route(L, x)
        h = mlp_in if self.var["expertin"] == "mlp" else self.rms(x, self.vec(p + ".pre_feedforward_layernorm_2.weight"))
        out = np.zeros_like(x)
        for e in sorted(set(int(i) for i in ids.reshape(-1))):
            rows, slot = np.nonzero(ids == e)
            ep = f"{p}.moe.experts.{e}"
            hr = h[rows]
            g = self.act(hr @ self.linear(ep + ".gate_proj").T)
            g *= hr @ self.linear(ep + ".up_proj").T
            out[rows] += w[rows, slot][:, None] * (g @ self.linear(ep + ".down_proj").T)
        if tap is not None:
            tap(f"L{L:02d}_router_logits", logits)
            tap(f"L{L:02d}_router_ids", ids)
            tap(f"L{L:02d}_router_w", w)
            tap(f"L{L:02d}_moe_out", out)
        return out

    def new_state(self):
        return State(self.n_layers)

    def forward(self, ids, st, tap=None, detail_layers=(), on_layer=None):
        """Feeds `ids` at positions st.pos.. ; returns the residual stream [T, H] after the last
        layer run (BEFORE the final norm)."""
        x = self.embed(ids)
        T = x.shape[0]
        if tap is not None:
            tap("h_00", x)
        for L in range(self.n_layers):
            if st.layers[L] is None:
                st.layers[L] = {}
            p = self.layer_prefix(L)
            t = tap if (tap is not None and L in detail_layers) else None
            h = self.rms(x, self.vec(p + ".input_layernorm.weight"))
            if t is not None:
                t(f"L{L:02d}_attn_in", h)
            a = self.attention(L, h, st.layers[L], st.pos, tap=t)
            if t is not None:
                t(f"L{L:02d}_attn_out", a)
            x = x + self.rms(a, self.vec(p + ".post_attention_layernorm.weight"))
            if t is not None:
                t(f"L{L:02d}_mid", x)
            h = self.rms(x, self.vec(p + ".pre_feedforward_layernorm.weight"))
            m = self.mlp(L, h)
            if t is not None:
                t(f"L{L:02d}_mlp_in", h)
                t(f"L{L:02d}_mlp_out", m)
            if self.moe_on:
                m1 = self.rms(m, self.vec(p + ".post_feedforward_layernorm_1.weight"))
                m2 = self.rms(self.moe(L, x, h, tap=t), self.vec(p + ".post_feedforward_layernorm_2.weight"))
                m = {"sum": m1 + m2, "moeonly": m2, "mlponly": m1}[self.var["moecombine"]]
            x = x + self.rms(m, self.vec(p + ".post_feedforward_layernorm.weight"))
            if self.var["scalar"] == "on":
                x = x * self.vec(p + ".layer_scalar").reshape(())
            if tap is not None:
                tap(f"h_{L + 1:02d}", x)
            if not self._cache:
                self.ck.release_pages()
            if on_layer is not None:
                on_layer(L + 1)
        st.pos += T
        return x

    def final_norm(self, x):
        return self.rms(x, self.vec(self.mp + "norm.weight"))

    def logits(self, x, rows_per_step=16384):
        """[T, H] residual -> [T, vocab] logits: the tied head applied in row chunks of the BF16
        table, then the soft-cap."""
        hn = self.final_norm(x)
        raw = self.ck.raw(self.mp + "embed_tokens.weight")
        bf16 = self.ck.index[self.mp + "embed_tokens.weight"][1] == "BF16"
        out = np.empty((x.shape[0], self.vocab), self.dt)
        for r0 in range(0, self.vocab, rows_per_step):
            rows = np.ascontiguousarray(raw[r0:r0 + rows_per_step])
            w = (bf16_to_f32(rows) if bf16 else np.array(rows, np.float32)).astype(self.dt, copy=False)
            out[:, r0:r0 + rows_per_step] = hn @ w.T
        if self.cap and self.var["softcap"] == "on":
            cap = self.dt(self.cap)
            out = np.tanh(out / cap) * cap
        return out

    def logprobs(self, x):
        return log_softmax64(self.logits(x))


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def progress(stage, stages, done, total, err, t0):
    print(f"[{stage}/{stages}] {done}/{total} err={err} elapsed={int(time.time() - t0)}s", flush=True)


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 3) if sys.platform == "darwin" else r / (1024.0 * 1024.0)   # bytes | KiB


def load_ids(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        d = d.get("token_ids") or d.get("ids")
    if not isinstance(d, list) or not d or not all(isinstance(i, int) for i in d):
        raise ValueError(f"{path}: expected a non-empty JSON list of token ids")
    return d


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def make_model(a):
    variants = {}
    for kv in a.variant or []:
        k, _, v = kv.partition("=")
        if k not in VARIANT_DEFAULTS:
            raise SystemExit(f"unknown --variant {k!r}; known: {', '.join(sorted(VARIANT_DEFAULTS))}")
        variants[k] = v
    m = Model(a.ckpt, layers=a.layers, dtype=a.dtype, variants=variants, cache_gb=a.cache_gb)
    if variants:
        print(f"# DEBUG VARIANTS ACTIVE: {variants}", flush=True)
    return m


def run_forward(model, ids, chunk=None, on_layer=None, tap=None, detail_layers=()):
    """Feeds the whole sequence (in chunks of `chunk` tokens when given). Returns (x [T, H], state)."""
    st = model.new_state()
    step = len(ids) if not chunk else max(1, int(chunk))
    xs = []
    for s in range(0, len(ids), step):
        xs.append(model.forward(ids[s:s + step], st, tap=tap, detail_layers=detail_layers,
                                on_layer=on_layer if s + step >= len(ids) else None))
    return np.concatenate(xs), st


def score_ids(model, ids, chunk=None, on_layer=None, top=5, keep_hidden=False):
    """Teacher-forced scoring. Entry i (i >= 1) of every list is about log P(ids[i] | ids[<i])."""
    t0 = time.time()
    x, _ = run_forward(model, ids, chunk=chunk, on_layer=on_layer)
    T = len(ids)
    logprobs, tops, ranks, next_top = [None], [None], [None], None
    for b0 in range(0, T, 32):
        lp = model.logprobs(x[b0:b0 + 32])
        for j in range(lp.shape[0]):
            i = b0 + j + 1                               # row j predicts position i
            entry = [[int(t), float(lp[j, t])] for t in top_desc(lp[j], top)]
            if i < T:
                logprobs.append(float(lp[j, ids[i]]))
                tops.append(entry)
                ranks.append(int(np.count_nonzero(lp[j] > lp[j, ids[i]])))
            else:
                next_top = entry
    res = {"n_tokens": T, "layers": model.n_layers, "dtype": np.dtype(model.dt).name, "ids": list(ids),
           "logprobs": logprobs, "top": tops, "rank": ranks, "next_top": next_top,
           "seconds": round(time.time() - t0, 2), "peak_rss_gb": round(peak_rss_gb(), 2)}
    if keep_hidden:
        res["_hidden"] = x
    return res


def cmd_score(a):
    ids = load_ids(a.ids_json)
    model = make_model(a)
    t0 = time.time()
    res = score_ids(model, ids, chunk=a.chunk,
                    on_layer=lambda d: progress(1, 1, d, model.n_layers, 0, t0))
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


def greedy(model, ids, max_new, stop_eos=True, on_token=None, on_layer=None, top=5):
    """Greedy continuation. Returns (new ids, per-step records)."""
    st = model.new_state()
    x = model.forward(ids, st, on_layer=on_layer)[-1:]
    new, steps = [], []
    for step in range(max_new):
        lp = model.logprobs(x)[0]
        t = top_desc(lp, top)
        tok = int(t[0])
        new.append(tok)
        steps.append({"id": tok, "top": [[int(i), float(lp[i])] for i in t],
                      "margin": float(lp[t[0]] - lp[t[1]])})
        if on_token is not None:
            on_token(step + 1)
        if stop_eos and tok in model.eos:
            break
        if step + 1 < max_new:
            x = model.forward([tok], st)
    return new, steps


def cmd_generate(a):
    ids = load_ids(a.ids_json)
    model = make_model(a)
    t0 = time.time()
    new, steps = greedy(model, ids, a.max_new, stop_eos=not a.ignore_eos,
                        on_layer=lambda d: progress(1, 2, d, model.n_layers, 0, t0),
                        on_token=lambda d: progress(2, 2, d, a.max_new, 0, t0))
    if a.out:
        write_json(a.out, {"prompt_ids": ids, "new_ids": new, "steps": steps, "layers": model.n_layers,
                           "seconds": round(time.time() - t0, 2), "peak_rss_gb": round(peak_rss_gb(), 2)})
    print(json.dumps(new))


def cmd_dump(a):
    ids = load_ids(a.ids_json)
    model = make_model(a)
    os.makedirs(a.out, exist_ok=True)
    detail = sorted({int(s) for s in a.detail_layers.split(",") if s.strip() != ""} & set(range(model.n_layers)))
    files = {}

    def tap(name, arr):
        arr = np.asarray(arr)
        arr = arr.astype(np.int32) if arr.dtype.kind in "iu" else arr.astype(np.float32)
        np.save(os.path.join(a.out, name + ".npy"), arr)
        files[name + ".npy"] = list(arr.shape)

    t0 = time.time()
    x, _ = run_forward(model, ids, tap=tap, detail_layers=detail,
                       on_layer=lambda d: progress(1, 1, d, model.n_layers, 0, t0))
    tap("h_final_norm", model.final_norm(x))
    tap("logits_last", model.logits(x[-1:])[0])
    write_json(os.path.join(a.out, "meta.json"), {
        "ids": ids, "n_tokens": len(ids), "layers": model.n_layers, "dtype": np.dtype(model.dt).name,
        "detail_layers": detail, "files": files, "variants": model.var,
        "notes": {
            "h_NN": "residual stream [T, H]: h_00 after the scaled embedding, h_NN after layer NN-1 (layer_scalar applied)",
            "h_final_norm": "rms(h_last, norm) [T, H]",
            "logits_last": "soft-capped logits at the last position [vocab]",
            "LNN_attn_in": "rms(x, input_layernorm)", "LNN_attn_out": "o_proj output, before post_attention_layernorm",
            "LNN_attn_q / _k": "after q_norm / k_norm and RoPE [T, heads | kv_heads, head_dim]",
            "LNN_attn_v": "after the weightless value norm [T, kv_heads, head_dim]",
            "LNN_attn_ctx": "attention output per query head, before o_proj [T, heads, head_dim]",
            "LNN_mid": "residual stream after the attention sublayer",
            "LNN_mlp_in": "rms(x, pre_feedforward_layernorm)",
            "LNN_mlp_out": "down_proj output, before post_feedforward_layernorm (before _1 with the MoE block)",
            "LNN_router_logits": "[T, experts]", "LNN_router_ids": "top-k expert ids [T, k], best first",
            "LNN_router_w": "renormalized top-k weights times per_expert_scale [T, k]",
            "LNN_moe_out": "sum over the top-k experts, before post_feedforward_layernorm_2"},
        "seconds": round(time.time() - t0, 2), "peak_rss_gb": round(peak_rss_gb(), 2)})
    print(f"# wrote {len(files)} arrays to {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint snapshot directory")
        p.add_argument("--ids-json", required=True, help="JSON list of token ids (or {\"ids\": [...]})")
        p.add_argument("--layers", type=int, default=None, help="run the first N layers only")
        p.add_argument("--threads", type=int, default=_THREADS)
        p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
        p.add_argument("--cache-gb", type=float, default=0.0, help="dequantized matrices kept between calls")
        p.add_argument("--variant", action="append", help="debug: key=value (see VARIANT_DEFAULTS)")

    p = sub.add_parser("score")
    common(p)
    p.add_argument("--out")
    p.add_argument("--chunk", type=int, default=None, help="feed the sequence in chunks of N tokens")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("generate")
    common(p)
    p.add_argument("--max-new", type=int, default=16)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("dump")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--detail-layers", default="", help="comma-separated layers to dump sublayer tensors for")
    p.set_defaults(fn=cmd_dump)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
