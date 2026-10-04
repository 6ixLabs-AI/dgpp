#!/usr/bin/env python3
"""qwen3next_reference.py — host reference forward pass for Qwen3-Next-80B-A3B-Instruct (2026-10-03).

Pure Python + numpy, reading the NVIDIA NVFP4 checkpoint directly (safetensors headers parsed by
hand, shards mmap'd). It is the ground truth the DGPP engine port is compared against, so it is
written for clarity and checkability, not speed: one straightforward implementation of every
block, weights dequantized to float, activations in full precision (the checkpoint's
`input_scale` activation-quant scales and the `k_scale` / `v_scale` KV-cache scales are ignored).

  score    --ids-json F [--out F.json]   teacher-forced log P(ids[i] | ids[<i]) + top-5 per position
  generate --ids-json F --max-new N      greedy continuation, prints the new token ids
  dump     --ids-json F --out DIR        .npy of the residual stream after every layer (+ sublayers)
  accept   --refset F.jsonl --out DIR    the acceptance test against an Atlas capture
  mtp      --ids-json F                  the MTP draft head vs the main model's own next token
  all modes: --ckpt DIR, --layers N (first N layers only), --threads N, --dtype float32|float64

THE MODEL (what this file computes; names are checkpoint tensor names)

  x = embed_tokens[ids]                                   no scaling
  per layer L:  x += mixer(norm1p(x, input_layernorm))    mixer = full attention iff (L+1) % 4 == 0,
                x += moe(norm1p(x, post_attention_layernorm))        else Gated DeltaNet
  logits = lm_head @ norm1p(x, model.norm)                lm_head is untied

  norm1p(x, w) = x * rsqrt(mean(x^2) + eps) * (1 + w)     zero-centred weight, eps = 1e-6

  Full attention (16 query heads, 2 KV heads, head_dim 256):
    q_proj -> [16, 512] = per head [query 256 | gate 256];  k_proj, v_proj -> [2, 256]
    q = norm1p(q, q_norm), k = norm1p(k, k_norm)           per head, over 256 dims
    RoPE on dims 0..63 of q and k: pairs (i, i+32), angle = pos * theta^(-i/32), theta = 1e7
    causal softmax attention, scale 1/sqrt(256), query head h reads KV head h // 8
    out = attn * sigmoid(gate); flatten 16x256; o_proj

  Gated DeltaNet (16 key heads, 32 value heads, head dims 128, conv kernel 4):
    in_proj_qkvz -> [16 groups, 768] = per key-head group [q 128 | k 128 | v 2x128 | z 2x128]
    in_proj_ba   -> [16 groups, 4]   = per group [b, b, a, a]
    conv input   = [all q (2048) | all k (2048) | all v (4096)], depthwise causal conv over time
                   out[t] = sum_j w[c, j] * in[t - 3 + j]   (w[c, 3] multiplies the current token)
                   then SiLU
    q, k         = L2-normalized per key head: x * rsqrt(sum(x^2) + 1e-6);  q *= 1/sqrt(128)
    beta = sigmoid(b);  g = -exp(A_log) * softplus(a + dt_bias)            (log decay, per value head)
    value head vh uses key head vh // 2. State S[vh] is [128 (k), 128 (v)], zero at the start:
        S *= exp(g);  kv = S^T k;  delta = (v - kv) * beta;  S += outer(k, delta);  o = S^T q
    gated output norm, per value head over 128 dims:  o * rsqrt(mean(o^2) + eps) * w * silu(z)
                   (plain weight here, not 1 + w); flatten 32x128; out_proj

  MoE (512 experts, top-10):
    p = softmax(mlp.gate @ h) over all 512; top-10; weights = p[top] / sum(p[top])
    y = sum_e w_e * down_e(silu(gate_e @ h) * (up_e @ h))
      + sigmoid(shared_expert_gate @ h) * shared_expert(h)                 (same MLP form)

  NVFP4 (modelopt): `.weight` U8 [N, K/2], two E2M1 codes per byte, LOW nibble = EVEN column;
    `.weight_scale` F8_E4M3 [N, K/16]; `.weight_scale_2` F32 scalar:
        W[n, k] = e2m1(code[n, k]) * e4m3(weight_scale[n, k // 16]) * weight_scale_2
    E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 (bit 3 = sign).

  MTP draft head (`mtp.*`, all BF16): see Model.mtp_forward.

Arithmetic is float32 by default; the DeltaNet gates and state, the RoPE angles and the final
log-softmax are float64. `--dtype float64` runs everything in float64 (a noise-floor check).

Memory: non-expert weights are converted once and kept (about 8 GB float32); routed experts are
dequantized on demand and dropped. The mmap'd shard pages are released after every layer.
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


# The thread budget (a shared box: 8 unless told otherwise) is split between the BLAS pool, sized
# here before numpy loads, and the expert-dequant workers. OpenBLAS workers spin for a while after
# a call, so the two pools can be busy at once: the budget is the sum, not the maximum.
_THREADS = _early_threads()
_BLAS_THREADS = max(1, _THREADS // 2)
_DEQUANT_THREADS = max(1, _THREADS - _BLAS_THREADS)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(_BLAS_THREADS)

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import mmap  # noqa: E402
import resource  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

DEFAULT_CKPT = ("/home/mark/.cache/huggingface/hub/models--nvidia--Qwen3-Next-80B-A3B-Instruct-NVFP4/"
                "snapshots/8fb2682f136cf94d932a498f18cb1e428832a912")

assert sys.byteorder == "little", "the BF16 / NVFP4 decoders assume a little-endian host"

# ----------------------------------------------------------------------------------------------
# Debug variants. Every layout assumption can be flipped in isolation with `--variant key=value`
# (repeatable); the defaults are the conventions that were confirmed against the checkpoint and
# against Atlas. They exist to PROVE a convention (the wrong value must give garbage), not to be used.
# ----------------------------------------------------------------------------------------------
VARIANT_DEFAULTS = {
    "nibble": "lo",        # lo: low nibble = even column | hi: high nibble = even column
    "ws2": "mul",          # mul: W = e2m1 * e4m3 * weight_scale_2 | div: ... / weight_scale_2
    "qkvz": "interleaved",  # interleaved: per key-head group [q|k|v|z] | flat: [all q|all k|all v|all z]
    "ba": "bbaa",          # bbaa: per group [b,b,a,a] | baba: per group [b,a,b,a] | flat: [all b|all a]
    "gate": "perhead",     # perhead: q_proj = per head [q|gate] | flat: [all q|all gate]
    "rope": "half",        # half: pairs (i, i+32) | adjacent: pairs (2i, 2i+1) | none
    "kvmap": "div",        # div: query head h -> KV head h // 8 | mod: h % 2
    "gdnkey": "div",       # div: value head vh -> key head vh // 2 | mod: vh % 16
    "conv": "causal",      # causal: w[:, 3] multiplies the current token | reversed: w[:, 0] does
    "norm": "1p",          # 1p: (1 + w) | plain: w                      (the layer / q / k / final norms)
    "gdnnorm": "plain",    # plain: w | 1p: (1 + w)                      (the DeltaNet output norm)
    "qscale": "on",        # on: q *= 1/sqrt(128) in the DeltaNet | off
    "renorm": "on",        # on: top-k router weights renormalized | off
    # Not a layout assumption either: the YaRN ramp (the engine's rope_scaling). 0: the plain rope;
    # 2 or 4: Qwen's long-context recipe over max_position_embeddings (plain 1-D rope).
    "yarn_factor": "0",
    # Not a layout assumption: an NVFP4 encode/decode round trip of BF16 matrices, to measure what
    # an engine that re-quantizes them to 4 bits at load (Atlas does) loses. none | atlas (= all
    # four classes) | any of qkv, qkvz, router, lm_head joined by '+'.
    "requant": "none",
}
REQUANT_CLASSES = ("qkv", "qkvz", "router", "lm_head")

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


def _round_to_grid(x, grid):
    """Nearest value of an ascending non-negative grid, saturating, ties to the even grid index."""
    x = np.clip(x, grid[0], grid[-1])
    hi = np.clip(np.searchsorted(grid, x, side="left"), 1, grid.size - 1)
    lo = hi - 1
    dlo, dhi = x - grid[lo], grid[hi] - x
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (hi % 2 == 0))
    return np.where(pick_hi, grid[hi], grid[lo])


def nvfp4_roundtrip(w, rows_per_step=4096):
    """float [N, K] -> what an NVFP4 encode + decode of it yields, with the NVIDIA modelopt recipe:
        weight_scale_2 = amax(W) / (6 * 448);  s_b = e4m3(amax(block) / 6 / weight_scale_2)
        code = e2m1(w / (s_b * weight_scale_2));  w' = code * s_b * weight_scale_2"""
    n, k = w.shape
    e2 = _E2M1[:8]                                         # 0 .. 6, ascending, index = code
    e4 = _E4M3[:127]                                       # 0 .. 448, ascending, index = code
    amax = float(np.abs(w).max())
    ws2 = np.float32(amax / (6.0 * 448.0)) if amax > 0 else np.float32(1.0)
    out = np.empty((n, k), np.float32)
    for r0 in range(0, n, rows_per_step):
        blk = np.asarray(w[r0:r0 + rows_per_step], np.float32).reshape(-1, k // 16, 16)
        bmax = np.abs(blk).max(axis=-1)
        scale = _round_to_grid(bmax / np.float32(6.0) / ws2, e4) * ws2      # the block's absolute scale
        inv = np.where(scale > 0, np.float32(1.0) / np.where(scale > 0, scale, 1), 0).astype(np.float32)
        x = blk * inv[:, :, None]
        q = np.sign(x) * _round_to_grid(np.abs(x), e2)
        out[r0:r0 + rows_per_step] = (q * scale[:, :, None]).reshape(-1, k)
    return out


# ----------------------------------------------------------------------------------------------
# Elementwise maths (dtype-preserving)
# ----------------------------------------------------------------------------------------------
def sigmoid(x):
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    return np.logaddexp(0.0, x)


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
    """Per-sequence recurrent state, so a sequence can be fed in chunks (or token by token)."""

    def __init__(self, n_layers):
        self.pos = 0                      # tokens consumed so far
        self.layers = [None] * n_layers   # attention: {"k","v"}; DeltaNet: {"conv","S"}


class Model:
    def __init__(self, ckpt_dir, layers=None, dtype="float32", threads=_THREADS, variants=None):
        self.ck = Checkpoint(ckpt_dir)
        c = self.ck.cfg
        self.dt = np.dtype(dtype).type
        self.var = dict(VARIANT_DEFAULTS)
        self.var.update(variants or {})
        self.H = c["hidden_size"]
        self.vocab = c["vocab_size"]
        self.eps = float(c["rms_norm_eps"])
        self.total_layers = c["num_hidden_layers"]
        self.n_layers = self.total_layers if layers is None else max(0, min(int(layers), self.total_layers))
        self.is_attn = [t == "full_attention" for t in c["layer_types"]]
        self.eos = c.get("eos_token_id")
        # full attention
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c["num_key_value_heads"]
        self.head_dim = c["head_dim"]
        self.rot_dim = int(self.head_dim * c.get("partial_rotary_factor", 1.0))
        self.theta = float(c["rope_theta"])
        self.max_pos = int(c.get("max_position_embeddings", 262144))
        # gated deltanet
        self.nk = c["linear_num_key_heads"]
        self.nv = c["linear_num_value_heads"]
        self.dk = c["linear_key_head_dim"]
        self.dv = c["linear_value_head_dim"]
        self.conv_k = c["linear_conv_kernel_dim"]
        # moe
        self.n_exp = c["num_experts"]
        self.top_k = c["num_experts_per_tok"]
        self.norm_topk = bool(c.get("norm_topk_prob", True))

        self._cache = {}
        rq = self.var["requant"]
        self._requant = set(REQUANT_CLASSES) if rq == "atlas" else set(rq.split("+")) - {"none", ""}
        if self._requant - set(REQUANT_CLASSES):
            raise ValueError(f"--variant requant: unknown class in {rq!r}; known: {REQUANT_CLASSES}")
        self._pair = _pair_table(self.var["nibble"] == "lo")
        self.threads = max(1, int(threads) - _BLAS_THREADS)          # dequant workers
        self.pool = ThreadPoolExecutor(max_workers=self.threads)
        self.stats = {"experts_dequantized": 0}

    # ---------------- weights ----------------
    def dense(self, name):
        """A BF16 (or F32) tensor as float, converted once and kept."""
        w = self._cache.get(name)
        if w is None:
            w = self._dense_nocache(name)
            self._cache[name] = w
        return w

    def _dense_nocache(self, name):
        raw = self.ck.raw(name)
        st_dtype = self.ck.index[name][1]
        if st_dtype == "BF16":
            w = bf16_to_f32(raw)
        elif st_dtype == "F32":
            w = np.array(raw, np.float32)
        else:
            raise TypeError(f"{name}: expected BF16 or F32, found {st_dtype}")
        if self._requant and self._requant_class(name) in self._requant:   # debug variant
            w = nvfp4_roundtrip(w)
        return w if self.dt == np.float32 else w.astype(self.dt)

    @staticmethod
    def _requant_class(name):
        if not name.startswith("model.") and name != "lm_head.weight":
            return None
        if name == "lm_head.weight":
            return "lm_head"
        if name.endswith(".mlp.gate.weight"):
            return "router"
        if name.endswith(".in_proj_qkvz.weight"):
            return "qkvz"
        if name.endswith((".q_proj.weight", ".k_proj.weight", ".v_proj.weight")):
            return "qkv"
        return None

    def _fp4(self, prefix):
        """An NVFP4 linear dequantized to a dense float [N, K] matrix (not cached)."""
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
        """Weight of a linear layer, whichever way the checkpoint stores it (kept)."""
        w = self._cache.get(prefix)
        if w is None:
            if prefix + ".weight_scale" in self.ck:
                w = self._fp4(prefix)
            else:
                w = self._dense_nocache(prefix + ".weight")
            self._cache[prefix] = w
        return w

    def _mlp_weights(self, prefix):
        """(gate_proj, up_proj, down_proj) of one expert, not cached."""
        out = []
        for part in ("gate_proj", "up_proj", "down_proj"):
            p = f"{prefix}.{part}"
            out.append(self._fp4(p) if p + ".weight_scale" in self.ck else self._dense_nocache(p + ".weight"))
        return out

    # ---------------- blocks ----------------
    def norm1p(self, x, w):
        ms = np.mean(np.square(x), axis=-1, keepdims=True)
        g = (1.0 + w) if self.var["norm"] == "1p" else w
        return x * (1.0 / np.sqrt(ms + self.eps)) * g

    def embed(self, ids):
        raw = self.ck.raw("model.embed_tokens.weight")
        x = bf16_to_f32(np.ascontiguousarray(raw[np.asarray(ids, np.int64)]))
        return x if self.dt == np.float32 else x.astype(self.dt)

    def rope(self, x, pos):
        """x [n, heads, head_dim]; rotates the first rot_dim dims of every head."""
        r, h = self.rot_dim, self.rot_dim // 2
        if self.var["rope"] == "none":
            return x
        inv = self.theta ** (-np.arange(h, dtype=np.float64) / h)       # theta^(-2i/rot_dim)
        mscale = 1.0
        yf = float(self.var.get("yarn_factor") or 0.0)
        if yf > 1.0:
            # YaRN, as transformers' _compute_yarn_parameters and vLLM's
            # YaRNScalingRotaryEmbedding write it for a plain 1-D rope: interpolate the
            # frequencies by `factor`, keep the high ones (extrapolation) through a linear
            # ramp between the correction band's ends, and scale cos/sin by the attention
            # factor 0.1 ln(factor) + 1.
            orig = float(self.max_pos)
            def corr_dim(rot):
                return (r * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(self.theta))
            low = max(math.floor(corr_dim(32.0)), 0)
            high = min(math.ceil(corr_dim(1.0)), r - 1)
            if low == high:
                high += 0.001
            ramp = np.clip((np.arange(h, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
            extrap = 1.0 - ramp                                          # 1: keep the original frequency
            inv = (inv / yf) * (1.0 - extrap) + inv * extrap
            mscale = 0.1 * math.log(yf) + 1.0
        ang = np.asarray(pos, np.float64)[:, None] * inv[None, :]        # [n, h]
        cos = (np.cos(ang) * mscale).astype(self.dt)[:, None, :]
        sin = (np.sin(ang) * mscale).astype(self.dt)[:, None, :]
        out = x.copy()
        if self.var["rope"] == "half":
            x1, x2 = x[..., :h], x[..., h:r]
            out[..., :h] = x1 * cos - x2 * sin
            out[..., h:r] = x2 * cos + x1 * sin
        else:  # debug variant: adjacent pairs
            x1, x2 = x[..., 0:r:2], x[..., 1:r:2]
            out[..., 0:r:2] = x1 * cos - x2 * sin
            out[..., 1:r:2] = x2 * cos + x1 * sin
        return out

    def attention(self, prefix, h, st_layer, pos0, tap=None, self_only=False):
        """Full attention on a chunk. h [n, H] is the normed input; st_layer holds the K/V of the
        earlier tokens; pos0 is the position of the chunk's first token. self_only (a measurement
        switch for the MTP head) lets every token read only its own key, i.e. no K/V history."""
        n = h.shape[0]
        nh, nkv, hd = self.n_heads, self.n_kv, self.head_dim
        qg = h @ self.linear(prefix + ".q_proj").T                       # [n, 16 * 512]
        if self.var["gate"] == "perhead":
            qg = qg.reshape(n, nh, 2 * hd)
            q, gate = qg[..., :hd], qg[..., hd:]                         # per head [query | gate]
        else:  # debug variant
            q, gate = qg[:, :nh * hd].reshape(n, nh, hd), qg[:, nh * hd:].reshape(n, nh, hd)
        k = (h @ self.linear(prefix + ".k_proj").T).reshape(n, nkv, hd)
        v = (h @ self.linear(prefix + ".v_proj").T).reshape(n, nkv, hd)
        q = self.norm1p(q, self.dense(prefix + ".q_norm.weight"))
        k = self.norm1p(k, self.dense(prefix + ".k_norm.weight"))
        pos = np.arange(pos0, pos0 + n)
        q, k = self.rope(q, pos), self.rope(k, pos)

        if st_layer.get("k") is None:
            K, V = k, v
        else:
            K, V = np.concatenate([st_layer["k"], k]), np.concatenate([st_layer["v"], v])
        st_layer["k"], st_layer["v"] = K, V
        t0 = K.shape[0] - n                                              # keys before this chunk
        # query i (absolute index t0 + i) may read keys 0 .. t0 + i
        mask = np.arange(K.shape[0])[None, :] <= (t0 + np.arange(n))[:, None]
        if self_only:
            mask = np.arange(K.shape[0])[None, :] == (t0 + np.arange(n))[:, None]
        scale = hd ** -0.5
        ctx = np.empty((n, nh, hd), self.dt)
        rep = nh // nkv
        for head in range(nh):
            g = head // rep if self.var["kvmap"] == "div" else head % nkv
            s = (q[:, head] @ K[:, g].T) * scale                         # [n, t0 + n]
            s = np.where(mask, s, -np.inf)
            ctx[:, head] = softmax(s) @ V[:, g]
        out = ctx * sigmoid(gate)
        y = out.reshape(n, nh * hd) @ self.linear(prefix + ".o_proj").T
        if tap is not None:
            tap("attn_q", q)
            tap("attn_k", k)
            tap("attn_v", v)
            tap("attn_gate", gate)
            tap("attn_ctx", ctx)
        return y

    def gdn(self, prefix, h, st_layer, tap=None):
        """Gated DeltaNet on a chunk. h [n, H] is the normed input."""
        n = h.shape[0]
        nk, nv, dk, dv = self.nk, self.nv, self.dk, self.dv
        r = nv // nk                                                     # value heads per key head
        f64 = np.float64

        qkvz = h @ self.linear(prefix + ".in_proj_qkvz").T               # [n, 12288]
        ba = h @ self.linear(prefix + ".in_proj_ba").T                   # [n, 64]
        if self.var["qkvz"] == "interleaved":
            g_ = qkvz.reshape(n, nk, 2 * dk + 2 * r * dv)                # per key-head group
            q = g_[..., :dk].reshape(n, nk * dk)
            k = g_[..., dk:2 * dk].reshape(n, nk * dk)
            v = g_[..., 2 * dk:2 * dk + r * dv].reshape(n, nv * dv)
            z = g_[..., 2 * dk + r * dv:].reshape(n, nv, dv)
        else:  # debug variant
            q, k = qkvz[:, :nk * dk], qkvz[:, nk * dk:2 * nk * dk]
            v = qkvz[:, 2 * nk * dk:2 * nk * dk + nv * dv]
            z = qkvz[:, 2 * nk * dk + nv * dv:].reshape(n, nv, dv)
        if self.var["ba"] == "bbaa":
            g_ = ba.reshape(n, nk, 2 * r)
            b, a = g_[..., :r].reshape(n, nv), g_[..., r:].reshape(n, nv)
        elif self.var["ba"] == "baba":  # debug variant
            g_ = ba.reshape(n, nv, 2)
            b, a = g_[..., 0], g_[..., 1]
        else:  # debug variant
            b, a = ba[:, :nv], ba[:, nv:]

        # depthwise causal conv over time on [q | k | v], then SiLU
        mixed = np.concatenate([q, k, v], axis=-1)                       # [n, 8192]
        cw = self.dense(prefix + ".conv1d.weight").reshape(-1, self.conv_k)   # [8192, 4]
        if self.var["conv"] != "causal":  # debug variant
            cw = cw[:, ::-1]
        tail = st_layer.get("conv")
        if tail is None:
            tail = np.zeros((self.conv_k - 1, mixed.shape[1]), self.dt)
        ext = np.concatenate([tail, mixed])                              # [n + 3, 8192]
        st_layer["conv"] = ext[n:].copy()                                # the last 3 pre-conv inputs
        y = np.zeros_like(mixed)
        for j in range(self.conv_k):
            y += ext[j:j + n] * cw[:, j]
        y = silu(y)
        q = y[:, :nk * dk].reshape(n, nk, dk)
        k = y[:, nk * dk:2 * nk * dk].reshape(n, nk, dk)
        v = y[:, 2 * nk * dk:].reshape(n, nv, dv)

        # the recurrence runs in float64
        q, k, v = q.astype(f64), k.astype(f64), v.astype(f64)
        q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        if tap is not None:
            tap("gdn_q", q)
            tap("gdn_k", k)
            tap("gdn_v", v)
        if self.var["qscale"] == "on":
            q = q * dk ** -0.5
        if self.var["gdnkey"] == "div":
            q, k = np.repeat(q, r, axis=1), np.repeat(k, r, axis=1)      # value head vh <- key head vh // r
        else:  # debug variant
            q, k = np.tile(q, (1, r, 1)), np.tile(k, (1, r, 1))
        beta = sigmoid(b.astype(f64))                                    # [n, 32]
        a_log = self.dense(prefix + ".A_log").astype(f64)
        dt_bias = self.dense(prefix + ".dt_bias").astype(f64)
        g = -np.exp(a_log) * softplus(a.astype(f64) + dt_bias)           # [n, 32] log decay
        decay = np.exp(g)

        S = st_layer.get("S")
        if S is None:
            S = np.zeros((nv, dk, dv), f64)                              # [value head, k dim, v dim]
        o = np.empty((n, nv, dv), f64)
        outer = np.empty_like(S)             # reused: a fresh 4 MB temporary per token costs 25x the maths
        for t in range(n):
            np.multiply(S, decay[t][:, None, None], out=S)               # S *= exp(g)
            kt = k[t]
            kv = np.matmul(kt[:, None, :], S)[:, 0, :]                   # S^T k        [32, 128]
            delta = (v[t] - kv) * beta[t][:, None]
            np.multiply(kt[:, :, None], delta[:, None, :], out=outer)    # outer(k, delta)
            np.add(S, outer, out=S)                                      # S += outer(k, delta)
            o[t] = np.matmul(q[t][:, None, :], S)[:, 0, :]               # S^T q
        st_layer["S"] = S
        if tap is not None:
            tap("gdn_beta", beta)
            tap("gdn_g", g)
            tap("gdn_z", z)
            tap("gdn_core", o)
            tap("gdn_state", S)

        # gated RMS norm per value head (plain weight), SiLU gate from z
        o = o.astype(self.dt)
        w = self.dense(prefix + ".norm.weight")
        if self.var["gdnnorm"] != "plain":  # debug variant
            w = 1.0 + w
        o = o * (1.0 / np.sqrt(np.mean(np.square(o), axis=-1, keepdims=True) + self.eps)) * w * silu(z)
        return o.reshape(n, nv * dv) @ self.linear(prefix + ".out_proj").T

    def mlp(self, h, wg, wu, wd):
        return (silu(h @ wg.T) * (h @ wu.T)) @ wd.T

    def moe(self, prefix, h, tap=None):
        """Sparse MoE on a chunk. h [n, H] is the normed input."""
        n, kk = h.shape[0], self.top_k
        logits = h @ self.dense(prefix + ".gate.weight").T               # [n, 512]
        prob = softmax(logits)
        idx = np.empty((n, kk), np.int64)
        for i in range(n):
            idx[i] = top_desc(prob[i], kk)
        wts = np.take_along_axis(prob, idx, axis=-1)
        if self.norm_topk and self.var["renorm"] == "on":
            wts = wts / wts.sum(axis=-1, keepdims=True)

        out = np.zeros_like(h)
        flat_e = idx.reshape(-1)
        flat_t = np.repeat(np.arange(n), kk)
        flat_w = wts.reshape(-1)
        order = np.argsort(flat_e, kind="stable")
        uniq, start = np.unique(flat_e[order], return_index=True)
        bounds = list(start) + [order.size]
        batch = max(1, 2 * self.threads)
        for b0 in range(0, len(uniq), batch):
            group = [int(e) for e in uniq[b0:b0 + batch]]
            weights = list(self.pool.map(lambda e: self._mlp_weights(f"{prefix}.experts.{e}"), group))
            self.stats["experts_dequantized"] += len(group)
            for j, (wg, wu, wd) in enumerate(weights):
                sel = order[bounds[b0 + j]:bounds[b0 + j + 1]]
                rows = flat_t[sel]                                       # distinct tokens
                out[rows] += flat_w[sel][:, None] * self.mlp(h[rows], wg, wu, wd)
            del weights

        sg = sigmoid(h @ self.dense(prefix + ".shared_expert_gate.weight").T)   # [n, 1]
        shared = sg * self.mlp(h, self.linear(prefix + ".shared_expert.gate_proj"),
                               self.linear(prefix + ".shared_expert.up_proj"),
                               self.linear(prefix + ".shared_expert.down_proj"))
        if tap is not None:
            tap("router_logits", logits)
            tap("router_ids", idx.astype(np.int32))
            tap("router_w", wts)
            tap("moe_routed", out)
            tap("moe_shared", shared)
        return out + shared

    # ---------------- the stack ----------------
    def new_state(self):
        return State(self.total_layers)

    def forward(self, ids, st, tap=None, detail_layers=(), on_layer=None):
        """Feeds a chunk of token ids, advancing the state. Returns the residual stream after the
        last executed layer, [n, H] (before model.norm). tap(name, array) receives h_00 .. h_NN and,
        for the layers in detail_layers, the sublayer tensors as LNN_<name>."""
        n = len(ids)
        x = self.embed(ids)
        if tap is not None:
            tap("h_00", x)
        for L in range(self.n_layers):
            p = f"model.layers.{L}"
            if st.layers[L] is None:
                st.layers[L] = {}
            sub = None
            if tap is not None and L in detail_layers:
                sub = (lambda name, arr, L=L: tap(f"L{L:02d}_{name}", arr))
            h = self.norm1p(x, self.dense(p + ".input_layernorm.weight"))
            if self.is_attn[L]:
                y = self.attention(p + ".self_attn", h, st.layers[L], st.pos, sub)
            else:
                y = self.gdn(p + ".linear_attn", h, st.layers[L], sub)
            if sub is not None:
                sub("mixer_in", h)
                sub("mixer_out", y)
            x = x + y
            h = self.norm1p(x, self.dense(p + ".post_attention_layernorm.weight"))
            m = self.moe(p + ".mlp", h, sub)
            if sub is not None:
                sub("moe_in", h)
                sub("moe_out", m)
            x = x + m
            if tap is not None:
                tap(f"h_{L + 1:02d}", x)
            self.ck.release_pages()
            if on_layer is not None:
                on_layer(L + 1)
        st.pos += n
        return x

    def final_norm(self, x):
        return self.norm1p(x, self.dense("model.norm.weight"))

    def logits(self, x):
        """x [n, H] residual stream (before model.norm) -> logits [n, vocab]."""
        return self.final_norm(x) @ self.dense("lm_head.weight").T

    def logprobs(self, x):
        return log_softmax64(self.logits(x))

    # ---------------- MTP draft head ----------------
    def mtp_new_state(self):
        return {"pos": 0, "attn": {}}

    def mtp_forward(self, next_ids, hidden, st, history=True):
        """The MTP draft head on a chunk of (hidden, next token) pairs.
            x = mtp.fc @ concat[norm1p(embed(next_token), pre_fc_norm_embedding),
                                norm1p(hidden,            pre_fc_norm_hidden)]
        then one full-attention + MoE layer (mtp.layers.0, same maths as a main layer, BF16
        weights), then mtp.norm and the SHARED lm_head.
        Row i pairs hidden[i] (the main model's hidden state at position i) with next_ids[i] (the
        token at position i + 1) and drafts the token at position i + 2.
        `hidden` is whatever the caller's convention says feeds the head (see cmd_mtp): the residual
        stream before model.norm, or the final hidden after it.
        The draft layer's attention has its OWN K/V cache: with history=True row i reads the keys
        of rows 0 .. i (every earlier pair of the sequence), with history=False only its own.
        RoPE positions count the pairs from st["pos"]; only position differences matter.
        Returns log-probabilities [n, vocab]."""
        n = len(next_ids)
        e = self.norm1p(self.embed(next_ids), self.dense("mtp.pre_fc_norm_embedding.weight"))
        hh = self.norm1p(hidden, self.dense("mtp.pre_fc_norm_hidden.weight"))
        x = np.concatenate([e, hh], axis=-1) @ self.linear("mtp.fc").T
        p = "mtp.layers.0"
        h = self.norm1p(x, self.dense(p + ".input_layernorm.weight"))
        x = x + self.attention(p + ".self_attn", h, st["attn"], st["pos"], self_only=not history)
        h = self.norm1p(x, self.dense(p + ".post_attention_layernorm.weight"))
        x = x + self.moe(p + ".mlp", h)
        st["pos"] += n
        self.ck.release_pages()
        x = self.norm1p(x, self.dense("mtp.norm.weight"))
        out = np.empty((n, self.vocab), np.float64)
        for b0 in range(0, n, 32):
            out[b0:b0 + 32] = log_softmax64(x[b0:b0 + 32] @ self.dense("lm_head.weight").T)
        return out


# ----------------------------------------------------------------------------------------------
# Token strings (only for comparing with a capture that stores strings)
# ----------------------------------------------------------------------------------------------
def _byte_decoder():
    """The GPT-2 byte-level alphabet: printable bytes map to themselves, the rest to U+0100.."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class Vocab:
    """id -> the string a server prints for that single token (special tokens print as '')."""

    def __init__(self, ckpt_dir):
        with open(os.path.join(ckpt_dir, "tokenizer.json"), encoding="utf-8") as f:
            tj = json.load(f)
        bd = _byte_decoder()
        self.text = {}
        for s, i in tj["model"]["vocab"].items():
            self.text[i] = bytes(bd[ch] for ch in s).decode("utf-8", errors="replace")
        self.special = set()
        for a in tj.get("added_tokens", []):
            if a.get("special"):
                self.special.add(a["id"])
                self.text[a["id"]] = ""
            else:
                self.text[a["id"]] = a["content"]
        self.by_text = {}
        for i, s in self.text.items():
            self.by_text.setdefault(s, []).append(i)

    def s(self, i):
        return self.text.get(int(i), f"<{int(i)}>")


# ----------------------------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------------------------
def progress(stage, stages, done, total, err, t0):
    print(f"[{stage}/{stages}] {done}/{total} err={err} elapsed={int(time.time() - t0)}s", flush=True)


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)   # Linux: KiB


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
    m = Model(a.ckpt, layers=a.layers, dtype=a.dtype, threads=a.threads, variants=variants)
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
        if stop_eos and model.eos is not None and tok == model.eos:
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
    hn = model.final_norm(x)
    tap("h_final_norm", hn)
    lg = (hn[-1:] @ model.dense("lm_head.weight").T)[0]
    tap("logits_last", lg)
    write_json(os.path.join(a.out, "meta.json"), {
        "ids": ids, "n_tokens": len(ids), "layers": model.n_layers, "dtype": np.dtype(model.dt).name,
        "detail_layers": detail, "files": files, "variants": model.var,
        "notes": {
            "h_NN": "residual stream [T, 2048]: h_00 after the embedding, h_NN after layer NN-1",
            "h_final_norm": "norm1p(h_last, model.norm) [T, 2048]",
            "logits_last": "lm_head @ h_final_norm at the last position [vocab]",
            "LNN_mixer_in": "norm1p(x, input_layernorm)", "LNN_mixer_out": "mixer output before the residual add",
            "LNN_moe_in": "norm1p(x, post_attention_layernorm)", "LNN_moe_out": "MoE output (routed + shared)",
            "LNN_moe_routed": "sum over the top-k experts", "LNN_moe_shared": "sigmoid(shared_expert_gate) * shared expert",
            "LNN_router_logits": "[T, 512]", "LNN_router_ids": "top-k expert ids [T, 10], best first",
            "LNN_router_w": "renormalized top-k weights [T, 10]",
            "LNN_gdn_q / _k": "after conv + SiLU + L2 norm, before the 1/sqrt(128) scale [T, 16, 128]",
            "LNN_gdn_v": "after conv + SiLU [T, 32, 128]", "LNN_gdn_beta": "sigmoid(b) [T, 32]",
            "LNN_gdn_g": "log decay [T, 32]", "LNN_gdn_z": "gate input [T, 32, 128]",
            "LNN_gdn_core": "recurrence output before the gated norm [T, 32, 128]",
            "LNN_gdn_state": "final S [32, 128 (k), 128 (v)]",
            "LNN_attn_q / _k": "after q_norm / k_norm and RoPE [T, 16 | 2, 256]", "LNN_attn_v": "[T, 2, 256]",
            "LNN_attn_gate": "gate before the sigmoid [T, 16, 256]",
            "LNN_attn_ctx": "attention output before the gate [T, 16, 256]"},
        "seconds": round(time.time() - t0, 2), "peak_rss_gb": round(peak_rss_gb(), 2)})
    print(f"# wrote {len(files)} arrays to {a.out}", flush=True)


# ---------------- acceptance test against an Atlas capture ----------------
def _pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compare_echo(row, res, vocab):
    """Atlas echo row vs this reference. Atlas entry i is log P(token i | tokens < i)."""
    ids = row["token_ids"]
    T = len(ids)
    a_lp, a_top = row["token_logprobs"], row["top_logprobs"]
    mine, theirs, pos = [], [], []
    top1_agree = top1_n = ambiguous = 0
    per_pos = []
    for i in range(1, T):
        if i >= len(a_lp) or a_lp[i] is None:
            continue
        mine.append(res["logprobs"][i])
        theirs.append(a_lp[i])
        pos.append(i)
        my_top = res["top"][i][0][0]
        entry = {"pos": i, "id": ids[i], "ref": res["logprobs"][i], "atlas": a_lp[i],
                 "ref_top1": my_top, "ref_rank": res["rank"][i]}
        tl = a_top[i] if i < len(a_top) else None
        if tl:
            # Atlas keys its top list by token STRING, so tokens that print alike (specials print
            # as '') collapse into one key: fewer than 5 keys marks an unreliable entry.
            best_s, best_lp = max(tl.items(), key=lambda kv: kv[1])
            if len(tl) < 5:
                ambiguous += 1
            if a_lp[i] >= best_lp - 1e-6:
                agree = (my_top == ids[i])             # Atlas's top-1 is the actual token
                entry["atlas_top1"] = vocab.s(ids[i])
            else:
                agree = (vocab.s(my_top) == best_s)
                entry["atlas_top1"] = best_s
            top1_n += 1
            top1_agree += int(agree)
            entry["top1_agree"] = bool(agree)
        per_pos.append(entry)
    d = np.abs(np.array(mine) - np.array(theirs))
    out = {"id": row["id"], "kind": "echo", "tokens": T, "compared": len(mine),
           "mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max()),
           "median_abs_diff": float(np.median(d)), "max_abs_diff_pos": int(pos[int(d.argmax())]),
           "pearson": _pearson(mine, theirs),
           "mean_logprob_ref": float(np.mean(mine)), "mean_logprob_atlas": float(np.mean(theirs)),
           "top1_agree": top1_agree, "top1_n": top1_n, "top1_ambiguous_entries": ambiguous,
           "top1_frac": top1_agree / max(1, top1_n)}
    # Atlas's one generated token after the echo (its greedy pick) vs this reference's next-token top-1
    if len(row.get("tokens") or []) == T + 1 and res.get("next_top"):
        out["atlas_next"] = row["tokens"][T]
        out["ref_next"] = vocab.s(res["next_top"][0][0])
        out["next_agree"] = out["atlas_next"] == out["ref_next"]
    out["per_pos"] = per_pos
    return out


def compare_greedy(row, new_ids, steps, vocab):
    """Atlas greedy continuation (token strings) vs this reference's greedy ids."""
    a_tok, a_lp, a_top = row["tokens"], row["token_logprobs"], row["top_logprobs"]
    n = min(len(new_ids), len(a_tok))
    mine = [vocab.s(t) for t in new_ids]
    first = next((j for j in range(n) if mine[j] != a_tok[j]), None)
    out = {"id": row["id"], "kind": "completion", "prompt_tokens": len(row["token_ids"]),
           "generated": len(new_ids), "compared": n, "first_divergence": first,
           "ref_tokens": mine, "atlas_tokens": a_tok[:n], "ref_ids": new_ids}
    # logprob of the chosen token on the common prefix (same context on both sides)
    m = n if first is None else first
    pairs = [(steps[j]["top"][0][1], a_lp[j]) for j in range(m) if a_lp[j] is not None]
    if pairs:
        d = np.abs(np.array([p[0] for p in pairs]) - np.array([p[1] for p in pairs]))
        out["prefix_mean_abs_diff"], out["prefix_max_abs_diff"] = float(d.mean()), float(d.max())
    if first is not None:
        st = steps[first]
        div = {"step": first, "ref_token": mine[first], "atlas_token": a_tok[first],
               "ref_top": [[vocab.s(i), lp] for i, lp in st["top"]], "ref_margin": st["margin"]}
        want = [lp for i, lp in st["top"] if vocab.s(i) == a_tok[first]]
        div["ref_logprob_of_atlas_token"] = want[0] if want else None
        tl = a_top[first] if first < len(a_top) else None
        if tl:
            srt = sorted(tl.items(), key=lambda kv: -kv[1])
            div["atlas_top"] = srt
            if len(srt) > 1:
                div["atlas_margin"] = srt[0][1] - srt[1][1]
        out["divergence"] = div
    return out


def cmd_accept(a):
    model = make_model(a)
    vocab = Vocab(a.ckpt)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    with open(a.refset) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    want = set(a.rows.split(",")) if a.rows else None
    echo = [r for r in rows if r.get("kind") == "echo" and r.get("token_ids")]
    comp = [r for r in rows if r.get("kind") == "completion" and r.get("token_ids")]
    if want is not None:
        todo = [r for r in echo + comp if r["id"] in want]
    else:
        todo = echo + comp[:a.gen_rows]
    t0, err = time.time(), 0
    for n, row in enumerate(todo, 1):
        path = os.path.join(a.out, row["id"] + ".json")
        if os.path.exists(path) and not a.redo:
            progress(n, len(todo), 1, 1, err, t0)
            continue
        try:
            t1 = time.time()
            if row["kind"] == "echo":
                res = score_ids(model, row["token_ids"],
                                on_layer=lambda d: progress(n, len(todo), d, model.n_layers, err, t0)
                                if d % 8 == 0 or d == model.n_layers else None)
                out = compare_echo(row, res, vocab)
                out["ref"] = {k: res[k] for k in ("logprobs", "top", "rank", "next_top")}
            else:
                new, steps = greedy(model, row["token_ids"], a.max_new, stop_eos=True,
                                    on_token=lambda d: progress(n, len(todo), d, a.max_new, err, t0)
                                    if d % 8 == 0 or d == a.max_new else None)
                out = compare_greedy(row, new, steps, vocab)
                out["ref_steps"] = steps
            out["seconds"] = round(time.time() - t1, 2)
            out["peak_rss_gb"] = round(peak_rss_gb(), 2)
            out["layers"], out["dtype"], out["variants"] = model.n_layers, np.dtype(model.dt).name, model.var
            write_json(path, out)
        except Exception as e:  # one bad row must not lose the others
            err += 1
            write_json(path + ".error", {"id": row["id"], "error": f"{type(e).__name__}: {e}"})
            progress(n, len(todo), 0, 1, err, t0)
    summarize(a.out, [r["id"] for r in todo])
    print("DONE", flush=True)
    return 1 if err else 0


def summarize(out_dir, ids):
    table = []
    for i in ids:
        p = os.path.join(out_dir, i + ".json")
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            table.append({k: v for k, v in d.items() if k not in ("per_pos", "ref", "ref_steps")})
    write_json(os.path.join(out_dir, "summary.json"), table)
    print(f"{'row':<16}{'tok':>5}{'cmp':>5}{'mean|d|':>9}{'med|d|':>9}{'max|d|':>9}{'pearson':>9}{'top1':>10}{'next':>6}{'sec':>7}")
    for d in table:
        if d["kind"] != "echo":
            continue
        nxt = "-" if "next_agree" not in d else ("same" if d["next_agree"] else "DIFF")
        print(f"{d['id']:<16}{d['tokens']:>5}{d['compared']:>5}{d['mean_abs_diff']:>9.4f}{d['median_abs_diff']:>9.4f}"
              f"{d['max_abs_diff']:>9.4f}{d['pearson']:>9.5f}{d['top1_agree']:>6}/{d['top1_n']:<3}{nxt:>6}{d['seconds']:>7.0f}")
    for d in table:
        if d["kind"] != "completion":
            continue
        fd = d["first_divergence"]
        msg = f"all {d['compared']} tokens equal" if fd is None else f"first divergence at step {fd}"
        if fd is not None:
            v = d["divergence"]
            msg += (f": ref {v['ref_token']!r} vs atlas {v['atlas_token']!r}, ref margin {v['ref_margin']:.4f}"
                    f", atlas margin {v.get('atlas_margin', float('nan')):.4f}")
        print(f"{d['id']:<16} greedy {d['generated']} tokens: {msg}")


# ---------------- MTP ----------------
MTP_HIDDEN = (("pre_norm", "residual stream of the last layer BEFORE model.norm"),
              ("post_norm", "final hidden AFTER model.norm"))


def cmd_mtp(a):
    """How often the MTP head's greedy draft equals the main model's own greedy next token.
    For every position t of a text the head gets embed(token t+1) and the main model's hidden
    state at t and drafts position t+2; that is compared with the main model's argmax for
    position t+2 (its row t+1). Measured for both conventions of "hidden state" (before / after
    model.norm), each with and without the draft layer's own K/V history over the earlier pairs."""
    model = make_model(a)
    t0 = time.time()
    jobs = []
    if a.refset:
        with open(a.refset) as f:
            for line in f:
                r = json.loads(line) if line.strip() else None
                if r and r.get("kind") == "echo" and r.get("token_ids"):
                    jobs.append((r["id"], r["token_ids"]))
    else:
        jobs.append((os.path.basename(a.ids_json), load_ids(a.ids_json)))
    keys = [f"{hid}/{'history' if hist else 'no_history'}" for hid, _ in MTP_HIDDEN for hist in (True, False)]
    results = []
    for n, (name, ids) in enumerate(jobs, 1):
        x, _ = run_forward(model, ids, on_layer=lambda d: progress(n, len(jobs), d, model.n_layers, 0, t0)
                           if d % 8 == 0 or d == model.n_layers else None)
        T = len(ids)
        hn = model.final_norm(x)
        main_top = np.empty(T, np.int64)                       # main_top[i]: greedy token for position i + 1
        for b0 in range(0, T, 32):
            main_top[b0:b0 + 32] = (hn[b0:b0 + 32] @ model.dense("lm_head.weight").T).argmax(axis=-1)
        actual = np.array(ids[2:], np.int64)
        r = {"id": name, "tokens": T, "drafts": T - 1,
             "main_eq_text": int(np.count_nonzero(main_top[1:-1] == actual)), "text_n": int(actual.size)}
        for hid, hidden in (("pre_norm", x), ("post_norm", hn)):
            for hist in (True, False):
                lp = model.mtp_forward(ids[1:], hidden[:-1], model.mtp_new_state(), history=hist)
                draft = lp.argmax(axis=-1)                     # draft[i]: position i + 2
                key = f"{hid}/{'history' if hist else 'no_history'}"
                hit = int(np.count_nonzero(draft == main_top[1:]))
                r[key] = {"draft_eq_main": hit, "frac": hit / max(1, draft.size),
                          "draft_eq_text": int(np.count_nonzero(draft[:-1] == actual)),
                          "mean_logprob_of_main_token": float(lp[np.arange(draft.size), main_top[1:]].mean())}
        results.append(r)
        print(json.dumps(r), flush=True)
        if a.out:
            write_json(a.out, results)
    tot = sum(r["drafts"] for r in results)
    print(f"{'text':<16}{'drafts':>7}" + "".join(f"{k:>22}" for k in keys))
    for r in results:
        print(f"{r['id']:<16}{r['drafts']:>7}" + "".join(f"{r[k]['draft_eq_main']:>14} {100 * r[k]['frac']:5.1f}% " for k in keys))
    print(f"{'ALL':<16}{tot:>7}" + "".join(
        f"{sum(r[k]['draft_eq_main'] for r in results):>14} {100 * sum(r[k]['draft_eq_main'] for r in results) / max(1, tot):5.1f}% "
        for k in keys))
    print(f"# peak_rss_gb={peak_rss_gb():.2f} seconds={int(time.time() - t0)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Qwen3-Next-80B-A3B (NVFP4) host reference forward pass, pure numpy")
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p, ids=True):
        if ids:
            p.add_argument("--ids-json", required=True, help="JSON list of token ids")
        p.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint snapshot directory")
        p.add_argument("--layers", type=int, default=None, help="run only the first N layers")
        p.add_argument("--threads", type=int, default=_THREADS,
                       help="total thread budget, half BLAS and half expert dequant (default 8)")
        p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
        p.add_argument("--variant", action="append", help="debug: flip one layout assumption, key=value")

    p = sub.add_parser("score", help="teacher-forced log-probabilities")
    common(p)
    p.add_argument("--out", help="write the result as JSON")
    p.add_argument("--chunk", type=int, default=None, help="feed the sequence N tokens at a time (state check)")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("generate", help="greedy continuation")
    common(p)
    p.add_argument("--max-new", type=int, required=True)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--out", help="write ids, per-step top-5 and margins as JSON")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("dump", help="save the residual stream and sublayer tensors as .npy")
    common(p)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--detail-layers", default="0,3", help="layers whose sublayer tensors are saved")
    p.set_defaults(fn=cmd_dump)

    p = sub.add_parser("accept", help="acceptance test against an Atlas capture (.jsonl)")
    common(p, ids=False)
    p.add_argument("--refset", required=True)
    p.add_argument("--out", required=True, help="output directory, one JSON per row")
    p.add_argument("--rows", help="comma-separated row ids (default: every echo row + the first --gen-rows completions)")
    p.add_argument("--gen-rows", type=int, default=3)
    p.add_argument("--max-new", type=int, default=24)
    p.add_argument("--redo", action="store_true", help="recompute rows whose output exists")
    p.set_defaults(fn=cmd_accept)

    p = sub.add_parser("mtp", help="MTP draft head vs the main model's next token")
    common(p, ids=False)
    p.add_argument("--ids-json")
    p.add_argument("--refset", help="use every echo row of this capture")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_mtp)

    a = ap.parse_args()
    if a.mode == "mtp" and not (a.ids_json or a.refset):
        ap.error("mtp needs --ids-json or --refset")
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
