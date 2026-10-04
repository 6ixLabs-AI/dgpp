#!/usr/bin/env python3
"""qwen3_reference.py — host reference forward pass for the PLAIN Qwen3 family (2026-10-04): the
Qwen3 architecture without Gated DeltaNet — full attention in every layer, then a dense SwiGLU MLP
or a routed MoE. One file for the three model classes and their containers:

  Qwen3VLMoeForConditionalGeneration   ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4 (compressed-tensors
                                       NVFP4). TEXT PATH ONLY: the vision tower is not run, and a
                                       prompt that carries an image or video placeholder is refused.
  Qwen3MoeForCausalLM                  nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4 (modelopt NVFP4)
  Qwen3ForCausalLM                     Qwen/Qwen3-Reranker-0.6B, Qwen/Qwen3-Embedding-0.6B (BF16)

Pure Python + numpy, reading the checkpoint directly (safetensors headers parsed by hand, shards
mmap'd). It is the ground truth the engine port is compared against, so it is written for clarity
and checkability, not speed: weights dequantized to float, activations in full precision (the
recipes' `input_scale` / `input_global_scale` activation scales and modelopt's `k_scale` /
`v_scale` K/V-cache scales are ignored). It is held to transformers' own classes by
tools/qwen3_synth.py (`torch-check`: Qwen3ForCausalLM / Qwen3MoeForCausalLM / the Qwen3-VL-MoE
text model on a synthetic checkpoint).

  score    --ids-json F [--out F.json]   teacher-forced log P(ids[i] | ids[<i]) + top-5 per position
  generate --ids-json F --max-new N      greedy continuation, prints the new token ids
  dump     --ids-json F --out DIR        .npy of the residual stream after every layer (+ sublayers)
  retrieve --ids-json F [--yes N --no N] the retrieval models' two read-outs of ONE prefill: the
                                         last position's hidden state after model.norm (the
                                         embedding, raw and L2-normalized) and its yes/no logits
  all modes: --ckpt DIR, --layers N (first N layers only), --threads N, --dtype float32|float64

THE MODEL (names are checkpoint tensor names; P is the model prefix: "model.language_model." in
the VL release, "model." in the others, "" in Qwen3-Embedding, which stores the base model)

  x = P.embed_tokens[ids]                                 no scaling
  per layer L:  x += attn(norm(x, input_layernorm))
                x += mlp(norm(x, post_attention_layernorm))
  logits = lm_head @ norm(x, P.norm)                      lm_head = P.embed_tokens when tied

  norm(x, w) = x * rsqrt(mean(x^2) + eps) * w             a PLAIN weight (Qwen3-Next's is 1 + w)

  Attention (heads query heads, kv KV heads, head_dim 128):
    q_proj -> [heads, 128];  k_proj, v_proj -> [kv, 128]          no biases, no output gate
    q = norm(q, q_norm), k = norm(k, k_norm)                      per head, one weight for all heads
    RoPE on ALL 128 dims of q and k: pairs (i, i + 64), angle = pos * theta^(-i/64)
    causal softmax attention, scale 1/sqrt(128), query head h reads KV head h // (heads / kv)
    flatten heads x 128; o_proj

    The VL text model builds its rotary from three position axes (MROPE, interleaved sections
    [24, 20, 20]); a text-only prompt puts the same position on all three, which is exactly the
    1-D rope above (transformers' get_rope_index without images).

  Dense MLP:   down_proj @ (silu(gate_proj @ h) * (up_proj @ h))
  Routed MoE (no shared expert, no router bias):
    p = softmax(mlp.gate @ h) over all experts; top-k; weights = p[top] / sum(p[top])
    y = sum_e w_e * down_e @ (silu(gate_e @ h) * (up_e @ h))

  NVFP4 (modelopt): `.weight` U8 [N, K/2], two E2M1 codes per byte, LOW nibble = EVEN column;
    `.weight_scale` F8_E4M3 [N, K/16]; `.weight_scale_2` F32 scalar:
        W[n, k] = e2m1(code[n, k]) * e4m3(weight_scale[n, k // 16]) * weight_scale_2
    E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 (bit 3 = sign).
    The 235B release quantizes the routed experts and o_proj; q/k/v, the router and the head are BF16.
  NVFP4 (compressed-tensors `nvfp4-pack-quantized`): the same codes and block scales under other
    names — `.weight_packed` U8 [N, K/2], `.weight_scale` F8_E4M3 [N, K/16] — and
    `.weight_global_scale` F32 [1], a DIVISOR:
        W[n, k] = e2m1(code[n, k]) * (e4m3(weight_scale[n, k // 16]) / weight_global_scale)
    The VL-30B release quantizes q/k/v/o and the routed experts; the router, the head and the
    vision tower are BF16. Its experts are per-expert Linears: llm-compressor unpacks transformers'
    fused `gate_up_proj` [E, H, 2I] into gate_proj = gate_up[:, :I].T and up_proj = gate_up[:, I:].T.

Arithmetic is float32 by default and the final log-softmax float64. The RoPE angles are float32 —
transformers computes them in float32 whatever the model's dtype, and so does the engine's kernel;
`--variant ropeangle=f64` takes the exact angle instead. `--dtype float64` runs everything else in
float64 (a noise-floor check).

Memory: the norms, routers and the head are converted once and kept; the attention (and dense-MLP)
matrices are kept only by `generate`, which revisits every layer per token (27 GB of float32 on the
235B); routed experts are dequantized on demand and dropped. The mmap'd shard pages are released
after every layer.
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
# here before numpy loads, and the expert-dequant workers.
_THREADS = _early_threads()
_BLAS_THREADS = max(1, _THREADS // 2)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(_BLAS_THREADS))

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import mmap  # noqa: E402
import resource  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

assert sys.byteorder == "little", "the BF16 / NVFP4 decoders assume a little-endian host"

# ----------------------------------------------------------------------------------------------
# Debug variants. Every layout assumption can be flipped in isolation with `--variant key=value`
# (repeatable). The defaults are what transformers' classes do (tools/qwen3_synth.py torch-check);
# the container ones (nibble, ws2, gscale, gateup) are the 80B / Coder-Next ports' reading and are
# only PROVEN on a real checkpoint: the wrong value must give garbage there.
# ----------------------------------------------------------------------------------------------
VARIANT_DEFAULTS = {
    "nibble": "lo",        # lo: low nibble = even column | hi: high nibble = even column
    "ws2": "mul",          # mul: W = e2m1 * e4m3 * weight_scale_2 | div: ... / weight_scale_2
    "gscale": "div",       # div: W = e2m1 * (e4m3 / weight_global_scale) | mul: ... * weight_global_scale
    "rope": "half",        # half: pairs (i, i+64) | adjacent: pairs (2i, 2i+1) | none
    # Not a layout assumption: the precision the rotary ANGLES are computed in. f32 is transformers'
    # own arithmetic in every model dtype (float32 inv_freq, float32 pos * inv_freq, float32 cos / sin
    # — the engine's kernel does the same); f64 is the exact angle (the unit-test vectors use it: a
    # float32 pow is not the same bit on every libm).
    "ropeangle": "f32",
    "kvmap": "div",        # div: query head h -> KV head h // (heads / kv) | mod: h % kv
    "norm": "plain",       # plain: w | 1p: (1 + w)             (layer / q / k / final norms)
    "qknorm": "on",        # on: q_norm / k_norm applied | off
    "renorm": "on",        # on: top-k router weights renormalized | off
    "gateup": "gate",      # gate: silu(gate_proj h) * (up_proj h) | up: silu(up_proj h) * (gate_proj h)
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


# ----------------------------------------------------------------------------------------------
# Elementwise maths (dtype-preserving)
# ----------------------------------------------------------------------------------------------
def sigmoid(x):
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def silu(x):
    return x * sigmoid(x)


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
    # argpartition may split a tie group at the boundary: every entry equal to the k-th value is a
    # candidate, and the lower index wins.
    kth = row[idx].min()
    cand = np.flatnonzero(row >= kth)
    order = cand[np.lexsort((cand, -row[cand]))]
    return order[:k]


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
    """Per-sequence state (every layer's K/V), so a sequence can be fed in chunks or token by token."""

    def __init__(self, n_layers):
        self.pos = 0                      # tokens consumed so far
        self.layers = [None] * n_layers   # {"k", "v"}


class Model:
    def __init__(self, ckpt_dir, layers=None, dtype="float32", threads=_THREADS, variants=None, keep_linear=True):
        self.ck = Checkpoint(ckpt_dir)
        # Whether the attention / dense-MLP matrices stay converted after their layer ran. A
        # generation revisits every layer per token and wants them kept; a single scoring pass does
        # not, and on the 235B they are 27 GB of float32.
        self.keep_linear = keep_linear
        root = self.ck.cfg
        mt = root.get("model_type")
        if mt == "qwen3_vl_moe":
            c = root["text_config"]
            self.prefix = "model.language_model."
            # A prompt carrying one of these asks for the vision tower, which this file does not run.
            self.vision_ids = {int(root[k]) for k in ("image_token_id", "video_token_id") if root.get(k) is not None}
        elif mt in ("qwen3_moe", "qwen3"):
            c = root
            # Qwen3-Embedding stores the base model's state dict: no "model." prefix, no head.
            self.prefix = "model." if "model.embed_tokens.weight" in self.ck else ""
            self.vision_ids = set()
        else:
            raise ValueError(f"model_type {mt!r}: this reference covers qwen3, qwen3_moe and qwen3_vl_moe")
        self.model_type = mt
        self.tcfg = c
        self.dt = np.dtype(dtype).type
        self.var = dict(VARIANT_DEFAULTS)
        self.var.update(variants or {})
        self.H = c["hidden_size"]
        self.vocab = c["vocab_size"]
        self.eps = float(c["rms_norm_eps"])
        self.total_layers = c["num_hidden_layers"]
        self.n_layers = self.total_layers if layers is None else max(0, min(int(layers), self.total_layers))
        eos = c.get("eos_token_id")
        self.eos = eos[0] if isinstance(eos, list) and eos else eos
        if c.get("use_sliding_window"):
            raise ValueError("use_sliding_window: sliding-window attention is not part of this reference")
        if c.get("attention_bias"):
            raise ValueError("attention_bias: biased projections are not part of this reference")
        if c.get("hidden_act", "silu") != "silu":
            raise ValueError(f"hidden_act {c.get('hidden_act')!r}: only silu")
        # attention
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c["num_key_value_heads"]
        self.head_dim = c.get("head_dim") or self.H // self.n_heads
        self.theta = float(c["rope_theta"])
        rs = c.get("rope_scaling")
        if rs is not None and rs.get("rope_type", rs.get("type", "default")) != "default":
            raise ValueError(f"rope_scaling {rs!r}: only the unscaled rope")
        # mlp
        self.n_exp = int(c.get("num_experts") or 0) if mt != "qwen3" else 0
        self.top_k = int(c.get("num_experts_per_tok") or 0)
        # Qwen3MoeSparseMoeBlock honours the field; the VL block renormalizes unconditionally.
        self.norm_topk = True if mt == "qwen3_vl_moe" else bool(c.get("norm_topk_prob", False))
        if self.n_exp and (c.get("decoder_sparse_step", 1) != 1 or c.get("mlp_only_layers")):
            raise ValueError("decoder_sparse_step / mlp_only_layers: every layer is a routed MoE here")
        # the head: lm_head.weight when stored, the embedding when tied
        tied = bool(root.get("tie_word_embeddings", c.get("tie_word_embeddings", False)))
        self.head_name = "lm_head.weight" if ("lm_head.weight" in self.ck and not tied) else self.prefix + "embed_tokens.weight"
        if not tied and "lm_head.weight" not in self.ck:
            raise ValueError("lm_head.weight is absent and the embeddings are not tied")

        self._cache = {}
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
        return w if self.dt == np.float32 else w.astype(self.dt)

    def _fp4(self, prefix):
        """An NVFP4 linear dequantized to a dense float [N, K] matrix (not cached), from either
        container: modelopt's `.weight` + `.weight_scale_2` (a multiplier) or compressed-tensors'
        `.weight_packed` + `.weight_global_scale` (a divisor)."""
        packed = prefix + ".weight_packed" in self.ck
        codes = self.ck.raw(prefix + (".weight_packed" if packed else ".weight"))   # U8 [N, K/2]
        scale = self.ck.raw(prefix + ".weight_scale")        # E4M3    [N, K/16]
        n, half = codes.shape
        w = self._pair[codes].view(np.float32)               # [N, K]: byte j -> columns 2j, 2j+1
        s = _E4M3[scale]                                     # [N, K/16]
        if packed:
            g = np.float32(self.ck.raw(prefix + ".weight_global_scale").reshape(()))
            s = s / g if self.var["gscale"] == "div" else s * g
        else:
            ws2 = np.float32(self.ck.raw(prefix + ".weight_scale_2").reshape(()))
            s = s * ws2 if self.var["ws2"] == "mul" else s / ws2
        w = w.reshape(n, half // 8, 16)
        w *= s[:, :, None]
        w = w.reshape(n, 2 * half)
        return w if self.dt == np.float32 else w.astype(self.dt)

    def linear_nocache(self, prefix):
        return self._fp4(prefix) if prefix + ".weight_scale" in self.ck else self._dense_nocache(prefix + ".weight")

    def linear(self, prefix):
        """Weight of a linear layer, whichever way the checkpoint stores it (kept)."""
        w = self._cache.get(prefix)
        if w is None:
            w = self.linear_nocache(prefix)
            if self.keep_linear:
                self._cache[prefix] = w
        return w

    def _mlp_weights(self, prefix):
        """(gate_proj, up_proj, down_proj) of one MLP, not cached."""
        return [self.linear_nocache(f"{prefix}.{part}") for part in ("gate_proj", "up_proj", "down_proj")]

    # ---------------- blocks ----------------
    def norm(self, x, w):
        ms = np.mean(np.square(x), axis=-1, keepdims=True)
        g = (1.0 + w) if self.var["norm"] == "1p" else w
        return x * (1.0 / np.sqrt(ms + self.eps)) * g

    def embed(self, ids):
        ids = np.asarray(ids, np.int64)
        if self.vision_ids and np.isin(ids, list(self.vision_ids)).any():
            raise ValueError("the prompt carries an image / video placeholder: the vision tower is not part of "
                             "this reference (text path only)")
        if ids.size and (ids.min() < 0 or ids.max() >= self.vocab):
            raise ValueError("token id outside [0, vocab_size)")
        raw = self.ck.raw(self.prefix + "embed_tokens.weight")
        x = bf16_to_f32(np.ascontiguousarray(raw[ids]))
        return x if self.dt == np.float32 else x.astype(self.dt)

    def rope(self, x, pos):
        """x [n, heads, head_dim]; rotates ALL head_dim dims of every head."""
        r, h = self.head_dim, self.head_dim // 2
        if self.var["rope"] == "none":
            return x
        if self.var["ropeangle"] == "f32":
            # transformers: inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim)), then
            # (inv_freq.float() @ position_ids.float()).cos() — all float32, whatever the model dtype.
            inv = (np.float32(1.0) / (np.float32(self.theta) ** (np.arange(0, r, 2, dtype=np.float32) / np.float32(r))))
            ang = np.asarray(pos, np.float32)[:, None] * inv.astype(np.float32)[None, :]
        else:
            inv = self.theta ** (-np.arange(h, dtype=np.float64) / h)   # theta^(-2i/head_dim)
            ang = np.asarray(pos, np.float64)[:, None] * inv[None, :]    # [n, h]
        cos = np.cos(ang).astype(self.dt)[:, None, :]
        sin = np.sin(ang).astype(self.dt)[:, None, :]
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

    def attention(self, prefix, h, st_layer, pos0, tap=None):
        """Full attention on a chunk. h [n, H] is the normed input; st_layer holds the K/V of the
        earlier tokens; pos0 is the position of the chunk's first token."""
        n = h.shape[0]
        nh, nkv, hd = self.n_heads, self.n_kv, self.head_dim
        q = (h @ self.linear(prefix + ".q_proj").T).reshape(n, nh, hd)
        k = (h @ self.linear(prefix + ".k_proj").T).reshape(n, nkv, hd)
        v = (h @ self.linear(prefix + ".v_proj").T).reshape(n, nkv, hd)
        if self.var["qknorm"] == "on":
            q = self.norm(q, self.dense(prefix + ".q_norm.weight"))
            k = self.norm(k, self.dense(prefix + ".k_norm.weight"))
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
        scale = hd ** -0.5
        ctx = np.empty((n, nh, hd), self.dt)
        rep = nh // nkv
        for head in range(nh):
            g = head // rep if self.var["kvmap"] == "div" else head % nkv
            s = (q[:, head] @ K[:, g].T) * scale                         # [n, t0 + n]
            s = np.where(mask, s, -np.inf)
            ctx[:, head] = softmax(s) @ V[:, g]
        y = ctx.reshape(n, nh * hd) @ self.linear(prefix + ".o_proj").T
        if tap is not None:
            tap("attn_q", q)
            tap("attn_k", k)
            tap("attn_v", v)
            tap("attn_ctx", ctx)
        return y

    def mlp(self, h, wg, wu, wd):
        if self.var["gateup"] == "gate":
            return (silu(h @ wg.T) * (h @ wu.T)) @ wd.T
        return (silu(h @ wu.T) * (h @ wg.T)) @ wd.T          # debug variant

    def dense_mlp(self, prefix, h):
        return self.mlp(h, self.linear(prefix + ".gate_proj"), self.linear(prefix + ".up_proj"),
                        self.linear(prefix + ".down_proj"))

    def route(self, prefix, h):
        """(router logits [n, E], top-k ids [n, k] best first, weights [n, k])."""
        n, kk = h.shape[0], self.top_k
        logits = h @ self.dense(prefix + ".gate.weight").T
        prob = softmax(logits)
        idx = np.empty((n, kk), np.int64)
        for i in range(n):
            idx[i] = top_desc(prob[i], kk)
        wts = np.take_along_axis(prob, idx, axis=-1)
        if self.norm_topk and self.var["renorm"] == "on":
            wts = wts / wts.sum(axis=-1, keepdims=True)
        return logits, idx, wts

    def moe(self, prefix, h, tap=None):
        """Routed MoE on a chunk (no shared expert). h [n, H] is the normed input."""
        n, kk = h.shape[0], self.top_k
        logits, idx, wts = self.route(prefix, h)
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
                out[rows] += flat_w[sel][:, None].astype(self.dt) * self.mlp(h[rows], wg, wu, wd)
            del weights
        if tap is not None:
            tap("router_logits", logits)
            tap("router_ids", idx.astype(np.int32))
            tap("router_w", wts)
        return out

    # ---------------- the stack ----------------
    def new_state(self):
        return State(self.total_layers)

    def forward(self, ids, st, tap=None, detail_layers=(), on_layer=None):
        """Feeds a chunk of token ids, advancing the state. Returns the residual stream after the
        last executed layer, [n, H] (before the final norm). tap(name, array) receives h_00 .. h_NN
        and, for the layers in detail_layers, the sublayer tensors as LNN_<name>."""
        n = len(ids)
        x = self.embed(ids)
        if tap is not None:
            tap("h_00", x)
        for L in range(self.n_layers):
            p = f"{self.prefix}layers.{L}"
            if st.layers[L] is None:
                st.layers[L] = {}
            sub = None
            if tap is not None and L in detail_layers:
                sub = (lambda name, arr, L=L: tap(f"L{L:02d}_{name}", arr))
            h = self.norm(x, self.dense(p + ".input_layernorm.weight"))
            y = self.attention(p + ".self_attn", h, st.layers[L], st.pos, sub)
            if sub is not None:
                sub("attn_in", h)
                sub("attn_out", y)
            x = x + y
            h = self.norm(x, self.dense(p + ".post_attention_layernorm.weight"))
            m = self.moe(p + ".mlp", h, sub) if self.n_exp else self.dense_mlp(p + ".mlp", h)
            if sub is not None:
                sub("mlp_in", h)
                sub("mlp_out", m)
            x = x + m
            if tap is not None:
                tap(f"h_{L + 1:02d}", x)
            self.ck.release_pages()
            if on_layer is not None:
                on_layer(L + 1)
        st.pos += n
        return x

    def final_norm(self, x):
        return self.norm(x, self.dense(self.prefix + "norm.weight"))

    def logits(self, x):
        """x [n, H] residual stream (before the final norm) -> logits [n, vocab]."""
        return self.final_norm(x) @ self.dense(self.head_name).T

    def logprobs(self, x):
        return log_softmax64(self.logits(x))


# ----------------------------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------------------------
def progress(stage, stages, done, total, err, t0):
    print(f"[{stage}/{stages}] {done}/{total} err={err} elapsed={int(time.time() - t0)}s", flush=True)


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 ** 3) if sys.platform == "darwin" else r / (1024.0 * 1024.0)   # macOS: bytes; Linux: KiB


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
    # One pass over the stack (score, dump, retrieve without --chunk) keeps no layer's matrices.
    one_pass = a.mode != "generate" and not getattr(a, "chunk", None)
    m = Model(a.ckpt, layers=a.layers, dtype=a.dtype, threads=a.threads, variants=variants, keep_linear=not one_pass)
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
    lg = (hn[-1:] @ model.dense(model.head_name).T)[0]
    tap("logits_last", lg)
    write_json(os.path.join(a.out, "meta.json"), {
        "ids": ids, "n_tokens": len(ids), "layers": model.n_layers, "dtype": np.dtype(model.dt).name,
        "detail_layers": detail, "files": files, "variants": model.var,
        "notes": {
            "h_NN": "residual stream [T, H]: h_00 after the embedding, h_NN after layer NN-1",
            "h_final_norm": "norm(h_last, norm.weight) [T, H]",
            "logits_last": "lm_head @ h_final_norm at the last position [vocab]",
            "LNN_attn_in": "norm(x, input_layernorm)", "LNN_attn_out": "o_proj output before the residual add",
            "LNN_mlp_in": "norm(x, post_attention_layernorm)", "LNN_mlp_out": "dense MLP / routed MoE output",
            "LNN_router_logits": "[T, E]", "LNN_router_ids": "top-k expert ids [T, k], best first",
            "LNN_router_w": "renormalized top-k weights [T, k]",
            "LNN_attn_q / _k": "after q_norm / k_norm and RoPE [T, heads | kv, 128]", "LNN_attn_v": "[T, kv, 128]",
            "LNN_attn_ctx": "attention output before o_proj [T, heads, 128]"},
        "seconds": round(time.time() - t0, 2), "peak_rss_gb": round(peak_rss_gb(), 2)})
    print(f"# wrote {len(files)} arrays to {a.out}", flush=True)


def retrieve(model, ids, yes_id=None, no_id=None):
    """The two read-outs the retrieval models take from ONE prefill, both at the LAST position:
        embedding  norm(h_last, norm.weight)               Qwen3-Embedding: last-token pooling, then
                                                           L2-normalized (1_Pooling + 2_Normalize)
        score      logits[yes], logits[no] of that position  Qwen3-Reranker: P(yes) = softmax over
                                                           the two (1_LogitScore: true 9693, false 2152)
    """
    x, _ = run_forward(model, ids)
    hn = model.final_norm(x[-1:])[0].astype(np.float64)
    out = {"n_tokens": len(ids), "embedding": hn.tolist(),
           "embedding_l2": (hn / max(np.linalg.norm(hn), 1e-12)).tolist()}
    if yes_id is not None and no_id is not None:
        lg = (hn.astype(model.dt)[None, :] @ model.dense(model.head_name)[[yes_id, no_id]].T)[0].astype(np.float64)
        out.update({"yes_logit": float(lg[0]), "no_logit": float(lg[1]),
                    "p_yes": float(1.0 / (1.0 + np.exp(lg[1] - lg[0]))), "logit_diff": float(lg[0] - lg[1])})
    return out


def cmd_retrieve(a):
    ids = load_ids(a.ids_json)
    model = make_model(a)
    res = retrieve(model, ids, a.yes, a.no)
    if a.out:
        write_json(a.out, res)
    e = np.array(res["embedding"])
    print(f"# tokens={len(ids)} |embedding|={np.linalg.norm(e):.6f} first8={np.round(np.array(res['embedding_l2'][:8]), 6).tolist()}")
    if "yes_logit" in res:
        print(f"yes {res['yes_logit']:.5f} no {res['no_logit']:.5f} p_yes {res['p_yes']:.6f}")


def main():
    ap = argparse.ArgumentParser(description="plain Qwen3 / Qwen3-MoE / Qwen3-VL-MoE (text) host reference, pure numpy")
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--ids-json", required=True, help="JSON list of token ids")
        p.add_argument("--ckpt", required=True, help="checkpoint snapshot directory")
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
    p.add_argument("--detail-layers", default="0", help="layers whose sublayer tensors are saved")
    p.set_defaults(fn=cmd_dump)

    p = sub.add_parser("retrieve", help="last-position embedding and yes/no logits (the retrieval models)")
    common(p)
    p.add_argument("--yes", type=int, default=None, help="the 'yes' token id (Qwen3-Reranker: 9693)")
    p.add_argument("--no", type=int, default=None, help="the 'no' token id (Qwen3-Reranker: 2152)")
    p.add_argument("--out", help="write the result as JSON")
    p.set_defaults(fn=cmd_retrieve)

    a = ap.parse_args()
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
