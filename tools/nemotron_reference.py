#!/usr/bin/env python3
"""nemotron_reference.py — host reference forward pass for NVIDIA Nemotron-3 (NemotronHForCausalLM):
Nano 30B-A3B and Super 120B-A12B, the NVFP4 releases (2026-10-04).

Pure Python + numpy, reading the checkpoint directly (safetensors headers parsed by hand, shards
mmap'd). It is the ground truth the DGPP engine port is compared against, so it is written for
clarity and checkability, not speed: one straightforward implementation of every block, weights
dequantized to float, activations in full precision (the checkpoint's `input_scale` activation-quant
scales and the `k_scale` / `v_scale` KV-cache scales are ignored). Transcribed from the release's own
modeling_nemotron_h.py (remote code in the snapshot directory).

  score    --ids-json F [--out F.json]   teacher-forced log P(ids[i] | ids[<i]) + top-5 per position
  generate --ids-json F --max-new N      greedy continuation, prints the new token ids
  dump     --ids-json F --out DIR        .npy of the residual stream after every layer (+ sublayers)
  mtp      --ids-json F                  Super's draft block vs the main model's own next token
  all modes: --ckpt DIR, --layers N (first N layers only), --threads N, --dtype float32|float64,
             --no-keep (drop every dequantized weight after its layer: Super on a small box)

THE MODEL (what this file computes; names are checkpoint tensor names under `backbone.`)

  x = embeddings[ids]                                     no scaling
  per layer L:  x += mixer(rms(x, layers.L.norm))         ONE mixer per layer, its kind the L-th
                                                          character of hybrid_override_pattern:
                                                          M Mamba2, E MoE, * attention
  logits = lm_head @ rms(x, norm_f)                       lm_head is untied

  rms(x, w) = x * rsqrt(mean(x^2) + eps) * w              plain weight, eps = layer_norm_epsilon (1e-5)

  Mamba2 (heads x head_dim = inner; n_groups groups; ssm_state_size N; conv kernel 4):
    in_proj -> [gate z (inner) | xBC (inner + 2*groups*N) | dt (heads)]
    xBC: depthwise causal conv over time with bias, then SiLU
         out[t] = b[c] + sum_j w[c, j] * in[t - 3 + j]    (w[c, 3] multiplies the current token)
    split -> x [heads, head_dim], B [groups, N], C [groups, N]; head h uses group h // (heads/groups)
    dt = softplus(dt + dt_bias);  dA = exp(-exp(A_log) * dt)                 (per head, no clamp)
    state S[h] is [head_dim, N], zero at the start:
        S[h] = S[h] * dA[h] + outer(dt[h] * x[h], B[g]);   y[h] = S[h] @ C[g] + D[h] * x[h]
    gated norm: u = y * silu(z);  per group of inner/groups channels u * rsqrt(mean(u^2) + eps);  * norm.weight
    out_proj

  Attention (GQA; Nano and Super: 32 query heads, 2 KV heads, head_dim 128):
    q_proj, k_proj, v_proj; NO positional encoding of any kind (config.json carries rope_theta and
    partial_rotary_factor; the modeling code never reads them); causal softmax, scale
    1/sqrt(head_dim), query head h reads KV head h // (heads / kv heads); o_proj

  MoE (Nano 128 experts top-6; Super 512 top-22 in a 1024-wide latent):
    s = sigmoid(gate.weight @ h);  pick the top-k of s + gate.e_score_correction_bias
    w = s[picked] / (sum(s[picked]) + 1e-20) * routed_scaling_factor
    u = fc1_latent_proj @ h  (Super; u = h without a latent)
    y = sum_e w_e * down_e(relu(up_e @ u)^2)             two matrices per expert, no gate_proj
    y = fc2_latent_proj @ y  (Super)
    y += shared_experts.down(relu(shared_experts.up @ h)^2)                  weight 1, reads h

  Weight formats (modelopt), told apart per module by the tensors present:
    NVFP4  `.weight` U8 [N, K/2], two E2M1 codes per byte, LOW nibble = EVEN column;
           `.weight_scale` F8_E4M3 [N, K/16]; `.weight_scale_2` F32 scalar:
               W[n, k] = e2m1(code[n, k]) * e4m3(weight_scale[n, k // 16]) * weight_scale_2
           E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 (bit 3 = sign).
    FP8    `.weight` F8_E4M3 [N, K]; `.weight_scale` F32 scalar:  W = e4m3(weight) * weight_scale
    BF16 / F32 as stored (Nano's router gate is F32).

  MTP draft block (Super, `mtp.layers.{0,1}`, all BF16): see Model.mtp_forward. The release's
  modeling code does not run it, so its conventions here are NOT transcribed from a reference;
  `mtp` measures the candidates.

Arithmetic is float32 by default; the Mamba time step, decay and state, and the final log-softmax
are float64. `--dtype float64` runs everything in float64 (a noise-floor check).

Memory: non-expert weights are converted once and kept (about 7 GB float32 on Nano, about 29 GB on
Super; `--no-keep` drops them after every layer instead); routed experts are dequantized on demand
and dropped. The mmap'd shard pages are released after every layer.
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
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(_BLAS_THREADS)

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import mmap  # noqa: E402
import resource  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

DEFAULT_CKPT = ("/home/mark/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/"
                "snapshots/6efb4a2a1c1fa277ce7b3df7a1416255011b1c99")

assert sys.byteorder == "little", "the BF16 / NVFP4 decoders assume a little-endian host"

# ----------------------------------------------------------------------------------------------
# Debug variants. Every layout assumption can be flipped in isolation with `--variant key=value`
# (repeatable); the defaults are the conventions read off the reference code. They exist to PROVE a
# convention on the real checkpoint (the wrong value must give garbage), not to be used.
# ----------------------------------------------------------------------------------------------
VARIANT_DEFAULTS = {
    "nibble": "lo",        # lo: low nibble = even column | hi: high nibble = even column
    "ws2": "mul",          # mul: W = e2m1 * e4m3 * weight_scale_2 | div: ... / weight_scale_2
    "fp8scale": "mul",     # mul: W = e4m3(weight) * weight_scale | div: ... / weight_scale
    "conv": "causal",      # causal: w[:, K-1] multiplies the current token | reversed: w[:, 0] does
    "grpmap": "div",       # div: Mamba head h -> B/C group h // (heads/groups) | mod: h % groups
    "dtclamp": "none",     # none: dt = softplus(.) (the CUDA path) | min: max(dt, time_step_min) (Super's CPU fallback)
    "gatenorm": "gate_first",  # gate_first: norm(y * silu(z)) | norm_first: norm(y) * silu(z)
    "kvmap": "div",        # div: query head h -> KV head h // (heads/kv) | mod: h % kv
    "pos": "none",         # none: no positional encoding | rope: rotate-half RoPE over the whole head, theta = rope_theta
    "rbias": "on",         # on: e_score_correction_bias on the selection key | off
    "renorm": "on",        # on: picked scores renormalized | off
    "act": "relu2",        # relu2: relu(x)^2 | relu | silu                 (the expert activation)
    "mtpcat": "eh",        # eh: eh_proj @ [enorm(embed) | hnorm(hidden)] | he: [hnorm(hidden) | enorm(embed)]
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
                if name in self.index:
                    raise ValueError(f"tensor {name!r} is in two shards")
                b, e = meta["data_offsets"]
                self.index[name] = (len(self.maps), meta["dtype"], tuple(meta["shape"]), base + b, base + e)
            self.maps.append(mm)

    def __contains__(self, name):
        return name in self.index

    def dtype(self, name):
        return self.index[name][1]

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
        self.layers = [None] * n_layers   # attention: {"k","v"}; Mamba: {"conv","ssm"}


_KIND = {"M": "mamba", "E": "moe", "*": "attention"}


def _layer_kinds(cfg, pattern_key, list_key):
    """Layer kinds from the pattern string (the released config.json) or the list a re-saved
    config writes. '-' is the reference's dense-MLP block; no release uses it."""
    if cfg.get(list_key) is not None and cfg.get(pattern_key) is None:
        return list(cfg[list_key])
    pat = cfg.get(pattern_key)
    if pat is None:
        return []
    bad = sorted(set(pat) - set(_KIND))
    if bad:
        raise ValueError(f"{pattern_key}: unsupported layer characters {bad}")
    return [_KIND[ch] for ch in pat]


class Model:
    def __init__(self, ckpt_dir, layers=None, dtype="float32", threads=_THREADS, variants=None, keep=True):
        self.ck = Checkpoint(ckpt_dir)
        c = self.ck.cfg
        self.dt = np.dtype(dtype).type
        self.var = dict(VARIANT_DEFAULTS)
        self.var.update(variants or {})
        self.keep = keep
        self.H = c["hidden_size"]
        self.vocab = c["vocab_size"]
        self.eps = float(c["layer_norm_epsilon"])
        self.kinds = _layer_kinds(c, "hybrid_override_pattern", "layers_block_type")
        self.total_layers = len(self.kinds)
        if c.get("num_hidden_layers") not in (None, self.total_layers):
            raise ValueError("hybrid_override_pattern: length differs from num_hidden_layers")
        self.n_layers = self.total_layers if layers is None else max(0, min(int(layers), self.total_layers))
        # Stop ids: config.json's eos (2, </s>) and generation_config.json's ([2, 11]: 11 is
        # <|im_end|>, what a chat turn actually ends with).
        self.eos = set()
        gen_path = os.path.join(ckpt_dir, "generation_config.json")
        gen = {}
        if os.path.exists(gen_path):
            with open(gen_path) as f:
                gen = json.load(f)
        for eos in (c.get("eos_token_id"), gen.get("eos_token_id")):
            self.eos |= set(eos) if isinstance(eos, list) else ({eos} if eos is not None else set())
        # Mamba2
        self.m_heads = c["mamba_num_heads"]
        self.m_hd = c["mamba_head_dim"]
        self.m_state = c["ssm_state_size"]
        self.m_groups = c.get("n_groups", c.get("mamba_n_groups"))
        self.m_conv = c.get("conv_kernel", c.get("mamba_d_conv"))
        self.m_inner = self.m_heads * self.m_hd
        self.m_conv_dim = self.m_inner + 2 * self.m_groups * self.m_state
        self.dt_min = float(c.get("time_step_min", c.get("mamba_dt_min", 0.001)))
        if self.m_heads % self.m_groups:
            raise ValueError("n_groups must divide mamba_num_heads")
        # attention
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c.get("num_key_value_heads") or self.n_heads
        self.head_dim = c.get("head_dim") or self.H // self.n_heads
        self.theta = float(c.get("rope_theta", 10000.0))           # the `pos=rope` debug variant only
        # MoE
        self.n_exp = c["n_routed_experts"]
        self.top_k = c["num_experts_per_tok"]
        self.route_scale = float(c.get("routed_scaling_factor", 1.0))
        self.norm_topk = bool(c.get("norm_topk_prob", True))
        self.latent = c.get("moe_latent_size") or 0
        if c.get("n_group", 1) != 1 or c.get("topk_group", 1) != 1:
            raise ValueError("group-limited routing (n_group / topk_group != 1) is not implemented")
        # MTP
        self.mtp_kinds = (_layer_kinds(c, "mtp_hybrid_override_pattern", "mtp_layers_block_type")
                          or ["attention", "moe"]) if c.get("num_nextn_predict_layers", 0) else []

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
        st_dtype = self.ck.dtype(name)
        if st_dtype == "BF16":
            w = bf16_to_f32(raw)
        elif st_dtype == "F32":
            w = np.array(raw, np.float32)
        else:
            raise TypeError(f"{name}: expected BF16 or F32, found {st_dtype}")
        return w if self.dt == np.float32 else w.astype(self.dt)

    def _fp4(self, prefix):
        """A modelopt NVFP4 linear dequantized to a dense float [N, K] matrix (not cached)."""
        codes = self.ck.raw(prefix + ".weight")              # U8 [N, K/2]
        scale = self.ck.raw(prefix + ".weight_scale")        # E4M3 [N, K/16]
        n, half = codes.shape
        w = self._pair[codes].view(np.float32)               # [N, K]: byte j -> columns 2j, 2j+1
        s = _E4M3[scale]                                     # [N, K/16]
        ws2 = np.float32(self.ck.raw(prefix + ".weight_scale_2").reshape(()))
        s = s * ws2 if self.var["ws2"] == "mul" else s / ws2
        w = w.reshape(n, half // 8, 16)
        w *= s[:, :, None]
        w = w.reshape(n, 2 * half)
        return w if self.dt == np.float32 else w.astype(self.dt)

    def _fp8(self, prefix):
        """A modelopt per-tensor FP8 linear dequantized to a dense float [N, K] matrix (not cached)."""
        s = np.float32(self.ck.raw(prefix + ".weight_scale").reshape(()))
        w = _E4M3[self.ck.raw(prefix + ".weight")]
        w = w * s if self.var["fp8scale"] == "mul" else w / s
        return w if self.dt == np.float32 else w.astype(self.dt)

    def fmt(self, prefix):
        """How the checkpoint stores the linear `prefix`: nvfp4 | fp8 | dense."""
        if prefix + ".weight_scale_2" in self.ck:
            return "nvfp4"
        if self.ck.dtype(prefix + ".weight") == "F8_E4M3":
            return "fp8"
        return "dense"

    def _linear_nocache(self, prefix):
        f = self.fmt(prefix)
        if f == "nvfp4":
            return self._fp4(prefix)
        if f == "fp8":
            return self._fp8(prefix)
        return self._dense_nocache(prefix + ".weight")

    def linear(self, prefix):
        """Weight of a linear layer, whichever way the checkpoint stores it (kept)."""
        w = self._cache.get(prefix)
        if w is None:
            w = self._linear_nocache(prefix)
            self._cache[prefix] = w
        return w

    def _expert_weights(self, prefix):
        """(up_proj, down_proj) of one routed expert, not cached."""
        return self._linear_nocache(prefix + ".up_proj"), self._linear_nocache(prefix + ".down_proj")

    def end_layer(self):
        """Releases the mapped pages and, under --no-keep, every converted weight but the head."""
        self.ck.release_pages()
        if not self.keep:
            for k in [k for k in self._cache if k != "lm_head.weight"]:
                del self._cache[k]

    # ---------------- blocks ----------------
    def rms(self, x, w):
        ms = np.mean(np.square(x), axis=-1, keepdims=True)
        return x * (1.0 / np.sqrt(ms + self.eps)) * w

    def embed(self, ids):
        raw = self.ck.raw("backbone.embeddings.weight")
        x = bf16_to_f32(np.ascontiguousarray(raw[np.asarray(ids, np.int64)]))
        return x if self.dt == np.float32 else x.astype(self.dt)

    def act(self, x):
        a = self.var["act"]
        if a == "relu2":
            return np.square(np.maximum(x, 0))
        return np.maximum(x, 0) if a == "relu" else silu(x)   # debug variants

    def mamba_core(self, prefix, proj, st_layer, tap=None):
        """Mamba2 between its two projections. proj [n, inner + conv_dim + heads] is in_proj's
        output; returns the gated-norm output [n, inner] (out_proj's input). st_layer holds the
        last conv_kernel - 1 pre-conv rows ("conv") and the SSM state ("ssm")."""
        n = proj.shape[0]
        nh, hd, N, G, K = self.m_heads, self.m_hd, self.m_state, self.m_groups, self.m_conv
        inner, cd = self.m_inner, self.m_conv_dim
        f64 = np.float64
        if proj.shape[1] != inner + cd + nh:
            raise ValueError(f"{prefix}: in_proj has {proj.shape[1]} rows, expected {inner + cd + nh} (d_mlp must be 0)")
        z, xbc, dt_raw = proj[:, :inner], proj[:, inner:inner + cd], proj[:, inner + cd:]

        # depthwise causal conv over time on [x | B | C], bias, then SiLU
        cw = self.dense(prefix + ".conv1d.weight").reshape(cd, K)
        if self.var["conv"] != "causal":  # debug variant
            cw = cw[:, ::-1]
        tail = st_layer.get("conv")
        if tail is None:
            tail = np.zeros((K - 1, cd), proj.dtype)
        ext = np.concatenate([tail, xbc])                                # [n + K - 1, conv_dim]
        st_layer["conv"] = ext[n:].copy()                                # the last K - 1 pre-conv rows
        c = np.zeros_like(xbc)
        for j in range(K):
            c += ext[j:j + n] * cw[:, j]
        if prefix + ".conv1d.bias" in self.ck:
            c += self.dense(prefix + ".conv1d.bias")
        c = silu(c)
        x = c[:, :inner].reshape(n, nh, hd).astype(f64)
        B = c[:, inner:inner + G * N].reshape(n, G, N).astype(f64)
        C = c[:, inner + G * N:].reshape(n, G, N).astype(f64)
        rep = nh // G
        if self.var["grpmap"] == "div":
            B, C = np.repeat(B, rep, axis=1), np.repeat(C, rep, axis=1)  # head h <- group h // rep
        else:  # debug variant
            B, C = np.tile(B, (1, rep, 1)), np.tile(C, (1, rep, 1))

        # the time step, the decay and the recurrence run in float64
        dt = softplus(dt_raw.astype(f64) + self.dense(prefix + ".dt_bias").astype(f64))   # [n, heads]
        if self.var["dtclamp"] == "min":  # Super's CPU fallback (torch_forward)
            dt = np.maximum(dt, self.dt_min)
        dA = np.exp(-np.exp(self.dense(prefix + ".A_log").astype(f64)) * dt)             # [n, heads]
        D = self.dense(prefix + ".D").astype(f64)
        S = st_layer.get("ssm")
        if S is None:
            S = np.zeros((nh, hd, N), f64)                               # [head, head_dim, state]
        y = np.empty((n, nh, hd), f64)
        dtx = dt[:, :, None] * x                                         # [n, heads, head_dim]
        for t in range(n):
            np.multiply(S, dA[t][:, None, None], out=S)                  # S *= dA
            S += dtx[t][:, :, None] * B[t][:, None, :]                   # S += outer(dt * x, B)
            y[t] = np.matmul(S, C[t][:, :, None])[:, :, 0] + D[:, None] * x[t]
        st_layer["ssm"] = S

        # gated RMS norm over n_groups channel groups: the gate first, then the norm
        u = y.reshape(n, inner).astype(self.dt)
        w = self.dense(prefix + ".norm.weight")
        if self.var["gatenorm"] == "gate_first":
            u = u * silu(z)
        g = u.reshape(n, G, inner // G)
        g = g * (1.0 / np.sqrt(np.mean(np.square(g), axis=-1, keepdims=True) + self.eps))
        out = g.reshape(n, inner) * w
        if self.var["gatenorm"] != "gate_first":  # debug variant
            out = out * silu(z)
        if tap is not None:
            tap("mamba_gate", z)
            tap("mamba_conv", c)
            tap("mamba_dt", dt)
            tap("mamba_scan", y)
            tap("mamba_normed", out)
            tap("mamba_state", S)
        return out

    def mamba(self, prefix, h, st_layer, tap=None):
        """Mamba2 on a chunk. h [n, H] is the normed input."""
        proj = h @ self.linear(prefix + ".in_proj").T
        return self.mamba_core(prefix, proj, st_layer, tap) @ self.linear(prefix + ".out_proj").T

    def _rope(self, x, pos):
        """Debug variant only (`pos=rope`): rotate-half RoPE over the whole head."""
        hd = x.shape[-1]
        h = hd // 2
        inv = self.theta ** (-np.arange(h, dtype=np.float64) / h)
        ang = np.asarray(pos, np.float64)[:, None] * inv[None, :]
        cos, sin = np.cos(ang).astype(self.dt)[:, None, :], np.sin(ang).astype(self.dt)[:, None, :]
        out = x.copy()
        out[..., :h] = x[..., :h] * cos - x[..., h:] * sin
        out[..., h:] = x[..., h:] * cos + x[..., :h] * sin
        return out

    def attention(self, prefix, h, st_layer, pos0, tap=None, self_only=False):
        """Attention on a chunk. h [n, H] is the normed input; st_layer holds the K/V of the
        earlier tokens; pos0 is the position of the chunk's first token (read by the `pos=rope`
        debug variant alone). self_only (a measurement switch for the draft block) lets every
        token read only its own key, i.e. no K/V history."""
        n = h.shape[0]
        nh, nkv, hd = self.n_heads, self.n_kv, self.head_dim
        q = (h @ self.linear(prefix + ".q_proj").T).reshape(n, nh, hd)
        k = (h @ self.linear(prefix + ".k_proj").T).reshape(n, nkv, hd)
        v = (h @ self.linear(prefix + ".v_proj").T).reshape(n, nkv, hd)
        if self.var["pos"] == "rope":  # debug variant
            pos = np.arange(pos0, pos0 + n)
            q, k = self._rope(q, pos), self._rope(k, pos)

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
        y = ctx.reshape(n, nh * hd) @ self.linear(prefix + ".o_proj").T
        if tap is not None:
            tap("attn_q", q)
            tap("attn_k", k)
            tap("attn_v", v)
            tap("attn_ctx", ctx)
        return y

    def moe(self, prefix, h, tap=None):
        """Routed MoE on a chunk. h [n, H] is the normed input."""
        n, kk = h.shape[0], self.top_k
        # The reference runs the router in float32 on its BF16 activations; here it runs in the
        # run's own dtype, like everything else.
        logits = h @ self.dense(prefix + ".gate.weight").T               # [n, experts]
        score = sigmoid(logits)
        key = score
        if self.var["rbias"] == "on":
            key = score + self.dense(prefix + ".gate.e_score_correction_bias")
        idx = np.empty((n, kk), np.int64)
        for i in range(n):
            idx[i] = top_desc(key[i], kk)
        wts = np.take_along_axis(score, idx, axis=-1)                    # the UNCORRECTED scores
        if self.norm_topk and self.var["renorm"] == "on":
            wts = wts / (wts.sum(axis=-1, keepdims=True) + 1e-20)
        wts = (wts * self.route_scale).astype(self.dt)

        u = h @ self.linear(prefix + ".fc1_latent_proj").T if self.latent else h
        out = np.zeros_like(u)
        flat_e = idx.reshape(-1)
        flat_t = np.repeat(np.arange(n), kk)
        flat_w = wts.reshape(-1)
        order = np.argsort(flat_e, kind="stable")
        uniq, start = np.unique(flat_e[order], return_index=True)
        bounds = list(start) + [order.size]
        batch = max(1, 2 * self.threads)
        for b0 in range(0, len(uniq), batch):
            group = [int(e) for e in uniq[b0:b0 + batch]]
            weights = list(self.pool.map(lambda e: self._expert_weights(f"{prefix}.experts.{e}"), group))
            self.stats["experts_dequantized"] += len(group)
            for j, (wu, wd) in enumerate(weights):
                sel = order[bounds[b0 + j]:bounds[b0 + j + 1]]
                rows = flat_t[sel]                                       # distinct tokens
                out[rows] += flat_w[sel][:, None] * (self.act(u[rows] @ wu.T) @ wd.T)
            del weights
        routed = out @ self.linear(prefix + ".fc2_latent_proj").T if self.latent else out
        shared = self.act(h @ self.linear(prefix + ".shared_experts.up_proj").T) @ \
            self.linear(prefix + ".shared_experts.down_proj").T
        if tap is not None:
            tap("router_logits", logits)
            tap("router_ids", idx.astype(np.int32))
            tap("router_w", wts)
            tap("moe_routed", routed)
            tap("moe_shared", shared)
        return routed + shared

    # ---------------- the stack ----------------
    def new_state(self):
        return State(self.total_layers)

    def mixer(self, kind, prefix, h, st_layer, pos0, tap=None):
        if kind == "mamba":
            return self.mamba(prefix, h, st_layer, tap)
        if kind == "attention":
            return self.attention(prefix, h, st_layer, pos0, tap)
        return self.moe(prefix, h, tap)

    def forward(self, ids, st, tap=None, detail_layers=(), on_layer=None):
        """Feeds a chunk of token ids, advancing the state. Returns the residual stream after the
        last executed layer, [n, H] (before norm_f). tap(name, array) receives h_00 .. h_NN and,
        for the layers in detail_layers, the sublayer tensors as LNN_<name>."""
        n = len(ids)
        x = self.embed(ids)
        if tap is not None:
            tap("h_00", x)
        for L in range(self.n_layers):
            p = f"backbone.layers.{L}"
            if st.layers[L] is None:
                st.layers[L] = {}
            sub = None
            if tap is not None and L in detail_layers:
                sub = (lambda name, arr, L=L: tap(f"L{L:02d}_{name}", arr))
            h = self.rms(x, self.dense(p + ".norm.weight"))
            y = self.mixer(self.kinds[L], p + ".mixer", h, st.layers[L], st.pos, sub)
            if sub is not None:
                sub("mixer_in", h)
                sub("mixer_out", y)
            x = x + y
            if tap is not None:
                tap(f"h_{L + 1:02d}", x)
            self.end_layer()
            if on_layer is not None:
                on_layer(L + 1)
        st.pos += n
        return x

    def final_norm(self, x):
        return self.rms(x, self.dense("backbone.norm_f.weight"))

    def logits(self, x):
        """x [n, H] residual stream (before norm_f) -> logits [n, vocab]."""
        return self.final_norm(x) @ self.dense("lm_head.weight").T

    def logprobs(self, x):
        return log_softmax64(self.logits(x))

    # ---------------- MTP draft block (Super) ----------------
    def mtp_new_state(self):
        return {"pos": 0, "attn": {}}

    def mtp_forward(self, next_ids, hidden, st, history=True):
        """The draft block on a chunk of (hidden, next token) pairs — the DeepSeek-V3 form its
        tensor names spell (enorm, hnorm, eh_proj, final_layernorm):
            x = eh_proj @ concat[rms(embed(next_token), enorm), rms(hidden, hnorm)]
        then mtp.layers.0 (an attention layer) and mtp.layers.1 (a MoE layer), each the backbone's
        own pre-norm residual block, then final_layernorm and the SHARED lm_head.
        NOT transcribed from a reference: the release's modeling code lists `mtp.*` as ignored on
        load. The concat order (`--variant mtpcat`), whether `hidden` is taken before or after
        norm_f, and whether the draft attention keeps a K/V history are the open conventions;
        cmd_mtp measures them. Row i pairs hidden[i] with next_ids[i] (the token at position i + 1)
        and drafts the token at position i + 2. Returns log-probabilities [n, vocab]."""
        n = len(next_ids)
        p0, p1 = "mtp.layers.0", "mtp.layers.1"
        e = self.rms(self.embed(next_ids), self.dense(p0 + ".enorm.weight"))
        hh = self.rms(hidden, self.dense(p0 + ".hnorm.weight"))
        cat = [e, hh] if self.var["mtpcat"] == "eh" else [hh, e]
        x = np.concatenate(cat, axis=-1) @ self.linear(p0 + ".eh_proj").T
        h = self.rms(x, self.dense(p0 + ".norm.weight"))
        x = x + self.attention(p0 + ".mixer", h, st["attn"], st["pos"], self_only=not history)
        h = self.rms(x, self.dense(p1 + ".norm.weight"))
        x = x + self.moe(p1 + ".mixer", h)
        st["pos"] += n
        self.ck.release_pages()
        x = self.rms(x, self.dense(p1 + ".final_layernorm.weight"))
        out = np.empty((n, self.vocab), np.float64)
        for b0 in range(0, n, 32):
            out[b0:b0 + 32] = log_softmax64(x[b0:b0 + 32] @ self.dense("lm_head.weight").T)
        return out


# ----------------------------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------------------------
def progress(stage, stages, done, total, err, t0):
    print(f"[{stage}/{stages}] {done}/{total} err={err} elapsed={int(time.time() - t0)}s", flush=True)


def peak_rss_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2.0 ** 30 if sys.platform == "darwin" else rss / (1024.0 * 1024.0)   # bytes | KiB


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
    m = Model(a.ckpt, layers=a.layers, dtype=a.dtype, threads=a.threads, variants=variants, keep=not a.no_keep)
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
    hn = model.final_norm(x)
    tap("h_final_norm", hn)
    lg = (hn[-1:] @ model.dense("lm_head.weight").T)[0]
    tap("logits_last", lg)
    write_json(os.path.join(a.out, "meta.json"), {
        "ids": ids, "n_tokens": len(ids), "layers": model.n_layers, "dtype": np.dtype(model.dt).name,
        "detail_layers": detail, "layer_kinds": model.kinds[:model.n_layers], "files": files,
        "variants": model.var,
        "notes": {
            "h_NN": "residual stream [T, H]: h_00 after the embedding, h_NN after layer NN-1",
            "h_final_norm": "rms(h_last, norm_f) [T, H]",
            "logits_last": "lm_head @ h_final_norm at the last position [vocab]",
            "LNN_mixer_in": "rms(x, layers.NN.norm)", "LNN_mixer_out": "mixer output before the residual add",
            "LNN_mamba_gate": "in_proj's gate slice z [T, inner]",
            "LNN_mamba_conv": "[x | B | C] after the conv and SiLU [T, inner + 2*groups*N]",
            "LNN_mamba_dt": "softplus(dt + dt_bias) [T, heads]",
            "LNN_mamba_scan": "recurrence output S C + D x, before the gated norm [T, heads, head_dim]",
            "LNN_mamba_normed": "gated-norm output, out_proj's input [T, inner]",
            "LNN_mamba_state": "final S [heads, head_dim, N]",
            "LNN_attn_q": "[T, heads, head_dim]", "LNN_attn_k / _v": "[T, kv heads, head_dim]",
            "LNN_attn_ctx": "attention output before o_proj [T, heads, head_dim]",
            "LNN_router_logits": "[T, experts]", "LNN_router_ids": "top-k expert ids [T, k], best first",
            "LNN_router_w": "normalized top-k weights x routed_scaling_factor [T, k]",
            "LNN_moe_routed": "weighted expert sum (after fc2_latent_proj on Super) [T, H]",
            "LNN_moe_shared": "shared expert [T, H]"},
        "seconds": round(time.time() - t0, 2), "peak_rss_gb": round(peak_rss_gb(), 2)})
    print(f"# wrote {len(files)} arrays to {a.out}", flush=True)


# ---------------- MTP ----------------
MTP_HIDDEN = (("pre_norm", "residual stream of the last layer BEFORE norm_f"),
              ("post_norm", "final hidden AFTER norm_f"))


def cmd_mtp(a):
    """How often the draft block's greedy draft equals the main model's own greedy next token.
    For every position t of a text the block gets embed(token t+1) and the main model's hidden
    state at t and drafts position t+2; that is compared with the main model's argmax for
    position t+2 (its row t+1). Measured for both conventions of "hidden state" (before / after
    norm_f), each with and without the draft attention's own K/V history over the earlier pairs.
    Run it once per `--variant mtpcat=eh|he`: the right conventions are the ones that agree."""
    model = make_model(a)
    if "mtp.layers.0.eh_proj.weight" not in model.ck:
        raise SystemExit("this checkpoint carries no MTP draft block (no mtp.* tensors): nothing to measure")
    if model.n_layers != model.total_layers:
        raise SystemExit("mtp needs the whole backbone (no --layers)")
    t0 = time.time()
    name, ids = os.path.basename(a.ids_json), load_ids(a.ids_json)
    keys = [f"{hid}/{'history' if hist else 'no_history'}" for hid, _ in MTP_HIDDEN for hist in (True, False)]
    x, _ = run_forward(model, ids, on_layer=lambda d: progress(1, 1, d, model.n_layers, 0, t0)
                       if d % 8 == 0 or d == model.n_layers else None)
    T = len(ids)
    hn = model.final_norm(x)
    main_top = np.empty(T, np.int64)                       # main_top[i]: greedy token for position i + 1
    for b0 in range(0, T, 32):
        main_top[b0:b0 + 32] = (hn[b0:b0 + 32] @ model.dense("lm_head.weight").T).argmax(axis=-1)
    actual = np.array(ids[2:], np.int64)
    r = {"id": name, "tokens": T, "drafts": T - 1, "mtpcat": model.var["mtpcat"],
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
    print(json.dumps(r), flush=True)
    if a.out:
        write_json(a.out, r)
    print(f"{'text':<16}{'drafts':>7}" + "".join(f"{k:>22}" for k in keys))
    print(f"{r['id']:<16}{r['drafts']:>7}" + "".join(f"{r[k]['draft_eq_main']:>14} {100 * r[k]['frac']:5.1f}% " for k in keys))
    print(f"# peak_rss_gb={peak_rss_gb():.2f} seconds={int(time.time() - t0)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Nemotron-3 Nano / Super (NVFP4) host reference forward pass, pure numpy")
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--ids-json", required=True, help="JSON list of token ids")
        p.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint snapshot directory")
        p.add_argument("--layers", type=int, default=None, help="run only the first N layers")
        p.add_argument("--threads", type=int, default=_THREADS,
                       help="total thread budget, half BLAS and half expert dequant (default 8)")
        p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
        p.add_argument("--no-keep", action="store_true",
                       help="drop every dequantized weight after its layer (slower chunked feeding, far less memory)")
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
    p.add_argument("--detail-layers", default="0,1,5", help="layers whose sublayer tensors are saved")
    p.set_defaults(fn=cmd_dump)

    p = sub.add_parser("mtp", help="Super's MTP draft block vs the main model's next token")
    common(p)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_mtp)

    a = ap.parse_args()
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
