#!/usr/bin/env python3
"""DeepSeek-V4-Flash independent reference: the release's OWN layer code
(inference/model.py of the deepseek-ai/DeepSeek-V4-Flash-0731 snapshot) over
a checkpoint — the real one or the test fixture — against the engine's
per-layer dump (dsv4_forward_check --out states.bin). The rule that found
GLM-4.7's RoPE bug: run the reference's own layer code on the weights
before trusting a self-written reference.

model.py drives its GEMMs and a few fused steps through tilelang kernels
(kernel.py). Those are replaced here by exact torch math on the CPU:
  * every Linear's weight is dequantized ONCE — e4m3 x e8m0 block scales
    for the fp8 tensors, e2m1 nibbles x e8m0 for the MXFP4 experts (both
    exactly representable in bf16, so nothing is lost) — and `linear()` is
    an fp32-accumulated product rounded to the activation dtype (the
    kernels' accumulator and output types); with --act-quant the activation
    is first put through act_quant's per-128 e4m3 / power-of-two-scale
    quantize-dequantize (the kernel's own rounding);
  * act_quant / fp4_act_quant (the cached rows, the indexer's q and k),
    sparse_attn (the attention with the sink: fp32 scores, P rounded to
    bf16 before PV as the kernel does, the unrounded denominator),
    hc_split_sinkhorn and the Hadamard rotation are ports of the kernels'
    arithmetic.
Three comparisons, each per layer as the relative l2 over the [T, 4, H]
streams:
  * CHAINED prefill (the reference's own inputs; errors accumulate),
  * ISOLATED prefill (block l fed the engine's layer l-1 output: one
    layer's own error),
  * ISOLATED decode (when the dump has a decode section): the block's
    caches built by a prefill of the first P rows, then every decode step
    fed the engine's layer l-1 row — the engine's decode path (rings,
    incremental compressors, the decode select) against the reference's.
With --draft the DSpark stages run too (the block's base logits, row 0's
Markov-biased logits and confidence).

  .venv/bin/python3 tools/dsv4_torch_reference.py --checkpoint-dir DIR \
      --engine-dump states.bin [--model-py PATH/inference/model.py]
      [--layers 0,1,2] [--act-quant] [--hc-bf16] [--threads 20]
"""
import argparse
import glob
import importlib.util
import json
import os
import struct
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# the release's quantization arithmetic in torch (kernel.py's semantics)
# ---------------------------------------------------------------------------
def e8m0_ceil_pow2(x):
    """2^ceil(log2(x)) from the fp32 bit fields (kernel.py fast_round_scale)."""
    bits = x.float().contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127
    man = bits & 0x7FFFFF
    k = exp + (man != 0).to(torch.int32)
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), k)


def e4m3_round(x):
    """The nearest e4m3 value (RNE, saturating at 448, subnormals at 2^-9), fp32 in and out."""
    a = x.abs()
    s = torch.sign(x)
    sub = a < 2.0 ** -6
    r_sub = torch.round(a * 512.0) * 2.0 ** -9
    m, e = torch.frexp(a.clamp_min(2.0 ** -6))  # a = m * 2^e, m in [0.5, 1)
    q = torch.round((m * 2.0 - 1.0) * 8.0)
    r = torch.clamp((1.0 + q / 8.0) * torch.ldexp(torch.ones_like(a), e - 1), max=448.0)
    return s * torch.where(sub, r_sub, r)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1_round(x):
    """The nearest e2m1 value on {0, .5, 1, 1.5, 2, 3, 4, 6} (ties to the even code)."""
    a = x.abs()
    s = torch.sign(x)
    out = torch.full_like(a, 6.0)
    for hi, v in ((5.0, 4.0), (3.5, 3.0), (2.5, 2.0), (1.75, 1.5), (1.25, 1.0), (0.75, 0.5), (0.25, 0.0)):
        out = torch.where(a <= hi if v in (0.0, 1.0, 2.0, 4.0) else a < hi, torch.full_like(a, v), out)
    return s * out


def act_quant_dequant(x, block, round_scale=True):
    """act_quant(block, ue8m0) quantize-dequantize per row block: values in, values out (x's dtype)."""
    xf = x.float()
    blocks = xf.reshape(*xf.shape[:-1], -1, block)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = e8m0_ceil_pow2(amax * (1.0 / 448.0)) if round_scale else amax * (1.0 / 448.0)
    q = e4m3_round(torch.clamp(blocks / s, -448.0, 448.0)) * s
    return q.reshape(xf.shape).to(x.dtype)


def fp4_quant_dequant(x, block):
    """fp4_act_quant quantize-dequantize with e8m0 scales per `block`."""
    xf = x.float()
    blocks = xf.reshape(*xf.shape[:-1], -1, block)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(6.0 * 2.0 ** -126)
    s = e8m0_ceil_pow2(amax * (1.0 / 6.0))
    q = e2m1_round(torch.clamp(blocks / s, -6.0, 6.0)) * s
    return q.reshape(xf.shape).to(x.dtype)


def dequant_fp8(codes, scales, block=128):
    """e4m3 codes [N, K] x e8m0 scales [ceil(N/128), ceil(K/128)] -> bf16 (exact)."""
    n, k = codes.shape
    v = codes.float()
    s = scales.float().repeat_interleave(block, dim=0)[:n].repeat_interleave(block, dim=1)[:, :k]
    return (v * s).to(torch.bfloat16)


def dequant_fp4(packed, scales, block=32):
    """e2m1 nibbles [N, K/2] (low nibble first) x e8m0 scales [N, K/32] -> bf16 (exact)."""
    b = packed.view(torch.uint8)
    n = b.shape[0]
    lo = (b & 0xF).to(torch.long)
    hi = (b >> 4).to(torch.long)
    nib = torch.stack([lo, hi], dim=-1).reshape(n, -1)  # [N, K]
    mag = E2M1[nib & 0x7]
    val = torch.where(nib & 0x8 != 0, -mag, mag)
    k = val.shape[1]
    s = scales.float().reshape(n, k // block, 1)
    return (val.reshape(n, k // block, block) * s).reshape(n, k).to(torch.bfloat16)


def sylvester(n):
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


# ---------------------------------------------------------------------------
# the kernel module stand-ins (bound before model.py is imported)
# ---------------------------------------------------------------------------
ACT_QUANT = False
ATTN_FP32_P = False


def make_kernel_module():
    m = types.ModuleType("kernel")

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        y = act_quant_dequant(x, block_size, round_scale=scale_fmt is not None)
        if inplace:
            x.copy_(y)
            return x
        return y, None

    def fp4_act_quant(x, block_size=32, inplace=False):
        y = fp4_quant_dequant(x, block_size)
        if inplace:
            x.copy_(y)
            return x
        return y, None

    def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
        return F.linear(a.float(), b.float()).to(a.dtype)

    def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
        return F.linear(a.float(), b.float()).to(a.dtype)

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        """The kernel's arithmetic: fp32 scores, P in bf16 for the PV product, the unrounded
        denominator plus the sink term, the query's dtype out."""
        b, s, h, d = q.shape
        out = torch.empty_like(q)
        kvf = kv.float()
        for bi in range(b):
            idx = topk_idxs[bi].long()  # [s, topk]
            valid = idx >= 0
            chunk = 256
            for r0 in range(0, s, chunk):
                r1 = min(s, r0 + chunk)
                rows = kvf[bi][idx[r0:r1].clamp_min(0)]  # [c, topk, d]
                sc = torch.einsum("shd,std->sht", q[bi, r0:r1].float(), rows) * softmax_scale
                sc = sc.masked_fill(~valid[r0:r1].unsqueeze(1), float("-inf"))
                mx = sc.amax(dim=-1, keepdim=True).clamp_min(-1e30)
                p = torch.exp(sc - mx)
                denom = p.sum(dim=-1) + torch.exp(attn_sink.float().view(1, h) - mx.squeeze(-1))
                pv = torch.einsum("sht,std->shd", p if ATTN_FP32_P else p.to(torch.bfloat16).float(), rows)
                out[bi, r0:r1] = (pv / denom.unsqueeze(-1)).to(q.dtype)
        return out

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        b, s, _ = mixes.shape
        m_ = mixes.float()
        pre = torch.sigmoid(m_[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
        post = 2 * torch.sigmoid(m_[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])
        comb = (m_[..., 2 * hc_mult:] * hc_scale[2] + hc_base[2 * hc_mult:]).reshape(b, s, hc_mult, hc_mult)
        comb = torch.softmax(comb, dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb

    m.act_quant = act_quant
    m.fp4_act_quant = fp4_act_quant
    m.fp8_gemm = fp8_gemm
    m.fp4_gemm = fp4_gemm
    m.sparse_attn = sparse_attn
    m.hc_split_sinkhorn = hc_split_sinkhorn
    return m


def make_hadamard_module():
    m = types.ModuleType("fast_hadamard_transform")
    cache = {}

    def hadamard_transform(x, scale=1.0):
        n = x.shape[-1]
        if n not in cache:
            cache[n] = sylvester(n)
        return ((x.float() @ cache[n]) * scale).to(x.dtype)

    m.hadamard_transform = hadamard_transform
    return m


def load_release_model(model_py):
    sys.modules["kernel"] = make_kernel_module()
    sys.modules["fast_hadamard_transform"] = make_hadamard_module()
    spec = importlib.util.spec_from_file_location("model", model_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# the checkpoint
# ---------------------------------------------------------------------------
class Shards:
    def __init__(self, d):
        from safetensors import safe_open

        self.d = d
        self.index = {}
        self.files = {}
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.index = json.load(open(idx))["weight_map"]
        else:
            for p in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
                f = safe_open(p, framework="pt", device="cpu")
                self.files[os.path.basename(p)] = f
                for k in f.keys():
                    self.index[k] = os.path.basename(p)
        self._open = safe_open

    def file(self, name):
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = self._open(os.path.join(self.d, shard), framework="pt", device="cpu")
        return self.files[shard]

    def has(self, name):
        return name in self.index

    def get(self, name):
        return self.file(name).get_tensor(name)

    def linear_weight(self, prefix):
        """A Linear's weight dequantized to bf16 (fp8 x e8m0, fp4 x e8m0) or as stored (bf16/f32)."""
        w = self.get(prefix + ".weight")
        if self.has(prefix + ".scale"):
            s = self.get(prefix + ".scale").float()  # e8m0: 2^(byte - 127)
            if w.dtype == torch.int8:
                return dequant_fp4(w, s)
            return dequant_fp8(w, s)
        return w


def read_engine_dump(path):
    out = {}
    with open(path, "rb") as f:
        if f.read(8) != b"DSV4ST01":
            raise SystemExit(f"{path}: not a dsv4_forward_check dump")
        L, T, W = struct.unpack("<iii", f.read(12))
        raw = np.frombuffer(f.read(L * T * W * 2), dtype=np.uint16).astype(np.uint32) << 16
        out["states"] = torch.from_numpy(raw.view(np.float32).reshape(L, T, W).copy())
        out["ids"] = np.frombuffer(f.read(T * 8), dtype=np.int64).copy()
        K = struct.unpack("<i", f.read(4))[0]
        out["routes"] = np.frombuffer(f.read(L * T * K * 4), dtype=np.int32).reshape(L, T, K).copy()
        Li, ms = struct.unpack("<ii", f.read(8))
        out["sels"] = np.frombuffer(f.read(Li * T * ms * 4), dtype=np.int32).reshape(Li, T, ms).copy()
        V, R = struct.unpack("<ii", f.read(8))
        out["logits"] = torch.from_numpy(np.frombuffer(f.read(R * V * 4), dtype=np.float32).reshape(R, V).copy())
        magic = f.read(8)
        if magic == b"DSV4DEC1":
            N, Ld, Wd, Vd, Lid, msd = struct.unpack("<iiiiii", f.read(24))
            st, lg, sl = [], [], []
            for _ in range(N):
                raw = np.frombuffer(f.read(Ld * Wd * 2), dtype=np.uint16).astype(np.uint32) << 16
                st.append(raw.view(np.float32).reshape(Ld, Wd).copy())
                lg.append(np.frombuffer(f.read(Vd * 4), dtype=np.float32).copy())
                sl.append(np.frombuffer(f.read(Lid * msd * 4), dtype=np.int32).reshape(Lid, msd).copy())
            out["dec_states"] = torch.from_numpy(np.stack(st))  # [N, L, W]
            out["dec_logits"] = torch.from_numpy(np.stack(lg))
            out["dec_sels"] = np.stack(sl)
            magic = f.read(8)
        if magic == b"DSV4DRF1":
            N, B, Vd = struct.unpack("<iii", f.read(12))
            nx, row, base, conf = [], [], [], []
            for _ in range(N):
                nx.append(struct.unpack("<i", f.read(4))[0])
                row.append(np.frombuffer(f.read(Vd * 4), dtype=np.float32).copy())
                base.append(np.frombuffer(f.read(B * Vd * 4), dtype=np.float32).reshape(B, Vd).copy())
                conf.append(struct.unpack("<f", f.read(4))[0])
            out["draft_next"] = nx
            out["draft_row"] = torch.from_numpy(np.stack(row))
            out["draft_base"] = torch.from_numpy(np.stack(base))
            out["draft_conf"] = conf
    return out


def rel_l2(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def model_args(model, hf, max_seq_len):
    n_mtp = len(hf["compress_ratios"]) - hf["num_hidden_layers"]
    rs = hf["rope_scaling"]
    return model.ModelArgs(
        max_batch_size=1, max_seq_len=max_seq_len, temperature=0,
        dtype="fp8", scale_fmt="ue8m0", expert_dtype="fp4", scale_dtype="fp8",
        vocab_size=hf["vocab_size"], dim=hf["hidden_size"], moe_inter_dim=hf["moe_intermediate_size"],
        n_layers=hf["num_hidden_layers"], n_hash_layers=hf.get("num_hash_layers", 0), n_mtp_layers=n_mtp,
        n_heads=hf["num_attention_heads"], n_routed_experts=hf["n_routed_experts"],
        n_shared_experts=hf["n_shared_experts"], n_activated_experts=hf["num_experts_per_tok"],
        score_func=hf["scoring_func"], route_scale=hf["routed_scaling_factor"], swiglu_limit=hf["swiglu_limit"],
        q_lora_rank=hf["q_lora_rank"], head_dim=hf["head_dim"], rope_head_dim=hf["qk_rope_head_dim"],
        norm_eps=hf["rms_norm_eps"], o_groups=hf["o_groups"], o_lora_rank=hf["o_lora_rank"],
        window_size=hf["sliding_window"], compress_ratios=tuple(hf["compress_ratios"]),
        compress_rope_theta=hf["compress_rope_theta"], original_seq_len=rs["original_max_position_embeddings"],
        rope_theta=hf["rope_theta"], rope_factor=rs["factor"], beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"],
        index_n_heads=hf["index_n_heads"], index_head_dim=hf["index_head_dim"], index_topk=hf["index_topk"],
        hc_mult=hf["hc_mult"], hc_sinkhorn_iters=hf["hc_sinkhorn_iters"], hc_eps=hf["hc_eps"],
        dspark_block_size=hf.get("dspark_block_size", 0), dspark_noise_token_id=hf.get("dspark_noise_token_id", 0),
        dspark_target_layer_ids=tuple(hf.get("dspark_target_layer_ids", [])),
        dspark_markov_rank=hf.get("dspark_markov_rank", 256))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--engine-dump", required=True)
    ap.add_argument("--model-py", default=None, help="the release's inference/model.py (default: the checkpoint's own)")
    ap.add_argument("--layers", default=None, help="the layers to run (default: every dumped layer)")
    ap.add_argument("--act-quant", action="store_true", help="quantize every Linear's activation as the kernels do")
    ap.add_argument("--hc-bf16", action="store_true",
                    help="round the mHC coefficient matrices (hc_*_fn) to bf16 as the engine's loader does")
    ap.add_argument("--attn-fp32-p", action="store_true",
                    help="the attention stand-in keeps its probabilities in fp32 for the PV product")
    ap.add_argument("--no-decode", action="store_true", help="skip the dump's decode section")
    ap.add_argument("--draft-only", action="store_true",
                    help="skip the main layers and the head: only the DSpark draft, fed the engine's main hidden "
                         "(needs a dump with --decode-steps and --draft); adds the block's argmax agreement per row")
    ap.add_argument("--chained-only", action="store_true",
                    help="run only the chained prefill (the reference's own states end to end) and the chained "
                         "head: the reference's own next-token logits beside the engine's")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--budget", type=float, default=0.01,
                    help="the isolated per-layer relative-l2 budget over the rows routed and selected alike")
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    global ACT_QUANT, ATTN_FP32_P
    ACT_QUANT = args.act_quant
    ATTN_FP32_P = args.attn_fp32_p
    snap = args.checkpoint_dir
    dump = read_engine_dump(args.engine_dump)
    states = dump["states"]
    ids = dump["ids"]
    L_dump, T, W = states.shape
    model_py = args.model_py or os.path.join(snap, "inference", "model.py")
    model = load_release_model(model_py)
    hf = json.load(open(os.path.join(snap, "config.json")))
    margs = model_args(model, hf, max(4096, ((T + 127) // 128) * 128 + 128))
    torch.set_default_dtype(torch.bfloat16)
    model.world_size, model.rank = 1, 0
    model.default_dtype = torch.bfloat16
    model.scale_fmt = "ue8m0"
    model.scale_dtype = "e8m0"  # only compared against inside the kernel stand-ins (ignored)
    H, hc = margs.dim, margs.hc_mult
    assert W == hc * H, (W, hc, H)
    shards = Shards(snap)
    print(f"engine dump: {L_dump} layers x {T} rows x {W}; act_quant {'on' if ACT_QUANT else 'off'}, hc_fn "
          f"{'bf16 (as loaded by the engine)' if args.hc_bf16 else 'fp32 (as stored)'}; decode steps "
          f"{0 if 'dec_states' not in dump else dump['dec_states'].shape[0]}")

    # linear(): fp32 accumulation, the activation dtype out; quantized weights optionally take a
    # quantized activation (the kernels' dispatch by weight dtype).
    def linear(x, weight, bias=None):
        assert bias is None
        if ACT_QUANT and getattr(weight, "_dgpp_quantized", False):
            x = act_quant_dequant(x, 128, True)
        return F.linear(x.float(), weight.float()).to(x.dtype)

    model.linear = linear

    # Lean constructors: model.py's Linear allocates its storage at construction (an expert layer's
    # gigabytes of placeholders); the weights come from the shards, so the constructor keeps shapes.
    def lean_linear_init(self, in_features, out_features, bias=False, dtype=None):
        torch.nn.Module.__init__(self)
        self.in_features, self.out_features = in_features, out_features
        self._dtype = dtype
        self.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)
        self.register_parameter("scale", None)
        self.register_parameter("bias", None)

    model.Linear.__init__ = lean_linear_init
    # ColumnParallelLinear / RowParallelLinear call super().__init__(in, out, bias, dtype) with the
    # part sizes at world 1: the lean init takes them.
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def set_linear(lin, prefix, fp32=False):
        w = shards.linear_weight(prefix)
        quant = shards.has(prefix + ".scale")
        w = w.float() if fp32 else w.to(torch.bfloat16)
        lin.weight = torch.nn.Parameter(w, requires_grad=False)
        lin.weight._dgpp_quantized = quant

    def set_param(module, attr, name, dtype=None, hc_fn=False):
        w = shards.get(name)
        if hc_fn and args.hc_bf16:
            w = w.to(torch.bfloat16)
        w = w.to(dtype) if dtype is not None else w
        setattr(module, attr, torch.nn.Parameter(w, requires_grad=False))

    def load_compressor(comp, p):
        set_param(comp, "ape", p + "ape", torch.float32)
        set_linear(comp.wkv, p + "wkv", fp32=True)
        set_linear(comp.wgate, p + "wgate", fp32=True)
        set_param(comp.norm, "weight", p + "norm.weight", torch.float32)

    def load_attn(attn, p):
        set_param(attn, "attn_sink", p + "attn_sink", torch.float32)
        set_linear(attn.wq_a, p + "wq_a")
        set_param(attn.q_norm, "weight", p + "q_norm.weight", torch.float32)
        set_linear(attn.wq_b, p + "wq_b")
        set_linear(attn.wkv, p + "wkv")
        set_param(attn.kv_norm, "weight", p + "kv_norm.weight", torch.float32)
        set_linear(attn.wo_a, p + "wo_a")
        set_linear(attn.wo_b, p + "wo_b")
        if attn.compress_ratio:
            load_compressor(attn.compressor, p + "compressor.")
            if attn.indexer is not None:
                set_linear(attn.indexer.wq_b, p + "indexer.wq_b")
                set_linear(attn.indexer.weights_proj, p + "indexer.weights_proj")
                load_compressor(attn.indexer.compressor, p + "indexer.compressor.")

    def load_block(blk, p, hashed):
        for site in ("attn", "ffn"):
            set_param(blk, f"hc_{site}_fn", p + f"hc_{site}_fn", torch.float32, hc_fn=True)
            set_param(blk, f"hc_{site}_base", p + f"hc_{site}_base", torch.float32)
            set_param(blk, f"hc_{site}_scale", p + f"hc_{site}_scale", torch.float32)
        set_param(blk.attn_norm, "weight", p + "attn_norm.weight", torch.float32)
        set_param(blk.ffn_norm, "weight", p + "ffn_norm.weight", torch.float32)
        load_attn(blk.attn, p + "attn.")
        gate = blk.ffn.gate
        set_param(gate, "weight", p + "ffn.gate.weight")
        if hashed:
            gate.tid2eid = torch.nn.Parameter(shards.get(p + "ffn.gate.tid2eid").long(), requires_grad=False)
        else:
            set_param(gate, "bias", p + "ffn.gate.bias", torch.float32)
        for nm in ("w1", "w2", "w3"):
            set_linear(getattr(blk.ffn.shared_experts, nm), p + f"ffn.shared_experts.{nm}")
        cache = {}

        def make_forward(i, expert):
            def fwd(x, weights=None):
                if i not in cache:
                    cache[i] = {nm: shards.linear_weight(p + f"ffn.experts.{i}.{nm}") for nm in ("w1", "w2", "w3")}
                for nm in ("w1", "w2", "w3"):
                    lin = getattr(expert, nm)
                    lin.weight = torch.nn.Parameter(cache[i][nm], requires_grad=False)
                    lin.weight._dgpp_quantized = True
                return model.Expert.forward(expert, x, weights)
            return fwd

        for i, expert in enumerate(blk.ffn.experts):
            if expert is not None:
                expert.forward = make_forward(i, expert)

    def new_block(l):
        blk = model.Block(l, margs)
        load_block(blk, f"layers.{l}.", l < margs.n_hash_layers)
        return blk

    # The reference's routing and selections, captured by hooks.
    gate_log, sel_log = {}, {}

    def hook(blk, key):
        gate = blk.ffn.gate
        orig = gate.forward

        def gfwd(x, input_ids=None):
            weights, indices = orig(x, input_ids)
            gate_log[key] = indices.sort(dim=-1).values.cpu().numpy()
            return weights, indices

        gate.forward = gfwd
        if getattr(blk.attn, "indexer", None) is not None:
            idx = blk.attn.indexer
            oidx = idx.forward

            def ifwd(x, qr, start_pos, offset):
                r = oidx(x, qr, start_pos, offset)
                sel_log[key] = (r - offset).masked_fill(r < 0, -1)[0].cpu().numpy()
                return r

            idx.forward = ifwd

    routes, sels = dump["routes"], dump["sels"]
    indexed = [l for l in range(margs.n_layers) if margs.compress_ratios[l] == 4]
    layers = [int(v) for v in args.layers.split(",")] if args.layers else list(range(L_dump))
    embed = shards.get("embed.weight")
    h0 = embed[input_ids].unsqueeze(2).repeat(1, 1, hc, 1)
    results = []

    def compare(l, mode, got, want, key, sel_engine, t0, rows_label="rows"):
        """Relative l2 over every row and over the rows routed AND selected alike."""
        n = got.shape[0]
        d = rel_l2(got, want)
        rows = np.array([rel_l2(got[t], want[t]) for t in range(n)])
        line = f"layer {l:2d} {mode:14s}: rel l2 {d:.5f} (worst row {rows.max():.5f} at t{int(rows.argmax())})"
        flips = set()
        if key in gate_log and routes is not None and mode != "decode":
            ref = gate_log[key]
            flips |= {t for t in range(n) if not np.array_equal(ref[t], np.sort(routes[l, t]))}
            if flips:
                line += f"; route flips {len(flips)}"
        if key in sel_log and sel_engine is not None:
            ref = sel_log[key]
            sf = set()
            for t in range(n):
                r = set(int(v) for v in ref[t] if v >= 0)
                g = set(int(v) for v in sel_engine[t] if v >= 0)
                if r != g:
                    sf.add(t)
            line += f"; selections {n - len(sf)}/{n} equal"
            flips |= sf
        kept = [t for t in range(n) if t not in flips]
        dk = rel_l2(got[kept], want[kept]) if kept else float("nan")
        if flips and kept:
            line += f"; {len(kept)} {rows_label} alike: rel l2 {dk:.5f}, worst {rows[kept].max():.5f}"
        print(line + f" in {time.time() - t0:.1f} s", flush=True)
        return dk if kept else d

    dec = dump.get("dec_states") if not args.no_decode else None
    N = 0 if dec is None else dec.shape[0]
    P = T - N
    h_chain = h0
    chain_ok = True
    main_hidden_pre = {}
    main_hidden_dec = {}
    for l in range(L_dump):
        if l in margs.dspark_target_layer_ids:
            main_hidden_pre[l] = states[l].reshape(T, hc, H).to(torch.bfloat16).mean(dim=1)
            if N > 0:
                main_hidden_dec[l] = dec[:, l].reshape(N, hc, H).to(torch.bfloat16).mean(dim=1)
        if l not in layers or args.draft_only:
            chain_ok = False
            continue
        want = states[l].reshape(T, hc, H)
        x_iso = h0 if l == 0 else states[l - 1].reshape(1, T, hc, H).to(torch.bfloat16)
        with torch.inference_mode():
            if chain_ok:
                t0 = time.time()
                blk = new_block(l)
                hook(blk, ("c", l))
                h_chain = blk(h_chain, 0, input_ids)
                results.append(("chained", l, compare(l, "chained", h_chain[0], want, ("c", l),
                                                      sels[indexed.index(l)] if l in indexed else None, t0)))
                del blk
            if args.chained_only:
                continue
            t0 = time.time()
            blk = new_block(l)
            hook(blk, ("i", l))
            out = blk(x_iso, 0, input_ids)
            results.append(("isolated", l, compare(l, "isolated", out[0], want, ("i", l),
                                                   sels[indexed.index(l)] if l in indexed else None, t0)))
            del blk
            if N > 0:
                # The decode path: this block's caches from a prefill of the first P rows (fed the
                # engine's layer l-1 rows), then every step fed the engine's layer l-1 decode row.
                t0 = time.time()
                blk = new_block(l)
                blk(x_iso[:, :P].clone(), 0, input_ids[:, :P])
                hook(blk, ("d", l))
                got, sel_e, sel_r = [], [], []
                for s in range(N):
                    x = (embed[input_ids[:, P + s:P + s + 1]].unsqueeze(2).repeat(1, 1, hc, 1) if l == 0
                         else dec[s, l - 1].reshape(1, 1, hc, H).to(torch.bfloat16))
                    y = blk(x, P + s, input_ids[:, P + s:P + s + 1])
                    got.append(y[0, 0])
                    if ("d", l) in sel_log:
                        sel_r.append(sel_log[("d", l)][0])
                        sel_e.append(dump["dec_sels"][s, indexed.index(l)])
                got = torch.stack(got)
                if sel_r:
                    sel_log[("d", l)] = np.stack([np.pad(r, (0, max(0, len(sel_e[0]) - len(r))), constant_values=-1)
                                                  for r in sel_r])
                results.append(("decode", l, compare(l, "decode", got, dec[:, l].reshape(N, hc, H), ("d", l),
                                                     np.stack(sel_e) if sel_e else None, t0, "steps")))
                del blk

    # ---- the head ---------------------------------------------------------------------------
    ok = True
    if dump["logits"].shape[0] > 0 and L_dump == margs.n_layers and not args.draft_only:
        t0 = time.time()
        R = dump["logits"].shape[0]
        last = model.Block.__new__(model.Block)
        torch.nn.Module.__init__(last)
        last.norm_eps, last.hc_eps = margs.norm_eps, margs.hc_eps
        fn = shards.get("hc_head_fn")
        if args.hc_bf16:
            fn = fn.to(torch.bfloat16)
        fn = fn.float()
        base, scale = shards.get("hc_head_base").float(), shards.get("hc_head_scale").float()
        norm = model.RMSNorm(H, margs.norm_eps)
        norm.weight = torch.nn.Parameter(shards.get("norm.weight").float(), requires_grad=False)
        head_w = shards.get("head.weight").float()
        with torch.inference_mode():
            x = states[L_dump - 1].reshape(1, T, hc, H).to(torch.bfloat16)[:, T - R:]
            y = model.Block.hc_head(last, x, fn, scale, base)
            logits = F.linear(norm(y).float(), head_w)[0]
        d = rel_l2(logits, dump["logits"])
        agree = int((logits.argmax(-1) == dump["logits"].argmax(-1)).sum())
        print(f"head (isolated)        : logits rel l2 {d:.5f}, argmax {agree}/{R} equal in {time.time() - t0:.1f} s")
        results.append(("head", -1, d))
        if chain_ok:
            # The reference's OWN prediction: its chained states through the head, beside the engine's
            # logits (a decision the engine takes at a small margin is the model's when both agree).
            with torch.inference_mode():
                yc = model.Block.hc_head(last, h_chain[:, T - R:], fn, scale, base)
                lc = F.linear(norm(yc).float(), head_w)[0]
            dc = rel_l2(lc, dump["logits"])
            agree_c = int((lc.argmax(-1) == dump["logits"].argmax(-1)).sum())
            print(f"head (chained)         : logits rel l2 {dc:.5f}, argmax {agree_c}/{R} equal")

            def top(v):
                vals, ids = torch.topk(v.float(), 5)
                return " ".join(f"{int(i)}:{float(x):.3f}" for i, x in zip(ids, vals))

            print(f"  last row, reference : {top(lc[-1])}")
            print(f"  last row, engine    : {top(dump['logits'][-1])}")
        if N > 0:
            with torch.inference_mode():
                x = dec[:, L_dump - 1].reshape(N, 1, hc, H).to(torch.bfloat16)
                y = model.Block.hc_head(last, x, fn, scale, base)
                dl = F.linear(norm(y).float(), head_w)[:, 0]
            d = rel_l2(dl, dump["dec_logits"])
            agree = int((dl.argmax(-1) == dump["dec_logits"].argmax(-1)).sum())
            print(f"head (decode steps)    : logits rel l2 {d:.5f}, argmax {agree}/{N} equal")
            results.append(("head-decode", -1, d))

    # ---- the DSpark draft ---------------------------------------------------------------------
    if "draft_row" in dump and N > 0 and len(main_hidden_dec) == len(margs.dspark_target_layer_ids):
        t0 = time.time()
        B, n_mtp = margs.dspark_block_size, margs.n_mtp_layers
        emb = model.ParallelEmbedding.__new__(model.ParallelEmbedding)
        torch.nn.Module.__init__(emb)
        emb.vocab_size, emb.dim = margs.vocab_size, H
        emb.weight = torch.nn.Parameter(embed, requires_grad=False)
        head = model.ParallelHead.__new__(model.ParallelHead)
        torch.nn.Module.__init__(head)
        head.vocab_size, head.dim = margs.vocab_size, H
        head.weight = torch.nn.Parameter(shards.get("head.weight").float(), requires_grad=False)
        stages = []
        for s in range(n_mtp):
            blk = model.DSparkBlock(margs.n_layers + s, margs)
            p = f"mtp.{s}."
            load_block(blk, p, False)
            if s == 0:
                set_linear(blk.main_proj, p + "main_proj")
                set_param(blk.main_norm, "weight", p + "main_norm.weight", torch.float32)
            if s == n_mtp - 1:
                set_param(blk.norm, "weight", p + "norm.weight", torch.float32)
                blk.markov_head.markov_w1.weight = torch.nn.Parameter(shards.get(p + "markov_head.markov_w1.weight"),
                                                                      requires_grad=False)
                blk.markov_head.markov_w2.weight = torch.nn.Parameter(
                    shards.get(p + "markov_head.markov_w2.weight").float(), requires_grad=False)
                set_linear(blk.confidence_head.proj, p + "confidence_head.proj", fp32=True)
                set_param(blk, "hc_head_fn", p + "hc_head_fn", torch.float32, hc_fn=True)
                set_param(blk, "hc_head_base", p + "hc_head_base", torch.float32)
                set_param(blk, "hc_head_scale", p + "hc_head_scale", torch.float32)
            blk.embed, blk.head = emb, head
            blk.temperature = 0
            stages.append(blk)
        targets = list(margs.dspark_target_layer_ids)
        with torch.inference_mode():
            # the rings from the prefill rows (start_pos 0), then a draft per step
            mh = torch.cat([main_hidden_pre[l][:P] for l in targets], dim=-1).unsqueeze(0)
            h, main_x = stages[0].forward_embed(mh, input_ids[:, :1].new_zeros(1))
            for blk in stages:
                h = blk(h, 0, None, main_x)
            base_d, row_d, conf_d = [], [], []
            for s in range(N):
                pos = P + s
                mh = torch.cat([main_hidden_dec[l][s] for l in targets], dim=-1).reshape(1, 1, -1)
                nxt = torch.tensor([dump["draft_next"][s]], dtype=torch.long)
                h, main_x = stages[0].forward_embed(mh, nxt)
                for blk in stages:
                    h = blk(h, pos, nxt, main_x)
                lastb = stages[-1]
                x = lastb.hc_head(h, lastb.hc_head_fn, lastb.hc_head_scale, lastb.hc_head_base)
                logits = head(lastb.norm(x), full_logits=True)[0]  # [B, V]
                bias, m_embed = lastb.markov_head(nxt)
                conf = lastb.confidence_head(x[:, :1], m_embed.unsqueeze(1))
                base_d.append(rel_l2(logits, dump["draft_base"][s]))
                row_d.append(rel_l2(logits[0] + bias[0], dump["draft_row"][s]))
                conf_d.append(abs(float(conf.flatten()[0]) - dump["draft_conf"][s]))
                eng_base = dump["draft_base"][s]
                eng_base = eng_base if torch.is_tensor(eng_base) else torch.from_numpy(np.asarray(eng_base))
                same = (logits.float().argmax(-1) == eng_base.float().argmax(-1)).tolist()
                row0 = int((logits[0] + bias[0]).float().argmax()) == int(
                    (dump["draft_row"][s] if torch.is_tensor(dump["draft_row"][s])
                     else torch.from_numpy(np.asarray(dump["draft_row"][s]))).float().argmax())
                print(f"  draft step {s}: block base argmax equal per row {same}, row 0 (biased) argmax equal {row0}, "
                      f"base rel l2 {base_d[-1]:.5f}, confidence {float(conf.flatten()[0]):.3f} vs engine "
                      f"{float(dump['draft_conf'][s]):.3f}", flush=True)
        print(f"dspark draft ({N} steps): base logits rel l2 max {max(base_d):.5f}, row 0 (Markov-biased) max "
              f"{max(row_d):.5f}, confidence |diff| max {max(conf_d):.4f} in {time.time() - t0:.1f} s")
        results.append(("draft", -1, max(base_d)))

    worst = max((v for m, _, v in results if m in ("isolated", "decode")), default=0.0)
    ok = ok and worst <= args.budget
    print(f"{'OK' if ok else 'FAIL'}: worst isolated / decode relative l2 {worst:.5f} over the rows routed and "
          f"selected alike against the budget {args.budget}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
