#!/usr/bin/env python3
"""gemma4_torch_check.py — the numpy reference against the model's own class (2026-10-04).

Loads a SMALL Gemma 4 checkpoint in the releases' container (tools/gemma4_synth.py make) into
transformers' Gemma4ForCausalLM — the text model the released Gemma4ForConditionalGeneration wraps
as `model.language_model`, with the same tied head and logit soft-cap — runs it in float32 on the
CPU and compares its logits, position by position, with tools/gemma4_reference.py on the same
files. The NVFP4 matrices are dequantized here by a second, independent implementation (plain
arithmetic, no lookup table), so the reference's decoder is checked too.

  gemma4_torch_check.py --ckpt DIR [--ids-json F] [--tol 2e-4]

What it shows: the reference's reading of the architecture (norm placement, the weightless value
norm, k_proj as the full layers' value, both rotary kinds, the window, the unscaled dot product,
layer_scalar, the soft-cap) agrees with modeling_gemma4.py. What it cannot show: anything about a
released checkpoint's values. It needs torch and transformers >= 5.5; nothing here touches a GPU.

The class is run in float32, where its embedding scale is sqrt(H) exactly; the reference is told
the same (--variant embscale=exact). The bf16 rounding of that scale (73.5 for H = 5376) is the
class's behaviour on a bf16 checkpoint and is checked separately below.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemma4_reference as ref  # noqa: E402


def e2m1(code):
    """One 4-bit code -> value: sign (bit 3), 2-bit exponent, 1-bit mantissa."""
    sign = -1.0 if code & 8 else 1.0
    e, m = (code >> 1) & 3, code & 1
    return sign * (0.5 * m if e == 0 else (1.0 + 0.5 * m) * 2.0 ** (e - 1))


def e4m3(code):
    sign = -1.0 if code & 0x80 else 1.0
    e, m = (code >> 3) & 0xF, code & 7
    return sign * ((m / 8.0) * 2.0 ** -6 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7))


def dequant_nvfp4(ck, base):
    codes = np.array(ck.raw(base + ".weight"))
    scales = np.array(ck.raw(base + ".weight_scale"))
    ws2 = float(np.array(ck.raw(base + ".weight_scale_2")).reshape(()))
    n, half = codes.shape
    w = np.zeros((n, 2 * half), np.float64)
    for r in range(n):
        for j in range(half):
            b = int(codes[r, j])
            s0 = e4m3(int(scales[r, (2 * j) // 16])) * ws2
            w[r, 2 * j] = e2m1(b & 0xF) * s0            # low nibble: the even column
            w[r, 2 * j + 1] = e2m1(b >> 4) * s0
    return w.astype(np.float32)


def torch_state_dict(ck):
    """Checkpoint tensors -> Gemma4ForCausalLM names (model.language_model.X -> model.X), floats.
    The release's unfused experts (layers.L.moe.experts.E.{gate,up,down}_proj) are stacked back into
    the class's two parameters: experts.gate_up_proj [E, 2I, H] — gate rows first — and
    experts.down_proj [E, H, I]."""
    import re
    import torch
    sd, experts = {}, {}
    prefix = "model.language_model."
    for name in ck.index:
        if not name.startswith(prefix):
            continue                                     # the encoders
        short = "model." + name[len(prefix):]
        dtype = ck.index[name][1]
        if name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
            continue
        if dtype == "U8":
            w = dequant_nvfp4(ck, name[:-len(".weight")])
        elif dtype == "BF16":
            w = ref.bf16_to_f32(np.array(ck.raw(name))).copy()
        else:
            raise SystemExit(f"{name}: unexpected dtype {dtype}")
        m = re.fullmatch(r"model\.layers\.(\d+)\.moe\.experts\.(\d+)\.(gate|up|down)_proj\.weight", short)
        if m:
            experts.setdefault(int(m.group(1)), {}).setdefault(int(m.group(2)), {})[m.group(3)] = w
        else:
            sd[short] = torch.from_numpy(w)
    for layer, by_expert in experts.items():
        order = sorted(by_expert)
        assert order == list(range(len(order))), f"layer {layer}: expert ids are not 0..E-1"
        sd[f"model.layers.{layer}.experts.gate_up_proj"] = torch.from_numpy(
            np.stack([np.concatenate([by_expert[e]["gate"], by_expert[e]["up"]]) for e in order]))
        sd[f"model.layers.{layer}.experts.down_proj"] = torch.from_numpy(np.stack([by_expert[e]["down"] for e in order]))
    return sd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ids-json")
    ap.add_argument("--tol", type=float, default=2e-4, help="max |logit difference| allowed")
    a = ap.parse_args()

    import torch
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    import transformers

    torch.manual_seed(0)
    torch.set_num_threads(4)
    with open(a.ids_json or os.path.join(a.ckpt, "ids.json")) as f:
        ids = json.load(f)
    ck = ref.Checkpoint(a.ckpt)
    tc = dict(ck.cfg["text_config"])
    tc.pop("model_type", None)
    tc.pop("dtype", None)
    cfg = Gemma4TextConfig(**tc)
    model = Gemma4ForCausalLM(cfg).float().eval()
    sd = torch_state_dict(ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The only key the checkpoint may lack is the tied head.
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing {missing[:8]} unexpected {list(unexpected)[:8]}")
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr() or torch.equal(
        model.lm_head.weight, model.model.embed_tokens.weight), "the head is not the embedding"

    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
        t_logits = out.logits[0].double().numpy()
        # The same sequence through the class's KV cache, one token at a time.
        past, steps = None, []
        for t in ids:
            o = model(input_ids=torch.tensor([[t]]), past_key_values=past, use_cache=True)
            past = o.past_key_values
            steps.append(o.logits[0, -1].double().numpy())
        t_step_logits = np.stack(steps)

    # The class in float32 multiplies by the exact Python constants; on a bf16 checkpoint both the
    # embedding scale and the router's H^-0.5 come out rounded to bf16 (checked at the end).
    exact = {"embscale": "exact", "routerscale": "exact"}
    results = []
    for dtype in ("float32", "float64"):
        m = ref.Model(a.ckpt, dtype=dtype, variants=exact)
        x, _ = ref.run_forward(m, ids)
        n_logits = m.logits(x).astype(np.float64)
        x1, _ = ref.run_forward(m, ids, chunk=1)
        n_step_logits = m.logits(x1).astype(np.float64)
        d_full = float(np.abs(n_logits - t_logits).max())
        d_step = float(np.abs(n_step_logits - t_step_logits).max())
        d_self = float(np.abs(n_step_logits - n_logits).max())
        argmax_same = bool((n_logits.argmax(-1) == t_logits.argmax(-1)).all())
        t_lp = t_logits - np.log(np.exp(t_logits - t_logits.max(-1, keepdims=True)).sum(-1, keepdims=True)) - t_logits.max(-1, keepdims=True)
        n_lp = ref.log_softmax64(n_logits)
        d_lp = float(np.abs(n_lp - t_lp).max())
        results.append((dtype, d_full, d_step, d_self, d_lp, argmax_same))
        print(f"reference {dtype}: max |logit diff| vs transformers {d_full:.3e} (whole sequence), "
              f"{d_step:.3e} (token by token, both through their caches), chunked vs whole {d_self:.3e}, "
              f"max |logprob diff| {d_lp:.3e}, argmax equal at every position: {argmax_same}")

    # The variants must MATTER on this fixture: each wrong convention has to move the logits, or
    # the fixture would not be telling the conventions apart.
    weak = []
    wrong = [("norm", "1p"), ("qkscale", "sqrt"), ("rope", "adjacent"), ("rope", "none"),
             ("globalrope", "full"), ("kvmap", "mod"), ("keqv", "normed"), ("keqv", "roped"),
             ("vnorm", "off"), ("window", "plus1"), ("window", "none"), ("scalar", "off"),
             ("softcap", "off"), ("act", "gelu"), ("act", "silu"), ("nibble", "hi"), ("ws2", "div"),
             ("embscale", "none")]
    if cfg.enable_moe_block:
        wrong += [("routerscale", "none"), ("routerin", "normed"), ("routernorm", "off"), ("expertin", "mlp"),
                  ("pes", "off"), ("moecombine", "moeonly"), ("moecombine", "mlponly")]
    for key, value in wrong:
        v = ref.Model(a.ckpt, variants={**exact, key: value})
        xv, _ = ref.run_forward(v, ids)
        d = float(np.abs(v.logits(xv).astype(np.float64) - t_logits).max())
        if d < 5 * a.tol:      # (erf GELU is within 2e-3 of the tanh form by construction)
            weak.append(f"{key}={value} ({d:.2e})")
        print(f"  variant {key}={value}: max |logit diff| vs transformers {d:.3e}")
    if cfg.enable_moe_block:
        # Not a convention the output can show: the renormalization multiplies all of a token's
        # expert weights by one factor, and post_feedforward_layernorm_2 divides it straight out
        # (to within eps). Reported, never counted.
        v = ref.Model(a.ckpt, variants={**exact, "renorm": "off"})
        xv, _ = ref.run_forward(v, ids)
        d = float(np.abs(v.logits(xv).astype(np.float64) - t_logits).max())
        print(f"  variant renorm=off: max |logit diff| vs transformers {d:.3e} (invisible by construction: "
              f"the norm after the experts removes a per-token factor)")

    # The embedding scale under the checkpoint's dtype: the class casts sqrt(H) to bf16.
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding
    scale_ok = True
    for H in (5376, 2816, cfg.hidden_size):
        emb = Gemma4TextScaledWordEmbedding(4, H, 0, embed_scale=H ** 0.5).to(torch.bfloat16)
        with torch.no_grad():
            emb.weight.fill_(1.0)
            got = float(emb(torch.tensor([1]))[0, 0].float())
        want = float(ref.f32_to_bf16_value(np.float32(math.sqrt(H))))
        scale_ok = scale_ok and got == want
        print(f"  bf16 embedding scale, H = {H}: transformers {got}, reference {want}")
        # The router's scalar: a Python float against a bf16 tensor is rounded to bf16 by torch.
        got = float((torch.ones(1, dtype=torch.bfloat16) * (H ** -0.5))[0].float())
        want = float(ref.f32_to_bf16_value(np.float32(H ** -0.5)))
        scale_ok = scale_ok and got == want
        print(f"  bf16 router input scale, H = {H}: torch {got}, reference {want}")

    ok = all(r[1] <= a.tol and r[2] <= a.tol and r[3] <= a.tol and r[5] for r in results) and not weak and scale_ok
    if weak:
        print("variants this fixture does not tell apart: " + ", ".join(weak))
    print(f"transformers {transformers.__version__}, torch {torch.__version__}, {len(ids)} tokens, "
          f"{cfg.num_hidden_layers} layers" + (f", MoE {cfg.num_experts} experts top-{cfg.top_k_experts}"
                                               if cfg.enable_moe_block else "") + ": " + ("AGREE" if ok else "DISAGREE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
