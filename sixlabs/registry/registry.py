#!/usr/bin/env python3
"""registry.py — 6ix.cpp's model registry: what the engine serves, how each model is launched,
and how far each one has been verified. The data is models.json beside this file; README.md
describes the format.

  python3 sixlabs/registry/registry.py check
  python3 sixlabs/registry/registry.py table [--status working,compiled]
  python3 sixlabs/registry/registry.py show MODEL
  python3 sixlabs/registry/registry.py resolve MODEL [--world N] [--checkpoint REPO] [--variant NAME]
                                       [--set key=value ...] [--allow-unverified] [--out FILE]
  python3 sixlabs/registry/registry.py launch MODEL [the same options] [--dry-run]

`resolve` writes the deployment JSON `scripts/dgpp-cluster` takes: the checkpoint's template,
the family's default engine settings for keys the template leaves out, then --set. It refuses a
model that has not been verified on a GPU unless told otherwise, and one that cannot be served
always. `launch` is resolve + `dgpp-cluster up`. Standard library only.
"""
import argparse, copy, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ORDER = ["working", "upstream", "compiled", "groundwork", "refused"]   # best first
LAUNCHABLE = ("working", "upstream")
REPO_PATH = re.compile(r"^(sixlabs|docs|deploy|benchmarks)/\S+$|^[A-Z0-9_]+\.md$")


class RegistryError(Exception):
    pass


def load(path=None):
    with open(path or os.path.join(HERE, "models.json")) as f:
        return json.load(f)


def model_status(model):
    """A model is as far along as its best checkpoint."""
    have = {c.get("status") for c in model.get("checkpoints", [])}
    return next((s for s in ORDER if s in have), "refused")


def default_checkpoint(model):
    d = [c for c in model.get("checkpoints", []) if c.get("default")]
    return d[0] if d else None


def server_families(root):
    """The family names the serve binary can return, read from its source."""
    out = set()
    with open(os.path.join(root, "apps", "dgpp_serve.cpp")) as f:
        for line in f:
            # `name()` itself, not `kv_dtype_name()` and the like.
            if re.search(r"(?<!\w)name\(\) const override", line) and "return" in line:
                out.update(re.findall(r'"([a-z0-9_]+)"', line.split("return", 1)[1]))
    return out


# ── check ────────────────────────────────────────────────────────────────────────────────────
def prefill_budget_families(root):
    """The families the serve binary accepts an explicit prefill budget for, read from the refusal
    in apps/dgpp_serve.cpp; None when that code cannot be found."""
    with open(os.path.join(root, "apps/dgpp_serve.cpp")) as f:
        m = re.search(r"prefill_budget_tokens > 0 &&(.{0,400}?)\{", f.read(), re.S)
    return set(re.findall(r'!= "([a-z0-9_]+)"', m.group(1))) if m else None


def check(reg, root=ROOT):
    """Every inconsistency in the registry, and between it and the tree. Empty list = clean."""
    bad = []
    if reg.get("schema") != 1:
        bad.append("schema must be 1")
    statuses, formats, fams = set(reg.get("statuses", {})), set(reg.get("formats", [])), reg.get("families", {})
    if statuses != set(ORDER):
        bad.append(f"statuses must be exactly {ORDER}")

    def text(rel):
        try:
            with open(os.path.join(root, rel)) as f:
                return f.read()
        except OSError:
            bad.append(f"cannot read {rel}")
            return ""

    arch_src, cfg_src = text("src/loaders/architecture.cpp"), text("src/serve/cluster_config.cpp")
    for name, fam in fams.items():
        for key, typ in (("architectures", list), ("served", bool), ("on_main", bool), ("engine_defaults", dict), ("tokenizer", str)):
            if not isinstance(fam.get(key), typ):
                bad.append(f"family {name}: '{key}' missing or not a {typ.__name__}")
        if fam.get("on_main"):
            for a in fam.get("architectures", []):
                if f'"{a}' not in arch_src:
                    bad.append(f"family {name}: architecture prefix '{a}' is not recognised in src/loaders/architecture.cpp")
        for k in fam.get("engine_defaults", {}):
            if f'"{k}"' not in cfg_src:
                bad.append(f"family {name}: engine default '{k}' is not a key src/serve/cluster_config.cpp parses")
    served = {n for n, f in fams.items() if f.get("served")}
    try:
        in_server = server_families(root)
        for n in sorted(served - in_server):
            bad.append(f"family {n} is marked served but apps/dgpp_serve.cpp has no family of that name")
        for n in sorted(in_server - served):
            bad.append(f"apps/dgpp_serve.cpp serves family {n}, which the registry does not list as served")
        budget_ok = prefill_budget_families(root)
        for n, f in fams.items():
            if budget_ok is not None and "prefill_budget_tokens" in f.get("engine_defaults", {}) and n not in budget_ok:
                bad.append(f"family {n}: engine default 'prefill_budget_tokens' is refused by apps/dgpp_serve.cpp for this family "
                           f"(accepted: {', '.join(sorted(budget_ok))})")
    except OSError:
        bad.append("cannot read apps/dgpp_serve.cpp")

    aliases = {}
    for mid, m in reg.get("models", {}).items():
        where = f"model {mid}"
        fam = fams.get(m.get("family"))
        if fam is None:
            bad.append(f"{where}: unknown family '{m.get('family')}'")
            continue
        for key in ("tool_calls", "thinking", "vision"):
            if key not in m.get("text", {}):
                bad.append(f"{where}: text.{key} missing")
        cps = m.get("checkpoints", [])
        if sum(1 for c in cps if c.get("default")) != 1:
            bad.append(f"{where}: exactly one checkpoint must be the default")
        alias = m.get("fleet", {}).get("alias")
        if alias:
            if alias in aliases:
                bad.append(f"{where}: fleet alias '{alias}' is also {aliases[alias]}'s")
            aliases[alias] = mid
        for c in cps:
            cw = f"{where} / {c.get('repo', '?')}"
            st = c.get("status")
            if "repo" not in c or st not in statuses:
                bad.append(f"{cw}: needs a repo and a status from {ORDER}")
                continue
            if c.get("format") not in formats:
                bad.append(f"{cw}: format '{c.get('format')}' is not in formats")
            if st in ("working", "upstream", "compiled") and (not fam.get("on_main") or m.get("branch")):
                bad.append(f"{cw}: status {st} but the family is not on main")
            if st in LAUNCHABLE and not fam.get("served"):
                bad.append(f"{cw}: status {st} but family {m['family']} is not served")
            if st == "working":
                v = c.get("verified", {})
                if not all(v.get(k) for k in ("date", "box", "engine_commit", "record")):
                    bad.append(f"{cw}: working needs verified.date, box, engine_commit and record")
                elif not os.path.exists(os.path.join(root, v["record"])):
                    bad.append(f"{cw}: verified.record {v['record']} does not exist")
                if not any("measured" in w for w in c.get("worlds", {}).values()):
                    bad.append(f"{cw}: working needs a measured block in at least one world")
            if st == "compiled" and not os.path.exists(os.path.join(root, c.get("gpu_steps", "\0"))):
                bad.append(f"{cw}: compiled needs gpu_steps pointing at the verification steps")
            if st in ("groundwork", "refused") and not c.get("notes"):
                bad.append(f"{cw}: {st} needs notes saying why")
            for world, w in c.get("worlds", {}).items():
                for label, rel in [("template", w.get("template"))] + [(f"variant {k}", v) for k, v in w.get("variants", {}).items()]:
                    if not rel or not os.path.exists(os.path.join(root, rel)):
                        bad.append(f"{cw}: world {world} {label} {rel} does not exist")
                        continue
                    try:
                        with open(os.path.join(root, rel)) as f:
                            t = json.load(f)
                    except ValueError as e:
                        bad.append(f"{cw}: {rel} is not JSON ({e})")
                        continue
                    if t.get("model") != c["repo"]:
                        bad.append(f"{cw}: {rel} names model '{t.get('model')}'")
                    if str(t.get("world_size")) != str(world):
                        bad.append(f"{cw}: {rel} has world_size {t.get('world_size')}, registered under world {world}")
                src = w.get("measured", {}).get("source", "")
                if REPO_PATH.match(src) and not os.path.exists(os.path.join(root, src)):
                    bad.append(f"{cw}: measured.source {src} does not exist")
        for b in m.get("baselines", []):
            if not b.get("engine") or not b.get("source"):
                bad.append(f"{where}: every baseline needs an engine and a source")
    return bad


# ── table ────────────────────────────────────────────────────────────────────────────────────
def _num(v):
    if isinstance(v, list):
        return f"{v[0]:g}–{v[1]:g}"
    return "" if v is None else f"{v:g}"


def _speed(t):
    if not t:
        return "", ""
    single = " / ".join(x for x in (_num(t.get("single_prose")), _num(t.get("single_code"))) if x)
    multi = " / ".join(_num(t.get(k)) or "–" for k in ("c2", "c4", "c8")) if any(k in t for k in ("c2", "c4", "c8")) else ""
    return single, multi


def table(reg, only=None):
    """One row per engine per model: 6ix.cpp first, then the baselines it is compared with."""
    rows = ["| Model | Engine | Status | Load cold / warm (s) | Memory (GiB) | Single stream tok/s (prose / code) | 2 / 4 / 8 streams |",
            "|---|---|---|---|---|---|---|"]
    for mid, m in sorted(reg["models"].items(), key=lambda kv: (ORDER.index(model_status(kv[1])), kv[0])):
        st = model_status(m)
        if only and st not in only:
            continue
        c = default_checkpoint(m) or {}
        worlds = c.get("worlds", {})
        world = min(worlds, key=int) if worlds else None
        ms = worlds.get(world, {}).get("measured", {}) if world else {}
        load_ = " / ".join(x for x in (_num(ms.get("load_cold_s")), _num(ms.get("load_warm_s"))) if x)
        mem = _num(ms.get("gpu_gib")) or (_num(ms.get("plan_gib")) + " planned" if ms.get("plan_gib") else "")
        single, multi = _speed(ms.get("tok_s"))
        label = f"6ix.cpp, {c.get('format', '?')}" + (f", {world} Spark{'s' if world != '1' else ''}" if world else "")
        rows.append(f"| {m['title']} | {label} | {st} | {load_} | {mem} | {single} | {multi} |")
        for b in m.get("baselines", []):
            single, multi = _speed(b.get("tok_s"))
            rows.append(f"| | {b['engine']} | baseline | {_num(b.get('load_cold_s'))} | {_num(b.get('gpu_gib'))} | {single} | {multi} |")
    return "\n".join(rows)


# ── resolve / launch ─────────────────────────────────────────────────────────────────────────
def resolve(reg, model_id, world=None, checkpoint=None, variant=None, sets=(), allow_unverified=False, root=ROOT):
    """(deployment config, provenance) for one model, or RegistryError saying why not."""
    m = reg.get("models", {}).get(model_id)
    if m is None:
        raise RegistryError(f"no model '{model_id}' in the registry (try: registry.py table)")
    cps = m["checkpoints"]
    c = next((x for x in cps if x["repo"] == checkpoint), None) if checkpoint else default_checkpoint(m)
    if c is None:
        raise RegistryError(f"{model_id}: no checkpoint '{checkpoint}'; it has {[x['repo'] for x in cps]}")
    st = c["status"]
    if st in ("groundwork", "refused"):
        raise RegistryError(f"{model_id} ({c['repo']}) cannot be served: {st}. {c.get('notes', '')}")
    if st not in LAUNCHABLE and not allow_unverified:
        raise RegistryError(f"{model_id} ({c['repo']}) is '{st}': {reg['statuses'][st]} "
                            f"Verify it with {c.get('gpu_steps', 'its gpu-steps file')}, or pass --allow-unverified.")
    worlds = c.get("worlds", {})
    if not worlds:
        raise RegistryError(f"{model_id} ({c['repo']}) has no deployment template registered")
    world = str(world) if world else min(worlds, key=int)
    if world not in worlds:
        raise RegistryError(f"{model_id} ({c['repo']}) is registered for {sorted(worlds, key=int)} Spark(s), not {world}")
    w = worlds[world]
    rel = w["template"]
    if variant:
        if variant not in w.get("variants", {}):
            raise RegistryError(f"{model_id}: no variant '{variant}' at world {world}; it has {sorted(w.get('variants', {}))}")
        rel = w["variants"][variant]
    with open(os.path.join(root, rel)) as f:
        cfg = json.load(f)
    engine = dict(cfg.get("engine", {}))
    from_family = {k: v for k, v in reg["families"][m["family"]].get("engine_defaults", {}).items() if k not in engine}
    engine.update(from_family)
    engine.update(w.get("engine", {}))
    overrides = {}
    for kv in sets:
        if "=" not in kv:
            raise RegistryError(f"--set wants key=value, got '{kv}'")
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except ValueError:
            pass
        overrides[k] = v
    engine.update(overrides)
    cfg = copy.deepcopy(cfg)
    cfg["engine"] = engine
    prov = {"model": model_id, "checkpoint": c["repo"], "revision": c.get("revision"), "status": st, "family": m["family"],
            "world": int(world), "template": rel, "from_family_defaults": from_family, "from_set": overrides}
    return cfg, prov


def site_config_path(reg, model_id, prov, root=ROOT):
    # deploy/cluster_*.json (not *.example.json) is git-ignored: site copies live there.
    fmt = next(c["format"] for c in reg["models"][model_id]["checkpoints"] if c["repo"] == prov["checkpoint"])
    return os.path.join(root, "deploy", f"cluster_{model_id}_{fmt}_w{prov['world']}.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description="6ix.cpp model registry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    t = sub.add_parser("table")
    t.add_argument("--status", default="")
    s = sub.add_parser("show")
    s.add_argument("model")
    for name in ("resolve", "launch"):
        p = sub.add_parser(name)
        p.add_argument("model")
        p.add_argument("--world")
        p.add_argument("--checkpoint")
        p.add_argument("--variant")
        p.add_argument("--set", action="append", default=[])
        p.add_argument("--allow-unverified", action="store_true")
        p.add_argument("--out")
        if name == "launch":
            p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    reg = load()

    if a.cmd == "check":
        bad = check(reg)
        for b in bad:
            print("PROBLEM:", b)
        counts = {}
        for m in reg["models"].values():
            counts[model_status(m)] = counts.get(model_status(m), 0) + 1
        print(f"{len(reg['models'])} models, {len(reg['families'])} families: " + ", ".join(f"{counts[s]} {s}" for s in ORDER if s in counts))
        print("registry OK" if not bad else f"{len(bad)} problem(s)")
        return 1 if bad else 0
    if a.cmd == "table":
        print(table(reg, [x for x in a.status.split(",") if x] or None))
        return 0
    if a.cmd == "show":
        if a.model not in reg["models"]:
            print(f"no model '{a.model}'", file=sys.stderr)
            return 2
        m = reg["models"][a.model]
        print(json.dumps({"status": model_status(m), "family_settings": reg["families"][m["family"]], **m}, indent=2))
        return 0
    try:
        cfg, prov = resolve(reg, a.model, a.world, a.checkpoint, a.variant, a.set, a.allow_unverified)
    except RegistryError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    out = a.out or site_config_path(reg, a.model, prov)
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(json.dumps(prov, indent=2))
    print(f"wrote {os.path.relpath(out, ROOT)}")
    if a.cmd == "launch":
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "dgpp-cluster"), "up", "--config", out]
        if a.dry_run:
            print("would run:", " ".join(cmd))
            return 0
        return subprocess.call(cmd, cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
