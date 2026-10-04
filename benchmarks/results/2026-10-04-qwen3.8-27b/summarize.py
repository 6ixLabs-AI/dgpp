#!/usr/bin/env python3
"""Derive current benchmark tables from the saved request-level measurements."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CLASSES = ("prose", "code", "json", "math", "chat")
# Presentation order groups model families, then node counts, with options adjacent.
# The measurement matrix retains its original campaign order.
DEPLOYMENTS = {
    "glm-flash-hybrid-w2": ("GLM-5.3-Flash NVFP4/FP8", "160K FP8 KV, 4 slots"),
    "glm-flash-hybrid-256k-w2": ("GLM-5.3-Flash NVFP4/FP8", "256K FP8 KV, 2 slots"),
    "glm-flash-hybrid-w4": ("GLM-5.3-Flash NVFP4/FP8", "768K BF16 KV, 4 slots"),
    "glm-flash-fp8-w4": ("GLM-5.3-Flash FP8", "384K BF16 KV, 4 slots"),
    "qwen-nvfp4-w1": ("Qwen3.8-Flash-Next NVFP4", "FP8 dense"),
    "qwen-nvfp4-bf16-w1": ("Qwen3.8-Flash-Next NVFP4", "BF16 dense"),
    "qwen-nvfp4-w2": ("Qwen3.8-Flash-Next NVFP4", "FP8 dense"),
    "qwen-yarn-w2": ("Qwen3.8-Flash-Next NVFP4", "FP8 dense, YaRN 512K, 2 slots"),
    "qwen-fp8-w2": ("Qwen3.8-Flash-Next FP8", "Default"),
    "qwen-fp8-w4": ("Qwen3.8-Flash-Next FP8", "Default"),
    "glm53-w4": ("GLM-5.3 int4/int8", "120K BF16 KV"),
    "glm53-fp8kv-w4": ("GLM-5.3 int4/int8", "208K FP8 KV"),
    "deepseek-w4": ("DeepSeek-V4.1-Flash MXFP4/FP8", "Default"),
    "mimo-w4": ("MiMo-V2.6-Flash MXFP4/FP8", "128K BF16 KV, 4 slots"),
    "mimo-w2": ("MiMo-V2.6-Flash MXFP4/FP8", "128K BF16 KV, 4 slots"),
    "mimo-w2-fp8kv": ("MiMo-V2.6-Flash MXFP4/FP8", "256K FP8 KV, 4 slots"),
    "dsv4-w4": ("DeepSeek-V4-Flash MXFP4/FP8", "1M BF16 KV, 6 slots"),
    "dsv4-w2": ("DeepSeek-V4-Flash MXFP4/FP8", "1M BF16 KV, 4 slots"),
    "qwen27b-fp8-w1": ("Qwen3.8-27B FP8", "DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots"),
    "qwen27b-fp8-w2": ("Qwen3.8-27B FP8", "MTP depth 3, FP8 head, 256K BF16 KV, 8 slots"),
    "qwen27b-fp8-w4": ("Qwen3.8-27B FP8", "MTP depth 3, FP8 head, 256K BF16 KV, 8 slots"),
}
DEPLOYMENT_HEADERS = ["Model / weights", "Nodes", "Options"]


def deployment_cells(entry):
    model, options = DEPLOYMENTS[entry["id"]]
    return [model, entry["world"], options]


def deployment_name(entry):
    model, nodes, options = deployment_cells(entry)
    return f"{model} · {nodes} {'node' if nodes == 1 else 'nodes'} · {options}"


def ordered_deployments(result):
    order = {key: i for i, key in enumerate(DEPLOYMENTS)}
    return sorted(result["deployments"], key=lambda entry: order[entry["id"]])


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def distribution(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def phases(path):
    data = read(path)
    if not data:
        return []
    groups = defaultdict(list)
    for phase in data["phases"]:
        groups[(phase["class"], phase["concurrency"])].append(phase)
    result = []
    for (name, concurrency), samples in groups.items():
        values = {"class": name, "concurrency": concurrency}
        measures = {
            "engine_tokens_per_s": lambda p: p["engine"]["tokens_per_s"],
            "wall_tokens_per_s": lambda p: p["metrics"]["wall_tokens_per_s"],
            "ms_per_pass": lambda p: p["engine"]["step_ms"] / p["engine"]["decode_steps"],
            "tokens_per_pass_per_request": lambda p: p["engine"]["decode_tokens"] / p["engine"]["decode_rows"],
            "ttft_ms": lambda p: statistics.mean((r["first"] - r["t0"]) * 1000 for r in p["requests"]),
        }
        for key, measurement in measures.items():
            values[key] = distribution([measurement(p) for p in samples])
        values["completion_tokens"] = [p["metrics"]["completion_tokens"] for p in samples]
        values["cached_prompt_tokens"] = [sum(r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                                               for r in p["requests"]) for p in samples]
        values["all_http_200"] = all(r["status"] == 200 for p in samples for r in p["requests"])
        result.append(values)
    return result


def prefill(path):
    data = read(path)
    if not data:
        return []
    groups = defaultdict(list)
    for sample in data["samples"]:
        groups[sample["requested_tokens"]].append(sample)
    return [{"target": length, "samples": len(samples),
             "prefill_s": distribution([s["prefill_ms"] / 1000 for s in samples]),
             "ttft_s": distribution([s["ttft_ms"] / 1000 for s in samples]),
             "ms_per_token": distribution([s["prefill_ms_per_token"] for s in samples]),
             "actual_prompt_tokens": [s["prompt_tokens"] for s in samples],
             "all_cold": all(s["cached_tokens"] == 0 for s in samples)}
            for length, samples in groups.items()]


def parity(default, other):
    a, b = read(default), read(other)
    if not a or not b:
        return None
    def answers(data):
        return {name: [(p["requests"][0]["text"], p["requests"][0]["tokens"])
                       for p in data["phases"] if p["class"] == name and p["concurrency"] == 1]
                for name in CLASSES}
    aa, bb = answers(a), answers(b)
    return {name: {"default_repeats": len(aa[name]), "mode_repeats": len(bb[name]),
                   "identical": bool(aa[name] and bb[name]) and len(set(aa[name] + bb[name])) == 1}
            for name in CLASSES}


def collect():
    result = {"manifest": read(HERE / "manifest.json"), "deployments": []}
    for entry in read(HERE / "matrix.json"):
        base = HERE / "raw" / entry["id"]
        default = base / "performance/default"
        pfile = base / "prefill/default/prefill.json"
        if not pfile.exists():
            pfile = default / "prefill.json"
        item = {**entry, "name": deployment_name(entry), "settings": read(HERE / entry["config"])["engine"],
                "greedy": phases(default / "greedy.json"), "sampled": phases(default / "sampled.json"),
                "prefill": prefill(pfile), "quality": read(base / "quality/default/eval/summary.json"),
                "modes": {}, "rank_checks": [], "isolation": None}
        for directory in sorted((base / "modes").glob("*")):
            if directory.is_dir():
                item["modes"][directory.name] = {
                    "greedy": phases(directory / "greedy.json"),
                    "parity": parity(default / "greedy.json", directory / "greedy.json")}
        for path in sorted(base.glob("**/rank-identity.json")):
            if any(part.startswith("attempt-") for part in path.relative_to(base).parts):
                continue
            check = read(path)
            # A world of one rank has nothing to compare; its eager engine (the
            # drafter template's default) writes no op stream at all. Only
            # launches with a collected stream, or a real world, are checks.
            if check.get("expected_ranks", 0) <= 1 and not check.get("md5"):
                continue
            item["rank_checks"].append({"path": str(path.relative_to(HERE)), **check})
        isolation = default / "isolation.log"
        if isolation.exists():
            lines = [s for s in isolation.read_text().splitlines() if "== isolation:" in s]
            item["isolation"] = lines[-1] if lines else None
        result["deployments"].append(item)
    return result


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)]) + "\n"


def span(samples, key, concurrency=1):
    values = [s[key]["median"] for s in samples if s["concurrency"] == concurrency]
    if not values:
        return "—"
    low, high = f"{min(values):.1f}", f"{max(values):.1f}"
    return low if low == high else f"{low}–{high}"


def render(result):
    deployments = ordered_deployments(result)
    commit = result["manifest"]["commit"]
    started = datetime.fromisoformat(result["manifest"]["started_at"]).date()
    ended = datetime.fromisoformat(result["manifest"].get("completed_at", result["manifest"]["started_at"])).date()
    dates = started.isoformat() if started == ended else f"{started.isoformat()} through {ended.isoformat()}"
    record = "../benchmarks/results/2026-10-04-qwen3.8-27b/"
    lines = ["# Benchmarks", "",
        f"Measured {dates} (UTC), using source revision `{commit[:12]}` "
        "and one release binary across the cluster. All results on this page come "
        "from this campaign. "
        f"[Raw results, exact configurations and commands]({record}README.md).", "",
        "[Performance](#serving-performance) · [Configuration](#hardware-and-configuration) · "
        "[Method](#workload-and-timing) · [Decode modes](#decode-modes) · "
        "[Quality](#quality-and-correctness) · [Long context](#long-context) · "
        "[Reproduce](#reproduce)", "",
        "## Serving performance", "",
        "Rates are tokens per second. Each range spans the five prompt classes' "
        "medians, with three repetitions per class. **C** is concurrent requests. "
        "The single-request column measures engine decode; the loaded column "
        "measures completed output over full request wall time, including admission "
        "and prefill. The two columns have different timing scopes.", "",
        "Rows are grouped by model family and node count, with configuration "
        "options next to each other throughout this page. **Nodes** is the tensor-parallel "
        "world size. **KV** is the shared key/value-cache token pool; **slots** "
        "is the configured concurrent-request limit. In configuration labels, "
        "K = 1,024 tokens.", ""]
    overview = []
    for d in deployments:
        p = {s["target"]: s["prefill_s"]["median"] for s in d["prefill"]}
        overview.append([*deployment_cells(d), span(d["greedy"], "engine_tokens_per_s"),
                         f"C{d['slots']}: " + span(d["greedy"], "wall_tokens_per_s", d["slots"]),
                         " / ".join(f"{p[t]:.3f}" for t in (2048, 8192, 32768))])
    lines += [table([*DEPLOYMENT_HEADERS, "C1 engine tok/s", "Loaded wall tok/s", "Cold prefill seconds: ~2K / ~8K / ~32K"], overview),
        "Rows cover the checked-in deployment templates, the GLM Flash FP8 "
        "checkpoint, and the three configuration variants described below. "
        "For Qwen NVFP4, the Options column identifies the dense projection "
        "format; the expert weights remain NVFP4 in both cases. All Qwen "
        "NVFP4 rows map the n-gram table from NVMe. YaRN 512K denotes the "
        "extended-context configuration.", "",
        "The two-node GLM-5.3-Flash NVFP4/FP8 rows use the same checkpoint: "
        "one has a 163,840-token FP8 KV pool and four request slots; the other "
        "has a 262,144-token FP8 KV pool and two slots. NVFP4/FP8 identifies "
        "the [mixed-weight checkpoint](model_cards/GLM-5.3-Flash-NVFP4-FP8.md), "
        "which combines NVFP4 main-stack routed experts with the remaining "
        "tensors from the FP8 release, retaining their original formats.", "",
        "## Hardware and configuration", "",
        "The cluster consists of four GB10 systems: MSI EdgeXpert (rank 0), "
        "ASUS GX10 (rank 1), NVIDIA DGX Spark (rank 2), and AI TOP ATOM (rank 3). "
        "Each has about 121 GiB of OS-visible unified memory and uses the "
        "200 Gb/s RoCE fabric. Measurements use one, "
        "two or four nodes, CUDA 13.0, driver 580.173.02 and the CMake `release` "
        "preset.", "",
        "Only one benchmark uses the cluster at a time; no profiler is "
        "attached. Models and decode graphs are loaded before timed requests. "
        "Rank 0 and the local microbenchmarks set `CUDA_DEVICE_MAX_CONNECTIONS=32`; "
        "peer processes inherit their login environment. The nodes differ in "
        "firmware and some OS packages; the active kernel, GPU driver, CUDA and "
        "RDMA versions match. "
        f"[Full environment comparison]({record}environment.md).", "",
        "Four-node launches after Node 3's second recovery also collect host "
        "memory and pressure approximately every second, GPU state approximately "
        "every five seconds, and Node 3's live kernel journal. Node 3's telemetry "
        "uses its separate Wi-Fi management link. These probes run "
        "during timing; earlier launches have no continuous telemetry.", "",
        "MTP is speculative multi-token prediction. A pass can commit several "
        "output tokens. DeepSeek uses its configured adaptive depth schedule. "
        "The Qwen3.8-27B single-node template drafts with the DFlash2 block drafter "
        "instead of MTP: one block forward proposes seven drafts a pass, the "
        "target verifies up to eight rows; its MTP depth-2 world is measured as the "
        "template's mode (the decode-modes table). "
        "KV capacity below is the deployment's shared token pool; it is not a "
        "promise that every concurrent request can use that full context.", ""]
    lines.append(table([*DEPLOYMENT_HEADERS, "Request slots", "KV pool tokens", "KV format", "Default MTP depth"],
        [[*deployment_cells(d), d["slots"], f"{d['settings']['kv_capacity']:,}",
          d["settings"].get("kv_dtype", "model default"),
          ("block drafter, 7 drafts" if d["settings"].get("dflash_model") else
           str(d["settings"].get("mtp_depth", 1)) + (" (adaptive)" if d["settings"].get("mtp_schedule") else ""))]
         for d in deployments]))
    lines += ["GLM-5.3-Flash FP8 uses the four-node Flash template with the FP8 model ID "
              "and `kv_capacity=393216`. The alternative Qwen BF16 row selects "
              "`dense_weights=checkpoint` and `fp8_head=gemv`; full GLM's FP8 KV row "
              "selects `kv_dtype=fp8`, `kv_capacity=212992`, and a 1.5 GiB prefix cache; "
              "the two-node GLM-5.3-Flash NVFP4/FP8 256K KV row selects two slots, "
              "`kv_capacity=262144`, and a 2 GiB prefix cache (the 160K KV row "
              "uses 1.5 GiB). Every effective configuration is saved in "
              f"the [configuration matrix]({record}matrix.json).", "",
        "## Workload and timing", "",
        "- **Decode:** the repository's prose, code, JSON, math and chat corpus; "
        "256 maximum output tokens, three repetitions. Greedy decoding chooses "
        "the highest-scoring next token (temperature zero). Sampled decoding "
        "draws from the token distribution (temperature one here); other sampling "
        "parameters retain each model's server defaults, recorded in its log. Thinking is disabled "
        "where the model exposes that setting; GLM-5.3 uses its default reasoning mode.",
        "- **Concurrency:** C1/C2/C4, plus the template's maximum if it exceeds four; "
        "the two-slot templates use C1/C2. Requests within a phase have distinct "
        "prompts, and each concurrency uses a different subset of its class. "
        "Repetitions reuse prompts with the configured prefix cache enabled. "
        "Loaded wall rates therefore describe this repeated-request workload.",
        "- **Cold prefill:** three different prompts per target length, constructed "
        "from the prepared GSM8K text with a distinct prefix. Each sample must "
        "report zero cached tokens. Targets are approximate; actual tokenizer "
        "counts, engine prefill time and client time to first token are retained.",
        "- **Engine decode rate:** `1000 × (tokens_generated − prompts_prefilled) / "
        "step_ms`. It excludes prefill and the first token emitted by each prefill. "
        "`ms/pass = step_ms / decode_steps`; at C1, `tokens/pass = decode_tokens / "
        "decode_steps`. Scheduler counters are reconciled against completed requests.",
        "- **Client wall rate:** total completion tokens divided by the interval "
        "from launching a request group to its last completion. An SSE update may "
        "contain multiple tokens; update counts are never used as token counts.", "",
        f"The [detailed tables]({record}README.md) include every measured concurrency, "
        "sampled results, per-class engine pass times, time to first token, and "
        "the minimum/median/maximum across greedy repetitions. Ranges in the overview are "
        "variation across prompt classes, not confidence intervals.", "",
        "## Single-request decode by class", "",
        "Engine tokens/s, greedy; median of three repetitions.", ""]
    lines.append(table([*DEPLOYMENT_HEADERS, *["JSON" if c == "json" else c.title() for c in CLASSES]],
        [[*deployment_cells(d), *[f"{next(s for s in d['greedy'] if s['class'] == c and s['concurrency'] == 1)['engine_tokens_per_s']['median']:.1f}"
                       for c in CLASSES]] for d in deployments]))
    lines += ["## Decode modes", "",
              "Single-request engine tokens/s, shown as the range of the five "
              "class medians. Plain decoding has MTP disabled. The deeper run uses "
              "depth two, or fixed depth five for DeepSeek. Full GLM's depth-two "
              "and DeepSeek's depth-five runs use two request slots. Other "
              "settings are saved with each run.", ""]
    lines.append(table([*DEPLOYMENT_HEADERS, "Plain", "Template default", "Other MTP depth", "DFlash2 drafter"],
        [[*deployment_cells(d), span(d["modes"].get("plain", {}).get("greedy", []), "engine_tokens_per_s"),
          span(d["greedy"], "engine_tokens_per_s"),
          span(next((d["modes"][k] for k in ("depth2", "depth3", "depth5", "mtp2") if k in d["modes"]), {}).get("greedy", []), "engine_tokens_per_s"),
          ("the template" if d["settings"].get("dflash_model") else
           span(d["modes"].get("dflash2", {}).get("greedy", []), "engine_tokens_per_s"))]
         for d in deployments]))
    lines += ["A dash means that mode was not part of that deployment's "
              "mode sweep. \"Other MTP depth\" is depth two for the Qwen3.8-27B "
              "two- and four-node templates (whose default is depth three) and the "
              "MTP depth-two world for its one-node drafter template; \"DFlash2 "
              "drafter\" is the block drafter run as a two- or four-node template's "
              "mode (the one-node template is the drafter itself). Prompt-dependent "
              "acceptance is included in these rates; pass times and committed "
              "tokens per pass are recorded separately.", "",
              "## Quality and correctness", "",
              "All deployments use all 164 HumanEval problems, the same first "
              "300 GSM8K test problems and 100 deterministic schema-extraction "
              "records. Evaluation concurrency equals each deployment's request-slot "
              "count. Temperature is zero; reasoning effort is low with "
              "thinking enabled and none with thinking disabled. The "
              "generation cap is 2,048 tokens (512 for extraction), and thinking "
              "uses the mode described above.", "",
              "HumanEval executes generated code "
              "in a container without network access or host-file mounts. Scores "
              "are specific to the repository's chat prompts, execution wrapper "
              "and token budgets. The HumanEval wrapper executes the returned "
              "code block independently; omitted context helpers can cause a "
              "failure, as observed for `encode_cyclic` in HumanEval/38. Raw "
              "responses and execution errors are retained.", ""]
    quality_rows = []
    for d in deployments:
        tasks = d["quality"]["tasks"]
        quality_rows.append([*deployment_cells(d), *[f"{tasks[t]['passed']}/{tasks[t]['n']}" for t in ("humaneval", "gsm8k", "extract")],
                             sum(tasks[t]["truncated"] for t in tasks)])
    lines.append(table([*DEPLOYMENT_HEADERS, "HumanEval", "GSM8K", "Schema extraction", "Responses at token cap"], quality_rows))
    checks = [c for d in deployments for c in d["rank_checks"]]
    lines.append(f"Complete matching operation streams were collected for {sum(c['passed'] for c in checks)}/{len(checks)} "
                 "recorded launches. The detailed record reports greedy "
                 "plain/MTP transcript comparisons and solo/batched text checks "
                 "for each deployment, including any failures.")
    identity_rows = []
    for d in deployments:
        comparisons = d["modes"].get("plain", {}).get("parity") or {}
        matched = sum(value["identical"] for value in comparisons.values())
        isolation = d["isolation"] or ""
        identity_rows.append([*deployment_cells(d),
            f"{sum(c['passed'] for c in d['rank_checks'])}/{len(d['rank_checks'])}",
            f"{matched}/{len(comparisons)} classes",
            "different" if "DIFFERENT" in isolation else "identical" if "IDENTICAL" in isolation else "not completed"])
    lines += ["", table([*DEPLOYMENT_HEADERS, "Rank checks passed", "Plain/default greedy match", "Solo/batched greedy text"], identity_rows),
              "A class counts as a plain/default match only when all three "
              "repetitions in both modes produce the same text and token count. "
              "A differing transcript fails the exact-text check. Current dense "
              "kernels can use different floating-point reduction orders for "
              "different batch shapes, so identical greedy text across batch "
              "sizes is not guaranteed. A text mismatch alone does not establish "
              "request-state contamination. Rank operation-stream agreement "
              "checks execution order, not numerical equality or request isolation; "
              "single-node launches have only one stream to record.", "",
              "For MiMo-V2.6-Flash every deployment's solo and batched decode "
              "transcripts matched (the isolation column above). The check does "
              "not measure logit error or prove isolation for all workloads. "
              "Performance tables use the default kernel setting."]
    if result["manifest"].get("interruptions"):
        lines += ["", "GLM-4.7 attempts were interrupted when a node became "
                  "unreachable during cold prefill. GLM-4.7 is excluded from "
                  "this benchmark matrix at the user's request. Its failed "
                  "attempts are retained separately from the reported results. "
                  "The available logs do not establish the outages' cause. "
                  f"[Interruption and recovery record]({record}interruptions.md)."]
    lines += ["", "## Long context", "",
              "Each deterministic parcel document is generated once cold and "
              "twice more with the prefix cache enabled. Decode columns are "
              "medians over those three 256-token requests; cold prefill is one "
              "sample. These requests use the model's default reasoning mode. "
              "These are timing measurements; retrieval accuracy is "
              "measured separately.", ""]
    long_rows = []
    for d in deployments:
        data = read(HERE / "raw" / d["id"] / "long/default/long.json")
        if not data:
            continue
        groups = defaultdict(list)
        for s in data["samples"]:
            groups[s["requested_tokens"]].append(s)
        for _, samples in groups.items():
            first = samples[0]
            long_rows.append([*deployment_cells(d), f"{first['answer']['usage']['prompt_tokens']:,}",
                f"{first['engine']['prefill_ms'] / 1000:.3f}",
                f"{statistics.median(s['ms_per_pass'] for s in samples):.2f}",
                f"{statistics.median(s['tokens_per_pass'] for s in samples):.2f}",
                f"{statistics.median(s['engine_tokens_per_s'] for s in samples):.1f}"])
    lines.append(table([*DEPLOYMENT_HEADERS, "Prompt tokens", "Cold prefill (s)", "Decode ms/pass", "Tokens/pass", "Engine tok/s"], long_rows))
    retrieval = read(HERE / "raw/qwen-yarn-w2/long/default/retrieval.json")
    if retrieval:
        lane = retrieval["results"]["dgpp"]
        retrieval_rows = []
        for target, slot in lane["lengths"].items():
            probes = slot["retrieval"]
            lengths = [p["prompt_tokens"] for p in probes["probes"]]
            streams = slot["concurrent"]
            prompt_length = f"{min(lengths):,}" if min(lengths) == max(lengths) else f"{min(lengths):,}–{max(lengths):,}"
            retrieval_rows.append([f"{int(target):,}", prompt_length,
                f"{probes['hits']}/{probes['requests']}",
                f"{100 * slot['cache_reuse']['cached_ratio']:.3f}%",
                f"{streams['completed']}/{streams['requests']}"])
        control = lane["short_context"]
        lines += ["YaRN retrieval uses five planted numeric codes at 5%, 25%, "
                  "50%, 75% and 95% of each document, with a 768-token response "
                  "budget including reasoning. A hit means the expected code "
                  "occurs in the response, including reasoning, after removing "
                  "non-digit characters. Prefix reuse repeats the final probe. "
                  "The two-stream test checks completion; admission may queue "
                  "a request when the shared KV pool is full.", "",
                  table(["Target tokens", "Actual prompt tokens", "Retrieval hits", "Repeated prefix cached", "Concurrent requests completed"], retrieval_rows),
                  f"The ~4K control retrieved {control['hits']}/{control['requests']} codes. "
                  f"[Full retrieval and concurrency record]({record}raw/qwen-yarn-w2/long/default/retrieval.json)."]
    lines += ["", "## Selection microbenchmarks", "",
              "The Qwen QSA and DeepSeek CSA2 benchmarks run on an idle GB10, "
              "using CUDA graphs, five warmups and thirty timed iterations per "
              "shape. QSA checks scores and selections against a host oracle; "
              "CSA2 checks selection against a host sort of the production "
              "score keys, not an independent score-arithmetic oracle. Warm "
              "inputs are repeated; cold samples evict L2 before timing. These "
              "kernel timings are separate from end-to-end service throughput.", "",
              f"[Complete current selection timings and correctness results]({record}README.md#selection-microbenchmarks).", "",
              "## Reproduce", "",
              "Use the same revision, prepared checkpoints and cluster site "
              "settings. Stop any serving workload before the campaign; only "
              "one world can own the benchmark fabric at a time.", "",
              "The following runs one deployment. Use a clean checkout of the "
              "recorded source revision for an exact code match.", "",
              "```bash", "cmake --preset release",
              "cmake --build --preset release -j 4",
              "python3 scripts/prepare_data.py download",
              "export CUDA_DEVICE_MAX_CONNECTIONS=32",
              "CONFIG=deploy/cluster_qwen-3.8-flash-next_nvfp4_w2.example.json",
              "python3 scripts/dgpp-cluster up --config \"$CONFIG\"",
              "python3 scripts/timed_load.py 127.0.0.1 18080 \\",
              "  --concurrency 1,2,4 --classes all --repeat 3 --max-tokens 256 \\",
              "  --json-out /tmp/decode.json",
              "python3 scripts/serve_prefill_probe.py 127.0.0.1 18080 2048 8192 32768 \\",
              "  --repeat 3 --no-think --seed 7 --tag current-20260922 \\",
              "  --json-out /tmp/prefill.json",
              "python3 scripts/dgpp-cluster down --config \"$CONFIG\"", "```", "",
              "Use the configured HTTP host/port if they differ. Omit `--no-think` "
              "for GLM-5.3. To repeat a cold-prefill run on the same server, change "
              "the tag so it cannot reuse a previous prefix. Sampled sweeps add "
              "`--temperature 1`. Mode sweeps use the saved configurations: "
              "plain decoding sets `mtp=false`, `mtp_depth=1` and "
              "`mtp_schedule=false`; deeper runs use the recorded depth, "
              "scheduling and slot settings.", "",
              f"The [campaign runner]({record}run.py), [isolated evaluator]({record}isolated_eval.py) "
              f"and [long-context client]({record}long_context.py) preserve the full "
              "procedure. Quality evaluation requires Docker and the Python "
              "image pinned by both registry digest and local image ID in "
              "`manifest.json`. Every invocation, including the microbenchmark shapes, "
              "is recorded in `raw/**/*.command.json`. The runner is resumable "
              "and checks this campaign's binary hash; do not reuse its completed "
              "result directory for a new measurement."]
    # This campaign adds two deployments to the page the current-code campaign
    # publishes; its overview is written beside the record and merged into
    # docs/benchmarks.md by hand (the page keeps the other deployments' rows).
    (HERE / "overview.md").write_text("\n".join(lines) + "\n")
    detailed(result)


def detailed(result):
    lines = ["# Current benchmark results", "", "Source: `" + result["manifest"]["commit"] + "`.", "",
             "All values below are calculated by `summarize.py` from this "
             "directory's raw results. The saved `*.command.json` files identify "
             "commands, start/end times and exit codes. `manifest.json` identifies "
             "the binary, hardware records, checkpoints and datasets.", ""]
    if not result["manifest"].get("completed_at"):
        lines += ["**Run in progress.** These tables contain completed measurement "
                  "groups only; remaining performance, mode, long-context and "
                  "quality runs are still pending. GLM-4.7 is excluded at the "
                  "user's request.", ""]
    side = [text for name, text in (("notes.md", "[Reference harness, HumanEval fences and the reference cross-check](notes.md)"),
                                    ("environment.md", "[Environment comparison](environment.md)"),
                                    ("interruptions.md", "[Interruption record](interruptions.md)"),
                                    ("isolation.md", "[Batching diagnostic](isolation.md)"))
            if (HERE / name).exists()]
    if side:
        lines += [" · ".join(side), ""]
    lines += [
              "Deployments follow the [overview](../../../docs/benchmarks.md) order: "
              "model family, node count, then configuration options. KV labels "
              "describe the shared key/value-cache token pool (K = 1,024 tokens); "
              "slots are the configured concurrent-request limit.", ""]
    for d in ordered_deployments(result):
        if not (d["greedy"] or d["sampled"] or d["prefill"] or d["modes"] or d["quality"]):
            continue
        lines += ["## " + d["name"], "", f"Configuration: [{d['id']}]({d['config']}).", "",
                  "### Greedy serving", "", "Rates are tokens/s. Brackets show "
                  "minimum–maximum across three repetitions; the leading value "
                  "is the median. TTFT is the median of per-phase mean request TTFTs.", ""]
        rows = []
        for s in d["greedy"]:
            def spread(key):
                v = s[key]
                return f"{v['median']:.1f} [{v['min']:.1f}–{v['max']:.1f}]"
            rows.append([s["class"], s["concurrency"], spread("engine_tokens_per_s"),
                         spread("wall_tokens_per_s"), f"{s['ms_per_pass']['median']:.2f}",
                         f"{s['tokens_per_pass_per_request']['median']:.2f}", f"{s['ttft_ms']['median']:.0f}"])
        lines.append(table(["Class", "C", "Engine tok/s", "Wall tok/s", "ms/pass", "Tokens/pass/request", "TTFT ms"], rows))
        lines += ["### Sampled serving", "", "Temperature one; medians of three repetitions. "
                  "Other sampling parameters use each model's server defaults.", "",
                  table(["Class", "C", "Engine tok/s", "Wall tok/s"],
                  [[s["class"], s["concurrency"], f"{s['engine_tokens_per_s']['median']:.1f}",
                    f"{s['wall_tokens_per_s']['median']:.1f}"] for s in d["sampled"]]),
                  "### Cold prefill", "", table(["Target tokens", "Actual prompt tokens (all samples)", "Prefill median s", "Prefill min–max s", "ms/token", "TTFT median s"],
                  [[s["target"], ", ".join(map(str, s["actual_prompt_tokens"])),
                    f"{s['prefill_s']['median']:.3f}", f"{s['prefill_s']['min']:.3f}–{s['prefill_s']['max']:.3f}",
                    f"{s['ms_per_token']['median']:.3f}", f"{s['ttft_s']['median']:.3f}"] for s in d["prefill"]]),
                  "### Decode mode checks", ""]
        rows = []
        for mode, values in d["modes"].items():
            for s in values["greedy"]:
                same = values["parity"][s["class"]]["identical"] if values["parity"] else None
                rows.append([mode, s["class"], f"{s['engine_tokens_per_s']['median']:.1f}",
                             f"{s['ms_per_pass']['median']:.2f}", f"{s['tokens_per_pass_per_request']['median']:.2f}",
                             "yes" if same else "no"])
        lines += [table(["Mode", "Class", "Engine tok/s", "ms/pass", "Tokens/pass", "All greedy repetitions match default"], rows),
                  "Solo/batched exact-text check: " + (d["isolation"] or "No completed check."), ""]
        if d["quality"]:
            lines += ["### Quality", "", "One greedy response per problem, using the repository's "
                      f"chat prompts and concurrency {d['slots']}. Token caps include reasoning. HumanEval runs generated Python "
                      "in the pinned container recorded in `manifest.json`.", "",
                      table(["Task", "Passed / evaluated", "At token cap", "Mean completion tokens"],
                            [[task, f"{score['passed']}/{score['n']}", score["truncated"],
                              score["mean_completion_tokens"]]
                             for task, score in d["quality"]["tasks"].items()])]
        lines += [f"Raw results: [raw/{d['id']}/](raw/{d['id']}/).", ""]
    if not (HERE / "raw/micro").exists():
        # A family campaign without the selection microbenchmarks.
        (HERE / "README.md").write_text("\n".join(lines).rstrip() + "\n")
        return
    lines += ["## Selection microbenchmarks", "",
              "Medians in microseconds. Cold denotes L2 eviction outside the "
              "timed interval. All commands use five warmups and thirty iterations.", ""]
    rows = []
    for path in sorted((HERE / "raw/micro").glob("qsa-*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.startswith("{"):
                continue
            s = json.loads(line)
            assert s["mismatches"] == 0, path
            rows.append([s["rows"], s["pools"], s["cache"], s["stage"], s["median_us"], s["min_us"], s["p95_us"]])
    lines += ["### Qwen QSA", "", table(["Rows", "Pools", "Cache", "Stage", "Median µs", "Min µs", "p95 µs"], rows)]
    rows = []
    for path in sorted((HERE / "raw/micro").glob("csa2-*.jsonl")):
        assert "PASS selections" in path.read_text(), path
        for line in path.read_text().splitlines():
            if not line.startswith("{"):
                continue
            s = json.loads(line)
            rows.append([s["rows"], s["ctx"], s["cache"], s["stage"], s["median_us"], s["min_us"], s["max_us"]])
    lines += ["### DeepSeek CSA2", "", "`combined` times one candidate and one "
              "restricted call; the model has five restricted calls per full "
              "decoder pass. These component timings must not be added to predict "
              "service throughput.", "",
              table(["Rows", "Context", "Cache", "Stage", "Median µs", "Min µs", "Max µs"], rows)]
    (HERE / "README.md").write_text("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true", help="save summary.json without publishing Markdown")
    args = parser.parse_args()
    result = collect()
    (HERE / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.partial:
        return
    # A complete document is generated only from full measurement groups.
    for item in result["deployments"]:
        base = HERE / "raw" / item["id"]
        performance = read(base / "performance/complete.json")
        assert performance and all(performance["checks"][name] for name in
                                   ("greedy", "prefill", "sampled")), item["id"]
        concurrencies = {1, 2, min(item["slots"], 4), item["slots"]}
        assert {(s["class"], s["concurrency"]) for s in item["greedy"]} == {
            (name, c) for name in CLASSES for c in concurrencies}, item["id"]
        assert all(s["engine_tokens_per_s"]["n"] == 3 and s["all_http_200"] for s in item["greedy"]), item["id"]
        assert len(item["sampled"]) == 10 and all(s["engine_tokens_per_s"]["n"] == 3 and s["all_http_200"]
                                                 for s in item["sampled"]), item["id"]
        assert len(item["prefill"]) == 3 and all(s["samples"] == 3 and s["all_cold"] for s in item["prefill"]), item["id"]
        assert "plain" in item["modes"], item["id"]
        if item["id"] == "qwen27b-fp8-w1":
            assert "mtp2" in item["modes"], item["id"]  # the MTP depth-2 world as the drafter template's mode
        elif item["id"] in ("qwen27b-fp8-w2", "qwen27b-fp8-w4"):
            assert "depth2" in item["modes"] and "dflash2" in item["modes"], item["id"]
        elif item["id"] == "qwen27b-fp8-w4":
            assert "depth2" in item["modes"], item["id"]
        elif item["id"].startswith("qwen") or item["id"] in ("glm47-w4", "glm53-w4"):
            assert "depth2" in item["modes"], item["id"]
        if item["id"] == "deepseek-w4" or item["id"].startswith("dsv4"):
            assert "depth5" in item["modes"], item["id"]
        for name, mode in item["modes"].items():
            marker = read(base / "modes" / name / "complete.json")
            assert marker and marker["passed"], (item["id"], name)
            assert len(mode["greedy"]) == 5 and all(s["engine_tokens_per_s"]["n"] == 3 and s["all_http_200"]
                                                     for s in mode["greedy"]), item["id"]
        quality = read(base / "quality/complete.json")
        assert quality and quality["passed"], item["id"]
        assert item["quality"], item["id"]
        assert set(item["quality"]["tasks"]) == {"humaneval", "gsm8k", "extract"}, item["id"]
        assert item["quality"]["model"] == item["model"], item["id"]
        settings = item["quality"]["settings"]
        no_think = "GLM-5.3" not in item["model"]
        assert settings == {"max_tokens": 2048, "reasoning_effort": "none" if no_think else "low",
                            "no_think": no_think, "limit": 300, "seed": 20260908}, item["id"]
        for task, count in (("humaneval", 164), ("gsm8k", 300), ("extract", 100)):
            score = item["quality"]["tasks"][task]
            answers = [json.loads(line) for line in
                       (HERE / "raw" / item["id"] / "quality/default/eval" / f"{task}.jsonl").read_text().splitlines()]
            assert len(answers) == len({answer["id"] for answer in answers}) == score["n"] == count, (item["id"], task)
            assert sum(answer["ok"] for answer in answers) == score["passed"], (item["id"], task)
            assert sum(answer["finish"] == "length" for answer in answers) == score["truncated"], (item["id"], task)
        assert item["isolation"], item["id"]
    campaign_ids = {item["id"] for item in result["deployments"]}
    for model, lengths in (("glm-flash-hybrid-w4", {2048, 8192, 32768}),
                           ("glm-flash-fp8-w4", {2048, 8192, 32768}),
                           ("deepseek-w4", {4096, 32768, 122000}),
                           ("qwen-yarn-w2", {4096, 32768, 131072, 262144, 524288}),
                           ("mimo-w4", {4096, 32768, 122000}),
                           ("mimo-w2-fp8kv", {4096, 32768, 131072, 240000}),
                           ("dsv4-w4", {4096, 32768, 122000}),
                           ("dsv4-w2", {4096, 32768, 122000})):
        if model not in campaign_ids:
            continue
        marker = read(HERE / "raw" / model / "long/complete.json")
        assert marker and all(marker["checks"].values()), model
        data = read(HERE / "raw" / model / "long/default/long.json")
        assert data and {(s["requested_tokens"], s["repeat"]) for s in data["samples"]} == {
            (length, repeat) for length in lengths for repeat in range(3)}, model
    retrieval = read(HERE / "raw/qwen-yarn-w2/long/default/retrieval.json")
    assert "qwen-yarn-w2" not in campaign_ids or (retrieval and retrieval.get("status") == "completed"), \
        "YaRN retrieval did not finish"
    lane = retrieval["results"]["dgpp"] if retrieval else {"lengths": {}}
    assert "qwen-yarn-w2" not in campaign_ids or set(lane["lengths"]) == {"262144", "524288"}, \
        "YaRN retrieval lengths are incomplete"
    for target, sample in lane["lengths"].items():
        probes = sample["retrieval"]
        assert probes["requests"] == len(probes["probes"]) == 5, target
        assert probes["hits"] == sum(p["hit"] for p in probes["probes"]), target
        assert sample["concurrent"]["requests"] == 2, target
        assert sample["cache_reuse"]["prompt_sha256"] == probes["probes"][-1]["prompt_sha256"], target
    if retrieval:
        control = lane["short_context"]
        assert control["requests"] == len(control["probes"]) == 5, "YaRN short control is incomplete"
        assert control["hits"] == sum(p["hit"] for p in control["probes"]), "YaRN short control score differs"
    from summarize_telemetry import main as summarize_telemetry
    summarize_telemetry()
    render(result)


if __name__ == "__main__":
    main()
