#!/bin/bash
# prefill_sweep.sh — which scheduler settings keep answers moving beside long prompts, and what
# do they cost the prompts? One 80B test engine on DGXone, restarted per setting; mixed_load.py
# (4 writers streaming, 2 readers sending fresh 17k-token prompts) on each.
#   setsid bash sixlabs/bench/prefill_sweep.sh > ~/dgpp-refset/prefill_sweep.log 2>&1 </dev/null &
# DGXone's GPU is shared with the core services, so every run records how many scoring calls
# HEM took and which other processes used the GPU while it ran: a run with any is not clean.
# The engine is stopped and the config put back at the end.
set -u
R=/home/mark/dgpp-refset
O=$R/prefill_sweep
D=/home/mark/dgpp-next
CFG=deploy/cluster_qwen3-next-80b_nvfp4_w1.json
BASE=http://172.17.0.1:18090
T0=$(date +%s); ERR=0
# label                budget idle passes order
CONFIGS="old_4096_fair:4096:0:1:fair
cur_256_p4:256:4096:4:shortest
b512_p4:512:4096:4:shortest
b1024_p4:1024:4096:4:shortest
b256_p2:256:4096:2:shortest
b1024_p8:1024:4096:8:shortest
b2048_p8:2048:4096:8:shortest"
N=$(echo "$CONFIGS" | wc -l)
say() { echo "[1/1] $1/$N err=$ERR elapsed=$(( $(date +%s) - T0 ))s :: $(date +%H:%M:%S) $2"; }
cd $D || exit 2
mkdir -p $O
[ "$(free -g | awk 'NR==2 {print $7}')" -ge 75 ] || { echo "refusing: under 75 GiB available"; exit 3; }
pgrep -x 6ix-Serve > /dev/null && { echo "refusing: an engine is already running on this box"; exit 3; }
cp $CFG $O/standard_config.json
say 0 "sweep starting"
i=0
for c in $CONFIGS; do
  IFS=: read -r label budget idle passes order <<< "$c"
  i=$((i + 1))
  if ! bash $R/restart_test.sh prefill_budget_tokens=$budget prefill_idle_budget_tokens=$idle \
        decode_passes_per_prefill=$passes prefill_order=$order > $O/up_$label.log 2>&1; then
    ERR=$((ERR + 1)); say $i "$label: engine did not come up: $(tail -2 $O/up_$label.log | tr '\n' ' ' | cut -c1-200)"; continue
  fi
  t_run=$(date -u +%Y-%m-%dT%H:%M:%SZ)   # with the Z: docker reads a bare timestamp as local time and counts nothing
  ( nvidia-smi pmon -c 150 -s u 2>/dev/null | awk '$4+0 > 5 && $NF !~ /6ix-Serve/ {print $NF}' | sort | uniq -c > $O/pmon_$label.txt ) &
  python3 sixlabs/bench/mixed_load.py --base $BASE --model Qwen3-Next-80B --label $label --writers 4 --readers 2 \
      --lines 530 --seconds 50 --out $O/mixed.jsonl > $O/run_$label.log 2>&1 || ERR=$((ERR + 1))
  wait
  hem=$(docker logs --since $t_run HEMnliOne 2>&1 | grep -c POST)
  others=$(tr '\n' ' ' < $O/pmon_$label.txt)
  say $i "$label done; HEM scoring calls during it: $hem; other GPU users above 5 %: ${others:-none}"
done
python3 scripts/dgpp-cluster down --config $CFG > /dev/null 2>&1
cp $O/standard_config.json $D/$CFG
sleep 3
say $N "engine $(pgrep -x 6ix-Serve > /dev/null && echo STILL RUNNING || echo stopped), config back, $(free -g | awk 'NR==2 {print $7}') GiB available"
python3 - <<'PY'
import json
rows = [json.loads(l) for l in open("/home/mark/dgpp-refset/prefill_sweep/mixed.jsonl")]
by = {}
for r in rows:
    by.setdefault(r["label"], {})[r["phase"]] = r
print("%-16s | %-22s | %-34s | %-40s" % ("setting", "4 writers alone", "2 readers alone (17k prompts)", "both together"))
print("%-16s | %-22s | %-34s | %-40s" % ("", "tok/s each, gap p95 ms", "prompt tok/s, first token s", "writer tok/s each, gap p95/max ms; reader prompt tok/s, first token s"))
for label, ph in by.items():
    w, r, m = ph.get("writers_alone", {}), ph.get("readers_alone", {}), ph.get("mixed", {})
    print("%-16s | %6s  %6s        | %8s  %8s                | %6s  %6s/%-6s  %8s  %8s" % (
        label, w.get("per_writer_tok_s"), w.get("gap_ms_p95"), r.get("reader_prompt_tok_s"), r.get("reader_ttft_s_p50"),
        m.get("per_writer_tok_s"), m.get("gap_ms_p95"), m.get("gap_ms_max"), m.get("reader_prompt_tok_s"), m.get("reader_ttft_s_p50")))
PY
echo "ALLDONE err=$ERR"
