#!/bin/bash
# group_window.sh — the GPU check of reading several prompts in one walk (engine.prefill_group),
# on DGXone's test engine: the same checks with it off and on.
#   setsid bash sixlabs/bench/group_window.sh > ~/dgpp-refset/group_window.log 2>&1 </dev/null &
# Per setting: group_check.py (right answers one at a time, together, mixed lengths, beside a
# streaming answer; a burst of eight) and mixed_load.py (4 answers beside 2 fresh 17k prompts).
# Each run records HEM's scoring calls and other GPU users, as the sweep does. The engine is
# stopped and the config put back at the end.
set -u
R=/home/mark/dgpp-refset
O=$R/group
D=/home/mark/dgpp-next
CFG=deploy/cluster_qwen3-next-80b_nvfp4_w1.json
BASE=http://172.17.0.1:18090
T0=$(date +%s); ERR=0
MODES="${MODES:-off on}"
N=$(echo $MODES | wc -w)
say() { echo "[1/1] $1/$N err=$ERR elapsed=$(( $(date +%s) - T0 ))s :: $(date +%H:%M:%S) $2"; }
cd $D || exit 2
mkdir -p $O
[ "$(free -g | awk 'NR==2 {print $7}')" -ge 75 ] || { echo "refusing: under 75 GiB available"; exit 3; }
pgrep -x 6ix-Serve > /dev/null && { echo "refusing: an engine is already running on this box"; exit 3; }
cp $CFG $O/standard_config.json
say 0 "starting"
i=0
for mode in $MODES; do
  i=$((i + 1))
  if ! bash $R/restart_test.sh prefill_group=$mode > $O/up_$mode.log 2>&1; then
    ERR=$((ERR + 1)); say $i "$mode: engine did not come up: $(tail -2 $O/up_$mode.log | tr '\n' ' ' | cut -c1-240)"; continue
  fi
  L=$(ls -t $D/log/deployments/*/serve_r0.log | head -1)
  echo "   $(grep 'prefill budget' "$L" | tail -1 | cut -c40-300)"
  t_run=$(date -u +%Y-%m-%dT%H:%M:%SZ)   # with the Z: docker reads a bare timestamp as local time and counts nothing
  ( nvidia-smi pmon -c 240 -s u 2>/dev/null | awk '$4+0 > 5 && $NF !~ /6ix-Serve/ {print $NF}' | sort | uniq -c > $O/pmon_$mode.txt ) &
  python3 sixlabs/bench/group_check.py --base $BASE --model Qwen3-Next-80B --label $mode --out $O/group.jsonl 2>&1 | tee $O/check_$mode.txt
  grep -q "^RESULT: ok" $O/check_$mode.txt || ERR=$((ERR + 1))
  python3 sixlabs/bench/mixed_load.py --base $BASE --model Qwen3-Next-80B --label group_$mode --writers 4 --readers 2 \
      --lines 530 --seconds 50 --out $O/mixed.jsonl > $O/mixed_$mode.log 2>&1 || ERR=$((ERR + 1))
  wait
  echo "   engine errors in its log: $(grep -c ' ERROR ' "$L"); failed: $(curl -s -m 5 $BASE/metrics/prometheus | awk '/^dgpp_engine_failed/ {print $2}')"
  say $i "$mode done; HEM scoring calls during it: $(docker logs --since $t_run HEMnliOne 2>&1 | grep -c POST); other GPU users above 5 %: $(tr '\n' ' ' < $O/pmon_$mode.txt)"
done
python3 scripts/dgpp-cluster down --config $CFG > /dev/null 2>&1
cp $O/standard_config.json $D/$CFG
sleep 3
say $N "engine $(pgrep -x 6ix-Serve > /dev/null && echo STILL RUNNING || echo stopped), config back, $(free -g | awk 'NR==2 {print $7}') GiB available"
python3 - <<'PY'
import json
print("--- burst of eight and the busy case, per setting")
for l in open("/home/mark/dgpp-refset/group/group.jsonl"):
    r = json.loads(l)
    print("%-5s burst: first tokens %s (mean %s s), all done in %s s, %s/8 right | beside an answer: first tokens %s | checks failed: %s"
          % (r["label"], r["burst_ttft_s"], r["burst_ttft_mean_s"], r["burst_wall_s"], r["burst_right"], r["busy_ttft_s"], r["fails"]))
print("--- 4 answers beside 2 fresh 17k prompts")
for l in open("/home/mark/dgpp-refset/group/mixed.jsonl"):
    r = json.loads(l)
    print("%-11s %-14s answers %s tok/s each, pause p95 %s / max %s ms | prompts %s tok/s, first token %s s"
          % (r["label"], r["phase"], r["per_writer_tok_s"], r["gap_ms_p95"], r["gap_ms_max"], r["reader_prompt_tok_s"], r["reader_ttft_s_p50"]))
PY
echo "ALLDONE err=$ERR"
