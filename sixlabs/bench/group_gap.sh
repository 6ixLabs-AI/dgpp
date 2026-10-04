#!/bin/bash
# group_gap.sh — the one outstanding measurement (shared walk with the busy budget as the walk's
# total), taken in the gap between two of the RAG session's benchmarks, with no session awake.
# Run detached on DGXone:
#   setsid bash ~/dgpp-next/sixlabs/bench/group_gap.sh > ~/dgpp-refset/group_gap.log 2>&1 </dev/null &
#   1  wait for "[done]" in the running benchmark's log (never act on the process merely gone)
#   2  restart the DGXtwo engine with prefill_group on, prefill_budget_per_reader false; run
#      group_check.py and mixed_load.py (group_two.sh)
#   3  keep that setting if every check passed and answers beside two prompts ran at 17 tokens/s
#      each or better; otherwise restart on the setting the engine had (per reader)
#   4  write READY (a file) naming what the engine is left on: the next benchmark waits for it
set -u
GRID_LOG=${GRID_LOG:-/home/mark/rag-grid-pchunks/run.log}
READY=/home/mark/dgpp-refset/engine_ready
D=/home/mark/dgpp-next
O=/home/mark/dgpp-refset/group
BASE=http://10.77.0.2:18090
T0=$(date +%s)
say() { echo "[1/1] $1/3 err=${ERR:-0} elapsed=$(( $(date +%s) - T0 ))s :: $(date +%H:%M:%S) $2"; }
ERR=0
rm -f $READY
say 0 "waiting for [done] in $GRID_LOG"
gone=0
until grep -q '^\[done\]' $GRID_LOG; do
  if pgrep -f "rag/grid.py" > /dev/null; then gone=0; else gone=$((gone + 1)); fi
  if [ $gone -ge 12 ]; then say 0 "the benchmark's process is gone with no [done] line: engine left as it is"; echo "engine untouched: benchmark ended without [done]" > $READY; exit 2; fi
  sleep 15
done
for i in $(seq 1 24); do
  busy=$(curl -s -m 5 $BASE/metrics/prometheus | awk '/^dgpp_num_requests_(running|waiting|prefilling)/ {s += $2} END {print s + 0}')
  [ "$busy" = "0" ] && break
  sleep 5
done
say 1 "benchmark finished; requests in flight now: $busy"
cd $D
MODES=on EXTRA="prefill_budget_per_reader=false" TAG=_total bash sixlabs/bench/group_two.sh > /home/mark/dgpp-refset/group_two_total.log 2>&1 || ERR=$((ERR + 1))
tail -12 /home/mark/dgpp-refset/group_two_total.log | cut -c1-300
answers=$(python3 - <<'PY'
import json
best = 0.0
for l in open("/home/mark/dgpp-refset/group/mixed_two.jsonl"):
    r = json.loads(l)
    if r["label"] == "two_group_on_total" and r["phase"] == "mixed":
        best = r["per_writer_tok_s"]
print(best)
PY
)
ok=$(grep -c "^RESULT: ok" $O/check_two_on.txt 2>/dev/null)
say 2 "measured: checks ok=$ok, answers beside two prompts $answers tok/s each"
if [ "$ERR" = "0" ] && [ "$ok" = "1" ] && python3 -c "import sys; sys.exit(0 if float('$answers') >= 17 else 1)"; then
  final="prefill_group on, prefill_budget_per_reader false (answers $answers tok/s each beside two prompts)"
else
  bash sixlabs/bench/six_on_dgxtwo.sh prefill_group=on prefill_budget_per_reader=true > /home/mark/dgpp-refset/group_gap_revert.log 2>&1 || ERR=$((ERR + 1))
  final="prefill_group on, prefill_budget_per_reader true (the earlier setting; the total-budget run gave $answers tok/s, checks ok=$ok)"
fi
alive=$(curl -s -m 10 $BASE/v1/models | grep -c Qwen3-Next-80B)
say 3 "engine left on: $final; answering: $alive"
echo "$(date +%H:%M:%S) $final; answering=$alive; err=$ERR" > $READY
echo "ALLDONE err=$ERR"
