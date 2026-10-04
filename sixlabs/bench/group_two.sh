#!/bin/bash
# group_two.sh — engine.prefill_group off against on, on the serving engine's own box (DGXtwo),
# where nothing else uses the GPU. Run on DGXone:
#   setsid bash sixlabs/bench/group_two.sh > ~/dgpp-refset/group_two.log 2>&1 </dev/null &
# It restarts the DGXtwo engine once per setting (six_on_dgxtwo.sh) and LEAVES it running on the
# last one in MODES. Refuses to start while requests are in flight there. Per setting:
# group_check.py and mixed_load.py, with DGXtwo's other GPU users recorded. EXTRA: more engine
# settings for every restart (KEY=VALUE ...); TAG: a suffix for the labels of this run.
set -u
R=/home/mark/dgpp-refset
O=$R/group
D=/home/mark/dgpp-next
BASE=http://10.77.0.2:18090
TWO=marktwo@10.77.0.2
SSH="ssh -o BatchMode=yes -o ConnectTimeout=6"
MODES="${MODES:-off on}"
N=$(echo $MODES | wc -w)
T0=$(date +%s); ERR=0
say() { echo "[1/1] $1/$N err=$ERR elapsed=$(( $(date +%s) - T0 ))s :: $(date +%H:%M:%S) $2"; }
cd $D || exit 2
mkdir -p $O
busy=$(curl -s -m 5 $BASE/metrics/prometheus | awk '/^dgpp_num_requests_(running|waiting|prefilling)/ {s += $2} END {print s + 0}')
[ "$busy" = "0" ] || { echo "refusing: $busy requests in flight on $BASE"; exit 3; }
say 0 "starting"
i=0
for mode in $MODES; do
  i=$((i + 1))
  if ! bash sixlabs/bench/six_on_dgxtwo.sh prefill_group=$mode ${EXTRA:-} > $O/up_two_$mode.log 2>&1; then
    ERR=$((ERR + 1)); say $i "$mode: engine did not come up: $(tail -3 $O/up_two_$mode.log | tr '\n' ' ' | cut -c1-260)"; continue
  fi
  echo "   $(grep 'budget' $O/up_two_$mode.log | tail -1 | cut -c1-260)"
  ( $SSH $TWO "nvidia-smi pmon -c 240 -s u 2>/dev/null" | awk '$4+0 > 5 && $NF !~ /6ix-Serve/ {print $NF}' | sort | uniq -c > $O/pmon_two_$mode.txt ) &
  python3 sixlabs/bench/group_check.py --base $BASE --model Qwen3-Next-80B --label two_$mode${TAG:-} --out $O/group_two.jsonl 2>&1 | tee $O/check_two_$mode.txt
  grep -q "^RESULT: ok" $O/check_two_$mode.txt || ERR=$((ERR + 1))
  python3 sixlabs/bench/mixed_load.py --base $BASE --model Qwen3-Next-80B --label two_group_$mode${TAG:-} --writers 4 --readers 2 \
      --lines 530 --seconds 50 --out $O/mixed_two.jsonl > $O/mixed_two_$mode.log 2>&1 || ERR=$((ERR + 1))
  wait
  say $i "$mode done; engine failed flag: $(curl -s -m 5 $BASE/metrics/prometheus | awk '/^dgpp_engine_failed/ {print $2}'); other GPU users above 5 % on DGXtwo: $(tr '\n' ' ' < $O/pmon_two_$mode.txt)"
done
python3 - <<'PY'
import json
print("--- burst of eight and the busy case, per setting")
for l in open("/home/mark/dgpp-refset/group/group_two.jsonl"):
    r = json.loads(l)
    print("%-8s burst: first tokens %s (mean %s s), all done in %s s, %s/8 right | beside an answer: first tokens %s | checks failed: %s"
          % (r["label"], r["burst_ttft_s"], r["burst_ttft_mean_s"], r["burst_wall_s"], r["burst_right"], r["busy_ttft_s"], r["fails"]))
print("--- 4 answers beside 2 fresh 17k prompts")
for l in open("/home/mark/dgpp-refset/group/mixed_two.jsonl"):
    r = json.loads(l)
    print("%-15s %-14s answers %s tok/s each, pause p95 %s / max %s ms | prompts %s tok/s, first token %s s"
          % (r["label"], r["phase"], r["per_writer_tok_s"], r["gap_ms_p95"], r["gap_ms_max"], r["reader_prompt_tok_s"], r["reader_ttft_s_p50"]))
PY
echo "ALLDONE err=$ERR (the DGXtwo engine is left running on prefill_group=$mode)"
