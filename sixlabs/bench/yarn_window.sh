#!/bin/bash
# yarn_window.sh — the whole GPU check of YaRN x2 (512k per request) for Qwen3-Next-80B, in one
# engine window. Run on DGXone:
#   setsid bash sixlabs/bench/yarn_window.sh > ~/dgpp-refset/yarn_window.log 2>&1 </dev/null &
# It takes the TEST deployment down, so it refuses to start while requests are in flight
# (FORCE=1 overrides). The standard config is put back and restarted at the end, also on failure.
#   1  engine down
#   2  forward check with and without the ramp, against the numpy reference (yarn_compare.py)
#   3  engine up with rope_scaling factor 2, pool for one 512k request
#   4  planted-fact retrieval at 300k / 400k / 500k prompt tokens (long_context_check.py)
#   5  short prompts with the ramp on
#   6  standard config back up, the same short prompts
set -u
R=/home/mark/dgpp-refset
O=$R/yarn
D=/home/mark/dgpp-next
CFG=deploy/cluster_qwen3-next-80b_nvfp4_w1.json
BASE=http://172.17.0.1:18090
MODEL=nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4
TEXTS="c_fib echo_chat_raw echo_prose_en echo_code_py"
T0=$(date +%s)
ERR=0
say() { echo "[$1/6] $2 err=$ERR elapsed=$(( $(date +%s) - T0 ))s :: $3"; }
cd $D || exit 2
mkdir -p $O

busy=$(curl -s -m 5 $BASE/metrics/prometheus | awk '/^dgpp_num_requests_(running|waiting|prefilling)/ {s += $2} END {print s + 0}')
if [ "$busy" != "0" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "refusing: $busy requests in flight on $BASE (another job is using the engine; FORCE=1 overrides)"; exit 3
fi

cp $CFG $O/standard_config.json
restore() {
  cp $O/standard_config.json $D/$CFG
  bash $R/restart_test.sh > $O/restore.log 2>&1 || { ERR=$((ERR + 1)); cat $O/restore.log; }
}

say 1 0/1 "engine down"
python3 scripts/dgpp-cluster down --config $CFG > /dev/null 2>&1
sleep 3
say 1 1/1 "engine down, $(free -g | awk 'NR==2 {print $7}') GiB available"

n=0
for t in $TEXTS; do
  build-release/qwen35_forward_check --model $MODEL --ids-json $R/ids/$t.json --rope-scaling 2 > $O/eng_yarn2_$t.txt 2> $O/eng_yarn2_$t.err || ERR=$((ERR + 1))
  n=$((n + 1)); say 2 $n/5 "forward check with the ramp: $t"
done
build-release/qwen35_forward_check --model $MODEL --ids-json $R/ids/echo_prose_en.json > $O/eng_plain_echo_prose_en.txt 2> $O/eng_plain_echo_prose_en.err || ERR=$((ERR + 1))
say 2 5/5 "forward check without the ramp: echo_prose_en"
python3 sixlabs/bench/yarn_compare.py $O | tee $O/compare.txt
grep -q "^RESULT: ok" $O/compare.txt || ERR=$((ERR + 1))

say 3 0/1 "engine up with YaRN x2"
if bash $R/restart_test.sh 'rope_scaling={"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144}' \
     max_concurrency=4 kv_capacity=565248 > $O/up_yarn.log 2>&1; then
  say 3 1/1 "$(tr '\n' ' ' < $O/up_yarn.log | cut -c1-200)"
  L=$(ls -t $D/log/deployments/*/serve_r0.log | head -1)
  grep -i "rope\|yarn\|context" "$L" | tail -4 | cut -c1-220

  say 4 0/3 "retrieval at 300k / 400k / 500k"
  python3 sixlabs/bench/long_context_check.py --base $BASE --model Qwen3-Next-80B \
      --tokens 300000,400000,500000 --out $O/long_yarn.jsonl || ERR=$((ERR + 1))

  say 5 0/1 "short prompts with the ramp on"
  python3 sixlabs/bench/short_prompts.py --base $BASE --model Qwen3-Next-80B --out $O/short_yarn.jsonl || ERR=$((ERR + 1))
  say 5 1/1 "short prompts with the ramp on"
else
  ERR=$((ERR + 1)); say 3 1/1 "YaRN engine did not come up: $(tail -3 $O/up_yarn.log | tr '\n' ' ' | cut -c1-300)"
fi

say 6 0/2 "standard config back"
restore
say 6 1/2 "standard engine: $(tr '\n' ' ' < $O/restore.log | cut -c1-160)"
python3 sixlabs/bench/short_prompts.py --base $BASE --model Qwen3-Next-80B --out $O/short_plain.jsonl || ERR=$((ERR + 1))
say 6 2/2 "short prompts without the ramp"
echo "ALLDONE err=$ERR"
