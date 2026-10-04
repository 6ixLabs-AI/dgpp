#!/bin/bash
# score_window.sh — the GPU check of POST /v1/score, in one short engine window on DGXone:
#   setsid bash sixlabs/bench/score_window.sh > ~/dgpp-refset/score_window.log 2>&1 </dev/null &
#   1  forward check of one text with the serving dense format (the expected log-probabilities)
#   2  a small test engine up (64k pool, 2 slots: about 54 GiB), DGXone's test address
#   3  score_check.py against it
#   4  engine down, the config put back as it was
# It refuses to start when the box does not have the memory for the engine (NEED_GIB, default
# 60): earlyoom prefers python, and its first victims here would be other people's jobs.
set -u
R=/home/mark/dgpp-refset
O=$R/score
D=/home/mark/dgpp-next
CFG=deploy/cluster_qwen3-next-80b_nvfp4_w1.json
BASE=http://172.17.0.1:18090
MODEL=nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4
T0=$(date +%s)
ERR=0
say() { echo "[$1/4] $2 err=$ERR elapsed=$(( $(date +%s) - T0 ))s :: $3"; }
cd $D || exit 2
mkdir -p $O
avail=$(free -g | awk 'NR==2 {print $7}')
if [ "$avail" -lt "${NEED_GIB:-60}" ]; then echo "refusing: $avail GiB available, need ${NEED_GIB:-60}"; exit 3; fi
if pgrep -x 6ix-Serve > /dev/null; then echo "refusing: an engine is already running on this box"; exit 3; fi

say 1 0/1 "forward check (fp8 dense), $avail GiB available"
if ! grep -q '^token logprobs:' $O/eng_fp8_echo_prose_en.txt 2>/dev/null; then   # kept from an earlier window
  build-release/qwen35_forward_check --model $MODEL --ids-json $R/ids/echo_prose_en.json --dense-weights fp8 \
      > $O/eng_fp8_echo_prose_en.txt 2> $O/eng_fp8_echo_prose_en.err || ERR=$((ERR + 1))
fi
say 1 1/1 "forward check: $(grep -c '^token logprobs:' $O/eng_fp8_echo_prose_en.txt) logprob line(s)"

cp $CFG $O/standard_config.json
say 2 0/1 "test engine up"
if bash $R/restart_test.sh kv_capacity=65536 max_concurrency=2 > $O/up.log 2>&1; then
  say 2 1/1 "$(tr '\n' ' ' < $O/up.log | cut -c1-200)"
  say 3 0/1 "score_check.py"
  python3 sixlabs/bench/score_check.py --base $BASE --model Qwen3-Next-80B --ids-json $R/ids/echo_prose_en.json \
      --filler-json $R/ids/echo_code_py.json --expected $O/eng_fp8_echo_prose_en.txt 2>&1 | tee $O/check.txt
  grep -q "^RESULT: ok" $O/check.txt || ERR=$((ERR + 1))
  say 3 1/1 "score_check.py"
else
  ERR=$((ERR + 1)); say 2 1/1 "the test engine did not come up: $(tail -3 $O/up.log | tr '\n' ' ' | cut -c1-300)"
fi

say 4 0/1 "engine down, config back"
python3 scripts/dgpp-cluster down --config $CFG > /dev/null 2>&1
cp $O/standard_config.json $D/$CFG
sleep 3
say 4 1/1 "engine $(pgrep -x 6ix-Serve > /dev/null && echo STILL RUNNING || echo stopped), $(free -g | awk 'NR==2 {print $7}') GiB available"
echo "ALLDONE err=$ERR"
