#!/bin/bash
# after_papers.sh — what follows the RAG papers benchmark on the DGXtwo engine, with no session
# awake (2026-10-04). Run detached on DGXone:
#   setsid bash ~/dgpp-next/sixlabs/bench/after_papers.sh > ~/dgpp-refset/after_papers.log 2>&1 </dev/null &
#
#   1  wait for the benchmark's "[done]" line (never act on the process merely being gone)
#   2  put the /v1/score build on DGXtwo and restart the engine there; if it does not come up,
#      put the previous binary back and restart that, so the box is never left without an engine
#   3  wait for the RAG session's evidence job (it starts by itself when /v1/score answers and
#      wants the engine to itself)
#   4  the clean benchmarks, one at a time, nothing else on that GPU: decode at 1/2/4/8 streams,
#      answers beside long prompts (mixed_load), the 30-minute soak, agentic streams, judge sweep
# No engine setting is changed and nothing restarts after step 2: the prefill sweep, which needs
# restarts, is left for an attended session.
set -u
GRID_LOG=/home/mark/rag-grid-papers/run.log
BIN=/home/mark/dgpp-refset/release/6ix-Serve-5c20f70
TWO=marktwo@10.77.0.2
SSH="ssh -o BatchMode=yes -o ConnectTimeout=6"
H=10.77.0.2; P=18090; M=Qwen3-Next-80B; BASE=http://$H:$P
D=/home/mark/dgpp-next
O=/home/mark/dgpp-refset/dgxtwo_benches_20261004
mkdir -p $O
T0=$(date +%s); ERR=0
say() { echo "[$1/4] $2 err=$ERR elapsed=$(( $(date +%s) - T0 ))s :: $(date +%H:%M:%S) $3"; }
metric() { curl -s -m 5 $BASE/metrics/prometheus | awk -v k="$1" '$1 ~ "^"k {s += $2} END {print s + 0}'; }
gpu_others() { $SSH $TWO "nvidia-smi --query-compute-apps=name --format=csv,noheader | grep -vc 6ix-Serve"; }

say 1 0/1 "waiting for [done] in $GRID_LOG"
gone=0
until grep -q '^\[done\]' $GRID_LOG; do
  if pgrep -f "rag/grid.py --n=100 --workers=2 --papers=1" > /dev/null; then gone=0; else gone=$((gone + 1)); fi
  if [ $gone -ge 6 ]; then say 1 1/1 "the benchmark's process is gone and there is no [done] line: engine left as it is"; echo "ALLDONE err=1"; exit 2; fi
  sleep 30
done
say 1 1/1 "the papers benchmark is finished"

say 2 0/1 "the /v1/score build to DGXtwo, engine restart"
[ "$(md5sum $BIN | cut -c1-32)" = "582c1f53e05f63e4a81886ca77612cfb" ] || { say 2 1/1 "the saved binary is not the tested build; nothing changed"; echo "ALLDONE err=1"; exit 3; }
scp -q -o BatchMode=yes $BIN $TWO:dgpp-next/build-release/6ix-Serve.new || { say 2 1/1 "copy failed; nothing changed"; echo "ALLDONE err=1"; exit 3; }
rsync -a -e "ssh -o BatchMode=yes" $D/scripts/ $TWO:dgpp-next/scripts/
$SSH $TWO "cd ~/dgpp-next/build-release && cp -p 6ix-Serve 6ix-Serve.prev && mv 6ix-Serve.new 6ix-Serve && chmod +x 6ix-Serve"
if bash $D/sixlabs/bench/six_on_dgxtwo.sh > $O/restart.log 2>&1 && \
   curl -s -m 60 -X POST $BASE/v1/score -H "Content-Type: application/json" \
        -d '{"model":"Qwen3-Next-80B","prompt":"The capital of France is","continuation":" Paris"}' | tee $O/score_probe.json | grep -q sum_logprob; then
  say 2 1/1 "new build serving: $(tr '\n' ' ' < $O/restart.log | cut -c1-160) | score probe: $(cut -c1-200 $O/score_probe.json)"
else
  ERR=$((ERR + 1))
  say 2 1/1 "the new build did not come up or /v1/score did not answer: $(tail -3 $O/restart.log | tr '\n' ' ' | cut -c1-240). Previous binary back."
  $SSH $TWO "cd ~/dgpp-next/build-release && cp -p 6ix-Serve.prev 6ix-Serve"
  bash $D/sixlabs/bench/six_on_dgxtwo.sh > $O/restart_prev.log 2>&1 || ERR=$((ERR + 1))
  say 2 1/1 "previous build: $(tr '\n' ' ' < $O/restart_prev.log | cut -c1-200)"
  echo "ALLDONE err=$ERR"; exit 4
fi

say 3 0/1 "waiting for the evidence job to start and finish (it wants the engine alone)"
for i in $(seq 1 20); do pgrep -f "evidence_locate.py" > /dev/null && break; sleep 15; done    # up to 5 min for it to see /v1/score
for i in $(seq 1 360); do
  pgrep -f "evidence_locate.py|run_when_ready.sh" > /dev/null || break
  sleep 15
done
pgrep -f "evidence_locate.py|run_when_ready.sh" > /dev/null && { ERR=$((ERR + 1)); say 3 1/1 "the evidence job is still running after 90 minutes: my benchmarks not started"; echo "ALLDONE err=$ERR"; exit 5; }
say 3 1/1 "evidence job gone: $(tail -1 /home/mark/evidence-locate-gain/run.log 2>/dev/null | cut -c1-160)"

# Nothing else on that GPU, engine idle: or these are not clean numbers.
others=$(gpu_others); busy=$(( $(metric dgpp_num_requests_running) + $(metric dgpp_num_requests_waiting) ))
say 4 0/5 "preflight: other GPU processes on DGXtwo $others, requests in flight $busy"
if [ "$others" != "0" ] || [ "$busy" != "0" ]; then ERR=$((ERR + 1)); say 4 0/5 "NOT CLEAN: benchmarks not started"; echo "ALLDONE err=$ERR"; exit 6; fi
run() {   # run N NAME CMD...: one benchmark, with the engine's request counter around it
  local n=$1 name=$2; shift 2
  local r0=$(metric dgpp_requests_total)
  ( "$@" ) > $O/$name.log 2>&1 || ERR=$((ERR + 1))
  say 4 $n/5 "$name done; engine requests during it: $(( $(metric dgpp_requests_total) - r0 )); other GPU processes: $(gpu_others)"
}
cd $D
run 1 decode python3 sixlabs/bench/bench_decode.py --base $BASE --model $M --streams 1,2,4,8
run 2 mixed_load python3 sixlabs/bench/mixed_load.py --base $BASE --model $M --label dgxtwo_256_idle4096_p4_shortest --writers 4 --readers 2 --lines 530 --seconds 60 --out $O/mixed_load.jsonl
run 3 soak bash -c "cd $D/scripts && python3 serve_soak.py $H $P 30 $O/soak $M"
run 4 agentic bash -c "cd $D/scripts && python3 serve_agentic_streams.py $H $P --model $M --streams 4 --turns 8 --out $O/agentic --label 6ix-80b-dgxtwo"
run 5 judge_sweep bash -c "cd /home/mark/griff-research && JUDGE_URL=$BASE/v1/chat/completions python3 conc_sweep.py --work pilot_dither_work.jsonl --model $M --concs 4,8,16,32 --rows 24"
echo "ALLDONE err=$ERR"
