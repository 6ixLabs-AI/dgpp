#!/bin/bash
# Session X4 (2026-09-30): the reference under nsys with periodic CUDA buffer flushes (the X3 report lost the probe
# phase at shutdown): one 32K and one 8K prefill, a 15 s settle, SIGINT; production back after.
set -u
R=/home/stephen/claude-scratch/2026-09-30-reference-vllm
OUT=$R/run_nsys_$(date +%m%d_%H%M); mkdir -p $OUT; echo "OUT=$OUT"
PROD_CFG=/home/stephen/dgpp/cluster_glm-5.3-flash_nvfp4-fp8_w4.as-running-2026-09-28.json
PROD_BIN=/tmp/dgpp-pr65-maintenance/production-dgpp-serve
H=127.0.0.1; P=18300
wait_ready() { for i in $(seq 1 240); do curl -sf -m 5 http://$H:$P/health >/dev/null 2>&1 && return 0; if ! docker ps --format '{{.Names}}' | grep -q "^qwen38-ref$"; then echo "container exited"; docker logs qwen38-ref > $OUT/boot_failed.log 2>&1; grep -n -A30 "Traceback" $OUT/boot_failed.log | grep EngineCore | tail -20 | cut -c1-200; return 1; fi; sleep 5; done; echo "not ready"; return 1; }
warm() { for i in 1 2 3; do curl -s -m 600 http://$H:$P/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32,"temperature":0}' > /dev/null; done; echo "   warmed"; }
cd /home/stephen/workspace/dgpp && python3 scripts/dgpp-cluster down --config $PROD_CFG > $OUT/prod_down.log 2>&1
echo "production down $(date +%T)"
echo "== boot (nsys, flush 5 s) $(date +%T)"
OUT=$OUT NSYS=1 bash $R/run_ref.sh > $OUT/run.log 2>&1 || { echo "run failed"; tail -5 $OUT/run.log; }
if wait_ready; then
  echo "ready $(date +%T)"; warm
  python3 $R/ref_prefill_probe.py $H $P 32768 8192 --repeat 1 --seed 7 --tag nsys4-$(date +%s) --json-out $OUT/prefill_nsys.json 2>&1 | tee $OUT/prefill_nsys.log | grep -E "median|calibrate|\[n"
  echo "== settling 20 s before the stop $(date +%T)"; sleep 20
  docker logs qwen38-ref > $OUT/server.log 2>&1
  echo "== stopping under nsys (SIGINT) $(date +%T)"
  docker kill -s INT qwen38-ref >/dev/null 2>&1
  for i in $(seq 1 120); do docker ps --format '{{.Names}}' | grep -q "^qwen38-ref$" || break; sleep 5; done
  ls -la $OUT/r0.nsys-rep 2>/dev/null | awk '{print "report", $5, "bytes"}' || echo "NO NSYS REPORT"
fi
docker rm -f qwen38-ref >/dev/null 2>&1
cd /home/stephen/workspace/dgpp && python3 scripts/dgpp-cluster up --config $PROD_CFG --bin $PROD_BIN > $OUT/prod_up.log 2>&1 && echo "production restored $(date +%T)" || echo "PRODUCTION RESTORE FAILED"
echo "session done $(date +%T)"
