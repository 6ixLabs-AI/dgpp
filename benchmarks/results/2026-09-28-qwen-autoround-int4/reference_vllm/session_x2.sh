#!/bin/bash
# Session X (2026-09-30): the author's vLLM stack on this box — cold prefill at 2K/8K/32K (our prompts, TTFT), the
# smoke-test's 8k-word prefill, then a second boot under nsys with one 32K and one 8K prefill; production back after.
set -u
R=/home/stephen/claude-scratch/2026-09-30-reference-vllm
OUT=$R/run_$(date +%m%d_%H%M); mkdir -p $OUT; echo "OUT=$OUT"
PROD_CFG=/home/stephen/dgpp/cluster_glm-5.3-flash_nvfp4-fp8_w4.as-running-2026-09-28.json
PROD_BIN=/tmp/dgpp-pr65-maintenance/production-dgpp-serve
H=127.0.0.1; P=18300
wait_ready() { for i in $(seq 1 240); do curl -sf -m 5 http://$H:$P/health >/dev/null 2>&1 && return 0; if ! docker ps --format '{{.Names}}' | grep -q "^qwen38-ref$"; then echo "container exited"; docker logs qwen38-ref > $OUT/boot_failed_$(date +%H%M%S).log 2>&1; grep -n -A30 "Traceback" $OUT/boot_failed_*.log | grep EngineCore | tail -25 | cut -c1-220; return 1; fi; sleep 5; done; echo "not ready after 20 min"; docker logs qwen38-ref > $OUT/boot_failed_$(date +%H%M%S).log 2>&1; grep -n -A30 "Traceback" $OUT/boot_failed_*.log | grep EngineCore | tail -25 | cut -c1-220; return 1; }
warm() { for i in 1 2 3; do curl -s -m 600 http://$H:$P/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32,"temperature":0}' | python3 -c 'import json,sys;d=json.load(sys.stdin);print("   warm:",repr(d["choices"][0]["message"]["content"][:60]))' 2>&1; done; }
cd /home/stephen/workspace/dgpp && python3 scripts/dgpp-cluster down --config $PROD_CFG > $OUT/prod_down.log 2>&1
echo "production down $(date +%T)"
echo "== boot 1 (plain) $(date +%T)"
OUT=$OUT NSYS=0 bash $R/run_ref.sh > $OUT/run1.log 2>&1 || { echo "run failed"; cat $OUT/run1.log | tail -5; }
if wait_ready; then
  echo "ready $(date +%T)"; docker logs qwen38-ref 2>&1 | grep -iE "prewarm|ple|marlin|graph|KV cache|Maximum concurrency|loaded|took" | tail -12 | cut -c1-180
  warm
  echo "== probe $(date +%T)"
  python3 $R/ref_prefill_probe.py $H $P 2048 8192 32768 --repeat 2 --seed 7 --tag ref-$(date +%s) --json-out $OUT/prefill_ref.json 2>&1 | tee $OUT/prefill_ref.log | grep -E "median|calibrate"
  echo "== smoke-style (word x8000, max_tokens 1, non-streamed) $(date +%T)"
  python3 - $H $P <<'PY' 2>&1 | tee $OUT/smoke_prefill.log
import json,sys,time,urllib.request
base=f"http://{sys.argv[1]}:{sys.argv[2]}"
for rep in range(2):
    prompt="word "*8000
    t=time.time()
    req=urllib.request.Request(base+"/v1/completions",data=json.dumps({"model":"qwen","prompt":prompt+str(rep),"max_tokens":1,"temperature":0}).encode(),headers={"Content-Type":"application/json"})
    u=json.load(urllib.request.urlopen(req,timeout=600))["usage"]; dt=time.time()-t
    print(f"   {u['prompt_tokens']} tok in {dt:.2f}s => {u['prompt_tokens']/dt:.0f} tok/s prefill (smoke definition)")
PY
  docker logs qwen38-ref > $OUT/server1.log 2>&1
fi
docker rm -f qwen38-ref >/dev/null 2>&1
echo "== boot 2 (nsys) $(date +%T)"
OUT=$OUT NSYS=1 bash $R/run_ref.sh > $OUT/run2.log 2>&1 || { echo "run failed"; tail -5 $OUT/run2.log; }
if wait_ready; then
  echo "ready $(date +%T)"; warm
  python3 $R/ref_prefill_probe.py $H $P 32768 8192 --repeat 1 --seed 7 --tag nsys-$(date +%s) --json-out $OUT/prefill_nsys.json 2>&1 | tee $OUT/prefill_nsys.log | grep -E "median|calibrate"
  docker logs qwen38-ref > $OUT/server2.log 2>&1
  echo "== stopping under nsys (SIGINT) $(date +%T)"
  docker kill -s INT qwen38-ref >/dev/null 2>&1
  for i in $(seq 1 120); do docker ps --format '{{.Names}}' | grep -q "^qwen38-ref$" || break; sleep 5; done
  ls -la $OUT/r0.nsys-rep 2>/dev/null | awk '{print "report", $5, "bytes"}' || echo "NO NSYS REPORT"
fi
docker rm -f qwen38-ref >/dev/null 2>&1
cd /home/stephen/workspace/dgpp && python3 scripts/dgpp-cluster up --config $PROD_CFG --bin $PROD_BIN > $OUT/prod_up.log 2>&1 && echo "production restored $(date +%T)" || echo "PRODUCTION RESTORE FAILED"
echo "session done $(date +%T)"
