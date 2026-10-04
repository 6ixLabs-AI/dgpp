#!/bin/bash
# restart_test.sh KEY=VALUE ... — restart the dgpp-next TEST deployment (never the production one)
# with engine settings changed, and wait for READY. Run on DGXone.
# The launcher keys a deployment on its config PATH, so every variant is written to the one
# canonical test config; `down` with that same path then always finds the running server.
#   bash restart_test.sh mtp_depth=2 dense_weights=fp8 max_concurrency=8
set -u
CFG=deploy/cluster_qwen3-next-80b_nvfp4_w1.json
cd /home/mark/dgpp-next || exit 2
python3 scripts/dgpp-cluster down --config "$CFG" > /dev/null 2>&1
python3 - "$CFG" "$@" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    try:
        v = json.loads(v)
    except Exception:
        pass
    c["engine"][k] = v
json.dump(c, open(p, "w"), indent=2)
print("engine:", {k: c["engine"][k] for k in ("max_concurrency", "kv_capacity", "mtp", "mtp_depth", "dense_weights") if k in c["engine"]})
PY
# Only this start's lines count: the server log of a failed start keeps its ERROR line, and a
# launcher whose rank died stays up holding the deployment lock (kill it by PID, then `down`).
T0=$(date -u "+%Y-%m-%d %H:%M:%S")
: > ~/dgpp-next-up.log
ssh -o BatchMode=yes -o ConnectTimeout=6 mark@10.77.0.1 "cd ~/dgpp-next && (setsid python3 scripts/dgpp-cluster up --config $CFG > ~/dgpp-next-up.log 2>&1 < /dev/null &); sleep 1"
for i in $(seq 1 50); do
  if grep -q "READY" ~/dgpp-next-up.log 2>/dev/null; then
    L=$(ls -t ~/dgpp-next/log/deployments/*/serve_r0.log | head -1)
    grep -o "listening on.*boot [0-9.]*s" "$L" | tail -1 | sed 's/.*(boot/boot/'
    grep "memory plan total" "$L" | tail -1 | sed 's/.*memory plan total/memory plan total/' | cut -c1-60
    exit 0
  fi
  if grep -q "^FAIL" ~/dgpp-next-up.log 2>/dev/null; then grep "^FAIL" ~/dgpp-next-up.log | cut -c1-240; exit 1; fi
  L=$(ls -t ~/dgpp-next/log/deployments/*/serve_r0.log 2>/dev/null | head -1)
  if [ -n "$L" ] && awk -v t="$T0" '$0 >= t && / ERROR /' "$L" | grep -q .; then
    awk -v t="$T0" '$0 >= t && / ERROR /' "$L" | head -3 | cut -c1-300
    echo "the rank exited; its launcher is still holding the lock: kill the dgpp-cluster up PID, then run down"
    exit 1
  fi
  if grep -q "another launcher is starting or stopping" ~/dgpp-next-up.log 2>/dev/null; then
    echo "a previous launcher still holds this deployment (find it with ps, kill it by PID, run down, retry)"; exit 1
  fi
  sleep 5
done
echo "timed out waiting for READY"; tail -3 ~/dgpp-next-up.log; exit 1
