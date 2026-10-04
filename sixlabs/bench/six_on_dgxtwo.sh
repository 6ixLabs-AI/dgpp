#!/bin/bash
# six_on_dgxtwo.sh [KEY=VALUE ...] — (re)start the 6ix 80B engine on DGXtwo, from DGXone.
#
# DGXtwo runs nothing else on its GPU, so the engine is not slowed by DGXone's scoring and
# embedding services (2026-10-04: HEM on DGXone took about 90 % of that GPU and halved the
# engine, 39 against 80 tokens/s single stream). The engine binds 10.77.0.2:18090 (the fabric
# address DGXone's gateway already reaches); weights are read over /mnt/dgxone-hf.
#
# Run on DGXone. The launcher must start from an OpenSSH session (memlock), which the fabric
# ssh below gives it. KEY=VALUE pairs change engine settings in DGXtwo's config first, as
# restart_test.sh does on DGXone. The config and .env on DGXtwo are created on first use from
# DGXone's live config.
set -u
TWO=marktwo@10.77.0.2
SSH="ssh -o BatchMode=yes -o ConnectTimeout=6"
CFG=deploy/cluster_qwen3-next-80b_nvfp4_w1.json
D=/home/marktwo/dgpp-next

if ! $SSH $TWO "test -f $D/$CFG"; then
  python3 - <<'PY' > /tmp/six_two_cfg.json
import json
c = json.load(open("/home/mark/dgpp-next/deploy/cluster_qwen3-next-80b_nvfp4_w1.json"))
c["http"] = {"bind_host": "10.77.0.2", "port": 18090}
print(json.dumps(c, indent=2))
PY
  scp -q -o BatchMode=yes /tmp/six_two_cfg.json $TWO:$D/$CFG || exit 2
fi
$SSH $TWO "cat > $D/.env" <<'ENV'
# The 6ix 80B engine on DGXtwo, world 1 (written by sixlabs/bench/six_on_dgxtwo.sh).
DGPP_NODES="127.0.0.1"
DGPP_SSH_USER="marktwo"
DGPP_HTTP_PORT=18090
DGPP_HTTP_BIND=10.77.0.2
DGPP_FABRIC_PORT=29980
DGPP_JOURNAL_PORT=29981
DGPP_LOG_DIR="~/dgpp-next/log"
DGPP_STAGE_DIR="/tmp/bus4-next"
DGPP_RELEASE_DIR="~/dgpp-next/releases"
DGPP_RESIDENT_CACHE_DIR="~/.cache/dgpp-next/resident"
ENV

$SSH $TWO "cd $D && python3 scripts/dgpp-cluster down --config $CFG > /dev/null 2>&1; python3 - $CFG $(printf '%q ' "$@")" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
for kv in sys.argv[2:]:
    if "=" not in kv:
        continue
    k, v = kv.split("=", 1)
    try:
        v = json.loads(v)
    except Exception:
        pass
    c["engine"][k] = v
# The config's own http block wins over .env, and a config copied from DGXone carries DGXone's
# bridge address (2026-10-04: the engine came up on DGXtwo's 172.17.0.1, reachable by nobody).
c["http"] = {"bind_host": "10.77.0.2", "port": 18090}
json.dump(c, open(p, "w"), indent=2)
e = c["engine"]
print("engine:", {k: e.get(k) for k in ("max_concurrency", "kv_capacity", "default_max_tokens", "prefill_budget_tokens",
                                        "prefill_idle_budget_tokens", "decode_passes_per_prefill", "prefill_order")}, "bind", c["http"])
PY

# The preflight checks ssh to the node list; rank 0 runs beside the launcher and DGXtwo does
# not ssh to itself, so it is skipped here.
$SSH $TWO "cd $D && : > ~/dgpp-next-up.log && (setsid python3 scripts/dgpp-cluster up --skip-preflight --config $CFG > ~/dgpp-next-up.log 2>&1 < /dev/null &); sleep 1"
for i in $(seq 1 130); do
  S=$($SSH $TWO "grep -m1 -E 'READY|^FAIL|Traceback|another launcher|ValueError' ~/dgpp-next-up.log 2>/dev/null")
  case "$S" in
    *READY*) $SSH $TWO "L=\$(ls -t $D/log/deployments/*/serve_r0.log | head -1); grep -o 'listening on.*boot [0-9.]*s' \$L | tail -1 | sed 's/.*(boot/boot/'; grep 'memory plan total' \$L | tail -1 | sed 's/.*memory plan total/memory plan total/' | cut -c1-60; grep 'prefill budget' \$L | tail -1 | cut -c40-260; echo errors: \$(grep -c ' ERROR ' \$L)"; exit 0 ;;
    "") ;;
    *) echo "start failed: $S"; $SSH $TWO "tail -5 ~/dgpp-next-up.log; L=\$(ls -t $D/log/deployments/*/serve_r0.log 2>/dev/null | head -1); [ -n \"\$L\" ] && grep ' ERROR ' \$L | tail -3 | cut -c1-300"; exit 1 ;;
  esac
  sleep 5
done
echo "timed out waiting for READY"; $SSH $TWO "tail -3 ~/dgpp-next-up.log"; exit 1
