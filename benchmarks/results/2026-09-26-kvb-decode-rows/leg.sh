#!/usr/bin/env bash
# One fabric leg: up (with --bin) -> timed_load -> /v1/metrics -> down; raw files under raw/NAME.*
# usage: leg.sh NAME CONFIG BIN CONCURRENCY_CSV [extra timed_load args]
set -u
NAME=$1; CONFIG=$2; BIN=$3; CONC=$4; shift 4
OUT=~/claude-scratch/2026-09-26-kvb-rows-ab/raw
TREE=/home/stephen/workspace/dgpp
PORT=18080
cd "$TREE"
echo "== leg $NAME  config=$CONFIG bin=$BIN conc=$CONC  $(date -u +%FT%TZ)" | tee "$OUT/$NAME.log"
strings "$BIN" | grep -m1 -E "0\.1\.0\+g[0-9a-f]{6,}" > "$OUT/$NAME.version.txt"
timeout 600 python3 scripts/dgpp-cluster up --config "$CONFIG" --bin "$BIN" > "$OUT/$NAME.up.txt" 2>&1
UP=$?
echo "up exit $UP" | tee -a "$OUT/$NAME.log"
if [ $UP -ne 0 ]; then
  tail -20 "$OUT/$NAME.up.txt" | tee -a "$OUT/$NAME.log"
  python3 scripts/dgpp-cluster down --config "$CONFIG" > "$OUT/$NAME.down.txt" 2>&1
  exit 1
fi
python3 scripts/timed_load.py 127.0.0.1 $PORT --concurrency "$CONC" --classes all --repeat 2 --json-out "$OUT/$NAME.json" "$@" > "$OUT/$NAME.load.txt" 2>&1
echo "timed_load exit $?" | tee -a "$OUT/$NAME.log"
curl -s "http://127.0.0.1:$PORT/v1/metrics" > "$OUT/$NAME.metrics.json"
python3 scripts/dgpp-cluster down --config "$CONFIG" > "$OUT/$NAME.down.txt" 2>&1
echo "down exit $?  $(date -u +%FT%TZ)" | tee -a "$OUT/$NAME.log"
grep -i 'identity\|md5\|op stream' "$OUT/$NAME.down.txt" | head -3 | tee -a "$OUT/$NAME.log"
