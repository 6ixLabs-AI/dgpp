#!/bin/bash
# The DeepSeek-V4-Flash (deepseek_v4, the 0731 release) serving gates on the fabric:
# boot the world from a cluster config (deploy/cluster_deepseek-v4-flash_mxfp4-fp8_w4.json
# or _w2.json: the DSpark block draft + the decode graph; --knobs "--no-mtp" for the T=1
# world), then the API check, the greedy transcripts (compared against a reference file
# when given), the client-side pace, the MTP acceptance per prompt class (read from the
# scheduler's retired lines), the prefill cost by length through the endpoint, optionally
# the task evals, and the world down with its op-stream md5s. One world at a time on the
# fabric.
#   fabric_dsv4_serve.sh CONFIG OUT_DIR [--compare REF_TRANSCRIPTS.json] [--eval]
#                          [--allow-code-execution] [--prefill "LEN..."] [--knobs "FLAGS"]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/cluster_env.sh" || exit 1
CONFIG=${1:?CONFIG}; OUT=${2:?OUT_DIR}; shift 2
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
export DGPP_CLUSTER_CONFIG="$CONFIG"
case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
COMPARE=""; EVAL=0; ALLOW_CODE=0; KNOBS=""; PREFILL="512 2048 8192"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --compare) COMPARE="$2"; shift 2;;
    --eval) EVAL=1; shift;;
    --allow-code-execution) ALLOW_CODE=1; shift;;
    --prefill) PREFILL="$2"; shift 2;;
    --knobs) KNOBS="$2"; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
if [[ $EVAL -eq 1 && $ALLOW_CODE -ne 1 ]]; then
  echo "--eval includes HumanEval: use an isolated evaluation environment and pass --allow-code-execution" >&2
  exit 2
fi
mkdir -p "$OUT"
# The endpoint: the config's bind host when it names one, else the site's client host.
HOST=$(jq -r '.http.bind_host // empty' "$CONFIG"); [[ -z "$HOST" || "$HOST" == "0.0.0.0" ]] && { HOST=$(dgpp_client_host) || exit 1; }
PORT=$(jq -r '.http.port // empty' "$CONFIG"); [[ -z "$PORT" ]] && { PORT=$(dgpp_http_port) || exit 1; }
export DGPP_DATA_DIR="${DGPP_DATA_DIR:-$ROOT/build-ci/eval_data}"
cd "$ROOT" || exit 1
up=(python3 scripts/dgpp-cluster up --config "$CONFIG" --log-dir "$OUT/world")
[[ -n "$KNOBS" ]] && up+=("--knobs=$KNOBS")  # the = form: a knob that starts with -- is a value, not a flag
"${up[@]}" > "$OUT/up.log" 2>&1 || { echo "world did not come up (see $OUT/up.log)"; tail -20 "$OUT/up.log"; exit 1; }
echo "== world up ($CONFIG)"; grep -h 'model constructed\|graph variants warm\|memory plan total\|listening' "$OUT/world/serve_r0.log" | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-200
rc=0
echo "== greedy transcripts"
cmp=(); [[ -n "$COMPARE" ]] && cmp=(--compare "$COMPARE")
python3 scripts/serve_greedy_transcript.py "$HOST" "$PORT" --out "$OUT/transcripts.json" "${cmp[@]}" 2>&1 | tee "$OUT/transcripts.log" || rc=1
echo "== pace (serve_bench, 200 tokens)"
python3 scripts/serve_bench.py "$HOST" "$PORT" 200 dsv4 2>&1 | tail -4 | tee "$OUT/bench.log"
echo "== api check"
python3 scripts/serve_api_check.py "$HOST" "$PORT" > "$OUT/api_check.log" 2>&1 && echo "api check: OK" || { echo "api check: FAILED (see $OUT/api_check.log)"; tail -5 "$OUT/api_check.log"; rc=1; }
echo "== MTP acceptance per class (greedy, 300 tokens; the scheduler's retired lines)"
python3 scripts/serve_mtp_classes.py "$HOST" "$PORT" --out "$OUT/mtp_classes.json" --log "$OUT/world/serve_r0.log" 2>&1 | tee "$OUT/mtp_classes.log"
if [[ -n "$PREFILL" ]]; then
  echo "== prefill by length (server-side, median of 3)"
  # shellcheck disable=SC2086
  python3 scripts/serve_prefill_probe.py "$HOST" "$PORT" $PREFILL --repeat 3 2>&1 | grep -v '^  ' | tee "$OUT/prefill_probe.log"
fi
if [[ $EVAL -eq 1 ]]; then
  echo "== eval (gsm8k 60, humaneval 40, extract 30; thinking off)"
  python3 scripts/serve_eval.py "$HOST" "$PORT" --out "$OUT/eval" --tasks gsm8k --limit 60 --concurrency 4 --no-think 2>&1 | tail -3
  python3 scripts/serve_eval.py "$HOST" "$PORT" --out "$OUT/eval" --tasks humaneval --allow-code-execution --limit 40 --concurrency 4 --no-think 2>&1 | tail -3
  python3 scripts/serve_eval.py "$HOST" "$PORT" --out "$OUT/eval" --tasks extract --limit 30 --concurrency 4 --no-think 2>&1 | tail -3
fi
echo "== world down"
python3 scripts/dgpp-cluster down --config "$CONFIG" --log-dir "$OUT/world" 2>&1 | tail -6 | tee "$OUT/down.log"
grep -h 'stats: rank 0' "$OUT/world/serve_r0.log" | grep -v 'decode 0.0 tok/s' | tail -2 | sed 's/^[0-9-]* [0-9:.]* //' | cut -c1-220
exit $rc
