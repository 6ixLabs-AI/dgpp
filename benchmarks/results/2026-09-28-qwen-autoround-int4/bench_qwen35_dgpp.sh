#!/bin/bash
# The author's bench_qwen35.sh protocol (albond's script the model card
# cites) against dgpp: five prompts, one chat completion each, two runs,
# temperature 0.0, no thinking flag, model "qwen", tok/s = completion tokens
# over the whole request's wall time (TTFT inside).
#   bench_qwen35_dgpp.sh [HOST] [PORT]      NOTHINK=1 adds enable_thinking=false
HOST=${1:-127.0.0.1}; PORT=${2:-18080}; MODEL=${MODEL:-qwen}
declare -a NAMES=("Q&A" "Code" "JSON" "Math" "LongCode")
declare -a MAXT=(256 512 1024 64 2048)
declare -a PROMPTS=(
  "What are the main differences between TCP and UDP? Be concise."
  "Write a Python function that implements binary search on a sorted list. Include type hints and docstring."
  "Generate a JSON array of 10 fictional employees with fields: name, age, department, salary, email, skills (array of 3). Output ONLY valid JSON, no explanation."
  "What is 7823 * 4519? Show only the answer."
  "Write a complete Python implementation of a red-black tree with insert, delete, search, and in-order traversal. Include all rotation methods."
)
echo "Qwen bench_qwen35 protocol against $HOST:$PORT model=$MODEL nothink=${NOTHINK:-0}  $(date)"
for run in 1 2; do
  echo "── Run $run/2 ──"
  for i in 0 1 2 3 4; do
    body=$(python3 - "$MODEL" "${PROMPTS[$i]}" "${MAXT[$i]}" "${NOTHINK:-0}" <<'PY'
import json, sys
model, prompt, maxt, nothink = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
req = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": maxt}
if nothink: req["chat_template_kwargs"] = {"enable_thinking": False}
print(json.dumps(req))
PY
)
    t0=$(date +%s.%N)
    resp=$(curl -s -m 600 "http://$HOST:$PORT/v1/chat/completions" -H 'content-type: application/json' -d "$body")
    t1=$(date +%s.%N)
    RESP="$resp" python3 - "${NAMES[$i]}" "$t0" "$t1" <<'PY'
import json, os, sys
name, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
d = json.loads(os.environ["RESP"])
u = d.get("usage", {}); n = u.get("completion_tokens", 0); p = u.get("prompt_tokens", 0)
dt = t1 - t0
print(f"  [{name}] {n} tokens in {dt:.2f}s = {n / dt:.1f} tok/s (prompt: {p})")
PY
  done
done
echo "=== Done ==="
