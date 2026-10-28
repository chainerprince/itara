#!/usr/bin/env bash
# The whole Day 5-6 pipeline at full scale on simulated models: checks the code, not the idea.
# CPU only, no GPU or Ollama; the simulated models answer from what they're shown (eval/sim.py).
#
#   docker compose start && scripts/sim_full.sh       # SIM_ID to override the run-id prefix
#
# Every SWE-bench Lite item through four baselines, a LoCoMo sample, team replay (related and
# random task order, shared and no_sharing), then the invariant checks. Results are labelled
# simulated; open eval/day6_results.ipynb to look at them.
set -euo pipefail
cd "$(dirname "$0")/.."

ID=${SIM_ID:-sim1}
PORT=8099
export LLAMA_BASE_URL=http://localhost:$PORT/v1 LLAMA_MODEL=sim-weak LLAMA_BACKEND=vllm
export MUSE_BASE_URL=http://localhost:$PORT/v1 MUSE_MODEL=sim-strong MUSE_IS_STANDIN=false
export MODEL_CACHE=
# A fresh database of its own, so simulated memory never mixes with other runs'.
export DATABASE_URL=${SIM_DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/gateway_sim}
uv run python -m scripts.init_db --fresh

uv run python -m eval.sim --port $PORT > eval/data/sim_server.log 2>&1 &
SIM_PID=$!
trap 'kill $SIM_PID 2>/dev/null' EXIT
until curl -s -o /dev/null localhost:$PORT/docs; do sleep 1; done

for bench in "swebench -n 300" "locomo -n 300"; do
  name=${bench%% *}
  for baseline in llama_only escalate_compact muse_direct rule_gate; do
    uv run python -m eval.run $bench --baseline $baseline --resume "$ID-$name-$baseline" --yes
  done
  uv run python -m eval.checks "$ID-$name-muse_direct" "$ID-$name-escalate_compact" \
    "$ID-$name-llama_only" "$ID-$name-rule_gate" || FAILED=1
done

for order in related random; do
  for baseline in rule_gate no_sharing; do
    uv run python -m eval.run replay --baseline $baseline --order $order \
      --users 1,2,4,8,16 --per-user 2 --yes
  done
done
echo "Done. Open eval/day6_results.ipynb (BENCH = swebench or locomo)."
exit ${FAILED:-0}
