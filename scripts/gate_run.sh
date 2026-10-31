#!/usr/bin/env bash
# The learned gate, after scripts/full_run.sh: adds the escalate_raw baseline (Muse with the
# raw pages behind the retrieved memory, the gate's third route) on the full run's database,
# then trains the gate, replays it offline and marks the north star (eval/learn_gate.py).
#
#   docker compose start && caffeinate -i scripts/gate_run.sh      # FULL_ID / FULL_N as for full_run.sh
#   GATE_ONLY=1 scripts/gate_run.sh      # retrain only (no model calls), e.g. with --budget changed
#
# - Same database, spend cap and settings as the full run, so memory is the one it answered
#   from. The cap counts the full run's spend too (FULL_SPEND_CAP, default $50 here).
# - Rerunning continues: answered items are skipped. learn_gate itself takes ~5-10 minutes
#   (it searches memory again for every item) and calls no model.
set -euo pipefail
cd "$(dirname "$0")/.."

ID=${FULL_ID:-full1}
N=${FULL_N:-300}
BASE=${FULL_DATABASE_BASE:-postgresql://postgres:postgres@localhost:5432}
export DATABASE_URL="$BASE/gateway_${ID//-/_}_full"
export INGEST_CONCURRENCY=${INGEST_CONCURRENCY:-16}
export SPEND_CAP_USD=${FULL_SPEND_CAP:-50}
export MODEL_CACHE=

if [ -z "${GATE_ONLY:-}" ]; then
  for bench in swebench locomo; do
    uv run python -m eval.run $bench -n "$N" --baseline escalate_raw --resume "$ID-$bench-escalate_raw" --yes
  done
fi
uv run python -m eval.learn_gate --id "$ID" -n "$N" "$@"
echo "Results notebook: eval/day6_results.ipynb with BENCH = swebench or locomo and"
echo "RUN_IDS = [f\"$ID-{BENCH}-{b}\" for b in (\"muse_direct\", \"full\", \"escalate_raw\", \"escalate_compact\", \"rule_gate\", \"llama_only\")]"
