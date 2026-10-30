#!/usr/bin/env bash
# The full eval on the real models .env points at: every SWE-bench Lite item and a LoCoMo
# sample through four baselines, then team replay. The same steps as scripts/sim_full.sh,
# without the simulator or its invariant checks (real models aren't "right iff shown").
#
#   docker compose start && caffeinate -i scripts/full_run.sh
#   FULL_ID=seq1 INGEST_CONCURRENCY=1 caffeinate -i scripts/full_run.sh   # sequential reference
#
# - A fresh database per FULL_ID (gateway_<id>_full), so the spend cap and memory are the run's
#   own. Rerunning with the same FULL_ID continues: finished items are skipped (replay restarts).
# - SPEND_CAP_USD is raised for the run (FULL_SPEND_CAP, default $30; ~$15-20 of Muse expected).
# - MODEL_CACHE is off, so a repeat run is a real repeat, not a replay of this one.
# - Muse-only baselines (escalate_compact, muse_direct) run back to back, so the GPU can scale
#   to zero while they run.
set -euo pipefail
cd "$(dirname "$0")/.."

ID=${FULL_ID:-full1}
N=${FULL_N:-300}
BASE=${FULL_DATABASE_BASE:-postgresql://postgres:postgres@localhost:5432}
export DATABASE_URL="$BASE/gateway_${ID//-/_}_full"
export INGEST_CONCURRENCY=${INGEST_CONCURRENCY:-16}
export SPEND_CAP_USD=${FULL_SPEND_CAP:-30}
export MODEL_CACHE=
echo "== $ID: $N items, INGEST_CONCURRENCY=$INGEST_CONCURRENCY, cap \$$SPEND_CAP_USD, $DATABASE_URL"
uv run python -m scripts.init_db --create

for bench in swebench locomo; do
  # llama_only first: it fills memory (the ingest) for the baselines after it.
  for baseline in llama_only rule_gate escalate_compact muse_direct; do
    uv run python -m eval.run $bench -n "$N" --baseline $baseline --resume "$ID-$bench-$baseline" --yes
  done
done

for order in related random; do
  for baseline in rule_gate no_sharing; do
    uv run python -m eval.run replay --baseline $baseline --order $order \
      --users 1,2,4,8,16 --per-user 2 --yes
  done
done
echo "Done. Open eval/day6_results.ipynb with BENCH = swebench or locomo and"
echo "RUN_IDS = [f\"$ID-{BENCH}-{b}\" for b in (\"llama_only\", \"rule_gate\", \"escalate_compact\", \"muse_direct\")]"
