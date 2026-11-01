#!/usr/bin/env bash
# Muse at a lower reasoning effort, after scripts/gate_run.sh. Muse's reasoning is most of an
# escalation's tokens (eval/full_results.ipynb, section 3), so this reruns every condition
# that calls Muse (muse_direct, escalate_compact, escalate_raw) with MUSE_REASONING_EFFORT
# set, on the full run's database and memory, under the run id <FULL_ID>-<effort>. Llama's
# side doesn't see Muse's effort, so the full run's drafts and confidence probes are reused.
# Then the gates are trained and scored on it (eval/learn_gate.py).
#
#   docker compose start && caffeinate -i scripts/effort_run.sh       # EFFORT=minimal
#   EFFORT=low scripts/effort_run.sh
#   GATE_ONLY=1 scripts/effort_run.sh      # retrain only (no model calls)
#
# - Added on 31 Oct, after the full run's results: report it as such. muse_direct is rerun at
#   the same effort, so the comparison stays like for like; the default-effort muse_direct
#   stays the stronger reference.
# - Cost at minimal: ~$15 of Muse (mostly muse_direct's prompts), ~2 hours (items are
#   answered one at a time). The LoCoMo judge keeps its own minimal effort, as before.
# - The cap counts all of this database's spend so far (~$40): FULL_SPEND_CAP, default $60.
# - Rerunning continues: answered items are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

ID=${FULL_ID:-full1}
N=${FULL_N:-300}
EFFORT=${EFFORT:-minimal}
NEW="$ID-$EFFORT"
BASE=${FULL_DATABASE_BASE:-postgresql://postgres:postgres@localhost:5432}
export DATABASE_URL="$BASE/gateway_${ID//-/_}_full"
export INGEST_CONCURRENCY=${INGEST_CONCURRENCY:-16}
export SPEND_CAP_USD=${FULL_SPEND_CAP:-60}
export MUSE_REASONING_EFFORT=$EFFORT
export MODEL_CACHE=
R=eval/results

for bench in swebench locomo; do
  if [ -z "${GATE_ONLY:-}" ]; then
    for baseline in muse_direct escalate_compact escalate_raw; do
      uv run python -m eval.run $bench -n "$N" --baseline $baseline --resume "$NEW-$bench-$baseline" --yes
    done
  fi
  cp "$R/raw/${bench}_llama_only_$ID-$bench-llama_only.jsonl" "$R/raw/${bench}_llama_only_$NEW-$bench-llama_only.jsonl"
  cp "$R/${bench}_llama_only_$ID-$bench-llama_only.json" "$R/${bench}_llama_only_$NEW-$bench-llama_only.json"
  cp "$R/raw/${bench}_confidence_$ID.jsonl" "$R/raw/${bench}_confidence_$NEW.jsonl"
done
uv run python -m eval.learn_gate --id "$NEW" -n "$N" --database "$DATABASE_URL" "$@"
echo "Results notebook: eval/full_results.ipynb (set RUN_ID = \"$NEW\" for this run)"
