#!/usr/bin/env bash
# Is concurrent ingest as good as sequential, on the models .env points at? Three fresh
# databases get the same items: seqa and seqb one page at a time (INGEST_CONCURRENCY=1; seqb
# measures run-to-run noise) and conc at $CONCURRENCY. Each then answers them with llama_only
# (the local model; LoCoMo's judge adds a few cents of Muse), and eval.ingest_check compares
# the three. Run it before choosing the mode for the full run.
#
#   docker compose start && scripts/ingest_check.sh    # CHECK_ID / CHECK_N / CONCURRENCY to override
#
# - MODEL_CACHE is off: replayed responses would make the three runs identical by construction.
# - Rerunning with the same CHECK_ID continues: docs already in memory and items already
#   answered are skipped. An interrupted sequential ingest can leave one doc half stored, which
#   the page-set check will flag; use a new CHECK_ID to start over.
# - pages_per_second is indicative: later modes can hit earlier modes' prefix cache.
set -euo pipefail
cd "$(dirname "$0")/.."

ID=${CHECK_ID:-check1}
N=${CHECK_N:-100}
CONCURRENCY=${CONCURRENCY:-16}
BASE=${CHECK_DATABASE_BASE:-postgresql://postgres:postgres@localhost:5432}
export MODEL_CACHE=

for mode in seqa seqb conc; do
  export DATABASE_URL="$BASE/gateway_${ID//-/_}_${mode}_check"
  export INGEST_CONCURRENCY=$([ "$mode" = conc ] && echo "$CONCURRENCY" || echo 1)
  echo "== $mode (INGEST_CONCURRENCY=$INGEST_CONCURRENCY)"
  uv run python -m scripts.init_db --create
  for bench in swebench locomo; do
    uv run python -m eval.run "$bench" -n "$N" --ingest-only --resume "$ID-$mode"
    uv run python -m eval.run "$bench" -n "$N" --baseline llama_only --resume "$ID-$mode" --yes
  done
done
uv run python -m eval.ingest_check --id "$ID" -n "$N"
