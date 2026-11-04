#!/usr/bin/env bash
# Another open model drafting in place of Llama 3.1 8B, everything else held fixed: a copy of
# a full run's memory (so nothing is ingested again), the same questions, and that run's Muse
# answers for every escalation (learn_gate/cost_compare --muse-id). Muse is called only to judge
# LoCoMo answers (minimal effort, as in the full run): the spend cap allows what the copied
# ledger already holds plus RUN_MUSE_BUDGET. Run big1 (Llama 3.3 70B) was made this way.
#
#   docker compose start
#   LLAMA_BASE_URL=<url>/v1 LLAMA_MODEL=meta-models/Muse-Glimmer-30B LLAMA_REASONING_STRENGTH=low \
#       RUN_ID=glim1 caffeinate -i scripts/open_model_run.sh
#
# RUN_ID (required), FROM_ID (default full1), RUN_N (default 300), RUN_GPUS (default 1: GPUs
# serving the open model, for its cost), RUN_BENCHES (default "swebench locomo"),
# RUN_MUSE_BUDGET (default $3). Rerunning continues: drafted and probed items are skipped. The
# run stops if any page is ingested: the memory would then differ from the source run's.
set -euo pipefail
cd "$(dirname "$0")/.."

ID=${RUN_ID:?set RUN_ID, e.g. glim1}
FROM=${FROM_ID:-full1}
N=${RUN_N:-300}
GPUS=${RUN_GPUS:-1}
BENCHES=${RUN_BENCHES:-swebench locomo}
BASE=${FULL_DATABASE_BASE:-postgresql://postgres:postgres@localhost:5432}
export DATABASE_URL="$BASE/gateway_${ID//-/_}_full"
export MODEL_CACHE=
echo "== $ID: ${LLAMA_MODEL:-the open model in .env} drafts $N items of: $BENCHES, on a copy of $FROM's memory"

uv run python - "$BASE" "gateway_${FROM//-/_}_full" "gateway_${ID//-/_}_full" <<'EOF'
import sys
import psycopg
from psycopg import sql

base, source, target = sys.argv[1:]
with psycopg.connect(f"{base}/postgres", autocommit=True) as admin:
    if admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target,)).fetchone():
        print(f"{target} exists: continuing")
    else:
        admin.execute(sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
            sql.Identifier(target), sql.Identifier(source)))
        print(f"{target}: a copy of {source}")
EOF
# The copy keeps the schema of its source's day; bring it to today's (idempotent, adds columns
# only, so the copied memory is unchanged). The source database is never touched.
uv run python -m scripts.init_db
SPENT=$(uv run python -c "
import psycopg, sys
with psycopg.connect(sys.argv[1]) as c:
    print(c.execute('SELECT coalesce(sum(cost_usd), 0) FROM usage').fetchone()[0])" "$DATABASE_URL")
export SPEND_CAP_USD=$(python3 -c "print($SPENT + ${RUN_MUSE_BUDGET:-3})")
echo "Spend cap \$$SPEND_CAP_USD: \$$SPENT copied from $FROM's ledger + \$${RUN_MUSE_BUDGET:-3} for judging"

for bench in $BENCHES; do
  uv run python -m eval.run $bench -n "$N" --baseline llama_only --resume "$ID-$bench-llama_only" --yes
done
uv run python - "$ID" $BENCHES <<'EOF'
import json
import sys

from eval.common import RESULTS

run, *benches = sys.argv[1:]
for bench in benches:
    summary = json.loads((RESULTS / f"{bench}_llama_only_{run}-{bench}-llama_only.json").read_text())
    if summary["ingest_pages"]:
        sys.exit(f"{bench}: {summary['ingest_pages']} pages ingested, so this run's memory differs "
                 "from the source run's and its Muse answers can't be reused")
    print(f"{bench}: memory as copied (0 pages ingested)")
EOF
uv run python -m eval.confidence --id "$ID" -n "$N"
if [ "$N" -lt 50 ]; then  # too few items for the gate's five folds
  echo "Smoke run: drafts and probes only. Seconds and tokens per item: results/runs/raw/*_$ID-*"
  exit 0
fi
uv run python -m eval.learn_gate --id "$ID" -n "$N" --muse-id "$FROM"
uv run python -m eval.cost_compare --id "$ID" -n "$N" --muse-id "$FROM" --gpus "$GPUS"
