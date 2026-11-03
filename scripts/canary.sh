#!/usr/bin/env bash
# Real-model canary on the laptop: how the local stand-ins actually behave (JSON compliance,
# answer parsing, truncation, rough accuracy) on a few items, gently.
#
#   docker compose start && scripts/canary.sh        # CANARY_ID / CANARY_N to override
#
# - Every model response is cached (MODEL_CACHE), so a rerun or a resumed phase costs nothing
#   for calls already made. Ctrl-C at any time; rerun the script to continue.
# - Phases are ordered by model so Ollama loads each model about once (one model in memory
#   at a time): ingest + llama_only on the 3B, then the escalations on the 8B. rule_gate comes
#   last and is answered almost entirely from the cache.
# - LoCoMo is ingest-only here (its judge would swap models on every item); LoCoMo answers and
#   team replay are covered by the simulator until vLLM on Modal.
# - Runs a private Ollama on port 11435 with a 16K context, and stops it when done.
set -euo pipefail
cd "$(dirname "$0")/.."

ID=${CANARY_ID:-canary1}
N=${CANARY_N:-20}
PORT=11435
export LLAMA_BASE_URL=http://localhost:$PORT/v1 MUSE_BASE_URL=http://localhost:$PORT/v1
export MODEL_CACHE=eval/data/model_cache.sqlite
# Its own database (kept between runs so phases can resume).
export DATABASE_URL=${CANARY_DATABASE_URL:-postgresql://postgres:postgres@localhost:5432/gateway_canary}
uv run python -m scripts.init_db --create

OLLAMA_HOST=127.0.0.1:$PORT OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=5m ollama serve > eval/data/canary_ollama.log 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null' EXIT
until curl -s localhost:$PORT/api/version > /dev/null; do sleep 1; done

run() { caffeinate -i nice -n 10 uv run python -m eval.run "$@"; }
SWE=(swebench -n "$N" --repo django/django)

echo "== 1/5 ingest (local 3B model)"
run "${SWE[@]}" --ingest-only --resume "$ID-swebench-ingest"
run locomo -n "$N" --ingest-only --resume "$ID-locomo-ingest"

echo "== 2/5 llama_only (3B)"
run "${SWE[@]}" --baseline llama_only --resume "$ID-llama_only"

echo "== 3/5 escalate_compact (8B)"
run "${SWE[@]}" --baseline escalate_compact --resume "$ID-escalate_compact" --yes

echo "== 4/5 muse_direct (8B, ~2 min per item)"
run "${SWE[@]}" --baseline muse_direct --resume "$ID-muse_direct" --yes

echo "== 5/5 rule_gate (mostly cached)"
run "${SWE[@]}" --baseline rule_gate --resume "$ID-rule_gate" --yes

uv run python -m eval.checks "$ID-muse_direct" "$ID-escalate_compact" "$ID-llama_only" \
  "$ID-rule_gate" || true
echo "Done. Per-run summaries: results/runs/swebench_*_$ID-*.json."
