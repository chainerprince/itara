# Results

Everything measured for Itara, on the real models: Llama 3.1 8B Instruct (vLLM on a Modal H100) and Muse Spark 1.2 (Meta Model API). Nothing in this folder calls a model.

## Headline

| | Code (SWE-bench Lite, 300) | Conversations (LoCoMo, 300) |
|---|---|---|
| Muse with the full context (reference) | 86.0% | 87.0% |
| **Itara, best within 25% of Muse tokens** (budget chosen after the run) | **86.3% at 23%** (every question to Muse with team memory) | 62.7% at 16% |
| Itara, cascade gate at 10% of Muse tokens (the north star, set in advance) | 74.0%: not reached yet | 54.7%: not reached yet |
| Llama alone, with team memory | 55.7% | 38.3% |

- **Memory:** with the team's memory summaries, Muse reads 8.9% of the context on code and loses no accuracy.
- **The gate's signal:** Llama's confidence separates its right answers from wrong ones with AUC 0.68 (code) and 0.80 (conversations).
- **Sharing:** shared memory cuts ingest work per engineer by 35% at 16 engineers, against 13% without sharing.

All of this is explained, with intervals, in the notebook and the paper.

## What's here

| Path | What it is |
|---|---|
| [`full_run.ipynb`](full_run.ipynb) | The analysis. Sections: the north star, cost, where LoCoMo loses, routing headroom, team scaling, the gates, larger budgets and Muse at minimal effort (both chosen after the run). |
| [`bringup.ipynb`](bringup.ipynb) | The checks before the full run: deployment, the scoped KV cache, whether concurrent ingest gives the same memory as sequential ingest, and the memory fixes for dates. Some cells need the check databases (`docker compose start`). |
| `runs/` | Every run's output (below). |
| [`../paper/`](../paper/) | The write-up. `paper/generate.py` builds every table and figure from `runs/`. |

## `runs/`: one file per run

| File | Contents |
|---|---|
| `<bench>_<condition>_<run id>.json` | A run's summary: models, settings, accuracy and token totals. `muse_is_standin` and `simulated` say whether a number may be reported. |
| `raw/<bench>_<condition>_<run id>.jsonl` | One line per item: the answer, its score, the route taken, memory retrieved and exact tokens. |
| `gate_<run id>.json`, `gate_<run id>_sweep.csv` | The gates' out-of-fold replay: λ and threshold sweeps, operating points, verdicts, AUCs (`eval/learn_gate.py`). |
| `gate_<run id>_model.json`, `gate_<run id>_routes_model.json` | The trained cascade and per-route gates: what `itara up --gate` loads. |
| `raw/<bench>_confidence_<run id>.jsonl` | Llama's confidence probes per item (`eval/confidence.py`). |
| `replay_<baseline>_<order>_<timestamp>.csv` | Team replay: per-engineer cost at 1–16 engineers (`rule_gate` = one team sharing; `no_sharing` = everyone alone). |
| `handcheck_locomo_full1.csv` | 50 LoCoMo judge verdicts checked by hand (48 agree). |
| `effort_check_full1-minimal.json` | Intervals for the minimal-effort result (`eval/effort_check.py`). |
| `cost_compare_full1.json` | Dollars (Muse and GPU) and latency per question for each option, including simpler alternatives, RouteLLM's router and team-scale ingest dollars (`eval/cost_compare.py`). |
| `routellm_bert_full1.jsonl` | RouteLLM's BERT router score for each question (`eval/routellm_scores.py`). |

**Run ids:**
- `full1`: the full run. `full1-minimal`: the same with Muse at minimal effort, chosen after the run.
- `big1`: code only, with Llama 3.3 70B (FP8, 2×H100) drafting instead of 8B, on a copy of full1's memory, added after the run. Its gate files take Muse's answers from `full1` (`eval/learn_gate.py --muse-id full1`).
- `check1-*`, `smoke-*`: the ingest checks.
- `dates`, `facts`: the memory fixes.

Only real-model runs are committed. Simulated runs (`sim1`) and stand-in runs stay local: they check the pipeline and are never results.

## Regenerate

```bash
uv run python -m eval.learn_gate --id full1 -n 300      # gates (needs the run's database)
uv run python -m eval.effort_check                      # minimal-effort intervals (same)
uv run python -m eval.cost_compare                      # cost per question (same)
uv run python -m scripts.dashboard_data                 # the dashboard's numbers
uv run python paper/generate.py                         # the paper's tables and figures
uv run jupyter nbconvert --execute --to notebook --inplace results/full_run.ipynb
```

The commands that produce new runs (and spend money) are listed in the top-level README.
