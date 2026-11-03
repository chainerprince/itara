# Itara

**One open-weight model, one memory and one KV cache for a whole team. The closed model only when it's needed.**

Every engineer on a team sends the same codebase, docs and decisions to an expensive closed model, over and over. The company pays for the same context once per person, per question. Itara is a self-hosted gateway that a whole team shares:

- An open-weight model (Llama) answers what it can.
- The team's knowledge is compressed once into a memory everyone can read.
- Work one person's request already did is cached, so teammates reuse it instead of recomputing it.
- The closed model (Muse Spark) is called only when Llama isn't sure, and gets compact memory instead of raw files.

Itara speaks the OpenAI API, so any app, chat UI or coding agent uses it by changing its base URL.

## What it achieves

Measured on the real models (Llama 3.1 8B and Muse Spark 1.2), on all 300 SWE-bench Lite issues and 300 LoCoMo questions:

- **On code, Itara matches the closed model with about a quarter of its tokens.** Muse given the team's memory scores 86.3% against 86.0% with the full context, while reading under 9% of that context.
- **Llama knows when it's unsure.** Its own confidence separates its right answers from its wrong ones (AUC 0.68 on code, 0.80 on conversations), and the gate it drives beats escalating at random by 5 points.
- **Sharing scales with the team.** At 16 engineers, shared memory cuts the work per engineer by 35%, against 13% without sharing.
- **Teams stay isolated.** A teammate reused 6,144 cached prompt tokens; someone on another team reused none, by design.

## The north star

Before running anything, we set a more ambitious goal:

> For a team sharing one codebase, Itara matches the accuracy of sending context straight to the closed model, while sending it **≤ 10% of the tokens**. It knows when not to escalate, and the shared cache makes each extra teammate cheaper.

We haven't reached it yet: at 10% of the tokens, Itara gets 74% on code against Muse's 86%. It is still what Itara is built toward, and the results point to a way there: see [the path to the north star](#the-path-to-the-north-star).

## How it works

```
 any app · chat UI · coding agent
        │  OpenAI-compatible API · identity from the API key
        ▼
┌──────────────────────────── Itara (self-hosted) ─────────────────────────────┐
│  Llama on vLLM ── shared KV cache, keyed by permission (cache_salt per team) │
│        │  the cascade gate: Llama drafts; unsure drafts escalate             │
│        ▼                                                                     │
│  shared memory: Postgres + pgvector, row-level security per team             │
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
                       Muse Spark (closed), with compact memory
```

| Layer | Cost it removes |
|---|---|
| **Llama gateway** (vLLM; Modal or your own GPU) | Closed-model calls: it answers the questions it's sure of. |
| **Shared memory** (Postgres + pgvector) | Closed-model tokens: escalations carry the team's summaries, not raw files. Hybrid vector and keyword search. |
| **Shared KV cache** (vLLM prefix cache, scoped per team) | GPU time: a teammate's request loads the prefix someone already computed. Other teams never hit it, so its timing can't leak their content. |
| **The cascade gate** | Wasted escalations. Llama scores its own draft (token probabilities, 4 resamples, a self-check), and only unsure drafts go to Muse. |
| **Ledger and hard cap** | Surprise bills. Exact token counts per person and team, and a spending cap checked before every closed-model call. |

## Results

Measured on the real models: Llama 3.1 8B on vLLM (a Modal H100) and Muse Spark 1.2 (Meta Model API). The test sets were all 300 SWE-bench Lite issues (name the file to change, scored without a judge) and 300 LoCoMo questions about long conversations, judged by Muse; 48 of 50 judge verdicts agreed with a hand check. Gates are scored out of fold, on repos and conversations they didn't train on. The details are in [`results/full_run.ipynb`](results/full_run.ipynb).

**On code, Itara already matches the closed model, at about a quarter of the tokens.**

| SWE-bench Lite (300 issues) | Accuracy | Muse tokens |
|---|---|---|
| Muse with the full context (the reference) | 86.0% | 100% |
| **Itara: every question to Muse with team memory** | **86.3%** | **23%** (about a quarter; 36% of the dollars) |
| Itara, cascade gate at 20% of the tokens | 82.7% | 20% |
| Itara, cascade gate at 10% (the north star, not reached yet) | 74.0% | 10% |
| Llama alone with team memory | 55.7% | 0% |

- **Memory carries most of the saving.** With the team's memory summaries, Muse reads **under 9% of the context** it would otherwise get and loses no accuracy. What remains of the bill is mostly Muse's own reasoning, about 2,800 tokens per escalation.
- **The gate's signal is real.** Llama's confidence separates its right drafts from wrong ones with an AUC of 0.68 on code and 0.80 on conversations. At 10% of the tokens, the cascade beats escalating at random by 5 points on code.
- **What was decided after the run.** The 10% north star is the only budget set in advance; the 20% and 25% budgets were chosen after seeing the results. A follow-up at Muse's minimal reasoning effort, also chosen after the run, comes within about 2 points on code at 10% of the tokens, but its 95% interval includes misses of up to 5 points (section 10 of the notebook).

**Long conversations are harder, and the reason is retrieval.** Itara reaches 62.7% against Muse's 87.0% with the full conversation. Its memory search misses the right session for 37% of questions. When it finds that session, Muse with the raw pages scores 85.3%.

**Sharing pays off as the team grows.** In replays of 1 to 16 engineers on one repo, the gateway's ingest work per engineer falls **35%** with shared memory at 16 engineers, against 13% without sharing (related tasks; on random tasks, 7% vs 1%). A teammate's request reused **6,144** cached prompt tokens; someone on another team reused **0**.

## The path to the north star

These runs used an 8B open model on one GPU over nine days. The numbers suggest the goal is within reach with a bigger open model, more GPUs and more time:

- **The goal is reachable in principle.** An oracle router, which knows each answer in advance, scores 90.3% on code at 8.8% of Muse's tokens: inside the target.
- **The open model is the limit.** Llama 8B's draft is right on only 56% of issues. Even a perfect check that escalated exactly its wrong drafts would use 12.1% of the tokens, just over the target. If a bigger open model (70B-class) got 70% right, the same check would use about 8%. That is a rough estimate, but it lands inside the target. Bigger models in the same family score much higher, and they judge their own answers better, which is the signal the gate relies on.
- **The gate needs more data.** It learned from about 480 questions at a time. A deployed Itara collects questions as a matter of course.
- **Cheaper escalations already help.** At Muse's minimal reasoning effort, Itara came within about 2 points on code at 10% of the tokens (chosen after the run, and the interval allows misses of up to 5 points).

A 70B open model needs at least two H100s, which was beyond this project's budget. It is the first experiment on the roadmap. For conversations, a bigger model isn't enough: retrieval has to improve first. These are expectations to test, not results.

## Install

Itara installs from GitHub for now (PyPI is on the roadmap).

**Needs:** Python 3.12+, [uv](https://docs.astral.sh/uv/) and Docker (for Postgres). It also needs two OpenAI-compatible model endpoints:
- **an open-weight model:** vLLM on [Modal](modal_app.py), vLLM on your own GPU, or Ollama on a laptop;
- **a closed model to escalate to:** Muse Spark through the Meta Model API, or a larger local model standing in.

```bash
git clone https://github.com/chainerprince/itara.git itara && cd itara
uv sync                         # installs the `itara` command
cp .env.example .env            # point LLAMA_* and MUSE_* at your models (see the comments)
uv run itara up                 # Postgres + pgvector in Docker, the schema, the gateway on :8000
```

To serve Llama on Modal instead of locally, follow the steps at the top of [`modal_app.py`](modal_app.py). Then set `LLAMA_BASE_URL` to the printed URL and `LLAMA_BACKEND=vllm`.

To use the learned gate rather than the rule-based one, train it with `scripts/gate_run.sh` (or use a saved one), then:

```bash
uv run itara up --gate results/runs/gate_full1_model.json
```

## Use

```bash
uv run itara key add --team web --user alice           # one key per person; only its hash is stored
uv run itara key add --team web --user you --admin     # admins see usage and the dashboard
uv run itara memory add src/ docs/ --key mg-…          # the team's code and docs, compressed once
uv run itara dashboard                                 # opens http://localhost:8000/dashboard
uv run itara status --key mg-…                         # the same, in the terminal
uv run itara demo                                      # two teammates share memory; another team can't
```

Any OpenAI client then points at Itara:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="mg-…")  # the person's Itara key
reply = client.chat.completions.create(
    model="itara", messages=[{"role": "user", "content": "Where do we enforce the spend cap?"}])
```

Each response says where it went in its `X-Gateway-Route` header: `local_memory` (Llama), `escalate_compact` or `escalate_raw` (Muse with summaries or raw pages), or `passthrough` (requests with tool calls go to Muse unchanged).

## The dashboard

The gateway serves the dashboard at `/dashboard`, so there's nothing extra to run. It has two pages:

- **Live** (needs an admin key):
  - **Topology:** each person, their team's memory and KV cache (sealed by row-level security and the team's own cache key), and the models every team shares. Each new request lights up the path it took: green when Llama answered, violet when it escalated to Muse, dark when memory was added. Edges show the cached tokens each person reused from work their teammates already paid for.
  - **Totals:** questions kept on Llama, the share of Llama's prompt served from the team cache, and Muse spend against the cap.
  - **The latest requests.**
- **Results:** the north-star verdict, accuracy against Muse tokens for each benchmark, the best accuracy at each budget, and team scaling. They come from [`dashboard/public/results.json`](dashboard/public/results.json), which `scripts/dashboard_data.py` writes.

The dashboard is a small Vite + React + TypeScript app in [`dashboard/`](dashboard/). The built files (`dashboard/dist`) are committed, so Itara runs without Node. To change it:

```bash
cd dashboard && npm install
npm run dev      # http://localhost:5173/dashboard/, talking to the gateway on :8000
npm run build    # rebuilds dashboard/dist, which the gateway serves
```

## Reproduce the results

| What | Command | Calls models? |
|---|---|---|
| Check the pipeline on simulated models | `scripts/sim_full.sh` | No |
| Full run: every condition, both benchmarks | `scripts/full_run.sh` | Yes (Modal + Muse) |
| Llama's confidence probes and the gates | `scripts/gate_run.sh` | Llama only |
| Muse at minimal reasoning effort | `scripts/effort_run.sh` | Yes (Muse) |
| Intervals for the minimal-effort result | `uv run python -m eval.effort_check` | No |
| Dashboard numbers | `uv run python -m scripts.dashboard_data` | No |
| Tests | `uv run pytest` (needs `docker compose up -d`) | No |

Everything measured is in [`results/`](results/): the analysis notebooks, and every real run's per-item answers and summaries in `results/runs/`. The notebooks call no model. [`paper/`](paper/) has the write-up.

## Roadmap

- **Reach the north star.** Rerun the evaluation with a 70B-class open model on more GPUs, train the gate on more questions, and improve retrieval for conversations.
- **`pip install itara`.** Publish to PyPI, with the dashboard and schema shipped inside the package.
- **A one-command self-host.** A Docker Compose file that runs the gateway alongside Postgres.
- **Better retrieval for conversations.** It is what limits LoCoMo.
- **Improve your own open-weight model.** Every escalation pairs Llama's draft with the closed model's answer on the team's own questions. That is training data for fine-tuning the team's model, so it needs the closed model less over time. Not built yet.
- **An MCP server for coding agents, and LMCache to share the KV cache across several GPU servers.**

## Status

Itara was built during the Meta Global AI Developer Hackathon (26 Oct – 4 Nov 2026), under the working name Memory Gateway. [`DISCLOSURE.md`](DISCLOSURE.md) separates the code reused from an earlier project from the code written during the event. Teams in the results are simulated, and only one model family was tested.
