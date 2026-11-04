# Itara

**One open model, one memory and one KV cache for your whole team. The closed model only when it's needed.**

On code, Itara cuts what your team spends on a closed model by **57% at the same accuracy**: 86.3% against 86.0% for sending the full context, with **Llama 3.1 8B** as the team's open model and **Muse Spark** as the closed one. In fast mode it cuts **68%**, and the median answer comes back in **2.3 s** instead of 12.2 s, at 74% accuracy.

Measured on all 300 SWE-bench Lite issues, with the closed model's tokens and the GPU both paid for ([how](paper/itara.pdf)).

| Mode | Spend per 1,000 questions | Accuracy | Median wait |
|---|---|---|---|
| No Itara: Muse with the full context | $37.09 | 86.0% | 12.2 s |
| `itara up` (accurate) | **$16.11** (−57%) | **86.3%** | 9.8 s |
| `itara up --mode fast` | **$11.87** (−68%) | 74.0% | **2.3 s** |

**A bigger open model makes it better.** With Llama 3.3 70B drafting instead of the 8B (same questions, same memory, same Muse answers; tested after the run):

| Code accuracy | Llama 3.1 8B | Llama 3.3 70B |
|---|---|---|
| Llama alone, with team memory | 55.7% | **67.0%** (+11.3 points, 95% interval +6.7 to +16.0) |
| Fast mode, 10% of Muse's tokens | 74.0% | **77.7%** (+3.7, interval −0.7 to +8.0) |
| 20% of Muse's tokens | 82.7% | **85.7%**: within 0.3 points of Muse with the full context |

The 70B runs on two H100s, so it costs more GPU time per question. A simpler gate that adds a second opinion (does the 8B name the same file?) takes the 70B's fast mode to 81.0% at 10%. That gate was designed after the run, so it's the next step rather than a result ([paper](paper/itara.pdf), `eval/simple_gate.py`).

Itara is a self-hosted gateway that speaks the OpenAI API: any app, chat UI or coding agent uses it by changing its base URL.

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

1. **Shared team memory.** The open model compresses the team's code and docs once, into short summaries that everyone on the team searches (by meaning and by keyword). The closed model reads these instead of raw files: **under 9%** of the context, with no loss of accuracy. Postgres row-level security keeps each team's memory to itself.
2. **A KV cache keyed by permission.** When a teammate asks about the same code, Llama reuses the attention state someone else's request already computed. The cache key includes the team, so other teams never hit it and can't time it. In our test a teammate reused **6,144** cached tokens; someone on another team reused **0**.
3. **A confidence cascade** (fast mode). Llama drafts an answer and scores it with its own token probabilities, four resamples and a one-token self-check. Only unsure drafts go to the closed model, so most questions never leave your GPU.
4. **An exact ledger with a hard cap.** Every token is counted per person and per team, and a spending cap is checked before every closed-model call.

## Why not something simpler

Measured on the same questions:

- **Instead of sending the full context:** 57% cheaper at the same accuracy, and answers come back sooner.
- **Instead of an off-the-shelf router:** RouteLLM decides from the question alone and can barely tell when Llama will be wrong (AUC 0.53, where 0.5 is chance). Itara's confidence signal gets 0.65. On a tight budget (10% of the closed model's tokens), that is **74.0% against 68.3%**.
- **Instead of per-person memory tools:** one memory and one cache for the whole team. At 16 engineers, the work per engineer to build memory falls **35%**, against 13% when each works alone, and teams stay isolated.
- **Instead of a hosted gateway:** it runs on your hardware, with your open model, and plugs into anything that speaks the OpenAI API.

Where simpler is enough: on code, Muse with raw retrieved pages costs only a little more than team memory, because most of a reasoning model's bill is its own output. With a budget of 20% of the tokens, RouteLLM matches the cascade for less. On long conversations, memory search misses the right session a third of the time, and Itara reaches 62.7% against 87.0%. All of this is in [the paper](paper/itara.pdf).

## Install

**Needs:** Python 3.12+, [uv](https://docs.astral.sh/uv/) and Docker (for Postgres), plus two OpenAI-compatible model endpoints:
- **an open-weight model:** vLLM on [Modal](modal_app.py), vLLM on your own GPU, or Ollama on a laptop;
- **a closed model to escalate to:** Muse Spark through the Meta Model API, or a larger local model standing in.

1. **Clone it:** `git clone https://github.com/chainerprince/itara.git itara && cd itara`
2. **Install the `itara` command:** `uv sync`
3. **Point it at your models:** `cp .env.example .env`, then set `LLAMA_*` and `MUSE_*` (the comments explain each).
4. **Start it:** `uv run itara up`. This starts Postgres + pgvector in Docker, applies the schema, and runs the gateway on :8000.

PyPI (`pip install itara`) is next on the roadmap. To serve Llama on Modal, follow the steps at the top of [`modal_app.py`](modal_app.py), then set `LLAMA_BASE_URL` to the printed URL and `LLAMA_BACKEND=vllm`.

## Use

```bash
uv run itara up                         # accurate: every question to Muse with team memory
uv run itara up --mode fast             # fast: Llama answers what it's sure of (the trained cascade)

uv run itara key add --team web --user alice           # one key per person; only its hash is stored
uv run itara key add --team web --user you --admin     # admins see usage and the dashboard
uv run itara memory add src/ docs/ --key mg-…          # the team's code and docs, compressed once
uv run itara dashboard                                 # opens http://localhost:8000/dashboard
uv run itara status --key mg-…                         # who asked what, and where it went
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

To draft with Llama 3.3 70B, deploy it with `MODEL=RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic GPU=H100:2 MODAL_APP=itara-llama-70b modal deploy modal_app.py`, point `LLAMA_*` at it, and start `itara up --gate results/runs/gate_big1_model.json`.

Fast mode uses the cascade trained on the full run (`results/runs/gate_full1_model.json`) at its 10% operating point on code. `GATE_THRESHOLD` moves it between cheaper (lower) and more accurate (higher). `--gate <file>` loads another gate at its own saved threshold.

## The dashboard

The gateway serves a dashboard at `/dashboard`. **Live** shows each person, their team's memory and KV cache, and the models every team shares. Each request lights up the path it took: green when Llama answered, violet when it went to Muse. It also shows questions kept on Llama, the share of prompts served from the team cache, and spend against the cap. **Results** shows the benchmarks behind the numbers above.

The dashboard is a small Vite + React + TypeScript app in [`dashboard/`](dashboard/). The built files (`dashboard/dist`) are committed, so Itara runs without Node. To change it:

```bash
cd dashboard && npm install
npm run dev      # http://localhost:5173/dashboard/, talking to the gateway on :8000
npm run build    # rebuilds dashboard/dist, which the gateway serves
```

## The research behind it

Before building Itara we set a north star: **match the closed model's accuracy while sending it at most 10% of its tokens.** We haven't reached it yet. At 10%, Itara gets 74% on code against 86%. The measurements point to the way there: a router that knew every answer in advance would pass (90.3% at 8.8%), and the open model is the main limit. Swapping the 8B for a 70B already moves fast mode from 74.0% to 77.7%. Most of the results above come from measuring that goal. The goal, every result, the comparisons and their limits are in the paper:

- **[`paper/itara.pdf`](paper/itara.pdf):** the system, the evaluation on SWE-bench Lite and LoCoMo, the gate, costs, team scaling, and what we chose after seeing results.
- **[`results/`](results/):** every real run's per-item answers, and the notebooks that analyse them. Nothing there calls a model.

### Reproduce the results

| What | Command | Calls models? |
|---|---|---|
| Check the pipeline on simulated models | `scripts/sim_full.sh` | No |
| Full run: every condition, both benchmarks | `scripts/full_run.sh` | Yes (Modal + Muse) |
| Llama's confidence probes and the gates | `scripts/gate_run.sh` | Llama only |
| Muse at minimal reasoning effort | `scripts/effort_run.sh` | Yes (Muse) |
| Intervals for the minimal-effort result | `uv run python -m eval.effort_check` | No |
| Cost per question, and the comparison with simpler options | `uv run python -m eval.cost_compare` | No |
| RouteLLM's router scores | `eval/routellm_scores.py` (see its docstring) | Its router, on the CPU |
| Dashboard numbers | `uv run python -m scripts.dashboard_data` | No |
| Tests | `uv run pytest` (needs `docker compose up -d`) | No |

## Roadmap

- **Reach the north star.** A 70B open model already moves fast mode from 74.0% to 77.7% at 10%. Next: train the gate on more questions, try larger open models, and improve retrieval for conversations.
- **`pip install itara`.** Publish to PyPI, with the dashboard and schema shipped inside the package.
- **A one-command self-host.** A Docker Compose file that runs the gateway alongside Postgres.
- **Better retrieval for conversations.** It is what limits LoCoMo.
- **Improve your own open-weight model.** Every escalation pairs Llama's draft with the closed model's answer on the team's own questions. That is training data for fine-tuning the team's model, so it needs the closed model less over time. Not built yet.
- **An MCP server for coding agents, and LMCache to share the KV cache across several GPU servers.**

## Credits

Itara was built by Prince Mutegetsi, with Claude Opus 5.5, during the Meta Global AI Developer Hackathon (26 Oct – 4 Nov 2026; Indie Developer track, Push the Frontier). The only earlier code it uses is about 20 lines of a compression prompt from Prince's [ai-os](https://github.com/chainerprince/ai-os); [`DISCLOSURE.md`](DISCLOSURE.md) lists what was built during the event.

Built with Llama.
