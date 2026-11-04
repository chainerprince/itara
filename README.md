# Itara

**Shared repository context for your team's AI tools.**

Itara is a self-hosted gateway that speaks the OpenAI API. Ingest your team's code and docs once, and every teammate's chat app, script or coding agent gets that context with each request, scoped to the team by Postgres row-level security. An open model on your own GPU (anything vLLM or Ollama serves) builds the memory and can answer on its own. **Muse Spark** answers what it can't.

**Measured on all 300 SWE-bench Lite issues**, as file localization: the model names up to five files, and an answer counts when every file the real fix changed is among them. Costs are estimated from exact tokens at Muse's list prices plus H100 time at Modal's rate, memory build included ([paper](paper/itara.pdf)).

| | Localization success | Est. cost per 1,000 questions | Median wait |
|---|---|---|---|
| Muse with the issue alone, no context | 81.3% (244/300) | $20.64 | |
| Muse with the benchmark's full retrieved context (no Itara) | 86.0% (258/300) | $37.09 | 12.2 s |
| **`itara up`**: Muse with the team's memory | **86.3% (259/300)** | **$16.11** (−57%) | 9.8 s |
| **`itara up --mode fast`**: Llama 3.1 8B answers what it's sure of, run live | 72.7% (218/300) | **$11.34** (−69%) | **1.9 s** |
| **`itara up --mode local`**: Muse Glimmer 30B, nothing sent to Muse | 76.0% | GPU only | 6.1 s |

- **Comparable accuracy at 57% lower estimated cost.** Muse reads the team's compact memory instead of 13K tokens of retrieved code. Without the two reference answers that ran to Muse's 262K-token output limit, the saving is 46% ($15.98 against $29.75). Itara caps output by default, so it would have stopped both.
- **The memory adds accuracy and saves money.** Muse already knows these public repositories well: from the issue text alone it localizes 81.3%. The team's memory adds 5.0 points (95% interval +2.0 to +8.0). It is also cheaper than no context at all ($16.11 against $20.64), because Muse reasons less when it has what it needs. On private code, where the model knows nothing in advance, memory is the only source.
- **Summaries match plain retrieval and send a third of the prompt.** Searching the raw files directly, with no summaries, gets 85.3% at $15.79 per 1,000. That is about the same as summaries once their one-time build is charged to these 300 questions. Summaries send Muse 1,004 prompt tokens instead of 3,027 and cut its bill by 16%, so the build pays for itself after about 340 questions on the same memory.
- **Fast mode, measured live in the gateway:** 72.7% of issues, with a median wait of 1.9 s end to end and a 69% lower estimated cost. Llama keeps 63% of questions on your GPU; the rest go to Muse with the team's memory and take as long as Muse does (90th percentile 32 s).

**Status:** an experimental prototype built in nine days for the Meta AI Developer Hackathon. [What's built, measured and planned](#status-built-measured-planned) is spelled out below.

## Install

**Needs:** Python 3.12+, [uv](https://docs.astral.sh/uv/) and Docker (for Postgres), plus two OpenAI-compatible model endpoints:
- **an open-weight model:** vLLM on [Modal](modal_app.py), vLLM on your own GPU, or Ollama on a laptop;
- **a closed model to escalate to:** Muse Spark through the Meta Model API, or a larger local model standing in.

```bash
git clone https://github.com/chainerprince/itara.git itara && cd itara
uv sync                       # installs the itara command
cp .env.example .env          # then set LLAMA_* and MUSE_* (the comments explain each)
uv run itara up               # Postgres + pgvector in Docker, the schema, the gateway on :8000
```

Then give each person a key, add the repository, and ask:

```bash
uv run itara key add --team web --user alice          # one key per person; only its hash is stored
uv run itara memory add src/ docs/ --key mg-…         # compressed once for the whole team
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="mg-…")  # alice's Itara key
reply = client.chat.completions.create(
    model="itara", messages=[{"role": "user", "content": "Where do we enforce the spend cap?"}])
```

`uv run itara demo` runs the whole story against your gateway. Alice adds the team's code and notes, and Bob asks about a release decision that exists only in those notes. A coding agent with tools attached gets the same memory. Olga, on another team, gets none of it. To serve Llama on Modal, follow the steps at the top of [`modal_app.py`](modal_app.py), then set `LLAMA_BASE_URL` to the printed URL and `LLAMA_BACKEND=vllm`.

## Modes

| Mode | Plain questions | Requests with tools (coding agents) | Sent to Muse |
|---|---|---|---|
| `itara up` (accurate, the default) | Muse + the team's memory | Muse + the team's memory | everything |

When Muse gets the team's memory, code comes as summaries and docs and conversations as the raw pages behind them. Summaries of dense prose can drop a date or a threshold, and each choice is the one that measured best: summaries on code, raw pages on conversations.
| `itara up --mode fast` | the open model drafts with memory; a confidence gate keeps the draft or escalates to Muse + memory | Muse + the team's memory | unsure questions and tool calls |
| `itara up --mode local` | the open model + memory | the open model + memory | **nothing** |
| `itara up --mode rule` | a hand-written rule picks the model | Muse + the team's memory | long or code-heavy questions, tool calls |

Requests with tools skip the gate, which learned to judge plain answers, not tool calls. They still get the team's memory, which is prepended as a system message; their tools and messages go on unchanged. A tool request that matches no memory passes through untouched. In local mode the open model's server must support tool calling: `modal_app.py` enables vLLM's tool parser for Llama models.

Each response says where it went in its `X-Gateway-Route` header (`local_memory`, `escalate_compact`, `passthrough`, …) and how many memories it drew on in `X-Gateway-Memory-Hits`. Admins can force a route with `X-Gateway-Route`; `escalate_pages` (Muse with raw pages searched directly, no summaries) exists for comparisons like the one above.

Fast mode uses the cascade trained on the full run (`results/runs/gate_full1_model.json`) at the threshold of its 10%-of-tokens operating point. `GATE_THRESHOLD` moves it between cheaper (lower) and more accurate (higher). `--gate <file>` loads another gate at its own saved threshold.

## How it works

```
 chat apps · scripts · coding agents  (OpenAI API, one key per person)
        │
        ▼
┌──────────────────────────── Itara (self-hosted) ─────────────────────────────┐
│  team memory: Postgres + pgvector, row-level security per team               │
│     summaries + the raw pages behind them · hybrid search (vectors + words)  │
│  open model on vLLM: builds the memory; drafts in fast mode, answers in      │
│     local mode · prefix cache keyed by team (cache_salt)                     │
│  ledger: exact tokens per person · output limit · hard spend cap             │
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        ▼
                         Muse Spark, with the team's memory
```

1. **Team memory, kept current.** `itara memory add` splits each file into pages. The open model writes a short summary of each, and Postgres keeps the summary, the raw page and their embeddings. A page the team already has is never compressed again: adding the same repo twice, or two teammates adding it at once, costs one compression. A changed file replaces its old version. `itara memory rm` forgets a source. Row-level security keeps each team's memory, and each person's private memory, to them.
2. **A prefix cache keyed by permission.** Memory goes first, in a stable order, so teammates asking about the same code share a byte-identical prefix and reuse each other's prefill on vLLM. The cache key includes the team, so other teams never hit it and can't time it. In our test a teammate reused **6,144** cached tokens; someone on another team reused **0**.
3. **A confidence cascade** (fast mode). The open model drafts an answer and scores it with its own token probabilities, four resamples and a one-token self-check. Only unsure drafts go to Muse.
4. **An exact ledger with a hard cap.** Every token is counted per person and per team. Each Muse call is limited to `MUSE_MAX_OUTPUT_TOKENS` (32,768 by default; callers can ask for less). Its worst case is reserved against `SPEND_CAP_USD` before it is sent: every prompt byte counted as a token, times `n` choices.

## Why not something simpler

Measured on the same 300 issues, with the same memory:

- **Instead of sending the full context:** comparable accuracy at 57% lower estimated cost, and a shorter wait.
- **Instead of plain retrieval over raw files:** the same accuracy (86.3% against 85.3%, within noise), a third of Muse's prompt, and 16% less Muse spend per question. Plain retrieval needs no build, so it is cheaper for memory that gets asked fewer than about 340 questions.
- **Instead of an off-the-shelf router:** we compared one router, RouteLLM's BERT checkpoint used as shipped. It decides from the question alone and barely predicts when Llama will be wrong (AUC 0.53, where 0.5 is chance), against 0.65 for Itara's confidence signal. At 10% of Muse's tokens that is **74.0% against 68.3%**. At 20% they are level, and the router is cheaper per correct answer.
- **Instead of per-person memory tools:** one memory and one cache for the whole team. In a replay of up to 16 simulated engineers, the work per engineer to build memory falls **35%** at 16 engineers, against 13% when each works alone.
- **Instead of a hosted gateway:** it runs on your hardware, with your open model.

Where simpler is enough: provider prompt caching. Muse's API bills cached prompt tokens at $0.15 per million instead of $1.25. On code each issue's context differs, and Muse served 0% of it from its cache. On LoCoMo conversations, many questions share one long transcript: 93% was cached, and full context would cost about $8.58 per 1,000 instead of $27.22. Itara costs $7.30 there, sending Muse the raw pages behind the retrieved memory (62.7%), but full context is far more accurate (87.0%), because retrieval misses the right session a third of the time.

## Status: built, measured, planned

| | |
|---|---|
| **Built and tested** | The OpenAI-compatible gateway: chat completions, tool calls, and streaming as one chunk. Per-person keys. Team and private memory under row-level security. Ingest that compresses each page once, replaces changed files and forgets sources. The prefix cache keyed by team. Accurate, fast, local and rule modes. The ledger, output limit and spend cap. The CLI and dashboard. 85 tests. |
| **Measured on the real models** | SWE-bench Lite file localization (300 issues) and LoCoMo conversations (300 questions). Every condition above. Fast mode live in the gateway. Llama 3.3 70B and Muse Glimmer 30B drafting. Cache reuse across teammates and teams. Team scaling with simulated engineers. |
| **Planned, not yet done** | A coding agent's full tool loop, measured end to end. Token-by-token streaming. A spend cap shared across several workers. KV cache shared across servers (LMCache). Retrieval across several branches of one repo. `pip install itara`. Fine-tuning the open model from escalations. |

**Know before you deploy:**
- Run one worker: the spend reservation lives in the process.
- The gate learned from file-localization and conversation answers. For other work, such as design advice or long structured outputs, use accurate or local mode, or retrain the gate on your own traffic (`scripts/open_model_run.sh`). The "10% of tokens" figure is where the threshold landed on the benchmark; the live gate doesn't enforce a ratio.
- Retrieved memory goes into a system message: anyone who can add memory for a team can influence its answers. Add only content you'd trust in a prompt.
- Pages are cut by size and blank lines, not by syntax. Memory keeps one version per file per team (the latest added). Permissions are per team or per person, not per document.
- The benchmark measures finding the right files, not writing the fix. Benchmark memory pools each repository's files across all its sampled issues; the paper discusses what that changes.

## Choose your open model

A stronger open model makes fast and local modes better. We measured three, each drafting the same questions with the same memory and the same Muse answers (the 70B and Glimmer after the main run; fast mode replayed offline from saved drafts):

| Code | Llama 3.1 8B | Llama 3.3 70B | Muse Glimmer 30B |
|---|---|---|---|
| The open model alone, with team memory | 55.7% | 67.0% (+11.3 points, 95% interval +6.7 to +16.0) | **76.0%** (+20.3, interval +14.7 to +26.0) |
| Fast mode at 10% of Muse's tokens (offline replay) | 74.0% | 77.7% (+3.7, interval −0.7 to +8.0) | **80.3%** at 6.8% of the tokens (+6.3, interval +1.7 to +11.0) |
| 20% of Muse's tokens (offline replay) | 82.7% | **85.7%** | 84.3% |
| GPUs | one H100 | two H100s (FP8) | one H100 |

| You want | Run | Open model |
|---|---|---|
| Muse's accuracy for less | `itara up` | **Llama 3.1 8B.** Muse answers every question; the open model only builds the memory. |
| The fastest, cheapest answers | `itara up --mode fast` | **Llama 3.1 8B.** It drafts in 1.0 s at the median on one GPU. |
| Nothing sent to the closed model | `itara up --mode local` | **Muse Glimmer 30B.** 76.0% on its own, with drafts in 6.1 s at the median. |

Glimmer drafts best, but it reasons before it answers, which makes its fast mode slower and costlier than accurate mode. On conversations no open model helps much (Glimmer 38.7%, the 8B 38.3%): retrieval is the limit there. A simpler gate that adds a second opinion (does the 8B name the same file?) takes the 70B's fast mode to 81.0% at 10%. It was designed after the run, so it's the next step rather than a result.

**To switch models**, deploy one with [`modal_app.py`](modal_app.py), point `LLAMA_*` in `.env` at it, and start Itara with the gate trained for that model:

- **Llama 3.3 70B:** `MODEL=RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic GPU=H100:2 MODAL_APP=itara-llama-70b modal deploy modal_app.py`, then `itara up --gate results/runs/gate_big1_model.json`.
- **Muse Glimmer 30B:** `MODEL=meta-models/Muse-Glimmer-30B MODAL_APP=itara-glimmer modal deploy modal_app.py` (after `modal run modal_app.py::download` with the same variables) and `LLAMA_REASONING_STRENGTH=low` in `.env`, then `itara up --mode local`, or `itara up --gate results/runs/gate_glim1_model.json` for fast mode.
- **Another model:** accurate and local modes work as they are. Fast mode needs a gate trained on that model's drafts: `scripts/open_model_check.py` checks its endpoint, and `scripts/open_model_run.sh` measures it and trains its gate.

## The dashboard

The gateway serves a dashboard at `/dashboard` (`uv run itara dashboard`; it asks for an admin key). **Live** shows each person, their team's memory and cache, and the models every team shares. Each request lights up the path it took: green when the open model answered, violet when it went to Muse. It also shows questions kept local, the share of prompts served from the team cache, and spend against the cap. **Results** shows the benchmarks behind the numbers above. The built files are committed, so Itara runs without Node; to change it, `cd dashboard && npm install && npm run dev`.

## The research behind it

Before building Itara we set a north star: **match the closed model's accuracy while sending it at most 10% of its tokens.** We haven't reached it yet: at 10%, fast mode with the 8B gets 74.0% replayed offline and 72.7% live, against 86%. A router that knew every answer in advance would pass (90.3% at 8.8%), so the open model and the judge of its drafts are the limits. Stronger open models already move fast mode to 77.7% (Llama 3.3 70B) and 80.3% (Glimmer). The goal, every result, the comparisons and their limits are in the paper:

- **[`paper/itara.pdf`](paper/itara.pdf):** the system, the evaluation on SWE-bench Lite and LoCoMo, the gate, costs, team scaling, and what we chose after seeing results.
- **[`results/`](results/):** every real run's per-item answers, and the notebooks that analyse them. Nothing there calls a model.

### Reproduce the results

| What | Command | Calls models? |
|---|---|---|
| Check the pipeline on simulated models | `scripts/sim_full.sh` | No |
| Full run: every condition, both benchmarks | `scripts/full_run.sh` | Yes (Modal + Muse) |
| The checks added on 4 Nov: question only, raw pages searched directly | `uv run python -m eval.run swebench -n 300 --baseline muse_question_only` (or `escalate_pages`) `--resume full1-swebench-<baseline> --concurrency 8 --yes` on the full run's database | Muse |
| Fast mode live in the gateway | `GATE_PATH=results/runs/gate_full1_model.json GATE_THRESHOLD=0.4568 uv run python -m eval.run swebench -n 300 --baseline full --resume live2-swebench-full --concurrency 6 --yes` | Modal + Muse |
| Llama's confidence probes and the gates | `scripts/gate_run.sh` | Llama only |
| Another open model drafting (runs `big1`, `glim1`) | `RUN_ID=glim1 scripts/open_model_run.sh` (settings in its header) | The open model; Muse only to judge LoCoMo |
| Cost per question, sensitivity, and the comparison with simpler options | `uv run python -m eval.cost_compare` | No |
| The paper's numbers | `uv run python paper/generate.py && cd paper && tectonic itara.tex` | No |
| Tests | `uv run pytest` (needs `docker compose up -d`) | No |

## Roadmap

- **One coding agent, end to end.** Measure a real agent's tool loop through Itara: accuracy, tokens and cost per task.
- **Reach the north star.** A better judge of the open model's drafts, the gate trained on more and broader traffic, and better retrieval for conversations.
- **Operations.** `pip install itara` with the dashboard and schema inside; a Compose file that runs the gateway beside Postgres; a spend cap shared across workers; real streaming.
- **Memory across branches,** and LMCache to share the KV cache across GPU servers.
- **Improve your own open model.** Escalations pair the open model's draft with Muse's answer on the team's own questions, which is a start on training data, though Muse's answer isn't always right. Not built yet.

## Credits

Itara was built by Prince Mutegetsi, with Claude Opus 5.5, during the Meta Global AI Developer Hackathon (26 Oct – 4 Nov 2026; Indie Developer track, Push the Frontier). The only earlier code it uses is about 20 lines of a compression prompt from Prince's [ai-os](https://github.com/chainerprince/ai-os); [`DISCLOSURE.md`](DISCLOSURE.md) lists what was built during the event.

Built with Llama.
