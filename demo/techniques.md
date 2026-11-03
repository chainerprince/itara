# Techniques used in Itara

## Serving and routing
- **OpenAI-compatible gateway** (FastAPI). Any client switches by changing its base URL. Tool calls go to Muse unchanged.
- **Identity from API keys only.** Only each key's SHA-256 hash is stored. Headers can't claim another team. Admin keys are needed to read usage or force a route.
- **Llama 3.1 8B on vLLM on Modal** (H100). Weights live in a Modal Volume and the GPU scales to zero after 5 idle minutes. **Muse Spark 1.2** is reached through the Meta Model API, with a reasoning-effort setting.

## Shared memory
- **Two-level memory.** Llama compresses each page into a summary plus keywords; the compression logic is from ai-os. The raw page is kept behind its summary. Escalations send either the **compact** summaries or the **raw** pages.
- **Compression prompt:** JSON output, one fact per sentence (who, what, when), session dates in the source labels. If the JSON is unusable twice, it falls back to the raw page start.
- **Hybrid retrieval:** pgvector cosine search (bge-small embeddings) and Postgres full-text search, merged with **reciprocal rank fusion**. Ties break on id, so results are deterministic.
- **Row-level security** (`FORCE ROW LEVEL SECURITY`, `SET LOCAL ROLE`). Each team reads only its own memory, and personal memory is private by default.
- **Concurrent ingest** (16 pages in flight, so vLLM batches them). A pre-registered check against sequential ingest showed it gives the same memory.

## Shared KV cache
- **Permission-scoped prefix caching.** A hash of the team, plus the user when memory is private, goes into vLLM's `cache_salt`. Teammates reuse each other's cached prefix; other teams never hit it, which closes the timing side channel. Muse never gets the scope.

## Deciding when to escalate
- **Rule gate** (baseline): escalate long or code-heavy questions.
- **Per-route learned gate:** one logistic model per route on 18 features. Labels come from **escalation regret**, and each route is picked by P(right) − λ × cost. Sweeping λ draws the cost/accuracy curve.
- **Cascade gate (shipped):** Llama drafts, then one logistic score of P(draft right) from **Llama's own confidence**:
  - token log-probabilities;
  - agreement of 4 resampled answers;
  - a yes/no self-check.

  The probes reuse the draft's cached prefix, so they cost Llama tokens and no Muse. Unsure drafts escalate: code memory with summaries, everything else with raw pages.

## Cost control
- **Exact token accounting** from each model response, in a Postgres ledger. Each call is tagged with its request, so the dashboard counts questions, not calls.
- **Hard spend cap:** the worst-case cost is reserved under a lock before every paid call, so concurrent requests can't overshoot it.

## Evaluation
- **SWE-bench Lite file localisation:** all 300 items, scored objectively with no judge.
- **LoCoMo:** 300 questions, judged by Muse. 50 verdicts were checked by hand, and 48 of 50 agreed.
- **Baselines** on the same items: Muse with full context, Llama only, rule gate, always-escalate (compact and raw), no sharing, and an **oracle** that shows the best any gate could do.
- **Out-of-fold scoring grouped by team.** A repo or conversation is never in both training and test.
- **Paired bootstrap** confidence intervals; answers at temperature 0.
- **Team replay:** 1–16 simulated engineers on one repo, related vs random tasks.
- **Pipeline checks** before spending money: a deterministic simulated model with invariant checks, small real-model canaries, and a response cache.
- **Honesty rules:** the north star was stated before the run and reported as failed. Larger budgets and minimal reasoning effort are labelled as chosen after the run, and runaway Muse answers are called out.

## Dashboard and CLI
- **Dashboard,** served by the gateway at `/dashboard`:
  - **Live topology:** in the style of ArgoCD. It shows people, each team's sealed memory and KV cache, and the shared models. Each request lights up its path (Llama, Muse, or memory added), and edges show the cache tokens each person reused.
  - **Live totals:** company-wide, from `/v1/usage` with an admin key, refreshed every 3 s.
  - **Benchmarks:** from `dashboard/results.json` (`scripts/dashboard_data.py`).
  - Plain HTML and SVG with no external dependencies. The colour palette was validated for colour-blind readers, and it has a dark mode.
- **`itara` CLI,** in the style of Docker: `up`, `key add`, `memory add`, `status`, `dashboard`, `demo`. It installs from a GitHub clone with `uv sync`; PyPI is next.
