# Disclosure: pre-existing work vs. hackathon work

**Project:** Itara (built under the working name Memory Gateway): a self-hosted Llama gateway with shared team memory
**Event:** Meta Global AI Developer Hackathon (hacking window: 26 Oct – 4 Nov 2026, 9:00 PM ET)
**Repository created:** 26 Oct 2026, inside the hacking window

## Summary

This repository was created during the hackathon. It reuses a small amount of code from an earlier project by the same author, [`chainerprince/ai-os`](https://github.com/chainerprince/ai-os). All of that code is added in **one import commit** (commit 1), and nothing else enters the repository that way. **Every commit after the import commit is work done during the hackathon.**

To see exactly what was built during the hackathon:

```bash
git log --reverse --format='%h %ad %s' --date=iso   # commit 1 = import, everything after = hackathon
git diff 34a377c..HEAD --stat                       # all hackathon changes
```

Import commit: `34a377c89ef95f73db5e00d658c973efabfa6d29`. The imported files are unmodified copies under [`imported/ai-os/`](imported/ai-os/), with their original paths.

## Pre-existing work (before 26 Oct 2026)

Source: `chainerprince/ai-os`, commit [`5967f9b`](https://github.com/chainerprince/ai-os/commit/5967f9b690fe1260514cce3abbaa338d0703b003) (3 Jun 2026), built on [`d77c79b`](https://github.com/chainerprince/ai-os/commit/d77c79bf8f370b86ab117a2dd84ca64cd1cd3e7a) (3 Jun 2026).

| Imported in commit 1 | From `ai-os` | What it is |
|---|---|---|
| Data models | `src/agent_os/models.py` (126 lines) | Pydantic models for messages and compressed memory "atoms" |
| Compression logic | `src/agent_os/daemon.py` (331 lines; only the compression part is reused) | Turns batches of messages into compact summaries with keywords |
| Monitor UI | `examples/rich_chat.py` (122 lines) | Terminal monitor, imported as a possible starting point for the dashboard. The final dashboard (`dashboard/`) is a new web app and doesn't use it. |

Not imported: the `ai-os` Elasticsearch storage, MCP client, Gemini integration, router and tests. Those stay in `ai-os`, which remains a separate research project.

**Prior research (not code):** a literature and landscape review of LLM memory systems (9 Oct 2026), kept in the `ai-os` repository. It informed the design but isn't part of this submission.

## Built during the hackathon (26 Oct – 4 Nov 2026)

| Component | Status |
|---|---|
| Project plan (`research/index.html`) and this disclosure | Done (26 Oct) |
| OpenAI-compatible gateway server (rule-based gate, tool-call passthrough, API-key identity) | Done (27 Oct) |
| Shared text memory: Postgres + pgvector with row-level security | Done (27 Oct) |
| Exact token accounting and spending cap | Done (27 Oct) |
| Evaluation harness: SWE-bench Lite localisation, LoCoMo, team replay | Done (27 Oct); checked against a deterministic simulated model with invariant checks (28 Oct) |
| Llama 3.1 8B served with vLLM on Modal (`modal_app.py`) | Deployed (29 Oct) |
| Escalation to Muse Spark 1.2 (Meta Model API) | Done; live API from 29 Oct |
| Shared KV cache with permission-scoped keys | Done: on the H100 deployment a teammate reused 6,144 cached prompt tokens and another team 0 (`scripts/cache_check.py`). LMCache across several servers not tested. |
| Concurrent ingest check, memory fixes for dates | Done (29–30 Oct, `results/bringup.ipynb`) |
| Full run on the real models: 300 SWE-bench Lite issues, 300 LoCoMo questions, team replay | Done (30–31 Oct, `results/full_run.ipynb`) |
| Learned gates: per-route gate and the cascade gate on Llama's confidence | Done (31 Oct) |
| LoCoMo judge hand check (50 verdicts) | Done (31 Oct): 48 of 50 agree |
| Muse at minimal reasoning effort, and its intervals | Done (1 Nov; analysed 3 Nov). Chosen after the results, and reported as such. |
| `itara` CLI, dashboard (Vite + React), README | Done (3 Nov) |
| Paper (`paper/`) | Done (3 Nov) |
| Demo video | Planned (3–4 Nov) |

The north star set in the plan (within 3 points of Muse with the full context at ≤ 10% of its tokens) was **not met**. Budgets of 20% and 25%, and Muse's minimal reasoning effort, were chosen after the results and are labelled that way everywhere they appear.

The Official Rules still need to be read and the pre-existing-work wording confirmed.

Update each status as work lands. The commit history is the authoritative record.

## Third-party components

Used as dependencies, not authored by us. Each is used under its own license.

- **Models:** Meta Llama open weights (Llama license); Muse Spark via the Meta Model API
- **Serving and caching:** vLLM, LMCache
- **Storage and retrieval:** PostgreSQL, pgvector; BAAI bge-small-en-v1.5 embeddings via fastembed
- **Dashboard:** React, React Router, Vite
- **Infrastructure:** Modal (GPU), Neon (hosted Postgres for the demo)
- **Datasets:** SWE-bench Lite, LoCoMo (see each dataset's license)

## AI assistance

AI coding assistants were used during the hackathon for research, planning and code. All design decisions, experiments and results were reviewed by the author.
