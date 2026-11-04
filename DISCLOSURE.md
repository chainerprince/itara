# Disclosure: what was built during the hackathon

**Project:** Itara (built under the working name Memory Gateway): a self-hosted gateway that gives a team one open model, one memory and one KV cache, and calls the closed model only when needed
**Event:** Meta Global AI Developer Hackathon (26 Oct – 4 Nov 2026, 9:00 PM ET)
**Track:** Indie Developer (solo: Prince Mutegetsi)
**Theme:** Push the Frontier: Meta's models (Llama and Muse Spark) on a hard problem in software
**Repository:** [github.com/chainerprince/itara](https://github.com/chainerprince/itara), public, created 26 Oct 2026

## Summary

Everything Itara does was built during the hackathon. The repository's first commit (`34a377c`, 26 Oct) copies three files from an earlier project of mine; **every commit after it is hackathon work**, and so is everything the demo video shows.

```bash
git log --reverse --format='%h %ad %s' --date=iso   # commit 1 = the import; everything after = hackathon
git diff 34a377c..HEAD --stat                       # all hackathon changes
```

## Reused from before the event

Commit 1 copies three files, unmodified, from [`chainerprince/ai-os`](https://github.com/chainerprince/ai-os) at [`5967f9b`](https://github.com/chainerprince/ai-os/commit/5967f9b690fe1260514cce3abbaa338d0703b003) (3 Jun 2026) into [`imported/ai-os/`](imported/ai-os/). Only one piece is used in Itara: a first draft of the compression prompt and its JSON parsing, about 20 lines. [`memory/compress.py`](memory/compress.py) rewrites and extends that prompt (one fact per sentence, dates resolved, worked examples). The other two files, data models and a terminal monitor, are not used.

ai-os is a separate single-user memory project. A literature review I wrote for it (9 Oct 2026) informed the design; it isn't part of this submission.

## Built during the hackathon (26 Oct – 4 Nov 2026)

| Component | Date |
|---|---|
| Project plan ([`research/index.html`](research/index.html)) and this disclosure | 26 Oct |
| OpenAI-compatible gateway server: API-key identity, rule-based gate, tool-call passthrough | 27 Oct |
| Shared team memory: Postgres + pgvector with row-level security, hybrid search | 27 Oct |
| Exact token accounting and a hard spending cap | 27 Oct |
| Evaluation harness: SWE-bench Lite localisation, LoCoMo, team replay; checked on simulated models | 27–28 Oct |
| Llama 3.1 8B on vLLM on Modal ([`modal_app.py`](modal_app.py)); escalation to Muse Spark | 29 Oct |
| Permission-scoped KV cache: a teammate reused 6,144 cached tokens, another team 0 ([`scripts/cache_check.py`](scripts/cache_check.py)) | 29 Oct |
| Concurrent-ingest check and memory fixes for dates ([`results/bringup.ipynb`](results/bringup.ipynb)) | 29–30 Oct |
| Full run on the real models: 300 SWE-bench Lite issues, 300 LoCoMo questions, team replay ([`results/`](results/)) | 30–31 Oct |
| Learned gates: per-route gate and the cascade gate on Llama's confidence; LoCoMo judge hand check | 31 Oct |
| Muse at minimal reasoning effort (chosen after the results, reported as such) | 1–3 Nov |
| `itara` CLI with accurate, fast and rule modes; dashboard (Vite + React); README | 3 Nov |
| Cost per question and the comparison with RouteLLM ([`eval/cost_compare.py`](eval/cost_compare.py)) | 3 Nov |
| Llama 3.3 70B as the open model, a simpler gate with a second opinion, and further confidence probes (after the results, reported as such) | 3 Nov |
| Muse Glimmer 30B as the open model, on code and conversations (run `glim1`, after the results, reported as such) | 3 Nov |
| Paper ([`paper/itara.pdf`](paper/itara.pdf)) | 3 Nov |
| Demo video | 4 Nov |

The north star set in the plan on 26 Oct (within 3 points of Muse with the full context at ≤ 10% of its tokens) has **not been reached yet**. Budgets of 20% and 25%, Muse's minimal reasoning effort and the 70B and Glimmer runs were chosen after the results and are labelled that way everywhere they appear.

## Third-party components

Used as dependencies, not authored by us, each under its own license.

- **Models:** Meta Llama 3.1 8B and Llama 3.3 70B (Llama Community License; Built with Llama); Muse Glimmer 30B (Apache 2.0); Muse Spark through the Meta Model API
- **Serving and caching:** vLLM, LMCache
- **Storage and retrieval:** PostgreSQL, pgvector; BAAI bge-small-en-v1.5 embeddings via fastembed
- **Dashboard:** React, React Router, Vite
- **Infrastructure:** Modal (GPU), Neon (hosted Postgres for the demo)
- **Datasets:** SWE-bench Lite and LoCoMo, used under their licenses. The repository stores item ids, model answers and scores, not copies of the datasets; the LoCoMo hand check quotes its 50 questions and their gold answers.

## AI assistance

AI coding assistants were used during the hackathon for research, planning, code and writing. All design decisions, experiments and results were reviewed by the author.
