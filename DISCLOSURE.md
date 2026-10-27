# Disclosure: pre-existing work vs. hackathon work

**Project:** Memory Gateway: a self-hosted Llama gateway with shared team memory
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
| Monitor UI | `examples/rich_chat.py` (122 lines) | Terminal monitor, used as the starting point for the dashboard |

Not imported: the `ai-os` Elasticsearch storage, MCP client, Gemini integration, router and tests. Those stay in `ai-os`, which remains a separate research project.

**Prior research (not code):** a literature and landscape review of LLM memory systems (9 Oct 2026), kept in the `ai-os` repository. It informed the design but isn't part of this submission.

## Built during the hackathon (26 Oct – 4 Nov 2026)

| Component | Status |
|---|---|
| Project plan (`research/index.html`) and this disclosure | Done (26 Oct) |
| Llama gateway served with vLLM on Modal | Planned |
| Escalation to Muse Spark (Meta Model API) | Code done (27 Oct); not yet run against the live API |
| OpenAI-compatible gateway server (rule-based gate, tool-call passthrough) | Done (27 Oct) |
| Shared text memory: Postgres + pgvector with row-level security | Done (27 Oct) |
| Shared KV cache (vLLM + LMCache) with permission-scoped cache keys | Planned |
| Learned escalation gate | Planned |
| Exact token accounting and spending cap | Done (27 Oct) |
| Evaluation harness: SWE-bench Lite localisation, LoCoMo subset, team replay | Planned |
| Results, dashboard, write-up, demo video | Planned |

Update each status as work lands. The commit history is the authoritative record.

## Third-party components

Used as dependencies, not authored by us. Each is used under its own license.

- **Models:** Meta Llama open weights (Llama license); Muse Spark via the Meta Model API
- **Serving and caching:** vLLM, LMCache
- **Storage:** PostgreSQL, pgvector
- **Infrastructure:** Modal (GPU), Neon (hosted Postgres for the demo)
- **Datasets:** SWE-bench Lite, LoCoMo (see each dataset's license)

## AI assistance

AI coding assistants were used during the hackathon for research, planning and code. All design decisions, experiments and results were reviewed by the author.
