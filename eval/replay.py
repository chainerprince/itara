"""Team replay: SWE-bench tasks on one repo, worked on by N simulated engineers at once.

Every engineer handles the same number of tasks at every team size, so per-user cost falls
only if the team really shares work (memory ingested once, KV cache reused). Each run uses
a fresh team so earlier runs can't warm memory or cache. no_sharing gives each engineer
their own team, which is the comparison curve.

On Ollama requests are served one at a time, so the seconds are only indicative until vLLM
on Modal; Muse tokens per user and ingest-once are valid locally.
"""

from __future__ import annotations

import asyncio
import csv
import time

from eval import swebench_loc
from eval.common import RESULTS, Harness, Item, Result, write


async def run_team(harness: Harness, items: list[Item], users: int, per_user: int,
                   baseline: str, team: str) -> list[Result]:
    async def engineer(u: int) -> list[Result]:
        results = []
        for item in items[u * per_user:(u + 1) * per_user]:
            result = await harness.ask(item, baseline, user=f"eng{u}", team=team)
            result.score = swebench_loc.score(item, result.answer)
            results.append(result)
        return results

    per_engineer = await asyncio.gather(*(engineer(u) for u in range(users)))
    return [r for results in per_engineer for r in results]


async def replay(harness: Harness, items: list[Item], team_sizes: list[int], per_user: int,
                 baseline: str, repo: str) -> list[dict]:
    rows, everything = [], []
    stamp = int(time.time())
    for n in team_sizes:
        results = await run_team(harness, items, n, per_user, baseline,
                                 team=f"eval:replay:{repo}:{n}:{stamp}")
        everything += results
        prompt = sum(r.llama_prompt_tokens for r in results)
        rows.append({
            "team_size": n,
            "baseline": baseline,
            "tasks": len(results),
            "accuracy": sum(r.score["hit"] for r in results) / len(results),
            "muse_tokens_per_user": sum(r.muse_prompt_tokens + r.muse_completion_tokens
                                        for r in results) / n,
            "llama_seconds_per_user": sum(r.seconds for r in results
                                          if not r.muse_prompt_tokens) / n,
            "ingest_seconds_per_user": sum(r.ingest_seconds for r in results) / n,
            "cached_ratio": sum(r.cached_tokens for r in results) / prompt if prompt else 0.0,
            **harness.labels(),
        })
        print(rows[-1])
    write(f"replay_{baseline}", everything, {"rows": rows})
    path = RESULTS / f"replay_{baseline}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")
    return rows
