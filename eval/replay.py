"""Team replay: SWE-bench tasks on one repo, worked on by N simulated engineers at once.

Every engineer handles the same number of tasks at every team size, so per-user cost falls
only if the team really shares work (memory ingested once, KV cache reused). Each run uses
a fresh team so earlier runs can't warm memory or cache. no_sharing gives each engineer
their own team, which is the comparison curve.

Task order: `related` (default) models a team working in one area of the codebase: tasks
are chosen greedily so each one overlaps most with the files the team has already touched.
`random` is the conservative check. Both are reported; the order is in every row.

Compute tokens (uncached prompt + completion on the local model) are the cost measure that
works on any backend; seconds are only meaningful on vLLM, which batches requests.
"""

from __future__ import annotations

import asyncio
import csv
import time

from eval import swebench_loc
from eval.common import RESULTS, Harness, Item, Result, RunLog, summarise


def order_related(items: list[Item]) -> list[Item]:
    """Greedy: start where files overlap most, then always add the most overlapping task."""
    files = {item.id: set(item.meta["retrieved"]) for item in items}

    def overlap_with_others(item: Item) -> int:
        return sum(len(files[item.id] & files[other.id]) for other in items if other is not item)

    remaining = sorted(items, key=lambda i: (-overlap_with_others(i), i.id))
    ordered, touched = [remaining.pop(0)], set()
    touched |= files[ordered[0].id]
    while remaining:
        best = max(remaining, key=lambda i: (len(files[i.id] & touched), -remaining.index(i)))
        remaining.remove(best)
        ordered.append(best)
        touched |= files[best.id]
    return ordered


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
                 baseline: str, repo: str, order: str) -> list[dict]:
    rows = []
    run_id = harness.run_id
    for n in team_sizes:
        results = await run_team(harness, items, n, per_user, baseline,
                                 team=f"eval:replay:{repo}:{order}:{n}:{run_id}")
        log = RunLog(f"replay_{baseline}_{order}_n{n}", run_id)
        for result in results:
            log.append(result)
        log.write_summary({"team_size": n, "order": order, **harness.labels(),
                           **summarise(results, "hit")})
        prompt = sum(r.llama_prompt_tokens for r in results)
        rows.append({
            "team_size": n,
            "baseline": baseline,
            "order": order,
            "tasks": len(results),
            "accuracy": sum(r.score["hit"] for r in results) / len(results),
            "muse_tokens_per_user": sum(r.muse_prompt_tokens + r.muse_completion_tokens
                                        for r in results) / n,
            "compute_tokens_per_user": sum(r.compute_tokens for r in results) / n,
            "ingest_pages_per_user": sum(r.ingest_pages for r in results) / n,
            "llama_seconds_per_user": sum(r.seconds for r in results
                                          if not r.muse_prompt_tokens) / n,
            "ingest_seconds_per_user": sum(r.ingest_seconds for r in results) / n,
            "cached_ratio": sum(r.cached_tokens for r in results) / prompt if prompt else 0.0,
            **harness.labels(),
        })
        print(rows[-1])
    path = RESULTS / f"replay_{baseline}_{order}_{run_id or time.strftime('%Y%m%d-%H%M%S')}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")
    return rows
