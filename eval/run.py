"""One command for every benchmark and baseline.

    uv run python -m eval.run swebench --baseline rule_gate -n 20
    uv run python -m eval.run swebench --ingest-only -n 20      # fill memory first (local model only)
    uv run python -m eval.run locomo   --baseline muse_direct -n 20 --yes
    uv run python -m eval.run swebench --baseline muse_direct --resume 20261028-101500
    uv run python -m eval.run replay   --baseline no_sharing --users 1,2,4 --per-user 2

Baselines: muse_direct (no gateway, raw context to Muse), escalate_compact (always escalate
with memory summaries), llama_only (never escalate), rule_gate (the gateway as it is),
no_sharing (replay only: no shared memory or cache), full (the learned gate, day 7).

Results are appended item by item to eval/results/raw/<bench>_<baseline>_<run_id>.jsonl,
so a run can be stopped and resumed with --resume <run_id>. Each run uses its own user id
(eval-<run_id>), so its rows in the usage ledger can be matched to its results.
"""

from __future__ import annotations

import argparse
import asyncio

from eval import locomo, replay, swebench_loc
from eval.common import BASELINES, Harness, RunLog, check_cost, new_run_id, summarise


def worst_case_bodies(items, baseline: str, judged: bool) -> list[dict]:
    if baseline == "llama_only":
        bodies = []
    else:  # upper bound: everything escalates with the full retrieved context
        bodies = [{"messages": [{"role": "user", "content": i.raw_context}]} for i in items]
    if judged:
        bodies += [{"messages": [{"role": "user", "content": i.question}], "max_tokens": 4}
                   for i in items]
    return bodies


async def run_bench(args) -> None:
    bench = swebench_loc if args.bench == "swebench" else locomo
    items = (swebench_loc.load(args.n, args.seed, args.repo) if bench is swebench_loc
             else locomo.load(args.n, args.seed))
    run_id = args.resume or new_run_id()
    user = f"eval-{run_id}"
    log = RunLog(f"{args.bench}_{'ingest' if args.ingest_only else args.baseline}", run_id)
    todo = [i for i in items if (i.id, user) not in log.done()]
    async with Harness(run_id=run_id) as harness:
        if args.ingest_only:  # idempotent: docs already in memory are skipped
            totals = [0.0, 0, 0]
            for n, item in enumerate(todo, 1):
                seconds, pages, fallbacks = await harness.ingest(item, item.team_id, user)
                totals = [totals[0] + seconds, totals[1] + pages, totals[2] + fallbacks]
                print(f"[{n}/{len(todo)}] {item.id}: {pages} pages, {fallbacks} fallbacks, "
                      f"{seconds:.0f}s")
            path = log.write_summary({
                "bench": args.bench, "phase": "ingest", **harness.labels(),
                "items": len(todo), "ingest_seconds": round(totals[0], 1),
                "ingest_pages": totals[1],
                "ingest_fallback_rate": totals[2] / totals[1] if totals[1] else None})
            print(f"Wrote {path}")
            return
        check_cost(harness.ledger, worst_case_bodies(todo, args.baseline, bench is locomo),
                   args.yes)
        print(f"Run {run_id}: {len(items) - len(todo)} done, {len(todo)} to go")
        if args.baseline != "muse_direct":
            await harness.ingest_all(items, user)
        for n, item in enumerate(todo, 1):
            result = await harness.ask(item, args.baseline, user=user)
            if bench is swebench_loc:
                result.score = swebench_loc.score(item, result.answer)
            else:
                result.score = await locomo.judge(harness, item, result.answer, user)
            log.append(result)
            print(f"[{n}/{len(todo)}] {item.id} {result.route} {result.score}")
        headline = "hit" if bench is swebench_loc else "correct"
        summary = {"bench": args.bench, "baseline": args.baseline, "seed": args.seed,
                   **harness.labels(), **summarise(log.results(), headline)}
        path = log.write_summary(summary)
    print({k: v for k, v in summary.items() if k != "routes"})
    print(f"Wrote {path}")


async def run_replay(args) -> None:
    sizes = [int(n) for n in args.users.split(",")]
    needed = max(sizes) * args.per_user
    candidates = swebench_loc.load(10_000, args.seed, args.repo)
    if args.order == "related":
        candidates = replay.order_related(candidates)
    items = candidates[:needed]
    async with Harness(run_id=new_run_id()) as harness:
        check_cost(harness.ledger, worst_case_bodies(items, args.baseline, False) * len(sizes),
                   args.yes)
        await replay.replay(harness, items, sizes, args.per_user, args.baseline, args.repo,
                            args.order)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench", choices=["swebench", "locomo", "replay"])
    parser.add_argument("--baseline", choices=BASELINES, default="rule_gate")
    parser.add_argument("-n", type=int, default=20, help="items (swebench, locomo)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo", default=None, help="swebench: only this repo")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    parser.add_argument("--ingest-only", action="store_true",
                        help="only add the items' docs to memory (local model only)")
    parser.add_argument("--users", default="1,2,4,8,16", help="replay: team sizes")
    parser.add_argument("--per-user", type=int, default=2, help="replay: tasks per engineer")
    parser.add_argument("--order", choices=["related", "random"], default="related",
                        help="replay: how tasks are picked")
    parser.add_argument("--yes", action="store_true", help="allow a worst case over $5")
    args = parser.parse_args()
    if args.bench == "replay":
        args.repo = args.repo or "django/django"
        asyncio.run(run_replay(args))
    else:
        if args.baseline == "no_sharing":
            parser.error("no_sharing only applies to replay")
        asyncio.run(run_bench(args))


if __name__ == "__main__":
    main()
