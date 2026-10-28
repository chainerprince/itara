"""One command for every benchmark and baseline.

    uv run python -m eval.run swebench --baseline rule_gate -n 20
    uv run python -m eval.run locomo   --baseline muse_direct -n 20 --yes
    uv run python -m eval.run replay   --baseline no_sharing --users 1,2,4 --per-user 2

Baselines: muse_direct (no gateway, raw context to Muse), llama_only (never escalate),
rule_gate (the gateway as it is), no_sharing (replay only: no shared memory or cache),
full (the learned gate, day 7).
"""

from __future__ import annotations

import argparse
import asyncio

from eval import locomo, replay, swebench_loc
from eval.common import BASELINES, Harness, check_cost, summarise, write


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
    async with Harness() as harness:
        check_cost(harness.ledger, worst_case_bodies(items, args.baseline, bench is locomo),
                   args.yes)
        results = []
        for i, item in enumerate(items, 1):
            result = await harness.ask(item, args.baseline, user="eval-user")
            if bench is swebench_loc:
                result.score = swebench_loc.score(item, result.answer)
            else:
                result.score = await locomo.judge(harness, item, result.answer)
            results.append(result)
            print(f"[{i}/{len(items)}] {item.id} {result.route} {result.score}")
        headline = "hit" if bench is swebench_loc else "correct"
        summary = {"bench": args.bench, "baseline": args.baseline, "seed": args.seed,
                   **harness.labels(), **summarise(results, headline)}
        path = write(f"{args.bench}_{args.baseline}", results, summary)
    print({k: v for k, v in summary.items() if k != "routes"})
    print(f"Wrote {path}")


async def run_replay(args) -> None:
    sizes = [int(n) for n in args.users.split(",")]
    items = swebench_loc.load(max(sizes) * args.per_user, args.seed, args.repo)
    async with Harness() as harness:
        check_cost(harness.ledger, worst_case_bodies(items, args.baseline, False) * len(sizes),
                   args.yes)
        await replay.replay(harness, items, sizes, args.per_user, args.baseline, args.repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench", choices=["swebench", "locomo", "replay"])
    parser.add_argument("--baseline", choices=BASELINES, default="rule_gate")
    parser.add_argument("-n", type=int, default=20, help="items (swebench, locomo)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo", default=None, help="swebench: only this repo")
    parser.add_argument("--users", default="1,2,4,8,16", help="replay: team sizes")
    parser.add_argument("--per-user", type=int, default=2, help="replay: tasks per engineer")
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
