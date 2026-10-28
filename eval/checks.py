"""Invariants that must hold across runs of the same items: the "is the code right?" checks.

On the simulator (eval/sim.py) a model's answer is a known function of what it was shown,
so these hold exactly; any failure is a harness or gateway bug and names the items. On real
models only the structural checks apply (token budget, accounting).

    uv run python -m eval.checks <run_id> [<run_id> ...]   # the runs to compare
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict

from eval.common import RESULTS, Result
from eval.sim import WEAK_SUCCESS

TOKEN_BUDGET = 0.10  # north star: at most 10% of Muse-direct's Muse tokens


def correct(r: Result) -> bool:
    return bool(r.score.get("hit", r.score.get("correct")))


def check(runs: dict[str, list[Result]], simulated: bool) -> list[str]:
    """Failures, as readable lines. `runs` maps a baseline to its results."""
    failures = []
    by_item = {b: {r.item_id: r for r in rs} for b, rs in runs.items()}

    def expect(name: str, bad: list[str]) -> None:
        if bad:
            failures.append(f"{name}: {len(bad)} items, e.g. {bad[:5]}")

    if simulated and "muse_direct" in by_item:
        expect("muse_direct right iff evidence in the full context",
               [i for i, r in by_item["muse_direct"].items() if correct(r) != r.in_context])
    if simulated and "escalate_compact" in by_item:
        expect("escalate_compact right iff evidence in retrieved memory",
               [i for i, r in by_item["escalate_compact"].items()
                if correct(r) != bool(r.gold_in_hits)])
    if simulated and "llama_only" in by_item:
        rs = by_item["llama_only"].values()
        expect("llama_only right only when evidence in retrieved memory",
               [r.item_id for r in rs if correct(r) and not r.gold_in_hits])
        seen = [r for r in rs if r.gold_in_hits]
        if len(seen) >= 20:
            rate = sum(map(correct, seen)) / len(seen)
            if abs(rate - WEAK_SUCCESS) > 0.2:
                failures.append(f"llama_only success when evidence seen is {rate:.0%}, "
                                f"expected about {WEAK_SUCCESS:.0%}")
    if simulated and "rule_gate" in by_item:
        forced = {"local_memory": "llama_only", "escalate_compact": "escalate_compact"}
        expect("rule_gate answers equal the forced route it chose",
               [i for i, r in by_item["rule_gate"].items()
                if forced.get(r.route) in by_item and i in by_item[forced[r.route]]
                and by_item[forced[r.route]][i].answer != r.answer])
    if "muse_direct" in by_item and "escalate_compact" in by_item:
        shared = set(by_item["muse_direct"]) & set(by_item["escalate_compact"])
        tokens = lambda b: sum(by_item[b][i].muse_prompt_tokens  # noqa: E731
                               + by_item[b][i].muse_completion_tokens for i in shared)
        if shared and tokens("escalate_compact") > TOKEN_BUDGET * tokens("muse_direct"):
            failures.append(f"escalate_compact uses {tokens('escalate_compact')} Muse tokens, "
                            f"over {TOKEN_BUDGET:.0%} of muse_direct's {tokens('muse_direct')}")
    return failures


async def check_ledger(pool, run_id: str, results: list[Result]) -> list[str]:
    """The run's answer calls in the usage ledger match its results, token for token."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT coalesce(sum(prompt_tokens), 0), coalesce(sum(completion_tokens), 0),"
            " count(*) FROM usage WHERE user_id = %s AND route NOT IN ('ingest', 'judge')",
            (f"eval-{run_id}",))
        prompt, completion, calls = await cur.fetchone()
    expected = (sum(r.muse_prompt_tokens + r.llama_prompt_tokens for r in results),
                sum(r.muse_completion_tokens + r.llama_completion_tokens for r in results),
                len(results))
    if (prompt, completion, calls) != expected:
        return [f"ledger for run {run_id}: (prompt, completion, calls) = "
                f"{(prompt, completion, calls)}, results say {expected}"]
    return []


def load_run(run_id: str) -> tuple[str, list[Result], dict]:
    """(baseline, results, summary) for one run id."""
    (path,) = (RESULTS / "raw").glob(f"*_{run_id}.jsonl")
    results = [Result(**json.loads(line)) for line in path.read_text().splitlines()]
    summary_path = RESULTS / f"{path.stem}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return results[0].baseline, results, summary


async def main(run_ids: list[str]) -> None:
    from psycopg_pool import AsyncConnectionPool

    from gateway.config import Settings

    runs, simulated, failures = defaultdict(list), True, []
    async with AsyncConnectionPool(Settings().database_url) as pool:
        for run_id in run_ids:
            baseline, results, summary = load_run(run_id)
            runs[baseline] += results
            simulated &= bool(summary.get("simulated"))
            failures += await check_ledger(pool, run_id, results)
    failures += check(runs, simulated)
    print(f"Checked {', '.join(f'{b} ({len(r)})' for b, r in runs.items())}; "
          f"{'simulated: all invariants' if simulated else 'real models: structural checks only'}")
    for line in failures or ["all checks passed"]:
        print(("FAIL " if failures else "OK   ") + line)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+")
    asyncio.run(main(parser.parse_args().run_ids))
