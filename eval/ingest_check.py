"""Is concurrent ingest as good as sequential? Compares three databases that ingested the same
items: seqa and seqb one page at a time (INGEST_CONCURRENCY=1), conc concurrently.

seqb is the control: sampling and batching make even two sequential runs differ, so conc
passes a measure if it is no further from seqa than seqb is, plus a small tolerance. The page
set must match exactly (splitting is deterministic). On the simulator compression is
deterministic too, so every summary must match. scripts/ingest_check.sh fills the databases.

    uv run python -m eval.ingest_check --id check1 -n 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict

import numpy as np
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from eval import locomo, swebench_loc
from eval.common import RESULTS
from gateway.config import Settings
from memory.store import MemoryStore, Scope

MODES = ("seqa", "seqb", "conc")
BENCHES = ("swebench", "locomo")
# How much further from seqa than seqb is conc may be. completion_tokens is relative to seqa.
TOLERANCE = {"fallback_rate": 0.02, "calls_per_page": 0.02, "completion_tokens": 0.05,
             "summary_similarity": 0.02, "accuracy": 0.03, "gold_in_hits_rate": 0.03,
             "source_in_hits_rate": 0.03}
RELATIVE = {"completion_tokens"}

_PAGES_SQL = """
SELECT p.source, p.content, s.content, s.keywords, s.embedding
FROM memories AS s JOIN memories AS p ON p.parent_id = s.id
WHERE s.kind = 'summary' AND p.kind = 'page'
"""


def database(run_id: str, mode: str) -> str:
    return f"gateway_{run_id.replace('-', '_')}_{mode}_check"


async def team_pages(store: MemoryStore, team: str) -> list[tuple]:
    """(source, page, summary, keywords, summary embedding) for every page the team holds."""
    async with store.scoped(Scope(team, "ingest-check")) as conn:
        cur = await conn.execute(_PAGES_SQL)
        return await cur.fetchall()


async def read(url: str, teams: set[str]) -> tuple[dict, dict]:
    """Summaries by (team, source, page), and the ingest calls' token usage."""
    memory = defaultdict(list)
    async with AsyncConnectionPool(url, configure=register_vector_async) as pool:
        store = MemoryStore(pool, embed=None)  # only scoped() is used, so reads obey RLS
        for team in sorted(teams):
            for source, page, summary, keywords, embedding in await team_pages(store, team):
                memory[team, source or "", page].append((summary, keywords, embedding.to_numpy()))
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT count(*), coalesce(avg(completion_tokens), 0)"
                                     " FROM usage WHERE route = 'ingest'")
            calls, completion = await cur.fetchone()
    return memory, {"calls": calls, "completion_tokens": float(completion)}


def is_fallback(page: str, summary: str, keywords: list[str]) -> bool:
    return summary == page[:600] and not keywords  # see compress() in memory/compress.py


def similarity(memory: dict, base: dict) -> tuple[float, float]:
    """Mean cosine of summary embeddings and the share of identical summaries, over shared pages."""
    cosines, same = [], 0
    for key, rows in memory.items():
        if key not in base:
            continue
        (summary, _, vec), (base_summary, _, base_vec) = rows[0], base[key][0]
        cosines.append(float(vec @ base_vec / (np.linalg.norm(vec) * np.linalg.norm(base_vec))))
        same += summary == base_summary
    n = len(cosines) or 1
    return sum(cosines) / n, same / n


def run_summary(name: str) -> dict:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


async def main(args) -> None:
    items = swebench_loc.load(args.n, args.seed, args.repo) + locomo.load(args.n, args.seed)
    teams = {item.team_id for item in items}
    base = Settings().database_url.rsplit("/", 1)[0]
    data = {m: await read(f"{base}/{database(args.id, m)}", teams) for m in MODES}

    rows: dict[str, dict] = defaultdict(dict)
    labels = {m: run_summary(f"swebench_ingest_{args.id}-{m}") for m in MODES}
    for m in MODES:
        memory, usage = data[m]
        pages = sum(len(r) for r in memory.values())
        rows["pages"][m] = pages
        rows["fallback_rate"][m] = sum(is_fallback(key[2], summary, keywords)
                                       for key, rs in memory.items()
                                       for summary, keywords, _ in rs) / (pages or 1)
        rows["calls_per_page"][m] = usage["calls"] / (pages or 1)
        rows["completion_tokens"][m] = usage["completion_tokens"]
        rows["summary_similarity"][m], rows["identical_summaries"][m] = (
            similarity(memory, data["seqa"][0]))
        for bench in BENCHES:
            answers = run_summary(f"{bench}_llama_only_{args.id}-{m}")
            for k in ("accuracy", "gold_in_hits_rate", "source_in_hits_rate"):
                rows[f"{bench}_{k}"][m] = answers.get(k)
            rows[f"{bench}_pages_per_second"][m] = (
                run_summary(f"{bench}_ingest_{args.id}-{m}").get("pages_per_second"))

    failures = []
    concurrency = {m: labels[m].get("ingest_concurrency") for m in MODES}
    if concurrency["seqa"] != 1 or concurrency["seqb"] != 1 or not (concurrency["conc"] or 0) > 1:
        failures.append(f"modes ran at the wrong concurrency: {concurrency}")
    page_set = {m: Counter({k: len(v) for k, v in data[m][0].items()}) for m in MODES}
    for m in ("seqb", "conc"):
        if page_set[m] != page_set["seqa"]:
            failures.append(f"{m} holds different pages from seqa: "
                            f"{sum((page_set['seqa'] - page_set[m]).values())} missing, "
                            f"{sum((page_set[m] - page_set['seqa']).values())} extra")
    for measure, values in rows.items():
        kind = measure.split("_", 1)[1] if measure.startswith(BENCHES) else measure
        if kind not in TOLERANCE or any(values.get(m) is None for m in MODES):
            continue
        a, b, c = (values[m] for m in MODES)
        tolerance = TOLERANCE[kind] * (abs(a) if kind in RELATIVE else 1)
        if (c < b - tolerance) if kind == "summary_similarity" else (
                abs(c - a) > abs(b - a) + tolerance):
            failures.append(f"{measure}: conc {fmt(c)} vs seqa {fmt(a)}, where seqb is "
                            f"{fmt(b)} (tolerance {tolerance:.3f})")
    simulated = all(labels[m].get("simulated") for m in MODES)
    if simulated and min(rows["identical_summaries"].values()) < 1:
        failures.append("simulated, so every summary must match: "
                        f"identical shares {dict(rows['identical_summaries'])}")

    print(f"{'measure':28}" + "".join(f"{m:>12}" for m in MODES))
    for measure, values in rows.items():
        if all(v is None for v in values.values()):  # e.g. source_in_hits on SWE-bench
            continue
        print(f"{measure:28}" + "".join(f"{fmt(values.get(m)):>12}" for m in MODES))
    print(f"\nconcurrency {concurrency}; {labels['seqa'].get('llama_model')}"
          f"{' (simulated)' if simulated else ''}")
    for line in failures or ["concurrent ingest is as close to seqa as seqb is on every measure"]:
        print(("FAIL " if failures else "OK   ") + line)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True, help="the CHECK_ID scripts/ingest_check.sh used")
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo", default=None)
    asyncio.run(main(parser.parse_args()))
