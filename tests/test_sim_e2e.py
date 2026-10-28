"""The invariants (eval/checks.py) on 30 real SWE-bench Lite items, with simulated models.

Slow: loads the dataset and ingests ~30 items. Run with `uv run pytest -m slow`.
"""

import pytest

from eval import checks, replay, swebench_loc
from eval.common import Harness
from eval.sim import SimIndex, SimModel
from gateway.accounting import Ledger
from gateway.auth import Principals
from gateway.server import Services
from memory.store import MemoryStore
from tests.conftest import fake_embed

pytestmark = pytest.mark.slow
BASELINES = ["muse_direct", "escalate_compact", "llama_only", "rule_gate"]


@pytest.fixture(scope="module")
def items():
    return swebench_loc.load(30, seed=1, repo="django/django")


def services(pool, items, backend="vllm"):
    index = SimIndex(items)
    return Services(store=MemoryStore(pool, fake_embed), ledger=Ledger(pool, 1000.0, 1.0, 4.0),
                    llama=SimModel("sim-weak", index), muse=SimModel("sim-strong", index),
                    principals=Principals(pool), llama_backend=backend)


async def run(harness, items, baseline):
    await harness.ingest_all(items, f"eval-{harness.run_id}")
    results = []
    for item in items:
        result = await harness.ask(item, baseline, user=f"eval-{harness.run_id}")
        result.score = swebench_loc.score(item, result.answer)
        results.append(result)
    return results


async def test_invariants_hold_on_real_items(pool, items):
    runs = {}
    for baseline in BASELINES:
        async with Harness(services(pool, items), run_id=f"e2e-{baseline}") as h:
            runs[baseline] = await run(h, items, baseline)
            assert await checks.check_ledger(pool, h.run_id, runs[baseline]) == []
    assert checks.check(runs, simulated=True) == []
    # The comparison is informative: some items are lost to retrieval, some aren't.
    assert 0 < sum(r.gold_in_hits for r in runs["escalate_compact"]) < len(items)


async def test_runs_are_deterministic(pool, items):
    answers = []
    for run_id in ("det-1", "det-2"):
        async with Harness(services(pool, items), run_id=run_id) as h:
            answers.append([(r.answer, r.muse_prompt_tokens, r.gold_in_hits)
                            for r in await run(h, items[:10], "escalate_compact")])
    assert answers[0] == answers[1]


@pytest.mark.parametrize("backend", ["vllm", "ollama"])
async def test_cache_never_crosses_teams(pool, items, backend):
    async with Harness(services(pool, items, backend), run_id=f"iso-{backend}") as h:
        item = items[0]
        first = await h.ask(item, "llama_only", user="alice", team=f"iso-a-{backend}")
        mate = await h.ask(item, "llama_only", user="bob", team=f"iso-a-{backend}")
        other = await h.ask(item, "llama_only", user="carol", team=f"iso-b-{backend}")
    assert first.cached_tokens == 0
    assert mate.cached_tokens > mate.llama_prompt_tokens // 2
    assert other.cached_tokens == 0


async def test_sharing_ingests_less_than_no_sharing(pool, items):
    tasks = replay.order_related(items)[:4]
    async with Harness(services(pool, items), run_id="share") as h:
        shared = await replay.run_team(h, tasks, 2, 2, "rule_gate", team="share-team")
        alone = await replay.run_team(h, tasks, 2, 2, "no_sharing", team="alone-team")
    # A team never ingests more than its members would separately (equal only if their
    # tasks share no file versions).
    assert sum(r.ingest_pages for r in shared) <= sum(r.ingest_pages for r in alone)
