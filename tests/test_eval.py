import json

import pytest

from eval import common, locomo, replay, swebench_loc
from eval.common import Harness, Item, check_cost, summarise
from gateway.accounting import Ledger, SpendCapExceeded
from gateway.auth import Principals
from gateway.server import Services
from memory.store import MemoryStore
from tests.conftest import FakeModel, fake_embed

PATCH = """<patch>
diff --git a/django/db/models/query.py b/django/db/models/query.py
--- a/django/db/models/query.py
diff --git a/docs/ref.txt b/docs/ref.txt
"""
BM25_TEXT = """You will be provided with a partial code base.
<issue>
QuerySet breaks
</issue>
<code>
[start of README.rst]
1 Django
[end of README.rst]
[start of django/db/models/query.py]
1 class QuerySet:
2     pass
[end of django/db/models/query.py]
</code>
I need you to solve this issue by generating a single patch file"""
ROW = {"instance_id": "django__django-1", "text": BM25_TEXT, "repo": "django/django",
       "problem_statement": "QuerySet breaks", "patch": PATCH}


def test_gold_and_predicted_files():
    assert swebench_loc.gold_files(PATCH) == ["django/db/models/query.py", "docs/ref.txt"]
    answer = "The fix is in:\n1. `django/db/models/query.py`\n- ./docs/ref.txt (docs)\n"
    assert swebench_loc.predicted_files(answer) == ["django/db/models/query.py", "docs/ref.txt"]


def test_swebench_item_and_score():
    item = swebench_loc.to_item(ROW)
    assert [s.split("@")[0] for s, _ in item.docs] == [
        "README.rst", "django/db/models/query.py"]
    assert item.raw_context.rstrip().endswith(swebench_loc.INSTRUCTION)
    assert "</code>" in item.raw_context and "single patch file" not in item.raw_context
    assert swebench_loc.score(item, "django/db/models/query.py") == {
        "hit": False, "recall": 0.5, "ceiling": False}
    assert swebench_loc.score(item, "django/db/models/query.py\ndocs/ref.txt")["hit"]


def test_locomo_items_skip_adversarial():
    sample = {"sample_id": "c1", "conversation": {
        "speaker_a": "A", "speaker_b": "B", "session_1_date_time": "1 May 2023",
        "session_1": [{"speaker": "A", "text": "I adopted a cat"}],
        "session_2": [{"speaker": "B", "text": "Nice"}]},
        "qa": [{"question": "What did A adopt?", "answer": "a cat", "category": 1},
               {"question": "Trick?", "adversarial_answer": "x", "category": 5}]}
    (item,) = locomo.items_from([sample])
    assert item.gold == "a cat" and len(item.docs) == 2
    assert "1 May 2023" in item.docs[0][1] and "A: I adopted a cat" in item.raw_context
    assert locomo.parse_verdict(" correct.") and not locomo.parse_verdict("WRONG")


@pytest.fixture
async def harness(pool, tmp_path, monkeypatch):
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE usage")
    monkeypatch.setattr(common, "RESULTS", tmp_path)
    monkeypatch.setattr(replay, "RESULTS", tmp_path)
    llama = FakeModel("llama", reply="django/db/models/query.py\ndocs/ref.txt")

    async def compress_or_answer(body, _chat=llama.chat):
        if "memory compression" in body["messages"][0]["content"]:
            llama.requests.append(body)
            return {"choices": [{"message": {"content": '{"summary": "s", "keywords": []}'}}]}
        return await _chat(body)

    llama.chat = compress_or_answer
    services = Services(store=MemoryStore(pool, fake_embed), ledger=Ledger(pool, 1.0, 1.0, 2.0),
                        llama=llama, muse=FakeModel("muse"), principals=Principals(pool),
                        muse_is_standin=True)
    async with Harness(services) as h:
        yield h


def items(n, team="eval:test"):
    return [Item(id=f"t{i}", team_id=team, docs=[("shared.py", "shared code"),
                                                 (f"own{i}.py", f"code {i}")],
                 question="Which file?", raw_context="ctx\nWhich file?",
                 gold=["django/db/models/query.py"], meta={"retrieved": []})
            for i in range(n)]


async def test_baselines_end_to_end(harness, tmp_path):
    local = await harness.ask(items(1)[0], "llama_only", user="u")
    assert local.route == "local_memory" and local.llama_prompt_tokens == 100
    direct = await harness.ask(items(1)[0], "muse_direct", user="u")
    assert direct.route == "muse_direct" and direct.muse_prompt_tokens == 100
    summary = summarise([local, direct], "hit")
    path = common.write("swebench_test", [local, direct], {**harness.labels(), **summary})
    assert json.loads(path.read_text())["muse_is_standin"] is True
    assert len((tmp_path / "raw").glob("*.jsonl").__next__().read_text().splitlines()) == 2


async def test_spend_cap_blocks_muse_direct(harness):
    harness.ledger.cap_usd = 0.0
    with pytest.raises(SpendCapExceeded):
        await harness.ask(items(1)[0], "muse_direct", user="u")


def test_cost_guard():
    ledger = Ledger(pool=None, cap_usd=1, input_price_per_m=1e6, output_price_per_m=0)
    with pytest.raises(SystemExit):
        check_cost(ledger, [{"messages": "x" * 100}], yes=False)
    assert check_cost(ledger, [{"messages": "x" * 100}], yes=True) > 5


def compressions(harness):
    return sum("memory compression" in r["messages"][0]["content"]
               for r in harness.services.llama.requests)


async def test_replay_shares_ingest_only_with_sharing(harness):
    tasks = items(4)
    await replay.run_team(harness, tasks, users=2, per_user=2, baseline="rule_gate", team="t-a")
    shared = compressions(harness)
    await replay.run_team(harness, tasks, users=2, per_user=2, baseline="no_sharing", team="t-b")
    assert shared == 5                            # shared.py once + 4 own files
    assert compressions(harness) - shared == 6    # shared.py once per engineer + 4 own files


def test_summary_averages_only_numeric_scores():
    results = [common.Result("a", "b", "u", "local", "", {"correct": True, "judge": "CORRECT"})]
    assert summarise(results, "correct")["accuracy"] == 1.0
