import json
from dataclasses import replace

import pytest

from eval import common, locomo, replay, swebench_loc
from eval.common import Harness, Item, RunLog, check_cost, evidence_visible, summarise
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
    # Seen in the full context; not seen just because the issue names it.
    assert evidence_visible(swebench_loc.to_item({**ROW, "patch": PATCH.split("diff --git a/docs")[0]}),
                            item.raw_context)
    assert not evidence_visible(item, "the bug is in django/db/models/query.py")
    assert evidence_visible(item, "[django/db/models/query.py@1234abcd] s\n[docs/ref.txt@00000000] s")
    assert swebench_loc.score(item, "django/db/models/query.py") == {
        "hit": False, "recall": 0.5, "ceiling": False}
    assert swebench_loc.score(item, "django/db/models/query.py\ndocs/ref.txt")["hit"]


def test_locomo_items_skip_adversarial():
    sample = {"sample_id": "c1", "conversation": {
        "speaker_a": "A", "speaker_b": "B", "session_1_date_time": "1 May 2023",
        "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "I adopted a cat"}],
        "session_2": [{"speaker": "B", "text": "Nice"}]},
        "qa": [{"question": "What did A adopt?", "answer": "a cat", "category": 1,
                "evidence": ["D1:1"]},
               {"question": "Trick?", "adversarial_answer": "x", "category": 5}]}
    (item,) = locomo.items_from([sample])
    assert item.gold == "a cat" and len(item.docs) == 2
    assert item.evidence == [["A: I adopted a cat"]]
    assert item.evidence_sources == ["locomo:c1:session_1"]
    assert evidence_visible(item, item.raw_context) and not evidence_visible(item, item.question)
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
    async with Harness(services, run_id="test") as h:
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
    assert local.compute_tokens == 100 - 40 + 10 and local.gold_in_hits is not None
    assert local.ingest_pages == 2 and local.retrieved_sources == []  # "s" never matches
    direct = await harness.ask(items(1)[0], "muse_direct", user="u")
    assert direct.route == "muse_direct" and direct.muse_prompt_tokens == 100
    assert direct.compute_tokens == 0 and direct.gold_in_hits is None
    # Answers are never sampled; source_in_hits only applies to items that name evidence docs.
    assert harness.services.llama.requests[-1]["temperature"] == 0
    assert harness.services.muse.requests[-1]["temperature"] == 0
    assert local.source_in_hits is None
    sourced = await harness.ask(replace(items(1)[0], evidence_sources=["shared.py"]),
                                "llama_only", user="u")
    assert sourced.source_in_hits is False  # retrieved_sources is empty here
    log = RunLog("swebench_test", "r1")
    for result in (local, direct):
        log.append(result)
    assert [r.baseline for r in log.results()] == ["llama_only", "muse_direct"]
    assert log.done() == {("t0", "u")}
    path = log.write_summary({**harness.labels(), **summarise(log.results(), "hit")})
    summary = json.loads(path.read_text())
    assert summary["muse_is_standin"] is True and summary["run_id"] == "r1"


async def test_spend_cap_blocks_muse_direct(harness):
    harness.ledger.cap_usd = 0.0
    with pytest.raises(SpendCapExceeded):
        await harness.ask(items(1)[0], "muse_direct", user="u")


def test_cost_guard():
    ledger = Ledger(pool=None, cap_usd=1, input_price_per_m=1e6, output_price_per_m=0)
    with pytest.raises(SystemExit):
        check_cost(ledger, [{"messages": "x" * 100}], yes=False)
    assert check_cost(ledger, [{"messages": "x" * 100}], yes=True) > 5


async def test_forced_compact_escalation(harness):
    result = await harness.ask(items(1)[0], "escalate_compact", user="u")
    assert result.route == "escalate_compact" and result.muse_prompt_tokens == 100


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


def test_sim_prefix_cache_is_scoped_by_salt():
    from eval.sim import PrefixCache

    cache, text = PrefixCache(), "shared team context " * 50
    assert cache.lookup_and_insert(text, "team-a") == 0
    assert cache.lookup_and_insert(text, "team-a") > 200
    assert cache.lookup_and_insert(text, "team-b") == 0
    assert cache.lookup_and_insert(text + " more", "team-a") > 200


async def test_sim_answers_only_from_visible_evidence():
    from eval.sim import SimIndex, SimModel, weak_succeeds

    item = swebench_loc.to_item(ROW)
    strong = SimModel("sim-strong", SimIndex([item]))
    ask = lambda text: strong.chat({"messages": [{"role": "user", "content": text}]})  # noqa: E731
    seen = await ask(item.raw_context.replace("[end of django/db/models/query.py]",
                                              "[start of docs/ref.txt]\n[end of docs/ref.txt]"))
    assert seen["choices"][0]["message"]["content"].splitlines() == item.gold
    unseen = await ask(item.question)  # the issue alone shows no files
    assert swebench_loc.score(item, unseen["choices"][0]["message"]["content"])["hit"] is False
    assert weak_succeeds("x") == weak_succeeds("x")


async def test_sim_compression_and_judge():
    from eval.sim import SimIndex, SimModel

    sim = SimModel("sim-weak", SimIndex([]))
    compressed = await sim.chat({"messages": [
        {"role": "system", "content": "You are a memory compression agent. ..."},
        {"role": "user", "content": "Compress the following text:\n\nFile a.py:\n1 def refund():\n"}]})
    assert json.loads(compressed["choices"][0]["message"]["content"])["keywords"] == ["refund"]
    verdict = await sim.chat({"messages": [{"role": "user", "content": locomo.JUDGE_PROMPT.format(
        question="q", gold="7 May 2023", answer="On 7 May, 2023.")}]})
    assert verdict["choices"][0]["message"]["content"] == "CORRECT"


def test_order_related_follows_file_overlap():
    def item(i, files):
        return Item(id=i, team_id="t", docs=[], question="", raw_context="", gold=[],
                    meta={"retrieved": files})

    ordered = replay.order_related([item("lonely", ["z.py"]), item("a", ["x.py", "y.py"]),
                                    item("b", ["x.py"]), item("c", ["y.py", "w.py"])])
    assert [i.id for i in ordered][-1] == "lonely" and ordered[0].id == "a"
