import asyncio
import json

from memory.compress import ingest, parse_compression, split_pages
from memory.store import MemoryStore, Scope
from tests.conftest import fake_embed


def test_split_pages():
    text = "\n\n".join(["a" * 900] * 5)
    pages = split_pages(text, max_chars=2000)
    assert len(pages) == 3
    assert "".join(pages).replace("\n", "") == "a" * 4500
    assert all(len(p) <= 2000 for p in split_pages("b" * 5000))


def test_parse_compression_strips_fences():
    assert parse_compression('```json\n{"summary": "s", "keywords": ["k"]}\n```') == ("s", ["k"])


class FakeLlama:
    async def chat(self, body):
        page = body["messages"][1]["content"]
        content = '{"summary": "summary of: %s", "keywords": ["fake"]}' % page[-20:].strip()
        return {"choices": [{"message": {"content": content}}]}


async def test_ingest(pool):
    store = MemoryStore(pool, fake_embed)
    scope = Scope("ingest-team", "carol")
    ids, usages, fallbacks = await ingest(store, FakeLlama(), scope, "deploy steps for billing",
                                          source="runbook.md")
    assert len(ids) == len(usages) == 1 and fallbacks == 0
    pages = await store.pages(scope, ids)
    assert pages[0].content == "deploy steps for billing"


class BadJsonLlama:
    calls = 0

    async def chat(self, body):
        self.calls += 1
        return {"choices": [{"message": {"content": "Sure! Here is a summary"}}], "usage": {}}


async def test_bad_json_falls_back_to_page_start(pool):
    store = MemoryStore(pool, fake_embed)
    scope = Scope("ingest-team", "carol")
    llm = BadJsonLlama()
    ids, _, fallbacks = await ingest(store, llm, scope, "rotate keys monthly " * 50)
    assert llm.calls == 2 and fallbacks == 1
    hits = await store.search(scope, "rotate keys monthly")
    assert hits and hits[0].content.startswith("rotate keys monthly")


class SlowLlama:
    """Compresses after a short wait, counting how many calls overlap."""

    def __init__(self):
        self.in_flight = self.most = 0
        self.pages: list[str] = []

    async def chat(self, body):
        self.in_flight += 1
        self.most = max(self.most, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        page = body["messages"][1]["content"].split("\n\n", 1)[1]
        self.pages.append(page)
        content = json.dumps({"summary": f"summary of {page[:7]}", "keywords": ["k"]})
        return {"choices": [{"message": {"content": content}}]}


EIGHT_PAGES = "\n\n".join(f"page {i} " + "x" * 1900 for i in range(8))


async def test_ingest_without_limit_is_one_page_at_a_time(pool):
    store, llm = MemoryStore(pool, fake_embed), SlowLlama()
    await ingest(store, llm, Scope("ingest-team", "carol"), EIGHT_PAGES)
    assert llm.most == 1
    assert [p[:6] for p in llm.pages] == [f"page {i}" for i in range(8)]


async def test_concurrent_ingest_respects_limit_and_page_order(pool):
    store, llm = MemoryStore(pool, fake_embed), SlowLlama()
    scope = Scope("ingest-team", "carol")
    ids, usages, _ = await ingest(store, llm, scope, EIGHT_PAGES, limit=asyncio.Semaphore(3))
    assert llm.most == 3 and len(ids) == len(usages) == 8
    stored = [(await store.pages(scope, [i]))[0].content[:6] for i in ids]
    assert stored == [f"page {i}" for i in range(8)]
