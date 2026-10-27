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
    ids = await ingest(store, FakeLlama(), scope, "deploy steps for billing", source="runbook.md")
    assert len(ids) == 1
    pages = await store.pages(scope, ids)
    assert pages[0].content == "deploy steps for billing"
