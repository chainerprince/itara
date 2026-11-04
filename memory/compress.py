"""Turn raw text into a compact summary + keywords, and add it to shared memory.

The prompt and JSON parsing are adapted from the ai-os MemoryDaemon
(imported/ai-os/src/agent_os/daemon.py, commit 1); here the local Llama model does the
compression and the result goes to Postgres instead of Elasticsearch. The prompt asks for
one fact per sentence, with worked examples: given rules alone, Llama 3.1 8B wrote topic
summaries ("A and B discuss ...") that lost who did what and when. The examples are made up,
not taken from any benchmark.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import nullcontext
from uuid import UUID

from gateway.llm import ChatClient
from memory.store import MemoryStore, Scope

COMPRESSION_SYSTEM_PROMPT = """\
You are a memory compression agent. Your job is to compress a chunk of text
into a concise, information-dense summary that can be used for future context retrieval.

Requirements:
1. Write the summary as short factual sentences, one fact per sentence: who did what,
   and when. Never write "A and B discuss ..."; state what was actually said or done.
2. Capture ALL important facts, decisions, preferences, and action items.
3. Preserve specific details: names, numbers, dates, code identifiers, technical terms.
4. Keep when things happened. If the text or its source says when it was written, turn
   relative times ("yesterday", "last week", "two years ago") into dates.
5. Use clear, searchable language — this summary will be indexed for hybrid search.
6. Respond with ONLY a JSON object:

{
  "summary": "concise but thorough summary of the chunk",
  "keywords": ["keyword1", "keyword2", "keyword3", ...]
}

The keywords should include important nouns, technical terms, and topic identifiers
that would help a search engine find this memory later.

Example. Source: chat, 3 March 2024
Priya: I finally started glassblowing classes last week! Tomas: Nice. We adopted a dog two days ago, her name is Biscuit.
{"summary": "Priya started glassblowing classes the week before 3 March 2024. Tomas adopted a dog named Biscuit on 1 March 2024.", "keywords": ["Priya", "glassblowing classes", "Tomas", "dog", "Biscuit", "adoption"]}

Example. Source: billing/retry.py
def charge(card, amount, attempts=3): ... retries a declined charge up to three times, waiting 2 s, then raises PaymentError.
{"summary": "charge() in billing/retry.py retries a declined charge up to 3 times, 2 seconds apart, then raises PaymentError.", "keywords": ["charge", "billing/retry.py", "retry", "declined charge", "PaymentError"]}"""


def split_pages(text: str, max_chars: int = 2000) -> list[str]:
    """Split on blank lines into pages of at most ~max_chars (~500 tokens)."""
    pages: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        if current and len(current) + len(block) > max_chars:
            pages.append(current)
            current = ""
        current = f"{current}\n\n{block}" if current else block
        while len(current) > max_chars:
            pages.append(current[:max_chars])
            current = current[max_chars:]
    if current.strip():
        pages.append(current)
    return pages


def parse_compression(text: str) -> tuple[str, list[str]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(cleaned)
    return data.get("summary", ""), data.get("keywords", [])


async def compress(llm: ChatClient, page: str, source: str | None = None,
                   ) -> tuple[str, list[str], list[dict], bool]:
    """Summary and keywords for one page, the usage of each model call, and whether it fell back.

    The source label goes with every page, so a page that isn't the first of its doc still
    knows where it came from (e.g. the date of the conversation, for relative times).

    Small models sometimes return invalid JSON: retry once, then fall back to the start of
    the page as its own summary so the page is still stored and searchable.
    """
    usages = []
    text = f"Compress the following text:\n\n{page}"
    if source:
        text = f"Source: {source}\n\n{text}"
    for _ in range(2):
        response = await llm.chat({
            "messages": [
                {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        })
        usages.append(response.get("usage", {}))
        try:
            summary, keywords = parse_compression(response["choices"][0]["message"]["content"])
        except (ValueError, AttributeError):
            continue
        if isinstance(summary, str) and summary.strip():
            return summary, [k for k in keywords if isinstance(k, str)], usages, False
    return page[:600], [], usages, True


def page_hash(source: str | None, page: str) -> str:
    """What makes two pages the same memory: the same text from the same source."""
    return hashlib.sha256(f"{source or ''}\0{page}".encode()).hexdigest()


async def ingest_page(
    store: MemoryStore,
    llm: ChatClient,
    scope: Scope,
    page: str,
    source: str | None,
    private: bool,
    limit: asyncio.Semaphore | None,
) -> tuple[UUID, list[dict], bool, bool]:
    """Compress and store one page, unless the caller's scope already holds it.

    A page another request is compressing right now is waited for, not compressed again
    (in this process; the database's unique index catches the same race across processes).
    Returns the summary id, the compression calls' usage, whether it fell back, and whether
    the page was already there.
    """
    content_hash = page_hash(source, page)
    key = (scope.team_id, scope.user_id if private else None, content_hash)
    if key not in store.in_flight:
        existing = await store.summary_of_page(scope, content_hash, private)
        if existing is not None:
            return existing, [], False, True
    if key in store.in_flight:  # checked again: another ingest may have started meanwhile
        return await asyncio.shield(store.in_flight[key]), [], False, True
    done = store.in_flight[key] = asyncio.get_running_loop().create_future()
    try:
        async with limit or nullcontext():
            summary, keywords, usages, fell_back = await compress(llm, page, source)
        summary_id = await store.add(scope, summary, keywords, page, source=source,
                                     private=private, content_hash=content_hash)
        done.set_result(summary_id)
        return summary_id, usages, fell_back, False
    except BaseException as e:
        done.set_exception(e)
        done.exception()  # retrieved here, so nobody waiting isn't an error
        raise
    finally:
        del store.in_flight[key]


async def ingest(
    store: MemoryStore,
    llm: ChatClient,
    scope: Scope,
    text: str,
    source: str | None = None,
    private: bool = False,
    limit: asyncio.Semaphore | None = None,
) -> tuple[list[UUID], list[dict], int, int]:
    """Split text into pages, compress each with the local model, store summary + page.

    Each page is compressed once per scope: a page already in the caller's memory (same
    source, same text) is reused, not compressed or stored again, so re-ingesting a repo
    only pays for what changed. Without `limit`, pages are compressed one at a time. With it,
    concurrently (at most `limit` calls in flight, shared with any other ingest holding the
    same semaphore).

    Returns the summary ids in page order, the usage of every compression call, how many
    pages fell back to raw text because the model's JSON was unusable, and how many pages
    were already in memory.
    """
    pages = split_pages(text)
    if limit is None:
        results = [await ingest_page(store, llm, scope, page, source, private, None)
                   for page in pages]
    else:
        results = await asyncio.gather(*(ingest_page(store, llm, scope, page, source, private,
                                                     limit) for page in pages))
    return ([r[0] for r in results], [u for r in results for u in r[1]],
            sum(r[2] for r in results), sum(r[3] for r in results))
