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
import json
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


async def ingest(
    store: MemoryStore,
    llm: ChatClient,
    scope: Scope,
    text: str,
    source: str | None = None,
    private: bool = False,
    limit: asyncio.Semaphore | None = None,
) -> tuple[list[UUID], list[dict], int]:
    """Split text into pages, compress each with the local model, store summary + page.

    Without `limit`, pages are compressed and stored one at a time. With it, they are
    compressed concurrently (at most `limit` calls in flight, shared with any other ingest
    holding the same semaphore) and then stored in page order, so memory ends up the same.

    Returns the summary ids, the usage of every compression call, and how many pages fell
    back to raw text because the model's JSON was unusable.
    """
    ids, usages, fallbacks = [], [], 0
    pages = split_pages(text)
    if limit is None:
        for page in pages:
            summary, keywords, page_usages, fell_back = await compress(llm, page, source)
            usages += page_usages
            fallbacks += fell_back
            ids.append(await store.add(scope, summary, keywords, page, source=source,
                                       private=private))
        return ids, usages, fallbacks

    async def limited(page: str):
        async with limit:
            return await compress(llm, page, source)

    compressed = await asyncio.gather(*(limited(page) for page in pages))
    for page, (summary, keywords, page_usages, fell_back) in zip(pages, compressed):
        usages += page_usages
        fallbacks += fell_back
        ids.append(await store.add(scope, summary, keywords, page, source=source, private=private))
    return ids, usages, fallbacks
