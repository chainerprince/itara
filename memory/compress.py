"""Turn raw text into a compact summary + keywords, and add it to shared memory.

The prompt and JSON parsing are adapted from the ai-os MemoryDaemon
(imported/ai-os/src/agent_os/daemon.py, commit 1); here the local Llama model does the
compression and the result goes to Postgres instead of Elasticsearch.
"""

from __future__ import annotations

import json
from uuid import UUID

from gateway.llm import ChatClient
from memory.store import MemoryStore, Scope

COMPRESSION_SYSTEM_PROMPT = """\
You are a memory compression agent. Your job is to compress a chunk of text
into a concise, information-dense summary that can be used for future context retrieval.

Requirements:
1. Capture ALL important facts, decisions, preferences, and action items.
2. Preserve specific details: names, numbers, dates, code identifiers, technical terms.
3. Use clear, searchable language — this summary will be indexed for hybrid search.
4. Respond with ONLY a JSON object:

{
  "summary": "concise but thorough summary of the chunk",
  "keywords": ["keyword1", "keyword2", "keyword3", ...]
}

The keywords should include important nouns, technical terms, and topic identifiers
that would help a search engine find this memory later."""


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


async def compress(llm: ChatClient, page: str) -> tuple[str, list[str]]:
    response = await llm.chat({
        "messages": [
            {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Compress the following text:\n\n{page}"},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    })
    return parse_compression(response["choices"][0]["message"]["content"])


async def ingest(
    store: MemoryStore,
    llm: ChatClient,
    scope: Scope,
    text: str,
    source: str | None = None,
    private: bool = False,
) -> list[UUID]:
    """Split text into pages, compress each with the local model, store summary + page."""
    ids = []
    for page in split_pages(text):
        summary, keywords = await compress(llm, page)
        ids.append(await store.add(scope, summary, keywords, page, source=source, private=private))
    return ids
