"""Shared text memory in Postgres + pgvector, scoped by row-level security.

Every query runs inside `scoped()`, which switches to the `gateway_app` role and sets
the caller's team and user, so Postgres only returns rows that caller may read.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from psycopg import AsyncConnection
from psycopg.errors import UniqueViolation
from psycopg.rows import class_row
from psycopg_pool import AsyncConnectionPool

Embedder = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class Scope:
    team_id: str
    user_id: str


@dataclass
class Memory:
    id: UUID
    owner_id: str | None
    kind: str
    parent_id: UUID | None
    source: str | None
    content: str
    keywords: list[str]
    token_count: int


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token). Billing uses exact counts from the API."""
    return max(1, len(text) // 4)


def default_embedder() -> Embedder:
    """BAAI/bge-small-en-v1.5 via fastembed: 384 dimensions, runs on CPU."""
    from fastembed import TextEmbedding

    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return lambda texts: [vector.tolist() for vector in model.embed(texts)]


_COLUMNS = "id, owner_id, kind, parent_id, source, content, keywords, token_count"

# Hybrid search: reciprocal rank fusion of vector and keyword rankings over one kind of row
# (summaries, or the raw pages for the route that skips them).
# Keyword matching uses OR across query words; ts_rank still favours matching more of them.
# Every ranking breaks ties by id: summaries of near-identical file versions tie exactly, and
# without a tie-break the same question can get different memory from one run (or teammate)
# to the next, which breaks reproducible evals and shared-prefix cache hits.
_SEARCH_SQL = f"""
WITH vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(vector)s, id) AS rank
    FROM memories
    WHERE kind = %(kind)s AND embedding <=> %(vector)s < %(max_distance)s
    ORDER BY embedding <=> %(vector)s, id
    LIMIT 20
), kw AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank(tsv, query) DESC, id) AS rank
    FROM memories, to_tsquery('simple', replace(plainto_tsquery('english', %(text)s)::text, '&', '|')) AS query
    WHERE kind = %(kind)s AND tsv @@ query
    ORDER BY ts_rank(tsv, query) DESC, id
    LIMIT 20
)
SELECT {", ".join(f"m.{c}" for c in _COLUMNS.split(", "))}
FROM (SELECT * FROM vec UNION ALL SELECT * FROM kw) AS hits
JOIN memories AS m USING (id)
GROUP BY m.id
ORDER BY sum(1.0 / (60 + hits.rank)) DESC, m.id
LIMIT %(k)s
"""


class MemoryStore:
    def __init__(self, pool: AsyncConnectionPool, embed: Embedder) -> None:
        self._pool = pool
        self._embed = embed
        # Pages being compressed now, by (team, owner, content hash): see compress.ingest_page.
        self.in_flight: dict[tuple[str, str | None, str], asyncio.Future] = {}

    @asynccontextmanager
    async def scoped(self, scope: Scope) -> AsyncIterator[AsyncConnection]:
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute("SET LOCAL ROLE gateway_app")
            # HNSW returns a fixed number of candidates and RLS filters them afterwards; keep
            # scanning until enough visible rows are found, or other teams' rows crowd ours out.
            await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            await conn.execute(
                "SELECT set_config('app.team_id', %s, true), set_config('app.user_id', %s, true)",
                (scope.team_id, scope.user_id),
            )
            yield conn

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        return await self._vectors(texts)

    async def _vectors(self, texts: list[str]) -> list[np.ndarray]:
        vectors = await asyncio.to_thread(self._embed, texts)
        return [np.array(v, dtype=np.float32) for v in vectors]

    async def add(
        self,
        scope: Scope,
        summary: str,
        keywords: list[str],
        page: str,
        source: str | None = None,
        private: bool = False,
        content_hash: str | None = None,
    ) -> UUID:
        """Store one summary and the raw page it was made from. Returns the summary id.

        With a content_hash, a page the scope already holds isn't stored twice: the existing
        summary's id comes back instead.
        """
        owner = scope.user_id if private else None
        summary_vec, page_vec = await self._vectors([summary, page])
        insert = """
            INSERT INTO memories (team_id, owner_id, kind, parent_id, source, content, keywords,
                                  token_count, embedding, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        try:
            async with self.scoped(scope) as conn:
                cur = await conn.execute(
                    insert,
                    (scope.team_id, owner, "summary", None, source, summary, keywords,
                     estimate_tokens(summary), summary_vec, None),
                )
                (summary_id,) = await cur.fetchone()
                await conn.execute(
                    insert,
                    (scope.team_id, owner, "page", summary_id, source, page, keywords,
                     estimate_tokens(page), page_vec, content_hash),
                )
        except UniqueViolation:  # stored meanwhile by another process; this one rolled back
            existing = await self.summary_of_page(scope, content_hash, private)
            if existing is None:
                raise
            return existing
        return summary_id

    async def summary_of_page(self, scope: Scope, content_hash: str,
                              private: bool = False) -> UUID | None:
        """The summary of the scope's page with this content hash, if it has one."""
        async with self.scoped(scope) as conn:
            cur = await conn.execute(
                "SELECT parent_id FROM memories WHERE kind = 'page' AND content_hash = %s"
                " AND owner_id IS NOT DISTINCT FROM %s",
                (content_hash, scope.user_id if private else None),
            )
            row = await cur.fetchone()
        return row[0] if row else None

    async def delete_source(self, scope: Scope, source: str, private: bool = False,
                            keep: list[UUID] | None = None) -> list[str]:
        """Forget a source: its summaries and (by cascade) their pages. Returns the sources removed.

        With `keep` (the summaries of the version just added), forget every version of the
        source instead (the same name before any "@", as in path@hash) except those summaries:
        memory then holds only the current version.
        """
        match = ("regexp_replace(source, '@[^@]*$', '') = %(path)s AND NOT id = ANY(%(keep)s)"
                 if keep is not None else "source = %(source)s")
        async with self.scoped(scope) as conn:
            cur = await conn.execute(
                f"DELETE FROM memories WHERE kind = 'summary' AND {match}"
                " AND owner_id IS NOT DISTINCT FROM %(owner)s RETURNING source",
                {"source": source, "path": source.rsplit("@", 1)[0], "keep": keep or [],
                 "owner": scope.user_id if private else None},
            )
            return sorted({row[0] for row in await cur.fetchall()})

    async def search(
        self, scope: Scope, text: str, k: int = 5, max_distance: float = 0.5,
        kind: str = "summary",
    ) -> list[Memory]:
        """Best-matching summaries (or, with kind="page", raw pages) the caller can read."""
        (vector,) = await self._vectors([text])
        params = {"vector": vector, "text": text, "k": k, "max_distance": max_distance,
                  "kind": kind}
        async with self.scoped(scope) as conn:
            cur = conn.cursor(row_factory=class_row(Memory))
            await cur.execute(_SEARCH_SQL, params)
            return await cur.fetchall()

    async def has_source(self, scope: Scope, source: str) -> bool:
        async with self.scoped(scope) as conn:
            cur = await conn.execute(
                "SELECT exists(SELECT 1 FROM memories WHERE kind = 'summary' AND source = %s)",
                (source,),
            )
            return (await cur.fetchone())[0]

    async def pages(self, scope: Scope, summary_ids: list[UUID]) -> list[Memory]:
        """Raw pages behind the given summaries, for when exact details matter."""
        async with self.scoped(scope) as conn:
            cur = conn.cursor(row_factory=class_row(Memory))
            await cur.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE kind = 'page' AND parent_id = ANY(%s)",
                (summary_ids,),
            )
            return await cur.fetchall()
