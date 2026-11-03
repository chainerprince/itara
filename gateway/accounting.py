"""Exact token accounting and a hard cap on closed-model spend.

Costs come from the token counts each API response reports. Before a paid call, the
worst-case cost (estimated prompt + max_tokens) is reserved, so concurrent requests
can't push total spend past the cap. The reservation lives in this process, so run the
server with a single worker.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from uuid import UUID

from psycopg_pool import AsyncConnectionPool

from memory.store import Scope

DEFAULT_MAX_TOKENS = 1024


class SpendCapExceeded(Exception):
    pass


class Ledger:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        cap_usd: float,
        input_price_per_m: float,
        output_price_per_m: float,
    ) -> None:
        self._pool = pool
        self.cap_usd = cap_usd
        self._input_price = input_price_per_m
        self._output_price = output_price_per_m
        self._reserved = 0.0
        self._lock = asyncio.Lock()

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self._input_price + completion_tokens * self._output_price) / 1e6

    def worst_case_cost(self, body: dict) -> float:
        # ~2 characters per token: an upper bound even for code, which tokenizes densely.
        prompt = len(str(body.get("messages", "")) + str(body.get("tools", ""))) // 2 + 1
        completion = body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS
        return self.cost(prompt, completion)

    async def spent(self) -> float:
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT coalesce(sum(cost_usd), 0) FROM usage")
            return float((await cur.fetchone())[0])

    @asynccontextmanager
    async def reserve(self, estimate_usd: float) -> AsyncIterator[None]:
        """Hold estimate_usd of the budget for the duration of one paid call."""
        async with self._lock:
            spent = await self.spent()
            if spent + self._reserved + estimate_usd > self.cap_usd:
                raise SpendCapExceeded(
                    f"Spend cap reached: ${spent:.4f} spent of ${self.cap_usd:.2f}"
                )
            self._reserved += estimate_usd
        try:
            yield
        finally:
            self._reserved -= estimate_usd

    async def record(
        self,
        scope: Scope,
        model: str,
        route: str,
        usage: dict,
        latency_ms: int,
        paid: bool,
        estimate_usd: float = 0.0,
        request_id: UUID | None = None,
    ) -> None:
        """Record one call. A paid call with no token counts is charged estimate_usd."""
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        estimated = paid and not (prompt or completion)
        if not paid:
            cost = 0.0
        elif estimated:
            cost = estimate_usd
        else:
            cost = self.cost(prompt, completion)
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO usage (team_id, user_id, model, route, prompt_tokens, completion_tokens,"
                " cached_tokens, cost_usd, latency_ms, estimated, request_id)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (scope.team_id, scope.user_id, model, route, prompt, completion, cached, cost,
                 latency_ms, estimated, request_id),
            )

    async def summary(self) -> list[dict]:
        """Totals per model and route, for the dashboard."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT model, route, count(*), sum(prompt_tokens), sum(completion_tokens),"
                " sum(cached_tokens), sum(latency_ms), sum(cost_usd), max(created_at)"
                " FROM usage GROUP BY model, route ORDER BY model, route"
            )
            keys = ("model", "route", "requests", "prompt_tokens", "completion_tokens",
                    "cached_tokens", "latency_ms", "cost_usd")
            return [
                {**dict(zip(keys, row)), "cost_usd": float(row[7]), "last_at": row[8].isoformat()}
                for row in await cur.fetchall()
            ]

    async def requests(self, paid_routes: Iterable[str], recent: int = 20) -> dict:
        """Per gateway request, for the dashboard: its final route and what it cost.

        A request's route is its paid (Muse) call if it made one, else its Llama call, so a
        cascade question kept on Llama counts as local_memory however many probes it ran.
        Returns the totals per person (team, user) and route, and the most recent requests.
        """
        per_request = (
            "SELECT request_id, min(created_at) AS at, min(team_id) AS team_id,"
            " min(user_id) AS user_id,"
            " coalesce(max(route) FILTER (WHERE route = ANY(%(paid)s)),"
            "          max(route) FILTER (WHERE route <> 'gate_probe')) AS route,"
            " count(*) AS calls,"
            " coalesce(sum(prompt_tokens + completion_tokens)"
            "          FILTER (WHERE route = ANY(%(paid)s)), 0) AS muse_tokens,"
            " coalesce(sum(prompt_tokens) FILTER (WHERE route <> ALL(%(paid)s)), 0)"
            "   AS llama_prompt_tokens,"
            " sum(cached_tokens) AS cached_tokens,"
            " coalesce(sum(latency_ms) FILTER (WHERE route <> ALL(%(paid)s)), 0) AS llama_ms,"
            " sum(latency_ms) AS latency_ms, sum(cost_usd) AS cost_usd"
            " FROM usage WHERE request_id IS NOT NULL GROUP BY request_id"
        )
        keys = ("team_id", "user_id", "route", "requests", "calls", "muse_tokens",
                "llama_prompt_tokens", "cached_tokens", "llama_ms", "cost_usd", "last_at")
        params = {"paid": list(paid_routes), "recent": recent}
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT team_id, user_id, route, count(*), sum(calls), sum(muse_tokens),"
                " sum(llama_prompt_tokens), sum(cached_tokens), sum(llama_ms), sum(cost_usd),"
                f" max(at) FROM ({per_request}) r"
                " GROUP BY team_id, user_id, route ORDER BY team_id, user_id, route",
                params,
            )
            totals = [{**dict(zip(keys, (*row[:3], *map(int, row[3:9])))),
                       "cost_usd": float(row[9]), "last_at": row[10].isoformat()}
                      for row in await cur.fetchall()]
            cur = await conn.execute(f"{per_request} ORDER BY at DESC LIMIT %(recent)s", params)
            columns = [c.name for c in cur.description]
            latest = []
            for row in await cur.fetchall():
                r = dict(zip(columns, row))
                latest.append({**r, "request_id": str(r["request_id"]), "at": r["at"].isoformat(),
                               **{k: int(r[k]) for k in columns[5:-1]},
                               "cost_usd": float(r["cost_usd"])})
        return {"by_user_route": totals, "recent": latest}
