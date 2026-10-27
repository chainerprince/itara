"""Exact token accounting and a hard cap on closed-model spend.

Costs come from the token counts each API response reports. Before a paid call, the
worst-case cost (estimated prompt + max_tokens) is reserved, so concurrent requests
can't push total spend past the cap.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg_pool import AsyncConnectionPool

from memory.store import Scope, estimate_tokens

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
        prompt = estimate_tokens(str(body.get("messages", "")) + str(body.get("tools", "")))
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
        self, scope: Scope, model: str, route: str, usage: dict, latency_ms: int, paid: bool
    ) -> None:
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        cost = self.cost(prompt, completion) if paid else 0.0
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO usage (team_id, user_id, model, route, prompt_tokens,"
                " completion_tokens, cost_usd, latency_ms) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (scope.team_id, scope.user_id, model, route, prompt, completion, cost, latency_ms),
            )

    async def summary(self) -> list[dict]:
        """Totals per model and route, for the dashboard."""
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT model, route, count(*), sum(prompt_tokens), sum(completion_tokens),"
                " sum(cost_usd) FROM usage GROUP BY model, route ORDER BY model, route"
            )
            keys = ("model", "route", "requests", "prompt_tokens", "completion_tokens", "cost_usd")
            return [
                {**dict(zip(keys, row)), "cost_usd": float(row[5])}
                for row in await cur.fetchall()
            ]
