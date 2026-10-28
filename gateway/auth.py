"""API keys: each key maps to one (team, user). Only a hash of the key is stored."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

from memory.store import Scope


@dataclass(frozen=True)
class Principal:
    scope: Scope
    is_admin: bool


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Principals:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def create(self, team_id: str, user_id: str, is_admin: bool = False) -> str:
        """Make a new key. It's returned once and can't be recovered later."""
        key = "mg-" + secrets.token_urlsafe(32)
        async with self._pool.connection() as conn:
            await conn.execute(
                "INSERT INTO principals (key_sha256, team_id, user_id, is_admin)"
                " VALUES (%s, %s, %s, %s)",
                (_hash(key), team_id, user_id, is_admin),
            )
        return key

    async def lookup(self, key: str) -> Principal | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT team_id, user_id, is_admin FROM principals WHERE key_sha256 = %s",
                (_hash(key),),
            )
            row = await cur.fetchone()
        return Principal(Scope(row[0], row[1]), row[2]) if row else None
