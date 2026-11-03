"""Apply schema.sql to a database, creating it first if asked. Every statement is idempotent."""

from pathlib import Path

import psycopg
from psycopg import sql

SCHEMA = Path(__file__).parent / "schema.sql"


def init_database(url: str, create: bool = False, fresh: bool = False) -> None:
    """fresh drops and recreates, and only for *_sim, *_test or *_check databases."""
    base, name = url.rsplit("/", 1)
    name = name.split("?")[0]
    if fresh and not name.endswith(("_sim", "_test", "_check")):
        raise SystemExit(f"--fresh only drops *_sim, *_test or *_check databases, not {name!r}")
    if create or fresh:
        with psycopg.connect(f"{base}/postgres", autocommit=True) as admin:
            if fresh:
                admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(name)))
            exists = admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
            if not exists:
                admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(SCHEMA.read_text())
