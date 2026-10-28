"""Apply memory/schema.sql to DATABASE_URL. Safe to rerun: every statement is idempotent.

    uv run python -m scripts.init_db             # apply the schema
    uv run python -m scripts.init_db --create    # create the database first if it's missing
    uv run python -m scripts.init_db --fresh     # drop and recreate it (only *_sim / *_test)

Experiments use their own databases (gateway_sim, gateway_canary) so their memory never
mixes with other runs'.
"""

import argparse
from pathlib import Path

import psycopg
from psycopg import sql

from gateway.config import Settings

SCHEMA = Path(__file__).parent.parent / "memory" / "schema.sql"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="create the database if missing")
    parser.add_argument("--fresh", action="store_true", help="drop and recreate (*_sim, *_test only)")
    args = parser.parse_args()

    url = Settings().database_url
    base, name = url.rsplit("/", 1)
    name = name.split("?")[0]
    if args.fresh and not name.endswith(("_sim", "_test")):
        raise SystemExit(f"--fresh only drops *_sim or *_test databases, not {name!r}")
    if args.create or args.fresh:
        with psycopg.connect(f"{base}/postgres", autocommit=True) as admin:
            if args.fresh:
                admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(name)))
            exists = admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
            if not exists:
                admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(SCHEMA.read_text())
    print(f"Schema applied to {url.rsplit('@', 1)[-1]}")
