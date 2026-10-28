"""Apply memory/schema.sql to DATABASE_URL. Safe to rerun: every statement is idempotent.

    uv run python -m scripts.init_db
"""

from pathlib import Path

import psycopg

from gateway.config import Settings

SCHEMA = Path(__file__).parent.parent / "memory" / "schema.sql"

if __name__ == "__main__":
    url = Settings().database_url
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(SCHEMA.read_text())
    print(f"Schema applied to {url.rsplit('@', 1)[-1]}")
