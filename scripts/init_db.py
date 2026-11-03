"""Apply memory/schema.sql to DATABASE_URL. Safe to rerun: every statement is idempotent.

    uv run python -m scripts.init_db             # apply the schema
    uv run python -m scripts.init_db --create    # create the database first if it's missing
    uv run python -m scripts.init_db --fresh     # drop and recreate (*_sim / *_test / *_check only)

Experiments use their own databases (gateway_sim, gateway_canary, gateway_*_check) so their
memory never mixes with other runs'.
"""

import argparse

from gateway.config import Settings
from memory.schema import init_database

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="create the database if missing")
    parser.add_argument("--fresh", action="store_true",
                        help="drop and recreate (*_sim, *_test, *_check only)")
    args = parser.parse_args()

    url = Settings().database_url
    init_database(url, create=args.create, fresh=args.fresh)
    print(f"Schema applied to {url.rsplit('@', 1)[-1]}")
