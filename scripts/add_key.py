"""Create a gateway API key for one (team, user). The key is printed once; only its hash is kept.

    uv run python -m scripts.add_key --team web --user alice [--admin]
"""

import argparse
import asyncio

from psycopg_pool import AsyncConnectionPool

from gateway.auth import Principals
from gateway.config import Settings


async def main(team: str, user: str, admin: bool) -> None:
    async with AsyncConnectionPool(Settings().database_url) as pool:
        print(await Principals(pool).create(team, user, is_admin=admin))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--admin", action="store_true", help="may read /v1/usage and force routes")
    args = parser.parse_args()
    asyncio.run(main(args.team, args.user, args.admin))
