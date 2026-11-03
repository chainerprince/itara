"""itara: run a team's shared gateway, give people keys, add memory, watch it work.

    itara up                         # Postgres (Docker), the schema, then the gateway on :8000
    itara key add --team web --user alice [--admin]
    itara memory add src/ docs/ --key mg-…   # a team's code and docs, compressed once for all
    itara status --key mg-…          # admin: who asked what, and where it went
    itara dashboard                  # the live topology and the benchmark results
    itara demo                       # two teammates share memory; another team can't see it

Settings come from the environment or a .env file in the current directory (.env.example
lists them). ITARA_URL and ITARA_KEY save typing --url and --key.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent.parent
DEFAULT_URL = "http://localhost:8000/v1"
# Text worth remembering; code sources get @<commit> so escalations send them as summaries.
CODE = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".rb", ".php", ".c",
        ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift", ".scala", ".sql", ".sh", ".toml", ".yaml",
        ".yml", ".json", ".html", ".css"}
DOCS = {".md", ".rst", ".txt", ".adoc"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build", ".mypy_cache",
             ".pytest_cache", ".ruff_cache"}
MAX_BYTES = 200_000


def url_option(p: argparse.ArgumentParser) -> None:
    p.add_argument("--url", default=os.environ.get("ITARA_URL", DEFAULT_URL), help="the gateway's /v1")


def key_option(p: argparse.ArgumentParser, admin: bool = False) -> None:
    p.add_argument("--key", default=os.environ.get("ITARA_KEY"),
                   help=f"a gateway key{' with admin rights' if admin else ''} (or ITARA_KEY)")


def need_key(args) -> str:
    if not args.key:
        raise SystemExit("Needs a gateway key: --key, or ITARA_KEY (itara key add makes one)")
    return args.key


# ───────── up ─────────

def wait_for_postgres(url: str, seconds: float = 60) -> None:
    import psycopg

    base = url.rsplit("/", 1)[0]
    deadline = time.monotonic() + seconds
    while True:
        try:
            psycopg.connect(f"{base}/postgres").close()
            return
        except psycopg.OperationalError:
            if time.monotonic() > deadline:
                raise SystemExit(f"Postgres isn't answering at {urlparse(url).hostname}") from None
            time.sleep(1)


def up(args) -> None:
    import uvicorn

    from gateway.config import Settings
    from memory.schema import init_database

    if args.gate:
        os.environ["GATE_PATH"] = args.gate
    settings = Settings()
    if urlparse(settings.database_url).hostname in ("localhost", "127.0.0.1") and not args.no_docker:
        print("Starting Postgres + pgvector (docker compose)…")
        subprocess.run(["docker", "compose", "up", "-d", "--wait"], cwd=ROOT, check=True)
    wait_for_postgres(settings.database_url)
    init_database(settings.database_url, create=True)
    gate = settings.gate_path or "the rule-based gate (set GATE_PATH or --gate for the learned one)"
    print(f"\n  itara is up on http://{args.host}:{args.port}/v1  ·  gate: {gate}"
          f"\n  dashboard:  http://localhost:{args.port}/dashboard"
          f"\n  Llama: {settings.llama_model} ({settings.llama_backend})  ·  Muse: {settings.muse_model}"
          f"{' (a local stand-in)' if settings.muse_is_standin else ''}\n")
    # One worker: the spend cap's reservations live in this process (gateway/accounting.py).
    uvicorn.run("gateway.server:app", host=args.host, port=args.port, workers=1)


# ───────── key ─────────

def key_add(args) -> None:
    from psycopg_pool import AsyncConnectionPool

    from gateway.auth import Principals
    from gateway.config import Settings

    async def create() -> str:
        async with AsyncConnectionPool(Settings().database_url) as pool:
            return await Principals(pool).create(args.team, args.user, is_admin=args.admin)

    key = asyncio.run(create())
    print(key)
    print(f"# {args.user} on team {args.team}{' (admin)' if args.admin else ''}. Shown once; only "
          "its hash is stored.", file=sys.stderr)


# ───────── memory ─────────

def commit_of(path: Path) -> str | None:
    out = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], cwd=path if path.is_dir() else path.parent,
                         capture_output=True, text=True, check=False)
    return out.stdout.strip() or None


def collect(paths: list[str]) -> list[tuple[str, Path]]:
    """(source label, file) for every text file under paths. Code is labelled path@commit."""
    files = []
    for arg in paths:
        p = Path(arg)
        found = [p] if p.is_file() else sorted(
            f for f in p.rglob("*")
            if f.is_file() and not any(part in SKIP_DIRS or part.startswith(".") for part in f.relative_to(p).parts))
        for f in found:
            suffix = f.suffix.lower()
            if suffix not in CODE | DOCS or f.stat().st_size > MAX_BYTES:
                continue
            label = f.as_posix() if not f.is_absolute() else f.name
            if suffix in CODE:
                label += f"@{commit_of(f) or 'workdir'}"
            files.append((label, f))
    return files


def memory_add(args) -> None:
    import httpx

    files = collect(args.paths)
    if not files:
        raise SystemExit("No text files found (code, Markdown, reStructuredText or plain text)")
    headers = {"Authorization": f"Bearer {need_key(args)}"}

    async def send() -> None:
        limit = asyncio.Semaphore(8)  # files in flight; the gateway caps compression calls
        async with httpx.AsyncClient(base_url=args.url, timeout=900, headers=headers) as gw:
            async def one(source: str, path: Path) -> int:
                async with limit:
                    r = await gw.post("/memory", json={"text": path.read_text(errors="replace"),
                                                       "source": source, "private": args.private})
                r.raise_for_status()
                print(f"  {source}: {len(r.json()['ids'])} pages")
                return len(r.json()["ids"])

            start = time.monotonic()
            pages = sum(await asyncio.gather(*(one(s, p) for s, p in files)))
            print(f"{pages} pages from {len(files)} files in {time.monotonic() - start:.0f}s, "
                  f"{'private to you' if args.private else 'shared with your team'}")

    asyncio.run(send())


# ───────── status ─────────

def status(args) -> None:
    import httpx

    r = httpx.get(f"{args.url}/usage", headers={"Authorization": f"Bearer {need_key(args)}"}, timeout=30)
    if r.status_code == 403:
        raise SystemExit("status needs an admin key (itara key add --admin)")
    r.raise_for_status()
    u = r.json()
    print(f"Llama  {u['llama_model']} ({u['llama_backend']})   Muse  {u['muse_model']}"
          f"{' (stand-in)' if u['muse_is_standin'] else ''}   gate  {u['gate']}")
    print(f"Muse spend  ${u['spent_usd']:.4f} of ${u['cap_usd']:.0f}\n")
    people: dict[tuple[str, str], dict] = {}
    for row in u["by_user_route"]:
        p = people.setdefault((row["team_id"], row["user_id"]), {
            "questions": 0, "llama": 0, "muse": 0, "cached": 0, "prompt": 0, "pages": 0, "last": ""})
        if row["route"] == "ingest":
            p["pages"] += row["calls"]
        else:
            p["questions"] += row["requests"]
            p["llama" if row["route"].startswith("local") else "muse"] += row["requests"]
            p["cached"] += row["cached_tokens"]
            p["prompt"] += row["llama_prompt_tokens"]
        p["last"] = max(p["last"], row["last_at"])
    print(f"{'TEAM':10}{'USER':14}{'QUESTIONS':>10}{'ON LLAMA':>10}{'TO MUSE':>9}{'CACHED':>9}{'PAGES':>7}  LAST")
    for (team, user), p in sorted(people.items()):
        cached = f"{p['cached'] / p['prompt']:.0%}" if p["prompt"] else "-"
        print(f"{team:10}{user:14}{p['questions']:>10}{p['llama']:>10}{p['muse']:>9}{cached:>9}"
              f"{p['pages']:>7}  {p['last'][:19].replace('T', ' ')}")
    for team, scope in sorted(u["cache_scopes"].items()):
        print(f"\nteam {team}: KV cache scope {scope[:8]}", end="")
    print()


# ───────── dashboard, demo ─────────

def dashboard(args) -> None:
    url = args.url.rsplit("/v1", 1)[0] + "/dashboard"
    print(url)
    webbrowser.open(url)


def demo(args) -> None:
    sys.exit(subprocess.run([sys.executable, "-m", "scripts.demo", "--url", args.url], cwd=ROOT, check=False).returncode)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="itara", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("up", help="start Postgres (Docker), apply the schema, run the gateway")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gate", help="a gate saved by eval/learn_gate.py (sets GATE_PATH)")
    p.add_argument("--no-docker", action="store_true", help="Postgres is already running")
    p.set_defaults(run=up)

    keys = sub.add_parser("key", help="gateway keys").add_subparsers(dest="action", required=True)
    p = keys.add_parser("add", help="make a key for one person on one team")
    p.add_argument("--team", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--admin", action="store_true", help="may read usage and force routes")
    p.set_defaults(run=key_add)

    mem = sub.add_parser("memory", help="team memory").add_subparsers(dest="action", required=True)
    p = mem.add_parser("add", help="add files or folders (code and docs) to memory")
    p.add_argument("paths", nargs="+")
    p.add_argument("--private", action="store_true", help="only you can read it (default: your team)")
    url_option(p)
    key_option(p)
    p.set_defaults(run=memory_add)

    p = sub.add_parser("status", help="who asked what, and where it went (admin)")
    url_option(p)
    key_option(p, admin=True)
    p.set_defaults(run=status)

    p = sub.add_parser("dashboard", help="open the dashboard")
    url_option(p)
    p.set_defaults(run=dashboard)

    p = sub.add_parser("demo", help="run the two-teams demo against a running gateway")
    url_option(p)
    p.set_defaults(run=demo)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
