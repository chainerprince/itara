"""The demo: two engineers on one team share memory, a third on another team can't see it.

    export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/gateway_demo
    uv run python -m scripts.init_db --create
    GATE_PATH=results/runs/gate_full1_model.json uv run uvicorn gateway.server:app --port 8000
    uv run python -m scripts.demo                  # in another shell, same DATABASE_URL

Or with the CLI: `uv run itara up --gate results/runs/gate_full1_model.json`, then `uv run itara demo`.
The dashboard is at http://localhost:8000/dashboard; it asks for an admin key
(uv run itara key add --team web --user you --admin) and updates every 3 s.

It makes fresh keys for alice and bob (team web), olga (team ops) and an admin (team web),
straight in the database; they are never printed. Then:
1. alice adds the team's code (this repo's gateway/, memory/ and cache/ packages, labelled
   with a hash of each file, as `itara memory add` does) and the team's notes
   (demo/web_team_notes.md) to web's memory. The gateway compresses each page once: on a
   rerun every page is already there, and the output says so.
2. The scripted questions run as each engineer, and each answer shows the route the gateway
   took (X-Gateway-Route), the model, its tokens and the time it took. One comes from a coding
   agent, with tools attached: it skips the gate but still gets the team's memory.
3. The admin reads /v1/usage: what Muse cost, and what Llama did.

With the cascade gate (GATE_PATH), a question Llama is sure of stays on Llama; one it isn't
goes to Muse with the team's memory: summaries for code, raw pages for the notes. Without
GATE_PATH the rule-based gate decides. Answers depend on the models, so the routes may
differ from the comments below; the script prints what actually happened.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import time
from pathlib import Path

import httpx
from psycopg_pool import AsyncConnectionPool

from gateway.auth import Principals
from gateway.config import Settings

ROOT = Path(__file__).parent.parent
CODE = ["gateway", "memory", "cache"]
NOTES = ROOT / "demo" / "web_team_notes.md"

# (engineer, question, what it shows)
SCRIPT = [
    ("alice", "Where does the gateway enforce the Muse spend cap, and what HTTP status does a "
              "caller get when it is reached?",
     "a question the team's code answers: Llama with shared memory"),
    ("bob", "What status code comes back when we hit the spend cap, and who signs off on "
            "raising it?",
     "a teammate's related question: the same memory, and Llama's cached prefix"),
    ("bob", "Design per-team spend caps instead of one company-wide cap. Which functions, "
            "tables and SQL queries have to change, and how do concurrent reservations stay "
            "correct?",
     "a hard design question: the gate should send it to Muse with compact memory"),
    ("bob", "When does the Kestrel release ship, who owns it, and when do we roll it back?",
     "a decision recorded only in the team's notes: no model knows it on its own"),
    ("bob", "Open the file where the gateway reserves Muse spend before a call.",
     "a coding agent, with tools attached: no gate, but the team's memory still goes with it"),
    ("olga", "When does the web team's Kestrel release ship, and when do they roll it back?",
     "another team: none of web's memory is visible to ops"),
]
# The tool a coding agent sends with its request (the gateway passes tools through unchanged).
AGENT_TOOLS = [{"type": "function", "function": {
    "name": "open_file", "description": "Open a file in the repository",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}]


def team_docs() -> list[tuple[str, str]]:
    """(source, text): code sources end in @<hash of the file>, like the SWE-bench memory."""
    docs = [(f"{p.relative_to(ROOT)}@{hashlib.sha256(p.read_bytes()).hexdigest()[:8]}",
             p.read_text())
            for package in CODE for p in sorted((ROOT / package).glob("*.py"))
            if p.read_text().strip()]
    return docs + [(f"{NOTES.relative_to(ROOT)}", NOTES.read_text())]


def show(who: str, why: str, question: str, r: httpx.Response, seconds: float) -> None:
    body = r.json()
    usage = body.get("usage", {})
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    message = body["choices"][0]["message"]
    answer = " ".join((message.get("content") or "").split())
    if message.get("tool_calls"):
        answer = (answer + " " if answer else "") + "tool call: " + ", ".join(
            f"{c['function']['name']}({c['function']['arguments']})" for c in message["tool_calls"])
    print(f"\n── {who}: {why}")
    print(f"   Q: {question}")
    print(f"   route {r.headers.get('x-gateway-route')} · {r.headers.get('x-gateway-memory-hits', 0)} "
          f"memories · model {body.get('model')} · "
          f"{usage.get('prompt_tokens', 0):,} prompt"
          + (f" ({cached:,} cached)" if cached else "")
          + f" + {usage.get('completion_tokens', 0):,} completion tokens · {seconds:.1f}s")
    print(f"   A: {answer[:400]}{'…' if len(answer) > 400 else ''}")


async def main(args) -> None:
    settings = Settings()
    async with AsyncConnectionPool(settings.database_url) as pool:
        principals = Principals(pool)
        keys = {"alice": await principals.create("web", "alice"),
                "bob": await principals.create("web", "bob"),
                "olga": await principals.create("ops", "olga"),
                "admin": await principals.create("web", "demo-admin", is_admin=True)}
    docs = team_docs()

    def auth(who: str) -> dict:
        return {"Authorization": f"Bearer {keys[who]}"}

    async with httpx.AsyncClient(base_url=args.url, timeout=900) as gw:
        print(f"Adding {len(docs)} docs to team web's memory (alice)…")
        start = time.monotonic()
        limit = asyncio.Semaphore(8)  # docs in flight; the gateway caps compression calls

        async def add(source: str, text: str) -> tuple[int, int]:
            async with limit:
                r = await gw.post("/memory", json={"text": text, "source": source,
                                                   "replace": True}, headers=auth("alice"))
            r.raise_for_status()
            return len(r.json()["ids"]), r.json().get("reused_pages", 0)

        counts = await asyncio.gather(*(add(s, t) for s, t in docs))
        pages, reused = sum(c[0] for c in counts), sum(c[1] for c in counts)
        print(f"  {pages} memory pages in {time.monotonic() - start:.0f}s: {pages - reused} compressed "
              f"now, {reused} already in memory (each page is compressed once, for the team)")

        for who, question, why in SCRIPT:
            agent = "tools attached" in why
            start = time.monotonic()
            r = await gw.post("/chat/completions", headers=auth(who), json={
                "model": "itara", "messages": [{"role": "user", "content": question}],
                "temperature": 0, **({"tools": AGENT_TOOLS} if agent else {})})
            r.raise_for_status()
            show(who, why, question, r, time.monotonic() - start)

        usage = (await gw.get("/usage", headers=auth("admin"))).json()
    print(f"\n── Company usage (admin): ${usage['spent_usd']:.4f} of Muse "
          f"(cap ${usage['cap_usd']:.0f}), {usage['muse_model']}"
          + (" (a local stand-in)" if usage["muse_is_standin"] else ""))
    print(f"   {'model':34}{'route':18}{'requests':>9}{'prompt':>10}{'cached':>9}{'completion':>12}{'$':>9}")
    for row in usage["by_model_route"]:
        print(f"   {row['model'][:33]:34}{row['route']:18}{row['requests']:>9}{row['prompt_tokens']:>10,}"
              f"{row['cached_tokens']:>9,}{row['completion_tokens']:>12,}{row['cost_usd']:>9.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000/v1", help="the gateway's /v1")
    asyncio.run(main(parser.parse_args()))
