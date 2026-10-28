"""Day 4 check: a teammate's repeat request is faster; another team's isn't.

Puts the same ~6K-token document in two teams' memory, then asks the same question as
alice@web (cold), bob@web (same scope: should reuse alice's KV cache) and carol@ops (same
text, different scope: must not). The document goes in as large summaries directly, skipping
compression, so the shared prefix is big enough to time.

    uv run python -m scripts.cache_check
"""

import asyncio
import time

import httpx

from gateway.config import Settings
from gateway.server import build_services, create_app
from memory.store import Scope

SECTION = (
    "Service {n} handles {topic}. It is deployed from the main branch by the release pipeline, "
    "reads its configuration from config/{topic}.yaml, retries failed jobs three times with "
    "exponential backoff, and pages the on-call engineer after five consecutive failures. "
)
TOPICS = ["billing", "refunds", "search", "auth", "email", "exports"]
QUESTION = "Summarise how the billing refunds search auth email exports services retry jobs."


def doc_section(topic: str) -> str:
    return " ".join(SECTION.format(n=i, topic=topic) for i in range(25))


async def main() -> None:
    settings = Settings()
    services, pool = await build_services(settings)
    run = str(int(time.time()))
    web, ops = f"web-{run}", f"ops-{run}"
    for team in (web, ops):
        for topic in TOPICS:
            await services.store.add(Scope(team, "setup"), doc_section(topic), [topic],
                                     doc_section(topic), source=f"{topic}.md")
    users = [("warm-up", f"warmup-{run}", "w"), ("alice@web", web, "alice"),
             ("bob@web", web, "bob"), ("carol@ops", ops, "carol")]
    keys = {label: await services.principals.create(team, user, is_admin=True)
            for label, team, user in users}

    app = create_app(services)
    rows = []
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw", timeout=600) as gw:
            for label, _, _ in users:
                body = {"messages": [{"role": "user", "content": QUESTION}], "max_tokens": 16}
                start = time.monotonic()
                r = await gw.post("/v1/chat/completions", json=body, headers={
                    "Authorization": f"Bearer {keys[label]}", "X-Gateway-Route": "local_memory"})
                r.raise_for_status()
                usage = r.json().get("usage", {})
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                rows.append((label, time.monotonic() - start, usage.get("prompt_tokens"), cached))
    await services.llama.close()
    await services.muse.close()
    await pool.close()

    print(f"backend: {settings.llama_backend} ({settings.llama_model})\n")
    print(f"{'caller':<11}{'seconds':>9}{'prompt':>8}{'cached':>8}")
    for label, seconds, prompt, cached in rows:
        print(f"{label:<11}{seconds:>9.2f}{prompt or 0:>8}{'n/a' if cached is None else cached:>8}")
    _, alice, _, _ = rows[1]
    _, bob, bob_prompt, bob_cached = rows[2]
    _, carol, _, carol_cached = rows[3]
    print(f"\nbob vs alice: {alice / bob:.1f}x faster; carol vs bob: {carol / bob:.1f}x slower")
    if bob_cached is None:
        print("This backend doesn't report cached tokens: isolation is shown by latency only.")
        return
    # Carol may reuse the chat template's few header tokens, never the team's content.
    assert bob_cached > bob_prompt // 2, "expected bob to reuse alice's cached prefix"
    assert carol_cached < 64, "carol reused another team's cache"
    print(f"cached tokens: bob reused {bob_cached}/{bob_prompt}; carol only {carol_cached} "
          "(chat template header): isolation confirmed")

if __name__ == "__main__":
    asyncio.run(main())
