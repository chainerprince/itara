"""Llama's confidence in its drafts, for the learned gate (gateway/gate.py: draft_confidence).

    uv run python -m eval.confidence --id full1 -n 300

For every item of a full run: rebuild the exact llama_only request (the same memory search,
prompt and cache scope as the gateway's), draft it again at temperature 0 with logprobs,
then resample it and ask Llama whether the draft is right, the calls the live gateway makes.
Writes results/runs/raw/<bench>_confidence_<id>.jsonl, one row per item, and skips items
already there, so it can stop and resume. Llama only: no Muse, nothing against the cap.

At temperature 0 the new draft should be the run's llama_only answer; `same_as_run` records
whether it is (vLLM batching can change a token now and then). The gate's labels stay the
run's: whether the run's answer was right.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from cache.lmcache import scoped_body
from eval import locomo, swebench_loc
from eval.common import ANSWER_TEMPERATURE, RESULTS, RunLog
from gateway.config import Settings
from gateway.gate import draft_confidence
from gateway.llm import ChatClient
from memory.store import MemoryStore, Scope, default_embedder

BENCHES = ("swebench", "locomo")


def confidence_path(run_id: str, bench: str):
    return RESULTS / "raw" / f"{bench}_confidence_{run_id}.jsonl"


def load_confidence(run_id: str, bench: str) -> dict[str, dict]:
    """item id -> its row; {} if this run has none."""
    path = confidence_path(run_id, bench)
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text().splitlines())
    return {r["item_id"]: r for r in rows}


def usage_of(responses: list[dict]) -> dict[str, int]:
    """Local tokens of the extra calls, counted the way eval results count them."""
    prompt = completion = cached = 0
    for r in responses:
        usage = r.get("usage", {})
        prompt += usage.get("prompt_tokens", 0)
        completion += usage.get("completion_tokens", 0)
        cached += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return {"llama_prompt_tokens": prompt, "llama_completion_tokens": completion,
            "cached_tokens": cached, "compute_tokens": prompt - cached + completion}


async def main(args) -> None:
    settings = Settings()
    llama = ChatClient(settings.llama_base_url, settings.llama_api_key, settings.llama_model)
    url = args.database or f"{settings.database_url.rsplit('/', 1)[0]}/gateway_{args.id.replace('-', '_')}_full"
    limit = asyncio.Semaphore(args.concurrency)
    async with AsyncConnectionPool(url, configure=register_vector_async) as pool:
        store = MemoryStore(pool, default_embedder())
        for bench in BENCHES:
            items = (swebench_loc.load(args.n, args.seed) if bench == "swebench"
                     else locomo.load(args.n, args.seed))
            run = {r.item_id: r for r in
                   RunLog(f"{bench}_llama_only", f"{args.id}-{bench}-llama_only").results()}
            done = load_confidence(args.id, bench)
            todo = [i for i in items if i.id in run and i.id not in done]
            print(f"{bench}: {len(todo)} items to probe ({len(done)} done already)")
            path = confidence_path(args.id, bench)
            finished = 0
            start = time.monotonic()

            async def one(item):
                nonlocal finished
                async with limit:
                    scope = Scope(item.team_id, "confidence")
                    hits = await store.search(scope, item.question)
                    body = {"messages": [{"role": "user", "content": item.question}],
                            "temperature": ANSWER_TEMPERATURE}
                    local = scoped_body(body, scope, hits, settings.llama_backend)
                    calls = []

                    async def chat(b):
                        calls.append(await llama.chat(b))
                        return calls[-1]

                    draft = await llama.chat({**local, "logprobs": True})
                    features = await draft_confidence(chat, store.embed, local, draft)
                text = draft["choices"][0]["message"].get("content") or ""
                row = {"item_id": item.id, "bench": bench, "features": features, "draft": text,
                       "same_as_run": text == run[item.id].answer, **usage_of(calls)}
                with path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                finished += 1
                if finished % 50 == 0 or finished == len(todo):
                    print(f"  {finished}/{len(todo)}  {time.monotonic() - start:.0f}s")

            await asyncio.gather(*(one(item) for item in todo))
            rows = load_confidence(args.id, bench).values()
            if rows:
                same = sum(r["same_as_run"] for r in rows) / len(rows)
                print(f"{bench}: {len(rows)} probed; the new draft is the run's answer for {same:.0%}")
    await llama.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True, help="the FULL_ID of scripts/full_run.sh")
    parser.add_argument("-n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--database", help="default: gateway_<id>_full on DATABASE_URL's server")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="items in flight (Modal serves at most 32 requests at once)")
    asyncio.run(main(parser.parse_args()))
