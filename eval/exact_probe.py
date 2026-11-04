"""Sharper probes of the open model's drafts, on code: the exact draft, and a check with evidence.

    uv run python -m eval.exact_probe --id big1          # LLAMA_* pointing at that run's model

Added on 3 Nov, after the 70B run, as two fixes to eval/confidence.py's probes:

1. **The exact draft.** confidence.py drafts again before probing, and under batching the new
   draft differed from the scored one for 40% of the 70B's issues. Here the probes read the
   run's own answer: the yes/no self-check, and whether 4 resamples name the same first file.
   The draft's token log-probabilities are not re-scored: vLLM's prompt_logprobs computes the
   whole vocabulary for every prompt token, and on long prompts that ran the 70B out of GPU
   memory even one request at a time (they were the weakest signal, AUC 0.51).
2. **A check with evidence.** The self-check sees only the draft. This one shows the model the
   issue and the team memory's summaries of the file the draft names first (at the issue's
   commit), and asks whether that is the file to change. If memory has nothing on the file,
   the prompt says so.

Writes results/runs/raw/swebench_exact_probe_<id>.jsonl and skips items already there. Llama
only: no Muse. eval/simple_gate.py scores gates built on these features.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from cache.lmcache import scoped_body
from eval import swebench_loc
from eval.common import ANSWER_TEMPERATURE, RESULTS, RunLog
from eval.confidence import usage_of
from gateway.config import Settings
from gateway.gate import (CHECK_DRAFT_CHARS, SAMPLE_MAX_TOKENS, SAMPLE_TEMPERATURE, SAMPLES,
                          SELF_CHECK, first_line, yes_probability)
from gateway.llm import ChatClient
from memory.store import MemoryStore, Scope, default_embedder

EVIDENCE_CHARS = 3000
EVIDENCE = ("A teammate says the file to change for this issue is `{path}`.\n\n"
            "Issue:\n{issue}\n\nWhat the team memory says about `{path}`:\n{memory}\n\n"
            "Is `{path}` a file that must change to fix this issue? Reply with only yes or no.")


def path_of(run_id: str):
    return RESULTS / "raw" / f"swebench_exact_probe_{run_id}.jsonl"


def load_exact(run_id: str) -> dict[str, dict]:
    path = path_of(run_id)
    return ({r["item_id"]: r for r in map(json.loads, path.open())} if path.exists() else {})


def files(answer: str) -> list[str]:
    return [line.strip() for line in answer.splitlines() if line.strip()]


async def memory_about(store: MemoryStore, scope: Scope, path: str, commit: str | None) -> list[str]:
    """The team memory's summaries of one file, at the issue's commit if it has any there."""
    async with store.scoped(scope) as conn:
        cur = await conn.execute(
            "SELECT content, source FROM memories WHERE kind = 'summary' AND split_part(source, '@', 1) = %s"
            " ORDER BY (split_part(source, '@', 2) = %s) DESC, source, id LIMIT 4", (path, commit or ""))
        return [content for content, _ in await cur.fetchall()]


async def main(args) -> None:
    settings = Settings()
    llama = ChatClient(settings.llama_base_url, settings.llama_api_key, settings.llama_model,
                       defaults=settings.llama_defaults())
    url = args.database or f"{settings.database_url.rsplit('/', 1)[0]}/gateway_{args.id.replace('-', '_')}_full"
    limit = asyncio.Semaphore(args.concurrency)
    items = swebench_loc.load(args.n, args.seed)
    run = {r.item_id: r for r in RunLog("swebench_llama_only", f"{args.id}-swebench-llama_only").results()}
    done = load_exact(args.id)
    todo = [i for i in items if i.id in run and i.id not in done]
    print(f"swebench: {len(todo)} items to probe ({len(done)} done already)")
    finished, start = 0, time.monotonic()
    async with AsyncConnectionPool(url, configure=register_vector_async) as pool:
        store = MemoryStore(pool, default_embedder())

        async def one(item):
            nonlocal finished
            async with limit:
                result = run[item.id]
                draft = result.answer
                scope = Scope(item.team_id, "confidence")
                hits = await store.search(scope, item.question)
                local = scoped_body({"messages": [{"role": "user", "content": item.question}],
                                     "temperature": ANSWER_TEMPERATURE}, scope, hits, settings.llama_backend)
                named = files(draft)
                commit = next((s.split("@", 1)[1] for s in result.retrieved_sources if "@" in s), None)
                memory = await memory_about(store, scope, named[0], commit) if named else []
                evidence = "\n".join(f"- {m}" for m in memory)[:EVIDENCE_CHARS] or "(nothing: no team memory mentions this file)"
                as_turn = [*local["messages"], {"role": "assistant", "content": draft[:CHECK_DRAFT_CHARS]}]
                samples, check, grounded = await asyncio.gather(
                    llama.chat({**local, "n": SAMPLES, "temperature": SAMPLE_TEMPERATURE,
                                "max_tokens": min(SAMPLE_MAX_TOKENS, max(64, 2 * max(1, result.llama_completion_tokens)))}),
                    llama.chat({**local, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
                                "messages": [*as_turn, {"role": "user", "content": SELF_CHECK}]}),
                    llama.chat({**local, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
                                "messages": [{"role": "user", "content": EVIDENCE.format(
                                    path=named[0] if named else "(none)", issue=item.question, memory=evidence)}]}))
            texts = [c["message"].get("content") or "" for c in samples.get("choices", [])]
            features = {
                "exact_self_check": yes_probability(check),
                "exact_same_answer": (sum(first_line(t) == first_line(draft) for t in texts) / len(texts)
                                      if texts else 1.0),
                "evidence_check": yes_probability(grounded),
                "evidence_found": float(bool(memory)),
            }
            row = {"item_id": item.id, "features": features, **usage_of([samples, check, grounded])}
            with path_of(args.id).open("a") as f:
                f.write(json.dumps(row) + "\n")
            finished += 1
            if finished % 50 == 0 or finished == len(todo):
                print(f"  {finished}/{len(todo)}  {time.monotonic() - start:.0f}s")

        await asyncio.gather(*(one(item) for item in todo))
    await llama.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True, help="the run whose drafts to probe (full1, big1)")
    parser.add_argument("-n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--database", help="default: gateway_<id>_full on DATABASE_URL's server")
    parser.add_argument("--concurrency", type=int, default=16)
    asyncio.run(main(parser.parse_args()))
