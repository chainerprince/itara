"""Shared pieces of the eval harness: items, baselines, and a gateway that runs in-process.

Every baseline goes through the real gateway code and the real ledger (so the spend cap
applies), against whatever models .env points at. Results are labelled with the model
names and whether "Muse" is a local stand-in.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from gateway.accounting import Ledger
from gateway.config import Settings
from gateway.gate import ESCALATIONS
from gateway.server import Services, build_services, create_app
from memory.store import Scope

RESULTS = Path(__file__).parent / "results"
BASELINES = ("muse_direct", "llama_only", "rule_gate", "no_sharing", "full")
COST_GUARD_USD = 5.0


@dataclass
class Item:
    id: str
    team_id: str
    docs: list[tuple[str, str]]  # (source, text) added to the team's memory
    question: str                # what a user sends to the gateway
    raw_context: str             # Muse-direct sends this instead: the full retrieved context
    gold: object
    meta: dict = field(default_factory=dict)


@dataclass
class Result:
    item_id: str
    baseline: str
    user: str
    route: str
    answer: str
    score: dict
    muse_prompt_tokens: int = 0
    muse_completion_tokens: int = 0
    llama_prompt_tokens: int = 0
    llama_completion_tokens: int = 0
    cached_tokens: int = 0
    seconds: float = 0.0          # model latency for the answer
    ingest_seconds: float = 0.0   # local compression time this request triggered


class Harness:
    """The gateway in-process, with one API key per simulated user."""

    def __init__(self, services: Services | None = None) -> None:
        self._given = services
        self._keys: dict[tuple[str, str], str] = {}
        self._ingest_locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def __aenter__(self) -> Harness:
        if self._given is not None:
            self.services, self._pool = self._given, None
        else:
            self.services, self._pool = await build_services(Settings())
        app = create_app(self.services)
        self._lifespan = app.router.lifespan_context(app)
        await self._lifespan.__aenter__()
        self.gw = httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://gw", timeout=1800)
        return self

    async def __aexit__(self, *exc) -> None:
        await self.gw.aclose()
        await self._lifespan.__aexit__(*exc)
        if self._pool is not None:
            await self.services.llama.close()
            await self.services.muse.close()
            await self._pool.close()

    @property
    def ledger(self) -> Ledger:
        return self.services.ledger

    async def headers(self, team: str, user: str) -> dict:
        # Eval keys are admin so baselines can force a route.
        if (team, user) not in self._keys:
            self._keys[team, user] = await self.services.principals.create(team, user, True)
        return {"Authorization": f"Bearer {self._keys[team, user]}"}

    async def ingest(self, item: Item, team: str, user: str) -> float:
        """Add the item's docs to the team's memory unless already there. Returns seconds spent."""
        seconds = 0.0
        for source, text in item.docs:
            lock = self._ingest_locks.setdefault((team, source), asyncio.Lock())
            async with lock:  # concurrent teammates ingest a shared doc once
                if await self.services.store.has_source(Scope(team, user), source):
                    continue
                start = time.monotonic()
                r = await self.gw.post("/v1/memory", json={"text": text, "source": source},
                                       headers=await self.headers(team, user))
                r.raise_for_status()
                seconds += time.monotonic() - start
        return seconds

    async def ask(self, item: Item, baseline: str, user: str, team: str | None = None) -> Result:
        """Answer one item with one baseline (scoring is up to the benchmark)."""
        team = team or item.team_id
        if baseline == "muse_direct":
            response, seconds = await self.paid_call(
                Scope(team, user), "muse_direct",
                {"messages": [{"role": "user", "content": item.raw_context}]})
            return self._result(item, baseline, user, "muse_direct", response, seconds, 0.0)
        if baseline == "full":
            raise NotImplementedError("the learned gate arrives on day 7")
        if baseline == "no_sharing":
            team = f"{team}:{user}"
        ingest_seconds = await self.ingest(item, team, user)
        headers = await self.headers(team, user)
        if baseline == "llama_only":
            headers["X-Gateway-Route"] = "local_memory"
        body = {"messages": [{"role": "user", "content": item.question}]}
        start = time.monotonic()
        r = await self.gw.post("/v1/chat/completions", json=body, headers=headers)
        r.raise_for_status()
        return self._result(item, baseline, user, r.headers["x-gateway-route"], r.json(),
                            time.monotonic() - start, ingest_seconds)

    async def paid_call(self, scope: Scope, route: str, body: dict) -> tuple[dict, float]:
        """Call Muse directly (Muse-direct baseline, judge) through the spend cap."""
        muse = self.services.muse
        estimate = self.ledger.worst_case_cost(body)
        async with self.ledger.reserve(estimate):
            start = time.monotonic()
            response = await muse.chat(body)
            seconds = time.monotonic() - start
            await self.ledger.record(scope, muse.model, route, response.get("usage", {}),
                                     int(seconds * 1000), paid=True, estimate_usd=estimate)
        return response, seconds

    @staticmethod
    def _result(item, baseline, user, route, response, seconds, ingest_seconds) -> Result:
        usage = response.get("usage", {})
        prompt, completion = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        paid = route == "muse_direct" or route in ESCALATIONS
        return Result(
            item_id=item.id, baseline=baseline, user=user, route=route,
            answer=response["choices"][0]["message"].get("content") or "", score={},
            muse_prompt_tokens=prompt if paid else 0,
            muse_completion_tokens=completion if paid else 0,
            llama_prompt_tokens=0 if paid else prompt,
            llama_completion_tokens=0 if paid else completion,
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            seconds=seconds, ingest_seconds=ingest_seconds,
        )

    def labels(self) -> dict:
        s = self.services
        return {"llama_model": s.llama.model, "muse_model": s.muse.model,
                "muse_is_standin": s.muse_is_standin}


def check_cost(ledger: Ledger, bodies: list[dict], yes: bool) -> float:
    """Worst-case Muse cost if every body went to Muse; refuse large runs without --yes."""
    estimate = sum(ledger.worst_case_cost(b) for b in bodies)
    print(f"Worst-case Muse cost: ${estimate:.2f}")
    if estimate > COST_GUARD_USD and not yes:
        raise SystemExit(f"Over ${COST_GUARD_USD:.0f}: rerun with --yes to proceed")
    return estimate


def summarise(results: list[Result], headline: str) -> dict:
    n = len(results) or 1
    total = lambda attr: sum(getattr(r, attr) for r in results)  # noqa: E731
    routes: dict[str, int] = {}
    for r in results:
        routes[r.route] = routes.get(r.route, 0) + 1
    scores = sorted({k for r in results for k, v in r.score.items()
                     if isinstance(v, (bool, int, float))})
    return {
        "items": len(results),
        "accuracy": sum(bool(r.score.get(headline)) for r in results) / n,
        **{f"mean_{k}": sum(float(r.score.get(k, 0)) for r in results) / n
           for k in scores if k != headline},
        "muse_prompt_tokens": total("muse_prompt_tokens"),
        "muse_completion_tokens": total("muse_completion_tokens"),
        "llama_prompt_tokens": total("llama_prompt_tokens"),
        "cached_tokens": total("cached_tokens"),
        "llama_seconds": round(total("seconds") - sum(
            r.seconds for r in results if r.muse_prompt_tokens), 2),
        "ingest_seconds": round(total("ingest_seconds"), 2),
        "routes": routes,
    }


def write(name: str, results: list[Result], summary: dict) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "raw" / f"{name}_{stamp}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    path = RESULTS / f"{name}_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path
