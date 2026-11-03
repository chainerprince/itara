"""Shared pieces of the eval harness: items, baselines, and a gateway that runs in-process.

Every baseline goes through the real gateway code and the real ledger (so the spend cap
applies), against whatever models .env points at: the simulator (eval/sim.py), local
stand-ins, or the real models. Results are labelled with the model names so the three can
never be mixed up.

Each result also records what the model could see: whether the evidence was in the full
retrieved context (`in_context`, the ceiling) and in the memory the gateway retrieved
(`gold_in_hits`). That separates "retrieval missed" from "the model missed". Where markers
can't survive summarising (LoCoMo turns), `source_in_hits` asks the looser question: did
the gateway retrieve memory from every doc holding the evidence?

Answers are requested at temperature 0, so a difference in accuracy comes from what the
model was shown, not from sampling.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from gateway.accounting import Ledger
from gateway.config import Settings
from gateway.gate import ESCALATIONS
from gateway.server import Services, build_services, create_app
from memory.store import Memory, Scope

RESULTS = Path(__file__).parent.parent / "results" / "runs"
BASELINES = ("muse_direct", "escalate_compact", "escalate_raw", "llama_only", "rule_gate",
             "no_sharing", "full")
FORCED_ROUTES = {"llama_only": "local_memory", "escalate_compact": "escalate_compact",
                 "escalate_raw": "escalate_raw"}
COST_GUARD_USD = 5.0
ANSWER_TEMPERATURE = 0.0  # without it vLLM samples Llama 3.1 at its default 0.6


@dataclass
class Item:
    id: str
    team_id: str
    docs: list[tuple[str, str]]  # (source, text) added to the team's memory
    question: str                # what a user sends to the gateway
    raw_context: str             # Muse-direct sends this instead: the full retrieved context
    gold: object
    meta: dict = field(default_factory=dict)
    # What a model must see to answer: one list of acceptable markers per piece of evidence
    # (e.g. a gold file's path as it appears in context or in a memory source label).
    evidence: list[list[str]] = field(default_factory=list)
    # The docs (sources) that hold the evidence, where markers can't survive summarising.
    evidence_sources: list[str] = field(default_factory=list)


def evidence_visible(item: Item, text: str) -> bool:
    return bool(item.evidence) and all(any(m in text for m in alts) for alts in item.evidence)


def render_memory(memories: list[Memory]) -> str:
    """Memory as the gateway shows it to a model (see cache/lmcache.py with_memory)."""
    return "\n\n".join(f"[{m.source or 'memory'}] {m.content}" for m in memories)


@dataclass
class Result:
    item_id: str
    baseline: str
    user: str
    route: str
    answer: str
    score: dict
    run_id: str = ""
    in_context: bool = False             # evidence is in the full retrieved context (ceiling)
    gold_in_hits: bool | None = None     # evidence is in the memory the gateway retrieved
    source_in_hits: bool | None = None   # memory from every evidence doc was retrieved
    retrieved_sources: list[str] = field(default_factory=list)
    muse_prompt_tokens: int = 0
    muse_completion_tokens: int = 0
    llama_prompt_tokens: int = 0
    llama_completion_tokens: int = 0
    cached_tokens: int = 0
    compute_tokens: int = 0              # local model work: uncached prompt + completion tokens
    seconds: float = 0.0                 # model latency for the answer
    ingest_seconds: float = 0.0          # local compression time this request triggered
    ingest_pages: int = 0
    ingest_fallbacks: int = 0            # pages whose compression fell back to raw text


class Harness:
    """The gateway in-process, with one API key per simulated user."""

    def __init__(self, services: Services | None = None, run_id: str = "") -> None:
        self._given = services
        self.run_id = run_id
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

    async def ingest(self, item: Item, team: str, user: str) -> tuple[float, int, int]:
        """Add the item's docs to the team's memory unless already there.

        Returns seconds spent, pages stored and pages whose compression fell back.
        """
        seconds, pages, fallbacks = 0.0, 0, 0
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
                pages += len(r.json()["ids"])
                fallbacks += r.json().get("fallback_pages", 0)
        return seconds, pages, fallbacks

    async def ingest_many(self, items: list[Item], user: str,
                          report: Callable[[Item, tuple[float, int, int]], None] | None = None,
                          ) -> list[tuple[float, int, int]]:
        """ingest() for each item into its own team: one at a time, or with INGEST_CONCURRENCY
        above 1, several at once. The gateway's limit caps compression calls in flight; this
        only keeps enough docs queued to fill it. `report` is called as each item finishes."""
        n = self.services.ingest_concurrency
        if n <= 1:
            results = []
            for item in items:
                results.append(await self.ingest(item, item.team_id, user))
                if report:
                    report(item, results[-1])
            return results
        queued = asyncio.Semaphore(4 * n)

        async def one(item: Item) -> tuple[float, int, int]:
            async with queued:
                result = await self.ingest(item, item.team_id, user)
            if report:
                report(item, result)
            return result

        return list(await asyncio.gather(*(one(item) for item in items)))

    async def ingest_all(self, items: list[Item], user: str) -> None:
        """Fill memory for every item before answering any, so all memory baselines see the
        same memory (otherwise a baseline run first would see less than one run later)."""
        await self.ingest_many(items, user)

    async def ask(self, item: Item, baseline: str, user: str, team: str | None = None) -> Result:
        """Answer one item with one baseline (scoring is up to the benchmark)."""
        team = team or item.team_id
        in_context = evidence_visible(item, item.raw_context)
        if baseline == "muse_direct":
            response, seconds = await self.paid_call(
                Scope(team, user), "muse_direct",
                {"messages": [{"role": "user", "content": item.raw_context}],
                 "temperature": ANSWER_TEMPERATURE})
            result = self._result(item, baseline, user, "muse_direct", response, seconds)
            result.in_context = in_context
            return result
        if baseline == "full" and self.services.gate is None:
            raise SystemExit("full needs a learned gate: set GATE_PATH (see eval/learn_gate.py)")
        if baseline == "no_sharing":
            team = f"{team}:{user}"
        ingest_seconds, pages, fallbacks = await self.ingest(item, team, user)
        # The same search the gateway runs for this question, kept for diagnosis.
        hits = await self.services.store.search(Scope(team, user), item.question)
        headers = await self.headers(team, user)
        if baseline in FORCED_ROUTES:
            headers["X-Gateway-Route"] = FORCED_ROUTES[baseline]
        body = {"messages": [{"role": "user", "content": item.question}],
                "temperature": ANSWER_TEMPERATURE}
        start = time.monotonic()
        r = await self.gw.post("/v1/chat/completions", json=body, headers=headers)
        r.raise_for_status()
        result = self._result(item, baseline, user, r.headers["x-gateway-route"], r.json(),
                              time.monotonic() - start)
        result.in_context = in_context
        result.gold_in_hits = evidence_visible(item, render_memory(hits))
        result.retrieved_sources = [m.source or "" for m in hits]
        if item.evidence_sources:
            result.source_in_hits = set(item.evidence_sources) <= set(result.retrieved_sources)
        result.ingest_seconds, result.ingest_pages, result.ingest_fallbacks = (
            ingest_seconds, pages, fallbacks)
        return result

    async def paid_call(self, scope: Scope, route: str, body: dict) -> tuple[dict, float]:
        """Call Muse directly (Muse-direct baseline, judge) through the spend cap."""
        muse = self.services.muse
        estimate = self.ledger.worst_case_cost(body)
        async with self.ledger.reserve(estimate):
            start = time.monotonic()
            response = await muse.chat(body)
            latency_ms = response.pop("_replayed_latency_ms", None)
            if latency_ms is None:
                latency_ms = int((time.monotonic() - start) * 1000)
            await self.ledger.record(scope, muse.model, route, response.get("usage", {}),
                                     latency_ms, paid=True, estimate_usd=estimate)
        return response, latency_ms / 1000

    def _result(self, item, baseline, user, route, response, seconds) -> Result:
        usage = response.get("usage", {})
        prompt, completion = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        paid = route == "muse_direct" or route in ESCALATIONS
        return Result(
            item_id=item.id, baseline=baseline, user=user, route=route,
            answer=response["choices"][0]["message"].get("content") or "", score={},
            run_id=self.run_id,
            muse_prompt_tokens=prompt if paid else 0,
            muse_completion_tokens=completion if paid else 0,
            llama_prompt_tokens=0 if paid else prompt,
            llama_completion_tokens=0 if paid else completion,
            cached_tokens=0 if paid else cached,
            compute_tokens=0 if paid else prompt - cached + completion,
            seconds=seconds,
        )

    def labels(self) -> dict:
        s = self.services
        effort = getattr(s.muse, "defaults", {}).get("reasoning_effort", "")  # "": Muse's default
        return {"llama_model": s.llama.model, "muse_model": s.muse.model,
                "muse_is_standin": s.muse_is_standin, "muse_reasoning_effort": effort,
                "simulated": s.llama.model.startswith("sim-"),
                "ingest_concurrency": s.ingest_concurrency,
                "answer_temperature": ANSWER_TEMPERATURE,
                "replayed_calls": getattr(s.llama, "replayed", 0)
                + getattr(s.muse, "replayed", 0)}


class RunLog:
    """Per-item results appended as they finish, so a run can stop and resume."""

    def __init__(self, name: str, run_id: str) -> None:
        self.name, self.run_id = name, run_id
        self.path = RESULTS / "raw" / f"{name}_{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def results(self) -> list[Result]:
        if not self.path.exists():
            return []
        return [Result(**json.loads(line)) for line in self.path.read_text().splitlines()]

    def done(self) -> set[tuple[str, str]]:
        return {(r.item_id, r.user) for r in self.results()}

    def append(self, result: Result) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def write_summary(self, summary: dict) -> Path:
        path = RESULTS / f"{self.name}_{self.run_id}.json"
        path.write_text(json.dumps({"run_id": self.run_id, **summary}, indent=2) + "\n")
        return path


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


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
    diagnosed = [r for r in results if r.gold_in_hits is not None]
    sourced = [r for r in results if r.source_in_hits is not None]
    pages = total("ingest_pages")
    return {
        "items": len(results),
        "accuracy": sum(bool(r.score.get(headline)) for r in results) / n,
        **{f"mean_{k}": sum(float(r.score.get(k, 0)) for r in results) / n
           for k in scores if k != headline},
        "in_context_rate": total("in_context") / n,
        "gold_in_hits_rate": (sum(bool(r.gold_in_hits) for r in diagnosed) / len(diagnosed)
                              if diagnosed else None),
        "source_in_hits_rate": (sum(r.source_in_hits for r in sourced) / len(sourced)
                                if sourced else None),
        "muse_prompt_tokens": total("muse_prompt_tokens"),
        "muse_completion_tokens": total("muse_completion_tokens"),
        "llama_prompt_tokens": total("llama_prompt_tokens"),
        "cached_tokens": total("cached_tokens"),
        "compute_tokens": total("compute_tokens"),
        "llama_seconds": round(sum(r.seconds for r in results if not r.muse_prompt_tokens), 2),
        # Sum of each request's ingest time: only comparable at the same ingest_concurrency.
        "ingest_seconds": round(total("ingest_seconds"), 2),
        "ingest_pages": pages,
        "ingest_fallback_rate": total("ingest_fallbacks") / pages if pages else None,
        "routes": routes,
    }
