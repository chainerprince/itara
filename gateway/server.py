"""OpenAI-compatible gateway: point any app's base URL at /v1.

Callers authenticate with `Authorization: Bearer <gateway key>` (`itara key add`), which
every OpenAI SDK already sends; the key decides the caller's team and user. Clients that ask
for streaming get the whole answer as a single chunk. Run with a single worker: the spend
reservation lives in this process.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from cache.lmcache import cache_scope, scoped_body, with_memory
from gateway.accounting import Ledger, SpendCapExceeded
from gateway.auth import Principal, Principals
from gateway.config import Settings
from gateway.gate import (ESCALATIONS, FEATURES, CascadeGate, LearnedGate, Route,
                          draft_confidence, estimated_prompt_tokens, gate_features, has_tools,
                          load_gate, rule_gate)
from gateway.llm import ChatClient
from memory.compress import ingest
from memory.store import MemoryStore, Scope, default_embedder

DASHBOARD = Path(__file__).parent.parent / "dashboard"
# The gateway request every model call belongs to, so the ledger can group a question's
# draft, probes and escalation (the probes run in tasks, which copy this context).
REQUEST_ID: ContextVar[UUID | None] = ContextVar("request_id", default=None)


@dataclass
class Services:
    store: MemoryStore
    ledger: Ledger
    llama: ChatClient
    muse: ChatClient
    principals: Principals
    llama_backend: str = "ollama"
    muse_is_standin: bool = False
    ingest_concurrency: int = 1
    gate: LearnedGate | CascadeGate | None = None  # None: the rule-based gate
    always_escalate: bool = False  # every question to Muse with memory summaries, no gate
    # One limit for the whole process, so concurrent ingests can't multiply it.
    ingest_limit: asyncio.Semaphore | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.ingest_concurrency > 1:
            self.ingest_limit = asyncio.Semaphore(self.ingest_concurrency)


async def build_services(settings: Settings) -> tuple[Services, AsyncConnectionPool]:
    pool = AsyncConnectionPool(settings.database_url, open=False, configure=register_vector_async)
    await pool.open()
    services = Services(
        store=MemoryStore(pool, default_embedder()),
        ledger=Ledger(pool, settings.spend_cap_usd,
                      settings.muse_input_price_per_m, settings.muse_output_price_per_m),
        llama=ChatClient(settings.llama_base_url, settings.llama_api_key, settings.llama_model,
                         cache_path=settings.model_cache, defaults=settings.llama_defaults()),
        muse=ChatClient(settings.muse_base_url, settings.muse_api_key, settings.muse_model,
                        cache_path=settings.model_cache,
                        defaults=({"reasoning_effort": settings.muse_reasoning_effort}
                                  if settings.muse_reasoning_effort else None)),
        principals=Principals(pool),
        llama_backend=settings.llama_backend,
        muse_is_standin=settings.muse_is_standin,
        ingest_concurrency=settings.ingest_concurrency,
        gate=load_gate(settings.gate_path) if settings.gate_path and not settings.always_escalate else None,
        always_escalate=settings.always_escalate,
    )
    if isinstance(services.gate, CascadeGate) and settings.gate_threshold is not None:
        services.gate.threshold = settings.gate_threshold
    return services, pool


class MemoryIn(BaseModel):
    text: str
    source: str | None = None
    private: bool = False


async def principal_of(request: Request) -> Principal:
    scheme, _, key = request.headers.get("authorization", "").partition(" ")
    principal = None
    if scheme.lower() == "bearer" and key:
        principal = await request.app.state.services.principals.lookup(key.strip())
    if principal is None:
        raise HTTPException(401, "A valid gateway API key is required (Authorization: Bearer ...)")
    return principal


def route_override(request: Request, principal: Principal) -> Route | None:
    """X-Gateway-Route forces a route, for running baselines and debugging. Admin keys only."""
    value = request.headers.get("x-gateway-route")
    if value is None:
        return None
    if not principal.is_admin:
        raise HTTPException(403, "X-Gateway-Route needs an admin key")
    if value not in Route.__members__.values() or value == Route.PASSTHROUGH:
        raise HTTPException(400, f"Unknown X-Gateway-Route: {value}")
    return Route(value)


def last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content") or ""
            if isinstance(content, list):  # content parts
                return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
            return content
    return ""


def as_sse(response: dict) -> StreamingResponse:
    """Replay a complete response as a one-chunk stream."""
    choices = []
    for choice in response.get("choices", []):
        message = choice.get("message", {})
        delta = {"role": "assistant", "content": message.get("content")}
        if message.get("tool_calls"):
            delta["tool_calls"] = [{**c, "index": i} for i, c in enumerate(message["tool_calls"])]
        choices.append({"index": choice.get("index", 0), "delta": delta,
                        "finish_reason": choice.get("finish_reason")})
    chunk = {**response, "object": "chat.completion.chunk", "choices": choices}
    chunk.pop("usage", None)

    def events():
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if services is not None:
            app.state.services = services
            yield
            return
        app.state.services, pool = await build_services(Settings())
        yield
        await app.state.services.llama.close()
        await app.state.services.muse.close()
        await pool.close()

    app = FastAPI(title="Itara", lifespan=lifespan)

    @app.exception_handler(SpendCapExceeded)
    async def spend_cap(request: Request, exc: SpendCapExceeded):
        return JSONResponse({"error": {"message": str(exc), "type": "spend_cap"}}, status_code=402)

    @app.exception_handler(httpx.HTTPStatusError)
    async def upstream_error(request: Request, exc: httpx.HTTPStatusError):
        # Keep rate limits as 429; anything else is the model's failure, not the caller's
        # (an upstream 401 must not look like the caller's gateway key was rejected).
        status = exc.response.status_code
        return JSONResponse(
            {"error": {"message": f"Model returned {status}: {exc.response.text}",
                       "type": "upstream_error"}},
            status_code=429 if status == 429 else 502,
        )

    @app.exception_handler(httpx.RequestError)
    async def upstream_unreachable(request: Request, exc: httpx.RequestError):
        return JSONResponse(
            {"error": {"message": f"Model unreachable: {exc!r}", "type": "upstream_error"}},
            status_code=502,
        )

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "itara", "object": "model"}]}

    async def prepare(s: Services, scope, route: Route, body: dict, hits: list) -> dict:
        """The request as the route's model gets it: with its memory (and cache scope)."""
        if route == Route.PASSTHROUGH:
            return body
        if route == Route.ESCALATE_RAW:
            memories = await s.store.pages(scope, [h.id for h in hits])
        elif route in (Route.LOCAL_MEMORY, Route.ESCALATE_COMPACT):
            memories = hits
        else:
            memories = []
        if route in ESCALATIONS:  # Muse gets memory but never the cache scope
            return {**body, "messages": with_memory(body["messages"], memories)}
        return scoped_body(body, scope, memories, s.llama_backend)

    async def answer(s: Services, scope, route: Route, body: dict, hits: list) -> dict:
        """Send the request down one route, with its memory, and record the usage."""
        return await send(s, scope, route, await prepare(s, scope, route, body, hits))

    async def send(s: Services, scope, route: str, body: dict) -> dict:
        """Call the route's model with a prepared request and record the usage."""
        paid = route in ESCALATIONS
        client = s.muse if paid else s.llama
        estimate = s.ledger.worst_case_cost(body) if paid else 0.0
        async with s.ledger.reserve(estimate) if paid else nullcontext():
            start = time.monotonic()
            response = await client.chat(body)
            # A response replayed from the model cache keeps the latency of the real call.
            latency_ms = response.pop("_replayed_latency_ms", None)
            if latency_ms is None:
                latency_ms = int((time.monotonic() - start) * 1000)
            await s.ledger.record(scope, client.model, route, response.get("usage", {}),
                                  latency_ms, paid=paid, estimate_usd=estimate,
                                  request_id=REQUEST_ID.get())
        return response

    async def learned_route(s: Services, scope, question: str, hits: list, local: dict,
                            draft: dict) -> Route:
        """`local`: the draft's request as sent; `draft`: Llama's answer to it."""
        text = draft["choices"][0]["message"].get("content") or ""
        features = {}
        if any(f in FEATURES for f in s.gate.features):
            vectors = await s.store.embed([question, text, *(h.content for h in hits)])
            features = gate_features(question, hits, text, vectors)
        if s.gate.needs_confidence:  # local GPU work, recorded as its own route
            features |= await draft_confidence(lambda b: send(s, scope, "gate_probe", b),
                                               s.store.embed, local, draft)
        if isinstance(s.gate, CascadeGate):
            return s.gate.decide(s.gate.p_right(features), hits)
        pages = (await s.store.pages(scope, [h.id for h in hits])
                 if Route.ESCALATE_RAW in s.gate.routes else [])
        probabilities = s.gate.probabilities(features)
        costs = s.gate.costs(estimated_prompt_tokens(question, hits, pages), s.ledger.cost)
        return s.gate.decide(probabilities, costs)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        s: Services = request.app.state.services
        principal = await principal_of(request)
        scope = principal.scope
        REQUEST_ID.set(uuid4())
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            raise HTTPException(400, "Request body needs a messages list")

        if has_tools(body):
            route = Route.PASSTHROUGH
            response = await answer(s, scope, route, body, [])
        else:
            question = last_user_text(body["messages"])
            hits = await s.store.search(scope, question)
            route = route_override(request, principal)
            if route is None and s.always_escalate:
                route = Route.ESCALATE_COMPACT
            if route is None and s.gate is not None:
                # Cascade: Llama drafts with memory; the gate keeps the draft or escalates.
                local = await prepare(s, scope, Route.LOCAL_MEMORY, body, hits)
                logprobs = s.gate.needs_confidence and not body.get("logprobs")
                response = await send(s, scope, Route.LOCAL_MEMORY,
                                      {**local, "logprobs": True} if logprobs else local)
                route = await learned_route(s, scope, question, hits, local, response)
                if route != Route.LOCAL_MEMORY:
                    response = await answer(s, scope, route, body, hits)
                elif logprobs:  # asked for the gate, not by the caller
                    for choice in response.get("choices", []):
                        choice.pop("logprobs", None)
            else:
                route = route or rule_gate(question, hits)
                response = await answer(s, scope, route, body, hits)

        headers = {"X-Gateway-Route": route.value}
        if route != Route.PASSTHROUGH:  # how much team memory the answer drew on
            headers["X-Gateway-Memory-Hits"] = str(len(hits))
        if body.get("stream"):
            sse = as_sse(response)
            sse.headers.update(headers)
            return sse
        return JSONResponse(response, headers=headers)

    @app.post("/v1/memory")
    async def add_memory(item: MemoryIn, request: Request):
        """Add text (docs, code, decisions) to memory: team-shared, or private to the caller."""
        s: Services = request.app.state.services
        scope = (await principal_of(request)).scope
        request_id = uuid4()
        start = time.monotonic()
        ids, usages, fallbacks = await ingest(s.store, s.llama, scope, item.text,
                                              source=item.source, private=item.private,
                                              limit=s.ingest_limit)
        # Wall time per call: with concurrent ingest that's throughput, not one call's latency.
        latency_ms = int((time.monotonic() - start) * 1000) // max(1, len(usages))
        for page_usage in usages:  # compression is local work the team pays for once
            await s.ledger.record(scope, s.llama.model, "ingest", page_usage, latency_ms,
                                  paid=False, request_id=request_id)
        return {"ids": [str(i) for i in ids], "fallback_pages": fallbacks}

    @app.get("/v1/usage")
    async def usage(request: Request):
        """Company-wide spend. Admin keys only."""
        if not (await principal_of(request)).is_admin:
            raise HTTPException(403, "/v1/usage needs an admin key")
        s: Services = request.app.state.services
        requests = await s.ledger.requests([r.value for r in ESCALATIONS])
        teams = sorted({r["team_id"] for r in requests["by_user_route"]})
        return {"spent_usd": await s.ledger.spent(), "cap_usd": s.ledger.cap_usd,
                "muse_model": s.muse.model, "muse_is_standin": s.muse_is_standin,
                "llama_model": s.llama.model, "llama_backend": s.llama_backend,
                "gate": ("always escalate" if s.always_escalate
                         else type(s.gate).__name__ if s.gate else "rule"),
                # The KV cache key each team's shared memory is cached under (cache/lmcache.py).
                "cache_scopes": {t: cache_scope(Scope(t, ""), []) for t in teams},
                "by_model_route": await s.ledger.summary(), **requests}

    @app.get("/dashboard/results.json")
    async def dashboard_results():
        """The benchmark results the dashboard shows (scripts/dashboard_data.py)."""
        return FileResponse(DASHBOARD / "public" / "results.json")

    @app.get("/dashboard")
    @app.get("/dashboard/{path:path}")
    async def dashboard(path: str = ""):
        """The dashboard app (dashboard/, built to dashboard/dist). It asks for an admin key and
        reads /v1/usage. Paths that aren't files are the app's own pages."""
        dist = (DASHBOARD / "dist").resolve()
        file = (dist / path).resolve()
        if path and file.is_file() and file.is_relative_to(dist):
            return FileResponse(file)
        if not (dist / "index.html").is_file():
            raise HTTPException(404, "The dashboard isn't built: cd dashboard && npm install && npm run build")
        return FileResponse(dist / "index.html")

    return app


app = create_app()
