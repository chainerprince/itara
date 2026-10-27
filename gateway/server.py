"""OpenAI-compatible gateway: point any app's base URL at /v1.

Callers identify themselves with X-Team-Id and X-User-Id headers. Clients that ask for
streaming get the whole answer as a single chunk.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from gateway.accounting import Ledger, SpendCapExceeded
from gateway.config import Settings
from gateway.gate import ESCALATIONS, Route, has_tools, rule_gate
from gateway.llm import ChatClient
from memory.compress import ingest
from memory.store import Memory, MemoryStore, Scope, default_embedder


@dataclass
class Services:
    store: MemoryStore
    ledger: Ledger
    llama: ChatClient
    muse: ChatClient


async def build_services(settings: Settings) -> tuple[Services, AsyncConnectionPool]:
    pool = AsyncConnectionPool(settings.database_url, open=False, configure=register_vector_async)
    await pool.open()
    services = Services(
        store=MemoryStore(pool, default_embedder()),
        ledger=Ledger(pool, settings.spend_cap_usd,
                      settings.muse_input_price_per_m, settings.muse_output_price_per_m),
        llama=ChatClient(settings.llama_base_url, settings.llama_api_key, settings.llama_model),
        muse=ChatClient(settings.muse_base_url, settings.muse_api_key, settings.muse_model),
    )
    return services, pool


class MemoryIn(BaseModel):
    text: str
    source: str | None = None
    private: bool = False


def scope_of(request: Request) -> Scope:
    team, user = request.headers.get("x-team-id"), request.headers.get("x-user-id")
    if not team or not user:
        raise HTTPException(400, "X-Team-Id and X-User-Id headers are required")
    return Scope(team_id=team, user_id=user)


def route_override(request: Request) -> Route | None:
    """X-Gateway-Route forces a route, for running baselines and debugging."""
    value = request.headers.get("x-gateway-route")
    if value is None:
        return None
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


def with_memory(messages: list[dict], memories: list[Memory]) -> list[dict]:
    """Add memories to the system message (or a new one at the start)."""
    if not memories:
        return messages
    context = "Relevant team memory:\n\n" + "\n\n".join(
        f"[{m.source or 'memory'}] {m.content}" for m in memories
    )
    first = messages[0] if messages else {}
    if first.get("role") == "system" and isinstance(first.get("content"), str):
        return [{**first, "content": f"{first['content']}\n\n{context}"}, *messages[1:]]
    return [{"role": "system", "content": context}, *messages]


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

    app = FastAPI(title="Memory Gateway", lifespan=lifespan)

    @app.exception_handler(SpendCapExceeded)
    async def spend_cap(request: Request, exc: SpendCapExceeded):
        return JSONResponse({"error": {"message": str(exc), "type": "spend_cap"}}, status_code=402)

    @app.exception_handler(httpx.HTTPStatusError)
    async def upstream_error(request: Request, exc: httpx.HTTPStatusError):
        return JSONResponse(
            {"error": {"message": exc.response.text, "type": "upstream_error"}},
            status_code=exc.response.status_code,
        )

    @app.exception_handler(httpx.RequestError)
    async def upstream_unreachable(request: Request, exc: httpx.RequestError):
        return JSONResponse(
            {"error": {"message": f"Model unreachable: {exc!r}", "type": "upstream_error"}},
            status_code=502,
        )

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "memory-gateway", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        s: Services = request.app.state.services
        scope = scope_of(request)
        body = await request.json()

        if has_tools(body):
            route, memories = Route.PASSTHROUGH, []
        else:
            question = last_user_text(body.get("messages", []))
            hits = await s.store.search(scope, question)
            route = route_override(request) or rule_gate(question, hits)
            if route == Route.ESCALATE_RAW:
                memories = await s.store.pages(scope, [h.id for h in hits])
            elif route in (Route.LOCAL_MEMORY, Route.ESCALATE_COMPACT):
                memories = hits
            else:
                memories = []
            body = {**body, "messages": with_memory(body["messages"], memories)}

        paid = route in ESCALATIONS
        client = s.muse if paid else s.llama
        async with s.ledger.reserve(s.ledger.worst_case_cost(body)) if paid else nullcontext():
            start = time.monotonic()
            response = await client.chat(body)
            latency_ms = int((time.monotonic() - start) * 1000)
            await s.ledger.record(scope, client.model, route, response.get("usage", {}),
                                  latency_ms, paid=paid)

        headers = {"X-Gateway-Route": route.value}
        if body.get("stream"):
            sse = as_sse(response)
            sse.headers.update(headers)
            return sse
        return JSONResponse(response, headers=headers)

    @app.post("/v1/memory")
    async def add_memory(item: MemoryIn, request: Request):
        """Add text (docs, code, decisions) to memory: team-shared, or private to the caller."""
        s: Services = request.app.state.services
        ids = await ingest(s.store, s.llama, scope_of(request), item.text,
                           source=item.source, private=item.private)
        return {"ids": [str(i) for i in ids]}

    @app.get("/v1/usage")
    async def usage(request: Request):
        ledger: Ledger = request.app.state.services.ledger
        return {"spent_usd": await ledger.spent(), "cap_usd": ledger.cap_usd,
                "by_model_route": await ledger.summary()}

    return app


app = create_app()
