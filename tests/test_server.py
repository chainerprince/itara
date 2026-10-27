import httpx
import pytest

from gateway.accounting import Ledger
from gateway.server import Services, create_app
from memory.store import MemoryStore, Scope
from tests.conftest import fake_embed

HEADERS = {"X-Team-Id": "web", "X-User-Id": "frank"}


class FakeModel:
    def __init__(self, model: str):
        self.model = model
        self.requests: list[dict] = []

    async def chat(self, body):
        self.requests.append(body)
        message = {"role": "assistant", "content": f"answer from {self.model}"}
        if body.get("tools"):
            message["tool_calls"] = [{"id": "c1", "type": "function",
                                      "function": {"name": "f", "arguments": "{}"}}]
        return {"id": "x", "object": "chat.completion", "created": 0, "model": self.model,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}


@pytest.fixture
async def gateway(pool):
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE usage")
    store = MemoryStore(pool, fake_embed)
    await store.add(Scope("web", "frank"), "The frontend deploys with Vercel from the main branch",
                    ["deploy", "vercel"], "vercel.json: {...}", source="deploy.md")
    services = Services(store=store, ledger=Ledger(pool, 1.0, 1.0, 2.0),
                        llama=FakeModel("llama"), muse=FakeModel("muse"))
    app = create_app(services)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
            yield client, services


def chat(text, **extra):
    return {"model": "memory-gateway", "messages": [{"role": "user", "content": text}], **extra}


async def test_requires_identity(gateway):
    client, _ = gateway
    assert (await client.post("/v1/chat/completions", json=chat("hi"))).status_code == 400


async def test_easy_question_with_memory_stays_local(gateway):
    client, services = gateway
    r = await client.post("/v1/chat/completions", json=chat("how do we deploy the frontend"),
                          headers=HEADERS)
    assert r.status_code == 200
    assert r.headers["x-gateway-route"] == "local_memory"
    assert r.json()["model"] == "llama"
    assert "Vercel" in services.llama.requests[0]["messages"][0]["content"]
    assert services.muse.requests == []


async def test_escalation_is_paid_and_carries_memory(gateway):
    client, services = gateway
    r = await client.post("/v1/chat/completions", json=chat("deploy frontend " + "x " * 900),
                          headers=HEADERS)
    assert r.headers["x-gateway-route"] == "escalate_compact"
    assert "Vercel" in services.muse.requests[0]["messages"][0]["content"]
    usage = (await client.get("/v1/usage")).json()
    assert usage["spent_usd"] == pytest.approx((100 * 1.0 + 10 * 2.0) / 1e6)


async def test_raw_override_sends_pages(gateway):
    client, services = gateway
    r = await client.post("/v1/chat/completions", json=chat("frontend deploy config"),
                          headers={**HEADERS, "X-Gateway-Route": "escalate_raw"})
    assert r.headers["x-gateway-route"] == "escalate_raw"
    assert "vercel.json" in services.muse.requests[0]["messages"][0]["content"]


async def test_tool_calls_pass_through_unchanged(gateway):
    client, services = gateway
    body = chat("deploy the frontend", tools=[{"type": "function", "function": {"name": "f"}}])
    r = await client.post("/v1/chat/completions", json=body, headers=HEADERS)
    assert r.headers["x-gateway-route"] == "passthrough"
    assert services.muse.requests[0]["messages"] == body["messages"]
    assert r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "f"


async def test_streaming_clients_get_one_chunk(gateway):
    client, _ = gateway
    r = await client.post("/v1/chat/completions", json=chat("hello", stream=True), headers=HEADERS)
    lines = [l for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    assert '"delta"' in lines[0] and "answer from llama" in lines[0]


async def test_spend_cap_returns_402(gateway):
    client, services = gateway
    services.ledger.cap_usd = 0.0
    r = await client.post("/v1/chat/completions", json=chat("hi", tools=[{"type": "function"}]),
                          headers=HEADERS)
    assert r.status_code == 402


async def test_add_memory(gateway):
    client, services = gateway

    async def compress(body):
        return {"choices": [{"message": {"content": '{"summary": "s", "keywords": []}'}}]}

    services.llama.chat = compress
    r = await client.post("/v1/memory", json={"text": "notes", "private": True}, headers=HEADERS)
    assert r.status_code == 200 and len(r.json()["ids"]) == 1
