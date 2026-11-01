import httpx
import pytest

from gateway.accounting import Ledger
from gateway.auth import Principals
from gateway.server import Services, create_app
from memory.store import MemoryStore, Scope
from tests.conftest import FakeModel, fake_embed
from gateway.gate import CONFIDENCE_FEATURES, FEATURES, CascadeGate
from tests.test_gate import constant_gate


def bearer(key):
    return {"Authorization": f"Bearer {key}"}

@pytest.fixture
async def gateway(pool):
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE usage")
    store = MemoryStore(pool, fake_embed)
    await store.add(Scope("web", "frank"), "The frontend deploys with Vercel from the main branch",
                    ["deploy", "vercel"], "vercel.json: {...}", source="deploy.md")
    principals = Principals(pool)
    services = Services(store=store, ledger=Ledger(pool, 1.0, 1.0, 2.0),
                        llama=FakeModel("llama"), muse=FakeModel("muse"), principals=principals)
    services.keys = {
        "frank": bearer(await principals.create("web", "frank")),
        "admin": bearer(await principals.create("web", "root", is_admin=True)),
        "olga": bearer(await principals.create("ops", "olga")),
    }
    app = create_app(services)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw",
                                     headers=services.keys["frank"]) as client:
            yield client, services


def chat(text, **extra):
    return {"model": "memory-gateway", "messages": [{"role": "user", "content": text}], **extra}


async def test_requires_a_valid_key(gateway):
    client, _ = gateway
    for headers in ({"Authorization": ""}, bearer("mg-wrong")):
        r = await client.post("/v1/chat/completions", json=chat("hi"), headers=headers)
        assert r.status_code == 401


async def test_identity_headers_are_ignored(gateway):
    client, services = gateway
    await client.post("/v1/chat/completions", json=chat("how do we deploy the frontend"),
                      headers={**services.keys["olga"], "X-Team-Id": "web"})
    assert "Vercel" not in services.llama.requests[0]["messages"][0]["content"]


async def test_malformed_body_is_400(gateway):
    client, _ = gateway
    assert (await client.post("/v1/chat/completions", json={"model": "x"})).status_code == 400


async def test_easy_question_with_memory_stays_local(gateway):
    client, services = gateway
    r = await client.post("/v1/chat/completions", json=chat("how do we deploy the frontend"))
    assert r.status_code == 200
    assert r.headers["x-gateway-route"] == "local_memory"
    assert int(r.headers["x-gateway-memory-hits"]) >= 1  # the shared database keeps earlier tests' memory
    assert r.json()["model"] == "llama"
    assert "Vercel" in services.llama.requests[0]["messages"][0]["content"]
    assert services.muse.requests == []


async def test_escalation_is_paid_and_carries_memory(gateway):
    client, services = gateway
    r = await client.post("/v1/chat/completions", json=chat("deploy frontend " + "x " * 900))
    assert r.headers["x-gateway-route"] == "escalate_compact"
    assert "Vercel" in services.muse.requests[0]["messages"][0]["content"]
    assert (await client.get("/v1/usage")).status_code == 403
    usage = (await client.get("/v1/usage", headers=services.keys["admin"])).json()
    assert usage["spent_usd"] == pytest.approx((100 * 1.0 + 10 * 2.0) / 1e6)


async def test_raw_override_sends_pages(gateway):
    client, services = gateway
    r = await client.post("/v1/chat/completions", json=chat("frontend deploy config"),
                          headers={**services.keys["admin"], "X-Gateway-Route": "escalate_raw"})
    assert r.headers["x-gateway-route"] == "escalate_raw"
    assert "vercel.json" in services.muse.requests[0]["messages"][0]["content"]


async def test_tool_calls_pass_through_unchanged(gateway):
    client, services = gateway
    body = chat("deploy the frontend", tools=[{"type": "function", "function": {"name": "f"}}])
    r = await client.post("/v1/chat/completions", json=body)
    assert r.headers["x-gateway-route"] == "passthrough"
    assert services.muse.requests[0]["messages"] == body["messages"]
    assert r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "f"


async def test_streaming_clients_get_one_chunk(gateway):
    client, _ = gateway
    r = await client.post("/v1/chat/completions", json=chat("hello", stream=True))
    lines = [l for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    assert '"delta"' in lines[0] and "answer from llama" in lines[0]


async def test_spend_cap_returns_402(gateway):
    client, services = gateway
    services.ledger.cap_usd = 0.0
    r = await client.post("/v1/chat/completions", json=chat("hi", tools=[{"type": "function"}]))
    assert r.status_code == 402


async def test_add_memory(gateway):
    client, services = gateway

    async def compress(body):
        return {"choices": [{"message": {"content": '{"summary": "s", "keywords": []}'}}]}

    services.llama.chat = compress
    r = await client.post("/v1/memory", json={"text": "notes", "private": True})
    assert r.status_code == 200 and len(r.json()["ids"]) == 1
    usage = (await client.get("/v1/usage", headers=services.keys["admin"])).json()
    assert [row["route"] for row in usage["by_model_route"]] == ["ingest"]


async def test_route_override_needs_admin(gateway):
    client, _ = gateway
    r = await client.post("/v1/chat/completions", json=chat("hi"),
                          headers={"X-Gateway-Route": "escalate_raw"})
    assert r.status_code == 403


async def test_local_calls_are_cache_scoped_and_muse_calls_are_not(gateway):
    client, services = gateway
    await client.post("/v1/chat/completions", json=chat("how do we deploy the frontend"))
    await client.post("/v1/chat/completions", json=chat("how do we deploy the frontend"),
                      headers=services.keys["olga"])
    await client.post("/v1/chat/completions", json=chat("deploy frontend " + "x " * 900))
    web, ops = (r["messages"][0]["content"] for r in services.llama.requests)
    assert web.startswith("[scope:") and ops.startswith("[scope:")
    assert web.split("]")[0] != ops.split("]")[0]
    assert "[scope:" not in services.muse.requests[0]["messages"][0]["content"]


async def test_upstream_auth_failure_is_502(gateway):
    client, services = gateway

    async def unauthorized(body):
        request = httpx.Request("POST", "http://muse")
        raise httpx.HTTPStatusError("401", request=request,
                                    response=httpx.Response(401, request=request))

    services.llama.chat = unauthorized
    assert (await client.post("/v1/chat/completions", json=chat("hi"))).status_code == 502


class LogprobModel(FakeModel):
    """Returns token logprobs when asked, as vLLM does."""

    async def chat(self, body):
        response = await super().chat(body)
        if body.get("logprobs"):
            response["choices"][0]["logprobs"] = {"content": [
                {"token": "yes", "logprob": -0.1, "top_logprobs": [{"token": "yes", "logprob": -0.1}]}]}
        return response


async def test_gate_with_confidence_probes_the_draft(gateway, pool):
    client, services = gateway
    services.llama = LogprobModel("llama")
    services.gate = constant_gate(0.9, 0.8, features=FEATURES + CONFIDENCE_FEATURES)
    r = await client.post("/v1/chat/completions", json=chat("how do we deploy the frontend"))
    assert r.headers["x-gateway-route"] == "local_memory"
    assert "logprobs" not in r.json()["choices"][0]  # asked for by the gate, not the caller
    draft, *probes = services.llama.requests
    assert draft["logprobs"] is True and len(probes) == 2
    assert all(p["messages"][0] == draft["messages"][0] for p in probes)  # the same cached prefix
    assert all(p.get("cache_salt") == draft.get("cache_salt") for p in probes)
    async with pool.connection() as conn:
        routes = [row[0] for row in await (await conn.execute(
            "SELECT route FROM usage ORDER BY id")).fetchall()]
    assert routes == ["local_memory", "gate_probe", "gate_probe"]


async def test_learned_gate_keeps_the_draft_or_escalates(gateway):
    client, services = gateway
    for p_local, route, model in ((0.9, "local_memory", "llama"), (0.1, "escalate_compact", "muse")):
        services.gate = constant_gate(p_local, 0.8)
        services.llama.requests.clear()
        services.muse.requests.clear()
        r = await client.post("/v1/chat/completions", json=chat("how do we deploy the frontend"))
        assert r.headers["x-gateway-route"] == route and r.json()["model"] == model
        assert len(services.llama.requests) == 1  # Llama always drafts first
        assert len(services.muse.requests) == (model == "muse")
    r = await client.post("/v1/chat/completions", json=chat("deploy"),  # an override skips the gate
                          headers={**services.keys["admin"], "X-Gateway-Route": "escalate_raw"})
    assert r.headers["x-gateway-route"] == "escalate_raw"


async def test_cascade_gate_escalates_an_unsure_draft_with_raw_pages(gateway):
    client, services = gateway
    services.llama = LogprobModel("llama")
    zeros = [0.0] * len(CONFIDENCE_FEATURES)
    for threshold, route in ((0.4, "local_memory"), (0.6, "escalate_raw")):  # P(draft right) 0.5
        services.gate = CascadeGate(mean=zeros, scale=[1.0] * len(zeros), weights=[*zeros, 0.0],
                                    threshold=threshold)
        services.llama.requests.clear()
        services.muse.requests.clear()
        r = await client.post("/v1/chat/completions", json=chat("how do we deploy the frontend"))
        assert r.headers["x-gateway-route"] == route
        assert len(services.llama.requests) == 3  # the draft and its two probes
    assert "vercel.json" in services.muse.requests[0]["messages"][0]["content"]  # deploy.md: raw
