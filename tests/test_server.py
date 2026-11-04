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
    return {"model": "itara", "messages": [{"role": "user", "content": text}], **extra}


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


async def test_accurate_mode_sends_every_question_to_muse_with_memory(gateway):
    client, services = gateway
    services.always_escalate = True
    r = await client.post("/v1/chat/completions", json=chat("how do we deploy the frontend"))
    assert r.headers["x-gateway-route"] == "escalate_compact"
    assert services.llama.requests == []
    assert "vercel.json" in services.muse.requests[0]["messages"][0]["content"]
    usage = (await client.get("/v1/usage", headers=services.keys["admin"])).json()
    assert usage["gate"] == "always escalate"


async def test_local_mode_keeps_every_question_on_the_open_model(gateway):
    client, services = gateway
    services.never_escalate = True
    r = await client.post("/v1/chat/completions", json=chat("deploy frontend " + "x " * 900))
    assert r.headers["x-gateway-route"] == "local_memory"  # the rule gate would escalate this
    assert "Vercel" in services.llama.requests[0]["messages"][0]["content"]
    assert services.muse.requests == []
    usage = (await client.get("/v1/usage", headers=services.keys["admin"])).json()
    assert usage["gate"] == "never escalate"


async def test_escalation_is_paid_and_carries_memory(gateway):
    client, services = gateway
    r = await client.post("/v1/chat/completions", json=chat("deploy frontend " + "x " * 900))
    assert r.headers["x-gateway-route"] == "escalate_compact"
    assert "vercel.json" in services.muse.requests[0]["messages"][0]["content"]
    assert (await client.get("/v1/usage")).status_code == 403
    usage = (await client.get("/v1/usage", headers=services.keys["admin"])).json()
    assert usage["spent_usd"] == pytest.approx((100 * 1.0 + 10 * 2.0) / 1e6)


async def test_raw_override_sends_pages(gateway):
    client, services = gateway
    r = await client.post("/v1/chat/completions", json=chat("frontend deploy config"),
                          headers={**services.keys["admin"], "X-Gateway-Route": "escalate_raw"})
    assert r.headers["x-gateway-route"] == "escalate_raw"
    assert "vercel.json" in services.muse.requests[0]["messages"][0]["content"]


async def test_tool_calls_get_team_memory_and_skip_the_gate(gateway):
    client, services = gateway
    tools = [{"type": "function", "function": {"name": "f"}}]
    body = chat("how do we deploy the frontend", tools=tools)
    r = await client.post("/v1/chat/completions", json=body)
    assert r.headers["x-gateway-route"] == "escalate_compact"
    sent = services.muse.requests[0]
    assert "vercel.json" in sent["messages"][0]["content"] and sent["tools"] == tools
    assert sent["messages"][-1] == body["messages"][-1]
    assert services.llama.requests == []
    assert r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "f"


async def test_tool_calls_without_memory_pass_through_unchanged(gateway):
    client, services = gateway
    body = chat("zzqx", tools=[{"type": "function", "function": {"name": "f"}}])
    r = await client.post("/v1/chat/completions", json=body)
    assert r.headers["x-gateway-route"] == "passthrough"
    assert services.muse.requests[0]["messages"] == body["messages"]


async def test_local_mode_never_calls_muse_even_with_tools(gateway):
    client, services = gateway
    services.never_escalate = True
    tools = [{"type": "function", "function": {"name": "f"}}]
    for text in ("how do we deploy the frontend", "zzqx"):
        r = await client.post("/v1/chat/completions", json=chat(text, tools=tools))
        assert r.status_code == 200 and r.headers["x-gateway-route"].startswith("local")
    assert "Vercel" in services.llama.requests[0]["messages"][0]["content"]
    assert services.llama.requests[0]["tools"] == tools
    assert services.muse.requests == []


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


async def test_usage_counts_requests_not_calls(gateway):
    client, services = gateway
    services.llama = LogprobModel("llama")
    zeros = [0.0] * len(CONFIDENCE_FEATURES)
    for threshold in (0.6, 0.4):  # P(draft right) 0.5: escalated with raw pages, then kept
        services.gate = CascadeGate(mean=zeros, scale=[1.0] * len(zeros), weights=[*zeros, 0.0],
                                    threshold=threshold)
        await client.post("/v1/chat/completions", json=chat("how do we deploy the frontend"))
    await client.post("/v1/chat/completions", json=chat("hi"), headers=services.keys["olga"])
    usage = (await client.get("/v1/usage", headers=services.keys["admin"])).json()
    totals = {(r["team_id"], r["route"]): r for r in usage["by_user_route"]}
    assert {k: r["requests"] for k, r in totals.items()} == {
        ("ops", "local_memory"): 1, ("web", "escalate_raw"): 1, ("web", "local_memory"): 1}
    assert {r["user_id"] for r in usage["by_user_route"]} == {"frank", "olga"}
    assert set(usage["cache_scopes"]) == {"ops", "web"}
    assert usage["cache_scopes"]["ops"] != usage["cache_scopes"]["web"]
    assert totals["web", "local_memory"]["calls"] == 3  # the draft and its two probes
    assert totals["web", "escalate_raw"]["calls"] == 4  # and the escalation
    assert totals["web", "escalate_raw"]["muse_tokens"] == 110
    assert totals["web", "local_memory"]["muse_tokens"] == 0
    assert [r["user_id"] for r in usage["recent"]] == ["olga", "frank", "frank"]
    assert usage["recent"][0]["route"] == "local_memory" and usage["gate"] == "CascadeGate"


async def test_dashboard_serves_the_app_and_its_pages(gateway):
    client, _ = gateway
    for path in ("/dashboard", "/dashboard/", "/dashboard/results"):  # the app's own routes
        r = await client.get(path)
        assert r.status_code == 200 and '<div id="root">' in r.text
    assert (await client.get("/dashboard/results.json")).json()["run_id"]
    r = await client.get("/dashboard/..%2F..%2Fgateway%2Fconfig.py")  # never outside dist
    assert "database_url" not in r.text


async def test_muse_calls_carry_the_reserved_output_limit(gateway):
    client, services = gateway
    services.ledger.max_output_tokens = 64
    await client.post("/v1/chat/completions", json=chat("deploy frontend " + "x " * 900))
    await client.post("/v1/chat/completions",
                      json=chat("deploy frontend " + "x " * 900, max_completion_tokens=10_000))
    assert [r["max_tokens"] for r in services.muse.requests] == [64, 64]


async def test_memory_is_compressed_once_and_kept_current(gateway):
    client, services = gateway
    calls = []

    async def compress(body):
        calls.append(body)
        page = body["messages"][1]["content"].rsplit("\n\n", 1)[1]
        return {"choices": [{"message": {"content": '{"summary": "%s", "keywords": []}' % page}}]}

    services.llama.chat = compress
    v1 = {"text": "rollback if p95 over 300ms", "source": "release.py@v1", "replace": True}
    first = (await client.post("/v1/memory", json=v1)).json()
    again = (await client.post("/v1/memory", json=v1)).json()
    assert len(calls) == 1 and again["ids"] == first["ids"] and again["reused_pages"] == 1
    # A new version replaces the old one: only the current text stays searchable.
    v2 = {**v1, "text": "rollback if p95 over 250ms", "source": "release.py@v2"}
    assert (await client.post("/v1/memory", json=v2)).json()["replaced_sources"] == ["release.py@v1"]
    hits = await services.store.search(Scope("web", "frank"), "rollback p95")
    assert [h.source for h in hits if h.source.startswith("release.py")] == ["release.py@v2"]
    # And a source can be forgotten, by its team only.
    r = await client.delete("/v1/memory", params={"source": "release.py@v2"},
                            headers=services.keys["olga"])
    assert r.json()["deleted_sources"] == []
    r = await client.delete("/v1/memory", params={"source": "release.py@v2"})
    assert r.json()["deleted_sources"] == ["release.py@v2"]
    assert not [h for h in await services.store.search(Scope("web", "frank"), "rollback p95")
                if h.source.startswith("release.py")]


async def test_muse_gets_summaries_for_code_and_pages_for_docs(gateway):
    client, services = gateway
    team = Scope("ship", "sam")
    await services.store.add(team, "deploy() in ship.py pushes the frontend build",
                             ["deploy"], "def deploy(): push('frontend')", source="ship.py@1a2b3c4d")
    await services.store.add(team, "The frontend deploys from main", ["deploy"],
                             "release.md: deploy the frontend on Thursdays at 10:00",
                             source="release.md")
    services.always_escalate = True
    await client.post("/v1/chat/completions", json=chat("how do we deploy the frontend"),
                      headers=bearer(await services.principals.create("ship", "sam")))
    sent = services.muse.requests[0]["messages"][0]["content"]
    assert "deploy() in ship.py" in sent and "push('frontend')" not in sent  # code: summary
    assert "Thursdays at 10:00" in sent and "deploys from main" not in sent  # doc: raw page
