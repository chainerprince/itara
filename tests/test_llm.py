import json

import httpx

from gateway.llm import ChatClient


async def test_defaults_are_sent_unless_the_request_sets_them():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": []})

    client = ChatClient("http://muse/v1", "", "muse", defaults={"reasoning_effort": "minimal"})
    client._http = httpx.AsyncClient(base_url="http://muse/v1", transport=httpx.MockTransport(handler))
    await client.chat({"messages": []})
    await client.chat({"messages": [], "reasoning_effort": "high"})
    assert [b["reasoning_effort"] for b in sent] == ["minimal", "high"]
    await client.close()
