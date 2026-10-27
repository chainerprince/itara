"""Client for any OpenAI-compatible chat API: vLLM/Ollama (Llama) or the Meta Model API (Muse)."""

from __future__ import annotations

import httpx


class ChatClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0) -> None:
        self.model = model
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def chat(self, body: dict) -> dict:
        """Send a chat completion request (never streamed) and return the response JSON."""
        body = {**body, "model": self.model, "stream": False}
        body.pop("stream_options", None)
        response = await self._http.post("/chat/completions", json=body)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self._http.aclose()
