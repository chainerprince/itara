"""Client for any OpenAI-compatible chat API: vLLM/Ollama (Llama) or the Meta Model API (Muse).

With `cache_path` set (MODEL_CACHE in .env), every response is stored in a sqlite file
keyed by endpoint, model and request body, and an identical request is answered from it.
It's for experiments only: re-running a baseline after a scoring fix, or reusing all local
calls when only Muse changes, costs no model time. A replayed response carries the latency
of the original call in `_replayed_latency_ms`, so accounting stays truthful.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import httpx


class ChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 600.0,
        cache_path: str = "",
    ) -> None:
        self.model = model
        self.replayed = 0
        self._base_url = base_url
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=timeout,
            # Modal answers a request still running after 150s (e.g. during a cold start)
            # with a 303 to a URL that waits for the result.
            follow_redirects=True,
        )
        self._cache = None
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._cache = sqlite3.connect(cache_path)
            self._cache.execute("CREATE TABLE IF NOT EXISTS responses"
                                " (key TEXT PRIMARY KEY, response TEXT, latency_ms INTEGER)")

    async def chat(self, body: dict) -> dict:
        """Send a chat completion request (never streamed) and return the response JSON."""
        body = {**body, "model": self.model, "stream": False}
        body.pop("stream_options", None)
        key = None
        if self._cache is not None:
            key = hashlib.sha256(json.dumps([self._base_url, body], sort_keys=True).encode()
                                 ).hexdigest()
            row = self._cache.execute("SELECT response, latency_ms FROM responses WHERE key = ?",
                                      (key,)).fetchone()
            if row:
                self.replayed += 1
                return {**json.loads(row[0]), "_replayed_latency_ms": row[1]}
        start = time.monotonic()
        response = await self._http.post("/chat/completions", json=body)
        response.raise_for_status()
        if key is not None:
            self._cache.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
                                (key, response.text, int((time.monotonic() - start) * 1000)))
            self._cache.commit()
        return response.json()

    async def close(self) -> None:
        await self._http.aclose()
        if self._cache is not None:
            self._cache.close()
