import hashlib
import os
from pathlib import Path

import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

ADMIN_URL = os.environ.get("TEST_ADMIN_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
TEST_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/gateway_test")
SCHEMA = Path(__file__).parent.parent / "memory" / "schema.sql"


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Bag-of-words hashed into 384 dimensions: similar words, similar vectors."""
    out = []
    for text in texts:
        v = np.zeros(384)
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 384] += 1
        out.append((v / (np.linalg.norm(v) or 1)).tolist())
    return out


@pytest.fixture(scope="session")
async def pool():
    """A fresh gateway_test database with the schema applied."""
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("Postgres not running (docker compose up -d)")
    with admin:
        admin.execute("DROP DATABASE IF EXISTS gateway_test WITH (FORCE)")
        admin.execute("CREATE DATABASE gateway_test")
    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        conn.execute(SCHEMA.read_text())
    pool = AsyncConnectionPool(TEST_URL, open=False, configure=register_vector_async)
    await pool.open()
    yield pool
    await pool.close()


class FakeModel:
    """Stands in for any OpenAI-compatible model; remembers every request."""

    def __init__(self, model: str, reply: str | None = None):
        self.model = model
        self.reply = reply
        self.requests: list[dict] = []

    async def chat(self, body):
        self.requests.append(body)
        message = {"role": "assistant", "content": self.reply or f"answer from {self.model}"}
        if body.get("tools"):
            message["tool_calls"] = [{"id": "c1", "type": "function",
                                      "function": {"name": "f", "arguments": "{}"}}]
        return {"id": "x", "object": "chat.completion", "created": 0, "model": self.model,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                          "prompt_tokens_details": {"cached_tokens": 40}}}
