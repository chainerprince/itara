"""Permission-scoped KV cache keys and a cache-friendly prompt layout.

A prefix cache (vLLM, LMCache, Ollama) reuses work only for an identical token prefix, and
each block's key is chained to everything before it. So scoping the very start of the
prompt scopes the whole cache entry: cache hits only happen between people with the same
access, which closes the timing side channel across teams.

- vllm: the scope goes in the `cache_salt` request field, which vLLM mixes into the first
  block hash.
- ollama: the scope goes in a marker at the start of the system message.

Memory comes first and in a stable order, so the same team context is a byte-identical
prefix for every member of the team.
"""

from __future__ import annotations

import hashlib

from memory.store import Memory, Scope


def cache_scope(scope: Scope, memories: list[Memory]) -> str:
    """Shared memory only: keyed by team. Any private memory: keyed by team and user."""
    key = scope.team_id
    if any(m.owner_id is not None for m in memories):
        key += "\0" + scope.user_id
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def with_memory(messages: list[dict], memories: list[Memory], marker: str = "") -> list[dict]:
    """One system message: marker, then memory sorted by (source, id), then the caller's system text."""
    parts = [marker] if marker else []
    if memories:
        ordered = sorted(memories, key=lambda m: (m.source or "", str(m.id)))
        parts.append("Relevant team memory:\n\n" + "\n\n".join(
            f"[{m.source or 'memory'}] {m.content}" for m in ordered
        ))
    if not parts:
        return messages
    first = messages[0] if messages else {}
    if first.get("role") == "system" and isinstance(first.get("content"), str):
        parts.append(first["content"])
        messages = messages[1:]
    return [{"role": "system", "content": "\n\n".join(parts)}, *messages]


def scoped_body(body: dict, scope: Scope, memories: list[Memory], backend: str) -> dict:
    """The request for the local model, with memory added and the KV cache scoped."""
    salt = cache_scope(scope, memories)
    if backend == "vllm":
        return {**body, "messages": with_memory(body["messages"], memories), "cache_salt": salt}
    return {**body, "messages": with_memory(body["messages"], memories, f"[scope:{salt}]")}
