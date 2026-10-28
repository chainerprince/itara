"""A deterministic stand-in for every model, for checking the harness without a GPU.

The simulated models know the gold answers, but only use them when the evidence is visible
in their prompt (the same `evidence_visible` rule the harness reports). So the expected
results follow exactly from what each condition shows the model:

- `sim-strong` (the Muse role) is right iff the evidence is in its prompt.
- `sim-weak` (the Llama role) is right iff the evidence is in its prompt and the item is
  one of a fixed 70% (hash of the item id).
- Compression returns valid JSON: the start of the page plus the names it defines.
- The LoCoMo judge says CORRECT iff the gold answer appears in the response.
- Usage counts ~4 characters per token. `cached_tokens` comes from a vLLM-style prefix cache
  (16-token blocks, chained hashes seeded by `cache_salt`, LRU), so hits only happen within
  one permission scope, as on the real server.

Results from it are labelled simulated (model names start with `sim-`). They check the
code, not the idea: accuracy depends on the simulator's rules.

    uv run python -m eval.sim --port 8099    # OpenAI-compatible; see the SIM block in .env.example
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import OrderedDict

from eval import locomo, swebench_loc
from eval.common import Item, evidence_visible

BLOCK_TOKENS = 16
CHARS_PER_TOKEN = 4
CACHE_BLOCKS = 500_000
WEAK_SUCCESS = 0.7

_ISSUE = re.compile(r"<issue>\n?(.*?)\n?</issue>", re.DOTALL)
_SEEN_PATHS = re.compile(r"\[start of (.+?)\]|\[([^\[\]\s]+?)@[0-9a-f]{8}\]")
_DEFINES = re.compile(r"^\s*\d*\s*(?:def|class)\s+(\w+)", re.MULTILINE)


def _key(text: str) -> str:
    return " ".join(text.split())[:300]


def weak_succeeds(item_id: str) -> bool:
    return int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 100 < WEAK_SUCCESS * 100


class SimIndex:
    """Finds the item a prompt is about: by its issue text (SWE-bench) or question (LoCoMo)."""

    def __init__(self, items: list[Item]) -> None:
        self.by_issue: dict[str, Item] = {}
        self.by_question: dict[str, list[Item]] = {}  # a few questions recur across chats
        for item in items:
            for text in (item.question, item.raw_context):
                if match := _ISSUE.search(text):
                    self.by_issue.setdefault(_key(match.group(1)), item)
            if "question" in item.meta:
                self.by_question.setdefault(_key(item.meta["question"]), []).append(item)

    @classmethod
    def from_datasets(cls) -> SimIndex:
        return cls(swebench_loc.load(10_000) + locomo.load(10_000))

    def find(self, last_user: str, prompt: str = "") -> Item | None:
        if match := _ISSUE.search(last_user):
            return self.by_issue.get(_key(match.group(1)))
        if locomo.INSTRUCTION in last_user:
            question = last_user.split(locomo.INSTRUCTION)[0].strip().split("\n\n")[-1]
            candidates = self.by_question.get(_key(question), [])
            # Same question in several conversations: the one whose evidence is in view.
            return next((i for i in candidates if evidence_visible(i, prompt)),
                        candidates[0] if candidates else None)
        return None


class PrefixCache:
    """vLLM-style automatic prefix caching: chained block hashes, seeded by the cache salt."""

    def __init__(self, capacity: int = CACHE_BLOCKS) -> None:
        self._blocks: OrderedDict[str, None] = OrderedDict()
        self._capacity = capacity

    def lookup_and_insert(self, text: str, salt: str) -> int:
        """Cached tokens for this prompt; then the prompt's blocks are cached."""
        size = BLOCK_TOKENS * CHARS_PER_TOKEN
        digest, hits, missed = hashlib.sha256(salt.encode()).hexdigest(), 0, False
        for start in range(0, len(text) - size + 1, size):
            digest = hashlib.sha256((digest + text[start:start + size]).encode()).hexdigest()
            if not missed and digest in self._blocks:
                hits += 1
                self._blocks.move_to_end(digest)
            else:
                missed = True
                self._blocks[digest] = None
                if len(self._blocks) > self._capacity:
                    self._blocks.popitem(last=False)
        return hits * BLOCK_TOKENS


class SimModel:
    """Same interface as ChatClient: `await chat(body)` returns an OpenAI-style response."""

    def __init__(self, model: str, index: SimIndex) -> None:
        self.model = model
        self.index = index
        self.cache = PrefixCache()
        self.requests = 0

    async def chat(self, body: dict) -> dict:
        self.requests += 1
        messages = body.get("messages", [])
        prompt = "".join(f"<{m.get('role')}>{m.get('content') or ''}" for m in messages)
        system = next((m.get("content") or "" for m in messages if m.get("role") == "system"), "")
        last_user = next((m.get("content") or "" for m in reversed(messages)
                          if m.get("role") == "user"), "")
        if system.startswith("You are a memory compression agent"):
            content = self._compress(last_user.split("Compress the following text:\n\n", 1)[-1])
        elif "Reply with exactly one word: CORRECT or WRONG" in last_user:
            content = self._judge(last_user)
        else:
            content = self._answer(self.index.find(last_user, prompt), prompt)
        prompt_tokens = math.ceil(len(prompt) / CHARS_PER_TOKEN)
        cached = min(self.cache.lookup_and_insert(prompt, body.get("cache_salt", "")),
                     prompt_tokens)
        completion_tokens = math.ceil(len(content) / CHARS_PER_TOKEN)
        return {
            "id": f"sim-{self.requests}", "object": "chat.completion", "created": 0,
            "model": self.model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens,
                      "prompt_tokens_details": {"cached_tokens": cached}},
        }

    def _answer(self, item: Item | None, prompt: str) -> str:
        if item is None:
            return "I can't tell from this."
        right = evidence_visible(item, prompt) and (
            self.model != "sim-weak" or weak_succeeds(item.id))
        if isinstance(item.gold, list):  # SWE-bench: file paths
            if right:
                return "\n".join(item.gold)
            seen = [a or b for a, b in _SEEN_PATHS.findall(prompt)]
            wrong = next((p for p in seen if p not in item.gold), "unknown/file.py")
            return wrong
        return str(item.gold) if right else "I don't know."

    @staticmethod
    def _compress(page: str) -> str:
        names = list(dict.fromkeys(_DEFINES.findall(page)))[:15]
        summary = page[:300].strip()
        if names:
            summary += "\nDefines: " + ", ".join(names)
        keywords = names or [w for w in re.findall(r"[A-Za-z]{5,}", page)][:5]
        return json.dumps({"summary": summary, "keywords": keywords})

    @staticmethod
    def _judge(prompt: str) -> str:
        gold = re.search(r"Gold answer: (.*)", prompt)
        response = re.search(r"Response: (.*)", prompt)
        norm = lambda s: " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())  # noqa: E731
        if gold and response and norm(gold.group(1)) and norm(gold.group(1)) in norm(
                response.group(1)):
            return "CORRECT"
        return "WRONG"


def create_app():
    from fastapi import FastAPI

    app = FastAPI(title="Simulated models")
    state: dict = {}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        if "index" not in state:  # load the datasets on first use
            state["index"] = SimIndex.from_datasets()
        model = body.get("model", "sim-strong")
        if model not in state:
            state[model] = SimModel(model, state["index"])
        return await state[model].chat(body)

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
