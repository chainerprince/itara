"""Before a run with a new open model: does it answer the way the gate reads it?

    uv run python -m scripts.open_model_check          # LLAMA_* (and LLAMA_REASONING_STRENGTH) from .env

For a model that reasons (Muse Glimmer): the reasoning comes back apart from the answer, the
answer's logprobs can be found after it, the yes/no self-check gives a probability, and the
KV cache is scoped by cache_salt (a repeat with the same salt reuses the prompt, another
salt reuses none). Prints what it saw and a verdict per check; calls the open model only.
"""

import asyncio
import time

from gateway.config import Settings
from gateway.gate import SELF_CHECK, answer_logprobs, reasoning_of, yes_probability
from gateway.llm import ChatClient

MEMORY = "\n".join(f"Memory {i}: billing/retry.py retries a declined charge up to {i} times." for i in range(80))
QUESTION = ("Which file retries a declined charge? Reply with the path only.")


def show(name: str, ok: bool, detail: str) -> bool:
    print(f"{'OK  ' if ok else 'FAIL'} {name}: {detail}")
    return ok


async def main() -> None:
    s = Settings()
    llama = ChatClient(s.llama_base_url, s.llama_api_key, s.llama_model, defaults=s.llama_defaults())
    print(f"{s.llama_model} at {s.llama_base_url}, reasoning_strength={s.llama_reasoning_strength or '-'}")
    messages = [{"role": "system", "content": MEMORY}, {"role": "user", "content": QUESTION}]
    salt = f"check-{time.time_ns()}"  # new each time, so an earlier check's cache can't answer
    body = {"messages": messages, "temperature": 0, "logprobs": True, "cache_salt": salt}

    start = time.monotonic()
    draft = await llama.chat(body)
    seconds = time.monotonic() - start
    content = draft["choices"][0]["message"].get("content") or ""
    reasoning = reasoning_of(draft)
    usage = draft.get("usage", {})
    answer = answer_logprobs(draft)
    written = len((draft["choices"][0].get("logprobs") or {}).get("content") or [])
    results = [
        show("answer", "billing/retry.py" in content and "to=self" not in content,
             f"{content!r} in {seconds:.1f}s, {usage.get('completion_tokens')} tokens written"),
        show("reasoning apart from the answer", bool(reasoning) or not s.llama_reasoning_strength,
             f"{len(reasoning)} characters of reasoning"),
        show("answer logprobs", 0 < len(answer) < written if reasoning else len(answer) > 0,
             f"{len(answer)} of {written} tokens: {''.join(t['token'] for t in answer)!r}"),
    ]

    check = await llama.chat({**body, "logprobs": True, "top_logprobs": 10,
                              "max_tokens": 4096 if reasoning else 1,
                              "messages": [*messages, {"role": "assistant", "content": content},
                                           {"role": "user", "content": SELF_CHECK}]})
    p = yes_probability(check)
    results.append(show("self-check", p != 0.5,
                        f"P(yes) = {p:.3f} from {check['choices'][0]['message'].get('content')!r}"))

    same = await llama.chat({**body, "max_tokens": 1})
    other = await llama.chat({**body, "max_tokens": 1, "cache_salt": f"{salt}-other"})

    def cached(r: dict) -> int:
        return (r.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0

    results.append(show("cache scoped by salt", cached(same) > 0 and cached(other) == 0,
                        f"same salt reused {cached(same)} tokens, another salt {cached(other)}"))
    await llama.close()
    print("Ready for the run." if all(results) else "Fix the failures before the run.")


if __name__ == "__main__":
    asyncio.run(main())
