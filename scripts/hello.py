"""Day 1 check: both models answer, and long prompts aren't silently truncated.

    uv run python -m scripts.hello
"""

import asyncio

from gateway.config import Settings
from gateway.llm import ChatClient

# ~12K tokens of filler with a fact at the very start: if the server truncates the context
# (Ollama's default is short), the model can't see it and prompt_tokens comes back small.
LONG_PROMPT = "The secret word is PELICAN.\n" + "Filler sentence number one two three. " * 1500
LONG_EXPECTED_TOKENS = 10_000


async def check(name: str, client: ChatClient) -> None:
    # Room for a reasoning model (Muse Spark), whose max_tokens includes its reasoning.
    short = await client.chat({"messages": [{"role": "user", "content": "Say hello in five words."}],
                               "max_tokens": 1024})
    print(f"{name} ({client.model}): {short['choices'][0]['message']['content']!r}")
    print(f"  usage: {short.get('usage')}")
    long = await client.chat({"messages": [{"role": "user",
                                            "content": LONG_PROMPT + "\nWhat is the secret "
                                                       "word? Reply with only the word."}],
                              "max_tokens": 1024})
    tokens = long.get("usage", {}).get("prompt_tokens", 0)
    answer = long["choices"][0]["message"]["content"]
    ok = tokens >= LONG_EXPECTED_TOKENS
    print(f"  long prompt: {tokens} prompt tokens -> "
          f"{'OK' if ok else 'TRUNCATED: start Ollama with OLLAMA_CONTEXT_LENGTH=16384'}"
          f" (answer {answer!r})")
    await client.close()


async def main() -> None:
    s = Settings()
    await check("Llama", ChatClient(s.llama_base_url, s.llama_api_key, s.llama_model))
    label = "Muse (local stand-in)" if s.muse_is_standin else "Muse"
    await check(label, ChatClient(s.muse_base_url, s.muse_api_key, s.muse_model))


if __name__ == "__main__":
    asyncio.run(main())
