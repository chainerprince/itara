"""Settings, read from environment variables or a .env file (see .env.example)."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/gateway"

    llama_base_url: str = "http://localhost:11434/v1"
    llama_api_key: str = "none"
    llama_model: str = "llama3.2:3b"
    # vllm: scope the KV cache with the cache_salt field and read cached_tokens.
    # ollama: scope it with a marker at the start of the prompt.
    llama_backend: Literal["ollama", "vllm"] = "ollama"
    # For an open model that reasons (Muse Glimmer): its reasoning_strength (low, medium, high,
    # xhigh), sent as a chat-template argument with every request. Empty: nothing sent (Llama).
    llama_reasoning_strength: str = ""

    muse_base_url: str = "https://api.llama.com/compat/v1"
    muse_api_key: str = ""
    muse_model: str = "muse-spark"
    muse_input_price_per_m: float = 0.0
    muse_output_price_per_m: float = 0.0
    # Prompt tokens the provider served from its cache, when it bills them lower. Unset: billed
    # at muse_input_price_per_m.
    muse_cached_input_price_per_m: float | None = None
    # The most output one Muse call may write (sent as max_tokens), and what the spend cap
    # reserves for it. Callers can ask for less.
    muse_max_output_tokens: int = 32_768
    # True while "Muse" is a local open model; reported with every usage and eval number.
    muse_is_standin: bool = False
    # Muse's reasoning effort (e.g. minimal, low) for every request sent to it that doesn't set
    # its own: escalations, tool calls and eval's muse_direct. Empty: Muse's default.
    # Reasoning is most of an escalation's Muse tokens (results/full_run.ipynb, section 3).
    muse_reasoning_effort: str = ""

    spend_cap_usd: float = 150.0

    # Compression calls in flight at once while ingesting (shared by every request). 1 is the
    # original one-page-at-a-time path; more lets vLLM batch them (see memory/compress.py).
    ingest_concurrency: int = 1

    # A learned gate saved by eval/learn_gate.py (JSON). Empty: the rule-based gate.
    gate_path: str = ""
    # Send every question to Muse with the team's memory summaries, with no gate: the most
    # accurate setting measured (`itara up --mode accurate`). Overrides gate_path.
    always_escalate: bool = False
    # Answer every request with the open model and team memory, tool calls included: nothing
    # is sent to Muse (`itara up --mode local`). Overrides gate_path.
    never_escalate: bool = False
    # The cascade gate's threshold on P(draft right), instead of the saved one: higher
    # escalates more. eval/learn_gate.py prints the threshold for each budget.
    gate_threshold: float | None = None

    # Optional sqlite file that stores every model response, so repeated experiments don't
    # call the model again (eval only; see gateway/llm.py).
    model_cache: str = ""

    def llama_defaults(self) -> dict | None:
        """What every request to the open model carries unless it sets its own."""
        if not self.llama_reasoning_strength:
            return None
        return {"chat_template_kwargs": {"reasoning_strength": self.llama_reasoning_strength}}
