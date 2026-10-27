"""Settings, read from environment variables or a .env file (see .env.example)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/gateway"

    llama_base_url: str = "http://localhost:11434/v1"
    llama_api_key: str = "none"
    llama_model: str = "llama3.1:8b"

    muse_base_url: str = "https://api.llama.com/compat/v1"
    muse_api_key: str = ""
    muse_model: str = "muse-spark"
    muse_input_price_per_m: float = 0.0
    muse_output_price_per_m: float = 0.0

    spend_cap_usd: float = 150.0
