"""Settings and shared LLM model factory. The only place that knows about OpenRouter."""
from functools import lru_cache

from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

ROLES = ("scope", "research", "compress", "summarize", "supervisor", "report")

ROLE_DEFAULTS: dict[str, dict] = {
    "scope": {"temperature": 0.0},
    "compress": {"max_tokens": 32000},
    "report": {"max_tokens": 32000},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "openai/gpt-4.1"

    model_scope: str | None = None
    model_research: str | None = None
    model_compress: str | None = None
    model_summarize: str | None = None
    model_supervisor: str | None = None
    model_report: str | None = None

    tavily_api_key: str

    max_concurrent_researchers: int = 2
    max_researcher_iterations: int = 6

    sqlite_path: str = "data/threads.db"

    gdrive_client_id: str | None = None
    gdrive_client_secret: str | None = None
    gdrive_refresh_token: str | None = None
    gdrive_base_path: str = "deep_research_agents"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_model(role: str, **overrides) -> ChatOpenAI:
    """Build the ChatOpenAI client for a model-usage role, routed through OpenRouter."""
    if role not in ROLES:
        raise ValueError(f"Unknown model role: {role!r}, expected one of {ROLES}")
    settings = get_settings()
    model_name = getattr(settings, f"model_{role}") or settings.default_model
    kwargs = {**ROLE_DEFAULTS.get(role, {}), **overrides}
    return ChatOpenAI(
        model=model_name,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        **kwargs,
    )
