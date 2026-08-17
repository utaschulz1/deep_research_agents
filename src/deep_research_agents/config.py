"""Settings and shared LLM model factory. The only place that knows about OpenRouter."""
from functools import lru_cache

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

ROLES = ("scope", "research", "compress", "extract", "supervisor", "report")

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
    model_extract: str | None = None
    model_supervisor: str | None = None
    model_report: str | None = None

    tavily_api_key: str

    max_concurrent_researchers: int = 2
    max_researcher_iterations: int = 6

    # research_agent_sub safety net (point 5) — deliberately not reusing
    # max_researcher_iterations, which is the supervisor's own, differently-scoped cap.
    max_subagent_links: int = 10
    max_subagent_tool_iterations: int = 11
    subagent_time_budget_seconds: int = 600
    subagent_call_timeout_seconds: int = 90

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


def thread_id_from_config(config: RunnableConfig | None) -> str | None:
    """Pull the LangGraph thread_id out of a node's RunnableConfig, if present.

    Absent when a graph is invoked directly (e.g. a script/smoke test) rather
    than through the API's /threads routes, which always set it.
    """
    if not config:
        return None
    return config.get("configurable", {}).get("thread_id")


def session_kwargs(session_id: str | None) -> dict:
    """extra_body kwargs so an OpenRouter call is grouped under session_id.

    Pass session_id=thread_id at each call site so every OpenRouter call made
    while processing one thread — across every node and any spawned sub-agents —
    shows up as one session in OpenRouter's dashboard. No-op (empty dict) when
    session_id is falsy, so callers can unconditionally splat this in.
    """
    if not session_id:
        return {}
    return {"extra_body": {"session_id": session_id}}
