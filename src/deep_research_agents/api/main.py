"""FastAPI app: compiles all 5 registered graphs with a shared checkpointer at startup."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deep_research_agents.api.routers import agents, export, revise, threads
from deep_research_agents.config import get_settings
from deep_research_agents.db import ThreadStore
from deep_research_agents.graphs import AGENT_REGISTRY


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(settings.sqlite_path) as checkpointer:
        app.state.graphs = {
            agent_id: entry.builder.compile(checkpointer=checkpointer)
            for agent_id, entry in AGENT_REGISTRY.items()
        }
        app.state.store = await ThreadStore.connect(settings.sqlite_path)
        try:
            yield
        finally:
            await app.state.store.close()


app = FastAPI(title="deep_research_agents", lifespan=lifespan)

app.include_router(agents.router)
app.include_router(threads.router)
app.include_router(revise.router)
app.include_router(export.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
