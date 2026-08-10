"""POST /threads/{id}/export — write report locally and upload to Drive."""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from deep_research_agents import gdrive
from deep_research_agents.config import get_settings

router = APIRouter(tags=["export"])

EXPORTS_DIR = Path("data/exports")


@router.post("/threads/{thread_id}/export")
async def export_thread(thread_id: str, request: Request):
    store = request.app.state.store
    thread = await store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(404, "Thread not found")

    graph = request.app.state.graphs[thread["agent_id"]]
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    values = state.values

    # final_report for research_agent_full; joined notes as fallback for other agents
    report = values.get("final_report") or "\n\n".join(values.get("notes", []))
    if not report:
        raise HTTPException(400, "Nothing to export yet — thread has no final_report or notes")

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = EXPORTS_DIR / f"{thread_id}.md"
    local_path.write_text(report)

    settings = get_settings()
    await run_in_threadpool(
        gdrive.export_report,
        local_path,
        settings.gdrive_base_path,
        refresh_token=settings.gdrive_refresh_token,
        client_id=settings.gdrive_client_id,
        client_secret=settings.gdrive_client_secret,
    )

    return {"local_path": str(local_path), "gdrive_base_path": settings.gdrive_base_path}
