"""Thread lifecycle: create/list/get, state inspection, and background-run dispatch."""
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Request

from deep_research_agents.api.schemas import CreateThreadRequest, RunRequest, StateUpdateRequest, ThreadResponse
from deep_research_agents.graphs import AGENT_REGISTRY

router = APIRouter(tags=["threads"])


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@router.post("/threads", response_model=ThreadResponse)
async def create_thread(body: CreateThreadRequest, request: Request):
    if body.agent_id not in AGENT_REGISTRY:
        raise HTTPException(400, f"Unknown agent_id: {body.agent_id!r}")
    return await request.app.state.store.create_thread(body.name, body.agent_id)


@router.get("/threads", response_model=list[ThreadResponse])
async def list_threads(request: Request):
    return await request.app.state.store.list_threads()


@router.get("/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(thread_id: str, request: Request):
    thread = await request.app.state.store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(404, "Thread not found")
    return thread


@router.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str, request: Request):
    """Return graph.aget_state(config).values. Shape varies by agent_id — only
    research_agent_full has all of notes/final_report/research_brief."""
    thread = await request.app.state.store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(404, "Thread not found")
    graph = request.app.state.graphs[thread["agent_id"]]
    state = await graph.aget_state(_config(thread_id))
    return state.values


async def _run_agent(thread_id: str, agent_id: str, message: str, app: FastAPI) -> None:
    store = app.state.store
    graph = app.state.graphs[agent_id]
    adapter = AGENT_REGISTRY[agent_id].input_adapter
    try:
        result = await graph.ainvoke(adapter(message), _config(thread_id))
    except Exception as e:
        await store.update_status(thread_id, "failed", error_detail=str(e))
        return

    # Surfacing (not fixing) the supervisor's silent-failure bug: a rate-limited
    # parallel research call can make the graph return successfully with an
    # empty/partial report. Purely observational string-length check.
    if agent_id == "research_agent_full":
        final_report = result.get("final_report") or ""
        notes = result.get("notes") or []
        if len(final_report) < 100 or not notes:
            await store.update_status(thread_id, "done_empty")
            return

    await store.update_status(thread_id, "done")


@router.post("/threads/{thread_id}/runs", status_code=202)
async def run_thread(thread_id: str, body: RunRequest, background_tasks: BackgroundTasks, request: Request):
    """Adapt message via the registry, dispatch as a background task, return immediately.

    Reruns the entire pipeline from START every time (architectural fact, not a
    bug) — none of the 5 graphs support interrupt()-based resume.
    """
    thread = await request.app.state.store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(404, "Thread not found")
    await request.app.state.store.update_status(thread_id, "running")
    background_tasks.add_task(_run_agent, thread_id, thread["agent_id"], body.message, request.app)
    return {"status": "running"}


@router.post("/threads/{thread_id}/state")
async def update_thread_state(thread_id: str, body: StateUpdateRequest, request: Request):
    """Raw passthrough to graph.aupdate_state(config, values).

    Reducer-free fields (research_brief, final_report) get replaced cleanly;
    operator.add fields (notes, raw_notes) and add_messages fields append
    rather than replace — framework behavior, not worked around here.
    """
    thread = await request.app.state.store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(404, "Thread not found")
    graph = request.app.state.graphs[thread["agent_id"]]
    await graph.aupdate_state(_config(thread_id), body.values)
    return {"status": "updated"}
