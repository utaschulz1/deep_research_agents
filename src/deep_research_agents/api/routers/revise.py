"""POST /threads/{id}/revise — the cheap-correction path.

Calls final_report_generation() directly, bypassing the graph/supervisor
entirely, with the current notes + a "USER CORRECTION" entry + research_brief.
One cheap LLM call instead of a full research re-run.
"""
from fastapi import APIRouter, HTTPException, Request

from deep_research_agents.api.schemas import ReviseRequest
from deep_research_agents.research_agent_full import final_report_generation

router = APIRouter(tags=["revise"])


@router.post("/threads/{thread_id}/revise")
async def revise_report(thread_id: str, body: ReviseRequest, request: Request):
    store = request.app.state.store
    thread = await store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(404, "Thread not found")
    if thread["agent_id"] != "research_agent_full":
        raise HTTPException(
            400,
            "revise is only valid for research_agent_full threads — no other "
            "graph has a final_report_generation step",
        )

    graph = request.app.state.graphs["research_agent_full"]
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)

    notes = list(state.values.get("notes", []))
    notes.append(f"USER CORRECTION:\n{body.instruction}")

    result = await final_report_generation({
        "notes": notes,
        "research_brief": state.values.get("research_brief", ""),
    })

    await graph.aupdate_state(config, {"final_report": result["final_report"]})
    return {"final_report": result["final_report"]}
