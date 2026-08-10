"""GET /agents — list the graphs exposed via AGENT_REGISTRY."""
from fastapi import APIRouter

from deep_research_agents.api.schemas import AgentInfo
from deep_research_agents.graphs import AGENT_REGISTRY

router = APIRouter(tags=["agents"])


@router.get("/agents", response_model=list[AgentInfo])
async def list_agents():
    return [
        AgentInfo(id=agent_id, name=entry.name, description=entry.description)
        for agent_id, entry in AGENT_REGISTRY.items()
    ]
