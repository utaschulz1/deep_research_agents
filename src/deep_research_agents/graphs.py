"""Agent registry — maps agent_id to its graph builder, display metadata, and input adapter.

The 5 graphs do not share one input schema: scope_research/agent take
{"messages": [...]}, research_agent_sub/agent_mcp take {"researcher_messages": [...],
"research_topic": str}, supervisor_agent takes {"supervisor_messages": [...],
"research_brief": str}. Each registry entry carries an input_adapter that turns
POST /threads/{id}/runs' plain {"message": "..."} body into the right shape.
"""
from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph

from deep_research_agents.research_agent_scope import scope_research_builder
from deep_research_agents.research_agent_sub import researcher_agent_sub_builder
from deep_research_agents.research_agent_mcp import agent_builder_mcp
from deep_research_agents.multi_agent_supervisor import supervisor_builder
from deep_research_agents.research_agent_full import deep_researcher_builder


@dataclass(frozen=True)
class AgentEntry:
    builder: StateGraph
    name: str
    description: str
    input_adapter: Callable[[str], dict]


def _messages_input(message: str) -> dict:
    return {"messages": [HumanMessage(content=message)]}


def _researcher_input(message: str) -> dict:
    return {"researcher_messages": [HumanMessage(content=message)], "research_topic": message}


def _supervisor_input(message: str) -> dict:
    return {"supervisor_messages": [HumanMessage(content=message)], "research_brief": message}


AGENT_REGISTRY: dict[str, AgentEntry] = {
    "scope_research": AgentEntry(
        scope_research_builder,
        "Scoping",
        "Clarifies the user's request and produces a research brief.",
        _messages_input,
    ),
    "research_agent_sub": AgentEntry(
        researcher_agent_sub_builder,
        "Sub-Researcher",
        "Single agent performing iterative web search and reflection on one topic.",
        _researcher_input,
    ),
    "research_agent_mcp": AgentEntry(
        agent_builder_mcp,
        "Research Agent (MCP)",
        "Single agent researching a local document corpus via an MCP filesystem server. Requires Node/npx.",
        _researcher_input,
    ),
    "research_agent_supervisor": AgentEntry(
        supervisor_builder,
        "Supervisor",
        "Coordinates parallel research sub-agents against a research brief. Does not scope or write a final report.",
        _supervisor_input,
    ),
    "research_agent_full": AgentEntry(
        deep_researcher_builder,
        "Full Deep Research Agent",
        "End-to-end pipeline: clarify -> brief -> supervised research -> final report.",
        _messages_input,
    ),
}
