"""Research Sub-Agent Implementation.

This module implements a research sub-agent that can perform iterative web searches
and per-link, brief-aware extraction to answer complex research questions. It is
spawned by the supervisor (multi_agent_supervisor.py) to research one sub-topic at
a time, and is also independently reachable as its own graph via AGENT_REGISTRY.
"""

import asyncio
import time

from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig

from deep_research_agents.config import get_model, get_settings, session_kwargs, thread_id_from_config
from deep_research_agents.state_research import ResearcherState, ResearcherOutputState, ResearchChecklist
from deep_research_agents.utils import tavily_search, get_today_str, think_tool
from deep_research_agents.prompts import research_agent_prompt, derive_checklist_prompt

# ===== CONFIGURATION =====

# Set up tools and model binding
tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

# Initialize models
model = get_model("research")
model_with_tools = model.bind_tools(tools)
checklist_model = get_model("scope")  # temperature=0.0, same lightweight-judgment role as scoping

# ===== AGENT NODES =====

async def derive_checklist(state: ResearcherState, config: RunnableConfig) -> dict:
    """One-time, run-start structured-output call turning research_topic into a checklist.

    Also stamps started_at for point 5's time budget, since this is always the
    first node the graph runs.
    """
    structured_model = checklist_model.with_structured_output(ResearchChecklist)
    result = await structured_model.ainvoke([
        HumanMessage(content=derive_checklist_prompt.format(
            research_topic=state["research_topic"],
            date=get_today_str(),
        ))
    ], **session_kwargs(thread_id_from_config(config)))
    return {
        "checklist": result.checklist,
        "started_at": time.time(),
    }

async def llm_call(state: ResearcherState, config: RunnableConfig):
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response. A per-call timeout treats a
    hung call as "no tool calls" (routes to finalize_research) rather than raising —
    the caller's bare `except Exception` would otherwise mark the whole thread failed
    for what's really just one slow call.
    """
    settings = get_settings()
    try:
        response = await asyncio.wait_for(
            model_with_tools.ainvoke(
                [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"],
                **session_kwargs(thread_id_from_config(config)),
            ),
            timeout=settings.subagent_call_timeout_seconds,
        )
    except asyncio.TimeoutError:
        response = AIMessage(content="[llm_call timed out — stopping research for this sub-agent]")

    return {"researcher_messages": [response]}

async def tool_node(state: ResearcherState, config: RunnableConfig):
    """Execute all tool calls from the previous LLM response.

    tavily_search needs research_topic/checklist/already_visited/session_id that
    the LLM doesn't (and shouldn't) supply — this hand-rolled tool_node injects
    them from state/config at the call site, same special-casing pattern
    research_agent_mcp.py already uses for think_tool vs MCP tools.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls
    session_id = thread_id_from_config(config)

    async def _invoke_tool(tool_call):
        tool = tools_by_name[tool_call["name"]]
        if tool_call["name"] == "tavily_search":
            call_args = {
                **tool_call["args"],
                "research_topic": state["research_topic"],
                "checklist": state.get("checklist", []),
                "already_visited": state.get("visited_urls", set()),
                "session_id": session_id,
            }
            return await tool.ainvoke(call_args)
        return tool.invoke(tool_call["args"])  # think_tool: sync, no I/O, negligible either way

    # Run all tool calls in this round concurrently — e.g. two tavily_search calls
    # (one per country) no longer run back-to-back. asyncio.gather preserves input
    # order, so the zip(observations, tool_calls) below needs no changes.
    observations = await asyncio.gather(*[_invoke_tool(tc) for tc in tool_calls])

    tool_outputs = []
    new_extractions = []
    new_urls = set()
    for observation, tool_call in zip(observations, tool_calls):
        if tool_call["name"] == "tavily_search":
            formatted_text, extraction_records = observation
            content = formatted_text
            new_extractions.extend(extraction_records)
            new_urls.update(record["url"] for record in extraction_records)
        else:
            content = observation
        tool_outputs.append(
            ToolMessage(content=content, name=tool_call["name"], tool_call_id=tool_call["id"])
        )

    return {
        "researcher_messages": tool_outputs,
        "extractions": new_extractions,
        "visited_urls": new_urls,
        "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
    }

def finalize_research(state: ResearcherState) -> dict:
    """Concatenate every relevant extraction into research_findings — pure Python, no LLM call.

    Also computes coverage_gaps against the derived checklist (point 4) —
    informational only, does not affect routing.
    """
    extractions = state.get("extractions", [])
    checklist = state.get("checklist", [])

    relevant = [e for e in extractions if e.get("relevant")]
    research_findings = "\n\n".join(
        f"SOURCE: {e['title']} — {e['url']}\n{e['extracted_content']}"
        for e in relevant
    ) or "No relevant findings were extracted."

    covered = set()
    for e in relevant:
        covered.update(e.get("covers", []))
    coverage_gaps = [item for item in checklist if item not in covered]

    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"],
            include_types=["tool", "ai"]
        )
    ]

    return {
        "research_findings": research_findings,
        "coverage_gaps": coverage_gaps,
        "raw_notes": ["\n".join(raw_notes)],
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "finalize_research"]:
    """Determine whether to continue research or finalize.

    Point 5 safety net is checked first and, unlike point 4's coverage flag, DOES
    override the model's own stop decision — it's a resource cap, not a quality
    judgment. Otherwise, falls through to the existing tool-call-based routing.
    """
    settings = get_settings()
    visited_urls = state.get("visited_urls", set())
    tool_call_iterations = state.get("tool_call_iterations", 0)
    started_at = state.get("started_at", time.time())

    if (
        len(visited_urls) >= settings.max_subagent_links
        or tool_call_iterations >= settings.max_subagent_tool_iterations
        or time.time() - started_at >= settings.subagent_time_budget_seconds
    ):
        return "finalize_research"

    last_message = state["researcher_messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return "finalize_research"

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
researcher_agent_sub_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
researcher_agent_sub_builder.add_node("derive_checklist", derive_checklist)
researcher_agent_sub_builder.add_node("llm_call", llm_call)
researcher_agent_sub_builder.add_node("tool_node", tool_node)
researcher_agent_sub_builder.add_node("finalize_research", finalize_research)

# Add edges to connect nodes
researcher_agent_sub_builder.add_edge(START, "derive_checklist")
researcher_agent_sub_builder.add_edge("derive_checklist", "llm_call")
researcher_agent_sub_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",  # Continue research loop
        "finalize_research": "finalize_research",  # Provide final answer
    },
)
researcher_agent_sub_builder.add_edge("tool_node", "llm_call")  # Loop back for more research
researcher_agent_sub_builder.add_edge("finalize_research", END)

# Compile the agent
researcher_agent_sub = researcher_agent_sub_builder.compile()
