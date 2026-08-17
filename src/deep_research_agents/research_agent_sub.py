"""Research Sub-Agent Implementation.

This module implements a research sub-agent that can perform iterative web searches
and per-link, brief-aware extraction to answer complex research questions. It is
spawned by the supervisor (multi_agent_supervisor.py) to research one sub-topic at
a time, and is also independently reachable as its own graph via AGENT_REGISTRY.
"""

import asyncio
import logging
import time

from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig

from deep_research_agents.config import get_model, get_settings, session_kwargs, thread_id_from_config
from deep_research_agents.state_research import ResearcherState, ResearcherOutputState, ResearchChecklist
from deep_research_agents.utils import tavily_search, get_today_str, think_tool
from deep_research_agents.prompts import research_agent_prompt, derive_checklist_prompt

logger = logging.getLogger(__name__)

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
    first node the graph runs. Wrapped in the same timeout as every other model
    call in this file — a hang here used to skip stamping started_at entirely,
    which would silently disable the whole time-budget safety net. On timeout we
    fall back to checklist=[], the same value a single-topic question already
    produces, so this degrades gracefully rather than failing the run.

    TODO: when multi_agent_supervisor.py / research_agent_mcp.py get their own
    redesign pass, apply this same asyncio.wait_for(subagent_call_timeout_seconds)
    pattern to their LLM calls too — nothing wraps them today.
    """
    thread_id = thread_id_from_config(config)
    settings = get_settings()
    structured_model = checklist_model.with_structured_output(ResearchChecklist)
    try:
        result = await asyncio.wait_for(
            structured_model.ainvoke([
                HumanMessage(content=derive_checklist_prompt.format(
                    research_topic=state["research_topic"],
                    date=get_today_str(),
                ))
            ], **session_kwargs(thread_id)),
            timeout=settings.subagent_call_timeout_seconds,
        )
        checklist = result.checklist
        logger.info("thread=%s derive_checklist -> %r", thread_id, checklist)
    except asyncio.TimeoutError:
        checklist = []
        logger.warning("thread=%s derive_checklist timed out, falling back to checklist=[]", thread_id)
    return {
        "checklist": checklist,
        "started_at": time.time(),
    }

async def llm_call(state: ResearcherState, config: RunnableConfig) -> Command[Literal["tool_node", "finalize_research"]]:
    """Analyze current state, decide whether to keep researching, and route accordingly.

    This node owns the full stop/continue decision and returns a `Command` that
    both picks the next node *and* writes `stop_reason` in one atomic step —
    replacing a separate `should_continue` conditional-edge function plus a
    shared `_stop_reason(state)` helper called independently from two places
    (routing and `finalize_research`). That split had a real bug: `_stop_reason`
    read `time.time()` fresh on each call, so the routing call and the
    persisting call could disagree near the `subagent_time_budget_seconds`
    boundary if real time (e.g. an async checkpoint write) elapsed between them
    — mislabeling a genuine model-decided stop as a forced cutoff. Deciding
    once, here, makes that race structurally impossible: there's only one
    `time.time()` read and one place `stop_reason` is ever set.

    Resource caps (links/iterations/time budget) are checked *before* calling
    the model at all — state, previously, had already skipped a wasted LLM call
    since should_continue only ever looked at these to override the model's
    tool-call decision after the fact. Checking them first also means a capped
    run doesn't spend one last (immediately-discarded) round-trip on a
    now-moot response.

    A per-call timeout routes straight to `finalize_research` with
    `stop_reason="llm_timeout"` rather than raising — the caller's bare
    `except Exception` would otherwise mark the whole thread failed for what's
    really just one slow call. No fabricated AIMessage is added to history:
    researcher_messages is checkpointed per-thread, so a synthetic "I decided
    to stop" message would persist and be replayed to the model on any later
    resume of this thread, indistinguishable from a genuine self-produced
    decision.
    """
    thread_id = thread_id_from_config(config)
    settings = get_settings()

    visited_urls = state.get("visited_urls", set())
    tool_call_iterations = state.get("tool_call_iterations", 0)
    started_at = state.get("started_at", time.time())

    if len(visited_urls) >= settings.max_subagent_links:
        stop_reason = "link_cap"
    elif tool_call_iterations >= settings.max_subagent_tool_iterations:
        stop_reason = "iteration_cap"
    elif time.time() - started_at >= settings.subagent_time_budget_seconds:
        stop_reason = "time_budget"
    else:
        stop_reason = None

    if stop_reason is not None:
        logger.warning("thread=%s llm_call: stop_reason=%s (resource cap, skipping model call)", thread_id, stop_reason)
        return Command(goto="finalize_research", update={"stop_reason": stop_reason})

    try:
        response = await asyncio.wait_for(
            model_with_tools.ainvoke(
                [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"],
                **session_kwargs(thread_id),
            ),
            timeout=settings.subagent_call_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("thread=%s llm_call timed out after %ss", thread_id, settings.subagent_call_timeout_seconds)
        return Command(goto="finalize_research", update={"stop_reason": "llm_timeout"})
    except Exception as e:
        # Broader than the timeout above on purpose: a rate limit, a malformed
        # response, or any other model-call failure would otherwise propagate
        # uncaught out of this node and fail the whole run for what, from the
        # caller's perspective, is the same class of problem as a timeout --
        # one bad round shouldn't lose whatever research this sub-agent has
        # already gathered. Distinct stop_reason from llm_timeout so it's not
        # mistaken for the (expected, budgeted-for) timeout case in logs/output.
        logger.error("thread=%s llm_call failed: %s", thread_id, e)
        return Command(goto="finalize_research", update={"stop_reason": "llm_error"})

    # response is appended, not replacing prior history, via researcher_messages'
    # add_messages reducer (state_research.py) -- returning [response] here is
    # correct even though it looks like it would overwrite everything else.
    if response.tool_calls:
        return Command(goto="tool_node", update={"researcher_messages": [response]})

    return Command(
        goto="finalize_research",
        update={"researcher_messages": [response], "stop_reason": "model_decided"},
    )

async def tool_node(state: ResearcherState, config: RunnableConfig):
    """Execute all tool calls from the previous LLM response.

    tavily_search needs research_topic/checklist/already_visited/session_id that
    the LLM doesn't (and shouldn't) supply — this hand-rolled tool_node injects
    them from state/config at the call site, same special-casing pattern
    research_agent_mcp.py already uses for think_tool vs MCP tools.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls
    session_id = thread_id_from_config(config)
    iteration = state.get("tool_call_iterations", 0) + 1
    logger.info(
        "thread=%s tool_node round %d: dispatching %s",
        session_id, iteration, [tc["name"] for tc in tool_calls],
    )

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

    # Union, not a sum: new_urls is disjoint from the prior visited_urls today
    # (to_extract already filters out already_visited), but a union is
    # correct regardless of that invariant holding in the future -- e.g. a
    # refetch-style tool that deliberately reprocesses an already-visited URL
    # would make new_urls overlap, and len(a) + len(b) would then overcount.
    total_visited = state.get("visited_urls", set()) | new_urls
    logger.info(
        "thread=%s tool_node round %d done: %d new URL(s) extracted, %d total visited",
        session_id, iteration, len(new_urls), len(total_visited),
    )

    return {
        # extractions/visited_urls have reducers (operator.add/operator.or_ in
        # state_research.py) that merge this round's values into the running
        # total automatically -- these two fields are the round's *increment*,
        # not the accumulated total. tool_call_iterations has no reducer (plain
        # replace), so `iteration` above is computed as the accumulated total
        # itself (old value + 1) and returned as-is -- an intentionally
        # different pattern from its two neighbors here, not an oversight.
        "researcher_messages": tool_outputs,
        "extractions": new_extractions,
        "visited_urls": new_urls,
        "tool_call_iterations": iteration,
    }

def finalize_research(state: ResearcherState, config: RunnableConfig) -> dict:
    """Concatenate every relevant extraction into research_findings — pure Python, no LLM call.

    Also computes coverage_gaps against the derived checklist (point 4) —
    informational only, does not affect routing within this graph. stop_reason
    records *why* this node was reached at all — model_decided is the only
    "research complete" case; the other four (llm_timeout/link_cap/
    iteration_cap/time_budget) are a forced cutoff, which matters to anything
    downstream that might otherwise read "research finished" as "the model was
    satisfied it had enough."

    TODO: coverage_gaps is not read anywhere yet. It's meant to eventually be
    surfaced to multi_agent_supervisor.py (e.g. folded into the ToolMessage
    content alongside research_findings) so the supervisor can judge whether a
    sub-agent's research actually covered its assigned sub-topic and decide
    whether to re-dispatch. Wire this up when multi_agent_supervisor.py gets
    its own redesign pass — today result.get("research_findings", ...) is the
    only field multi_agent_supervisor.py reads off a sub-agent's output.
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

    # llm_call is the only place stop_reason is ever set, and it does so on
    # every path that routes here via Command — guaranteed present by
    # construction, no recomputation or fallback needed.
    stop_reason = state["stop_reason"]
    log = logger.warning if stop_reason != "model_decided" else logger.info
    log(
        "thread=%s finalize_research: stop_reason=%s, %d/%d relevant extraction(s), coverage_gaps=%r",
        thread_id_from_config(config), stop_reason, len(relevant), len(extractions), coverage_gaps,
    )

    return {
        "research_findings": research_findings,
        "coverage_gaps": coverage_gaps,
        "stop_reason": stop_reason,
        "raw_notes": ["\n".join(raw_notes)],
    }

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
researcher_agent_sub_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
researcher_agent_sub_builder.add_node("derive_checklist", derive_checklist)
researcher_agent_sub_builder.add_node("llm_call", llm_call)
researcher_agent_sub_builder.add_node("tool_node", tool_node)
researcher_agent_sub_builder.add_node("finalize_research", finalize_research)

# Add edges to connect nodes. llm_call routes itself via the Command it
# returns (to "tool_node" or "finalize_research"), same pattern already used
# by multi_agent_supervisor.py's supervisor/supervisor_tools — no
# add_conditional_edges needed for it.
researcher_agent_sub_builder.add_edge(START, "derive_checklist")
researcher_agent_sub_builder.add_edge("derive_checklist", "llm_call")
researcher_agent_sub_builder.add_edge("tool_node", "llm_call")  # Loop back for more research
researcher_agent_sub_builder.add_edge("finalize_research", END)

# Compile the agent
researcher_agent_sub = researcher_agent_sub_builder.compile()
