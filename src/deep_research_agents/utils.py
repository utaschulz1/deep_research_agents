"""Research Utilities and Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and brief-aware content extraction tools.
"""

import asyncio
from pathlib import Path
from datetime import datetime
from typing_extensions import Annotated, List, Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from tavily import AsyncTavilyClient

from deep_research_agents.config import get_model, get_settings, session_kwargs
from deep_research_agents.state_research import RelevantExtraction
from deep_research_agents.prompts import extract_relevant_content_prompt

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def get_current_dir() -> Path:
    """Get the current directory of the module.

    This function is compatible with Jupyter notebooks and regular Python scripts.

    Returns:
        Path object representing the current directory
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ is not defined
        return Path.cwd()

# ===== CONFIGURATION =====

extraction_model = get_model("extract")
tavily_client = AsyncTavilyClient(api_key=get_settings().tavily_api_key)

# ===== SEARCH FUNCTIONS =====

async def extract_relevant_content(
    webpage_content: str,
    research_topic: str,
    checklist: List[str],
    search_query: str,
    session_id: str | None = None,
) -> RelevantExtraction | None:
    """Extract topic-relevant, verbatim content from a webpage via the configured extraction model.

    Args:
        webpage_content: Raw webpage content to extract from
        research_topic: The specific research topic/brief guiding relevance
        checklist: Sub-topics/entities this content should be checked for coverage of
        search_query: The specific tavily_search query that surfaced this URL — the
            primary relevance anchor (see extract_relevant_content_prompt), with
            research_topic kept as broader context so on-brief-but-off-query content
            (e.g. a page that corrects a wrong assumption in the query) isn't dropped.
        session_id: OpenRouter session grouping key (the LangGraph thread_id), if known

    Returns:
        A RelevantExtraction (which may itself have relevant=False — a considered
        "not relevant" judgment). Returns None if both attempts below fail — the
        caller falls back to a raw excerpt of webpage_content rather than losing
        the page's content outright.
    """
    structured_model = extraction_model.with_structured_output(RelevantExtraction)
    prompt = [HumanMessage(content=extract_relevant_content_prompt.format(
        webpage_content=webpage_content,
        research_topic=research_topic,
        checklist=", ".join(checklist) if checklist else "(none specified)",
        search_query=search_query,
        date=get_today_str(),
    ))]
    for attempt in range(2):  # one retry, in case the first failure was transient
        try:
            return await asyncio.wait_for(
                structured_model.ainvoke(prompt, **session_kwargs(session_id)),
                timeout=get_settings().subagent_call_timeout_seconds,
            )
        except Exception as e:
            print(f"extract_relevant_content attempt {attempt + 1} failed: {e}")
    return None

def deduplicate_search_results(search_results: List[dict]) -> dict:
    """Deduplicate search results by URL to avoid processing duplicate content.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results = {}

    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result

    return unique_results

# ===== RESEARCH TOOLS =====

@tool(parse_docstring=True)
async def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    research_topic: Annotated[str, InjectedToolArg] = "",
    checklist: Annotated[List[str] | None, InjectedToolArg] = None,
    already_visited: Annotated[set, InjectedToolArg] = frozenset(),
    session_id: Annotated[str | None, InjectedToolArg] = None,
) -> tuple[str, list[dict]]:
    """Fetch results from Tavily search API with brief-aware content extraction.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')
        research_topic: The research brief guiding what content is relevant (injected by tool_node)
        checklist: Sub-topics/entities to tag extracted content against (injected by tool_node)
        already_visited: URLs already extracted in this run, to skip re-extracting (injected by tool_node)
        session_id: OpenRouter session grouping key, the LangGraph thread_id (injected by tool_node)

    Returns:
        A tuple of (formatted string of relevant extracted content for the LLM,
        list of per-URL extraction records for state aggregation).
    """
    checklist = checklist or []  # default is None, not [], to avoid a shared mutable default

    # Execute search, bounded by the same per-call timeout as everything downstream of it
    try:
        result = await asyncio.wait_for(
            tavily_client.search(
                query,
                max_results=max_results,
                include_raw_content=True,
                topic=topic,
            ),
            timeout=get_settings().subagent_call_timeout_seconds,
        )
    except asyncio.TimeoutError:
        return "Search timed out — no results for this query.\n", []

    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = deduplicate_search_results([result])

    # Extract every not-yet-visited URL concurrently — total wait for the batch is
    # bounded by the slowest single extraction call's timeout, not their sum.
    to_extract = [(url, r) for url, r in unique_results.items() if url not in already_visited]
    skipped_count = len(unique_results) - len(to_extract)

    extractions = await asyncio.gather(*[
        extract_relevant_content(
            r.get("raw_content") or r.get("content") or "",
            research_topic, checklist, query, session_id,
        )
        for url, r in to_extract
    ])

    extraction_records = []
    relevant_count = 0
    formatted_output = "Search results: \n\n"

    for (url, r), extraction in zip(to_extract, extractions):
        if extraction is None:
            # Extraction failed even after retrying — fall back to a raw excerpt
            # rather than losing the page's content entirely. Included in
            # extraction_records (and therefore visited_urls) so it isn't
            # reprocessed forever; extraction_failed marks it as unreviewed/raw
            # rather than a considered relevance judgment.
            raw = (r.get("raw_content") or r.get("content") or "")[:1000]
            extraction_records.append({
                "url": url,
                "title": r["title"],
                "relevant": True,
                "extracted_content": raw,
                "covers": [],
                "extraction_failed": True,
            })
            relevant_count += 1
            formatted_output += f"\n\n--- SOURCE {relevant_count}: {r['title']} (raw excerpt — extraction failed) ---\n"
            formatted_output += f"URL: {url}\n\n{raw}\n\n"
            formatted_output += "-" * 80 + "\n"
            continue

        extraction_records.append({
            "url": url,
            "title": r["title"],
            "relevant": extraction.relevant,
            "extracted_content": extraction.extracted_content,
            "covers": extraction.covers,
        })

        if extraction.relevant:
            relevant_count += 1
            formatted_output += f"\n\n--- SOURCE {relevant_count}: {r['title']} ---\n"
            formatted_output += f"URL: {url}\n\n"
            formatted_output += f"{extraction.extracted_content}\n\n"
            formatted_output += "-" * 80 + "\n"

    if skipped_count:
        formatted_output += f"\n({skipped_count} result(s) skipped — already visited in this research run.)\n"

    if relevant_count == 0:
        formatted_output += "\nNo relevant new information found in this search's results.\n"

    return formatted_output, extraction_records

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
