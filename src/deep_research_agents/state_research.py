"""
State Definitions and Pydantic Schemas for Research Agent

This module defines the state objects and structured schemas used for
the research agent workflow, including researcher state management and output schemas.
"""

import operator
from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ===== STATE DEFINITIONS =====

class ResearcherState(TypedDict):
    """
    State for the research agent containing message history and research metadata.

    This state tracks the researcher's conversation, iteration count for limiting
    tool calls, the research topic being investigated, per-link extractions,
    cross-call link dedup, and raw research notes for detailed analysis.
    """
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    checklist: List[str]
    started_at: float
    visited_urls: Annotated[set[str], operator.or_]
    extractions: Annotated[List[dict], operator.add]
    research_findings: str
    coverage_gaps: List[str]
    raw_notes: Annotated[List[str], operator.add]
    timed_out: bool

class ResearcherOutputState(TypedDict):
    """
    Output state for the research agent containing final research results.

    This represents the final output of the research process: the concatenated,
    per-link extraction findings, informational coverage gaps against the
    derived checklist, and all raw notes from the research process.
    """
    research_findings: str
    coverage_gaps: List[str]
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decisions during scoping phase."""
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Schema for research brief generation."""
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class RelevantExtraction(BaseModel):
    """Schema for brief-aware per-link content extraction."""
    relevant: bool = Field(
        description="Whether this webpage contains information relevant to the research topic/checklist."
    )
    extracted_content: str = Field(
        description="Verbatim facts, figures, names, and dates relevant to the research topic — no paraphrasing away specifics. Empty string if not relevant."
    )
    covers: list[str] = Field(
        default=[],
        description="Which checklist items (if any) this content addresses.",
    )

class ResearchChecklist(BaseModel):
    """Schema for deriving a coverage checklist from a research topic."""
    checklist: list[str] = Field(
        default=[],
        description=(
            "Distinct sub-topics or entities that must each be covered to fully answer the "
            "research topic (e.g. ['Germany', 'Portugal'] for a multi-country comparison). "
            "Empty list if the topic is a single unified question with no natural sub-parts."
        ),
    )
