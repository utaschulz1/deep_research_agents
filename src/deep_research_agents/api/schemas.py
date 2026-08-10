"""Pydantic request/response schemas for the FastAPI routes."""
from typing import Any

from pydantic import BaseModel


class AgentInfo(BaseModel):
    id: str
    name: str
    description: str


class CreateThreadRequest(BaseModel):
    name: str
    agent_id: str


class ThreadResponse(BaseModel):
    thread_id: str
    name: str
    agent_id: str
    status: str
    error_detail: str | None = None
    created_at: str
    updated_at: str


class RunRequest(BaseModel):
    message: str


class StateUpdateRequest(BaseModel):
    values: dict[str, Any]


class ReviseRequest(BaseModel):
    instruction: str
