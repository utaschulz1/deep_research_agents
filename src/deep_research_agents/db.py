"""Thread metadata store.

A plain table in the same sqlite file the checkpointer uses, but through its own
aiosqlite connection — not routed through the checkpointer's own tables.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error_detail TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


class ThreadStore:
    """Async CRUD over the `threads` metadata table.

    Known, accepted phase-1 limitation: `status` isn't tied to any in-process
    task registry, so a server restart mid-run can leave a thread stuck showing
    "running" forever. Fine for local iteration; worth a startup sweep before
    deploying to Railway (see federated-hopping-walrus.md plan).
    """

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn
        self._conn.row_factory = aiosqlite.Row

    @classmethod
    async def connect(cls, sqlite_path: str) -> "ThreadStore":
        conn = await aiosqlite.connect(sqlite_path)
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute(_CREATE_TABLE)
        await conn.commit()
        return cls(conn)

    async def close(self) -> None:
        await self._conn.close()

    async def create_thread(self, name: str, agent_id: str) -> dict[str, Any]:
        thread_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            "INSERT INTO threads (thread_id, name, agent_id, status, error_detail, created_at, updated_at) "
            "VALUES (?, ?, ?, 'idle', NULL, ?, ?)",
            (thread_id, name, agent_id, now, now),
        )
        await self._conn.commit()
        return await self.get_thread(thread_id)

    async def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        async with self._conn.execute(
            "SELECT * FROM threads WHERE thread_id = ?", (thread_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_threads(self) -> list[dict[str, Any]]:
        async with self._conn.execute(
            "SELECT * FROM threads ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def update_status(self, thread_id: str, status: str, error_detail: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            "UPDATE threads SET status = ?, error_detail = ?, updated_at = ? WHERE thread_id = ?",
            (status, error_detail, now, thread_id),
        )
        await self._conn.commit()
