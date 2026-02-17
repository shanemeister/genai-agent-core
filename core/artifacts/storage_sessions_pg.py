"""PostgreSQL storage for Chat Sessions.

Replaces storage_sessions.py SQLite with PostgreSQL backend.
"""

from __future__ import annotations

import json
from typing import List, Optional
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from core.db.postgres import get_pool


class ChatMessage(BaseModel):
    """Chat message model (same as storage_sessions.py)."""
    role: str  # "user" | "assistant"
    text: str
    model: Optional[str] = None
    processing_time: Optional[float] = None
    retrieved_context: List[dict] = Field(default_factory=list)
    session_id: Optional[str] = None
    proposed_memories: List[dict] = Field(default_factory=list)
    reasoning: Optional[str] = None


class ChatSession(BaseModel):
    """Chat session model (same as storage_sessions.py)."""
    id: str
    title: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    messages: List[ChatMessage] = Field(default_factory=list)


async def save_session(session: ChatSession) -> None:
    """Save or update chat session in PostgreSQL."""
    pool = await get_pool()
    session.updated_at = datetime.now(timezone.utc)

    messages_json = [m.model_dump(mode="json") for m in session.messages]

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO chat_sessions(id, title, created_at, updated_at, messages)
               VALUES($1, $2, $3, $4, $5)
               ON CONFLICT(id) DO UPDATE SET
                   title = EXCLUDED.title,
                   updated_at = EXCLUDED.updated_at,
                   messages = EXCLUDED.messages
            """,
            session.id,
            session.title,
            session.created_at,
            session.updated_at,
            json.dumps(messages_json),  # JSONB in PostgreSQL
        )


async def load_session(session_id: str) -> Optional[ChatSession]:
    """Load chat session from PostgreSQL."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, title, created_at, updated_at, messages FROM chat_sessions WHERE id = $1",
            session_id,
        )

        if not row:
            return None

        # PostgreSQL returns JSONB as parsed Python dict
        messages_data = row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"])

        return ChatSession(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            messages=[ChatMessage.model_validate(m) for m in messages_data],
        )


async def list_sessions(limit: int = 50) -> List[dict]:
    """Return session summaries (without full messages) sorted newest first."""
    pool = await get_pool()
    sessions: List[dict] = []

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, title, created_at, updated_at, messages
               FROM chat_sessions
               ORDER BY updated_at DESC
               LIMIT $1
            """,
            limit,
        )

        for row in rows:
            messages_data = row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"])

            sessions.append({
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "message_count": len(messages_data),
                "preview": messages_data[0]["text"][:80] if messages_data else "",
            })

    return sessions


async def delete_session(session_id: str) -> bool:
    """Delete chat session from PostgreSQL."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM chat_sessions WHERE id = $1",
            session_id,
        )
        return "1" in result
