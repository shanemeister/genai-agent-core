from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import aiosqlite
from pydantic import BaseModel, Field
from datetime import datetime, timezone


DB_PATH = Path("data/memory/memory_cards.sqlite3")

CREATE_SESSIONS_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  messages TEXT NOT NULL
);
"""


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    text: str
    model: Optional[str] = None
    processing_time: Optional[float] = None
    retrieved_context: List[dict] = Field(default_factory=list)
    session_id: Optional[str] = None
    proposed_memories: List[dict] = Field(default_factory=list)
    reasoning: Optional[str] = None


class ChatSession(BaseModel):
    id: str
    title: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    messages: List[ChatMessage] = Field(default_factory=list)


async def init_sessions_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        await db.execute(CREATE_SESSIONS_SQL)
        await db.commit()


async def save_session(session: ChatSession) -> None:
    await init_sessions_db()
    session.updated_at = datetime.now(timezone.utc)
    messages_json = json.dumps([m.model_dump(mode="json") for m in session.messages])
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        await db.execute(
            "INSERT INTO chat_sessions(id, title, created_at, updated_at, messages) "
            "VALUES(?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET title=excluded.title, updated_at=excluded.updated_at, messages=excluded.messages",
            (
                session.id,
                session.title,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                messages_json,
            ),
        )
        await db.commit()


async def load_session(session_id: str) -> Optional[ChatSession]:
    await init_sessions_db()
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        async with db.execute(
            "SELECT id, title, created_at, updated_at, messages FROM chat_sessions WHERE id = ?",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return ChatSession(
                id=row[0],
                title=row[1],
                created_at=datetime.fromisoformat(row[2]),
                updated_at=datetime.fromisoformat(row[3]),
                messages=[ChatMessage.model_validate(m) for m in json.loads(row[4])],
            )


async def list_sessions(limit: int = 50) -> List[dict]:
    """Return session summaries (without full messages) sorted newest first."""
    await init_sessions_db()
    sessions: List[dict] = []
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        async with db.execute(
            "SELECT id, title, created_at, updated_at, messages FROM chat_sessions "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            async for row in cur:
                messages = json.loads(row[4])
                sessions.append({
                    "id": row[0],
                    "title": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                    "message_count": len(messages),
                    "preview": messages[0]["text"][:80] if messages else "",
                })
    return sessions


async def delete_session(session_id: str) -> bool:
    await init_sessions_db()
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        cursor = await db.execute(
            "DELETE FROM chat_sessions WHERE id = ?", (session_id,)
        )
        await db.commit()
        return cursor.rowcount > 0
