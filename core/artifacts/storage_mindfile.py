from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from core.artifacts.mindfile_entry import MindFileEntry, MindFileEntryCategory

# Same database as memory cards and sessions
DB_PATH = Path("data/memory/memory_cards.sqlite3")

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS mindfile_entries (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  category TEXT NOT NULL,
  json TEXT NOT NULL
);
"""


async def init_mindfile_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        await db.execute(CREATE_SQL)
        await db.commit()


async def save_entry(entry: MindFileEntry) -> None:
    await init_mindfile_db()
    payload = entry.model_dump(mode="json")
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        await db.execute(
            "INSERT INTO mindfile_entries(id, created_at, category, json) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET category=excluded.category, json=excluded.json",
            (entry.id, entry.created_at.isoformat(), entry.category.value, json.dumps(payload)),
        )
        await db.commit()


async def load_all_entries(
    category: Optional[MindFileEntryCategory] = None,
) -> list[MindFileEntry]:
    await init_mindfile_db()
    entries: list[MindFileEntry] = []
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        if category:
            sql = "SELECT json FROM mindfile_entries WHERE category = ? ORDER BY created_at DESC"
            cursor = await db.execute(sql, (category.value,))
        else:
            sql = "SELECT json FROM mindfile_entries ORDER BY created_at DESC"
            cursor = await db.execute(sql)
        async for (json_str,) in cursor:
            data = json.loads(json_str)
            entries.append(MindFileEntry.model_validate(data))
    return entries


async def load_entry(entry_id: str) -> Optional[MindFileEntry]:
    await init_mindfile_db()
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        cursor = await db.execute(
            "SELECT json FROM mindfile_entries WHERE id = ?", (entry_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return MindFileEntry.model_validate(json.loads(row[0]))


async def update_note(entry_id: str, note: str) -> Optional[MindFileEntry]:
    entry = await load_entry(entry_id)
    if not entry:
        return None
    entry.note = note
    entry.updated_at = datetime.utcnow()
    await save_entry(entry)
    return entry


async def delete_entry(entry_id: str) -> bool:
    await init_mindfile_db()
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        cursor = await db.execute(
            "DELETE FROM mindfile_entries WHERE id = ?", (entry_id,)
        )
        await db.commit()
        return cursor.rowcount > 0


async def entry_exists_for_card(card_id: str) -> bool:
    """Check if a Mind File entry already exists for a given memory card."""
    await init_mindfile_db()
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        cursor = await db.execute(
            "SELECT 1 FROM mindfile_entries WHERE json LIKE ? LIMIT 1",
            (f'%"source_memory_card_id": "{card_id}"%',),
        )
        return await cursor.fetchone() is not None
