"""PostgreSQL storage for Mind File entries.

Replaces storage_mindfile.py SQLite with PostgreSQL backend.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from core.artifacts.mindfile_entry import MindFileEntry, MindFileEntryCategory
from core.db.postgres import get_pool


async def save_entry(entry: MindFileEntry) -> None:
    """Save or update mind file entry in PostgreSQL."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO mindfile_entries(
                id, created_at, updated_at, category, text,
                source_memory_card_id, note
            ) VALUES($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT(id) DO UPDATE SET
                updated_at = EXCLUDED.updated_at,
                category = EXCLUDED.category,
                text = EXCLUDED.text,
                note = EXCLUDED.note
            """,
            entry.id,
            entry.created_at,
            entry.updated_at,
            entry.category.value,
            entry.text,
            entry.source_memory_card_id,
            entry.note or "",
        )


async def load_all_entries(
    category: Optional[MindFileEntryCategory] = None,
) -> list[MindFileEntry]:
    """Load mind file entries, optionally filtered by category."""
    pool = await get_pool()
    entries: list[MindFileEntry] = []

    async with pool.acquire() as conn:
        if category:
            rows = await conn.fetch(
                "SELECT * FROM mindfile_entries WHERE category = $1 ORDER BY created_at DESC",
                category.value,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM mindfile_entries ORDER BY created_at DESC"
            )

        for row in rows:
            entries.append(MindFileEntry(
                id=row["id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                category=MindFileEntryCategory(row["category"]),
                text=row["text"],
                source_memory_card_id=row["source_memory_card_id"],
                note=row["note"] if row["note"] else None,
            ))

    return entries


async def load_entry(entry_id: str) -> Optional[MindFileEntry]:
    """Load single mind file entry by ID."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM mindfile_entries WHERE id = $1",
            entry_id,
        )

        if not row:
            return None

        return MindFileEntry(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            category=MindFileEntryCategory(row["category"]),
            text=row["text"],
            source_memory_card_id=row["source_memory_card_id"],
            note=row["note"] if row["note"] else None,
        )


async def update_note(entry_id: str, note: str) -> Optional[MindFileEntry]:
    """Update note on a mind file entry."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE mindfile_entries SET note = $1, updated_at = $2 WHERE id = $3",
            note,
            datetime.utcnow(),
            entry_id,
        )

        if "0" in result:
            return None

    return await load_entry(entry_id)


async def delete_entry(entry_id: str) -> bool:
    """Delete mind file entry."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM mindfile_entries WHERE id = $1",
            entry_id,
        )
        return "1" in result


async def entry_exists_for_card(card_id: str) -> bool:
    """Check if a Mind File entry already exists for a given memory card."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM mindfile_entries WHERE source_memory_card_id = $1 LIMIT 1",
            card_id,
        )
        return row is not None
