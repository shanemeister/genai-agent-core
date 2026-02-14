from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import aiosqlite

from core.artifacts.memory_card import MemoryCard

DB_PATH = Path("data/memory/memory_cards.sqlite3")

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS memory_cards (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  approval TEXT NOT NULL,
  json TEXT NOT NULL
);
"""

async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        await db.execute(CREATE_SQL)
        await db.commit()

async def load_all_cards() -> Dict[str, MemoryCard]:
    await init_db()
    cards: Dict[str, MemoryCard] = {}
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        async with db.execute("SELECT json FROM memory_cards") as cur:
            async for (json_str,) in cur:
                data = json.loads(json_str)
                card = MemoryCard.model_validate(data)
                cards[card.id] = card
    return cards

async def upsert_card(card: MemoryCard) -> None:
    await init_db()
    payload = card.model_dump(mode="json")
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        await db.execute(
            "INSERT INTO memory_cards(id, created_at, approval, json) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET approval=excluded.approval, json=excluded.json",
            (card.id, card.created_at.isoformat(), card.approval.value, json.dumps(payload)),
        )
        await db.commit()