"""PostgreSQL storage for Memory Cards.

Replaces storage_sqlite.py with PostgreSQL backend on the workstation.
"""

from __future__ import annotations

import json
from typing import Dict, Optional
from datetime import datetime

from core.artifacts.memory_card import MemoryCard, MemoryApproval
from core.db.postgres import get_pool


async def load_all_cards() -> Dict[str, MemoryCard]:
    """Load all memory cards from PostgreSQL."""
    pool = await get_pool()
    cards: Dict[str, MemoryCard] = {}

    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM memory_cards")

        for row in rows:
            # Reconstruct card from PostgreSQL row
            card_data = {
                "id": row["id"],
                "created_at": row["created_at"].isoformat(),
                "category": row["category"],
                "scope": row["scope"],
                "text": row["text"],
                "approval": row["approval"],
                "approved_at": row["approved_at"].isoformat() if row["approved_at"] else None,
                "rejected_at": row["rejected_at"].isoformat() if row["rejected_at"] else None,
                "provenance": json.loads(row["provenance"]) if isinstance(row["provenance"], str) else row["provenance"],
            }
            card = MemoryCard.model_validate(card_data)
            cards[card.id] = card

    return cards


async def upsert_card(card: MemoryCard) -> None:
    """Save or update memory card in PostgreSQL."""
    pool = await get_pool()
    payload = card.model_dump(mode="json")

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO memory_cards(
                id, created_at, category, scope, text, approval,
                approved_at, rejected_at, provenance
            ) VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT(id) DO UPDATE SET
                category = EXCLUDED.category,
                scope = EXCLUDED.scope,
                text = EXCLUDED.text,
                approval = EXCLUDED.approval,
                approved_at = EXCLUDED.approved_at,
                rejected_at = EXCLUDED.rejected_at,
                provenance = EXCLUDED.provenance
            """,
            card.id,
            card.created_at,
            card.category.value,
            card.scope.value,
            card.text,
            card.approval.value,
            card.approved_at,
            card.rejected_at,
            json.dumps(payload["provenance"]),
        )


async def delete_card(card_id: str) -> bool:
    """Delete memory card from PostgreSQL."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM memory_cards WHERE id = $1",
            card_id,
        )
        # result is like "DELETE 1" - extract the count
        return "1" in result


async def get_card_by_id(card_id: str) -> Optional[MemoryCard]:
    """Load single memory card by ID."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM memory_cards WHERE id = $1",
            card_id,
        )

        if not row:
            return None

        card_data = {
            "id": row["id"],
            "created_at": row["created_at"].isoformat(),
            "category": row["category"],
            "scope": row["scope"],
            "text": row["text"],
            "approval": row["approval"],
            "approved_at": row["approved_at"].isoformat() if row["approved_at"] else None,
            "rejected_at": row["rejected_at"].isoformat() if row["rejected_at"] else None,
            "provenance": json.loads(row["provenance"]) if isinstance(row["provenance"], str) else row["provenance"],
        }
        return MemoryCard.model_validate(card_data)


async def get_approved_cards() -> Dict[str, MemoryCard]:
    """Load only approved memory cards (for RAG indexing)."""
    pool = await get_pool()
    cards: Dict[str, MemoryCard] = {}

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM memory_cards WHERE approval = $1",
            MemoryApproval.APPROVED.value,
        )

        for row in rows:
            card_data = {
                "id": row["id"],
                "created_at": row["created_at"].isoformat(),
                "category": row["category"],
                "scope": row["scope"],
                "text": row["text"],
                "approval": row["approval"],
                "approved_at": row["approved_at"].isoformat() if row["approved_at"] else None,
                "rejected_at": row["rejected_at"].isoformat() if row["rejected_at"] else None,
                "provenance": json.loads(row["provenance"]) if isinstance(row["provenance"], str) else row["provenance"],
            }
            card = MemoryCard.model_validate(card_data)
            cards[card.id] = card

    return cards
