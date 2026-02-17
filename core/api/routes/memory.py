"""Memory card endpoints: propose, list, approve, reject, seed."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.artifacts.memory_card import (
    MemoryApproval,
    MemoryCard,
    MemoryCategory,
    MemoryProvenance,
    MemoryScope,
)
from core.artifacts.mindfile_entry import MindFileEntry, MindFileEntryCategory
from core.artifacts.storage_memory_pg import upsert_card
from core.artifacts.storage_mindfile_pg import entry_exists_for_card, save_entry as save_mindfile_entry
from core.graph import queries as graph_queries
from core.api.shared import MEMORY_CARDS

log = logging.getLogger("noesis.memory")

router = APIRouter(tags=["memory"])


class ProposeMemoryRequest(BaseModel):
    text: str
    category: MemoryCategory
    scope: MemoryScope = MemoryScope.PROJECT
    reason: str
    derived_from_artifact_ids: List[str] = []
    tools_used: List[str] = []
    model: Optional[str] = None


@router.post("/memory/propose", response_model=MemoryCard)
async def propose_memory(req: ProposeMemoryRequest):
    card = MemoryCard(
        text=req.text.strip(),
        category=req.category,
        scope=req.scope,
        provenance=MemoryProvenance(
            reason=req.reason.strip(),
            derived_from_artifact_ids=req.derived_from_artifact_ids,
            tools_used=req.tools_used,
            model=req.model,
            sources=[],
        ),
    )
    MEMORY_CARDS[card.id] = card
    await upsert_card(card)
    return card


@router.get("/memory/cards", response_model=List[MemoryCard])
def list_cards(
    approval: Optional[MemoryApproval] = Query(default=None),
    category: Optional[MemoryCategory] = Query(default=None),
    q: Optional[str] = Query(default=None, description="substring search"),
):
    items = list(MEMORY_CARDS.values())

    if approval:
        items = [c for c in items if c.approval == approval]
    if category:
        items = [c for c in items if c.category == category]
    if q:
        needle = q.lower()
        items = [c for c in items if needle in c.text.lower() or needle in c.provenance.reason.lower()]

    # newest first
    items.sort(key=lambda c: c.created_at, reverse=True)
    return items


@router.post("/memory/cards/{card_id}/approve", response_model=MemoryCard)
async def approve_card(card_id: str):
    card = MEMORY_CARDS.get(card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    if card.approval != MemoryApproval.PENDING:
        return card
    card.approval = MemoryApproval.APPROVED
    card.approved_at = datetime.utcnow()
    MEMORY_CARDS[card_id] = card
    await upsert_card(card)
    # Auto-sync approved card to knowledge graph
    try:
        await graph_queries.sync_memory_card(card)
    except Exception as e:
        log.warning("Graph sync failed for card %s: %s", card_id, e)
    # Auto-promote to Mind File
    try:
        if not await entry_exists_for_card(card.id):
            entry = MindFileEntry(
                category=MindFileEntryCategory(card.category.value),
                text=card.text,
                source_memory_card_id=card.id,
            )
            await save_mindfile_entry(entry)
            # Sync Mind File entry to Neo4j graph
            try:
                await graph_queries.sync_mindfile_entry(entry)
            except Exception as e:
                log.warning("Graph sync failed for mindfile entry: %s", e)
    except Exception as e:
        log.warning("Mind File promotion failed for card %s: %s", card_id, e)
    return card


@router.post("/memory/cards/{card_id}/reject", response_model=MemoryCard)
async def reject_card(card_id: str):
    card = MEMORY_CARDS.get(card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    if card.approval != MemoryApproval.PENDING:
        return card
    card.approval = MemoryApproval.REJECTED
    card.rejected_at = datetime.utcnow()
    MEMORY_CARDS[card.id] = card
    await upsert_card(card)
    return card


@router.post("/dev/seed")
async def dev_seed():
    seeds = [
        (
            "A system that cannot recognize values worth dying for cannot recognize values worth preserving.",
            MemoryCategory.PRINCIPLES_VALUES,
            "User-approved value anchor; informs ethical boundaries and system behavior.",
        ),
        (
            "Nothing is new in the human mind, but our tools are sharper than ever.",
            MemoryCategory.COGNITIVE_FRAMING,
            "User-approved framing principle; emphasizes tool power vs human novelty.",
        ),
    ]
    created = []
    skipped = []

    for text, cat, reason in seeds:
        # Check if a card with this exact text already exists
        existing = [c for c in MEMORY_CARDS.values() if c.text == text]
        if existing:
            skipped.append({"text": text[:50] + "...", "reason": "already exists"})
            continue

        card = MemoryCard(
            text=text,
            category=cat,
            scope=MemoryScope.PROJECT,
            provenance=MemoryProvenance(
                reason=reason,
                derived_from_artifact_ids=[],
                tools_used=["dev_seed"],
                model=None,
                sources=[],
            ),
        )
        MEMORY_CARDS[card.id] = card
        await upsert_card(card)
        created.append(card.id)

    return {"seeded": created, "count": len(created), "skipped": skipped}
