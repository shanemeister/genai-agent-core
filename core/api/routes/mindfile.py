"""Mind File endpoints: CRUD, stats, export, backfill, cognitive profile, timeline, patterns."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.artifacts.memory_card import MemoryApproval
from core.artifacts.mindfile_entry import MindFileEntry, MindFileEntryCategory
from core.artifacts.storage_mindfile_pg import (
    save_entry as save_mindfile_entry,
    load_all_entries as load_mindfile_entries,
    load_entry as load_mindfile_entry,
    update_note as update_mindfile_note,
    delete_entry as delete_mindfile_entry,
    entry_exists_for_card,
)
from core.graph import queries as graph_queries
from core.api.shared import MEMORY_CARDS

log = logging.getLogger("noesis.mindfile")

router = APIRouter(prefix="/mindfile", tags=["mindfile"])


@router.get("")
async def list_mindfile(
    category: Optional[str] = Query(default=None),
):
    """List all Mind File entries, optionally filtered by category."""
    cat = MindFileEntryCategory(category) if category else None
    entries = await load_mindfile_entries(category=cat)
    return [e.model_dump(mode="json") for e in entries]


@router.get("/stats")
async def mindfile_stats():
    """Return Mind File statistics."""
    entries = await load_mindfile_entries()
    by_category: dict[str, int] = {}
    for e in entries:
        by_category[e.category.value] = by_category.get(e.category.value, 0) + 1
    first_date = min((e.created_at for e in entries), default=None)
    latest_date = max((e.created_at for e in entries), default=None)
    return {
        "total": len(entries),
        "by_category": by_category,
        "first_entry_date": first_date.isoformat() if first_date else None,
        "latest_entry_date": latest_date.isoformat() if latest_date else None,
    }


@router.get("/export")
async def export_mindfile():
    """Export the full Mind File as structured markdown."""
    entries = await load_mindfile_entries()
    lines: list[str] = []
    lines.append("# Mind File")
    lines.append("")
    lines.append(f"**Total entries:** {len(entries)}")
    lines.append("")

    # Group by category
    grouped: dict[str, list[MindFileEntry]] = {}
    for e in entries:
        grouped.setdefault(e.category.value, []).append(e)

    category_labels = {
        "principles_values": "Principles & Values",
        "cognitive_framing": "Cognitive Framing",
        "decision_heuristics": "Decision Heuristics",
        "preferences": "Preferences",
        "vocabulary": "Vocabulary",
    }

    for cat_value, label in category_labels.items():
        cat_entries = grouped.get(cat_value, [])
        if not cat_entries:
            continue
        lines.append(f"## {label}")
        lines.append("")
        for e in cat_entries:
            lines.append(f"- {e.text}")
            if e.note:
                lines.append(f"  - *Note: {e.note}*")
            lines.append(f"  - Added: {e.created_at.strftime('%Y-%m-%d')}")
            lines.append("")
        lines.append("")

    lines.append(f"*Exported from Noesis on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")

    markdown = "\n".join(lines)
    return {
        "markdown": markdown,
        "filename": "mind_file.md",
        "entry_count": len(entries),
    }


@router.post("/backfill")
async def backfill_mindfile():
    """One-time backfill: create Mind File entries for all approved cards that don't have one."""
    approved = [c for c in MEMORY_CARDS.values() if c.approval == MemoryApproval.APPROVED]
    created = 0
    skipped = 0
    for card in approved:
        if await entry_exists_for_card(card.id):
            skipped += 1
            continue
        entry = MindFileEntry(
            category=MindFileEntryCategory(card.category.value),
            text=card.text,
            source_memory_card_id=card.id,
        )
        await save_mindfile_entry(entry)
        try:
            await graph_queries.sync_mindfile_entry(entry)
        except Exception as e:
            log.warning("Graph sync failed for mindfile entry %s: %s", entry.id, e)
        created += 1
    return {"created": created, "skipped": skipped, "total_approved": len(approved)}


@router.post("/sync-graph")
async def sync_mindfile_to_graph():
    """Sync all existing Mind File entries to Neo4j graph."""
    entries = await load_mindfile_entries()
    synced = 0
    errors = 0
    for entry in entries:
        try:
            await graph_queries.sync_mindfile_entry(entry)
            synced += 1
        except Exception as e:
            log.warning("Graph sync failed for mindfile entry %s: %s", entry.id, e)
            errors += 1
    return {"synced": synced, "errors": errors, "total": len(entries)}


# ---------------------------------------------------------------------------
# Room 1: Timeline, Patterns, Cognitive Profile
# (These MUST be declared before the /{entry_id} wildcard route)
# ---------------------------------------------------------------------------

@router.get("/timeline/{concept}")
async def concept_timeline(concept: str):
    """Return a chronological timeline of all artifacts related to a concept."""
    events = await graph_queries.get_concept_timeline(concept)
    return events


@router.get("/patterns")
async def mindfile_patterns():
    """Return concept co-occurrence patterns and top concepts."""
    top_concepts = await graph_queries.get_top_concepts(limit=20)
    co_occurrences = await graph_queries.get_concept_cooccurrences(limit=20)
    category_trend = await graph_queries.get_category_trend()
    return {
        "top_concepts": top_concepts,
        "co_occurrences": co_occurrences,
        "category_trend": category_trend,
    }


@router.get("/cognitive-profile")
async def cognitive_profile():
    """Return an aggregate cognitive style summary."""
    entries = await load_mindfile_entries()
    category_breakdown: dict[str, int] = {}
    for e in entries:
        category_breakdown[e.category.value] = category_breakdown.get(e.category.value, 0) + 1

    top_concepts = await graph_queries.get_top_concepts(limit=10)
    co_occurrences = await graph_queries.get_concept_cooccurrences(limit=10)
    category_trend = await graph_queries.get_category_trend()

    first_date = min((e.created_at for e in entries), default=None)
    latest_date = max((e.created_at for e in entries), default=None)

    return {
        "total_entries": len(entries),
        "category_breakdown": category_breakdown,
        "top_concepts": top_concepts,
        "co_occurrences": co_occurrences,
        "category_trend": category_trend,
        "first_entry_date": first_date.isoformat() if first_date else None,
        "latest_entry_date": latest_date.isoformat() if latest_date else None,
    }


# Wildcard entry routes — MUST come after all specific routes

class UpdateNoteRequest(BaseModel):
    note: str


@router.get("/{entry_id}")
async def get_mindfile_entry(entry_id: str):
    """Get a single Mind File entry."""
    entry = await load_mindfile_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry.model_dump(mode="json")


@router.put("/{entry_id}/note")
async def update_mindfile_entry_note(entry_id: str, req: UpdateNoteRequest):
    """Update the user annotation on a Mind File entry."""
    entry = await update_mindfile_note(entry_id, req.note)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry.model_dump(mode="json")


@router.delete("/{entry_id}")
async def remove_mindfile_entry(entry_id: str):
    """Remove an entry from the Mind File. The source memory card stays approved."""
    deleted = await delete_mindfile_entry(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "deleted", "entry_id": entry_id}
