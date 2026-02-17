"""Knowledge graph endpoints: data, search, concepts, relationships, sync."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Query

from core.artifacts.memory_card import MemoryApproval
from core.graph.models import (
    ConceptCreate,
    DiagramToGraphRequest,
    GraphData,
    GraphEdge,
    GraphNode,
    GraphToDiagramRequest,
    NeighborRequest,
    RelationshipCreate,
)
from core.graph import queries as graph_queries
from core.api.shared import MEMORY_CARDS

log = logging.getLogger("noesis.graph")

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/data", response_model=GraphData)
async def graph_data(limit: int = Query(default=100, ge=1, le=500)):
    """Return the full graph up to a limit."""
    return await graph_queries.get_full_graph(limit=limit)


@router.post("/neighbors", response_model=GraphData)
async def graph_neighbors(req: NeighborRequest):
    """Expand neighborhood around a node."""
    return await graph_queries.get_neighbors(
        node_id=req.node_id, depth=req.depth, limit=req.limit
    )


@router.get("/search", response_model=GraphData)
async def graph_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Search nodes by name/text."""
    return await graph_queries.search_nodes(query=q, limit=limit)


@router.post("/concepts", response_model=GraphNode)
async def graph_create_concept(req: ConceptCreate):
    """Create or merge a Concept node."""
    return await graph_queries.create_concept(
        name=req.name, description=req.description, source=req.source
    )


@router.post("/relationships", response_model=GraphEdge)
async def graph_create_relationship(req: RelationshipCreate):
    """Create a relationship between two nodes."""
    return await graph_queries.create_relationship(
        source_id=req.source_id,
        target_id=req.target_id,
        rel_type=req.rel_type,
        strength=req.strength,
    )


@router.get("/stats")
async def graph_stats():
    """Return node and edge counts."""
    return await graph_queries.get_stats()


@router.post("/sync-memories")
async def graph_sync_memories():
    """Bulk sync all approved memory cards to the knowledge graph."""
    approved = [c for c in MEMORY_CARDS.values() if c.approval == MemoryApproval.APPROVED]
    synced = 0
    errors = 0
    for card in approved:
        try:
            await graph_queries.sync_memory_card(card)
            synced += 1
        except Exception as e:
            log.warning("Graph sync failed for card %s: %s", card.id, e)
            errors += 1
    return {"synced": synced, "errors": errors, "total_approved": len(approved)}


@router.post("/seed")
async def graph_seed():
    """Seed the knowledge graph with demo data."""
    return await graph_queries.seed_demo_data()


@router.get("/session/{session_id}", response_model=GraphData)
async def graph_session_subgraph(session_id: str):
    """Return a focused subgraph around a chat session (1-2 hops)."""
    return await graph_queries.get_session_subgraph(session_id=session_id)


@router.get("/scoped", response_model=GraphData)
async def graph_scoped(
    scope: str = Query(default="session", regex="^(session|question|artifact)$"),
    session_id: Optional[str] = Query(default=None),
    node_id: Optional[str] = Query(default=None),
    depth: int = Query(default=1, ge=1, le=3),
    view_mode: str = Query(default="provenance", regex="^(provenance|lineage|full)$"),
):
    """Return a scoped, predictable subgraph."""
    return await graph_queries.get_scoped_graph(
        scope=scope,
        session_id=session_id,
        node_id=node_id,
        depth=depth,
        view_mode=view_mode,
    )


@router.get("/lineage/{node_id:path}", response_model=GraphData)
async def graph_lineage(node_id: str):
    """Get artifact lineage chain for a node."""
    return await graph_queries.get_artifact_lineage(node_id=node_id)


@router.post("/from-diagram", response_model=GraphData)
async def graph_from_diagram(req: DiagramToGraphRequest):
    """Import a Mermaid diagram into the knowledge graph as Concept nodes + edges."""
    return await graph_queries.import_diagram_to_graph(
        diagram_code=req.diagram_code,
        source=req.source or "diagram_import",
    )


@router.post("/to-diagram")
async def graph_to_diagram(req: GraphToDiagramRequest):
    """Export graph nodes/edges as Mermaid flowchart syntax."""
    ids = req.node_ids if req.node_ids is not None else None
    code = await graph_queries.export_graph_to_mermaid(
        node_ids=ids,
        depth=req.depth,
        layout=req.layout,
    )
    return {"code": code}
