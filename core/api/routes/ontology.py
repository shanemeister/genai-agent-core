"""Ontology management API — SNOMED CT import, search, and browsing.

Endpoints:
    POST /ontology/snomed/import   — trigger SNOMED CT import (background)
    GET  /ontology/snomed/status   — import progress
    GET  /ontology/snomed/info     — ontology summary (counts)
    GET  /ontology/snomed/search   — full-text search for concepts
    GET  /ontology/snomed/concept/{sctid} — concept detail + relationships
    GET  /ontology/snomed/hierarchy/{sctid} — IS_A ancestors and children
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from core.config import settings
from core.graph.neo4j_client import get_session
from core.ontology.models import OntologyInfo, SnomedConceptDetail, SnomedImportStats
from core.rag.vector_store import PgVectorStore

log = logging.getLogger("noesis.ontology")

router = APIRouter(prefix="/ontology", tags=["ontology"])

# Module-level state for tracking the running import
_import_stats = SnomedImportStats()
_import_lock = asyncio.Lock()

_store = PgVectorStore()


# ── Import ───────────────────────────────────────────────────────────────

async def _run_import(rf2_dir: str) -> None:
    """Run the full SNOMED import pipeline as a background task."""
    global _import_stats
    try:
        from core.ontology.snomed_loader import load_snomed

        _import_stats = SnomedImportStats(status="running", phase="Starting import")
        stats = await load_snomed(rf2_dir=rf2_dir, stats=_import_stats)

        # Phase 5: Embedding
        stats.phase = "Embedding concepts into pgvector"
        log.info("Phase 5: %s", stats.phase)

        from core.ontology.snomed_embedder import embed_snomed_concepts

        embedded = await embed_snomed_concepts()
        stats.concepts_embedded = embedded
        stats.phase = "Complete"
        stats.status = "complete"

    except Exception as e:
        _import_stats.status = "error"
        _import_stats.error = str(e)
        log.error("SNOMED import background task failed: %s", e, exc_info=True)


@router.post("/snomed/import")
async def import_snomed(
    background_tasks: BackgroundTasks,
    rf2_dir: str | None = None,
):
    """Trigger SNOMED CT import from RF2 files.

    This runs as a background task. Use GET /ontology/snomed/status to
    monitor progress.
    """
    global _import_stats

    if _import_stats.status == "running":
        raise HTTPException(
            status_code=409,
            detail=f"Import already in progress: {_import_stats.phase}",
        )

    rf2_dir = rf2_dir or settings.snomed_rf2_dir
    _import_stats = SnomedImportStats(status="starting")
    background_tasks.add_task(_run_import, rf2_dir)
    return {"status": "started", "rf2_dir": rf2_dir}


@router.get("/snomed/status")
async def import_status() -> SnomedImportStats:
    """Get the current/last SNOMED CT import status."""
    return _import_stats


# ── Info ─────────────────────────────────────────────────────────────────

@router.get("/snomed/info")
async def snomed_info() -> OntologyInfo:
    """Get summary information about the loaded SNOMED CT ontology."""
    try:
        async with get_session() as session:
            # Ontology metadata
            result = await session.run(
                """
                MATCH (o:Ontology {name: "SNOMED CT"})
                RETURN o.version AS version, o.edition AS edition
                """
            )
            record = await result.single()
            if not record:
                raise HTTPException(status_code=404, detail="SNOMED CT not loaded")

            # Concept count
            result = await session.run(
                "MATCH (s:SnomedConcept) RETURN count(s) AS cnt"
            )
            concept_count = (await result.single())["cnt"]

            # Relationship count (exclude BELONGS_TO)
            result = await session.run(
                """
                MATCH (:SnomedConcept)-[r]->(:SnomedConcept)
                RETURN count(r) AS cnt
                """
            )
            rel_count = (await result.single())["cnt"]

        # Embedding count
        embed_count = await _store.count(source_type="ontology")

        return OntologyInfo(
            name="SNOMED CT",
            version=record["version"],
            edition=record["edition"],
            concept_count=concept_count,
            relationship_count=rel_count,
            embedded_count=embed_count,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to get SNOMED info: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Search ───────────────────────────────────────────────────────────────

@router.get("/snomed/search")
async def search_snomed(
    q: str = Query(..., min_length=2, description="Search term"),
    limit: int = Query(10, ge=1, le=100),
):
    """Search SNOMED CT concepts using the full-text index.

    Falls back to CONTAINS match if full-text index is not available.
    """
    try:
        async with get_session() as session:
            # Try full-text index first
            try:
                result = await session.run(
                    """
                    CALL db.index.fulltext.queryNodes(
                        'snomed_term_search', $query
                    ) YIELD node, score
                    RETURN node.sctid AS sctid,
                           node.fsn AS fsn,
                           node.preferred_term AS preferred_term,
                           node.semantic_tag AS semantic_tag,
                           score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    query=q,
                    limit=limit,
                )
                records = [record.data() async for record in result]
                if records:
                    return {"results": records, "method": "fulltext"}
            except Exception:
                pass

            # Fallback: case-insensitive CONTAINS
            result = await session.run(
                """
                MATCH (s:SnomedConcept)
                WHERE toLower(s.preferred_term) CONTAINS toLower($query)
                   OR toLower(s.fsn) CONTAINS toLower($query)
                RETURN s.sctid AS sctid,
                       s.fsn AS fsn,
                       s.preferred_term AS preferred_term,
                       s.semantic_tag AS semantic_tag,
                       1.0 AS score
                LIMIT $limit
                """,
                query=q,
                limit=limit,
            )
            records = [record.data() async for record in result]
            return {"results": records, "method": "contains"}

    except Exception as e:
        log.error("SNOMED search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Concept Detail ───────────────────────────────────────────────────────

@router.get("/snomed/concept/{sctid}")
async def get_concept(sctid: str) -> SnomedConceptDetail:
    """Get full detail for a SNOMED CT concept including relationships."""
    try:
        async with get_session() as session:
            # Core concept
            result = await session.run(
                """
                MATCH (s:SnomedConcept {sctid: $sctid})
                RETURN s.sctid AS sctid, s.fsn AS fsn,
                       s.preferred_term AS pt, s.semantic_tag AS tag,
                       s.definition_status AS def_status
                """,
                sctid=sctid,
            )
            record = await result.single()
            if not record:
                raise HTTPException(status_code=404, detail=f"Concept {sctid} not found")

            # IS_A parents
            result = await session.run(
                """
                MATCH (s:SnomedConcept {sctid: $sctid})-[:IS_A]->(parent:SnomedConcept)
                RETURN parent.sctid AS sctid, parent.preferred_term AS term,
                       parent.semantic_tag AS tag
                """,
                sctid=sctid,
            )
            parents = [r.data() async for r in result]

            # IS_A children
            result = await session.run(
                """
                MATCH (child:SnomedConcept)-[:IS_A]->(s:SnomedConcept {sctid: $sctid})
                RETURN child.sctid AS sctid, child.preferred_term AS term,
                       child.semantic_tag AS tag
                LIMIT 50
                """,
                sctid=sctid,
            )
            children = [r.data() async for r in result]

            # Non-IS_A relationships (outgoing)
            result = await session.run(
                """
                MATCH (s:SnomedConcept {sctid: $sctid})-[r]->(t:SnomedConcept)
                WHERE type(r) <> 'IS_A' AND type(r) <> 'BELONGS_TO'
                RETURN type(r) AS rel_type, t.sctid AS target_sctid,
                       t.preferred_term AS target_term,
                       r.rel_group AS rel_group
                """,
                sctid=sctid,
            )
            relationships = [r.data() async for r in result]

            # ICD-10 codes
            result = await session.run(
                """
                MATCH (s:SnomedConcept {sctid: $sctid})-[r:MAPS_TO]->(i:ICD10Code)
                RETURN i.code AS code, r.map_group AS map_group,
                       r.map_priority AS priority, r.map_advice AS advice
                ORDER BY r.map_group, r.map_priority
                """,
                sctid=sctid,
            )
            icd10_codes = [r.data() async for r in result]

        return SnomedConceptDetail(
            sctid=record["sctid"],
            fsn=record["fsn"] or "",
            preferred_term=record["pt"] or "",
            semantic_tag=record["tag"] or "",
            definition_status=record["def_status"] or "",
            parents=parents,
            children=children,
            relationships=relationships,
            icd10_codes=icd10_codes,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to get concept %s: %s", sctid, e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Hierarchy ────────────────────────────────────────────────────────────

@router.get("/snomed/hierarchy/{sctid}")
async def get_hierarchy(
    sctid: str,
    depth: int = Query(2, ge=1, le=5, description="Number of IS_A hops"),
):
    """Get IS_A ancestor and descendant tree for a concept."""
    safe_depth = max(1, min(depth, 5))
    try:
        async with get_session() as session:
            # Ancestors (concept -[:IS_A]-> parent chain)
            result = await session.run(
                f"""
                MATCH path = (s:SnomedConcept {{sctid: $sctid}})-[:IS_A*1..{safe_depth}]->(ancestor:SnomedConcept)
                RETURN ancestor.sctid AS sctid, ancestor.preferred_term AS term,
                       ancestor.semantic_tag AS tag, length(path) AS distance
                ORDER BY distance
                """,
                sctid=sctid,
            )
            ancestors = [r.data() async for r in result]

            # Descendants (child -[:IS_A]-> concept chain)
            result = await session.run(
                f"""
                MATCH path = (descendant:SnomedConcept)-[:IS_A*1..{safe_depth}]->(s:SnomedConcept {{sctid: $sctid}})
                RETURN descendant.sctid AS sctid, descendant.preferred_term AS term,
                       descendant.semantic_tag AS tag, length(path) AS distance
                ORDER BY distance
                LIMIT 200
                """,
                sctid=sctid,
            )
            descendants = [r.data() async for r in result]

        return {
            "sctid": sctid,
            "depth": safe_depth,
            "ancestors": ancestors,
            "descendants": descendants,
        }
    except Exception as e:
        log.error("Failed to get hierarchy for %s: %s", sctid, e)
        raise HTTPException(status_code=500, detail=str(e))
