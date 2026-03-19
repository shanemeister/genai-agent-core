"""Ontology management API — SNOMED CT and RxNorm import, search, and browsing.

Endpoints:
    POST /ontology/snomed/import   — trigger SNOMED CT import (background)
    GET  /ontology/snomed/status   — import progress
    GET  /ontology/snomed/info     — ontology summary (counts)
    GET  /ontology/snomed/search   — full-text search for concepts
    GET  /ontology/snomed/concept/{sctid} — concept detail + relationships
    GET  /ontology/snomed/hierarchy/{sctid} — IS_A ancestors and children
    POST /ontology/rxnorm/import   — trigger RxNorm import (background)
    GET  /ontology/rxnorm/status   — import progress
    GET  /ontology/rxnorm/info     — ontology summary (counts)
    GET  /ontology/rxnorm/search   — full-text search for drug concepts
    GET  /ontology/rxnorm/concept/{rxcui} — concept detail + SNOMED cross-refs
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from core.config import settings
from core.graph.neo4j_client import get_session
from core.ontology.models import (
    OntologyInfo,
    RxNormConceptDetail,
    RxNormImportStats,
    SnomedConceptDetail,
    SnomedImportStats,
)
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
                        'snomed_term_search', $search_term
                    ) YIELD node, score
                    RETURN node.sctid AS sctid,
                           node.fsn AS fsn,
                           node.preferred_term AS preferred_term,
                           node.semantic_tag AS semantic_tag,
                           score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    search_term=q,
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
                WHERE toLower(s.preferred_term) CONTAINS toLower($search_term)
                   OR toLower(s.fsn) CONTAINS toLower($search_term)
                RETURN s.sctid AS sctid,
                       s.fsn AS fsn,
                       s.preferred_term AS preferred_term,
                       s.semantic_tag AS semantic_tag,
                       1.0 AS score
                LIMIT $limit
                """,
                search_term=q,
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


# ══════════════════════════════════════════════════════════════════════════
# RxNorm Endpoints
# ══════════════════════════════════════════════════════════════════════════

_rxnorm_import_stats = RxNormImportStats()


async def _run_rxnorm_import(rrf_dir: str) -> None:
    """Run the full RxNorm import pipeline as a background task."""
    global _rxnorm_import_stats
    try:
        from core.ontology.rxnorm_loader import load_rxnorm

        _rxnorm_import_stats = RxNormImportStats(status="running", phase="Starting import")
        stats = await load_rxnorm(rrf_dir=rrf_dir, stats=_rxnorm_import_stats)

        # Phase 5: Embedding — reset status since loader marks it "complete"
        stats.status = "running"
        stats.phase = "Embedding concepts into pgvector"
        log.info("RxNorm Phase 5: %s", stats.phase)

        from core.ontology.rxnorm_embedder import embed_rxnorm_concepts

        embedded = await embed_rxnorm_concepts()
        stats.concepts_embedded = embedded
        stats.phase = "Complete"
        stats.status = "complete"

    except Exception as e:
        _rxnorm_import_stats.status = "error"
        _rxnorm_import_stats.error = str(e)
        log.error("RxNorm import background task failed: %s", e, exc_info=True)


@router.post("/rxnorm/import")
async def import_rxnorm(
    background_tasks: BackgroundTasks,
    rrf_dir: str | None = None,
):
    """Trigger RxNorm import from RRF files.

    This runs as a background task. Use GET /ontology/rxnorm/status to
    monitor progress.
    """
    global _rxnorm_import_stats

    if _rxnorm_import_stats.status == "running":
        raise HTTPException(
            status_code=409,
            detail=f"Import already in progress: {_rxnorm_import_stats.phase}",
        )

    rrf_dir = rrf_dir or settings.rxnorm_rrf_dir
    _rxnorm_import_stats = RxNormImportStats(status="starting")
    background_tasks.add_task(_run_rxnorm_import, rrf_dir)
    return {"status": "started", "rrf_dir": rrf_dir}


@router.get("/rxnorm/status")
async def rxnorm_import_status() -> RxNormImportStats:
    """Get the current/last RxNorm import status."""
    return _rxnorm_import_stats


@router.get("/rxnorm/info")
async def rxnorm_info() -> OntologyInfo:
    """Get summary information about the loaded RxNorm ontology."""
    try:
        async with get_session() as session:
            result = await session.run(
                """
                MATCH (o:Ontology {name: "RxNorm"})
                RETURN o.version AS version
                """
            )
            record = await result.single()
            if not record:
                raise HTTPException(status_code=404, detail="RxNorm not loaded")

            result = await session.run(
                "MATCH (r:RxNormConcept) RETURN count(r) AS cnt"
            )
            concept_count = (await result.single())["cnt"]

            result = await session.run(
                """
                MATCH (:RxNormConcept)-[r]->(:RxNormConcept)
                RETURN count(r) AS cnt
                """
            )
            rel_count = (await result.single())["cnt"]

        # Get embedded count from pgvector
        from core.db.postgres import get_pool

        pool = await get_pool()
        row = await pool.fetchrow(
            "SELECT COUNT(*) AS cnt FROM embeddings WHERE doc_id LIKE 'ontology:rxnorm:%'"
        )
        embed_count = row["cnt"] if row else 0

        return OntologyInfo(
            name="RxNorm",
            version=record["version"],
            edition="",
            concept_count=concept_count,
            relationship_count=rel_count,
            embedded_count=embed_count,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to get RxNorm info: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rxnorm/search")
async def search_rxnorm(
    q: str = Query(..., min_length=2, description="Search term"),
    limit: int = Query(10, ge=1, le=100),
):
    """Search RxNorm concepts using the full-text index."""
    try:
        async with get_session() as session:
            try:
                result = await session.run(
                    """
                    CALL db.index.fulltext.queryNodes(
                        'rxnorm_term_search', $search_term
                    ) YIELD node, score
                    RETURN node.rxcui AS rxcui,
                           node.preferred_term AS preferred_term,
                           node.tty AS tty,
                           score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    search_term=q,
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
                MATCH (r:RxNormConcept)
                WHERE toLower(r.preferred_term) CONTAINS toLower($search_term)
                RETURN r.rxcui AS rxcui,
                       r.preferred_term AS preferred_term,
                       r.tty AS tty,
                       1.0 AS score
                LIMIT $limit
                """,
                search_term=q,
                limit=limit,
            )
            records = [record.data() async for record in result]
            return {"results": records, "method": "contains"}

    except Exception as e:
        log.error("RxNorm search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rxnorm/concept/{rxcui}")
async def get_rxnorm_concept(rxcui: str) -> RxNormConceptDetail:
    """Get full detail for an RxNorm concept including SNOMED cross-references."""
    try:
        async with get_session() as session:
            result = await session.run(
                """
                MATCH (r:RxNormConcept {rxcui: $rxcui})
                RETURN r.rxcui AS rxcui, r.preferred_term AS pt,
                       r.tty AS tty, r.synonyms AS synonyms
                """,
                rxcui=rxcui,
            )
            record = await result.single()
            if not record:
                raise HTTPException(status_code=404, detail=f"RxNorm concept {rxcui} not found")

            # RxNorm relationships (outgoing)
            result = await session.run(
                """
                MATCH (r:RxNormConcept {rxcui: $rxcui})-[rel]->(t:RxNormConcept)
                WHERE type(rel) <> 'BELONGS_TO'
                RETURN type(rel) AS rel_type, t.rxcui AS target_rxcui,
                       t.preferred_term AS target_term, t.tty AS target_tty
                LIMIT 50
                """,
                rxcui=rxcui,
            )
            relationships = [r.data() async for r in result]

            # SNOMED cross-references
            result = await session.run(
                """
                MATCH (r:RxNormConcept {rxcui: $rxcui})-[:MAPS_TO_SNOMED]->(s:SnomedConcept)
                RETURN s.sctid AS sctid, s.preferred_term AS term,
                       s.semantic_tag AS semantic_tag
                """,
                rxcui=rxcui,
            )
            snomed_xrefs = [r.data() async for r in result]

        return RxNormConceptDetail(
            rxcui=record["rxcui"],
            preferred_term=record["pt"] or "",
            tty=record["tty"] or "",
            synonyms=record["synonyms"] or [],
            relationships=relationships,
            snomed_crossrefs=snomed_xrefs,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to get RxNorm concept %s: %s", rxcui, e)
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════
# Specificity Rules Endpoints
# ══════════════════════════════════════════════════════════════════════════

@router.get("/rules")
async def list_rules():
    """List all loaded specificity rules."""
    from core.validation.specificity_rules import get_rules
    rules = get_rules()
    return {
        "total": len(rules),
        "rules": [
            {
                "rule_id": r.rule_id,
                "condition": r.condition,
                "priority": r.priority,
                "layer": r.layer,
                "attributes": [a.name for a in r.attributes],
                "icd10_unspecified": r.icd10_unspecified,
            }
            for r in rules
        ],
    }


@router.get("/rules/{rule_id}")
async def get_rule(rule_id: str):
    """Get a single specificity rule by ID."""
    from core.validation.specificity_rules import get_rule as _get_rule
    rule = _get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    return rule.model_dump()


@router.post("/rules")
async def create_rule(rule_data: dict):
    """Add a hospital-specific specificity rule (Layer 3).

    The rule will be saved to the hospital rules directory and loaded
    into the active rule set.
    """
    import json as json_mod
    from pathlib import Path

    from core.validation.specificity_rules import SpecificityRule, load_rules

    try:
        rule = SpecificityRule(**rule_data)
        rule.layer = 3  # Force Layer 3 for API-created rules
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid rule data: {e}")

    # Save to hospital rules directory
    hospital_dir = Path(__file__).parent.parent.parent / "validation" / "rules" / "hospital"
    hospital_dir.mkdir(parents=True, exist_ok=True)

    rule_file = hospital_dir / f"{rule.rule_id.lower()}.json"
    rule_file.write_text(json_mod.dumps([rule.model_dump()], indent=2))

    # Reload all rules
    load_rules(hospital_dir=hospital_dir)

    return {"status": "created", "rule_id": rule.rule_id}


@router.put("/rules/{rule_id}")
async def update_rule(rule_id: str, rule_data: dict):
    """Update a hospital-specific specificity rule (Layer 3 only)."""
    import json as json_mod
    from pathlib import Path

    from core.validation.specificity_rules import SpecificityRule, get_rule as _get_rule, load_rules

    existing = _get_rule(rule_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    if existing.layer != 3:
        raise HTTPException(status_code=403, detail="Only Layer 3 (hospital) rules can be modified via API")

    try:
        rule_data["rule_id"] = rule_id
        rule = SpecificityRule(**rule_data)
        rule.layer = 3
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid rule data: {e}")

    hospital_dir = Path(__file__).parent.parent.parent / "validation" / "rules" / "hospital"
    rule_file = hospital_dir / f"{rule_id.lower()}.json"
    rule_file.write_text(json_mod.dumps([rule.model_dump()], indent=2))

    load_rules(hospital_dir=hospital_dir)

    return {"status": "updated", "rule_id": rule_id}


@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """Delete a hospital-specific specificity rule (Layer 3 only)."""
    from pathlib import Path

    from core.validation.specificity_rules import get_rule as _get_rule, load_rules

    existing = _get_rule(rule_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    if existing.layer != 3:
        raise HTTPException(status_code=403, detail="Only Layer 3 (hospital) rules can be deleted via API")

    hospital_dir = Path(__file__).parent.parent.parent / "validation" / "rules" / "hospital"
    rule_file = hospital_dir / f"{rule_id.lower()}.json"
    if rule_file.exists():
        rule_file.unlink()

    load_rules(hospital_dir=hospital_dir)

    return {"status": "deleted", "rule_id": rule_id}
