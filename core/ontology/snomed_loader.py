"""SNOMED CT → Neo4j batch loader.

Parses RF2 files and bulk-loads concepts, relationships, and ICD-10 mappings
into the Neo4j `noesis` database using UNWIND batch MERGE patterns.

Usage:
    from core.ontology.snomed_loader import load_snomed
    stats = await load_snomed("/path/to/Snapshot")
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from core.config import settings
from core.graph.neo4j_client import get_session
from core.ontology.models import SnomedImportStats
from core.ontology.rf2_parser import (
    build_concept_lookup,
    parse_concepts,
    parse_icd10_map,
    parse_relationships,
)

log = logging.getLogger("noesis.ontology")


# ── Batch helpers ────────────────────────────────────────────────────────

def _batches(iterable, size: int):
    """Yield successive batches from an iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ── Cypher batch operations ──────────────────────────────────────────────

async def _create_ontology_node(session) -> None:
    """Create or update the :Ontology metadata node for SNOMED CT."""
    await session.run(
        """
        MERGE (o:Ontology {name: "SNOMED CT"})
        SET o.version = $version,
            o.edition = $edition,
            o.updated_at = datetime()
        """,
        version="US 20250901",
        edition="US1000124",
    )
    log.info("Created/updated :Ontology metadata node for SNOMED CT")


async def _batch_merge_concepts(session, batch: list[dict]) -> int:
    """UNWIND batch of concept dicts into :SnomedConcept nodes."""
    result = await session.run(
        """
        UNWIND $batch AS row
        MERGE (c:SnomedConcept {sctid: row.sctid})
        SET c.fsn = row.fsn,
            c.preferred_term = row.pt,
            c.semantic_tag = row.tag,
            c.definition_status = row.def_status,
            c.module_id = row.module_id,
            c.updated_at = datetime()
        RETURN count(c) AS cnt
        """,
        batch=batch,
    )
    record = await result.single()
    return record["cnt"] if record else 0


async def _batch_merge_relationships(
    session, rel_type: str, batch: list[dict]
) -> int:
    """UNWIND batch of relationships of a single type.

    Neo4j does not support dynamic relationship types in MERGE,
    so we run one query per relationship type with the type interpolated.
    The type name is validated against the known RELATIONSHIP_TYPE_NAMES dict.
    """
    query = f"""
        UNWIND $batch AS row
        MATCH (s:SnomedConcept {{sctid: row.source}})
        MATCH (t:SnomedConcept {{sctid: row.target}})
        MERGE (s)-[r:{rel_type}]->(t)
        SET r.rel_group = row.rel_group,
            r.rel_id = row.rel_id
        RETURN count(r) AS cnt
    """
    result = await session.run(query, batch=batch)
    record = await result.single()
    return record["cnt"] if record else 0


async def _batch_merge_icd10(session, batch: list[dict]) -> int:
    """UNWIND batch of ICD-10 mappings into :ICD10Code nodes + :MAPS_TO edges."""
    result = await session.run(
        """
        UNWIND $batch AS row
        MERGE (i:ICD10Code {code: row.code})
        SET i.updated_at = datetime()
        WITH i, row
        MATCH (s:SnomedConcept {sctid: row.snomed_id})
        MERGE (s)-[r:MAPS_TO]->(i)
        SET r.map_group = row.map_group,
            r.map_priority = row.map_priority,
            r.map_rule = row.map_rule,
            r.map_advice = row.map_advice
        RETURN count(r) AS cnt
        """,
        batch=batch,
    )
    record = await result.single()
    return record["cnt"] if record else 0


async def _batch_link_to_ontology(session, batch: list[str]) -> int:
    """Link a batch of SnomedConcept sctids to the :Ontology node via :BELONGS_TO."""
    result = await session.run(
        """
        MATCH (o:Ontology {name: "SNOMED CT"})
        UNWIND $sctids AS sctid
        MATCH (c:SnomedConcept {sctid: sctid})
        MERGE (c)-[:BELONGS_TO]->(o)
        RETURN count(*) AS cnt
        """,
        sctids=batch,
    )
    record = await result.single()
    return record["cnt"] if record else 0


# ── Main import orchestrator ─────────────────────────────────────────────

async def load_snomed(
    rf2_dir: str | None = None,
    batch_size: int | None = None,
    stats: SnomedImportStats | None = None,
) -> SnomedImportStats:
    """Import SNOMED CT from RF2 files into Neo4j.

    Args:
        rf2_dir: Path to the Snapshot directory. Defaults to settings.snomed_rf2_dir.
        batch_size: Nodes per UNWIND batch. Defaults to settings.ontology_import_batch_size.
        stats: Optional stats object for progress tracking (mutated in-place).

    Returns:
        SnomedImportStats with final counts and timing.
    """
    rf2_dir = rf2_dir or settings.snomed_rf2_dir
    batch_size = batch_size or settings.ontology_import_batch_size

    if stats is None:
        stats = SnomedImportStats()

    stats.status = "running"
    t0 = time.monotonic()

    try:
        # ── Phase 0: Build concept lookup from descriptions ──────
        stats.phase = "Building concept lookup from descriptions"
        log.info("Phase 0: %s", stats.phase)
        concept_lookup = build_concept_lookup(rf2_dir)

        # ── Phase 1: Load concepts ───────────────────────────────
        stats.phase = "Loading concepts into Neo4j"
        log.info("Phase 1: %s", stats.phase)

        async with get_session() as session:
            await _create_ontology_node(session)

        concept_count = 0
        for batch_num, batch in enumerate(
            _batches(parse_concepts(rf2_dir), batch_size)
        ):
            # Enrich each concept with description data
            enriched = []
            for concept in batch:
                sctid = concept["sctid"]
                desc = concept_lookup.get(sctid, {})
                enriched.append({
                    "sctid": sctid,
                    "fsn": desc.get("fsn", ""),
                    "pt": desc.get("preferred_term", ""),
                    "tag": desc.get("semantic_tag", ""),
                    "def_status": concept["definition_status"],
                    "module_id": concept["module_id"],
                })

            async with get_session() as session:
                n = await _batch_merge_concepts(session, enriched)
                concept_count += n

            if (batch_num + 1) % 50 == 0:
                log.info(
                    "  Concepts: %d loaded (%d batches)",
                    concept_count,
                    batch_num + 1,
                )

        stats.concepts_loaded = concept_count
        log.info("Phase 1 complete: %d concepts loaded", concept_count)

        # ── Phase 2: Load relationships (grouped by type) ────────
        stats.phase = "Loading relationships into Neo4j"
        log.info("Phase 2: %s", stats.phase)

        # Group relationships by type for efficient per-type UNWIND
        rel_by_type: dict[str, list[dict]] = defaultdict(list)
        for rel in parse_relationships(rf2_dir):
            rel_by_type[rel["type_name"]].append({
                "rel_id": rel["rel_id"],
                "source": rel["source_id"],
                "target": rel["dest_id"],
                "rel_group": rel["rel_group"],
            })

        rel_count = 0
        for rel_type, rels in rel_by_type.items():
            type_count = 0
            for batch_num, batch in enumerate(_batches(rels, batch_size)):
                async with get_session() as session:
                    n = await _batch_merge_relationships(session, rel_type, batch)
                    type_count += n
                    rel_count += n

            log.info("  %s: %d relationships", rel_type, type_count)

        stats.relationships_loaded = rel_count
        log.info("Phase 2 complete: %d relationships loaded", rel_count)

        # ── Phase 3: ICD-10 crosswalk ────────────────────────────
        stats.phase = "Loading ICD-10 crosswalk"
        log.info("Phase 3: %s", stats.phase)

        icd10_count = 0
        icd10_batch = []
        for mapping in parse_icd10_map(rf2_dir):
            icd10_batch.append({
                "snomed_id": mapping["snomed_id"],
                "code": mapping["icd10_code"],
                "map_group": mapping["map_group"],
                "map_priority": mapping["map_priority"],
                "map_rule": mapping["map_rule"],
                "map_advice": mapping["map_advice"],
            })
            if len(icd10_batch) >= batch_size:
                async with get_session() as session:
                    n = await _batch_merge_icd10(session, icd10_batch)
                    icd10_count += n
                icd10_batch = []

        if icd10_batch:
            async with get_session() as session:
                n = await _batch_merge_icd10(session, icd10_batch)
                icd10_count += n

        stats.icd10_mappings_loaded = icd10_count
        log.info("Phase 3 complete: %d ICD-10 mappings loaded", icd10_count)

        # ── Phase 4: BELONGS_TO links ────────────────────────────
        stats.phase = "Linking concepts to Ontology node"
        log.info("Phase 4: %s", stats.phase)

        all_sctids = list(concept_lookup.keys())
        link_count = 0
        for batch in _batches(all_sctids, batch_size * 2):
            async with get_session() as session:
                n = await _batch_link_to_ontology(session, batch)
                link_count += n

        log.info("Phase 4 complete: %d BELONGS_TO links", link_count)

        # ── Done ─────────────────────────────────────────────────
        stats.status = "complete"
        stats.phase = "Import complete"
        stats.elapsed_seconds = round(time.monotonic() - t0, 1)
        log.info(
            "SNOMED CT import complete in %.1fs — %d concepts, %d relationships, %d ICD-10 maps",
            stats.elapsed_seconds,
            stats.concepts_loaded,
            stats.relationships_loaded,
            stats.icd10_mappings_loaded,
        )

    except Exception as e:
        stats.status = "error"
        stats.error = str(e)
        stats.elapsed_seconds = round(time.monotonic() - t0, 1)
        log.error("SNOMED CT import failed: %s", e, exc_info=True)
        raise

    return stats
