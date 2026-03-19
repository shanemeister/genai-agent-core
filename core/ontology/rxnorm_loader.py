"""RxNorm → Neo4j batch loader.

Parses RRF files and bulk-loads concepts, relationships, and SNOMED
crosswalk into the Neo4j `noesis` database using UNWIND batch MERGE.

Usage:
    from core.ontology.rxnorm_loader import load_rxnorm
    stats = await load_rxnorm("/path/to/rrf")
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from core.config import settings
from core.graph.neo4j_client import get_session
from core.ontology.models import RxNormImportStats
from core.ontology.rxnorm_parser import (
    build_rxnorm_lookup,
    parse_rxn_concepts,
    parse_rxn_relationships,
    parse_snomed_crosswalk,
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
    """Create or update the :Ontology metadata node for RxNorm."""
    await session.run(
        """
        MERGE (o:Ontology {name: "RxNorm"})
        SET o.version = $version,
            o.updated_at = datetime()
        """,
        version="Monthly 2025",
    )
    log.info("Created/updated :Ontology metadata node for RxNorm")


async def _batch_merge_concepts(session, batch: list[dict]) -> int:
    """UNWIND batch of concept dicts into :RxNormConcept nodes."""
    result = await session.run(
        """
        UNWIND $batch AS row
        MERGE (c:RxNormConcept {rxcui: row.rxcui})
        SET c.preferred_term = row.pt,
            c.tty = row.tty,
            c.synonyms = row.synonyms,
            c.updated_at = datetime()
        RETURN count(c) AS cnt
        """,
        batch=batch,
    )
    record = await result.single()
    return record["cnt"] if record else 0


async def _batch_merge_relationships(
    session, rela_type: str, batch: list[dict]
) -> int:
    """UNWIND batch of relationships of a single RELA type.

    Neo4j doesn't support dynamic relationship types in MERGE,
    so we run one query per RELA type with the type interpolated.
    RELA values are validated against RXNORM_RELA_TYPES in the parser.
    """
    # Convert RELA to Neo4j-safe relationship name (uppercase, no spaces)
    rel_name = rela_type.upper()

    query = f"""
        UNWIND $batch AS row
        MATCH (s:RxNormConcept {{rxcui: row.rxcui1}})
        MATCH (t:RxNormConcept {{rxcui: row.rxcui2}})
        MERGE (s)-[r:{rel_name}]->(t)
        RETURN count(r) AS cnt
    """
    result = await session.run(query, batch=batch)
    record = await result.single()
    return record["cnt"] if record else 0


async def _batch_merge_snomed_crosswalk(session, batch: list[dict]) -> int:
    """UNWIND batch of SNOMED crosswalk entries into MAPS_TO_SNOMED edges."""
    result = await session.run(
        """
        UNWIND $batch AS row
        MATCH (r:RxNormConcept {rxcui: row.rxcui})
        MATCH (s:SnomedConcept {sctid: row.snomed_sctid})
        MERGE (r)-[m:MAPS_TO_SNOMED]->(s)
        RETURN count(m) AS cnt
        """,
        batch=batch,
    )
    record = await result.single()
    return record["cnt"] if record else 0


async def _batch_link_to_ontology(session, batch: list[str]) -> int:
    """Link a batch of RxNormConcept rxcuis to the :Ontology node."""
    result = await session.run(
        """
        MATCH (o:Ontology {name: "RxNorm"})
        UNWIND $rxcuis AS rxcui
        MATCH (c:RxNormConcept {rxcui: rxcui})
        MERGE (c)-[:BELONGS_TO]->(o)
        RETURN count(*) AS cnt
        """,
        rxcuis=batch,
    )
    record = await result.single()
    return record["cnt"] if record else 0


# ── Main import orchestrator ─────────────────────────────────────────────

async def load_rxnorm(
    rrf_dir: str | None = None,
    batch_size: int | None = None,
    stats: RxNormImportStats | None = None,
) -> RxNormImportStats:
    """Import RxNorm from RRF files into Neo4j.

    Args:
        rrf_dir: Path to the RRF directory. Defaults to settings.rxnorm_rrf_dir.
        batch_size: Nodes per UNWIND batch. Defaults to settings.ontology_import_batch_size.
        stats: Optional stats object for progress tracking (mutated in-place).

    Returns:
        RxNormImportStats with final counts and timing.
    """
    rrf_dir = rrf_dir or settings.rxnorm_rrf_dir
    batch_size = batch_size or settings.ontology_import_batch_size

    if stats is None:
        stats = RxNormImportStats()

    stats.status = "running"
    t0 = time.monotonic()

    try:
        # ── Phase 0: Build concept lookup ───────────────────────
        stats.phase = "Building concept lookup from RXNCONSO"
        log.info("Phase 0: %s", stats.phase)
        concept_lookup = build_rxnorm_lookup(rrf_dir)

        # ── Phase 1: Load concepts ──────────────────────────────
        stats.phase = "Loading concepts into Neo4j"
        log.info("Phase 1: %s", stats.phase)

        async with get_session() as session:
            await _create_ontology_node(session)

        concept_count = 0
        all_rxcuis = list(concept_lookup.keys())

        for batch_num, batch_rxcuis in enumerate(
            _batches(all_rxcuis, batch_size)
        ):
            enriched = []
            for rxcui in batch_rxcuis:
                info = concept_lookup[rxcui]
                enriched.append({
                    "rxcui": rxcui,
                    "pt": info["preferred_term"],
                    "tty": info["tty"],
                    "synonyms": info["synonyms"],
                })

            async with get_session() as session:
                n = await _batch_merge_concepts(session, enriched)
                concept_count += n

            if (batch_num + 1) % 50 == 0:
                log.info(
                    "  Concepts: %d loaded (%d batches)",
                    concept_count, batch_num + 1,
                )

        stats.concepts_loaded = concept_count
        log.info("Phase 1 complete: %d concepts loaded", concept_count)

        # ── Phase 2: Load relationships (grouped by RELA) ──────
        stats.phase = "Loading relationships into Neo4j"
        log.info("Phase 2: %s", stats.phase)

        rel_by_type: dict[str, list[dict]] = defaultdict(list)
        for rel in parse_rxn_relationships(rrf_dir):
            rel_by_type[rel["rela"]].append({
                "rxcui1": rel["rxcui1"],
                "rxcui2": rel["rxcui2"],
            })

        rel_count = 0
        for rela_type, rels in rel_by_type.items():
            type_count = 0
            for batch_num, batch in enumerate(_batches(rels, batch_size)):
                async with get_session() as session:
                    n = await _batch_merge_relationships(session, rela_type, batch)
                    type_count += n
                    rel_count += n

            log.info("  %s: %d relationships", rela_type, type_count)

        stats.relationships_loaded = rel_count
        log.info("Phase 2 complete: %d relationships loaded", rel_count)

        # ── Phase 3: SNOMED CT crosswalk ────────────────────────
        stats.phase = "Loading SNOMED CT crosswalk"
        log.info("Phase 3: %s", stats.phase)

        xref_count = 0
        xref_batch = []
        for mapping in parse_snomed_crosswalk(rrf_dir):
            xref_batch.append(mapping)
            if len(xref_batch) >= batch_size:
                async with get_session() as session:
                    n = await _batch_merge_snomed_crosswalk(session, xref_batch)
                    xref_count += n
                xref_batch = []

        if xref_batch:
            async with get_session() as session:
                n = await _batch_merge_snomed_crosswalk(session, xref_batch)
                xref_count += n

        stats.snomed_crosswalk_loaded = xref_count
        log.info("Phase 3 complete: %d SNOMED crosswalk entries loaded", xref_count)

        # ── Phase 4: BELONGS_TO links ───────────────────────────
        stats.phase = "Linking concepts to Ontology node"
        log.info("Phase 4: %s", stats.phase)

        link_count = 0
        for batch in _batches(all_rxcuis, batch_size * 2):
            async with get_session() as session:
                n = await _batch_link_to_ontology(session, batch)
                link_count += n

        log.info("Phase 4 complete: %d BELONGS_TO links", link_count)

        # ── Done ────────────────────────────────────────────────
        stats.status = "complete"
        stats.phase = "Import complete"
        stats.elapsed_seconds = round(time.monotonic() - t0, 1)
        log.info(
            "RxNorm import complete in %.1fs — %d concepts, %d relationships, %d SNOMED xrefs",
            stats.elapsed_seconds,
            stats.concepts_loaded,
            stats.relationships_loaded,
            stats.snomed_crosswalk_loaded,
        )

    except Exception as e:
        stats.status = "error"
        stats.error = str(e)
        stats.elapsed_seconds = round(time.monotonic() - t0, 1)
        log.error("RxNorm import failed: %s", e, exc_info=True)
        raise

    return stats
