"""Batch embedding pipeline for SNOMED CT concepts into pgvector.

Reads :SnomedConcept nodes from Neo4j, embeds their preferred terms
using nomic-embed-text-v1.5, and upserts into the embeddings table
with source_type='ontology'.

Usage:
    from core.ontology.snomed_embedder import embed_snomed_concepts
    count = await embed_snomed_concepts()
"""

from __future__ import annotations

import asyncio
import logging
import time
from functools import partial

from core.config import settings
from core.graph.neo4j_client import get_session
from core.ontology.rf2_parser import SKIP_SEMANTIC_TAGS
from core.rag.embeddings import embed_texts_batch
from core.rag.vector_store import PgVectorStore

log = logging.getLogger("noesis.ontology")

_store = PgVectorStore()


async def _fetch_concepts_to_embed() -> list[dict]:
    """Fetch all SnomedConcept nodes from Neo4j that should be embedded.

    Filters out non-clinical semantic tags (metadata, linkage, etc.).
    """
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (s:SnomedConcept)
            WHERE s.preferred_term IS NOT NULL AND s.preferred_term <> ''
            RETURN s.sctid AS sctid, s.fsn AS fsn,
                   s.preferred_term AS pt, s.semantic_tag AS tag
            """
        )
        records = [record.data() async for record in result]

    # Filter out non-clinical tags
    filtered = [
        r for r in records
        if r.get("tag", "").lower() not in SKIP_SEMANTIC_TAGS
    ]
    log.info(
        "Fetched %d concepts from Neo4j, %d after filtering non-clinical tags",
        len(records),
        len(filtered),
    )
    return filtered


async def _get_existing_ontology_ids() -> set[str]:
    """Get set of doc_ids already embedded with source_type='ontology'."""
    from core.db.postgres import get_pool

    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT doc_id FROM embeddings WHERE source_type = 'ontology'"
    )
    return {row["doc_id"] for row in rows}


async def embed_snomed_concepts(
    batch_size: int | None = None,
) -> int:
    """Embed SNOMED CT concept terms into pgvector.

    Skips concepts that are already embedded (idempotent).
    Runs embedding on CPU in a thread pool to avoid blocking the event loop.

    Args:
        batch_size: Number of texts per embedding batch. Defaults to
            settings.ontology_embed_batch_size.

    Returns:
        Number of newly embedded concepts.
    """
    batch_size = batch_size or settings.ontology_embed_batch_size
    t0 = time.monotonic()

    # Fetch concepts from Neo4j
    concepts = await _fetch_concepts_to_embed()
    if not concepts:
        log.warning("No SNOMED concepts found in Neo4j to embed")
        return 0

    # Check what's already embedded
    existing = await _get_existing_ontology_ids()
    log.info("Found %d existing ontology embeddings in pgvector", len(existing))

    # Filter to only new concepts
    to_embed = [
        c for c in concepts
        if f"ontology:snomed:{c['sctid']}" not in existing
    ]
    log.info("%d new concepts to embed (skipping %d already embedded)",
             len(to_embed), len(concepts) - len(to_embed))

    if not to_embed:
        log.info("All SNOMED concepts already embedded — nothing to do")
        return 0

    # Process in batches
    embedded_count = 0
    total_batches = (len(to_embed) + batch_size - 1) // batch_size
    loop = asyncio.get_event_loop()

    for batch_idx in range(0, len(to_embed), batch_size):
        batch = to_embed[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        # Build embedding texts: "preferred_term (semantic_tag)"
        texts = [
            f"{c['pt']} ({c['tag']})" if c.get("tag") else c["pt"]
            for c in batch
        ]

        # Run embedding in thread pool (CPU-bound, don't block event loop)
        vectors = await loop.run_in_executor(
            None,
            partial(embed_texts_batch, texts, batch_size),
        )

        # Upsert into pgvector
        for concept, vector in zip(batch, vectors):
            doc_id = f"ontology:snomed:{concept['sctid']}"
            metadata = {
                "source_type": "ontology",
                "ontology": "snomed",
                "sctid": concept["sctid"],
                "semantic_tag": concept.get("tag", ""),
                "fsn": concept.get("fsn", ""),
            }
            text = f"{concept['pt']} ({concept.get('tag', '')})"
            await _store.add(doc_id, vector, text, metadata)

        embedded_count += len(batch)

        if batch_num % 10 == 0 or batch_num == total_batches:
            elapsed = time.monotonic() - t0
            rate = embedded_count / elapsed if elapsed > 0 else 0
            log.info(
                "  Embedded %d/%d (batch %d/%d, %.0f concepts/sec)",
                embedded_count, len(to_embed), batch_num, total_batches, rate,
            )

    elapsed = round(time.monotonic() - t0, 1)
    log.info(
        "SNOMED embedding complete: %d concepts in %.1fs",
        embedded_count, elapsed,
    )
    return embedded_count
