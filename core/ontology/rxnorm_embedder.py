"""Batch embedding pipeline for RxNorm concepts into pgvector.

Reads :RxNormConcept nodes from Neo4j, embeds their preferred terms
using nomic-embed-text-v1.5, and upserts into the embeddings table
with source_type='ontology' and ontology='rxnorm'.

Usage:
    from core.ontology.rxnorm_embedder import embed_rxnorm_concepts
    count = await embed_rxnorm_concepts()
"""

from __future__ import annotations

import asyncio
import logging
import time
from functools import partial

from core.config import settings
from core.graph.neo4j_client import get_session
from core.ontology.rxnorm_parser import SKIP_EMBED_TTYS
from core.rag.embeddings import embed_texts_batch
from core.rag.vector_store import PgVectorStore

log = logging.getLogger("noesis.ontology")

_store = PgVectorStore()


async def _fetch_concepts_to_embed() -> list[dict]:
    """Fetch RxNormConcept nodes from Neo4j that should be embedded.

    Filters out dose form TTYs (DF, DFG, etc.) that aren't useful
    for clinical text matching.
    """
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (r:RxNormConcept)
            WHERE r.preferred_term IS NOT NULL AND r.preferred_term <> ''
            RETURN r.rxcui AS rxcui, r.preferred_term AS pt, r.tty AS tty
            """
        )
        records = [record.data() async for record in result]

    # Filter out non-clinical TTYs
    filtered = [
        r for r in records
        if r.get("tty", "").upper() not in SKIP_EMBED_TTYS
    ]
    log.info(
        "Fetched %d RxNorm concepts from Neo4j, %d after filtering non-clinical TTYs",
        len(records),
        len(filtered),
    )
    return filtered


async def _get_existing_rxnorm_ids() -> set[str]:
    """Get set of doc_ids already embedded for RxNorm."""
    from core.db.postgres import get_pool

    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT doc_id FROM embeddings WHERE doc_id LIKE 'ontology:rxnorm:%'"
    )
    return {row["doc_id"] for row in rows}


async def embed_rxnorm_concepts(
    batch_size: int | None = None,
) -> int:
    """Embed RxNorm concept terms into pgvector.

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
        log.warning("No RxNorm concepts found in Neo4j to embed")
        return 0

    # Check what's already embedded
    existing = await _get_existing_rxnorm_ids()
    log.info("Found %d existing RxNorm embeddings in pgvector", len(existing))

    # Filter to only new concepts
    to_embed = [
        c for c in concepts
        if f"ontology:rxnorm:{c['rxcui']}" not in existing
    ]
    log.info("%d new RxNorm concepts to embed (skipping %d already embedded)",
             len(to_embed), len(concepts) - len(to_embed))

    if not to_embed:
        log.info("All RxNorm concepts already embedded — nothing to do")
        return 0

    # Process in batches
    embedded_count = 0
    total_batches = (len(to_embed) + batch_size - 1) // batch_size
    loop = asyncio.get_event_loop()

    for batch_idx in range(0, len(to_embed), batch_size):
        batch = to_embed[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        # Build embedding texts — just the preferred term (no semantic tag for RxNorm)
        texts = [c["pt"] for c in batch]

        # Run embedding in thread pool (CPU-bound, don't block event loop)
        vectors = await loop.run_in_executor(
            None,
            partial(embed_texts_batch, texts, batch_size),
        )

        # Upsert into pgvector
        for concept, vector in zip(batch, vectors):
            doc_id = f"ontology:rxnorm:{concept['rxcui']}"
            metadata = {
                "source_type": "ontology",
                "ontology": "rxnorm",
                "rxcui": concept["rxcui"],
                "tty": concept.get("tty", ""),
            }
            await _store.add(doc_id, vector, concept["pt"], metadata)

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
        "RxNorm embedding complete: %d concepts in %.1fs",
        embedded_count, elapsed,
    )
    return embedded_count
