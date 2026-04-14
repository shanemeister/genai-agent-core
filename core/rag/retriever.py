"""RAG retrieval layer — async, backed by pgvector.

Functions:
  seed_store()               — upsert 3 system seed docs (idempotent)
  index_memory_cards(cards)  — upsert approved memory cards
  retrieve_context(query)    — vector search + optional reranking
  reindex_document_chunks()  — re-embed document chunks missing from embeddings table
"""

from __future__ import annotations

import logging

from core.config import settings
from core.rag.embeddings import embed_text
from core.rag.vector_store import PgVectorStore

log = logging.getLogger("noesis.rag")
_STORE = PgVectorStore()


async def seed_store() -> None:
    """Seed vector store with initial system docs (idempotent via upsert)."""
    seed_docs = [
        ("doc:v0-loop", "Ask -> Retrieve -> Reason -> Produce Artifact -> Propose Memory"),
        ("doc:v0-artifacts", "V0 centers diagram canvas, graph explorer, and memory deck"),
        ("doc:v0-local", "System should be local-first, private, and explicit about memory"),
    ]
    for doc_id, text in seed_docs:
        if not await _STORE.exists(doc_id):
            await _STORE.add(
                doc_id,
                embed_text(text),
                text,
                metadata={"source_type": "seed"},
            )
    log.info("Seed docs ready")


async def index_memory_cards(cards) -> int:
    """Index approved memory cards into pgvector (idempotent via upsert).

    Skips cards already in the store to avoid redundant embedding calls.
    Returns count of newly indexed cards.
    """
    added = 0
    for card in cards:
        card_id = f"memory:{card.id}"
        if not await _STORE.exists(card_id):
            await _STORE.add(
                card_id,
                embed_text(card.text),
                card.text,
                metadata={
                    "source_type": "memory",
                    "category": card.category.value,
                    "scope": card.scope.value,
                },
            )
            added += 1
    return added


async def retrieve_context(
    query: str, k: int = 5, sources: list[str] | None = None
) -> list[dict]:
    """Retrieve relevant context using vector search + optional reranking.

    Pipeline:
      1. Vector search: retrieve top candidates by inner-product similarity
      2. Rerank: cross-encoder reranks candidates, returns top k

    Special behavior: always try to include up to 3 product_documentation
    chunks in the result set when they're relevant. This prevents the
    ~445K SNOMED/RxNorm concept embeddings from crowding out the small
    (~14 chunk) curated product documentation corpus for questions about
    the Noesis product itself. Product docs are retrieved from their own
    scoped search and merged into the final result, deduplicated by doc_id.

    Args:
        query: Search query text
        k: Number of results to return in the final merged set
        sources: Optional list of source types to filter by. If None,
                 searches all sources AND also pulls from product_documentation.
                 If explicitly passed, only those sources are used.
    """
    await seed_store()
    query_embedding = embed_text(query)

    # Decide on per-call k values. We want room for product docs
    # alongside general retrieval.
    if sources is None:
        # General + product doc boost: 3 product chunks + (k-3) general
        product_k = 3
        general_k = max(k - product_k, k // 2)  # leave room for at least half general

        # Pull product docs first
        product_results = await _STORE.search(
            query_embedding,
            k=product_k,
            source_types=["product_documentation"],
        )
    else:
        product_results = []
        general_k = k

    if settings.noesis_use_reranker:
        # Retrieve more candidates than needed, then rerank
        candidates = await _STORE.search(
            query_embedding,
            k=max(general_k * 3, 15),
            source_types=sources,
        )
        try:
            from core.rag.reranker import rerank
            general_ranked = rerank(query, candidates, top_k=general_k)
        except Exception as e:
            log.warning("Reranker failed, falling back to vector search: %s", e)
            general_ranked = candidates[:general_k]
    else:
        general_ranked = await _STORE.search(
            query_embedding, k=general_k, source_types=sources
        )

    # Merge product docs with general retrieval, deduplicate by doc_id.
    # Product docs go first so the LLM sees them before ontology concepts.
    seen_ids = set()
    merged: list[dict] = []
    for doc in product_results:
        if doc["doc_id"] not in seen_ids:
            seen_ids.add(doc["doc_id"])
            merged.append(doc)
    for doc in general_ranked:
        if doc["doc_id"] not in seen_ids:
            seen_ids.add(doc["doc_id"])
            merged.append(doc)
        if len(merged) >= k:
            break

    return merged[:k]


async def reindex_document_chunks() -> int:
    """Re-embed document chunks that exist in PostgreSQL but not in embeddings table.

    Called on server startup to recover vectors lost before pgvector migration.
    Returns count of chunks re-indexed.
    """
    from core.db.postgres import get_pool

    pool = await get_pool()

    # Find chunks in document_chunks table whose embedding is missing
    rows = await pool.fetch("""
        SELECT dc.id, dc.document_id, dc.chunk_index, dc.text, dc.page_number,
               d.filename, d.file_type
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE d.status = 'indexed'
          AND NOT EXISTS (
              SELECT 1 FROM embeddings e WHERE e.doc_id = 'chunk:' || dc.id
          )
        ORDER BY dc.document_id, dc.chunk_index
    """)

    if not rows:
        log.info("All document chunks already indexed in pgvector")
        return 0

    log.info("Re-indexing %d document chunks...", len(rows))
    count = 0
    for row in rows:
        vector = embed_text(row["text"])
        await _STORE.add(
            doc_id=f"chunk:{row['id']}",
            vector=vector,
            text=row["text"],
            metadata={
                "source_type": "document_chunk",
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "filename": row["filename"],
                "file_type": row["file_type"],
                "page_number": row["page_number"],
            },
        )
        count += 1
        if count % 50 == 0:
            log.info("  ...%d/%d chunks re-indexed", count, len(rows))

    log.info("Re-indexed %d document chunks into pgvector", count)
    return count
