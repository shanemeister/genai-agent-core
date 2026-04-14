"""Load Noesis product documentation into pgvector RAG store.

Reads tools/product_docs/noesis_product_corpus.json (a curated set of
14 product description chunks), embeds each chunk with nomic-embed-text,
and upserts into the embeddings table with source_type='product_documentation'.

The chat pipeline's RAG retriever picks these up alongside other sources,
so questions like "What does Noesis do?", "What's Axiom Core?", or
"How does CMS Hospital Compare benchmarking work?" get high-grounding
answers from authoritative product docs instead of LLM pretraining.

Every chunk has a stable doc_id (e.g., 'product_doc:noesis-overview'),
so this script is fully re-runnable. Existing chunks are UPSERTed; no
duplicates accumulate on repeated loads.

Usage:
    python tools/product_docs/load_product_docs.py
    python tools/product_docs/load_product_docs.py --clear-first  # wipe old product docs first
    python tools/product_docs/load_product_docs.py --verify       # test grounding on key questions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.db.postgres import get_pool, init_database
from core.rag.embeddings import embed_text
from core.rag.vector_store import PgVectorStore

log = logging.getLogger("noesis.product_docs_loader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

CORPUS_PATH = Path(__file__).parent / "noesis_product_corpus.json"
SOURCE_TYPE = "product_documentation"
DOC_ID_PREFIX = "product_doc:"


async def clear_existing(store: PgVectorStore) -> int:
    """Remove all product_documentation entries before re-loading."""
    count = await store.remove_by_prefix(DOC_ID_PREFIX)
    log.info("Cleared %d existing product_documentation entries", count)
    return count


async def load_corpus(store: PgVectorStore, corpus: dict) -> int:
    """Embed and upsert every chunk in the corpus."""
    chunks = corpus.get("chunks", [])
    if not chunks:
        log.warning("Corpus contains no chunks")
        return 0

    log.info("Loading %d product documentation chunks ...", len(chunks))

    items = []
    for chunk in chunks:
        chunk_id = chunk.get("id")
        title = chunk.get("title", "")
        text = chunk.get("text", "")
        if not chunk_id or not text:
            log.warning("Skipping chunk with missing id or text: %s", chunk)
            continue

        # Embed the title + body so the vector captures the topic framing
        # as well as the content. Short title at the front boosts recall
        # for direct-question searches like "What is Noesis Health?".
        embedding_text = f"{title}\n\n{text}" if title else text
        vector = embed_text(embedding_text)

        items.append({
            "doc_id": f"{DOC_ID_PREFIX}{chunk_id}",
            "vector": vector,
            "text": text,
            "metadata": {
                "source_type": SOURCE_TYPE,
                "title": title,
                "chunk_id": chunk_id,
                "corpus_version": corpus.get("version", "unknown"),
            },
        })

    added = await store.add_batch(items)
    log.info("Loaded %d product documentation chunks into pgvector", added)
    return added


async def verify(store: PgVectorStore) -> None:
    """Smoke-test that the loaded chunks retrieve correctly for key questions."""
    test_queries = [
        ("What is Noesis Health?", ["noesis-overview"]),
        ("What does Noesis do?", ["noesis-core-capabilities", "noesis-overview"]),
        ("What is Axiom Core?", ["noesis-axiom-core"]),
        ("What is Noesis Gateway?", ["noesis-gateway"]),
        ("How does the DRG financial impact calculation work?", ["noesis-drg-financial-impact"]),
        ("What is CMS Hospital Compare benchmarking?", ["noesis-cms-hospital-compare-benchmark"]),
        ("How does Noesis handle Boston Children's Hospital?", ["noesis-cms-hospital-compare-benchmark"]),
        ("What CDI specificity rules does Noesis have?", ["noesis-cdi-specificity-rules"]),
        ("Is Noesis cloud-based?", ["noesis-privacy-and-local-inference"]),
        ("What does Noesis NOT do?", ["noesis-what-it-does-not-do"]),
    ]

    log.info("━" * 70)
    log.info("Retrieval verification — testing top-1 match against expected chunk IDs")
    log.info("━" * 70)

    passed = 0
    failed = 0
    for query, expected_ids in test_queries:
        query_vec = embed_text(query)
        results = await store.search(query_vec, k=3, source_types=[SOURCE_TYPE])
        if not results:
            log.warning("  ❌ '%s' → NO results", query)
            failed += 1
            continue

        top_id = results[0]["doc_id"].replace(DOC_ID_PREFIX, "")
        top_score = results[0]["score"]

        if top_id in expected_ids:
            log.info("  ✓ '%s'", query[:60])
            log.info("      → %s (score %.3f)", top_id, top_score)
            passed += 1
        else:
            log.warning("  ⚠ '%s'", query[:60])
            log.warning("      → got %s (expected one of %s)", top_id, expected_ids)
            log.warning("      → top 3: %s",
                       [r["doc_id"].replace(DOC_ID_PREFIX, "") for r in results[:3]])
            failed += 1

    log.info("━" * 70)
    log.info("Verification: %d/%d passed", passed, passed + failed)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Remove existing product_documentation entries before loading",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run retrieval verification after loading",
    )
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        log.error("Corpus file not found: %s", CORPUS_PATH)
        sys.exit(1)

    corpus = json.loads(CORPUS_PATH.read_text())
    log.info("Loaded corpus: %s (version %s, %d chunks)",
             corpus.get("description", "")[:60],
             corpus.get("version", "unknown"),
             len(corpus.get("chunks", [])))

    await init_database()
    store = PgVectorStore()

    if args.clear_first:
        await clear_existing(store)

    await load_corpus(store, corpus)

    total = await store.count(source_type=SOURCE_TYPE)
    log.info("Total product_documentation entries in pgvector: %d", total)

    if args.verify:
        await verify(store)


if __name__ == "__main__":
    asyncio.run(main())
