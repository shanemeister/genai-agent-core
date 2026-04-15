"""Load curated documentation corpora into pgvector RAG store.

Supports two corpora:

1. Product documentation (noesis_product_corpus.json) — describes Noesis
   Health itself. Loaded with source_type='product_documentation' and
   doc_id prefix 'product_doc:'. Questions like "What does Noesis do?"
   and "What is Axiom Core?" retrieve these chunks.

2. Clinical guidelines (clinical_guidelines_corpus.json) — authoritative
   summaries of Sepsis-3, SOFA, CDI query compliance, heart failure
   staging, MS-DRG, POA/HAC, and related topics drawn from public
   consensus guidelines and CMS documentation. Loaded with
   source_type='clinical_guideline' and doc_id prefix 'clin_guide:'.
   Questions like "What is Sepsis-3?" or "What makes a CDI query
   compliant?" retrieve these chunks.

Both corpora share the same JSON schema (description, version, chunks[]
with id/title/text). The loader is fully re-runnable — existing entries
are UPSERTed by stable doc_id, so no duplicates accumulate.

Usage:
    # Product docs (default)
    python tools/product_docs/load_product_docs.py
    python tools/product_docs/load_product_docs.py --verify

    # Clinical guidelines
    python tools/product_docs/load_product_docs.py --corpus clinical_guidelines_corpus.json --verify

    # Wipe and reload
    python tools/product_docs/load_product_docs.py --clear-first
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

log = logging.getLogger("noesis.corpus_loader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DEFAULT_CORPUS = "noesis_product_corpus.json"

# Corpus registry: filename → (source_type, doc_id_prefix, verification_queries)
# Verification queries map each user question to a list of acceptable
# top-1 chunk IDs. These are smoke tests, not benchmarks.
CORPUS_REGISTRY = {
    "noesis_product_corpus.json": {
        "source_type": "product_documentation",
        "doc_id_prefix": "product_doc:",
        "verification": [
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
        ],
    },
    "clinical_guidelines_corpus.json": {
        "source_type": "clinical_guideline",
        "doc_id_prefix": "clin_guide:",
        "verification": [
            ("What is Sepsis-3?", ["sepsis-3-definition"]),
            ("What is the SOFA score?", ["sofa-score"]),
            ("What is qSOFA?", ["qsofa-screening"]),
            ("What makes a CDI query compliant?", ["cdi-query-compliance"]),
            ("How is heart failure classified?", ["heart-failure-staging"]),
            ("What is MS-DRG?", ["cms-ms-drg-overview"]),
            ("What is the difference between CC and MCC?", ["cms-mcc-cc-distinction"]),
            ("What is a POA indicator?", ["poa-indicator-and-hac"]),
            ("What are ICD-10-CM coding principles?", ["icd-10-cm-coding-principles"]),
            ("What is a RAC audit?", ["rac-audits"]),
            ("What qualifies as a secondary diagnosis?", ["secondary-diagnosis-reporting"]),
            ("What is case mix index?", ["cmi-case-mix-index"]),
        ],
    },
}


async def clear_existing(store: PgVectorStore, doc_id_prefix: str) -> int:
    """Remove all entries with the given doc_id prefix before re-loading."""
    count = await store.remove_by_prefix(doc_id_prefix)
    log.info("Cleared %d existing entries with prefix %s", count, doc_id_prefix)
    return count


async def load_corpus(
    store: PgVectorStore,
    corpus: dict,
    source_type: str,
    doc_id_prefix: str,
) -> int:
    """Embed and upsert every chunk in the corpus."""
    chunks = corpus.get("chunks", [])
    if not chunks:
        log.warning("Corpus contains no chunks")
        return 0

    log.info("Loading %d chunks (source_type=%s) ...", len(chunks), source_type)

    items = []
    for chunk in chunks:
        chunk_id = chunk.get("id")
        title = chunk.get("title", "")
        text = chunk.get("text", "")
        if not chunk_id or not text:
            log.warning("Skipping chunk with missing id or text: %s", chunk)
            continue

        # Embed title + body so the vector captures topic framing plus content.
        embedding_text = f"{title}\n\n{text}" if title else text
        vector = embed_text(embedding_text)

        items.append({
            "doc_id": f"{doc_id_prefix}{chunk_id}",
            "vector": vector,
            "text": text,
            "metadata": {
                "source_type": source_type,
                "title": title,
                "chunk_id": chunk_id,
                "corpus_version": corpus.get("version", "unknown"),
            },
        })

    added = await store.add_batch(items)
    log.info("Loaded %d chunks into pgvector", added)
    return added


async def verify(
    store: PgVectorStore,
    source_type: str,
    doc_id_prefix: str,
    test_queries: list,
) -> None:
    """Smoke-test that the loaded chunks retrieve correctly."""
    log.info("━" * 70)
    log.info("Retrieval verification — top-1 match against expected chunk IDs")
    log.info("source_type=%s", source_type)
    log.info("━" * 70)

    passed = 0
    failed = 0
    for query, expected_ids in test_queries:
        query_vec = embed_text(query)
        results = await store.search(query_vec, k=3, source_types=[source_type])
        if not results:
            log.warning("  ❌ '%s' → NO results", query)
            failed += 1
            continue

        top_id = results[0]["doc_id"].replace(doc_id_prefix, "")
        top_score = results[0]["score"]

        if top_id in expected_ids:
            log.info("  ✓ '%s'", query[:60])
            log.info("      → %s (score %.3f)", top_id, top_score)
            passed += 1
        else:
            log.warning("  ⚠ '%s'", query[:60])
            log.warning("      → got %s (expected one of %s)", top_id, expected_ids)
            log.warning("      → top 3: %s",
                       [r["doc_id"].replace(doc_id_prefix, "") for r in results[:3]])
            failed += 1

    log.info("━" * 70)
    log.info("Verification: %d/%d passed", passed, passed + failed)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help=f"Corpus filename under tools/product_docs/ (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Remove existing entries (matching this corpus's doc_id prefix) before loading",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run retrieval verification after loading",
    )
    args = parser.parse_args()

    corpus_filename = Path(args.corpus).name  # guard against path traversal
    if corpus_filename not in CORPUS_REGISTRY:
        log.error(
            "Unknown corpus '%s'. Registered corpora: %s",
            corpus_filename,
            list(CORPUS_REGISTRY.keys()),
        )
        sys.exit(1)

    corpus_path = Path(__file__).parent / corpus_filename
    if not corpus_path.exists():
        log.error("Corpus file not found: %s", corpus_path)
        sys.exit(1)

    registry_entry = CORPUS_REGISTRY[corpus_filename]
    source_type = registry_entry["source_type"]
    doc_id_prefix = registry_entry["doc_id_prefix"]
    test_queries = registry_entry["verification"]

    corpus = json.loads(corpus_path.read_text())
    log.info(
        "Loaded corpus: %s (version %s, %d chunks, source_type=%s)",
        corpus.get("description", "")[:60],
        corpus.get("version", "unknown"),
        len(corpus.get("chunks", [])),
        source_type,
    )

    await init_database()
    store = PgVectorStore()

    if args.clear_first:
        await clear_existing(store, doc_id_prefix)

    await load_corpus(store, corpus, source_type, doc_id_prefix)

    total = await store.count(source_type=source_type)
    log.info("Total %s entries in pgvector: %d", source_type, total)

    if args.verify:
        await verify(store, source_type, doc_id_prefix, test_queries)


if __name__ == "__main__":
    asyncio.run(main())
