from __future__ import annotations

import os
from functools import lru_cache
from sentence_transformers import CrossEncoder

# Configurable via environment variable
RERANKER_MODEL = os.getenv("NOESIS_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    """Load the reranker model once and cache it."""
    model = CrossEncoder(RERANKER_MODEL)
    print(f"[reranker] Loaded {RERANKER_MODEL}")
    return model


def rerank(query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
    """Rerank retrieved documents by relevance to the query.

    Args:
        query: The user's question
        docs: List of dicts with at least "text" key (from vector search)
        top_k: Number of top results to return

    Returns:
        Top-k documents sorted by reranker score, with "rerank_score" added
    """
    if not docs:
        return []

    reranker = _get_reranker()

    # Build query-document pairs for the cross-encoder
    pairs = [[query, doc["text"]] for doc in docs]
    scores = reranker.predict(pairs)

    # Attach scores and sort
    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)

    ranked = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)
    return ranked[:top_k]
