from __future__ import annotations

from core.rag.embeddings import embed_text
from core.rag.vector_store import VectorStore

_STORE = VectorStore()
_SEEDED = False
_INDEXED_CARD_IDS: set[str] = set()


def _seed_store() -> None:
    global _SEEDED
    if _SEEDED:
        return

    seed_docs = [
        ("doc:v0-loop", "Ask -> Retrieve -> Reason -> Produce Artifact -> Propose Memory"),
        ("doc:v0-artifacts", "V0 centers diagram canvas, graph explorer, and memory deck"),
        ("doc:v0-local", "System should be local-first, private, and explicit about memory"),
    ]
    for doc_id, text in seed_docs:
        _STORE.add(doc_id, embed_text(text), text)

    _SEEDED = True


def index_memory_cards(cards) -> int:
    """Index approved memory cards into the vector store. Returns count of newly indexed."""
    _seed_store()
    added = 0
    for card in cards:
        card_id = f"memory:{card.id}"
        if card_id not in _INDEXED_CARD_IDS:
            _STORE.add(card_id, embed_text(card.text), card.text)
            _INDEXED_CARD_IDS.add(card_id)
            added += 1
    return added


def retrieve_context(query: str, k: int = 5) -> list[dict]:
    _seed_store()
    query_embedding = embed_text(query)
    return _STORE.search(query_embedding, k=k)
