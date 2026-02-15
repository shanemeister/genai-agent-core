from __future__ import annotations

import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer

# Configurable via environment variable. Options:
#   "nomic-ai/nomic-embed-text-v1.5"  — 768d, 8192 token context, best quality
#   "all-MiniLM-L6-v2"                — 384d, 512 token context, fastest
EMBED_MODEL = os.getenv("NOESIS_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it."""
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    print(f"[embeddings] Loaded {EMBED_MODEL} ({model.get_sentence_embedding_dimension()}d)")
    return model


def embed_text(text: str) -> list[float]:
    """Convert text to a semantic embedding vector.

    Returns a vector whose dimensionality depends on the configured model:
      - nomic-embed-text-v1.5: 768 dimensions
      - all-MiniLM-L6-v2: 384 dimensions
    """
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def get_embedding_dim() -> int:
    """Return the dimensionality of the current embedding model."""
    return _get_model().get_sentence_embedding_dimension()
