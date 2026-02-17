from __future__ import annotations

import logging
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from core.config import settings

log = logging.getLogger("noesis.rag")


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it."""
    model = SentenceTransformer(settings.noesis_embed_model, trust_remote_code=True)
    log.info("Loaded %s (%dd)", settings.noesis_embed_model, model.get_sentence_embedding_dimension())
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
