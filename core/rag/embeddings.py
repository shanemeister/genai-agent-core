from __future__ import annotations

import logging
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from core.config import settings

log = logging.getLogger("noesis.rag")


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it.

    Forces CPU device since GPUs are dedicated to vLLM inference.
    """
    model = SentenceTransformer(
        settings.noesis_embed_model, trust_remote_code=True, device="cpu"
    )
    log.info("Loaded %s (%dd) on CPU", settings.noesis_embed_model, model.get_sentence_embedding_dimension())
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


def embed_texts_batch(texts: list[str], batch_size: int = 256) -> list[list[float]]:
    """Batch-embed multiple texts at once.

    Uses SentenceTransformer's native batch processing for much higher
    throughput than calling embed_text() in a loop.

    Args:
        texts: List of strings to embed.
        batch_size: Internal batch size for the model encoder.

    Returns:
        List of embedding vectors (same order as input texts).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def get_embedding_dim() -> int:
    """Return the dimensionality of the current embedding model."""
    return _get_model().get_sentence_embedding_dimension()
