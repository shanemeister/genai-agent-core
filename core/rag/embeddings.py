from __future__ import annotations

from functools import lru_cache
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Load the embedding model once and cache it.

    Model: all-MiniLM-L6-v2
    - Size: ~80MB
    - Dimensions: 384
    - Speed: ~1ms per sentence on CPU
    - Quality: Good for semantic search, trained on 1B+ pairs

    The @lru_cache decorator ensures we only load the model once,
    even if this function is called multiple times.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')


def embed_text(text: str) -> list[float]:
    """
    Convert text to a semantic embedding vector.

    Args:
        text: Input text to embed

    Returns:
        384-dimensional vector representing the semantic meaning

    Example:
        >>> embed_text("hello world")
        [0.123, -0.456, 0.789, ...]  # 384 numbers

    The model maps semantically similar sentences to nearby vectors:
        embed_text("car") ≈ embed_text("automobile")
        embed_text("car") ≠ embed_text("banana")
    """
    model = _get_model()
    # convert_to_numpy=True ensures we get a numpy array
    embedding = model.encode(text, convert_to_numpy=True)
    # Convert to list for JSON serialization
    return embedding.tolist()
