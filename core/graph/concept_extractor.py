from __future__ import annotations

import logging
import httpx
import re

from core.config import settings

log = logging.getLogger("noesis.graph")


async def extract_concepts(text: str, max_concepts: int = 5) -> list[str]:
    """Use the local LLM to extract concept names from text.

    Returns a list of lowercase concept strings. Falls back to
    simple keyword extraction if the LLM is unreachable.
    """
    prompt = (
        "Extract the main concepts from the following text. "
        f"Return at most {max_concepts} concepts as a comma-separated list. "
        "Only return the list, nothing else.\n\n"
        f"Text: {text}"
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.vllm_base_url}/v1/chat/completions",
                json={
                    "model": settings.vllm_model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 120,
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip()
            concepts = [c.strip().lower() for c in answer.split(",") if c.strip()]
            return concepts[:max_concepts]
    except Exception as e:
        log.info("LLM concept extraction unavailable, using fallback: %s", e)
        return _fallback_extract(text, max_concepts)


def _fallback_extract(text: str, max_concepts: int = 5) -> list[str]:
    """Simple keyword extraction when LLM is unavailable."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "and",
        "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "just", "because", "about", "between",
        "this", "that", "these", "those", "it", "its", "they", "them",
        "their", "we", "our", "you", "your", "i", "my", "me", "he",
        "she", "his", "her", "which", "what", "who", "whom", "how",
    }
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    seen = set()
    concepts = []
    for w in words:
        if w not in stop_words and w not in seen:
            seen.add(w)
            concepts.append(w)
        if len(concepts) >= max_concepts:
            break
    return concepts
