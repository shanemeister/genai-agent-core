"""Shared state and helpers used across multiple API routers."""

from __future__ import annotations

import logging
import time

import httpx

from core.artifacts.memory_card import MemoryApproval, MemoryCard
from core.config import settings

log = logging.getLogger("noesis")

# In-memory mirror of approved+pending memory cards (loaded at startup from PG)
MEMORY_CARDS: dict[str, MemoryCard] = {}


async def ask_llm(
    question: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> dict:
    """Call vLLM directly and return response with answer/model/processing_time."""
    start = time.time()
    payload = {
        "model": settings.vllm_model_name,
        "messages": [{"role": "user", "content": question}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
        response = await client.post(
            f"{settings.vllm_base_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    answer = data["choices"][0]["message"]["content"].strip()
    elapsed = round(time.time() - start, 2)
    return {"answer": answer, "model": "DeepSeek-R1-70B", "processing_time": elapsed}
