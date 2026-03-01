"""Shared state and helpers used across multiple API routers."""

from __future__ import annotations

import logging
import re
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
    """Call local LLM and return response with answer/model/processing_time.

    Uses the OpenAI-compatible endpoint. For thinking models (Qwen3.5),
    the model may think first — content may be in the reasoning field.
    """
    start = time.time()
    payload = {
        "model": settings.llm_model_name,
        "messages": [{"role": "user", "content": question}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
        response = await client.post(
            f"{settings.llm_base_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    msg = data["choices"][0]["message"]
    answer = (msg.get("content") or "").strip()

    # Thinking models may put everything in reasoning with content empty
    if not answer and msg.get("reasoning"):
        answer = msg["reasoning"].strip()
        # Strip any <think> wrapper if present
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    elapsed = round(time.time() - start, 2)
    return {"answer": answer, "model": settings.llm_model_display, "processing_time": elapsed}


def strip_llm_wrapper(text: str) -> str:
    """Strip <think> blocks and markdown code fences from LLM output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        if lines and lines[0].strip().lower() in ("json",):
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return text


async def ask_llm_nothink(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    timeout: float = 90.0,
) -> str:
    """Call LLM with thinking disabled — for structured JSON extraction tasks.

    Uses Ollama's native /api/chat with think=false to suppress chain-of-thought.
    The /nothink soft switch does NOT work via the OpenAI-compatible endpoint,
    so we must use the native API for structured output tasks.

    Returns the raw text content from the model (caller handles JSON parsing).
    Falls back to OpenAI-compatible endpoint reasoning extraction if native
    API is unavailable (e.g., running against vLLM instead of Ollama).
    """
    # ── Primary: Ollama native API with think=false ──────────────────
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{settings.llm_base_url}/api/chat",
                json={
                    "model": settings.llm_model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "think": False,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            answer = (data.get("message", {}).get("content") or "").strip()
            if answer:
                return answer
            log.warning("Ollama native API returned empty content")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # /api/chat not available — not Ollama (maybe vLLM)
            log.info("Ollama native API not available, falling back to OpenAI endpoint")
        else:
            raise
    except Exception as e:
        log.warning("Ollama native API call failed: %s", e)

    # ── Fallback: OpenAI-compatible endpoint with reasoning extraction ──
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{settings.llm_base_url}/v1/chat/completions",
            json={
                "model": settings.llm_model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        answer = (msg.get("content") or "").strip()
        if not answer and msg.get("reasoning"):
            answer = msg["reasoning"].strip()
        return answer
