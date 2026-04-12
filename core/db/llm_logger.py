"""Structured logging for every LLM call.

Every call to the LLM — chat, validation, concept extraction, claim extraction,
memory proposal, diagram generation — gets a row in the llm_calls table.

Usage:
    from core.db.llm_logger import log_llm_call

    call_id = await log_llm_call(
        caller="chat_stream",
        session_id="chat-abc123",
        model="qwen3.5:35b-a3b",
        prompt="You are Noesis... [full prompt text]",
        response="Based on the Sepsis-3 criteria...",
        temperature=0.3,
        max_tokens=2000,
        duration_ms=7200,
        prompt_tokens=1500,
        completion_tokens=800,
        grounding_score=0.51,
    )

The returned UUID is the llm_calls.id — use it to link feedback records.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional
from uuid import UUID

from core.db.postgres import get_pool

log = logging.getLogger("noesis.observability")


async def log_llm_call(
    *,
    caller: str,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    reasoning: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    duration_ms: Optional[int] = None,
    grounding_score: Optional[float] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
    error: Optional[str] = None,
) -> Optional[UUID]:
    """Log an LLM call to the llm_calls table.

    All parameters except `caller` are optional — log whatever you have.
    Failures are caught and logged, never raised, so observability never
    breaks the pipeline it's observing.

    Returns the UUID of the inserted row, or None if logging failed.
    """
    try:
        pool = await get_pool()
        row_id = await pool.fetchval(
            """
            INSERT INTO llm_calls
                (caller, session_id, model, prompt, response, reasoning,
                 temperature, max_tokens, prompt_tokens, completion_tokens,
                 duration_ms, grounding_score, tool_calls, error)
            VALUES ($1, $2, $3, $4, $5, $6,
                    $7, $8, $9, $10,
                    $11, $12, $13::jsonb, $14)
            RETURNING id
            """,
            caller,
            session_id,
            model,
            prompt,
            response,
            reasoning,
            temperature,
            max_tokens,
            prompt_tokens,
            completion_tokens,
            duration_ms,
            grounding_score,
            json.dumps(tool_calls) if tool_calls else None,
            error,
        )
        return row_id
    except Exception as e:
        # Observability must never break the thing it's observing.
        # Log the failure and move on.
        log.warning("Failed to log LLM call (%s): %s", caller, e)
        return None


async def log_feedback(
    *,
    rating: str,
    llm_call_id: Optional[UUID] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
) -> Optional[UUID]:
    """Log user feedback (thumbs up/down) linked to an LLM call.

    Returns the UUID of the inserted row, or None if logging failed.
    """
    try:
        pool = await get_pool()
        row_id = await pool.fetchval(
            """
            INSERT INTO feedback
                (llm_call_id, session_id, rating, user_id, context)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            RETURNING id
            """,
            llm_call_id,
            session_id,
            rating,
            user_id,
            json.dumps(context) if context else None,
        )
        return row_id
    except Exception as e:
        log.warning("Failed to log feedback: %s", e)
        return None
