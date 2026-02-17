from __future__ import annotations

import json
import logging
import re

import httpx

from core.config import settings

log = logging.getLogger("noesis.validation")


async def extract_claims(
    response_text: str,
    user_question: str,
    max_claims: int = 6,
) -> list[str]:
    """Extract atomic factual claims from an LLM response.

    Returns a list of claim strings (3-8 items).
    Falls back to sentence splitting if LLM fails.
    """
    truncated = response_text[:1500]

    prompt = f"""Break the following response into atomic factual claims.

A claim is a single, verifiable statement of fact. NOT opinions, questions, or hedging language.

User asked: {user_question[:200]}

Response to analyze:
{truncated}

Rules:
- Extract {max_claims} or fewer claims
- Each claim must be one sentence, self-contained
- Skip subjective opinions ("I think", "probably")
- Skip meta-statements ("As I mentioned", "To summarize")
- Keep the original meaning — do not add information
- If the response is purely conversational (greeting, apology), return an empty array

Return a JSON array of strings. Example:
["The V0 loop consists of five stages", "Memory cards require human approval"]

Return ONLY the JSON array, nothing else."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.vllm_base_url}/v1/chat/completions",
                json={
                    "model": settings.vllm_model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 600,
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip <think> reasoning blocks
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
        if "</think>" in answer:
            answer = answer.split("</think>", 1)[-1]

        # Strip markdown code fences
        answer = answer.strip()
        if answer.startswith("```"):
            lines = answer.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            if lines and lines[0].strip().lower() in ("json",):
                lines = lines[1:]
            answer = "\n".join(lines).strip()

        claims = json.loads(answer)
        if not isinstance(claims, list):
            return _fallback_split(response_text, max_claims)

        result = []
        for c in claims[:max_claims]:
            if isinstance(c, str) and len(c.strip()) > 10:
                result.append(c.strip())

        return result if result else _fallback_split(response_text, max_claims)

    except Exception as e:
        log.info("LLM claim extraction unavailable, using fallback: %s", e)
        return _fallback_split(response_text, max_claims)


def _fallback_split(text: str, max_claims: int) -> list[str]:
    """Simple sentence splitter as fallback when LLM is unavailable."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = []
    skip_prefixes = ("I think", "Perhaps", "Maybe", "Note:", "Okay,", "So,", "Well,")
    for s in sentences:
        s = s.strip()
        if len(s) > 20 and not s.startswith(skip_prefixes):
            claims.append(s)
        if len(claims) >= max_claims:
            break
    return claims
