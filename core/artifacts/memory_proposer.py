from __future__ import annotations

import json
import logging
import re

import httpx

from core.config import settings

log = logging.getLogger("noesis.memory")


async def propose_memories(
    user_message: str,
    assistant_response: str,
    max_proposals: int = 2,
) -> list[dict]:
    """Ask the LLM to extract memory-worthy insights from a conversation turn.

    Returns a list of dicts with keys: text, category, reason.
    Categories must be one of: principles_values, cognitive_framing,
    decision_heuristics, preferences, vocabulary.
    """
    prompt = f"""Analyze this conversation and identify any insights worth remembering long-term.

User said: {user_message}

Assistant responded: {assistant_response[:800]}

Extract at most {max_proposals} memory-worthy insights. These should be:
- Enduring principles, values, or preferences expressed by the user
- Cognitive framings or mental models discussed
- Decision heuristics or rules of thumb mentioned
- Important vocabulary or definitions introduced

Return a JSON array (no markdown, no code fences). Each item must have:
- "text": the insight to remember (1-2 sentences)
- "category": one of "principles_values", "cognitive_framing", "decision_heuristics", "preferences", "vocabulary"
- "reason": why this is worth remembering (1 sentence)

If nothing is worth remembering, return an empty array: []

Return ONLY the JSON array, nothing else."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.vllm_base_url}/v1/chat/completions",
                json={
                    "model": settings.vllm_model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip()

        # Strip <think> reasoning blocks (DeepSeek-R1)
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
        if "</think>" in answer:
            answer = answer.split("</think>", 1)[-1]

        # Strip markdown fences if present
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

        proposals = json.loads(answer)
        if not isinstance(proposals, list):
            return []

        # Validate each proposal
        valid_categories = {
            "principles_values", "cognitive_framing",
            "decision_heuristics", "preferences", "vocabulary",
        }
        result = []
        for p in proposals[:max_proposals]:
            if (
                isinstance(p, dict)
                and p.get("text")
                and p.get("category") in valid_categories
                and p.get("reason")
            ):
                result.append({
                    "text": str(p["text"]).strip(),
                    "category": p["category"],
                    "reason": str(p["reason"]).strip(),
                })
        return result

    except Exception as e:
        log.info("Memory proposal LLM call failed: %s", e)
        return []
