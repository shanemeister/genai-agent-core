"""Refusal as a first-class output path.

When an agent's confidence falls below its manifest ``grounding_threshold`` it
must REFUSE — emit a structured refusal and escalate to the human-approval path
declared in ``escalation_to`` — rather than silently emit a low-confidence
result. This module gives every conformant agent the same refusal shape.

PHI safety: a :class:`RefusalSpan` carries no note text and no patient data.
``subject`` is an opaque, non-PHI label for *what* was refused (e.g. a concept
category or an ordinal), never the clinical term itself.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RefusalSpan(BaseModel):
    """A structured refusal emitted in place of a below-threshold result.

    Mirrors the manifest's ``refusal.emits: refusal-span`` declaration. The
    caller routes a refusal to ``escalation_to`` (the human-approval control)
    instead of treating the absence of a mapping as a clean negative.
    """

    agent_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    reason: str
    escalation_to: str
    subject: str | None = None  # opaque, non-PHI label for what was refused


def should_refuse(confidence: float | None, threshold: float) -> bool:
    """True when ``confidence`` is missing or below the manifest threshold.

    A missing confidence (no candidate mapping at all) is treated as a refusal:
    the agent could not ground the input, which is exactly what the human should
    review.
    """
    if confidence is None:
        return True
    return confidence < threshold
