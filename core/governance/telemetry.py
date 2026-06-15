"""PHI-free telemetry spans for registry-conformant agents.

Each agent manifest declares a ``telemetry.emits`` block — e.g. ``snomed.map``
with attributes ``agent_id``, ``confidence``, ``refusal_flag``, ``tool_calls``,
``latency``. :func:`governance_span` emits exactly one such span per agent
invocation, with those attribute names, and nothing that could carry PHI.

Latency is measured by the context manager. ``confidence``, ``refusal_flag``,
and ``tool_calls`` are recorded by the agent through the yielded span object
during the invocation. Emission failures never propagate — observability must
not break the pipeline it observes (same rule as ``core.db.llm_logger``).

Latency uses :func:`time.monotonic`, not wall-clock, so it is unaffected by the
clock-skew tooling restrictions elsewhere in the harness.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Iterator

log = logging.getLogger("noesis.observability")


class _Span:
    """Mutable accumulator for the attributes an agent records mid-invocation."""

    def __init__(self, name: str, agent_id: str) -> None:
        self.name = name
        self.agent_id = agent_id
        self.confidence: float | None = None
        self.refusal_flag: bool = False
        self.tool_calls: int = 0

    def record(
        self,
        *,
        confidence: float | None = None,
        refusal_flag: bool | None = None,
    ) -> None:
        """Record the per-invocation outcome attributes."""
        if confidence is not None:
            self.confidence = confidence
        if refusal_flag is not None:
            self.refusal_flag = refusal_flag

    def add_tool_call(self, n: int = 1) -> None:
        """Increment the tool-call count for this invocation."""
        self.tool_calls += n


@contextmanager
def governance_span(name: str, agent_id: str) -> Iterator[_Span]:
    """Emit one PHI-free telemetry span for an agent invocation.

    Usage::

        with governance_span("snomed.map", agent_id="snomed-mapper") as span:
            span.add_tool_call()              # per ontology lookup
            span.record(confidence=0.82, refusal_flag=False)

    The span is emitted once on exit, carrying the manifest's declared
    attributes (``agent_id``, ``confidence``, ``refusal_flag``, ``tool_calls``,
    ``latency``) plus a millisecond ``latency``. Emission never raises.
    """
    span = _Span(name, agent_id)
    start = time.monotonic()
    try:
        yield span
    finally:
        latency_ms = int((time.monotonic() - start) * 1000)
        try:
            payload = {
                "span": span.name,
                "agent_id": span.agent_id,
                "confidence": span.confidence,
                "refusal_flag": span.refusal_flag,
                "tool_calls": span.tool_calls,
                "latency_ms": latency_ms,
            }
            # PHI-free by construction: no term text, no note text, no patient
            # data ever enters this payload.
            log.info("governance.span %s", json.dumps(payload))
        except Exception as e:  # observability must never break the pipeline
            log.warning("Failed to emit governance span (%s): %s", name, e)
