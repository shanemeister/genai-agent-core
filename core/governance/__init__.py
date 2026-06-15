"""Shared governance primitives for registry-conformant agents.

The registry under ``registry/`` is the source of truth (see
``governance/REGISTRY_CONTRACT.md``). Each agent manifest declares a
``grounding_threshold``, a ``refusal`` block, ``telemetry`` it emits, and an
``escalation_to`` target. These primitives let an existing worker module honor
that contract without re-implementing the plumbing per agent:

- :class:`RefusalSpan` — the structured refusal output an agent emits when its
  confidence falls below the manifest threshold (instead of silently dropping a
  low-confidence result). Carries the escalation target so the caller can route
  to the human-approval path.
- :func:`governance_span` — a context manager that emits one PHI-free telemetry
  span per agent invocation with the manifest's declared attributes
  (``agent_id``, ``confidence``, ``refusal_flag``, ``tool_calls``, ``latency``).

Telemetry must never break the pipeline it observes (same rule as
``core.db.llm_logger``): emission failures are swallowed and logged.
"""

from core.governance.refusal import RefusalSpan, should_refuse
from core.governance.telemetry import governance_span

__all__ = ["RefusalSpan", "should_refuse", "governance_span"]
