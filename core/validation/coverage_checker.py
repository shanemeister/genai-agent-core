"""Check ontology coverage for documented clinical concepts.

For each condition/diagnosis in the note, traverses the SNOMED CT graph
to find expected related concepts (finding site, morphology, etc.) and
checks whether they are documented.
"""

from __future__ import annotations

import logging

from core.graph.neo4j_client import get_session
from core.validation.clinical_models import (
    ConceptCategory,
    CoverageResult,
    MappedConcept,
    MissingConcept,
)

log = logging.getLogger("noesis.validation")

# Relationship types that indicate expected documentation, with priority
_EXPECTED_RELATIONSHIPS: dict[str, str] = {
    "FINDING_SITE": "high",
    "ASSOCIATED_MORPHOLOGY": "high",
    "CAUSATIVE_AGENT": "medium",
    "PATHOLOGICAL_PROCESS": "medium",
    "METHOD": "low",
    "INTERPRETS": "low",
    "DUE_TO": "medium",
    "AFTER": "low",
    "DIRECT_MORPHOLOGY": "high",
    "PROCEDURE_SITE_DIRECT": "medium",
}

# Categories that should trigger coverage checking
_CHECKABLE_CATEGORIES = {
    ConceptCategory.CONDITION,
    ConceptCategory.PROCEDURE,
}

# Minimum mapping confidence to include in coverage checking
_MIN_CONFIDENCE = 0.3


async def check_coverage(
    mapped_concepts: list[MappedConcept],
) -> CoverageResult:
    """Check ontology coverage for documented concepts.

    For each condition or procedure with a SNOMED mapping, queries the
    ontology graph for expected related concepts and checks whether
    they appear among the note's other mapped concepts.

    Args:
        mapped_concepts: Concepts already mapped to SNOMED CT.

    Returns:
        CoverageResult with matched/missing concepts, score, suggestions.
    """
    # Build set of all SCTIDs documented in the note
    documented_sctids: set[str] = set()
    documented_terms: set[str] = set()
    for mc in mapped_concepts:
        if mc.sctid and mc.confidence >= _MIN_CONFIDENCE:
            documented_sctids.add(mc.sctid)
            documented_terms.add(mc.concept.term.lower())
            if mc.snomed_term:
                documented_terms.add(mc.snomed_term.lower())

    # Concepts to check coverage for
    checkable = [
        mc for mc in mapped_concepts
        if mc.sctid
        and mc.confidence >= _MIN_CONFIDENCE
        and mc.concept.category in _CHECKABLE_CATEGORIES
        and not mc.concept.negated
    ]

    if not checkable:
        return CoverageResult(
            matched_concepts=mapped_concepts,
            coverage_score=1.0,
        )

    missing: list[MissingConcept] = []
    expected_count = 0
    present_count = 0

    for mc in checkable:
        expected = await _get_expected_concepts(mc.sctid)

        for exp in expected:
            expected_count += 1
            target_sctid = exp["target_sctid"]
            target_term = exp["target_term"].lower() if exp["target_term"] else ""

            # Check if this expected concept is documented
            if target_sctid in documented_sctids:
                present_count += 1
            elif target_term and target_term in documented_terms:
                present_count += 1
            else:
                rel_type = exp["rel_type"]
                priority = _EXPECTED_RELATIONSHIPS.get(rel_type, "low")
                missing.append(
                    MissingConcept(
                        term=exp["target_term"] or f"SCTID:{target_sctid}",
                        sctid=target_sctid,
                        semantic_tag=exp.get("target_tag", ""),
                        relationship=rel_type,
                        source_concept=mc.concept.term,
                        priority=priority,
                    )
                )

    # Compute coverage score
    total = expected_count
    if total > 0:
        coverage_score = round(present_count / total, 3)
    else:
        coverage_score = 1.0

    # Generate suggestions from high-priority missing concepts
    suggestions: list[str] = []
    high_missing = [m for m in missing if m.priority == "high"]
    medium_missing = [m for m in missing if m.priority == "medium"]

    for m in high_missing[:5]:
        suggestions.append(
            f"Consider documenting {m.relationship.lower().replace('_', ' ')} "
            f"for {m.source_concept}: {m.term}"
        )
    for m in medium_missing[:3]:
        suggestions.append(
            f"{m.source_concept} may need {m.relationship.lower().replace('_', ' ')}: {m.term}"
        )

    # Sort missing by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    missing.sort(key=lambda m: priority_order.get(m.priority, 3))

    log.info(
        "Coverage check: %d/%d expected concepts documented (%.0f%%), %d missing",
        present_count, total, coverage_score * 100, len(missing),
    )

    return CoverageResult(
        matched_concepts=mapped_concepts,
        missing_concepts=missing,
        coverage_score=coverage_score,
        suggestions=suggestions,
    )


async def _get_expected_concepts(sctid: str) -> list[dict]:
    """Query Neo4j for concepts expected to be documented alongside this concept.

    Returns list of dicts with: rel_type, target_sctid, target_term, target_tag
    """
    rel_types = list(_EXPECTED_RELATIONSHIPS.keys())

    try:
        async with get_session() as session:
            # Build WHERE clause for relationship types
            # We query outgoing relationships from the concept
            result = await session.run(
                """
                MATCH (s:SnomedConcept {sctid: $sctid})-[r]->(t:SnomedConcept)
                WHERE type(r) IN $rel_types
                RETURN type(r) AS rel_type,
                       t.sctid AS target_sctid,
                       t.preferred_term AS target_term,
                       t.semantic_tag AS target_tag
                """,
                sctid=sctid,
                rel_types=rel_types,
            )
            return [record.data() async for record in result]

    except Exception as e:
        log.warning("Failed to get expected concepts for %s: %s", sctid, e)
        return []
