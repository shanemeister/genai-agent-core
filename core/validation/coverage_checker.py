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

# Relationship types that indicate expected documentation, with priority.
# HIGH + MEDIUM relationships count toward the coverage score.
# LOW relationships are reported as gaps/suggestions but don't penalise the score.
_EXPECTED_RELATIONSHIPS: dict[str, str] = {
    "FINDING_SITE": "high",
    "ASSOCIATED_MORPHOLOGY": "high",
    "DIRECT_MORPHOLOGY": "high",
    "CAUSATIVE_AGENT": "medium",
    "PATHOLOGICAL_PROCESS": "medium",
    "DUE_TO": "medium",
    "PROCEDURE_SITE_DIRECT": "medium",
    "METHOD": "low",
    "INTERPRETS": "low",
    "AFTER": "low",
}

# Only high and medium relationships affect the coverage score denominator
_SCORED_PRIORITIES = {"high", "medium"}

# Max expected concepts per checkable concept to prevent one complex
# SNOMED code from dominating the denominator
_MAX_EXPECTED_PER_CONCEPT = 5

# Categories that should trigger coverage checking
_CHECKABLE_CATEGORIES = {
    ConceptCategory.CONDITION,
    ConceptCategory.PROCEDURE,
}

# Minimum mapping confidence to include in coverage checking
_MIN_CONFIDENCE = 0.3

# Minimum stem length for word-stem matching (6 chars handles medical roots
# like "append" matching "appendicitis"/"appendectomy"/"appendix")
_MIN_STEM_LEN = 6


def _stem_match(
    target_core: str,
    target_synonyms: list[str],
    documented_text_blob: str,
    source_term: str,
) -> bool:
    """Check if target is documented via word-stem matching.

    For medical terms, shared roots like "append-" connect "appendix",
    "appendicitis", and "appendectomy". This catches implicit documentation
    that substring matching misses.
    """
    combined = documented_text_blob + " ||| " + source_term

    # Collect all candidate terms (target + synonyms)
    candidates = [target_core] + [s.lower() for s in target_synonyms]

    for term in candidates:
        words = term.split()
        # Only use significant words (>= 4 chars, skip common modifiers)
        sig_words = [
            w for w in words
            if len(w) >= 4 and w not in {"with", "from", "that", "this", "more"}
        ]
        if not sig_words:
            continue

        matched = 0
        for word in sig_words:
            if len(word) < _MIN_STEM_LEN:
                # Short words: require exact match (already checked above)
                continue
            stem = word[:_MIN_STEM_LEN]
            if stem in combined:
                matched += 1

        # Require at least one significant stem match
        if matched > 0:
            return True

    return False


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
    # Build set of all SCTIDs and terms documented in the note
    documented_sctids: set[str] = set()
    documented_terms: set[str] = set()
    # Concatenated text of all documented terms for partial matching
    documented_text_parts: list[str] = []
    for mc in mapped_concepts:
        if (mc.sctid or mc.rxcui) and mc.confidence >= _MIN_CONFIDENCE:
            if mc.sctid:
                documented_sctids.add(mc.sctid)
            term_lower = mc.concept.term.lower()
            documented_terms.add(term_lower)
            documented_text_parts.append(term_lower)
            if mc.snomed_term:
                snomed_lower = mc.snomed_term.lower()
                documented_terms.add(snomed_lower)
                documented_text_parts.append(snomed_lower)
    documented_text_blob = " ||| ".join(documented_text_parts)

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
    concept_scores: list[float] = []  # per-concept coverage for averaging

    for mc in checkable:
        expected = await _get_expected_concepts(mc.sctid)

        # Cap expected concepts per checkable concept
        scored_expected = [
            e for e in expected
            if _EXPECTED_RELATIONSHIPS.get(e["rel_type"], "low") in _SCORED_PRIORITIES
        ][:_MAX_EXPECTED_PER_CONCEPT]

        concept_present = 0
        concept_total = len(scored_expected)

        for exp in scored_expected:
            target_sctid = exp["target_sctid"]
            target_term = exp["target_term"].lower() if exp["target_term"] else ""
            # Strip SNOMED semantic tags like "(body structure)" for matching
            target_core = target_term.split("(")[0].strip() if target_term else ""

            # Check if this expected concept is documented:
            # 1. Exact SCTID match
            # 2. Exact term match
            # 3. Partial match: expected term appears within a documented term
            #    e.g., "kidney" found in "chronic kidney disease"
            # 4. Partial match: expected term found in the source concept name
            #    e.g., FINDING_SITE "kidney" for source "chronic kidney disease"
            # Also check synonyms from SNOMED (e.g., "Airways" for "Tracheobronchial tree")
            target_synonyms = [
                s.lower() for s in (exp.get("target_synonyms") or [])
            ]

            if target_sctid in documented_sctids:
                concept_present += 1
            elif target_term and target_term in documented_terms:
                concept_present += 1
            elif target_core and len(target_core) >= 3 and (
                target_core in documented_text_blob
                or target_core in mc.concept.term.lower()
            ):
                concept_present += 1
            elif any(
                syn in documented_text_blob or syn in mc.concept.term.lower()
                for syn in target_synonyms if len(syn) >= 3
            ):
                concept_present += 1
            elif _stem_match(
                target_core, target_synonyms,
                documented_text_blob, mc.concept.term.lower(),
            ):
                concept_present += 1
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

        # Per-concept score (concepts with no expected relationships get 1.0)
        if concept_total > 0:
            concept_scores.append(concept_present / concept_total)
        else:
            concept_scores.append(1.0)

        # Still report low-priority gaps as suggestions (don't count in score)
        low_expected = [
            e for e in expected
            if _EXPECTED_RELATIONSHIPS.get(e["rel_type"], "low") not in _SCORED_PRIORITIES
        ]
        for exp in low_expected:
            target_sctid = exp["target_sctid"]
            target_term = exp["target_term"].lower() if exp["target_term"] else ""
            target_core = target_term.split("(")[0].strip() if target_term else ""
            if _is_documented(
                target_sctid, target_term, target_core,
                documented_sctids, documented_terms, documented_text_blob, checkable,
                target_synonyms=exp.get("target_synonyms"),
            ):
                continue
            rel_type = exp["rel_type"]
            missing.append(
                MissingConcept(
                    term=exp["target_term"] or f"SCTID:{target_sctid}",
                    sctid=target_sctid,
                    semantic_tag=exp.get("target_tag", ""),
                    relationship=rel_type,
                    source_concept=mc.concept.term,
                    priority="low",
                )
            )

    # Average per-concept coverage scores
    if concept_scores:
        coverage_score = round(sum(concept_scores) / len(concept_scores), 3)
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
        "Coverage check: %.0f%% avg across %d concepts, %d missing",
        coverage_score * 100, len(concept_scores), len(missing),
    )

    return CoverageResult(
        matched_concepts=mapped_concepts,
        missing_concepts=missing,
        coverage_score=coverage_score,
        suggestions=suggestions,
    )


def _is_documented(
    target_sctid: str,
    target_term: str,
    target_core: str,
    documented_sctids: set[str],
    documented_terms: set[str],
    documented_text_blob: str,
    checkable: list[MappedConcept],
    target_synonyms: list[str] | None = None,
) -> bool:
    """Check if an expected concept is documented (exact, partial, or synonym match)."""
    if target_sctid in documented_sctids:
        return True
    if target_term and target_term in documented_terms:
        return True
    if target_core and len(target_core) >= 3 and (
        target_core in documented_text_blob
        or any(target_core in mc.concept.term.lower() for mc in checkable)
    ):
        return True
    # Check synonyms
    if target_synonyms:
        for syn in target_synonyms:
            syn_lower = syn.lower()
            if len(syn_lower) >= 3 and (
                syn_lower in documented_text_blob
                or any(syn_lower in mc.concept.term.lower() for mc in checkable)
            ):
                return True
    # Stem matching fallback
    source_terms = " ||| ".join(mc.concept.term.lower() for mc in checkable)
    syn_list = [s.lower() for s in (target_synonyms or [])]
    if _stem_match(target_core, syn_list, documented_text_blob, source_terms):
        return True
    return False


async def _get_expected_concepts(sctid: str) -> list[dict]:
    """Query Neo4j for concepts expected to be documented alongside this concept.

    Returns list of dicts with: rel_type, target_sctid, target_term, target_tag, target_synonyms
    """
    rel_types = list(_EXPECTED_RELATIONSHIPS.keys())

    try:
        async with get_session() as session:
            result = await session.run(
                """
                MATCH (s:SnomedConcept {sctid: $sctid})-[r]->(t:SnomedConcept)
                WHERE type(r) IN $rel_types
                RETURN type(r) AS rel_type,
                       t.sctid AS target_sctid,
                       t.preferred_term AS target_term,
                       t.semantic_tag AS target_tag,
                       t.synonyms AS target_synonyms
                """,
                sctid=sctid,
                rel_types=rel_types,
            )
            return [record.data() async for record in result]

    except Exception as e:
        log.warning("Failed to get expected concepts for %s: %s", sctid, e)
        return []
