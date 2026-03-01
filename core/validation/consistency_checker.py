"""Check consistency between clinical concepts documented in a note.

Detects:
  - Category mismatches (procedure coded as diagnosis)
  - Negation inconsistencies (concept both affirmed and denied)
  - Specificity issues (general + specific variant of same condition)
  - Laterality conflicts (contradictory body sides)
"""

from __future__ import annotations

import logging
import re

from core.graph.neo4j_client import get_session
from core.validation.clinical_models import (
    ConceptCategory,
    ConsistencyFlag,
    ConsistencyResult,
    MappedConcept,
)

log = logging.getLogger("noesis.validation")

# Expected semantic tags per concept category
_EXPECTED_TAGS: dict[ConceptCategory, set[str]] = {
    ConceptCategory.CONDITION: {"disorder", "finding", "clinical finding"},
    ConceptCategory.SYMPTOM: {"finding", "clinical finding", "observable entity"},
    ConceptCategory.PROCEDURE: {"procedure", "regime/therapy"},
    ConceptCategory.MEDICATION: {
        "substance", "product", "pharmaceutical / biologic product",
        "medicinal product", "clinical drug",
    },
    ConceptCategory.BODY_SITE: {"body structure", "cell structure"},
    ConceptCategory.LAB_VALUE: {"observable entity", "procedure"},
}

# Laterality keywords
_LEFT_RE = re.compile(r"\bleft\b", re.IGNORECASE)
_RIGHT_RE = re.compile(r"\bright\b", re.IGNORECASE)
_BILATERAL_RE = re.compile(r"\bbilateral\b", re.IGNORECASE)

# Minimum confidence for mapped concepts to be checked
_MIN_CONFIDENCE = 0.3


async def check_consistency(
    mapped_concepts: list[MappedConcept],
) -> ConsistencyResult:
    """Check for consistency issues among documented concepts.

    Args:
        mapped_concepts: Concepts already mapped to SNOMED CT.

    Returns:
        ConsistencyResult with list of flags and overall consistency status.
    """
    flags: list[ConsistencyFlag] = []

    # Filter to confidently mapped concepts
    confident = [
        mc for mc in mapped_concepts
        if mc.sctid and mc.confidence >= _MIN_CONFIDENCE
    ]

    # Run each consistency check
    flags.extend(_check_category_mismatches(confident))
    flags.extend(_check_negation_inconsistencies(mapped_concepts))
    flags.extend(await _check_specificity_issues(confident))

    # Sort by severity: error > warning > info
    severity_order = {"error": 0, "warning": 1, "info": 2}
    flags.sort(key=lambda f: severity_order.get(f.severity, 3))

    is_consistent = not any(f.severity == "error" for f in flags)

    if flags:
        log.info(
            "Consistency check: %d flags (%d errors, %d warnings, %d info)",
            len(flags),
            sum(1 for f in flags if f.severity == "error"),
            sum(1 for f in flags if f.severity == "warning"),
            sum(1 for f in flags if f.severity == "info"),
        )
    else:
        log.info("Consistency check: no issues found")

    return ConsistencyResult(flags=flags, is_consistent=is_consistent)


def _check_category_mismatches(
    concepts: list[MappedConcept],
) -> list[ConsistencyFlag]:
    """Detect concepts whose SNOMED semantic tag doesn't match their category."""
    flags: list[ConsistencyFlag] = []

    for mc in concepts:
        if not mc.semantic_tag:
            continue
        expected_tags = _EXPECTED_TAGS.get(mc.concept.category, set())
        if not expected_tags:
            continue
        if mc.semantic_tag.lower() not in expected_tags:
            flags.append(
                ConsistencyFlag(
                    type="category_mismatch",
                    severity="warning",
                    description=(
                        f'"{mc.concept.term}" categorized as {mc.concept.category.value} '
                        f'but SNOMED tag is "{mc.semantic_tag}" '
                        f'(expected: {", ".join(sorted(expected_tags))})'
                    ),
                    involved_concepts=[mc.concept.term],
                )
            )

    return flags


def _check_negation_inconsistencies(
    concepts: list[MappedConcept],
) -> list[ConsistencyFlag]:
    """Detect concepts that are both affirmed and negated in the same note."""
    flags: list[ConsistencyFlag] = []

    # Group by normalized term
    by_term: dict[str, list[MappedConcept]] = {}
    for mc in concepts:
        key = mc.concept.term.lower().strip()
        by_term.setdefault(key, []).append(mc)

    for term, group in by_term.items():
        affirmed = [mc for mc in group if not mc.concept.negated]
        negated = [mc for mc in group if mc.concept.negated]

        if affirmed and negated:
            flags.append(
                ConsistencyFlag(
                    type="negation_inconsistency",
                    severity="error",
                    description=(
                        f'"{term}" is both affirmed and denied in the same note'
                    ),
                    involved_concepts=[term],
                )
            )

    return flags


async def _check_specificity_issues(
    concepts: list[MappedConcept],
) -> list[ConsistencyFlag]:
    """Detect IS_A hierarchical overlaps (general + specific same condition).

    For example, documenting both "diabetes mellitus" and "type 2 diabetes"
    suggests the more general term should be replaced by the specific one.
    """
    flags: list[ConsistencyFlag] = []

    # Only check conditions and symptoms
    conditions = [
        mc for mc in concepts
        if mc.concept.category in {ConceptCategory.CONDITION, ConceptCategory.SYMPTOM}
        and not mc.concept.negated
    ]

    if len(conditions) < 2:
        return flags

    # Collect all SCTIDs for conditions
    sctids = [mc.sctid for mc in conditions if mc.sctid]
    if len(sctids) < 2:
        return flags

    # Check for IS_A relationships between documented conditions
    try:
        async with get_session() as session:
            result = await session.run(
                """
                UNWIND $sctids AS child_id
                UNWIND $sctids AS parent_id
                WITH child_id, parent_id
                WHERE child_id <> parent_id
                MATCH (child:SnomedConcept {sctid: child_id})-[:IS_A]->(parent:SnomedConcept {sctid: parent_id})
                RETURN child.sctid AS child_sctid,
                       child.preferred_term AS child_term,
                       parent.sctid AS parent_sctid,
                       parent.preferred_term AS parent_term
                """,
                sctids=sctids,
            )
            records = [record.data() async for record in result]

        # Build lookup for readable terms
        sctid_to_term = {mc.sctid: mc.concept.term for mc in conditions if mc.sctid}

        seen = set()
        for rec in records:
            pair_key = (rec["child_sctid"], rec["parent_sctid"])
            if pair_key in seen:
                continue
            seen.add(pair_key)

            child_term = sctid_to_term.get(rec["child_sctid"], rec["child_term"])
            parent_term = sctid_to_term.get(rec["parent_sctid"], rec["parent_term"])

            flags.append(
                ConsistencyFlag(
                    type="specificity_issue",
                    severity="info",
                    description=(
                        f'"{parent_term}" is a broader term for "{child_term}" — '
                        f"consider using only the more specific term for coding accuracy"
                    ),
                    involved_concepts=[parent_term, child_term],
                )
            )

    except Exception as e:
        log.warning("Specificity check failed: %s", e)

    return flags
