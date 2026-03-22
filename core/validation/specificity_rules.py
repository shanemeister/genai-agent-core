"""CDI Specificity Rules Engine.

Evaluates mapped clinical concepts against ICD-10 specificity requirements
and generates CDI queries when documentation gaps are found.

Three-layer architecture:
  Layer 1 — Base rules (ICD-10/CMS guidelines, ships with system)
  Layer 2 — Specialty rules (enabled per facility)
  Layer 3 — Hospital-specific rules (full CRUD via API)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from core.graph.neo4j_client import get_session
from core.validation.clinical_models import (
    ConceptCategory,
    MappedConcept,
    SpecificityGap,
    SpecificityResult,
)

log = logging.getLogger("noesis.validation")

# ── Rule data models ────────────────────────────────────────────────────

class SpecificityAttribute(BaseModel):
    """A single specificity dimension required for a condition."""
    name: str
    description: str
    required: bool = True
    evidence_patterns: list[str] = Field(default_factory=list)
    snomed_descendants: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)


class SpecificityRule(BaseModel):
    """A specificity rule for a clinical condition."""
    rule_id: str
    condition: str
    snomed_codes: list[str] = Field(default_factory=list)
    icd10_unspecified: list[str] = Field(default_factory=list)
    icd10_specific: dict[str, str] = Field(default_factory=dict)
    attributes: list[SpecificityAttribute] = Field(default_factory=list)
    query_template: str = ""
    priority: str = "high"
    layer: int = 1


# ── Module state ────────────────────────────────────────────────────────

_rules: list[SpecificityRule] = []
_rules_loaded = False

# Base rules directory (relative to this file)
_BASE_RULES_DIR = Path(__file__).parent / "rules" / "base"
_HOSPITAL_RULES_DIR = Path(__file__).parent / "rules" / "hospital"


# ── Rule loading ────────────────────────────────────────────────────────

def load_rules(
    base_dir: Path | None = None,
    hospital_dir: Path | None = None,
) -> list[SpecificityRule]:
    """Load specificity rules from JSON files.

    Loads Layer 1 (base) rules, then optionally merges Layer 3 (hospital)
    rules. Later layers override earlier ones for the same rule_id.

    Args:
        base_dir: Directory containing base rule JSON files.
        hospital_dir: Optional directory with hospital-specific rules.

    Returns:
        List of all loaded SpecificityRule objects.
    """
    global _rules, _rules_loaded

    base_dir = base_dir or _BASE_RULES_DIR
    rules_by_id: dict[str, SpecificityRule] = {}

    # Layer 1: Base rules
    if base_dir.is_dir():
        for json_file in sorted(base_dir.glob("*.json")):
            try:
                raw = json.loads(json_file.read_text())
                if not isinstance(raw, list):
                    raw = [raw]
                for item in raw:
                    rule = SpecificityRule(**item)
                    rule.layer = 1
                    rules_by_id[rule.rule_id] = rule
            except Exception as e:
                log.warning("Failed to load rule file %s: %s", json_file.name, e)

    # Layer 3: Hospital-specific rules (override by rule_id)
    if hospital_dir and hospital_dir.is_dir():
        for json_file in sorted(hospital_dir.glob("*.json")):
            try:
                raw = json.loads(json_file.read_text())
                if not isinstance(raw, list):
                    raw = [raw]
                for item in raw:
                    rule = SpecificityRule(**item)
                    rule.layer = 3
                    rules_by_id[rule.rule_id] = rule
            except Exception as e:
                log.warning("Failed to load hospital rule %s: %s", json_file.name, e)

    _rules = list(rules_by_id.values())
    _rules_loaded = True

    log.info("Loaded %d specificity rules (%d base, %d hospital)",
             len(_rules),
             sum(1 for r in _rules if r.layer == 1),
             sum(1 for r in _rules if r.layer == 3))

    return _rules


def get_rules() -> list[SpecificityRule]:
    """Get all loaded rules, loading from disk if needed."""
    if not _rules_loaded:
        load_rules(hospital_dir=_HOSPITAL_RULES_DIR)
    return _rules


def get_rule(rule_id: str) -> SpecificityRule | None:
    """Get a specific rule by ID."""
    for rule in get_rules():
        if rule.rule_id == rule_id:
            return rule
    return None


def save_rule(rule: SpecificityRule) -> SpecificityRule:
    """Save a rule to the hospital rules directory (Layer 3).

    Creates or overwrites a JSON file named after the rule_id.
    Then reloads all rules so the in-memory cache is updated.
    """
    _HOSPITAL_RULES_DIR.mkdir(parents=True, exist_ok=True)

    rule.layer = 3
    filename = rule.rule_id.lower().replace(" ", "_") + ".json"
    filepath = _HOSPITAL_RULES_DIR / filename

    filepath.write_text(json.dumps([rule.dict()], indent=2))
    log.info("Saved hospital rule %s to %s", rule.rule_id, filepath.name)

    # Reload all rules to pick up the change
    load_rules(hospital_dir=_HOSPITAL_RULES_DIR)
    return rule


def delete_rule(rule_id: str) -> bool:
    """Delete a hospital rule (Layer 3 only).

    Base rules (Layer 1) cannot be deleted through the API.
    Returns True if deleted, False if not found or not deletable.
    """
    existing = get_rule(rule_id)
    if not existing:
        return False

    if existing.layer == 1:
        # Cannot delete base rules — only override via Layer 3
        return False

    # Find and remove the file
    if _HOSPITAL_RULES_DIR.is_dir():
        for json_file in _HOSPITAL_RULES_DIR.glob("*.json"):
            try:
                raw = json.loads(json_file.read_text())
                if not isinstance(raw, list):
                    raw = [raw]
                # Check if this file contains the rule
                if any(item.get("rule_id") == rule_id for item in raw):
                    remaining = [item for item in raw if item.get("rule_id") != rule_id]
                    if remaining:
                        json_file.write_text(json.dumps(remaining, indent=2))
                    else:
                        json_file.unlink()
                    log.info("Deleted hospital rule %s", rule_id)
                    load_rules(hospital_dir=_HOSPITAL_RULES_DIR)
                    return True
            except Exception as e:
                log.warning("Error checking %s for deletion: %s", json_file.name, e)

    return False


# ── Main specificity check ──────────────────────────────────────────────

async def check_specificity(
    mapped_concepts: list[MappedConcept],
    note_text: str,
) -> SpecificityResult:
    """Check specificity of documented conditions against ICD-10 requirements.

    For each mapped condition with a SNOMED code, finds matching specificity
    rules and checks the note text for evidence of required attributes.
    Generates CDI queries for any gaps found.

    Args:
        mapped_concepts: Concepts mapped to SNOMED CT / RxNorm.
        note_text: Original clinical note text.

    Returns:
        SpecificityResult with gaps, score, and query count.
    """
    rules = get_rules()
    if not rules:
        return SpecificityResult()

    # Filter to conditions/diagnoses with SNOMED codes
    checkable = [
        mc for mc in mapped_concepts
        if mc.sctid
        and mc.concept.category in {ConceptCategory.CONDITION, ConceptCategory.SYMPTOM}
        and not mc.concept.negated
        and mc.confidence >= 0.5
    ]

    if not checkable:
        return SpecificityResult()

    gaps: list[SpecificityGap] = []
    total_checked = 0
    fully_specified = 0
    note_lower = note_text.lower()
    # Track which rules have already been evaluated to avoid duplicates
    # (e.g., "pneumonia" extracted twice should only generate one CDI query)
    evaluated_rules: set[str] = set()

    for mc in checkable:
        matching_rules = await _find_matching_rules(mc.sctid, mc.concept.term, rules)

        for rule in matching_rules:
            if rule.rule_id in evaluated_rules:
                continue
            evaluated_rules.add(rule.rule_id)

            total_checked += 1
            present, missing = _check_evidence(rule, note_lower, mapped_concepts, mc)

            if not missing:
                fully_specified += 1
                continue

            # Build gap
            query_text = _generate_query(rule, missing, present, mc.concept.term)

            gaps.append(SpecificityGap(
                rule_id=rule.rule_id,
                condition=rule.condition,
                mapped_concept=mc.concept.term,
                sctid=mc.sctid,
                missing_attributes=[attr.name for attr in missing],
                present_attributes={
                    name: value for name, value in present.items()
                },
                current_icd10=rule.icd10_unspecified,
                potential_icd10=rule.icd10_specific,
                query_text=query_text,
                priority=rule.priority,
                impact=_assess_impact(rule),
            ))

    # Compute specificity score
    if total_checked > 0:
        specificity_score = round(fully_specified / total_checked, 3)
    else:
        specificity_score = 1.0

    queries_generated = len([g for g in gaps if g.query_text])

    if gaps:
        log.info(
            "Specificity check: %d/%d conditions fully specified, %d gaps, %d queries",
            fully_specified, total_checked, len(gaps), queries_generated,
        )
    else:
        log.info("Specificity check: all %d conditions fully specified", total_checked)

    return SpecificityResult(
        gaps=gaps,
        specificity_score=specificity_score,
        total_conditions_checked=total_checked,
        fully_specified_count=fully_specified,
        queries_generated=queries_generated,
    )


# ── Rule matching ───────────────────────────────────────────────────────

async def _find_matching_rules(
    sctid: str,
    concept_term: str,
    rules: list[SpecificityRule],
) -> list[SpecificityRule]:
    """Find specificity rules that apply to a given concept.

    Matching strategy:
      1. Direct SCTID match against rule's snomed_codes
      2. IS_A ancestor traversal (2 hops) in Neo4j
      3. Term-based match (condition name appears in concept term)
    """
    matched: list[SpecificityRule] = []
    term_lower = concept_term.lower()

    # 1. Direct SCTID match
    for rule in rules:
        if sctid in rule.snomed_codes:
            matched.append(rule)

    if matched:
        return matched

    # 2. Ancestor traversal
    ancestor_sctids = await _get_ancestor_sctids(sctid, max_hops=2)
    if ancestor_sctids:
        for rule in rules:
            if any(anc in rule.snomed_codes for anc in ancestor_sctids):
                matched.append(rule)

    if matched:
        return matched

    # 3. Term-based fallback
    for rule in rules:
        rule_condition = rule.condition.lower()
        # Check if the rule's condition name appears in the concept term
        # or the concept term appears in the rule's condition name
        if (rule_condition in term_lower
                or term_lower in rule_condition
                or _terms_overlap(rule_condition, term_lower)):
            matched.append(rule)

    return matched


def _terms_overlap(rule_condition: str, concept_term: str) -> bool:
    """Check if rule condition and concept term share significant words.

    Requires that the majority of the rule's significant words appear
    in the concept term to avoid false matches like "acute exacerbation
    of COPD" matching "acute kidney injury" (only "acute" overlaps).
    """
    stopwords = {"of", "the", "a", "an", "in", "on", "with", "and", "or",
                 "to", "for", "acute", "chronic", "type", "unspecified"}
    rule_words = set(rule_condition.split()) - stopwords
    concept_words = set(concept_term.split()) - stopwords
    # Only keep significant words (>= 4 chars)
    rule_sig = {w for w in rule_words if len(w) >= 4}
    if not rule_sig:
        return False
    shared = rule_sig & concept_words
    # Require at least half of the rule's significant words to match
    return len(shared) >= max(1, len(rule_sig) // 2 + 1)


async def _get_ancestor_sctids(sctid: str, max_hops: int = 2) -> set[str]:
    """Query Neo4j for IS_A ancestors up to max_hops."""
    try:
        async with get_session() as session:
            result = await session.run(
                f"""
                MATCH (s:SnomedConcept {{sctid: $sctid}})-[:IS_A*1..{max_hops}]->(a:SnomedConcept)
                RETURN DISTINCT a.sctid AS sctid
                """,
                sctid=sctid,
            )
            return {record["sctid"] async for record in result}
    except Exception as e:
        log.debug("Ancestor lookup failed for %s: %s", sctid, e)
        return set()


# ── Evidence checking ───────────────────────────────────────────────────

def _check_evidence(
    rule: SpecificityRule,
    note_lower: str,
    mapped_concepts: list[MappedConcept],
    target_concept: MappedConcept,
) -> tuple[dict[str, str], list[SpecificityAttribute]]:
    """Check the note for evidence satisfying each attribute in a rule.

    Returns:
        Tuple of (present_attributes dict, missing_attributes list).
        present_attributes maps attribute name → detected value.
    """
    present: dict[str, str] = {}
    missing: list[SpecificityAttribute] = []

    for attr in rule.attributes:
        value = _find_evidence_for_attribute(
            attr, note_lower, mapped_concepts, target_concept,
        )
        if value:
            present[attr.name] = value
        elif attr.required:
            missing.append(attr)
        # Non-required attributes with no evidence are silently skipped

    return present, missing


def _find_evidence_for_attribute(
    attr: SpecificityAttribute,
    note_lower: str,
    mapped_concepts: list[MappedConcept],
    target_concept: MappedConcept,
) -> str | None:
    """Search for evidence of a single attribute in the note.

    Evidence sources (in order):
      1. Regex patterns on note text
      2. Concept qualifier field
      3. SNOMED descendants in mapped concepts
    """
    # 1. Regex patterns on note text
    for pattern in attr.evidence_patterns:
        try:
            match = re.search(pattern, note_lower)
            if match:
                # Return the matched group (captured value) or the full match
                if match.lastindex:
                    return match.group(1).strip()
                return match.group(0).strip()
        except re.error:
            log.debug("Invalid regex in rule attribute %s: %s", attr.name, pattern)

    # 2. Check the concept's qualifier (e.g., qualifier="acute")
    if target_concept.concept.qualifier and attr.values:
        qualifier_lower = target_concept.concept.qualifier.lower()
        for value in attr.values:
            if value.lower() == qualifier_lower:
                return qualifier_lower

    # 3. Check other mapped concepts for SNOMED descendants
    if attr.snomed_descendants:
        for mc in mapped_concepts:
            if mc.sctid and mc.sctid in attr.snomed_descendants and not mc.concept.negated:
                return mc.concept.term

    return None


# ── Query generation ────────────────────────────────────────────────────

def _generate_query(
    rule: SpecificityRule,
    missing_attrs: list[SpecificityAttribute],
    present_attrs: dict[str, str],
    concept_term: str,
) -> str:
    """Generate a CDI query from the rule template and missing attributes."""
    # Build the missing questions list
    questions = []
    for attr in missing_attrs:
        question = f"- {attr.description}?"
        if attr.values:
            options = ", ".join(attr.values)
            question += f" (e.g., {options})"
        questions.append(question)

    missing_questions = "\n".join(questions)

    # Build present summary
    if present_attrs:
        present_parts = [f"{k}: {v}" for k, v in present_attrs.items()]
        present_summary = "; ".join(present_parts)
    else:
        present_summary = "no specificity details found"

    # Fill template
    try:
        query = rule.query_template.format(
            condition=concept_term,
            missing_questions=missing_questions,
            present_summary=present_summary,
        )
    except (KeyError, IndexError):
        # Fallback if template has unexpected placeholders
        query = (
            f"Documentation indicates {concept_term}. "
            f"Please clarify:\n{missing_questions}\n"
            f"Currently documented: {present_summary}"
        )

    return query


# ── Impact assessment ───────────────────────────────────────────────────

def _assess_impact(rule: SpecificityRule) -> str:
    """Assess the CDI impact of a specificity gap."""
    # High-impact conditions that affect DRG/CC/MCC
    drg_impact_conditions = {
        "heart failure", "sepsis", "respiratory failure",
        "acute kidney injury", "pneumonia", "encephalopathy",
        "malnutrition", "pressure injury", "coagulopathy",
        "deep vein thrombosis", "pulmonary embolism",
        "cerebrovascular accident", "anemia",
        "alcohol dependence", "COPD",
        "hypertensive heart disease",
        "hyponatremia", "hypokalemia",
    }
    # Conditions that affect HCC risk scores
    hcc_impact_conditions = {
        "diabetes mellitus", "chronic kidney disease", "heart failure",
        "COPD", "major depressive disorder", "atrial fibrillation",
        "obesity",
    }

    condition = rule.condition.lower()
    impacts = []
    if condition in {c.lower() for c in drg_impact_conditions}:
        impacts.append("DRG impact")
    if condition in {c.lower() for c in hcc_impact_conditions}:
        impacts.append("HCC impact")
    if not impacts:
        impacts.append("specificity")

    return ", ".join(impacts)
