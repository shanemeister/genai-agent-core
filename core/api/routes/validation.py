"""Clinical note validation API.

Endpoints:
    POST /validate/note      — full validation report (extract → map → coverage → consistency)
    POST /validate/concepts  — extract and map concepts only (lighter)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.validation.clinical_models import (
    MappedConcept,
    NoteValidationReport,
)

log = logging.getLogger("noesis.validation")

router = APIRouter(prefix="/validate", tags=["validation"])


# ── Request models ────────────────────────────────────────────────────────

class NoteValidationRequest(BaseModel):
    note_text: str = Field(..., min_length=10, description="Clinical note text")
    include_icd10: bool = Field(True, description="Include ICD-10 code suggestions")


class ConceptExtractionRequest(BaseModel):
    note_text: str = Field(..., min_length=10, description="Clinical note text")


# ── Full validation ──────────────────────────────────────────────────────

@router.post("/note", response_model=NoteValidationReport)
async def validate_note(req: NoteValidationRequest):
    """Validate a clinical note against SNOMED CT.

    Pipeline:
      1. Extract clinical concepts from note text (LLM)
      2. Map concepts to SNOMED CT codes (embedding similarity)
      3. Check ontology coverage (graph traversal)
      4. Check consistency (contradiction detection)
      5. Check specificity (CDI rules engine)
      6. Assemble validation report
    """
    try:
        from core.validation.clinical_concept_extractor import extract_clinical_concepts
        from core.validation.snomed_mapper import map_concepts_to_snomed
        from core.validation.coverage_checker import check_coverage
        from core.validation.consistency_checker import check_consistency

        # Step 1: Extract clinical concepts
        log.info("Extracting clinical concepts from note (%d chars)", len(req.note_text))
        concepts = await extract_clinical_concepts(req.note_text)
        if not concepts:
            return NoteValidationReport(
                note_hash=_hash_note(req.note_text),
                summary="No clinical concepts could be extracted from this note.",
                validated_at=_now_iso(),
            )

        log.info("Extracted %d clinical concepts", len(concepts))

        # Step 2: Map to SNOMED CT
        mapped = await map_concepts_to_snomed(concepts)
        mapped_count = sum(1 for m in mapped if m.sctid or m.rxcui)
        log.info("Mapped %d/%d concepts to ontology", mapped_count, len(concepts))

        # Step 3: Coverage check
        coverage = await check_coverage(mapped)

        # Step 4: Consistency check
        consistency = await check_consistency(mapped)

        # Step 5: Specificity check (CDI rules engine)
        from core.validation.specificity_rules import check_specificity
        specificity = await check_specificity(mapped, req.note_text)

        # Step 6: Collect ICD-10 suggestions
        icd10_suggestions: list[dict] = []
        if req.include_icd10:
            for mc in mapped:
                if mc.icd10_codes and mc.confidence >= 0.3:
                    icd10_suggestions.append({
                        "concept": mc.concept.term,
                        "snomed_code": mc.sctid,
                        "icd10_codes": mc.icd10_codes,
                        "confidence": mc.confidence,
                    })

        # Step 7: Compute overall score
        overall = _compute_overall_score(
            mapped_count, len(concepts),
            coverage.coverage_score,
            consistency.is_consistent,
            specificity.specificity_score,
        )

        # Step 8: Build summary
        summary = _build_summary(
            concept_count=len(concepts),
            mapped_count=mapped_count,
            coverage=coverage,
            consistency=consistency,
            specificity=specificity,
        )

        return NoteValidationReport(
            note_hash=_hash_note(req.note_text),
            extracted_concepts=concepts,
            mapped_concepts=mapped,
            coverage=coverage,
            consistency=consistency,
            specificity=specificity,
            icd10_suggestions=icd10_suggestions,
            overall_score=overall,
            summary=summary,
            validated_at=_now_iso(),
        )

    except Exception as e:
        log.error("Note validation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Concept extraction only ──────────────────────────────────────────────

@router.post("/concepts")
async def extract_and_map_concepts(req: ConceptExtractionRequest) -> list[MappedConcept]:
    """Extract clinical concepts and map to SNOMED CT (without coverage/consistency).

    Lighter endpoint for UI concept highlighting and quick lookups.
    """
    try:
        from core.validation.clinical_concept_extractor import extract_clinical_concepts
        from core.validation.snomed_mapper import map_concepts_to_snomed

        concepts = await extract_clinical_concepts(req.note_text)
        if not concepts:
            return []

        return await map_concepts_to_snomed(concepts)

    except Exception as e:
        log.error("Concept extraction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Helpers ──────────────────────────────────────────────────────────────

def _hash_note(text: str) -> str:
    """SHA256 hash of note text for audit trail."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _compute_overall_score(
    mapped_count: int,
    total_concepts: int,
    coverage_score: float,
    is_consistent: bool,
    specificity_score: float = 1.0,
) -> float:
    """Composite quality score from mapping, coverage, consistency, and specificity."""
    if total_concepts == 0:
        return 0.0

    mapping_score = mapped_count / total_concepts
    consistency_score = 1.0 if is_consistent else 0.5

    # Weighted composite (4 components)
    overall = (
        0.25 * mapping_score
        + 0.25 * coverage_score
        + 0.25 * consistency_score
        + 0.25 * specificity_score
    )
    return round(overall, 3)


def _build_summary(
    concept_count: int,
    mapped_count: int,
    coverage,
    consistency,
    specificity=None,
) -> str:
    """Build a human-readable validation summary."""
    parts = []

    # Mapping summary
    parts.append(f"{mapped_count}/{concept_count} concepts mapped to SNOMED CT / RxNorm")

    # Coverage summary
    pct = round(coverage.coverage_score * 100)
    parts.append(f"{pct}% ontology coverage")

    high_missing = [m for m in coverage.missing_concepts if m.priority == "high"]
    if high_missing:
        terms = ", ".join(m.term for m in high_missing[:3])
        parts.append(f"{len(high_missing)} high-priority gaps: {terms}")

    # Consistency summary
    if consistency.is_consistent:
        parts.append("no consistency issues detected")
    else:
        error_count = sum(1 for f in consistency.flags if f.severity == "error")
        warning_count = sum(1 for f in consistency.flags if f.severity == "warning")
        if error_count:
            parts.append(f"{error_count} consistency error(s)")
        if warning_count:
            parts.append(f"{warning_count} consistency warning(s)")

    # Specificity summary
    if specificity and specificity.total_conditions_checked > 0:
        spec_pct = round(specificity.specificity_score * 100)
        parts.append(f"{spec_pct}% documentation specificity ({specificity.fully_specified_count}/{specificity.total_conditions_checked} conditions)")
        if specificity.queries_generated > 0:
            conditions = ", ".join(g.condition for g in specificity.gaps[:3])
            parts.append(f"{specificity.queries_generated} CDI query(ies) generated: {conditions}")

    return ". ".join(parts) + "."
