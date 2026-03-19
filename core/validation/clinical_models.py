"""Pydantic models for clinical note validation.

These models represent the full pipeline:
  ClinicalConcept → MappedConcept → CoverageResult / ConsistencyResult → NoteValidationReport
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class ConceptCategory(str, Enum):
    CONDITION = "condition"
    SYMPTOM = "symptom"
    PROCEDURE = "procedure"
    MEDICATION = "medication"
    BODY_SITE = "body_site"
    LAB_VALUE = "lab_value"


class ClinicalConcept(BaseModel):
    """A clinical concept extracted from note text."""
    term: str
    category: ConceptCategory
    negated: bool = False
    qualifier: str | None = None


class MappedConcept(BaseModel):
    """A clinical concept mapped to a SNOMED CT or RxNorm code."""
    concept: ClinicalConcept
    sctid: str | None = None
    snomed_term: str | None = None
    semantic_tag: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    icd10_codes: List[str] = Field(default_factory=list)
    rxcui: str | None = None
    source_ontology: str = "snomed"  # "snomed" or "rxnorm"


class MissingConcept(BaseModel):
    """An expected ontology concept not found in the note."""
    term: str
    sctid: str
    semantic_tag: str
    relationship: str
    source_concept: str
    priority: str = "medium"


class ConsistencyFlag(BaseModel):
    """A consistency issue detected in the note."""
    type: str
    severity: str
    description: str
    involved_concepts: List[str] = Field(default_factory=list)


class CoverageResult(BaseModel):
    """Result of checking ontology coverage for documented concepts."""
    matched_concepts: List[MappedConcept] = Field(default_factory=list)
    missing_concepts: List[MissingConcept] = Field(default_factory=list)
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    suggestions: List[str] = Field(default_factory=list)


class ConsistencyResult(BaseModel):
    """Result of checking consistency between documented concepts."""
    flags: List[ConsistencyFlag] = Field(default_factory=list)
    is_consistent: bool = True


class SpecificityGap(BaseModel):
    """A detected specificity gap in clinical documentation."""
    rule_id: str
    condition: str
    mapped_concept: str
    sctid: str | None = None
    missing_attributes: List[str] = Field(default_factory=list)
    present_attributes: dict[str, str] = Field(default_factory=dict)
    current_icd10: List[str] = Field(default_factory=list)
    potential_icd10: dict[str, str] = Field(default_factory=dict)
    query_text: str = ""
    priority: str = "medium"
    impact: str = ""


class SpecificityResult(BaseModel):
    """Result of specificity analysis for the full note."""
    gaps: List[SpecificityGap] = Field(default_factory=list)
    specificity_score: float = Field(default=1.0, ge=0.0, le=1.0)
    total_conditions_checked: int = 0
    fully_specified_count: int = 0
    queries_generated: int = 0


class NoteValidationReport(BaseModel):
    """Full validation report for a clinical note."""
    note_hash: str
    ontology_version: str = "SNOMED CT US 20250901"
    extracted_concepts: List[ClinicalConcept] = Field(default_factory=list)
    mapped_concepts: List[MappedConcept] = Field(default_factory=list)
    coverage: CoverageResult = Field(default_factory=CoverageResult)
    consistency: ConsistencyResult = Field(default_factory=ConsistencyResult)
    specificity: SpecificityResult = Field(default_factory=SpecificityResult)
    icd10_suggestions: List[dict] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""
    validated_at: str = ""
