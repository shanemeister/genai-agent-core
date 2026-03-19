"""Pydantic models for ontology import and management."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SnomedImportStats(BaseModel):
    """Tracks progress and results of a SNOMED CT import operation."""

    concepts_loaded: int = 0
    relationships_loaded: int = 0
    icd10_mappings_loaded: int = 0
    concepts_embedded: int = 0
    elapsed_seconds: float = 0.0
    status: str = "pending"  # pending | running | complete | error
    phase: str = ""  # current phase description
    error: str | None = None


class OntologyInfo(BaseModel):
    """Summary information about a loaded ontology."""

    name: str
    version: str
    edition: str
    concept_count: int = 0
    relationship_count: int = 0
    embedded_count: int = 0


class RxNormImportStats(BaseModel):
    """Tracks progress and results of an RxNorm import operation."""

    concepts_loaded: int = 0
    relationships_loaded: int = 0
    snomed_crosswalk_loaded: int = 0
    concepts_embedded: int = 0
    elapsed_seconds: float = 0.0
    status: str = "pending"  # pending | running | complete | error
    phase: str = ""  # current phase description
    error: str | None = None


class RxNormConceptDetail(BaseModel):
    """Full detail for a single RxNorm concept."""

    rxcui: str
    preferred_term: str
    tty: str = ""
    synonyms: list[str] = Field(default_factory=list)
    relationships: list[dict] = Field(default_factory=list)
    snomed_crossrefs: list[dict] = Field(default_factory=list)


class SnomedConceptDetail(BaseModel):
    """Full detail for a single SNOMED CT concept."""

    sctid: str
    fsn: str
    preferred_term: str
    semantic_tag: str
    definition_status: str = ""
    parents: list[dict] = Field(default_factory=list)
    children: list[dict] = Field(default_factory=list)
    relationships: list[dict] = Field(default_factory=list)
    icd10_codes: list[dict] = Field(default_factory=list)
