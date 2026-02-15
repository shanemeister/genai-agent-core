from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class ClaimEvidence(BaseModel):
    """A single piece of evidence for or against a claim."""
    doc_id: str
    text: str
    relevance_score: float  # 0-1 normalized
    relationship: str = "supports"  # supports | contradicts | tangential


class ValidatedClaim(BaseModel):
    """An atomic factual claim with evidence and status."""
    text: str
    status: ClaimStatus
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[ClaimEvidence] = []
    graph_concepts_found: List[str] = []
    graph_concepts_missing: List[str] = []


class ValidationResult(BaseModel):
    """Full validation result for an LLM response."""
    claims: List[ValidatedClaim]
    summary_score: float = Field(ge=0.0, le=1.0)
    supported_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    label: str = "Medium"
    detail: str = ""
