from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryCategory(str, Enum):
    PRINCIPLES_VALUES = "principles_values"
    COGNITIVE_FRAMING = "cognitive_framing"
    DECISION_HEURISTICS = "decision_heuristics"
    PREFERENCES = "preferences"
    VOCABULARY = "vocabulary"


class MemoryScope(str, Enum):
    PROJECT = "project"
    GLOBAL = "global"


class MemoryApproval(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class MemoryProvenance(BaseModel):
    """Why the system proposed this memory."""
    reason: str = Field(..., description="Short explanation for why this was proposed")
    derived_from_artifact_ids: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)  # document chunks, urls, etc.
    tools_used: List[str] = Field(default_factory=list)
    model: Optional[str] = None


class MemoryCard(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    category: MemoryCategory
    scope: MemoryScope = MemoryScope.PROJECT

    # The actual candidate memory
    text: str = Field(..., min_length=1)

    # Governance
    approval: MemoryApproval = MemoryApproval.PENDING
    approved_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None

    provenance: MemoryProvenance