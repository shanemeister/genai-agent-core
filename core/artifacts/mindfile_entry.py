from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MindFileEntryCategory(str, Enum):
    PRINCIPLES_VALUES = "principles_values"
    COGNITIVE_FRAMING = "cognitive_framing"
    DECISION_HEURISTICS = "decision_heuristics"
    PREFERENCES = "preferences"
    VOCABULARY = "vocabulary"


class MindFileEntry(BaseModel):
    """A curated entry. Created only from approved MemoryCards."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    category: MindFileEntryCategory
    text: str = Field(..., min_length=1)

    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Traceability
    source_memory_card_id: str
    note: Optional[str] = None