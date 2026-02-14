from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class Artifact:
    type: str
    title: str
    body: str
    source_ids: list[str] = field(default_factory=list)
    artifact_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "draft"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
