from __future__ import annotations

from dataclasses import dataclass, field

from core.artifacts.base import Artifact


@dataclass(slots=True)
class DiagramArtifact(Artifact):
    syntax: str = "mermaid"
    model: str = "placeholder-model"
    type: str = field(default="diagram", init=False)
