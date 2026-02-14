from __future__ import annotations

from dataclasses import dataclass, field

from core.artifacts.base import Artifact


@dataclass(slots=True)
class GraphArtifact(Artifact):
    directed: bool = True
    type: str = field(default="graph", init=False)
