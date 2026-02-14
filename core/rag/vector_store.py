from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VectorStore:
    vectors: list[tuple[str, list[float], str]] = field(default_factory=list)

    def add(self, doc_id: str, vector: list[float], text: str) -> None:
        self.vectors.append((doc_id, vector, text))

    def search(self, query: list[float], k: int = 3) -> list[dict]:
        def score(vector: list[float]) -> float:
            return sum(a * b for a, b in zip(query, vector))

        ranked = sorted(self.vectors, key=lambda item: score(item[1]), reverse=True)
        return [
            {"doc_id": doc_id, "score": score(vector), "text": text}
            for doc_id, vector, text in ranked[:k]
        ]

    def save(self, path: Path) -> None:
        payload = [
            {"doc_id": doc_id, "vector": vector, "text": text}
            for doc_id, vector, text in self.vectors
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.vectors = [
            (item["doc_id"], item["vector"], item["text"])
            for item in data
        ]
