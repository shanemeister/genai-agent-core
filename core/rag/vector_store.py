"""Persistent vector store backed by PostgreSQL + pgvector.

Replaces the in-memory VectorStore with ACID-consistent storage.
Vectors survive server restarts and are included in database backups.
"""

from __future__ import annotations

import json
import numpy as np

from core.db.postgres import get_pool


class PgVectorStore:
    """Async vector store using pgvector for similarity search."""

    async def add(self, doc_id: str, vector: list[float], text: str, metadata: dict | None = None) -> None:
        """Upsert a document embedding. Overwrites if doc_id exists."""
        pool = await get_pool()
        meta = metadata or {}
        source_type = meta.get("source_type", "unknown")
        embedding = np.array(vector, dtype=np.float32)

        await pool.execute(
            """
            INSERT INTO embeddings (doc_id, embedding, text, metadata, source_type)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (doc_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                source_type = EXCLUDED.source_type
            """,
            doc_id,
            embedding,
            text,
            json.dumps(meta),
            source_type,
        )

    async def search(
        self, query: list[float], k: int = 3, source_types: list[str] | None = None
    ) -> list[dict]:
        """Search by inner-product similarity with optional source filtering.

        Uses pgvector <#> operator (negative inner product).
        """
        pool = await get_pool()
        query_vec = np.array(query, dtype=np.float32)

        if source_types:
            rows = await pool.fetch(
                """
                SELECT doc_id, text, metadata,
                       (embedding <#> $1) * -1 AS score
                FROM embeddings
                WHERE source_type = ANY($3)
                ORDER BY embedding <#> $1
                LIMIT $2
                """,
                query_vec,
                k,
                source_types,
            )
        else:
            rows = await pool.fetch(
                """
                SELECT doc_id, text, metadata,
                       (embedding <#> $1) * -1 AS score
                FROM embeddings
                ORDER BY embedding <#> $1
                LIMIT $2
                """,
                query_vec,
                k,
            )

        return [
            {
                "doc_id": row["doc_id"],
                "text": row["text"],
                "score": float(row["score"]),
                "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
            }
            for row in rows
        ]

    async def remove(self, doc_id: str) -> None:
        """Remove a document embedding by doc_id."""
        pool = await get_pool()
        await pool.execute("DELETE FROM embeddings WHERE doc_id = $1", doc_id)

    async def remove_by_prefix(self, prefix: str) -> int:
        """Remove all embeddings whose doc_id starts with prefix. Returns count deleted."""
        pool = await get_pool()
        result = await pool.execute(
            "DELETE FROM embeddings WHERE doc_id LIKE $1", f"{prefix}%"
        )
        return int(result.split()[-1])  # "DELETE N"

    async def exists(self, doc_id: str) -> bool:
        """Check if a doc_id exists in the store."""
        pool = await get_pool()
        row = await pool.fetchval(
            "SELECT 1 FROM embeddings WHERE doc_id = $1", doc_id
        )
        return row is not None

    async def count(self, source_type: str | None = None) -> int:
        """Count embeddings, optionally filtered by source_type."""
        pool = await get_pool()
        if source_type:
            return await pool.fetchval(
                "SELECT count(*) FROM embeddings WHERE source_type = $1",
                source_type,
            )
        return await pool.fetchval("SELECT count(*) FROM embeddings")
