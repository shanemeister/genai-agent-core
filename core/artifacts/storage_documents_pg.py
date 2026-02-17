"""PostgreSQL storage for Document artifacts.

Stores documents and chunks in PostgreSQL on the workstation (192.168.4.25:5433).
This ensures data is accessible from any client machine running Noesis.
"""

from __future__ import annotations

from typing import List, Optional
from datetime import datetime

from core.artifacts.document import Document, DocumentChunk, DocumentType, DocumentStatus
from core.db.postgres import get_pool


async def save_document(doc: Document) -> None:
    """Save or update document in PostgreSQL."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO documents(
                id, created_at, filename, filepath, file_type, file_size, file_hash,
                status, chunk_count, error_message, tags, notes
            ) VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT(id) DO UPDATE SET
                filename = EXCLUDED.filename,
                status = EXCLUDED.status,
                chunk_count = EXCLUDED.chunk_count,
                error_message = EXCLUDED.error_message,
                tags = EXCLUDED.tags,
                notes = EXCLUDED.notes
            """,
            doc.id,
            doc.created_at if isinstance(doc.created_at, datetime) else datetime.fromisoformat(doc.created_at),
            doc.filename,
            doc.filepath,
            doc.file_type.value if isinstance(doc.file_type, DocumentType) else doc.file_type,
            doc.file_size,
            doc.file_hash,
            doc.status.value if isinstance(doc.status, DocumentStatus) else doc.status,
            doc.chunk_count,
            doc.error_message,
            doc.tags,
            doc.notes,
        )


async def load_document(doc_id: str) -> Document | None:
    """Load document by ID from PostgreSQL."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM documents WHERE id = $1",
            doc_id,
        )

        if not row:
            return None

        return _document_from_row(row)


async def load_all_documents(status: Optional[DocumentStatus] = None, limit: int = 100) -> List[Document]:
    """Load all documents, optionally filtered by status."""
    pool = await get_pool()
    docs: List[Document] = []

    async with pool.acquire() as conn:
        if status:
            rows = await conn.fetch(
                "SELECT * FROM documents WHERE status = $1 ORDER BY created_at DESC LIMIT $2",
                status.value,
                limit,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT $1",
                limit,
            )

        for row in rows:
            docs.append(_document_from_row(row))

    return docs


async def save_chunk(chunk: DocumentChunk) -> None:
    """Save document chunk to PostgreSQL."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO document_chunks(
                id, document_id, chunk_index, text, page_number, heading, image_caption, ocr_text
            ) VALUES($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT(id) DO UPDATE SET
                text = EXCLUDED.text,
                page_number = EXCLUDED.page_number,
                heading = EXCLUDED.heading,
                image_caption = EXCLUDED.image_caption,
                ocr_text = EXCLUDED.ocr_text
            """,
            chunk.id,
            chunk.document_id,
            chunk.chunk_index,
            chunk.text,
            chunk.page_number,
            chunk.heading,
            chunk.image_caption,
            chunk.ocr_text,
        )


async def load_chunks_by_document(doc_id: str, limit: Optional[int] = None) -> List[DocumentChunk]:
    """Load all chunks for a document, ordered by chunk_index."""
    pool = await get_pool()
    chunks: List[DocumentChunk] = []

    async with pool.acquire() as conn:
        if limit:
            rows = await conn.fetch(
                "SELECT * FROM document_chunks WHERE document_id = $1 ORDER BY chunk_index LIMIT $2",
                doc_id,
                limit,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM document_chunks WHERE document_id = $1 ORDER BY chunk_index",
                doc_id,
            )

        for row in rows:
            chunks.append(_chunk_from_row(row))

    return chunks


async def delete_document(doc_id: str) -> None:
    """Delete document and all its chunks (CASCADE handles chunks automatically)."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM documents WHERE id = $1", doc_id)


async def find_document_by_hash(file_hash: str) -> Document | None:
    """Find document by file hash (for deduplication)."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM documents WHERE file_hash = $1",
            file_hash,
        )

        if not row:
            return None

        return _document_from_row(row)


def _document_from_row(row) -> Document:
    """Reconstruct Document from PostgreSQL row."""
    return Document(
        id=row["id"],
        created_at=row["created_at"],
        filename=row["filename"],
        filepath=row["filepath"],
        file_type=DocumentType(row["file_type"]),
        file_size=row["file_size"],
        file_hash=row["file_hash"],
        status=DocumentStatus(row["status"]),
        chunk_count=row["chunk_count"],
        error_message=row["error_message"] or "",
        tags=list(row["tags"]) if row["tags"] else [],
        notes=row["notes"] or "",
    )


def _chunk_from_row(row) -> DocumentChunk:
    """Reconstruct DocumentChunk from PostgreSQL row."""
    return DocumentChunk(
        id=row["id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        text=row["text"],
        page_number=row["page_number"],
        heading=row["heading"] or "",
        image_caption=row["image_caption"] or "",
        ocr_text=row["ocr_text"] or "",
    )
