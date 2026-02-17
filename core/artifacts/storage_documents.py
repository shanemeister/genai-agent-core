"""SQLite storage for Document artifacts.

Follows the same pattern as storage_sqlite.py for memory cards.
Two tables: documents (metadata) and document_chunks (indexed chunks).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import aiosqlite

from core.artifacts.document import Document, DocumentChunk, DocumentType, DocumentStatus

DB_PATH = Path("data/documents/documents.sqlite3")

CREATE_DOCUMENTS_SQL = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  filename TEXT NOT NULL,
  filepath TEXT NOT NULL,
  file_hash TEXT NOT NULL,
  status TEXT NOT NULL,
  json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
"""

CREATE_CHUNKS_SQL = """
CREATE TABLE IF NOT EXISTS document_chunks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
"""


async def init_documents_db() -> None:
    """Initialize documents database and tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        await db.execute(CREATE_DOCUMENTS_SQL)
        await db.execute(CREATE_CHUNKS_SQL)
        await db.commit()


async def save_document(doc: Document) -> None:
    """Save or update document in database."""
    await init_documents_db()
    payload = doc.to_dict()

    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        await db.execute(
            """INSERT INTO documents(id, created_at, filename, filepath, file_hash, status, json)
               VALUES(?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   filename=excluded.filename,
                   status=excluded.status,
                   json=excluded.json""",
            (
                doc.id,
                doc.created_at.isoformat() if isinstance(doc.created_at, datetime) else doc.created_at,
                doc.filename,
                doc.filepath,
                doc.file_hash,
                doc.status.value if isinstance(doc.status, DocumentStatus) else doc.status,
                json.dumps(payload),
            ),
        )
        await db.commit()


async def load_document(doc_id: str) -> Document | None:
    """Load document by ID."""
    await init_documents_db()
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        async with db.execute("SELECT json FROM documents WHERE id = ?", (doc_id,)) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            data = json.loads(row[0])
            return _document_from_dict(data)


async def load_all_documents(status: Optional[DocumentStatus] = None) -> List[Document]:
    """Load all documents, optionally filtered by status."""
    await init_documents_db()
    docs: List[Document] = []

    query = "SELECT json FROM documents"
    params = ()

    if status:
        query += " WHERE status = ?"
        params = (status.value,)

    query += " ORDER BY created_at DESC"

    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        async with db.execute(query, params) as cur:
            async for (json_str,) in cur:
                data = json.loads(json_str)
                docs.append(_document_from_dict(data))

    return docs


async def save_chunk(chunk: DocumentChunk) -> None:
    """Save document chunk to database."""
    await init_documents_db()
    payload = chunk.to_dict()

    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        await db.execute(
            """INSERT INTO document_chunks(id, document_id, chunk_index, json)
               VALUES(?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET json=excluded.json""",
            (chunk.id, chunk.document_id, chunk.chunk_index, json.dumps(payload)),
        )
        await db.commit()


async def load_chunks_by_document(doc_id: str, limit: Optional[int] = None) -> List[DocumentChunk]:
    """Load all chunks for a document, ordered by chunk_index."""
    await init_documents_db()
    chunks: List[DocumentChunk] = []

    query = "SELECT json FROM document_chunks WHERE document_id = ? ORDER BY chunk_index"
    if limit:
        query += f" LIMIT {limit}"

    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        async with db.execute(query, (doc_id,)) as cur:
            async for (json_str,) in cur:
                data = json.loads(json_str)
                chunks.append(_chunk_from_dict(data))

    return chunks


async def delete_document(doc_id: str) -> None:
    """Delete document and all its chunks (cascade)."""
    await init_documents_db()
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        # Delete chunks first
        await db.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
        # Then delete document
        await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await db.commit()


async def find_document_by_hash(file_hash: str) -> Document | None:
    """Find document by file hash (for deduplication)."""
    await init_documents_db()
    async with aiosqlite.connect(DB_PATH.as_posix()) as db:
        async with db.execute("SELECT json FROM documents WHERE file_hash = ?", (file_hash,)) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            data = json.loads(row[0])
            return _document_from_dict(data)


def _document_from_dict(data: dict) -> Document:
    """Reconstruct Document from dict (handles enum conversion)."""
    # Convert string enums back to enum types
    if "file_type" in data and isinstance(data["file_type"], str):
        data["file_type"] = DocumentType(data["file_type"])
    if "status" in data and isinstance(data["status"], str):
        data["status"] = DocumentStatus(data["status"])
    if "created_at" in data and isinstance(data["created_at"], str):
        data["created_at"] = datetime.fromisoformat(data["created_at"])

    return Document(**data)


def _chunk_from_dict(data: dict) -> DocumentChunk:
    """Reconstruct DocumentChunk from dict."""
    return DocumentChunk(**data)
