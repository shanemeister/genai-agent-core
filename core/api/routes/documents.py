"""Document upload/ingest, filesystem browse/import/scan, vision, and settings endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form

from core.artifacts.document import Document, DocumentChunk, DocumentType, DocumentStatus
from core.artifacts.storage_documents_pg import (
    save_document,
    load_document,
    load_all_documents,
    save_chunk,
    load_chunks_by_document,
    delete_document,
    find_document_by_hash,
)
from core.tools.mcp_filesystem import FilesystemAccess
from core.tools.document_ingest import DocumentIngestor

log = logging.getLogger("noesis.documents")

router = APIRouter(tags=["documents"])


# ---------------------------------------------------------------------------
# Document upload & management
# ---------------------------------------------------------------------------

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    tags: str = Form(default=""),
    notes: str = Form(default=""),
):
    """Upload document for indexing."""
    content = await file.read()

    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    file_hash = hashlib.sha256(content).hexdigest()

    existing = await find_document_by_hash(file_hash)
    if existing:
        return {"status": "duplicate", "document_id": existing.id}

    ext = Path(file.filename).suffix[1:].lower()
    try:
        file_type = DocumentType(ext)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Supported: PDF, DOCX, TXT, MD, CSV, images, SVG"
        )

    doc = Document(
        filename=file.filename,
        filepath=f"uploads/{file.filename}",
        file_type=file_type,
        file_size=len(content),
        file_hash=file_hash,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
        notes=notes,
    )
    await save_document(doc)

    upload_dir = Path("data/documents/uploads") / doc.id
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / file.filename).write_bytes(content)

    asyncio.create_task(_index_document_task(doc.id, content))

    return {"status": "uploaded", "document_id": doc.id}


async def _index_document_task(doc_id: str, content: bytes):
    """Background task: ingest document into RAG system."""
    from core.rag.vector_store import PgVectorStore

    try:
        doc = await load_document(doc_id)
        if not doc:
            log.error("Document %s not found for indexing", doc_id)
            return

        ingestor = DocumentIngestor(PgVectorStore())
        await ingestor.ingest(doc, content)
        log.info("Indexed %s (%d chunks)", doc.filename, doc.chunk_count)
    except Exception as e:
        log.error("Failed to index document %s: %s", doc_id, e, exc_info=True)


@router.get("/documents")
async def list_documents(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    """List documents with optional status filter."""
    doc_status = DocumentStatus(status) if status else None
    docs = await load_all_documents(status=doc_status, limit=limit)
    return [doc.to_dict() for doc in docs]


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document metadata by ID."""
    doc = await load_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc.to_dict()


@router.get("/documents/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    limit: Optional[int] = Query(default=None, ge=1, le=1000),
):
    """Get all chunks for a document."""
    doc = await load_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = await load_chunks_by_document(doc_id, limit=limit)
    return [chunk.to_dict() for chunk in chunks]


@router.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    """Delete document and all its chunks."""
    from core.rag.vector_store import PgVectorStore

    doc = await load_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove chunks from vector store
    store = PgVectorStore()
    chunks = await load_chunks_by_document(doc_id)
    for chunk in chunks:
        await store.remove(f"chunk:{chunk.id}")

    # Delete from PostgreSQL (CASCADE handles chunks)
    await delete_document(doc_id)

    # Delete uploaded file from server disk
    upload_dir = Path("data/documents/uploads") / doc_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)

    return {"status": "deleted", "document_id": doc_id}


# ---------------------------------------------------------------------------
# Filesystem browse/import/scan
# ---------------------------------------------------------------------------

@router.post("/documents/browse")
async def browse_filesystem(req: dict):
    """Browse allowed directories on server filesystem."""
    fs = FilesystemAccess()

    directory = req.get("directory", str(Path.home() / "Documents"))
    recursive = req.get("recursive", False)

    try:
        files = fs.list_files(directory, recursive)
        return {"files": files}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/documents/import")
async def import_from_filesystem(req: dict):
    """Import file from server filesystem path."""
    fs = FilesystemAccess()
    filepath = req.get("filepath")

    if not filepath:
        raise HTTPException(status_code=400, detail="filepath required")

    try:
        content = fs.read_file(filepath)
        file_hash = fs.compute_hash(content)

        existing = await find_document_by_hash(file_hash)
        if existing:
            return {"status": "duplicate", "document_id": existing.id}

        ext = Path(filepath).suffix[1:].lower()
        try:
            file_type = DocumentType(ext)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: .{ext}"
            )

        doc = Document(
            filename=Path(filepath).name,
            filepath=filepath,
            file_type=file_type,
            file_size=len(content),
            file_hash=file_hash,
        )
        await save_document(doc)

        asyncio.create_task(_index_document_task(doc.id, content))

        return {"status": "imported", "document_id": doc.id}

    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/documents/scan")
async def scan_and_import_directory(req: dict):
    """Scan a directory tree and import all supported files."""
    fs = FilesystemAccess()
    directory = req.get("directory")

    if not directory:
        raise HTTPException(status_code=400, detail="directory required")

    try:
        files = fs.list_files(directory, recursive=True)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    results = []
    imported = 0
    duplicates = 0
    errors = 0

    for f in files:
        filepath = f["path"]
        try:
            content = fs.read_file(filepath)
            file_hash = fs.compute_hash(content)

            existing = await find_document_by_hash(file_hash)
            if existing:
                results.append({"path": filepath, "status": "duplicate", "document_id": existing.id})
                duplicates += 1
                continue

            ext = Path(filepath).suffix[1:].lower()
            try:
                file_type = DocumentType(ext)
            except ValueError:
                results.append({"path": filepath, "status": "error", "document_id": None})
                errors += 1
                continue

            doc = Document(
                filename=Path(filepath).name,
                filepath=filepath,
                file_type=file_type,
                file_size=len(content),
                file_hash=file_hash,
            )
            await save_document(doc)

            asyncio.create_task(_index_document_task(doc.id, content))

            results.append({"path": filepath, "status": "imported", "document_id": doc.id})
            imported += 1

        except Exception as e:
            results.append({"path": filepath, "status": "error", "document_id": None})
            errors += 1

    return {
        "scanned": len(files),
        "imported": imported,
        "duplicates": duplicates,
        "errors": errors,
        "files": results,
    }


# ---------------------------------------------------------------------------
# Vision endpoints
# ---------------------------------------------------------------------------

@router.post("/vision/caption")
async def caption_image(file: UploadFile = File(...)):
    """Generate a caption and OCR text for an uploaded image."""
    from core.vision.captioner import caption_and_ocr

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 50MB)")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(status_code=400, detail="Unsupported image format. Use JPG or PNG.")

    try:
        result = caption_and_ocr(content)
        return {
            "caption": result["caption"],
            "ocr_text": result["ocr_text"],
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision processing error: {str(e)}")


# ---------------------------------------------------------------------------
# Filesystem settings
# ---------------------------------------------------------------------------

@router.get("/settings/filesystem/allowed-roots")
async def get_allowed_roots():
    """Get list of allowed directory roots."""
    from core.settings.filesystem_settings import load_allowed_roots
    return {"roots": load_allowed_roots()}


@router.put("/settings/filesystem/allowed-roots")
async def update_allowed_roots(req: dict):
    """Update allowed directory roots."""
    from core.settings.filesystem_settings import save_allowed_roots

    roots = req.get("roots", [])
    if not isinstance(roots, list):
        raise HTTPException(status_code=400, detail="roots must be a list")

    save_allowed_roots(roots)
    return {"status": "updated"}
