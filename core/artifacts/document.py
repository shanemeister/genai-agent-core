"""Document artifact models for Room 2 MCP Filesystem integration.

Document: Metadata about uploaded/imported files
DocumentChunk: Individual chunks with embeddings for RAG retrieval
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from enum import Enum


class DocumentType(str, Enum):
    """Supported document file types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    CSV = "csv"
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    SVG = "svg"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"      # Uploaded, not yet indexed
    INDEXING = "indexing"    # Chunks being embedded
    INDEXED = "indexed"      # Available in RAG
    ERROR = "error"          # Failed to process


@dataclass
class Document:
    """Document metadata artifact (stored in SQLite)."""

    # Identity
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    # File metadata
    filename: str = ""
    filepath: str = ""  # Original location (for re-indexing)
    file_type: DocumentType = DocumentType.TXT
    file_size: int = 0  # bytes
    file_hash: str = ""  # SHA-256 for deduplication

    # Processing state
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    error_message: str = ""

    # User annotations
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "filename": self.filename,
            "filepath": self.filepath,
            "file_type": self.file_type.value if isinstance(self.file_type, DocumentType) else self.file_type,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "status": self.status.value if isinstance(self.status, DocumentStatus) else self.status,
            "chunk_count": self.chunk_count,
            "error_message": self.error_message,
            "tags": self.tags,
            "notes": self.notes,
        }


@dataclass
class DocumentChunk:
    """Individual chunk with embedding (stored in vector store + SQLite)."""

    # Identity
    id: str = field(default_factory=lambda: str(uuid4()))
    document_id: str = ""  # Parent document

    # Chunk content
    text: str = ""
    chunk_index: int = 0  # Position in document (0, 1, 2, ...)

    # Context preservation
    page_number: int = 0  # For PDFs (0 if not applicable)
    heading: str = ""  # Section context (future enhancement)

    # Vision-specific (Phase 2b)
    image_caption: str = ""  # For image chunks
    ocr_text: str = ""  # Tesseract fallback

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "heading": self.heading,
            "image_caption": self.image_caption,
            "ocr_text": self.ocr_text,
        }
