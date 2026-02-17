"""Document ingestion pipeline: parse → chunk → embed → index.

Integrates with existing RAG infrastructure (vector_store, retriever).
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import settings
from core.rag.embeddings import embed_text
from core.rag.vector_store import PgVectorStore
from core.tools.document_parser import DocumentParser
from core.artifacts.document import Document, DocumentChunk, DocumentStatus
from core.artifacts.storage_documents_pg import save_document, save_chunk


class DocumentIngestor:
    """Ingest documents into the RAG system."""

    def __init__(self, vector_store: PgVectorStore):
        self.parser = DocumentParser()
        self.vector_store = vector_store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    async def ingest(self, document: Document, content: bytes) -> List[DocumentChunk]:
        """Ingest a document: parse → chunk → embed → store.

        Updates document status in place (INDEXING → INDEXED or ERROR).

        Returns:
            List of created DocumentChunk objects
        """
        try:
            # 1. Update status to INDEXING
            document.status = DocumentStatus.INDEXING
            await save_document(document)

            # 2. Parse document
            parsed = self.parser.parse(document.filename, content)

            # 3. Create chunks with context preservation
            chunks = self._create_chunks(document, parsed)

            # 4. Embed each chunk and index in vector store
            for chunk in chunks:
                # Generate embedding
                vector = embed_text(chunk.text)

                # Add to vector store with metadata
                await self.vector_store.add(
                    doc_id=f"chunk:{chunk.id}",
                    vector=vector,
                    text=chunk.text,
                    metadata={
                        "source_type": "document_chunk",
                        "document_id": document.id,
                        "chunk_index": chunk.chunk_index,
                        "filename": document.filename,
                        "file_type": document.file_type.value,
                        "page_number": chunk.page_number,
                    }
                )

                # Save chunk to PostgreSQL
                await save_chunk(chunk)

            # 5. Update document status to INDEXED
            document.status = DocumentStatus.INDEXED
            document.chunk_count = len(chunks)
            document.error_message = ""
            await save_document(document)

            return chunks

        except Exception as e:
            # Update document status to ERROR
            document.status = DocumentStatus.ERROR
            document.error_message = str(e)
            await save_document(document)
            raise

    def _create_chunks(self, doc: Document, parsed: dict) -> List[DocumentChunk]:
        """Create chunks with context preservation.

        For PDFs: Track page numbers
        For other formats: Simple sequential chunking
        """
        chunks = []

        if "pages" in parsed:  # PDF with page tracking
            for page_data in parsed["pages"]:
                page_text = page_data["text"]
                if not page_text.strip():
                    continue  # Skip empty pages

                # Split page into chunks
                page_chunks = self.splitter.split_text(page_text)

                for text in page_chunks:
                    chunks.append(DocumentChunk(
                        document_id=doc.id,
                        text=text,
                        chunk_index=len(chunks),
                        page_number=page_data["page_number"],
                    ))

        else:  # Other formats (DOCX, TXT, MD, CSV, images, SVG)
            full_text = parsed["text"]
            # Vision fields (populated for images by Florence-2)
            image_caption = parsed.get("image_caption", "")
            ocr_text = parsed.get("ocr_text", "")

            if not full_text.strip():
                # Empty document - create single empty chunk
                chunks.append(DocumentChunk(
                    document_id=doc.id,
                    text="[Empty document]",
                    chunk_index=0,
                ))
            else:
                # Split into chunks
                text_chunks = self.splitter.split_text(full_text)

                for i, text in enumerate(text_chunks):
                    chunks.append(DocumentChunk(
                        document_id=doc.id,
                        text=text,
                        chunk_index=i,
                        page_number=0,
                        image_caption=image_caption if i == 0 else "",
                        ocr_text=ocr_text if i == 0 else "",
                    ))

        return chunks
