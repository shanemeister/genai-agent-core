# Tools (Room 2)

This folder holds MCP tool adapters and capability wrappers for external data access.

## Phase 2a: MCP Filesystem — COMPLETE

The MCP filesystem server gives Noesis read access to local and network-shared files
without those files ever leaving the system. This is a Zone 1 (fully private) integration.

### What was built:

| File | Purpose |
|------|---------|
| `mcp_filesystem.py` | Privacy-controlled filesystem access with allowlist |
| `document_parser.py` | PDF (PyMuPDF), DOCX, TXT, MD, CSV, Images (OCR/Tesseract), SVG parsing |
| `document_ingest.py` | Chunking (RecursiveCharacterTextSplitter, 2000 chars, 25% overlap), embedding, indexing |

### How it integrates:
- MCP filesystem server → document ingestion pipeline → chunk → embed → RAG index
- Chat queries search ingested documents via existing `retrieve_context()`
- Documents stored in PostgreSQL (`documents` + `document_chunks` tables)
- Chunks embedded with nomic-embed-text-v1.5 (768d) into vector store
- SHA-256 deduplication prevents redundant uploads

### API endpoints (on server.py port 8008):
- `POST /documents/upload` — Upload and index a document
- `GET /documents` — List all documents with status
- `GET /documents/{id}` — Get document details
- `DELETE /documents/{id}` — Remove document and chunks
- `POST /documents/browse` — Browse allowed filesystem directories
- `POST /documents/import` — Import from filesystem path
- `GET /settings/filesystem/allowed-roots` — Get allowed directories
- `PUT /settings/filesystem/allowed-roots` — Update allowed directories

**See:** `docs/room2-phase2a-implementation.md` for complete technical reference.

## Phase 2b: Advanced Document Processing — NEXT

- Vision models (BLIP-2) for semantic image understanding beyond OCR
- Neo4j graph sync: extract entities/relationships from documents
- Auto memory proposal from document insights
- Advanced chunking: heading-aware, semantic chunking
- Filesystem watcher: auto-reindex on file changes
- Batch import: import entire directories at once
- Document similarity search via embeddings

## Phase 2c: External Connectors (future)

Zone 2 integrations — user authorizes pulling THEIR data from external services
INTO the private system. Data flows inward only. Nothing goes out.

- **MCP Client for MS 365** — Graph API: emails, documents, calendar → local index
- **MCP Client for Google Workspace** — Drive, Docs, Sheets → local index
- **MCP Client for GitHub** — repos, issues, PRs → local index

## Privacy Zones (guiding principle)

- **Zone 1 — Fully Private:** Data on your hardware. Core promise. (MCP filesystem)
- **Zone 2 — User-Controlled External:** Data pulled IN from user's own accounts. (MS 365, Google, GitHub)
- **Zone 3 — Breaks Promise:** Data sent OUT for processing. NEVER do this.

## Tool Registry (Room 3 — planned)

Central registry for agents to discover available tools dynamically.

## MCP Server (Room 3 — planned)

Expose Noesis capabilities (search memories, create diagrams, query graph, propose cards)
as MCP tools for other AI systems to use.
