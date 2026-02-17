# Artifact Model

All artifacts inherit from `core/artifacts/base.py` and share:

- `artifact_id`: stable id
- `type`: diagram, graph, memory_card, mindfile_entry
- `title` and `body`: user-visible content
- `source_ids`: provenance references
- `status`: draft/accepted/rejected lifecycle
- `created_at`: UTC timestamp

## Current Types

### 1. DiagramArtifact (`core/artifacts/diagram.py`)
- Mermaid syntax by default
- Generated from natural language via chat or manually edited
- Rendered live in split-pane DiagramCanvas (source + SVG preview)
- Sanitization handles LLM truncation and syntax errors
- Zoom controls (25%-300% with +/-/Reset)

### 2. GraphArtifact (`core/artifacts/graph.py`)
- Directed graph representation
- Backed by Neo4j (database: `noesis`)
- 5 node labels: Concept, MemoryCard, ChatSession, Diagram, Session
- 7 relationship types: RELATED_TO, MENTIONS, USED_CONTEXT, DISCUSSES, PRODUCED, PROPOSED, CURATED_FROM
- Visualized with d3-force in GraphExplorer
- Scoped queries: scope (session|question|artifact), view_mode (provenance|lineage|full), depth (1-3)

### 3. MemoryCard (`core/artifacts/memory_card.py`)
- Pydantic model with validation
- Categories: principles_values, cognitive_framing, decision_heuristics, preferences, vocabulary
- Scopes: project, global
- Approval states: pending, approved, rejected
- Provenance tracking: reason, tools_used, model
- Stored in PostgreSQL via `storage_memory_pg.py` (migrated from SQLite in Phase 2a)
- Indexed in RAG vector store when approved
- Auto-synced to Neo4j graph on approval
- Auto-promoted to Mind File on approval

### 4. MindFileEntry (`core/artifacts/mindfile_entry.py`)
- Curated entries promoted from approved MemoryCards
- Tracks `source_memory_card_id` for traceability
- Same category structure as MemoryCard
- User-editable annotations (notes)
- Persisted in PostgreSQL (mind file entries table)
- Three views: Entries (browse/search/filter), Timeline (concept evolution), Profile (cognitive breakdown)

### 5. ValidatedClaim (`core/validation/models.py`)
- Extracted from AI responses via LLM-based claim extraction
- Each claim has: text, status (supported|unsupported|contradicted|unverifiable), confidence (0-1)
- Evidence: per-claim RAG results with relevance scores
- Graph coverage: concepts found/missing in Neo4j
- Aggregated into ValidationResult with summary_score and label (High/Medium/Low/Ungrounded)
- On-demand only — user clicks "Validate" per message

### 6. Document + DocumentChunk (`core/artifacts/document.py`)
- Ingested from local filesystem via MCP or uploaded via UI
- Document fields: id, filename, filepath, file_type, file_size, file_hash (SHA-256), status, chunk_count, tags, notes
- file_type enum: PDF, DOCX, TXT, MD, CSV, IMAGE, SVG
- status enum: PENDING, INDEXING, INDEXED, ERROR
- Chunked via RecursiveCharacterTextSplitter (2000 chars, 25% overlap)
- Each DocumentChunk tracks: document_id, chunk_index, page_number, text
- Chunks embedded with nomic-embed-text-v1.5 (768d) and indexed in vector store
- SHA-256 deduplication prevents re-uploading the same file
- Stored in PostgreSQL via `storage_documents_pg.py`
- Entities extracted from documents → Neo4j graph nodes (Phase 2b)
- Facts from documents → proposed memory cards (Phase 2b)

## Memory Proposal Flow

1. User sends chat message
2. DeepSeek-R1-70B generates response via SSE streaming
3. `_propose_memories_for_session()` analyzes the conversation
4. `propose_memories()` calls LLM to extract memory-worthy facts
5. Proposed cards saved as **pending** in PostgreSQL
6. User reviews in Memory Deck: approve, edit, or reject
7. On approval: card indexed in RAG vector store + synced to Neo4j graph + promoted to Mind File

## Validation Flow (on-demand)

1. User clicks "Validate" on an AI response
2. `claim_extractor.py` calls LLM to break response into 3-8 atomic claims
3. `claim_validator.py` checks each claim:
   - RAG evidence retrieval (`retrieve_context(claim, k=3)`)
   - Graph concept coverage (`extract_concepts(claim)` → `search_nodes()`)
   - Status determination based on evidence strength
4. ValidationResult returned to frontend with per-claim breakdown
5. Displayed as expandable panel with color-coded status dots

## Storage

| Artifact | Storage | Location |
|----------|---------|----------|
| Memory Cards | PostgreSQL | Axiom Core port 5433, `memory_cards` table |
| Mind File Entries | PostgreSQL | Axiom Core port 5433 |
| Chat Sessions | PostgreSQL | Axiom Core port 5433, `chat_sessions` table |
| Documents | PostgreSQL | Axiom Core port 5433, `documents` table |
| Document Chunks | PostgreSQL | Axiom Core port 5433, `document_chunks` table |
| Graph Nodes/Edges | Neo4j | Axiom Core port 7687 |
| Diagram Source | In-memory | Generated on demand |
| Embeddings | In-memory vector store | `data/vector_store.pkl` (pickle persistence) |
| Validated Claims | In-memory | Attached to message, not persisted |
| Document Uploads | Filesystem | `data/documents/uploads/{doc_id}/` |
| ~~SQLite~~ | Deprecated | `data/memory/memory_cards.sqlite3` (backup only) |
