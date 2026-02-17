# Roadmap

## Room 0 (V0) — COMPLETE

### Completed
- Tauri desktop shell (migrated from Electron)
- FastAPI single-backend architecture (port 8008 services, calls vLLM directly on 8081)
- vLLM serving DeepSeek-R1-70B (migrated from Llama-3-8B via HuggingFace transformers)
- SSE streaming chat with animated thinking indicator (`<think>` tag parsing)
- Diagram Canvas with Mermaid live rendering + LLM diagram generation
- Memory Deck with approval workflow + automatic memory proposals from chat
- Graph Explorer with Neo4j backend + d3-force visualization
- RAG retrieval loop wired (approved memories indexed and retrieved)
- Artifact data models (base, diagram, graph, memory card)
- SQLite persistence for memory cards
- LLM-powered concept extraction for graph
- Diagram <-> Graph bidirectional conversion (Mermaid parse/export + Neo4j import)
- Session persistence in SQLite (auto-save with debounce, session picker, load/delete)
- Export session as markdown (backend export endpoint + frontend download button)
- Embedding model upgrade (nomic-ai/nomic-embed-text-v1.5, 768d)
- RAG reranker (BAAI/bge-reranker-v2-m3, retrieve 15 → rerank to top 5)
- Cypher injection prevention (whitelist + regex validation for relationship types)
- Diagram Canvas zoom controls (25%-300% with +/−/Reset)

---

## Room 1 — COMPLETE

### Completed
- Mind File system — auto-promotes approved memory cards to curated MindFileEntries
- Mind File UI tab with category filters, search, inline note editing, export as markdown
- Temporal thinking — timeline view showing concept evolution across artifacts
- Pattern recognition — top concepts, concept co-occurrences, category trends from Neo4j
- Cognitive profile — aggregate category breakdown, top concepts bar chart, co-occurring concept pairs
- 7 new API endpoints: /mindfile (CRUD), /mindfile/timeline, /mindfile/patterns, /mindfile/cognitive-profile

### Hardening (completed post-Room 1)
- Response validation system — claim-level evidence checking (LLM extracts claims, per-claim RAG + graph evidence, on-demand via /validate endpoint and UI button)
- Graph scoping model — scope (session|question|artifact), view_mode (provenance|lineage|full), depth (1-3 hops), /graph/scoped endpoint
- Graph Explorer filtering — toolbar dropdowns for view mode and depth, click-to-expand uses scoped queries
- On-demand validation — removed auto-validation from SSE stream, added per-message Validate button in chat

---

## Room 2 — IN PROGRESS

External knowledge connectors. The goal: give Noesis access to real-world documents
and data sources while maintaining the privacy guarantee. Data flows IN to the system;
questions, reasoning, and artifacts stay local.

### Phase 2a: MCP Filesystem — COMPLETE ✅

**Completed:** 2026-02-16

#### Multi-Client PostgreSQL Architecture
- Migrated from SQLite to PostgreSQL for centralized storage on Axiom Core
- Multi-client access: Mac Studio, laptop, etc. all share same data
- PostgreSQL connection pooling (asyncpg)
- Migration script: 10 memory cards, 5 chat sessions migrated

#### Document Ingestion Pipeline
- MCP filesystem with privacy-controlled allowlist access
- Document parser: PDF (PyMuPDF), DOCX, TXT, MD, CSV, Images (OCR), SVG
- Chunking: RecursiveCharacterTextSplitter (2000 chars, 25% overlap)
- Embedding: nomic-embed-text-v1.5 (768d)
- Vector store with metadata: source_type, filename, page_number, etc.
- PostgreSQL storage: documents + document_chunks tables

#### API & Frontend
- 8 new endpoints: /documents/upload, /documents, /documents/{id}, /documents/browse, /documents/import, /settings/filesystem/allowed-roots
- DocumentsPanel UI: upload, status tracking (PENDING → INDEXING → INDEXED), settings
- Real-time status polling, delete documents, manage allowed directories

#### RAG Integration
- Extended retrieve_context() with source filtering (memory/documents/seed)
- Documents indexed alongside memory cards in vector store
- Chat queries automatically search documents
- **Tested:** Receipt PDF uploaded, indexed (1 chunk), authorization ID successfully retrieved via chat

**See:** `docs/room2-phase2a-implementation.md` for full technical details

### Phase 2a+ (completed since initial Phase 2a)

- SQLite fully eliminated — all storage on PostgreSQL (memory cards, sessions, mind file, documents)
- Client-side local folder scan via Tauri FS plugin — native folder picker, recursive scan, preview with checkboxes, batch upload to Axiom Core
- Server-side directory scan endpoint (`/documents/scan`) for Axiom Core filesystem
- DocumentType enum fix — all file extensions now match correctly (md, jpg, jpeg, png)

### Phase 2b: Persistent Vector Store + Vision — IN PROGRESS

#### Completed
- **pgvector migration** — replaced in-memory VectorStore with PostgreSQL pgvector extension (HNSW index, vector_ip_ops). 73 embeddings persisted across restarts. Startup reindex recovers any missing document chunks automatically.
- **LLM proxy elimination** — removed main_vllm.py (port 8080). server.py now calls vLLM directly on 8081. One fewer process, one fewer port.
- **Async RAG pipeline** — all retriever/vector store functions converted to async for pgvector compatibility.

- **Vision model (Florence-2 Large)** — semantic image captioning + OCR via Microsoft Florence-2 (770M params, MIT license). Runs on CPU alongside vLLM (GPU). Generates detailed scene descriptions and text extraction for uploaded images. Tesseract retained as fallback. New `/vision/caption` endpoint. `image_caption` and `ocr_text` fields populated on document chunks.

#### Remaining
- Neo4j graph sync: extract entities/relationships from documents
- Auto memory proposal from document insights
- Advanced chunking: heading-aware, better context preservation
- Filesystem watcher: auto-reindex on file changes
- Document similarity search via embeddings

### Phase 2b-hardening: Foundation Hardening — IN PROGRESS

Before adding new features, harden the platform so new capabilities don't
cause failures or require refactoring. Every decision is guided by the
ontology validation design (`docs/design-ontology-validation.md`).

- [x] Centralized config (`core/config.py`) — Pydantic BaseSettings, includes ontology settings
- [x] Replace silent failures with structured logging — 30+ bare `except: pass` blocks eliminated
- [ ] Split server.py into routers — graph.py, validation.py, ontology.py (future), etc.
- [ ] Health endpoint + startup hardening — PostgreSQL, Neo4j, vLLM, pgvector checks
- [ ] Integration tests for critical paths — chat+RAG, document pipeline, graph sync, validation

### Phase 2c: External Connectors (future)
- MCP client for MS 365 (Graph API — pull emails, documents, calendar)
- MCP client for Google Workspace (Drive, Docs, Sheets)
- MCP client for GitHub (repos, issues, PRs)
- All connectors follow Zone 2 privacy model: data comes IN, nothing goes OUT

### Privacy Zones (guiding principle for Room 2)
- **Zone 1 — Fully Private:** Data on your hardware, never leaves. Core promise. (MCP filesystem)
- **Zone 2 — User-Controlled External:** User authorizes pulling THEIR data from external services INTO the private system. Data flows inward only. (MS 365, Google, GitHub connectors)
- **Zone 3 — Breaks Promise:** Data sent to external services for processing. NEVER do this.

---

## Room 3 — Planned: Ontology-Backed Validation

The graph earns its place as a **structural reasoning engine**, not a visualization tool.
Full design: `docs/design-ontology-validation.md`

### Level 2: Ontology Coverage Check
- Import curated FIBO subgraphs into `noesis` as `:OntologyNode` (separate from user `:Concept` nodes)
- Embed ontology node names in pgvector (`source_type='ontology'`) for semantic matching
- `/validate/coverage` endpoint: map response concepts → ontology region → find gaps
- Structured gap report: matched, missing, novel concepts + coverage score
- Frontend: coverage panel alongside existing claim validation

### Level 3: Multi-Ontology Intelligence
- Ontology registry (FIBO, SNOMED CT, ICD-10, LOINC, custom)
- Auto-detect domain from query → select appropriate ontology
- Composite validation: personal KG + domain ontology
- Feedback loop: missing concepts → proposed memory cards → enriches personal KG

### Agent Workflows
- Agent workflows with LangGraph (visible reasoning)
- Research agents with tool use (ReAct pattern)
- Memory curation agents
- MCP server (expose Noesis capabilities to other AI systems)

---

## Room 4 — Future

- Visual simulations / game-like modes
- Animated avatar (TTS/STT + talking head generation — separate from vision/captioning)

---

## Business Context

### Target Markets
- **Law firms** — document-heavy, privacy-critical, budget for $15-25K systems
- **Medical offices** — HIPAA compliance makes local-first essential
- **SMBs with compliance needs** — financial services, government contractors
- **Developer teams** — offline-capable AI coding assistant (VS Code extension, future)

### Delivery Model
- Managed private AI service: $10-15K setup + $1,500-3,000/month recurring
- Hardware tiers: Entry ($4K, 7B models), Mid ($8-12K, 70B quantized), Pro ($20-25K, full 70B)
- Each "room" adds capabilities and justifies ongoing service fees
