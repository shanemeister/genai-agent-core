# Architecture

## System Overview

Noesis is a two-machine system communicating over LAN. No internet required.

```
┌─────────────────────────────┐         ┌──────────────────────────────────────┐
│   Noesis (Mac 192.168.4.22) │   LAN   │   Axiom Core (Linux 192.168.4.25)   │
│                             │ ──────> │                                      │
│   Tauri + React + Vite      │  REST   │   FastAPI + vLLM + Neo4j + Postgres  │
│   Port 1420 (dev)           │  + SSE  │   Ports 8008, 8081, 7687, 5433       │
└─────────────────────────────┘         └──────────────────────────────────────┘
```

## Layers

1. `app/` — Tauri desktop shell and visual workspace (React + TypeScript).
2. `core/` — Python backend: two FastAPI services, LLM integration, graph, RAG, memory, validation.
3. `data/` — Local persistence for vectors, document uploads, and settings.

## Backend Services (Axiom Core)

Two processes run on Axiom Core:

| Service | Port | Entry Point | Role |
|---------|------|-------------|------|
| **Noesis Services** | 8008 | `core/api/server.py` | All backend features: streaming chat, memory, RAG, graph, diagrams, validation, mind file, documents. Calls vLLM directly. |
| **vLLM** | 8081 | vLLM OpenAI API server | DeepSeek-R1-70B inference (tensor-parallel, 2x A6000) |

## Runtime Flow

1. User sends message from `ChatPanel` (React).
2. Noesis hits `POST /chat/stream` on port 8008 (SSE connection).
3. `server.py` retrieves context from approved memory cards via RAG (`core/rag/`).
4. `server.py` calls vLLM (port 8081) with context-augmented prompt.
5. vLLM streams tokens back; `server.py` re-streams via SSE to Noesis.
6. DeepSeek-R1 `<think>` tokens are parsed client-side for the thinking indicator.
7. After response completes, `server.py` calls `_propose_memories_for_session()`.
8. Proposed memory cards appear in Memory Deck as pending.
9. When user approves a card, it is auto-synced to Neo4j graph and promoted to Mind File.
10. User can optionally click "Validate" to run claim-level evidence checking on any response.

## Data Flow

```
User Message
    │
    ▼
ChatPanel.tsx ──SSE──> server.py (8008)
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              RAG Retrieval    vLLM (8081)
              (vector_store)   DeepSeek-R1-70B
                    │             │
                    └──────┬──────┘
                           ▼
                    Streaming Response
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              Memory Proposer  Diagram Generator
              (auto-propose)   (if requested)
                    │             │
                    ▼             ▼
              Memory Deck     Diagram Canvas
              (pending cards) (Mermaid render)
                    │
                    ▼ (on approve)
              ┌─────┴─────┐
              ▼           ▼
        Neo4j Graph   Mind File
        (concept sync) (curated entry)
```

### Validation Flow (on-demand)

```
User clicks "Validate" on a message
    │
    ▼
POST /validate { response_text, user_question }
    │
    ├─── claim_extractor.py: LLM extracts 3-8 atomic claims
    │
    ├─── claim_validator.py (per claim):
    │       ├── retrieve_context(claim, k=3) — RAG evidence
    │       ├── extract_concepts(claim) → search_nodes() — graph coverage
    │       └── determine status: supported | unsupported | contradicted | unverifiable
    │
    └─── ValidationResult returned to frontend
         └── Displayed as expandable claim-by-claim panel
```

### Graph Scoping Model

The Graph Explorer supports scoped queries to show focused, bounded subgraphs:

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `scope` | session, question, artifact | What to center the graph on |
| `view_mode` | provenance, lineage, full | Which relationship types to traverse |
| `depth` | 1, 2, 3 | How many hops from the center node |

**Relationship filters by view_mode:**
- **provenance**: USED_CONTEXT, MENTIONS, ABOUT, CURATED_FROM — "where did this come from?"
- **lineage**: PRODUCED, PROPOSED, DISCUSSES — "what did this create?"
- **full**: union of both

## Database Architecture

| Database | Location | Port | Purpose |
|----------|----------|------|---------|
| **PostgreSQL** | Axiom Core (Docker: numerai-postgres) | 5433 | Memory cards, chat sessions, documents, document chunks (primary storage, multi-client) |
| **Neo4j** | Axiom Core | 7687 | Knowledge graph (database: `noesis`) |
| **pgvector** | PostgreSQL extension (embeddings table) | 5433 | Persistent vector embeddings for RAG (nomic-embed-text-v1.5, 768d, HNSW index) |
| ~~SQLite~~ | `data/memory/memory_cards.sqlite3` | N/A | Deprecated — migrated to PostgreSQL in Phase 2a |

## Frontend Components

| Component | File | Role |
|-----------|------|------|
| `App.tsx` | Root | Tab management (chat/memory/diagram/graph/mindfile/documents), app-level zoom |
| `ChatPanel.tsx` | Chat | SSE streaming, thinking indicator, RAG context display, on-demand validation |
| `MemoryDeck.tsx` | Memory | Card listing, approve/reject workflow, pending proposals |
| `DiagramCanvas.tsx` | Diagrams | Mermaid source editor + live SVG preview |
| `GraphExplorer.tsx` | Graph | d3-force visualization, zoom/pan, node detail panel, scoped queries |
| `MindFile.tsx` | Mind File | Curated entries, timeline, cognitive profile |
| `DocumentsPanel.tsx` | Documents | Upload, status tracking, filesystem browsing, settings |

## Key API Endpoints (Port 8008 — Noesis Services)

### Chat & Streaming
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat/stream` | POST | SSE streaming chat with RAG + memory proposals |

### Memory
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/memory/cards` | GET | List all memory cards |
| `/memory/card` | POST | Create/update memory card |
| `/memory/card/{id}/approve` | POST | Approve pending card (auto-syncs to Neo4j + Mind File) |
| `/memory/card/{id}/reject` | POST | Reject pending card |

### Mind File
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/mindfile` | GET | List mind file entries (with optional category filter) |
| `/mindfile/{id}/note` | PUT | Update annotation on an entry |
| `/mindfile/{id}` | DELETE | Remove entry from mind file |
| `/mindfile/stats` | GET | Category breakdown and date range |
| `/mindfile/export` | GET | Export mind file as markdown |
| `/mindfile/timeline/{concept}` | GET | Temporal evolution of a concept |
| `/mindfile/cognitive-profile` | GET | Aggregate cognitive profile |

### Graph
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/graph/data` | GET | Full graph for initial load |
| `/graph/scoped` | GET | Scoped subgraph (scope, view_mode, depth) |
| `/graph/session/{id}` | GET | Session subgraph (legacy, uses scoped internally) |
| `/graph/neighbors` | POST | Click-to-expand node neighbors |
| `/graph/search` | GET | Text search across graph nodes |
| `/graph/concepts` | POST | Create concept manually |
| `/graph/relationships` | POST | Create edge |
| `/graph/stats` | GET | Node/edge counts |
| `/graph/sync-memories` | POST | Bulk sync approved memory cards to graph |
| `/graph/seed` | POST | Seed demo data |

### Validation
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/validate` | POST | On-demand claim-level evidence checking |

### Sessions
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/sessions` | GET | List saved sessions |
| `/sessions/{id}` | GET | Load a session |
| `/sessions/{id}` | DELETE | Delete a session |
| `/sessions/save` | POST | Save current session |
| `/sessions/{id}/export` | GET | Export session as markdown |

### Documents (Phase 2a)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/documents/upload` | POST | Upload and index a document (multipart/form-data) |
| `/documents` | GET | List all documents with status |
| `/documents/{id}` | GET | Get document details |
| `/documents/{id}` | DELETE | Remove document and its chunks |
| `/documents/browse` | POST | Browse allowed filesystem directories |
| `/documents/import` | POST | Import document from filesystem path |
| `/settings/filesystem/allowed-roots` | GET | Get allowed directories |
| `/settings/filesystem/allowed-roots` | PUT | Update allowed directories |

## Key Configuration

All settings are centralized in `core/config.py` via Pydantic `BaseSettings`,
loaded from `.env` file on Axiom Core. No `os.getenv()` calls in application code.

```python
from core.config import settings
settings.vllm_base_url    # "http://127.0.0.1:8081"
settings.postgres_host     # "192.168.4.25"
settings.neo4j_database    # "noesis"
```

| Setting | Env Var | Default | Purpose |
|---------|---------|---------|---------|
| `vllm_base_url` | `VLLM_BASE_URL` | `http://127.0.0.1:8081` | vLLM endpoint (internal) |
| `vllm_model_name` | `VLLM_MODEL_NAME` | `./models/deepseek-r1-70b-w4a16` | Model path for vLLM |
| `llm_timeout` | `LLM_TIMEOUT` | `180.0` | Timeout for LLM responses |
| `postgres_*` | `POSTGRES_*` | See .env | PostgreSQL connection |
| `neo4j_*` | `NEO4J_*` | See .env | Neo4j connection |
| `noesis_embed_model` | `NOESIS_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model |
| `noesis_use_reranker` | `NOESIS_USE_RERANKER` | `true` | Enable cross-encoder reranking |
| `noesis_vision_model` | `NOESIS_VISION_MODEL` | `microsoft/Florence-2-large` | Vision captioning model |
| `chunk_size` | `CHUNK_SIZE` | `2000` | Document chunk size (chars) |
| `chunk_overlap` | `CHUNK_OVERLAP` | `500` | Chunk overlap (chars) |
| `max_upload_size_mb` | `MAX_UPLOAD_SIZE_MB` | `50` | Max upload file size |
| `API_BASE` | — | `http://192.168.4.25:8008` | Frontend connects to Noesis Services |

## Room 2 Architecture (Phase 2a — COMPLETE)

### Multi-Client Design

```
┌─────────────┐         ┌─────────────┐
│  Mac Studio │         │   Laptop    │
│ (Frontend)  │         │ (Frontend)  │
└──────┬──────┘         └──────┬──────┘
       │                       │
       └───────────┬───────────┘
                   │ HTTP
                   ▼
           ┌───────────────┐
           │  Axiom Core   │
           │  (Backend)    │
           │  :8008        │
           └───────┬───────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
       ▼           ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │PostgreSQL│ │ Neo4j  │ │ Vector  │
  │  :5433   │ │  :7687 │ │  Store  │
  └─────────┘ └─────────┘ └─────────┘
```

### Document Ingestion Pipeline

```
Upload/Import from client or filesystem
        │
        ▼
   MCP Filesystem (privacy-controlled allowlist access)
        │
        ▼
   Document Parser (PDF/DOCX/TXT/MD/CSV/Image/SVG)
        │
        ▼
   Chunking (RecursiveCharacterTextSplitter, 2000 chars, 25% overlap)
        │
        ▼
   Embedding (nomic-embed-text-v1.5, 768d)
        │
        ▼
   Storage: pgvector (PostgreSQL embeddings table, HNSW index) + PostgreSQL (chunks + metadata)
        │
        ▼
   Existing Noesis Pipeline
   ├── retrieve_context() — searches ingested docs + approved memories
   ├── Graph extraction — entities from documents → Neo4j (Phase 2b)
   ├── Memory proposals — facts from documents → pending cards (Phase 2b)
   └── Validation — claims checked against document evidence
```

All processing runs locally on Axiom Core. Files never leave the network.
This is a Zone 1 (fully private) integration.

**See:** `docs/room2-phase2a-implementation.md` for full technical details.
