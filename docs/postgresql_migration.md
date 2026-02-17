# PostgreSQL Migration Guide

## Context

Noesis is designed to run from multiple desktop clients (Mac Studio, laptop, etc.) connecting to **Axiom Core** (Linux workstation at 192.168.4.25).

**Problem with SQLite**: Data was stored locally on whichever machine ran `server.py`, making it inaccessible from other clients.

**Solution**: Centralize all data in **PostgreSQL** on the workstation. This ensures:
- ✅ Data accessible from any client machine
- ✅ Single source of truth for all users
- ✅ Proper multi-client architecture

---

## Architecture

### Before (SQLite)
```
Desktop (Mac Studio)
├── server.py (port 8008)
└── data/memory/memory_cards.sqlite3  ❌ Local only!
```

### After (PostgreSQL)
```
Desktop (Mac Studio, Laptop, etc.)
└── server.py (port 8008) ──────┐
                                 │
Desktop (Laptop)                 │
└── server.py (port 8008) ───────┼──> PostgreSQL (192.168.4.25:5433)
                                 │    ├── memory_cards
                                 │    ├── chat_sessions
                                 └────├── documents
                                      └── document_chunks
```

---

## Setup

### 1. PostgreSQL Configuration

**On Axiom Core (192.168.4.25):**

```bash
# PostgreSQL should already be running in Docker
docker ps | grep postgres

# If not, start it:
docker run -d \
  --name postgres-noesis \
  -e POSTGRES_DB=noesis \
  -e POSTGRES_USER=noesis_user \
  -e POSTGRES_PASSWORD=noesis_pass \
  -p 5433:5432 \
  -v /home/exx/data/postgres:/var/lib/postgresql/data \
  postgres:16
```

### 2. Environment Variables

**On Axiom Core**, create/update `.env`:

```bash
# PostgreSQL connection
POSTGRES_HOST=192.168.4.25
POSTGRES_PORT=5433
POSTGRES_DB=noesis
POSTGRES_USER=noesis_user
POSTGRES_PASSWORD=noesis_pass

# Neo4j (existing)
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASS=your_password
NEO4J_DATABASE=noesis
```

### 3. Install Dependencies

```bash
cd /home/exx/myCode/genai-agent-core
conda activate genai-core
pip install asyncpg==0.30.0
```

### 4. Initialize Database Schema

```python
# Run once to create tables
from core.db.postgres import init_database
import asyncio

asyncio.run(init_database())
```

---

## Migration

### Run Migration Script

**On Axiom Core:**

```bash
cd /home/exx/myCode/genai-agent-core
python scripts/migrate_to_postgres.py
```

This will:
1. Create PostgreSQL schema (tables, indexes)
2. Copy all memory cards from SQLite → PostgreSQL
3. Copy all chat sessions from SQLite → PostgreSQL
4. Verify migration by counting records

### Verify Migration

```sql
-- Connect to PostgreSQL
psql -h 192.168.4.25 -p 5433 -U noesis_user -d noesis

-- Check data
SELECT COUNT(*) FROM memory_cards;
SELECT COUNT(*) FROM chat_sessions;
SELECT COUNT(*) FROM documents;
```

---

## Code Changes

### Update Imports in server.py

**Before:**
```python
from core.artifacts.storage_sqlite import load_all_cards, upsert_card
from core.artifacts.storage_sessions import save_session, load_session, list_sessions
```

**After:**
```python
from core.artifacts.storage_memory_pg import load_all_cards, upsert_card
from core.artifacts.storage_sessions_pg import save_session, load_session, list_sessions
from core.artifacts.storage_documents_pg import save_document, load_document, load_all_documents
from core.db.postgres import init_database, close_pool
```

### Update Startup/Shutdown Hooks

```python
@app.on_event("startup")
async def _startup():
    global MEMORY_CARDS

    # Initialize PostgreSQL
    await init_database()

    # Load from PostgreSQL (not SQLite)
    MEMORY_CARDS = await load_all_cards()

    # ... existing Neo4j init

@app.on_event("shutdown")
async def _shutdown():
    await close_driver()  # Neo4j
    await close_pool()    # PostgreSQL - NEW
```

---

## Storage Module Mapping

| Data Type | SQLite (Old) | PostgreSQL (New) |
|-----------|-------------|------------------|
| Memory Cards | `storage_sqlite.py` | `storage_memory_pg.py` |
| Chat Sessions | `storage_sessions.py` | `storage_sessions_pg.py` |
| Documents | ~~`storage_documents.py`~~ | `storage_documents_pg.py` |

---

## Rollback Plan

If migration fails, restore SQLite:

1. Keep old imports in `server.py`
2. SQLite files remain at `data/memory/*.sqlite3`
3. No data loss - migration script doesn't delete SQLite files

---

## Testing

### Test from Multiple Clients

1. **On Mac Studio:**
   ```bash
   cd /home/exx/myCode/genai-agent-core
   uvicorn core.api.server:app --host 0.0.0.0 --port 8008
   ```

2. **On Laptop (connect to same server):**
   - Frontend connects to `http://192.168.4.25:8008`
   - Upload a document
   - Create a memory card

3. **Back to Mac Studio:**
   - Restart server
   - Verify document and memory card are still there (from PostgreSQL!)

---

## Benefits

✅ **Multi-client access**: Data available from any machine
✅ **Centralized**: Single source of truth
✅ **Scalable**: PostgreSQL handles concurrent writes
✅ **Backup-friendly**: One database to backup
✅ **ACID compliance**: Better data integrity

---

## Next Steps

1. ✅ Run migration script
2. ✅ Update server.py imports
3. ✅ Test from multiple clients
4. ✅ Backup old SQLite files
5. ⏸️ (Optional) Delete SQLite files after confirming migration
