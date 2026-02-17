"""Noesis API — FastAPI application entry point.

Thin shell: startup/shutdown lifecycle, CORS, logging config, health checks,
LLM test endpoint. All domain endpoints live in core/api/routes/*.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.db.postgres import init_database, close_pool
from core.graph.neo4j_client import close_driver, init_driver
from core.graph.schema import ensure_schema
from core.artifacts.storage_memory_pg import load_all_cards

from core.api.shared import MEMORY_CARDS, ask_llm
from core.api.health import run_all_checks

# Routers
from core.api.routes.memory import router as memory_router
from core.api.routes.chat import router as chat_router
from core.api.routes.graph import router as graph_router
from core.api.routes.sessions import router as sessions_router
from core.api.routes.mindfile import router as mindfile_router
from core.api.routes.documents import router as documents_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("noesis")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Noesis API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(memory_router)
app.include_router(chat_router)
app.include_router(graph_router)
app.include_router(sessions_router)
app.include_router(mindfile_router)
app.include_router(documents_router)

# Startup readiness — tracks which subsystems initialized successfully
_ready: dict[str, bool] = {
    "postgres": False,
    "neo4j": False,
}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    t0 = time.monotonic()

    # ── PostgreSQL (critical — nothing works without it) ──────────
    try:
        await init_database()
        _ready["postgres"] = True
        log.info("PostgreSQL initialized")
    except Exception as e:
        log.error("PostgreSQL not available at startup: %s", e, exc_info=True)
        log.error("Server is running but most endpoints will fail without PostgreSQL")

    # Load memory cards into shared state (skip if PG failed)
    if _ready["postgres"]:
        try:
            MEMORY_CARDS.update(await load_all_cards())
            log.info("Loaded %d memory cards", len(MEMORY_CARDS))
        except Exception as e:
            log.error("Failed to load memory cards: %s", e, exc_info=True)

    # ── Neo4j (non-critical — graph endpoints degrade gracefully) ─
    try:
        await init_driver()
        await ensure_schema()
        _ready["neo4j"] = True
        log.info("Neo4j initialized")
    except Exception as e:
        log.error("Neo4j not available at startup: %s", e, exc_info=True)
        log.warning("Graph endpoints will return errors until Neo4j is available")

    # ── Background reindex (non-blocking, only if PG is up) ──────
    if _ready["postgres"]:
        asyncio.create_task(_reindex_background())

    # ── Startup summary ──────────────────────────────────────────
    elapsed = round((time.monotonic() - t0) * 1000)
    subsystems = ", ".join(f"{k}={'ok' if v else 'FAILED'}" for k, v in _ready.items())
    log.info("Startup complete in %dms — %s", elapsed, subsystems)


async def _reindex_background():
    """Background task: re-index any document chunks missing from pgvector."""
    try:
        from core.rag.retriever import reindex_document_chunks, seed_store
        await seed_store()
        count = await reindex_document_chunks()
        if count:
            log.info("Re-indexed %d document chunks into pgvector", count)
    except Exception as e:
        log.error("Background reindex failed: %s", e, exc_info=True)


@app.on_event("shutdown")
async def _shutdown():
    await close_driver()   # Neo4j
    await close_pool()     # PostgreSQL


# ---------------------------------------------------------------------------
# Health & readiness endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness():
    """Readiness probe — checks all infrastructure dependencies.

    Returns 200 if healthy/degraded, 503 if unhealthy (PostgreSQL down).
    """
    from fastapi.responses import JSONResponse

    result = await run_all_checks()
    status_code = 200 if result["status"] != "unhealthy" else 503
    return JSONResponse(content=result, status_code=status_code)


@app.get("/health/startup")
def startup_status():
    """Shows which subsystems initialized at startup."""
    all_ok = all(_ready.values())
    return {
        "ready": all_ok,
        "subsystems": _ready,
    }


# ---------------------------------------------------------------------------
# LLM test
# ---------------------------------------------------------------------------

@app.get("/llm/test")
async def test_llm():
    try:
        result = await ask_llm("Say 'Hello from the LLM!'", max_tokens=50)
        return {
            "status": "ok",
            "response": result["answer"],
            "model": result.get("model"),
            "processing_time": result.get("processing_time"),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
