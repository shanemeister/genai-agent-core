"""Health check probes for all Noesis infrastructure dependencies.

Each checker returns (ok: bool, detail: str) — fast, non-destructive, read-only.
"""

from __future__ import annotations

import logging
import time

import httpx

from core.config import settings

log = logging.getLogger("noesis.health")


async def check_postgres() -> tuple[bool, str]:
    """Ping PostgreSQL with SELECT 1."""
    try:
        from core.db.postgres import get_pool
        pool = await get_pool()
        t0 = time.monotonic()
        row = await pool.fetchval("SELECT 1")
        ms = round((time.monotonic() - t0) * 1000)
        if row == 1:
            return True, f"ok ({ms}ms)"
        return False, f"unexpected result: {row}"
    except Exception as e:
        return False, str(e)


async def check_pgvector() -> tuple[bool, str]:
    """Verify pgvector extension is installed and embeddings table exists."""
    try:
        from core.db.postgres import get_pool
        pool = await get_pool()
        t0 = time.monotonic()
        count = await pool.fetchval("SELECT count(*) FROM embeddings")
        ms = round((time.monotonic() - t0) * 1000)
        return True, f"ok ({count} embeddings, {ms}ms)"
    except Exception as e:
        return False, str(e)


async def check_neo4j() -> tuple[bool, str]:
    """Verify Neo4j connectivity."""
    try:
        from core.graph.neo4j_client import get_session
        t0 = time.monotonic()
        async with get_session() as session:
            result = await session.run("RETURN 1 AS n")
            record = await result.single()
            ms = round((time.monotonic() - t0) * 1000)
            if record and record["n"] == 1:
                return True, f"ok ({ms}ms)"
            return False, "unexpected result"
    except Exception as e:
        return False, str(e)


async def check_llm() -> tuple[bool, str]:
    """Check LLM inference server is reachable via /v1/models (no GPU load)."""
    try:
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{settings.llm_base_url}/v1/models")
            ms = round((time.monotonic() - t0) * 1000)
            if resp.status_code == 200:
                data = resp.json()
                models = [m["id"] for m in data.get("data", [])]
                return True, f"ok ({', '.join(models)}, {ms}ms)"
            return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)


async def run_all_checks() -> dict:
    """Run all health checks and return structured result.

    Returns:
        {
            "status": "healthy" | "degraded" | "unhealthy",
            "checks": {
                "postgres": {"ok": bool, "detail": str},
                "pgvector": {"ok": bool, "detail": str},
                "neo4j":    {"ok": bool, "detail": str},
                "llm":      {"ok": bool, "detail": str},
            }
        }

    - healthy:  all checks pass
    - degraded: postgres OK but something else is down (server can partially function)
    - unhealthy: postgres is down (nothing works without it)
    """
    pg_ok, pg_detail = await check_postgres()
    pgv_ok, pgv_detail = await check_pgvector()
    neo_ok, neo_detail = await check_neo4j()
    llm_ok, llm_detail = await check_llm()

    checks = {
        "postgres": {"ok": pg_ok, "detail": pg_detail},
        "pgvector": {"ok": pgv_ok, "detail": pgv_detail},
        "neo4j": {"ok": neo_ok, "detail": neo_detail},
        "llm": {"ok": llm_ok, "detail": llm_detail},
    }

    all_ok = all(c["ok"] for c in checks.values())
    if all_ok:
        status = "healthy"
    elif pg_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return {"status": status, "checks": checks}
