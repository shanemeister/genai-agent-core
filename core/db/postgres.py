"""PostgreSQL connection pool management.

Centralized connection pooling for all PostgreSQL storage layers.
"""

import logging

import asyncpg
from pgvector.asyncpg import register_vector
from typing import Optional

from core.config import settings

log = logging.getLogger("noesis.db")

# Global connection pool
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create PostgreSQL connection pool."""
    global _pool

    if _pool is None:
        async def _init_connection(conn):
            """Register pgvector type on each new connection."""
            await register_vector(conn)

        _pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            min_size=settings.postgres_pool_min,
            max_size=settings.postgres_pool_max,
            command_timeout=settings.postgres_command_timeout,
            init=_init_connection,
        )

        log.info("Connected to %s:%d/%s", settings.postgres_host, settings.postgres_port, settings.postgres_db)

    return _pool


async def close_pool() -> None:
    """Close PostgreSQL connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        log.info("Connection pool closed")


async def init_database() -> None:
    """Initialize database schema (create tables if not exist)."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Documents table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                error_message TEXT DEFAULT '',
                tags TEXT[] DEFAULT '{}',
                notes TEXT DEFAULT ''
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_hash
            ON documents(file_hash)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_status
            ON documents(status)
        """)

        # Document chunks table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                page_number INTEGER DEFAULT 0,
                heading TEXT DEFAULT '',
                image_caption TEXT DEFAULT '',
                ocr_text TEXT DEFAULT ''
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document
            ON document_chunks(document_id, chunk_index)
        """)

        # Memory cards table (for migration)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_cards (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                category TEXT NOT NULL,
                scope TEXT NOT NULL,
                text TEXT NOT NULL,
                approval TEXT NOT NULL,
                approved_at TIMESTAMP,
                rejected_at TIMESTAMP,
                provenance JSONB NOT NULL
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_cards_approval
            ON memory_cards(approval)
        """)

        # Chat sessions table (for migration)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                messages JSONB NOT NULL
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_updated
            ON chat_sessions(updated_at DESC)
        """)

        # Mind file entries table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS mindfile_entries (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                category TEXT NOT NULL,
                text TEXT NOT NULL,
                source_memory_card_id TEXT NOT NULL,
                note TEXT DEFAULT ''
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mindfile_category
            ON mindfile_entries(category)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mindfile_source_card
            ON mindfile_entries(source_memory_card_id)
        """)

        # Embeddings table (pgvector)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                embedding vector(768) NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}',
                source_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT now()
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_source_type
            ON embeddings(source_type)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
            ON embeddings USING hnsw (embedding vector_ip_ops)
            WITH (m = 16, ef_construction = 64)
        """)

        # ── Observability: LLM call log ──────────────────────
        # Every LLM call gets a row here. This is the ground-truth
        # record for debugging, eval, cost tracking, and agent tuning.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                created_at        TIMESTAMPTZ DEFAULT NOW(),
                caller            VARCHAR(64) NOT NULL,
                session_id        VARCHAR(128),
                model             VARCHAR(128),
                prompt            TEXT,
                temperature       FLOAT,
                max_tokens        INT,
                response          TEXT,
                reasoning         TEXT,
                prompt_tokens     INT,
                completion_tokens INT,
                duration_ms       INT,
                grounding_score   FLOAT,
                tool_calls        JSONB,
                error             TEXT
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_llm_calls_created
            ON llm_calls(created_at DESC)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_llm_calls_caller
            ON llm_calls(caller)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_llm_calls_session
            ON llm_calls(session_id)
            WHERE session_id IS NOT NULL
        """)

        # ── Observability: User feedback ──────────────────────
        # Binary thumbs-up/down linked to specific LLM calls.
        # The join between feedback and llm_calls is the dataset
        # for measuring "does better prompting = happier users?"
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                llm_call_id UUID REFERENCES llm_calls(id),
                session_id  VARCHAR(128),
                rating      VARCHAR(16) NOT NULL,
                user_id     VARCHAR(128),
                context     JSONB
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_llm_call
            ON feedback(llm_call_id)
            WHERE llm_call_id IS NOT NULL
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_session
            ON feedback(session_id)
            WHERE session_id IS NOT NULL
        """)

        # ── CMS MS-DRG v42 (FY2025) weights ───────────────────
        # Source: https://www.cms.gov/medicare/payment/prospective-payment-systems/
        #         acute-inpatient-pps/fy-2025-ipps-final-rule-home-page
        # File: fy-2025-ipps-final-rule-table-5.zip
        # One row per MS-DRG (~773 rows). The triplet_base groups DRGs that
        # differ only by CC/MCC status (e.g., DRGs 291/292/293 for heart
        # failure with MCC, CC, and without). This is the key for the
        # CC/MCC impact model in the Dashboard.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS cms_drg_weights (
                drg INTEGER PRIMARY KEY,
                mdc TEXT,
                drg_type TEXT,                  -- MED, SURG, or PRE
                title TEXT NOT NULL,
                triplet_base TEXT,              -- Title with "WITH MCC/CC" stripped
                cc_mcc_status TEXT,             -- 'MCC', 'CC', 'NONE', or NULL
                weight_uncapped NUMERIC(10, 4),
                weight_capped NUMERIC(10, 4),   -- 10% cap applied
                gmlos NUMERIC(6, 2),
                alos NUMERIC(6, 2),
                post_acute BOOLEAN,
                special_pay BOOLEAN,
                fiscal_year INTEGER DEFAULT 2025,
                loaded_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_drg_triplet
            ON cms_drg_weights(triplet_base)
            WHERE triplet_base IS NOT NULL
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_drg_mdc
            ON cms_drg_weights(mdc)
        """)

        # ── CMS CC/MCC designation list ───────────────────────
        # Source: fy-2025-ipps-final-rule-tables-6a-6k-and-tables-6p1a-6p4d.zip
        # Files: Table 6I (Complete MCC List) + Table 6J (Complete CC List)
        # ~18,300 ICD-10-CM codes designated as CC or MCC.
        # A code present in both tables is rare — we only store the strongest
        # designation (MCC > CC).
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS cms_cc_mcc_codes (
                icd10_code TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                designation TEXT NOT NULL,       -- 'MCC' or 'CC'
                fiscal_year INTEGER DEFAULT 2025,
                loaded_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cc_mcc_designation
            ON cms_cc_mcc_codes(designation)
        """)

        log.info("Database schema initialized")
