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

        log.info("Database schema initialized")
