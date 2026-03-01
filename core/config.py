"""Centralized configuration for Noesis / Axiom Core.

All settings are read from environment variables (with .env file support).
Import `settings` from this module — never call os.getenv() directly.

Usage:
    from core.config import settings

    pool = await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        ...
    )
"""

from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ── PostgreSQL ──────────────────────────────────────────────
    postgres_host: str = "192.168.4.25"
    postgres_port: int = 5433
    postgres_db: str = "noesis"
    postgres_user: str = "noesis_user"
    postgres_password: str = "noesis_pass"
    postgres_pool_min: int = 2
    postgres_pool_max: int = 10
    postgres_command_timeout: int = 60

    # ── Neo4j ───────────────────────────────────────────────────
    neo4j_uri: str = "bolt://192.168.4.25:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "ZGRkXDGr9wcRbIs6yLg9yPwpPnDNYKzp"
    neo4j_database: str = "noesis"

    # ── LLM Inference (Ollama or vLLM — both OpenAI-compatible) ─
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_model_name: str = "axiom-primary"
    llm_model_display: str = "Qwen3.5-35B-A3B"
    llm_timeout: float = 180.0


    # ── Embedding Model ─────────────────────────────────────────
    noesis_embed_model: str = "nomic-ai/nomic-embed-text-v1.5"

    # ── Reranker ────────────────────────────────────────────────
    noesis_use_reranker: bool = True
    noesis_reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # ── Vision Model ────────────────────────────────────────────
    noesis_vision_model: str = "microsoft/Florence-2-large"

    # ── Document Ingestion ──────────────────────────────────────
    chunk_size: int = 2000
    chunk_overlap: int = 500
    max_upload_size_mb: int = 50

    # ── Ontology ─────────────────────────────────────────────────
    ontology_import_batch_size: int = 1000
    ontology_embed_batch_size: int = 256
    coverage_threshold: float = 0.6
    snomed_rf2_dir: str = "/home/exx/data_sdb/ontologies/snomed/SnomedCT_ManagedServiceUS_PRODUCTION_US1000124_20250901T120000Z/Snapshot"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton — import this everywhere
settings = Settings()
