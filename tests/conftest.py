"""Shared fixtures for Noesis tests.

Strategy: mock all external infrastructure (PostgreSQL, Neo4j, vLLM)
so tests run locally on the Mac without needing the Axiom Core workstation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pytest_asyncio

from core.artifacts.memory_card import (
    MemoryApproval,
    MemoryCard,
    MemoryCategory,
    MemoryProvenance,
    MemoryScope,
)


@pytest.fixture(autouse=True)
def _clear_memory_cards():
    """Clear shared MEMORY_CARDS before each test."""
    from core.api.shared import MEMORY_CARDS
    MEMORY_CARDS.clear()
    yield
    MEMORY_CARDS.clear()


def make_memory_card(
    text: str = "Test insight",
    category: MemoryCategory = MemoryCategory.PRINCIPLES_VALUES,
    approval: MemoryApproval = MemoryApproval.PENDING,
) -> MemoryCard:
    """Factory for test memory cards."""
    return MemoryCard(
        text=text,
        category=category,
        scope=MemoryScope.PROJECT,
        approval=approval,
        provenance=MemoryProvenance(
            reason="test reason",
            derived_from_artifact_ids=[],
            tools_used=["test"],
            model=None,
            sources=[],
        ),
    )


# ---------------------------------------------------------------------------
# Mock PostgreSQL pool
# ---------------------------------------------------------------------------

class MockPool:
    """Minimal asyncpg pool mock."""

    async def fetchval(self, query, *args):
        if query.strip().startswith("SELECT 1"):
            return 1
        if "count(*)" in query.lower():
            return 42
        return None

    async def fetch(self, query, *args):
        return []

    async def execute(self, query, *args):
        return "OK"

    async def acquire(self):
        return MockConnection()

    async def close(self):
        pass


class MockConnection:
    """Minimal asyncpg connection mock for `async with pool.acquire()`."""

    async def execute(self, query, *args):
        return "OK"

    async def fetchval(self, query, *args):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_pg_pool(monkeypatch):
    """Patch get_pool to return a MockPool (no real PostgreSQL needed)."""
    pool = MockPool()

    async def _get_pool():
        return pool

    monkeypatch.setattr("core.db.postgres.get_pool", _get_pool)
    monkeypatch.setattr("core.db.postgres._pool", pool)
    return pool


# ---------------------------------------------------------------------------
# Mock Neo4j
# ---------------------------------------------------------------------------

class MockNeo4jSession:
    async def run(self, query, **kwargs):
        return MockNeo4jResult()

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class MockNeo4jResult:
    async def single(self):
        return {"n": 1}

    async def data(self):
        return []


@pytest.fixture
def mock_neo4j(monkeypatch):
    """Patch Neo4j to avoid real connections."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _mock_get_session():
        yield MockNeo4jSession()

    monkeypatch.setattr("core.graph.neo4j_client.get_session", _mock_get_session)


# ---------------------------------------------------------------------------
# Mock vLLM
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vllm(monkeypatch):
    """Patch ask_llm to return a canned response (no real vLLM needed)."""

    async def _fake_ask_llm(question, temperature=0.7, max_tokens=2000):
        return {
            "answer": "This is a test response from the mock LLM.",
            "model": "mock-model",
            "processing_time": 0.1,
        }

    monkeypatch.setattr("core.api.shared.ask_llm", _fake_ask_llm)
    return _fake_ask_llm
