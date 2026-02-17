"""Tests for health check logic."""

import pytest
from unittest.mock import AsyncMock, patch

from core.api.health import run_all_checks


class TestHealthAggregation:
    """Test the health status aggregation logic without real infrastructure."""

    @pytest.mark.asyncio
    async def test_all_healthy(self):
        with patch("core.api.health.check_postgres", return_value=(True, "ok (1ms)")), \
             patch("core.api.health.check_pgvector", return_value=(True, "ok (42 embeddings, 2ms)")), \
             patch("core.api.health.check_neo4j", return_value=(True, "ok (5ms)")), \
             patch("core.api.health.check_vllm", return_value=(True, "ok (model, 10ms)")):
            result = await run_all_checks()

        assert result["status"] == "healthy"
        assert all(c["ok"] for c in result["checks"].values())

    @pytest.mark.asyncio
    async def test_neo4j_down_is_degraded(self):
        with patch("core.api.health.check_postgres", return_value=(True, "ok")), \
             patch("core.api.health.check_pgvector", return_value=(True, "ok")), \
             patch("core.api.health.check_neo4j", return_value=(False, "connection refused")), \
             patch("core.api.health.check_vllm", return_value=(True, "ok")):
            result = await run_all_checks()

        assert result["status"] == "degraded"
        assert result["checks"]["postgres"]["ok"] is True
        assert result["checks"]["neo4j"]["ok"] is False

    @pytest.mark.asyncio
    async def test_vllm_down_is_degraded(self):
        with patch("core.api.health.check_postgres", return_value=(True, "ok")), \
             patch("core.api.health.check_pgvector", return_value=(True, "ok")), \
             patch("core.api.health.check_neo4j", return_value=(True, "ok")), \
             patch("core.api.health.check_vllm", return_value=(False, "timeout")):
            result = await run_all_checks()

        assert result["status"] == "degraded"
        assert result["checks"]["vllm"]["ok"] is False

    @pytest.mark.asyncio
    async def test_postgres_down_is_unhealthy(self):
        with patch("core.api.health.check_postgres", return_value=(False, "connection refused")), \
             patch("core.api.health.check_pgvector", return_value=(False, "no connection")), \
             patch("core.api.health.check_neo4j", return_value=(True, "ok")), \
             patch("core.api.health.check_vllm", return_value=(True, "ok")):
            result = await run_all_checks()

        assert result["status"] == "unhealthy"
        assert result["checks"]["postgres"]["ok"] is False

    @pytest.mark.asyncio
    async def test_everything_down_is_unhealthy(self):
        with patch("core.api.health.check_postgres", return_value=(False, "down")), \
             patch("core.api.health.check_pgvector", return_value=(False, "down")), \
             patch("core.api.health.check_neo4j", return_value=(False, "down")), \
             patch("core.api.health.check_vllm", return_value=(False, "down")):
            result = await run_all_checks()

        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_checks_include_detail_strings(self):
        with patch("core.api.health.check_postgres", return_value=(True, "ok (1ms)")), \
             patch("core.api.health.check_pgvector", return_value=(True, "ok (100 embeddings, 2ms)")), \
             patch("core.api.health.check_neo4j", return_value=(False, "bolt://192.168.4.25:7687 unreachable")), \
             patch("core.api.health.check_vllm", return_value=(True, "ok (model-name, 15ms)")):
            result = await run_all_checks()

        assert "1ms" in result["checks"]["postgres"]["detail"]
        assert "100 embeddings" in result["checks"]["pgvector"]["detail"]
        assert "unreachable" in result["checks"]["neo4j"]["detail"]
