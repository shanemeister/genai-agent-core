"""Tests for centralized configuration."""

from core.config import Settings, settings


class TestConfig:
    """Verify the Pydantic Settings singleton loads correctly."""

    def test_settings_is_singleton(self):
        """The module-level `settings` should be a Settings instance."""
        assert isinstance(settings, Settings)

    def test_defaults_are_set(self):
        """Critical defaults should exist even without .env."""
        assert settings.postgres_host
        assert settings.postgres_port > 0
        assert settings.neo4j_uri.startswith("bolt://")
        assert settings.vllm_base_url.startswith("http")

    def test_llm_timeout_is_positive(self):
        assert settings.llm_timeout > 0

    def test_chunk_settings_sane(self):
        assert settings.chunk_size > 0
        assert settings.chunk_overlap >= 0
        assert settings.chunk_overlap < settings.chunk_size

    def test_coverage_threshold_in_range(self):
        assert 0.0 <= settings.coverage_threshold <= 1.0

    def test_env_override(self, monkeypatch):
        """Settings should be overridable via environment variables."""
        monkeypatch.setenv("POSTGRES_PORT", "9999")
        fresh = Settings()
        assert fresh.postgres_port == 9999

    def test_extra_env_vars_ignored(self, monkeypatch):
        """Unknown env vars should not cause errors (extra='ignore')."""
        monkeypatch.setenv("TOTALLY_UNKNOWN_SETTING", "whatever")
        fresh = Settings()  # Should not raise
        assert isinstance(fresh, Settings)
