"""Tests for grounding score computation."""

import pytest
from unittest.mock import patch, AsyncMock

from core.api.routes.chat import _compute_grounding, GroundingScore


class TestGroundingScore:
    """_compute_grounding produces a composite score from retrieval, coverage,
    diversity, and reasoning dimensions."""

    @pytest.mark.asyncio
    async def test_no_context_no_reasoning(self):
        with patch("core.graph.concept_extractor.extract_concepts", return_value=[]):
            score = await _compute_grounding(
                query="test query",
                context_docs=[],
                has_reasoning=False,
            )
        assert isinstance(score, GroundingScore)
        assert score.overall == 0.0
        assert score.label == "Ungrounded"
        assert score.retrieval == 0.0
        assert score.reasoning == 0.0

    @pytest.mark.asyncio
    async def test_high_retrieval_scores(self):
        docs = [
            {"doc_id": "memory:1", "text": "relevant doc", "score": 0.9},
            {"doc_id": "memory:2", "text": "another doc", "score": 0.8},
            {"doc_id": "seed:1", "text": "seed doc", "score": 0.7},
        ]
        with patch("core.graph.concept_extractor.extract_concepts", return_value=[]):
            score = await _compute_grounding(
                query="test",
                context_docs=docs,
                has_reasoning=True,
            )
        assert score.retrieval > 0.7
        assert score.reasoning == 1.0
        assert score.diversity > 0.0  # 3 unique sources

    @pytest.mark.asyncio
    async def test_labels_correct_thresholds(self):
        """Verify label thresholds: >=0.7=High, >=0.4=Medium, >0.1=Low, else Ungrounded."""
        docs = [
            {"doc_id": f"memory:{i}", "text": "doc", "score": 0.95}
            for i in range(5)
        ]
        with patch("core.graph.concept_extractor.extract_concepts", return_value=[]):
            score = await _compute_grounding("q", docs, has_reasoning=True)
        # retrieval=0.95*0.4=0.38, diversity=5/4=1.0*0.2=0.2, reasoning=0.1 => ~0.68+
        assert score.label in ("High", "Medium")

    @pytest.mark.asyncio
    async def test_dot_product_normalization(self):
        """Scores > 1.5 are treated as dot products and normalized relative to max."""
        docs = [
            {"doc_id": "memory:1", "text": "a", "score": 100.0},
            {"doc_id": "memory:2", "text": "b", "score": 50.0},
        ]
        with patch("core.graph.concept_extractor.extract_concepts", return_value=[]):
            score = await _compute_grounding("q", docs, has_reasoning=False)
        # 100/100=1.0, 50/100=0.5 => avg=0.75
        assert score.retrieval == 0.75

    @pytest.mark.asyncio
    async def test_single_source_low_diversity(self):
        docs = [
            {"doc_id": "seed:system", "text": "only one", "score": 0.8},
        ]
        with patch("core.graph.concept_extractor.extract_concepts", return_value=[]):
            score = await _compute_grounding("q", docs, has_reasoning=False)
        assert score.diversity == 0.25  # 1 source => 1/4 = 0.25

    @pytest.mark.asyncio
    async def test_detail_string_content(self):
        docs = [
            {"doc_id": "memory:1", "text": "doc", "score": 0.9},
        ]
        with patch("core.graph.concept_extractor.extract_concepts", return_value=[]):
            score = await _compute_grounding("q", docs, has_reasoning=True)
        assert "retrieval relevance" in score.detail
        assert "chain-of-thought" in score.detail
