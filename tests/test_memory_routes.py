"""Integration tests for memory card API routes.

Uses FastAPI's TestClient with mocked PostgreSQL/Neo4j — no real infrastructure needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from core.api.shared import MEMORY_CARDS
from tests.conftest import make_memory_card


def _make_app():
    """Create a fresh FastAPI app for testing, with startup events skipped."""
    from core.api.routes.memory import router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client():
    app = _make_app()
    return TestClient(app)


class TestProposeMemory:
    """POST /memory/propose"""

    def test_propose_creates_card(self, client):
        with patch("core.api.routes.memory.upsert_card", new_callable=AsyncMock):
            resp = client.post("/memory/propose", json={
                "text": "Privacy is non-negotiable.",
                "category": "principles_values",
                "reason": "User stated explicitly.",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Privacy is non-negotiable."
        assert data["category"] == "principles_values"
        assert data["approval"] == "pending"
        assert data["id"]  # UUID assigned

    def test_propose_adds_to_shared_state(self, client):
        with patch("core.api.routes.memory.upsert_card", new_callable=AsyncMock):
            resp = client.post("/memory/propose", json={
                "text": "Test card",
                "category": "preferences",
                "reason": "Testing",
            })
        card_id = resp.json()["id"]
        assert card_id in MEMORY_CARDS

    def test_propose_with_all_fields(self, client):
        with patch("core.api.routes.memory.upsert_card", new_callable=AsyncMock):
            resp = client.post("/memory/propose", json={
                "text": "Full card",
                "category": "cognitive_framing",
                "scope": "global",
                "reason": "Testing all fields",
                "derived_from_artifact_ids": ["chat-abc123"],
                "tools_used": ["auto_propose"],
                "model": "test-model",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["scope"] == "global"
        assert data["provenance"]["model"] == "test-model"

    def test_propose_rejects_empty_text(self):
        """Empty text passes request validation but fails MemoryCard min_length=1."""
        from pydantic import ValidationError
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/memory/propose", json={
            "text": "",
            "category": "preferences",
            "reason": "This should fail",
        })
        assert resp.status_code == 500

    def test_propose_strips_whitespace(self, client):
        with patch("core.api.routes.memory.upsert_card", new_callable=AsyncMock):
            resp = client.post("/memory/propose", json={
                "text": "  Trimmed text  ",
                "category": "preferences",
                "reason": "  Trimmed reason  ",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Trimmed text"
        assert data["provenance"]["reason"] == "Trimmed reason"


class TestListCards:
    """GET /memory/cards"""

    def test_empty_list(self, client):
        resp = client.get("/memory/cards")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_lists_all_cards(self, client):
        card1 = make_memory_card(text="Card one")
        card2 = make_memory_card(text="Card two")
        MEMORY_CARDS[card1.id] = card1
        MEMORY_CARDS[card2.id] = card2

        resp = client.get("/memory/cards")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_filter_by_approval(self, client):
        pending = make_memory_card(text="Pending")
        approved = make_memory_card(text="Approved", approval="approved")
        MEMORY_CARDS[pending.id] = pending
        MEMORY_CARDS[approved.id] = approved

        resp = client.get("/memory/cards?approval=pending")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["text"] == "Pending"

    def test_filter_by_category(self, client):
        pv = make_memory_card(text="Value", category="principles_values")
        pref = make_memory_card(text="Preference", category="preferences")
        MEMORY_CARDS[pv.id] = pv
        MEMORY_CARDS[pref.id] = pref

        resp = client.get("/memory/cards?category=preferences")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["text"] == "Preference"

    def test_search_by_text(self, client):
        card = make_memory_card(text="Privacy matters most")
        MEMORY_CARDS[card.id] = card

        resp = client.get("/memory/cards?q=privacy")
        assert len(resp.json()) == 1

        resp = client.get("/memory/cards?q=nonexistent")
        assert len(resp.json()) == 0

    def test_newest_first_ordering(self, client):
        import time
        card1 = make_memory_card(text="Older")
        time.sleep(0.01)
        card2 = make_memory_card(text="Newer")
        MEMORY_CARDS[card1.id] = card1
        MEMORY_CARDS[card2.id] = card2

        resp = client.get("/memory/cards")
        data = resp.json()
        assert data[0]["text"] == "Newer"
        assert data[1]["text"] == "Older"


class TestApproveReject:
    """POST /memory/cards/{card_id}/approve and /reject"""

    def test_approve_card(self, client):
        card = make_memory_card(text="Approve me")
        MEMORY_CARDS[card.id] = card

        with patch("core.api.routes.memory.upsert_card", new_callable=AsyncMock), \
             patch("core.api.routes.memory.graph_queries.sync_memory_card", new_callable=AsyncMock), \
             patch("core.api.routes.memory.entry_exists_for_card", new_callable=AsyncMock, return_value=True):
            resp = client.post(f"/memory/cards/{card.id}/approve")

        assert resp.status_code == 200
        data = resp.json()
        assert data["approval"] == "approved"
        assert data["approved_at"] is not None

    def test_reject_card(self, client):
        card = make_memory_card(text="Reject me")
        MEMORY_CARDS[card.id] = card

        with patch("core.api.routes.memory.upsert_card", new_callable=AsyncMock):
            resp = client.post(f"/memory/cards/{card.id}/reject")

        assert resp.status_code == 200
        data = resp.json()
        assert data["approval"] == "rejected"
        assert data["rejected_at"] is not None

    def test_approve_nonexistent_returns_404(self, client):
        resp = client.post("/memory/cards/nonexistent-id/approve")
        assert resp.status_code == 404

    def test_reject_nonexistent_returns_404(self, client):
        resp = client.post("/memory/cards/nonexistent-id/reject")
        assert resp.status_code == 404

    def test_double_approve_is_idempotent(self, client):
        card = make_memory_card(text="Approve once")
        MEMORY_CARDS[card.id] = card

        with patch("core.api.routes.memory.upsert_card", new_callable=AsyncMock), \
             patch("core.api.routes.memory.graph_queries.sync_memory_card", new_callable=AsyncMock), \
             patch("core.api.routes.memory.entry_exists_for_card", new_callable=AsyncMock, return_value=True):
            resp1 = client.post(f"/memory/cards/{card.id}/approve")
            resp2 = client.post(f"/memory/cards/{card.id}/approve")

        assert resp1.json()["approval"] == "approved"
        assert resp2.json()["approval"] == "approved"


class TestDevSeed:
    """POST /dev/seed"""

    def test_seed_creates_two_cards(self, client):
        with patch("core.api.routes.memory.upsert_card", new_callable=AsyncMock):
            resp = client.post("/dev/seed")

        data = resp.json()
        assert data["count"] == 2
        assert len(data["seeded"]) == 2
        assert len(MEMORY_CARDS) == 2

    def test_seed_is_idempotent(self, client):
        with patch("core.api.routes.memory.upsert_card", new_callable=AsyncMock):
            resp1 = client.post("/dev/seed")
            resp2 = client.post("/dev/seed")

        assert resp1.json()["count"] == 2
        assert resp2.json()["count"] == 0
        assert len(resp2.json()["skipped"]) == 2
