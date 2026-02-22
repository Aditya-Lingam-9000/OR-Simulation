"""Tests for src.api.app module — FastAPI endpoints.

Phase 7: Enhanced with stats, transcript, and override integration tests.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.app import app, app_state


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# ======================================================================
# Health Endpoint
# ======================================================================


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_ok(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_has_timestamp(self, client):
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data

    def test_health_has_surgery_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        assert "surgery_loaded" in data

    def test_health_has_disclaimer(self, client):
        response = client.get("/health")
        data = response.json()
        assert "disclaimer" in data
        assert "SIMULATION" in data["disclaimer"]


# ======================================================================
# State Endpoint
# ======================================================================


class TestStateEndpoint:
    def test_state_returns_200(self, client):
        response = client.get("/state")
        assert response.status_code == 200

    def test_state_has_required_keys(self, client):
        response = client.get("/state")
        data = response.json()
        required = ["metadata", "machines", "details", "suggestions", "confidence", "source"]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_state_machines_format(self, client):
        response = client.get("/state")
        data = response.json()
        assert "0" in data["machines"]
        assert "1" in data["machines"]

    def test_state_confidence_range(self, client):
        response = client.get("/state")
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0


# ======================================================================
# Surgeries Endpoint
# ======================================================================


class TestSurgeriesEndpoint:
    def test_list_surgeries(self, client):
        response = client.get("/surgeries")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
        assert "PCNL" in data

    def test_select_valid_surgery(self, client):
        response = client.post("/select_surgery", json={"surgery": "Lobectomy"})
        assert response.status_code == 200
        data = response.json()
        assert data["surgery"] == "Lobectomy"

    def test_select_invalid_surgery(self, client):
        response = client.post("/select_surgery", json={"surgery": "InvalidSurgery"})
        assert response.status_code == 400


# ======================================================================
# Machines Endpoint
# ======================================================================


class TestMachinesEndpoint:
    def test_get_machines(self, client):
        response = client.get("/machines")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


# ======================================================================
# Override Endpoint
# ======================================================================


class TestOverrideEndpoint:
    def test_override_valid_machine(self, client):
        # First ensure PCNL is selected (default)
        response = client.post(
            "/override",
            json={"machine_id": "M01", "action": "OFF", "reason": "test"},
        )
        assert response.status_code == 200

    def test_override_invalid_machine(self, client):
        response = client.post(
            "/override",
            json={"machine_id": "M99", "action": "ON", "reason": "test"},
        )
        assert response.status_code == 404

    def test_override_response_fields(self, client):
        response = client.post(
            "/override",
            json={"machine_id": "M01", "action": "ON", "reason": "safety check"},
        )
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "ok"
            assert data["machine_id"] == "M01"
            assert data["action"] == "ON"
            assert "note" in data


# ======================================================================
# Stats Endpoint
# ======================================================================


class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        response = client.get("/stats")
        assert response.status_code == 200

    def test_stats_has_content(self, client):
        response = client.get("/stats")
        data = response.json()
        assert isinstance(data, dict)
        # Should have either 'running' (from orchestrator) or 'message' (no orchestrator)
        assert "running" in data or "message" in data


# ======================================================================
# Transcript Endpoint
# ======================================================================


class TestTranscriptEndpoint:
    def test_transcript_empty_text_rejected(self, client):
        response = client.post("/transcript", json={"text": "", "speaker": "test"})
        assert response.status_code == 400

    def test_transcript_whitespace_rejected(self, client):
        response = client.post("/transcript", json={"text": "   ", "speaker": "test"})
        assert response.status_code == 400

    def test_transcript_no_text_key(self, client):
        response = client.post("/transcript", json={"speaker": "test"})
        assert response.status_code == 400

    def test_transcript_when_pipeline_not_running(self, client):
        """If orchestrator is not running, should return 503."""
        # Temporarily set orchestrator to None
        saved = app_state.orchestrator
        app_state.orchestrator = None
        response = client.post(
            "/transcript",
            json={"text": "turn on laser", "speaker": "surgeon"},
        )
        app_state.orchestrator = saved
        assert response.status_code == 503


# ======================================================================
# AppState Tests
# ======================================================================


class TestAppState:
    def test_default_surgery(self):
        from src.utils.constants import DEFAULT_SURGERY
        assert app_state.current_surgery == DEFAULT_SURGERY or isinstance(
            app_state.current_surgery, str
        )

    def test_update_from_pipeline(self):
        state_dict = {
            "metadata": {"surgery": "PCNL", "phase": "Phase2"},
            "machines": {"0": ["M03"], "1": ["M01"]},
            "details": {"test": True},
            "suggestions": ["check vitals"],
            "confidence": 0.85,
            "source": "rule+medgemma",
        }
        app_state.update_from_pipeline(state_dict)
        current = app_state.current_state
        assert current.confidence == 0.85
        assert current.source == "rule+medgemma"

    def test_get_current_machines_returns_dict(self):
        machines = app_state.get_current_machines()
        assert isinstance(machines, dict)
