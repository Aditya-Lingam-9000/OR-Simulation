"""Tests for src.api.app module — FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


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


class TestMachinesEndpoint:
    def test_get_machines(self, client):
        response = client.get("/machines")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


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
