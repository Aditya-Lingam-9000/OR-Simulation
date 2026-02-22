"""
OR-Symphony: FastAPI Application Skeleton

Main API server with REST and WebSocket endpoints.
Drives the frontend operating-room visualization.

Run:
    uvicorn src.api.app:app --reload --port 8000

Sanity check:
    curl http://127.0.0.1:8000/health
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.constants import (
    DEFAULT_SURGERY,
    SAFETY_DISCLAIMER,
    SUPPORTED_SURGERIES,
    SURGERIES_MACHINES_PATH,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    timestamp: str
    surgery_loaded: str
    disclaimer: str = SAFETY_DISCLAIMER


class SurgerySelectRequest(BaseModel):
    """Request to select a surgery type."""

    surgery: str = Field(..., description="Surgery name (PCNL, Partial Hepatectomy, Lobectomy)")


class OverrideRequest(BaseModel):
    """Manual override request for machine state."""

    machine_id: str = Field(..., description="Machine ID (e.g., M01)")
    action: str = Field(..., description="Action: ON, OFF, STANDBY")
    reason: str = Field(default="Manual override", description="Reason for override")


class SurgeryStateResponse(BaseModel):
    """Current surgery state response — matches JSON output contract."""

    metadata: Dict[str, Any] = Field(default_factory=dict)
    machines: Dict[str, List[str]] = Field(
        default_factory=lambda: {"0": [], "1": []},
        description="0=OFF, 1=ON machine IDs",
    )
    details: Dict[str, Any] = Field(default_factory=dict)
    suggestions: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = Field(default="rule", description="rule | medgemma | rule+medgemma")


# ---------------------------------------------------------------------------
# Application State (in-memory, thread-safe via async)
# ---------------------------------------------------------------------------


class AppState:
    """Holds the application state in memory."""

    def __init__(self) -> None:
        self.current_surgery: str = DEFAULT_SURGERY
        self.current_state: SurgeryStateResponse = SurgeryStateResponse(
            metadata={
                "surgery": DEFAULT_SURGERY,
                "phase": "Phase1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reasoning": "normal",
            }
        )
        self.machines_data: Dict[str, Any] = {}
        self.connected_clients: List[WebSocket] = []
        self._load_machines_data()

    def _load_machines_data(self) -> None:
        """Load surgery machines data from JSON file."""
        try:
            if SURGERIES_MACHINES_PATH.exists():
                with open(SURGERIES_MACHINES_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Index by surgery name
                self.machines_data = {entry["surgery"]: entry for entry in data}
                logger.info(
                    "Loaded machines data for %d surgeries: %s",
                    len(self.machines_data),
                    list(self.machines_data.keys()),
                )
            else:
                logger.warning("Machines data file not found: %s", SURGERIES_MACHINES_PATH)
        except Exception as e:
            logger.error("Failed to load machines data: %s", e)

    def get_current_machines(self) -> Dict[str, Any]:
        """Get machines dictionary for currently selected surgery."""
        entry = self.machines_data.get(self.current_surgery, {})
        return entry.get("machines", {})


# Global app state (created before lifespan so endpoints can reference it)
app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — startup and shutdown logic."""
    # --- startup ---
    logger.info("OR-Symphony API starting — Surgery: %s", app_state.current_surgery)
    logger.info("⚠️  %s", SAFETY_DISCLAIMER)
    yield
    # --- shutdown ---
    logger.info("OR-Symphony API shutting down")
    for client in app_state.connected_clients:
        try:
            await client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OR-Symphony API",
    description="Predictive Surgical State Engine — Simulation & Research Only",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint — verifies API is running."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        surgery_loaded=app_state.current_surgery,
    )


@app.get("/state", response_model=SurgeryStateResponse)
async def get_state() -> SurgeryStateResponse:
    """Get the current surgery state as structured JSON."""
    return app_state.current_state


@app.get("/surgeries", response_model=List[str])
async def list_surgeries() -> List[str]:
    """List all supported surgery types."""
    return SUPPORTED_SURGERIES


@app.get("/machines")
async def get_machines() -> Dict[str, Any]:
    """Get machines dictionary for the currently selected surgery."""
    return app_state.get_current_machines()


@app.post("/select_surgery")
async def select_surgery(request: SurgerySelectRequest) -> Dict[str, str]:
    """
    Select a surgery type. Switches the machines dictionary and resets state.

    This changes which machine-set the rule engine and LLM use.
    """
    if request.surgery not in SUPPORTED_SURGERIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported surgery: {request.surgery}. Choose from {SUPPORTED_SURGERIES}",
        )

    app_state.current_surgery = request.surgery
    app_state.current_state = SurgeryStateResponse(
        metadata={
            "surgery": request.surgery,
            "phase": "Phase1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reasoning": "normal",
        }
    )
    logger.info("Surgery switched to: %s", request.surgery)

    # Notify connected WebSocket clients
    await _broadcast_state()

    return {"status": "ok", "surgery": request.surgery}


@app.post("/override")
async def manual_override(request: OverrideRequest) -> Dict[str, str]:
    """
    Manual override for a machine state.

    Requires human confirmation — this is a safety feature.
    All overrides are logged to the audit trail.
    """
    machines = app_state.get_current_machines()
    if request.machine_id not in machines:
        raise HTTPException(
            status_code=404,
            detail=f"Machine {request.machine_id} not found in {app_state.current_surgery}",
        )

    # Log the override (audit trail)
    logger.info(
        "OVERRIDE | machine=%s | action=%s | reason=%s | surgery=%s",
        request.machine_id,
        request.action,
        request.reason,
        app_state.current_surgery,
    )

    # TODO: Apply override to current state and broadcast
    return {
        "status": "ok",
        "machine_id": request.machine_id,
        "action": request.action,
        "note": "Override logged. State update will be broadcast via WebSocket.",
    }


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/state")
async def websocket_state(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time state updates.

    Clients connect here to receive surgery state changes as JSON.
    """
    await websocket.accept()
    app_state.connected_clients.append(websocket)
    logger.info("WebSocket client connected. Total clients: %d", len(app_state.connected_clients))

    try:
        # Send current state immediately on connect
        await websocket.send_json(app_state.current_state.model_dump())

        # Keep connection alive, receive messages (e.g., ping/pong)
        while True:
            data = await websocket.receive_text()
            # Handle client messages if needed (e.g., override requests via WS)
            logger.debug("WS received: %s", data)
    except WebSocketDisconnect:
        app_state.connected_clients.remove(websocket)
        logger.info(
            "WebSocket client disconnected. Total clients: %d",
            len(app_state.connected_clients),
        )


async def _broadcast_state() -> None:
    """Broadcast current state to all connected WebSocket clients."""
    state_json = app_state.current_state.model_dump()
    disconnected: List[WebSocket] = []

    for client in app_state.connected_clients:
        try:
            await client.send_json(state_json)
        except Exception:
            disconnected.append(client)

    for client in disconnected:
        app_state.connected_clients.remove(client)


