#!/usr/bin/env python3
"""
frontend_simulator.py — Sends fake state updates over WebSocket for frontend testing.

Usage:
    python scripts/frontend_simulator.py [--port 8000] [--interval 3]

This starts a minimal FastAPI server that:
  1. Serves GET /surgeries, GET /machines, POST /select_surgery, POST /override
  2. Opens WebSocket at /ws/state and pushes simulated state every N seconds
  3. Cycles through phases and toggles machines ON/OFF

Useful for developing the frontend without the full backend pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Find project root and load machine configs
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MACHINES_DIR = PROJECT_ROOT / "configs" / "machines"

SURGERIES = ["PCNL", "Partial Hepatectomy", "Lobectomy"]
SURGERY_FILES = {
    "PCNL": "pcnl.json",
    "Partial Hepatectomy": "partial_hepatectomy.json",
    "Lobectomy": "lobectomy.json",
}


def load_machines(surgery: str) -> Dict[str, Any]:
    """Load machines config for a given surgery."""
    fname = SURGERY_FILES.get(surgery, "pcnl.json")
    path = MACHINES_DIR / fname
    if not path.exists():
        logger.warning("Machine config not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("machines", {})


# ---------------------------------------------------------------------------
# Simulator State
# ---------------------------------------------------------------------------

class SimState:
    """Holds the simulated state."""

    def __init__(self) -> None:
        self.surgery = "PCNL"
        self.machines = load_machines(self.surgery)
        self.phase_index = 0
        self.phases = ["Phase1", "Phase2", "Phase3", "Phase4", "Phase5", "Phase6"]
        self.tick = 0
        self.clients: List[WebSocket] = []

    def switch_surgery(self, surgery: str) -> None:
        self.surgery = surgery
        self.machines = load_machines(surgery)
        self.phase_index = 0
        self.tick = 0

    def generate_state(self) -> Dict[str, Any]:
        """Generate a simulated state dict."""
        self.tick += 1
        # Advance phase every 4 ticks
        if self.tick % 4 == 0 and self.phase_index < len(self.phases) - 1:
            self.phase_index += 1

        phase = self.phases[self.phase_index]
        machine_ids = list(self.machines.keys())

        # Determine which machines are ON based on phase_usage
        on_ids = []
        off_ids = []
        for mid, mdata in self.machines.items():
            usage = mdata.get("phase_usage", [])
            if phase in usage:
                on_ids.append(mid)
            else:
                off_ids.append(mid)

        # Randomly toggle one machine for visual interest
        if random.random() < 0.2 and on_ids:
            toggled = random.choice(on_ids)
            on_ids.remove(toggled)
            off_ids.append(toggled)

        sources = ["rule", "medgemma", "rule+medgemma"]
        source = random.choices(sources, weights=[0.5, 0.3, 0.2])[0]
        confidence = round(random.uniform(0.65, 0.98), 2)

        suggestions = []
        if self.tick % 3 == 0:
            suggestions = [f"Consider preparing for {self.phases[min(self.phase_index + 1, len(self.phases) - 1)]}"]

        return {
            "metadata": {
                "surgery": self.surgery,
                "phase": phase,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reasoning": "simulated" if self.tick % 5 != 0 else "normal",
            },
            "machines": {
                "0": sorted(off_ids),
                "1": sorted(on_ids),
            },
            "details": {
                "tick": self.tick,
                "phase_index": self.phase_index,
                "source": source,
            },
            "suggestions": suggestions,
            "confidence": confidence,
            "source": source,
        }


sim = SimState()

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="OR-Symphony Frontend Simulator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SurgerySelectRequest(BaseModel):
    surgery: str


class OverrideRequest(BaseModel):
    machine_id: str
    action: str
    reason: str = "Manual override"


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat(),
            "surgery_loaded": sim.surgery, "disclaimer": "SIMULATION ONLY"}


@app.get("/surgeries")
async def surgeries():
    return SURGERIES


@app.get("/machines")
async def machines():
    return sim.machines


@app.get("/state")
async def state():
    return sim.generate_state()


@app.get("/stats")
async def stats():
    return {"running": True, "tick": sim.tick, "mode": "simulator"}


@app.post("/select_surgery")
async def select_surgery(req: SurgerySelectRequest):
    if req.surgery not in SURGERIES:
        return {"status": "error", "detail": f"Unknown surgery: {req.surgery}"}
    sim.switch_surgery(req.surgery)
    logger.info("Switched to %s", req.surgery)
    # Broadcast new state to all clients
    state_json = sim.generate_state()
    for client in list(sim.clients):
        try:
            await client.send_json(state_json)
        except Exception:
            sim.clients.remove(client)
    return {"status": "ok", "surgery": req.surgery}


@app.post("/transcript")
async def transcript(body: dict):
    logger.info("Transcript received: %s", body.get("text", "")[:80])
    return {"status": "ok", "text": body.get("text", "")[:100]}


@app.post("/override")
async def override(req: OverrideRequest):
    logger.info("Override: %s → %s (%s)", req.machine_id, req.action, req.reason)
    return {"status": "ok", "machine_id": req.machine_id, "action": req.action,
            "note": "Simulated override applied."}


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    sim.clients.append(websocket)
    logger.info("Client connected (%d total)", len(sim.clients))
    try:
        # Send initial state
        await websocket.send_json(sim.generate_state())
        while True:
            # Wait for client messages (keepalive)
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in sim.clients:
            sim.clients.remove(websocket)
        logger.info("Client disconnected (%d remaining)", len(sim.clients))


# ---------------------------------------------------------------------------
# Background broadcaster
# ---------------------------------------------------------------------------

async def broadcast_loop(interval: float):
    """Periodically broadcast simulated state to all connected clients."""
    while True:
        await asyncio.sleep(interval)
        if sim.clients:
            state_json = sim.generate_state()
            disconnected = []
            for client in sim.clients:
                try:
                    await client.send_json(state_json)
                except Exception:
                    disconnected.append(client)
            for c in disconnected:
                sim.clients.remove(c)


@app.on_event("startup")
async def startup():
    interval = float(getattr(app.state, "_sim_interval", 3))
    asyncio.create_task(broadcast_loop(interval))
    logger.info("Simulator broadcasting every %.1fs", interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OR-Symphony Frontend Simulator")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Broadcast interval in seconds (default: 3)")
    args = parser.parse_args()

    app.state._sim_interval = args.interval
    logger.info("Starting simulator on port %d (interval=%.1fs)", args.port, args.interval)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
