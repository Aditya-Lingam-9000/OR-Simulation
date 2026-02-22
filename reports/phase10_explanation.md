# Phase 10 — Frontend Architecture Explanation

## Overview

Phase 10 delivers the **visual frontend** for OR-Symphony — a React single-page application
that connects to the FastAPI backend via WebSocket and renders a real-time operating room
visualisation with machine state tracking, override controls, and accessibility support.

## Component Architecture

```
App.jsx
├── StateProvider (Context + WebSocket)
│   └── AppContent
│       ├── Header
│       │   └── SurgerySelector
│       ├── Main (flex row)
│       │   ├── OPRoom (SVG)
│       │   │   ├── Zone labels
│       │   │   ├── Operating table
│       │   │   ├── Agent positions
│       │   │   └── MachineIcon × 9
│       │   ├── AgentOverlay
│       │   ├── MachineList (table)
│       │   ├── Suggestions
│       │   └── Raw Details (collapsible)
│       ├── StatusBar
│       └── OverrideDialog (modal)
```

## Data Flow

1. **StateProvider** opens a WebSocket to `/ws/state` (proxied to backend :8000)
2. Backend broadcasts `SurgeryStateResponse` JSON on every pipeline cycle
3. React Context distributes `{ state, connected, reconnectCount }` to all children
4. **OPRoom** reads `state.machines` → determines ON/OFF per machine → colours icons
5. **StatusBar** reads `state.metadata.phase`, `state.confidence`, `state.source`
6. **MachineList** provides the same data as a text table (accessibility)

## Surgery Selection Flow

1. **SurgerySelector** fetches `GET /surgeries` → populates dropdown
2. User selects → `POST /select_surgery { surgery: "Lobectomy" }`
3. Backend switches orchestrator + broadcasts new state via WebSocket
4. **App** detects `state.metadata.surgery` changed → loads matching layout JSON
5. **OPRoom** re-renders with new zone/machine positions

## Override Flow

1. User clicks a machine icon (SVG) or the Override button (table)
2. **OverrideDialog** modal opens with machine details
3. User picks ON/OFF/STANDBY, types a reason
4. `POST /override { machine_id, action, reason }` → backend applies + logs audit
5. Next WebSocket broadcast reflects the override

## Layout System

Each surgery has a layout JSON in `configs/layouts/`:
```json
{
  "surgery": "PCNL",
  "room": { "width": 800, "height": 600, "background": "#1e293b" },
  "zones": { "head_side": { "x": 400, "y": 80, "label": "Head" }, ... },
  "bed": { "x": 300, "y": 220, "width": 200, "height": 120 },
  "agents": [{ "id": "surgeon", "label": "Surgeon", "x": 400, "y": 380, "color": "#3b82f6" }],
  "machines": { "M01": { "x": 420, "y": 80 }, ... }
}
```

The `ui.icon` field in machine configs maps to SVG shape functions in `MachineIcon.jsx`.
14 distinct icon shapes cover all machine categories across all three surgeries.

## Accessibility

- **MachineList** text table provides screen-reader access to all machine states
- Machine icons have `role="button"`, `tabIndex={0}`, and `aria-label`
- **StatusBar** uses colour + text (not colour alone) for state indication
- Override dialog uses `aria-modal` and `aria-label`
- All interactive elements are keyboard-navigable

## Frontend Simulator

`scripts/frontend_simulator.py` runs a standalone FastAPI on port 8000 that:
- Serves all REST endpoints with mock data
- Opens WebSocket and broadcasts simulated state every N seconds
- Cycles through phases, toggles machines, randomises source/confidence
- Allows developing the frontend without the full ML pipeline

Usage: `python scripts/frontend_simulator.py --port 8000 --interval 2`

## Build Pipeline

- **Vite 6** bundles React + Tailwind → `dist/` (163KB JS + 14KB CSS gzipped)
- `@configs` alias resolves `configs/layouts/*.json` at build time
- `fs.allow: ['..']` lets Vite dev server access parent-level config files
- Proxy config routes all API calls and WebSocket to backend on :8000
