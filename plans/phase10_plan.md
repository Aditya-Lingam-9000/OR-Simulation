# Phase 10 Plan — Final Frontend

**Date**: 2026-02-22  
**Phase**: 10 of 11 (Final)  

---

## Requirements (from Master Plan)

1. React + Vite frontend skeleton
2. WebSocket StateProvider connecting to `/ws/state`
3. Surgery selection dropdown (PCNL, Partial Hepatectomy, Lobectomy)
4. OPRoom component with configurable layout per surgery
5. Entity rendering (machines with ON/OFF/STANDBY states)
6. Agent animation (rule → pending, medgemma → confirmed, degraded → caution)
7. Manual override dialog (click machine → POST /override)
8. Accessibility: textual machine list, keyboard navigation
9. Frontend simulator script
10. Cross-browser sanity

## Implementation Approach

Since this is a simulation/research project, we'll build a **self-contained React SPA** with:
- Vite for bundling
- Pure CSS for animations (no heavy deps like pixi.js)
- SVG-based OR room rendering (clean, scalable, accessible)
- Tailwind CSS for styling
- WebSocket for real-time state

## Deliverables

| # | Path | Description |
|---|------|-------------|
| 1 | `frontend/` | Vite + React project skeleton |
| 2 | `frontend/src/providers/StateProvider.jsx` | WebSocket connection + state context |
| 3 | `frontend/src/components/OPRoom.jsx` | SVG operating room visualization |
| 4 | `frontend/src/components/MachineIcon.jsx` | Individual machine with state coloring |
| 5 | `frontend/src/components/SurgerySelector.jsx` | Dropdown to switch surgery |
| 6 | `frontend/src/components/OverrideDialog.jsx` | Modal for manual overrides |
| 7 | `frontend/src/components/MachineList.jsx` | Accessible text list fallback |
| 8 | `frontend/src/components/AgentOverlay.jsx` | Agent animation overlay |
| 9 | `frontend/src/components/StatusBar.jsx` | Pipeline status + phase display |
| 10 | `configs/layouts/*.json` | Zone coordinates per surgery |
| 11 | `scripts/frontend_simulator.py` | Fake WS state update sender |

## Implementation Order

1. Create Vite + React skeleton (package.json, vite.config, index.html)
2. Create layout configs for all 3 surgeries
3. Build StateProvider (WebSocket + context)
4. Build OPRoom + MachineIcon components
5. Build SurgerySelector + OverrideDialog
6. Build AgentOverlay + StatusBar
7. Build MachineList (accessibility fallback)
8. Build frontend_simulator.py
9. Create App.jsx assembling everything
10. Run backend tests, commit, report
