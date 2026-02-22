# Phase 7 Report: Orchestrator, API, WebSocket & Overrides

**Date:** 2025-06-22  
**Commit:** `c769ca3`  
**Status:** ✅ COMPLETE  

---

## Summary

Phase 7 wires the full pipeline together: ASR → Rule Engine → LLM Dispatcher →
State Writer → WebSocket broadcast. A central Orchestrator coordinates all workers,
the API is enhanced with real pipeline integration, and a manual override system
with audit logging is fully operational.

---

## Deliverables

### New Files
| File | Lines | Purpose |
|------|-------|---------|
| `src/workers/orchestrator.py` | 371 | Central pipeline coordinator — manages all workers, shared queues, lifecycle |
| `tests/test_workers.py` | 590 | 59 tests covering LLMDispatcher, StateWriter, Orchestrator |
| `plans/phase7_plan.md` | 116 | Implementation plan |

### Modified Files
| File | Lines | Change |
|------|-------|--------|
| `src/workers/llm_dispatcher.py` | 296 | Full rewrite — processing loop, RollingBuffer, rate-limited dispatch, fallback |
| `src/workers/state_writer.py` | 369 | Full rewrite — merge loop, atomic write, override system, audit logging |
| `src/api/app.py` | 417 | Orchestrator integration, new endpoints (/stats, /transcript), real overrides |
| `src/workers/__init__.py` | 21 | Proper exports for all worker + orchestrator classes |
| `tests/test_api.py` | 196 | Enhanced from 13 to 25 tests — stats, transcript, override, AppState |

---

## Architecture

```
                        ┌──────────────────────────────────────┐
                        │            Orchestrator              │
                        │   (creates queues, manages workers)  │
                        └────────┬─────────────────────────────┘
                                 │
           ┌─────────────────────┼──────────────────────┐
           │                     │                      │
     feed_audio()          feed_transcript()       apply_override()
           │                     │                      │
     ┌─────▼──────┐    ┌────────▼────────┐       ┌─────▼──────┐
     │ ASR Worker  │    │  Fan-out to     │       │ Override   │
     │ audio→text  │    │  both workers   │       │ queue      │
     └─────┬──────┘    └───┬─────────┬───┘       └─────┬──────┘
           │               │         │                  │
           │          ┌────▼───┐ ┌───▼──────────┐       │
           │          │ Rule   │ │ LLM          │       │
           │          │ Worker │ │ Dispatcher   │       │
           │          └───┬────┘ └──────┬───────┘       │
           │              │             │               │
           │         rule_q         llm_q               │
           │              │             │               │
           │       ┌──────▼─────────────▼───────────────▼──┐
           │       │          State Writer                  │
           │       │  merge → override → atomic write      │
           │       │  → broadcast callback                 │
           │       └───────────────┬────────────────────────┘
           │                       │
           │              on_state_update()
           │                       │
           │              ┌────────▼──────────┐
           │              │ WebSocket clients  │
           │              │ (real-time push)   │
           │              └───────────────────┘
```

---

## Key Design Decisions

### 1. Separate Transcript Queues (Fan-out)
- Rule Worker and LLM Dispatcher each have their own `transcript_queue`
- `Orchestrator.feed_transcript()` enqueues to both, preventing one worker from stealing items from the other
- ASR Worker outputs to a shared queue; fan-out copies to both workers

### 2. Rate-Limited LLM Dispatch
- `dispatch_interval_s=2.0` — LLM dispatch at most every 2 seconds
- Single-threaded GGUF model cannot handle every transcript
- Transcripts accumulate in RollingBuffer; only the latest context is sent

### 3. Atomic State Writes
- `StateWriter.write_state()` writes to `.json.tmp` then calls `os.replace()`
- Prevents partial writes from being read by the API or frontend

### 4. Override Audit Trail
- All overrides logged as JSON lines to `logs/overrides.log`
- Each entry: `{machine_id, action, reason, operator, timestamp}`
- Overrides applied during the next merge cycle

### 5. Orchestrator Lifecycle Order
- Start: StateWriter → Rule Worker + LLM Dispatcher → ASR Worker
- Stop: ASR Worker → Rule Worker + LLM Dispatcher → StateWriter
- Consumers start before producers; producers stop before consumers

---

## API Endpoints (v0.7.0)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check with surgery loaded |
| GET | `/state` | Current surgery state (reads from orchestrator) |
| GET | `/surgeries` | List supported surgery types |
| GET | `/machines` | Machines for current surgery |
| GET | `/stats` | Aggregated pipeline statistics |
| POST | `/select_surgery` | Switch surgery type (propagates to all workers) |
| POST | `/transcript` | Feed text into pipeline (bypasses ASR) |
| POST | `/override` | Manual machine override with audit |
| WS | `/ws/state` | Real-time state updates |

---

## Test Summary

| File | Tests | Category |
|------|-------|----------|
| `tests/test_workers.py` | 59 | LLMDispatcher, StateWriter, Orchestrator |
| `tests/test_api.py` | 25 | All REST endpoints + AppState |
| **Phase 7 new** | **84** | |
| **Total suite** | **582** | All passing |

---

## Files Changed

```
 8 files changed, 2150 insertions(+), 88 deletions(-)
 plans/phase7_plan.md          | 116 +++
 src/api/app.py                | 417 (was 314)
 src/workers/__init__.py       |  21 (was 0)
 src/workers/llm_dispatcher.py | 296 (was 89)
 src/workers/orchestrator.py   | 371 +++
 src/workers/state_writer.py   | 369 (was 117)
 tests/test_api.py             | 196 (was 109)
 tests/test_workers.py         | 590 +++
```
