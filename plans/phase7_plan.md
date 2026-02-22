# Phase 7 Plan: Orchestrator, API, WebSocket & Overrides

## Goal
Wire ASR → Rule Engine → LLM → State Writer → WebSocket push into a cohesive orchestrator,
enhance the API with real state broadcasting, and add manual override with audit logging.

## Existing State
- `src/workers/asr_worker.py` — fully functional (audio_queue → ASR → transcript_queue)
- `src/workers/rule_worker.py` — fully functional (transcript_queue → RuleEngine → state_queue)
- `src/workers/llm_dispatcher.py` — stub (has LLMManager but no processing loop)
- `src/workers/state_writer.py` — partial (atomic write works, no merge loop, no WS broadcast)
- `src/api/app.py` — REST + WS endpoints exist, override endpoint is stub, no worker integration

## Implementation Steps

### 1. Rewrite LLM Dispatcher (`src/workers/llm_dispatcher.py`)
- Add transcript_queue input, state_queue output
- Processing loop: poll transcripts → build LLMRequest → manager.submit() → enqueue result
- Configurable context window from RollingBuffer
- Fallback mode: skip LLM, pass None to state writer
- Rate limiting: submit to LLM every N transcripts (not every single one)
- Stats tracking

### 2. Enhance State Writer (`src/workers/state_writer.py`)
- Add processing loop consuming from rule_state_queue + llm_state_queue
- Merge rule + LLM outputs via StateSerializer.build_current_state()
- Atomic write to output path
- WebSocket broadcast callback
- Override integration (apply manual overrides)
- Stats tracking

### 3. Build Orchestrator (`src/workers/orchestrator.py`) — NEW
- Central coordinator that creates and manages all workers
- Creates shared queues: audio_q, transcript_q, rule_state_q, llm_state_q
- Lifecycle: start() → runs all workers → stop() graceful shutdown
- Surgery switching at runtime
- Combined stats from all workers
- Serves as the entry point for the full pipeline

### 4. Enhance API + WebSocket (`src/api/app.py`)
- Integrate Orchestrator into FastAPI lifespan
- GET /state → reads from state writer's latest state
- WS /ws/state → broadcasts on state changes
- POST /override → applies override, logs to audit file, broadcasts
- GET /stats → returns orchestrator stats
- Pipeline start/stop via API (optional)

### 5. Add Override Audit Logging
- Write overrides to `logs/overrides.log` (append-only)
- JSON format: timestamp, machine_id, action, reason, operator
- Override applies to current state and triggers broadcast

### 6. Update workers __init__.py
- Export all worker classes + Orchestrator

### 7. Tests (`tests/test_workers.py` + enhance `tests/test_api.py`)
- ASR Worker: mock runner, queue flow
- Rule Worker: queue flow, text extraction, surgery switching
- LLM Dispatcher: queue flow, rate limiting, fallback mode
- State Writer: atomic write, merge, broadcast callback
- Orchestrator: start/stop, pipeline flow, stats
- API: enhanced override, stats endpoint, state broadcast

## Key Design Decisions
- Workers communicate via asyncio.Queue (no external message broker)
- LLM dispatcher rate limits (every 3rd transcript or 2s debounce) to avoid flooding
- State writer merges latest rule + LLM outputs (rule takes priority for toggles)
- Atomic writes via temp file + os.replace()
- WebSocket broadcast is fire-and-forget to disconnected clients
