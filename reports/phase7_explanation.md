# Phase 7 Explanation: Orchestrator, API, WebSocket & Overrides

## What Was Built

Phase 7 is the integration layer — it takes the four separate workers
(ASR, Rule Engine, LLM, State Writer) and wires them into a cohesive
pipeline managed by a central Orchestrator. The API is enhanced to serve
as the real-time interface for the frontend.

## The Three Core Modules

### 1. LLM Dispatcher (`src/workers/llm_dispatcher.py`)
- Consumes transcripts from a queue, adds them to a `RollingBuffer`
- Every `dispatch_interval_s` seconds (default 2), builds an `LLMRequest`
  from the buffer context and submits it to `LLMManager`
- The rate limit prevents flooding the single-threaded GGUF model —
  the model takes 1-5 seconds per inference, so sending every transcript
  would create an unbounded backlog
- **Fallback mode**: if the LLM model fails to load, the dispatcher
  enters fallback mode where it skips all dispatching. The pipeline
  continues with rule-engine-only output.
- `_extract_text()` handles multiple input types: plain strings, dicts
  with "text" keys, `ASRResult` objects — making the dispatcher flexible

### 2. State Writer (`src/workers/state_writer.py`)
- Drains both `rule_state_queue` and `llm_state_queue` non-blocking
- Merges rule + LLM outputs via `StateSerializer.merge()`:
  - Rule engine has priority for machine toggles (ON/OFF)
  - LLM adds suggestions, phase reasoning, and confidence scores
  - Source field becomes `"rule+medgemma"` when both contribute
- Applies pending overrides (modifies the machine lists)
- Writes JSON atomically: writes to `.json.tmp` then `os.replace()`
  swaps it in — this prevents any reader from seeing a half-written file
- Fires the `on_update` callback with the merged state dict, which the
  API uses to broadcast via WebSocket

### 3. Orchestrator (`src/workers/orchestrator.py`)
- Creates all shared `asyncio.Queue` instances
- Instantiates all four workers with the correct queue wiring
- `start()` starts workers in dependency order: consumers first
  (StateWriter), then intermediate (Rule Worker, LLM Dispatcher),
  then producers (ASR Worker)
- `stop()` reverses the order so no data is lost in transit
- `feed_transcript(text)` sends the same text to **both** the Rule Worker
  and LLM Dispatcher queues — this is the "fan-out" pattern
- `apply_override()` delegates to StateWriter's override queue
- `switch_surgery()` propagates the change to all workers
- `stats` property aggregates metrics from all workers into one dict

## How the API Uses the Orchestrator

The FastAPI app creates an `Orchestrator` instance during its lifespan
startup. All endpoints now delegate to the orchestrator:

- **GET /state** → `orchestrator.get_current_state()`
- **POST /transcript** → `orchestrator.feed_transcript(text)`
- **POST /override** → `orchestrator.apply_override(machine_id, action)`
- **GET /stats** → `orchestrator.stats`
- **POST /select_surgery** → `orchestrator.switch_surgery(surgery)`
- **WS /ws/state** → receives broadcasts from the StateWriter callback

The `_broadcast_state()` function is passed as the `on_state_update`
callback to the Orchestrator. When StateWriter merges new state, it
calls this function, which pushes the state JSON to all connected
WebSocket clients.

## Override System

Overrides represent human corrections to the AI's machine state
predictions. The flow:

1. Operator sends POST /override with `{machine_id, action, reason}`
2. API logs it and calls `orchestrator.apply_override()`
3. StateWriter queues the override in `_overrides` list
4. On next merge cycle, `_apply_overrides()` modifies the machine lists
5. Override is logged to `logs/overrides.log` as a JSON line
6. Updated state is broadcast via WebSocket

This audit trail is critical — in a real OR, every human intervention
must be traceable.

## Why asyncio.Queue (Not Redis/Kafka)

For a single-process simulation, `asyncio.Queue` provides:
- Zero latency between workers (in-process memory)
- No external dependencies to install or configure
- Natural backpressure via `maxsize` parameter
- Clean cancellation via poison pills (`None` sentinel)

The architecture is designed so that swapping to an external broker
(Redis Streams, NATS, etc.) would only require changing the queue
creation in the Orchestrator — the workers don't care what queue
implementation they use.

## Test Strategy

59 new worker tests cover:
- **LLMDispatcher**: init, start/stop, transcript extraction, dispatch
  to state queue, rate limiting, fallback mode, surgery switching
- **StateWriter**: init, atomic writes, override queueing/applying/logging,
  merge loop, broadcast callback
- **Orchestrator**: init, lifecycle, feed_transcript fan-out, override
  delegation, surgery switching, stats, full pipeline integration

All tests use mock LLMManagers to avoid loading the 2GB model.
