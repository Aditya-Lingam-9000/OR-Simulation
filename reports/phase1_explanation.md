# Phase 1 Explanation — Local Architecture & Skeleton Code

## What Was Done

Phase 1 built the **complete structural skeleton** of OR-Symphony. Every module that will eventually process audio, run inference, manage state, and serve the frontend now exists as a well-defined Python file with clear interfaces, type hints, docstrings, and placeholder implementations where actual model logic will go in later phases.

---

## Module-by-Module Breakdown

### 1. Utility Layer (`src/utils/`)

**`device.py`** — Solves the "what hardware do we have?" problem once for the entire project. It probes for CUDA GPUs, Apple MPS, or falls back to CPU. Returns a `DeviceInfo` dataclass that every other module can query:
- PyTorch device string (`"cpu"`, `"cuda"`, `"mps"`)
- ONNX Runtime execution providers list (for MedASR)
- GGUF GPU layer count (for MedGemma via llama-cpp-python)

**`constants.py`** — Single source of truth for every magic value: file paths, audio parameters (16 kHz, mono, 0.5s chunks), latency targets (ASR < 400ms, rule engine < 500ms), LLM settings (batch=4, temp=0.1, context=4096), supported surgery list, and JSON output key names.

**`config.py`** — Uses `pydantic-settings` to create a `Settings` class that reads from environment variables (prefix `OR_SYMPHONY_`) with sensible defaults. This means you can override any setting without editing code — just set `OR_SYMPHONY_DEFAULT_SURGERY=Lobectomy` in your environment.

**`logging_config.py`** — Structured logging with two handlers (console at INFO, file at DEBUG) plus dedicated loggers for transcripts and state changes. The audit loggers will be critical for post-hoc analysis of surgical sessions.

### 2. API Server (`src/api/app.py`)

The FastAPI application is the **single entry point** to the system. It provides:

- **REST endpoints** for external tools and the frontend to query system state
- **WebSocket endpoint** (`/ws/state`) for real-time state broadcasting — the frontend connects here and receives JSON state updates as they happen
- **AppState class** that loads the machine dictionary from `surgeries_machines.json` and maintains current surgery selection, machine states, and connected WebSocket clients

The `/state` endpoint returns the exact JSON contract that MedGemma will eventually produce, so the frontend can be built against a stable API from day one.

### 3. ASR Layer (`src/asr/`)

**`runner.py`** defines the abstract interface (`BaseASRRunner`) with three methods: `load_model()`, `transcribe(audio_chunk)`, and `transcribe_streaming(chunks)`. Any ASR backend must implement this interface.

**`onnx_runner.py`** is the concrete implementation for our MedASR INT8 ONNX model. It already loads the model file and vocabulary (`tokens.txt`), verifies I/O shapes, and reports model info. The actual `transcribe()` method returns a placeholder — real inference comes in Phase 3.

### 4. Audio Ingest (`src/ingest/mic_server.py`)

Defines the `AudioChunk` dataclass (raw bytes + metadata) and `MicrophoneCapture` class skeleton. Phase 2 will fill in the VAD (Voice Activity Detection) and chunking logic using `sounddevice` and `webrtcvad`.

### 5. State Management (`src/state/`)

This is the **brain** of the system — three cooperating modules:

**`rules.py`** — The deterministic "fast path." When a transcript says "turn on the ventilator," this module instantly matches keywords against the machine trigger map from `surgeries_machines.json`. It handles:
- Activation keywords ("turn on", "activate", "start", "power up")
- Deactivation keywords ("turn off", "deactivate", "stop", "shut down")
- Negation detection ("don't turn on" → no action)
- Debounce (won't toggle the same machine twice within 2 seconds)

**`rolling_buffer.py`** — Maintains a sliding 180-second window of transcripts. When MedGemma needs context for reasoning, it pulls the full buffer. Old entries are automatically evicted by timestamp.

**`serializer.py`** — The JSON contract enforcer. Takes raw outputs from the rule engine and/or MedGemma and normalizes them into the strict output format:
```json
{
  "metadata": {"surgery": "PCNL", "phase": "...", "timestamp": "..."},
  "machines": {"M01": 0, "M02": 1, ...},
  "details": {...},
  "suggestions": [...],
  "confidence": 0.85,
  "source": "rule|llm|merged"
}
```
It also validates outputs (confidence 0.0-1.0, valid source strings, all required keys) and merges rule+LLM outputs with configurable weighting.

### 6. LLM Layer (`src/llm/`)

**`manager.py`** — Async queue-based dispatcher. When a transcript arrives, it creates an `LLMRequest` and enqueues it. A background worker dequeues and sends to the GGUF runner. This decouples the fast rule path from slow LLM inference.

**`gguf_runner.py`** — Interface to MedGemma via `llama-cpp-python`. Already has the model path, context window, temperature, and GPU layer settings wired up. Actual model loading comes in Phase 6.

### 7. Workers (`src/workers/`)

Four async worker stubs that will form the pipeline in Phase 7:
- **ASR Worker**: Consumes audio chunks → produces transcripts
- **Rule Worker**: Consumes transcripts → produces fast state updates
- **LLM Dispatcher**: Consumes transcripts → produces reasoned state updates
- **State Writer**: Consumes state updates → writes atomic JSON files + broadcasts via WebSocket

The State Writer already implements **atomic file writing** (write to temp file, then rename) to prevent partial reads.

### 8. Tests

55 unit tests covering:
- Device detection (10 tests)
- API endpoints including error paths (15 tests)
- Configuration defaults and caching (7 tests)
- Rule engine with all 3 surgeries (13 tests)
- State serializer normalize/validate/merge (10 tests)
- Project structure smoke tests (2 tests from Phase 0, still passing)

---

## Data Flow (Current State)

```
[Not yet implemented]          [Skeleton ready]              [Working]
                                                              
Microphone ──→ VAD ──→ ASR ──→ Rule Engine ──→ Serializer ──→ API ──→ Frontend
                       │                                       ↑
                       └──→ Rolling Buffer ──→ MedGemma ──→────┘
```

The right side (Serializer → API → Frontend) is functional now. The left side (audio → ASR → inference) contains well-defined interfaces that Phase 2-6 will implement.

---

## How This Feeds Phase 2

Phase 2 will implement **Audio Ingest & VAD**:
1. Fill in `MicrophoneCapture.start_capture()` with `sounddevice` streaming
2. Add WebRTC VAD for speech/silence detection
3. Chunk audio into 0.5s segments as `AudioChunk` objects
4. Push chunks to an async queue consumed by `ASRWorker`

All the infrastructure (constants, config, logging, queue interfaces) is already in place.
