# OR-Symphony: Developer Guide

> **Version:** 0.8.0  
> **Last Updated:** 2026-02-22  

---

## 1. Project Overview

OR-Symphony is a research-grade AI system that tracks surgical state by
processing live audio from an operating room microphone. It transcribes
speech, matches equipment commands via a deterministic rule engine, and
uses a medical language model (MedGemma) for contextual reasoning.

**This is a simulation system. It does NOT control real medical devices.**

See [SAFETY.md](../SAFETY.md) for the full safety policy.

---

## 2. Architecture

```
Audio Input → VAD → Chunking → MedASR (ONNX) → Transcript
                                                    │
                                 ┌──────────────────┤
                                 │                  │
                            Rule Engine       LLM Dispatcher
                           (deterministic)    (MedGemma GGUF)
                                 │                  │
                                 └───────┬──────────┘
                                         │
                                    State Writer
                                   (merge + audit)
                                         │
                                    WebSocket Push
                                         │
                                    Frontend UI
```

### Pipeline Components

| Component | Module | Latency |
|-----------|--------|---------|
| Audio capture | `src/ingest/mic_stream.py` | Real-time |
| VAD + chunking | `src/ingest/mic_stream.py` | < 30ms |
| ASR (MedASR) | `src/asr/onnx_runner.py` | < 400ms |
| Rule Engine | `src/state/rules.py` | < 500ms |
| LLM (MedGemma) | `src/llm/gguf_runner.py` | 1-5s |
| State merge | `src/state/serializer.py` | < 10ms |
| State write | `src/workers/state_writer.py` | Atomic |
| WebSocket push | `src/api/app.py` | < 10ms |

### Worker Pipeline

The Orchestrator (`src/workers/orchestrator.py`) manages four workers:

1. **ASR Worker**: Consumes audio chunks → produces transcripts
2. **Rule Worker**: Consumes transcripts → produces machine state patches
3. **LLM Dispatcher**: Accumulates transcripts → rate-limited LLM inference
4. **State Writer**: Merges rule + LLM outputs → atomic JSON write → broadcast

---

## 3. Quick Start

### Prerequisites

- Python 3.11+
- Windows 10/11 (tested), Linux/macOS (should work)
- ~4 GB disk space (for models)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd OR-Simulation

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the API Server

```bash
uvicorn src.api.app:app --reload --port 8000
```

Then visit:
- Health check: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs
- Current state: http://127.0.0.1:8000/state

### Run Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_rules.py -v

# With coverage
pytest --cov=src --cov-report=term-missing
```

---

## 4. Project Structure

```
OR-Simulation/
├── configs/
│   ├── machines/           # Surgery-specific machine dictionaries
│   │   ├── pcnl.json
│   │   ├── partial_hepatectomy.json
│   │   └── lobectomy.json
│   └── layouts/            # Frontend layout configs (Phase 10)
├── data/
│   └── surgeries_machines.json
├── docs/
│   └── DEVELOPER_GUIDE.md  # This file
├── logs/
│   ├── transcripts/        # Daily transcript logs (append-only)
│   ├── state_changes.log   # State audit log (SHA-256 chain)
│   ├── overrides.log       # Override audit log
│   └── app_YYYYMMDD.log    # Application log
├── onnx_models/
│   ├── medasr/             # MedASR INT8 ONNX model
│   └── medgemma/           # MedGemma GGUF model
├── plans/                  # Phase implementation plans
├── reports/                # Phase completion reports
├── schemas/
│   └── surgery_state.schema.json
├── scripts/                # Utility scripts
├── src/
│   ├── api/
│   │   └── app.py          # FastAPI + WebSocket server
│   ├── asr/
│   │   ├── runner.py       # ASRResult dataclass
│   │   ├── onnx_runner.py  # ONNX ASR inference
│   │   ├── feature_extractor.py
│   │   └── ctc_decoder.py
│   ├── ingest/
│   │   └── mic_stream.py   # Microphone capture + VAD
│   ├── llm/
│   │   ├── gguf_runner.py  # GGUF model loading + inference
│   │   ├── prompts.py      # Surgery-aware prompt engineering
│   │   ├── batcher.py      # Async micro-batcher
│   │   └── manager.py      # LLM pipeline orchestrator
│   ├── state/
│   │   ├── rules.py        # Deterministic rule engine
│   │   ├── rolling_buffer.py
│   │   └── serializer.py   # State normalization + merge
│   ├── utils/
│   │   ├── audit.py        # SHA-256 checksummed audit logging
│   │   ├── safety.py       # Output safety validation
│   │   ├── constants.py    # All project constants
│   │   ├── config.py       # Configuration management
│   │   ├── device.py       # CPU/GPU detection
│   │   └── logging_config.py
│   └── workers/
│       ├── orchestrator.py # Central pipeline coordinator
│       ├── asr_worker.py   # Audio → transcript
│       ├── rule_worker.py  # Transcript → rule state
│       ├── llm_dispatcher.py # Transcript → LLM state
│       └── state_writer.py # Merge → write → broadcast
├── tests/                  # pytest test suite
├── SAFETY.md               # Safety policy document
├── README.md
├── LICENSE
├── requirements.txt
└── pytest.ini
```

---

## 5. API Reference

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + disclaimer |
| GET | `/state` | Current surgery state JSON |
| GET | `/surgeries` | List supported surgery types |
| GET | `/machines` | Machines for current surgery |
| GET | `/stats` | Pipeline statistics |
| POST | `/select_surgery` | Switch surgery type |
| POST | `/transcript` | Feed text into pipeline |
| POST | `/override` | Manual machine override |

### WebSocket

| Path | Description |
|------|-------------|
| `/ws/state` | Real-time state push (JSON) |

### State JSON Schema

```json
{
  "metadata": {
    "surgery": "PCNL",
    "phase": "Phase1",
    "timestamp": "2026-02-22T10:30:00Z",
    "reasoning": "normal"
  },
  "machines": {
    "0": ["M03", "M05"],
    "1": ["M01", "M09"]
  },
  "details": {},
  "suggestions": ["Consider preparing the fluoroscopy"],
  "confidence": 0.85,
  "source": "rule+medgemma"
}
```

---

## 6. Supported Surgeries

| Surgery | Config | Machines |
|---------|--------|----------|
| PCNL | `configs/machines/pcnl.json` | 12 machines |
| Partial Hepatectomy | `configs/machines/partial_hepatectomy.json` | 14 machines |
| Lobectomy | `configs/machines/lobectomy.json` | 13 machines |

---

## 7. Testing

### Test Organization

| File | Tests | Coverage |
|------|-------|----------|
| `test_audio.py` | Audio capture, VAD, chunking |
| `test_asr.py` | ASR inference, feature extraction |
| `test_rules.py` | Rule engine (170 tests) |
| `test_buffer.py` | Rolling buffer |
| `test_serializer.py` | State serialization + merge |
| `test_schema.py` | JSON schema validation |
| `test_llm.py` | LLM runner, prompts, batcher, manager |
| `test_workers.py` | Worker lifecycle, pipeline flow |
| `test_api.py` | API endpoints |
| `test_safety.py` | Safety validation, audit logging |
| `test_config.py` | Configuration |
| `test_device.py` | Device detection |
| `test_smoke.py` | Basic import tests |

### Running Specific Categories

```bash
# Rule engine only
pytest tests/test_rules.py -v

# Workers + API
pytest tests/test_workers.py tests/test_api.py -v

# Safety + audit
pytest tests/test_safety.py -v
```

---

## 8. Configuration

All constants are in `src/utils/constants.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `AUDIO_SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `VAD_AGGRESSIVENESS` | 2 | WebRTC VAD level (0-3) |
| `ASR_CHUNK_LATENCY_TARGET_MS` | 400 | Max ASR latency |
| `RULE_DEBOUNCE_SECONDS` | 3.0 | Debounce repeated commands |
| `LLM_MAX_BATCH_SIZE` | 4 | Micro-batcher max batch |
| `LLM_TEMPERATURE` | 0.1 | LLM temperature |
| `ROLLING_BUFFER_DURATION_S` | 180 | Context window (seconds) |
| `API_PORT` | 8000 | API server port |

---

## 9. Safety & Audit

### Output Validation

The `SafetyValidator` (`src/utils/safety.py`) checks every output for:
- Required fields (`source`, `confidence`, `metadata`)
- Valid source values (`rule`, `medgemma`, `rule+medgemma`)
- Confidence in [0.0, 1.0]
- No banned patterns (executable commands, API calls, clinical dosages)
- Suggestions use advisory language

### Audit Trail

The `AuditLogger` (`src/utils/audit.py`) provides:
- SHA-256 hash of each log entry payload
- Chain hashing (each entry references previous hash)
- `verify_chain()` method for tamper detection
- Separate loggers for state changes and overrides

### Override Logging

Every manual override is logged with:
- Machine ID, action, reason, operator, timestamp
- SHA-256 checksum
- Chain integrity reference

---

## 10. Contributing

1. Create a feature branch from `main`
2. Follow existing code patterns and type hints
3. Add tests for new functionality
4. Run `pytest` and ensure all tests pass
5. Update relevant documentation
6. Submit a pull request

### Code Style

- Python 3.11+ type hints throughout
- Docstrings on all public methods
- `from __future__ import annotations` in all modules
- Constants in `src/utils/constants.py` (no magic values)
- Async workers with `start()`/`stop()` lifecycle
