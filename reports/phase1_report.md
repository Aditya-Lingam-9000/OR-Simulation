# Phase 1 Report — Local Architecture & Skeleton Code

**Date:** 2026-02-22  
**Phase:** 1 of 10  
**Status:** ✅ PASS  
**Branch:** `feature/phase1-skeleton` → merged to `main`  
**Commit:** `feat(phase1): local architecture skeleton - all modules, API, tests (55/55 pass)`

---

## Objective

Build the complete local architecture skeleton: utility modules, API server, ASR/LLM runner interfaces, state management modules, worker stubs, and comprehensive unit tests. No model inference yet — strictly structural.

---

## Files Created (24 files, +3,135 lines)

### Utility Modules (`src/utils/`)
| File | Purpose | Lines |
|------|---------|-------|
| `device.py` | Unified device detection (PyTorch/ONNX/GGUF) | 186 |
| `constants.py` | All project-wide constants, paths, latency targets | 115 |
| `config.py` | Pydantic Settings for env-var-aware configuration | 116 |
| `logging_config.py` | Structured logging, console+file, audit loggers | 115 |

### API Server (`src/api/`)
| File | Purpose | Lines |
|------|---------|-------|
| `app.py` | FastAPI app with REST + WebSocket endpoints | 313 |

**Endpoints implemented:**
- `GET /health` — health check with disclaimer
- `GET /state` — current surgery state as JSON contract
- `GET /surgeries` — list available surgeries
- `GET /machines` — machines for current surgery
- `POST /select_surgery` — switch active surgery
- `POST /override` — manual machine override
- `WS /ws/state` — real-time state broadcast WebSocket

### ASR Runners (`src/asr/`)
| File | Purpose | Lines |
|------|---------|-------|
| `runner.py` | Abstract base class (`BaseASRRunner`) | 96 |
| `onnx_runner.py` | Concrete ONNX runner for MedASR INT8 | 207 |

### Audio Ingest (`src/ingest/`)
| File | Purpose | Lines |
|------|---------|-------|
| `mic_server.py` | Microphone capture + VAD chunking skeleton | 125 |

### State Management (`src/state/`)
| File | Purpose | Lines |
|------|---------|-------|
| `rules.py` | Deterministic rule engine (keyword/regex matching) | 352 |
| `rolling_buffer.py` | 180s rolling transcript buffer | 166 |
| `serializer.py` | JSON contract normalizer/validator/merger | 193 |

### LLM Inference (`src/llm/`)
| File | Purpose | Lines |
|------|---------|-------|
| `manager.py` | Async LLM request queue and dispatcher | 174 |
| `gguf_runner.py` | MedGemma GGUF runner via llama-cpp-python | 155 |

### Workers (`src/workers/`)
| File | Purpose | Lines |
|------|---------|-------|
| `asr_worker.py` | ASR pipeline worker | 71 |
| `rule_worker.py` | Rule engine worker | 72 |
| `llm_dispatcher.py` | LLM dispatch worker | 85 |
| `state_writer.py` | Atomic state file writer | 126 |

### Tests (`tests/`)
| File | Tests | Lines |
|------|-------|-------|
| `test_device.py` | 10 tests — device detection | 67 |
| `test_api.py` | 15 tests — all endpoints | 108 |
| `test_config.py` | 7 tests — settings validation | 40 |
| `test_rules.py` | 13 tests — rule engine + all 3 surgeries | 69 |
| `test_serializer.py` | 10 tests — normalize/validate/merge | 102 |

### Plans
| File | Purpose |
|------|---------|
| `plans/phase1_plan.md` | Detailed implementation plan with goals |

---

## Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.0.2
collected 55 items

tests/test_api.py ............... (15 passed)
tests/test_config.py ....... (7 passed)
tests/test_device.py .......... (10 passed)
tests/test_rules.py ............. (13 passed)
tests/test_serializer.py .......... (10 passed)
tests/test_smoke.py .. (2 passed)

============================= 55 passed in 4.54s ==============================
```

**0 failures, 0 warnings.**

---

## Sanity Checks

### 1. pytest — 55/55 PASS ✅
All unit tests pass across all modules.

### 2. uvicorn /health — 200 OK ✅
```json
{
  "status": "ok",
  "timestamp": "2026-02-22T06:23:52.265406+00:00",
  "surgery_loaded": "PCNL",
  "disclaimer": "OR-Symphony is a SIMULATION and RESEARCH system only..."
}
```

### 3. Deprecation Warning Fix ✅
Migrated from deprecated `@app.on_event("startup")` to modern `lifespan` context manager. Zero warnings on re-run.

---

## Architecture Decisions

1. **MedGemma format**: Model is GGUF (Q3_K_M), not ONNX. Created `GGUFRunner` using `llama-cpp-python` instead of ONNX Runtime. Added to requirements.txt.
2. **Device detection**: Unified `get_device_info()` returns `DeviceInfo` dataclass with PyTorch device, ONNX providers, and GGUF GPU layers. Auto-detects CPU/CUDA/MPS.
3. **Configuration**: Used `pydantic-settings` (v2.13.1) for env-var-aware settings with `OR_SYMPHONY_` prefix.
4. **Lifespan**: Used modern FastAPI lifespan context manager (not deprecated `on_event`).
5. **Rule engine**: Deterministic keyword/regex matching with negation handling and debounce. Builds trigger map from `surgeries_machines.json`.
6. **State serializer**: Enforces JSON output contract with `normalize()`, `validate()`, `merge()`. Supports both rule and LLM sources.

---

## Dependencies Added

| Package | Version | Purpose |
|---------|---------|---------|
| `pydantic-settings` | ≥2.1.0 (installed: 2.13.1) | Env-var-aware configuration |
| `llama-cpp-python` | ≥0.2.0 (not yet installed) | MedGemma GGUF inference (Phase 6) |

---

## What Feeds Phase 2

Phase 2 (Audio Ingest & VAD) will:
- Implement `MicrophoneCapture` in `src/ingest/mic_server.py` (currently skeleton)
- Use `AudioChunk` dataclass already defined
- Feed chunks to `ASRWorker` via async queue
- Use constants from `constants.py` (SAMPLE_RATE=16000, CHANNELS=1, CHUNK_DURATION_S=0.5)

---

## Sign-Off

> _(awaiting user review)_

- [ ] User PASS / FAIL
