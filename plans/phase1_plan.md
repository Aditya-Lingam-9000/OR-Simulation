# Phase 1 — Local Architecture & Skeleton Code — Detailed Plan

> **Project**: OR-Symphony: Predictive Surgical State Engine  
> **Phase**: 1 — Architecture & Skeleton  
> **Created**: 2026-02-22  
> **Status**: IN PROGRESS  
> **Gate**: Must PASS before Phase 2 (Audio Capture)

---

## Objective

Create the full directory layout, module interfaces, API endpoint stubs, config, schema artifacts, and placeholder workers. All modules must have proper type hints, docstrings, and be importable.

---

## Key Discovery: Model Formats

- **MedASR**: `onnx_models/medasr/model.int8.onnx` (147 MB, INT8, ONNX) — use `onnxruntime`
- **MedGemma**: `onnx_models/medgemma/medgemma-4b-it-Q3_K_M.gguf` (1.95 GB, Q3, GGUF) — use `llama-cpp-python`
- Need to add `llama-cpp-python` to `requirements.txt`

---

## Subphases & Checklist

### 1.1 — Utilities Layer
- [ ] `src/utils/device.py` — CPU/GPU detection, ONNX provider selection, GGUF device config
- [ ] `src/utils/config.py` — Configuration management via pydantic Settings
- [ ] `src/utils/logging_config.py` — Structured JSON logging setup
- [ ] `src/utils/constants.py` — Project-wide constants (no magic values)

### 1.2 — API Layer
- [ ] `src/api/app.py` — FastAPI app with:
  - `GET /health` — health check
  - `GET /state` — current surgery state (stub)
  - `WS /ws/state` — WebSocket state push (stub)
  - `POST /override` — manual override (stub)
  - `POST /select_surgery` — surgery selection (stub)

### 1.3 — ASR Layer
- [ ] `src/asr/runner.py` — Abstract ASR runner + ONNX-based implementation skeleton
- [ ] `src/asr/onnx_runner.py` — Concrete ONNX ASR runner using `model.int8.onnx`

### 1.4 — Ingest Layer
- [ ] `src/ingest/mic_server.py` — Microphone capture + WebRTC stub

### 1.5 — State Layer
- [ ] `src/state/rules.py` — Rule engine skeleton (machines dict loader + keyword mapper)
- [ ] `src/state/rolling_buffer.py` — Transcript rolling buffer skeleton
- [ ] `src/state/serializer.py` — Output normalizer skeleton

### 1.6 — LLM Layer
- [ ] `src/llm/manager.py` — LLM request queue + batcher skeleton
- [ ] `src/llm/gguf_runner.py` — MedGemma GGUF runner via llama-cpp-python

### 1.7 — Workers Layer
- [ ] `src/workers/asr_worker.py` — ASR worker placeholder
- [ ] `src/workers/rule_worker.py` — Rule worker placeholder
- [ ] `src/workers/llm_dispatcher.py` — LLM dispatcher placeholder
- [ ] `src/workers/state_writer.py` — State writer placeholder

### 1.8 — Tests
- [ ] `tests/test_device.py` — Device helper tests
- [ ] `tests/test_api.py` — API endpoint tests (health, state stub)
- [ ] `tests/test_config.py` — Config loading tests

### 1.9 — Sanity Checks
- [ ] `pytest` all passes
- [ ] `uvicorn src.api.app:app --reload` starts
- [ ] `curl http://127.0.0.1:8000/health` returns `{"status":"ok"}`
- [ ] All modules importable without error

### 1.10 — Commit & Report
- [ ] Feature branch, commit, merge
- [ ] `reports/phase1_report.md`
- [ ] `reports/phase1_explanation.md`
- [ ] **PHASE 1 PASS: ______ (initials/date/time)**
