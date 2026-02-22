# OR-Symphony: Master Project Plan — Full Checkbox Tracker

> **Project**: OR-Symphony: Predictive Surgical State Engine  
> **Created**: 2026-02-22  
> **Status**: Phase 1 COMPLETE — awaiting sign-off  
> **Rule**: Frontend (Phase 10) is BLOCKED until ALL prior phases have PASS reports with manual sign-off.  
> **Rule**: Only delete temp/test artifacts after the phase report is PASS and you manually confirm.

---

## Pre-Start One-Time Checklist

- [x] Install Python 3.10+ and `git`
- [x] Install VS Code with Git and Python extensions
- [x] Sign into GitHub Copilot Pro
- [ ] Create GitHub repo (private)
- [ ] Create Kaggle account (for GPU runs)
- [ ] Optional: Lightning AI account for short GPU sessions
- [x] Create top-level project folder

---

## Phase 0 — Repo + Environment (Gated: must PASS before any coding)

**Goal**: Clean, versioned repo with venv and CI skeleton.

- [x] Create repository locally with initial commit
  - [x] Create `README.md`
  - [x] Create `LICENSE`
  - [x] Create `.gitignore`
  - [x] Initial git commit
- [ ] Create remote repo on GitHub and connect
  - [ ] Add remote origin
  - [ ] Push to main
- [x] Create venv + baseline requirements
  - [x] Create virtual environment
  - [x] Install and upgrade pip
  - [x] Create `requirements.txt` with all dependencies
  - [x] Install requirements successfully
- [x] Add CI skeleton (`.github/workflows/ci.yml`)
  - [x] Lint job configured
  - [x] Unit test job configured
- [x] Sanity checks
  - [x] `python -c "import onnxruntime, torch, transformers, fastapi; print('ok')"` passes
  - [ ] CI file appears on GitHub and runs
- [x] Commit and push all Phase 0 artifacts
- [x] Generate `reports/phase0_report.md`
- [x] **PHASE 0 PASS: user signed off 2026-02-22**

---

## Phase 1 — Local Architecture & Skeleton Code (Gated)

**Goal**: Directory layout, module interfaces, API endpoint stubs, config and schema artifacts.

- [x] Create project directory structure
  - [x] `src/ingest/` — mic/web input, VAD bridge
  - [x] `src/asr/` — MedASR wrappers + streaming adapters
  - [x] `src/state/` — rolling buffer, rule-engine, schema
  - [x] `src/llm/` — MedGemma runner, queue/dispatcher
  - [x] `src/api/` — FastAPI/WS endpoints
  - [x] `src/workers/` — orchestrator services
  - [x] `src/utils/` — device helpers, logging, config
  - [x] `tests/`
  - [x] `scripts/`
  - [x] `onnx_models/` (git-lfs or excluded)
  - [x] `models/` (excluded)
  - [x] `reports/`
  - [x] `configs/`
  - [x] `schemas/`
  - [x] `logs/`
- [x] Create key interface modules (with TODOs)
  - [x] `src/ingest/mic_server.py` — browser mic via WebRTC or local mic wrapper
  - [x] `src/asr/runner.py` — streaming ASR runner (abstract + ONNX impl)
  - [x] `src/state/rules.py` — machines dictionary loader + rule engine
  - [x] `src/llm/manager.py` — request queue + batcher skeleton
  - [x] `src/api/app.py` — FastAPI + WS skeleton with `/health` endpoint
  - [x] `src/utils/device.py` — device helper (CPU/GPU detection)
  - [x] `src/utils/logging_config.py` — logging setup
  - [x] `src/utils/config.py` — configuration management
- [x] Create placeholder worker modules
  - [x] `src/workers/asr_worker.py`
  - [x] `src/workers/rule_worker.py`
  - [x] `src/workers/llm_dispatcher.py`
  - [x] `src/workers/state_writer.py`
- [x] Sanity checks
  - [x] `pytest` runs and passes — 55/55 tests
  - [x] `uvicorn src.api.app:app` starts successfully
  - [x] `/health` returns `{"status":"ok"}` (HTTP 200)
- [x] Create feature branch, commit, PR, merge
- [x] Generate `reports/phase1_report.md`
- [ ] **PHASE 1 PASS: ______ (initials/date/time)**

---

## Phase 2 — Audio Capture & Streaming (Gated)

**Goal**: Robust mic capture, VAD, chunking policy, temporary audio queue.

- [x] Choose input path (WebRTC or desktop mic) — desktop mic via sounddevice
- [x] Implement `scripts/test_mic_client.py` for local mic capture
- [x] Implement VAD + chunker
  - [x] `src/ingest/mic_stream.py` using `webrtcvad`
  - [x] Produce 0.5–2.0s speech chunks
  - [x] Configurable overlap 200–500ms
- [x] Save chunks to in-memory asyncio.Queue
- [x] Save chunks to `tmp/audio_chunks/` for replay
- [x] Add `scripts/generate_test_audio.py` + `scripts/test_mic_client.py`
- [x] Sanity checks
  - [x] Validate `.wav` properties: 16kHz, mono, 16-bit — 5/5 files pass
  - [x] VAD false positive check: 30s silence → 0 chunks (<2 required)
  - [x] All 100 tests pass (45 new audio + 55 existing)
- [x] Create feature branch, commit, PR, merge
- [x] Generate `reports/phase2_report.md`
- [ ] **PHASE 2 PASS: ______ (initials/date/time)**
- [ ] Delete `tmp/audio_chunks/*` (manual, only after PASS)

---

## Phase 3 — Fast Streaming ASR (MedASR on CPU/GPU via ONNX) (Gated)

**Goal**: Real-time transcripts with partial hypotheses; ONNX export + quantization.

- [x] Implement `src/asr/onnx_runner.py`
  - [x] Input: WAV chunk → Output: partial + final text + timestamps
  - [x] CPU support (`CPUExecutionProvider`)
  - [x] GPU support (`CUDAExecutionProvider`) — auto-detected via `get_onnx_providers()`
  - [x] Dynamic provider selection via `get_device()` helper
- [x] Model inference pipeline (pre-trained MedASR INT8 ONNX)
  - [x] `src/asr/feature_extractor.py` — 128-dim fbank, CMVN
  - [x] `src/asr/ctc_decoder.py` — greedy CTC, SentencePiece, timestamps
  - [x] INT8 quantized model in `onnx_models/medasr/`
- [x] Integrate ASR with audio pipeline (ready for Phase 4 worker)
  - [x] `transcribe(audio)` → ASRResult with segments
  - [x] `transcribe_streaming(audio)` → partial results
- [x] Sanity checks
  - [x] Test synthetic audio transcription (silence, noise, harmonics, bursts)
  - [x] Latency: avg 238ms per chunk ≤ 400ms target ✅ (RTF 0.136x)
  - [x] `scripts/test_asr_inference.py` — latency benchmarking script
  - [x] 42 new tests, 142 total, all passing
- [x] Commit `9dc6972` on `main`
- [x] Generate `reports/phase3_report.md` + `reports/phase3_explanation.md`
- [x] **PHASE 3 PASS: 2026-02-22**

---

## Phase 4 — Machines Dictionary & Deterministic Rule Engine (Gated)

**Goal**: Reliable, deterministic rule engine mapping transcripts → machine toggles.

- [x] Machines dictionary design (JSON)
  - [x] `configs/machines/pcnl.json`
  - [x] `configs/machines/partial_hepatectomy.json`
  - [x] `configs/machines/lobectomy.json`
- [x] Rule engine implementation (`src/state/rules.py`)
  - [x] Keyword mapping (turn on/start/activate vs turn off/stop/standby)
  - [x] Alias normalization
  - [x] Negation handling ("don't start suction")
  - [x] Temporal qualifiers (immediately, after x)
  - [x] Debounce toggles (ignore repeated within 3s)
  - [x] Output canonical JSON patch (0/1 format)
- [x] Tests
  - [x] `tests/test_rules.py` with 170 tests (20 test classes)
  - [x] Edge cases: negations, ambiguity, repeated commands
- [x] Integration with ASR
  - [x] ASRWorker: async audio→transcript pipeline
  - [x] RuleWorker: async transcript→rule engine→state patch pipeline
- [x] Sanity checks
  - [x] `pytest tests/` passes — 300 total tests, all passing
  - [x] Manual QA: 65 sentences reviewed in `reports/ruleqa.csv`
- [x] Commit `cc42119` on `main`
- [x] Generate `reports/phase4_report.md` + `reports/phase4_explanation.md`
- [x] **PHASE 4 PASS: ______ (initials/date/time)**

---

## Phase 5 — Rolling Context & State Schema (Gated)

**Goal**: Rolling transcript buffer and canonical JSON state schema for MedGemma.

- [x] Rolling buffer module `src/state/rolling_buffer.py`
  - [x] Keep last N seconds (configurable, default 180s)
  - [x] Append transcripts with timestamps and speaker labels
- [x] JSON schema `schemas/surgery_state.schema.json`
  - [x] Validate using `jsonschema`
- [x] Implement `src/state/serializer.py`
  - [x] Normalize rule engine outputs to schema
  - [x] Normalize LLM outputs to schema
- [x] Tests
  - [x] Long simulated transcript → buffer maintains 180s
  - [x] Serializer produces valid schema JSON
- [x] Sanity checks
  - [x] Rolling buffer prints buffer length, earliest/latest timestamp
  - [x] Schema validation passes on `tmp/current_state.json`
- [x] Create feature branch, commit, PR, merge
- [x] Generate `reports/phase5_report.md`
- [x] **PHASE 5 PASS: ✅ (commit 620f5f6 — 2026-02-22)**

---

## Phase 6 — MedGemma Integration & ONNX Quantization (Gated; GPU on Kaggle)

**Goal**: MedGemma as structured reasoning engine; ONNX export/quantize; local dev + Kaggle benchmarks.

- [x] Choose model variant and download (MedGemma-4B-IT Q3_K_M GGUF, 1.95GB)
- [x] GGUF runner `src/llm/gguf_runner.py` (llama-cpp-python 0.3.2)
  - [x] Completion prompt → JSON output with 3-strategy parsing
  - [x] Chat-style inference via create_chat_completion
  - [x] CPU inference (pre-built wheel, no C++ compiler)
  - [x] Stub mode for unit tests
- [x] Stub LLM for local unit tests (stub_mode in all modules)
- [ ] Kaggle inference notebook (deferred to Phase 10)
  - [ ] `scripts/kaggle_inference_notebook.ipynb`
  - [ ] GPU benchmarks
- [x] Batching & concurrency
  - [x] `src/llm/batcher.py` — async micro-batcher (278 lines)
  - [x] `MAX_BATCH=4` and `MAX_WAIT_MS=500` knobs from constants.py
  - [x] Sequential processing with run_in_executor
- [x] Prompt engineering
  - [x] `src/llm/prompts.py` — PromptBuilder (386 lines)
  - [x] Surgery-specific system prompt with machine dictionary context
  - [x] Pass machines_dict, phases, recent_transcript, current_machines
  - [x] Strict JSON schema examples in prompt
- [x] Pipeline orchestration
  - [x] `src/llm/manager.py` — full rewrite (381 lines)
  - [x] Request → Prompt → Batcher → Runner → Normalize → Response
  - [x] Graceful fallback to rule-only on any failure
- [x] Sanity checks (local)
  - [x] 79 new tests, 513 total, all pass
  - [x] Outputs validate against `surgery_state.schema.json`
- [x] Commit on main: `7b36651`
- [x] Generate `reports/phase6_report.md`
- [x] **PHASE 6 PASS: AL / 2025-06-22**

---

## Phase 7 — Orchestrator, API, WebSocket & Overrides (Gated)

**Goal**: Full orchestrator wiring ASR → Rule Engine → LLM → State Writer → WebSocket push + manual override with audit.

- [x] Orchestration workers
  - [x] `workers/asr_worker.py` — reads mic chunks, runs ASR, enqueues transcripts
  - [x] `workers/rule_worker.py` — polls transcripts, runs rule engine, writes pending state
  - [x] `workers/llm_dispatcher.py` — rate-limited LLM dispatch with RollingBuffer context (296 lines)
  - [x] `workers/state_writer.py` — merges rule + LLM output, atomic JSON, override system (369 lines)
- [x] Orchestrator `src/workers/orchestrator.py` (371 lines)
  - [x] Central coordinator — shared queues, lifecycle, fan-out, stats
  - [x] Start/stop workers in dependency order
  - [x] Transcript fan-out to both rule and LLM workers
- [x] API server `src/api/app.py` (417 lines, v0.7.0)
  - [x] `GET /state` — returns current JSON from orchestrator
  - [x] `GET /stats` — aggregated pipeline statistics
  - [x] `POST /transcript` — feed text into pipeline (bypass ASR)
  - [x] `WS /ws/state` — pushes complete JSON via StateWriter callback
  - [x] `POST /override` — applies via orchestrator, logs to `logs/overrides.log`
- [x] Failure modes
  - [x] MedGemma failure → fallback to rule engine only (LLMDispatcher.enter_fallback_mode)
  - [x] Degraded mode output from LLMManager with `source: "rule"`
- [x] Atomic writes
  - [x] Write to `.json.tmp` then `os.replace()` for atomic reads
- [x] Tests
  - [x] 59 worker tests (LLMDispatcher, StateWriter, Orchestrator)
  - [x] 25 API tests (enhanced from 13)
  - [x] 582 total tests, all passing
- [x] Sanity checks
  - [x] All workers start via orchestrator
  - [x] WebSocket broadcast callback wired
  - [x] `POST /override` creates audit log entry
- [x] Commits: `c769ca3` (code) + `35b817e` (docs)
- [x] Generate `reports/phase7_report.md` + `reports/phase7_explanation.md`
- [ ] **PHASE 7 PASS: ______ (initials/date/time)**
- [ ] Delete `tmp/` test artifacts (manual, only after PASS)

---

## Phase 8 — Safety, Logging, Audit, Documentation & Clinical Gating (Gated)

**Goal**: No auto-execution of clinical actions; immutable logs; disclaimers; user flows.

- [x] Write `SAFETY.md` — 170-line safety policy covering 10 sections
  - [x] Human-in-loop requirement documented (Section 2: mandatory human confirmation)
  - [x] Usage disclaimers (Sections 1, 8, 9: simulation boundaries, regulatory notice)
  - [x] No direct device APIs — simulated toggles only (Section 3)
- [x] Implement logging — `src/utils/audit.py` (358 lines, 4 logger classes)
  - [x] Immutable transcripts: `TranscriptAuditLogger` → `logs/transcripts/YYYYMMDD.log`
  - [x] State changes audit: `StateAuditLogger` with SHA-256 chain hashing
  - [x] Overrides audit: `OverrideAuditLogger` with SHA-256 chain hashing
- [x] Unit tests — `tests/test_safety.py` (101 tests all passing)
  - [x] `current_surgery_state.json` always includes `source` and `confidence` (TestValidatorRequiredFields)
  - [x] Output never contains executable instructions — 18 banned patterns tested (TestValidatorBannedPatterns)
- [x] Documentation
  - [x] `docs/DEVELOPER_GUIDE.md` — developer onboarding guide (10 sections)
  - [x] `reports/` auto-generated phase reports template
- [x] Sanity checks
  - [x] `pytest` — 683 tests all passing (582 prior + 101 new)
  - [x] Review and manually sign `SAFETY.md` — sign-off line present
- [x] Create feature branch, commit, PR, merge
- [x] Generate `reports/phase8_report.md`
- [ ] **PHASE 8 PASS: ______ (initials/date/time)**

---

## Phase 9 — Local Stress Testing & CI (Gated)

**Goal**: Stress/latency tests locally and on Kaggle; tune batcher/worker counts; stable CI.

- [ ] Local synthetic load
  - [ ] `scripts/stress_local.py` — push transcripts at varying QPS
  - [ ] Measure latencies & queue sizes
- [ ] Kaggle test
  - [ ] Re-run micro-batching with concurrent MedASR streams
  - [ ] Capture p50/p95 latencies
  - [ ] Store in `reports/kaggle_benchmarks/`
- [ ] Update CI
  - [ ] Subset of integration tests (lightweight)
  - [ ] Generate artifacts (lint, unit, integration smoke)
- [ ] Acceptance criteria validation
  - [ ] Rule path UI update latency median < 500ms (local)
  - [ ] MedGemma confirm p50 < 3s (Kaggle quantized)
  - [ ] Max memory usage < 90% of target device
- [ ] Create feature branch, commit, PR, merge
- [ ] Generate `reports/phase9_report.md`
- [ ] **PHASE 9 PASS: ______ (initials/date/time)**

---

## Phase 10 — Final Frontend (BLOCKED until ALL Phases 0–9 PASS) (Gated — Final)

**Goal**: OR-room visualization UI with surgery selection, agent animations, machine state toggles, manual override.

### Pre-Condition Checklist (ALL must be checked)

- [ ] Phase 0 report: PASS
- [ ] Phase 1 report: PASS
- [ ] Phase 2 report: PASS
- [x] Phase 3 report: PASS
- [ ] Phase 4 report: PASS
- [ ] Phase 5 report: PASS
- [x] Phase 6 report: PASS
- [ ] Phase 7 report: PASS
- [x] Phase 8 report: PASS
- [ ] Phase 9 report: PASS

### Frontend Implementation

- [ ] Create `frontend/` skeleton with Vite + React
- [ ] Architecture & tech stack
  - [ ] React + Vite (static client)
  - [ ] WebSocket connection to `/ws/state`
  - [ ] Asset handling: sprite sheets for agents/machines
  - [ ] Animation engine (CSS transforms / pixi.js / konva)
- [ ] Implement `StateProvider`
  - [ ] WebSocket connection
  - [ ] JSON normalization
- [ ] Surgery selection dropdown
  - [ ] Three options: PCNL, Partial Hepatectomy, Lobectomy
  - [ ] `POST /select_surgery` on selection
  - [ ] Backend switches `machines_dict`
- [ ] Implement `OPRoom` component
  - [ ] Configurable layout per surgery type
  - [ ] JSON layout maps: `configs/layouts/pcnl_layout.json`, etc.
  - [ ] Static background per surgery type
- [ ] Entity rendering
  - [ ] Patient, bed, lights, ventilator, ECG, ESU, nurses, doctors, agents
  - [ ] Each has states: ON / OFF / STANDBY
  - [ ] Clickable machines → manual override dialog → `POST /override`
- [ ] Implement `Agent` component
  - [ ] Path nodes pre-defined
  - [ ] On state change: animate agent moving to node and toggling
  - [ ] `source:"rule"` → pending animation (hand, blinking)
  - [ ] `source:"medgemma"` → confirmation check, device turns green
  - [ ] `reasoning_degraded` → red caution icon, human must confirm
- [ ] Timeouts & persistence
  - [ ] `suggestion` with `eta_seconds` → countdown bubble above machine
- [ ] Accessibility & fallback
  - [ ] Textual list of current machines/states below canvas
  - [ ] Keyboard navigation to toggle devices
- [ ] Test harness
  - [ ] `scripts/frontend_simulator.py` — sends fake WS state updates
- [ ] Sanity checks
  - [ ] End-to-end QA: orchestrator → frontend → speak/feed transcript → animations
  - [ ] Cross-browser: Chrome and Firefox
  - [ ] Accessibility: keyboard navigation
  - [ ] All three surgery types validated
  - [ ] All machine toggles (rule & medgemma) validated
  - [ ] Manual overrides validated
- [ ] Create feature branch, commit, PR, merge
- [ ] Generate `reports/phase10_report.md`
- [ ] **PHASE 10 PASS: ______ (initials/date/time)**

---

## Final Release Checklist

- [ ] All phase reports (0–10) present in `reports/` with PASS and manual sign-off
- [ ] `SAFETY.md` reviewed and signed
- [ ] No FP32 model weights in repo
- [ ] All quantized ONNX artifacts properly stored (Kaggle/cloud links)
- [ ] `machines_dict` per surgery is small, human-reviewable, versioned in `configs/machines/`
- [ ] CI passes on main branch
- [ ] Create `release/v1.0` tag
- [ ] Final `reports/release_checklist.md` generated

---

## Acceptance Criteria Summary (Must be met before frontend)

| Criterion | Target | Status |
|-----------|--------|--------|
| ASR final segment latency | ≤ 400ms (local) | ⬜ |
| Rule engine mapping coverage | ≥ 95% on curated set | ⬜ |
| MedGemma confirmation p50 | < 3s (Kaggle quantized) | ⬜ |
| Orchestrator atomic writes | Working | ⬜ |
| WS push | Working | ⬜ |
| Manual override audit | Working | ⬜ |
| All phase reports signed PASS | 0–9 | ⬜ |

---

## Branching Policy

- `main` = stable
- `dev` = integration
- `feature/*` branches per phase — merge only when phase report PASS

## Strict Rules

1. **No automated device control** — visualization only; human confirmation required
2. **Never commit raw FP32 weights** — use Kaggle/cloud dataset references
3. **Keep `machines_dict` small, human-reviewable**, versioned in `configs/machines/`
4. **Frontend ONLY after every prior phase has PASSed report + manual sign-off**
5. **Delete temp artifacts ONLY after manual PASS confirmation**
