# OR-Symphony: Master Project Plan — Full Checkbox Tracker

> **Project**: OR-Symphony: Predictive Surgical State Engine  
> **Created**: 2026-02-22  
> **Status**: NOT STARTED  
> **Rule**: Frontend (Phase 10) is BLOCKED until ALL prior phases have PASS reports with manual sign-off.  
> **Rule**: Only delete temp/test artifacts after the phase report is PASS and you manually confirm.

---

## Pre-Start One-Time Checklist

- [ ] Install Python 3.10+ and `git`
- [ ] Install VS Code with Git and Python extensions
- [ ] Sign into GitHub Copilot Pro
- [ ] Create GitHub repo (private)
- [ ] Create Kaggle account (for GPU runs)
- [ ] Optional: Lightning AI account for short GPU sessions
- [ ] Create top-level project folder

---

## Phase 0 — Repo + Environment (Gated: must PASS before any coding)

**Goal**: Clean, versioned repo with venv and CI skeleton.

- [ ] Create repository locally with initial commit
  - [ ] Create `README.md`
  - [ ] Create `LICENSE`
  - [ ] Create `.gitignore`
  - [ ] Initial git commit
- [ ] Create remote repo on GitHub and connect
  - [ ] Add remote origin
  - [ ] Push to main
- [ ] Create venv + baseline requirements
  - [ ] Create virtual environment
  - [ ] Install and upgrade pip
  - [ ] Create `requirements.txt` with all dependencies
  - [ ] Install requirements successfully
- [ ] Add CI skeleton (`.github/workflows/ci.yml`)
  - [ ] Lint job configured
  - [ ] Unit test job configured
- [ ] Sanity checks
  - [ ] `python -c "import onnxruntime, torch, transformers, fastapi; print('ok')"` passes
  - [ ] CI file appears on GitHub and runs
- [ ] Commit and push all Phase 0 artifacts
- [ ] Generate `reports/phase0_report.md`
- [ ] **PHASE 0 PASS: ______ (initials/date/time)**

---

## Phase 1 — Local Architecture & Skeleton Code (Gated)

**Goal**: Directory layout, module interfaces, API endpoint stubs, config and schema artifacts.

- [ ] Create project directory structure
  - [ ] `src/ingest/` — mic/web input, VAD bridge
  - [ ] `src/asr/` — MedASR wrappers + streaming adapters
  - [ ] `src/state/` — rolling buffer, rule-engine, schema
  - [ ] `src/llm/` — MedGemma runner, queue/dispatcher
  - [ ] `src/api/` — FastAPI/WS endpoints
  - [ ] `src/workers/` — orchestrator services
  - [ ] `src/utils/` — device helpers, logging, config
  - [ ] `tests/`
  - [ ] `scripts/`
  - [ ] `onnx_models/` (git-lfs or excluded)
  - [ ] `models/` (excluded)
  - [ ] `reports/`
  - [ ] `configs/`
  - [ ] `schemas/`
  - [ ] `logs/`
- [ ] Create key interface modules (with TODOs)
  - [ ] `src/ingest/mic_server.py` — browser mic via WebRTC or local mic wrapper
  - [ ] `src/asr/runner.py` — streaming ASR runner (abstract + ONNX impl)
  - [ ] `src/state/rules.py` — machines dictionary loader + rule engine
  - [ ] `src/llm/manager.py` — request queue + batcher skeleton
  - [ ] `src/api/app.py` — FastAPI + WS skeleton with `/health` endpoint
  - [ ] `src/utils/device.py` — device helper (CPU/GPU detection)
  - [ ] `src/utils/logging_config.py` — logging setup
  - [ ] `src/utils/config.py` — configuration management
- [ ] Create placeholder worker modules
  - [ ] `src/workers/asr_worker.py`
  - [ ] `src/workers/rule_worker.py`
  - [ ] `src/workers/llm_dispatcher.py`
  - [ ] `src/workers/state_writer.py`
- [ ] Sanity checks
  - [ ] `pytest` runs and passes (empty/stub tests)
  - [ ] `uvicorn src.api.app:app --reload` starts successfully
  - [ ] `curl http://127.0.0.1:8000/health` returns `{"status":"ok"}`
- [ ] Create feature branch, commit, PR, merge
- [ ] Generate `reports/phase1_report.md`
- [ ] **PHASE 1 PASS: ______ (initials/date/time)**

---

## Phase 2 — Audio Capture & Streaming (Gated)

**Goal**: Robust mic capture, VAD, chunking policy, temporary audio queue.

- [ ] Choose input path (WebRTC or desktop mic)
- [ ] Implement `scripts/test_mic_client.py` for local mic capture
- [ ] Implement VAD + chunker
  - [ ] `src/ingest/mic_stream.py` using `webrtcvad`
  - [ ] Produce 0.5–1.5s speech chunks
  - [ ] Configurable overlap 200–500ms
- [ ] Save chunks to in-memory deque
- [ ] Save chunks to `tmp/audio_chunks/` for replay
- [ ] Add `scripts/mic_test.sh` (record 10s, create sample chunks)
- [ ] Sanity checks
  - [ ] Validate `.wav` properties: 16kHz, mono, 16-bit
  - [ ] VAD false positive check: 30s silence → <2 chunks
  - [ ] Latency check: audio end → chunk in queue < 50ms
- [ ] Create feature branch, commit, PR, merge
- [ ] Generate `reports/phase2_report.md`
- [ ] **PHASE 2 PASS: ______ (initials/date/time)**
- [ ] Delete `tmp/audio_chunks/*` (manual, only after PASS)

---

## Phase 3 — Fast Streaming ASR (MedASR on CPU/GPU via ONNX) (Gated)

**Goal**: Real-time transcripts with partial hypotheses; ONNX export + quantization.

- [ ] Implement `src/asr/onnx_runner.py`
  - [ ] Input: WAV chunk → Output: partial + final text + timestamps
  - [ ] CPU support (`CPUExecutionProvider`)
  - [ ] GPU support (`CUDAExecutionProvider`)
  - [ ] Dynamic provider selection via `get_device()` helper
- [ ] Export & quantize ASR model
  - [ ] Create `scripts/export_asr_to_onnx.py`
  - [ ] Export with dynamic axes
  - [ ] Quantize to INT8/4-bit
  - [ ] Store in `onnx_models/`
- [ ] Provide lightweight fallback ASR (Whisper tiny or similar)
- [ ] Integrate ASR with mic chunking
  - [ ] Final chunk → transcript file in `tmp/transcripts/`
  - [ ] Enqueue to LLM queue
- [ ] Sanity checks
  - [ ] Test recorded file transcription
  - [ ] Latency: per 1s chunk ≤ 400ms on dev CPU
  - [ ] Record timings to `logs/asr_latency.csv`
  - [ ] WER test on small annotated sample
- [ ] Create feature branch, commit, PR, merge
- [ ] Generate `reports/phase3_report.md`
- [ ] **PHASE 3 PASS: ______ (initials/date/time)**
- [ ] Delete `tmp/transcripts/` (manual, only after PASS)

---

## Phase 4 — Machines Dictionary & Deterministic Rule Engine (Gated)

**Goal**: Reliable, deterministic rule engine mapping transcripts → machine toggles.

- [ ] Machines dictionary design (JSON)
  - [ ] `configs/machines/pcnl.json`
  - [ ] `configs/machines/partial_hepatectomy.json`
  - [ ] `configs/machines/lobectomy.json`
- [ ] Rule engine implementation (`src/state/rules.py`)
  - [ ] Keyword mapping (turn on/start/activate vs turn off/stop/standby)
  - [ ] Alias normalization
  - [ ] Negation handling ("don't start suction")
  - [ ] Temporal qualifiers (immediately, after x)
  - [ ] Debounce toggles (ignore repeated within 3s)
  - [ ] Output canonical JSON patch (0/1 format)
- [ ] Tests
  - [ ] `tests/rules/test_rules.py` with ~100 synthetic utterances
  - [ ] Edge cases: negations, ambiguity, repeated commands
- [ ] Integration with ASR
  - [ ] Final transcript → rule engine → `tmp/pending_update.json` with `source:"rule"`
- [ ] Sanity checks
  - [ ] `pytest tests/rules` passes
  - [ ] Manual QA: ~50 real-life sample sentences reviewed in `reports/ruleqa.csv`
- [ ] Create feature branch, commit, PR, merge
- [ ] Generate `reports/phase4_report.md`
- [ ] **PHASE 4 PASS: ______ (initials/date/time)**
- [ ] Delete `reports/ruleqa.csv` (manual, only after PASS)

---

## Phase 5 — Rolling Context & State Schema (Gated)

**Goal**: Rolling transcript buffer and canonical JSON state schema for MedGemma.

- [ ] Rolling buffer module `src/state/rolling_buffer.py`
  - [ ] Keep last N seconds (configurable, default 180s)
  - [ ] Append transcripts with timestamps and speaker labels
- [ ] JSON schema `schemas/surgery_state.schema.json`
  - [ ] Validate using `jsonschema`
- [ ] Implement `src/state/serializer.py`
  - [ ] Normalize rule engine outputs to schema
  - [ ] Normalize LLM outputs to schema
- [ ] Tests
  - [ ] Long simulated transcript → buffer maintains 180s
  - [ ] Serializer produces valid schema JSON
- [ ] Sanity checks
  - [ ] Rolling buffer prints buffer length, earliest/latest timestamp
  - [ ] Schema validation passes on `tmp/current_state.json`
- [ ] Create feature branch, commit, PR, merge
- [ ] Generate `reports/phase5_report.md`
- [ ] **PHASE 5 PASS: ______ (initials/date/time)**

---

## Phase 6 — MedGemma Integration & ONNX Quantization (Gated; GPU on Kaggle)

**Goal**: MedGemma as structured reasoning engine; ONNX export/quantize; local dev + Kaggle benchmarks.

- [ ] Choose model variant and download (license compliance)
- [ ] Local ONNX runner `src/llm/onnx_runner.py`
  - [ ] Batch prompt input → JSON output
  - [ ] `CUDAExecutionProvider` on Kaggle
  - [ ] `CPUExecutionProvider` locally
  - [ ] `device_id` option for multi-GPU
- [ ] Stub LLM for local unit tests (tiny distilled Gemma or mock)
- [ ] Kaggle inference notebook
  - [ ] `scripts/kaggle_inference_notebook.ipynb`
  - [ ] Upload quantized ONNX artifact to Kaggle datasets
  - [ ] Load model, run tests, record GPU memory & latency
  - [ ] Save logs to `reports/kaggle_benchmarks/`
- [ ] Batching & concurrency
  - [ ] `src/llm/batcher.py` — micro-batcher
  - [ ] `MAX_BATCH` and `MAX_WAIT_MS` knobs
  - [ ] Test on Kaggle with multiple replicas
- [ ] Prompt engineering
  - [ ] System prompt template with `system_ruleset_id`
  - [ ] Pass `machines_dict` and `recent_transcript`
  - [ ] Strict JSON schema examples in prompt
- [ ] Validation
  - [ ] N curated transcripts → MedGemma → compare JSON to expected state
  - [ ] Compute match rate and confidence threshold
- [ ] Sanity checks (Kaggle)
  - [ ] 100 prompts: median latency, p95 latency, GPU memory reported
  - [ ] `reports/kaggle_benchmarks/bench_YYYYMMDD.json` uploaded
- [ ] Sanity checks (local)
  - [ ] Fallback small LLM integration tests pass
  - [ ] Outputs validate against `surgery_state.schema.json`
- [ ] Create feature branch, commit, PR, merge
- [ ] Generate `reports/phase6_report.md`
- [ ] **PHASE 6 PASS: ______ (initials/date/time)**
- [ ] Delete local large artifact copies (manual, only after PASS; keep Kaggle dataset link)

---

## Phase 7 — Orchestrator, API, WebSocket & Overrides (Gated)

**Goal**: Full orchestrator wiring ASR → Rule Engine → LLM → State Writer → WebSocket push + manual override with audit.

- [ ] Orchestration workers
  - [ ] `workers/asr_worker.py` — reads mic chunks, runs ASR, enqueues transcripts
  - [ ] `workers/rule_worker.py` — polls transcripts, runs rule engine, writes pending state
  - [ ] `workers/llm_dispatcher.py` — batches requests to MedGemma workers
  - [ ] `workers/state_writer.py` — merges rule + medgemma output, writes atomic JSON
- [ ] API server `src/api/app.py`
  - [ ] `GET /state` — returns current JSON
  - [ ] `WS /ws/state` — pushes complete JSON or diffs
  - [ ] `POST /override` — requires auth token, appends to `logs/overrides.log`
- [ ] Failure modes
  - [ ] MedGemma failure → fallback to rule engine only
  - [ ] Add `meta.reasoning="degraded"` in JSON on fallback
  - [ ] p95 latency threshold triggers degraded mode
- [ ] Atomic writes
  - [ ] Write to `.tmp` then `os.replace()` for atomic reads
- [ ] Tests
  - [ ] End-to-end integration with `samples/pcnl_recording.wav`
  - [ ] Full PCNL workflow simulation
- [ ] Sanity checks
  - [ ] All workers start locally (separate terminals)
  - [ ] WebSocket client receives updates
  - [ ] `POST /override` creates audit log entry
- [ ] Create feature branch, commit, PR, merge
- [ ] Generate `reports/phase7_report.md`
- [ ] **PHASE 7 PASS: ______ (initials/date/time)**
- [ ] Delete `tmp/` test artifacts (manual, only after PASS)

---

## Phase 8 — Safety, Logging, Audit, Documentation & Clinical Gating (Gated)

**Goal**: No auto-execution of clinical actions; immutable logs; disclaimers; user flows.

- [ ] Write `SAFETY.md`
  - [ ] Human-in-loop requirement documented
  - [ ] Usage disclaimers
  - [ ] No direct device APIs — simulated toggles only
- [ ] Implement logging
  - [ ] Immutable transcripts: `logs/transcripts/YYYYMMDD.log` (append only)
  - [ ] State changes audit with `sha256` checksums
  - [ ] Overrides audit with `sha256` checksums
- [ ] Unit tests
  - [ ] `current_surgery_state.json` always includes `source` and `confidence`
  - [ ] Output never contains executable instructions (suggestions only)
- [ ] Documentation
  - [ ] `docs/` developer onboarding guide
  - [ ] `reports/` auto-generated phase reports template
- [ ] Sanity checks
  - [ ] `pytest` for safety tests passes
  - [ ] Review and manually sign `SAFETY.md`
- [ ] Create feature branch, commit, PR, merge
- [ ] Generate `reports/phase8_report.md`
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
- [ ] Phase 3 report: PASS
- [ ] Phase 4 report: PASS
- [ ] Phase 5 report: PASS
- [ ] Phase 6 report: PASS
- [ ] Phase 7 report: PASS
- [ ] Phase 8 report: PASS
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
