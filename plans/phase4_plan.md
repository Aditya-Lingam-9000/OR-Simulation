# Phase 4 Plan: Machines Dictionary & Deterministic Rule Engine

**Date:** 2026-02-22
**Depends on:** Phase 3 (ASR inference pipeline) ✅

---

## What Exists (from Phase 1 skeleton)

- `data/surgeries_machines.json` — 3 surgeries × 9 machines, basic triggers
- `src/state/rules.py` — RuleEngine with keyword matching, negation, debounce
- `tests/test_rules.py` — 13 basic tests
- `src/workers/asr_worker.py` / `rule_worker.py` — placeholder stubs

## What Phase 4 Builds

### 1. Per-Surgery Machine Config Files
- Split `data/surgeries_machines.json` into `configs/machines/{pcnl,hepatectomy,lobectomy}.json`
- Add rich alias sets for each machine (multiple trigger phrases)
- Add regex trigger patterns for flexible matching
- Update RuleEngine to load from new config path

### 2. Enhanced Rule Engine (`src/state/rules.py`)
- **Regex-based machine matching** — not just substring `in`, use word-boundary regex
- **Rich alias expansion** — e.g., "vent", "ventilator", "breathing machine" all match M03
- **Temporal qualifier detection** — detect "immediately", "after X", "when ready"
- **Phase-aware filtering** — only match machines valid for the declared surgical phase
- **Confidence scoring** — grade match quality (exact trigger > alias > partial match)
- **Multi-machine commands** — "turn on suction and irrigation"
- **Standby action** — separate from OFF; "put ventilator on standby"

### 3. ASR→Rules Integration (`src/workers/asr_worker.py`, `rule_worker.py`)
- Wire OnnxASRRunner → RuleEngine via async queues
- ASRWorker: consume audio chunks → transcribe → enqueue transcript
- RuleWorker: consume transcripts → rule engine → produce state updates
- TranscriptPipeline: end-to-end connector class

### 4. Comprehensive Test Suite
- Expand to 100+ tests covering:
  - All 3 surgeries × all machines
  - Activate/deactivate/standby actions
  - Negation variants
  - Multi-machine commands
  - Temporal qualifiers
  - Phase filtering
  - Debounce
  - Edge cases (empty, gibberish, ambiguous)

### 5. QA Validation
- `reports/ruleqa.csv` — 50 sample sentences with expected outputs

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `configs/machines/pcnl.json` | Create |
| `configs/machines/partial_hepatectomy.json` | Create |
| `configs/machines/lobectomy.json` | Create |
| `src/state/rules.py` | Major rewrite |
| `src/workers/asr_worker.py` | Implement real worker |
| `src/workers/rule_worker.py` | Implement real worker |
| `tests/test_rules.py` | Expand to 100+ tests |
| `reports/ruleqa.csv` | Create |

---

## Acceptance Criteria

- [ ] 3 per-surgery config files with 5+ triggers per machine
- [ ] Rule engine handles: aliases, negation, temporal, phase-aware, multi-machine
- [ ] ASR→Rules pipeline works end-to-end
- [ ] 100+ tests, all passing
- [ ] Latency < 500ms for rule processing
- [ ] QA CSV with 50+ sentences
