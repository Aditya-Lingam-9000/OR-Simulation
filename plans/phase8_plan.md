# Phase 8 Plan: Safety, Logging, Audit, Documentation & Clinical Gating

## Objective
No auto-execution of clinical actions; immutable logs; disclaimers; user flows.
SHA-256 checksums on state changes and overrides for tamper detection.

## Deliverables

### 1. SAFETY.md (new)
- Human-in-loop requirement documented
- Usage disclaimers and regulatory notices
- No direct device APIs — simulated toggles only
- Operator responsibilities

### 2. Immutable Transcript Logging (enhance existing)
- `src/utils/audit.py` — new module
  - `TranscriptLogger`: writes to `logs/transcripts/YYYYMMDD.log` (append-only)
  - `StateAuditLogger`: writes state changes with SHA-256 checksums
  - `OverrideAuditLogger`: writes overrides with SHA-256 checksums
  - Each log entry includes a SHA-256 hash of the payload
  - Chain hashing: each entry includes hash of previous entry (tamper chain)
- Enhance `logging_config.py` to use audit loggers
- Wire `StateWriter` to call audit logger on each write
- Wire `StateWriter` override log to use checksummed entries

### 3. Safety Validation Module
- `src/utils/safety.py` — new module
  - `validate_output_safe(state_dict)` → ensures no executable instructions
  - `validate_has_source_and_confidence(state_dict)` → checks required fields
  - `is_suggestion_only(text)` → verifies text is a suggestion, not a command
  - `BANNED_PATTERNS` — regex list of unsafe output patterns
  - Called by StateWriter before broadcasting

### 4. Documentation
- `docs/DEVELOPER_GUIDE.md` — developer onboarding
  - Project overview, architecture, setup instructions
  - Running tests, API usage, pipeline explanation
  - Safety considerations

### 5. Tests
- `tests/test_safety.py` — safety + audit tests
  - Output always has `source` and `confidence`
  - Output never contains executable instructions
  - Transcript logger produces append-only files
  - State audit logger includes SHA-256 checksums
  - Override audit logger includes SHA-256 checksums
  - Safety validator catches banned patterns
  - Chain hash integrity verification

## Existing Infrastructure
- `logging_config.py` already has `get_transcript_logger()` and `get_state_change_logger()`
- `constants.py` has `SAFETY_DISCLAIMER`, `LOGS_DIR`, `JSON_OUTPUT_KEYS`, `VALID_SOURCES`
- `state_writer.py` already has `_log_override()` writing to `logs/overrides.log`
- `app.py` already shows disclaimer in /health endpoint
- `README.md` and `LICENSE` have safety notices

## Implementation Order
1. Create `SAFETY.md`
2. Create `src/utils/audit.py` with checksummed loggers
3. Create `src/utils/safety.py` with output validators
4. Enhance `StateWriter` to use audit loggers + safety validation
5. Create `docs/DEVELOPER_GUIDE.md`
6. Create `tests/test_safety.py`
7. Run all tests
8. Commit + reports
