# Phase 8 Explanation — Safety, Logging, Audit, Documentation & Clinical Gating

## What Was Built

Phase 8 adds the safety and compliance layer that ensures OR-Symphony never issues executable instructions or controls real devices. It consists of five interconnected components:

### 1. Safety Policy (SAFETY.md)

The safety policy document is the single source of truth for how the system must behave in a medical environment. It establishes:

- **Human-in-the-loop is mandatory** — every suggestion requires a human operator to read, evaluate, and manually act. The system never executes actions autonomously.
- **Simulation boundaries** — the system simulates equipment state tracking. It does NOT interface with real medical devices, patient monitors, or hospital systems.
- **Output format** — every output includes its source (rule engine, LLM, or both), a confidence score, and uses advisory language ("consider", "suggest", "recommend").
- **Degraded mode** — if the LLM fails or connectivity is lost, the rule engine continues providing deterministic state tracking with its own confidence scores.

### 2. Audit Logging (src/utils/audit.py)

The audit module provides tamper-evident logging using SHA-256 hash chains:

```
Entry 1: payload → sha256(payload) = H1, prev_hash = "genesis"
Entry 2: payload → sha256(payload) = H2, prev_hash = H1
Entry 3: payload → sha256(payload) = H3, prev_hash = H2
```

If an attacker modifies Entry 2's payload, H2 changes, and Entry 3's `prev_hash` no longer matches — the chain is broken. The `verify_chain()` method detects this by re-computing every hash and checking the chain references.

**Four specialized loggers:**
- `AuditLogger` — base class with the chain mechanism
- `TranscriptAuditLogger` — writes daily transcript files (no chain, but each entry is individually hashed)
- `StateAuditLogger` — chains state change events, extracting essential fields (surgery, phase, source, confidence, machine states)
- `OverrideAuditLogger` — chains manual override events (machine, action, reason, operator)

### 3. Safety Validation (src/utils/safety.py)

The `SafetyValidator` runs five checks on every state output before it reaches the frontend:

1. **Required fields** — `source`, `confidence`, and `metadata` must be present
2. **Valid source** — must be `rule`, `medgemma`, or `rule+medgemma`
3. **Confidence range** — must be between 0.0 and 1.0
4. **Banned patterns** — 18 compiled regular expressions catch:
   - Device control language: "execute command", "force on", "auto-activate", "bypass safety"
   - Hardware API calls: "serial.write", "gpio", "api.send"
   - Clinical decision language: "administer 50mg", "prescribe", "diagnose: X"
5. **Advisory language** — warns (but doesn't block) if suggestions don't start with advisory prefixes

### 4. StateWriter Integration

The StateWriter now gates its WebSocket broadcast on safety validation:

```
state_dict → SafetyValidator.validate_output()
  ├── Safe → write to disk → log to audit → broadcast via WebSocket
  └── Unsafe → write to disk (with violations in metadata) → log to audit → NO broadcast
```

This "write-then-gate" pattern ensures:
- Unsafe outputs never reach the UI
- The raw data is preserved for debugging/investigation
- The audit trail captures everything, including safety failures

### 5. Developer Guide (docs/DEVELOPER_GUIDE.md)

A comprehensive onboarding document covering project architecture, setup, API reference, testing, configuration, and safety considerations. This helps new developers understand the codebase quickly and follow established patterns.

## Why This Design

**Hash chains over simple logging:** Regular log files can be edited silently. SHA-256 chains make tampering detectable — any modification to an entry invalidates the chain from that point forward. This is critical for clinical environments where audit integrity is non-negotiable.

**Regex-based pattern blocking:** Rather than trying to understand intent (which could miss edge cases), we pattern-match against known-dangerous constructs. This is conservative but reliable — a banned pattern will always be caught, at the cost of potential false positives (which are acceptable; safety is the priority).

**Warnings vs violations:** Not all non-advisory language is dangerous. "Turn on the monitor" is safe even though it doesn't start with "consider." Advisory language checks produce warnings for review but don't block output. Only banned patterns (actual executable/clinical language) produce violations.

## Test Coverage

101 new tests cover:
- Hash computation correctness and determinism
- Chain integrity verification (including tamper detection)
- Chain resumption from existing files
- All 18 banned patterns individually
- Required field validation
- Source and confidence range checking
- Advisory language warning generation
- Convenience function behavior
- Custom pattern injection
