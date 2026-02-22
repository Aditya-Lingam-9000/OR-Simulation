# Phase 5 Explanation — Rolling Context & State Schema

## What Was Built

Phase 5 establishes the **data contract and context infrastructure** that connects the deterministic rule engine (Phase 4) with the upcoming MedGemma LLM integration (Phase 6). Three components were enhanced:

### 1. JSON Schema — The Output Contract

Every state update in OR-Symphony — whether from the rule engine, MedGemma, or both — must conform to a single JSON schema. This schema is defined in `schemas/surgery_state.schema.json` using JSON Schema Draft-07.

**Why a formal schema?**
- Prevents schema drift as the system grows
- Catches errors early (bad machine IDs, invalid confidence, wrong source)
- Validates both rule engine and LLM outputs through the same contract
- Frontend can rely on consistent structure

**Key design choices:**
- `additionalProperties: false` at the top level and in machines/toggles prevents undocumented fields
- Machine IDs validated by regex pattern `^M[0-9]{2}$` (e.g., M01, M09)
- Metadata allows additional properties for extensibility
- Confidence clamped to `[0.0, 1.0]` at schema level

### 2. Rolling Buffer — LLM-Ready Context

The rolling buffer was enhanced from a simple deque-based transcript store to a full **context provider for MedGemma**:

**`get_context_for_llm()`** produces output like:
```
--- Recent OR Transcript --- | Surgery: PCNL | Phase: Phase3 | Window: 45s (12 segments)

[120.0s] surgeon: Starting tract dilation
[125.0s] nurse: Irrigation pump is running
[130.5s] surgeon: Insert the nephroscope

--- End Transcript ---
```

This block can be inserted directly into MedGemma prompt templates. Only final (non-partial) transcripts are included.

**Other additions:**
- `get_entries_since(timestamp)` — for incremental state updates (only process new transcript since last check)
- `get_entries_by_speaker()` — enables speaker-specific reasoning
- `get_summary()` — operational monitoring (entry count, duration, speakers, eviction stats)
- `to_dict()` / `from_dict()` / `save()` / `load()` — full persistence for session recovery

### 3. Serializer — The Normalization Pipeline

The serializer now handles the complete **data flow from both sources** through to validated output:

```
RuleEngineResult ──→ normalize_rule_output() ──┐
                                               ├─→ merge() ──→ validate_schema() ──→ save_state()
MedGemma Output ───→ normalize_llm_output() ──┘
```

**`build_current_state()`** orchestrates this entire pipeline:
1. Normalizes rule engine result (if available)
2. Merges with LLM output (if available)  
3. Enriches metadata with buffer statistics
4. Records total processing time
5. Returns complete, schema-conformant state dict

**LLM output handling:**
- Tolerates missing keys (LLMs may omit `machines` or `details`)
- Extracts `reasoning` and `next_phase` from either top-level or metadata
- Converts string confidence to float
- Forces `source: "medgemma"` regardless of LLM output

### 4. Schema Validation Integration

The serializer uses `jsonschema.validate()` against the loaded schema:
- `validate(state)` — uses schema if available, falls back to manual checks
- `validate_schema(state)` — strict schema validation only
- `get_validation_errors(state)` — returns all validation errors for debugging
- Schema loaded once and cached at module level

## How It Connects

```
Phase 4: Rule Engine                     Phase 6: MedGemma (next)
  ↓ RuleEngineResult                       ↓ raw LLM dict
  ↓                                        ↓
  └──→ normalize_rule_output() ←─ buffer ──→ normalize_llm_output()
                ↓                                    ↓
                └──────────→ merge() ←───────────────┘
                               ↓
                        validate_schema()
                               ↓
                         save_state()
                               ↓
                     WebSocket → Frontend
```

The buffer's `get_context_for_llm()` feeds MedGemma's prompt. The serializer's `normalize_llm_output()` handles whatever MedGemma returns. The schema guarantees the frontend always gets consistent JSON.

## Test Strategy

- **Buffer tests (54):** Creation, append, eviction boundary, LLM formatting, incremental queries, speaker filtering, serialization round-trips, file persistence, long simulations (200+ entries, 500-entry performance)
- **Schema tests (43):** Schema loading, valid states, missing keys (parametrized), machine ID patterns, toggle details, source/confidence enums, temporal enums, additionalProperties restrictions
- **Serializer tests (38):** Basic normalization, schema validation, rule normalization, LLM normalization, build_current_state, persistence, full integration (rule → serializer → schema)

Total: 134 new tests, 434 overall.
