# Phase 5 Plan: Rolling Context & State Schema

**Date:** 2026-02-22
**Prerequisites:** Phase 4 signed off (cc42119)

---

## Objectives

1. **JSON Schema** — Formal schema for surgery state (`schemas/surgery_state.schema.json`)
   - Validate all state updates with `jsonschema`
   - Covers machines, metadata, details, suggestions, confidence, source
   - Extensions: phase info, temporal, toggle details, buffer context

2. **Rolling Buffer Enhancement** — Upgrade `src/state/rolling_buffer.py`
   - Structured entries with speaker labels, source tracking
   - `get_context_for_llm()` — formatted prompt context for MedGemma
   - `get_entries_since(timestamp)` — incremental query
   - `get_summary()` — stats dict (count, duration, earliest, latest)
   - Serialization: `to_dict()` / `from_dict()` for persistence

3. **Serializer Enhancement** — Upgrade `src/state/serializer.py`
   - Schema validation via `jsonschema.validate()`
   - `normalize_rule_output()` — convert RuleEngineResult to schema-valid state
   - `normalize_llm_output()` — convert raw LLM JSON to schema-valid state
   - `build_current_state()` — full state snapshot with buffer context
   - `save_state()` / `load_state()` — persist to `tmp/current_state.json`

4. **Tests** — Expand test coverage
   - Schema validation tests (valid/invalid states)
   - Rolling buffer: long simulation (180s+), eviction, serialization
   - Serializer: normalization, merge, schema validation, persistence
   - Integration: rule engine → serializer → schema validation

5. **Sanity Checks**
   - Buffer demo with 200 entries → verify window maintained
   - Schema validation on live state file
   - Full pipeline: rule engine → serializer → validate → save

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `schemas/surgery_state.schema.json` | CREATE — JSON Schema definition |
| `src/state/rolling_buffer.py` | ENHANCE — LLM context, serialization |
| `src/state/serializer.py` | ENHANCE — schema validation, persistence |
| `tests/test_serializer.py` | EXPAND — 50+ tests |
| `tests/test_buffer.py` | CREATE — 40+ buffer tests |
| `tests/test_schema.py` | CREATE — 30+ schema tests |

---

## Deliverables

- [ ] `schemas/surgery_state.schema.json`
- [ ] Enhanced `rolling_buffer.py` with LLM context formatting
- [ ] Enhanced `serializer.py` with schema validation + persistence
- [ ] 120+ new tests (across 3 test files)
- [ ] Sanity checks passing
- [ ] Git commit on main
- [ ] `reports/phase5_report.md` + `reports/phase5_explanation.md`
