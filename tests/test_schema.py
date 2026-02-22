"""Tests for schemas/surgery_state.schema.json — JSON Schema validation.

Covers:
  - Schema loads correctly
  - Valid states pass validation
  - Missing required keys fail
  - Machine ID pattern (M##) validation
  - Toggle detail structure validation
  - Source enum validation
  - Confidence range validation
  - Metadata field types
  - additionalProperties restrictions
  - Edge cases
"""

import json
from pathlib import Path

import jsonschema
import pytest

# Load schema once
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "surgery_state.schema.json"


@pytest.fixture
def schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _valid_state(**overrides) -> dict:
    """Create a minimal valid state dict with optional overrides."""
    state = {
        "metadata": {
            "timestamp": "2025-01-01T00:00:00Z",
            "surgery": "PCNL",
            "phase": "Phase2",
        },
        "machines": {"0": ["M03"], "1": ["M01", "M09"]},
        "details": {
            "toggles": [
                {
                    "machine_id": "M01",
                    "name": "C-Arm Fluoroscopy",
                    "action": "ON",
                    "trigger": "fluoroscopy",
                    "confidence": 0.9,
                    "match_type": "trigger",
                }
            ],
            "negations": [],
            "debounced": [],
        },
        "suggestions": ["Prepare lithotripter"],
        "confidence": 0.85,
        "source": "rule",
    }
    state.update(overrides)
    return state


# ── Schema loading ───────────────────────────────────────────────────────


class TestSchemaLoading:
    def test_schema_file_exists(self):
        assert SCHEMA_PATH.exists()

    def test_schema_is_valid_json(self, schema):
        assert isinstance(schema, dict)
        assert "$schema" in schema

    def test_schema_has_required_fields(self, schema):
        required = schema.get("required", [])
        for key in ["metadata", "machines", "details", "suggestions", "confidence", "source"]:
            assert key in required

    def test_schema_is_valid_draft7(self, schema):
        # Should not raise
        jsonschema.Draft7Validator.check_schema(schema)


# ── Valid states ─────────────────────────────────────────────────────────


class TestValidStates:
    def test_minimal_valid(self, schema):
        state = _valid_state()
        jsonschema.validate(instance=state, schema=schema)

    def test_empty_machines(self, schema):
        state = _valid_state(machines={"0": [], "1": []})
        jsonschema.validate(instance=state, schema=schema)

    def test_empty_suggestions(self, schema):
        state = _valid_state(suggestions=[])
        jsonschema.validate(instance=state, schema=schema)

    def test_source_rule(self, schema):
        state = _valid_state(source="rule")
        jsonschema.validate(instance=state, schema=schema)

    def test_source_medgemma(self, schema):
        state = _valid_state(source="medgemma")
        jsonschema.validate(instance=state, schema=schema)

    def test_source_combined(self, schema):
        state = _valid_state(source="rule+medgemma")
        jsonschema.validate(instance=state, schema=schema)

    def test_confidence_zero(self, schema):
        state = _valid_state(confidence=0.0)
        jsonschema.validate(instance=state, schema=schema)

    def test_confidence_one(self, schema):
        state = _valid_state(confidence=1.0)
        jsonschema.validate(instance=state, schema=schema)

    def test_all_surgeries(self, schema):
        for surgery in ["PCNL", "Partial Hepatectomy", "Lobectomy", ""]:
            state = _valid_state()
            state["metadata"]["surgery"] = surgery
            jsonschema.validate(instance=state, schema=schema)

    def test_details_with_all_toggle_fields(self, schema):
        state = _valid_state()
        state["details"] = {
            "toggles": [
                {
                    "machine_id": "M05",
                    "name": "Irrigation Pump",
                    "action": "STANDBY",
                    "trigger": "irrigation",
                    "confidence": 0.75,
                    "match_type": "alias",
                }
            ],
            "negations": ["don't turn on"],
            "debounced": ["M03"],
        }
        jsonschema.validate(instance=state, schema=schema)

    def test_metadata_with_optional_fields(self, schema):
        state = _valid_state()
        state["metadata"].update({
            "processing_time_ms": 42.5,
            "matched_keywords": ["fluoroscopy", "c-arm"],
            "temporal": "immediately",
            "reasoning": "Phase indicates imaging needed",
            "next_phase": "Phase3",
            "transcript_context": "turn on the c-arm",
            "buffer_entries": 15,
            "buffer_duration_s": 45.2,
            "session_elapsed_s": 120.5,
        })
        jsonschema.validate(instance=state, schema=schema)


# ── Missing required keys ────────────────────────────────────────────────


class TestMissingKeys:
    @pytest.mark.parametrize("missing_key", [
        "metadata", "machines", "details", "suggestions", "confidence", "source"
    ])
    def test_missing_required_key(self, schema, missing_key):
        state = _valid_state()
        del state[missing_key]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)


# ── Machine ID pattern ───────────────────────────────────────────────────


class TestMachineIdPattern:
    def test_valid_machine_ids(self, schema):
        state = _valid_state(machines={"0": ["M01", "M09"], "1": ["M03", "M10"]})
        jsonschema.validate(instance=state, schema=schema)

    def test_invalid_machine_id_lowercase(self, schema):
        state = _valid_state(machines={"0": ["m01"], "1": []})
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_invalid_machine_id_no_prefix(self, schema):
        state = _valid_state(machines={"0": ["01"], "1": []})
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_invalid_machine_id_three_digits(self, schema):
        state = _valid_state(machines={"0": ["M001"], "1": []})
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_missing_machine_key_0(self, schema):
        state = _valid_state(machines={"1": ["M01"]})
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_missing_machine_key_1(self, schema):
        state = _valid_state(machines={"0": ["M01"]})
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_extra_machine_key(self, schema):
        state = _valid_state(machines={"0": [], "1": [], "2": []})
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)


# ── Toggle details ───────────────────────────────────────────────────────


class TestToggleDetails:
    def test_toggle_missing_required_field(self, schema):
        state = _valid_state()
        state["details"]["toggles"] = [
            {"machine_id": "M01", "name": "Test"}  # missing "action"
        ]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_invalid_action_enum(self, schema):
        state = _valid_state()
        state["details"]["toggles"] = [
            {"machine_id": "M01", "name": "Test", "action": "RESET"}
        ]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_invalid_match_type(self, schema):
        state = _valid_state()
        state["details"]["toggles"] = [
            {"machine_id": "M01", "name": "Test", "action": "ON", "match_type": "fuzzy"}
        ]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_toggle_confidence_out_of_range(self, schema):
        state = _valid_state()
        state["details"]["toggles"] = [
            {"machine_id": "M01", "name": "Test", "action": "ON", "confidence": 1.5}
        ]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_valid_action_values(self, schema):
        for action in ["ON", "OFF", "STANDBY"]:
            state = _valid_state()
            state["details"]["toggles"] = [
                {"machine_id": "M01", "name": "Test", "action": action}
            ]
            jsonschema.validate(instance=state, schema=schema)

    def test_valid_match_types(self, schema):
        for mt in ["trigger", "alias", "name", "llm"]:
            state = _valid_state()
            state["details"]["toggles"] = [
                {"machine_id": "M01", "name": "Test", "action": "ON", "match_type": mt}
            ]
            jsonschema.validate(instance=state, schema=schema)


# ── Source & confidence ──────────────────────────────────────────────────


class TestSourceAndConfidence:
    def test_invalid_source(self, schema):
        state = _valid_state(source="gpt4")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_confidence_above_one(self, schema):
        state = _valid_state(confidence=1.1)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_confidence_below_zero(self, schema):
        state = _valid_state(confidence=-0.1)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_confidence_string(self, schema):
        state = _valid_state(confidence="high")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)


# ── Temporal enum ────────────────────────────────────────────────────────


class TestTemporalEnum:
    def test_valid_temporals(self, schema):
        for t in ["none", "immediately", "after", "when", "in_time", "before", "during"]:
            state = _valid_state()
            state["metadata"]["temporal"] = t
            jsonschema.validate(instance=state, schema=schema)

    def test_invalid_temporal(self, schema):
        state = _valid_state()
        state["metadata"]["temporal"] = "later"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)


# ── additionalProperties ─────────────────────────────────────────────────


class TestAdditionalProperties:
    def test_extra_top_level_key(self, schema):
        state = _valid_state()
        state["extra_field"] = "not allowed"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_extra_machines_key(self, schema):
        state = _valid_state()
        state["machines"]["standby"] = ["M05"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)

    def test_extra_toggle_field(self, schema):
        state = _valid_state()
        state["details"]["toggles"] = [
            {"machine_id": "M01", "name": "Test", "action": "ON", "extra": True}
        ]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=state, schema=schema)
