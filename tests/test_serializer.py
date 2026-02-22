"""Tests for src.state.serializer — Enhanced State Serializer.

Covers:
  - Basic normalization (existing)
  - Schema-based validation (jsonschema)
  - normalize_rule_output from RuleEngineResult
  - normalize_llm_output from raw LLM response
  - build_current_state with buffer context
  - merge rule + LLM outputs
  - save_state / load_state persistence
  - get_validation_errors
  - Edge cases and error handling
  - Integration: rule engine → serializer → schema validation
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.state.rolling_buffer import RollingBuffer
from src.state.serializer import StateSerializer


# ── Mock RuleEngineResult ────────────────────────────────────────────────

@dataclass
class MockTemporalQualifier:
    qualifier: str = "none"


@dataclass
class MockMachineToggle:
    machine_id: str = "M01"
    machine_name: str = "C-Arm Fluoroscopy"
    action: str = "ON"
    trigger_text: str = "fluoroscopy"
    confidence: float = 0.9
    match_type: str = "trigger"


@dataclass
class MockRuleEngineResult:
    toggles: List[MockMachineToggle] = field(default_factory=list)
    processing_time_ms: float = 12.5
    source: str = "rule"
    matched_keywords: List[str] = field(default_factory=list)
    negations_detected: List[str] = field(default_factory=list)
    debounced: List[str] = field(default_factory=list)
    temporal: Optional[MockTemporalQualifier] = None
    current_phase: str = ""

    def to_json_patch(self) -> Dict[str, Any]:
        machines_on = []
        machines_off = []
        for t in self.toggles:
            if t.action == "ON":
                machines_on.append(t.machine_id)
            elif t.action == "STANDBY":
                machines_off.append(t.machine_id)
            else:
                machines_off.append(t.machine_id)
        return {
            "metadata": {
                "processing_time_ms": self.processing_time_ms,
                "matched_keywords": self.matched_keywords,
                "phase": self.current_phase,
                "temporal": self.temporal.qualifier if self.temporal else "none",
            },
            "machines": {"0": machines_off, "1": machines_on},
            "details": {
                "toggles": [
                    {
                        "machine_id": t.machine_id,
                        "name": t.machine_name,
                        "action": t.action,
                        "trigger": t.trigger_text,
                        "confidence": t.confidence,
                        "match_type": t.match_type,
                    }
                    for t in self.toggles
                ],
                "negations": self.negations_detected,
                "debounced": self.debounced,
            },
            "suggestions": [],
            "confidence": min((t.confidence for t in self.toggles), default=0.0),
            "source": self.source,
        }


# ── Basic normalization (existing tests) ─────────────────────────────────


class TestStateSerializer:
    def setup_method(self):
        self.serializer = StateSerializer()

    def test_normalize_basic(self):
        raw = {
            "machines": {"0": ["M03"], "1": ["M01"]},
            "confidence": 0.85,
            "source": "rule",
        }
        result = self.serializer.normalize(raw, surgery="PCNL", phase="Phase2")
        assert result["metadata"]["surgery"] == "PCNL"
        assert result["machines"]["0"] == ["M03"]
        assert result["machines"]["1"] == ["M01"]
        assert result["confidence"] == 0.85
        assert result["source"] == "rule"

    def test_normalize_adds_timestamp(self):
        raw = {"machines": {"0": [], "1": []}, "confidence": 0.0, "source": "rule"}
        result = self.serializer.normalize(raw)
        assert "timestamp" in result["metadata"]

    def test_validate_valid_state(self):
        state = {
            "metadata": {"surgery": "PCNL"},
            "machines": {"0": [], "1": ["M01"]},
            "details": {},
            "suggestions": [],
            "confidence": 0.9,
            "source": "rule",
        }
        assert self.serializer.validate(state) is True

    def test_validate_missing_key(self):
        state = {"metadata": {}, "machines": {"0": [], "1": []}}
        assert self.serializer.validate(state) is False

    def test_validate_invalid_confidence(self):
        state = {
            "metadata": {},
            "machines": {"0": [], "1": []},
            "details": {},
            "suggestions": [],
            "confidence": 1.5,
            "source": "rule",
        }
        assert self.serializer.validate(state) is False

    def test_validate_invalid_source(self):
        state = {
            "metadata": {},
            "machines": {"0": [], "1": []},
            "details": {},
            "suggestions": [],
            "confidence": 0.5,
            "source": "invalid",
        }
        assert self.serializer.validate(state) is False

    def test_merge_rule_only(self):
        rule = {
            "metadata": {"surgery": "PCNL"},
            "machines": {"0": [], "1": ["M01"]},
            "details": {},
            "suggestions": [],
            "confidence": 0.8,
            "source": "rule",
        }
        merged = self.serializer.merge(rule, None)
        assert merged == rule

    def test_merge_rule_and_llm(self):
        rule = {
            "metadata": {"surgery": "PCNL"},
            "machines": {"0": [], "1": ["M01"]},
            "details": {},
            "suggestions": [],
            "confidence": 0.8,
            "source": "rule",
        }
        llm = {
            "metadata": {"phase": "Phase3", "reasoning": "normal"},
            "suggestions": ["Prepare lithotripter"],
            "confidence": 0.95,
            "source": "medgemma",
        }
        merged = self.serializer.merge(rule, llm)
        assert merged["source"] == "rule+medgemma"
        assert merged["confidence"] == 0.95
        assert "Prepare lithotripter" in merged["suggestions"]
        assert merged["metadata"]["phase"] == "Phase3"

    def test_clamp_confidence(self):
        assert self.serializer._clamp_confidence(1.5) == 1.0
        assert self.serializer._clamp_confidence(-0.5) == 0.0


# ── Schema-based validation ──────────────────────────────────────────────


class TestSchemaValidation:
    def setup_method(self):
        self.serializer = StateSerializer()

    def test_schema_loaded(self):
        assert self.serializer._schema is not None

    def test_validate_schema_valid(self):
        state = {
            "metadata": {"surgery": "PCNL", "phase": "Phase1"},
            "machines": {"0": [], "1": ["M01"]},
            "details": {"toggles": [], "negations": [], "debounced": []},
            "suggestions": [],
            "confidence": 0.85,
            "source": "rule",
        }
        assert self.serializer.validate_schema(state) is True

    def test_validate_schema_invalid(self):
        state = {"metadata": {}, "machines": {"0": [], "1": []}}
        assert self.serializer.validate_schema(state) is False

    def test_get_validation_errors_valid(self):
        state = {
            "metadata": {"surgery": "PCNL"},
            "machines": {"0": [], "1": []},
            "details": {},
            "suggestions": [],
            "confidence": 0.5,
            "source": "rule",
        }
        errors = self.serializer.get_validation_errors(state)
        assert len(errors) == 0

    def test_get_validation_errors_multiple(self):
        state = {"metadata": {}}  # Missing many keys
        errors = self.serializer.get_validation_errors(state)
        assert len(errors) > 0

    def test_validate_detects_bad_machine_id(self):
        state = {
            "metadata": {"surgery": "PCNL"},
            "machines": {"0": ["invalid"], "1": []},
            "details": {},
            "suggestions": [],
            "confidence": 0.5,
            "source": "rule",
        }
        assert self.serializer.validate(state) is False

    def test_validate_detects_bad_confidence(self):
        state = {
            "metadata": {},
            "machines": {"0": [], "1": []},
            "details": {},
            "suggestions": [],
            "confidence": 2.0,
            "source": "rule",
        }
        assert self.serializer.validate(state) is False


# ── normalize_rule_output ────────────────────────────────────────────────


class TestNormalizeRuleOutput:
    def setup_method(self):
        self.serializer = StateSerializer()

    def test_basic_rule_output(self):
        toggle = MockMachineToggle(
            machine_id="M01", machine_name="C-Arm",
            action="ON", trigger_text="fluoroscopy",
            confidence=0.9, match_type="trigger"
        )
        result = MockRuleEngineResult(
            toggles=[toggle],
            matched_keywords=["fluoroscopy"],
            current_phase="Phase2",
        )
        state = self.serializer.normalize_rule_output(result, surgery="PCNL", phase="Phase2")
        assert state["machines"]["1"] == ["M01"]
        assert state["source"] == "rule"
        assert state["confidence"] == 0.9
        assert state["metadata"]["surgery"] == "PCNL"
        assert state["metadata"]["phase"] == "Phase2"

    def test_rule_output_with_buffer(self):
        toggle = MockMachineToggle()
        result = MockRuleEngineResult(toggles=[toggle])
        buf = RollingBuffer(max_duration_s=30)
        buf.append("turn on fluoroscopy", timestamp=5.0, speaker="surgeon")
        buf.append("position the c-arm", timestamp=8.0, speaker="nurse")

        state = self.serializer.normalize_rule_output(
            result, surgery="PCNL", phase="Phase2", buffer=buf
        )
        assert state["metadata"]["buffer_entries"] == 2
        assert state["metadata"]["buffer_duration_s"] == 3.0
        assert "transcript_context" in state["metadata"]

    def test_rule_output_no_toggles(self):
        result = MockRuleEngineResult(toggles=[], current_phase="Phase1")
        state = self.serializer.normalize_rule_output(result, surgery="PCNL")
        assert state["machines"]["0"] == []
        assert state["machines"]["1"] == []
        assert state["confidence"] == 0.0

    def test_rule_output_multiple_toggles(self):
        toggles = [
            MockMachineToggle(machine_id="M01", action="ON"),
            MockMachineToggle(machine_id="M03", action="OFF"),
            MockMachineToggle(machine_id="M05", action="STANDBY"),
        ]
        result = MockRuleEngineResult(toggles=toggles)
        state = self.serializer.normalize_rule_output(result, surgery="PCNL")
        assert "M01" in state["machines"]["1"]
        assert "M03" in state["machines"]["0"]
        assert "M05" in state["machines"]["0"]


# ── normalize_llm_output ────────────────────────────────────────────────


class TestNormalizeLLMOutput:
    def setup_method(self):
        self.serializer = StateSerializer()

    def test_basic_llm_output(self):
        raw = {
            "metadata": {"reasoning": "Patient ready for imaging"},
            "machines": {"0": [], "1": ["M01"]},
            "suggestions": ["Prepare lithotripter"],
            "confidence": 0.7,
        }
        state = self.serializer.normalize_llm_output(raw, surgery="PCNL", phase="Phase3")
        assert state["source"] == "medgemma"
        assert state["metadata"]["surgery"] == "PCNL"
        assert state["metadata"]["reasoning"] == "Patient ready for imaging"
        assert state["confidence"] == 0.7
        assert "Prepare lithotripter" in state["suggestions"]

    def test_llm_output_with_reasoning_at_top_level(self):
        raw = {
            "reasoning": "Top-level reasoning",
            "next_phase": "Phase4",
            "machines": {"0": [], "1": []},
            "confidence": 0.6,
        }
        state = self.serializer.normalize_llm_output(raw)
        assert state["metadata"]["reasoning"] == "Top-level reasoning"
        assert state["metadata"]["next_phase"] == "Phase4"

    def test_llm_output_missing_machines(self):
        raw = {"confidence": 0.5, "suggestions": ["check vitals"]}
        state = self.serializer.normalize_llm_output(raw)
        assert state["machines"] == {"0": [], "1": []}

    def test_llm_output_string_confidence(self):
        raw = {"confidence": "0.8", "machines": {"0": [], "1": []}}
        state = self.serializer.normalize_llm_output(raw)
        assert state["confidence"] == 0.8

    def test_llm_output_invalid_machines_type(self):
        raw = {"machines": "not a dict", "confidence": 0.5}
        state = self.serializer.normalize_llm_output(raw)
        assert state["machines"] == {"0": [], "1": []}


# ── build_current_state ──────────────────────────────────────────────────


class TestBuildCurrentState:
    def setup_method(self):
        self.serializer = StateSerializer()

    def test_rule_only(self):
        toggle = MockMachineToggle(machine_id="M01", action="ON")
        result = MockRuleEngineResult(toggles=[toggle], current_phase="Phase2")
        state = self.serializer.build_current_state(
            rule_result=result, surgery="PCNL", phase="Phase2"
        )
        assert state["machines"]["1"] == ["M01"]
        assert state["source"] == "rule"
        assert "processing_time_ms" in state["metadata"]

    def test_rule_with_llm(self):
        toggle = MockMachineToggle(machine_id="M01", action="ON")
        result = MockRuleEngineResult(toggles=[toggle])
        llm = {
            "metadata": {"reasoning": "imaging phase"},
            "suggestions": ["prep lithotripter"],
            "confidence": 0.95,
            "source": "medgemma",
        }
        state = self.serializer.build_current_state(
            rule_result=result, llm_result=llm, surgery="PCNL"
        )
        assert state["source"] == "rule+medgemma"
        assert state["confidence"] == 0.95
        assert "prep lithotripter" in state["suggestions"]

    def test_with_buffer_context(self):
        toggle = MockMachineToggle()
        result = MockRuleEngineResult(toggles=[toggle])
        buf = RollingBuffer(max_duration_s=30)
        buf.append("turn on fluoroscopy", timestamp=1.0)
        buf.append("position c-arm", timestamp=3.0)

        state = self.serializer.build_current_state(
            rule_result=result, buffer=buf, surgery="PCNL"
        )
        assert state["metadata"]["buffer_entries"] == 2
        assert "buffer_duration_s" in state["metadata"]
        assert "session_elapsed_s" in state["metadata"]

    def test_no_rule_no_llm(self):
        state = self.serializer.build_current_state(surgery="PCNL", phase="Phase1")
        assert state["machines"] == {"0": [], "1": []}
        assert state["confidence"] == 0.0
        assert state["source"] == "rule"

    def test_empty_state_structure(self):
        state = self.serializer._empty_state(surgery="PCNL", phase="Phase1")
        assert state["metadata"]["surgery"] == "PCNL"
        assert state["metadata"]["phase"] == "Phase1"
        assert state["metadata"]["temporal"] == "none"
        assert state["machines"] == {"0": [], "1": []}
        assert state["details"]["toggles"] == []
        assert state["suggestions"] == []
        assert state["confidence"] == 0.0
        assert state["source"] == "rule"


# ── Persistence: save_state / load_state ─────────────────────────────────


class TestStatePersistence:
    def setup_method(self):
        self.serializer = StateSerializer()

    def test_save_and_load(self, tmp_path):
        state = {
            "metadata": {"surgery": "PCNL", "phase": "Phase2"},
            "machines": {"0": ["M03"], "1": ["M01"]},
            "details": {"toggles": []},
            "suggestions": ["check vitals"],
            "confidence": 0.85,
            "source": "rule",
        }
        path = tmp_path / "state.json"
        self.serializer.save_state(state, path=path)
        loaded = self.serializer.load_state(path=path)
        assert loaded is not None
        assert loaded["metadata"]["surgery"] == "PCNL"
        assert loaded["machines"]["1"] == ["M01"]
        assert loaded["confidence"] == 0.85

    def test_save_creates_parent_dirs(self, tmp_path):
        state = {"metadata": {}, "machines": {"0": [], "1": []},
                 "details": {}, "suggestions": [], "confidence": 0.0, "source": "rule"}
        path = tmp_path / "nested" / "deep" / "state.json"
        self.serializer.save_state(state, path=path)
        assert path.exists()

    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        result = self.serializer.load_state(path=path)
        assert result is None

    def test_roundtrip_preserves_all_fields(self, tmp_path):
        state = {
            "metadata": {
                "surgery": "Lobectomy",
                "phase": "Phase3",
                "timestamp": "2025-01-01T00:00:00Z",
                "processing_time_ms": 15.2,
                "matched_keywords": ["stapler", "vats"],
            },
            "machines": {"0": ["M02", "M05"], "1": ["M01", "M07"]},
            "details": {
                "toggles": [
                    {"machine_id": "M01", "name": "VATS Camera", "action": "ON",
                     "trigger": "camera", "confidence": 0.9, "match_type": "trigger"}
                ],
                "negations": [],
                "debounced": ["M03"],
            },
            "suggestions": ["Prepare chest drain"],
            "confidence": 0.88,
            "source": "rule",
        }
        path = tmp_path / "full_state.json"
        self.serializer.save_state(state, path=path)
        loaded = self.serializer.load_state(path=path)
        assert loaded == state

    def test_saved_file_is_valid_json(self, tmp_path):
        state = {"metadata": {}, "machines": {"0": [], "1": []},
                 "details": {}, "suggestions": [], "confidence": 0.5, "source": "rule"}
        path = tmp_path / "test.json"
        self.serializer.save_state(state, path=path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)


# ── Integration: rule engine → serializer → schema ───────────────────────


class TestIntegration:
    def setup_method(self):
        self.serializer = StateSerializer()

    def test_rule_result_validates_against_schema(self):
        """Full pipeline: MockRuleEngineResult → normalize → validate_schema."""
        toggle = MockMachineToggle(
            machine_id="M01", machine_name="C-Arm Fluoroscopy",
            action="ON", trigger_text="fluoroscopy",
            confidence=0.9, match_type="trigger"
        )
        result = MockRuleEngineResult(
            toggles=[toggle],
            matched_keywords=["fluoroscopy"],
            current_phase="Phase2",
            temporal=MockTemporalQualifier(qualifier="immediately"),
        )
        state = self.serializer.normalize_rule_output(
            result, surgery="PCNL", phase="Phase2"
        )
        assert self.serializer.validate_schema(state) is True

    def test_build_state_validates_against_schema(self):
        """Full pipeline: build_current_state → validate_schema."""
        toggle = MockMachineToggle(
            machine_id="M03", machine_name="Irrigation Pump",
            action="ON", trigger_text="irrigation",
            confidence=0.85, match_type="alias"
        )
        result = MockRuleEngineResult(
            toggles=[toggle],
            matched_keywords=["irrigation"],
            current_phase="Phase3",
        )
        buf = RollingBuffer(max_duration_s=30)
        buf.append("start irrigation", timestamp=10.0, speaker="surgeon")

        state = self.serializer.build_current_state(
            rule_result=result, buffer=buf,
            surgery="PCNL", phase="Phase3"
        )
        assert self.serializer.validate_schema(state) is True

    def test_merged_state_validates(self):
        """Rule + LLM merge → validate."""
        toggle = MockMachineToggle()
        result = MockRuleEngineResult(toggles=[toggle], current_phase="Phase2")
        llm = {
            "metadata": {"reasoning": "standard imaging", "next_phase": "Phase3"},
            "suggestions": ["Prepare lithotripter"],
            "confidence": 0.92,
            "source": "medgemma",
        }
        state = self.serializer.build_current_state(
            rule_result=result, llm_result=llm,
            surgery="PCNL", phase="Phase2"
        )
        assert state["source"] == "rule+medgemma"
        assert self.serializer.validate_schema(state) is True

    def test_empty_state_validates(self):
        """Empty state should still pass validation."""
        state = self.serializer._empty_state(surgery="PCNL", phase="Phase1")
        assert self.serializer.validate_schema(state) is True

    def test_save_load_validates(self, tmp_path):
        """Save → Load → Validate round-trip."""
        toggle = MockMachineToggle()
        result = MockRuleEngineResult(toggles=[toggle])
        state = self.serializer.normalize_rule_output(result, surgery="PCNL")

        path = tmp_path / "pipeline.json"
        self.serializer.save_state(state, path=path)
        loaded = self.serializer.load_state(path=path)
        assert self.serializer.validate_schema(loaded) is True

        assert self.serializer._clamp_confidence(0.7) == 0.7
        assert self.serializer._clamp_confidence("invalid") == 0.0
