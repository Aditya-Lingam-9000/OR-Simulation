"""Tests for src.state.serializer — State Serializer."""

from src.state.serializer import StateSerializer


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
            "confidence": 1.5,  # Invalid
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
        assert self.serializer._clamp_confidence(0.7) == 0.7
        assert self.serializer._clamp_confidence("invalid") == 0.0
