"""Tests for src.state.rules — Rule Engine."""

from src.state.rules import RuleEngine, RuleEngineResult


class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine(surgery="PCNL")

    def test_loads_machines(self):
        assert len(self.engine.machines) > 0

    def test_machines_have_names(self):
        for mid, mdata in self.engine.machines.items():
            assert "name" in mdata

    def test_empty_transcript(self):
        result = self.engine.process("")
        assert isinstance(result, RuleEngineResult)
        assert len(result.toggles) == 0

    def test_activate_ventilator(self):
        result = self.engine.process("start the ventilator")
        patch = result.to_json_patch()
        assert isinstance(patch, dict)
        assert "machines" in patch
        assert "0" in patch["machines"]
        assert "1" in patch["machines"]

    def test_negation_handling(self):
        result = self.engine.process("don't start the suction")
        # With negation, the action should be OFF
        for toggle in result.toggles:
            if "suction" in toggle.machine_name.lower():
                assert toggle.action == "OFF"

    def test_json_patch_format(self):
        result = self.engine.process("turn on the monitor")
        patch = result.to_json_patch()
        required_keys = ["metadata", "machines", "details", "suggestions", "confidence", "source"]
        for key in required_keys:
            assert key in patch, f"Missing key: {key}"

    def test_processing_time_recorded(self):
        result = self.engine.process("start irrigation")
        assert result.processing_time_ms >= 0

    def test_switch_surgery(self):
        self.engine.switch_surgery("Lobectomy")
        assert self.engine.surgery == "Lobectomy"
        assert len(self.engine.machines) > 0

    def test_source_is_rule(self):
        result = self.engine.process("camera on")
        assert result.source == "rule"


class TestRuleEngineAllSurgeries:
    def test_pcnl_loads(self):
        engine = RuleEngine(surgery="PCNL")
        assert len(engine.machines) == 9

    def test_hepatectomy_loads(self):
        engine = RuleEngine(surgery="Partial Hepatectomy")
        assert len(engine.machines) == 9

    def test_lobectomy_loads(self):
        engine = RuleEngine(surgery="Lobectomy")
        assert len(engine.machines) == 9
