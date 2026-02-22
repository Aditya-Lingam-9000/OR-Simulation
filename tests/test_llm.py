"""Tests for src.llm — MedGemma GGUF Integration (Phase 6).

Covers:
  - GGUFRunner: stub mode, generate, chat, _parse_json_output (3 strategies), stats, load/unload
  - PromptBuilder: all 3 surgeries, system/user prompts, messages, completion prompt, set_surgery
  - LLMBatcher: single request, batch collection, timeout flush, stats
  - LLMManager: start/stop, submit (stub), fallback on failure, surgery switching, stats
  - Integration: PromptBuilder → GGUFRunner(stub) → StateSerializer.normalize_llm_output
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.llm.batcher import BatchRequest, BatchResult, LLMBatcher
from src.llm.gguf_runner import GGUFRunner
from src.llm.manager import LLMManager, LLMRequest, LLMResponse
from src.llm.prompts import PromptBuilder
from src.state.serializer import StateSerializer


# ═══════════════════════════════════════════════════════════════════════
#  GGUFRunner Tests
# ═══════════════════════════════════════════════════════════════════════


class TestGGUFRunnerStubMode:
    """GGUFRunner in stub_mode — no model file required."""

    def setup_method(self) -> None:
        self.runner = GGUFRunner(stub_mode=True)

    def test_init_stub(self) -> None:
        assert self.runner.stub_mode is True
        assert self.runner.is_loaded() is False

    def test_load_model_stub(self) -> None:
        result = self.runner.load_model()
        assert result is True
        # Stub mode does not set _loaded to True
        assert self.runner.is_loaded() is False

    def test_generate_stub(self) -> None:
        self.runner.load_model()
        out = self.runner.generate("any prompt")
        assert isinstance(out, dict)
        assert "machines" in out
        assert "source" in out
        assert out["source"] == "medgemma"

    def test_generate_stub_increments_stats(self) -> None:
        self.runner.load_model()
        self.runner.generate("prompt 1")
        self.runner.generate("prompt 2")
        assert self.runner.stats["total_inferences"] == 2
        assert self.runner.stats["total_failures"] == 0

    def test_chat_stub(self) -> None:
        self.runner.load_model()
        messages = [
            {"role": "system", "content": "You are a test"},
            {"role": "user", "content": "Hello"},
        ]
        out = self.runner.chat(messages)
        assert isinstance(out, dict)
        assert "machines" in out

    def test_generate_raw_stub(self) -> None:
        out = self.runner.generate_raw("prompt")
        assert isinstance(out, str)
        parsed = json.loads(out)
        assert "machines" in parsed

    def test_unload_stub(self) -> None:
        self.runner.load_model()
        self.runner.unload()
        assert self.runner.is_loaded() is False

    def test_generate_not_loaded(self) -> None:
        # Without load_model, returns stub if stub_mode, otherwise failure
        runner = GGUFRunner(stub_mode=True)
        out = runner.generate("test")
        assert isinstance(out, dict)

    def test_stats_initial(self) -> None:
        stats = self.runner.stats
        assert stats["total_inferences"] == 0
        assert stats["total_failures"] == 0
        assert stats["stub_mode"] is True
        assert stats["model_loaded"] is False

    def test_stats_after_usage(self) -> None:
        self.runner.load_model()
        self.runner.generate("p1")
        self.runner.generate("p2")
        self.runner.generate("p3")
        stats = self.runner.stats
        assert stats["total_inferences"] == 3

    def test_multiple_generate_returns_copy(self) -> None:
        """Each generate() should return a separate dict."""
        self.runner.load_model()
        out1 = self.runner.generate("p1")
        out2 = self.runner.generate("p2")
        assert out1 is not out2
        out1["test_key"] = "modified"
        assert "test_key" not in out2


class TestGGUFRunnerModelNotFound:
    """GGUFRunner when model file doesn't exist."""

    def test_load_model_missing_file(self) -> None:
        runner = GGUFRunner(model_path="nonexistent.gguf", stub_mode=False)
        result = runner.load_model()
        assert result is False
        assert runner.is_loaded() is False

    def test_generate_without_model_returns_stub(self) -> None:
        runner = GGUFRunner(model_path="nonexistent.gguf", stub_mode=False)
        out = runner.generate("test")
        assert isinstance(out, dict)
        assert runner.stats["total_failures"] == 1


class TestGGUFRunnerJsonParsing:
    """Test _parse_json_output with various strategies."""

    def test_strategy1_direct_json(self) -> None:
        """Strategy 1: Direct JSON parse."""
        text = '{"machines": {"0": [], "1": ["M01"]}, "confidence": 0.85}'
        result = GGUFRunner._parse_json_output(text)
        assert result["confidence"] == 0.85
        assert result["machines"]["1"] == ["M01"]

    def test_strategy2_brace_extraction(self) -> None:
        """Strategy 2: Find outermost { ... }."""
        text = 'Here is the analysis:\n{"machines": {"0": ["M02"], "1": []}, "confidence": 0.5}\nDone.'
        result = GGUFRunner._parse_json_output(text)
        assert result["machines"]["0"] == ["M02"]

    def test_strategy3_json_block(self) -> None:
        """Strategy 3: Extract from ```json block."""
        text = 'Here is the output:\n```json\n{"machines": {"0": [], "1": ["M03"]}, "confidence": 0.7}\n```'
        result = GGUFRunner._parse_json_output(text)
        assert result["machines"]["1"] == ["M03"]

    def test_strategy3_bare_backticks(self) -> None:
        """Strategy 3: Extract from ``` block (no json specifier)."""
        text = '```\n{"phase": "Phase2", "confidence": 0.6}\n```'
        result = GGUFRunner._parse_json_output(text)
        assert result["phase"] == "Phase2"

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty model output"):
            GGUFRunner._parse_json_output("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty model output"):
            GGUFRunner._parse_json_output("   \n  ")

    def test_no_valid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No valid JSON"):
            GGUFRunner._parse_json_output("This is just plain text with no JSON.")

    def test_nested_json(self) -> None:
        """Deeply nested JSON should parse."""
        obj = {
            "metadata": {"phase": "Phase3", "reasoning": "test"},
            "machines": {"0": [], "1": ["M01", "M05"]},
            "details": {"toggles": [{"machine_id": "M01", "action": "ON"}]},
            "confidence": 0.9,
        }
        text = json.dumps(obj)
        result = GGUFRunner._parse_json_output(text)
        assert result["metadata"]["phase"] == "Phase3"
        assert len(result["machines"]["1"]) == 2

    def test_json_with_leading_text(self) -> None:
        text = 'Response:\n\n{"key": "value"}'
        result = GGUFRunner._parse_json_output(text)
        assert result["key"] == "value"

    def test_json_array_not_accepted(self) -> None:
        """Arrays should not be accepted (we need dicts)."""
        with pytest.raises(ValueError):
            GGUFRunner._parse_json_output('[1, 2, 3]')

    def test_malformed_json_in_braces_raises(self) -> None:
        """Invalid JSON within braces should raise."""
        with pytest.raises(ValueError):
            GGUFRunner._parse_json_output('{key: "no quotes on key"}')

    def test_complex_multiline(self) -> None:
        """Multi-line JSON with whitespace."""
        text = """The model determined:
```json
{
    "metadata": {
        "phase": "Phase4",
        "reasoning": "The surgeon mentioned fluoroscopy"
    },
    "machines": {
        "0": ["M02"],
        "1": ["M01", "M06"]
    },
    "confidence": 0.88
}
```
"""
        result = GGUFRunner._parse_json_output(text)
        assert result["metadata"]["phase"] == "Phase4"
        assert result["confidence"] == 0.88


# ═══════════════════════════════════════════════════════════════════════
#  PromptBuilder Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPromptBuilderPCNL:
    """PromptBuilder with PCNL surgery config."""

    def setup_method(self) -> None:
        self.builder = PromptBuilder(surgery="PCNL")

    def test_init(self) -> None:
        assert self.builder.surgery == "PCNL"
        assert self.builder.machine_count > 0
        assert self.builder.phase_count > 0

    def test_machine_ids(self) -> None:
        ids = self.builder.machine_ids
        assert all(mid.startswith("M") for mid in ids)
        assert len(ids) == self.builder.machine_count

    def test_system_prompt_contains_surgery(self) -> None:
        prompt = self.builder.build_system_prompt()
        assert "PCNL" in prompt
        assert "OR-Symphony" in prompt

    def test_system_prompt_contains_machines(self) -> None:
        prompt = self.builder.build_system_prompt()
        for mid in self.builder.machine_ids:
            assert mid in prompt

    def test_system_prompt_contains_phases(self) -> None:
        prompt = self.builder.build_system_prompt()
        assert "Phase" in prompt

    def test_system_prompt_contains_json_schema(self) -> None:
        prompt = self.builder.build_system_prompt()
        assert '"machines"' in prompt
        assert '"metadata"' in prompt
        assert '"confidence"' in prompt

    def test_user_prompt_basic(self) -> None:
        prompt = self.builder.build_user_prompt(
            transcript_context="surgeon: turn on the fluoroscopy",
            current_phase="Phase3",
            session_time="5m 30s",
        )
        assert "fluoroscopy" in prompt
        assert "Phase3" in prompt
        assert "5m 30s" in prompt

    def test_user_prompt_with_machines(self) -> None:
        prompt = self.builder.build_user_prompt(
            transcript_context="test transcript",
            current_phase="Phase2",
            current_machines={"0": ["M03", "M05"], "1": ["M01"]},
        )
        assert "M01" in prompt
        assert "M03" in prompt

    def test_user_prompt_empty_transcript(self) -> None:
        prompt = self.builder.build_user_prompt(
            transcript_context="",
            current_phase="Phase1",
        )
        assert "no transcript available" in prompt

    def test_build_messages(self) -> None:
        messages = self.builder.build_messages(
            transcript_context="test",
            current_phase="Phase2",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "test" in messages[1]["content"]

    def test_build_completion_prompt(self) -> None:
        prompt = self.builder.build_completion_prompt(
            transcript_context="turn on insufflator",
            current_phase="Phase4",
            session_time="10m 0s",
        )
        assert isinstance(prompt, str)
        assert "insufflator" in prompt
        assert "Phase4" in prompt
        assert "JSON Response:" in prompt

    def test_config_property(self) -> None:
        config = self.builder.config
        assert isinstance(config, dict)
        assert "machines" in config or "surgery" in config


class TestPromptBuilderAllSurgeries:
    """PromptBuilder across all supported surgery types."""

    @pytest.mark.parametrize(
        "surgery", ["PCNL", "Partial Hepatectomy", "Lobectomy"]
    )
    def test_loads_config(self, surgery: str) -> None:
        builder = PromptBuilder(surgery=surgery)
        assert builder.surgery == surgery
        assert builder.machine_count > 0

    @pytest.mark.parametrize(
        "surgery", ["PCNL", "Partial Hepatectomy", "Lobectomy"]
    )
    def test_builds_system_prompt(self, surgery: str) -> None:
        builder = PromptBuilder(surgery=surgery)
        prompt = builder.build_system_prompt()
        assert len(prompt) > 100
        assert surgery in prompt

    @pytest.mark.parametrize(
        "surgery", ["PCNL", "Partial Hepatectomy", "Lobectomy"]
    )
    def test_builds_messages(self, surgery: str) -> None:
        builder = PromptBuilder(surgery=surgery)
        messages = builder.build_messages("test transcript")
        assert len(messages) == 2


class TestPromptBuilderSetSurgery:
    """Test surgery switching."""

    def test_set_surgery_reloads(self) -> None:
        builder = PromptBuilder(surgery="PCNL")
        pcnl_count = builder.machine_count

        builder.set_surgery("Lobectomy")
        assert builder.surgery == "Lobectomy"
        lob_count = builder.machine_count
        # Different surgeries should have different machine counts (or at least work)
        assert lob_count > 0

    def test_set_surgery_same_noop(self) -> None:
        builder = PromptBuilder(surgery="PCNL")
        builder.set_surgery("PCNL")
        assert builder.surgery == "PCNL"

    def test_unknown_surgery(self) -> None:
        builder = PromptBuilder(surgery="UnknownSurgery")
        assert builder.machine_count == 0
        assert builder.phase_count == 0
        prompt = builder.build_system_prompt()
        assert "No machines defined" in prompt


# ═══════════════════════════════════════════════════════════════════════
#  LLMBatcher Tests
# ═══════════════════════════════════════════════════════════════════════


class TestLLMBatcher:
    """LLMBatcher with a mock runner."""

    @staticmethod
    def _make_mock_runner() -> MagicMock:
        """Create a mock runner that returns fixed JSON."""
        runner = MagicMock()
        runner.generate.return_value = {
            "metadata": {"phase": "Phase2", "reasoning": "test"},
            "machines": {"0": [], "1": ["M01"]},
            "confidence": 0.8,
            "source": "medgemma",
        }
        runner.chat.return_value = {
            "metadata": {"phase": "Phase2"},
            "machines": {"0": [], "1": []},
            "confidence": 0.7,
            "source": "medgemma",
        }
        return runner

    @pytest.mark.asyncio
    async def test_batcher_init(self) -> None:
        batcher = LLMBatcher(runner=self._make_mock_runner())
        assert batcher.is_running is False
        assert batcher.queue_size == 0

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        batcher = LLMBatcher(runner=self._make_mock_runner())
        await batcher.start()
        assert batcher.is_running is True
        await batcher.stop()
        assert batcher.is_running is False

    @pytest.mark.asyncio
    async def test_single_request(self) -> None:
        runner = self._make_mock_runner()
        batcher = LLMBatcher(runner=runner, max_batch=4, max_wait_ms=100)
        await batcher.start()

        result = await batcher.submit(
            prompt="test prompt",
            request_id="r1",
            timeout_s=5.0,
        )

        assert result.success is True
        assert result.request_id == "r1"
        assert result.output["machines"]["1"] == ["M01"]
        assert result.processing_time_ms > 0

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_multiple_requests(self) -> None:
        runner = self._make_mock_runner()
        batcher = LLMBatcher(runner=runner, max_batch=4, max_wait_ms=100)
        await batcher.start()

        results = await asyncio.gather(
            batcher.submit("p1", "r1"),
            batcher.submit("p2", "r2"),
            batcher.submit("p3", "r3"),
        )

        assert len(results) == 3
        assert all(r.success for r in results)
        assert {r.request_id for r in results} == {"r1", "r2", "r3"}

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_stats(self) -> None:
        runner = self._make_mock_runner()
        batcher = LLMBatcher(runner=runner, max_batch=4, max_wait_ms=100)
        await batcher.start()

        await batcher.submit("p1", "r1")
        await batcher.submit("p2", "r2")

        stats = batcher.stats
        assert stats["total_processed"] == 2
        assert stats["total_failed"] == 0
        assert stats["total_batches"] >= 1

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_chat_mode(self) -> None:
        runner = self._make_mock_runner()
        batcher = LLMBatcher(runner=runner, max_batch=4, max_wait_ms=100)
        await batcher.start()

        result = await batcher.submit(
            prompt="",
            request_id="chat1",
            use_chat=True,
            messages=[{"role": "user", "content": "test"}],
        )

        assert result.success is True
        runner.chat.assert_called_once()

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_runner_error_returns_failure(self) -> None:
        runner = MagicMock()
        runner.generate.side_effect = RuntimeError("GGUF inference error")
        batcher = LLMBatcher(runner=runner, max_batch=4, max_wait_ms=100)
        await batcher.start()

        result = await batcher.submit("p1", "r1")

        assert result.success is False
        assert "GGUF inference error" in result.error

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_custom_process_fn(self) -> None:
        def custom_fn(prompt: str) -> Dict[str, Any]:
            return {"custom": True, "prompt_len": len(prompt)}

        batcher = LLMBatcher(process_fn=custom_fn, max_batch=2, max_wait_ms=100)
        await batcher.start()

        result = await batcher.submit("hello", "c1")
        assert result.success is True
        assert result.output["custom"] is True
        assert result.output["prompt_len"] == 5

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_async_process_fn(self) -> None:
        async def async_fn(prompt: str) -> Dict[str, Any]:
            return {"async": True}

        batcher = LLMBatcher(process_fn=async_fn, max_batch=2, max_wait_ms=100)
        await batcher.start()

        result = await batcher.submit("test", "a1")
        assert result.success is True
        assert result.output["async"] is True

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_auto_request_id(self) -> None:
        runner = self._make_mock_runner()
        batcher = LLMBatcher(runner=runner, max_batch=4, max_wait_ms=100)
        await batcher.start()

        result = await batcher.submit("test")
        assert result.request_id.startswith("req_")

        await batcher.stop()

    @pytest.mark.asyncio
    async def test_double_start(self) -> None:
        batcher = LLMBatcher(runner=self._make_mock_runner())
        await batcher.start()
        await batcher.start()  # Should not crash
        assert batcher.is_running is True
        await batcher.stop()


# ═══════════════════════════════════════════════════════════════════════
#  LLMManager Tests
# ═══════════════════════════════════════════════════════════════════════


class TestLLMManagerStub:
    """LLMManager in stub mode."""

    @pytest.mark.asyncio
    async def test_init(self) -> None:
        mgr = LLMManager(surgery="PCNL", stub_mode=True)
        assert mgr.surgery == "PCNL"
        assert mgr.is_running is False
        assert mgr.is_model_loaded is False

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        mgr = LLMManager(surgery="PCNL", stub_mode=True)
        await mgr.start()
        assert mgr.is_running is True
        await mgr.stop()
        assert mgr.is_running is False

    @pytest.mark.asyncio
    async def test_submit_stub(self) -> None:
        mgr = LLMManager(surgery="PCNL", stub_mode=True, validate_output=False)
        await mgr.start()

        request = LLMRequest(
            request_id="test1",
            transcript_context="turn on the fluoroscopy",
            surgery_type="PCNL",
            machines_dict={},
            current_phase="Phase3",
            session_time_s=150.0,
        )

        response = await mgr.submit(request)
        assert response.request_id == "test1"
        assert response.success is True
        assert isinstance(response.output, dict)
        assert "machines" in response.output
        assert response.processing_time_ms > 0

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_submit_multiple(self) -> None:
        mgr = LLMManager(surgery="PCNL", stub_mode=True, validate_output=False)
        await mgr.start()

        for i in range(5):
            request = LLMRequest(
                request_id=f"multi_{i}",
                transcript_context=f"test transcript {i}",
                surgery_type="PCNL",
                machines_dict={},
            )
            response = await mgr.submit(request)
            assert response.success is True

        assert mgr.total_requests == 5
        assert mgr.stats["total_failures"] == 0

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_submit_with_machines(self) -> None:
        mgr = LLMManager(surgery="PCNL", stub_mode=True, validate_output=False)
        await mgr.start()

        request = LLMRequest(
            request_id="mach1",
            transcript_context="check insufflator",
            surgery_type="PCNL",
            machines_dict={},
            current_phase="Phase4",
            current_machines={"0": ["M03"], "1": ["M01", "M02"]},
        )

        response = await mgr.submit(request)
        assert response.success is True

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_stats(self) -> None:
        mgr = LLMManager(surgery="PCNL", stub_mode=True, validate_output=False)
        await mgr.start()

        request = LLMRequest(
            request_id="s1",
            transcript_context="test",
            surgery_type="PCNL",
            machines_dict={},
        )
        await mgr.submit(request)

        stats = mgr.stats
        assert stats["total_requests"] == 1
        assert stats["stub_mode"] is True
        assert stats["surgery"] == "PCNL"
        assert "batcher" in stats
        assert "runner" in stats

        await mgr.stop()


class TestLLMManagerFallback:
    """LLMManager fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_model_not_loaded(self) -> None:
        """When model not loaded and not stub, should fallback."""
        mgr = LLMManager(
            surgery="PCNL",
            model_path="nonexistent.gguf",
            stub_mode=False,
            validate_output=False,
        )
        await mgr.start()
        assert mgr.is_model_loaded is False

        request = LLMRequest(
            request_id="f1",
            transcript_context="test",
            surgery_type="PCNL",
            machines_dict={},
        )
        response = await mgr.submit(request)

        assert response.success is False
        assert response.degraded is True
        assert response.output["source"] == "rule"
        assert "degraded" in response.output["metadata"]["reasoning"].lower()

        await mgr.stop()

    @pytest.mark.asyncio
    async def test_fallback_output_has_surgery(self) -> None:
        mgr = LLMManager(
            surgery="PCNL",
            model_path="nonexistent.gguf",
            stub_mode=False,
            validate_output=False,
        )
        await mgr.start()

        request = LLMRequest(
            request_id="f2",
            transcript_context="test",
            surgery_type="Lobectomy",
            machines_dict={},
            current_phase="Phase5",
        )
        response = await mgr.submit(request)

        assert response.output["metadata"]["surgery"] == "Lobectomy"
        assert response.output["metadata"]["phase"] == "Phase5"

        await mgr.stop()


class TestLLMManagerSurgerySwitching:
    """LLMManager surgery type switching."""

    @pytest.mark.asyncio
    async def test_set_surgery(self) -> None:
        mgr = LLMManager(surgery="PCNL", stub_mode=True)
        assert mgr.surgery == "PCNL"

        mgr.set_surgery("Lobectomy")
        assert mgr.surgery == "Lobectomy"


class TestLLMManagerDoubleStart:
    """LLMManager double start/stop."""

    @pytest.mark.asyncio
    async def test_double_start(self) -> None:
        mgr = LLMManager(surgery="PCNL", stub_mode=True)
        await mgr.start()
        await mgr.start()  # Should not crash
        assert mgr.is_running is True
        await mgr.stop()

    @pytest.mark.asyncio
    async def test_double_stop(self) -> None:
        mgr = LLMManager(surgery="PCNL", stub_mode=True)
        await mgr.start()
        await mgr.stop()
        await mgr.stop()  # Should not crash
        assert mgr.is_running is False


# ═══════════════════════════════════════════════════════════════════════
#  Integration Tests
# ═══════════════════════════════════════════════════════════════════════


class TestIntegrationPromptToRunner:
    """PromptBuilder → GGUFRunner(stub) → StateSerializer pipeline."""

    def test_prompt_to_generate(self) -> None:
        """Build prompt, send to runner, normalize output."""
        builder = PromptBuilder(surgery="PCNL")
        runner = GGUFRunner(stub_mode=True)
        runner.load_model()
        serializer = StateSerializer()

        # Build prompt
        prompt = builder.build_completion_prompt(
            transcript_context="turn on the C-arm",
            current_phase="Phase3",
            session_time="5m 0s",
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 100

        # Run inference
        raw = runner.generate(prompt)
        assert isinstance(raw, dict)

        # Normalize
        normalized = serializer.normalize_llm_output(
            raw=raw, surgery="PCNL", phase="Phase3"
        )
        assert normalized["source"] == "medgemma"
        assert "metadata" in normalized
        assert "machines" in normalized

    def test_messages_to_chat(self) -> None:
        """Build messages, send to chat, normalize output."""
        builder = PromptBuilder(surgery="Partial Hepatectomy")
        runner = GGUFRunner(stub_mode=True)
        runner.load_model()
        serializer = StateSerializer()

        messages = builder.build_messages(
            transcript_context="mobilize the liver",
            current_phase="Phase4",
        )
        assert len(messages) == 2

        raw = runner.chat(messages)
        normalized = serializer.normalize_llm_output(
            raw=raw, surgery="Partial Hepatectomy", phase="Phase4"
        )
        assert normalized["source"] == "medgemma"

    def test_full_manager_pipeline_stub(self) -> None:
        """End-to-end: LLMManager → submit → response."""
        async def _run():
            mgr = LLMManager(surgery="PCNL", stub_mode=True, validate_output=False)
            await mgr.start()

            request = LLMRequest(
                request_id="e2e_1",
                transcript_context="activate the electrosurgical unit",
                surgery_type="PCNL",
                machines_dict={},
                current_phase="Phase3",
                session_time_s=300.0,
            )

            response = await mgr.submit(request)
            assert response.success is True
            assert isinstance(response.output, dict)
            assert response.model_name == "medgemma-stub"
            assert response.processing_time_ms > 0

            await mgr.stop()

        asyncio.get_event_loop().run_until_complete(_run())


class TestIntegrationSessionTime:
    """Verify session_time_s float → string conversion in manager."""

    @pytest.mark.asyncio
    async def test_session_time_formatting(self) -> None:
        mgr = LLMManager(surgery="PCNL", stub_mode=True, validate_output=False)
        await mgr.start()

        # 150 seconds = 2m 30s
        request = LLMRequest(
            request_id="time1",
            transcript_context="test",
            surgery_type="PCNL",
            machines_dict={},
            session_time_s=150.0,
        )
        response = await mgr.submit(request)
        assert response.success is True

        # 0 seconds = 0m 0s
        request2 = LLMRequest(
            request_id="time2",
            transcript_context="test",
            surgery_type="PCNL",
            machines_dict={},
            session_time_s=0.0,
        )
        response2 = await mgr.submit(request2)
        assert response2.success is True

        await mgr.stop()


# ═══════════════════════════════════════════════════════════════════════
#  BatchRequest / BatchResult data class tests
# ═══════════════════════════════════════════════════════════════════════


class TestDataClasses:
    """Verify data class construction."""

    def test_batch_request(self) -> None:
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        req = BatchRequest(
            request_id="br1",
            prompt="test prompt",
            future=future,
        )
        assert req.request_id == "br1"
        assert req.prompt == "test prompt"
        assert req.use_chat is False
        assert req.messages is None
        assert req.submitted_at > 0
        loop.close()

    def test_batch_result(self) -> None:
        res = BatchResult(
            request_id="br1",
            output={"key": "value"},
            processing_time_ms=42.5,
            success=True,
        )
        assert res.request_id == "br1"
        assert res.output["key"] == "value"
        assert res.processing_time_ms == 42.5
        assert res.error is None

    def test_llm_request(self) -> None:
        req = LLMRequest(
            request_id="lr1",
            transcript_context="test",
            surgery_type="PCNL",
            machines_dict={"M01": {"name": "Light"}},
        )
        assert req.request_id == "lr1"
        assert req.current_phase == "Phase1"
        assert req.session_time_s == 0.0

    def test_llm_response(self) -> None:
        resp = LLMResponse(
            request_id="lr1",
            output={"machines": {}},
            processing_time_ms=100.0,
        )
        assert resp.success is True
        assert resp.degraded is False
        assert resp.error is None
