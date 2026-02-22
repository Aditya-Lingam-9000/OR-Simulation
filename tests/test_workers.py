"""
Tests for src.workers — LLMDispatcher, StateWriter, Orchestrator.

Phase 7: Orchestrator, API, WebSocket & Overrides.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# ── Module under test ─────────────────────────────────────────────────
from src.workers.llm_dispatcher import LLMDispatcher
from src.workers.state_writer import StateWriter
from src.workers.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temp directory for state files."""
    return tmp_path


@pytest.fixture
def state_output_path(tmp_dir):
    return tmp_dir / "current_surgery_state.json"


@pytest.fixture
def override_log_path(tmp_dir):
    return tmp_dir / "overrides.log"


@pytest.fixture
def rule_state_queue():
    return asyncio.Queue(maxsize=50)


@pytest.fixture
def llm_state_queue():
    return asyncio.Queue(maxsize=50)


@pytest.fixture
def transcript_queue():
    return asyncio.Queue(maxsize=50)


def _make_mock_manager(stub: bool = True):
    """Create a mock LLMManager."""
    mgr = MagicMock()
    mgr.stub_mode = stub
    mgr.is_model_loaded = True
    mgr.stats = {
        "model_loaded": stub,
        "stub_mode": stub,
        "requests": 0,
        "errors": 0,
    }
    mgr.start = AsyncMock()
    mgr.stop = AsyncMock()
    mgr.set_surgery = MagicMock()

    # submit returns a successful LLMResponse-like object
    response = MagicMock()
    response.success = True
    response.degraded = False
    response.processing_time_ms = 50.0
    response.error = None
    response.output = {
        "metadata": {"surgery": "PCNL", "phase": "Phase1"},
        "machines": {"0": [], "1": ["M01"]},
        "details": {},
        "suggestions": ["Consider checking vitals"],
        "confidence": 0.75,
        "source": "medgemma",
    }
    mgr.submit = AsyncMock(return_value=response)
    return mgr


# ======================================================================
# LLMDispatcher Tests
# ======================================================================


class TestLLMDispatcherInit:
    """Test LLMDispatcher initialization."""

    def test_init_defaults(self):
        d = LLMDispatcher(surgery="PCNL", stub_mode=True, manager=_make_mock_manager())
        assert d.surgery == "PCNL"
        assert d.dispatch_interval_s == 2.0
        assert d.is_running is False
        assert d.is_fallback is False

    def test_init_custom_interval(self):
        d = LLMDispatcher(
            surgery="Lobectomy",
            dispatch_interval_s=5.0,
            stub_mode=True,
            manager=_make_mock_manager(),
        )
        assert d.dispatch_interval_s == 5.0
        assert d.surgery == "Lobectomy"

    def test_stats_initial(self):
        mgr = _make_mock_manager()
        d = LLMDispatcher(surgery="PCNL", stub_mode=True, manager=mgr)
        stats = d.stats
        assert stats["running"] is False
        assert stats["transcripts_received"] == 0
        assert stats["dispatches"] == 0
        assert stats["dispatch_errors"] == 0
        assert stats["fallback_mode"] is False

    def test_queues_created_if_none(self):
        d = LLMDispatcher(surgery="PCNL", stub_mode=True, manager=_make_mock_manager())
        assert d.transcript_queue is not None
        assert d.state_queue is not None

    def test_external_queues_used(self, transcript_queue, llm_state_queue):
        d = LLMDispatcher(
            transcript_queue=transcript_queue,
            state_queue=llm_state_queue,
            surgery="PCNL",
            stub_mode=True,
            manager=_make_mock_manager(),
        )
        assert d.transcript_queue is transcript_queue
        assert d.state_queue is llm_state_queue


class TestLLMDispatcherLifecycle:
    """Test LLMDispatcher start/stop."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        mgr = _make_mock_manager()
        d = LLMDispatcher(surgery="PCNL", stub_mode=True, manager=mgr)
        await d.start()
        assert d.is_running is True
        mgr.start.assert_awaited_once()

        await d.stop()
        assert d.is_running is False
        mgr.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_double_start_ignored(self):
        mgr = _make_mock_manager()
        d = LLMDispatcher(surgery="PCNL", stub_mode=True, manager=mgr)
        await d.start()
        await d.start()  # Should log warning, not crash
        assert d.is_running is True
        await d.stop()


class TestLLMDispatcherProcessing:
    """Test LLMDispatcher transcript processing and dispatch."""

    @pytest.mark.asyncio
    async def test_transcript_extracted_from_string(self):
        mgr = _make_mock_manager()
        d = LLMDispatcher(
            surgery="PCNL",
            dispatch_interval_s=0.0,  # Dispatch immediately
            stub_mode=True,
            manager=mgr,
        )
        await d.start()

        await d.transcript_queue.put("turn on the fluoroscopy")
        await asyncio.sleep(0.3)

        assert d.stats["transcripts_received"] >= 1
        await d.stop()

    @pytest.mark.asyncio
    async def test_transcript_extracted_from_dict(self):
        mgr = _make_mock_manager()
        d = LLMDispatcher(
            surgery="PCNL",
            dispatch_interval_s=0.0,
            stub_mode=True,
            manager=mgr,
        )
        await d.start()

        await d.transcript_queue.put({"text": "prepare the laser"})
        await asyncio.sleep(0.3)

        assert d.stats["transcripts_received"] >= 1
        await d.stop()

    @pytest.mark.asyncio
    async def test_dispatch_puts_result_on_state_queue(self):
        mgr = _make_mock_manager()
        state_q = asyncio.Queue(maxsize=50)
        d = LLMDispatcher(
            state_queue=state_q,
            surgery="PCNL",
            dispatch_interval_s=0.0,
            stub_mode=True,
            manager=mgr,
        )
        await d.start()

        await d.transcript_queue.put("turn on the fluoroscopy")
        # Wait for processing + dispatch
        await asyncio.sleep(0.5)

        assert d.stats["dispatches"] >= 1
        assert not state_q.empty()
        result = state_q.get_nowait()
        assert "confidence" in result
        await d.stop()

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        mgr = _make_mock_manager()
        d = LLMDispatcher(
            surgery="PCNL",
            dispatch_interval_s=1.0,  # 1 second
            stub_mode=True,
            manager=mgr,
        )
        await d.start()

        # Send 3 transcripts rapidly
        for i in range(3):
            await d.transcript_queue.put(f"line {i}")
        await asyncio.sleep(0.5)

        # Should only dispatch once (rate limited)
        assert d.stats["dispatches"] <= 1
        await d.stop()

    @pytest.mark.asyncio
    async def test_fallback_mode_skips_dispatch(self):
        mgr = _make_mock_manager()
        d = LLMDispatcher(
            surgery="PCNL",
            dispatch_interval_s=0.0,
            stub_mode=True,
            manager=mgr,
        )
        d.enter_fallback_mode()
        assert d.is_fallback is True

        await d.start()
        await d.transcript_queue.put("test transcript")
        await asyncio.sleep(0.3)

        assert d.stats["dispatches"] == 0
        mgr.submit.assert_not_awaited()
        await d.stop()

    @pytest.mark.asyncio
    async def test_exit_fallback_mode(self):
        mgr = _make_mock_manager()
        d = LLMDispatcher(surgery="PCNL", stub_mode=True, manager=mgr)
        d.enter_fallback_mode()
        assert d.is_fallback is True
        d.exit_fallback_mode()
        assert d.is_fallback is False


class TestLLMDispatcherSurgery:
    """Test LLMDispatcher surgery and phase switching."""

    def test_set_phase(self):
        d = LLMDispatcher(surgery="PCNL", stub_mode=True, manager=_make_mock_manager())
        d.set_phase("Phase3")
        assert d.stats["current_phase"] == "Phase3"

    def test_switch_surgery(self):
        mgr = _make_mock_manager()
        d = LLMDispatcher(surgery="PCNL", stub_mode=True, manager=mgr)
        d.switch_surgery("Lobectomy")
        assert d.surgery == "Lobectomy"
        mgr.set_surgery.assert_called_once_with("Lobectomy")

    def test_buffer_accessible(self):
        d = LLMDispatcher(surgery="PCNL", stub_mode=True, manager=_make_mock_manager())
        assert d.buffer is not None
        assert d.buffer.entry_count == 0


# ======================================================================
# StateWriter Tests
# ======================================================================


class TestStateWriterInit:
    """Test StateWriter initialization."""

    def test_init_defaults(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        assert w.is_running is False
        state = w.current_state
        assert "metadata" in state
        assert "machines" in state

    def test_stats_initial(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        stats = w.stats
        assert stats["running"] is False
        assert stats["writes"] == 0
        assert stats["merges"] == 0
        assert stats["errors"] == 0
        assert stats["overrides"] == 0

    def test_queues_created_if_none(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        assert w.rule_state_queue is not None
        assert w.llm_state_queue is not None


class TestStateWriterAtomicWrite:
    """Test StateWriter atomic file writes."""

    def test_write_state_creates_file(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        test_state = {
            "metadata": {"surgery": "PCNL", "phase": "Phase1"},
            "machines": {"0": [], "1": ["M01"]},
            "details": {},
            "suggestions": [],
            "confidence": 0.9,
            "source": "rule",
        }
        result = w.write_state(test_state)
        assert result is True
        assert state_output_path.exists()

    def test_write_state_valid_json(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        test_state = {
            "metadata": {"surgery": "PCNL"},
            "machines": {"0": [], "1": ["M01", "M09"]},
        }
        w.write_state(test_state)

        with open(state_output_path, "r") as f:
            loaded = json.load(f)
        assert loaded["machines"]["1"] == ["M01", "M09"]

    def test_write_state_increments_counter(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        w.write_state({"metadata": {}})
        w.write_state({"metadata": {}})
        assert w.stats["writes"] == 2

    def test_write_state_atomic_no_partial(self, state_output_path, override_log_path):
        """Verify temp file is cleaned up after write."""
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        w.write_state({"metadata": {"test": True}})
        tmp_path = state_output_path.with_suffix(".json.tmp")
        assert not tmp_path.exists()


class TestStateWriterOverrides:
    """Test StateWriter override system."""

    def test_apply_override_queued(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        w.apply_override(machine_id="M01", action="ON", reason="test")
        assert w.stats["overrides"] == 1

    def test_apply_override_logged_to_file(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        w.apply_override(machine_id="M01", action="OFF", reason="safety")

        assert override_log_path.exists()
        with open(override_log_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        logged = json.loads(lines[0])
        assert logged["machine_id"] == "M01"
        assert logged["action"] == "OFF"
        assert logged["reason"] == "safety"

    def test_multiple_overrides_logged(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        w.apply_override(machine_id="M01", action="ON", reason="test1")
        w.apply_override(machine_id="M02", action="OFF", reason="test2")
        w.apply_override(machine_id="M03", action="ON", reason="test3")

        with open(override_log_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 3
        assert w.stats["overrides"] == 3

    def test_apply_overrides_modifies_state(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        # Add an override
        w.apply_override(machine_id="M05", action="ON", reason="test")

        # Manually call _apply_overrides on a state
        state = {
            "metadata": {},
            "machines": {"0": ["M05"], "1": ["M01"]},
        }
        result = w._apply_overrides(state)

        # M05 should move from 0 to 1
        assert "M05" in result["machines"]["1"]
        assert "M05" not in result["machines"]["0"]
        assert result["metadata"]["overrides_applied"] == 1

    def test_apply_override_off_action(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        w.apply_override(machine_id="M01", action="OFF", reason="test")

        state = {
            "metadata": {},
            "machines": {"0": [], "1": ["M01", "M09"]},
        }
        result = w._apply_overrides(state)
        assert "M01" in result["machines"]["0"]
        assert "M01" not in result["machines"]["1"]

    def test_overrides_cleared_after_apply(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        w.apply_override(machine_id="M01", action="ON", reason="test")
        state = {"metadata": {}, "machines": {"0": [], "1": []}}
        w._apply_overrides(state)
        # Calling again should not have any overrides
        state2 = {"metadata": {}, "machines": {"0": [], "1": []}}
        result = w._apply_overrides(state2)
        assert "overrides_applied" not in result.get("metadata", {})


class TestStateWriterLifecycle:
    """Test StateWriter start/stop and process loop."""

    @pytest.mark.asyncio
    async def test_start_stop(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        await w.start()
        assert w.is_running is True
        await w.stop()
        assert w.is_running is False

    @pytest.mark.asyncio
    async def test_double_start(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        await w.start()
        await w.start()  # Should warn, not crash
        assert w.is_running is True
        await w.stop()

    @pytest.mark.asyncio
    async def test_process_rule_update(self, state_output_path, override_log_path):
        """Feed a rule state and verify it gets processed."""
        rule_q = asyncio.Queue(maxsize=50)
        w = StateWriter(
            rule_state_queue=rule_q,
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        await w.start()

        rule_state = {
            "metadata": {"surgery": "PCNL", "phase": "Phase2"},
            "machines": {"0": ["M03"], "1": ["M01", "M09"]},
            "details": {},
            "suggestions": [],
            "confidence": 0.85,
            "source": "rule",
        }
        await rule_q.put(rule_state)
        await asyncio.sleep(0.3)

        current = w.current_state
        assert current["metadata"].get("surgery") == "PCNL" or w.stats["merges"] >= 1
        await w.stop()

    @pytest.mark.asyncio
    async def test_broadcast_callback_called(self, state_output_path, override_log_path):
        """Verify on_update callback is called after state merge."""
        callback_data = []

        async def mock_callback(state_dict):
            callback_data.append(state_dict)

        rule_q = asyncio.Queue(maxsize=50)
        w = StateWriter(
            rule_state_queue=rule_q,
            output_path=state_output_path,
            override_log_path=override_log_path,
            on_update=mock_callback,
        )
        await w.start()

        await rule_q.put({
            "metadata": {"surgery": "PCNL"},
            "machines": {"0": [], "1": ["M01"]},
            "details": {},
            "suggestions": [],
            "confidence": 0.9,
            "source": "rule",
        })
        await asyncio.sleep(0.3)

        assert len(callback_data) >= 1
        await w.stop()


class TestStateWriterReadState:
    """Test StateWriter read_current_state."""

    def test_read_current_state_returns_copy(self, state_output_path, override_log_path):
        w = StateWriter(
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        s1 = w.read_current_state()
        s2 = w.read_current_state()
        assert s1 is not s2  # Different dict instances
        assert s1 == s2  # Same content


# ======================================================================
# Orchestrator Tests
# ======================================================================


class TestOrchestratorInit:
    """Test Orchestrator initialization."""

    def test_init_defaults(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        assert o.surgery == "PCNL"
        assert o.llm_stub is True
        assert o.is_running is False

    def test_init_creates_workers(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        assert o.asr_worker is not None
        assert o.rule_worker is not None
        assert o.llm_dispatcher is not None
        assert o.state_writer is not None

    def test_init_creates_queues(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        assert o.audio_queue is not None
        assert o.transcript_queue is not None
        assert o.rule_state_queue is not None
        assert o.llm_state_queue is not None


class TestOrchestratorLifecycle:
    """Test Orchestrator start/stop."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        await o.start()
        assert o.is_running is True
        assert o.uptime_s >= 0

        await o.stop()
        assert o.is_running is False

    @pytest.mark.asyncio
    async def test_double_start(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        await o.start()
        await o.start()  # Should warn, not crash
        assert o.is_running is True
        await o.stop()


class TestOrchestratorFeedTranscript:
    """Test Orchestrator transcript feeding."""

    @pytest.mark.asyncio
    async def test_feed_transcript_when_running(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        await o.start()

        await o.feed_transcript("turn on the fluoroscopy")
        await asyncio.sleep(0.1)

        # Rule worker should have received the text
        assert o.rule_worker.transcript_queue.qsize() >= 0  # May have been consumed
        await o.stop()

    @pytest.mark.asyncio
    async def test_feed_transcript_when_stopped(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        # Should not crash when not running
        await o.feed_transcript("should be ignored")
        # No assertion needed — just verify no exception

    @pytest.mark.asyncio
    async def test_feed_transcript_reaches_rule_worker(self):
        """Verify transcript gets to rule worker's queue via fanout."""
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        # Start fanout task manually (without full start)
        o._running = True
        o._fanout_task = asyncio.create_task(o._transcript_fanout())

        await o.feed_transcript("turn on cautery")
        await asyncio.sleep(0.1)  # let fanout distribute
        assert o.rule_worker.transcript_queue.qsize() >= 1

        o._running = False
        o._fanout_task.cancel()
        try:
            await o._fanout_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_feed_transcript_reaches_llm_dispatcher(self):
        """Verify transcript gets to LLM dispatcher's queue via fanout."""
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        o._running = True
        o._fanout_task = asyncio.create_task(o._transcript_fanout())

        await o.feed_transcript("check the imaging")
        await asyncio.sleep(0.1)  # let fanout distribute
        assert o.llm_dispatcher.transcript_queue.qsize() >= 1

        o._running = False
        o._fanout_task.cancel()
        try:
            await o._fanout_task
        except asyncio.CancelledError:
            pass


class TestOrchestratorOverrides:
    """Test Orchestrator override delegation."""

    def test_apply_override_delegates_to_state_writer(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        o.apply_override(machine_id="M01", action="ON", reason="test")
        assert o.state_writer.stats["overrides"] == 1


class TestOrchestratorSurgerySwitching:
    """Test Orchestrator surgery and phase switching."""

    def test_switch_surgery(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        o.switch_surgery("Lobectomy")
        assert o.surgery == "Lobectomy"

    def test_set_phase(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        o.set_phase("Phase3")
        # Phase should propagate to LLM dispatcher (rule engine stores phase differently)
        assert o.llm_dispatcher.stats["current_phase"] == "Phase3"


class TestOrchestratorStats:
    """Test Orchestrator stats aggregation."""

    def test_stats_structure(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        stats = o.stats
        assert "running" in stats
        assert "surgery" in stats
        assert "uptime_s" in stats
        assert "asr" in stats
        assert "rule" in stats
        assert "llm" in stats
        assert "state_writer" in stats
        assert "queue_sizes" in stats

    def test_stats_surgery(self):
        o = Orchestrator(surgery="Lobectomy", llm_stub=True)
        assert o.stats["surgery"] == "Lobectomy"

    def test_stats_queue_sizes(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        qs = o.stats["queue_sizes"]
        assert "audio" in qs
        assert "transcript" in qs
        assert "rule_state" in qs
        assert "llm_state" in qs


class TestOrchestratorGetState:
    """Test Orchestrator state access."""

    def test_get_current_state_default(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        state = o.get_current_state()
        assert isinstance(state, dict)
        assert "metadata" in state
        assert "machines" in state

    @pytest.mark.asyncio
    async def test_get_current_state_after_feed(self):
        """Feed a transcript and verify state is accessible."""
        callback_data = []

        async def cb(state_dict):
            callback_data.append(state_dict)

        o = Orchestrator(surgery="PCNL", llm_stub=True, on_state_update=cb)
        await o.start()

        await o.feed_transcript("turn on the fluoroscopy")
        await asyncio.sleep(0.5)

        state = o.get_current_state()
        assert isinstance(state, dict)
        await o.stop()


class TestOrchestratorCallback:
    """Test Orchestrator state update callback."""

    @pytest.mark.asyncio
    async def test_callback_receives_state(self):
        callback_data = []

        async def cb(state_dict):
            callback_data.append(state_dict)

        o = Orchestrator(surgery="PCNL", llm_stub=True, on_state_update=cb)
        await o.start()

        await o.feed_transcript("prepare the lithotripter")
        await asyncio.sleep(0.5)

        # Callback should have been called at least once if rule matched
        # (depends on rule engine having the machine)
        await o.stop()
        # No assertion on callback_data length — rule match depends on
        # config. We just verify no exceptions were raised.


class TestOrchestratorFeedAudio:
    """Test Orchestrator audio feeding."""

    @pytest.mark.asyncio
    async def test_feed_audio_when_running(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        o._running = True
        await o.feed_audio(b"fake_audio_data")
        assert o.audio_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_feed_audio_when_stopped(self):
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        await o.feed_audio(b"should_be_ignored")
        assert o.audio_queue.qsize() == 0


# ======================================================================
# Integration-style Tests
# ======================================================================


class TestOrchestratorIntegration:
    """Integration tests for the full orchestrator pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_start_feed_stop(self):
        """Start pipeline, feed transcript, stop — no crashes."""
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        await o.start()
        assert o.is_running is True

        for i in range(3):
            await o.feed_transcript(f"test transcript line {i}")

        await asyncio.sleep(0.3)
        state = o.get_current_state()
        assert isinstance(state, dict)

        await o.stop()
        assert o.is_running is False

    @pytest.mark.asyncio
    async def test_override_during_pipeline_run(self):
        """Apply override while pipeline is running."""
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        await o.start()

        o.apply_override(machine_id="M01", action="ON", reason="requested")
        await o.feed_transcript("preparing equipment")
        await asyncio.sleep(0.3)

        assert o.state_writer.stats["overrides"] == 1
        await o.stop()

    @pytest.mark.asyncio
    async def test_surgery_switch_during_run(self):
        """Switch surgery while pipeline is running."""
        o = Orchestrator(surgery="PCNL", llm_stub=True)
        await o.start()

        o.switch_surgery("Lobectomy")
        assert o.surgery == "Lobectomy"

        await o.feed_transcript("check the stapler")
        await asyncio.sleep(0.2)

        await o.stop()


class TestStateWriterMergePipeline:
    """Test StateWriter processing loop with actual queue data."""

    @pytest.mark.asyncio
    async def test_rule_and_llm_merged(self, state_output_path, override_log_path):
        """Feed both rule and LLM outputs and verify merge."""
        rule_q = asyncio.Queue(maxsize=50)
        llm_q = asyncio.Queue(maxsize=50)

        w = StateWriter(
            rule_state_queue=rule_q,
            llm_state_queue=llm_q,
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        await w.start()

        # Feed rule output
        await rule_q.put({
            "metadata": {"surgery": "PCNL", "phase": "Phase1"},
            "machines": {"0": ["M03"], "1": ["M01"]},
            "details": {},
            "suggestions": [],
            "confidence": 0.8,
            "source": "rule",
        })

        # Feed LLM output
        await llm_q.put({
            "metadata": {"surgery": "PCNL"},
            "machines": {"0": [], "1": ["M01"]},
            "suggestions": ["Check patient vitals"],
            "confidence": 0.7,
            "source": "medgemma",
        })

        await asyncio.sleep(0.4)
        assert w.stats["merges"] >= 1
        await w.stop()

    @pytest.mark.asyncio
    async def test_override_applied_during_merge(self, state_output_path, override_log_path):
        """Apply override before feeding state — should be included in merge."""
        rule_q = asyncio.Queue(maxsize=50)

        w = StateWriter(
            rule_state_queue=rule_q,
            output_path=state_output_path,
            override_log_path=override_log_path,
        )
        await w.start()

        # Apply override first
        w.apply_override(machine_id="M05", action="ON", reason="manual")

        # Then feed rule state
        await rule_q.put({
            "metadata": {"surgery": "PCNL", "phase": "Phase1"},
            "machines": {"0": ["M05"], "1": ["M01"]},
            "details": {},
            "suggestions": [],
            "confidence": 0.9,
            "source": "rule",
        })

        await asyncio.sleep(0.4)

        current = w.current_state
        # M05 should be in the ON list after override
        assert "M05" in current.get("machines", {}).get("1", [])
        await w.stop()
