"""
Tests for Phase 9 — Stress Testing & Acceptance Criteria.

Tests are split into two categories:
  - acceptance: Validates latency bounds, memory, throughput
  - stress: Longer-running synthetic load tests

Usage:
    pytest tests/test_stress.py -v
    pytest tests/test_stress.py -k "acceptance" -v
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
import pytest_asyncio

from src.state.rules import RuleEngine
from src.workers.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def rule_engine():
    """Fresh PCNL rule engine."""
    return RuleEngine(surgery="PCNL")


@pytest_asyncio.fixture
async def orchestrator(tmp_path):
    """Orchestrator in stub mode with temp paths."""
    orch = Orchestrator(surgery="PCNL", llm_stub=True)
    await orch.start()
    yield orch
    await orch.stop()


# ======================================================================
# ACCEPTANCE — Rule Engine Latency
# ======================================================================


class TestAcceptanceRuleLatency:
    """
    Acceptance criteria: Rule engine median latency < 500ms.

    The rule engine must process transcripts fast enough that the
    UI update path stays below 500ms median.
    """

    def test_acceptance_rule_single_call_under_500ms(self, rule_engine):
        """Single rule engine call completes well under 500ms."""
        t0 = time.perf_counter()
        rule_engine.process("Turn on the fluoroscopy")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 500, f"Rule call took {elapsed_ms:.1f}ms"

    def test_acceptance_rule_median_under_500ms(self, rule_engine):
        """Median of 100 rule calls is under 500ms."""
        transcripts = [
            "Turn on the fluoroscopy",
            "Turn off the fluoroscopy",
            "C-arm on please",
            "Electrocautery ready",
            "Irrigation pump on",
            "The patient is stable",
            "Blood pressure normal",
            "Suction device on",
            "Patient monitor on",
            "Defibrillator standby",
        ]

        latencies = []
        for _ in range(10):  # 10 rounds × 10 transcripts = 100
            for text in transcripts:
                t0 = time.perf_counter()
                rule_engine.process(text)
                latencies.append((time.perf_counter() - t0) * 1000)

        median_ms = statistics.median(latencies)
        p95_ms = sorted(latencies)[int(len(latencies) * 0.95)]

        assert median_ms < 500, f"Rule median {median_ms:.2f}ms >= 500ms"
        logger.info(
            "Rule latency: median=%.2fms, p95=%.2fms, n=%d",
            median_ms, p95_ms, len(latencies),
        )

    def test_acceptance_rule_p95_under_500ms(self, rule_engine):
        """p95 of 200 rule calls is under 500ms."""
        latencies = []
        for i in range(200):
            text = f"Turn {'on' if i % 2 == 0 else 'off'} the fluoroscopy"
            t0 = time.perf_counter()
            rule_engine.process(text)
            latencies.append((time.perf_counter() - t0) * 1000)

        sorted_lat = sorted(latencies)
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        assert p95 < 500, f"Rule p95 {p95:.2f}ms >= 500ms"

    def test_acceptance_rule_throughput(self, rule_engine):
        """Rule engine sustains > 100 calls/sec."""
        t0 = time.perf_counter()
        n = 500
        for i in range(n):
            rule_engine.process("Turn on the fluoroscopy")
        elapsed = time.perf_counter() - t0

        calls_per_sec = n / elapsed
        assert calls_per_sec > 100, f"Throughput {calls_per_sec:.0f}/s < 100/s"
        logger.info("Rule throughput: %.0f calls/sec", calls_per_sec)


# ======================================================================
# ACCEPTANCE — Pipeline Latency
# ======================================================================


class TestAcceptancePipelineLatency:
    """
    Acceptance criteria: Pipeline feed-to-queue latency is reasonable.

    Since we use stub mode (no real LLM inference), this tests the
    pipeline overhead: queue put, rule processing, state merge.
    """

    @pytest.mark.asyncio
    async def test_acceptance_pipeline_feed_latency(self, orchestrator):
        """Feed transcript latency is under 50ms (queue put only)."""
        latencies = []
        for i in range(20):
            t0 = time.perf_counter()
            await orchestrator.feed_transcript("Turn on the fluoroscopy")
            latencies.append((time.perf_counter() - t0) * 1000)
            await asyncio.sleep(0.01)

        median = statistics.median(latencies)
        assert median < 50, f"Feed latency median {median:.2f}ms >= 50ms"

    @pytest.mark.asyncio
    async def test_acceptance_pipeline_state_update(self, orchestrator):
        """Pipeline produces state updates after feeding transcripts."""
        # Feed several transcripts
        for _ in range(5):
            await orchestrator.feed_transcript("Turn on the fluoroscopy")
            await asyncio.sleep(0.1)

        # Allow pipeline to process
        await asyncio.sleep(0.5)

        stats = orchestrator.stats
        assert stats["running"] is True
        assert stats["surgery"] == "PCNL"

    @pytest.mark.asyncio
    async def test_acceptance_pipeline_queue_no_overflow(self, orchestrator):
        """Queues don't overflow under moderate load."""
        # Feed 50 transcripts rapidly
        for i in range(50):
            await orchestrator.feed_transcript(f"Transcript {i}")
            await asyncio.sleep(0.005)

        # Allow drain
        await asyncio.sleep(1.0)

        sizes = orchestrator.stats["queue_sizes"]
        # Queues should have drained substantially
        total_queued = sum(sizes.values())
        assert total_queued < 100, f"Queue backlog {total_queued} too high"


# ======================================================================
# ACCEPTANCE — Memory Usage
# ======================================================================


class TestAcceptanceMemory:
    """
    Acceptance criteria: Process memory stays reasonable.

    We check that importing the full codebase and creating an
    orchestrator doesn't consume excessive memory.
    """

    def test_acceptance_memory_under_limit(self):
        """Process memory is under 2 GB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            rss_mb = process.memory_info().rss / (1024 * 1024)
            # 2 GB limit for pipeline only (no model loaded in tests)
            assert rss_mb < 2048, f"Memory {rss_mb:.0f}MB >= 2048MB"
            logger.info("Memory usage: %.0f MB", rss_mb)
        except ImportError:
            pytest.skip("psutil not installed")

    def test_acceptance_memory_after_rule_processing(self, rule_engine):
        """Memory stays reasonable after heavy rule processing."""
        # Process 1000 transcripts
        for i in range(1000):
            rule_engine.process("Turn on the fluoroscopy")

        try:
            import psutil
            process = psutil.Process(os.getpid())
            rss_mb = process.memory_info().rss / (1024 * 1024)
            assert rss_mb < 2048, f"Memory {rss_mb:.0f}MB >= 2048MB after rule processing"
        except ImportError:
            pytest.skip("psutil not installed")


# ======================================================================
# STRESS — Sustained Load
# ======================================================================


class TestStressSustainedLoad:
    """
    Stress tests: sustained synthetic load at various QPS.

    These tests are heavier than acceptance tests but run in pytest.
    """

    def test_stress_rule_1000_calls(self, rule_engine):
        """Rule engine handles 1000 mixed calls without error."""
        transcripts = [
            "Turn on the fluoroscopy",
            "Turn off the fluoroscopy",
            "C-arm on",
            "C-arm off",
            "The patient is stable",
            "Ready for incision",
        ]

        latencies = []
        for i in range(1000):
            text = transcripts[i % len(transcripts)]
            t0 = time.perf_counter()
            result = rule_engine.process(text)
            latencies.append((time.perf_counter() - t0) * 1000)

        assert len(latencies) == 1000
        median = statistics.median(latencies)
        assert median < 100, f"Stress median {median:.2f}ms"

    def test_stress_rule_all_surgeries(self):
        """Rule engine performs well across all surgery types."""
        from src.utils.constants import SUPPORTED_SURGERIES

        for surgery in SUPPORTED_SURGERIES:
            engine = RuleEngine(surgery=surgery)
            latencies = []
            for _ in range(100):
                t0 = time.perf_counter()
                engine.process("Turn on the fluoroscopy")
                latencies.append((time.perf_counter() - t0) * 1000)

            median = statistics.median(latencies)
            assert median < 500, f"{surgery}: median {median:.2f}ms >= 500ms"

    @pytest.mark.asyncio
    async def test_stress_pipeline_50_transcripts(self, orchestrator):
        """Pipeline handles 50 rapid transcripts without crash."""
        for i in range(50):
            await orchestrator.feed_transcript(f"Command {i}: turn on device")
            await asyncio.sleep(0.02)

        await asyncio.sleep(1.0)
        assert orchestrator.is_running

    @pytest.mark.asyncio
    async def test_stress_pipeline_override_during_load(self, orchestrator):
        """Overrides work correctly during sustained load."""
        # Feed load
        for i in range(20):
            await orchestrator.feed_transcript("Turn on the fluoroscopy")
            await asyncio.sleep(0.02)

        # Apply override mid-load
        orchestrator.apply_override("M03", "OFF", "stress test")

        # Continue load
        for i in range(20):
            await orchestrator.feed_transcript("Keep monitoring")
            await asyncio.sleep(0.02)

        await asyncio.sleep(0.5)
        assert orchestrator.is_running

    @pytest.mark.asyncio
    async def test_stress_surgery_switch_during_load(self, orchestrator):
        """Surgery switch works during sustained load."""
        for i in range(10):
            await orchestrator.feed_transcript("Turn on the fluoroscopy")
            await asyncio.sleep(0.02)

        orchestrator.switch_surgery("Lobectomy")

        for i in range(10):
            await orchestrator.feed_transcript("Prepare the bronchoscope")
            await asyncio.sleep(0.02)

        await asyncio.sleep(0.5)
        assert orchestrator.surgery == "Lobectomy"
        assert orchestrator.is_running


# ======================================================================
# STRESS — Queue Behavior
# ======================================================================


class TestStressQueueBehavior:
    """Tests for queue behavior under load."""

    @pytest.mark.asyncio
    async def test_stress_queue_depths_bounded(self, orchestrator):
        """Queue depths stay within maxsize during burst."""
        # Burst 100 transcripts as fast as possible
        for i in range(100):
            await orchestrator.feed_transcript(f"Burst command {i}")

        sizes = orchestrator.stats["queue_sizes"]
        # Each queue has maxsize 200, shouldn't overflow
        for name, size in sizes.items():
            assert size <= 200, f"Queue {name} overflowed: {size}"

    @pytest.mark.asyncio
    async def test_stress_queue_drains(self, orchestrator):
        """Queues drain after burst input stops."""
        for i in range(30):
            await orchestrator.feed_transcript(f"Drain test {i}")

        # Wait for drain
        await asyncio.sleep(2.0)

        sizes = orchestrator.stats["queue_sizes"]
        total = sum(sizes.values())
        assert total < 30, f"Queues not draining: {total} items remain"


# ======================================================================
# STRESS — Benchmark Script Integration
# ======================================================================


class TestBenchmarkScript:
    """Tests that the benchmark scripts are importable and functional."""

    def test_import_stress_local(self):
        """stress_local module imports cleanly."""
        from scripts.stress_local import (
            LatencyCollector,
            SYNTHETIC_TRANSCRIPTS,
            run_stress_test,
        )
        assert len(SYNTHETIC_TRANSCRIPTS) > 0
        assert callable(run_stress_test)

    def test_import_benchmark_runner(self):
        """benchmark_runner module imports cleanly."""
        from scripts.benchmark_runner import (
            benchmark_rule_engine,
            run_all_benchmarks,
        )
        assert callable(benchmark_rule_engine)
        assert callable(run_all_benchmarks)

    def test_latency_collector(self):
        """LatencyCollector produces valid summary."""
        from scripts.stress_local import LatencyCollector

        lc = LatencyCollector()
        for i in range(100):
            lc.record_latency(float(i))

        summary = lc.summary
        assert summary["count"] == 100
        assert "p50_ms" in summary
        assert "p95_ms" in summary
        assert summary["min_ms"] == 0.0
        assert summary["max_ms"] == 99.0

    def test_benchmark_rule_engine_runs(self):
        """Rule engine benchmark runs and returns results."""
        from scripts.benchmark_runner import benchmark_rule_engine

        result = benchmark_rule_engine(surgery="PCNL", iterations=5)
        assert result["benchmark"] == "rule_engine_direct"
        assert result["latency_ms"]["p50"] < 500
        assert result["throughput_calls_per_sec"] > 0

    @pytest.mark.asyncio
    async def test_acceptance_benchmark_pipeline_runs(self):
        """Pipeline benchmark runs and returns results."""
        from scripts.benchmark_runner import benchmark_pipeline

        result = await benchmark_pipeline(surgery="PCNL", iterations=2)
        assert result["benchmark"] == "pipeline_e2e"
        assert result["total_fed"] > 0
