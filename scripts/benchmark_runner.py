"""
OR-Symphony: Benchmark Runner

Runs the full pipeline in-process with controlled inputs and measures
end-to-end latencies from transcript feed to state output.

Captures:
  - Rule engine latency per transcript
  - State writer merge + write latency
  - End-to-end pipeline latency (feed → state change)
  - Memory usage profile

Usage:
    python -m scripts.benchmark_runner
    python -m scripts.benchmark_runner --iterations 200 --surgery PCNL
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state.rules import RuleEngine
from src.workers.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Benchmark transcripts — one per well-known machine command
# ---------------------------------------------------------------------------

RULE_ENGINE_TRANSCRIPTS = [
    "Turn on the fluoroscopy",
    "Turn off the fluoroscopy",
    "C-arm on please",
    "C-arm off",
    "Electrocautery ready",
    "Stop the electrocautery",
    "Irrigation pump on",
    "Irrigation pump off",
    "Suction device on",
    "Patient monitor on",
]

NEUTRAL_TRANSCRIPTS = [
    "The patient is stable",
    "Blood pressure is 120 over 80",
    "Proceeding with the next step",
    "Positioning the patient",
    "Ready for incision",
]


# ---------------------------------------------------------------------------
# Direct rule engine benchmark
# ---------------------------------------------------------------------------


def benchmark_rule_engine(
    surgery: str = "PCNL",
    iterations: int = 100,
) -> Dict[str, Any]:
    """
    Benchmark the rule engine directly (no async, no pipeline).

    Returns latency statistics for rule matching.
    """
    engine = RuleEngine(surgery=surgery)
    latencies: List[float] = []

    transcripts = RULE_ENGINE_TRANSCRIPTS + NEUTRAL_TRANSCRIPTS
    total = len(transcripts) * iterations

    for _ in range(iterations):
        for transcript in transcripts:
            t0 = time.perf_counter()
            engine.process(transcript)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    return {
        "benchmark": "rule_engine_direct",
        "surgery": surgery,
        "iterations": iterations,
        "total_calls": total,
        "latency_ms": {
            "min": round(sorted_lat[0], 3),
            "max": round(sorted_lat[-1], 3),
            "mean": round(statistics.mean(sorted_lat), 3),
            "median": round(statistics.median(sorted_lat), 3),
            "p50": round(sorted_lat[int(n * 0.50)], 3),
            "p90": round(sorted_lat[int(n * 0.90)], 3),
            "p95": round(sorted_lat[int(n * 0.95)], 3),
            "p99": round(sorted_lat[min(int(n * 0.99), n - 1)], 3),
            "stddev": round(statistics.stdev(sorted_lat), 3) if n > 1 else 0.0,
        },
        "throughput_calls_per_sec": round(total / (sum(sorted_lat) / 1000), 1),
    }


# ---------------------------------------------------------------------------
# Pipeline benchmark (async orchestrator)
# ---------------------------------------------------------------------------


async def benchmark_pipeline(
    surgery: str = "PCNL",
    iterations: int = 50,
) -> Dict[str, Any]:
    """
    Benchmark the full pipeline (orchestrator in stub mode).

    Feeds transcripts and measures time until state update callback fires.
    """
    latencies: List[float] = []
    update_event = asyncio.Event()

    async def on_state_update(state: Dict[str, Any]) -> None:
        update_event.set()

    orch = Orchestrator(
        surgery=surgery,
        llm_stub=True,
        on_state_update=on_state_update,
    )

    await orch.start()

    transcripts = RULE_ENGINE_TRANSCRIPTS
    total = 0

    for _ in range(iterations):
        for transcript in transcripts:
            update_event.clear()

            t0 = time.perf_counter()
            await orch.feed_transcript(transcript)

            # Wait for state update (with timeout)
            try:
                await asyncio.wait_for(update_event.wait(), timeout=2.0)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed_ms)
            except asyncio.TimeoutError:
                # State update didn't fire — count it but don't add latency
                pass

            total += 1

            # Small pause to avoid flooding
            await asyncio.sleep(0.01)

    await orch.stop()

    if not latencies:
        return {
            "benchmark": "pipeline_e2e",
            "surgery": surgery,
            "iterations": iterations,
            "total_fed": total,
            "note": "No state updates received",
        }

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    return {
        "benchmark": "pipeline_e2e",
        "surgery": surgery,
        "iterations": iterations,
        "total_fed": total,
        "state_updates_received": len(latencies),
        "latency_ms": {
            "min": round(sorted_lat[0], 3),
            "max": round(sorted_lat[-1], 3),
            "mean": round(statistics.mean(sorted_lat), 3),
            "median": round(statistics.median(sorted_lat), 3),
            "p50": round(sorted_lat[int(n * 0.50)], 3),
            "p90": round(sorted_lat[int(n * 0.90)], 3),
            "p95": round(sorted_lat[int(n * 0.95)], 3),
            "p99": round(sorted_lat[min(int(n * 0.99), n - 1)], 3),
            "stddev": round(statistics.stdev(sorted_lat), 3) if n > 1 else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Memory benchmark
# ---------------------------------------------------------------------------


def benchmark_memory() -> Dict[str, Any]:
    """Measure current process memory usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        return {
            "rss_mb": round(mem.rss / (1024 * 1024), 1),
            "vms_mb": round(mem.vms / (1024 * 1024), 1),
        }
    except ImportError:
        return {"note": "psutil not installed — memory stats unavailable"}


# ---------------------------------------------------------------------------
# Full benchmark suite
# ---------------------------------------------------------------------------


async def run_all_benchmarks(
    surgery: str = "PCNL",
    rule_iterations: int = 100,
    pipeline_iterations: int = 50,
) -> Dict[str, Any]:
    """Run all benchmarks and return combined results."""

    logger.info("Running rule engine benchmark...")
    rule_result = benchmark_rule_engine(surgery, rule_iterations)
    logger.info("Rule engine: p50=%.3fms, p95=%.3fms",
                rule_result["latency_ms"]["p50"],
                rule_result["latency_ms"]["p95"])

    logger.info("Running pipeline benchmark...")
    pipeline_result = await benchmark_pipeline(surgery, pipeline_iterations)
    if "latency_ms" in pipeline_result:
        logger.info("Pipeline: p50=%.3fms, p95=%.3fms",
                     pipeline_result["latency_ms"]["p50"],
                     pipeline_result["latency_ms"]["p95"])

    mem_result = benchmark_memory()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "surgery": surgery,
        "rule_engine": rule_result,
        "pipeline": pipeline_result,
        "memory": mem_result,
    }


def print_benchmark_report(results: Dict[str, Any]) -> None:
    """Print human-readable benchmark report."""
    print("\n" + "=" * 60)
    print("  OR-Symphony Benchmark Report")
    print("=" * 60)
    print(f"  Surgery:   {results['surgery']}")
    print(f"  Timestamp: {results['timestamp']}")

    # Rule engine
    rule = results["rule_engine"]
    print("\n  Rule Engine (direct):")
    print(f"    Calls:      {rule['total_calls']}")
    print(f"    Throughput: {rule['throughput_calls_per_sec']} calls/sec")
    rl = rule["latency_ms"]
    print(f"    p50:  {rl['p50']}ms  |  p90: {rl['p90']}ms  |  p95: {rl['p95']}ms")

    # Pipeline
    pipeline = results["pipeline"]
    if "latency_ms" in pipeline:
        print("\n  Pipeline (e2e, stub LLM):")
        print(f"    Transcripts: {pipeline['total_fed']}")
        print(f"    Updates:     {pipeline['state_updates_received']}")
        pl = pipeline["latency_ms"]
        print(f"    p50:  {pl['p50']}ms  |  p90: {pl['p90']}ms  |  p95: {pl['p95']}ms")
    else:
        print(f"\n  Pipeline: {pipeline.get('note', 'No data')}")

    # Memory
    mem = results["memory"]
    if "rss_mb" in mem:
        print(f"\n  Memory: RSS={mem['rss_mb']}MB, VMS={mem['vms_mb']}MB")
    else:
        print(f"\n  Memory: {mem.get('note', 'Unavailable')}")

    # Acceptance
    print("\n  Acceptance Criteria:")
    if "latency_ms" in rule:
        rule_pass = rule["latency_ms"]["median"] < 500
        print(f"    [{'PASS' if rule_pass else 'FAIL'}] Rule path median < 500ms"
              f" (actual: {rule['latency_ms']['median']}ms)")

    if "rss_mb" in mem:
        mem_pass = mem["rss_mb"] < 2048
        print(f"    [{'PASS' if mem_pass else 'FAIL'}] Memory < 2 GB"
              f" (actual: {mem['rss_mb']}MB)")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="OR-Symphony Benchmark Runner")
    parser.add_argument("--surgery", default="PCNL", help="Surgery type")
    parser.add_argument("--rule-iterations", type=int, default=100)
    parser.add_argument("--pipeline-iterations", type=int, default=50)
    parser.add_argument("--output", default="reports/benchmarks")
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args()

    results = await run_all_benchmarks(
        surgery=args.surgery,
        rule_iterations=args.rule_iterations,
        pipeline_iterations=args.pipeline_iterations,
    )

    print_benchmark_report(results)

    if not args.no_save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"benchmark_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", path)


if __name__ == "__main__":
    asyncio.run(main())
