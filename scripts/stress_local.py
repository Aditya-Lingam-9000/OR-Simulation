"""
OR-Symphony: Local Stress Test Script

Pushes synthetic transcripts at configurable QPS through the pipeline
(using the Orchestrator directly, no HTTP server required).

Measures:
  - Per-transcript rule-path latency
  - Queue depths over time
  - Memory usage
  - Throughput (transcripts/sec)

Usage:
    python -m scripts.stress_local --qps 5 --duration 30
    python -m scripts.stress_local --qps 10 --duration 60 --surgery PCNL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.workers.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("stress_local")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Synthetic transcript data — realistic OR commands
# ---------------------------------------------------------------------------

SYNTHETIC_TRANSCRIPTS = [
    "Turn on the fluoroscopy please",
    "We need the C-arm now",
    "Can we get the electrocautery ready",
    "Fluoroscopy off",
    "Let's start with the irrigation pump",
    "Turn on the patient monitor",
    "Defibrillator on standby",
    "We'll need the suction device",
    "Turn off the warming blanket",
    "Keep the ventilator running",
    "X-ray on",
    "Prepare the ultrasound machine",
    "Anesthesia machine check complete",
    "Blood pressure monitor is reading",
    "Turn on the overhead lights",
    "Switch to the high-flow insufflator",
    "The laser needs to be activated",
    "Bring in the harmonic scalpel",
    "Monitor the oxygen saturation",
    "We need another suction unit",
    "Blood warmer on",
    "Let me check the EKG monitor",
    "Adjust the operating table position",
    "The camera system needs calibrating",
    "Pulse oximeter reading is stable",
    "Turn on the nerve stimulator",
    "We're ready for the stent placement",
    "Prepare for fluoroscopy guidance",
    "Nephroscope is inserted",
    "Start the lithotripter",
]


# ---------------------------------------------------------------------------
# Latency collector
# ---------------------------------------------------------------------------


class LatencyCollector:
    """Collects per-transcript latency measurements."""

    def __init__(self) -> None:
        self.latencies: List[float] = []
        self.queue_snapshots: List[Dict[str, int]] = []
        self.state_update_count: int = 0
        self._last_update_time: float = 0.0

    def record_latency(self, latency_ms: float) -> None:
        self.latencies.append(latency_ms)

    def record_queue_snapshot(self, snapshot: Dict[str, int]) -> None:
        self.queue_snapshots.append(snapshot)

    def on_state_update(self) -> None:
        self.state_update_count += 1
        self._last_update_time = time.time()

    @property
    def summary(self) -> Dict[str, Any]:
        if not self.latencies:
            return {"count": 0, "note": "No latencies recorded"}

        sorted_lat = sorted(self.latencies)
        n = len(sorted_lat)

        return {
            "count": n,
            "min_ms": round(sorted_lat[0], 2),
            "max_ms": round(sorted_lat[-1], 2),
            "mean_ms": round(statistics.mean(sorted_lat), 2),
            "median_ms": round(statistics.median(sorted_lat), 2),
            "p50_ms": round(sorted_lat[int(n * 0.50)], 2),
            "p90_ms": round(sorted_lat[int(n * 0.90)], 2),
            "p95_ms": round(sorted_lat[int(n * 0.95)], 2),
            "p99_ms": round(sorted_lat[min(int(n * 0.99), n - 1)], 2),
            "stddev_ms": round(statistics.stdev(sorted_lat), 2) if n > 1 else 0.0,
            "state_updates": self.state_update_count,
        }


# ---------------------------------------------------------------------------
# Stress runner
# ---------------------------------------------------------------------------


async def run_stress_test(
    qps: float,
    duration_s: float,
    surgery: str,
) -> Dict[str, Any]:
    """
    Run a stress test against the pipeline.

    Args:
        qps: Target queries (transcripts) per second.
        duration_s: Test duration in seconds.
        surgery: Surgery type.

    Returns:
        Results dict with latency stats and queue metrics.
    """
    collector = LatencyCollector()

    # State update callback
    async def on_update(state: Dict[str, Any]) -> None:
        collector.on_state_update()

    # Create orchestrator in stub mode (no real LLM)
    orch = Orchestrator(
        surgery=surgery,
        llm_stub=True,
        on_state_update=on_update,
    )

    logger.info(
        "Starting stress test — QPS=%.1f, duration=%.0fs, surgery=%s",
        qps, duration_s, surgery,
    )

    await orch.start()

    interval = 1.0 / qps if qps > 0 else 1.0
    start_time = time.time()
    transcript_idx = 0
    sent_count = 0

    try:
        while (time.time() - start_time) < duration_s:
            text = SYNTHETIC_TRANSCRIPTS[transcript_idx % len(SYNTHETIC_TRANSCRIPTS)]
            transcript_idx += 1

            # Measure feed-to-queue latency
            t0 = time.perf_counter()
            await orch.feed_transcript(text)
            feed_latency_ms = (time.perf_counter() - t0) * 1000
            collector.record_latency(feed_latency_ms)
            sent_count += 1

            # Record queue depths
            stats = orch.stats
            collector.record_queue_snapshot(stats.get("queue_sizes", {}))

            # Pace to target QPS
            elapsed = time.time() - start_time
            expected = sent_count * interval
            sleep_time = expected - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        pass

    # Allow pipeline to drain
    await asyncio.sleep(0.5)

    # Collect final stats
    pipeline_stats = orch.stats

    await orch.stop()

    total_time = time.time() - start_time
    actual_qps = sent_count / total_time if total_time > 0 else 0

    # Memory usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        mem_mb = -1.0

    results = {
        "test_config": {
            "target_qps": qps,
            "duration_s": duration_s,
            "surgery": surgery,
        },
        "throughput": {
            "transcripts_sent": sent_count,
            "actual_qps": round(actual_qps, 2),
            "total_time_s": round(total_time, 2),
        },
        "latency": collector.summary,
        "memory_mb": round(mem_mb, 1),
        "pipeline_stats": pipeline_stats,
        "queue_max_depths": _max_queue_depths(collector.queue_snapshots),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return results


def _max_queue_depths(snapshots: List[Dict[str, int]]) -> Dict[str, int]:
    """Compute max queue depth across all snapshots."""
    if not snapshots:
        return {}
    result = {}
    for key in snapshots[0]:
        result[key] = max(s.get(key, 0) for s in snapshots)
    return result


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------


def print_report(results: Dict[str, Any]) -> None:
    """Print a human-readable stress test report."""
    config = results["test_config"]
    throughput = results["throughput"]
    latency = results["latency"]

    print("\n" + "=" * 60)
    print("  OR-Symphony Local Stress Test Report")
    print("=" * 60)
    print(f"  Surgery:     {config['surgery']}")
    print(f"  Target QPS:  {config['target_qps']}")
    print(f"  Duration:    {config['duration_s']}s")
    print(f"  Timestamp:   {results['timestamp']}")
    print("-" * 60)
    print(f"  Transcripts sent:  {throughput['transcripts_sent']}")
    print(f"  Actual QPS:        {throughput['actual_qps']}")
    print(f"  Total time:        {throughput['total_time_s']}s")
    print("-" * 60)
    print(f"  Latency (feed-to-queue):")
    print(f"    p50:    {latency.get('p50_ms', 'N/A')} ms")
    print(f"    p90:    {latency.get('p90_ms', 'N/A')} ms")
    print(f"    p95:    {latency.get('p95_ms', 'N/A')} ms")
    print(f"    p99:    {latency.get('p99_ms', 'N/A')} ms")
    print(f"    mean:   {latency.get('mean_ms', 'N/A')} ms")
    print(f"    median: {latency.get('median_ms', 'N/A')} ms")
    print(f"    min:    {latency.get('min_ms', 'N/A')} ms")
    print(f"    max:    {latency.get('max_ms', 'N/A')} ms")
    print(f"  State updates:     {latency.get('state_updates', 0)}")
    print("-" * 60)
    print(f"  Memory:  {results['memory_mb']} MB")
    print(f"  Max queue depths:  {results.get('queue_max_depths', {})}")
    print("=" * 60)

    # Acceptance criteria
    median = latency.get("median_ms", 999)
    print("\n  Acceptance Criteria:")
    rule_pass = median < 500
    print(f"    [{'PASS' if rule_pass else 'FAIL'}] Rule path median < 500ms"
          f" (actual: {median}ms)")
    print(f"    [INFO] MedGemma p50 < 3s — requires Kaggle GPU test")
    if results["memory_mb"] > 0:
        mem_pass = results["memory_mb"] < 2048  # 2 GB target for pipeline only
        print(f"    [{'PASS' if mem_pass else 'FAIL'}] Memory < 2 GB"
              f" (actual: {results['memory_mb']} MB)")
    print()


def save_results(results: Dict[str, Any], output_dir: Path) -> Path:
    """Save results as JSON to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stress_{ts}.json"
    path = output_dir / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="OR-Symphony Local Stress Test",
    )
    parser.add_argument("--qps", type=float, default=5.0,
                        help="Target queries per second (default: 5)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Test duration in seconds (default: 30)")
    parser.add_argument("--surgery", type=str, default="PCNL",
                        help="Surgery type (default: PCNL)")
    parser.add_argument("--output", type=str, default="reports/stress_results",
                        help="Output directory for results")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to file")

    args = parser.parse_args()

    results = await run_stress_test(
        qps=args.qps,
        duration_s=args.duration,
        surgery=args.surgery,
    )

    print_report(results)

    if not args.no_save:
        save_results(results, Path(args.output))


if __name__ == "__main__":
    asyncio.run(main())
