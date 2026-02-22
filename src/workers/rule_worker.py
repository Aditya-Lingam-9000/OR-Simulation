"""
OR-Symphony: Rule Engine Worker

Polls transcripts from the ASR output queue, runs the deterministic
rule engine, and writes pending state updates to the state queue.

Designed for single-writer, single-reader pattern with asyncio queues.

Usage:
    worker = RuleWorker(transcript_queue, state_queue, surgery="PCNL")
    await worker.start()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from src.asr.runner import ASRResult
from src.state.rules import RuleEngine, RuleEngineResult

logger = logging.getLogger(__name__)


class RuleWorker:
    """
    Rule engine processing worker.

    Consumes ASRResult transcripts, runs deterministic rule matching,
    and produces machine toggle state updates.
    """

    def __init__(
        self,
        transcript_queue: Optional[asyncio.Queue] = None,
        state_queue: Optional[asyncio.Queue] = None,
        surgery: str = "PCNL",
        phase_filter: bool = True,
    ) -> None:
        """
        Initialize the rule worker.

        Args:
            transcript_queue: Input queue of ASRResult objects.
            state_queue: Output queue for RuleEngineResult / JSON patches.
            surgery: Surgery type for the rule engine.
            phase_filter: Whether to enable phase-aware filtering.
        """
        self.transcript_queue = transcript_queue or asyncio.Queue(maxsize=200)
        self.state_queue = state_queue or asyncio.Queue(maxsize=200)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._processed_count = 0
        self._matched_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0
        self.engine = RuleEngine(surgery=surgery, phase_filter=phase_filter)
        logger.info("RuleWorker initialized — surgery=%s, machines=%d", surgery, len(self.engine.machines))

    async def start(self) -> None:
        """Start the rule worker processing loop."""
        if self._running:
            logger.warning("RuleWorker already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("RuleWorker started")

    async def _process_loop(self) -> None:
        """Main processing loop — consume transcripts, produce state updates."""
        while self._running:
            try:
                # Wait for transcript with timeout for clean shutdown
                try:
                    item = await asyncio.wait_for(
                        self.transcript_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                if item is None:
                    # Poison pill — stop processing
                    logger.info("RuleWorker received stop signal")
                    break

                # Extract text from item
                text = self._extract_text(item)
                if not text or not text.strip():
                    continue

                # Run rule engine
                t0 = time.perf_counter()
                result: RuleEngineResult = self.engine.process(text)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._total_latency_ms += elapsed_ms
                self._processed_count += 1

                if result.toggles:
                    self._matched_count += 1
                    patch = result.to_json_patch()
                    await self.state_queue.put(patch)
                    toggle_desc = ", ".join(
                        f"{t.machine_name} → {t.action}"
                        for t in result.toggles
                    )
                    logger.info(
                        "\u2699\ufe0f  Rule match: \"%s\" => %s",
                        text[:60], toggle_desc,
                    )
                else:
                    logger.info(
                        "\U0001f50d No rule match for: \"%s\"",
                        text[:60],
                    )

            except Exception:
                self._error_count += 1
                logger.exception("RuleWorker error processing transcript")

    @staticmethod
    def _extract_text(item: Any) -> str:
        """
        Extract text string from various input types.

        Args:
            item: ASRResult, dict with 'text' key, or bare string.

        Returns:
            Extracted text or empty string.
        """
        if isinstance(item, str):
            return item
        if isinstance(item, ASRResult):
            return item.full_text
        if isinstance(item, dict):
            return item.get("text", item.get("full_text", ""))
        if hasattr(item, "full_text"):
            return item.full_text
        return str(item)

    def switch_surgery(self, surgery: str) -> None:
        """
        Switch the rule engine to a different surgery at runtime.

        Args:
            surgery: New surgery type name.
        """
        self.engine = RuleEngine(surgery=surgery, phase_filter=True)
        logger.info("RuleWorker switched surgery to %s", surgery)

    def set_phase(self, phase: str) -> None:
        """
        Update the current surgical phase for phase-aware filtering.

        Args:
            phase: Phase identifier (e.g., 'access', 'resection').
        """
        self.engine.set_phase(phase)
        logger.debug("RuleWorker phase set to %s", phase)

    async def stop(self) -> None:
        """Stop the rule worker gracefully."""
        self._running = False
        if self._task is not None:
            # Send poison pill
            await self.transcript_queue.put(None)
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
        logger.info(
            "RuleWorker stopped — processed=%d, matched=%d, errors=%d",
            self._processed_count,
            self._matched_count,
            self._error_count,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> Dict[str, Any]:
        avg_ms = (
            self._total_latency_ms / self._processed_count
            if self._processed_count > 0
            else 0.0
        )
        return {
            "running": self._running,
            "processed": self._processed_count,
            "matched": self._matched_count,
            "errors": self._error_count,
            "surgery": self.engine.surgery,
            "machines_loaded": len(self.engine.machines),
            "avg_latency_ms": round(avg_ms, 2),
            "transcript_queue_size": self.transcript_queue.qsize(),
            "state_queue_size": self.state_queue.qsize(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker = RuleWorker(surgery="PCNL")
    print(f"RuleWorker stats: {worker.stats}")
