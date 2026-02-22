"""
OR-Symphony: Pipeline Orchestrator

Central coordinator that creates and manages all workers:
  ASR Worker → Rule Worker → LLM Dispatcher → State Writer

Creates shared asyncio queues, starts/stops all workers in order,
provides runtime surgery switching and aggregated stats.

Usage:
    from src.workers.orchestrator import Orchestrator
    orch = Orchestrator(surgery="PCNL", llm_stub=True)
    await orch.start()
    # ... application runs ...
    await orch.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional

from src.workers.asr_worker import ASRWorker
from src.workers.llm_dispatcher import LLMDispatcher
from src.workers.rule_worker import RuleWorker
from src.workers.state_writer import StateWriter

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Pipeline orchestrator — manages the full inference pipeline.

    Pipeline:
        audio_queue → ASRWorker → transcript_queue
                                        ├→ RuleWorker → rule_state_queue ─┐
                                        └→ LLMDispatcher → llm_state_queue ─┤
                                                                             └→ StateWriter → JSON + broadcast

    Args:
        surgery: Surgery type (PCNL, Partial Hepatectomy, Lobectomy).
        llm_stub: If True, LLM runs in stub mode (no model loading).
        llm_dispatch_interval_s: Seconds between LLM dispatch calls.
        on_state_update: Async callback fired on state changes.
        asr_runner: Optional pre-configured ASR runner (for testing).
    """

    def __init__(
        self,
        surgery: str = "PCNL",
        llm_stub: bool = False,
        llm_dispatch_interval_s: float = 2.0,
        on_state_update: Optional[Callable[[Dict[str, Any]], Coroutine]] = None,
        asr_runner: Any = None,
    ) -> None:
        self.surgery = surgery
        self.llm_stub = llm_stub

        # ── Shared queues ─────────────────────────────────────────────
        self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.transcript_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.rule_state_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.llm_state_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

        # ── Workers ──────────────────────────────────────────────────
        self.asr_worker = ASRWorker(
            audio_queue=self.audio_queue,
            transcript_queue=self.transcript_queue,
            runner=asr_runner,
        )

        self.rule_worker = RuleWorker(
            transcript_queue=asyncio.Queue(maxsize=200),  # own queue; fed by fanout
            state_queue=self.rule_state_queue,
            surgery=surgery,
        )

        self.llm_dispatcher = LLMDispatcher(
            transcript_queue=asyncio.Queue(maxsize=200),  # separate copy
            state_queue=self.llm_state_queue,
            surgery=surgery,
            dispatch_interval_s=llm_dispatch_interval_s,
            stub_mode=llm_stub,
        )

        self.state_writer = StateWriter(
            rule_state_queue=self.rule_state_queue,
            llm_state_queue=self.llm_state_queue,
            on_update=on_state_update,
        )

        # State
        self._running = False
        self._start_time: Optional[float] = None

        # Transcript fan-out task (duplicates transcripts to LLM dispatcher)
        self._fanout_task: Optional[asyncio.Task] = None

        logger.info(
            "Orchestrator initialized — surgery=%s, llm_stub=%s",
            surgery, llm_stub,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all workers in dependency order."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        self._start_time = time.time()
        logger.info("Orchestrator starting all workers...")

        # Start state writer first (consumer)
        await self.state_writer.start()

        # Start rule worker and LLM dispatcher (intermediate)
        await self.rule_worker.start()
        await self.llm_dispatcher.start()

        # Start transcript fan-out (copies transcripts to LLM dispatcher)
        self._fanout_task = asyncio.create_task(self._transcript_fanout())

        # Start ASR worker last (producer)
        # Note: ASR worker is only started if a runner is available
        # For API-driven usage without mic, skip ASR start
        try:
            await self.asr_worker.start()
        except Exception as e:
            logger.warning("ASR worker start skipped: %s", e)

        self._running = True
        logger.info("Orchestrator started — all workers running")

    async def stop(self) -> None:
        """Stop all workers in reverse order."""
        if not self._running:
            return

        self._running = False
        logger.info("Orchestrator stopping all workers...")

        # Stop ASR first (producer)
        try:
            await self.asr_worker.stop()
        except Exception as e:
            logger.warning("ASR worker stop error: %s", e)

        # Stop fan-out
        if self._fanout_task is not None:
            self._fanout_task.cancel()
            try:
                await self._fanout_task
            except asyncio.CancelledError:
                pass
            self._fanout_task = None

        # Stop rule worker and LLM dispatcher
        await self.rule_worker.stop()
        await self.llm_dispatcher.stop()

        # Stop state writer last (consumer)
        await self.state_writer.stop()

        elapsed = time.time() - (self._start_time or time.time())
        logger.info("Orchestrator stopped — ran for %.1fs", elapsed)

    # ------------------------------------------------------------------
    # Transcript fan-out
    # ------------------------------------------------------------------

    async def _transcript_fanout(self) -> None:
        """
        Fan out transcripts from the shared transcript queue to both
        the Rule Worker and LLM Dispatcher.

        The ASR Worker writes ASRResult objects into self.transcript_queue.
        This task reads from that queue and copies each item to both
        the Rule Worker's queue and the LLM Dispatcher's queue so
        neither worker steals items from the other.

        The Rule Worker's transcript_queue was originally self.transcript_queue,
        but now we give it its own internal queue (set up in __init__),
        and this fanout copies to both.
        """
        logger.info("Transcript fanout started")
        while self._running:
            try:
                try:
                    item = await asyncio.wait_for(
                        self.transcript_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                if item is None:
                    break

                # Copy to rule worker
                try:
                    self.rule_worker.transcript_queue.put_nowait(item)
                except asyncio.QueueFull:
                    logger.warning("Rule worker queue full — dropping transcript")

                # Copy to LLM dispatcher
                try:
                    self.llm_dispatcher.transcript_queue.put_nowait(item)
                except asyncio.QueueFull:
                    logger.warning("LLM dispatcher queue full — dropping transcript")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Fanout error")
        logger.info("Transcript fanout stopped")

    # ------------------------------------------------------------------
    # External transcript feed
    # ------------------------------------------------------------------

    async def feed_audio(self, audio_chunk: Any) -> None:
        """
        Feed an audio chunk into the pipeline.

        Args:
            audio_chunk: Audio data (numpy array or AudioChunk).
        """
        if not self._running:
            return
        await self.audio_queue.put(audio_chunk)

    async def feed_transcript(self, text: str, speaker: str = "asr") -> None:
        """
        Feed a transcript directly into both rule and LLM pipelines.

        This is the main entry point for text-based pipeline usage
        (e.g., from the API or tests, bypassing ASR).

        The text is enqueued to the shared transcript_queue, where the
        fanout task distributes it to both rule worker and LLM dispatcher.

        Args:
            text: Transcript text.
            speaker: Speaker label.
        """
        if not self._running:
            return

        # Push into shared queue — fanout distributes to both workers
        await self.transcript_queue.put(text)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current merged surgery state."""
        return self.state_writer.current_state

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def apply_override(
        self,
        machine_id: str,
        action: str,
        reason: str = "Manual override",
        operator: str = "unknown",
    ) -> None:
        """
        Apply a manual override for a machine state.

        Args:
            machine_id: Machine identifier.
            action: ON, OFF, or STANDBY.
            reason: Reason for the override.
            operator: Operator name/ID.
        """
        self.state_writer.apply_override(
            machine_id=machine_id,
            action=action,
            reason=reason,
            operator=operator,
        )

    # ------------------------------------------------------------------
    # Surgery switching
    # ------------------------------------------------------------------

    def switch_surgery(self, surgery: str) -> None:
        """
        Switch the surgery type across all workers.

        Args:
            surgery: New surgery type name.
        """
        self.surgery = surgery
        self.rule_worker.switch_surgery(surgery)
        self.llm_dispatcher.switch_surgery(surgery)
        logger.info("Orchestrator switched surgery to %s", surgery)

    def set_phase(self, phase: str) -> None:
        """
        Update the current surgical phase across all workers.

        Args:
            phase: New phase identifier.
        """
        self.rule_worker.set_phase(phase)
        self.llm_dispatcher.set_phase(phase)
        logger.debug("Orchestrator phase set to %s", phase)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def uptime_s(self) -> float:
        """Uptime in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def stats(self) -> Dict[str, Any]:
        """Aggregated stats from all workers."""
        return {
            "running": self._running,
            "surgery": self.surgery,
            "uptime_s": round(self.uptime_s, 1),
            "llm_stub": self.llm_stub,
            "asr": self.asr_worker.stats,
            "rule": self.rule_worker.stats,
            "llm": self.llm_dispatcher.stats,
            "state_writer": self.state_writer.stats,
            "queue_sizes": {
                "audio": self.audio_queue.qsize(),
                "transcript": self.transcript_queue.qsize(),
                "rule_state": self.rule_state_queue.qsize(),
                "llm_state": self.llm_state_queue.qsize(),
            },
        }
