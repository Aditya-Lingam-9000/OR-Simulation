"""
OR-Symphony: LLM Dispatcher Worker

Polls transcripts from a shared queue, accumulates context in a
RollingBuffer, and periodically submits requests to the MedGemma
LLM for structured reasoning.

Rate-limited: submits to LLM every `dispatch_interval_s` seconds
(default 2s) to avoid flooding the single-threaded GGUF model.

Usage:
    dispatcher = LLMDispatcher(transcript_queue, state_queue, surgery="PCNL")
    await dispatcher.start()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from src.asr.runner import ASRResult
from src.llm.manager import LLMManager, LLMRequest
from src.state.rolling_buffer import RollingBuffer
from src.utils.constants import ROLLING_BUFFER_DURATION_S

logger = logging.getLogger(__name__)


class LLMDispatcher:
    """
    LLM dispatch worker.

    Accumulates transcript context and periodically dispatches
    inference requests to the LLMManager. Rate-limited to avoid
    overwhelming the single-threaded GGUF model.

    Args:
        transcript_queue: Input queue of transcript items (ASRResult/str/dict).
        state_queue: Output queue for LLM state results (dict).
        surgery: Surgery type for prompt context.
        dispatch_interval_s: Minimum seconds between LLM dispatches.
        stub_mode: If True, LLMManager runs in stub mode.
        manager: Optional pre-configured LLMManager (for testing).
    """

    def __init__(
        self,
        transcript_queue: Optional[asyncio.Queue] = None,
        state_queue: Optional[asyncio.Queue] = None,
        surgery: str = "PCNL",
        dispatch_interval_s: float = 2.0,
        stub_mode: bool = False,
        manager: Optional[LLMManager] = None,
    ) -> None:
        self.transcript_queue = transcript_queue or asyncio.Queue(maxsize=200)
        self.state_queue = state_queue or asyncio.Queue(maxsize=200)
        self.surgery = surgery
        self.dispatch_interval_s = dispatch_interval_s

        self._buffer = RollingBuffer(max_duration_s=ROLLING_BUFFER_DURATION_S)
        self._manager = manager or LLMManager(
            surgery=surgery, stub_mode=stub_mode
        )

        self._running = False
        self._fallback_mode = False
        self._task: Optional[asyncio.Task] = None
        self._current_phase = "Phase1"
        self._session_start = time.time()

        # Stats
        self._transcripts_received = 0
        self._dispatches = 0
        self._dispatch_errors = 0
        self._last_dispatch_time = 0.0

        logger.info(
            "LLMDispatcher initialized — surgery=%s, interval=%.1fs, stub=%s",
            surgery, dispatch_interval_s, stub_mode,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the LLM dispatcher."""
        if self._running:
            logger.warning("LLMDispatcher already running")
            return

        await self._manager.start()

        # Check if model loaded — enter fallback if not
        if not self._manager.is_model_loaded and not self._manager.stub_mode:
            self.enter_fallback_mode()

        self._running = True
        self._session_start = time.time()
        self._task = asyncio.create_task(self._process_loop())
        logger.info("LLMDispatcher started (fallback=%s)", self._fallback_mode)

    async def stop(self) -> None:
        """Stop the LLM dispatcher gracefully."""
        self._running = False
        if self._task is not None:
            await self.transcript_queue.put(None)  # poison pill
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None

        await self._manager.stop()
        logger.info(
            "LLMDispatcher stopped — transcripts=%d, dispatches=%d, errors=%d",
            self._transcripts_received,
            self._dispatches,
            self._dispatch_errors,
        )

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    async def _process_loop(self) -> None:
        """Main loop: consume transcripts, accumulate, periodically dispatch."""
        while self._running:
            try:
                # Wait for transcript
                try:
                    item = await asyncio.wait_for(
                        self.transcript_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    # Check if we should dispatch based on time
                    await self._maybe_dispatch()
                    continue

                if item is None:
                    break

                # Add transcript to buffer
                text = self._extract_text(item)
                if text and text.strip():
                    elapsed_s = time.time() - self._session_start
                    self._buffer.append(
                        text=text,
                        timestamp=elapsed_s,
                        speaker="asr",
                        is_final=True,
                    )
                    self._transcripts_received += 1

                # Try to dispatch
                await self._maybe_dispatch()

            except asyncio.CancelledError:
                break
            except Exception:
                self._dispatch_errors += 1
                logger.exception("LLMDispatcher error in process loop")

    async def _maybe_dispatch(self) -> None:
        """Dispatch to LLM if enough time has elapsed since last dispatch."""
        if self._fallback_mode:
            return

        now = time.time()
        if now - self._last_dispatch_time < self.dispatch_interval_s:
            return

        # Only dispatch if we have content
        context = self._buffer.get_context_for_llm(
            surgery=self.surgery,
            phase=self._current_phase,
            max_entries=20,
        )
        if not context or not context.strip():
            return

        self._last_dispatch_time = now

        try:
            session_elapsed = now - self._session_start
            request = LLMRequest(
                request_id=f"llm_{self._dispatches}",
                transcript_context=context,
                surgery_type=self.surgery,
                machines_dict={},
                current_phase=self._current_phase,
                session_time_s=session_elapsed,
            )

            response = await self._manager.submit(request)
            self._dispatches += 1

            if response.success:
                await self.state_queue.put(response.output)
                logger.debug(
                    "LLM dispatch #%d: %.0fms, confidence=%.2f",
                    self._dispatches,
                    response.processing_time_ms,
                    response.output.get("confidence", 0),
                )
            else:
                self._dispatch_errors += 1
                if response.degraded:
                    await self.state_queue.put(response.output)
                logger.warning(
                    "LLM dispatch #%d failed: %s",
                    self._dispatches,
                    response.error,
                )

        except Exception as e:
            self._dispatch_errors += 1
            logger.error("LLM dispatch error: %s", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(item: Any) -> str:
        """Extract text from various input types."""
        if isinstance(item, str):
            return item
        if isinstance(item, ASRResult):
            return item.full_text
        if isinstance(item, dict):
            return item.get("text", item.get("full_text", ""))
        if hasattr(item, "full_text"):
            return item.full_text
        return str(item)

    def set_phase(self, phase: str) -> None:
        """Update the current surgical phase."""
        self._current_phase = phase
        logger.debug("LLMDispatcher phase set to %s", phase)

    def switch_surgery(self, surgery: str) -> None:
        """Switch the surgery type at runtime."""
        self.surgery = surgery
        self._manager.set_surgery(surgery)
        logger.info("LLMDispatcher switched surgery to %s", surgery)

    def enter_fallback_mode(self) -> None:
        """Switch to fallback (rule-only) mode when LLM is unavailable."""
        self._fallback_mode = True
        logger.warning("LLMDispatcher entering fallback mode — rule engine only")

    def exit_fallback_mode(self) -> None:
        """Exit fallback mode (re-enable LLM dispatching)."""
        self._fallback_mode = False
        logger.info("LLMDispatcher exiting fallback mode")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_fallback(self) -> bool:
        return self._fallback_mode

    @property
    def buffer(self) -> RollingBuffer:
        """Access the rolling buffer (for orchestrator integration)."""
        return self._buffer

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "transcripts_received": self._transcripts_received,
            "dispatches": self._dispatches,
            "dispatch_errors": self._dispatch_errors,
            "fallback_mode": self._fallback_mode,
            "buffer_entries": self._buffer.entry_count,
            "current_phase": self._current_phase,
            "surgery": self.surgery,
            "manager": self._manager.stats,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dispatcher = LLMDispatcher(surgery="PCNL", stub_mode=True)
    print(f"LLMDispatcher stats: {dispatcher.stats}")
