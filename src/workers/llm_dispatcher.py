"""
OR-Symphony: LLM Dispatcher Worker

Batches transcript context requests and dispatches them to the
MedGemma GGUF model for structured reasoning.

Full implementation in Phase 6/7.

Usage:
    python -m src.workers.llm_dispatcher
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from src.llm.manager import LLMManager

logger = logging.getLogger(__name__)


class LLMDispatcher:
    """
    LLM dispatch worker.

    Manages micro-batching of inference requests and dispatches
    them to the MedGemma model.
    """

    def __init__(self) -> None:
        self._running = False
        self._processed_count = 0
        self._error_count = 0
        self._fallback_mode = False
        self.manager = LLMManager()
        logger.info("LLMDispatcher initialized")

    async def start(self) -> None:
        """Start the LLM dispatcher."""
        self._running = True
        await self.manager.start()
        logger.info("LLMDispatcher started (placeholder)")
        # TODO: Phase 6/7 — implement batch processing loop

    async def stop(self) -> None:
        """Stop the LLM dispatcher."""
        self._running = False
        await self.manager.stop()
        logger.info(
            "LLMDispatcher stopped — processed=%d, errors=%d, fallback=%s",
            self._processed_count,
            self._error_count,
            self._fallback_mode,
        )

    def enter_fallback_mode(self) -> None:
        """Switch to fallback (rule-only) mode when LLM is unavailable."""
        self._fallback_mode = True
        logger.warning("LLMDispatcher entering fallback mode — rule engine only")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_fallback(self) -> bool:
        return self._fallback_mode

    @property
    def stats(self) -> dict:
        return {
            "running": self._running,
            "processed": self._processed_count,
            "errors": self._error_count,
            "fallback_mode": self._fallback_mode,
            "queue_size": self.manager.queue_size,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dispatcher = LLMDispatcher()
    print(f"LLMDispatcher stats: {dispatcher.stats}")
