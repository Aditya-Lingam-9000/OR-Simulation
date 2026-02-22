"""
OR-Symphony: State Writer Worker

Merges rule engine and MedGemma outputs, writes the canonical
current_surgery_state.json atomically, and broadcasts updates
via WebSocket.

Full implementation in Phase 7.

Usage:
    python -m src.workers.state_writer
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from src.state.serializer import StateSerializer
from src.utils.constants import TMP_DIR

logger = logging.getLogger(__name__)


class StateWriter:
    """
    Atomic state writer.

    Merges outputs from rule engine and LLM, validates against
    JSON schema, and writes the state file atomically using
    temp-file + rename pattern.
    """

    def __init__(self, output_path: Optional[Path] = None) -> None:
        self._running = False
        self._write_count = 0
        self._error_count = 0
        self.serializer = StateSerializer()
        self.output_path = output_path or (TMP_DIR / "current_surgery_state.json")

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("StateWriter initialized — output=%s", self.output_path)

    async def start(self) -> None:
        """Start the state writer."""
        self._running = True
        logger.info("StateWriter started (placeholder)")
        # TODO: Phase 7 — implement state merge + write loop

    async def stop(self) -> None:
        """Stop the state writer."""
        self._running = False
        logger.info(
            "StateWriter stopped — writes=%d, errors=%d",
            self._write_count,
            self._error_count,
        )

    def write_state(self, state: Dict[str, Any]) -> bool:
        """
        Write surgery state atomically.

        Uses temp file + os.replace() for atomic writes.

        Args:
            state: State dictionary to write.

        Returns:
            True if write succeeded.
        """
        if not self.serializer.validate(state):
            logger.error("Invalid state — refusing to write")
            self._error_count += 1
            return False

        try:
            tmp_path = self.output_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

            os.replace(str(tmp_path), str(self.output_path))
            self._write_count += 1
            logger.debug("State written atomically: %s", self.output_path)
            return True
        except Exception as e:
            logger.error("Failed to write state: %s", e)
            self._error_count += 1
            return False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        return {
            "running": self._running,
            "writes": self._write_count,
            "errors": self._error_count,
            "output_path": str(self.output_path),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    writer = StateWriter()
    print(f"StateWriter stats: {writer.stats}")

    # Test atomic write
    test_state = {
        "metadata": {"surgery": "PCNL", "phase": "Phase1"},
        "machines": {"0": [], "1": ["M01", "M09"]},
        "details": {},
        "suggestions": [],
        "confidence": 0.9,
        "source": "rule",
    }
    success = writer.write_state(test_state)
    print(f"Write test: {'PASS' if success else 'FAIL'}")
