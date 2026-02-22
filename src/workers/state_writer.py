"""
OR-Symphony: State Writer Worker

Consumes state updates from rule engine and LLM dispatcher queues,
merges them via StateSerializer, writes the canonical JSON atomically,
and fires a broadcast callback for WebSocket push.

Usage:
    writer = StateWriter(rule_state_queue, llm_state_queue, on_update=ws_broadcast)
    await writer.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from src.state.serializer import StateSerializer
from src.utils.audit import OverrideAuditLogger, StateAuditLogger
from src.utils.constants import LOGS_DIR, TMP_DIR
from src.utils.safety import SafetyValidator

logger = logging.getLogger(__name__)


class StateWriter:
    """
    Atomic state writer with merge and broadcast.

    Merges outputs from rule engine and LLM dispatcher, validates against
    JSON schema, writes the state file atomically, and fires a callback
    for WebSocket broadcasting.

    Args:
        rule_state_queue: Input queue for rule engine state patches.
        llm_state_queue: Input queue for LLM state results.
        output_path: Path to write the canonical state JSON.
        on_update: Async callback fired after each state write.
        override_log_path: Path for override audit log.
    """

    def __init__(
        self,
        rule_state_queue: Optional[asyncio.Queue] = None,
        llm_state_queue: Optional[asyncio.Queue] = None,
        output_path: Optional[Path] = None,
        on_update: Optional[Callable[[Dict[str, Any]], Coroutine]] = None,
        override_log_path: Optional[Path] = None,
    ) -> None:
        self.rule_state_queue = rule_state_queue or asyncio.Queue(maxsize=200)
        self.llm_state_queue = llm_state_queue or asyncio.Queue(maxsize=200)
        self.output_path = output_path or (TMP_DIR / "current_surgery_state.json")
        self._on_update = on_update
        self._override_log_path = override_log_path or (LOGS_DIR / "overrides.log")

        self.serializer = StateSerializer()

        # Latest state components
        self._latest_rule_state: Optional[Dict[str, Any]] = None
        self._latest_llm_state: Optional[Dict[str, Any]] = None
        self._current_state: Dict[str, Any] = self._default_state()
        self._overrides: List[Dict[str, Any]] = []

        # Accumulated machine states (persists across rule patches)
        self._accumulated_machines: Dict[str, str] = {}  # machine_id → "ON"/"OFF"

        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Stats
        self._write_count = 0
        self._merge_count = 0
        self._error_count = 0
        self._override_count = 0

        # Audit loggers (SHA-256 checksummed)
        self._state_audit = StateAuditLogger(
            log_path=self.output_path.parent / "state_audit.log"
        )
        self._override_audit = OverrideAuditLogger(
            log_path=self._override_log_path.parent / "overrides_audit.log"
        )

        # Safety validator
        self._safety = SafetyValidator()

        # Ensure output directories exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._override_log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("StateWriter initialized — output=%s", self.output_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the state writer processing loop."""
        if self._running:
            logger.warning("StateWriter already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("StateWriter started")

    async def stop(self) -> None:
        """Stop the state writer gracefully."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info(
            "StateWriter stopped — writes=%d, merges=%d, errors=%d, overrides=%d",
            self._write_count, self._merge_count, self._error_count, self._override_count,
        )

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    async def _process_loop(self) -> None:
        """Main loop: consume rule + LLM updates, merge, write, broadcast."""
        while self._running:
            try:
                updated = False

                # Drain rule state queue (non-blocking)
                while not self.rule_state_queue.empty():
                    try:
                        rule_patch = self.rule_state_queue.get_nowait()
                        if rule_patch is not None:
                            self._latest_rule_state = rule_patch
                            updated = True
                    except asyncio.QueueEmpty:
                        break

                # Drain LLM state queue (non-blocking)
                while not self.llm_state_queue.empty():
                    try:
                        llm_result = self.llm_state_queue.get_nowait()
                        if llm_result is not None:
                            self._latest_llm_state = llm_result
                            updated = True
                    except asyncio.QueueEmpty:
                        break

                # Merge and write if updated
                if updated:
                    await self._merge_and_write()

                # Small sleep to avoid busy-waiting
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception:
                self._error_count += 1
                logger.exception("StateWriter error in process loop")
                await asyncio.sleep(0.5)

    def _accumulate_machines(self, patch: Dict[str, Any]) -> None:
        """Accumulate machine ON/OFF states from a rule patch."""
        machines = patch.get("machines", {})
        for mid in machines.get("1", []):
            self._accumulated_machines[mid] = "ON"
        for mid in machines.get("0", []):
            self._accumulated_machines[mid] = "OFF"

    def _build_accumulated_machines(self) -> Dict[str, list]:
        """Build the machines dict from accumulated state."""
        on: list = []
        off: list = []
        for mid, action in sorted(self._accumulated_machines.items()):
            if action == "ON":
                on.append(mid)
            else:
                off.append(mid)
        return {"0": off, "1": on}

    async def _merge_and_write(self) -> None:
        """Merge rule + LLM outputs, apply overrides, write atomically, broadcast."""
        try:
            # Accumulate machine states from rule patch
            if self._latest_rule_state is not None:
                self._accumulate_machines(self._latest_rule_state)

            # Merge via serializer
            if self._latest_rule_state is not None or self._latest_llm_state is not None:
                merged = self.serializer.merge(
                    rule_output=self._latest_rule_state or self._default_state(),
                    llm_output=self._latest_llm_state,
                )
                # Replace machines with the full accumulated state
                merged["machines"] = self._build_accumulated_machines()
                self._merge_count += 1
            else:
                merged = self._current_state

            # Apply pending overrides
            merged = self._apply_overrides(merged)

            # Ensure timestamp
            merged.setdefault("metadata", {})
            merged["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()

            self._current_state = merged

            # Safety validation
            safety_result = self._safety.validate_output(merged)
            if not safety_result.is_safe:
                logger.error(
                    "Safety validation FAILED — %d violations, skipping broadcast",
                    len(safety_result.violations),
                )
                self._error_count += 1
                # Still write for debugging, but mark as unsafe
                merged.setdefault("metadata", {})
                merged["metadata"]["safety_violations"] = safety_result.violations

            # Write atomically
            self.write_state(merged)

            # Audit log (checksummed)
            try:
                self._state_audit.log_state_change(merged)
            except Exception as e:
                logger.error("State audit log failed: %s", e)

            # Fire broadcast callback (only if safe)
            if self._on_update is not None and safety_result.is_safe:
                on_list = merged.get("machines", {}).get("1", [])
                off_list = merged.get("machines", {}).get("0", [])
                logger.info(
                    "\U0001f4e1 State broadcast — ON: %s, OFF: %s",
                    on_list or "(none)", off_list or "(none)",
                )
                try:
                    await self._on_update(merged)
                except Exception as e:
                    logger.error("Broadcast callback error: %s", e)

        except Exception:
            self._error_count += 1
            logger.exception("Merge and write failed")

    # ------------------------------------------------------------------
    # Atomic write
    # ------------------------------------------------------------------

    def write_state(self, state: Dict[str, Any]) -> bool:
        """
        Write surgery state atomically using temp file + os.replace().

        Args:
            state: State dictionary to write.

        Returns:
            True if write succeeded.
        """
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

    def read_current_state(self) -> Dict[str, Any]:
        """Read the current in-memory state."""
        return dict(self._current_state)

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
        Queue a manual override for a machine state.

        Args:
            machine_id: Machine identifier (e.g., M01).
            action: New action (ON, OFF, STANDBY).
            reason: Reason for the override.
            operator: Operator identifier.
        """
        override = {
            "machine_id": machine_id,
            "action": action,
            "reason": reason,
            "operator": operator,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._overrides.append(override)
        self._override_count += 1

        # Log to audit file (legacy plain JSON)
        self._log_override(override)

        # Log to checksummed audit (SHA-256 chain)
        try:
            self._override_audit.log_override(
                machine_id=machine_id,
                action=action,
                reason=reason,
                operator=operator,
            )
        except Exception as e:
            logger.error("Override audit log failed: %s", e)

        logger.info(
            "Override queued: %s → %s (reason: %s)",
            machine_id, action, reason,
        )

    def _apply_overrides(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply pending overrides to the state and clear the queue."""
        if not self._overrides:
            return state

        state = dict(state)
        machines = dict(state.get("machines", {"0": [], "1": []}))
        machines["0"] = list(machines.get("0", []))
        machines["1"] = list(machines.get("1", []))

        for override in self._overrides:
            mid = override["machine_id"]
            action = override["action"].upper()

            # Remove from both lists
            if mid in machines["0"]:
                machines["0"].remove(mid)
            if mid in machines["1"]:
                machines["1"].remove(mid)

            # Add to appropriate list
            if action == "ON":
                machines["1"].append(mid)
            else:  # OFF or STANDBY
                machines["0"].append(mid)

        state["machines"] = machines

        # Record overrides in metadata
        state.setdefault("metadata", {})
        state["metadata"]["overrides_applied"] = len(self._overrides)

        self._overrides.clear()
        return state

    def _log_override(self, override: Dict[str, Any]) -> None:
        """Append override to the audit log file."""
        try:
            with open(self._override_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(override, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Failed to log override: %s", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_state() -> Dict[str, Any]:
        """Return a default empty state."""
        return {
            "metadata": {
                "surgery": "",
                "phase": "Phase1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reasoning": "normal",
            },
            "machines": {"0": [], "1": []},
            "details": {},
            "suggestions": [],
            "confidence": 0.0,
            "source": "rule",
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_state(self) -> Dict[str, Any]:
        return dict(self._current_state)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "writes": self._write_count,
            "merges": self._merge_count,
            "errors": self._error_count,
            "overrides": self._override_count,
            "output_path": str(self.output_path),
            "rule_queue_size": self.rule_state_queue.qsize(),
            "llm_queue_size": self.llm_state_queue.qsize(),
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
