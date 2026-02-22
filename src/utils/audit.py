"""
OR-Symphony: Audit Logging Module

Provides checksummed, append-only logging for:
  - Transcript entries
  - State change events
  - Manual override events

Each log entry includes a SHA-256 hash of the payload and a reference
to the previous entry's hash, forming a tamper-evident chain.

Usage:
    from src.utils.audit import AuditLogger, TranscriptAuditLogger
    audit = AuditLogger(Path("logs/state_changes.log"))
    audit.log({"surgery": "PCNL", "phase": "Phase2", ...})
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.constants import LOGS_DIR

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Append-only, checksummed audit logger.

    Each entry is a JSON line with:
      - timestamp: ISO 8601 UTC
      - payload: the logged data
      - sha256: hash of the payload JSON
      - prev_hash: hash of the previous entry (chain integrity)

    Args:
        log_path: Path to the log file.
        name: Logger name for identification.
    """

    def __init__(self, log_path: Path, name: str = "audit") -> None:
        self.log_path = log_path
        self.name = name
        self._prev_hash: str = "genesis"
        self._entry_count: int = 0

        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume chain from existing file
        self._resume_chain()

        logger.info(
            "AuditLogger[%s] initialized — path=%s, entries=%d",
            name, log_path, self._entry_count,
        )

    def _resume_chain(self) -> None:
        """Resume the hash chain from the last entry in the log file."""
        if not self.log_path.exists():
            return

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                last_line = ""
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
                        self._entry_count += 1

                if last_line:
                    entry = json.loads(last_line)
                    self._prev_hash = entry.get("sha256", "genesis")
        except Exception as e:
            logger.warning("AuditLogger[%s] chain resume failed: %s", self.name, e)

    @staticmethod
    def compute_hash(payload: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of a payload dictionary.

        Args:
            payload: Dictionary to hash.

        Returns:
            Hex digest of the SHA-256 hash.
        """
        payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    def log(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Append a checksummed entry to the log file.

        Args:
            payload: Data to log.

        Returns:
            The full log entry (with timestamp, sha256, prev_hash).
        """
        payload_hash = self.compute_hash(payload)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": self.name,
            "payload": payload,
            "sha256": payload_hash,
            "prev_hash": self._prev_hash,
            "seq": self._entry_count,
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._prev_hash = payload_hash
            self._entry_count += 1
        except Exception as e:
            logger.error("AuditLogger[%s] write failed: %s", self.name, e)

        return entry

    def verify_chain(self) -> bool:
        """
        Verify the integrity of the hash chain in the log file.

        Returns:
            True if the chain is intact, False if tampered.
        """
        if not self.log_path.exists():
            return True  # Empty log is valid

        prev_hash = "genesis"
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    entry = json.loads(line)

                    # Check prev_hash chain
                    if entry.get("prev_hash") != prev_hash:
                        logger.error(
                            "AuditLogger[%s] chain broken at line %d: "
                            "expected prev_hash=%s, got=%s",
                            self.name, line_no, prev_hash,
                            entry.get("prev_hash"),
                        )
                        return False

                    # Verify payload hash
                    payload = entry.get("payload", {})
                    expected_hash = self.compute_hash(payload)
                    actual_hash = entry.get("sha256", "")
                    if actual_hash != expected_hash:
                        logger.error(
                            "AuditLogger[%s] hash mismatch at line %d: "
                            "expected=%s, got=%s",
                            self.name, line_no, expected_hash, actual_hash,
                        )
                        return False

                    prev_hash = actual_hash

            return True
        except Exception as e:
            logger.error("AuditLogger[%s] verification failed: %s", self.name, e)
            return False

    def read_entries(self) -> List[Dict[str, Any]]:
        """
        Read all entries from the log file.

        Returns:
            List of log entry dicts.
        """
        entries = []
        if not self.log_path.exists():
            return entries

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    @property
    def entry_count(self) -> int:
        return self._entry_count


class TranscriptAuditLogger:
    """
    Immutable transcript logger.

    Writes transcripts to daily log files:
      logs/transcripts/YYYYMMDD.log

    Each entry is a JSON line with timestamp, speaker, text, and SHA-256.
    """

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        self.log_dir = log_dir or (LOGS_DIR / "transcripts")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._entry_count = 0

    def _get_log_path(self) -> Path:
        """Get today's transcript log file path."""
        return self.log_dir / f"{datetime.now().strftime('%Y%m%d')}.log"

    def log_transcript(
        self,
        text: str,
        speaker: str = "asr",
        confidence: float = 0.0,
        surgery: str = "",
    ) -> Dict[str, Any]:
        """
        Log a transcript entry.

        Args:
            text: Transcript text.
            speaker: Speaker identifier.
            confidence: ASR confidence score.
            surgery: Current surgery type.

        Returns:
            The log entry dict.
        """
        payload = {
            "text": text,
            "speaker": speaker,
            "confidence": confidence,
            "surgery": surgery,
        }

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "transcript",
            "payload": payload,
            "sha256": AuditLogger.compute_hash(payload),
        }

        try:
            log_path = self._get_log_path()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._entry_count += 1
        except Exception as e:
            logger.error("TranscriptAuditLogger write failed: %s", e)

        return entry

    def read_today(self) -> List[Dict[str, Any]]:
        """Read all entries from today's transcript log."""
        entries = []
        log_path = self._get_log_path()
        if not log_path.exists():
            return entries

        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    @property
    def entry_count(self) -> int:
        return self._entry_count


class StateAuditLogger(AuditLogger):
    """
    State change audit logger with SHA-256 chain.

    Logs every state update with the full state dict as payload.
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        path = log_path or (LOGS_DIR / "state_changes.log")
        super().__init__(log_path=path, name="state_changes")

    def log_state_change(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log a state change event.

        Args:
            state: Full state dict being written.

        Returns:
            The log entry dict.
        """
        # Include only essential fields to keep log manageable
        payload = {
            "surgery": state.get("metadata", {}).get("surgery", ""),
            "phase": state.get("metadata", {}).get("phase", ""),
            "source": state.get("source", ""),
            "confidence": state.get("confidence", 0.0),
            "machines_on": state.get("machines", {}).get("1", []),
            "machines_off": state.get("machines", {}).get("0", []),
            "suggestions_count": len(state.get("suggestions", [])),
        }
        return self.log(payload)


class OverrideAuditLogger(AuditLogger):
    """
    Override audit logger with SHA-256 chain.

    Logs every manual override with operator, reason, and action.
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        path = log_path or (LOGS_DIR / "overrides_audit.log")
        super().__init__(log_path=path, name="overrides")

    def log_override(
        self,
        machine_id: str,
        action: str,
        reason: str,
        operator: str = "unknown",
        surgery: str = "",
    ) -> Dict[str, Any]:
        """
        Log an override event.

        Args:
            machine_id: Machine identifier.
            action: ON, OFF, or STANDBY.
            reason: Human-provided reason.
            operator: Who performed the override.
            surgery: Current surgery type.

        Returns:
            The log entry dict.
        """
        payload = {
            "machine_id": machine_id,
            "action": action,
            "reason": reason,
            "operator": operator,
            "surgery": surgery,
        }
        return self.log(payload)
