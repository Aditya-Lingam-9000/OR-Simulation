"""
OR-Symphony: Rolling Transcript Buffer

Maintains a sliding window of recent transcript segments with timestamps.
Used to provide context to MedGemma for surgical phase reasoning.

Features:
  - Time-based windowing (default 180s)
  - Structured entries with speaker labels and source tracking
  - LLM context formatting for MedGemma prompts
  - Incremental queries (get entries since timestamp)
  - Serialization (to_dict / from_dict) for persistence
  - Summary statistics

Default window: 180 seconds (configurable).

Usage:
    from src.state.rolling_buffer import RollingBuffer
    buffer = RollingBuffer(max_duration_s=180)
    buffer.append("intubation complete", timestamp=42.5, speaker="surgeon")
    context = buffer.get_context_for_llm(surgery="PCNL", phase="Phase3")
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from src.utils.constants import ROLLING_BUFFER_DURATION_S

logger = logging.getLogger(__name__)


@dataclass
class BufferEntry:
    """A single transcript entry in the rolling buffer."""

    text: str
    timestamp: float  # seconds since session start
    speaker: str = "unknown"
    is_final: bool = True
    source: str = "asr"  # asr | override | system | rule

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BufferEntry":
        """Deserialize entry from dict."""
        return cls(
            text=data.get("text", ""),
            timestamp=data.get("timestamp", 0.0),
            speaker=data.get("speaker", "unknown"),
            is_final=data.get("is_final", True),
            source=data.get("source", "asr"),
        )

    def format_line(self, include_timestamp: bool = True) -> str:
        """Format entry as a readable line for LLM context.

        Args:
            include_timestamp: Whether to include the timestamp prefix.

        Returns:
            Formatted string like "[42.5s] surgeon: intubation complete"
        """
        parts = []
        if include_timestamp:
            parts.append(f"[{self.timestamp:.1f}s]")
        if self.speaker and self.speaker != "unknown":
            parts.append(f"{self.speaker}:")
        parts.append(self.text)
        return " ".join(parts)


class RollingBuffer:
    """
    Rolling transcript buffer with time-based windowing.

    Automatically evicts entries older than max_duration_s.
    Thread-safe for single-writer, multiple-reader pattern.
    """

    def __init__(self, max_duration_s: float = ROLLING_BUFFER_DURATION_S) -> None:
        """
        Initialize the rolling buffer.

        Args:
            max_duration_s: Maximum duration of transcript history to keep (seconds).
        """
        self.max_duration_s = max_duration_s
        self._entries: Deque[BufferEntry] = deque()
        self._session_start: float = time.time()
        self._total_appended: int = 0
        self._total_evicted: int = 0

        logger.info("RollingBuffer initialized — max_duration=%.0fs", max_duration_s)

    def append(
        self,
        text: str,
        timestamp: Optional[float] = None,
        speaker: str = "unknown",
        is_final: bool = True,
        source: str = "asr",
    ) -> None:
        """
        Append a transcript segment to the buffer.

        Args:
            text: Transcript text.
            timestamp: Seconds since session start. Auto-calculated if None.
            speaker: Speaker label (if ASR supports diarization).
            is_final: Whether this is a final (vs partial) transcript.
            source: Source of the transcript (asr, override, system, rule).
        """
        if timestamp is None:
            timestamp = time.time() - self._session_start

        entry = BufferEntry(
            text=text,
            timestamp=timestamp,
            speaker=speaker,
            is_final=is_final,
            source=source,
        )
        self._entries.append(entry)
        self._total_appended += 1
        self._evict_old()

        logger.debug("Buffer append: t=%.1fs, text='%s'", timestamp, text[:50])

    def _evict_old(self) -> None:
        """Remove entries older than max_duration_s from the latest entry."""
        if not self._entries:
            return

        latest = self._entries[-1].timestamp
        cutoff = latest - self.max_duration_s

        while self._entries and self._entries[0].timestamp < cutoff:
            self._entries.popleft()
            self._total_evicted += 1

    def get_context(self, max_entries: Optional[int] = None) -> str:
        """
        Get the full context string from the buffer.

        Args:
            max_entries: Maximum number of recent entries to include.

        Returns:
            Concatenated transcript text.
        """
        entries = list(self._entries)
        if max_entries is not None:
            entries = entries[-max_entries:]

        return " ".join(e.text for e in entries if e.text.strip())

    def get_context_for_llm(
        self,
        surgery: str = "",
        phase: str = "",
        max_entries: Optional[int] = None,
        include_timestamps: bool = True,
    ) -> str:
        """
        Get formatted context string for MedGemma LLM prompts.

        Produces a structured block suitable for insertion into LLM prompt templates.

        Args:
            surgery: Current surgery type for context header.
            phase: Current surgical phase for context header.
            max_entries: Maximum number of recent entries to include.
            include_timestamps: Whether to include timestamps in output.

        Returns:
            Formatted multi-line string with header and transcript lines.
        """
        entries = list(self._entries)
        if max_entries is not None:
            entries = entries[-max_entries:]

        # Filter to final transcripts only
        final_entries = [e for e in entries if e.is_final and e.text.strip()]

        if not final_entries:
            return "<no recent transcript>"

        lines: List[str] = []

        # Header with context
        header_parts = ["--- Recent OR Transcript ---"]
        if surgery:
            header_parts.append(f"Surgery: {surgery}")
        if phase:
            header_parts.append(f"Phase: {phase}")
        header_parts.append(
            f"Window: {self.duration_s:.0f}s ({len(final_entries)} segments)"
        )
        lines.append(" | ".join(header_parts))
        lines.append("")

        # Transcript lines
        for entry in final_entries:
            lines.append(entry.format_line(include_timestamp=include_timestamps))

        lines.append("")
        lines.append("--- End Transcript ---")

        return "\n".join(lines)

    def get_entries(self) -> List[BufferEntry]:
        """Get all current buffer entries as a list."""
        return list(self._entries)

    def get_entries_since(self, since_timestamp: float) -> List[BufferEntry]:
        """
        Get entries with timestamp >= since_timestamp.

        Args:
            since_timestamp: Minimum timestamp (inclusive) to filter by.

        Returns:
            List of matching BufferEntry objects.
        """
        return [e for e in self._entries if e.timestamp >= since_timestamp]

    def get_final_entries(self) -> List[BufferEntry]:
        """Get only final (non-partial) entries."""
        return [e for e in self._entries if e.is_final]

    def get_entries_by_speaker(self, speaker: str) -> List[BufferEntry]:
        """
        Get entries from a specific speaker.

        Args:
            speaker: Speaker label to filter by (case-insensitive).

        Returns:
            List of matching BufferEntry objects.
        """
        speaker_lower = speaker.lower()
        return [e for e in self._entries if e.speaker.lower() == speaker_lower]

    @property
    def duration_s(self) -> float:
        """Current duration span of buffered entries in seconds."""
        if len(self._entries) < 2:
            return 0.0
        return self._entries[-1].timestamp - self._entries[0].timestamp

    @property
    def entry_count(self) -> int:
        """Number of entries currently in buffer."""
        return len(self._entries)

    @property
    def earliest_timestamp(self) -> Optional[float]:
        """Timestamp of the earliest entry."""
        return self._entries[0].timestamp if self._entries else None

    @property
    def latest_timestamp(self) -> Optional[float]:
        """Timestamp of the latest entry."""
        return self._entries[-1].timestamp if self._entries else None

    @property
    def session_elapsed_s(self) -> float:
        """Total seconds elapsed since session start."""
        return time.time() - self._session_start

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary dict of the buffer state.

        Returns:
            Dict with count, duration, timestamps, and stats.
        """
        speakers = set()
        sources = set()
        for e in self._entries:
            speakers.add(e.speaker)
            sources.add(e.source)

        return {
            "entry_count": self.entry_count,
            "duration_s": round(self.duration_s, 1),
            "earliest_timestamp": self.earliest_timestamp,
            "latest_timestamp": self.latest_timestamp,
            "session_elapsed_s": round(self.session_elapsed_s, 1),
            "max_duration_s": self.max_duration_s,
            "total_appended": self._total_appended,
            "total_evicted": self._total_evicted,
            "speakers": sorted(speakers),
            "sources": sorted(sources),
        }

    def clear(self) -> None:
        """Clear all buffer entries."""
        self._entries.clear()
        self._session_start = time.time()

    def reset_session(self) -> None:
        """Reset the session timer and clear buffer."""
        self.clear()
        self._total_appended = 0
        self._total_evicted = 0
        logger.info("RollingBuffer session reset")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the entire buffer to a dict for persistence.

        Returns:
            Dict containing all buffer state.
        """
        return {
            "max_duration_s": self.max_duration_s,
            "session_start": self._session_start,
            "total_appended": self._total_appended,
            "total_evicted": self._total_evicted,
            "entries": [e.to_dict() for e in self._entries],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RollingBuffer":
        """
        Deserialize a buffer from a dict.

        Args:
            data: Dict produced by to_dict().

        Returns:
            Restored RollingBuffer instance.
        """
        buf = cls(max_duration_s=data.get("max_duration_s", ROLLING_BUFFER_DURATION_S))
        buf._session_start = data.get("session_start", time.time())
        buf._total_appended = data.get("total_appended", 0)
        buf._total_evicted = data.get("total_evicted", 0)
        for entry_data in data.get("entries", []):
            buf._entries.append(BufferEntry.from_dict(entry_data))
        return buf

    def save(self, path: Path) -> None:
        """
        Save buffer state to a JSON file.

        Args:
            path: File path to save to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Buffer saved to %s (%d entries)", path, self.entry_count)

    @classmethod
    def load(cls, path: Path) -> "RollingBuffer":
        """
        Load buffer state from a JSON file.

        Args:
            path: File path to load from.

        Returns:
            Restored RollingBuffer instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        buf = cls.from_dict(data)
        logger.info("Buffer loaded from %s (%d entries)", path, buf.entry_count)
        return buf


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    buf = RollingBuffer(max_duration_s=10)

    # Simulate 15 entries over 15 seconds
    speakers = ["surgeon", "anesthesiologist", "nurse", "surgeon", "surgeon"]
    phrases = [
        "Patient is prepped and draped",
        "Anesthesia is stable, ready for induction",
        "All instruments are accounted for",
        "Starting the incision now",
        "I need the suction and irrigation",
        "Ventilator settings look good",
        "Heart rate is elevated slightly",
        "Please start the fluoroscopy",
        "Needle placement looks good on imaging",
        "Beginning tract dilation",
        "Start the irrigation pump",
        "Inserting the nephroscope",
        "Camera view is clear",
        "Starting lithotripsy",
        "Fragment extraction in progress",
    ]

    for i in range(15):
        buf.append(
            phrases[i],
            timestamp=float(i * 2),
            speaker=speakers[i % len(speakers)],
        )

    print(f"\n=== Buffer Summary ===")
    summary = buf.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\n=== LLM Context ===")
    print(buf.get_context_for_llm(surgery="PCNL", phase="Phase3"))

    print(f"\n=== Entries since t=20s ===")
    recent = buf.get_entries_since(20.0)
    for e in recent:
        print(f"  {e.format_line()}")

    print(f"\n=== Surgeon entries ===")
    surgeon = buf.get_entries_by_speaker("surgeon")
    for e in surgeon:
        print(f"  {e.format_line()}")

