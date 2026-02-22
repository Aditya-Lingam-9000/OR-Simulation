"""
OR-Symphony: Rolling Transcript Buffer

Maintains a sliding window of recent transcript segments with timestamps.
Used to provide context to MedGemma for surgical phase reasoning.

Default window: 180 seconds (configurable).

Usage:
    from src.state.rolling_buffer import RollingBuffer
    buffer = RollingBuffer(max_duration_s=180)
    buffer.append("intubation complete", timestamp=42.5)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from src.utils.constants import ROLLING_BUFFER_DURATION_S

logger = logging.getLogger(__name__)


@dataclass
class BufferEntry:
    """A single transcript entry in the rolling buffer."""

    text: str
    timestamp: float  # seconds since session start
    speaker: str = "unknown"
    is_final: bool = True
    source: str = "asr"  # asr | override | system


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

        logger.info("RollingBuffer initialized — max_duration=%.0fs", max_duration_s)

    def append(
        self,
        text: str,
        timestamp: Optional[float] = None,
        speaker: str = "unknown",
        is_final: bool = True,
    ) -> None:
        """
        Append a transcript segment to the buffer.

        Args:
            text: Transcript text.
            timestamp: Seconds since session start. Auto-calculated if None.
            speaker: Speaker label (if ASR supports diarization).
            is_final: Whether this is a final (vs partial) transcript.
        """
        if timestamp is None:
            timestamp = time.time() - self._session_start

        entry = BufferEntry(
            text=text,
            timestamp=timestamp,
            speaker=speaker,
            is_final=is_final,
        )
        self._entries.append(entry)
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

    def get_entries(self) -> List[BufferEntry]:
        """Get all current buffer entries as a list."""
        return list(self._entries)

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

    def clear(self) -> None:
        """Clear all buffer entries."""
        self._entries.clear()
        self._session_start = time.time()

    def reset_session(self) -> None:
        """Reset the session timer and clear buffer."""
        self.clear()
        logger.info("RollingBuffer session reset")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    buf = RollingBuffer(max_duration_s=10)

    for i in range(15):
        buf.append(f"Segment {i}", timestamp=float(i))

    print(f"Entries: {buf.entry_count}")
    print(f"Duration: {buf.duration_s:.1f}s")
    print(f"Earliest: {buf.earliest_timestamp}")
    print(f"Latest: {buf.latest_timestamp}")
    print(f"Context: {buf.get_context()}")
