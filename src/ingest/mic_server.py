"""
OR-Symphony: Microphone Server / Audio Ingestion

Handles microphone capture (local desktop mic or WebRTC browser input),
applies VAD (Voice Activity Detection), and produces audio chunks
for the ASR pipeline.

This module is a skeleton — full implementation in Phase 2.

Usage:
    from src.ingest.mic_server import MicrophoneCapture
    mic = MicrophoneCapture(sample_rate=16000)
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional

import numpy as np

from src.utils.constants import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    CHUNK_MAX_DURATION_S,
    CHUNK_MIN_DURATION_S,
    CHUNK_OVERLAP_MS,
    VAD_AGGRESSIVENESS,
    VAD_FRAME_DURATION_MS,
)

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """A single audio chunk ready for ASR processing."""

    audio: np.ndarray
    sample_rate: int = AUDIO_SAMPLE_RATE
    duration_s: float = 0.0
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    is_speech: bool = True
    chunk_id: int = 0


class MicrophoneCapture:
    """
    Desktop microphone capture with VAD and chunking.

    Captures audio from the local microphone, applies WebRTC VAD,
    and produces speech chunks for the ASR pipeline.

    Full implementation in Phase 2.
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        vad_aggressiveness: int = VAD_AGGRESSIVENESS,
        chunk_min_s: float = CHUNK_MIN_DURATION_S,
        chunk_max_s: float = CHUNK_MAX_DURATION_S,
        on_chunk: Optional[Callable[[AudioChunk], None]] = None,
    ) -> None:
        """
        Initialize microphone capture.

        Args:
            sample_rate: Audio sample rate (default 16kHz).
            channels: Number of audio channels (default 1 = mono).
            vad_aggressiveness: VAD aggressiveness level (0-3).
            chunk_min_s: Minimum chunk duration in seconds.
            chunk_max_s: Maximum chunk duration in seconds.
            on_chunk: Callback when a new chunk is ready.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.vad_aggressiveness = vad_aggressiveness
        self.chunk_min_s = chunk_min_s
        self.chunk_max_s = chunk_max_s
        self.on_chunk = on_chunk

        self._chunk_queue: Deque[AudioChunk] = deque(maxlen=100)
        self._running = False
        self._chunk_counter = 0

        logger.info(
            "MicrophoneCapture initialized — sr=%d, vad=%d, chunk=%.1f-%.1fs",
            sample_rate,
            vad_aggressiveness,
            chunk_min_s,
            chunk_max_s,
        )

    async def start(self) -> None:
        """Start microphone capture. Full implementation in Phase 2."""
        self._running = True
        logger.info("Microphone capture started (placeholder)")
        # TODO: Phase 2 — implement actual mic capture with sounddevice + VAD

    async def stop(self) -> None:
        """Stop microphone capture."""
        self._running = False
        logger.info("Microphone capture stopped")

    def get_chunk(self) -> Optional[AudioChunk]:
        """Get the next audio chunk from the queue (non-blocking)."""
        if self._chunk_queue:
            return self._chunk_queue.popleft()
        return None

    @property
    def is_running(self) -> bool:
        """Check if capture is active."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Current number of chunks waiting in queue."""
        return len(self._chunk_queue)
