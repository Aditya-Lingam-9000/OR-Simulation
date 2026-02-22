"""
OR-Symphony: Microphone Server / Audio Ingestion

High-level microphone capture interface. Wraps the MicStream engine
and provides the AudioChunk dataclass used across the pipeline.

Usage:
    from src.ingest.mic_server import MicrophoneCapture, AudioChunk
    mic = MicrophoneCapture(sample_rate=16000)
    await mic.start()
    chunk = await mic.get_chunk()
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
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

    Wraps `MicStream` for real-time capture, or can be fed pre-recorded
    audio for offline processing via `feed_audio()`.
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        vad_aggressiveness: int = VAD_AGGRESSIVENESS,
        chunk_min_s: float = CHUNK_MIN_DURATION_S,
        chunk_max_s: float = CHUNK_MAX_DURATION_S,
        on_chunk: Optional[Callable[[AudioChunk], None]] = None,
        save_chunks: bool = False,
        save_dir: Optional[Path] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.vad_aggressiveness = vad_aggressiveness
        self.chunk_min_s = chunk_min_s
        self.chunk_max_s = chunk_max_s
        self.on_chunk = on_chunk
        self.save_chunks = save_chunks
        self.save_dir = save_dir

        self._mic_stream = None  # Lazy import to avoid sounddevice dependency at import time
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
        """Start real-time microphone capture with VAD."""
        from src.ingest.mic_stream import MicStream

        self._mic_stream = MicStream(
            sample_rate=self.sample_rate,
            channels=self.channels,
            vad_aggressiveness=self.vad_aggressiveness,
            chunk_min_s=self.chunk_min_s,
            chunk_max_s=self.chunk_max_s,
            save_chunks=self.save_chunks,
            save_dir=self.save_dir,
        )
        await self._mic_stream.start()
        self._running = True
        logger.info("MicrophoneCapture started — live mic capture active")

    async def stop(self) -> None:
        """Stop microphone capture."""
        self._running = False
        if self._mic_stream is not None:
            await self._mic_stream.stop()
            self._mic_stream = None
        logger.info("MicrophoneCapture stopped")

    async def get_chunk(self, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        """
        Get the next speech chunk (blocking).

        Args:
            timeout: Max seconds to wait; None = wait forever.

        Returns:
            AudioChunk or None on timeout.
        """
        if self._mic_stream is None:
            return None
        chunk = await self._mic_stream.get_chunk(timeout=timeout)
        if chunk is not None and self.on_chunk is not None:
            self.on_chunk(chunk)
        return chunk

    def get_chunk_nowait(self) -> Optional[AudioChunk]:
        """Non-blocking chunk retrieval."""
        if self._mic_stream is None:
            return None
        chunk = self._mic_stream.get_chunk_nowait()
        if chunk is not None and self.on_chunk is not None:
            self.on_chunk(chunk)
        return chunk

    def feed_audio(self, audio: np.ndarray) -> list[AudioChunk]:
        """
        Process pre-recorded audio through VAD + chunking (synchronous).

        Useful for testing or replaying recorded audio.

        Args:
            audio: Float32 numpy array (mono, 16kHz).

        Returns:
            List of AudioChunk objects.
        """
        from src.ingest.mic_stream import process_audio_buffer

        return process_audio_buffer(
            audio=audio,
            sample_rate=self.sample_rate,
            vad_aggressiveness=self.vad_aggressiveness,
            chunk_min_s=self.chunk_min_s,
            chunk_max_s=self.chunk_max_s,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        if self._mic_stream is not None:
            return self._mic_stream.queue_size
        return 0

    @property
    def stats(self) -> dict:
        if self._mic_stream is not None:
            return self._mic_stream.stats
        return {"running": False, "frames_processed": 0, "chunks_produced": 0}
