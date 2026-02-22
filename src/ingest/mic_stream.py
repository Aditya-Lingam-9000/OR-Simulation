"""
OR-Symphony: Microphone Stream Engine

Real-time microphone capture with WebRTC VAD and intelligent chunking.
Produces AudioChunk objects for the ASR pipeline.

Architecture:
    sounddevice.InputStream  →  VAD frame classifier  →  Chunk builder  →  asyncio.Queue

Usage:
    stream = MicStream()
    await stream.start()
    chunk = await stream.get_chunk()   # blocks until speech chunk ready
    await stream.stop()
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import sounddevice as sd
import webrtcvad

from src.ingest.mic_server import AudioChunk
from src.utils.constants import (
    AUDIO_BIT_DEPTH,
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    CHUNK_MAX_DURATION_S,
    CHUNK_MIN_DURATION_S,
    CHUNK_OVERLAP_MS,
    TMP_DIR,
    VAD_AGGRESSIVENESS,
    VAD_FRAME_DURATION_MS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VAD Frame Processor
# ---------------------------------------------------------------------------


@dataclass
class VADFrame:
    """A single VAD-classified audio frame."""

    audio: bytes  # raw PCM 16-bit mono
    timestamp: float  # seconds since stream start
    is_speech: bool


class VADProcessor:
    """
    WebRTC VAD wrapper that classifies audio frames as speech or silence.

    WebRTC VAD requires:
      - 16-bit signed PCM audio
      - Sample rates: 8000, 16000, 32000, 48000 Hz
      - Frame durations: 10, 20, or 30 ms
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        aggressiveness: int = VAD_AGGRESSIVENESS,
        frame_duration_ms: int = VAD_FRAME_DURATION_MS,
    ) -> None:
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"WebRTC VAD requires 8/16/32/48 kHz, got {sample_rate}")
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError(f"VAD frame must be 10/20/30 ms, got {frame_duration_ms}")

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)  # samples per frame

        self._vad = webrtcvad.Vad(aggressiveness)
        logger.info(
            "VADProcessor initialized — sr=%d, aggressiveness=%d, frame=%dms (%d samples)",
            sample_rate,
            aggressiveness,
            frame_duration_ms,
            self.frame_size,
        )

    def is_speech(self, pcm_bytes: bytes) -> bool:
        """
        Classify a single audio frame as speech or silence.

        Args:
            pcm_bytes: Raw 16-bit signed PCM audio (exactly frame_size * 2 bytes).

        Returns:
            True if frame contains speech.
        """
        expected_len = self.frame_size * 2  # 16-bit = 2 bytes per sample
        if len(pcm_bytes) != expected_len:
            raise ValueError(
                f"Frame must be exactly {expected_len} bytes "
                f"({self.frame_size} samples × 2), got {len(pcm_bytes)}"
            )
        return self._vad.is_speech(pcm_bytes, self.sample_rate)

    def process_audio(self, audio: np.ndarray, start_timestamp: float = 0.0) -> List[VADFrame]:
        """
        Process a numpy audio array into VAD-classified frames.

        Args:
            audio: float32 or int16 numpy array (mono).
            start_timestamp: Timestamp offset in seconds.

        Returns:
            List of VADFrame objects.
        """
        # Convert to int16 if needed
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio_int16 = (audio * 32767).astype(np.int16)
        elif audio.dtype == np.int16:
            audio_int16 = audio
        else:
            audio_int16 = audio.astype(np.int16)

        frames: List[VADFrame] = []
        n_samples = len(audio_int16)
        frame_samples = self.frame_size
        frame_duration_s = self.frame_duration_ms / 1000.0

        for i in range(0, n_samples - frame_samples + 1, frame_samples):
            frame_audio = audio_int16[i : i + frame_samples]
            pcm_bytes = frame_audio.tobytes()
            timestamp = start_timestamp + (i / self.sample_rate)

            try:
                speech = self._vad.is_speech(pcm_bytes, self.sample_rate)
            except Exception:
                speech = False

            frames.append(VADFrame(audio=pcm_bytes, timestamp=timestamp, is_speech=speech))

        return frames


# ---------------------------------------------------------------------------
# Ring-buffer based Chunk Builder
# ---------------------------------------------------------------------------


class ChunkBuilder:
    """
    Accumulates VAD speech frames into AudioChunk objects.

    Strategy:
      - Accumulate speech frames until chunk_max_s is reached or
        a silence gap exceeds the padding threshold.
      - Discard chunks shorter than chunk_min_s.
      - Optionally retain `overlap_ms` of trailing audio for the next chunk.
      - Tracks a "speech ring" — consecutive speech frame count with
        a trailing silence tolerance to bridge short pauses.

    Silence padding:
      Up to `_silence_pad_frames` silence frames are tolerated within
      a speech segment (bridges short pauses like "um" or breathing).
    """

    # How many consecutive silence frames to tolerate mid-speech
    SILENCE_PAD_FRAMES = 10  # ~300ms at 30ms/frame

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        frame_duration_ms: int = VAD_FRAME_DURATION_MS,
        chunk_min_s: float = CHUNK_MIN_DURATION_S,
        chunk_max_s: float = CHUNK_MAX_DURATION_S,
        overlap_ms: int = CHUNK_OVERLAP_MS,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.chunk_min_s = chunk_min_s
        self.chunk_max_s = chunk_max_s
        self.overlap_samples = int(sample_rate * overlap_ms / 1000)

        self._buffer: List[VADFrame] = []
        self._silence_count = 0
        self._in_speech = False
        self._chunk_counter = 0

        logger.info(
            "ChunkBuilder — min=%.1fs, max=%.1fs, overlap=%dms, silence_pad=%d frames",
            chunk_min_s,
            chunk_max_s,
            overlap_ms,
            self.SILENCE_PAD_FRAMES,
        )

    def feed(self, frame: VADFrame) -> Optional[AudioChunk]:
        """
        Feed a VAD frame and potentially produce a completed chunk.

        Returns:
            AudioChunk if a complete chunk is ready, else None.
        """
        if frame.is_speech:
            self._silence_count = 0
            self._in_speech = True
            self._buffer.append(frame)
        elif self._in_speech:
            self._silence_count += 1
            self._buffer.append(frame)  # keep silence within speech

            # End of speech segment if silence exceeds padding
            if self._silence_count > self.SILENCE_PAD_FRAMES:
                return self._finalize_chunk()
        # else: silence before speech — ignore

        # Force cut if max duration reached
        buffer_duration = len(self._buffer) * self.frame_duration_ms / 1000.0
        if buffer_duration >= self.chunk_max_s:
            return self._finalize_chunk()

        return None

    def flush(self) -> Optional[AudioChunk]:
        """Flush any remaining buffered speech as a final chunk."""
        if self._buffer and self._in_speech:
            return self._finalize_chunk()
        self._reset()
        return None

    def _finalize_chunk(self) -> Optional[AudioChunk]:
        """Build an AudioChunk from the accumulated buffer."""
        if not self._buffer:
            self._reset()
            return None

        # Strip trailing silence frames (beyond a small pad)
        while self._buffer and not self._buffer[-1].is_speech:
            self._buffer.pop()

        if not self._buffer:
            self._reset()
            return None

        # Concatenate PCM bytes → numpy
        pcm_all = b"".join(f.audio for f in self._buffer)
        audio = np.frombuffer(pcm_all, dtype=np.int16).astype(np.float32) / 32767.0

        duration_s = len(audio) / self.sample_rate
        if duration_s < self.chunk_min_s:
            self._reset()
            return None

        self._chunk_counter += 1
        timestamp_start = self._buffer[0].timestamp
        timestamp_end = self._buffer[-1].timestamp + (self.frame_duration_ms / 1000.0)

        chunk = AudioChunk(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_s=duration_s,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            is_speech=True,
            chunk_id=self._chunk_counter,
        )

        # Keep overlap for next chunk
        overlap_frames = max(1, self.overlap_samples // self.frame_size)
        overlap = self._buffer[-overlap_frames:] if overlap_frames < len(self._buffer) else []

        self._reset()
        if overlap:
            self._buffer = list(overlap)
            self._in_speech = True

        return chunk

    def _reset(self) -> None:
        """Reset builder state."""
        self._buffer.clear()
        self._silence_count = 0
        self._in_speech = False


# ---------------------------------------------------------------------------
# Main MicStream — ties it all together
# ---------------------------------------------------------------------------


class MicStream:
    """
    Real-time microphone stream with VAD and chunking.

    Captures audio from the default input device via sounddevice,
    classifies frames with WebRTC VAD, accumulates speech segments
    into AudioChunks, and pushes them to an async queue.

    Usage:
        stream = MicStream()
        await stream.start()

        # Consumer loop
        while True:
            chunk = await stream.get_chunk()
            # process chunk ...

        await stream.stop()
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        vad_aggressiveness: int = VAD_AGGRESSIVENESS,
        frame_duration_ms: int = VAD_FRAME_DURATION_MS,
        chunk_min_s: float = CHUNK_MIN_DURATION_S,
        chunk_max_s: float = CHUNK_MAX_DURATION_S,
        overlap_ms: int = CHUNK_OVERLAP_MS,
        queue_maxsize: int = 100,
        save_chunks: bool = False,
        save_dir: Optional[Path] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.save_chunks = save_chunks
        self.save_dir = save_dir or (TMP_DIR / "audio_chunks")

        self._vad = VADProcessor(sample_rate, vad_aggressiveness, frame_duration_ms)
        self._chunk_builder = ChunkBuilder(
            sample_rate, frame_duration_ms, chunk_min_s, chunk_max_s, overlap_ms
        )

        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=queue_maxsize)
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._start_time = 0.0
        self._frames_processed = 0
        self._chunks_produced = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(
            "MicStream initialized — sr=%d, ch=%d, save=%s",
            sample_rate,
            channels,
            save_chunks,
        )

    async def start(self) -> None:
        """Start the microphone capture stream."""
        if self._running:
            logger.warning("MicStream already running")
            return

        self._loop = asyncio.get_event_loop()
        self._start_time = time.monotonic()
        self._running = True

        if self.save_chunks:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Saving audio chunks to %s", self.save_dir)

        # Open sounddevice InputStream
        # The callback runs in a separate audio thread
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self._vad.frame_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("MicStream started — capturing from default input device")

    async def stop(self) -> None:
        """Stop the microphone capture stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Flush remaining audio
        final_chunk = self._chunk_builder.flush()
        if final_chunk is not None:
            await self._enqueue_chunk(final_chunk)

        logger.info(
            "MicStream stopped — frames=%d, chunks=%d",
            self._frames_processed,
            self._chunks_produced,
        )

    async def get_chunk(self, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        """
        Get the next speech chunk from the queue.

        Args:
            timeout: Max seconds to wait; None = wait forever.

        Returns:
            AudioChunk or None if timeout.
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._queue.get(), timeout)
            return await self._queue.get()
        except asyncio.TimeoutError:
            return None

    def get_chunk_nowait(self) -> Optional[AudioChunk]:
        """Non-blocking chunk retrieval."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "frames_processed": self._frames_processed,
            "chunks_produced": self._chunks_produced,
            "queue_size": self._queue.qsize(),
            "elapsed_s": round(elapsed, 1),
        }

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """
        sounddevice callback — runs in the audio thread.

        Converts indata to PCM bytes, runs VAD, feeds chunk builder.
        If a chunk is ready, schedules it on the event loop.
        """
        if status:
            logger.warning("Audio callback status: %s", status)

        if not self._running:
            return

        # indata shape: (frame_size, channels), dtype int16
        audio_mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        pcm_bytes = audio_mono.astype(np.int16).tobytes()

        timestamp = time.monotonic() - self._start_time
        self._frames_processed += 1

        # Classify frame
        try:
            speech = self._vad.is_speech(pcm_bytes)
        except Exception:
            speech = False

        frame = VADFrame(audio=pcm_bytes, timestamp=timestamp, is_speech=speech)

        # Feed to chunk builder
        chunk = self._chunk_builder.feed(frame)
        if chunk is not None:
            # Schedule async enqueue from audio thread
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    asyncio.ensure_future, self._enqueue_chunk(chunk)
                )

    async def _enqueue_chunk(self, chunk: AudioChunk) -> None:
        """Enqueue a chunk and optionally save to disk."""
        self._chunks_produced += 1

        # Save to disk if requested
        if self.save_chunks:
            self._save_chunk_wav(chunk)

        # Non-blocking put — drop oldest if full
        if self._queue.full():
            try:
                self._queue.get_nowait()
                logger.warning("Audio queue full — dropping oldest chunk")
            except asyncio.QueueEmpty:
                pass

        await self._queue.put(chunk)
        logger.debug(
            "Chunk #%d enqueued — %.2fs [%.1f-%.1f]",
            chunk.chunk_id,
            chunk.duration_s,
            chunk.timestamp_start,
            chunk.timestamp_end,
        )

    def _save_chunk_wav(self, chunk: AudioChunk) -> None:
        """Save a chunk as a WAV file for debugging/replay."""
        filename = self.save_dir / f"chunk_{chunk.chunk_id:05d}.wav"
        try:
            audio_int16 = (chunk.audio * 32767).astype(np.int16)
            with wave.open(str(filename), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            logger.debug("Saved chunk to %s", filename)
        except Exception as e:
            logger.error("Failed to save chunk WAV: %s", e)


# ---------------------------------------------------------------------------
# Convenience: process a file/buffer through VAD (for testing/replay)
# ---------------------------------------------------------------------------


def process_audio_buffer(
    audio: np.ndarray,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    vad_aggressiveness: int = VAD_AGGRESSIVENESS,
    frame_duration_ms: int = VAD_FRAME_DURATION_MS,
    chunk_min_s: float = CHUNK_MIN_DURATION_S,
    chunk_max_s: float = CHUNK_MAX_DURATION_S,
    overlap_ms: int = CHUNK_OVERLAP_MS,
) -> List[AudioChunk]:
    """
    Process a pre-recorded audio numpy array through VAD + chunking.

    This is the non-async equivalent of MicStream — useful for testing
    and replaying recorded audio through the pipeline.

    Args:
        audio: numpy array, float32 or int16.
        sample_rate: Audio sample rate.
        vad_aggressiveness: VAD level 0-3.
        frame_duration_ms: VAD frame duration.
        chunk_min_s: Min chunk duration.
        chunk_max_s: Max chunk duration.
        overlap_ms: Overlap between chunks.

    Returns:
        List of AudioChunk objects.
    """
    vad = VADProcessor(sample_rate, vad_aggressiveness, frame_duration_ms)
    builder = ChunkBuilder(sample_rate, frame_duration_ms, chunk_min_s, chunk_max_s, overlap_ms)

    frames = vad.process_audio(audio, start_timestamp=0.0)

    chunks: List[AudioChunk] = []
    for frame in frames:
        chunk = builder.feed(frame)
        if chunk is not None:
            chunks.append(chunk)

    # Flush remaining
    final = builder.flush()
    if final is not None:
        chunks.append(final)

    return chunks
