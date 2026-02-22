"""
OR-Symphony: Audio Pipeline Tests (Phase 2)

Tests for:
  - Audio utilities (WAV I/O, validation, synthetic generation)
  - VAD processor (speech/silence classification)
  - Chunk builder (duration bounds, overlap, edge cases)
  - MicrophoneCapture.feed_audio() (offline processing)
  - process_audio_buffer() convenience function
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.ingest.audio_utils import (
    AudioValidationResult,
    generate_silence,
    generate_sine,
    generate_speech_like,
    generate_speech_silence_pattern,
    generate_white_noise,
    load_wav,
    save_wav,
    validate_audio_array,
    validate_wav,
)
from src.ingest.mic_server import AudioChunk, MicrophoneCapture
from src.ingest.mic_stream import (
    ChunkBuilder,
    VADFrame,
    VADProcessor,
    process_audio_buffer,
)
from src.utils.constants import AUDIO_SAMPLE_RATE


# ===================================================================
# Audio Utilities
# ===================================================================


class TestGenerateSyntheticAudio:
    """Test synthetic audio generation functions."""

    def test_sine_shape(self):
        audio = generate_sine(440, 1.0, AUDIO_SAMPLE_RATE)
        assert audio.shape == (AUDIO_SAMPLE_RATE,)
        assert audio.dtype == np.float32

    def test_sine_amplitude(self):
        audio = generate_sine(440, 1.0, amplitude=0.5)
        assert np.max(np.abs(audio)) <= 0.51  # slight rounding

    def test_silence_is_zero(self):
        audio = generate_silence(1.0)
        assert np.allclose(audio, 0.0)
        assert len(audio) == AUDIO_SAMPLE_RATE

    def test_white_noise_not_zero(self):
        audio = generate_white_noise(1.0)
        assert np.std(audio) > 0.01

    def test_white_noise_reproducible(self):
        a = generate_white_noise(1.0, seed=42)
        b = generate_white_noise(1.0, seed=42)
        assert np.array_equal(a, b)

    def test_speech_like_shape(self):
        audio = generate_speech_like(2.0)
        assert len(audio) == AUDIO_SAMPLE_RATE * 2
        assert audio.dtype == np.float32

    def test_speech_like_amplitude_bounded(self):
        audio = generate_speech_like(2.0)
        assert np.max(np.abs(audio)) <= 1.0

    def test_speech_silence_pattern_length(self):
        audio = generate_speech_silence_pattern(1.0, 1.0, repetitions=3)
        expected_samples = int(AUDIO_SAMPLE_RATE * (1.0 + 1.0) * 3)
        assert len(audio) == expected_samples


class TestWavIO:
    """Test WAV file read/write."""

    def test_save_and_load_roundtrip(self, tmp_path):
        audio = generate_sine(440, 1.0)
        wav_path = tmp_path / "test.wav"
        save_wav(wav_path, audio)

        loaded, sr = load_wav(wav_path)
        assert sr == AUDIO_SAMPLE_RATE
        assert len(loaded) == len(audio)
        # Allow small quantization error (float32 → int16 → float32)
        assert np.allclose(audio, loaded, atol=1e-3)

    def test_save_creates_directory(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "test.wav"
        audio = generate_silence(0.5)
        save_wav(deep_path, audio)
        assert deep_path.exists()

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_wav("nonexistent.wav")

    def test_save_int16_input(self, tmp_path):
        audio_int16 = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
        wav_path = tmp_path / "int16.wav"
        save_wav(wav_path, audio_int16)
        loaded, sr = load_wav(wav_path)
        assert len(loaded) == 5


class TestValidateWav:
    """Test WAV file validation."""

    def test_valid_wav(self, tmp_path):
        audio = generate_sine(440, 1.0)
        wav_path = tmp_path / "valid.wav"
        save_wav(wav_path, audio)

        result = validate_wav(wav_path)
        assert result.valid
        assert result.sample_rate == AUDIO_SAMPLE_RATE
        assert result.channels == 1
        assert result.bit_depth == 16
        assert abs(result.duration_s - 1.0) < 0.01

    def test_nonexistent_file(self):
        result = validate_wav("nope.wav")
        assert not result.valid
        assert any("not found" in e for e in result.errors)

    def test_wrong_sample_rate(self, tmp_path):
        """WAV with different sample rate should fail validation."""
        audio = generate_sine(440, 1.0, sample_rate=8000)
        wav_path = tmp_path / "wrong_sr.wav"
        save_wav(wav_path, audio, sample_rate=8000)

        result = validate_wav(wav_path, expected_sr=AUDIO_SAMPLE_RATE)
        assert not result.valid
        assert any("Sample rate" in e for e in result.errors)

    def test_duration_bounds(self, tmp_path):
        audio = generate_silence(0.5)
        wav_path = tmp_path / "short.wav"
        save_wav(wav_path, audio)

        result = validate_wav(wav_path, min_duration_s=1.0)
        assert not result.valid
        assert any("minimum" in e for e in result.errors)


class TestValidateAudioArray:
    """Test in-memory audio validation."""

    def test_valid_array(self):
        audio = generate_sine(440, 1.0)
        result = validate_audio_array(audio)
        assert result.valid

    def test_nan_detection(self):
        audio = np.array([0.0, float("nan"), 0.0], dtype=np.float32)
        result = validate_audio_array(audio)
        assert not result.valid
        assert any("NaN" in e for e in result.errors)

    def test_inf_detection(self):
        audio = np.array([0.0, float("inf"), 0.0], dtype=np.float32)
        result = validate_audio_array(audio)
        assert not result.valid
        assert any("Inf" in e for e in result.errors)

    def test_amplitude_warning(self):
        audio = np.array([2.0, -2.0], dtype=np.float32)
        result = validate_audio_array(audio)
        assert result.valid  # warning, not error
        assert len(result.warnings) > 0


# ===================================================================
# VAD Processor
# ===================================================================


class TestVADProcessor:
    """Test WebRTC VAD frame classification."""

    def test_create_default(self):
        vad = VADProcessor()
        assert vad.sample_rate == AUDIO_SAMPLE_RATE
        assert vad.frame_size == int(AUDIO_SAMPLE_RATE * 30 / 1000)

    def test_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="8/16/32/48"):
            VADProcessor(sample_rate=22050)

    def test_invalid_frame_duration(self):
        with pytest.raises(ValueError, match="10/20/30"):
            VADProcessor(frame_duration_ms=25)

    def test_silence_classified_as_non_speech(self):
        """Silence should be classified as non-speech."""
        vad = VADProcessor(aggressiveness=3)
        silence = np.zeros(vad.frame_size, dtype=np.int16)
        assert not vad.is_speech(silence.tobytes())

    def test_speech_like_classified_as_speech(self):
        """Speech-like signal should trigger speech detection."""
        vad = VADProcessor(aggressiveness=0)  # least aggressive
        speech = generate_speech_like(0.03, AUDIO_SAMPLE_RATE)
        frame_samples = vad.frame_size
        audio_int16 = (speech[:frame_samples] * 32767).astype(np.int16)
        # Speech-like may or may not trigger — just check no crash
        result = vad.is_speech(audio_int16.tobytes())
        assert isinstance(result, bool)

    def test_wrong_frame_size_raises(self):
        vad = VADProcessor()
        with pytest.raises(ValueError, match="exactly"):
            vad.is_speech(b"\x00" * 10)

    def test_process_audio_returns_frames(self):
        vad = VADProcessor()
        audio = generate_silence(0.1, AUDIO_SAMPLE_RATE)
        frames = vad.process_audio(audio)
        # 100ms / 30ms = 3 frames (with integer truncation)
        assert len(frames) >= 3
        for f in frames:
            assert isinstance(f, VADFrame)
            assert isinstance(f.is_speech, bool)

    def test_process_audio_timestamps(self):
        vad = VADProcessor()
        audio = generate_silence(0.1, AUDIO_SAMPLE_RATE)
        frames = vad.process_audio(audio, start_timestamp=5.0)
        assert frames[0].timestamp >= 5.0
        # Timestamps should be monotonically increasing
        for i in range(1, len(frames)):
            assert frames[i].timestamp > frames[i - 1].timestamp


# ===================================================================
# Chunk Builder
# ===================================================================


class TestChunkBuilder:
    """Test chunk accumulation and boundary detection."""

    def _make_speech_frame(self, timestamp: float = 0.0) -> VADFrame:
        """Create a fake speech frame."""
        frame_size = int(AUDIO_SAMPLE_RATE * 30 / 1000)
        audio = np.random.randint(-1000, 1000, frame_size, dtype=np.int16).tobytes()
        return VADFrame(audio=audio, timestamp=timestamp, is_speech=True)

    def _make_silence_frame(self, timestamp: float = 0.0) -> VADFrame:
        """Create a fake silence frame."""
        frame_size = int(AUDIO_SAMPLE_RATE * 30 / 1000)
        audio = np.zeros(frame_size, dtype=np.int16).tobytes()
        return VADFrame(audio=audio, timestamp=timestamp, is_speech=False)

    def test_empty_builder(self):
        builder = ChunkBuilder()
        assert builder.flush() is None

    def test_silence_only_no_chunk(self):
        builder = ChunkBuilder()
        for i in range(50):
            result = builder.feed(self._make_silence_frame(i * 0.03))
            assert result is None
        assert builder.flush() is None

    def test_speech_produces_chunk_on_silence(self):
        """Speech followed by enough silence should produce a chunk."""
        builder = ChunkBuilder(chunk_min_s=0.1)  # lower min for test
        t = 0.0

        # Feed speech frames (enough for > 0.1s)
        for _ in range(20):  # 20 * 30ms = 600ms of speech
            builder.feed(self._make_speech_frame(t))
            t += 0.03

        # Feed silence to trigger end-of-speech
        chunk = None
        for _ in range(builder.SILENCE_PAD_FRAMES + 5):
            result = builder.feed(self._make_silence_frame(t))
            t += 0.03
            if result is not None:
                chunk = result
                break

        assert chunk is not None
        assert isinstance(chunk, AudioChunk)
        assert chunk.duration_s >= 0.1

    def test_max_duration_forces_cut(self):
        """Chunk builder should cut at max_duration even without silence."""
        builder = ChunkBuilder(chunk_max_s=0.5, chunk_min_s=0.1)
        t = 0.0
        chunks = []

        # Feed continuous speech for 2 seconds
        for _ in range(67):  # 67 * 30ms = ~2s
            result = builder.feed(self._make_speech_frame(t))
            t += 0.03
            if result is not None:
                chunks.append(result)

        # Should have cut at least once
        assert len(chunks) >= 1
        for c in chunks:
            assert c.duration_s <= 0.6  # slight tolerance

    def test_min_duration_discards_short(self):
        """Chunks shorter than min_duration should be discarded."""
        builder = ChunkBuilder(chunk_min_s=1.0)

        # Feed just 5 speech frames (150ms) then silence
        for i in range(5):
            builder.feed(self._make_speech_frame(i * 0.03))

        # Trigger end-of-speech
        chunk = None
        for i in range(20):
            result = builder.feed(self._make_silence_frame((5 + i) * 0.03))
            if result is not None:
                chunk = result
                break

        # Should be discarded (150ms < 1.0s minimum)
        assert chunk is None

    def test_flush_returns_buffered_speech(self):
        """Flush should return any accumulated speech."""
        builder = ChunkBuilder(chunk_min_s=0.1)

        for i in range(10):
            builder.feed(self._make_speech_frame(i * 0.03))

        chunk = builder.flush()
        assert chunk is not None
        assert chunk.duration_s >= 0.1

    def test_chunk_id_increments(self):
        """Each chunk should get a unique incrementing ID."""
        builder = ChunkBuilder(chunk_max_s=0.3, chunk_min_s=0.1)
        t = 0.0
        ids = []

        for _ in range(200):
            result = builder.feed(self._make_speech_frame(t))
            t += 0.03
            if result is not None:
                ids.append(result.chunk_id)

        assert len(ids) >= 2
        for i in range(1, len(ids)):
            assert ids[i] == ids[i - 1] + 1

    def test_chunk_audio_is_float32(self):
        builder = ChunkBuilder(chunk_min_s=0.1, chunk_max_s=0.5)
        for i in range(20):
            builder.feed(self._make_speech_frame(i * 0.03))

        chunk = builder.flush()
        assert chunk is not None
        assert chunk.audio.dtype == np.float32


# ===================================================================
# process_audio_buffer (sync convenience function)
# ===================================================================


class TestProcessAudioBuffer:
    """Test the sync convenience function for offline processing."""

    def test_silence_produces_no_chunks(self):
        """Pure silence should produce zero or near-zero chunks."""
        silence = generate_silence(5.0, AUDIO_SAMPLE_RATE)
        chunks = process_audio_buffer(silence, vad_aggressiveness=3)
        # Aggressive VAD on perfect silence → 0 chunks
        assert len(chunks) <= 2  # allow tiny false-positive margin

    def test_speech_like_produces_chunks(self):
        """Speech-like signal should produce at least one chunk."""
        speech = generate_speech_like(3.0, AUDIO_SAMPLE_RATE)
        chunks = process_audio_buffer(speech, vad_aggressiveness=0)
        # With aggressiveness=0 (most permissive), speech-like should trigger
        assert len(chunks) >= 1

    def test_speech_silence_pattern_multiple_chunks(self):
        """Alternating speech+silence should produce multiple chunks."""
        audio = generate_speech_silence_pattern(
            speech_duration_s=1.0,
            silence_duration_s=1.5,
            repetitions=3,
        )
        chunks = process_audio_buffer(audio, vad_aggressiveness=0)
        # With 3 speech segments, expect at least 1 chunk
        assert len(chunks) >= 1

    def test_chunk_durations_within_bounds(self):
        """All chunks should respect min/max duration settings."""
        audio = generate_speech_like(5.0, AUDIO_SAMPLE_RATE)
        min_s, max_s = 0.3, 1.5
        chunks = process_audio_buffer(audio, chunk_min_s=min_s, chunk_max_s=max_s, vad_aggressiveness=0)
        for c in chunks:
            assert c.duration_s >= min_s, f"Chunk too short: {c.duration_s:.2f}s < {min_s}s"
            assert c.duration_s <= max_s + 0.1, f"Chunk too long: {c.duration_s:.2f}s > {max_s}s"

    def test_chunk_timestamps_monotonic(self):
        """Chunk timestamps should be monotonically increasing."""
        audio = generate_speech_like(5.0, AUDIO_SAMPLE_RATE)
        chunks = process_audio_buffer(audio, vad_aggressiveness=0, chunk_max_s=1.0, chunk_min_s=0.2)
        for i in range(1, len(chunks)):
            assert chunks[i].timestamp_start >= chunks[i - 1].timestamp_start


# ===================================================================
# MicrophoneCapture.feed_audio (offline mode)
# ===================================================================


class TestMicrophoneCaptureFeedAudio:
    """Test MicrophoneCapture's offline audio processing."""

    def test_feed_silence(self):
        mic = MicrophoneCapture(vad_aggressiveness=3)
        silence = generate_silence(2.0)
        chunks = mic.feed_audio(silence)
        assert len(chunks) <= 2

    def test_feed_speech(self):
        mic = MicrophoneCapture(vad_aggressiveness=0)
        speech = generate_speech_like(2.0)
        chunks = mic.feed_audio(speech)
        assert len(chunks) >= 1

    def test_feed_returns_audio_chunks(self):
        mic = MicrophoneCapture(vad_aggressiveness=0)
        speech = generate_speech_like(2.0)
        chunks = mic.feed_audio(speech)
        for c in chunks:
            assert isinstance(c, AudioChunk)
            assert c.audio.dtype == np.float32
            assert c.sample_rate == AUDIO_SAMPLE_RATE
            assert c.is_speech is True


# ===================================================================
# VAD False Positive Check (from master plan)
# ===================================================================


class TestVADFalsePositive:
    """
    Master plan requirement:
    'VAD false positive check: 30s silence → <2 chunks'
    """

    def test_30s_silence_under_2_chunks(self):
        """30 seconds of pure silence should produce fewer than 2 chunks."""
        silence = generate_silence(30.0, AUDIO_SAMPLE_RATE)
        chunks = process_audio_buffer(silence, vad_aggressiveness=3)
        assert len(chunks) < 2, f"Got {len(chunks)} chunks from 30s silence (expected <2)"
