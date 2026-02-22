"""
OR-Symphony: Audio Utility Functions

WAV I/O, format validation, and synthetic audio generation.
Used by tests and scripts to work with audio data without a real microphone.

Usage:
    from src.ingest.audio_utils import load_wav, save_wav, validate_wav, generate_sine
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from src.utils.constants import (
    AUDIO_BIT_DEPTH,
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------


def load_wav(
    path: Union[str, Path],
    target_sr: int = AUDIO_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file and return (audio_float32, sample_rate).

    Args:
        path: Path to .wav file.
        target_sr: Expected sample rate (warning if mismatch).

    Returns:
        Tuple of (audio as float32 numpy array, actual sample rate).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"WAV file not found: {path}")

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sr != target_sr:
        logger.warning("WAV sample rate %d != target %d — no resampling applied", sr, target_sr)

    # Convert to numpy
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483647.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Mix to mono if stereo
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sr


def save_wav(
    path: Union[str, Path],
    audio: np.ndarray,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int = AUDIO_CHANNELS,
    sampwidth: int = 2,
) -> Path:
    """
    Save a numpy audio array as a WAV file.

    Args:
        path: Output path.
        audio: Float32 or int16 numpy array.
        sample_rate: Sample rate.
        channels: Number of channels.
        sampwidth: Sample width in bytes (2 = 16-bit).

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert float to int16
    if audio.dtype in (np.float32, np.float64):
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    elif audio.dtype == np.int16:
        audio_int16 = audio
    else:
        audio_int16 = audio.astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    logger.debug("Saved WAV: %s (%.2fs, %d Hz)", path, len(audio) / sample_rate, sample_rate)
    return path


def audio_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> bytes:
    """Convert numpy audio to in-memory WAV bytes."""
    if audio.dtype in (np.float32, np.float64):
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    else:
        audio_int16 = audio.astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class AudioValidationResult:
    """Result of audio format validation."""

    def __init__(self) -> None:
        self.valid = True
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.sample_rate: int = 0
        self.channels: int = 0
        self.bit_depth: int = 0
        self.duration_s: float = 0.0
        self.n_samples: int = 0

    def __bool__(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        return (
            f"AudioValidation({status}, sr={self.sample_rate}, "
            f"ch={self.channels}, bits={self.bit_depth}, "
            f"dur={self.duration_s:.2f}s)"
        )


def validate_wav(
    path: Union[str, Path],
    expected_sr: int = AUDIO_SAMPLE_RATE,
    expected_channels: int = AUDIO_CHANNELS,
    expected_bit_depth: int = AUDIO_BIT_DEPTH,
    min_duration_s: float = 0.0,
    max_duration_s: float = float("inf"),
) -> AudioValidationResult:
    """
    Validate a WAV file against expected audio properties.

    Args:
        path: Path to .wav file.
        expected_sr: Expected sample rate (16000).
        expected_channels: Expected channels (1 = mono).
        expected_bit_depth: Expected bit depth (16).
        min_duration_s: Minimum duration in seconds.
        max_duration_s: Maximum duration in seconds.

    Returns:
        AudioValidationResult with valid flag, errors, and metadata.
    """
    result = AudioValidationResult()
    path = Path(path)

    if not path.exists():
        result.valid = False
        result.errors.append(f"File not found: {path}")
        return result

    if path.suffix.lower() != ".wav":
        result.warnings.append(f"File extension is '{path.suffix}', expected '.wav'")

    try:
        with wave.open(str(path), "rb") as wf:
            result.sample_rate = wf.getframerate()
            result.channels = wf.getnchannels()
            result.bit_depth = wf.getsampwidth() * 8
            result.n_samples = wf.getnframes()
            result.duration_s = result.n_samples / result.sample_rate
    except wave.Error as e:
        result.valid = False
        result.errors.append(f"Invalid WAV file: {e}")
        return result

    # Check sample rate
    if result.sample_rate != expected_sr:
        result.valid = False
        result.errors.append(f"Sample rate {result.sample_rate} != expected {expected_sr}")

    # Check channels
    if result.channels != expected_channels:
        result.valid = False
        result.errors.append(f"Channels {result.channels} != expected {expected_channels}")

    # Check bit depth
    if result.bit_depth != expected_bit_depth:
        result.valid = False
        result.errors.append(f"Bit depth {result.bit_depth} != expected {expected_bit_depth}")

    # Check duration
    if result.duration_s < min_duration_s:
        result.valid = False
        result.errors.append(f"Duration {result.duration_s:.2f}s < minimum {min_duration_s:.2f}s")

    if result.duration_s > max_duration_s:
        result.valid = False
        result.errors.append(f"Duration {result.duration_s:.2f}s > maximum {max_duration_s:.2f}s")

    return result


def validate_audio_array(
    audio: np.ndarray,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    expected_dtype: type = np.float32,
) -> AudioValidationResult:
    """
    Validate an in-memory numpy audio array.

    Args:
        audio: Numpy audio array.
        sample_rate: Sample rate for duration computation.
        expected_dtype: Expected numpy dtype.

    Returns:
        AudioValidationResult.
    """
    result = AudioValidationResult()
    result.sample_rate = sample_rate
    result.channels = 1 if audio.ndim == 1 else audio.shape[1]
    result.n_samples = len(audio)
    result.duration_s = result.n_samples / sample_rate

    if audio.ndim > 2:
        result.valid = False
        result.errors.append(f"Audio has {audio.ndim} dimensions, expected 1 or 2")

    if audio.dtype != expected_dtype:
        result.warnings.append(f"Audio dtype {audio.dtype} != expected {expected_dtype}")

    # Check for NaN/Inf
    if np.any(np.isnan(audio)):
        result.valid = False
        result.errors.append("Audio contains NaN values")

    if np.any(np.isinf(audio)):
        result.valid = False
        result.errors.append("Audio contains Inf values")

    # Check amplitude range for float audio
    if audio.dtype in (np.float32, np.float64):
        max_amp = np.max(np.abs(audio))
        if max_amp > 1.0:
            result.warnings.append(f"Audio amplitude {max_amp:.2f} exceeds [-1, 1] range")

    return result


# ---------------------------------------------------------------------------
# Synthetic Audio Generation (for testing)
# ---------------------------------------------------------------------------


def generate_sine(
    frequency: float = 440.0,
    duration_s: float = 1.0,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate a sine wave (useful for testing).

    Args:
        frequency: Frequency in Hz.
        duration_s: Duration in seconds.
        sample_rate: Sample rate.
        amplitude: Signal amplitude (0.0 to 1.0).

    Returns:
        Float32 numpy array.
    """
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def generate_silence(
    duration_s: float = 1.0,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> np.ndarray:
    """Generate silence (zeros)."""
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


def generate_white_noise(
    duration_s: float = 1.0,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    amplitude: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Generate white noise."""
    rng = np.random.RandomState(seed)
    n_samples = int(sample_rate * duration_s)
    return (amplitude * rng.randn(n_samples)).astype(np.float32)


def generate_speech_like(
    duration_s: float = 2.0,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a speech-like signal (multiple harmonics + noise).

    Not real speech, but mimics the spectral characteristics of voice
    enough to trigger VAD speech detection. Fundamental ~150 Hz with
    harmonics at 300, 450, 600 Hz plus low-level noise.

    Args:
        duration_s: Duration in seconds.
        sample_rate: Sample rate.
        seed: Random seed for noise component.

    Returns:
        Float32 numpy array.
    """
    rng = np.random.RandomState(seed)
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)

    # Fundamental + harmonics (speech-like formants)
    signal = np.zeros(n_samples, dtype=np.float32)
    for freq, amp in [(150, 0.3), (300, 0.2), (450, 0.1), (600, 0.05)]:
        signal += amp * np.sin(2 * np.pi * freq * t)

    # Add some noise to make it more "speech-like"
    noise = 0.05 * rng.randn(n_samples).astype(np.float32)
    signal += noise

    # Normalize to prevent clipping
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.7

    return signal


def generate_speech_silence_pattern(
    speech_duration_s: float = 1.0,
    silence_duration_s: float = 1.0,
    repetitions: int = 3,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate alternating speech-like and silence segments.

    Useful for testing VAD chunk boundaries.

    Args:
        speech_duration_s: Duration of each speech segment.
        silence_duration_s: Duration of each silence segment.
        repetitions: Number of speech+silence pairs.
        sample_rate: Sample rate.
        seed: Random seed.

    Returns:
        Float32 numpy array.
    """
    segments = []
    for i in range(repetitions):
        speech = generate_speech_like(speech_duration_s, sample_rate, seed=seed + i)
        silence = generate_silence(silence_duration_s, sample_rate)
        segments.extend([speech, silence])

    return np.concatenate(segments)
