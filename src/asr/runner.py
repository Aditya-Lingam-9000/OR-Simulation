"""
OR-Symphony: ASR Runner Interface

Abstract base class for speech recognition runners.
Concrete implementations handle ONNX, Whisper, or other backends.

Usage:
    runner = OnnxASRRunner(model_path, tokens_path)
    result = runner.transcribe(audio_chunk)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A single transcript segment produced by ASR."""

    text: str
    is_final: bool
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    confidence: float = 0.0
    language: str = "en"
    tokens: List[str] = field(default_factory=list)


@dataclass
class ASRResult:
    """Complete ASR result for an audio chunk."""

    segments: List[TranscriptSegment]
    processing_time_ms: float = 0.0
    model_name: str = ""
    is_partial: bool = False

    @property
    def full_text(self) -> str:
        """Concatenate all segment texts."""
        return " ".join(seg.text for seg in self.segments if seg.text.strip())


class BaseASRRunner(ABC):
    """Abstract base class for ASR runners."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the ASR model into memory."""
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """
        Transcribe an audio chunk.

        Args:
            audio: Audio data as numpy array (float32, mono).
            sample_rate: Audio sample rate in Hz.

        Returns:
            ASRResult with transcript segments.
        """
        ...

    @abstractmethod
    def transcribe_streaming(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """
        Transcribe with streaming partial results.

        Args:
            audio: Audio data as numpy array (float32, mono).
            sample_rate: Audio sample rate in Hz.

        Returns:
            ASRResult with partial and/or final segments.
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Release model resources."""
        ...
