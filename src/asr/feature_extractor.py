"""
OR-Symphony: Audio Feature Extractor

Converts raw audio waveforms to 128-dim log-mel filterbank features
suitable for MedASR ONNX model input.

Feature specification (from model metadata):
  - 128 mel bins
  - 25 ms window length
  - 10 ms hop (frame shift)
  - Hamming window
  - 16 kHz sample rate

Usage:
    extractor = FeatureExtractor()
    features = extractor.extract(audio_float32)  # (T, 128) numpy
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio

from src.utils.constants import AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature extraction parameters (matching MedASR model expectations)
# ---------------------------------------------------------------------------

NUM_MEL_BINS = 128
FRAME_LENGTH_MS = 25.0
FRAME_SHIFT_MS = 10.0
WINDOW_TYPE = "hamming"
DITHER = 0.0  # No dither for deterministic output


class FeatureExtractor:
    """
    Extracts 128-dim log-mel filterbank features from audio.

    Uses torchaudio's Kaldi-compatible fbank implementation for
    consistency with the MedASR model training pipeline.
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        num_mel_bins: int = NUM_MEL_BINS,
        frame_length_ms: float = FRAME_LENGTH_MS,
        frame_shift_ms: float = FRAME_SHIFT_MS,
        window_type: str = WINDOW_TYPE,
        dither: float = DITHER,
        normalize: bool = True,
    ) -> None:
        """
        Initialize the feature extractor.

        Args:
            sample_rate: Audio sample rate (must be 16000 for MedASR).
            num_mel_bins: Number of mel filterbank bins.
            frame_length_ms: Window length in milliseconds.
            frame_shift_ms: Hop size in milliseconds.
            window_type: Window function type.
            dither: Dither amount (0.0 = off).
            normalize: Whether to apply utterance-level CMVN.
        """
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length_ms = frame_length_ms
        self.frame_shift_ms = frame_shift_ms
        self.window_type = window_type
        self.dither = dither
        self.normalize = normalize

        # Pre-compute expected samples per frame for validation
        self._samples_per_frame = int(sample_rate * frame_shift_ms / 1000)
        self._min_samples = int(sample_rate * frame_length_ms / 1000)

        logger.info(
            "FeatureExtractor initialized — %d mel bins, %.0fms window, %.0fms hop",
            num_mel_bins,
            frame_length_ms,
            frame_shift_ms,
        )

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log-mel filterbank features from audio.

        Args:
            audio: Float32 numpy array, mono, 16kHz.
                   Expected range: [-1.0, 1.0]

        Returns:
            Numpy array of shape (T, num_mel_bins) where
            T = floor((n_samples - frame_length) / frame_shift) + 1

        Raises:
            ValueError: If audio is too short (< 1 frame).
        """
        if len(audio) < self._min_samples:
            raise ValueError(
                f"Audio too short for feature extraction: {len(audio)} samples "
                f"(need at least {self._min_samples} for one frame)"
            )

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Convert to torch tensor (1, n_samples)
        waveform = torch.from_numpy(audio).unsqueeze(0)

        # Extract fbank features
        features = torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=self.sample_rate,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length_ms,
            frame_shift=self.frame_shift_ms,
            window_type=self.window_type,
            dither=self.dither,
        )  # shape: (T, num_mel_bins)

        # Optional utterance-level CMVN (cepstral mean and variance normalization)
        if self.normalize:
            features = self._apply_cmvn(features)

        return features.numpy()

    def extract_with_mask(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and create the corresponding mask.

        Returns:
            Tuple of:
              - features: (1, T, num_mel_bins) float32
              - mask: (1, T) int64
        """
        features = self.extract(audio)  # (T, C)
        T = features.shape[0]

        # Add batch dimension
        features_batch = features[np.newaxis, :, :]  # (1, T, C)
        mask = np.ones((1, T), dtype=np.int64)

        return features_batch, mask

    def compute_output_length(self, n_samples: int) -> int:
        """
        Compute the number of feature frames for a given number of audio samples.

        Args:
            n_samples: Number of audio samples.

        Returns:
            Number of feature frames.
        """
        frame_length_samples = int(self.sample_rate * self.frame_length_ms / 1000)
        frame_shift_samples = int(self.sample_rate * self.frame_shift_ms / 1000)
        if n_samples < frame_length_samples:
            return 0
        return (n_samples - frame_length_samples) // frame_shift_samples + 1

    @staticmethod
    def _apply_cmvn(features: torch.Tensor) -> torch.Tensor:
        """
        Apply utterance-level cepstral mean and variance normalization.

        Args:
            features: (T, C) tensor.

        Returns:
            Normalized features.
        """
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        # Avoid division by zero
        std = torch.clamp(std, min=1e-5)
        return (features - mean) / std

    @property
    def info(self) -> dict:
        """Feature extractor configuration info."""
        return {
            "sample_rate": self.sample_rate,
            "num_mel_bins": self.num_mel_bins,
            "frame_length_ms": self.frame_length_ms,
            "frame_shift_ms": self.frame_shift_ms,
            "window_type": self.window_type,
            "normalize": self.normalize,
        }
